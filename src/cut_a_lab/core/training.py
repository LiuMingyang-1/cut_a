"""Generic training runtime for method-based recipes."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cut_a_lab.core.artifacts import build_comparison_table, safe_metric_value
from cut_a_lab.core.contracts import SpanRecord
from cut_a_lab.core.evaluation import (
    aggregate_sample_predictions,
    build_group_folds,
    evaluate_binary_predictions,
    print_metrics_summary,
    summarize_metric_dicts,
)
from cut_a_lab.core.feature_views import build_feature_set_bundle
from cut_a_lab.core.io import dump_json, ensure_parent_dir, write_jsonl
from cut_a_lab.core.registry import get_method
from cut_a_lab.models.sklearn_models import build_sklearn_model_factories
from cut_a_lab.models.torch_models import build_torch_model_factories, require_torch
from cut_a_lab.recipes.base import RecipeSpec


TOP_K = 3
TORCH_EPOCHS = 50
TORCH_BATCH_SIZE = 64
TORCH_LEARNING_RATE = 1e-3
TORCH_PATIENCE = 8
TORCH_WEIGHT_DECAY = 1e-4


def _check_binary_training_labels(train_labels: np.ndarray, *, model_name: str, fold_id: int) -> None:
    unique_labels = np.unique(train_labels)
    if unique_labels.size < 2:
        raise ValueError(
            f"{model_name} fold {fold_id + 1} has only one training class in silver labels: {unique_labels.tolist()}"
        )


def _prediction_probability(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float32)
    scores = np.asarray(model.decision_function(features), dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)


def _factory_accepts_input_dim(model_factory: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(model_factory)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ) and parameter.default is inspect.Parameter.empty:
            return True
    return False


def _instantiate_model(model_factory: Callable[..., Any], *, input_dim: int) -> Any:
    if _factory_accepts_input_dim(model_factory):
        return model_factory(input_dim)
    return model_factory()


def _fit_torch_model(
    model_factory: Callable[[int], Any],
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val_labeled: np.ndarray,
    y_val_labeled: np.ndarray,
    x_val_all: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    weight_decay: float,
) -> np.ndarray:
    torch = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    criterion = torch.nn.BCELoss()
    model = model_factory(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=max(2, patience // 2),
        factor=0.5,
    )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = None
    if len(x_val_labeled):
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(x_val_labeled), torch.FloatTensor(y_val_labeled)),
            batch_size=batch_size,
            shuffle=False,
        )
    predict_loader = DataLoader(torch.FloatTensor(x_val_all), batch_size=batch_size, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    for _ in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_rows = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x).view(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item()) * len(batch_y)
            total_train_rows += len(batch_y)

        if val_loader is None:
            val_loss = total_train_loss / max(total_train_rows, 1)
        else:
            model.eval()
            total_val_loss = 0.0
            total_val_rows = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    pred = model(batch_x).view(-1)
                    loss = criterion(pred, batch_y)
                    total_val_loss += float(loss.item()) * len(batch_y)
                    total_val_rows += len(batch_y)
            val_loss = total_val_loss / max(total_val_rows, 1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        raise RuntimeError("Torch training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    outputs = []
    with torch.no_grad():
        for batch_x in predict_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).view(-1)
            outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def train_with_features(
    *,
    feature_set_name: str,
    method_names: tuple[str, ...],
    features: np.ndarray,
    feature_names: list[str],
    records: tuple[SpanRecord, ...],
    model_factories: dict[str, Callable[..., Any]],
    output_dir: Path,
    family_name: str,
    n_splits: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, dict[str, Any]]:
    """Train one model family on one feature set."""
    output_dir = Path(output_dir)
    ensure_parent_dir(output_dir / "placeholder")

    feature_matrix = np.asarray(features, dtype=np.float32)
    if feature_matrix.ndim != 2:
        raise ValueError(f"Expected features with shape [N, D], got {feature_matrix.shape}.")
    if len(records) != feature_matrix.shape[0]:
        raise ValueError(
            f"Record count and feature count must match, got records={len(records)} and features={feature_matrix.shape[0]}."
        )

    sample_ids = np.asarray([record.sample_id for record in records], dtype=object)
    sample_labels = np.asarray([record.sample_label for record in records], dtype=np.int32)
    silver_labels = np.asarray(
        [-1 if record.silver_label is None else int(record.silver_label) for record in records],
        dtype=np.int32,
    )

    use_torch = family_name == "torch"
    if use_torch:
        torch = require_torch()
        torch.manual_seed(seed)

    folds = build_group_folds(sample_ids, sample_labels, n_splits=n_splits, seed=seed)
    results: dict[str, dict[str, Any]] = {}

    for model_name, model_factory in model_factories.items():
        print(f"\n[{feature_set_name}] Training {family_name}/{model_name}")
        probabilities = np.full(len(records), np.nan, dtype=np.float32)
        fold_assignments = np.full(len(records), -1, dtype=np.int32)
        span_metrics: list[dict[str, float]] = []
        sample_metrics = {"max": [], "topk_mean": [], "noisy_or": []}

        for fold_id, (train_samples, val_samples) in enumerate(folds):
            train_mask = np.isin(sample_ids, list(train_samples)) & (silver_labels >= 0)
            val_mask_all = np.isin(sample_ids, list(val_samples))
            val_mask_labeled = val_mask_all & (silver_labels >= 0)

            _check_binary_training_labels(silver_labels[train_mask], model_name=model_name, fold_id=fold_id)

            if use_torch:
                train_x = feature_matrix[train_mask]
                val_x_all = feature_matrix[val_mask_all]
                val_x_labeled = feature_matrix[val_mask_labeled]
                train_y = silver_labels[train_mask].astype(np.float32)
                val_y_labeled = silver_labels[val_mask_labeled].astype(np.float32)

                mean = train_x.mean(axis=0, keepdims=True)
                std = train_x.std(axis=0, keepdims=True)
                std = np.where(std < 1e-6, 1.0, std)

                train_x = (train_x - mean) / std
                val_x_all = (val_x_all - mean) / std
                val_x_labeled = (val_x_labeled - mean) / std

                fold_probs = _fit_torch_model(
                    model_factory=model_factory,
                    x_train=train_x,
                    y_train=train_y,
                    x_val_labeled=val_x_labeled,
                    y_val_labeled=val_y_labeled,
                    x_val_all=val_x_all,
                    device=device,
                    epochs=TORCH_EPOCHS,
                    batch_size=TORCH_BATCH_SIZE,
                    learning_rate=TORCH_LEARNING_RATE,
                    patience=TORCH_PATIENCE,
                    weight_decay=TORCH_WEIGHT_DECAY,
                )
            else:
                model = _instantiate_model(model_factory, input_dim=feature_matrix.shape[1])
                model.fit(feature_matrix[train_mask], silver_labels[train_mask])
                fold_probs = _prediction_probability(model, feature_matrix[val_mask_all])

            probabilities[val_mask_all] = fold_probs
            fold_assignments[val_mask_all] = fold_id

            if val_mask_labeled.any():
                span_metrics.append(
                    evaluate_binary_predictions(
                        silver_labels[val_mask_labeled],
                        probabilities[val_mask_labeled],
                    )
                )

            fold_rows = [records[index].to_prediction_row(
                feature_set=feature_set_name,
                family=family_name,
                model=model_name,
                fold=fold_id,
                probability=None,
            ) for index in np.where(val_mask_all)[0]]
            aggregated = aggregate_sample_predictions(
                fold_rows,
                fold_probs,
                top_k=TOP_K,
            )
            for mode, payload in aggregated.items():
                sample_metrics[mode].append(evaluate_binary_predictions(payload["labels"], payload["probs"]))

        metrics_payload = {
            "family": family_name,
            "feature_set": feature_set_name,
            "model": model_name,
            "methods": list(method_names),
            "feature_names": feature_names,
            "feature_dim": len(feature_names),
            "n_rows": len(records),
            "n_labeled_rows": int((silver_labels >= 0).sum()),
            "span_level": summarize_metric_dicts(span_metrics),
            "sample_level": {
                mode: summarize_metric_dicts(metric_rows)
                for mode, metric_rows in sample_metrics.items()
            },
        }
        dump_json(output_dir / f"{model_name}.metrics.json", metrics_payload)
        write_jsonl(
            output_dir / f"{model_name}.oof_predictions.jsonl",
            [
                record.to_prediction_row(
                    feature_set=feature_set_name,
                    family=family_name,
                    model=model_name,
                    fold=int(fold_id),
                    probability=None if np.isnan(probability) else float(probability),
                )
                for record, probability, fold_id in zip(records, probabilities, fold_assignments)
            ],
        )
        results[model_name] = metrics_payload
        print_metrics_summary(metrics_payload, prefix=f"{feature_set_name} / {family_name} / {model_name}")

    return results


def run_recipe(
    *,
    recipe: RecipeSpec,
    method_inputs: dict[str, Path],
    output_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a recipe end to end and return the training summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_blocks: dict[str, FeatureBlock] = {}
    resolved_inputs: dict[str, str] = {}

    for method_name in recipe.method_names:
        if method_name not in method_inputs:
            raise ValueError(
                f"Recipe {recipe.name!r} requires method input {method_name!r}, but it was not provided."
            )
        method_input_path = Path(method_inputs[method_name]).expanduser().resolve()
        if not method_input_path.exists():
            raise FileNotFoundError(f"Method input for {method_name!r} not found: {method_input_path}")
        method = get_method(method_name)
        loaded_blocks[method_name] = method.load_feature_block(method_input_path)
        resolved_inputs[method_name] = str(method_input_path)

    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    comparison_rows: list[dict[str, Any]] = []

    for feature_set in recipe.feature_sets:
        records, features, feature_names = build_feature_set_bundle(
            feature_set,
            [loaded_blocks[method_name] for method_name in feature_set.methods]
        )
        feature_dir = output_dir / feature_set.name

        sklearn_results = train_with_features(
            feature_set_name=feature_set.name,
            method_names=feature_set.methods,
            features=features,
            feature_names=feature_names,
            records=records,
            model_factories=build_sklearn_model_factories(seed=42),
            output_dir=feature_dir / "sklearn",
            family_name="sklearn",
            n_splits=5,
            seed=42,
            device=device,
        )
        torch_results = train_with_features(
            feature_set_name=feature_set.name,
            method_names=feature_set.methods,
            features=features,
            feature_names=feature_names,
            records=records,
            model_factories=build_torch_model_factories(),
            output_dir=feature_dir / "torch",
            family_name="torch",
            n_splits=5,
            seed=42,
            device=device,
        )

        all_results[feature_set.name] = {"sklearn": sklearn_results, "torch": torch_results}

        for family_group, model_group in (("sklearn", sklearn_results), ("torch", torch_results)):
            for model_name, metrics in model_group.items():
                comparison_rows.append(
                    {
                        "feature_set": feature_set.name,
                        "family_group": family_group,
                        "model": model_name,
                        "methods": list(feature_set.methods),
                        "span_auroc": safe_metric_value(metrics, "span_level", "AUROC_mean"),
                        "sample_auroc": safe_metric_value(metrics, "sample_level", "AUROC_mean", mode="max"),
                        "sample_auprc": safe_metric_value(metrics, "sample_level", "AUPRC_mean", mode="max"),
                        "sample_f1": safe_metric_value(metrics, "sample_level", "F1_mean", mode="max"),
                    }
                )

    comparison_table = build_comparison_table(comparison_rows)
    print("\nRecipe comparison table")
    print(comparison_table)

    best_row = max(
        comparison_rows,
        key=lambda row: (
            float("-inf") if row["sample_auroc"] is None else float(row["sample_auroc"]),
            float("-inf") if row["span_auroc"] is None else float(row["span_auroc"]),
        ),
    )

    summary = {
        "recipe": recipe.name,
        "description": recipe.description,
        "device": device,
        "method_inputs": resolved_inputs,
        "feature_sets": [
            {"name": feature_set.name, "methods": list(feature_set.methods)}
            for feature_set in recipe.feature_sets
        ],
        "results": all_results,
        "comparison_rows": comparison_rows,
        "comparison_table": comparison_table,
        "best_model": best_row,
    }
    dump_json(output_dir / "training_summary.json", summary)
    (output_dir / "comparison_table.txt").write_text(comparison_table + "\n", encoding="utf-8")
    return summary
