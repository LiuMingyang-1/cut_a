"""Train-on-A, evaluate-on-B transfer evaluation for hallucination detection methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from cut_a_lab.core.artifacts import safe_metric_value
from cut_a_lab.core.contracts import SpanRecord
from cut_a_lab.core.evaluation import evaluate_binary_predictions, summarize_metric_dicts
from cut_a_lab.core.io import dump_json, write_jsonl
from cut_a_lab.models.torch_models import require_torch


TORCH_EPOCHS = 50
TORCH_BATCH_SIZE = 64
TORCH_LEARNING_RATE = 1e-3
TORCH_PATIENCE = 8
TORCH_WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.1


def _resolve_torch_device(device: str) -> str:
    if device != "auto":
        return device
    torch = require_torch()
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _split_train_val(
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/val at sample level to avoid within-sample leakage."""
    rng = np.random.default_rng(seed)
    unique_samples = np.unique(sample_ids)
    rng.shuffle(unique_samples)
    n_val_samples = max(1, int(len(unique_samples) * val_fraction))
    val_sample_set = set(unique_samples[:n_val_samples].tolist())
    train_mask = np.array([sid not in val_sample_set for sid in sample_ids])
    val_mask = ~train_mask
    return features[train_mask], labels[train_mask], features[val_mask], labels[val_mask]


def _fit_and_predict(
    model_factory: Callable[[int], Any],
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    device: str,
) -> np.ndarray:
    """Train an MLP and return probabilities on x_test."""
    torch = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    criterion = torch.nn.BCELoss()
    model = model_factory(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TORCH_LEARNING_RATE, weight_decay=TORCH_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=max(2, TORCH_PATIENCE // 2), factor=0.5
    )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)),
        batch_size=TORCH_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val)),
        batch_size=TORCH_BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(torch.FloatTensor(x_test), batch_size=TORCH_BATCH_SIZE, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    for _ in range(TORCH_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x).view(-1), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0.0
        total_val_rows = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                loss = criterion(model(batch_x).view(-1), batch_y)
                total_val_loss += float(loss.item()) * len(batch_y)
                total_val_rows += len(batch_y)
        val_loss = total_val_loss / max(total_val_rows, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= TORCH_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Transfer eval: training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    outputs = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            outputs.append(model(batch_x).view(-1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def run_transfer_eval(
    *,
    feature_set_name: str,
    method_names: tuple[str, ...],
    train_records: tuple[SpanRecord, ...],
    train_features: np.ndarray,
    test_records: tuple[SpanRecord, ...],
    test_features: np.ndarray,
    feature_names: list[str],
    model_factories: dict[str, Callable[[int], Any]],
    output_dir: Path,
    test_split_name: str,
    device: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    """Train on train split, predict on test split, write predictions + metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch = require_torch()
    device = _resolve_torch_device(device)
    torch.manual_seed(seed)

    train_features = np.asarray(train_features, dtype=np.float32)
    test_features = np.asarray(test_features, dtype=np.float32)

    train_sample_ids = np.array([r.sample_id for r in train_records], dtype=object)
    train_silver = np.array(
        [-1 if r.silver_label is None else int(r.silver_label) for r in train_records],
        dtype=np.int32,
    )
    test_silver = np.array(
        [-1 if r.silver_label is None else int(r.silver_label) for r in test_records],
        dtype=np.int32,
    )

    labeled_train_mask = train_silver >= 0
    if np.unique(train_silver[labeled_train_mask]).size < 2:
        status = {
            "feature_set": feature_set_name,
            "test_split": test_split_name,
            "status": "skipped",
            "skip_reason": "Train split has fewer than two labeled classes.",
        }
        dump_json(output_dir / "metrics.json", status)
        return status

    # Fit normalizer on labeled train rows, apply to all
    mean = train_features[labeled_train_mask].mean(axis=0, keepdims=True)
    std = train_features[labeled_train_mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    train_features_norm = (train_features - mean) / std
    test_features_norm = (test_features - mean) / std

    x_labeled = train_features_norm[labeled_train_mask]
    y_labeled = train_silver[labeled_train_mask].astype(np.float32)
    ids_labeled = train_sample_ids[labeled_train_mask]

    x_fit, y_fit, x_val, y_val = _split_train_val(
        x_labeled, y_labeled, ids_labeled, val_fraction=VAL_FRACTION, seed=seed
    )

    results: dict[str, Any] = {}

    for model_name, model_factory in model_factories.items():
        print(f"[transfer_eval] {feature_set_name} / {model_name} → {test_split_name}")

        test_probs = _fit_and_predict(
            model_factory,
            x_train=x_fit,
            y_train=y_fit,
            x_val=x_val,
            y_val=y_val,
            x_test=test_features_norm,
            device=device,
        )

        labeled_test_mask = test_silver >= 0
        metrics_payload: dict[str, Any] = {
            "feature_set": feature_set_name,
            "test_split": test_split_name,
            "model": model_name,
            "methods": list(method_names),
            "feature_dim": len(feature_names),
            "n_train_labeled": int(labeled_train_mask.sum()),
            "n_test_total": len(test_records),
            "n_test_labeled": int(labeled_test_mask.sum()),
            "status": "ok",
        }
        if labeled_test_mask.any():
            metrics_payload["test_metrics"] = evaluate_binary_predictions(
                test_silver[labeled_test_mask],
                test_probs[labeled_test_mask],
            )

        dump_json(output_dir / f"{model_name}.metrics.json", metrics_payload)

        prediction_rows = [
            {
                "feature_set": feature_set_name,
                "family": "torch",
                "model": model_name,
                "test_split": test_split_name,
                "span_id": record.span_id,
                "sample_id": record.sample_id,
                "sample_label": int(record.sample_label),
                "silver_label": record.silver_label,
                "is_labeled": record.silver_label is not None,
                "fold": -1,
                "probability": None if np.isnan(prob) else float(prob),
            }
            for record, prob in zip(test_records, test_probs)
        ]
        write_jsonl(output_dir / f"{model_name}.predictions.jsonl", prediction_rows)

        results[model_name] = metrics_payload
        if "test_metrics" in metrics_payload:
            m = metrics_payload["test_metrics"]
            print(
                f"  AUROC={m.get('AUROC', float('nan')):.4f} "
                f"AUPRC={m.get('AUPRC', float('nan')):.4f} "
                f"F1={m.get('F1', float('nan')):.4f}"
            )

    return results
