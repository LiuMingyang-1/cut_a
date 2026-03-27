"""Method-agnostic error analysis for selected model outputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from cut_a_lab.core.artifacts import load_metrics_from_prediction, safe_metric_value
from cut_a_lab.core.evaluation import aggregate_sample_predictions
from cut_a_lab.core.io import dump_json, read_jsonl


def _prediction_key(root: Path, prediction_path: Path) -> str:
    try:
        relative = prediction_path.relative_to(root)
    except ValueError:
        relative = prediction_path
    return str(relative).replace(".oof_predictions.jsonl", "")


def _aggregate_rows_to_sample_predictions(rows: list[dict[str, Any]], *, mode: str = "max") -> dict[str, dict[str, Any]]:
    usable_rows = [row for row in rows if row.get("probability") is not None]
    if not usable_rows:
        return {}

    probabilities = [float(row["probability"]) for row in usable_rows]
    aggregated = aggregate_sample_predictions(usable_rows, probabilities, top_k=3)[mode]
    span_counts = Counter(str(row["sample_id"]) for row in usable_rows)

    sample_predictions: dict[str, dict[str, Any]] = {}
    for sample_id, label, prob in zip(
        aggregated["sample_ids"],
        aggregated["labels"],
        aggregated["probs"],
    ):
        sample_id_str = str(sample_id)
        sample_predictions[sample_id_str] = {
            "sample_id": sample_id_str,
            "sample_label": int(label),
            "probability": float(prob),
            "n_spans": int(span_counts[sample_id_str]),
        }
    return sample_predictions


def load_prediction_sets(root_path: Path) -> dict[str, dict[str, Any]]:
    """Load one file or a directory of prediction artifacts."""
    root = Path(root_path)
    if root.is_file():
        prediction_paths = [root]
        root = root.parent
    else:
        prediction_paths = sorted(root.rglob("*.oof_predictions.jsonl"))

    if not prediction_paths:
        raise FileNotFoundError(f"No `.oof_predictions.jsonl` files found under {root_path}.")

    loaded: dict[str, dict[str, Any]] = {}
    for prediction_path in prediction_paths:
        rows = read_jsonl(prediction_path)
        metrics = load_metrics_from_prediction(prediction_path)
        model_name = (
            metrics.get("model")
            or (rows[0].get("model") if rows else None)
            or prediction_path.name.replace(".oof_predictions.jsonl", "")
        )
        feature_set = metrics.get("feature_set") or (rows[0].get("feature_set") if rows else None)
        key = _prediction_key(root, prediction_path)

        loaded[key] = {
            "key": key,
            "prediction_path": str(prediction_path),
            "metrics": metrics,
            "family": metrics.get("family") or prediction_path.parent.name,
            "feature_set": feature_set,
            "model": model_name,
            "span_rows": rows,
            "sample_predictions": _aggregate_rows_to_sample_predictions(rows, mode="max"),
            "sample_count": len({row["sample_id"] for row in rows}) if rows else 0,
            "sample_auroc": safe_metric_value(metrics, "sample_level", "AUROC_mean", mode="max"),
            "span_auroc": safe_metric_value(metrics, "span_level", "AUROC_mean"),
        }
    return loaded


def select_primary_prediction(prediction_sets: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Select the best available prediction set using sample AUROC first."""
    if not prediction_sets:
        raise ValueError("No prediction sets available.")

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, int]:
        payload = item[1]
        sample_auroc = payload.get("sample_auroc")
        span_auroc = payload.get("span_auroc")
        sample_count = payload.get("sample_count") or 0
        return (
            float("-inf") if sample_auroc is None else float(sample_auroc),
            float("-inf") if span_auroc is None else float(span_auroc),
            int(sample_count),
        )

    return max(prediction_sets.items(), key=sort_key)


def identify_errors(predictions: dict[str, Any], *, threshold: float = 0.5) -> dict[str, Any]:
    """Split sample-level predictions into TP, TN, FP, and FN groups."""
    sample_predictions = predictions.get("sample_predictions", predictions)
    groups = {"tp": [], "tn": [], "fp": [], "fn": []}

    for sample_id, payload in sorted(sample_predictions.items()):
        probability = float(payload["probability"])
        label = int(payload["sample_label"])
        predicted_label = int(probability >= threshold)
        entry = {
            "sample_id": str(sample_id),
            "sample_label": label,
            "predicted_label": predicted_label,
            "probability": probability,
        }
        if predicted_label == 1 and label == 1:
            groups["tp"].append(entry)
        elif predicted_label == 0 and label == 0:
            groups["tn"].append(entry)
        elif predicted_label == 1 and label == 0:
            groups["fp"].append(entry)
        else:
            groups["fn"].append(entry)

    total = sum(len(entries) for entries in groups.values())
    return {
        "threshold": float(threshold),
        "n_samples": total,
        "counts": {name: len(entries) for name, entries in groups.items()},
        "groups": groups,
    }


def compare_model_corrections(
    baseline_preds: dict[str, Any],
    candidate_preds: dict[str, Any],
) -> dict[str, Any]:
    """Compare baseline predictions against the selected recipe model."""
    baseline_samples = baseline_preds.get("sample_predictions", baseline_preds)
    candidate_samples = candidate_preds.get("sample_predictions", candidate_preds)

    baseline_ids = set(baseline_samples)
    candidate_ids = set(candidate_samples)
    shared_ids = sorted(baseline_ids & candidate_ids)

    groups = {
        "corrected": [],
        "introduced_errors": [],
        "remaining_errors": [],
        "both_correct": [],
        "corrected_false_positives": [],
        "corrected_false_negatives": [],
        "new_false_positives": [],
        "new_false_negatives": [],
    }

    for sample_id in shared_ids:
        baseline = baseline_samples[sample_id]
        candidate = candidate_samples[sample_id]

        label = int(baseline["sample_label"])
        baseline_prob = float(baseline["probability"])
        candidate_prob = float(candidate["probability"])
        baseline_pred = int(baseline_prob >= 0.5)
        candidate_pred = int(candidate_prob >= 0.5)
        baseline_error = baseline_pred != label
        candidate_error = candidate_pred != label

        entry = {
            "sample_id": sample_id,
            "sample_label": label,
            "baseline_probability": baseline_prob,
            "candidate_probability": candidate_prob,
            "baseline_predicted_label": baseline_pred,
            "candidate_predicted_label": candidate_pred,
        }

        if baseline_error and not candidate_error:
            groups["corrected"].append(entry)
            if label == 1:
                groups["corrected_false_negatives"].append(entry)
            else:
                groups["corrected_false_positives"].append(entry)
        elif not baseline_error and candidate_error:
            groups["introduced_errors"].append(entry)
            if label == 1:
                groups["new_false_negatives"].append(entry)
            else:
                groups["new_false_positives"].append(entry)
        elif baseline_error and candidate_error:
            groups["remaining_errors"].append(entry)
        else:
            groups["both_correct"].append(entry)

    return {
        "threshold": 0.5,
        "n_shared_samples": len(shared_ids),
        "baseline_only_sample_ids": sorted(baseline_ids - candidate_ids),
        "candidate_only_sample_ids": sorted(candidate_ids - baseline_ids),
        "counts": {name: len(entries) for name, entries in groups.items()},
        "groups": groups,
    }


def _candidate_summaries(prediction_sets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, payload in sorted(prediction_sets.items()):
        rows.append(
            {
                "key": key,
                "feature_set": payload.get("feature_set"),
                "family": payload.get("family"),
                "model": payload.get("model"),
                "prediction_path": payload.get("prediction_path"),
                "sample_count": payload.get("sample_count"),
                "sample_auroc": payload.get("sample_auroc"),
                "span_auroc": payload.get("span_auroc"),
            }
        )
    return rows


def run_error_analysis(
    *,
    training_dir: Path,
    output_dir: Path,
    baseline_dir: Path | None = None,
) -> dict[str, Any]:
    """Run selected-model error analysis, optionally against an external baseline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_candidates = load_prediction_sets(Path(training_dir))
    training_key, training_model = select_primary_prediction(training_candidates)
    training_errors = identify_errors(training_model, threshold=0.5)

    baseline_candidates = None
    baseline_key = None
    baseline_model = None
    baseline_errors = None
    corrections = None

    if baseline_dir is not None:
        baseline_candidates = load_prediction_sets(Path(baseline_dir))
        baseline_key, baseline_model = select_primary_prediction(baseline_candidates)
        baseline_errors = identify_errors(baseline_model, threshold=0.5)
        corrections = compare_model_corrections(baseline_model, training_model)

    report = {
        "training_dir": str(training_dir),
        "baseline_dir": None if baseline_dir is None else str(baseline_dir),
        "candidate_models": {
            "training": _candidate_summaries(training_candidates),
            "baseline": [] if baseline_candidates is None else _candidate_summaries(baseline_candidates),
        },
        "selected_models": {
            "training": {
                "key": training_key,
                "feature_set": training_model.get("feature_set"),
                "family": training_model.get("family"),
                "model": training_model.get("model"),
                "prediction_path": training_model.get("prediction_path"),
                "sample_auroc": training_model.get("sample_auroc"),
                "span_auroc": training_model.get("span_auroc"),
            },
            "baseline": None
            if baseline_model is None
            else {
                "key": baseline_key,
                "feature_set": baseline_model.get("feature_set"),
                "family": baseline_model.get("family"),
                "model": baseline_model.get("model"),
                "prediction_path": baseline_model.get("prediction_path"),
                "sample_auroc": baseline_model.get("sample_auroc"),
                "span_auroc": baseline_model.get("span_auroc"),
            },
        },
        "training_errors": training_errors,
        "baseline_errors": baseline_errors,
        "corrections": corrections,
    }

    dump_json(output_dir / "error_analysis.json", report)

    summary_lines = [
        f"Selected training model: {training_key} | sample(max) AUROC={training_model.get('sample_auroc')}",
        (
            "Training errors: "
            f"FP={training_errors['counts'].get('fp', 0)} "
            f"FN={training_errors['counts'].get('fn', 0)}"
        ),
    ]
    if baseline_model is not None and baseline_errors is not None and corrections is not None:
        summary_lines.extend(
            [
                f"Selected baseline model: {baseline_key} | sample(max) AUROC={baseline_model.get('sample_auroc')}",
                (
                    "Baseline errors: "
                    f"FP={baseline_errors['counts'].get('fp', 0)} "
                    f"FN={baseline_errors['counts'].get('fn', 0)}"
                ),
                (
                    "Correction summary: "
                    f"fixed={corrections['counts'].get('corrected', 0)} "
                    f"new_errors={corrections['counts'].get('introduced_errors', 0)} "
                    f"remaining={corrections['counts'].get('remaining_errors', 0)}"
                ),
            ]
        )

    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "error_analysis_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text.rstrip())
    return report
