"""Evaluation and aggregation helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def roc_auc_binary(y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray) -> float:
    """Compute AUROC without depending on sklearn metrics."""
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true_arr.sum())
    n_neg = int(len(y_true_arr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = _average_ranks(y_score_arr)
    rank_sum_pos = ranks[y_true_arr == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_binary(y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray) -> float:
    """Compute average precision for binary labels."""
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    total_pos = int(y_true_arr.sum())
    if total_pos == 0:
        return 0.0

    order = np.argsort(-y_score_arr, kind="mergesort")
    y_sorted = y_true_arr[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float((precision[y_sorted == 1]).sum() / total_pos)


def evaluate_binary_predictions(y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray) -> dict[str, float]:
    """Evaluate binary predictions at span or sample level."""
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    thresholds = np.unique(np.concatenate([y_prob_arr, np.array([0.5])]))
    best_f1 = 0.0
    best_threshold = 0.5
    best_accuracy = 0.0
    for threshold in thresholds:
        preds = (y_prob_arr >= threshold).astype(np.int32)
        tp = int(((preds == 1) & (y_true_arr == 1)).sum())
        fp = int(((preds == 1) & (y_true_arr == 0)).sum())
        fn = int(((preds == 0) & (y_true_arr == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        accuracy = float((preds == y_true_arr).mean())
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
            best_accuracy = accuracy

    return {
        "AUROC": roc_auc_binary(y_true_arr, y_prob_arr),
        "AUPRC": average_precision_binary(y_true_arr, y_prob_arr),
        "F1": best_f1,
        "Accuracy": best_accuracy,
        "Threshold": best_threshold,
    }


def summarize_metric_dicts(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    """Summarize fold-level metric dictionaries into mean and std entries."""
    if not metric_rows:
        return {}

    summary: dict[str, float] = {}
    for key in metric_rows[0]:
        if key == "Threshold":
            continue
        values = np.asarray([row[key] for row in metric_rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary


def build_group_folds(
    sample_ids: list[str] | np.ndarray,
    sample_labels: list[int] | np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[set[str], set[str]]]:
    """Build grouped folds that keep all spans from a sample together."""
    rng = np.random.default_rng(seed)
    grouped_counts = Counter(sample_ids)
    label_by_sample: dict[str, int] = {}
    for sample_id, label in zip(sample_ids, sample_labels):
        label_by_sample.setdefault(str(sample_id), int(label))

    samples_by_label: dict[int, list[str]] = defaultdict(list)
    for sample_id, label in label_by_sample.items():
        samples_by_label[label].append(sample_id)

    folds: list[dict[str, Any]] = [
        {"samples": set(), "label_counts": Counter(), "row_count": 0}
        for _ in range(n_splits)
    ]

    for label, grouped_samples in samples_by_label.items():
        shuffled = list(grouped_samples)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda sample_id: grouped_counts[sample_id], reverse=True)

        for sample_id in shuffled:
            target_fold = min(
                range(n_splits),
                key=lambda fold_idx: (
                    folds[fold_idx]["label_counts"][label],
                    folds[fold_idx]["row_count"],
                ),
            )
            folds[target_fold]["samples"].add(sample_id)
            folds[target_fold]["label_counts"][label] += 1
            folds[target_fold]["row_count"] += grouped_counts[sample_id]

    all_samples = set(label_by_sample.keys())
    return [
        (all_samples - fold_state["samples"], set(fold_state["samples"]))
        for fold_state in folds
    ]


def aggregate_probabilities(probabilities: list[float] | np.ndarray, mode: str, *, top_k: int = 3) -> float:
    """Aggregate span-level scores into a single sample-level score."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0:
        return 0.0
    if mode == "max":
        return float(probs.max())
    if mode == "topk_mean":
        k = min(top_k, probs.size)
        topk = np.partition(probs, -k)[-k:]
        return float(topk.mean())
    if mode == "noisy_or":
        return float(1.0 - np.prod(1.0 - probs))
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def aggregate_sample_predictions(
    rows: list[dict[str, Any]],
    probabilities: list[float] | np.ndarray,
    *,
    top_k: int = 3,
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate span rows into sample-level predictions."""
    grouped: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, int] = {}
    for row, probability in zip(rows, probabilities):
        grouped[str(row["sample_id"])].append(float(probability))
        labels[str(row["sample_id"])] = int(row["sample_label"])

    aggregated: dict[str, dict[str, np.ndarray]] = {}
    for mode in ("max", "topk_mean", "noisy_or"):
        sample_ids = sorted(grouped)
        probs = np.asarray(
            [aggregate_probabilities(grouped[sample_id], mode, top_k=top_k) for sample_id in sample_ids],
            dtype=np.float64,
        )
        sample_labels = np.asarray([labels[sample_id] for sample_id in sample_ids], dtype=np.int32)
        aggregated[mode] = {
            "sample_ids": np.asarray(sample_ids, dtype=object),
            "labels": sample_labels,
            "probs": probs,
        }
    return aggregated


def print_metrics_summary(metrics: dict[str, Any], *, prefix: str = "") -> None:
    """Pretty-print a training metrics summary."""
    if prefix:
        print(f"\n{'=' * 60}")
        print(f"  {prefix}")
        print(f"{'=' * 60}")

    if "span_level" in metrics:
        span_metrics = metrics["span_level"]
        print(
            "  Span-level:  "
            f"AUROC={span_metrics.get('AUROC_mean', 0):.4f}+-{span_metrics.get('AUROC_std', 0):.4f}  "
            f"AUPRC={span_metrics.get('AUPRC_mean', 0):.4f}  "
            f"F1={span_metrics.get('F1_mean', 0):.4f}"
        )

    if "sample_level" in metrics:
        for mode in ("max", "topk_mean", "noisy_or"):
            if mode not in metrics["sample_level"]:
                continue
            sample_metrics = metrics["sample_level"][mode]
            print(
                f"  Sample({mode}): "
                f"AUROC={sample_metrics.get('AUROC_mean', 0):.4f}+-{sample_metrics.get('AUROC_std', 0):.4f}  "
                f"AUPRC={sample_metrics.get('AUPRC_mean', 0):.4f}  "
                f"F1={sample_metrics.get('F1_mean', 0):.4f}"
            )
