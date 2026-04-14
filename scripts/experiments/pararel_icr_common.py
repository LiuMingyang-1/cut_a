"""Shared helpers for ParaRel ICR analysis scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from cut_a_lab.core.evaluation import roc_auc_binary


@dataclass
class IcrSplitData:
    split_name: str
    sample_ids: list[str]
    matrix: np.ndarray
    correctness: np.ndarray
    error: np.ndarray
    sample_label_agreement_with_contains_match: float
    sample_label_mode: str


def contains_match_correct(generated_text: str, expected_answer: str) -> int:
    return int(str(expected_answer).strip().lower() in str(generated_text).strip().lower())


def infer_sample_label_mode(sample_labels: np.ndarray, contains_match_correctness: np.ndarray) -> tuple[str, float]:
    agreement = float(np.mean(sample_labels == contains_match_correctness))
    mode = "correctness" if agreement >= 0.5 else "error"
    return mode, agreement


def load_icr_split(method_input_root: Path, split_name: str, *, label_source: str = "auto_sample") -> IcrSplitData:
    path = method_input_root / split_name / "combined_spans.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    sample_ids: list[str] = []
    sample_labels: list[int] = []
    contains_match_labels: list[int] = []
    vectors: list[list[float]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_ids.append(str(row["sample_id"]))
            sample_labels.append(int(row["sample_label"]))
            contains_match_labels.append(contains_match_correct(row["generated_text"], row["expected_answer"]))
            vectors.append(row["icr_vector"])

    matrix = np.asarray(vectors, dtype=np.float64)
    sample_label_arr = np.asarray(sample_labels, dtype=np.int32)
    contains_match_arr = np.asarray(contains_match_labels, dtype=np.int32)

    if label_source == "auto_sample":
        sample_label_mode, agreement = infer_sample_label_mode(sample_label_arr, contains_match_arr)
        correctness = sample_label_arr if sample_label_mode == "correctness" else 1 - sample_label_arr
    elif label_source == "contains_match":
        sample_label_mode, agreement = infer_sample_label_mode(sample_label_arr, contains_match_arr)
        correctness = contains_match_arr
    else:
        raise ValueError(f"Unsupported label_source={label_source!r}")

    error = 1 - correctness
    return IcrSplitData(
        split_name=split_name,
        sample_ids=sample_ids,
        matrix=matrix,
        correctness=np.asarray(correctness, dtype=np.int32),
        error=np.asarray(error, dtype=np.int32),
        sample_label_agreement_with_contains_match=agreement,
        sample_label_mode=sample_label_mode,
    )


def spearman_vector(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho = np.zeros(matrix.shape[1], dtype=np.float64)
    p = np.zeros(matrix.shape[1], dtype=np.float64)
    for layer_idx in range(matrix.shape[1]):
        result = stats.spearmanr(matrix[:, layer_idx], target)
        rho[layer_idx] = float(result.statistic)
        p[layer_idx] = float(result.pvalue)
    return rho, p


def pearson_vector(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corr = np.zeros(matrix.shape[1], dtype=np.float64)
    p = np.zeros(matrix.shape[1], dtype=np.float64)
    for layer_idx in range(matrix.shape[1]):
        result = stats.pearsonr(matrix[:, layer_idx], target)
        corr[layer_idx] = float(result.statistic)
        p[layer_idx] = float(result.pvalue)
    return corr, p


def signs_from_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isnan(values) | (values == 0.0), 1.0, np.sign(values))


def effective_auroc(raw_auroc: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_auroc, dtype=np.float64)
    return np.maximum(raw, 1.0 - raw)


def per_layer_error_auroc(matrix: np.ndarray, error: np.ndarray) -> np.ndarray:
    rows = [roc_auc_binary(error, matrix[:, layer_idx]) for layer_idx in range(matrix.shape[1])]
    return np.asarray(rows, dtype=np.float64)


def zscore_with_reference(reference: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (matrix - mean) / std


def local_signed_correlation_graph(matrix: np.ndarray, *, radius: int, method: str = "spearman") -> tuple[np.ndarray, np.ndarray]:
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}.")

    n_layers = matrix.shape[1]
    weights = np.zeros((n_layers, n_layers), dtype=np.float64)
    signs = np.zeros((n_layers, n_layers), dtype=np.float64)

    for left in range(n_layers):
        for right in range(max(0, left - radius), min(n_layers, left + radius + 1)):
            if left == right:
                continue
            if method == "spearman":
                corr = float(stats.spearmanr(matrix[:, left], matrix[:, right]).statistic)
            elif method == "pearson":
                corr = float(stats.pearsonr(matrix[:, left], matrix[:, right]).statistic)
            else:
                raise ValueError(f"Unsupported correlation method {method!r}.")
            weights[left, right] = abs(corr)
            signs[left, right] = 1.0 if corr >= 0.0 else -1.0
    return weights, signs

