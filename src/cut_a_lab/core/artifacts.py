"""Artifact helpers shared by training and analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cut_a_lab.core.io import load_json


def safe_metric_value(
    payload: dict[str, Any],
    section: str,
    key: str,
    *,
    mode: str | None = None,
) -> float | None:
    """Return a finite metric value if present."""
    if section == "sample_level":
        value = payload.get(section, {}).get(mode or "max", {}).get(key)
    else:
        value = payload.get(section, {}).get(key)
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def format_metric(value: float | None) -> str:
    """Format an optional metric for human-readable reports."""
    return "n/a" if value is None else f"{value:.4f}"


def build_comparison_table(rows: list[dict[str, Any]]) -> str:
    """Build a plain-text table for comparing trained models."""
    header = (
        f"{'Feature Set':<22} {'Model':<26} {'Span AUROC':>12} "
        f"{'Sample AUROC':>12} {'Sample AUPRC':>12} {'Sample F1':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        model_label = f"{row['family_group']}/{row['model']}"
        lines.append(
            f"{row['feature_set']:<22} {model_label:<26} "
            f"{format_metric(row['span_auroc']):>12} "
            f"{format_metric(row['sample_auroc']):>12} "
            f"{format_metric(row['sample_auprc']):>12} "
            f"{format_metric(row['sample_f1']):>10}"
        )
    return "\n".join(lines)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Load JSON if the file exists, otherwise return None."""
    return load_json(path) if path.exists() else None


def load_metrics_from_prediction(prediction_path: str | Path | None) -> dict[str, Any]:
    """Load a metrics artifact matching a prediction file if present."""
    if prediction_path is None:
        return {}
    prediction = Path(prediction_path)
    metrics_path = prediction.with_name(prediction.name.replace(".oof_predictions.jsonl", ".metrics.json"))
    if not metrics_path.exists():
        return {}
    return load_json(metrics_path)
