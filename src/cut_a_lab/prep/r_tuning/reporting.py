"""Reporting helpers for multi-dataset R-Tuning runs."""

from __future__ import annotations

from typing import Any


def build_best_model_table(rows: list[dict[str, Any]]) -> str:
    """Build a compact plain-text table across dataset splits."""
    header = (
        f"{'Dataset':<14} {'Split':<12} {'Feature Set':<22} "
        f"{'Model':<22} {'Sample AUROC':>12} {'Span AUROC':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        best = row.get("best_model") or {}
        feature_set = best.get("feature_set") or "n/a"
        model = (
            f"{best.get('family_group', 'n/a')}/{best.get('model', 'n/a')}"
            if best
            else "n/a"
        )
        sample_auroc = best.get("sample_auroc")
        span_auroc = best.get("span_auroc")
        sample_text = "n/a" if sample_auroc is None else f"{float(sample_auroc):.4f}"
        span_text = "n/a" if span_auroc is None else f"{float(span_auroc):.4f}"
        lines.append(
            f"{str(row.get('dataset_name', 'n/a')):<14} "
            f"{str(row.get('split_name', 'n/a')):<12} "
            f"{feature_set:<22} {model:<22} {sample_text:>12} {span_text:>10}"
        )
    return "\n".join(lines)
