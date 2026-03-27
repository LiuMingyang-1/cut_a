"""Visualization helpers for training and error analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cut_a_lab.core.artifacts import load_metrics_from_prediction


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


MODEL_COLORS = {
    "baseline": "#4C78A8",
    "selected": "#E45756",
}
ERROR_COLORS = {
    "tp": "#4C78A8",
    "tn": "#72B7B2",
    "fp": "#F58518",
    "fn": "#E45756",
}
CORRECTION_COLORS = {
    "corrected": "#54A24B",
    "introduced_errors": "#E45756",
    "remaining_errors": "#B279A2",
    "both_correct": "#9D755D",
}


def _require_matplotlib():
    if plt is None:
        raise RuntimeError("Missing dependency `matplotlib`. Install project dependencies before plotting.")
    return plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _metric_value(payload: dict[str, Any], metric_name: str) -> float | None:
    value = payload.get(metric_name)
    return _safe_float(value)


def plot_top_model_comparison(training_summary: dict[str, Any], output_path: Path, *, top_k: int = 8) -> None:
    """Plot the top models by sample AUROC."""
    plot = _require_matplotlib()
    rows = training_summary.get("comparison_rows", [])
    scored_rows = [row for row in rows if _metric_value(row, "sample_auroc") is not None]
    if not scored_rows:
        raise ValueError("No comparable model rows found in training summary.")

    scored_rows.sort(key=lambda row: float(_metric_value(row, "sample_auroc") or float("-inf")), reverse=True)
    selected_rows = scored_rows[:top_k]

    labels = [f"{row['feature_set']}\n{row['family_group']}/{row['model']}" for row in selected_rows]
    sample_aurocs = [float(_metric_value(row, "sample_auroc") or 0.0) for row in selected_rows]
    span_aurocs = [float(_metric_value(row, "span_auroc") or 0.0) for row in selected_rows]
    y_positions = np.arange(len(selected_rows), dtype=np.float32)
    x_min = min(sample_aurocs + span_aurocs + [0.7]) - 0.02
    x_min = max(0.0, x_min)

    plot.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plot.subplots(figsize=(13, 8.5))
    bar_height = 0.36
    sample_bars = ax.barh(
        y_positions + bar_height / 2,
        [max(value - x_min, 0.0) for value in sample_aurocs],
        left=x_min,
        height=bar_height,
        color=MODEL_COLORS["selected"],
        edgecolor="#A22C2C",
        linewidth=0.8,
        label="Sample AUROC",
    )
    span_bars = ax.barh(
        y_positions - bar_height / 2,
        [max(value - x_min, 0.0) for value in span_aurocs],
        left=x_min,
        height=bar_height,
        color=MODEL_COLORS["baseline"],
        edgecolor="#2F4B7C",
        linewidth=0.8,
        label="Span AUROC",
    )
    ax.set_yticks(y_positions, labels=labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(x_min, 1.0)
    ax.set_xlabel("AUROC")
    ax.set_title("Top Model Comparison")
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.8)

    for bars, values in ((sample_bars, sample_aurocs), (span_bars, span_aurocs)):
        for bar, value in zip(bars, values):
            ax.text(
                value + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )

    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plot.close(fig)


def plot_selected_model_metrics(error_analysis: dict[str, Any], output_path: Path) -> None:
    """Plot selected model metrics, with optional baseline comparison."""
    plot = _require_matplotlib()
    selected = error_analysis["selected_models"]
    training_metrics = load_metrics_from_prediction(selected["training"].get("prediction_path"))
    baseline_selection = selected.get("baseline")
    baseline_metrics = (
        load_metrics_from_prediction(baseline_selection.get("prediction_path"))
        if isinstance(baseline_selection, dict)
        else {}
    )

    metric_names = ["span_auroc", "sample_auroc", "sample_auprc", "sample_f1"]
    metric_labels = ["Span AUROC", "Sample AUROC", "Sample AUPRC", "Sample F1"]

    training_payload = {
        "label": selected["training"]["key"],
        "span_auroc": training_metrics.get("span_level", {}).get("AUROC_mean"),
        "sample_auroc": training_metrics.get("sample_level", {}).get("max", {}).get("AUROC_mean"),
        "sample_auprc": training_metrics.get("sample_level", {}).get("max", {}).get("AUPRC_mean"),
        "sample_f1": training_metrics.get("sample_level", {}).get("max", {}).get("F1_mean"),
    }

    plot.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plot.subplots(figsize=(10, 5.8))
    x_positions = np.arange(len(metric_names), dtype=np.float32)

    if baseline_metrics:
        baseline_payload = {
            "label": baseline_selection["key"],
            "span_auroc": baseline_metrics.get("span_level", {}).get("AUROC_mean"),
            "sample_auroc": baseline_metrics.get("sample_level", {}).get("max", {}).get("AUROC_mean"),
            "sample_auprc": baseline_metrics.get("sample_level", {}).get("max", {}).get("AUPRC_mean"),
            "sample_f1": baseline_metrics.get("sample_level", {}).get("max", {}).get("F1_mean"),
        }
        baseline_values = [float(_metric_value(baseline_payload, name) or 0.0) for name in metric_names]
        training_values = [float(_metric_value(training_payload, name) or 0.0) for name in metric_names]
        width = 0.36
        y_min = min(baseline_values + training_values + [0.7]) - 0.02
        y_min = max(0.0, y_min)

        ax.bar(
            x_positions - width / 2,
            [max(value - y_min, 0.0) for value in baseline_values],
            width=width,
            bottom=y_min,
            color=MODEL_COLORS["baseline"],
            edgecolor="#2F4B7C",
            linewidth=0.8,
            label="Baseline",
        )
        ax.bar(
            x_positions + width / 2,
            [max(value - y_min, 0.0) for value in training_values],
            width=width,
            bottom=y_min,
            color=MODEL_COLORS["selected"],
            edgecolor="#A22C2C",
            linewidth=0.8,
            label="Selected",
        )
        ax.set_ylim(y_min, 1.0)
        ax.legend(frameon=True)
        title = "Selected Model vs Baseline"
    else:
        training_values = [float(_metric_value(training_payload, name) or 0.0) for name in metric_names]
        y_min = min(training_values + [0.7]) - 0.02
        y_min = max(0.0, y_min)
        ax.bar(
            x_positions,
            [max(value - y_min, 0.0) for value in training_values],
            width=0.55,
            bottom=y_min,
            color=MODEL_COLORS["selected"],
            edgecolor="#A22C2C",
            linewidth=0.8,
        )
        ax.set_ylim(y_min, 1.0)
        title = "Selected Model Metrics"

    ax.set_xticks(x_positions, labels=metric_labels, rotation=10)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plot.close(fig)


def plot_error_count_comparison(error_analysis: dict[str, Any], output_path: Path) -> None:
    """Plot sample error counts for the selected model and optional baseline."""
    plot = _require_matplotlib()
    training_counts = error_analysis["training_errors"]["counts"]
    baseline_error_payload = error_analysis.get("baseline_errors")
    baseline_counts = baseline_error_payload.get("counts", {}) if isinstance(baseline_error_payload, dict) else {}

    labels = ["tp", "tn", "fp", "fn"]
    x_positions = np.arange(len(labels), dtype=np.float32)

    plot.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plot.subplots(figsize=(9, 5.2))

    if baseline_counts:
        width = 0.36
        ax.bar(
            x_positions - width / 2,
            [int(baseline_counts.get(label, 0)) for label in labels],
            width=width,
            color=MODEL_COLORS["baseline"],
            edgecolor="#2F4B7C",
            linewidth=0.8,
            label="Baseline",
        )
        ax.bar(
            x_positions + width / 2,
            [int(training_counts.get(label, 0)) for label in labels],
            width=width,
            color=MODEL_COLORS["selected"],
            edgecolor="#A22C2C",
            linewidth=0.8,
            label="Selected",
        )
        ax.legend(frameon=True)
        title = "Error Count Comparison"
    else:
        ax.bar(
            x_positions,
            [int(training_counts.get(label, 0)) for label in labels],
            width=0.55,
            color=[ERROR_COLORS[label] for label in labels],
            edgecolor="#555555",
            linewidth=0.8,
        )
        title = "Selected Model Error Counts"

    ax.set_xticks(x_positions, labels=[label.upper() for label in labels])
    ax.set_ylabel("Sample Count")
    ax.set_title(title)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plot.close(fig)


def plot_correction_summary(error_analysis: dict[str, Any], output_path: Path) -> None:
    """Plot how the selected model differs from the optional baseline."""
    plot = _require_matplotlib()
    corrections = error_analysis.get("corrections")
    if not corrections:
        raise ValueError("Correction summary is unavailable without a baseline comparison.")

    counts = corrections["counts"]
    labels = ["corrected", "introduced_errors", "remaining_errors", "both_correct"]
    values = [int(counts.get(label, 0)) for label in labels]

    plot.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plot.subplots(figsize=(9, 5.2))
    ax.bar(
        np.arange(len(labels), dtype=np.float32),
        values,
        width=0.55,
        color=[CORRECTION_COLORS[label] for label in labels],
        edgecolor="#555555",
        linewidth=0.8,
    )
    ax.set_xticks(np.arange(len(labels), dtype=np.float32), labels=[label.replace("_", "\n") for label in labels])
    ax.set_ylabel("Sample Count")
    ax.set_title("Baseline-to-Selected Correction Summary")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plot.close(fig)


def generate_figures(
    *,
    run_summary_path: Path,
    training_summary_path: Path,
    error_analysis_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Generate all supported figures from existing JSON artifacts."""
    del run_summary_path

    training_summary = json.loads(Path(training_summary_path).read_text(encoding="utf-8"))
    error_analysis = json.loads(Path(error_analysis_path).read_text(encoding="utf-8"))

    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    generated: dict[str, str] = {}

    top_models_path = output_dir / "top_model_comparison.png"
    plot_top_model_comparison(training_summary, top_models_path)
    generated["top_model_comparison"] = str(top_models_path)

    selected_metrics_path = output_dir / "selected_model_metrics.png"
    plot_selected_model_metrics(error_analysis, selected_metrics_path)
    generated["selected_model_metrics"] = str(selected_metrics_path)

    error_counts_path = output_dir / "error_count_comparison.png"
    plot_error_count_comparison(error_analysis, error_counts_path)
    generated["error_count_comparison"] = str(error_counts_path)

    if error_analysis.get("corrections") is not None:
        correction_path = output_dir / "correction_summary.png"
        plot_correction_summary(error_analysis, correction_path)
        generated["correction_summary"] = str(correction_path)

    return generated
