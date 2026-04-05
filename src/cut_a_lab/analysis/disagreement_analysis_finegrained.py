"""Fine-grained span-level disagreement analysis between ICR-only and entropy-only models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cut_a_lab.analysis.disagreement_analysis import (
    QUADRANT_COLORS,
    QUADRANT_LABELS,
    attach_vectors,
    classify_span_quadrants,
    load_span_vectors,
    plot_quadrant_counts,
    plot_trajectory_comparison,
)
from cut_a_lab.core.io import dump_json, write_jsonl


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


PAIRWISE_COMPARISONS = (
    ("group_a", "group_b", "group_a_vs_b", "#333333", "A vs B"),
    ("group_a", "group_d", "group_a_vs_d", QUADRANT_COLORS["group_a"], "A vs D"),
    ("group_b", "group_d", "group_b_vs_d", QUADRANT_COLORS["group_b"], "B vs D"),
)


def _require_matplotlib():
    if plt is None:
        raise RuntimeError("matplotlib not available")
    return plt


def build_group_matrices(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert attached group vectors into numpy matrices."""
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group_key, entries in groups.items():
        valid = [entry for entry in entries if "icr_vector" in entry and "entropy_vector" in entry]
        if not valid:
            continue
        icr_matrix = np.array([entry["icr_vector"] for entry in valid], dtype=np.float64)
        entropy_matrix = np.array([entry["entropy_vector"] for entry in valid], dtype=np.float64)
        matrices[group_key] = (icr_matrix, entropy_matrix)
    return matrices


def compute_sliding_window_matrix(
    matrix: np.ndarray,
    *,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return per-row sliding-window means with metadata for each window."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}.")
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    if matrix.shape[1] < window_size:
        raise ValueError(f"window_size={window_size} exceeds matrix width {matrix.shape[1]}.")

    columns: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for start in range(0, matrix.shape[1] - window_size + 1, stride):
        end = start + window_size
        columns.append(matrix[:, start:end].mean(axis=1))
        metadata.append({
            "index": len(metadata),
            "start_layer": start,
            "end_layer": end - 1,
            "label": f"L{start}-{end - 1}",
        })

    return np.column_stack(columns), metadata


def compute_localized_features(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Compute local timing features that are less lossy than broad early/mid/late means."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}.")
    row_index = np.arange(matrix.shape[0])
    argmin_layer = np.argmin(matrix, axis=1)
    min_value = matrix[row_index, argmin_layer]

    diffs = np.diff(matrix, axis=1)
    drop_magnitude = np.maximum(-diffs, 0.0)
    max_drop_layer = np.argmax(drop_magnitude, axis=1)
    max_drop_value = drop_magnitude[row_index, max_drop_layer]

    return {
        "argmin_layer": argmin_layer.astype(np.float64),
        "min_value": min_value.astype(np.float64),
        "max_drop_layer": max_drop_layer.astype(np.float64),
        "max_drop_value": max_drop_value.astype(np.float64),
        "last_minus_first": (matrix[:, -1] - matrix[:, 0]).astype(np.float64),
    }


def summarize_localized_features(
    localized_features: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Summarize group-local timing features."""
    summary: dict[str, dict[str, float]] = {}
    for feature_name, values in localized_features.items():
        summary[feature_name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
        }
    return summary


def _mannwhitney_u(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    try:
        from scipy import stats as scipy_stats
    except ImportError as exc:
        raise RuntimeError("scipy not available") from exc

    u_stat, p_value = scipy_stats.mannwhitneyu(left, right, alternative="two-sided")
    effect_r = float(1.0 - 2.0 * u_stat / (len(left) * len(right)))
    return float(u_stat), float(p_value), effect_r


def run_pairwise_tests(
    signal_matrices: dict[str, np.ndarray],
    *,
    metadata: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run pairwise Mann-Whitney tests column-wise across requested group comparisons."""
    results: dict[str, list[dict[str, Any]]] = {}

    for left_key, right_key, result_key, _color, _label in PAIRWISE_COMPARISONS:
        if left_key not in signal_matrices or right_key not in signal_matrices:
            continue

        left_matrix = signal_matrices[left_key]
        right_matrix = signal_matrices[right_key]
        if left_matrix.shape[1] != right_matrix.shape[1]:
            raise ValueError(
                f"Mismatched widths for {result_key}: {left_matrix.shape[1]} vs {right_matrix.shape[1]}."
            )

        series: list[dict[str, Any]] = []
        for column_index in range(left_matrix.shape[1]):
            left_values = left_matrix[:, column_index]
            right_values = right_matrix[:, column_index]
            u_stat, p_value, effect_r = _mannwhitney_u(left_values, right_values)
            item: dict[str, Any] = {
                "index": column_index,
                "U": u_stat,
                "p_value": p_value,
                "effect_r": effect_r,
                "mean_diff": float(left_values.mean() - right_values.mean()),
                "left_mean": float(left_values.mean()),
                "right_mean": float(right_values.mean()),
            }
            if metadata is not None:
                item.update(metadata[column_index])
            series.append(item)
        results[result_key] = series

    return results


def run_pairwise_feature_tests(
    localized_by_group: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, Any]]:
    """Run pairwise tests over local timing features for each group comparison."""
    results: dict[str, dict[str, Any]] = {}
    for left_key, right_key, result_key, _color, _label in PAIRWISE_COMPARISONS:
        if left_key not in localized_by_group or right_key not in localized_by_group:
            continue

        comp: dict[str, Any] = {}
        left_features = localized_by_group[left_key]
        right_features = localized_by_group[right_key]
        for feature_name, left_values in left_features.items():
            right_values = right_features[feature_name]
            u_stat, p_value, effect_r = _mannwhitney_u(left_values, right_values)
            comp[feature_name] = {
                "U": u_stat,
                "p_value": p_value,
                "effect_r": effect_r,
                "mean_diff": float(left_values.mean() - right_values.mean()),
                "left_mean": float(left_values.mean()),
                "right_mean": float(right_values.mean()),
                "left_median": float(np.median(left_values)),
                "right_median": float(np.median(right_values)),
            }
        results[result_key] = comp
    return results


def plot_pairwise_significance(
    pairwise_tests_by_signal: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    output_path: Path,
    x_key: str,
    xlabel: str,
    title_suffix: str,
) -> None:
    """Plot -log10(p) and mean difference for each pairwise comparison."""
    mpl = _require_matplotlib()
    mpl.style.use("seaborn-v0_8-whitegrid")

    fig, axes = mpl.subplots(2, 2, figsize=(14, 8), sharey="row")
    for col_idx, signal_name in enumerate(("icr", "entropy")):
        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]
        comp_results = pairwise_tests_by_signal[signal_name]
        n_values = len(next(iter(comp_results.values()))) if comp_results else 0
        bonferroni_line = -np.log10(0.05 / max(n_values, 1))

        for _left, _right, result_key, color, label in PAIRWISE_COMPARISONS:
            series = comp_results.get(result_key)
            if not series:
                continue
            x = [item[x_key] for item in series]
            neg_log_p = [-np.log10(max(item["p_value"], 1e-300)) for item in series]
            mean_diff = [item["mean_diff"] for item in series]
            ax_top.plot(x, neg_log_p, color=color, linewidth=1.8, marker="o", markersize=3, label=label)
            ax_bottom.plot(x, mean_diff, color=color, linewidth=1.8, marker="o", markersize=3, label=label)

        ax_top.axhline(-np.log10(0.05), color="#888888", linewidth=1.0, linestyle="--", label="p=0.05")
        ax_top.axhline(bonferroni_line, color="#444444", linewidth=1.0, linestyle=":", label="Bonferroni")
        ax_bottom.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")

        signal_upper = signal_name.upper()
        ax_top.set_title(f"{signal_upper}: -log10(p) {title_suffix}")
        ax_top.set_ylabel("-log10(p-value)")
        ax_top.legend(frameon=True, fontsize=8)
        ax_top.grid(color="#D0D0D0", linewidth=0.6)

        ax_bottom.set_title(f"{signal_upper}: mean difference {title_suffix}")
        ax_bottom.set_xlabel(xlabel)
        ax_bottom.set_ylabel("Mean difference")
        ax_bottom.legend(frameon=True, fontsize=8)
        ax_bottom.grid(color="#D0D0D0", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    mpl.close(fig)


def plot_discrete_feature_distribution(
    localized_by_group: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    signal_name: str,
    feature_name: str,
    max_index: int,
    output_path: Path,
    title: str,
) -> None:
    """Plot normalized discrete distributions for A/B/D to inspect timing shifts."""
    mpl = _require_matplotlib()
    mpl.style.use("seaborn-v0_8-whitegrid")

    fig, ax = mpl.subplots(figsize=(9, 4.5))
    for group_key in ("group_d", "group_b", "group_a"):
        group_features = localized_by_group.get(group_key)
        if group_features is None:
            continue
        values = group_features[signal_name][feature_name].astype(int)
        counts = np.bincount(values, minlength=max_index + 1)[: max_index + 1].astype(np.float64)
        counts /= counts.sum() if counts.sum() else 1.0
        ax.plot(
            np.arange(max_index + 1),
            counts,
            color=QUADRANT_COLORS[group_key],
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=QUADRANT_LABELS[group_key],
        )

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.legend(frameon=True)
    ax.grid(color="#D0D0D0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    mpl.close(fig)


def _top_k_series(series: list[dict[str, Any]], *, k: int = 5) -> list[dict[str, Any]]:
    return sorted(series, key=lambda item: item["p_value"])[:k]


def _top_k_features(feature_tests: dict[str, dict[str, Any]], *, k: int = 5) -> list[tuple[str, dict[str, Any]]]:
    return sorted(feature_tests.items(), key=lambda item: item[1]["p_value"])[:k]


def run_disagreement_analysis_finegrained(
    *,
    icr_oof_path: Path,
    entropy_oof_path: Path,
    input_data_path: Path,
    output_dir: Path,
    threshold: float = 0.5,
    window_size: int = 3,
    window_stride: int = 1,
) -> dict[str, Any]:
    icr_oof_path = Path(icr_oof_path)
    entropy_oof_path = Path(entropy_oof_path)
    input_data_path = Path(input_data_path)
    output_dir = Path(output_dir)

    figures_dir = output_dir / "figures"
    samples_dir = output_dir / "group_samples"
    figures_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    print("Classifying span quadrants...")
    groups = classify_span_quadrants(icr_oof_path, entropy_oof_path, threshold=threshold)
    counts = {group_key: len(entries) for group_key, entries in groups.items()}
    total = sum(counts.values())

    print("Loading span vectors...")
    all_span_ids = {entry["span_id"] for entries in groups.values() for entry in entries}
    vectors = load_span_vectors(input_data_path, all_span_ids)
    attach_vectors(groups, vectors)

    group_matrices = build_group_matrices(groups)
    icr_group_matrices = {group_key: mats[0] for group_key, mats in group_matrices.items()}
    entropy_group_matrices = {group_key: mats[1] for group_key, mats in group_matrices.items()}

    print("Computing sliding-window summaries and localized features...")
    window_metadata_by_signal: dict[str, list[dict[str, Any]]] = {}
    window_matrices_by_signal: dict[str, dict[str, np.ndarray]] = {"icr": {}, "entropy": {}}
    localized_by_group: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    localized_summary: dict[str, dict[str, Any]] = {}

    for group_key, (icr_matrix, entropy_matrix) in group_matrices.items():
        icr_windows, icr_window_meta = compute_sliding_window_matrix(
            icr_matrix, window_size=window_size, stride=window_stride
        )
        entropy_windows, entropy_window_meta = compute_sliding_window_matrix(
            entropy_matrix, window_size=window_size, stride=window_stride
        )
        window_metadata_by_signal["icr"] = icr_window_meta
        window_metadata_by_signal["entropy"] = entropy_window_meta
        window_matrices_by_signal["icr"][group_key] = icr_windows
        window_matrices_by_signal["entropy"][group_key] = entropy_windows

        localized_by_group[group_key] = {
            "icr": compute_localized_features(icr_matrix),
            "entropy": compute_localized_features(entropy_matrix),
        }
        localized_summary[group_key] = {
            "icr": summarize_localized_features(localized_by_group[group_key]["icr"]),
            "entropy": summarize_localized_features(localized_by_group[group_key]["entropy"]),
        }

    print("Running pairwise tests...")
    layer_tests_by_signal = {
        "icr": run_pairwise_tests(icr_group_matrices),
        "entropy": run_pairwise_tests(entropy_group_matrices),
    }
    window_tests_by_signal = {
        "icr": run_pairwise_tests(window_matrices_by_signal["icr"], metadata=window_metadata_by_signal["icr"]),
        "entropy": run_pairwise_tests(window_matrices_by_signal["entropy"], metadata=window_metadata_by_signal["entropy"]),
    }
    localized_feature_tests = {
        "icr": run_pairwise_feature_tests({group_key: feats["icr"] for group_key, feats in localized_by_group.items()}),
        "entropy": run_pairwise_feature_tests({group_key: feats["entropy"] for group_key, feats in localized_by_group.items()}),
    }

    print("Generating figures...")
    plot_quadrant_counts(groups, figures_dir / "quadrant_counts.png")
    plot_trajectory_comparison(
        icr_group_matrices,
        signal_name="ICR",
        ylabel="ICR value",
        output_path=figures_dir / "icr_trajectory_comparison.png",
    )
    plot_trajectory_comparison(
        entropy_group_matrices,
        signal_name="Entropy",
        ylabel="Entropy (nats)",
        output_path=figures_dir / "entropy_trajectory_comparison.png",
    )
    plot_pairwise_significance(
        layer_tests_by_signal,
        output_path=figures_dir / "pairwise_layer_significance.png",
        x_key="index",
        xlabel="Layer index",
        title_suffix="per layer",
    )
    plot_pairwise_significance(
        window_tests_by_signal,
        output_path=figures_dir / "pairwise_window_significance.png",
        x_key="index",
        xlabel=f"Sliding window index (size={window_size}, stride={window_stride})",
        title_suffix="per sliding window",
    )
    plot_discrete_feature_distribution(
        localized_by_group,
        signal_name="entropy",
        feature_name="argmin_layer",
        max_index=entropy_group_matrices["group_d"].shape[1] - 1,
        output_path=figures_dir / "entropy_argmin_layer_distribution.png",
        title="Entropy Argmin Layer Distribution",
    )
    plot_discrete_feature_distribution(
        localized_by_group,
        signal_name="entropy",
        feature_name="max_drop_layer",
        max_index=entropy_group_matrices["group_d"].shape[1] - 2,
        output_path=figures_dir / "entropy_max_drop_layer_distribution.png",
        title="Entropy Max Drop Layer Distribution",
    )

    print("Saving group samples and report...")
    for group_key, entries in groups.items():
        rows_to_save = [
            {key: value for key, value in entry.items() if key not in ("icr_vector", "entropy_vector")}
            for entry in entries
        ]
        write_jsonl(samples_dir / f"{group_key}_samples.jsonl", rows_to_save)

    report: dict[str, Any] = {
        "icr_oof_path": str(icr_oof_path),
        "entropy_oof_path": str(entropy_oof_path),
        "input_data_path": str(input_data_path),
        "threshold": threshold,
        "window_size": window_size,
        "window_stride": window_stride,
        "quadrant_counts": counts,
        "total_labeled_spans": total,
        "localized_summary": localized_summary,
        "pairwise_layer_tests": layer_tests_by_signal,
        "pairwise_window_tests": window_tests_by_signal,
        "localized_feature_tests": localized_feature_tests,
    }
    dump_json(output_dir / "disagreement_report_finegrained.json", report)

    summary_lines = [
        f"Fine-grained span-level disagreement analysis (threshold={threshold}, window_size={window_size}, stride={window_stride})",
        f"ICR OOF:     {icr_oof_path}",
        f"Entropy OOF: {entropy_oof_path}",
        f"Total labeled spans: {total}",
        "",
        "Quadrant counts:",
    ]
    for group_key in ("group_a", "group_b", "group_c", "group_d"):
        count = counts[group_key]
        summary_lines.append(f"  {QUADRANT_LABELS[group_key]}: {count} ({100 * count / max(total, 1):.1f}%)")

    summary_lines.extend(["", "Entropy localized summaries (group mean / median):"])
    for group_key in ("group_a", "group_b", "group_d"):
        if group_key not in localized_summary:
            continue
        entropy_summary = localized_summary[group_key]["entropy"]
        summary_lines.append(
            f"  {QUADRANT_LABELS[group_key]}: "
            f"argmin_layer={entropy_summary['argmin_layer']['mean']:.2f}/{entropy_summary['argmin_layer']['median']:.1f} "
            f"max_drop_layer={entropy_summary['max_drop_layer']['mean']:.2f}/{entropy_summary['max_drop_layer']['median']:.1f} "
            f"max_drop_value={entropy_summary['max_drop_value']['mean']:.4f}"
        )

    summary_lines.extend(["", "Top entropy layer tests:"])
    for result_key in ("group_a_vs_b", "group_a_vs_d", "group_b_vs_d"):
        series = layer_tests_by_signal["entropy"].get(result_key, [])
        if not series:
            continue
        top = _top_k_series(series)
        parts = [f"layer{item['index']}(p={item['p_value']:.2e}, diff={item['mean_diff']:+.4f})" for item in top]
        summary_lines.append(f"  {result_key}: {', '.join(parts)}")

    summary_lines.extend(["", "Top entropy sliding-window tests:"])
    for result_key in ("group_a_vs_b", "group_a_vs_d", "group_b_vs_d"):
        series = window_tests_by_signal["entropy"].get(result_key, [])
        if not series:
            continue
        top = _top_k_series(series)
        parts = [
            f"{item['label']}(p={item['p_value']:.2e}, diff={item['mean_diff']:+.4f})"
            for item in top
        ]
        summary_lines.append(f"  {result_key}: {', '.join(parts)}")

    summary_lines.extend(["", "Top entropy localized-feature tests:"])
    for result_key in ("group_a_vs_b", "group_a_vs_d", "group_b_vs_d"):
        comp = localized_feature_tests["entropy"].get(result_key, {})
        if not comp:
            continue
        top = _top_k_features(comp)
        parts = [
            f"{feature_name}(p={stats['p_value']:.2e}, diff={stats['mean_diff']:+.4f})"
            for feature_name, stats in top
        ]
        summary_lines.append(f"  {result_key}: {', '.join(parts)}")

    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "disagreement_summary_finegrained.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)

    return report
