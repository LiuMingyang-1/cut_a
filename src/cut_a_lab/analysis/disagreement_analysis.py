"""Span-level disagreement analysis between ICR-only and entropy-only models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cut_a_lab.core.io import dump_json, read_jsonl, write_jsonl
from cut_a_lab.core.feature_views import ICR_EARLY, ICR_MIDDLE, ICR_LATE, ENT_EARLY, ENT_MIDDLE, ENT_LATE


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


QUADRANT_COLORS = {
    "group_a": "#F58518",  # orange — entropy correct, ICR wrong
    "group_b": "#4C78A8",  # blue — ICR correct, entropy wrong
    "group_c": "#E45756",  # red — both wrong
    "group_d": "#72B7B2",  # teal — both correct (baseline)
}
QUADRANT_LABELS = {
    "group_a": "A: ICR-wrong Ent-right",
    "group_b": "B: ICR-right Ent-wrong",
    "group_c": "C: Both wrong",
    "group_d": "D: Both right",
}


# ---------------------------------------------------------------------------
# Data loading and quadrant classification
# ---------------------------------------------------------------------------


def classify_span_quadrants(
    icr_oof_path: Path,
    entropy_oof_path: Path,
    *,
    threshold: float = 0.5,
) -> dict[str, list[dict[str, Any]]]:
    """Join two OOF files by span_id and classify into 4 disagreement quadrants.

    Returns a dict with keys group_a, group_b, group_c, group_d, each containing
    a list of span dicts: {span_id, sample_id, silver_label, icr_prob, entropy_prob}.
    """
    icr_rows = read_jsonl(Path(icr_oof_path))
    ent_rows = read_jsonl(Path(entropy_oof_path))

    icr_by_span = {row["span_id"]: row for row in icr_rows if row.get("is_labeled") and row.get("probability") is not None}
    ent_by_span = {row["span_id"]: row for row in ent_rows if row.get("is_labeled") and row.get("probability") is not None}

    shared_span_ids = sorted(set(icr_by_span) & set(ent_by_span))

    groups: dict[str, list[dict[str, Any]]] = {
        "group_a": [],
        "group_b": [],
        "group_c": [],
        "group_d": [],
    }

    for span_id in shared_span_ids:
        icr_row = icr_by_span[span_id]
        ent_row = ent_by_span[span_id]

        silver_label = int(icr_row["silver_label"])
        icr_prob = float(icr_row["probability"])
        ent_prob = float(ent_row["probability"])

        icr_pred = int(icr_prob >= threshold)
        ent_pred = int(ent_prob >= threshold)
        icr_correct = icr_pred == silver_label
        ent_correct = ent_pred == silver_label

        entry: dict[str, Any] = {
            "span_id": span_id,
            "sample_id": str(icr_row["sample_id"]),
            "silver_label": silver_label,
            "icr_prob": icr_prob,
            "entropy_prob": ent_prob,
            "icr_correct": icr_correct,
            "entropy_correct": ent_correct,
        }

        if not icr_correct and ent_correct:
            groups["group_a"].append(entry)
        elif icr_correct and not ent_correct:
            groups["group_b"].append(entry)
        elif not icr_correct and not ent_correct:
            groups["group_c"].append(entry)
        else:
            groups["group_d"].append(entry)

    return groups


def load_span_vectors(
    input_data_path: Path,
    span_ids: set[str],
) -> dict[str, dict[str, list[float]]]:
    """Load span_vector and entropy_vector from input data for the requested span_ids.

    Returns {span_id: {"icr": [...], "entropy": [...]}}
    """
    result: dict[str, dict[str, list[float]]] = {}
    with Path(input_data_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            import json
            row = json.loads(line)
            sid = row.get("span_id")
            if sid in span_ids:
                result[sid] = {
                    "icr": row["span_vector"],
                    "entropy": row["entropy_vector"],
                }
                if len(result) == len(span_ids):
                    break
    return result


def attach_vectors(
    groups: dict[str, list[dict[str, Any]]],
    vectors: dict[str, dict[str, list[float]]],
) -> None:
    """Attach ICR and entropy vectors to each span entry in-place."""
    for entries in groups.values():
        for entry in entries:
            vec = vectors.get(entry["span_id"])
            if vec is not None:
                entry["icr_vector"] = vec["icr"]
                entry["entropy_vector"] = vec["entropy"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _slope(matrix: np.ndarray) -> np.ndarray:
    """Compute linear regression slope across layers for each row."""
    n_layers = matrix.shape[1]
    layers = np.arange(n_layers, dtype=np.float64)
    layer_mean = layers.mean()
    layer_var = ((layers - layer_mean) ** 2).sum()
    return ((matrix * (layers[None, :] - layer_mean)).sum(axis=1)) / layer_var


def compute_group_statistics(
    icr_matrix: np.ndarray,
    ent_matrix: np.ndarray,
) -> dict[str, Any]:
    """Compute summary statistics for one group given its feature matrices."""
    stats: dict[str, Any] = {"n": int(len(icr_matrix))}

    for name, mat, early, mid, late in [
        ("icr", icr_matrix, ICR_EARLY, ICR_MIDDLE, ICR_LATE),
        ("entropy", ent_matrix, ENT_EARLY, ENT_MIDDLE, ENT_LATE),
    ]:
        me = mat[:, early].mean(axis=1)
        mm = mat[:, mid].mean(axis=1)
        ml = mat[:, late].mean(axis=1)
        sl = _slope(mat)
        overall = mat.mean(axis=1)
        stats[name] = {
            "mean_early_mean": float(me.mean()),
            "mean_early_std": float(me.std()),
            "mean_mid_mean": float(mm.mean()),
            "mean_mid_std": float(mm.std()),
            "mean_late_mean": float(ml.mean()),
            "mean_late_std": float(ml.std()),
            "slope_mean": float(sl.mean()),
            "slope_std": float(sl.std()),
            "overall_mean": float(overall.mean()),
            "overall_std": float(overall.std()),
        }

    return stats


def run_per_layer_tests(
    group_vectors: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Per-layer Mann-Whitney U tests comparing each disagreement group against group_d.

    Returns per-layer p-values, effect sizes (rank-biserial r), and mean differences
    for both ICR (27 layers) and entropy (28 layers).
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return {"error": "scipy not available"}

    results: dict[str, Any] = {}
    d_icr, d_ent = group_vectors["group_d"]
    n_d_icr = len(d_icr)

    for group_name in ("group_a", "group_b", "group_c"):
        if group_name not in group_vectors:
            continue
        g_icr, g_ent = group_vectors[group_name]
        if len(g_icr) < 3:
            results[f"{group_name}_vs_d"] = {"skipped": "too few samples"}
            continue

        comp: dict[str, Any] = {}
        for signal_name, g_mat, d_mat in [
            ("icr", g_icr, d_icr),
            ("entropy", g_ent, d_ent),
        ]:
            n_layers = g_mat.shape[1]
            n_g = len(g_mat)
            layers: list[dict[str, float]] = []
            for layer_idx in range(n_layers):
                g_vals = g_mat[:, layer_idx]
                d_vals = d_mat[:, layer_idx]
                u_stat, p_val = scipy_stats.mannwhitneyu(g_vals, d_vals, alternative="two-sided")
                # rank-biserial correlation as effect size: r = 1 - 2U / (n_g * n_d)
                r = float(1.0 - 2.0 * u_stat / (n_g * n_d_icr))
                layers.append({
                    "layer": layer_idx,
                    "U": float(u_stat),
                    "p_value": float(p_val),
                    "effect_r": r,
                    "mean_diff": float(g_vals.mean() - d_vals.mean()),
                    "group_mean": float(g_vals.mean()),
                    "baseline_mean": float(d_vals.mean()),
                })
            comp[signal_name] = layers

        results[f"{group_name}_vs_d"] = comp

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _require_matplotlib():
    if plt is None:
        raise RuntimeError("matplotlib not available")
    return plt


def plot_quadrant_counts(groups: dict[str, list], output_path: Path) -> None:
    mpl = _require_matplotlib()
    mpl.style.use("seaborn-v0_8-whitegrid")
    fig, ax = mpl.subplots(figsize=(7, 4.5))
    group_keys = ["group_a", "group_b", "group_c", "group_d"]
    counts = [len(groups[k]) for k in group_keys]
    labels = [QUADRANT_LABELS[k] for k in group_keys]
    colors = [QUADRANT_COLORS[k] for k in group_keys]
    bars = ax.bar(np.arange(len(group_keys), dtype=np.float32), counts, width=0.6, color=colors, edgecolor="#555555", linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01, str(count), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(np.arange(len(group_keys), dtype=np.float32), labels=labels)
    ax.set_ylabel("Span Count")
    ax.set_title("Disagreement Quadrant Counts (Span Level)")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    mpl.close(fig)


def plot_trajectory_comparison(
    group_matrices: dict[str, np.ndarray],
    *,
    signal_name: str,
    ylabel: str,
    output_path: Path,
) -> None:
    mpl = _require_matplotlib()
    mpl.style.use("seaborn-v0_8-whitegrid")
    fig, ax = mpl.subplots(figsize=(10, 5.5))

    for group_key in ["group_d", "group_c", "group_b", "group_a"]:
        mat = group_matrices.get(group_key)
        if mat is None or len(mat) == 0:
            continue
        color = QUADRANT_COLORS[group_key]
        label = QUADRANT_LABELS[group_key]
        mean_curve = mat.mean(axis=0)
        std_curve = mat.std(axis=0)
        x = np.arange(len(mean_curve))
        ax.plot(x, mean_curve, color=color, linewidth=2.0, label=f"{label} (n={len(mat)})")
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{signal_name} Trajectory by Disagreement Group")
    ax.legend(frameon=True)
    ax.grid(color="#D0D0D0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    mpl.close(fig)


def plot_spaghetti(
    icr_matrix: np.ndarray,
    ent_matrix: np.ndarray,
    group_key: str,
    output_path: Path,
    *,
    max_curves: int = 40,
) -> None:
    mpl = _require_matplotlib()
    mpl.style.use("seaborn-v0_8-whitegrid")
    color = QUADRANT_COLORS[group_key]
    label = QUADRANT_LABELS[group_key]

    rng = np.random.default_rng(42)
    n = min(len(icr_matrix), max_curves)
    idx = rng.choice(len(icr_matrix), size=n, replace=False) if len(icr_matrix) > max_curves else np.arange(len(icr_matrix))

    fig, axes = mpl.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title in [
        (axes[0], icr_matrix[idx], "ICR Trajectory"),
        (axes[1], ent_matrix[idx], "Entropy Curve"),
    ]:
        for row in mat:
            ax.plot(row, color=color, alpha=0.12, linewidth=0.8)
        mean_curve = mat.mean(axis=0)
        ax.plot(mean_curve, color=color, linewidth=2.5, label="mean")
        ax.set_xlabel("Layer")
        ax.set_title(f"{title} — {label} (n={len(icr_matrix)} total, {n} shown)")
        ax.grid(color="#D0D0D0", linewidth=0.6)
        ax.legend(frameon=True)

    axes[0].set_ylabel("ICR value")
    axes[1].set_ylabel("Entropy (nats)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    mpl.close(fig)


def plot_summary_heatmap(
    group_stats: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    mpl = _require_matplotlib()
    mpl.style.use("seaborn-v0_8-whitegrid")

    stat_keys = ["mean_early_mean", "mean_mid_mean", "mean_late_mean", "slope_mean", "overall_mean"]
    stat_labels = ["mean early", "mean mid", "mean late", "slope", "overall"]
    group_keys = ["group_a", "group_b", "group_c", "group_d"]
    signals = ["icr", "entropy"]

    fig, axes = mpl.subplots(1, 2, figsize=(13, 5))
    for ax, signal in zip(axes, signals):
        data = np.array([
            [group_stats[g][signal][k] for k in stat_keys]
            for g in group_keys
            if g in group_stats
        ])
        present_groups = [g for g in group_keys if g in group_stats]
        row_labels = [QUADRANT_LABELS[g] for g in present_groups]

        # Normalize each column (stat) to z-score for visual comparison
        col_mean = data.mean(axis=0)
        col_std = data.std(axis=0)
        col_std = np.where(col_std == 0, 1.0, col_std)
        data_norm = (data - col_mean) / col_std

        im = ax.imshow(data_norm, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(np.arange(len(stat_labels)), labels=stat_labels, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
        ax.set_title(f"{signal.upper()} Summary Stats (z-score)")
        mpl.colorbar(im, ax=ax, shrink=0.8)

        for i in range(len(row_labels)):
            for j in range(len(stat_labels)):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(data_norm[i, j]) > 1.2 else "black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    mpl.close(fig)


def plot_layer_significance(
    per_layer_tests: dict[str, Any],
    output_path: Path,
) -> None:
    """Two-panel figure: per-layer -log10(p) and mean difference for ICR and entropy.

    Top row: -log10(p-value) across layers for each comparison group.
    Bottom row: mean difference (group - baseline) across layers.
    """
    mpl = _require_matplotlib()
    mpl.style.use("seaborn-v0_8-whitegrid")

    comparisons = [
        ("group_a_vs_d", QUADRANT_COLORS["group_a"], QUADRANT_LABELS["group_a"]),
        ("group_b_vs_d", QUADRANT_COLORS["group_b"], QUADRANT_LABELS["group_b"]),
        ("group_c_vs_d", QUADRANT_COLORS["group_c"], QUADRANT_LABELS["group_c"]),
    ]

    fig, axes = mpl.subplots(2, 2, figsize=(14, 8), sharey="row")
    alpha_threshold = -np.log10(0.05)
    bonferroni_icr = -np.log10(0.05 / 27)
    bonferroni_ent = -np.log10(0.05 / 28)

    for col_idx, (signal_name, bonferroni_line) in enumerate([("icr", bonferroni_icr), ("entropy", bonferroni_ent)]):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        for comp_key, color, label in comparisons:
            comp = per_layer_tests.get(comp_key)
            if not isinstance(comp, dict) or signal_name not in comp:
                continue
            layers_data = comp[signal_name]
            x = [d["layer"] for d in layers_data]
            neg_log_p = [-np.log10(max(d["p_value"], 1e-300)) for d in layers_data]
            mean_diff = [d["mean_diff"] for d in layers_data]

            ax_top.plot(x, neg_log_p, color=color, linewidth=1.8, label=label, marker="o", markersize=3)
            ax_bot.plot(x, mean_diff, color=color, linewidth=1.8, label=label, marker="o", markersize=3)
            ax_bot.axhline(0, color="#888888", linewidth=0.8, linestyle="--")

        # significance thresholds
        ax_top.axhline(alpha_threshold, color="#888888", linewidth=1.0, linestyle="--", label="p=0.05")
        ax_top.axhline(bonferroni_line, color="#444444", linewidth=1.0, linestyle=":", label="Bonferroni")

        signal_upper = signal_name.upper()
        ax_top.set_title(f"{signal_upper}: -log10(p) per layer vs Group D")
        ax_top.set_ylabel("-log10(p-value)")
        ax_top.legend(frameon=True, fontsize=8)
        ax_top.grid(color="#D0D0D0", linewidth=0.6)

        ax_bot.set_title(f"{signal_upper}: mean difference per layer (group - D)")
        ax_bot.set_xlabel("Layer index")
        ax_bot.set_ylabel("Mean difference")
        ax_bot.legend(frameon=True, fontsize=8)
        ax_bot.grid(color="#D0D0D0", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    mpl.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_disagreement_analysis(
    *,
    icr_oof_path: Path,
    entropy_oof_path: Path,
    input_data_path: Path,
    output_dir: Path,
    threshold: float = 0.5,
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

    counts = {k: len(v) for k, v in groups.items()}
    total = sum(counts.values())
    print(f"  Total labeled spans: {total}")
    for k, c in counts.items():
        print(f"  {QUADRANT_LABELS[k]}: {c} ({100*c/max(total,1):.1f}%)")

    print("Loading span vectors from input data...")
    all_span_ids = {entry["span_id"] for entries in groups.values() for entry in entries}
    vectors = load_span_vectors(input_data_path, all_span_ids)
    attach_vectors(groups, vectors)

    # Build numpy matrices per group (only spans that have vectors)
    group_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group_key, entries in groups.items():
        valid = [e for e in entries if "icr_vector" in e]
        if valid:
            icr_mat = np.array([e["icr_vector"] for e in valid], dtype=np.float64)
            ent_mat = np.array([e["entropy_vector"] for e in valid], dtype=np.float64)
            group_matrices[group_key] = (icr_mat, ent_mat)

    print("Computing group statistics...")
    group_stats: dict[str, Any] = {}
    for group_key, (icr_mat, ent_mat) in group_matrices.items():
        group_stats[group_key] = compute_group_statistics(icr_mat, ent_mat)

    print("Running per-layer statistical tests...")
    stat_tests = run_per_layer_tests(group_matrices)

    print("Generating figures...")
    plot_quadrant_counts(groups, figures_dir / "quadrant_counts.png")

    icr_group_matrices = {k: v[0] for k, v in group_matrices.items()}
    ent_group_matrices = {k: v[1] for k, v in group_matrices.items()}
    plot_trajectory_comparison(
        icr_group_matrices, signal_name="ICR", ylabel="ICR value",
        output_path=figures_dir / "icr_trajectory_comparison.png",
    )
    plot_trajectory_comparison(
        ent_group_matrices, signal_name="Entropy", ylabel="Entropy (nats)",
        output_path=figures_dir / "entropy_trajectory_comparison.png",
    )

    for group_key in ("group_a", "group_b"):
        if group_key in group_matrices:
            icr_mat, ent_mat = group_matrices[group_key]
            plot_spaghetti(icr_mat, ent_mat, group_key, figures_dir / f"{group_key}_spaghetti.png")

    plot_summary_heatmap(group_stats, figures_dir / "summary_statistics_heatmap.png")
    plot_layer_significance(stat_tests, figures_dir / "layer_significance.png")

    print("Saving group samples and report...")
    for group_key, entries in groups.items():
        rows_to_save = [{k: v for k, v in e.items() if k not in ("icr_vector", "entropy_vector")} for e in entries]
        write_jsonl(samples_dir / f"{group_key}_samples.jsonl", rows_to_save)

    report: dict[str, Any] = {
        "icr_oof_path": str(icr_oof_path),
        "entropy_oof_path": str(entropy_oof_path),
        "input_data_path": str(input_data_path),
        "threshold": threshold,
        "quadrant_counts": counts,
        "total_labeled_spans": total,
        "quadrant_statistics": group_stats,
        "statistical_tests": stat_tests,
    }
    dump_json(output_dir / "disagreement_report.json", report)

    summary_lines = [
        f"Span-level disagreement analysis (threshold={threshold})",
        f"ICR OOF:     {icr_oof_path}",
        f"Entropy OOF: {entropy_oof_path}",
        f"Total labeled spans: {total}",
        "",
        "Quadrant counts:",
    ]
    for k in ("group_a", "group_b", "group_c", "group_d"):
        c = counts[k]
        summary_lines.append(f"  {QUADRANT_LABELS[k]}: {c} ({100*c/max(total,1):.1f}%)")

    summary_lines.extend(["", "Key statistics (mean across spans):"])
    for group_key in ("group_a", "group_b", "group_d"):
        if group_key not in group_stats:
            continue
        s = group_stats[group_key]
        summary_lines.append(
            f"  {QUADRANT_LABELS[group_key]}: "
            f"ICR_overall={s['icr']['overall_mean']:.4f} "
            f"ICR_slope={s['icr']['slope_mean']:.4f} "
            f"Ent_overall={s['entropy']['overall_mean']:.4f} "
            f"Ent_slope={s['entropy']['slope_mean']:.4f}"
        )

    summary_lines.extend(["", "Per-layer tests (vs Group D) — top significant layers:"])
    for comp_key in ("group_a_vs_d", "group_b_vs_d"):
        comp = stat_tests.get(comp_key)
        if not isinstance(comp, dict):
            continue
        summary_lines.append(f"  {comp_key}:")
        for signal_name in ("icr", "entropy"):
            layers_data = comp.get(signal_name, [])
            if not layers_data:
                continue
            # sort by p-value ascending, show top 5
            top = sorted(layers_data, key=lambda d: d["p_value"])[:5]
            parts = [f"layer{d['layer']}(p={d['p_value']:.2e}, diff={d['mean_diff']:+.4f})" for d in top]
            summary_lines.append(f"    {signal_name}: {', '.join(parts)}")

    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "disagreement_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)

    return report
