"""Validate item-level sign flip and layer-block structure from ICR features.

This script operates directly on ParaRel combined method inputs, so it can reuse
existing cached features without rerunning inference.

It produces two validations:
  1. Item-level correlation between two selected ICR layers (default: L13 vs L27)
  2. Full layer x layer correlation matrices (Pearson + Spearman) with a simple
     contiguous two-block summary

Example:
    uv run python scripts/experiments/validate_sign_flip.py \
        --method-input-root outputs/experiments/llama-3.1-8b-instruct/method_inputs/pararel \
        --output-dir outputs/experiments/llama-3.1-8b-instruct/sign_flip_validation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DEFAULT_SPLITS = ("id_test", "ood_test")


@dataclass
class SplitData:
    sample_ids: list[str]
    labels: np.ndarray
    matrix: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method-input-root",
        type=Path,
        required=True,
        help="Path to method_inputs/<dataset>/, e.g. outputs/.../method_inputs/pararel",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for json/txt summaries and figures.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help=f"Splits to analyze. Defaults to: {' '.join(DEFAULT_SPLITS)}",
    )
    parser.add_argument(
        "--layer-a",
        type=int,
        default=13,
        help="First 0-based ICR layer index for the pair correlation check.",
    )
    parser.add_argument(
        "--layer-b",
        type=int,
        default=27,
        help="Second 0-based ICR layer index for the pair correlation check.",
    )
    parser.add_argument(
        "--label-source",
        choices=("sample_label", "silver_label", "contains_match"),
        default="sample_label",
        help="Label source. `sample_label` is the default to stay aligned with current saved layer-ablation results.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title prefix for figures. Defaults to output-dir name.",
    )
    return parser.parse_args()


def contains_match(generated: str, expected: str) -> bool:
    return str(expected).strip().lower() in str(generated).strip().lower()


def resolve_label(row: dict[str, Any], label_source: str) -> int:
    if label_source == "contains_match":
        return int(not contains_match(row["generated_text"], row["expected_answer"]))
    value = row.get(label_source)
    if value is None:
        raise KeyError(f"Row {row.get('sample_id', '<unknown>')} is missing label field {label_source!r}.")
    return int(value)


def load_split(method_input_root: Path, split_name: str, label_source: str) -> SplitData:
    path = method_input_root / split_name / "combined_spans.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    sample_ids: list[str] = []
    labels: list[int] = []
    vectors: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_ids.append(str(row["sample_id"]))
            labels.append(resolve_label(row, label_source))
            vectors.append(row["icr_vector"])

    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{path} produced an invalid ICR matrix with shape {matrix.shape}.")
    return SplitData(
        sample_ids=sample_ids,
        labels=np.asarray(labels, dtype=np.int32),
        matrix=matrix,
    )


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan"), float("nan")
    result = stats.pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def safe_spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def pair_correlation_summary(matrix: np.ndarray, labels: np.ndarray, layer_a: int, layer_b: int) -> dict[str, Any]:
    if layer_a >= matrix.shape[1] or layer_b >= matrix.shape[1]:
        raise ValueError(
            f"Requested layers L{layer_a} and L{layer_b}, but matrix width is {matrix.shape[1]}."
        )

    x = matrix[:, layer_a]
    y = matrix[:, layer_b]
    pearson_r, pearson_p = safe_pearsonr(x, y)
    spearman_rho, spearman_p = safe_spearmanr(x, y)

    by_label: dict[str, Any] = {}
    for label_value in sorted(np.unique(labels).tolist()):
        mask = labels == label_value
        if int(mask.sum()) < 3:
            continue
        sub_x = x[mask]
        sub_y = y[mask]
        sub_pearson_r, sub_pearson_p = safe_pearsonr(sub_x, sub_y)
        sub_spearman_rho, sub_spearman_p = safe_spearmanr(sub_x, sub_y)
        by_label[str(int(label_value))] = {
            "n_samples": int(mask.sum()),
            "pearson_r": sub_pearson_r,
            "pearson_p": sub_pearson_p,
            "spearman_rho": sub_spearman_rho,
            "spearman_p": sub_spearman_p,
        }

    return {
        "layer_a": int(layer_a),
        "layer_b": int(layer_b),
        "n_samples": int(matrix.shape[0]),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "by_label": by_label,
    }


def pearson_matrix(matrix: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def spearman_matrix(matrix: np.ndarray) -> np.ndarray:
    result = stats.spearmanr(matrix, axis=0)
    corr = np.asarray(result.statistic, dtype=np.float64)
    if corr.ndim == 0:
        corr = np.asarray([[1.0]], dtype=np.float64)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def off_diagonal_values(corr: np.ndarray) -> np.ndarray:
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return corr[mask]


def strongest_pair(corr: np.ndarray, *, mode: str) -> dict[str, Any]:
    work = corr.copy()
    upper = np.triu_indices_from(work, k=1)
    values = work[upper]
    if values.size == 0:
        return {"layer_i": None, "layer_j": None, "corr": float("nan")}
    index = int(np.argmax(values) if mode == "max" else np.argmin(values))
    layer_i = int(upper[0][index])
    layer_j = int(upper[1][index])
    return {"layer_i": layer_i, "layer_j": layer_j, "corr": float(values[index])}


def contiguous_two_block_summary(corr: np.ndarray) -> dict[str, Any]:
    n_layers = corr.shape[0]
    if n_layers < 3:
        return {
            "cut_after_layer": None,
            "left_range": None,
            "right_range": None,
            "mean_within": float("nan"),
            "mean_cross": float("nan"),
            "within_minus_cross": float("nan"),
            "negative_cross_fraction": float("nan"),
            "positive_within_fraction": float("nan"),
        }

    best: dict[str, Any] | None = None
    for cut in range(n_layers - 1):
        left = corr[: cut + 1, : cut + 1]
        right = corr[cut + 1 :, cut + 1 :]
        cross = corr[: cut + 1, cut + 1 :]
        if left.shape[0] < 2 or right.shape[0] < 2:
            continue

        within_values = np.concatenate([off_diagonal_values(left), off_diagonal_values(right)])
        cross_values = cross.ravel()
        mean_within = float(within_values.mean()) if within_values.size else float("nan")
        mean_cross = float(cross_values.mean()) if cross_values.size else float("nan")
        candidate = {
            "cut_after_layer": int(cut),
            "left_range": [0, int(cut)],
            "right_range": [int(cut + 1), int(n_layers - 1)],
            "mean_within": mean_within,
            "mean_cross": mean_cross,
            "within_minus_cross": float(mean_within - mean_cross),
            "negative_cross_fraction": float(np.mean(cross_values < 0.0)),
            "positive_within_fraction": float(np.mean(within_values > 0.0)),
        }
        if best is None or candidate["within_minus_cross"] > best["within_minus_cross"]:
            best = candidate

    if best is None:
        return {
            "cut_after_layer": None,
            "left_range": None,
            "right_range": None,
            "mean_within": float("nan"),
            "mean_cross": float("nan"),
            "within_minus_cross": float("nan"),
            "negative_cross_fraction": float("nan"),
            "positive_within_fraction": float("nan"),
        }
    return best


def matrix_summary(corr: np.ndarray) -> dict[str, Any]:
    off_diag = off_diagonal_values(corr)
    return {
        "n_layers": int(corr.shape[0]),
        "mean_offdiag": float(off_diag.mean()) if off_diag.size else float("nan"),
        "negative_offdiag_fraction": float(np.mean(off_diag < 0.0)) if off_diag.size else float("nan"),
        "positive_offdiag_fraction": float(np.mean(off_diag > 0.0)) if off_diag.size else float("nan"),
        "strongest_positive_pair": strongest_pair(corr, mode="max"),
        "strongest_negative_pair": strongest_pair(corr, mode="min"),
        "best_two_block": contiguous_two_block_summary(corr),
    }


def plot_pair_scatter(
    split_results: dict[str, dict[str, Any]],
    split_data: dict[str, SplitData],
    *,
    layer_a: int,
    layer_b: int,
    title_prefix: str,
    output_path: Path,
) -> None:
    n_splits = len(split_results)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5), squeeze=False)

    for ax, split_name in zip(axes[0], split_results):
        data = split_data[split_name]
        summary = split_results[split_name]["pair_correlation"]
        x = data.matrix[:, layer_a]
        y = data.matrix[:, layer_b]
        labels = data.labels

        for label_value, color in [(0, "#4C78A8"), (1, "#F58518")]:
            mask = labels == label_value
            if mask.any():
                ax.scatter(
                    x[mask],
                    y[mask],
                    s=12,
                    alpha=0.35,
                    c=color,
                    label=f"label={label_value}",
                    edgecolors="none",
                )

        if x.size >= 2:
            slope, intercept = np.polyfit(x, y, deg=1)
            xs = np.linspace(float(x.min()), float(x.max()), 200)
            ax.plot(xs, slope * xs + intercept, linestyle="--", linewidth=1.5, color="#222222")

        ax.set_title(
            f"{split_name}\n"
            f"Pearson={summary['pearson_r']:.3f}  Spearman={summary['spearman_rho']:.3f}"
        )
        ax.set_xlabel(f"ICR L{layer_a}")
        ax.set_ylabel(f"ICR L{layer_b}")
        ax.grid(alpha=0.2, linewidth=0.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.suptitle(f"{title_prefix}: item-level sign-flip check", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmaps(
    split_results: dict[str, dict[str, Any]],
    matrices: dict[str, dict[str, np.ndarray]],
    *,
    title_prefix: str,
    output_path: Path,
) -> None:
    split_names = list(split_results.keys())
    fig, axes = plt.subplots(len(split_names), 2, figsize=(10, 4.4 * len(split_names)), squeeze=False)
    image = None

    for row_idx, split_name in enumerate(split_names):
        for col_idx, method_name in enumerate(("pearson", "spearman")):
            ax = axes[row_idx][col_idx]
            corr = matrices[split_name][method_name]
            summary = split_results[split_name][f"{method_name}_matrix"]
            block = summary["best_two_block"]

            image = ax.imshow(
                corr,
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                origin="lower",
                aspect="equal",
            )
            if block["cut_after_layer"] is not None:
                boundary = block["cut_after_layer"] + 0.5
                ax.axhline(boundary, color="#111111", linestyle="--", linewidth=1.0, alpha=0.8)
                ax.axvline(boundary, color="#111111", linestyle="--", linewidth=1.0, alpha=0.8)

            ax.set_title(
                f"{split_name} | {method_name.capitalize()}\n"
                f"best cut=L{block['cut_after_layer']}  cross<0={block['negative_cross_fraction']:.2f}"
            )
            ax.set_xlabel("Layer")
            ax.set_ylabel("Layer")

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label="Correlation", fraction=0.03, pad=0.02)
    fig.suptitle(f"{title_prefix}: layer x layer ICR correlations", fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.88, right=0.9, wspace=0.25, hspace=0.35)
    fig.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt_p(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.3e}"


def build_summary_text(
    *,
    method_input_root: Path,
    label_source: str,
    split_results: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "Sign-flip validation summary",
        f"Method inputs: {method_input_root}",
        f"Label source: {label_source}",
        "",
    ]

    for split_name, payload in split_results.items():
        pair = payload["pair_correlation"]
        pearson_block = payload["pearson_matrix"]["best_two_block"]
        spearman_block = payload["spearman_matrix"]["best_two_block"]
        label_counts = payload["label_counts"]

        lines.extend(
            [
                f"[{split_name}] n={payload['n_samples']}  layers={payload['n_layers']}  label_counts={label_counts}",
                (
                    f"  Pair L{pair['layer_a']} vs L{pair['layer_b']}: "
                    f"Pearson={pair['pearson_r']:.4f} (p={fmt_p(pair['pearson_p'])}), "
                    f"Spearman={pair['spearman_rho']:.4f} (p={fmt_p(pair['spearman_p'])})"
                ),
            ]
        )
        for label_value, stats_dict in pair["by_label"].items():
            lines.append(
                f"  label={label_value}: "
                f"Pearson={stats_dict['pearson_r']:.4f} (p={fmt_p(stats_dict['pearson_p'])}), "
                f"Spearman={stats_dict['spearman_rho']:.4f} (p={fmt_p(stats_dict['spearman_p'])}), "
                f"n={stats_dict['n_samples']}"
            )
        lines.extend(
            [
                (
                    f"  Pearson matrix: neg_offdiag={payload['pearson_matrix']['negative_offdiag_fraction']:.3f}, "
                    f"best_cut=L{pearson_block['cut_after_layer']}, "
                    f"within-cross={pearson_block['within_minus_cross']:.3f}, "
                    f"cross<0={pearson_block['negative_cross_fraction']:.3f}"
                ),
                (
                    f"  Spearman matrix: neg_offdiag={payload['spearman_matrix']['negative_offdiag_fraction']:.3f}, "
                    f"best_cut=L{spearman_block['cut_after_layer']}, "
                    f"within-cross={spearman_block['within_minus_cross']:.3f}, "
                    f"cross<0={spearman_block['negative_cross_fraction']:.3f}"
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrices_path = args.output_dir / "correlation_matrices.npz"
    summary_json_path = args.output_dir / "sign_flip_validation.json"
    summary_txt_path = args.output_dir / "sign_flip_validation_summary.txt"
    title_prefix = args.title or args.output_dir.name

    split_data: dict[str, SplitData] = {}
    split_results: dict[str, dict[str, Any]] = {}
    matrix_payload: dict[str, np.ndarray] = {}

    for split_name in args.splits:
        data = load_split(args.method_input_root, split_name, args.label_source)
        split_data[split_name] = data

        p_matrix = pearson_matrix(data.matrix)
        s_matrix = spearman_matrix(data.matrix)
        matrix_payload[f"{split_name}_pearson"] = p_matrix.astype(np.float32)
        matrix_payload[f"{split_name}_spearman"] = s_matrix.astype(np.float32)

        label_values, label_counts = np.unique(data.labels, return_counts=True)
        split_results[split_name] = {
            "n_samples": int(data.matrix.shape[0]),
            "n_layers": int(data.matrix.shape[1]),
            "label_counts": {str(int(v)): int(c) for v, c in zip(label_values, label_counts)},
            "pair_correlation": pair_correlation_summary(
                data.matrix,
                data.labels,
                layer_a=args.layer_a,
                layer_b=args.layer_b,
            ),
            "pearson_matrix": matrix_summary(p_matrix),
            "spearman_matrix": matrix_summary(s_matrix),
        }

        print(
            f"{split_name}: n={data.matrix.shape[0]} layers={data.matrix.shape[1]} "
            f"L{args.layer_a}-L{args.layer_b} Pearson={split_results[split_name]['pair_correlation']['pearson_r']:.4f} "
            f"Spearman={split_results[split_name]['pair_correlation']['spearman_rho']:.4f}"
        )

    np.savez_compressed(matrices_path, **matrix_payload)

    plot_pair_scatter(
        split_results,
        split_data,
        layer_a=args.layer_a,
        layer_b=args.layer_b,
        title_prefix=title_prefix,
        output_path=args.output_dir / "pair_scatter",
    )
    plot_correlation_heatmaps(
        split_results,
        {
            split_name: {
                "pearson": matrix_payload[f"{split_name}_pearson"],
                "spearman": matrix_payload[f"{split_name}_spearman"],
            }
            for split_name in split_results
        },
        title_prefix=title_prefix,
        output_path=args.output_dir / "layer_correlation_matrices",
    )

    summary_payload = {
        "method_input_root": str(args.method_input_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "label_source": args.label_source,
        "pair_layers": [int(args.layer_a), int(args.layer_b)],
        "splits": split_results,
        "correlation_matrices_path": str(matrices_path.resolve()),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    summary_txt_path.write_text(
        build_summary_text(
            method_input_root=args.method_input_root.resolve(),
            label_source=args.label_source,
            split_results=split_results,
        ),
        encoding="utf-8",
    )

    print(f"Saved summary json: {summary_json_path}")
    print(f"Saved summary txt:  {summary_txt_path}")
    print(f"Saved matrices:      {matrices_path}")
    print(f"Saved figures under: {args.output_dir}")


if __name__ == "__main__":
    main()
