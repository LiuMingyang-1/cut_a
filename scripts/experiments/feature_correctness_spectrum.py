"""Compute layer-wise feature-correctness spectra from ParaRel ICR features.

This script standardizes label semantics first:
  - `correctness=1` always means the answer is correct
  - `error=1` always means the answer is wrong / hallucinated

By default it uses `sample_label` but automatically infers whether that field is
encoded as correctness or error by comparing it to a contains-match proxy.

Example:
    uv run python scripts/experiments/feature_correctness_spectrum.py \
        --method-input-root outputs/experiments/llama-3.1-8b-instruct/method_inputs/pararel \
        --output-dir outputs/experiments/llama-3.1-8b-instruct/feature_correctness_spectrum
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pararel_icr_common import (
    effective_auroc,
    load_icr_split,
    pearson_vector,
    per_layer_error_auroc,
    spearman_vector,
)


DEFAULT_SPLITS = ("train", "id_test", "ood_test")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--label-source",
        choices=("auto_sample", "contains_match"),
        default="auto_sample",
        help="`auto_sample` keeps sample_label but normalizes its polarity to correctness/error.",
    )
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def sign_change_boundaries(values: np.ndarray) -> list[int]:
    signs = np.sign(values)
    boundaries: list[int] = []
    for idx in range(len(signs) - 1):
        if signs[idx] == 0.0 or signs[idx + 1] == 0.0:
            continue
        if signs[idx] != signs[idx + 1]:
            boundaries.append(idx)
    return boundaries


def split_summary(split_name: str, layer_stats: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    spearman_values = np.asarray([row["spearman_rho_correctness"] for row in layer_stats], dtype=np.float64)
    effective_values = np.asarray([row["effective_error_auroc"] for row in layer_stats], dtype=np.float64)
    best_abs_idx = int(np.nanargmax(np.abs(spearman_values)))
    best_eff_idx = int(np.nanargmax(effective_values))
    return {
        "split_name": split_name,
        "n_samples": int(meta["n_samples"]),
        "n_layers": int(meta["n_layers"]),
        "sample_label_mode": meta["sample_label_mode"],
        "sample_label_agreement_with_contains_match": float(meta["sample_label_agreement_with_contains_match"]),
        "sign_change_boundaries": sign_change_boundaries(spearman_values),
        "best_abs_spearman_layer": {
            "layer": best_abs_idx,
            "spearman_rho_correctness": float(spearman_values[best_abs_idx]),
        },
        "best_effective_auroc_layer": {
            "layer": best_eff_idx,
            "effective_error_auroc": float(effective_values[best_eff_idx]),
        },
        "positive_spearman_layers": [int(i) for i in np.where(spearman_values > 0.0)[0]],
        "negative_spearman_layers": [int(i) for i in np.where(spearman_values < 0.0)[0]],
        "layer_stats": layer_stats,
    }


def plot_spectra(results: dict[str, dict[str, Any]], *, title_prefix: str, output_path: Path) -> None:
    fig, axes = plt.subplots(len(results), 2, figsize=(11, 4.0 * len(results)), squeeze=False)
    color_rho = "#2563eb"
    color_auc = "#dc2626"

    for row_idx, (split_name, payload) in enumerate(results.items()):
        rows = payload["layer_stats"]
        layers = [row["layer"] for row in rows]
        rho = [row["spearman_rho_correctness"] for row in rows]
        eff_auc = [row["effective_error_auroc"] for row in rows]
        raw_auc = [row["raw_error_auroc"] for row in rows]

        ax = axes[row_idx][0]
        ax.axhline(0.0, color="#111111", linewidth=1.0, alpha=0.6)
        ax.plot(layers, rho, "o-", color=color_rho, linewidth=1.8, markersize=4)
        for boundary in payload["sign_change_boundaries"]:
            ax.axvline(boundary + 0.5, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(
            f"{split_name}: Spearman(feature, correctness)\n"
            f"best |rho|=L{payload['best_abs_spearman_layer']['layer']} ({payload['best_abs_spearman_layer']['spearman_rho_correctness']:+.3f})"
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Spearman rho")
        ax.grid(alpha=0.2, linewidth=0.5)

        ax = axes[row_idx][1]
        ax.axhline(0.5, color="#111111", linewidth=1.0, alpha=0.6)
        ax.plot(layers, raw_auc, "o--", color="#f59e0b", linewidth=1.2, markersize=3.5, label="Raw error AUROC")
        ax.plot(layers, eff_auc, "o-", color=color_auc, linewidth=1.8, markersize=4, label="Effective error AUROC")
        best_eff = payload["best_effective_auroc_layer"]
        ax.scatter(
            [best_eff["layer"]],
            [best_eff["effective_error_auroc"]],
            color="#7f1d1d",
            s=40,
            zorder=5,
        )
        ax.set_title(
            f"{split_name}: Raw vs effective AUROC\n"
            f"best eff=L{best_eff['layer']} ({best_eff['effective_error_auroc']:.3f})"
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUROC")
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"{title_prefix}: feature-correctness spectra", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary_text(results: dict[str, dict[str, Any]], *, label_source: str, method_input_root: Path) -> str:
    lines = [
        "Feature-correctness spectrum summary",
        f"Method inputs: {method_input_root}",
        f"Label source: {label_source}",
        "",
    ]
    for split_name, payload in results.items():
        best_rho = payload["best_abs_spearman_layer"]
        best_eff = payload["best_effective_auroc_layer"]
        lines.extend(
            [
                (
                    f"[{split_name}] n={payload['n_samples']} layers={payload['n_layers']} "
                    f"sample_label_mode={payload['sample_label_mode']} "
                    f"sample-vs-contains-match={payload['sample_label_agreement_with_contains_match']:.3f}"
                ),
                f"  sign-change boundaries: {payload['sign_change_boundaries']}",
                (
                    f"  best |Spearman(feature, correctness)|: "
                    f"L{best_rho['layer']} ({best_rho['spearman_rho_correctness']:+.4f})"
                ),
                (
                    f"  best effective error AUROC: "
                    f"L{best_eff['layer']} ({best_eff['effective_error_auroc']:.4f})"
                ),
                f"  positive-rho layers: {payload['positive_spearman_layers']}",
                f"  negative-rho layers: {payload['negative_spearman_layers']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = args.title or args.output_dir.name

    results: dict[str, dict[str, Any]] = {}
    for split_name in args.splits:
        split = load_icr_split(args.method_input_root, split_name, label_source=args.label_source)
        spearman_rho, spearman_p = spearman_vector(split.matrix, split.correctness)
        pearson_r, pearson_p = pearson_vector(split.matrix, split.correctness)
        raw_error_auc = per_layer_error_auroc(split.matrix, split.error)
        eff_error_auc = effective_auroc(raw_error_auc)

        layer_stats = []
        for layer_idx in range(split.matrix.shape[1]):
            layer_stats.append(
                {
                    "layer": int(layer_idx),
                    "spearman_rho_correctness": float(spearman_rho[layer_idx]),
                    "spearman_p_correctness": float(spearman_p[layer_idx]),
                    "pearson_r_correctness": float(pearson_r[layer_idx]),
                    "pearson_p_correctness": float(pearson_p[layer_idx]),
                    "raw_error_auroc": float(raw_error_auc[layer_idx]),
                    "effective_error_auroc": float(eff_error_auc[layer_idx]),
                }
            )

        payload = split_summary(
            split_name,
            layer_stats,
            meta={
                "n_samples": split.matrix.shape[0],
                "n_layers": split.matrix.shape[1],
                "sample_label_mode": split.sample_label_mode,
                "sample_label_agreement_with_contains_match": split.sample_label_agreement_with_contains_match,
            },
        )
        results[split_name] = payload
        print(
            f"{split_name}: sample_label_mode={split.sample_label_mode} "
            f"best|rho|=L{payload['best_abs_spearman_layer']['layer']} "
            f"best_eff_auc=L{payload['best_effective_auroc_layer']['layer']}"
        )

    plot_spectra(results, title_prefix=title_prefix, output_path=args.output_dir / "feature_correctness_spectrum")

    summary_payload = {
        "method_input_root": str(args.method_input_root.resolve()),
        "label_source": args.label_source,
        "splits": results,
    }
    summary_json_path = args.output_dir / "feature_correctness_spectrum.json"
    summary_txt_path = args.output_dir / "feature_correctness_spectrum_summary.txt"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    summary_txt_path.write_text(
        build_summary_text(results, label_source=args.label_source, method_input_root=args.method_input_root.resolve()),
        encoding="utf-8",
    )
    print(f"Saved summary json: {summary_json_path}")
    print(f"Saved summary txt:  {summary_txt_path}")


if __name__ == "__main__":
    main()

