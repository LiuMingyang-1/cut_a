"""Cross-model comparison: final-layer phenomenon and OOD robustness.

Loads layer ablation results from two models and generates side-by-side
figures to validate that the final-layer jump is a universal property.

Usage:
    uv run python scripts/experiments/compare_models.py \\
        --model-a-results outputs/pararel_experiment/layer_ablation \\
        --model-a-name "Qwen2.5-7B-Instruct" \\
        --model-b-results outputs/experiments/llama-3.1-8b-instruct/layer_ablation \\
        --model-b-name "Llama-3.1-8B-Instruct" \\
        --output-dir outputs/experiments/comparison/qwen_vs_llama
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a-results", type=Path, required=True,
                        help="layer_ablation/ dir for model A")
    parser.add_argument("--model-a-name", type=str, required=True)
    parser.add_argument("--model-b-results", type=Path, required=True,
                        help="layer_ablation/ dir for model B")
    parser.add_argument("--model-b-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


COLORS = {"id": "#2563eb", "ood": "#dc2626"}


def load_exp1(results_dir: Path) -> dict:
    return json.load(open(results_dir / "exp1_layer_ablation_tf.json"))


def load_exp2(results_dir: Path) -> list:
    return json.load(open(results_dir / "exp2_ece_comparison.json"))


def plot_layer_curves(ax, rows: list[dict], title: str, show_ylabel: bool = True) -> None:
    layers = [r["layer"] for r in rows]
    id_auc = [r["id_auroc"] for r in rows]
    ood_auc = [r["ood_auroc"] for r in rows]
    n = len(layers)

    ax.plot(layers[:-1], id_auc[:-1], "o-", color=COLORS["id"], alpha=0.8,
            linewidth=1.5, markersize=3.5, label="ID test")
    ax.plot(layers[:-1], ood_auc[:-1], "s-", color=COLORS["ood"], alpha=0.8,
            linewidth=1.5, markersize=3.5, label="OOD test")
    # dashed connector to final layer
    ax.plot([layers[-2], layers[-1]], [id_auc[-2], id_auc[-1]], "--",
            color=COLORS["id"], alpha=0.4, linewidth=1)
    ax.plot([layers[-2], layers[-1]], [ood_auc[-2], ood_auc[-1]], "--",
            color=COLORS["ood"], alpha=0.4, linewidth=1)
    ax.plot(layers[-1], id_auc[-1], "*", color=COLORS["id"], markersize=13, zorder=5)
    ax.plot(layers[-1], ood_auc[-1], "*", color=COLORS["ood"], markersize=13, zorder=5)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvspan(n - 1.5, n - 0.5, color="gold", alpha=0.15)
    ax.set_xlabel("Layer index", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("AUROC", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0.30, 0.82)
    ax.grid(True, alpha=0.3)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    da = load_exp1(args.model_a_results)
    db = load_exp1(args.model_b_results)
    ea = load_exp2(args.model_a_results)
    eb = load_exp2(args.model_b_results)

    # ── Figure 1: 2x2 layer curves (ICR + Entropy, Model A + B) ──────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)

    for col, (model_name, d) in enumerate([(args.model_a_name, da), (args.model_b_name, db)]):
        for row, (feat_key, feat_label) in enumerate([("icr", "ICR"), ("entropy", "Entropy")]):
            ax = axes[row][col]
            plot_layer_curves(ax, d[feat_key],
                              f"{model_name}\n{feat_label}",
                              show_ylabel=(col == 0))

    fig.suptitle("Final-Layer Phenomenon: Cross-Model Validation\n"
                 "(★ = final layer, gold = final layer region)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(args.output_dir / "fig_cross_model_layer_curves.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(args.output_dir / "fig_cross_model_layer_curves.png", bbox_inches="tight", dpi=150)
    print("Saved fig_cross_model_layer_curves")
    plt.close()

    # ── Figure 2: OOD delta comparison bar chart ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (model_name, exp2) in zip(axes, [(args.model_a_name, ea), (args.model_b_name, eb)]):
        methods = [r["method"] for r in exp2]
        delta_auc = [r["delta_auroc"] for r in exp2]
        delta_ece = [r["delta_ece"] for r in exp2]
        x = np.arange(len(methods))
        colors = ["#16a34a" if d >= 0 else "#dc2626" for d in delta_auc]

        bars = ax.bar(x - 0.2, delta_auc, 0.35, color=colors, alpha=0.8, label="ΔAUROC")
        ax.bar(x + 0.2, delta_ece, 0.35,
               color=["#16a34a" if d <= 0 else "#dc2626" for d in delta_ece],
               alpha=0.5, hatch="///", label="ΔECE (lower=better, so negative=good)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Δ (OOD − ID)", fontsize=10)
        ax.set_title(f"{model_name}\nΔ AUROC and ΔECE (OOD − ID)", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("OOD Robustness: Δ(OOD−ID) for AUROC and ECE\n"
                 "Green = better on OOD, Red = worse on OOD",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(args.output_dir / "fig_cross_model_ood_delta.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(args.output_dir / "fig_cross_model_ood_delta.png", bbox_inches="tight", dpi=150)
    print("Saved fig_cross_model_ood_delta")
    plt.close()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Cross-model summary")
    print(f"{'='*70}")
    for model_name, d, exp2 in [
        (args.model_a_name, da, ea),
        (args.model_b_name, db, eb),
    ]:
        icr_final = d["icr"][-1]
        ent_final = d["entropy"][-1]
        print(f"\n{model_name}")
        print(f"  ICR  final layer:  ID={icr_final['id_auroc']:.4f}  OOD={icr_final['ood_auroc']:.4f}  Δ={icr_final['delta']:+.4f}")
        print(f"  Ent  final layer:  ID={ent_final['id_auroc']:.4f}  OOD={ent_final['ood_auroc']:.4f}  Δ={ent_final['delta']:+.4f}")
        for r in exp2:
            if r["method"].startswith("Sup-LR"):
                print(f"  {r['method']:28s}  ID={r['id_auroc']:.4f}  OOD={r['ood_auroc']:.4f}  Δ={r['delta_auroc']:+.4f}")

    print(f"\nFigures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
