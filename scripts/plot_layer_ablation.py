"""Generate figures for the layer ablation paper analysis.

Usage:
    uv run python scripts/plot_layer_ablation.py  # uses defaults
    uv run python scripts/plot_layer_ablation.py \\
        --results-dir outputs/experiments/llama-3.1-8b-instruct/layer_ablation \\
        --output-dir outputs/experiments/llama-3.1-8b-instruct/layer_ablation/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/pararel_experiment/layer_ablation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <results-dir>/figures/",
    )
    return parser.parse_args()


_args = _parse_args()
_results = _args.results_dir
OUT = _args.output_dir or (_results / "figures")
OUT.mkdir(parents=True, exist_ok=True)

d1 = json.load(open(_results / "exp1_layer_ablation_tf.json"))
d2 = json.load(open(_results / "exp2_ece_comparison.json"))
d3 = json.load(open(_results / "exp3_combination.json"))
d4 = json.load(open(_results / "exp4_supervised_layer_ablation.json"))

COLORS = {"id": "#2563eb", "ood": "#dc2626", "tf": "#16a34a", "sup": "#9333ea"}
ALPHA_LINE = 0.85

# ─── Figure 1: Layer AUROC curves (core paper figure) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

for ax, key, title, n_layers in [
    (axes[0], "icr", "ICR (Cosine Distance)", 28),
    (axes[1], "entropy", "Entropy (Logit Lens)", 29),
]:
    rows = d1[key]
    layers = [r["layer"] for r in rows]
    id_auc = [r["id_auroc"] for r in rows]
    ood_auc = [r["ood_auroc"] for r in rows]

    # All layers as scatter, final layer highlighted
    ax.plot(layers[:-1], id_auc[:-1], "o-", color=COLORS["id"], alpha=ALPHA_LINE,
            linewidth=1.5, markersize=4, label="ID test")
    ax.plot(layers[:-1], ood_auc[:-1], "s-", color=COLORS["ood"], alpha=ALPHA_LINE,
            linewidth=1.5, markersize=4, label="OOD test")

    # Final layer as star, connected with dashed line
    ax.plot([layers[-2], layers[-1]], [id_auc[-2], id_auc[-1]], "--",
            color=COLORS["id"], alpha=0.5, linewidth=1)
    ax.plot([layers[-2], layers[-1]], [ood_auc[-2], ood_auc[-1]], "--",
            color=COLORS["ood"], alpha=0.5, linewidth=1)
    ax.plot(layers[-1], id_auc[-1], "*", color=COLORS["id"], markersize=14,
            zorder=5, label=f"Final layer (ID={id_auc[-1]:.3f})")
    ax.plot(layers[-1], ood_auc[-1], "*", color=COLORS["ood"], markersize=14,
            zorder=5, label=f"Final layer (OOD={ood_auc[-1]:.3f})")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="Chance")
    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0.32, 0.80)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.grid(True, alpha=0.3)

    # Shade non-final region
    ax.axvspan(-0.5, n_layers - 1.5, color="gray", alpha=0.04)
    ax.axvspan(n_layers - 1.5, n_layers - 0.5, color="gold", alpha=0.12)
    ax.text(n_layers - 1, 0.34, "Final\nlayer", ha="center", fontsize=8, color="#92400e")

fig.suptitle("Training-Free AUROC by Layer: Final Layer is Uniquely Informative",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "fig1_layer_auroc_curves.pdf", bbox_inches="tight", dpi=150)
fig.savefig(OUT / "fig1_layer_auroc_curves.png", bbox_inches="tight", dpi=150)
print(f"Saved fig1")
plt.close()


# ─── Figure 2: OOD robustness — AUROC + ECE comparison bar chart ──────────────
methods_tf = [r for r in d2 if r["method"].startswith("TF")]
methods_sup = [r for r in d2 if r["method"].startswith("Sup")]

labels_tf = ["TF-ICR\n(final)", "TF-Ent\n(final)"]
labels_sup = ["Sup-LR\nICR-all", "Sup-LR\nEnt-all", "Sup-LR\nICR+Ent"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- AUROC panel ---
ax = axes[0]
x_tf = np.array([0, 1])
x_sup = np.array([3, 4, 5])
w = 0.35

for rows, x_pos, labels in [(methods_tf, x_tf, labels_tf), (methods_sup, x_sup, labels_sup)]:
    id_vals = [r["id_auroc"] for r in rows]
    od_vals = [r["ood_auroc"] for r in rows]
    ax.bar(x_pos - w/2, id_vals, w, color=COLORS["id"], alpha=0.8, label="ID" if x_pos[0] == 0 else "")
    ax.bar(x_pos + w/2, od_vals, w, color=COLORS["ood"], alpha=0.8, label="OOD" if x_pos[0] == 0 else "")
    for xi, iv, ov in zip(x_pos, id_vals, od_vals):
        ax.text(xi - w/2, iv + 0.003, f"{iv:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
        ax.text(xi + w/2, ov + 0.003, f"{ov:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

all_x = list(x_tf) + list(x_sup)
all_labels = labels_tf + labels_sup
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, fontsize=9)
ax.axvline(2, color="gray", linestyle="--", alpha=0.5)
ax.text(0.5, 0.77, "Training-free", ha="center", fontsize=9, color="green",
        transform=ax.get_xaxis_transform())
ax.text(3.5/5.5, 0.77, "Supervised", ha="center", fontsize=9, color="purple",
        transform=ax.get_xaxis_transform())
ax.set_ylabel("AUROC", fontsize=11)
ax.set_title("Discriminative Performance", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0.65, 0.85)
ax.grid(True, alpha=0.3, axis="y")

# Delta AUROC markers
for rows, x_pos in [(methods_tf, x_tf), (methods_sup, x_sup)]:
    for xi, r in zip(x_pos, rows):
        delta = r["delta_auroc"]
        color = "green" if delta >= 0 else "red"
        ax.text(xi, 0.672, f"Δ={delta:+.3f}", ha="center", fontsize=7.5, color=color, fontweight="bold")

# --- ECE panel ---
ax = axes[1]
for rows, x_pos, labels in [(methods_tf, x_tf, labels_tf), (methods_sup, x_sup, labels_sup)]:
    id_vals = [r["id_ece"] for r in rows]
    od_vals = [r["ood_ece"] for r in rows]
    ax.bar(x_pos - w/2, id_vals, w, color=COLORS["id"], alpha=0.8, label="ID" if x_pos[0] == 0 else "")
    ax.bar(x_pos + w/2, od_vals, w, color=COLORS["ood"], alpha=0.8, label="OOD" if x_pos[0] == 0 else "")
    for xi, iv, ov in zip(x_pos, id_vals, od_vals):
        ax.text(xi - w/2, iv + 0.001, f"{iv:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
        ax.text(xi + w/2, ov + 0.001, f"{ov:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, fontsize=9)
ax.axvline(2, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("ECE (lower = better)", fontsize=11)
ax.set_title("Calibration (ECE)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 0.115)
ax.grid(True, alpha=0.3, axis="y")

for rows, x_pos in [(methods_tf, x_tf), (methods_sup, x_sup)]:
    for xi, r in zip(x_pos, rows):
        delta = r["delta_ece"]
        color = "green" if delta <= 0 else "red"
        ax.text(xi, 0.003, f"Δ={delta:+.3f}", ha="center", fontsize=7.5, color=color, fontweight="bold")

fig.suptitle("OOD Robustness: Training-Free vs Supervised (ID→OOD Shift)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "fig2_ood_robustness_auroc_ece.pdf", bbox_inches="tight", dpi=150)
fig.savefig(OUT / "fig2_ood_robustness_auroc_ece.png", bbox_inches="tight", dpi=150)
print("Saved fig2")
plt.close()


# ─── Figure 3: Combination sweep + Exp4 dual-panel ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: combination sweep ---
ax = axes[0]
alphas = [r["alpha_icr"] for r in d3]
id_aucs = [r["id_auroc"] for r in d3]
ood_aucs = [r["ood_auroc"] for r in d3]

ax.plot(alphas, id_aucs, "o-", color=COLORS["id"], linewidth=2, markersize=5, label="ID AUROC")
ax.plot(alphas, ood_aucs, "s-", color=COLORS["ood"], linewidth=2, markersize=5, label="OOD AUROC")

best_ood_idx = int(np.argmax(ood_aucs))
best_id_idx = int(np.argmax(id_aucs))
ax.axvline(alphas[best_ood_idx], color=COLORS["ood"], linestyle="--", alpha=0.6)
ax.scatter([alphas[best_ood_idx]], [ood_aucs[best_ood_idx]], s=120, color=COLORS["ood"],
           zorder=6, label=f"Best OOD: α={alphas[best_ood_idx]:.2f}, AUC={ood_aucs[best_ood_idx]:.3f}")

# Reference lines (supervised baselines)
ax.axhline(0.7008, color=COLORS["sup"], linestyle=":", linewidth=1.5, alpha=0.8,
           label="Sup-LR-ICR-all OOD=0.701")
ax.axhline(0.7117, color="#ea580c", linestyle=":", linewidth=1.5, alpha=0.8,
           label="Sup-LR-Ent-all OOD=0.712")

ax.set_xlabel("α (weight on ICR, 1-α on Entropy)", fontsize=11)
ax.set_ylabel("AUROC", fontsize=11)
ax.set_title("Training-Free Combination Sweep\n(Final Layer ICR + Entropy)", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5)
ax.set_ylim(0.68, 0.76)
ax.grid(True, alpha=0.3)

# --- Right: Exp4 supervised single-layer vs TF (ICR) ---
ax = axes[1]
icr_sup = d4["icr"]
ent_sup = d4["entropy"]
icr_tf = d1["icr"]
ent_tf = d1["entropy"]

layers_icr = [r["layer"] for r in icr_sup]
sup_ood_icr = [r["ood_auroc"] for r in icr_sup]
tf_ood_icr = [r["ood_auroc"] for r in icr_tf]

layers_ent = [r["layer"] for r in ent_sup]
sup_ood_ent = [r["ood_auroc"] for r in ent_sup]
tf_ood_ent = [r["ood_auroc"] for r in ent_tf]

ax.plot(layers_icr[:-1], sup_ood_icr[:-1], "o-", color="#7c3aed", alpha=0.8,
        linewidth=1.5, markersize=4, label="Sup-LR (1-layer) ICR")
ax.plot(layers_icr[:-1], tf_ood_icr[:-1], "o--", color="#2563eb", alpha=0.6,
        linewidth=1, markersize=3, label="TF ICR")

ax.plot(layers_ent[:-1], sup_ood_ent[:-1], "s-", color="#dc2626", alpha=0.8,
        linewidth=1.5, markersize=4, label="Sup-LR (1-layer) Ent")
ax.plot(layers_ent[:-1], tf_ood_ent[:-1], "s--", color="#16a34a", alpha=0.6,
        linewidth=1, markersize=3, label="TF Ent")

# Final layer stars
ax.plot(layers_icr[-1], sup_ood_icr[-1], "*", color="#7c3aed", markersize=14, zorder=5)
ax.plot(layers_icr[-1], tf_ood_icr[-1], "*", color="#2563eb", markersize=14, zorder=5)
ax.plot(layers_ent[-1], sup_ood_ent[-1], "*", color="#dc2626", markersize=14, zorder=5)
ax.plot(layers_ent[-1], tf_ood_ent[-1], "*", color="#16a34a", markersize=14, zorder=5)

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Chance")
ax.set_xlabel("Layer index", fontsize=11)
ax.set_ylabel("OOD AUROC", fontsize=11)
ax.set_title("Supervised Single-Layer vs Training-Free\n(OOD AUROC per layer)", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left")
ax.set_ylim(0.32, 0.72)
ax.grid(True, alpha=0.3)

fig.suptitle("Combination & Single-Layer Ablation Analysis",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "fig3_combination_and_singlelayer.pdf", bbox_inches="tight", dpi=150)
fig.savefig(OUT / "fig3_combination_and_singlelayer.png", bbox_inches="tight", dpi=150)
print("Saved fig3")
plt.close()


# ─── Figure 4: Delta AUROC (ID→OOD) heatmap-style per layer ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for ax, key, tf_rows, sup_rows, title in [
    (axes[0], "icr", icr_tf, icr_sup, "ICR"),
    (axes[1], "entropy", ent_tf, ent_sup, "Entropy"),
]:
    layers = [r["layer"] for r in tf_rows]
    tf_delta = [r["delta"] for r in tf_rows]
    sup_delta = [r["delta"] for r in sup_rows]

    ax.bar([l - 0.2 for l in layers], tf_delta, 0.35, color=COLORS["tf"], alpha=0.8, label="Training-free")
    ax.bar([l + 0.2 for l in layers], sup_delta, 0.35, color=COLORS["sup"], alpha=0.8, label="Supervised (1-layer LR)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Δ AUROC (OOD − ID)", fontsize=11)
    ax.set_title(f"{title}: OOD Gap per Layer (Δ = OOD − ID)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(-1, len(layers))

    # Final layer annotation
    fl = layers[-1]
    ax.axvline(fl - 0.5, color="orange", linestyle="--", alpha=0.6)
    ax.text(fl, ax.get_ylim()[0] * 0.85, "Final\nlayer", ha="center", fontsize=8, color="#92400e")

fig.suptitle("OOD Gap (Δ AUROC) per Layer: Training-Free vs Supervised",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "fig4_delta_auroc_per_layer.pdf", bbox_inches="tight", dpi=150)
fig.savefig(OUT / "fig4_delta_auroc_per_layer.png", bbox_inches="tight", dpi=150)
print("Saved fig4")
plt.close()

print(f"\nAll figures saved to {OUT}")
print("Files:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")
