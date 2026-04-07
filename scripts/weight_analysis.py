"""Inspect what layers supervised models actually rely on.

Trains LR and MLP on all-layer ICR/Entropy features,
then examines which layer dimensions get the most weight.

Usage:
    uv run python scripts/weight_analysis.py  # uses defaults
    uv run python scripts/weight_analysis.py \\
        --features-path outputs/experiments/llama-3.1-8b-instruct/normalized_features.npz \\
        --output-dir outputs/experiments/llama-3.1-8b-instruct/weight_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("outputs/pararel_experiment/normalized_features.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <features-path parent>/weight_analysis/",
    )
    return parser.parse_args()


_args = _parse_args()
OUT = _args.output_dir or (_args.features_path.parent / "weight_analysis")
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(_args.features_path)
tr_icr = d["tr_icr_raw"]
tr_ent = d["tr_ent_raw"]
tr_lbl = d["tr_lbl"]
id_icr = d["id_icr_raw"]
id_ent = d["id_ent_raw"]
id_lbl = d["id_lbl"]
od_icr = d["od_icr_raw"]
od_ent = d["od_ent_raw"]
od_lbl = d["od_lbl"]

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


# ── Logistic Regression weights ────────────────────────────────────────────────
print("=" * 60)
print("LR coefficient analysis")
print("=" * 60)

results = {}
for name, Xtr, Xte_id, Xte_od, n_layers in [
    ("ICR",     tr_icr, id_icr, od_icr, tr_icr.shape[1]),
    ("Entropy", tr_ent, id_ent, od_ent, tr_ent.shape[1]),
]:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xid_s = scaler.transform(Xte_id)
    Xod_s = scaler.transform(Xte_od)

    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    lr.fit(Xtr_s, tr_lbl)

    coef = lr.coef_[0]          # (n_layers,)
    abs_coef = np.abs(coef)
    rank = np.argsort(-abs_coef)  # largest first

    id_auc = auroc(id_lbl, lr.predict_proba(Xid_s)[:, 1])
    od_auc = auroc(od_lbl, lr.predict_proba(Xod_s)[:, 1])

    print(f"\n{name} LR  ID={id_auc:.4f}  OOD={od_auc:.4f}")
    print(f"  Top-5 layers by |coef|: {rank[:5].tolist()}  (final layer = {n_layers-1})")
    print(f"  Coef values (all layers):")
    for l in range(n_layers):
        bar = "█" * int(abs_coef[l] / abs_coef.max() * 30)
        sign = "+" if coef[l] > 0 else "-"
        mark = " ◄ FINAL" if l == n_layers - 1 else ""
        print(f"    layer {l:>2}: {sign}{abs_coef[l]:.4f}  {bar}{mark}")

    # Share of final layer in total |coef| mass
    final_share = abs_coef[-1] / abs_coef.sum()
    print(f"  Final layer share of total |coef|: {final_share:.1%}")

    results[name] = {"coef": coef.tolist(), "abs_coef": abs_coef.tolist(),
                     "final_share": final_share, "id_auc": id_auc, "od_auc": od_auc}


# ── MLP weight analysis ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MLP first-layer weight analysis (input → hidden)")
print("=" * 60)

mlp_results = {}
for name, Xtr, Xte_id, Xte_od, n_layers in [
    ("ICR",     tr_icr, id_icr, od_icr, tr_icr.shape[1]),
    ("Entropy", tr_ent, id_ent, od_ent, tr_ent.shape[1]),
]:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xid_s = scaler.transform(Xte_id)
    Xod_s = scaler.transform(Xte_od)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    mlp.fit(Xtr_s, tr_lbl)

    id_auc = auroc(id_lbl, mlp.predict_proba(Xid_s)[:, 1])
    od_auc = auroc(od_lbl, mlp.predict_proba(Xod_s)[:, 1])

    # First layer weights: shape (n_input_features, n_hidden) = (n_layers, 64)
    W1 = mlp.coefs_[0]                    # (n_layers, 64)
    layer_importance = np.linalg.norm(W1, axis=1)  # L2 norm per input feature = per layer
    layer_importance_norm = layer_importance / layer_importance.sum()

    rank = np.argsort(-layer_importance)
    final_share = layer_importance[-1] / layer_importance.sum()

    print(f"\n{name} MLP  ID={id_auc:.4f}  OOD={od_auc:.4f}")
    print(f"  Top-5 layers by W1 L2 norm: {rank[:5].tolist()}  (final layer = {n_layers-1})")
    print(f"  Layer importance (L2 norm of input weights):")
    for l in range(n_layers):
        bar = "█" * int(layer_importance_norm[l] / layer_importance_norm.max() * 30)
        mark = " ◄ FINAL" if l == n_layers - 1 else ""
        print(f"    layer {l:>2}: {layer_importance_norm[l]:.4f}  {bar}{mark}")
    print(f"  Final layer share of total W1 L2 norm: {final_share:.1%}")

    mlp_results[name] = {"layer_importance": layer_importance.tolist(),
                         "final_share": final_share, "id_auc": id_auc, "od_auc": od_auc}


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for row, (name, key) in enumerate([("ICR", "ICR"), ("Entropy", "Entropy")]):
    n_layers = len(results[key]["abs_coef"])
    layers = list(range(n_layers))

    # LR coefficients
    ax = axes[row][0]
    abs_coef = results[key]["abs_coef"]
    colors = ["#f59e0b" if l == n_layers - 1 else "#3b82f6" for l in layers]
    ax.bar(layers, abs_coef, color=colors, alpha=0.85)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("|Coefficient|", fontsize=11)
    ax.set_title(f"{name} — LR |Coefficients| per Layer\n"
                 f"ID={results[key]['id_auc']:.4f}  OOD={results[key]['od_auc']:.4f}  "
                 f"Final layer share={results[key]['final_share']:.1%}",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    # Annotate final layer
    fl_val = abs_coef[-1]
    ax.annotate(f"layer {n_layers-1}\n({fl_val:.3f})",
                xy=(n_layers - 1, fl_val), xytext=(n_layers - 4, fl_val * 0.85),
                arrowprops=dict(arrowstyle="->", color="darkorange"),
                fontsize=9, color="darkorange", fontweight="bold")

    # MLP layer importance
    ax = axes[row][1]
    imp = mlp_results[key]["layer_importance"]
    imp_norm = np.array(imp) / sum(imp)
    colors = ["#f59e0b" if l == n_layers - 1 else "#8b5cf6" for l in layers]
    ax.bar(layers, imp_norm, color=colors, alpha=0.85)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("W1 L2 norm (normalized)", fontsize=11)
    ax.set_title(f"{name} — MLP First-Layer Weight Norm per Layer\n"
                 f"ID={mlp_results[key]['id_auc']:.4f}  OOD={mlp_results[key]['od_auc']:.4f}  "
                 f"Final layer share={mlp_results[key]['final_share']:.1%}",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fl_val = imp_norm[-1]
    ax.annotate(f"layer {n_layers-1}\n({fl_val:.3f})",
                xy=(n_layers - 1, fl_val), xytext=(n_layers - 4, fl_val * 0.85),
                arrowprops=dict(arrowstyle="->", color="darkorange"),
                fontsize=9, color="darkorange", fontweight="bold")

fig.suptitle("What Layers Do Supervised Models Actually Use?\n"
             "(Orange bar = final layer)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "weight_analysis.pdf", bbox_inches="tight", dpi=150)
fig.savefig(OUT / "weight_analysis.png", bbox_inches="tight", dpi=150)
print(f"\nFigure saved to {OUT}/weight_analysis.png")

with open(OUT / "weight_analysis.json", "w") as f:
    json.dump({"lr": results, "mlp": mlp_results}, f, indent=2)
