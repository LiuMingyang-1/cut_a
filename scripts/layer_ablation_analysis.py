"""Four-experiment analysis for the training-free OOD-robustness paper.

Experiments:
  1. Layer ablation — training-free AUROC per layer (id vs ood)
  2. Training-free ECE comparison vs supervised
  3. Training-free combination (ICR final + Ent final weighted search)
  4. Supervised single-layer LR ablation (id vs ood per layer)

All data from: outputs/pararel_experiment/normalized_features.npz
Output: outputs/pararel_experiment/layer_ablation/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("outputs/pararel_experiment/layer_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
d = np.load("outputs/pararel_experiment/normalized_features.npz")

tr_icr = d["tr_icr_raw"]   # (5575, 28)
tr_ent = d["tr_ent_raw"]   # (5575, 29)
tr_lbl = d["tr_lbl"]       # (5575,)

id_icr = d["id_icr_raw"]   # (5584, 28)
id_ent = d["id_ent_raw"]   # (5584, 29)
id_lbl = d["id_lbl"]       # (5584,)

od_icr = d["od_icr_raw"]   # (13974, 28)
od_ent = d["od_ent_raw"]   # (13974, 29)
od_lbl = d["od_lbl"]       # (13974,)

print(f"Train: {tr_lbl.shape[0]} samples, pos={tr_lbl.mean():.3f}")
print(f"ID:    {id_lbl.shape[0]} samples, pos={id_lbl.mean():.3f}")
print(f"OOD:   {od_lbl.shape[0]} samples, pos={od_lbl.mean():.3f}")
print(f"ICR layers: {tr_icr.shape[1]}, Ent layers: {tr_ent.shape[1]}")


# ── helpers ───────────────────────────────────────────────────────────────────
def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = probs[mask].mean()
        ece_val += mask.mean() * abs(acc - conf)
    return float(ece_val)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def score_to_prob(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0,1]."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.full_like(scores, 0.5, dtype=np.float64)
    return (scores - lo) / (hi - lo)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Training-free layer ablation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Experiment 1: Training-free layer ablation")
print("="*60)

icr_n_layers = tr_icr.shape[1]   # 28
ent_n_layers = tr_ent.shape[1]   # 29

icr_results = []
for l in range(icr_n_layers):
    id_auc = auroc(id_lbl, -id_icr[:, l])
    od_auc = auroc(od_lbl, -od_icr[:, l])
    icr_results.append({"layer": l, "id_auroc": id_auc, "ood_auroc": od_auc, "delta": od_auc - id_auc})

ent_results = []
for l in range(ent_n_layers):
    id_auc = auroc(id_lbl, -id_ent[:, l])
    od_auc = auroc(od_lbl, -od_ent[:, l])
    ent_results.append({"layer": l, "id_auroc": id_auc, "ood_auroc": od_auc, "delta": od_auc - id_auc})

print("\nICR — per-layer training-free AUROC:")
print(f"{'Layer':>6}  {'ID':>7}  {'OOD':>7}  {'Δ(OOD-ID)':>10}")
for r in icr_results:
    marker = " ←" if r["layer"] == icr_n_layers - 1 else ""
    print(f"  {r['layer']:>4}  {r['id_auroc']:>7.4f}  {r['ood_auroc']:>7.4f}  {r['delta']:>+10.4f}{marker}")

print("\nEntropy — per-layer training-free AUROC:")
print(f"{'Layer':>6}  {'ID':>7}  {'OOD':>7}  {'Δ(OOD-ID)':>10}")
for r in ent_results:
    marker = " ←" if r["layer"] == ent_n_layers - 1 else ""
    print(f"  {r['layer']:>4}  {r['id_auroc']:>7.4f}  {r['ood_auroc']:>7.4f}  {r['delta']:>+10.4f}{marker}")

# Best layer per split
icr_best_id = max(icr_results, key=lambda r: r["id_auroc"])
icr_best_ood = max(icr_results, key=lambda r: r["ood_auroc"])
ent_best_id = max(ent_results, key=lambda r: r["id_auroc"])
ent_best_ood = max(ent_results, key=lambda r: r["ood_auroc"])

print(f"\nICR  best ID  layer: {icr_best_id['layer']} ({icr_best_id['id_auroc']:.4f})")
print(f"ICR  best OOD layer: {icr_best_ood['layer']} ({icr_best_ood['ood_auroc']:.4f})")
print(f"Ent  best ID  layer: {ent_best_id['layer']} ({ent_best_id['id_auroc']:.4f})")
print(f"Ent  best OOD layer: {ent_best_ood['layer']} ({ent_best_ood['ood_auroc']:.4f})")

# Save
with open(OUTPUT_DIR / "exp1_layer_ablation_tf.json", "w") as f:
    json.dump({"icr": icr_results, "entropy": ent_results}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Training-free ECE vs supervised ECE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Experiment 2: Training-free ECE vs supervised ECE")
print("="*60)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def supervised_lr(Xtr, ytr, Xte):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    lr.fit(Xtr_s, ytr)
    return lr.predict_proba(Xte_s)[:, 1]

ece_rows = []

# Training-free: ICR final layer
icr_id_scores = score_to_prob(-id_icr[:, -1])
icr_od_scores = score_to_prob(-od_icr[:, -1])
ent_id_scores = score_to_prob(-id_ent[:, -1])
ent_od_scores = score_to_prob(-od_ent[:, -1])

for name, id_probs, od_probs in [
    ("TF-ICR-final",  icr_id_scores, icr_od_scores),
    ("TF-Ent-final",  ent_id_scores, ent_od_scores),
]:
    id_auc = auroc(id_lbl, id_probs)
    od_auc = auroc(od_lbl, od_probs)
    id_ece = ece(id_lbl, id_probs)
    od_ece = ece(od_lbl, od_probs)
    row = {"method": name, "id_auroc": id_auc, "ood_auroc": od_auc,
           "id_ece": id_ece, "ood_ece": od_ece,
           "delta_auroc": od_auc - id_auc, "delta_ece": od_ece - id_ece}
    ece_rows.append(row)
    print(f"{name:20s}  ID AUROC={id_auc:.4f} ECE={id_ece:.4f}  OOD AUROC={od_auc:.4f} ECE={od_ece:.4f}")

# Supervised: all-layer LR
print("  (fitting supervised LR all-layers…)")
for name, Xtr, Xte_id, Xte_od in [
    ("Sup-LR-ICR-all",     tr_icr, id_icr, od_icr),
    ("Sup-LR-Ent-all",     tr_ent, id_ent, od_ent),
    ("Sup-LR-ICR+Ent-all", np.hstack([tr_icr, tr_ent]), np.hstack([id_icr, id_ent]), np.hstack([od_icr, od_ent])),
]:
    id_probs = supervised_lr(Xtr, tr_lbl, Xte_id)
    od_probs = supervised_lr(Xtr, tr_lbl, Xte_od)
    id_auc = auroc(id_lbl, id_probs)
    od_auc = auroc(od_lbl, od_probs)
    id_ece = ece(id_lbl, id_probs)
    od_ece = ece(od_lbl, od_probs)
    row = {"method": name, "id_auroc": id_auc, "ood_auroc": od_auc,
           "id_ece": id_ece, "ood_ece": od_ece,
           "delta_auroc": od_auc - id_auc, "delta_ece": od_ece - id_ece}
    ece_rows.append(row)
    print(f"{name:20s}  ID AUROC={id_auc:.4f} ECE={id_ece:.4f}  OOD AUROC={od_auc:.4f} ECE={od_ece:.4f}")

with open(OUTPUT_DIR / "exp2_ece_comparison.json", "w") as f:
    json.dump(ece_rows, f, indent=2)

print("\nSummary: Δ(OOD-ID) AUROC and ECE")
print(f"{'Method':30s}  {'ΔAUROC':>8}  {'ΔECE':>8}")
for r in ece_rows:
    print(f"  {r['method']:28s}  {r['delta_auroc']:>+8.4f}  {r['delta_ece']:>+8.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Training-free combination
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Experiment 3: Training-free combination (ICR + Entropy, final layer)")
print("="*60)

# Grid search alpha in [0,1] where combined = alpha * (-icr_final) + (1-alpha) * (-ent_final)
# Use ID AUROC to pick alpha; report OOD AUROC

def normalize_01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)

id_icr_f = normalize_01(-id_icr[:, -1])
id_ent_f = normalize_01(-id_ent[:, -1])
od_icr_f = normalize_01(-od_icr[:, -1])
od_ent_f = normalize_01(-od_ent[:, -1])

alphas = np.linspace(0, 1, 21)
combo_rows = []
for alpha in alphas:
    id_combined = alpha * id_icr_f + (1 - alpha) * id_ent_f
    od_combined = alpha * od_icr_f + (1 - alpha) * od_ent_f
    combo_rows.append({
        "alpha_icr": float(alpha),
        "id_auroc": auroc(id_lbl, id_combined),
        "ood_auroc": auroc(od_lbl, od_combined),
    })

best_id = max(combo_rows, key=lambda r: r["id_auroc"])
best_ood = max(combo_rows, key=lambda r: r["ood_auroc"])

print(f"{'alpha_icr':>10}  {'ID AUROC':>10}  {'OOD AUROC':>10}")
for r in combo_rows:
    print(f"  {r['alpha_icr']:>8.2f}  {r['id_auroc']:>10.4f}  {r['ood_auroc']:>10.4f}")

print(f"\nBest for ID:  alpha_icr={best_id['alpha_icr']:.2f}  ID={best_id['id_auroc']:.4f}  OOD={best_id['ood_auroc']:.4f}")
print(f"Best for OOD: alpha_icr={best_ood['alpha_icr']:.2f}  OOD={best_ood['ood_auroc']:.4f}  ID={best_ood['id_auroc']:.4f}")

# Baseline singles
print(f"\nBaseline ICR-only (alpha=1): ID={combo_rows[20]['id_auroc']:.4f} OOD={combo_rows[20]['ood_auroc']:.4f}")
print(f"Baseline Ent-only (alpha=0): ID={combo_rows[0]['id_auroc']:.4f} OOD={combo_rows[0]['ood_auroc']:.4f}")

with open(OUTPUT_DIR / "exp3_combination.json", "w") as f:
    json.dump(combo_rows, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 4: Supervised single-layer LR ablation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Experiment 4: Supervised single-layer LR ablation")
print("="*60)

icr_sup_results = []
for l in range(icr_n_layers):
    Xtr = tr_icr[:, l:l+1]
    id_probs = supervised_lr(Xtr, tr_lbl, id_icr[:, l:l+1])
    od_probs = supervised_lr(Xtr, tr_lbl, od_icr[:, l:l+1])
    id_auc = auroc(id_lbl, id_probs)
    od_auc = auroc(od_lbl, od_probs)
    icr_sup_results.append({"layer": l, "id_auroc": id_auc, "ood_auroc": od_auc, "delta": od_auc - id_auc})

ent_sup_results = []
for l in range(ent_n_layers):
    Xtr = tr_ent[:, l:l+1]
    id_probs = supervised_lr(Xtr, tr_lbl, id_ent[:, l:l+1])
    od_probs = supervised_lr(Xtr, tr_lbl, od_ent[:, l:l+1])
    id_auc = auroc(id_lbl, id_probs)
    od_auc = auroc(od_lbl, od_probs)
    ent_sup_results.append({"layer": l, "id_auroc": id_auc, "ood_auroc": od_auc, "delta": od_auc - id_auc})

print("\nICR — supervised single-layer LR AUROC:")
print(f"{'Layer':>6}  {'ID':>7}  {'OOD':>7}  {'Δ(OOD-ID)':>10}  {'TF-OOD':>8}  {'Sup-TF gap':>10}")
for sup, tf in zip(icr_sup_results, icr_results):
    marker = " ←" if sup["layer"] == icr_n_layers - 1 else ""
    gap = sup["ood_auroc"] - tf["ood_auroc"]
    print(f"  {sup['layer']:>4}  {sup['id_auroc']:>7.4f}  {sup['ood_auroc']:>7.4f}  {sup['delta']:>+10.4f}  {tf['ood_auroc']:>8.4f}  {gap:>+10.4f}{marker}")

print("\nEntropy — supervised single-layer LR AUROC:")
print(f"{'Layer':>6}  {'ID':>7}  {'OOD':>7}  {'Δ(OOD-ID)':>10}  {'TF-OOD':>8}  {'Sup-TF gap':>10}")
for sup, tf in zip(ent_sup_results, ent_results):
    marker = " ←" if sup["layer"] == ent_n_layers - 1 else ""
    gap = sup["ood_auroc"] - tf["ood_auroc"]
    print(f"  {sup['layer']:>4}  {sup['id_auroc']:>7.4f}  {sup['ood_auroc']:>7.4f}  {sup['delta']:>+10.4f}  {tf['ood_auroc']:>8.4f}  {gap:>+10.4f}{marker}")

with open(OUTPUT_DIR / "exp4_supervised_layer_ablation.json", "w") as f:
    json.dump({"icr": icr_sup_results, "entropy": ent_sup_results}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Final summary table
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

def fmt(r):
    return f"ID={r['id_auroc']:.4f}  OOD={r['ood_auroc']:.4f}  Δ={r['delta']:+.4f}"

print(f"\nTraining-free final layer:")
print(f"  ICR-final:  {fmt(icr_results[-1])}")
print(f"  Ent-final:  {fmt(ent_results[-1])}")

print(f"\nTraining-free best layer (by OOD):")
print(f"  ICR best:   {fmt(icr_best_ood)}")
print(f"  Ent best:   {fmt(ent_best_ood)}")

print(f"\nSupervised all-layer LR:")
for r in ece_rows:
    if r["method"].startswith("Sup"):
        print(f"  {r['method']:28s}  ID={r['id_auroc']:.4f}  OOD={r['ood_auroc']:.4f}  Δ={r['delta_auroc']:+.4f}")

print(f"\nSupervised single-layer LR (final layer):")
icr_fl = icr_sup_results[-1]
ent_fl = ent_sup_results[-1]
print(f"  ICR layer-{icr_fl['layer']}:  {fmt(icr_fl)}")
print(f"  Ent layer-{ent_fl['layer']}:  {fmt(ent_fl)}")

print(f"\nBest TF combination: alpha_icr={best_ood['alpha_icr']:.2f}  OOD={best_ood['ood_auroc']:.4f}")
print(f"\nResults saved to: {OUTPUT_DIR}")
