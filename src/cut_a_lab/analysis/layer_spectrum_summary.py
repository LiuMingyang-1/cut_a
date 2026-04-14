"""Helpers for summarizing the third-model layer-spectrum experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def has_significant_negative_pair(payload: dict[str, Any], *, splits: tuple[str, ...] = ("id_test", "ood_test")) -> bool:
    """Return True when the pair correlation is significantly negative on every requested split."""
    for split_name in splits:
        pair = payload["splits"][split_name]["pair_correlation"]
        if not (pair["pearson_r"] < 0.0 and pair["pearson_p"] < 0.05):
            return False
    return True


def has_location_shift_and_sign_flip(payload: dict[str, Any]) -> bool:
    """Return True when OOD best-|rho| moves and flips sign versus both train and ID."""
    train_best = payload["splits"]["train"]["best_abs_spearman_layer"]
    id_best = payload["splits"]["id_test"]["best_abs_spearman_layer"]
    ood_best = payload["splits"]["ood_test"]["best_abs_spearman_layer"]
    train_sign = _sign(train_best["spearman_rho_correctness"])
    id_sign = _sign(id_best["spearman_rho_correctness"])
    ood_sign = _sign(ood_best["spearman_rho_correctness"])

    return (
        ood_best["layer"] != train_best["layer"]
        and ood_best["layer"] != id_best["layer"]
        and ood_sign != 0
        and train_sign != 0
        and id_sign != 0
        and ood_sign != train_sign
        and ood_sign != id_sign
    )


def has_final_layer_dominance(payload: dict[str, Any]) -> bool:
    """Return True when the best effective AUROC layer is the final layer on all splits."""
    for split_payload in payload["splits"].values():
        if split_payload["best_effective_auroc_layer"]["layer"] != split_payload["n_layers"] - 1:
            return False
    return True


def build_third_model_summary_text(
    *,
    model_key: str,
    model_name_or_path: str,
    output_root: Path,
    sign_flip_payload: dict[str, Any],
    spectrum_payload: dict[str, Any],
    propagation_payload: dict[str, Any],
) -> str:
    """Build the yes/no summary requested for the third-model experiment."""
    q1 = has_significant_negative_pair(sign_flip_payload)
    q2 = has_location_shift_and_sign_flip(spectrum_payload)
    q3 = has_final_layer_dominance(spectrum_payload)

    train_best = spectrum_payload["splits"]["train"]["best_abs_spearman_layer"]
    id_best = spectrum_payload["splits"]["id_test"]["best_abs_spearman_layer"]
    ood_best = spectrum_payload["splits"]["ood_test"]["best_abs_spearman_layer"]

    lines = [
        "Third-model layer-spectrum summary",
        f"Model key: {model_key}",
        f"Model path/name: {model_name_or_path}",
        f"Output root: {output_root.resolve()}",
        "",
        f"Q1. Early/mid vs late item-level Pearson significantly negative? {'YES' if q1 else 'NO'}",
        (
            f"  ID L{sign_flip_payload['pair_layers'][0]}-L{sign_flip_payload['pair_layers'][1]}: "
            f"Pearson={sign_flip_payload['splits']['id_test']['pair_correlation']['pearson_r']:.4f} "
            f"(p={sign_flip_payload['splits']['id_test']['pair_correlation']['pearson_p']:.3e})"
        ),
        (
            f"  OOD L{sign_flip_payload['pair_layers'][0]}-L{sign_flip_payload['pair_layers'][1]}: "
            f"Pearson={sign_flip_payload['splits']['ood_test']['pair_correlation']['pearson_r']:.4f} "
            f"(p={sign_flip_payload['splits']['ood_test']['pair_correlation']['pearson_p']:.3e})"
        ),
        "",
        f"Q2. Strongest spectrum layer shifts and flips sign on OOD? {'YES' if q2 else 'NO'}",
        f"  Train best |rho|: L{train_best['layer']} ({train_best['spearman_rho_correctness']:+.4f})",
        f"  ID best |rho|:    L{id_best['layer']} ({id_best['spearman_rho_correctness']:+.4f})",
        f"  OOD best |rho|:   L{ood_best['layer']} ({ood_best['spearman_rho_correctness']:+.4f})",
        "",
        f"Q3. Final layer dominates on all splits? {'YES' if q3 else 'NO'}",
    ]

    for split_name, split_payload in spectrum_payload["splits"].items():
        best_eff = split_payload["best_effective_auroc_layer"]
        lines.append(
            f"  {split_name}: best effective AUROC = L{best_eff['layer']} ({best_eff['effective_error_auroc']:.4f})"
        )

    lines.extend(
        [
            "",
            "Propagation reference",
            (
                f"  ID: train-sign={propagation_payload['target_splits']['id_test']['train_sign_transfer_auc']:.4f} "
                f"prop={propagation_payload['target_splits']['id_test']['propagated_auc']:.4f} "
                f"oracle={propagation_payload['target_splits']['id_test']['oracle_weighted_auc']:.4f}"
            ),
            (
                f"  OOD: train-sign={propagation_payload['target_splits']['ood_test']['train_sign_transfer_auc']:.4f} "
                f"prop={propagation_payload['target_splits']['ood_test']['propagated_auc']:.4f} "
                f"oracle={propagation_payload['target_splits']['ood_test']['oracle_weighted_auc']:.4f}"
            ),
            "",
            "Existing figure entry points",
            f"  Sign flip: {output_root / 'sign_flip_validation' / 'layer_correlation_matrices.png'}",
            f"  Spectrum:  {output_root / 'feature_correctness_spectrum' / 'feature_correctness_spectrum.png'}",
            f"  Propagate: {output_root / 'neighbor_layer_propagation' / 'propagation_signs.png'}",
            "",
        ]
    )
    return "\n".join(lines)
