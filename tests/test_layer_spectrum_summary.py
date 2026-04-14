"""Tests for third-model layer-spectrum summary helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from cut_a_lab.analysis.layer_spectrum_summary import (
    build_third_model_summary_text,
    has_final_layer_dominance,
    has_location_shift_and_sign_flip,
    has_significant_negative_pair,
)


def _sign_flip_payload() -> dict:
    return {
        "pair_layers": [13, 27],
        "splits": {
            "id_test": {"pair_correlation": {"pearson_r": -0.4, "pearson_p": 1e-6}},
            "ood_test": {"pair_correlation": {"pearson_r": -0.2, "pearson_p": 1e-5}},
        },
    }


def _spectrum_payload(*, final_layer: int = 31) -> dict:
    return {
        "splits": {
            "train": {
                "n_layers": 32,
                "best_abs_spearman_layer": {"layer": 15, "spearman_rho_correctness": -0.30},
                "best_effective_auroc_layer": {"layer": final_layer, "effective_error_auroc": 0.68},
            },
            "id_test": {
                "n_layers": 32,
                "best_abs_spearman_layer": {"layer": 15, "spearman_rho_correctness": -0.33},
                "best_effective_auroc_layer": {"layer": final_layer, "effective_error_auroc": 0.69},
            },
            "ood_test": {
                "n_layers": 32,
                "best_abs_spearman_layer": {"layer": 29, "spearman_rho_correctness": 0.17},
                "best_effective_auroc_layer": {"layer": final_layer, "effective_error_auroc": 0.61},
            },
        }
    }


def _propagation_payload() -> dict:
    return {
        "target_splits": {
            "id_test": {
                "train_sign_transfer_auc": 0.68,
                "propagated_auc": 0.65,
                "oracle_weighted_auc": 0.69,
            },
            "ood_test": {
                "train_sign_transfer_auc": 0.54,
                "propagated_auc": 0.57,
                "oracle_weighted_auc": 0.61,
            },
        }
    }


class LayerSpectrumSummaryTests(unittest.TestCase):
    def test_has_significant_negative_pair_requires_all_splits(self) -> None:
        self.assertTrue(has_significant_negative_pair(_sign_flip_payload()))

    def test_has_location_shift_and_sign_flip_detects_llama_like_pattern(self) -> None:
        self.assertTrue(has_location_shift_and_sign_flip(_spectrum_payload()))

    def test_has_location_shift_and_sign_flip_rejects_zero_ood_sign(self) -> None:
        payload = _spectrum_payload()
        payload["splits"]["ood_test"]["best_abs_spearman_layer"]["spearman_rho_correctness"] = 0.0
        self.assertFalse(has_location_shift_and_sign_flip(payload))

    def test_has_final_layer_dominance_requires_all_splits(self) -> None:
        self.assertTrue(has_final_layer_dominance(_spectrum_payload()))
        self.assertFalse(has_final_layer_dominance(_spectrum_payload(final_layer=29)))

    def test_build_summary_text_contains_yes_no_answers(self) -> None:
        text = build_third_model_summary_text(
            model_key="mistral-7b-instruct",
            model_name_or_path="/models/mistral",
            output_root=Path("outputs/experiments/mistral-7b-instruct"),
            sign_flip_payload=_sign_flip_payload(),
            spectrum_payload=_spectrum_payload(),
            propagation_payload=_propagation_payload(),
        )
        self.assertIn("Q1. Early/mid vs late item-level Pearson significantly negative? YES", text)
        self.assertIn("Q2. Strongest spectrum layer shifts and flips sign on OOD? YES", text)
        self.assertIn("Q3. Final layer dominates on all splits? YES", text)


if __name__ == "__main__":
    unittest.main()
