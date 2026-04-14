"""Tests for self-consistency experiment helpers."""

from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from cut_a_lab.analysis.self_consistency import (
    align_sample_metric,
    extract_first_non_empty_line,
    majority_vote,
    normalize_answer_key,
    sanitize_correlation_values,
)
from scripts.experiments.self_consistency_sign import _build_summary_text, _normalize_split_config


class SelfConsistencyHelperTests(unittest.TestCase):
    def test_extract_first_non_empty_line_skips_blanks(self) -> None:
        self.assertEqual(extract_first_non_empty_line("\n\nAnswer\nReason"), "Answer")

    def test_normalize_answer_key_uses_first_line_and_whitespace(self) -> None:
        self.assertEqual(normalize_answer_key("  New   York \nBecause..."), "new york")

    def test_majority_vote_uses_first_occurrence_tie_break(self) -> None:
        result = majority_vote(["B", "A", "a", "b"])
        self.assertEqual(result.majority_key, "b")
        self.assertEqual(result.majority_count, 2)
        self.assertEqual(result.tie_count, 2)
        self.assertAlmostEqual(result.agreement_rate, 0.5)

    def test_sanitize_correlation_values_replaces_nan(self) -> None:
        values = sanitize_correlation_values(np.asarray([0.1, np.nan, -0.2]))
        np.testing.assert_allclose(values, np.asarray([0.1, 0.0, -0.2]))

    def test_align_sample_metric_raises_on_missing_sample(self) -> None:
        with self.assertRaises(KeyError):
            align_sample_metric(["a", "b"], {"a": 1.0}, name="agreement_rate")

    def test_normalize_split_config_validates_eval_subset(self) -> None:
        args = Namespace(splits=["train", "custom"], eval_splits=["missing"], train_split="train")
        with self.assertRaises(ValueError):
            _normalize_split_config(args)

    def test_build_summary_text_supports_custom_eval_splits(self) -> None:
        config = Namespace(
            corr_method="spearman",
            k=3,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=16,
            label_source="auto_sample",
            splits=["train", "custom_eval"],
            eval_splits=["custom_eval"],
        )
        split_results = {
            "train": {
                "n_samples": 2,
                "agreement_mean": 0.75,
                "agreement_std": 0.25,
                "best_pseudo_abs_layer": 0,
                "best_true_abs_layer": 1,
                "pseudo_rho": [0.1, -0.2],
                "true_rho": [0.2, -0.3],
                "pseudo_sign_accuracy_vs_true": 1.0,
                "oracle_weighted_auc": float("nan"),
                "train_sign_transfer_auc": float("nan"),
                "pseudo_train_transfer_auc": float("nan"),
                "pseudo_on_split_auc": float("nan"),
                "best_single_effective_auc": 0.6,
            },
            "custom_eval": {
                "n_samples": 2,
                "agreement_mean": 0.5,
                "agreement_std": 0.0,
                "best_pseudo_abs_layer": 0,
                "best_true_abs_layer": 1,
                "pseudo_rho": [0.1, -0.2],
                "true_rho": [0.2, -0.3],
                "pseudo_sign_accuracy_vs_true": 0.5,
                "oracle_weighted_auc": 0.7,
                "train_sign_transfer_auc": 0.6,
                "pseudo_train_transfer_auc": 0.65,
                "pseudo_on_split_auc": 0.66,
                "best_single_effective_auc": 0.68,
            },
        }
        text = _build_summary_text(
            config=config,
            output_dir=Path("."),
            propagation_baseline=None,
            train_pseudo_signs=np.asarray([1.0, -1.0]),
            train_true_signs=np.asarray([1.0, -1.0]),
            split_results=split_results,
        )
        self.assertIn("custom_eval", text)
        self.assertIn("pseudo-sign (on-split infer)", text)


if __name__ == "__main__":
    unittest.main()
