"""Tests for multi-dataset reporting helpers."""

from __future__ import annotations

import unittest

from cut_a_lab.prep.r_tuning.reporting import build_best_model_table


class RTuningReportingTests(unittest.TestCase):
    def test_build_best_model_table_contains_dataset_rows(self) -> None:
        table = build_best_model_table(
            [
                {
                    "dataset_name": "HaluEvalQA",
                    "split_name": "default",
                    "best_model": {
                        "feature_set": "icr_only",
                        "family_group": "torch",
                        "model": "baseline_mlp",
                        "sample_auroc": 0.8123,
                        "span_auroc": 0.7456,
                    },
                }
            ]
        )
        self.assertIn("HaluEvalQA", table)
        self.assertIn("icr_only", table)
        self.assertIn("torch/baseline_mlp", table)


if __name__ == "__main__":
    unittest.main()
