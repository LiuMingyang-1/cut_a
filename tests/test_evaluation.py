"""Small smoke tests for evaluation helpers."""

from __future__ import annotations

import unittest

from cut_a_lab.core.evaluation import aggregate_probabilities, evaluate_binary_predictions


class EvaluationTests(unittest.TestCase):
    def test_aggregate_probabilities_max(self) -> None:
        self.assertAlmostEqual(aggregate_probabilities([0.1, 0.6, 0.2], "max"), 0.6)

    def test_aggregate_probabilities_topk_mean(self) -> None:
        self.assertAlmostEqual(aggregate_probabilities([0.1, 0.6, 0.2], "topk_mean", top_k=2), 0.4)

    def test_evaluate_binary_predictions_returns_expected_keys(self) -> None:
        metrics = evaluate_binary_predictions([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        for key in ("AUROC", "AUPRC", "F1", "Accuracy", "Threshold"):
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
