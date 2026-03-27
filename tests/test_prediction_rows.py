"""Tests for prediction-row serialization."""

from __future__ import annotations

import unittest

from cut_a_lab.core.contracts import INPUT_METADATA_KEY, SpanRecord


class PredictionRowTests(unittest.TestCase):
    def test_prediction_row_preserves_non_conflicting_metadata(self) -> None:
        record = SpanRecord(
            sample_id="s1",
            span_id="s1:0",
            sample_label=1,
            silver_label=1,
            metadata={"route": "window", "span_type": "window"},
        )

        payload = record.to_prediction_row(
            feature_set="icr_only",
            family="sklearn",
            model="logistic_regression",
            fold=2,
            probability=0.9,
        )

        self.assertEqual(payload["route"], "window")
        self.assertEqual(payload["span_type"], "window")
        self.assertNotIn(INPUT_METADATA_KEY, payload)

    def test_prediction_row_namespaces_conflicting_metadata(self) -> None:
        record = SpanRecord(
            sample_id="s1",
            span_id="s1:0",
            sample_label=1,
            silver_label=1,
            metadata={"model": "upstream", "fold": 99, "probability": 0.1},
        )

        payload = record.to_prediction_row(
            feature_set="icr_only",
            family="sklearn",
            model="logistic_regression",
            fold=2,
            probability=0.9,
        )

        self.assertEqual(payload["model"], "logistic_regression")
        self.assertEqual(payload["fold"], 2)
        self.assertEqual(payload["probability"], 0.9)
        self.assertEqual(
            payload[INPUT_METADATA_KEY],
            {"model": "upstream", "fold": 99, "probability": 0.1},
        )


if __name__ == "__main__":
    unittest.main()
