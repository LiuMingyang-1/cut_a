"""Tests for recipe feature views."""

from __future__ import annotations

import unittest

import numpy as np

from cut_a_lab.core.contracts import FeatureBlock, SpanRecord
from cut_a_lab.core.feature_views import build_feature_set_bundle
from cut_a_lab.recipes.base import FeatureSetSpec


def _records() -> tuple[SpanRecord, ...]:
    return (
        SpanRecord(sample_id="s1", span_id="s1:0", sample_label=1, silver_label=1),
        SpanRecord(sample_id="s2", span_id="s2:0", sample_label=0, silver_label=0),
    )


class FeatureViewTests(unittest.TestCase):
    def test_discrepancy_combined_builds_expected_shape(self) -> None:
        records = _records()
        icr_block = FeatureBlock(
            method_name="icr",
            level="span",
            feature_names=tuple(f"icr_{i}" for i in range(27)),
            features=np.tile(np.arange(27, dtype=np.float32), (2, 1)),
            records=records,
        )
        entropy_block = FeatureBlock(
            method_name="entropy",
            level="span",
            feature_names=tuple(f"ent_{i}" for i in range(28)),
            features=np.tile(np.arange(28, dtype=np.float32), (2, 1)),
            records=records,
        )

        feature_set = FeatureSetSpec(
            name="discrepancy_combined",
            methods=("icr", "entropy"),
            view_name="discrepancy_combined",
        )
        aligned_records, features, names = build_feature_set_bundle(feature_set, [icr_block, entropy_block])

        self.assertEqual(aligned_records, records)
        self.assertEqual(features.shape, (2, 14))
        self.assertEqual(len(names), 14)
        self.assertEqual(names[0], "icr_mean_early")
        self.assertEqual(names[-1], "ent_slope")


if __name__ == "__main__":
    unittest.main()
