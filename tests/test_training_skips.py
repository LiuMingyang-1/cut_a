"""Tests for training skip behavior on degenerate labels."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cut_a_lab.core.contracts import SpanRecord
from cut_a_lab.core.training import train_with_features
from cut_a_lab.models.torch_models import build_torch_model_factories


class TrainingSkipTests(unittest.TestCase):
    def test_train_with_features_skips_single_class_labels(self) -> None:
        records = (
            SpanRecord(sample_id="s1", span_id="s1:0", sample_label=1, silver_label=1),
            SpanRecord(sample_id="s2", span_id="s2:0", sample_label=1, silver_label=1),
        )
        features = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            results = train_with_features(
                feature_set_name="icr_only",
                method_names=("icr",),
                features=features,
                feature_names=["f0", "f1"],
                records=records,
                model_factories=build_torch_model_factories(),
                output_dir=Path(tmp_dir),
                family_name="torch",
                n_splits=2,
                seed=42,
                device="cpu",
            )

        payload = results["baseline_mlp"]
        self.assertEqual(payload["status"], "skipped")
        self.assertIn("Need at least two labeled classes", payload["skip_reason"])


if __name__ == "__main__":
    unittest.main()
