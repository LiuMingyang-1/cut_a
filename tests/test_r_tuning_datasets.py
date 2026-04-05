"""Tests for local R-Tuning dataset normalization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cut_a_lab.prep.r_tuning.datasets import discover_available_dataset_splits, load_normalized_samples


class RTuningDatasetTests(unittest.TestCase):
    def test_discover_available_dataset_splits_uses_local_files(self) -> None:
        available = discover_available_dataset_splits(Path("data/R-Tuning-data"))
        names = {(spec.dataset_name, spec.split_name) for spec in available}
        self.assertIn(("HaluEvalQA", "default"), names)
        self.assertIn(("HotpotQA", "train"), names)
        self.assertIn(("FEVER", "train"), names)

    def test_load_normalized_samples_keeps_expected_fields(self) -> None:
        available = discover_available_dataset_splits(Path("data/R-Tuning-data"))
        hotpot_spec = next(spec for spec in available if spec.dataset_name == "HotpotQA" and spec.split_name == "train")
        samples = load_normalized_samples(root_dir=Path("data/R-Tuning-data"), spec=hotpot_spec)
        self.assertTrue(samples)
        sample = samples[0]
        self.assertEqual(sample.dataset_name, "HotpotQA")
        self.assertEqual(sample.split_name, "train")
        self.assertEqual(sample.task_type, "qa_exact_match")
        self.assertIn("Question:", sample.prompt_text)
        self.assertTrue(sample.expected_answer)


if __name__ == "__main__":
    unittest.main()
