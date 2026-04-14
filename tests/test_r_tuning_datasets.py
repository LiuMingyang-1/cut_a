"""Tests for local R-Tuning dataset normalization."""

from __future__ import annotations

import unittest
from pathlib import Path

from cut_a_lab.prep.r_tuning.contracts import NormalizedSample
from cut_a_lab.prep.r_tuning.datasets import (
    discover_available_dataset_splits,
    load_normalized_samples,
    subset_normalized_samples,
)


class RTuningDatasetTests(unittest.TestCase):
    @staticmethod
    def _make_sample(index: int) -> NormalizedSample:
        return NormalizedSample(
            dataset_name="demo",
            split_name="train",
            sample_id=f"demo:{index}",
            prompt_text=f"Question {index}\nAnswer:",
            expected_answer=str(index),
            task_type="qa_exact_match",
            metadata={},
        )

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

    def test_subset_normalized_samples_is_reproducible(self) -> None:
        samples = [self._make_sample(i) for i in range(12)]
        subset_a = subset_normalized_samples(
            samples,
            subset_size=5,
            subset_seed=7,
            subset_namespace="pararel:train",
        )
        subset_b = subset_normalized_samples(
            samples,
            subset_size=5,
            subset_seed=7,
            subset_namespace="pararel:train",
        )
        self.assertEqual([sample.sample_id for sample in subset_a], [sample.sample_id for sample in subset_b])

    def test_subset_normalized_samples_preserves_original_order(self) -> None:
        samples = [self._make_sample(i) for i in range(20)]
        subset = subset_normalized_samples(
            samples,
            subset_size=6,
            subset_seed=11,
            subset_namespace="pararel:ood_test",
        )
        subset_positions = [int(sample.sample_id.split(":")[-1]) for sample in subset]
        self.assertEqual(subset_positions, sorted(subset_positions))

    def test_subset_normalized_samples_validates_fraction(self) -> None:
        samples = [self._make_sample(i) for i in range(4)]
        with self.assertRaises(ValueError):
            subset_normalized_samples(samples, subset_fraction=0.0, subset_namespace="demo")


if __name__ == "__main__":
    unittest.main()
