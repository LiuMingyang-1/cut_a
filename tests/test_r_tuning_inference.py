"""Tests for R-Tuning inference labeling logic."""

from __future__ import annotations

import unittest

from cut_a_lab.prep.r_tuning.contracts import NormalizedSample
from cut_a_lab.prep.r_tuning.inference import classify_generation


def _make_sample(
    *,
    task_type: str,
    expected_answer: str,
    choices: tuple[str, ...] = (),
) -> NormalizedSample:
    return NormalizedSample(
        dataset_name="demo",
        split_name="train",
        sample_id="demo:0",
        prompt_text="Question: demo\nAnswer:",
        expected_answer=expected_answer,
        task_type=task_type,
        choices=choices,
        metadata={},
    )


class RTuningInferenceTests(unittest.TestCase):
    def test_classification_label_uses_first_non_empty_line(self) -> None:
        sample = _make_sample(
            task_type="classification_label",
            expected_answer="SUPPORTS",
            choices=("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"),
        )
        sample_label, silver_label = classify_generation(sample, "\nSUPPORTS\nReasoning: claim is supported.")
        self.assertEqual(sample_label, 0)
        self.assertEqual(silver_label, 0)

    def test_classification_label_unknown_label_is_wrong(self) -> None:
        sample = _make_sample(
            task_type="classification_label",
            expected_answer="SUPPORTS",
            choices=("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"),
        )
        sample_label, silver_label = classify_generation(sample, "MAYBE\nReasoning: uncertain.")
        self.assertEqual(sample_label, 1)
        self.assertEqual(silver_label, 1)

    def test_qa_exact_match_allows_expected_substring_on_first_line(self) -> None:
        sample = _make_sample(task_type="qa_exact_match", expected_answer="yes")
        sample_label, silver_label = classify_generation(sample, "Yes, both are American.\nReasoning: ...")
        self.assertEqual(sample_label, 0)
        self.assertEqual(silver_label, 0)

    def test_qa_exact_match_marks_incorrect_answer_wrong(self) -> None:
        sample = _make_sample(task_type="qa_exact_match", expected_answer="Delhi")
        sample_label, silver_label = classify_generation(sample, "Mumbai\nReasoning: ...")
        self.assertEqual(sample_label, 1)
        self.assertEqual(silver_label, 1)

    def test_multiple_choice_letter_keeps_first_character_behavior(self) -> None:
        sample = _make_sample(task_type="multiple_choice_letter", expected_answer="B")
        sample_label, silver_label = classify_generation(sample, "B because option B is best.")
        self.assertEqual(sample_label, 0)
        self.assertEqual(silver_label, 0)


if __name__ == "__main__":
    unittest.main()
