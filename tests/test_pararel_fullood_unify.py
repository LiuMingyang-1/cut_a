"""Tests for ParaRel full-OOD unification helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.experiments.unify_pararel_full_ood import (
    ModelMetrics,
    _build_main_table,
    _build_report_markdown,
    _parse_split_sizes_from_agreement_summary,
    _parse_split_sizes_from_self_summary,
    _parse_two_column_method_table,
)


class PararelFullOODUnifyTests(unittest.TestCase):
    def test_parse_two_column_method_table(self) -> None:
        text = (
            "method                            id_test     ood_test\n"
            "train-label sign (oracle)          0.6902       0.6231\n"
            "propagation (minimal)                 nan          nan\n"
            "agreement rate (direct)            0.6900       0.7369\n"
            "\n"
        )
        rows = _parse_two_column_method_table(text)
        self.assertEqual(rows["train-label sign (oracle)"], (0.6902, 0.6231))
        self.assertTrue(str(rows["propagation (minimal)"][0]) == "nan")
        self.assertEqual(rows["agreement rate (direct)"], (0.69, 0.7369))

    def test_parse_split_sizes(self) -> None:
        self_summary = (
            "[train] n=5575 agreement_mean=0.45\n"
            "[id_test] n=5584 agreement_mean=0.44\n"
            "[ood_test] n=13974 agreement_mean=0.62\n"
        )
        agreement_summary = (
            "Agreement-rate AUROC vs true correctness\n"
            "  train    AUROC=0.6981 (n=5575, mean=0.4558, std=0.2948)\n"
            "  id_test  AUROC=0.6900 (n=5584, mean=0.4490, std=0.2924)\n"
            "  ood_test AUROC=0.7369 (n=13974, mean=0.6217, std=0.3101)\n"
        )
        self_sizes = _parse_split_sizes_from_self_summary(self_summary)
        agreement_sizes = _parse_split_sizes_from_agreement_summary(agreement_summary)
        self.assertEqual(self_sizes["ood_test"], 13974)
        self.assertEqual(agreement_sizes["ood_test"], 13974)

    def test_build_report_contains_required_sections(self) -> None:
        methods = {
            "train-label sign (oracle)": (0.60, 0.61),
            "train-sign transfer": (0.62, 0.63),
            "pseudo-sign (train infer)": (0.64, 0.65),
            "pseudo-sign (on-split infer)": (0.66, 0.67),
            "agreement rate (direct)": (0.68, 0.69),
        }
        llama = ModelMetrics(
            model_name="llama-3.1-8b-instruct",
            root=Path("outputs/experiments/llama-3.1-8b-instruct-fullood"),
            ood_n_samples_manifest=13974,
            ood_n_samples_method_input=13974,
            ood_n_samples_self_consistency=13974,
            ood_n_samples_agreement=13974,
            methods=methods,
            feature_spectrum_ood_n=13974,
            sign_flip_ood_n=13974,
        )
        mistral = ModelMetrics(
            model_name="mistral-7b-instruct",
            root=Path("outputs/experiments/mistral-7b-instruct"),
            ood_n_samples_manifest=13974,
            ood_n_samples_method_input=13974,
            ood_n_samples_self_consistency=13974,
            ood_n_samples_agreement=13974,
            methods=methods,
            feature_spectrum_ood_n=13974,
            sign_flip_ood_n=13974,
        )
        table = _build_main_table([llama, mistral])
        self.assertIn("agreement rate (direct)", table)
        self.assertIn("13974", table)

        markdown = _build_report_markdown(
            expected_counts={"train": 5575, "id_test": 5584, "ood_test": 13974},
            llama_metrics=llama,
            mistral_metrics=mistral,
            llama_source_root=Path("outputs/experiments/llama-3.1-8b-instruct"),
            llama_target_root=llama.root,
            mistral_source_root=mistral.root,
            mistral_target_root=mistral.root,
            ood5584_root=Path("outputs/experiments/mistral-7b-instruct-ood5584"),
        )
        self.assertIn("Main table (full OOD)", markdown)
        self.assertIn("mistral-7b-instruct-ood5584", markdown)
        self.assertIn("论文主表引用路径", markdown)


if __name__ == "__main__":
    unittest.main()
