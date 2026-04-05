"""CLI runner for validating whether disagreement yields combined-model gains."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate complementarity between ICR and entropy using combined models.")
    parser.add_argument(
        "--icr-oof",
        type=Path,
        default=Path("outputs/full_run/training/icr_only/torch/baseline_mlp.oof_predictions.jsonl"),
    )
    parser.add_argument(
        "--entropy-oof",
        type=Path,
        default=Path("outputs/full_run/training/entropy_only/torch/baseline_mlp.oof_predictions.jsonl"),
    )
    parser.add_argument(
        "--icr-entropy-oof",
        type=Path,
        default=Path("outputs/full_run/training/icr_entropy/torch/baseline_mlp.oof_predictions.jsonl"),
    )
    parser.add_argument(
        "--icr-delta-entropy-oof",
        type=Path,
        default=Path("outputs/full_run/training/icr_delta_entropy/torch/baseline_mlp.oof_predictions.jsonl"),
    )
    parser.add_argument(
        "--discrepancy-combined-oof",
        type=Path,
        default=Path("outputs/full_run/training/discrepancy_combined/torch/baseline_mlp.oof_predictions.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/full_run/complementarity_validation"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    from cut_a_lab.analysis.complementarity_validation import run_complementarity_validation

    run_complementarity_validation(
        icr_oof_path=args.icr_oof,
        entropy_oof_path=args.entropy_oof,
        target_model_paths={
            "icr_entropy": args.icr_entropy_oof,
            "icr_delta_entropy": args.icr_delta_entropy_oof,
            "discrepancy_combined": args.discrepancy_combined_oof,
        },
        output_dir=args.output_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
