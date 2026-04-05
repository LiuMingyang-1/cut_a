"""CLI runner for ICR vs entropy span-level disagreement analysis."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Span-level disagreement analysis between ICR and entropy models.")
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
        "--input-data",
        type=Path,
        default=Path("data/combined/tokenizer_windows_mean.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/full_run/disagreement_analysis"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    from cut_a_lab.analysis.disagreement_analysis import run_disagreement_analysis

    run_disagreement_analysis(
        icr_oof_path=args.icr_oof,
        entropy_oof_path=args.entropy_oof,
        input_data_path=args.input_data,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
