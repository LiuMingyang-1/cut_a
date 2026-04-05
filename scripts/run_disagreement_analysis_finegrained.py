"""CLI runner for fine-grained ICR vs entropy span-level disagreement analysis."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-grained span-level disagreement analysis between ICR and entropy models.")
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
        default=Path("outputs/full_run/disagreement_analysis_finegrained"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--window-stride", type=int, default=1)
    args = parser.parse_args()

    from cut_a_lab.analysis.disagreement_analysis_finegrained import run_disagreement_analysis_finegrained

    run_disagreement_analysis_finegrained(
        icr_oof_path=args.icr_oof,
        entropy_oof_path=args.entropy_oof,
        input_data_path=args.input_data,
        output_dir=args.output_dir,
        threshold=args.threshold,
        window_size=args.window_size,
        window_stride=args.window_stride,
    )


if __name__ == "__main__":
    main()
