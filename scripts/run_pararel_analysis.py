"""Run three-layer A/B analysis on pararel transfer-eval predictions.

Runs for both id_test and ood_test splits:
  1. Disagreement analysis (four-quadrant ICR vs entropy)
  2. Fine-grained disagreement analysis (layer-wise, sliding window, temporal)
  3. Complementarity validation (rescue / regression / net gain)

Output root: outputs/pararel_experiment/analysis/
"""

from __future__ import annotations

import argparse
from pathlib import Path


TEST_SPLITS = ("id_test", "ood_test")
MLP_MODEL = "baseline_mlp"


def _predictions_path(training_root: Path, feature_set: str, test_split: str) -> Path:
    return training_root / feature_set / test_split / f"{MLP_MODEL}.predictions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("outputs/pararel_experiment/training"),
    )
    parser.add_argument(
        "--method-input-root",
        type=Path,
        default=Path("outputs/pararel_experiment/method_inputs"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/pararel_experiment/analysis"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    from cut_a_lab.analysis.disagreement_analysis import run_disagreement_analysis
    from cut_a_lab.analysis.disagreement_analysis_finegrained import run_finegrained_disagreement_analysis
    from cut_a_lab.analysis.complementarity_validation import run_complementarity_validation

    for test_split in TEST_SPLITS:
        icr_path = _predictions_path(args.training_root, "icr_only", test_split)
        entropy_path = _predictions_path(args.training_root, "entropy_only", test_split)
        joint_path = _predictions_path(args.training_root, "icr_entropy", test_split)
        combined_data_path = args.method_input_root / "pararel" / test_split / "combined_spans.jsonl"

        missing = [p for p in (icr_path, entropy_path, joint_path, combined_data_path) if not p.exists()]
        if missing:
            print(f"[skip {test_split}] missing files:")
            for p in missing:
                print(f"  {p}")
            continue

        print(f"\n{'='*60}")
        print(f"Running analysis for: {test_split}")
        print(f"{'='*60}")

        split_output = args.output_root / test_split

        # Layer 1: disagreement analysis
        print("\n--- Layer 1: Disagreement analysis ---")
        run_disagreement_analysis(
            icr_oof_path=icr_path,
            entropy_oof_path=entropy_path,
            input_data_path=combined_data_path,
            output_dir=split_output / "disagreement",
            threshold=args.threshold,
        )

        # Layer 2: fine-grained disagreement analysis
        print("\n--- Layer 2: Fine-grained disagreement analysis ---")
        run_finegrained_disagreement_analysis(
            icr_oof_path=icr_path,
            entropy_oof_path=entropy_path,
            input_data_path=combined_data_path,
            output_dir=split_output / "disagreement_finegrained",
            threshold=args.threshold,
        )

        # Layer 3: complementarity validation
        print("\n--- Layer 3: Complementarity validation ---")
        run_complementarity_validation(
            icr_oof_path=icr_path,
            entropy_oof_path=entropy_path,
            target_model_paths={"icr_entropy": joint_path},
            output_dir=split_output / "complementarity",
            threshold=args.threshold,
        )

        print(f"\n[done] {test_split} → {split_output}")


if __name__ == "__main__":
    main()
