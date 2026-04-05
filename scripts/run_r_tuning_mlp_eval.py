"""Run MLP-only evaluation on prepared R-Tuning method inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from cut_a_lab.core.io import dump_json
from cut_a_lab.core.registry import get_recipe
from cut_a_lab.core.training import run_recipe
from cut_a_lab.prep.r_tuning.datasets import discover_available_dataset_splits
from cut_a_lab.prep.r_tuning.reporting import build_best_model_table


def _matches_filters(dataset_name: str, split_name: str, *, datasets: set[str], splits: set[str]) -> bool:
    if datasets and dataset_name not in datasets:
        return False
    if splits and split_name not in splits:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/R-Tuning-data"))
    parser.add_argument("--method-input-root", type=Path, default=Path("outputs/r_tuning/method_inputs"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/r_tuning/training"))
    parser.add_argument("--recipe", default="cut_a_default")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip dataset splits whose method-input directory is missing.",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Repeat to limit to specific datasets.")
    parser.add_argument("--split", action="append", default=[], help="Repeat to limit to specific splits.")
    args = parser.parse_args()

    selected_datasets = set(args.dataset)
    selected_splits = set(args.split)
    recipe = get_recipe(args.recipe)
    available = discover_available_dataset_splits(args.data_root)
    summary_rows: list[dict[str, object]] = []

    for spec in available:
        if not _matches_filters(spec.dataset_name, spec.split_name, datasets=selected_datasets, splits=selected_splits):
            continue
        base_dir = args.method_input_root / spec.dataset_name / spec.split_name
        method_inputs = {
            "icr": base_dir / "icr_spans.jsonl",
            "entropy": base_dir / "entropy_spans.jsonl",
        }
        missing_paths = [path for path in method_inputs.values() if not path.exists()]
        if missing_paths:
            if args.skip_missing:
                print(f"[eval] skip missing method inputs: {base_dir}")
                continue
            raise FileNotFoundError(
                "Missing method inputs for "
                f"{spec.dataset_name}/{spec.split_name}: {', '.join(str(path) for path in missing_paths)}"
            )
        output_dir = args.output_root / spec.dataset_name / spec.split_name
        summary = run_recipe(
            recipe=recipe,
            method_inputs=method_inputs,
            output_dir=output_dir,
            device=args.device,
            family_groups=("torch",),
        )
        summary_rows.append(
            {
                "dataset_name": spec.dataset_name,
                "split_name": spec.split_name,
                "output_dir": str(output_dir.resolve()),
                "best_model": summary.get("best_model"),
            }
        )
        print(f"[eval] {spec.dataset_name}/{spec.split_name}: {output_dir}")

    comparison_table = build_best_model_table(summary_rows)
    dump_json(
        args.output_root / "training_summary_all_datasets.json",
        {
            "recipe": recipe.name,
            "device": args.device,
            "runs": summary_rows,
            "comparison_table": comparison_table,
        },
    )
    (args.output_root / "training_summary_all_datasets.txt").parent.mkdir(parents=True, exist_ok=True)
    (args.output_root / "training_summary_all_datasets.txt").write_text(comparison_table + "\n", encoding="utf-8")
    print("\nBest-model summary across dataset splits")
    print(comparison_table)


if __name__ == "__main__":
    main()
