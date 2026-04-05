"""Build prepared method-input files from R-Tuning inference caches."""

from __future__ import annotations

import argparse
from pathlib import Path

from cut_a_lab.core.io import dump_json
from cut_a_lab.prep.r_tuning.datasets import discover_available_dataset_splits
from cut_a_lab.prep.r_tuning.methods import build_method_inputs_from_cache


def _matches_filters(dataset_name: str, split_name: str, *, datasets: set[str], splits: set[str]) -> bool:
    if datasets and dataset_name not in datasets:
        return False
    if splits and split_name not in splits:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/R-Tuning-data"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/r_tuning/inference"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/r_tuning/method_inputs"))
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip dataset splits whose inference cache directory is missing.",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Repeat to limit to specific datasets.")
    parser.add_argument("--split", action="append", default=[], help="Repeat to limit to specific splits.")
    args = parser.parse_args()

    selected_datasets = set(args.dataset)
    selected_splits = set(args.split)
    available = discover_available_dataset_splits(args.data_root)
    summary_rows: list[dict[str, str]] = []

    for spec in available:
        if not _matches_filters(spec.dataset_name, spec.split_name, datasets=selected_datasets, splits=selected_splits):
            continue
        cache_dir = args.cache_root / spec.dataset_name / spec.split_name
        if not cache_dir.exists():
            if args.skip_missing:
                print(f"[methods] skip missing cache: {cache_dir}")
                continue
            raise FileNotFoundError(f"Missing inference cache directory: {cache_dir}")
        output_dir = args.output_root / spec.dataset_name / spec.split_name
        paths = build_method_inputs_from_cache(cache_dir=cache_dir, output_dir=output_dir)
        summary_rows.append(
            {
                "dataset_name": spec.dataset_name,
                "split_name": spec.split_name,
                "cache_dir": str(cache_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
                **paths,
            }
        )
        print(f"[methods] {spec.dataset_name}/{spec.split_name}: {output_dir}")

    dump_json(args.output_root / "method_input_summary.json", {"runs": summary_rows})


if __name__ == "__main__":
    main()
