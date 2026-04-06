"""Build reusable inference caches for local R-Tuning datasets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from cut_a_lab.core.io import dump_json
from cut_a_lab.prep.r_tuning.datasets import discover_available_dataset_splits, load_normalized_samples
from cut_a_lab.prep.r_tuning.inference import ModelRunnerConfig, build_inference_cache


def _matches_filters(dataset_name: str, split_name: str, *, datasets: set[str], splits: set[str]) -> bool:
    if datasets and dataset_name not in datasets:
        return False
    if splits and split_name not in splits:
        return False
    return True


def _looks_like_7b_model(model_name_or_path: str) -> bool:
    normalized = model_name_or_path.lower()
    return re.search(r"(^|[^0-9])7b([^0-9]|$)", normalized) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/R-Tuning-data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/r_tuning/inference"))
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per dataset split for smoke testing.",
    )
    parser.add_argument(
        "--allow-local-7b",
        action="store_true",
        help="Explicitly allow running a local/model-path 7B checkpoint. Disabled by default.",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Repeat to limit to specific datasets.")
    parser.add_argument("--split", action="append", default=[], help="Repeat to limit to specific splits.")
    args = parser.parse_args()

    if _looks_like_7b_model(args.model_name_or_path) and not args.allow_local_7b:
        raise SystemExit(
            "Refusing to run a 7B model by default. "
            "Pick a smaller model for smoke tests, or pass --allow-local-7b if you really want it."
        )

    config = ModelRunnerConfig(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    selected_datasets = set(args.dataset)
    selected_splits = set(args.split)
    available = discover_available_dataset_splits(args.data_root)
    summary_rows: list[dict[str, str | int]] = []

    for spec in available:
        if not _matches_filters(spec.dataset_name, spec.split_name, datasets=selected_datasets, splits=selected_splits):
            continue
        samples = load_normalized_samples(root_dir=args.data_root, spec=spec)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        output_dir = args.output_root / spec.dataset_name / spec.split_name
        build_inference_cache(samples=samples, output_dir=output_dir, config=config)
        summary_rows.append(
            {
                "dataset_name": spec.dataset_name,
                "split_name": spec.split_name,
                "n_samples": len(samples),
                "cache_dir": str(output_dir.resolve()),
                "max_new_tokens": args.max_new_tokens,
            }
        )
        print(f"[cache] {spec.dataset_name}/{spec.split_name}: {len(samples)} samples -> {output_dir}")

    dump_json(args.output_root / "cache_summary.json", {"runs": summary_rows})


if __name__ == "__main__":
    main()
