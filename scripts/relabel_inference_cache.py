"""Re-apply fixed label extraction to existing inference cache samples.jsonl files.

Reads generated_text / expected_answer from cache, recomputes sample_label and
silver_label with the improved classify_generation logic, and overwrites
samples.jsonl. Does NOT touch layer_cache.npz (hidden states are unchanged).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from cut_a_lab.prep.r_tuning.contracts import NormalizedSample
from cut_a_lab.prep.r_tuning.datasets import discover_available_dataset_splits
from cut_a_lab.prep.r_tuning.inference import classify_generation

_DATASET_CHOICES: dict[str, tuple[str, ...]] = {
    "FEVER": ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"),
    "WiCE": ("supported", "partially_supported", "not_supported"),
}


def _choices_for_task(*, dataset_name: str, task_type: str) -> tuple[str, ...]:
    if task_type != "classification_label":
        return ()
    if dataset_name not in _DATASET_CHOICES:
        raise KeyError(
            f"Missing choices mapping for classification dataset {dataset_name!r}. "
            f"Add it to _DATASET_CHOICES before relabeling."
        )
    return _DATASET_CHOICES[dataset_name]


def _relabel_samples_file(samples_path: Path, dataset_name: str) -> dict[str, int | dict[str, int]]:
    rows: list[dict[str, object]] = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    before = {"0": 0, "1": 0}
    after = {"0": 0, "1": 0}

    for row in rows:
        before[str(int(row["sample_label"]))] += 1
        task_type = str(row["task_type"])
        choices = _choices_for_task(dataset_name=dataset_name, task_type=task_type)
        dummy_sample = NormalizedSample(
            dataset_name=str(row["dataset_name"]),
            split_name=str(row["split_name"]),
            sample_id=str(row["sample_id"]),
            prompt_text="",
            expected_answer=str(row["expected_answer"]),
            task_type=task_type,
            choices=choices,
            metadata={},
        )
        new_label, new_silver = classify_generation(dummy_sample, str(row["generated_text"]))
        row["sample_label"] = int(new_label)
        row["silver_label"] = int(new_silver)
        after[str(new_label)] += 1

    backup_path = samples_path.with_suffix(".jsonl.bak")
    if not backup_path.exists():
        shutil.copy2(samples_path, backup_path)

    with samples_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"before": before, "after": after, "n": len(rows)}


def _matches_filters(dataset_name: str, split_name: str, *, datasets: set[str], splits: set[str]) -> bool:
    if datasets and dataset_name not in datasets:
        return False
    if splits and split_name not in splits:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/r_tuning/inference"))
    parser.add_argument("--data-root", type=Path, default=Path("data/R-Tuning-data"))
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--split", action="append", default=[])
    args = parser.parse_args()

    selected_datasets = set(args.dataset)
    selected_splits = set(args.split)
    available = discover_available_dataset_splits(args.data_root)

    for spec in available:
        if not _matches_filters(
            spec.dataset_name,
            spec.split_name,
            datasets=selected_datasets,
            splits=selected_splits,
        ):
            continue

        samples_path = args.cache_root / spec.dataset_name / spec.split_name / "samples.jsonl"
        if not samples_path.exists():
            print(f"[relabel] skip missing: {samples_path}")
            continue

        stats = _relabel_samples_file(samples_path, spec.dataset_name)
        n = int(stats["n"])
        b0 = int(stats["before"]["0"])
        b1 = int(stats["before"]["1"])
        a0 = int(stats["after"]["0"])
        a1 = int(stats["after"]["1"])

        if n == 0:
            print(f"[relabel] {spec.dataset_name}/{spec.split_name} n=0 before: correct=0 wrong=0 after: correct=0 wrong=0")
            continue

        print(
            f"[relabel] {spec.dataset_name}/{spec.split_name} n={n} "
            f"before: correct={b0}({b0 / n * 100:.1f}%) wrong={b1}({b1 / n * 100:.1f}%) "
            f"after: correct={a0}({a0 / n * 100:.1f}%) wrong={a1}({a1 / n * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
