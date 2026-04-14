"""Create a reproducible stratified subset from normalized_features.npz.

This is meant for cheap pilot runs of:
  1. scripts/layer_ablation_analysis.py
  2. scripts/experiments/concentration_vs_ood.py
  3. scripts/experiments/compare_models_multi.py

It samples each split independently while preserving label balance as much as
possible, then writes a smaller .npz with the same keys as the original file.

Usage:
    uv run python scripts/experiments/make_feature_subset.py \\
        --features-path outputs/pararel_experiment/normalized_features.npz \\
        --output-path outputs/experiments/subsets/qwen2.5-7b-instruct-20p-seed7/normalized_features.npz \\
        --fraction 0.2 \\
        --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SPLIT_CONFIG = {
    "tr": ("tr_icr_raw", "tr_ent_raw", "tr_lbl"),
    "id": ("id_icr_raw", "id_ent_raw", "id_lbl"),
    "od": ("od_icr_raw", "od_ent_raw", "od_lbl"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Fraction of rows to keep per split. Must satisfy 0 < fraction <= 1.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _validate_fraction(fraction: float) -> None:
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"--fraction must satisfy 0 < fraction <= 1, got {fraction}")


def _validate_split_arrays(
    split_name: str,
    icr: np.ndarray,
    ent: np.ndarray,
    labels: np.ndarray,
) -> None:
    n_rows = labels.shape[0]
    if icr.shape[0] != n_rows or ent.shape[0] != n_rows:
        raise ValueError(
            f"{split_name}: row-count mismatch: "
            f"ICR={icr.shape[0]}, Ent={ent.shape[0]}, labels={n_rows}"
        )
    if n_rows == 0:
        raise ValueError(f"{split_name}: empty split is not supported.")


def _sample_class_indices(
    class_indices: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if class_indices.size == 0:
        return class_indices
    n_keep = int(round(class_indices.size * fraction))
    n_keep = max(1, n_keep)
    n_keep = min(class_indices.size, n_keep)
    chosen = rng.choice(class_indices, size=n_keep, replace=False)
    return np.sort(chosen.astype(np.int64, copy=False))


def _stratified_indices(labels: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    sampled_parts: list[np.ndarray] = []
    for label in sorted(np.unique(labels).tolist()):
        class_indices = np.flatnonzero(labels == label)
        sampled_parts.append(_sample_class_indices(class_indices, fraction=fraction, rng=rng))
    if not sampled_parts:
        raise ValueError("No labels found while building subset.")
    return np.sort(np.concatenate(sampled_parts))


def _describe_split(split_name: str, original_labels: np.ndarray, subset_labels: np.ndarray) -> str:
    return (
        f"{split_name}: {len(subset_labels)}/{len(original_labels)} rows kept "
        f"(pos_rate {float(original_labels.mean()):.4f} -> {float(subset_labels.mean()):.4f})"
    )


def main() -> None:
    args = _parse_args()
    _validate_fraction(args.fraction)
    rng = np.random.default_rng(args.seed)

    with np.load(args.features_path) as data:
        subset_arrays: dict[str, np.ndarray] = {}
        summaries: list[str] = []

        for split_name, (icr_key, ent_key, lbl_key) in SPLIT_CONFIG.items():
            if icr_key not in data or ent_key not in data or lbl_key not in data:
                raise KeyError(f"Missing keys for split {split_name!r} in {args.features_path}")

            icr = data[icr_key]
            ent = data[ent_key]
            labels = data[lbl_key]
            _validate_split_arrays(split_name=split_name, icr=icr, ent=ent, labels=labels)

            keep_indices = _stratified_indices(labels=labels, fraction=args.fraction, rng=rng)
            subset_arrays[icr_key] = icr[keep_indices]
            subset_arrays[ent_key] = ent[keep_indices]
            subset_arrays[lbl_key] = labels[keep_indices]
            summaries.append(_describe_split(split_name, labels, subset_arrays[lbl_key]))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_path, **subset_arrays)

    print(f"Saved subset features to {args.output_path}")
    print(f"fraction={args.fraction:.4f} seed={args.seed}")
    for line in summaries:
        print(line)


if __name__ == "__main__":
    main()
