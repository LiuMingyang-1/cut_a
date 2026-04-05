"""Cache readers and writers for the R-Tuning inference stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cut_a_lab.core.io import dump_json, write_jsonl
from cut_a_lab.prep.r_tuning.contracts import InferenceArrayBundle, InferenceSampleRecord, LayerCacheRecord


@dataclass(frozen=True)
class CacheArtifactPaths:
    """Paths for one dataset/split inference cache."""

    root_dir: Path
    manifest_path: Path
    samples_path: Path
    layer_cache_path: Path


def discover_cache_artifacts(root_dir: Path) -> CacheArtifactPaths:
    root = Path(root_dir)
    return CacheArtifactPaths(
        root_dir=root,
        manifest_path=root / "manifest.json",
        samples_path=root / "samples.jsonl",
        layer_cache_path=root / "layer_cache.npz",
    )


def write_inference_cache(
    *,
    output_dir: Path,
    records: list[LayerCacheRecord],
    manifest: dict[str, Any],
) -> CacheArtifactPaths:
    """Write one inference cache bundle to disk."""
    paths = discover_cache_artifacts(output_dir)
    paths.root_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        record.validate()

    sample_rows = [record.sample.to_json() for record in records]
    hidden = np.stack([np.asarray(record.layer_hidden_mean, dtype=np.float32) for record in records], axis=0)
    entropy = np.stack([np.asarray(record.layer_entropy, dtype=np.float32) for record in records], axis=0)
    bundle = InferenceArrayBundle(layer_hidden_mean=hidden, layer_entropy=entropy)
    bundle.validate()

    write_jsonl(paths.samples_path, sample_rows)
    np.savez_compressed(
        paths.layer_cache_path,
        layer_hidden_mean=bundle.layer_hidden_mean.astype(np.float32),
        layer_entropy=bundle.layer_entropy.astype(np.float32),
    )
    dump_json(paths.manifest_path, manifest)
    return paths


def load_layer_cache(cache_dir: Path) -> tuple[list[dict[str, Any]], InferenceArrayBundle]:
    """Load cached sample rows and dense arrays from one cache directory."""
    paths = discover_cache_artifacts(cache_dir)
    if not paths.samples_path.exists():
        raise FileNotFoundError(f"Missing cached samples: {paths.samples_path}")
    if not paths.layer_cache_path.exists():
        raise FileNotFoundError(f"Missing layer cache: {paths.layer_cache_path}")

    sample_rows: list[dict[str, Any]] = []
    with paths.samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                import json

                sample_rows.append(json.loads(line))

    with np.load(paths.layer_cache_path) as payload:
        bundle = InferenceArrayBundle(
            layer_hidden_mean=np.asarray(payload["layer_hidden_mean"], dtype=np.float32),
            layer_entropy=np.asarray(payload["layer_entropy"], dtype=np.float32),
        )
    bundle.validate()

    if len(sample_rows) != bundle.layer_hidden_mean.shape[0]:
        raise ValueError(
            f"Cached row count mismatch: {len(sample_rows)} sample rows vs {bundle.layer_hidden_mean.shape[0]} arrays."
        )
    return sample_rows, bundle
