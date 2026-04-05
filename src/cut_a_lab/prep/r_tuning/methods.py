"""Build method-input JSONL files from reusable inference caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cut_a_lab.core.io import dump_json, write_jsonl
from cut_a_lab.prep.r_tuning.cache import load_layer_cache


def _adjacent_cosine_distance(hidden: np.ndarray) -> np.ndarray:
    left = np.asarray(hidden[:-1], dtype=np.float32)
    right = np.asarray(hidden[1:], dtype=np.float32)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_safe = np.where(left_norm < 1e-6, 1.0, left_norm)
    right_safe = np.where(right_norm < 1e-6, 1.0, right_norm)
    cosine = (left * right).sum(axis=1) / (left_safe[:, 0] * right_safe[:, 0])
    cosine = np.clip(cosine, -1.0, 1.0)
    return (1.0 - cosine).astype(np.float32)


def _build_row(sample_row: dict[str, Any], *, icr_vector: np.ndarray, entropy_vector: np.ndarray, delta_entropy_vector: np.ndarray) -> dict[str, Any]:
    metadata = sample_row.get("metadata", {})
    row = {
        "sample_id": sample_row["sample_id"],
        "span_id": sample_row["span_id"],
        "sample_label": int(sample_row["sample_label"]),
        "silver_label": int(sample_row["silver_label"]),
        "span_type": "full_sample",
        "route": "r_tuning_inference_cache",
        "generated_text": sample_row["generated_text"],
        "expected_answer": sample_row["expected_answer"],
        "answer_token_count": int(sample_row["answer_token_count"]),
        "icr_vector": icr_vector.astype(np.float32).tolist(),
        "span_vector": icr_vector.astype(np.float32).tolist(),
        "entropy_vector": entropy_vector.astype(np.float32).tolist(),
        "delta_entropy_vector": delta_entropy_vector.astype(np.float32).tolist(),
    }
    row.update(metadata)
    return row


def build_method_inputs_from_cache(*, cache_dir: Path, output_dir: Path) -> dict[str, str]:
    """Build prepared method-input JSONL files from one inference cache."""
    sample_rows, bundle = load_layer_cache(cache_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, Any]] = []
    for index, sample_row in enumerate(sample_rows):
        hidden = bundle.layer_hidden_mean[index]
        entropy = bundle.layer_entropy[index]
        icr_vector = _adjacent_cosine_distance(hidden)
        delta_entropy = np.diff(entropy).astype(np.float32)
        combined_rows.append(
            _build_row(
                sample_row,
                icr_vector=icr_vector,
                entropy_vector=entropy,
                delta_entropy_vector=delta_entropy,
            )
        )

    icr_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"entropy_vector", "delta_entropy_vector"}
        }
        for row in combined_rows
    ]
    entropy_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"icr_vector", "span_vector", "delta_entropy_vector"}
        }
        for row in combined_rows
    ]
    delta_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"icr_vector", "span_vector", "entropy_vector"}
        }
        for row in combined_rows
    ]

    icr_path = output / "icr_spans.jsonl"
    entropy_path = output / "entropy_spans.jsonl"
    delta_path = output / "delta_entropy_spans.jsonl"
    combined_path = output / "combined_spans.jsonl"
    write_jsonl(icr_path, icr_rows)
    write_jsonl(entropy_path, entropy_rows)
    write_jsonl(delta_path, delta_rows)
    write_jsonl(combined_path, combined_rows)

    manifest = {
        "cache_dir": str(Path(cache_dir).resolve()),
        "n_rows": len(combined_rows),
        "icr_path": str(icr_path.resolve()),
        "entropy_path": str(entropy_path.resolve()),
        "delta_entropy_path": str(delta_path.resolve()),
        "combined_path": str(combined_path.resolve()),
        "status": "ok",
    }
    dump_json(output / "method_manifest.json", manifest)
    return {
        "icr": str(icr_path.resolve()),
        "entropy": str(entropy_path.resolve()),
        "delta_entropy": str(delta_path.resolve()),
        "combined": str(combined_path.resolve()),
    }
