"""Pure helpers for self-consistency pseudo-label experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def extract_first_non_empty_line(text: str) -> str:
    """Return the first non-empty line from a generation."""
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return str(text).strip()


def normalize_answer_key(text: str) -> str:
    """Normalize a generated answer for majority-vote grouping."""
    normalized = extract_first_non_empty_line(text).strip().lower()
    parts = normalized.split()
    return " ".join(parts)


@dataclass(frozen=True)
class MajorityVoteResult:
    """Majority-vote statistics for one sampled answer set."""

    majority_key: str
    majority_count: int
    agreement_rate: float
    tie_count: int
    answer_keys: list[str]


def majority_vote(answer_texts: Sequence[str]) -> MajorityVoteResult:
    """Resolve a majority answer with deterministic first-occurrence tie break."""
    if not answer_texts:
        raise ValueError("answer_texts must not be empty.")

    answer_keys = [normalize_answer_key(text) for text in answer_texts]
    counts = Counter(answer_keys)
    majority_count = max(counts.values())
    tied_keys = {key for key, count in counts.items() if count == majority_count}
    majority_key = next(key for key in answer_keys if key in tied_keys)
    return MajorityVoteResult(
        majority_key=majority_key,
        majority_count=majority_count,
        agreement_rate=float(majority_count / len(answer_keys)),
        tie_count=int(len(tied_keys)),
        answer_keys=list(answer_keys),
    )


def sanitize_correlation_values(values: np.ndarray) -> np.ndarray:
    """Convert nan correlations to zero so sign/weight logic stays stable."""
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)


def align_sample_metric(sample_ids: Sequence[str], value_by_sample_id: dict[str, float], *, name: str) -> np.ndarray:
    """Align a per-sample metric dictionary to the feature-matrix sample order."""
    missing = [sample_id for sample_id in sample_ids if sample_id not in value_by_sample_id]
    if missing:
        preview = ", ".join(missing[:3])
        raise KeyError(f"Missing {name} values for {len(missing)} samples, e.g. {preview}.")
    return np.asarray([value_by_sample_id[sample_id] for sample_id in sample_ids], dtype=np.float64)

