"""Feature helpers for the ICR method."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def coerce_feature_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate and convert a list of per-row vectors into a 2D float32 matrix."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D ICR feature matrix, got shape {matrix.shape}.")
    if matrix.shape[0] == 0:
        raise ValueError("ICR input produced zero rows.")
    if matrix.shape[1] == 0:
        raise ValueError("ICR vectors must contain at least one feature.")
    return matrix


def build_feature_names(width: int) -> tuple[str, ...]:
    """Build stable feature names for span-level ICR vectors."""
    if width <= 0:
        raise ValueError(f"ICR feature width must be positive, got {width}.")
    return tuple(f"icr_{index}" for index in range(width))
