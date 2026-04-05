"""Contracts shared by the R-Tuning preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NormalizedSample:
    """A dataset-agnostic sample used before model inference."""

    dataset_name: str
    split_name: str
    sample_id: str
    prompt_text: str
    expected_answer: str
    task_type: str
    choices: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "dataset_name": self.dataset_name,
            "split_name": self.split_name,
            "sample_id": self.sample_id,
            "prompt_text": self.prompt_text,
            "expected_answer": self.expected_answer,
            "task_type": self.task_type,
            "choices": list(self.choices),
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class InferenceSampleRecord:
    """One cached inference result for a normalized sample."""

    dataset_name: str
    split_name: str
    sample_id: str
    span_id: str
    prompt_text: str
    generated_text: str
    expected_answer: str
    sample_label: int
    silver_label: int
    task_type: str
    answer_token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "dataset_name": self.dataset_name,
            "split_name": self.split_name,
            "sample_id": self.sample_id,
            "span_id": self.span_id,
            "prompt_text": self.prompt_text,
            "generated_text": self.generated_text,
            "expected_answer": self.expected_answer,
            "sample_label": int(self.sample_label),
            "silver_label": int(self.silver_label),
            "task_type": self.task_type,
            "answer_token_count": int(self.answer_token_count),
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class LayerCacheRecord:
    """One cached per-sample layer summary used by downstream methods."""

    sample: InferenceSampleRecord
    layer_hidden_mean: np.ndarray
    layer_entropy: np.ndarray

    def validate(self) -> None:
        hidden = np.asarray(self.layer_hidden_mean, dtype=np.float32)
        entropy = np.asarray(self.layer_entropy, dtype=np.float32)
        if hidden.ndim != 2:
            raise ValueError(f"Expected layer_hidden_mean with shape [layers, hidden], got {hidden.shape}.")
        if entropy.ndim != 1:
            raise ValueError(f"Expected layer_entropy with shape [layers], got {entropy.shape}.")
        if hidden.shape[0] != entropy.shape[0]:
            raise ValueError(
                "layer_hidden_mean and layer_entropy must share the layer dimension, "
                f"got {hidden.shape[0]} vs {entropy.shape[0]}."
            )
        if hidden.shape[0] < 2:
            raise ValueError("At least two layers are required to build ICR and delta-entropy vectors.")
        if hidden.shape[1] == 0:
            raise ValueError("Hidden-state width must be positive.")


@dataclass(frozen=True)
class InferenceArrayBundle:
    """Dense arrays written alongside sample metadata for later method building."""

    layer_hidden_mean: np.ndarray
    layer_entropy: np.ndarray

    def validate(self) -> None:
        hidden = np.asarray(self.layer_hidden_mean, dtype=np.float32)
        entropy = np.asarray(self.layer_entropy, dtype=np.float32)
        if hidden.ndim != 3:
            raise ValueError(f"Expected layer_hidden_mean with shape [N, layers, hidden], got {hidden.shape}.")
        if entropy.ndim != 2:
            raise ValueError(f"Expected layer_entropy with shape [N, layers], got {entropy.shape}.")
        if hidden.shape[0] != entropy.shape[0]:
            raise ValueError(f"Row count mismatch between hidden and entropy arrays: {hidden.shape[0]} vs {entropy.shape[0]}.")
        if hidden.shape[1] != entropy.shape[1]:
            raise ValueError(
                f"Layer count mismatch between hidden and entropy arrays: {hidden.shape[1]} vs {entropy.shape[1]}."
            )
