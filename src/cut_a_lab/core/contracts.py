"""Core data contracts shared across the experiment runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


RowLevel = Literal["span", "sample"]

INPUT_METADATA_KEY = "input_metadata"
PREDICTION_ROW_RESERVED_KEYS = frozenset(
    {
        "feature_set",
        "family",
        "model",
        "span_id",
        "sample_id",
        "sample_label",
        "silver_label",
        "silver_confidence",
        "is_labeled",
        "fold",
        "probability",
        INPUT_METADATA_KEY,
    }
)
KNOWN_METHOD_VECTOR_FIELDS = frozenset(
    {
        "icr_vector",
        "span_vector",
        "entropy_vector",
        "delta_entropy_vector",
    }
)


def build_record_metadata(
    row: dict[str, Any],
    *,
    excluded_keys: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Extract stable record metadata while dropping known feature vectors."""
    return {
        key: value
        for key, value in row.items()
        if key not in excluded_keys and key not in KNOWN_METHOD_VECTOR_FIELDS
    }


def merge_prediction_metadata(payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Merge metadata into one prediction row without letting it override core keys."""
    collided_items: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in PREDICTION_ROW_RESERVED_KEYS:
            collided_items[key] = value
            continue
        payload[key] = value

    if collided_items:
        payload[INPUT_METADATA_KEY] = collided_items


@dataclass(frozen=True)
class MethodInputContract:
    """Human-readable description of a method's input contract."""

    method_name: str
    description: str
    required_fields: tuple[str, ...]
    required_any_of: tuple[tuple[str, ...], ...] = ()
    optional_fields: tuple[str, ...] = ()
    row_level: RowLevel = "span"
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        """Return a CLI-friendly contract description."""
        lines = [
            f"Method: {self.method_name}",
            f"Row level: {self.row_level}",
            "",
            self.description,
            "",
            "Required fields:",
        ]
        lines.extend(f"- {field_name}" for field_name in self.required_fields)
        if self.required_any_of:
            lines.append("")
            lines.append("Required alternatives:")
            for options in self.required_any_of:
                lines.append(f"- one of: {', '.join(options)}")
        if self.optional_fields:
            lines.append("")
            lines.append("Optional fields:")
            lines.extend(f"- {field_name}" for field_name in self.optional_fields)
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass(frozen=True)
class SpanRecord:
    """Minimal row metadata shared by training, outputs, and analysis."""

    sample_id: str
    span_id: str
    sample_label: int
    silver_label: int | None
    silver_confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def row_key(self) -> str:
        """Return the row key used for method alignment."""
        return self.span_id

    @property
    def is_labeled(self) -> bool:
        """Whether the span is available for span-level supervision."""
        return self.silver_label is not None

    def to_prediction_row(
        self,
        *,
        feature_set: str,
        family: str,
        model: str,
        fold: int,
        probability: float | None,
    ) -> dict[str, Any]:
        """Serialize a record into the standard OOF prediction row format."""
        payload = {
            "feature_set": feature_set,
            "family": family,
            "model": model,
            "span_id": self.span_id,
            "sample_id": self.sample_id,
            "sample_label": int(self.sample_label),
            "silver_label": self.silver_label,
            "silver_confidence": self.silver_confidence,
            "is_labeled": self.is_labeled,
            "fold": int(fold),
            "probability": probability,
        }
        merge_prediction_metadata(payload, self.metadata)
        return payload


@dataclass(frozen=True)
class FeatureBlock:
    """Row-aligned method output consumed by recipes."""

    method_name: str
    level: RowLevel
    feature_names: tuple[str, ...]
    features: np.ndarray
    records: tuple[SpanRecord, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate shape and row alignment invariants."""
        if self.level != "span":
            raise ValueError(f"Only span-level feature blocks are supported, got {self.level!r}.")
        matrix = np.asarray(self.features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(
                f"Feature block {self.method_name!r} must be 2D, got shape {matrix.shape}."
            )
        if matrix.shape[0] != len(self.records):
            raise ValueError(
                f"Feature rows for {self.method_name!r} do not match record count: "
                f"{matrix.shape[0]} vs {len(self.records)}."
            )
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature width for {self.method_name!r} does not match feature names: "
                f"{matrix.shape[1]} vs {len(self.feature_names)}."
            )
        if matrix.shape[1] == 0:
            raise ValueError(f"Feature block {self.method_name!r} has zero feature columns.")
        row_keys = [record.row_key for record in self.records]
        if len(set(row_keys)) != len(row_keys):
            raise ValueError(f"Feature block {self.method_name!r} contains duplicate row keys.")

    @property
    def row_keys(self) -> tuple[str, ...]:
        """Return row keys for alignment across methods."""
        return tuple(record.row_key for record in self.records)
