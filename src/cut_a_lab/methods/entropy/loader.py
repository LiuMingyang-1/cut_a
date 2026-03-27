"""Loader for prepared entropy span features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cut_a_lab.core.contracts import FeatureBlock, SpanRecord, build_record_metadata
from cut_a_lab.core.io import read_jsonl
from cut_a_lab.methods.base import BaseMethod
from cut_a_lab.methods.entropy.contract import ENTROPY_INPUT_CONTRACT, ENTROPY_VECTOR_FIELD_ALIASES
from cut_a_lab.methods.entropy.features import build_feature_names, coerce_feature_matrix
from cut_a_lab.methods.entropy.metadata import METHOD_NAME


def _coerce_binary_label(value: Any, *, field_name: str, row_index: int) -> int:
    label = int(value)
    if label not in {0, 1}:
        raise ValueError(f"Row {row_index} has invalid {field_name}={value!r}; expected 0 or 1.")
    return label


def _coerce_optional_binary_label(value: Any, *, field_name: str, row_index: int) -> int | None:
    if value is None:
        return None
    return _coerce_binary_label(value, field_name=field_name, row_index=row_index)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _pick_vector_field(row: dict[str, Any], *, row_index: int) -> np.ndarray:
    for field_name in ENTROPY_VECTOR_FIELD_ALIASES:
        if field_name in row:
            vector = np.asarray(row[field_name], dtype=np.float32)
            if vector.ndim != 1:
                raise ValueError(
                    f"Row {row_index} field {field_name!r} must be 1D, got shape {vector.shape}."
                )
            if vector.size == 0:
                raise ValueError(f"Row {row_index} field {field_name!r} is empty.")
            return vector
    raise ValueError(
        f"Row {row_index} is missing all accepted entropy vector fields: {', '.join(ENTROPY_VECTOR_FIELD_ALIASES)}."
    )


class EntropyMethod(BaseMethod):
    """Prepared span-level entropy feature method."""

    name = METHOD_NAME

    def input_contract(self):
        return ENTROPY_INPUT_CONTRACT

    def load_feature_block(self, path: Path) -> FeatureBlock:
        rows = read_jsonl(Path(path))
        if not rows:
            raise ValueError(f"No rows found in entropy input file {path}.")

        vectors: list[np.ndarray] = []
        records: list[SpanRecord] = []

        for row_index, row in enumerate(rows):
            missing_required = [
                field_name for field_name in ENTROPY_INPUT_CONTRACT.required_fields if field_name not in row
            ]
            if missing_required:
                raise ValueError(
                    f"Row {row_index} is missing required entropy fields: {', '.join(missing_required)}."
                )

            vector = _pick_vector_field(row, row_index=row_index)
            vectors.append(vector)

            metadata = build_record_metadata(
                row,
                excluded_keys={
                    "sample_id",
                    "span_id",
                    "sample_label",
                    "silver_label",
                    "silver_confidence",
                },
            )

            records.append(
                SpanRecord(
                    sample_id=str(row["sample_id"]),
                    span_id=str(row["span_id"]),
                    sample_label=_coerce_binary_label(row["sample_label"], field_name="sample_label", row_index=row_index),
                    silver_label=_coerce_optional_binary_label(
                        row.get("silver_label"),
                        field_name="silver_label",
                        row_index=row_index,
                    ),
                    silver_confidence=_coerce_optional_float(row.get("silver_confidence")),
                    metadata=metadata,
                )
            )

        matrix = coerce_feature_matrix(vectors)
        block = FeatureBlock(
            method_name=self.name,
            level="span",
            feature_names=build_feature_names(matrix.shape[1]),
            features=matrix,
            records=tuple(records),
            metadata={"source_path": str(Path(path).resolve())},
        )
        block.validate()
        return block
