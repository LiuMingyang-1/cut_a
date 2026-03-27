"""Feature-view builders used by recipes."""

from __future__ import annotations

from typing import Callable

import numpy as np

from cut_a_lab.core.contracts import FeatureBlock, SpanRecord
from cut_a_lab.recipes.base import FeatureSetSpec


FeatureSetBundle = tuple[tuple[SpanRecord, ...], np.ndarray, list[str]]

ICR_EARLY = slice(0, 9)
ICR_MIDDLE = slice(9, 18)
ICR_LATE = slice(18, 27)

ENT_EARLY = slice(0, 10)
ENT_MIDDLE = slice(10, 19)
ENT_LATE = slice(19, 28)


def _align_blocks(blocks: list[FeatureBlock]) -> tuple[tuple[SpanRecord, ...], list[np.ndarray], list[FeatureBlock]]:
    """Align multiple feature blocks by row key and validate shared metadata."""
    if not blocks:
        raise ValueError("At least one feature block is required.")
    for block in blocks:
        block.validate()

    anchor = blocks[0]
    anchor_records = list(anchor.records)
    anchor_row_keys = [record.row_key for record in anchor_records]
    anchor_lookup = {record.row_key: index for index, record in enumerate(anchor_records)}
    aligned_matrices = [np.asarray(anchor.features, dtype=np.float32)]
    aligned_blocks = [anchor]

    for block in blocks[1:]:
        block_lookup = {record.row_key: index for index, record in enumerate(block.records)}
        if set(block_lookup) != set(anchor_lookup):
            missing_in_block = sorted(set(anchor_lookup) - set(block_lookup))
            missing_in_anchor = sorted(set(block_lookup) - set(anchor_lookup))
            raise ValueError(
                f"Method {block.method_name!r} does not align with anchor {anchor.method_name!r}. "
                f"Missing in block: {missing_in_block[:5]}; missing in anchor: {missing_in_anchor[:5]}"
            )

        reorder = [block_lookup[row_key] for row_key in anchor_row_keys]
        reordered_records = [block.records[index] for index in reorder]
        for left, right in zip(anchor_records, reordered_records):
            if left.sample_id != right.sample_id:
                raise ValueError(
                    f"sample_id mismatch for row {left.row_key!r}: {left.sample_id!r} vs {right.sample_id!r}"
                )
            if left.sample_label != right.sample_label:
                raise ValueError(
                    f"sample_label mismatch for row {left.row_key!r}: {left.sample_label!r} vs {right.sample_label!r}"
                )
            if left.silver_label != right.silver_label:
                raise ValueError(
                    f"silver_label mismatch for row {left.row_key!r}: {left.silver_label!r} vs {right.silver_label!r}"
                )

        aligned_matrices.append(np.asarray(block.features, dtype=np.float32)[reorder])
        aligned_blocks.append(block)

    return tuple(anchor_records), aligned_matrices, aligned_blocks


def build_concat_feature_set(feature_set: FeatureSetSpec, blocks: list[FeatureBlock]) -> FeatureSetBundle:
    """Concatenate raw method feature blocks in recipe order."""
    records, aligned_matrices, aligned_blocks = _align_blocks(blocks)
    combined_names: list[str] = []
    for block in aligned_blocks:
        combined_names.extend(f"{block.method_name}.{name}" for name in block.feature_names)
    return records, np.hstack(aligned_matrices).astype(np.float32), combined_names


def _validate_matrix_width(matrix: np.ndarray, *, expected_width: int, method_name: str, feature_set_name: str) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"{feature_set_name!r} expected a 2D matrix for method {method_name!r}, got {matrix.shape}.")
    if matrix.shape[1] != expected_width:
        raise ValueError(
            f"{feature_set_name!r} expects method {method_name!r} to have width {expected_width}, "
            f"got {matrix.shape[1]}."
        )


def _extract_discrepancy_features(
    vectors: np.ndarray,
    *,
    early: slice,
    middle: slice,
    late: slice,
    expected_width: int,
    method_name: str,
    feature_set_name: str,
    name_prefix: str,
) -> tuple[np.ndarray, list[str]]:
    _validate_matrix_width(
        vectors,
        expected_width=expected_width,
        method_name=method_name,
        feature_set_name=feature_set_name,
    )

    mean_early = vectors[:, early].mean(axis=1)
    mean_mid = vectors[:, middle].mean(axis=1)
    mean_late = vectors[:, late].mean(axis=1)

    diff_mid_early = mean_mid - mean_early
    diff_late_mid = mean_late - mean_mid
    diff_late_early = mean_late - mean_early

    layers = np.arange(vectors.shape[1], dtype=np.float32)
    layer_mean = layers.mean()
    layer_var = ((layers - layer_mean) ** 2).sum()
    slopes = ((vectors * (layers[None, :] - layer_mean)).sum(axis=1)) / layer_var

    features = np.column_stack(
        [
            mean_early,
            mean_mid,
            mean_late,
            diff_mid_early,
            diff_late_mid,
            diff_late_early,
            slopes,
        ]
    ).astype(np.float32)
    names = [
        f"{name_prefix}mean_early",
        f"{name_prefix}mean_mid",
        f"{name_prefix}mean_late",
        f"{name_prefix}diff_mid_early",
        f"{name_prefix}diff_late_mid",
        f"{name_prefix}diff_late_early",
        f"{name_prefix}slope",
    ]
    return features, names


def build_discrepancy_combined_feature_set(feature_set: FeatureSetSpec, blocks: list[FeatureBlock]) -> FeatureSetBundle:
    """Build the legacy Cut A discrepancy-combined feature view from ICR and entropy."""
    records, aligned_matrices, aligned_blocks = _align_blocks(blocks)
    by_method = {block.method_name: matrix for block, matrix in zip(aligned_blocks, aligned_matrices)}

    required_methods = {"icr", "entropy"}
    if set(feature_set.methods) != required_methods:
        raise ValueError(
            f"{feature_set.name!r} requires methods {sorted(required_methods)}, got {list(feature_set.methods)}."
        )

    icr_features, icr_names = _extract_discrepancy_features(
        by_method["icr"],
        early=ICR_EARLY,
        middle=ICR_MIDDLE,
        late=ICR_LATE,
        expected_width=27,
        method_name="icr",
        feature_set_name=feature_set.name,
        name_prefix="icr_",
    )
    entropy_features, entropy_names = _extract_discrepancy_features(
        by_method["entropy"],
        early=ENT_EARLY,
        middle=ENT_MIDDLE,
        late=ENT_LATE,
        expected_width=28,
        method_name="entropy",
        feature_set_name=feature_set.name,
        name_prefix="ent_",
    )
    return records, np.hstack([icr_features, entropy_features]).astype(np.float32), icr_names + entropy_names


FEATURE_VIEW_BUILDERS: dict[str, Callable[[FeatureSetSpec, list[FeatureBlock]], FeatureSetBundle]] = {
    "concat": build_concat_feature_set,
    "discrepancy_combined": build_discrepancy_combined_feature_set,
}


def build_feature_set_bundle(feature_set: FeatureSetSpec, blocks: list[FeatureBlock]) -> FeatureSetBundle:
    """Build one feature-set bundle using the configured view."""
    builder = FEATURE_VIEW_BUILDERS.get(feature_set.view_name)
    if builder is None:
        available = ", ".join(sorted(FEATURE_VIEW_BUILDERS))
        raise ValueError(f"Unknown feature view {feature_set.view_name!r}. Available views: {available}")
    return builder(feature_set, blocks)
