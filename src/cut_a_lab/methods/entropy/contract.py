"""Entropy method contract."""

from __future__ import annotations

from cut_a_lab.core.contracts import MethodInputContract
from cut_a_lab.methods.entropy.metadata import METHOD_NAME


ENTROPY_VECTOR_FIELD_ALIASES = ("entropy_vector",)

ENTROPY_INPUT_CONTRACT = MethodInputContract(
    method_name=METHOD_NAME,
    description=(
        "Prepared span-level entropy vectors. The method consumes one JSONL row per span "
        "and treats entropy as an independent feature method."
    ),
    required_fields=("sample_id", "span_id", "sample_label", "entropy_vector"),
    optional_fields=(
        "silver_label",
        "silver_confidence",
        "route",
        "span_type",
        "sample_entropy_vector",
        "candidate_index",
        "source_sample_index",
        "window_size",
    ),
    notes=(
        "Entropy is modeled as a standalone method rather than a required auxiliary signal.",
    ),
)
