"""ICR method contract."""

from __future__ import annotations

from cut_a_lab.core.contracts import MethodInputContract
from cut_a_lab.methods.icr.metadata import METHOD_NAME


ICR_VECTOR_FIELD_ALIASES = ("icr_vector", "span_vector")

ICR_INPUT_CONTRACT = MethodInputContract(
    method_name=METHOD_NAME,
    description=(
        "Prepared span-level ICR vectors. The method consumes one JSONL row per span "
        "and does not ingest raw sample-level tensors in version 0.1.0."
    ),
    required_fields=("sample_id", "span_id", "sample_label"),
    required_any_of=(ICR_VECTOR_FIELD_ALIASES,),
    optional_fields=(
        "silver_label",
        "silver_confidence",
        "route",
        "span_type",
        "candidate_index",
        "source_sample_index",
        "window_size",
    ),
    notes=(
        "Use `icr_vector` for new data.",
        "The legacy alias `span_vector` is accepted for compatibility with prepared span datasets.",
    ),
)
