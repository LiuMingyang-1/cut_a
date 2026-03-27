"""Delta entropy method contract."""

from __future__ import annotations

from cut_a_lab.core.contracts import MethodInputContract
from cut_a_lab.methods.delta_entropy.metadata import METHOD_NAME


DELTA_ENTROPY_VECTOR_FIELD_ALIASES = ("delta_entropy_vector",)

DELTA_ENTROPY_INPUT_CONTRACT = MethodInputContract(
    method_name=METHOD_NAME,
    description=(
        "Prepared span-level delta entropy vectors. The method treats delta entropy "
        "as an independent method instead of an internal entropy implementation detail."
    ),
    required_fields=("sample_id", "span_id", "sample_label", "delta_entropy_vector"),
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
        "Delta entropy vectors are typically derived upstream, but the core does not assume how they were produced.",
    ),
)
