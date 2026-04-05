"""R-Tuning dataset preparation pipeline."""

from cut_a_lab.prep.r_tuning.cache import (
    CacheArtifactPaths,
    discover_cache_artifacts,
    load_layer_cache,
    write_inference_cache,
)
from cut_a_lab.prep.r_tuning.contracts import (
    InferenceArrayBundle,
    InferenceSampleRecord,
    LayerCacheRecord,
    NormalizedSample,
)
from cut_a_lab.prep.r_tuning.datasets import (
    DATASET_FILE_SPECS,
    discover_available_dataset_splits,
    load_normalized_samples,
)
from cut_a_lab.prep.r_tuning.methods import build_method_inputs_from_cache
from cut_a_lab.prep.r_tuning.reporting import build_best_model_table

__all__ = [
    "CacheArtifactPaths",
    "DATASET_FILE_SPECS",
    "InferenceArrayBundle",
    "InferenceSampleRecord",
    "LayerCacheRecord",
    "NormalizedSample",
    "build_method_inputs_from_cache",
    "build_best_model_table",
    "discover_available_dataset_splits",
    "discover_cache_artifacts",
    "load_layer_cache",
    "load_normalized_samples",
    "write_inference_cache",
]
