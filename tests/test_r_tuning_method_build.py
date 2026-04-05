"""Tests for cache-to-method-input conversion."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cut_a_lab.prep.r_tuning.cache import write_inference_cache
from cut_a_lab.prep.r_tuning.contracts import InferenceSampleRecord, LayerCacheRecord
from cut_a_lab.prep.r_tuning.methods import build_method_inputs_from_cache


class RTuningMethodBuildTests(unittest.TestCase):
    def test_build_method_inputs_from_cache_writes_all_expected_files(self) -> None:
        sample = InferenceSampleRecord(
            dataset_name="demo",
            split_name="train",
            sample_id="demo:0",
            span_id="demo:0:full",
            prompt_text="Question: demo\nAnswer:",
            generated_text="wrong",
            expected_answer="right",
            sample_label=1,
            silver_label=1,
            task_type="qa_exact_match",
            answer_token_count=2,
            metadata={"subject": "demo"},
        )
        record = LayerCacheRecord(
            sample=sample,
            layer_hidden_mean=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            layer_entropy=np.asarray([1.2, 0.9, 0.7], dtype=np.float32),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            cache_dir = tmp_root / "cache"
            output_dir = tmp_root / "methods"
            write_inference_cache(
                output_dir=cache_dir,
                records=[record],
                manifest={"status": "ok", "n_samples": 1},
            )
            paths = build_method_inputs_from_cache(cache_dir=cache_dir, output_dir=output_dir)

            for key in ("icr", "entropy", "delta_entropy", "combined"):
                self.assertTrue(Path(paths[key]).exists())

            with (output_dir / "combined_spans.jsonl").open("r", encoding="utf-8") as handle:
                row = json.loads(handle.readline())

        self.assertEqual(row["sample_id"], "demo:0")
        self.assertEqual(len(row["icr_vector"]), 2)
        self.assertEqual(len(row["entropy_vector"]), 3)
        self.assertEqual(len(row["delta_entropy_vector"]), 2)


if __name__ == "__main__":
    unittest.main()
