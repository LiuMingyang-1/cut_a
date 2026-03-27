"""Smoke tests for prepared vector methods."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cut_a_lab.core.registry import get_method, list_methods


class MethodLoaderTests(unittest.TestCase):
    def test_method_registry_contains_all_prepared_vector_methods(self) -> None:
        self.assertEqual(list_methods(), ["delta_entropy", "entropy", "icr"])

    def test_all_vector_methods_load_same_span_rows(self) -> None:
        rows = [
            {
                "sample_id": "s1",
                "span_id": "s1:0",
                "sample_label": 1,
                "silver_label": 1,
                "span_vector": [0.1, 0.2, 0.3],
                "entropy_vector": [2.0, 1.9, 1.8, 1.7],
                "delta_entropy_vector": [0.0, -0.1, -0.1],
            },
            {
                "sample_id": "s2",
                "span_id": "s2:0",
                "sample_label": 0,
                "silver_label": 0,
                "span_vector": [0.4, 0.5, 0.6],
                "entropy_vector": [1.5, 1.4, 1.3, 1.2],
                "delta_entropy_vector": [-0.2, -0.1, -0.1],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "prepared_vectors.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            icr_block = get_method("icr").load_feature_block(path)
            entropy_block = get_method("entropy").load_feature_block(path)
            delta_block = get_method("delta_entropy").load_feature_block(path)

        self.assertEqual(icr_block.row_keys, entropy_block.row_keys)
        self.assertEqual(icr_block.row_keys, delta_block.row_keys)
        self.assertEqual(icr_block.features.shape, (2, 3))
        self.assertEqual(entropy_block.features.shape, (2, 4))
        self.assertEqual(delta_block.features.shape, (2, 3))
        self.assertEqual(icr_block.feature_names, ("icr_0", "icr_1", "icr_2"))
        self.assertEqual(entropy_block.feature_names, ("ent_0", "ent_1", "ent_2", "ent_3"))
        self.assertEqual(delta_block.feature_names, ("dent_0", "dent_1", "dent_2"))
        self.assertEqual(icr_block.records[0].metadata, {})
        self.assertEqual(entropy_block.records[0].metadata, {})
        self.assertEqual(delta_block.records[0].metadata, {})


if __name__ == "__main__":
    unittest.main()
