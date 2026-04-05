"""Tests for fine-grained disagreement analysis helpers."""

from __future__ import annotations

import unittest

import numpy as np

from cut_a_lab.analysis.disagreement_analysis_finegrained import (
    compute_localized_features,
    compute_sliding_window_matrix,
)


class FineGrainedDisagreementAnalysisTests(unittest.TestCase):
    def test_compute_sliding_window_matrix_uses_requested_window(self) -> None:
        matrix = np.array(
            [
                [1.0, 3.0, 5.0, 7.0],
                [2.0, 4.0, 6.0, 8.0],
            ],
            dtype=np.float64,
        )

        windows, metadata = compute_sliding_window_matrix(matrix, window_size=2, stride=1)

        np.testing.assert_allclose(
            windows,
            np.array(
                [
                    [2.0, 4.0, 6.0],
                    [3.0, 5.0, 7.0],
                ],
                dtype=np.float64,
            ),
        )
        self.assertEqual(
            metadata,
            [
                {"index": 0, "start_layer": 0, "end_layer": 1, "label": "L0-1"},
                {"index": 1, "start_layer": 1, "end_layer": 2, "label": "L1-2"},
                {"index": 2, "start_layer": 2, "end_layer": 3, "label": "L2-3"},
            ],
        )

    def test_compute_localized_features_tracks_min_and_drop_timing(self) -> None:
        matrix = np.array(
            [
                [5.0, 4.0, 2.0, 3.0],
                [2.0, 2.5, 2.4, 1.0],
            ],
            dtype=np.float64,
        )

        features = compute_localized_features(matrix)

        np.testing.assert_array_equal(features["argmin_layer"], np.array([2.0, 3.0]))
        np.testing.assert_allclose(features["min_value"], np.array([2.0, 1.0]))
        np.testing.assert_array_equal(features["max_drop_layer"], np.array([1.0, 2.0]))
        np.testing.assert_allclose(features["max_drop_value"], np.array([2.0, 1.4]))
        np.testing.assert_allclose(features["last_minus_first"], np.array([-2.0, -1.0]))


if __name__ == "__main__":
    unittest.main()
