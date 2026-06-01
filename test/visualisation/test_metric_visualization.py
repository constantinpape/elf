import unittest

import numpy as np


class TestMetricVisualization(unittest.TestCase):
    def test_compute_matches(self):
        from elf.visualisation.metric_visualization import _compute_matches

        # Contiguous prediction / ground-truth ids (as produced by relabel_sequential).
        prediction = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [2, 2, 0, 0],
        ], dtype="uint32")
        ground_truth = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 3, 3],
        ], dtype="uint32")

        # overlap matrix indexed [pred_id, gt_id]; gt has ids 0, 1, 3.
        overlap_matrix = np.zeros((3, 4), dtype="float32")
        overlap_matrix[1, 1] = 0.9  # pred 1 strongly overlaps gt 1 -> TP
        overlap_matrix[2, 0] = 0.2  # pred 2 only overlaps background -> FP

        tp, fp, fn, ids = _compute_matches(prediction, ground_truth, overlap_matrix, 0.5)

        # Id classification (background id 0 is always stripped).
        np.testing.assert_array_equal(ids["tp"], np.array([1]))
        np.testing.assert_array_equal(ids["fp"], np.array([2]))
        self.assertIn(3, ids["fn"])
        self.assertNotIn(0, ids["fn"])

        # Masks must be boolean and have the input shape.
        for mask in (tp, fp, fn):
            self.assertEqual(mask.shape, prediction.shape)
            self.assertEqual(mask.dtype, np.bool_)

        # TP mask covers the prediction-1 region, FP the prediction-2 region.
        self.assertTrue(np.array_equal(tp, prediction == 1))
        self.assertTrue(np.array_equal(fp, prediction == 2))
        # FN mask covers the unmatched ground-truth region (id 3).
        self.assertTrue(np.array_equal(fn, ground_truth == 3))

    def test_compute_matches_threshold(self):
        from elf.visualisation.metric_visualization import _compute_matches

        prediction = np.array([[0, 1, 1], [0, 1, 1]], dtype="uint32")
        ground_truth = np.array([[0, 1, 1], [0, 1, 1]], dtype="uint32")
        overlap_matrix = np.zeros((2, 2), dtype="float32")
        overlap_matrix[1, 1] = 0.6

        # Below the value -> match counts as TP.
        _, _, _, ids_low = _compute_matches(prediction, ground_truth, overlap_matrix, 0.5)
        np.testing.assert_array_equal(ids_low["tp"], np.array([1]))
        self.assertEqual(len(ids_low["fp"]), 0)

        # Above the value -> no match, prediction becomes FP and gt becomes FN.
        _, _, _, ids_high = _compute_matches(prediction, ground_truth, overlap_matrix, 0.7)
        self.assertEqual(len(ids_high["tp"]), 0)
        np.testing.assert_array_equal(ids_high["fp"], np.array([1]))
        np.testing.assert_array_equal(ids_high["fn"], np.array([1]))


if __name__ == "__main__":
    unittest.main()
