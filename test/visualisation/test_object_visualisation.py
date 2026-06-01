import unittest

import numpy as np


class TestObjectVisualisation(unittest.TestCase):
    def _get_data_2d(self):
        seg = np.zeros((30, 30), dtype="uint32")
        seg[2:10, 2:10] = 1
        seg[2:10, 15:25] = 2
        seg[15:25, 2:10] = 3
        gt = np.zeros((30, 30), dtype="uint32")
        gt[2:10, 2:10] = 1     # perfect overlap with object 1
        gt[2:11, 15:26] = 2    # partial overlap with object 2
        gt[18:25, 2:10] = 3    # partial overlap with object 3
        return seg, gt

    def _get_data_3d(self):
        seg = np.zeros((6, 12, 12), dtype="uint32")
        seg[1:4, 1:5, 1:5] = 1
        seg[1:4, 7:11, 7:11] = 2
        gt = np.zeros((6, 12, 12), dtype="uint32")
        gt[1:4, 1:5, 1:5] = 1
        gt[1:4, 7:11, 7:11] = 2
        return seg, gt

    def test_visualise_iou_scores_2d(self):
        from elf.visualisation import visualise_iou_scores

        seg, gt = self._get_data_2d()
        scores = visualise_iou_scores(seg, gt)
        self.assertEqual(scores.shape, seg.shape)
        self.assertEqual(scores.dtype, np.float32)
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 1.0)
        # Background must be zero.
        self.assertEqual(float(scores[0, 0]), 0.0)
        # The perfectly matched object should score 1.0.
        self.assertAlmostEqual(float(scores[5, 5]), 1.0, places=5)

    def test_visualise_iou_scores_3d(self):
        from elf.visualisation import visualise_iou_scores

        seg, gt = self._get_data_3d()
        scores = visualise_iou_scores(seg, gt)
        self.assertEqual(scores.shape, seg.shape)
        self.assertEqual(scores.dtype, np.float32)
        self.assertAlmostEqual(float(scores.max()), 1.0, places=5)

    def test_visualise_dice_scores(self):
        from elf.visualisation import visualise_dice_scores

        seg, gt = self._get_data_2d()
        scores = visualise_dice_scores(seg, gt)
        self.assertEqual(scores.shape, seg.shape)
        self.assertEqual(scores.dtype, np.float32)
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 1.0)
        self.assertEqual(float(scores[0, 0]), 0.0)

    def test_visualise_voi_scores(self):
        from elf.visualisation import visualise_voi_scores

        seg, gt = self._get_data_2d()
        for voi in ("split", "merge", "full"):
            scores = visualise_voi_scores(seg, gt, voi=voi)
            self.assertEqual(scores.shape, seg.shape)
            self.assertEqual(scores.dtype, np.float32)

    def test_visualise_voi_invalid(self):
        from elf.visualisation import visualise_voi_scores

        seg, gt = self._get_data_2d()
        with self.assertRaises(AssertionError):
            visualise_voi_scores(seg, gt, voi="invalid")


if __name__ == "__main__":
    unittest.main()
