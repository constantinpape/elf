import unittest

import numpy as np


class TestEdgeVisualisation(unittest.TestCase):
    def _get_seg_2d(self):
        seg = np.zeros((20, 20), dtype="uint32")
        seg[:10, :10] = 0
        seg[:10, 10:] = 1
        seg[10:, :10] = 2
        seg[10:, 10:] = 3
        return seg

    def _get_seg_3d(self):
        seg = np.zeros((8, 8, 8), dtype="uint32")
        seg[:4] = 0
        seg[4:] = 1
        seg[:, 4:] = np.where(seg[:, 4:] == 0, 2, 3)
        return seg

    def test_visualise_edges_2d(self):
        from elf.segmentation.features import compute_rag
        from elf.visualisation import visualise_edges

        seg = self._get_seg_2d()
        rag = compute_rag(seg)
        edge_values = np.random.rand(rag.number_of_edges).astype("float32")

        vol = visualise_edges(rag, seg, edge_values)
        self.assertEqual(vol.shape, seg.shape)
        self.assertEqual(vol.dtype, edge_values.dtype)
        # The vast majority of pixels are off-boundary and stay zero.
        self.assertTrue((vol == 0).sum() > vol.size // 2)

    def test_visualise_edges_3d(self):
        from elf.segmentation.features import compute_rag
        from elf.visualisation import visualise_edges

        seg = self._get_seg_3d()
        rag = compute_rag(seg)
        edge_values = np.random.rand(rag.number_of_edges).astype("float32")

        vol = visualise_edges(rag, seg, edge_values)
        self.assertEqual(vol.shape, seg.shape)
        self.assertEqual(vol.dtype, edge_values.dtype)

    def test_visualise_edges_ignore(self):
        from elf.segmentation.features import compute_rag
        from elf.visualisation import visualise_edges

        seg = self._get_seg_2d()
        rag = compute_rag(seg)
        edge_values = np.ones(rag.number_of_edges, dtype="float32")

        ignore_edges = np.zeros(rag.number_of_edges, dtype="bool")
        ignore_edges[0] = True
        vol = visualise_edges(rag, seg, edge_values, ignore_edges=ignore_edges)
        self.assertEqual(vol.shape, seg.shape)
        # The original edge_values must not be modified in-place.
        self.assertTrue(np.all(edge_values == 1))

    def test_visualise_attractive_and_repulsive_edges(self):
        from elf.segmentation.features import compute_rag
        from elf.visualisation import visualise_attractive_and_repulsive_edges

        seg = self._get_seg_2d()
        rag = compute_rag(seg)
        edge_values = np.random.rand(rag.number_of_edges).astype("float32")

        attractive, repulsive = visualise_attractive_and_repulsive_edges(
            rag, seg, edge_values, threshold=0.5
        )
        for vol in (attractive, repulsive):
            self.assertEqual(vol.shape, seg.shape)
            self.assertGreaterEqual(float(vol.min()), 0.0)
            self.assertLessEqual(float(vol.max()), 1.0)

    def test_visualise_attractive_and_repulsive_edges_all_one_side(self):
        # All values above the threshold -> the repulsive group is empty.
        # This used to crash in _scale_values on a zero-size array.
        from elf.segmentation.features import compute_rag
        from elf.visualisation import visualise_attractive_and_repulsive_edges

        seg = self._get_seg_2d()
        rag = compute_rag(seg)
        edge_values = np.full(rag.number_of_edges, 0.9, dtype="float32")

        attractive, repulsive = visualise_attractive_and_repulsive_edges(
            rag, seg, edge_values, threshold=0.5
        )
        self.assertEqual(attractive.shape, seg.shape)
        self.assertEqual(repulsive.shape, seg.shape)

    def test_assert_on_size_mismatch(self):
        from elf.segmentation.features import compute_rag
        from elf.visualisation import visualise_edges

        seg = self._get_seg_2d()
        rag = compute_rag(seg)
        edge_values = np.random.rand(rag.number_of_edges + 1).astype("float32")
        with self.assertRaises(AssertionError):
            visualise_edges(rag, seg, edge_values)

    def test_scale_values(self):
        from elf.visualisation.edge_visualisation import _scale_values

        values = np.array([0.6, 0.8, 1.0], dtype="float32")
        scaled = _scale_values(values.copy(), threshold=0.5, invert=False)
        self.assertAlmostEqual(float(scaled.min()), 0.0)
        self.assertAlmostEqual(float(scaled.max()), 1.0)

        # Empty input must be returned as-is without raising.
        empty = np.array([], dtype="float32")
        self.assertEqual(_scale_values(empty, threshold=0.5, invert=False).size, 0)


if __name__ == "__main__":
    unittest.main()
