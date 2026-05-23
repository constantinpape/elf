import os
import unittest
import numpy as np


def _make_seg_2d(shape, n_labels):
    """Build a simple 2D segmentation with the given number of labels."""
    seg = np.zeros(shape, dtype="uint32")
    block_h = shape[0] // n_labels
    for i in range(n_labels):
        start, stop = i * block_h, (i + 1) * block_h
        seg[start:stop] = i
    return seg


class TestUtilsOverlap(unittest.TestCase):

    def test_compute_maximum_label_overlap_2d(self):
        from elf.segmentation.utils import compute_maximum_label_overlap
        # build two segs where seg_b is a perfect renaming of seg_a
        seg_a = _make_seg_2d((40, 20), n_labels=4)
        rename = {0: 10, 1: 20, 2: 30, 3: 40}
        seg_b = np.zeros_like(seg_a)
        for src, dst in rename.items():
            seg_b[seg_a == src] = dst
        overlaps = compute_maximum_label_overlap(seg_a, seg_b)
        for src, dst in rename.items():
            self.assertEqual(int(overlaps[src]), dst)

    def test_compute_maximum_label_overlap_3d(self):
        from elf.segmentation.utils import compute_maximum_label_overlap
        seg_a = np.zeros((4, 6, 6), dtype="uint32")
        seg_a[1:] = 1
        seg_a[2:] = 2
        seg_a[3:] = 3
        seg_b = (seg_a + 10).astype("uint32")
        overlaps = compute_maximum_label_overlap(seg_a, seg_b)
        for src in range(4):
            self.assertEqual(int(overlaps[src]), src + 10)

    def test_compute_maximum_label_overlap_ignore_zeros(self):
        from elf.segmentation.utils import compute_maximum_label_overlap
        seg_a = np.array([[1, 1, 1, 1],
                          [1, 1, 1, 1]], dtype="uint32")
        # for label 1 in a, b has mostly 0 (5 voxels) and a single 7 (3 voxels)
        seg_b = np.array([[0, 0, 7, 7],
                          [0, 0, 0, 7]], dtype="uint32")
        # with ignore_zeros=False, best overlap is 0
        ov = compute_maximum_label_overlap(seg_a, seg_b, ignore_zeros=False)
        self.assertEqual(int(ov[1]), 0)
        # with ignore_zeros=True, best overlap is 7
        ov = compute_maximum_label_overlap(seg_a, seg_b, ignore_zeros=True)
        self.assertEqual(int(ov[1]), 7)


class TestUtilsNormalize(unittest.TestCase):

    def test_normalize_input(self):
        from elf.segmentation.utils import normalize_input
        inp = np.array([2.0, 4.0, 6.0, 8.0], dtype="float32")
        out = normalize_input(inp)
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out.min()), 0.0)
        # max is close to 1 (eps shifts it slightly)
        self.assertAlmostEqual(float(out.max()), 1.0, places=5)

    def test_map_background_to_zero(self):
        from elf.segmentation.utils import map_background_to_zero
        seg = np.array([5, 5, 5, 1, 2, 0], dtype="uint32")
        out = map_background_to_zero(seg.copy())
        # background 5 (most-common) is mapped to 0; original 0 is preserved as 5
        self.assertEqual(int((out == 0).sum()), 3)  # original 5s
        self.assertEqual(int((out == 5).sum()), 1)  # original 0

    def test_map_background_to_zero_explicit(self):
        from elf.segmentation.utils import map_background_to_zero
        seg = np.array([1, 2, 3, 0], dtype="uint32")
        out = map_background_to_zero(seg.copy(), background_label=2)
        self.assertEqual(int(out[1]), 0)  # 2 → 0
        self.assertEqual(int(out[3]), 2)  # 0 swapped into the freed 2


class TestUtilsEdges(unittest.TestCase):

    def test_sharpen_edges(self):
        from elf.segmentation.utils import sharpen_edges
        edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype="float32")
        out = sharpen_edges(edges.copy(), percentile=100)  # normalize by max
        self.assertAlmostEqual(float(out.max()), 1.0, places=5)
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_smooth_edges(self):
        from elf.segmentation.utils import smooth_edges
        edges = np.zeros((9, 9), dtype="float32")
        edges[4, :] = 1.0
        out = smooth_edges(edges)
        # edge pixels stay at exp(0) == 1; far pixels decay
        self.assertAlmostEqual(float(out[4, 4]), 1.0, places=5)
        self.assertLess(float(out[0, 0]), float(out[4, 4]))

    def test_seg_to_edges_2d(self):
        from elf.segmentation.utils import seg_to_edges
        seg = np.zeros((6, 6), dtype="uint32")
        seg[3:, :] = 1
        edges = seg_to_edges(seg)
        self.assertEqual(edges.shape, seg.shape)
        # the edge crosses the boundary
        self.assertTrue(edges[2:4, :].any())

    def test_seg_to_edges_3d(self):
        from elf.segmentation.utils import seg_to_edges
        seg = np.zeros((5, 5, 5), dtype="uint32")
        seg[:, 3:, :] = 1
        edges = seg_to_edges(seg)
        self.assertEqual(edges.shape, seg.shape)
        self.assertTrue(edges.any())

    def test_seg_to_edges_only_in_plane(self):
        from elf.segmentation.utils import seg_to_edges
        seg = np.zeros((5, 5, 5), dtype="uint32")
        # only an across-z boundary
        seg[3:, :, :] = 1
        full = seg_to_edges(seg)
        in_plane = seg_to_edges(seg, only_in_plane_edges=True)
        # only_in_plane misses the across-z transition
        self.assertGreater(int(full.sum()), 0)
        self.assertEqual(int(in_plane.sum()), 0)

    def test_seg_to_edges_invalid_dim(self):
        from elf.segmentation.utils import seg_to_edges
        with self.assertRaises(ValueError):
            seg_to_edges(np.zeros((4,), dtype="uint32"))


class TestUtilsAnalyse(unittest.TestCase):

    def test_analyse_multicut_problem(self):
        import nifty.graph
        from elf.segmentation.utils import analyse_multicut_problem
        # simple 4-node graph with two CCs at threshold 0
        n_nodes = 4
        uv_ids = np.array([[0, 1], [1, 2], [2, 3]], dtype="uint64")
        graph = nifty.graph.undirectedGraph(n_nodes)
        graph.insertEdges(uv_ids)
        costs = np.array([1.0, -2.0, 1.0], dtype="float32")  # cut in the middle
        df = analyse_multicut_problem(graph, costs, verbose=False)
        self.assertEqual(int(df["n_nodes"].iloc[0]), n_nodes)
        self.assertEqual(int(df["n_edges"].iloc[0]), 3)
        self.assertEqual(int(df["n_components"].iloc[0]), 2)


@unittest.skipUnless(os.environ.get("ELF_TEST_NETWORK"), "skipped without ELF_TEST_NETWORK=1")
class TestUtilsDownload(unittest.TestCase):

    def test_load_multicut_problem(self):
        from elf.segmentation.utils import load_multicut_problem
        graph, costs = load_multicut_problem("A", "small")
        self.assertGreater(graph.numberOfNodes, 0)
        self.assertEqual(len(costs), graph.numberOfEdges)

    def test_load_mutex_watershed_problem(self):
        from elf.segmentation.utils import load_mutex_watershed_problem
        affs, offsets = load_mutex_watershed_problem("test")
        self.assertEqual(affs.ndim, 4)
        self.assertEqual(len(offsets), affs.shape[0])


if __name__ == "__main__":
    unittest.main()
