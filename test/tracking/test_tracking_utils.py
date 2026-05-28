import unittest

import numpy as np


class TestTrackingUtils(unittest.TestCase):
    def _get_segmentation(self):
        # Three timepoints, each containing a single 4x4 object with a unique id.
        # The objects move diagonally so that consecutive frames overlap partially.
        seg = np.zeros((3, 12, 12), dtype="uint32")
        seg[0, 2:6, 2:6] = 1    # centroid (3.5, 3.5)
        seg[1, 3:7, 3:7] = 2    # centroid (4.5, 4.5), overlaps id 1 in a 3x3 region
        seg[2, 8:11, 8:11] = 3  # centroid (9.0, 9.0), no overlap with id 2
        return seg

    def test_compute_edges_from_overlap(self):
        from elf.tracking.tracking_utils import compute_edges_from_overlap
        seg = self._get_segmentation()
        edges = compute_edges_from_overlap(seg, verbose=False)

        # Each edge needs source, target and score; targets must never be background.
        for edge in edges:
            self.assertEqual(set(edge.keys()), {"source", "target", "score"})
            self.assertNotEqual(edge["target"], 0)
            self.assertGreater(edge["score"], 0.0)
            self.assertLessEqual(edge["score"], 1.0)

        # id 1 and id 2 overlap in a 3x3 region; id 1 has 16 pixels -> fraction 9/16.
        # id 2 and id 3 do not overlap, so there is exactly one edge.
        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(edge["source"], 1)
        self.assertEqual(edge["target"], 2)
        self.assertAlmostEqual(edge["score"], 9.0 / 16.0)

    def test_compute_edges_from_centroid_distance(self):
        from elf.tracking.tracking_utils import compute_edges_from_centroid_distance
        seg = self._get_segmentation()

        # Distances: id1->id2 = sqrt(2), id2->id3 = sqrt(4.5**2 + 4.5**2).
        dist_12 = np.sqrt(2.0)
        dist_23 = np.sqrt(2 * 4.5 ** 2)

        # Without normalization the scores are the raw distances.
        edges = compute_edges_from_centroid_distance(
            seg, max_distance=10.0, normalize_distances=False, verbose=False
        )
        self.assertEqual(len(edges), 2)
        scores = {(e["source"], e["target"]): e["score"] for e in edges}
        self.assertAlmostEqual(scores[(1, 2)], dist_12)
        self.assertAlmostEqual(scores[(2, 3)], dist_23)

        # max_distance filters out the larger distance edge.
        edges_filtered = compute_edges_from_centroid_distance(
            seg, max_distance=2.0, normalize_distances=False, verbose=False
        )
        self.assertEqual(len(edges_filtered), 1)
        self.assertEqual(edges_filtered[0]["source"], 1)
        self.assertEqual(edges_filtered[0]["target"], 2)

        # Normalized scores lie in [0, 1] and the closer pair gets the higher score.
        edges_norm = compute_edges_from_centroid_distance(
            seg, max_distance=10.0, normalize_distances=True, verbose=False
        )
        scores_norm = {(e["source"], e["target"]): e["score"] for e in edges_norm}
        for score in scores_norm.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        self.assertGreater(scores_norm[(1, 2)], scores_norm[(2, 3)])

    def test_compute_node_costs_from_foreground_probabilities(self):
        from elf.tracking.tracking_utils import compute_node_costs_from_foreground_probabilities
        seg = self._get_segmentation()

        # Constant probability per object -> mean intensity equals that constant.
        probabilities = np.zeros(seg.shape, dtype="float32")
        probabilities[seg == 1] = 0.2
        probabilities[seg == 2] = 0.5
        probabilities[seg == 3] = 0.8

        costs = compute_node_costs_from_foreground_probabilities(seg, probabilities)
        self.assertEqual(len(costs), 3)
        np.testing.assert_allclose(costs, [0.2, 0.5, 0.8], rtol=1e-6)

    def test_relabel_segmentation_across_time(self):
        from elf.tracking.tracking_utils import relabel_segmentation_across_time
        # Reuse the same id 1 in every frame.
        seg = np.zeros((3, 8, 8), dtype="uint32")
        seg[0, 1:4, 1:4] = 1
        seg[1, 2:5, 2:5] = 1
        seg[2, 3:6, 3:6] = 1

        relabeled = relabel_segmentation_across_time(seg)
        self.assertEqual(relabeled.shape, seg.shape)

        # Background is preserved and foreground ids are unique across all frames.
        self.assertTrue(np.array_equal(relabeled == 0, seg == 0))
        foreground_ids = [np.unique(frame[frame != 0]) for frame in relabeled]
        for ids in foreground_ids:
            self.assertEqual(len(ids), 1)
        all_ids = np.concatenate(foreground_ids)
        self.assertEqual(len(all_ids), len(np.unique(all_ids)))

    def test_preprocess_closing_fills_gap(self):
        from elf.tracking.tracking_utils import preprocess_closing
        # Object present in slice 0 and slice 2, missing in slice 1.
        seg = np.zeros((3, 10, 10), dtype="uint32")
        seg[0, 3:7, 3:7] = 1
        seg[2, 3:7, 3:7] = 1
        self.assertEqual(seg[1].sum(), 0)

        closed = preprocess_closing(seg, gap_closing=1, verbose=False)
        self.assertEqual(closed.shape, seg.shape)
        # The gap in slice 1 has been filled in the object region.
        self.assertGreater(np.count_nonzero(closed[1, 3:7, 3:7]), 0)

    def test_preprocess_closing_prevents_merge(self):
        from elf.tracking.tracking_utils import preprocess_closing
        # Two touching objects in the processed slice. Closing along z must not
        # merge them into a single object: the initial segmentation is kept.
        seg = np.zeros((3, 10, 10), dtype="uint32")
        seg[1, 3:7, 2:5] = 1
        seg[1, 3:7, 5:8] = 2

        closed = preprocess_closing(seg, gap_closing=1, verbose=False)
        n_labels = len(np.unique(closed[1])) - 1  # subtract background
        self.assertEqual(n_labels, 2)


if __name__ == "__main__":
    unittest.main()
