import unittest

import networkx as nx
import numpy as np

try:
    import motile
except ImportError:
    motile = None


class TestMotileTracking(unittest.TestCase):
    def _get_linear_segmentation(self):
        # A single object that overlaps across three timepoints -> one track.
        seg = np.zeros((3, 10, 10), dtype="uint32")
        seg[0, 2:6, 2:6] = 1
        seg[1, 2:6, 2:6] = 1
        seg[2, 2:6, 2:6] = 1
        return seg

    def _get_division_lineage(self):
        # Lineage with a division at node 2: 1 -> 2, 2 -> {3, 4}, 4 -> 5.
        lineage_graph = nx.DiGraph()
        lineage_graph.add_nodes_from([1, 2, 3, 4, 5])
        lineage_graph.add_edges_from([(1, 2), (2, 3), (2, 4), (4, 5)])
        node_to_track = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
        return lineage_graph, node_to_track

    def test_get_node_assignment(self):
        from elf.tracking.motile_tracking import get_node_assignment
        node_ids = [1, 2, 3, 4]
        assignments = {1: [1, 2], 2: [3]}
        node_assignment = get_node_assignment(node_ids, assignments)
        self.assertEqual(node_assignment[1], 1)
        self.assertEqual(node_assignment[2], 1)
        self.assertEqual(node_assignment[3], 2)
        # Node 4 was not selected and must be mapped to background.
        self.assertEqual(node_assignment[4], 0)

    def test_recolor_segmentation(self):
        from elf.tracking.motile_tracking import recolor_segmentation
        seg = np.zeros((4, 4), dtype="uint32")
        seg[0] = 1
        seg[1] = 2
        seg[2] = 3
        node_to_assignment = {1: 10, 2: 20, 3: 0}
        recolored = recolor_segmentation(seg, node_to_assignment)
        self.assertEqual(recolored.shape, seg.shape)
        self.assertTrue(np.all(recolored[0] == 10))
        self.assertTrue(np.all(recolored[1] == 20))
        self.assertTrue(np.all(recolored[2] == 0))
        self.assertTrue(np.all(recolored[3] == 0))

    def test_lineage_graph_to_track_graph(self):
        from elf.tracking.motile_tracking import lineage_graph_to_track_graph
        lineage_graph, _ = self._get_division_lineage()
        track_graph, tracks = lineage_graph_to_track_graph(lineage_graph, None)

        # The track is broken at the division (node 2), giving tracks {1, 2}, {3}, {4, 5}.
        self.assertEqual(len(tracks), 3)
        track_sets = sorted([sorted(nodes) for nodes in tracks.values()])
        self.assertEqual(track_sets, [[1, 2], [3], [4, 5]])

    def test_create_data_for_track_layer(self):
        from elf.tracking.motile_tracking import create_data_for_track_layer
        lineage_graph, node_to_track = self._get_division_lineage()

        # Segmentation with one object per node placed in successive timepoints.
        seg = np.zeros((4, 12, 12), dtype="uint32")
        seg[0, 1:4, 1:4] = 1
        seg[1, 1:4, 1:4] = 2
        seg[2, 1:4, 1:4] = 3
        seg[2, 8:11, 8:11] = 4
        seg[3, 8:11, 8:11] = 5

        track_data, parent_graph = create_data_for_track_layer(seg, lineage_graph, node_to_track)
        # One row per node, columns: track_id, t, y, x.
        self.assertEqual(track_data.shape, (5, 4))
        # The division at node 2 (track 1) is the parent of tracks 2 and 3.
        self.assertEqual(parent_graph, {2: [1], 3: [1]})

    @unittest.skipUnless(motile, "Needs motile")
    def test_track_with_motile(self):
        from elf.tracking.motile_tracking import track_with_motile, get_representation_for_napari
        seg = self._get_linear_segmentation()

        segmentation, lineage_graph, lineages, track_graph, tracks = track_with_motile(seg)
        self.assertEqual(segmentation.shape, seg.shape)
        # After relabeling the ids are unique per frame.
        self.assertEqual(len(np.unique(segmentation)) - 1, 3)
        # The overlapping object yields a single lineage / track spanning all frames.
        self.assertEqual(len(lineages), 1)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(next(iter(tracks.values()))), 3)

        tracking_result, track_data, parent_graph = get_representation_for_napari(
            segmentation, lineage_graph, lineages, tracks
        )
        self.assertEqual(tracking_result.shape, seg.shape)
        self.assertEqual(track_data.shape, (3, 4))
        # A linear track has no divisions, so there is no parent graph.
        self.assertEqual(parent_graph, {})


if __name__ == "__main__":
    unittest.main()
