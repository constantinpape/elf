import unittest
import numpy as np


class TestPostprocess(unittest.TestCase):

    def _make_graph(self):
        # 6-node chain: 0-1-2-3-4-5
        import bioimage_cpp as bic
        uv = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype="uint64")
        graph = bic.graph.UndirectedGraph.from_edges(6, uv)
        return graph

    def test_graph_watershed(self):
        from elf.segmentation.postprocess import graph_watershed
        graph = self._make_graph()
        weights = np.array([0.1, 0.5, 0.05, 0.5, 0.1], dtype="float32")
        # Seeds at the two ends: 1 at node 0, 2 at node 5.
        seeds = np.array([1, 0, 0, 0, 0, 2], dtype="uint64")
        labels = graph_watershed(graph, weights, seeds)
        self.assertEqual(len(labels), 6)
        # node 0 and 5 must keep their seed
        self.assertEqual(int(labels[0]), 1)
        self.assertEqual(int(labels[5]), 2)
        # the middle should split somewhere
        self.assertIn(int(labels[1]), {1, 2})
        self.assertIn(int(labels[4]), {1, 2})
        # no zero left
        self.assertEqual(int((labels == 0).sum()), 0)

    def test_graph_size_filter(self):
        from elf.segmentation.postprocess import graph_size_filter
        graph = self._make_graph()
        weights = np.array([0.1, 0.5, 0.05, 0.5, 0.1], dtype="float32")
        # Each node is its own region (id 0..5) with sizes [10, 10, 1, 1, 10, 10].
        node_sizes = np.array([10, 10, 1, 1, 10, 10], dtype="uint64")
        labels = graph_size_filter(graph, weights, node_sizes, min_size=5)
        self.assertEqual(len(labels), 6)
        # The two small nodes should be merged into a neighbour.
        # The two end big nodes 0 and 5 are seeds.
        self.assertEqual(int((labels == 0).sum()), 0)

    def test_graph_size_filter_with_node_labels(self):
        from elf.segmentation.postprocess import graph_size_filter
        graph = self._make_graph()
        weights = np.array([0.1, 0.5, 0.05, 0.5, 0.1], dtype="float32")
        node_labels = np.array([1, 1, 2, 2, 3, 3], dtype="uint64")
        # sizes for each LABEL id (0..max_label)
        node_sizes = np.array([0, 20, 1, 20], dtype="uint64")
        labels = graph_size_filter(graph, weights, node_sizes, min_size=5, node_labels=node_labels)
        self.assertEqual(len(labels), 6)
        # label 2 was discarded; nodes 2 and 3 must be absorbed into 1 or 3
        self.assertTrue(int((labels == 2).sum()) == 0)

    def test_graph_size_filter_with_relabel(self):
        from elf.segmentation.postprocess import graph_size_filter
        graph = self._make_graph()
        weights = np.array([0.1, 0.5, 0.05, 0.5, 0.1], dtype="float32")
        node_sizes = np.array([10, 10, 1, 1, 10, 10], dtype="uint64")
        labels = graph_size_filter(graph, weights, node_sizes, min_size=5, relabel=True)
        self.assertEqual(len(labels), 6)


if __name__ == "__main__":
    unittest.main()
