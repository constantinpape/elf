import unittest
from functools import partial

import numpy as np

try:
    import bioimage_cpp as bic
except ImportError:
    bic = None


class TestClustering(unittest.TestCase):
    n_nodes = 250
    n_edges = 500

    def random_problem(self):
        rng = np.random.default_rng(42)
        seen = set()
        edges = []
        while len(edges) < self.n_edges:
            u, v = rng.integers(0, self.n_nodes, size=2)
            u, v = int(min(u, v)), int(max(u, v))
            if u == v or (u, v) in seen:
                continue
            seen.add((u, v))
            edges.append((u, v))
        uvs = np.asarray(edges, dtype="uint64")
        graph = bic.graph.UndirectedGraph.from_edges(self.n_nodes, uvs)
        features = rng.random(self.n_edges).astype("float32")
        return graph, features

    def _test_clustering(self, agglomerator):
        graph, features = self.random_problem()
        node_labels = agglomerator(graph, edge_features=features,
                                   edge_sizes=np.ones(self.n_edges, dtype="uint32"))
        self.assertEqual(len(node_labels), self.n_nodes)
        self.assertGreater(node_labels.max(), 2)  # make sure solution is not trivial
        return node_labels

    @unittest.skipUnless(bic, "Need bioimage-cpp for clustering functionality")
    def test_mala_clustering(self):
        from elf.segmentation.clustering import mala_clustering
        self._test_clustering(partial(mala_clustering, threshold=.75))

    @unittest.skipUnless(bic, "Need bioimage-cpp for clustering functionality")
    def test_agglomerative_clustering(self):
        from elf.segmentation.clustering import agglomerative_clustering
        n_clusters = 50
        labels = self._test_clustering(partial(agglomerative_clustering,
                                               node_sizes=np.ones(self.n_nodes, dtype="uint32"),
                                               n_stop=n_clusters, size_regularizer=1.))
        self.assertEqual(len(np.unique(labels)), n_clusters)

    @unittest.skipUnless(bic, "Need bioimage-cpp for clustering functionality")
    def test_cluster_segmentation(self):
        from elf.segmentation.clustering import cluster_segmentation
        rng = np.random.default_rng(0)
        # Block-structured 4x16x16 segmentation with ~32 distinct labels.
        shape = (4, 16, 16)
        seg = np.arange(int(np.prod(shape)) // 8, dtype="uint32").repeat(8).reshape(shape)
        seg = seg % 32
        boundary_map = rng.random(shape).astype("float32")
        result = cluster_segmentation(seg, boundary_map, n_stop=5, size_regularizer=0.5)
        self.assertEqual(result.shape, shape)
        # All resulting labels should be >= 1 since we relabel starting at 1.
        self.assertTrue((result >= 1).all())

    @unittest.skipUnless(bic, "Need bioimage-cpp for clustering functionality")
    def test_cluster_segmentation_mala(self):
        from elf.segmentation.clustering import cluster_segmentation_mala
        rng = np.random.default_rng(1)
        shape = (4, 16, 16)
        seg = np.arange(int(np.prod(shape)) // 8, dtype="uint32").repeat(8).reshape(shape)
        seg = seg % 32
        boundary_map = rng.random(shape).astype("float32")
        result = cluster_segmentation_mala(seg, boundary_map, threshold=0.6)
        self.assertEqual(result.shape, shape)
        self.assertTrue((result >= 1).all())


if __name__ == "__main__":
    unittest.main()
