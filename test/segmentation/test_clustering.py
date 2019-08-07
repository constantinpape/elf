import unittest
from functools import partial
import numpy as np
try:
    import nifty
except ImportError:
    nifty = None


class TestClustering(unittest.TestCase):
    n_nodes = 250
    n_edges = 500

    def random_problem(self):
        graph = nifty.graph.undirectedGraph(self.n_nodes)
        while graph.numberOfEdges < self.n_edges:
            u, v = np.random.randint(0, self.n_nodes, size=2)
            u, v = min(u, v), max(u, v)
            if u == v or graph.findEdge(u, v) != -1:
                continue
            graph.insertEdge(u, v)
        features = np.random.rand(self.n_edges).astype('float32')
        return graph, features

    def _test_clustering(self, agglomerator):
        graph, features = self.random_problem()
        node_labels = agglomerator(graph, edge_features=features,
                                   edge_sizes=np.ones(self.n_edges, dtype='uint32'))
        self.assertEqual(len(node_labels), self.n_nodes)
        self.assertGreater(node_labels.max(), 2)  # make sure solution is not trivial
        return node_labels

    @unittest.skipUnless(nifty, "Need nifty for clustering functionality")
    def test_mala_clustering(self):
        from elf.segmentation.clustering import mala_clustering
        self._test_clustering(partial(mala_clustering, threshold=.75))

    @unittest.skipUnless(nifty, "Need nifty for clustering functionality")
    def test_agglomerative_clustering(self):
        from elf.segmentation.clustering import agglomerative_clustering
        n_clusters = 50
        labels = self._test_clustering(partial(agglomerative_clustering,
                                               node_sizes=np.ones(self.n_nodes, dtype='uint32'),
                                               n_stop=n_clusters, size_regularizer=1.))
        self.assertEqual(len(np.unique(labels)), n_clusters)


if __name__ == '__main__':
    unittest.main()
