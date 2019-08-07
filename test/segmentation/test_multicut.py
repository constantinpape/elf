import unittest
import numpy as np
try:
    import nifty
except ImportError:
    nifty = None


class TestMulticut(unittest.TestCase):

    # cf https://github.com/constantinpape/graph/blob/master/src/andres/graph/unit-test/multicut/kernighan-lin.cxx
    def toy_problem(self):
        graph = nifty.graph.undirectedGraph(6)
        edges = np.array([[0, 1],
                          [0, 3],
                          [1, 2],
                          [1, 4],
                          [2, 5],
                          [3, 4],
                          [4, 5]])
        graph.insertEdges(edges)
        costs = np.array([5, -20, 5, 5, -20, 5, 5], dtype='float32')
        expected_result = np.array([0, 1, 0, 1, 1, 0, 0], dtype='bool')  # edge result
        return graph, costs, expected_result

    def _test_multicut(self, solver):
        graph, costs, expected_result = self.toy_problem()
        node_labels = solver(graph, costs)
        uv_ids = graph.uvIds()
        result = node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]
        self.assertTrue(np.array_equal(result, expected_result))

    @unittest.skipUnless(nifty, "Need nifty for multicut functionality")
    def test_gaec(self):
        from elf.segmentation.multicut import multicut_gaec
        self._test_multicut(multicut_gaec)

    @unittest.skipUnless(nifty, "Need nifty for multicut functionality")
    def test_kernighan_lin(self):
        from elf.segmentation.multicut import multicut_kernighan_lin
        self._test_multicut(multicut_kernighan_lin)

    @unittest.skipUnless(nifty, "Need nifty for multicut functionality")
    def test_decomposition(self):
        from elf.segmentation.multicut import multicut_decomposition
        self._test_multicut(multicut_decomposition)

    @unittest.skipUnless(nifty, "Need nifty for multicut functionality")
    def test_fusion_moves(self):
        from elf.segmentation.multicut import multicut_fusion_moves
        self._test_multicut(multicut_fusion_moves)

    @unittest.skipUnless(nifty, "Need nifty for multicut functionality")
    def test_transform_probabilities_to_costs(self):
        from elf.segmentation.multicut import transform_probabilities_to_costs
        probs = np.random.rand(100).astype('float32')
        costs = transform_probabilities_to_costs(probs)
        self.assertTrue(np.isfinite(costs).all())


if __name__ == '__main__':
    unittest.main()
