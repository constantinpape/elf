import unittest
import numpy as np
try:
    import nifty
except ImportError:
    nifty = None


class TestLiftedMulticut(unittest.TestCase):

    # https://github.com/constantinpape/graph/blob/master/src/andres/graph/unit-test/multicut-lifted/kernighan-lin.cxx
    def toy_problem(self):
        graph = nifty.graph.undirectedGraph(5)
        edges = np.array([[0, 1],
                          [0, 3],
                          [1, 2],
                          [1, 4],
                          [3, 4]])
        graph.insertEdges(edges)
        costs = np.array([10, -1, 10, 4, 10], dtype='float32')

        # complete graph excluding `edges`
        lifted_edges = np.array([[0, 2],
                                 [0, 4],
                                 [1, 3],
                                 [2, 3],
                                 [2, 4],
                                 [3, 4]])
        lifted_costs = np.array([-1, -1, -1, -1, -1, 10], dtype='float32')

        expected_result = np.array([0, 1, 0, 1, 0], dtype='bool')  # edge result
        return graph, lifted_edges, costs, lifted_costs, expected_result

    def _test_lifted_multicut(self, solver):
        graph, lifted_edges, costs, lifted_costs, expected_result = self.toy_problem()
        node_labels = solver(graph, costs, lifted_edges, lifted_costs)
        uv_ids = graph.uvIds()
        result = node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]
        self.assertTrue(np.array_equal(result, expected_result))

    @unittest.skipUnless(nifty, "Need nifty for multicut functionality")
    def test_gaec(self):
        from elf.segmentation.lifted_multicut import lifted_multicut_gaec
        self._test_lifted_multicut(lifted_multicut_gaec)

    @unittest.skipUnless(nifty, "Need nifty for multicut functionality")
    def test_kernighan_lin(self):
        from elf.segmentation.lifted_multicut import lifted_multicut_kernighan_lin
        self._test_lifted_multicut(lifted_multicut_kernighan_lin)

    @unittest.skipUnless(nifty, "Need nifty for multicut functionality")
    def test_fusion_moves(self):
        from elf.segmentation.lifted_multicut import lifted_multicut_fusion_moves
        self._test_lifted_multicut(lifted_multicut_fusion_moves)


if __name__ == '__main__':
    unittest.main()
