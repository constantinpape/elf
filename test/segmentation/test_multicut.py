import unittest
import numpy as np
try:
    import nifty
    import nifty.graph.opt.multicut as nmc
except ImportError:
    nmc = None


class TestMulticut(unittest.TestCase):
    # TODO download from Paul's website instead of hard-coding
    problem_root = '/home/pape/Work/data/multicut_problems/small/small_problem_sampleA.txt'
    upper_bound = -76900

    @staticmethod
    def load_problem(path):
        edges = []
        costs = []
        with open(path, 'r') as f:
            for l in f:
                r = l.split()
                edges.append([int(r[0]), int(r[1])])
                costs.append(float(r[2]))
        edges = np.array(edges, dtype='uint64')
        edges = np.sort(edges, axis=1)
        costs = np.array(costs)
        n_nodes = int(edges.max()) + 1
        graph = nifty.graph.undirectedGraph(n_nodes)
        graph.insertEdges(edges)
        return graph, costs

    # https://github.com/constantinpape/graph/blob/master/src/andres/graph/unit-test/multicut/kernighan-lin.cxx
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

    def _test_multicut_toy(self, solver):
        graph, costs, expected_result = self.toy_problem()
        node_labels = solver(graph, costs)
        uv_ids = graph.uvIds()
        result = node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]
        self.assertTrue(np.array_equal(result, expected_result))

    def _test_multicut(self, solver):
        # TODO remove the try-except once download is implemented
        try:
            graph, costs = self.load_problem()
        except FileNotFoundError:
            return
        node_labels = solver(graph, costs)
        obj = nmc.multicutObjective(graph, costs)
        energy = obj.evalNodeLabels(node_labels)
        self.assertGreater(self.upper_bound, energy)

    @unittest.skipUnless(nmc, "Need nifty for multicut functionality")
    def test_gaec(self):
        from elf.segmentation.multicut import multicut_gaec
        self._test_multicut(multicut_gaec)

    @unittest.skipUnless(nmc, "Need nifty for multicut functionality")
    def test_gaec_toy(self):
        from elf.segmentation.multicut import multicut_gaec
        self._test_multicut_toy(multicut_gaec)

    @unittest.skipUnless(nmc, "Need nifty for multicut functionality")
    def test_kernighan_lin(self):
        from elf.segmentation.multicut import multicut_kernighan_lin
        self._test_multicut(multicut_kernighan_lin)

    @unittest.skipUnless(nmc, "Need nifty for multicut functionality")
    def test_kernighan_lin_toy(self):
        from elf.segmentation.multicut import multicut_kernighan_lin
        self._test_multicut_toy(multicut_kernighan_lin)

    @unittest.skipUnless(nmc, "Need nifty for multicut functionality")
    def test_decomposition(self):
        from elf.segmentation.multicut import multicut_decomposition
        self._test_multicut(multicut_decomposition)

    @unittest.skipUnless(nmc, "Need nifty for multicut functionality")
    def test_decomposition_toy(self):
        from elf.segmentation.multicut import multicut_decomposition
        self._test_multicut_toy(multicut_decomposition)

    @unittest.skipUnless(nmc, "Need nifty for multicut functionality")
    def test_fusion_moves(self):
        from elf.segmentation.multicut import multicut_fusion_moves
        self._test_multicut(multicut_fusion_moves)

    @unittest.skipUnless(nmc, "Need nifty for multicut functionality")
    def test_fusion_moves_toy(self):
        from elf.segmentation.multicut import multicut_fusion_moves
        self._test_multicut_toy(multicut_fusion_moves)

    @unittest.skipUnless(nmc, "Need nifty for multicut functionality")
    def test_transform_probabilities_to_costs(self):
        from elf.segmentation.multicut import transform_probabilities_to_costs
        probs = np.random.rand(100).astype('float32')
        costs = transform_probabilities_to_costs(probs)
        self.assertTrue(np.isfinite(costs).all())


if __name__ == '__main__':
    unittest.main()
