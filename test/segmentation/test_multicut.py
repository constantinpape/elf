import os
import urllib.request

import unittest
import numpy as np
import nifty
import nifty.graph.opt.multicut as nmc


class TestMulticut(unittest.TestCase):
    upper_bound = -76900
    problem_url = 'https://oc.embl.de/index.php/s/yVKwyQ8VoPXYkft/download'
    problem_path = './tmp_mc_problem.txt'

    @classmethod
    def setUpClass(cls):
        with urllib.request.urlopen(cls.problem_url) as f:
            problem = f.read().decode('utf-8')
        with open(cls.problem_path, 'w') as f:
            f.write(problem)

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.problem_path)
        except OSError:
            pass

    @staticmethod
    def load_problem(path):
        problem = np.genfromtxt(path)
        uv_ids = problem[:, :2].astype('uint64')
        n_nodes = int(uv_ids.max()) + 1
        graph = nifty.graph.undirectedGraph(n_nodes)
        graph.insertEdges(uv_ids)
        costs = problem[:, -1].astype('float32')
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

    def _test_multicut(self, solver, **kwargs):
        graph, costs = self.load_problem(self.problem_path)
        node_labels = solver(graph, costs, **kwargs)
        obj = nmc.multicutObjective(graph, costs)
        energy = obj.evalNodeLabels(node_labels)
        self.assertGreater(self.upper_bound, energy)

    def test_gaec(self):
        from elf.segmentation.multicut import multicut_gaec
        self._test_multicut(multicut_gaec)

    def test_gaec_toy(self):
        from elf.segmentation.multicut import multicut_gaec
        self._test_multicut_toy(multicut_gaec)

    def test_kernighan_lin(self):
        from elf.segmentation.multicut import multicut_kernighan_lin
        self._test_multicut(multicut_kernighan_lin)

    def test_kernighan_lin_toy(self):
        from elf.segmentation.multicut import multicut_kernighan_lin
        self._test_multicut_toy(multicut_kernighan_lin)

    def test_decomposition(self):
        from elf.segmentation.multicut import multicut_decomposition
        self._test_multicut(multicut_decomposition)

    def test_decomposition_toy(self):
        from elf.segmentation.multicut import multicut_decomposition
        self._test_multicut_toy(multicut_decomposition)

    @unittest.skip("Fusion moves take to long for CI")
    def test_fusion_moves(self):
        from elf.segmentation.multicut import multicut_fusion_moves
        self._test_multicut(multicut_fusion_moves, internal_solver='greedy-additive',
                            num_it=250, num_it_stop=10, seed_fraction=.1)

    def test_fusion_moves_toy(self):
        from elf.segmentation.multicut import multicut_fusion_moves
        self._test_multicut_toy(multicut_fusion_moves)

    def test_transform_probabilities_to_costs(self):
        from elf.segmentation.multicut import transform_probabilities_to_costs
        probs = np.random.rand(100).astype('float32')
        costs = transform_probabilities_to_costs(probs)
        self.assertTrue(np.isfinite(costs).all())


if __name__ == '__main__':
    unittest.main()
