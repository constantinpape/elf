from sys import platform

import unittest
import bioimage_cpp as bic
import numpy as np


class TestLiftedMulticut(unittest.TestCase):

    # https://github.com/constantinpape/graph/blob/master/src/andres/graph/unit-test/multicut-lifted/kernighan-lin.cxx
    def toy_problem(self):
        edges = np.array([[0, 1],
                          [0, 3],
                          [1, 2],
                          [1, 4],
                          [3, 4]], dtype="uint64")
        graph = bic.graph.UndirectedGraph.from_edges(5, edges)
        costs = np.array([10, -1, 10, 4, 10], dtype='float32')

        # complete graph excluding `edges`
        lifted_edges = np.array([[0, 2],
                                 [0, 4],
                                 [1, 3],
                                 [2, 3],
                                 [2, 4],
                                 [3, 4]], dtype="uint64")
        lifted_costs = np.array([-1, -1, -1, -1, -1, 10], dtype='float32')

        expected_result = np.array([0, 1, 0, 1, 0], dtype='bool')  # edge result
        return graph, lifted_edges, costs, lifted_costs, expected_result

    def _test_lifted_multicut(self, solver):
        graph, lifted_edges, costs, lifted_costs, expected_result = self.toy_problem()
        node_labels = solver(graph, costs, lifted_edges, lifted_costs)
        uv_ids = graph.uv_ids()
        result = node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]
        self.assertTrue(np.array_equal(result, expected_result))

    def test_gaec(self):
        from elf.segmentation.lifted_multicut import lifted_multicut_gaec
        self._test_lifted_multicut(lifted_multicut_gaec)

    def test_kernighan_lin(self):
        from elf.segmentation.lifted_multicut import lifted_multicut_kernighan_lin
        self._test_lifted_multicut(lifted_multicut_kernighan_lin)

    def test_fusion_moves(self):
        from elf.segmentation.lifted_multicut import lifted_multicut_fusion_moves
        self._test_lifted_multicut(lifted_multicut_fusion_moves)


@unittest.skipIf(platform == "win32", "Download fails on windows")
class TestLiftedMulticutRealProblem(unittest.TestCase):
    """Exercise the lifted multicut solvers on the small example multicut problem
    augmented with a handful of synthetic lifted edges."""

    @classmethod
    def setUpClass(cls):
        from elf.segmentation.utils import load_multicut_problem
        graph, costs = load_multicut_problem('A', 'small')
        cls.graph = graph
        cls.costs = costs.astype('float64')

        rng = np.random.default_rng(0)
        n_nodes = graph.number_of_nodes
        # synthesize 200 random lifted node pairs; drop self-loops and any that
        # coincide with base edges. A small set is enough to exercise the solvers.
        sample = rng.integers(0, n_nodes, size=(400, 2), dtype='uint64')
        sample = sample[sample[:, 0] != sample[:, 1]]
        sample = np.sort(sample, axis=1)
        base_set = {(int(u), int(v)) for u, v in graph.uv_ids()}
        keep = np.array([(int(u), int(v)) not in base_set for u, v in sample], dtype=bool)
        cls.lifted_uvs = sample[keep][:200].astype('uint64')
        cls.lifted_costs = rng.uniform(-1.0, 1.0, size=cls.lifted_uvs.shape[0]).astype('float64')

    def _energy(self, labels):
        obj = bic.graph.lifted_multicut.LiftedMulticutObjective(
            self.graph, self.costs,
            lifted_uvs=self.lifted_uvs, lifted_costs=self.lifted_costs,
        )
        return obj.energy(labels.astype('uint64'))

    def test_gaec(self):
        from elf.segmentation.lifted_multicut import lifted_multicut_gaec
        labels = lifted_multicut_gaec(self.graph, self.costs, self.lifted_uvs, self.lifted_costs)
        self.assertEqual(labels.shape, (self.graph.number_of_nodes,))
        self.assertLess(self._energy(labels), 0)

    def test_kernighan_lin(self):
        from elf.segmentation.lifted_multicut import lifted_multicut_gaec, lifted_multicut_kernighan_lin
        labels = lifted_multicut_kernighan_lin(self.graph, self.costs, self.lifted_uvs, self.lifted_costs)
        gaec_labels = lifted_multicut_gaec(self.graph, self.costs, self.lifted_uvs, self.lifted_costs)
        # KL (warmstart from GAEC) should never be worse than GAEC alone, modulo numerical noise
        self.assertLessEqual(self._energy(labels), self._energy(gaec_labels) + 1e-6)

    def test_fusion_moves(self):
        from elf.segmentation.lifted_multicut import lifted_multicut_gaec, lifted_multicut_fusion_moves
        labels = lifted_multicut_fusion_moves(self.graph, self.costs, self.lifted_uvs, self.lifted_costs)
        gaec_labels = lifted_multicut_gaec(self.graph, self.costs, self.lifted_uvs, self.lifted_costs)
        self.assertLessEqual(self._energy(labels), self._energy(gaec_labels) + 1e-6)

    def test_blockwise_lifted_multicut(self):
        from elf.segmentation.lifted_multicut import blockwise_lifted_multicut
        n_nodes = self.graph.number_of_nodes
        side = int(np.ceil(n_nodes ** (1.0 / 3.0)))
        seg = np.zeros((side, side, side), dtype='uint64')
        seg.reshape(-1)[:n_nodes] = np.arange(n_nodes, dtype='uint64')
        block_shape = (max(side // 2, 1), max(side // 2, 1), max(side // 2, 1))
        labels = blockwise_lifted_multicut(
            self.graph, self.costs, self.lifted_uvs, self.lifted_costs, seg,
            internal_solver='greedy-additive',
            block_shape=block_shape,
            n_threads=1,
        )
        self.assertEqual(labels.shape, (n_nodes,))
        self.assertTrue(np.isfinite(self._energy(labels.astype('uint64'))))


if __name__ == '__main__':
    unittest.main()
