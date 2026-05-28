from sys import platform

import unittest
import numpy as np
import bioimage_cpp as bic
from elf.segmentation.utils import load_multicut_problem

try:
    import nifty as _nifty
    _HAS_NIFTY = True
    ILP_SOLVER = any((
        _nifty.Configuration.WITH_GLPK,
        _nifty.Configuration.WITH_GUROBI,
        _nifty.Configuration.WITH_CPLEX,
    ))
    QPBO = bool(_nifty.Configuration.WITH_QPBO)
except ImportError:
    _HAS_NIFTY = False
    ILP_SOLVER = False
    QPBO = False


@unittest.skipIf(platform == "win32", "Download fails on windows")
class TestMulticut(unittest.TestCase):
    upper_bound = -76900

    @classmethod
    def setUpClass(cls):
        load_multicut_problem('A', 'small')

    # https://github.com/constantinpape/graph/blob/master/src/andres/graph/unit-test/multicut/kernighan-lin.cxx
    def toy_problem(self):
        edges = np.array([[0, 1],
                          [0, 3],
                          [1, 2],
                          [1, 4],
                          [2, 5],
                          [3, 4],
                          [4, 5]], dtype="uint64")
        graph = bic.graph.UndirectedGraph.from_edges(6, edges)
        costs = np.array([5, -20, 5, 5, -20, 5, 5], dtype='float32')
        expected_result = np.array([0, 1, 0, 1, 1, 0, 0], dtype='bool')  # edge result
        return graph, costs, expected_result

    def _test_multicut_toy(self, solver):
        graph, costs, expected_result = self.toy_problem()
        node_labels = solver(graph, costs)
        uv_ids = graph.uv_ids()
        result = node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]
        self.assertTrue(np.array_equal(result, expected_result))

    def _test_multicut(self, solver, **kwargs):
        graph, costs = load_multicut_problem('A', 'small')
        node_labels = solver(graph, costs, **kwargs)
        obj = bic.graph.multicut.MulticutObjective(graph, costs)
        energy = obj.energy(node_labels)
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

    def test_greedy_fixation(self):
        from elf.segmentation.multicut import multicut_greedy_fixation
        self._test_multicut(multicut_greedy_fixation)

    @unittest.skipIf(platform == "darwin", "fails on macos")
    def test_greedy_fixation_toy(self):
        from elf.segmentation.multicut import multicut_greedy_fixation
        self._test_multicut_toy(multicut_greedy_fixation)

    @unittest.skipUnless(_HAS_NIFTY and QPBO, "CGC solver needs nifty built with QPBO")
    def test_cgc(self):
        from elf.segmentation.multicut import multicut_cgc
        self._test_multicut(multicut_cgc)

    @unittest.skipUnless(_HAS_NIFTY and QPBO, "CGC solver needs nifty built with QPBO")
    def test_cgc_toy(self):
        from elf.segmentation.multicut import multicut_cgc
        self._test_multicut_toy(multicut_cgc)

    def test_fusion_moves(self):
        from elf.segmentation.multicut import multicut_fusion_moves
        self._test_multicut(multicut_fusion_moves, internal_solver='greedy-additive', num_it=25, num_it_stop=5)

    def test_fusion_moves_toy(self):
        from elf.segmentation.multicut import multicut_fusion_moves
        self._test_multicut_toy(multicut_fusion_moves)

    @unittest.skipUnless(_HAS_NIFTY and ILP_SOLVER, "Needs nifty build with an ilp solver")
    def test_ilp(self):
        from elf.segmentation.multicut import multicut_ilp
        self._test_multicut(multicut_ilp)

    @unittest.skipUnless(_HAS_NIFTY and ILP_SOLVER, "Needs nifty build with an ilp solver")
    def test_ilp_toy(self):
        from elf.segmentation.multicut import multicut_ilp
        self._test_multicut_toy(multicut_ilp)

    def test_blockwise_multicut(self):
        # Reuse the small multicut problem, but project it into a synthetic 3D
        # segmentation so we can run the blockwise hierarchy. Each node id is
        # placed in its own voxel so every block touches a unique subset.
        from elf.segmentation.multicut import blockwise_multicut
        graph, costs = load_multicut_problem('A', 'small')
        n_nodes = graph.numberOfNodes
        # arrange node ids in a roughly cubic volume
        side = int(np.ceil(n_nodes ** (1.0 / 3.0)))
        seg = np.zeros((side, side, side), dtype='uint64')
        seg.reshape(-1)[:n_nodes] = np.arange(n_nodes, dtype='uint64')
        block_shape = (max(side // 2, 1), max(side // 2, 1), max(side // 2, 1))
        labels = blockwise_multicut(
            graph, costs, seg,
            internal_solver='greedy-additive',
            block_shape=block_shape,
            n_threads=1,
        )
        self.assertEqual(labels.shape, (n_nodes,))
        obj = bic.graph.multicut.MulticutObjective(graph, costs)
        energy = obj.energy(labels.astype('uint64'))
        # blockwise GAEC is an approximation; just require finite, sensibly-negative energy
        self.assertTrue(np.isfinite(energy))
        self.assertLess(energy, 0)

    def test_transform_probabilities_to_costs(self):
        from elf.segmentation.multicut import transform_probabilities_to_costs
        probs = np.random.rand(100).astype('float32')
        costs = transform_probabilities_to_costs(probs)
        self.assertTrue(np.isfinite(costs).all())


if __name__ == '__main__':
    unittest.main()
