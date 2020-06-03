import unittest
import numpy as np


class TestSkeletonIo(unittest.TestCase):
    n_nodes = 100

    def _get_skel(self):
        coords = np.random.rand(self.n_nodes, 3)
        # TODO generate non-loopy edges
        edges = np.random.randint(0, self.n_nodes)
        return coords, edges

    def test_swc(self):
        from elf.skeleton.io import read_swc, write_swc

    def test_nml(self):
        from elf.skeleton.io import read_nml, write_nml


if __name__ == '__main__':
    unittest.main()
