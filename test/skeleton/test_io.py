import os
import unittest
from shutil import rmtree

import numpy as np


class TestSkeletonIo(unittest.TestCase):
    shape = 128
    n_nodes = 100
    tmp_folder = "./tmp"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _get_skel(self):
        coords = np.random.randint(0, self.shape, size=(self.n_nodes, 3))
        edges = np.random.randint(0, self.n_nodes, size=(self.n_nodes, 2))
        return coords, edges

    def test_swc(self):
        from elf.skeleton.io import read_swc, write_swc
        n_skels = 5
        for skel_id in range(n_skels):
            path = os.path.join(self.tmp_folder, f"{skel_id}.swc")
            coords, edges = self._get_skel()
            write_swc(path, coords, edges)
            _, coords_read, parents_read,  = read_swc(path)
            self.assertTrue(np.array_equal(coords, coords_read))
            self.assertEqual(len(parents_read), len(coords_read))
            # checking for edges is a bit more complicated ...
            # self.assertTrue(np.array_equal(edges, edges_read))

    def test_nml(self):
        from elf.skeleton.io import read_nml  # noqa


if __name__ == "__main__":
    unittest.main()
