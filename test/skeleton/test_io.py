import os
import unittest
from shutil import rmtree

import numpy as np

try:
    import z5py
except ImportError:
    z5py = None


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
            coords_input = coords.copy()
            write_swc(path, coords, edges)
            # write_swc must not modify its input array in place
            self.assertTrue(np.array_equal(coords, coords_input))
            ids, coords_read, parents_read = read_swc(path)
            self.assertTrue(np.array_equal(coords, coords_read))
            self.assertEqual(len(ids), len(coords_read))
            # the parents are derived from the edges via the graph; we expect one per node
            self.assertEqual(len(parents_read), len(coords_read))

    def test_swc_resolution_invert(self):
        from elf.skeleton.io import read_swc, write_swc
        path = os.path.join(self.tmp_folder, "res.swc")
        coords, edges = self._get_skel()
        resolution = (2.0, 3.0, 4.0)
        write_swc(path, coords, edges, resolution=resolution, invert_coords=True)
        _, coords_read, _ = read_swc(path)
        expected = (coords * np.array(resolution))[:, ::-1]
        self.assertTrue(np.allclose(np.array(coords_read), expected))

    @unittest.skipUnless(z5py, "Needs z5py")
    def test_n5(self):
        from elf.skeleton.io import read_n5, write_n5
        path = os.path.join(self.tmp_folder, "skels.n5")
        f = z5py.File(path, "a")
        n_skels = 5
        ds = f.create_dataset("skeletons", shape=(n_skels,), chunks=(1,), dtype="uint64", compression="gzip")

        skels = {}
        for skel_id in range(n_skels):
            coords, edges = self._get_skel()
            coords = coords.astype("uint64")
            edges = edges.astype("uint64")
            skels[skel_id] = (coords, edges)
            write_n5(ds, skel_id, coords, edges)

        for skel_id in range(n_skels):
            coords, edges = skels[skel_id]
            nodes_read, edges_read = read_n5(ds, skel_id)
            self.assertTrue(np.array_equal(nodes_read, coords))
            self.assertTrue(np.array_equal(edges_read, edges))

        # reading an unwritten chunk should be reported as empty, i.e. (None, None)
        empty_ds = f.create_dataset("empty", shape=(1,), chunks=(1,), dtype="uint64", compression="gzip")
        self.assertEqual(read_n5(empty_ds, 0), (None, None))

    @unittest.skipUnless(z5py, "Needs z5py")
    def test_n5_coordinate_offset(self):
        from elf.skeleton.io import read_n5, write_n5
        path = os.path.join(self.tmp_folder, "skels_offset.n5")
        f = z5py.File(path, "a")
        ds = f.create_dataset("skeletons", shape=(1,), chunks=(1,), dtype="uint64", compression="gzip")
        coords, edges = self._get_skel()
        coords = coords.astype("uint64")
        coords_input = coords.copy()
        offset = [10, 20, 30]
        write_n5(ds, 0, coords, edges.astype("uint64"), coordinate_offset=offset)
        # write_n5 must not modify its input array in place
        self.assertTrue(np.array_equal(coords, coords_input))
        nodes_read, _ = read_n5(ds, 0)
        self.assertTrue(np.array_equal(nodes_read, coords + np.array(offset, dtype="uint64")))

    def test_nml(self):
        from elf.skeleton.io import read_nml  # noqa


if __name__ == "__main__":
    unittest.main()
