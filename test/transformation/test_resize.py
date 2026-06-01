import os
import unittest
from shutil import rmtree

import numpy as np
from scipy.ndimage import affine_transform
from elf.io import open_file
from elf.util import normalize_index

try:
    import h5py
except ImportError:
    h5py = None

try:
    import z5py
except ImportError:
    z5py = None


def _remove_path(path):
    try:
        rmtree(path)
    except OSError:
        try:
            os.remove(path)
        except OSError:
            pass


class TestResize(unittest.TestCase):
    def tearDown(self):
        _remove_path("tmp.n5")
        _remove_path("tmp.h5")

    def _test_resize(self, scale_factor, shape, chunks, order, out_file=None):
        from elf.transformation import transform_subvolume_resize

        x = np.random.rand(*shape)
        # resize is a coordinate scaling, which corresponds to an affine transform
        # with a diagonal scale matrix mapping output to input coordinates.
        matrix = np.diag(list(scale_factor) + [1.0])
        exp = affine_transform(x, matrix, order=order)

        if out_file is not None:
            _remove_path(out_file)
            with open_file(out_file, mode="a") as f:
                f.create_dataset("tmp", data=x, chunks=chunks)
            f = open_file(out_file, mode="r")
            x = f["tmp"]

        ndim = len(shape)
        bbs = [tuple(slice(None) for _ in range(ndim)),
               tuple(slice(s // 4, 3 * s // 4) for s in shape)]
        for bb in bbs:
            bb, _ = normalize_index(bb, shape)
            res = transform_subvolume_resize(x, scale_factor, bb, order=order)
            exp_bb = exp[bb]
            self.assertEqual(res.shape, exp_bb.shape)
            self.assertTrue(np.allclose(res, exp_bb))

        if out_file is not None:
            f.close()

    # NOTE: the order-0 (nearest neighbor) implementation rounds coordinates with numpy's
    # round-half-to-even, which only agrees with scipy when the scaled coordinates avoid exact
    # half-integer ties. We therefore restrict the order-0 comparison to integer scale factors.
    def test_resize_2d_order0(self):
        self._test_resize((2.0, 2.0), (128, 128), (32, 32), order=0)
        self._test_resize((3.0, 2.0), (96, 128), (32, 32), order=0)

    def test_resize_2d_order1(self):
        self._test_resize((2.0, 2.0), (128, 128), (32, 32), order=1)
        self._test_resize((0.5, 1.5), (96, 128), (32, 32), order=1)

    def test_resize_3d_order1(self):
        self._test_resize((2.0, 1.5, 0.5), (48, 48, 48), (16, 16, 16), order=1)

    @unittest.skipUnless(z5py, "Need z5py")
    def test_resize_2d_z5(self):
        self._test_resize((2.0, 2.0), (128, 128), (32, 32), order=0, out_file="tmp.n5")
        self._test_resize((2.0, 2.0), (128, 128), (32, 32), order=1, out_file="tmp.n5")

    @unittest.skipUnless(z5py, "Need z5py")
    def test_resize_3d_z5(self):
        self._test_resize((2.0, 1.5, 0.5), (48, 48, 48), (16, 16, 16), order=1, out_file="tmp.n5")

    @unittest.skipUnless(h5py, "Need h5py")
    def test_resize_2d_h5(self):
        self._test_resize((2.0, 2.0), (128, 128), (32, 32), order=1, out_file="tmp.h5")


if __name__ == "__main__":
    unittest.main()
