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


def _remove_path(path):
    try:
        rmtree(path)
    except OSError:
        try:
            os.remove(path)
        except OSError:
            pass


class TestAffine(unittest.TestCase):
    def tearDown(self):
        try:
            rmtree("tmp.n5")
        except OSError:
            pass
        try:
            os.remove("tmp.h5")
        except OSError:
            pass

    def _test_2d(self, matrix, out_file=None, **kwargs):
        from elf.transformation import transform_subvolume_affine
        shape = (512, 512)
        x = np.random.rand(*shape)
        exp = affine_transform(x, matrix, **kwargs)

        if out_file is not None:
            _remove_path(out_file)
            with open_file(out_file, mode="a") as f:
                f.create_dataset("tmp", data=x, chunks=(64, 64))
            f = open_file(out_file, mode="r")
            x = f["tmp"]

        bbs = [np.s_[:, :], np.s_[:256, :256], np.s_[37:115, 226:503],
               np.s_[:200, :], np.s_[:, 10:115]]
        for bb in bbs:
            bb, _ = normalize_index(bb, shape)
            res = transform_subvolume_affine(x, matrix, bb, **kwargs)
            exp_bb = exp[bb]

            self.assertEqual(res.shape, exp_bb.shape)
            self.assertTrue(np.allclose(res, exp_bb))

        if out_file is not None:
            f.close()

    def test_affine_subvolume_2d(self):
        from elf.transformation import compute_affine_matrix
        orders = [0, 1]
        matrices = [compute_affine_matrix(scale=(2, 2), rotation=(45,)),
                    compute_affine_matrix(scale=(1, 2), rotation=(33,)),
                    compute_affine_matrix(scale=(3, 2), rotation=(137,)),
                    compute_affine_matrix(scale=(.5, 1.5), rotation=(23,),
                                          translation=(23., -14.))]
        for mat in matrices:
            for order in orders:
                self._test_2d(mat, order=order)

    def _test_affine_subvolume_2d_chunked(self, out_file):
        from elf.transformation import compute_affine_matrix
        mat = compute_affine_matrix(scale=(2, 2), rotation=(45,))
        for order in (0, 1):
            self._test_2d(mat, order=order, out_file=out_file)

    def test_affine_subvolume_2d_z5(self):
        self._test_affine_subvolume_2d_chunked("tmp.n5")

    @unittest.skipUnless(h5py, "Need h5py")
    def test_affine_subvolume_2d_h5(self):
        self._test_affine_subvolume_2d_chunked("tmp.h5")

    # Pre-smoothing now works (via bioimage_cpp). It should run without error, preserve the output
    # shape, and (since it low-passes the input) produce a result that differs from the unsmoothed
    # transform on a region that lies inside the input.
    def _test_presmoothing(self, out_file=None):
        from elf.transformation import compute_affine_matrix, transform_subvolume_affine
        shape = (256, 256)
        x = np.random.rand(*shape)
        # a downscaling transform, where pre-smoothing (anti-aliasing) actually matters
        mat = compute_affine_matrix(scale=(2, 2))
        bb, _ = normalize_index(np.s_[:64, :64], shape)

        data = x
        if out_file is not None:
            _remove_path(out_file)
            with open_file(out_file, mode="a") as f:
                f.create_dataset("tmp", data=x, chunks=(64, 64))
            f = open_file(out_file, mode="r")
            data = f["tmp"]

        plain = transform_subvolume_affine(data, mat, bb, order=1)
        smoothed = transform_subvolume_affine(data, mat, bb, order=1, sigma=2.0)

        self.assertEqual(smoothed.shape, plain.shape)
        self.assertTrue(np.all(np.isfinite(smoothed)))
        self.assertFalse(np.allclose(smoothed, plain))

        if out_file is not None:
            f.close()

    def test_presmoothing(self):
        self._test_presmoothing()

    def test_presmoothing_z5(self):
        self._test_presmoothing(out_file="tmp.n5")

    def _test_3d(self, matrix, out_file=None, **kwargs):
        from elf.transformation import transform_subvolume_affine
        shape = 3 * (64,)
        x = np.random.rand(*shape)
        exp = affine_transform(x, matrix, **kwargs)

        if out_file is not None:
            _remove_path(out_file)
            with open_file(out_file, mode="a") as f:
                f.create_dataset("tmp", data=x, chunks=3 * (16,))
            f = open_file(out_file, mode="r")
            x = f["tmp"]

        bbs = [np.s_[:, :, :], np.s_[:32, :32, :32], np.s_[1:31, 5:27, 3:13],
               np.s_[4:19, :, 22:], np.s_[1:29], np.s_[:, 15:27, :], np.s_[:, 1:3, 4:14]]
        for bb in bbs:
            bb, _ = normalize_index(bb, shape)
            res = transform_subvolume_affine(x, matrix, bb, **kwargs)
            exp_bb = exp[bb]

            self.assertEqual(res.shape, exp_bb.shape)
            self.assertTrue(np.allclose(res, exp_bb))

        if out_file is not None:
            f.close()

    def test_affine_subvolume_3d(self):
        from elf.transformation import compute_affine_matrix
        orders = [0, 1]
        matrices = [compute_affine_matrix(scale=(1, 2, 1), rotation=(15, 30, 0)),
                    compute_affine_matrix(scale=(3, 2, .5), rotation=(24, 33, 99)),
                    compute_affine_matrix(scale=(1., 1.3, .79), rotation=(12, -4, 8),
                                          translation=(10.5, 18., -33.2))]
        for mat in matrices:
            for order in orders:
                self._test_3d(mat, order=order)

    def _test_affine_subvolume_3d_chunked(self, out_file):
        from elf.transformation import compute_affine_matrix
        mat = compute_affine_matrix(scale=(1, 2, 1), rotation=(15, 30, 0))
        for order in (0, 1):
            self._test_3d(mat, order=order, out_file=out_file)

    def test_affine_subvolume_3d_z5(self):
        self._test_affine_subvolume_3d_chunked("tmp.n5")

    @unittest.skipUnless(h5py, "Need h5py")
    def test_affine_subvolume_3d_h5(self):
        self._test_affine_subvolume_3d_chunked("tmp.h5")

    def test_higher_order_2d(self):
        from elf.transformation import compute_affine_matrix, transform_subvolume_affine
        shape = (256, 256)
        x = np.random.rand(*shape)
        mat = compute_affine_matrix(scale=(1.5, 1.5), rotation=(20,))
        # bioimage_cpp orders 2, 4, 5 evaluate the cardinal B-spline directly (no prefilter)
        # and use 'grid-constant' border handling, matching scipy with these settings.
        for order in (2, 4, 5):
            exp = affine_transform(x, mat, order=order, prefilter=False, mode="grid-constant")
            for bb in [np.s_[:, :], np.s_[20:200, 33:240]]:
                bb, _ = normalize_index(bb, shape)
                res = transform_subvolume_affine(x, mat, bb, order=order)
                self.assertEqual(res.shape, exp[bb].shape)
                self.assertTrue(np.allclose(res, exp[bb], atol=1e-5))

    def test_order_3_reproduces_input(self):
        from elf.transformation import transform_subvolume_affine
        # order 3 (Keys cubic) is interpolating: with the identity matrix it reproduces the input.
        x = np.random.rand(64, 64)
        identity = np.eye(3)
        res = transform_subvolume_affine(x, identity, np.s_[:, :], order=3)
        self.assertEqual(res.shape, x.shape)
        self.assertTrue(np.allclose(res, x))

    def _test_chunked_matches_memory(self, out_file):
        from elf.transformation import compute_affine_matrix, transform_subvolume_affine
        shape = (128, 128)
        x = np.random.rand(*shape)
        mat = compute_affine_matrix(scale=(1.7, 1.3), rotation=(35,), translation=(4, -6))

        _remove_path(out_file)
        with open_file(out_file, mode="a") as f:
            f.create_dataset("tmp", data=x, chunks=(32, 32))
        f = open_file(out_file, mode="r")
        ds = f["tmp"]

        bbs = [np.s_[:, :], np.s_[5:120, 8:122]]
        # The chunked workaround must agree with the in-memory transform for all interpolation
        # orders, independently of the interpolation semantics.
        for order in range(6):
            for bb in bbs:
                bb, _ = normalize_index(bb, shape)
                mem = transform_subvolume_affine(x, mat, bb, order=order)
                chunked = transform_subvolume_affine(ds, mat, bb, order=order)
                self.assertEqual(mem.shape, chunked.shape)
                self.assertTrue(np.allclose(mem, chunked), f"order={order}")
        f.close()

    def test_chunked_matches_memory_z5(self):
        self._test_chunked_matches_memory("tmp.n5")

    @unittest.skipUnless(h5py, "Need h5py")
    def test_chunked_matches_memory_h5(self):
        self._test_chunked_matches_memory("tmp.h5")

    def test_python_fallback(self):
        from elf.transformation import compute_affine_matrix, transform_subvolume_affine
        x = np.random.rand(64, 64)
        mat = compute_affine_matrix(scale=(2, 2), rotation=(45,), translation=(-1, 1))
        for order in (0, 1):
            exp = affine_transform(x, mat, order=order)
            res = transform_subvolume_affine(x, mat, np.s_[3:40, 5:50], order=order,
                                             use_python_fallback_impl=True)
            exp_bb = exp[3:40, 5:50]
            self.assertEqual(res.shape, exp_bb.shape)
            self.assertTrue(np.allclose(res, exp_bb))

    def test_toy(self):
        from elf.transformation import compute_affine_matrix
        from elf.transformation import transform_subvolume_affine
        mat = compute_affine_matrix(scale=(2, 2), rotation=(45,), translation=(-1, 1))
        x = np.random.rand(10, 10)

        bb = np.s_[0:3, 0:3]
        res = transform_subvolume_affine(x, mat, bb, order=1)
        exp = affine_transform(x, mat, order=1)[bb]

        self.assertTrue(np.allclose(res, exp))


class TestAffineUtils(unittest.TestCase):
    def test_transform_roi_with_affine(self):
        from elf.transformation.affine import transform_roi_with_affine, compute_affine_matrix

        # A pure translation shifts the roi corners by the translation vector.
        matrix = compute_affine_matrix(scale=(1, 1), translation=(5.0, -3.0))
        start, stop = transform_roi_with_affine([0, 0], [10, 20], matrix)
        self.assertTrue(np.allclose(start, [5.0, -3.0]))
        self.assertTrue(np.allclose(stop, [15.0, 17.0]))

        # A pure scaling scales the roi corners.
        matrix = compute_affine_matrix(scale=(2.0, 3.0))
        start, stop = transform_roi_with_affine([1, 1], [4, 5], matrix)
        self.assertTrue(np.allclose(start, [2.0, 3.0]))
        self.assertTrue(np.allclose(stop, [8.0, 15.0]))

    def test_translation_from_matrix(self):
        from elf.transformation.affine import (translation_from_matrix, compute_affine_matrix,
                                               affine_matrix_3d)

        matrix = compute_affine_matrix(scale=(2.0, 3.0), rotation=(30,), translation=(7.0, -5.0))
        self.assertTrue(np.allclose(translation_from_matrix(matrix), [7.0, -5.0]))

        matrix_3d = affine_matrix_3d(scale=(1, 1, 1), translation=(1.5, -2.5, 3.5))
        self.assertTrue(np.allclose(translation_from_matrix(matrix_3d), [1.5, -2.5, 3.5]))


if __name__ == "__main__":
    unittest.main()
