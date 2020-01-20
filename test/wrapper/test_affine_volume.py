import unittest
from math import ceil

import numpy as np
from scipy.ndimage import affine_transform
from elf.transformation.affine import compute_affine_matrix, transform_roi_with_affine


# TODO run tests for more transformation matrices
class TestAffineVolume(unittest.TestCase):
    def _check_index(self, out1, out2, index,
                     check_close=True, halo=4):
        o1 = out1[index]
        o2 = out2[index]
        self.assertEqual(o1.shape, o2.shape)
        if check_close:
            bb = tuple(slice(halo, sh - halo) for sh in o1.shape)
            o1, o2 = o1[bb], o2[bb]
            # print(o1.shape)
            # print(np.isclose(o1, o2).sum(), o1.size)
            # not_close = np.where(~np.isclose(o1, o2))
            # print(not_close)
            # print(o1[not_close], o2[not_close])
            # print("Expected")
            # print(o1[:5, :5])
            # print(o1[-10:, -10:])
            # print("Result")
            # print(o2[:5, :5])
            self.assertTrue(np.allclose(o1, o2))

    def get_shape_and_offset(self, matrix, shape):
        roi_start, roi_stop = transform_roi_with_affine([0] * len(shape), shape, matrix)
        offset = [-rs for rs in roi_start]
        return tuple(int(ceil(sto - sta)) for sta, sto in zip(roi_start, roi_stop)), offset

    def compute_scipy_mat(self, matrix, offset):
        ndim = matrix.shape[0] - 1
        tmp_mat = matrix.copy()
        tmp_mat[:ndim, ndim] = offset
        tmp_mat = np.linalg.inv(tmp_mat)
        return tmp_mat

    def apply_affine(self, data, matrix, shape, offset, order):
        mat_for_scipy = self.compute_scipy_mat(matrix, offset)
        out = affine_transform(data, mat_for_scipy, output_shape=shape,
                               order=order, mode='constant', cval=0, offset=offset)
        return out

    def test_affine_2d_full_volume(self):
        from elf.wrapper.affine_volume import AffineVolume

        matrices = [compute_affine_matrix(scale=(1, 1), rotation=(90,)),
                    compute_affine_matrix(scale=(1, 3), rotation=(60,)),
                    compute_affine_matrix(scale=(.5, 4), rotation=(173,))]
        orders = (0, 3)
        data = np.random.rand(128, 128)

        for matrix in matrices:
            out_shape, offset = self.get_shape_and_offset(matrix, data.shape)
            for order in orders:
                out1 = self.apply_affine(data, matrix, out_shape, offset, order)
                out2 = AffineVolume(data, affine_matrix=matrix, order=order)
                self._check_index(out1, out2, np.s_[:])

    @unittest.expectedFailure
    def test_affine_2d(self):
        from elf.wrapper.affine_volume import AffineVolume
        matrix = compute_affine_matrix(scale=(1, 1), rotation=(90,))
        orders = (0, 3)
        # data = np.random.rand(128, 128)
        data = np.arange(128**2).reshape((128, 128))

        out_shape, offset = self.get_shape_and_offset(matrix, data.shape)
        indices = (np.s_[1:-1, 2:-2], np.s_[:64, 64:], np.s_[12:53, 27:111])
        for order in orders:
            out1 = self.apply_affine(data, matrix, out_shape, offset, order)
            out2 = AffineVolume(data, affine_matrix=matrix, order=order)
            for index in indices:
                print("Check index:", index)
                self._check_index(out1, out2, index)

    def test_affine_3d_full_volume(self):
        from elf.wrapper.affine_volume import AffineVolume

        matrices = [compute_affine_matrix(scale=(2, 2, 2), rotation=(90, 0, 0)),
                    compute_affine_matrix(scale=(2, 2, 2), rotation=(60, 30, 0))]

        orders = (0, 3)
        data = np.random.rand(64, 64, 64)
        for matrix in matrices:
            out_shape, offset = self.get_shape_and_offset(matrix, data.shape)
            for order in orders:
                out1 = self.apply_affine(data, matrix, out_shape, offset, order)
                out2 = AffineVolume(data, affine_matrix=matrix, order=order)
                self._check_index(out1, out2, np.s_[:])

    @unittest.expectedFailure
    def test_affine_3d(self):
        from elf.wrapper.affine_volume import AffineVolume

        matrix = compute_affine_matrix(scale=(2, 2, 2), rotation=(90, 0, 0))

        orders = (0, 3)
        indices = (np.s_[1:-1, 2:-2, 3:-3], np.s_[:32, 32:, :],
                   np.s_[12:53, 27:54, 8:33])

        data = np.random.rand(64, 64, 64)
        out_shape = self._compute_shape(matrix, data.shape)

        for order in orders:
            out1 = affine_transform(data, np.linalg.inv(matrix), output_shape=out_shape,
                                    order=order, mode='constant', cval=0)
            out2 = AffineVolume(data, affine_matrix=matrix, order=order)
            for index in indices:
                self._check_index(out1, out2, index)


if __name__ == '__main__':
    unittest.main()
