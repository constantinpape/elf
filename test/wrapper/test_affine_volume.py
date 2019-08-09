import unittest
import numpy as np
from scipy.ndimage import affine_transform
from elf.transformation.affine import compute_affine_matrix, transform_roi


# TODO run tests for more transformation matrices
class TestAffineVolume(unittest.TestCase):
    def _check_index(self, out1, out2, index,
                     check_close=True, halo=8):
        o1 = out1[index]
        o2 = out2[index]
        self.assertEqual(o1.shape, o2.shape)
        if check_close:
            bb = tuple(slice(halo, sh - halo) for sh in o1.shape)
            self.assertTrue(np.allclose(o1[bb], o2[bb]))

    def _compute_shape(self, matrix, shape):
        roi_start, roi_stop = transform_roi([0] * len(shape), shape, matrix)
        return tuple(int(sto - sta) for sta, sto in zip(roi_start, roi_stop))

    def test_affine_2d_full_volume(self):
        from elf.wrapper.affine_volume import AffineVolume
        matrix = compute_affine_matrix(scale=(2, 2), rotation=(90,))
        orders = (0, 3)
        data = np.random.rand(128, 128)
        out_shape = self._compute_shape(matrix, data.shape)
        for order in orders:
            out1 = affine_transform(data, np.linalg.inv(matrix), output_shape=out_shape,
                                    order=order, mode='constant', cval=0)
            out2 = AffineVolume(data, affine_matrix=matrix, order=order)
            self._check_index(out1, out2, np.s_[:])

    @unittest.expectedFailure
    def test_affine_2d(self):
        from elf.wrapper.affine_volume import AffineVolume
        matrix = compute_affine_matrix(scale=(2, 2), rotation=(90,))
        orders = (0, 3)
        data = np.random.rand(128, 128)
        out_shape = self._compute_shape(matrix, data.shape)
        indices = (np.s_[1:-1, 2:-2], np.s_[:64, 64:], np.s_[12:53, 27:111])
        for order in orders:
            out1 = affine_transform(data, np.linalg.inv(matrix), output_shape=out_shape,
                                    order=order, mode='constant', cval=0)
            out2 = AffineVolume(data, affine_matrix=matrix, order=order)
            for index in indices:
                self._check_index(out1, out2, index)

    def test_affine_3d_full_volume(self):
        from elf.wrapper.affine_volume import AffineVolume

        # FIXME this fails !
        # matrix = compute_affine_matrix(scale=(2, 2, 2), rotation=(60, 30, 0))
        matrix = compute_affine_matrix(scale=(2, 2, 2), rotation=(90, 0, 0))

        orders = (0, 3)
        data = np.random.rand(64, 64, 64)
        out_shape = self._compute_shape(matrix, data.shape)
        for order in orders:
            out1 = affine_transform(data, np.linalg.inv(matrix), output_shape=out_shape,
                                    order=order, mode='constant', cval=0)
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
