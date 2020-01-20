import unittest

import numpy as np
from scipy.ndimage import affine_transform
from elf.transformation.affine import compute_affine_matrix


class TestAffineVolume(unittest.TestCase):
    def _check_index(self, out1, out2, index,
                     check_close=True, halo=4):
        o1 = out1[index]
        o2 = out2[index]
        self.assertEqual(o1.shape, o2.shape)
        if check_close:
            bb = tuple(slice(halo, sh - halo) for sh in o1.shape)
            o1, o2 = o1[bb], o2[bb]
            self.assertTrue(np.allclose(o1, o2))

    def test_affine_2d(self):
        from elf.wrapper.affine_volume import AffineVolume
        data = np.random.rand(128, 128)

        # TODO check more orders once we support this
        orders = [0]
        matrices = [compute_affine_matrix(scale=(1, 1), rotation=(45,)),
                    compute_affine_matrix(scale=(1, 3), rotation=(75,)),
                    compute_affine_matrix(scale=(2, 1), rotation=(127,))]
        indices = [np.s_[:], np.s_[1:-1, 2:-2], np.s_[:64, 64:], np.s_[12:53, 27:111]]

        for mat in matrices:
            for order in orders:
                out1 = affine_transform(data, mat, order=order)
                out2 = AffineVolume(data, affine_matrix=mat, order=order)
                for index in indices:
                    self._check_index(out1, out2, index)

    def test_affine_3d(self):
        from elf.wrapper.affine_volume import AffineVolume
        data = np.random.rand(64, 64, 64)

        # TODO check more orders once we support this
        orders = [0]
        matrices = [compute_affine_matrix(scale=(2, 2, 2), rotation=(90, 0, 0)),
                    compute_affine_matrix(scale=(1, 2, 2), rotation=(60, 30, 0))]
        indices = (np.s_[:],  np.s_[1:-1, 2:-2, 3:-3],
                   np.s_[:32, 32:, :], np.s_[12:53, 27:54, 8:33])

        for mat in matrices:
            for order in orders:
                out1 = affine_transform(data, mat, order=order)
                out2 = AffineVolume(data, affine_matrix=mat, order=order)
                for index in indices:
                    self._check_index(out1, out2, index)


if __name__ == '__main__':
    unittest.main()
