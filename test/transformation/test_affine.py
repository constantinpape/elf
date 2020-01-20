import unittest
import numpy as np
from scipy.ndimage import affine_transform
from elf.util import normalize_index


class TestAffine(unittest.TestCase):
    def _test_2d(self, matrix, **kwargs):
        from elf.transformation import transform_subvolume_with_affine
        shape = (512, 512)
        x = np.random.rand(*shape)
        exp = affine_transform(x, matrix, **kwargs)

        bbs = [np.s_[:, :], np.s_[:256, :256], np.s_[37:115, 226:503],
               np.s_[:200, :], np.s_[:, 10:115]]
        for bb in bbs:
            bb, _ = normalize_index(bb, shape)
            res = transform_subvolume_with_affine(x, matrix, bb, **kwargs)
            exp_bb = exp[bb]
            self.assertEqual(res.shape, exp_bb.shape)
            self.assertTrue(np.allclose(res, exp_bb))

    def test_affine_subvolume_2d(self):
        from elf.transformation import compute_affine_matrix
        # TODO test more matrices
        # TODO test more orders once implemented
        mat = compute_affine_matrix(scale=(2, 2), rotation=(45,))
        self._test_2d(mat, order=0)

    # FIXME 3d test is failing
    def _test_3d(self, matrix, **kwargs):
        from elf.transformation import transform_subvolume_with_affine
        shape = 3 * (64,)
        x = np.random.rand(*shape)
        exp = affine_transform(x, matrix, **kwargs)

        bbs = [np.s_[:, :, :], np.s_[:32, :32, :32], np.s_[1:31, 5:27, 3:13],
               np.s_[4:19, :, 22:], np.s_[1:29], np.s_[:, 15:27, :], np.s_[:, 1:3, 4:14]]
        for bb in bbs:
            bb, _ = normalize_index(bb, shape)
            res = transform_subvolume_with_affine(x, matrix, bb, **kwargs)
            exp_bb = exp[bb]
            self.assertEqual(res.shape, exp_bb.shape)
            self.assertTrue(np.allclose(res, exp_bb))

    def test_affine_subvolume_3d(self):
        from elf.transformation import compute_affine_matrix
        # TODO test more matrices
        # TODO test more orders once implemented
        mat = compute_affine_matrix(scale=(1, 2, 1), rotation=(5, 27, 3))
        self._test_3d(mat, order=0)


if __name__ == '__main__':
    unittest.main()
