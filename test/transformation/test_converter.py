import unittest
import numpy as np
from elf.transformation.affine import affine_matrix_3d


class TestConverter(unittest.TestCase):
    N = 10

    def test_bdv_to_native(self):
        from elf.transformation.converter import (bdv_to_native,
                                                  native_to_bdv,
                                                  matrix_to_parameters)

        for _ in range(self.N):
            mat = affine_matrix_3d(scale=2 * np.random.rand(3),
                                   rotation=180 * np.random.rand(3),
                                   translation=16 * np.random.rand(3))
            bdv_params = matrix_to_parameters(mat)
            mat_native = bdv_to_native(bdv_params)
            res = native_to_bdv(mat_native)

            self.assertEqual(len(res), len(bdv_params))
            for p1, p2 in zip(res, bdv_params):
                self.assertAlmostEqual(p1, p2)

    def test_native_to_bdv(self):
        from elf.transformation.converter import (bdv_to_native,
                                                  native_to_bdv)

        for _ in range(self.N):
            mat = affine_matrix_3d(scale=2 * np.random.rand(3),
                                   rotation=180 * np.random.rand(3),
                                   translation=16 * np.random.rand(3))
            bdv_params = native_to_bdv(mat)
            res = bdv_to_native(bdv_params)

            self.assertEqual(mat.shape, res.shape)
            self.assertTrue(np.allclose(mat, res))

    # TODO need elastix example files or generate them on the fly in elastix_parser
    def test_elastix_to_bdv(self):
        pass

    # TODO need elastix example files or generate them on the fly in elastix_parser
    def test_elastix_to_native(self):
        pass


if __name__ == '__main__':
    unittest.main()
