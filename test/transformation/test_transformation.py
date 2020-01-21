import unittest
import numpy as np


class TestTransformation(unittest.TestCase):
    def _test_coordinate_interpolation(self, interpolate, ndim, expected_coords,
                                       factor=1.):
        N = 10
        for _ in range(N):
            coord = [co + factor * np.random.rand() for co in range(ndim)]
            coords, weights = interpolate(coord)
            self.assertAlmostEqual(sum(weights), 1.)

            coord_set = set(tuple(coo) for coo in coords)
            exp_set = set(tuple(coo) for coo in expected_coords)
            self.assertEqual(coord_set, exp_set)

    def test_nn_coordinate_interpolation(self):
        from elf.transformation.transform_impl import interpolate_nn
        expected_2d = [[0, 1]]
        self._test_coordinate_interpolation(interpolate_nn, 2, expected_2d, factor=.5)
        expected_3d = [[0, 1, 2]]
        self._test_coordinate_interpolation(interpolate_nn, 3, expected_3d, factor=.5)

    def test_linear_coordinate_interpolation(self):
        from elf.transformation.transform_impl import interpolate_linear
        expected_2d = [[0, 1], [1, 1], [0, 2], [1, 2]]
        self._test_coordinate_interpolation(interpolate_linear, 2, expected_2d)
        expected_3d = [[0, 1, 2], [1, 1, 2], [0, 2, 2], [0, 1, 3],
                       [1, 2, 2], [1, 1, 3], [0, 2, 3], [1, 2, 3]]
        self._test_coordinate_interpolation(interpolate_linear, 3, expected_3d)


if __name__ == '__main__':
    unittest.main()
