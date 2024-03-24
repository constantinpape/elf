import unittest

import numpy as np
from skimage.data import binary_blobs
from scipy.ndimage import distance_transform_edt


class TestDistanceTransform(unittest.TestCase):
    def _check_result(self, data, result, tolerance):
        expected = distance_transform_edt(data)

        tolerance_mask = expected < tolerance
        self.assertTrue(np.allclose(result[tolerance_mask], expected[tolerance_mask]))

    def _check_result_with_indices(self, data, distances, indices, tolerance):
        expected_distances, expected_indices = distance_transform_edt(data, return_indices=True)

        tolerance_mask = expected_distances < tolerance
        self.assertTrue(np.allclose(distances[tolerance_mask], expected_distances[tolerance_mask]))
        tolerance_mask = np.stack([tolerance_mask] * indices.shape[0])
        self.assertTrue(np.allclose(indices[tolerance_mask], expected_indices[tolerance_mask]))

    def test_distance_transform_2d(self):
        from elf.parallel import distance_transform

        tolerance = 64
        data = binary_blobs(length=512, n_dim=2, volume_fraction=0.2)
        result = distance_transform(data, halo=(tolerance, tolerance), block_shape=(128, 128))
        self._check_result(data, result, tolerance)

    def test_distance_transform_3d(self):
        from elf.parallel import distance_transform

        tolerance = 16
        data = binary_blobs(length=128, n_dim=3, volume_fraction=0.2)
        result = distance_transform(data, halo=(tolerance, tolerance, tolerance), block_shape=(64, 64, 64))
        self._check_result(data, result, tolerance)

    def test_distance_transform_with_indices_2d(self):
        from elf.parallel import distance_transform

        tolerance = 64
        data = binary_blobs(length=512, n_dim=2, volume_fraction=0.2)
        distances, indices = distance_transform(
            data, halo=(tolerance, tolerance), return_indices=True, block_shape=(128, 128)
        )
        self._check_result_with_indices(data, distances, indices, tolerance)

    def test_distance_transform_with_indices_3d(self):
        from elf.parallel import distance_transform

        tolerance = 16
        data = binary_blobs(length=128, n_dim=3, volume_fraction=0.2)
        distances, indices = distance_transform(
            data, halo=(tolerance, tolerance, tolerance), return_indices=True, block_shape=(64, 64, 64)
        )
        self._check_result_with_indices(data, distances, indices, tolerance)


if __name__ == "__main__":
    unittest.main()
