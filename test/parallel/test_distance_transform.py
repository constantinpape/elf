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

    def test_map_points_to_objects(self):
        from elf.parallel.distance_transform import map_points_to_objects
        from skimage.measure import label as label_reference

        rng = np.random.default_rng(seed=7)
        size = 128
        blobs = binary_blobs(length=size, n_dim=2, volume_fraction=0.3, rng=11)
        segmentation = label_reference(blobs).astype("uint32")

        # Sample random points within the image bounds.
        n_points = 20
        points = rng.integers(0, size, size=(n_points, 2))

        halo = (24, 24)
        object_ids, object_distances = map_points_to_objects(
            segmentation, points, block_shape=(64, 64), halo=halo, n_threads=2,
        )

        # Ground truth: compute the global distance transform once and look up per point.
        bg_distances, bg_indices = distance_transform_edt(segmentation == 0, return_indices=True)
        expected_ids = np.zeros(n_points, dtype=segmentation.dtype)
        expected_dists = np.zeros(n_points, dtype="float32")
        for i, p in enumerate(points):
            expected_dists[i] = bg_distances[tuple(p)]
            expected_ids[i] = segmentation[tuple(bg_indices[:, p[0], p[1]])]

        # Points outside the halo distance to the nearest object may not be reached;
        # restrict comparison to points within the halo distance.
        max_halo_distance = min(halo)
        within_halo = expected_dists <= max_halo_distance
        self.assertTrue(np.all(object_ids[within_halo] == expected_ids[within_halo]))
        self.assertTrue(np.allclose(object_distances[within_halo], expected_dists[within_halo]))


if __name__ == "__main__":
    unittest.main()
