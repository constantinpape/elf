import unittest

import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max


class TestLocalMaxima(unittest.TestCase):
    def _test_find_local_maxima(self, shape, block_shape):
        from elf.parallel.local_maxima import find_local_maxima

        # Create test data.
        x = np.random.rand(*shape)
        x = (x > 0.995).astype("float32")
        x = gaussian(x)
        x /= x.max()

        min_distance = 1
        result = find_local_maxima(x, min_distance=min_distance, n_threads=1, block_shape=block_shape)
        expected_result = peak_local_max(x, min_distance=min_distance)

        # Apply consistent ordering.
        result = result[np.lexsort(result.T[::-1])]
        expected_result = expected_result[np.lexsort(expected_result.T[::-1])]

        # Check that result and expected results agree.
        self.assertEqual(result.shape, expected_result.shape)
        agreement = np.isclose(result, expected_result).all(axis=1)
        self.assertTrue(agreement.all())

    def test_find_local_maxima_2d(self):
        shape = (512, 512)
        block_shape = (128, 128)
        self._test_find_local_maxima(shape, block_shape)

    def test_find_local_maxima_3d(self):
        shape = 3 * (128,)
        block_shape = 3 * (32,)
        self._test_find_local_maxima(shape, block_shape)


if __name__ == "__main__":
    unittest.main()
