import unittest

import matplotlib
matplotlib.use("Agg")  # noqa: E402  Use a non-interactive backend for tests.

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


class TestSizeHistogram(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_size_histogram(self):
        from elf.visualisation import plot_size_histogram

        sizes = np.array([1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 300])
        threshold = plot_size_histogram(sizes, n_bins=8, bin_for_threshold=2)
        self.assertIsInstance(float(threshold), float)
        self.assertGreater(float(threshold), 0.0)

    def test_plot_size_histogram_min_max(self):
        from elf.visualisation import plot_size_histogram

        sizes = np.array([1, 5, 10, 50, 100, 500, 1000])
        threshold = plot_size_histogram(
            sizes, n_bins=4, bin_for_threshold=1, min_size=5, max_size=500
        )
        self.assertGreaterEqual(float(threshold), 5.0)
        self.assertLessEqual(float(threshold), 500.0)

    def test_size_histogram_from_segmentation(self):
        from elf.visualisation.size_histogram import size_histogram_from_segmentation

        seg = np.zeros((20, 20), dtype="uint32")
        seg[0:2, 0:2] = 1     # size 4
        seg[5:9, 5:9] = 2     # size 16
        seg[12:18, 12:19] = 3  # size 42
        threshold = size_histogram_from_segmentation(seg, n_bins=4, bin_for_threshold=1)
        self.assertGreater(float(threshold), 0.0)

    def test_ignore_background(self):
        from elf.visualisation.size_histogram import size_histogram_from_segmentation

        seg = np.zeros((10, 10), dtype="uint32")
        seg[1:3, 1:3] = 1
        seg[6:9, 6:9] = 2
        # Should not raise and should return a finite threshold with background ignored.
        threshold = size_histogram_from_segmentation(
            seg, n_bins=2, bin_for_threshold=1, ignore_background=True
        )
        self.assertTrue(np.isfinite(threshold))


if __name__ == "__main__":
    unittest.main()
