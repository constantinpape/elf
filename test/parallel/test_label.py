import unittest
import numpy as np
from skimage.data import binary_blobs
from skimage.measure import label as label_reference
from skimage.metrics import variation_of_information


class TestLabel(unittest.TestCase):
    def _check_labeling(self, data, res):
        exp = label_reference(data, connectivity=1)

        # debugging
        # import napari
        # v = napari.Viewer()
        # v.add_image(data)
        # v.add_labels(exp)
        # v.add_labels(res)
        # napari.run()

        vis, vim = variation_of_information(exp, res)
        self.assertAlmostEqual(vis + vim, 0)

    def test_label(self):
        from elf.parallel import label
        shape = 3 * (32,)
        block_shape = 3 * (16,)
        x = np.random.randint(1, 10, size=shape).astype("uint32")

        res = np.zeros_like(x)
        res = label(x, res, block_shape=block_shape, with_background=False)

        self._check_labeling(x, res)

    def test_label_with_background(self):
        from elf.parallel import label
        shape = 3 * (32,)
        block_shape = 3 * (16,)
        x = np.random.randint(0, 10, size=shape).astype("uint32")

        res = np.zeros_like(x)
        res = label(x, res, block_shape=block_shape, with_background=True)

        self._check_labeling(x, res)

    # stress test of the label function with binary blobs
    def _test_blobs(self, ndim, size):
        from elf.parallel import label
        block_shape = (size // 8,) * ndim
        for volume_fraction in (0.05, 0.1, 0.25, 0.5):
            data = binary_blobs(length=size, n_dim=ndim, volume_fraction=volume_fraction)
            res = np.zeros(data.shape, dtype="uint32")
            res = label(data, res, block_shape=block_shape, with_background=True)
            self._check_labeling(data, res)

    def test_label_blobs_2d(self):
        self._test_blobs(ndim=2, size=1024)

    def test_label_blobs_3d(self):
        self._test_blobs(ndim=3, size=256)

    def test_label_with_mask(self):
        from elf.parallel import label
        data = binary_blobs(length=1024, n_dim=2, volume_fraction=0.3)
        block_shape = (64, 64)

        bb = np.s_[96:-96, 132:-87]
        mask = np.zeros(data.shape, dtype="bool")
        mask[bb] = 1

        res = np.zeros(data.shape, dtype="uint32")
        res = label(data, res, block_shape=block_shape, with_background=True, mask=mask)

        self.assertTrue(np.allclose(res[~mask], 0))
        self._check_labeling(data[bb], res[bb])


if __name__ == "__main__":
    unittest.main()
