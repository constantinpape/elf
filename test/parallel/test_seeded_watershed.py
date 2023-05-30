import unittest
import numpy as np

from scipy.ndimage import binary_erosion
from skimage.data import binary_blobs
from skimage.filters import laplace
from skimage.measure import label
from skimage.metrics import variation_of_information
from skimage.segmentation import watershed


class TestSeededWatershed(unittest.TestCase):
    def test_seeded_watershed(self):
        from elf.parallel import seeded_watershed

        mask = binary_blobs(256)
        block_shape = (64, 64)
        halo = (16, 16)

        seeds = label(binary_erosion(mask, iterations=5))
        hmap = laplace(mask)

        res = np.zeros_like(seeds)
        res = seeded_watershed(hmap, seeds, res, block_shape, halo, mask=mask)
        exp = watershed(hmap, markers=seeds, mask=mask)

        vis, vim = variation_of_information(exp, res)
        self.assertLessEqual(vis + vim, 0.01)


if __name__ == "__main__":
    unittest.main()
