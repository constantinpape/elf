import unittest

import bioimage_cpp as bic
import numpy as np

from scipy.ndimage import binary_erosion
from skimage.data import binary_blobs
from skimage.filters import laplace
from skimage.measure import label
from skimage.metrics import variation_of_information


class TestSeededWatershed(unittest.TestCase):
    def test_seeded_watershed(self):
        from elf.parallel import seeded_watershed

        # Seed binary_blobs so the test data is deterministic across runs; otherwise the
        # blockwise/global discrepancy varies per run and occasionally exceeds the threshold.
        mask = binary_blobs(256, rng=0)
        block_shape = (64, 64)
        halo = (16, 16)

        seeds = label(binary_erosion(mask, iterations=5)).astype("uint32")
        # Add a tiny random perturbation to the heightmap to break ties — the bic watershed
        # uses unspecified tie-breaking, so equal-height plateaus would otherwise diverge
        # between the blockwise and global runs.
        rng = np.random.default_rng(seed=0)
        hmap = (laplace(mask) + 1e-3 * rng.random(mask.shape)).astype("float32")

        res = np.zeros_like(seeds)
        res = seeded_watershed(hmap, seeds, res, block_shape, halo, mask=mask)
        exp = bic.segmentation.watershed(hmap, seeds, mask=mask)

        # The blockwise watershed is an approximation of the global one: a few pixels near
        # block boundaries may be assigned to a different seed. The tolerance accounts for
        # this expected, small discrepancy.
        vis, vim = variation_of_information(exp, res)
        self.assertLessEqual(vis + vim, 0.01)


if __name__ == "__main__":
    unittest.main()
