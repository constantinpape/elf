import unittest
import numpy as np


class TestCopyDataset(unittest.TestCase):
    def test_copy_dataset(self):
        from elf.parallel import copy_dataset

        rng = np.random.default_rng(seed=0)
        shape = 3 * (32,)
        block_shape = 3 * (8,)
        src = rng.random(shape, dtype="float32")
        dst = np.zeros_like(src)

        out = copy_dataset(src, dst, block_shape=block_shape, n_threads=2)
        self.assertTrue(np.array_equal(out, src))
        self.assertTrue(np.array_equal(dst, src))

    def test_copy_dataset_with_roi(self):
        from elf.parallel import copy_dataset

        rng = np.random.default_rng(seed=1)
        shape = 3 * (32,)
        block_shape = 3 * (8,)
        src = rng.random(shape, dtype="float32")
        dst = np.zeros_like(src)

        roi_in = np.s_[8:24, 8:24, 8:24]
        roi_out = np.s_[8:24, 8:24, 8:24]
        copy_dataset(src, dst, roi_in=roi_in, roi_out=roi_out, block_shape=block_shape, n_threads=2)
        self.assertTrue(np.array_equal(dst[roi_out], src[roi_in]))
        # Pixels outside the ROI should remain zero.
        mask = np.ones(shape, dtype=bool)
        mask[roi_out] = False
        self.assertTrue(np.all(dst[mask] == 0))


if __name__ == "__main__":
    unittest.main()
