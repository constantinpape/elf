import os
import unittest
from shutil import rmtree

import numpy as np

try:
    import intern
except ImportError:
    intern = None


@unittest.skipIf(intern is None, "Needs intern (pip install intern)")
class TestInternWrapper(unittest.TestCase):
    def test_can_access_dataset(self):
        from elf.io.intern_wrapper import InternDataset

        # Choosing a dataset at random to make sure we can access shape and dtype
        ds = InternDataset("bossdb://witvliet2020/Dataset_1/em")
        self.assertEqual(ds.shape, (300, 36000, 22000))
        self.assertEqual(ds.dtype, np.uint8)
        self.assertEqual(ds.size, 300 * 36000 * 22000)
        self.assertEqual(ds.ndim, 3)

    def test_can_download_dataset(self):
        from elf.io.intern_wrapper import InternDataset

        ds = InternDataset("bossdb://witvliet2020/Dataset_1/em")
        cutout = ds[210:211, 7000:7064, 7000:7064]
        self.assertEqual(cutout.shape, (1, 64, 64))

    def test_file(self):
        from elf.io.intern_wrapper import InternFile, InternDataset

        f = InternFile("bossdb://witvliet2020/Dataset_1/em")
        ds = f["data"]
        self.assertIsInstance(ds, InternDataset)


if __name__ == "__main__":
    unittest.main()
