import os
import unittest
from shutil import rmtree

import numpy as np
import imageio

try:
    import tifffile
except ImportError:
    tifffile = None


class TestImageStackWrapper(unittest.TestCase):
    tmp_dir = './tmp'
    slice_dir = './tmp/slices'
    stack_path = './tmp/stack.tif'
    pattern = '*.tiff'
    shape = (16, 128, 128)

    def setUp(self):
        os.makedirs(self.slice_dir)
        self.data = np.random.randint(0, 128, dtype='uint8', size=self.shape)
        for z in range(self.data.shape[0]):
            name = 'z%03i.tiff' % z
            path = os.path.join(self.slice_dir, name)
            imageio.imwrite(path, self.data[z])

        imageio.volwrite(self.stack_path, self.data)

    def tearDown(self):
        try:
            rmtree(self.tmp_dir)
        except OSError:
            pass

    def _check_ds(self, ds):

        self.assertEqual(ds.shape, self.data.shape)
        self.assertEqual(ds.dtype, self.data.dtype)
        self.assertEqual(ds.size, self.data.size)
        self.assertEqual(ds.ndim, self.data.ndim)

        bbs = [np.s_[:], np.s_[1:5, 8:14, 6:12]]
        for bb in bbs:
            out = ds[bb]
            exp = self.data[bb]
            self.assertTrue(np.array_equal(out, exp))

    def test_dataset(self):
        from elf.io.image_stack_wrapper import ImageStackDataset
        ds = ImageStackDataset.from_pattern(self.slice_dir, self.pattern)
        self._check_ds(ds)

    @unittest.skipIf(tifffile is None, "Need tifffile")
    def test_dataset_tif(self):
        from elf.io.image_stack_wrapper import TifStackDataset
        ds = TifStackDataset.from_pattern(self.slice_dir, self.pattern)
        self._check_ds(ds)

    @unittest.skipIf(tifffile is None, "Need tifffile")
    def test_stack_tif(self):
        from elf.io.image_stack_wrapper import TifStackDataset
        ds = TifStackDataset.from_stack(self.stack_path)
        self._check_ds(ds)

    def test_file(self):
        from elf.io.image_stack_wrapper import ImageStackFile
        f = ImageStackFile(self.slice_dir)
        ds = f[self.pattern]
        self._check_ds(ds)

    def test_stack(self):
        from elf.io.image_stack_wrapper import ImageStackDataset
        ds = ImageStackDataset.from_stack(self.stack_path)
        self._check_ds(ds)


if __name__ == '__main__':
    unittest.main()
