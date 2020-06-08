import os
import unittest
from shutil import rmtree

import numpy as np
try:
    import mrcfile
except ImportError:
    mrcfile = None


@unittest.skipIf(mrcfile is None, "Needs mrcfile")
class TestMrcWrapper(unittest.TestCase):
    tmp_dir = './tmp'
    out = './tmp/data.mrc'
    out_compressed = './tmp/data_compressed.mrc'

    def setUp(self):
        os.makedirs(self.tmp_dir)
        shape = (64, 64, 64)
        self.data = np.random.rand(*shape).astype('float32')

        with mrcfile.new(self.out) as f:
            f.set_data(self.data)

        with mrcfile.new(self.out_compressed, compression='gzip') as f:
            f.set_data(self.data)

    def tearDown(self):
        rmtree(self.tmp_dir)

    def check_dataset(self, ds):
        self.assertEqual(ds.shape, self.data.shape)
        self.assertEqual(ds.dtype, self.data.dtype)
        self.assertEqual(ds.size, self.data.size)
        self.assertEqual(ds.ndim, self.data.ndim)

        bbs = [np.s_[:], np.s_[1:5, 8:14, 6:12], np.s_[:, 13:54, 4:6]]
        for bb in bbs:
            out = ds[bb]
            exp = self.data[bb]
            self.assertTrue(np.array_equal(out, exp))

    def test_dataset(self):
        from elf.io.mrc_wrapper import MRCDataset
        with mrcfile.mmap(self.out) as f:
            ds = MRCDataset(f.data)
            self.check_dataset(ds)

    def test_dataset_compressed(self):
        from elf.io.mrc_wrapper import MRCDataset
        with mrcfile.open(self.out_compressed) as f:
            ds = MRCDataset(f.data)
            self.check_dataset(ds)

    def test_file(self):
        from elf.io.mrc_wrapper import MRCFile
        with MRCFile(self.out) as f:
            ds = f['data']
            self.check_dataset(ds)

    def test_file_compressed(self):
        from elf.io.mrc_wrapper import MRCFile
        with MRCFile(self.out_compressed) as f:
            ds = f['data']
            self.check_dataset(ds)


if __name__ == '__main__':
    unittest.main()
