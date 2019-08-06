import os
import unittest
from shutil import rmtree
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import z5py
except ImportError:
    z5py = None


class TestFiles(unittest.TestCase):
    tmp_dir = './tmp'

    def setUp(self):
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_dir)
        except OSError:
            pass

    def _test_open(self, file_constructor, ext):
        from elf.io import open_file
        shape = (128,) * 2
        data = np.random.rand(*shape)
        fname = os.path.join(self.tmp_dir, 'data%s' % ext)
        with file_constructor(fname) as f:
            f.create_dataset('data', data=data)

        with open_file(fname) as f:
            out = f['data'][:]

        self.assertEqual(data.shape, out.shape)
        self.assertTrue(np.allclose(data, out))

    @unittest.skipIf(h5py is None, "Need h5py")
    def test_open_h5(self):
        self._test_open(h5py.File, '.h5')

    @unittest.skipIf(z5py is None, "Need z5py")
    def test_open_n5(self):
        self._test_open(z5py.N5File, '.n5')

    @unittest.skipIf(z5py is None, "Need z5py")
    def test_open_zarr(self):
        self._test_open(z5py.ZarrFile, '.zr')


if __name__ == '__main__':
    unittest.main()
