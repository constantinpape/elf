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
        with file_constructor(fname, 'a') as f:
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

    def _test_is_group(self, f):
        from elf.io import is_group
        g = f.create_group('group')
        ds = f.create_dataset('dataset', data=np.ones((100, 100)),
                              chunks=(10, 10))
        self.assertTrue(is_group(f))
        self.assertTrue(is_group(g))
        self.assertFalse(is_group(ds))

    @unittest.skipIf(h5py is None, "Need h5py")
    def test_is_group_h5py(self):
        f = h5py.File(os.path.join(self.tmp_dir, 'data.h5'), 'a')
        self._test_is_group(f)

    @unittest.skipIf(z5py is None, "Need z5py")
    def test_is_group_z5py(self):
        f = z5py.File(os.path.join(self.tmp_dir, 'data.n5'), 'a')
        self._test_is_group(f)


if __name__ == '__main__':
    unittest.main()
