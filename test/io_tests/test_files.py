import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from shutil import rmtree

import numpy as np
from elf.io.extensions import h5py, z5py, pyn5, zarr


@contextmanager
def patch_constructors(constructor, ext):
    with patch("elf.io.extensions.FILE_CONSTRUCTORS", {ext: constructor}):
        yield


class FileTestBase(unittest.TestCase):
    # todo: use https://github.com/clbarnes/tempcase/
    tmp_dir = "./tmp"

    def setUp(self):
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_dir)
        except OSError:
            pass

    def path_to(self, *args):
        return os.path.join("./tmp", *args)


class FileTestMixin:
    ext = None
    constructor = None

    def test_open(self):
        from elf.io import open_file
        shape = (128,) * 2
        data = np.random.rand(*shape)
        fname = self.path_to("data" + self.ext)
        with self.constructor(fname, 'a') as f:
            f.create_dataset('data', data=data)

        with patch("elf.io.extensions.FILE_CONSTRUCTORS", {self.ext: self.constructor}):
            with open_file(fname) as f:
                out = f['data'][:]

        self.assertEqual(data.shape, out.shape)
        self.assertTrue(np.allclose(data, out))

    def test_is_group(self):
        from elf.io import is_group
        f = self.constructor(self.path_to("data" + self.ext), mode="a")
        g = f.create_group('group')
        ds = f.create_dataset('dataset', data=np.ones((100, 100)),
                              chunks=(10, 10))
        self.assertTrue(is_group(f))
        self.assertTrue(is_group(g))
        self.assertFalse(is_group(ds))


@unittest.skipIf(h5py is None, "Need h5py")
class TestH5pyFiles(FileTestBase, FileTestMixin):
    ext = ".h5"
    constructor = getattr(h5py, "File", None)


@unittest.skipIf(z5py is None, "Need z5py")
class TestZ5pyN5Files(FileTestBase, FileTestMixin):
    ext = ".n5"
    constructor = getattr(z5py, "N5File", None)


@unittest.skipIf(z5py is None, "Need z5py")
class TestZ5pyZarrFiles(FileTestBase, FileTestMixin):
    ext = ".zr"
    constructor = getattr(z5py, "ZarrFile", None)


@unittest.skipIf(pyn5 is None, "Need pyn5")
class TestPyn5Files(FileTestBase, FileTestMixin):
    ext = ".n5"
    constructor = getattr(pyn5, "File", None)


@unittest.skipIf(zarr is None, "Need zarr")
class TestZarrFiles(FileTestBase, FileTestMixin):
    ext = ".zr"
    constructor = getattr(zarr, "open", None)


# todo: test loading N5 files using zarr-python

if __name__ == '__main__':
    unittest.main()
