import os
import unittest

from pathlib import Path
from shutil import rmtree
from unittest.mock import patch

import imageio.v3 as imageio
import numpy as np
from elf.io.extensions import h5py, z5py, pyn5, zarr, zarr_open, FILE_CONSTRUCTORS


class FileTestBase(unittest.TestCase):
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
        with self.constructor(fname, "a") as f:
            f.create_dataset("data", data=data)

        with patch("elf.io.extensions.FILE_CONSTRUCTORS", {self.ext: self.constructor}):
            with open_file(fname) as f:
                out = f["data"][:]

        self.assertEqual(data.shape, out.shape)
        self.assertTrue(np.allclose(data, out))

        fname_path = Path(fname)
        with patch("elf.io.extensions.FILE_CONSTRUCTORS", {self.ext: self.constructor}):
            with open_file(fname_path) as f:
                out = f["data"][:]

        self.assertEqual(data.shape, out.shape)
        self.assertTrue(np.allclose(data, out))

    def test_is_group(self):
        from elf.io import is_group

        f = self.constructor(self.path_to("data" + self.ext), mode="a")
        g = f.create_group("group")
        ds = f.create_dataset("dataset", data=np.ones((100, 100)), chunks=(10, 10))
        self.assertTrue(is_group(f))
        self.assertTrue(is_group(g))
        self.assertFalse(is_group(ds))


@unittest.skipUnless(h5py, "Need h5py")
class TestH5pyFiles(FileTestBase, FileTestMixin):
    ext = ".h5"
    constructor = getattr(h5py, "File", None)


@unittest.skipUnless(z5py, "Need z5py")
class TestZ5pyN5Files(FileTestBase, FileTestMixin):
    ext = ".n5"
    constructor = getattr(z5py, "N5File", None)


@unittest.skipUnless(z5py, "Need z5py")
class TestZ5pyZarrFiles(FileTestBase, FileTestMixin):
    ext = ".zr"
    constructor = getattr(z5py, "ZarrFile", None)


@unittest.skipUnless(pyn5, "Need pyn5")
class TestPyn5Files(FileTestBase, FileTestMixin):
    ext = ".n5"
    constructor = getattr(pyn5, "File", None)


@unittest.skipUnless(zarr, "Need zarr")
class TestZarrFiles(FileTestBase, FileTestMixin):
    ext = ".zr"
    constructor = staticmethod(zarr_open)


class TestBackendPreference(unittest.TestCase):
    @unittest.skipUnless(z5py and zarr, "Need z5py and zarr")
    def test_zarr_over_z5py(self):
        self.assertTrue(FILE_CONSTRUCTORS[".n5"] == zarr_open)

    @unittest.skipUnless(z5py and pyn5, "Need z5py and pyn5")
    def test_z5py_over_pyn5(self):
        self.assertTrue(issubclass(FILE_CONSTRUCTORS[".zr"], z5py.File))


class TestImageStack(unittest.TestCase):
    tmp_dir = "./tmp"
    pattern = "*.tiff"
    shape = (16, 128, 128)

    def tearDown(self):
        try:
            rmtree(self.tmp_dir)
        except OSError:
            pass

    def setUp(self):
        os.makedirs(self.tmp_dir)
        self.data = np.random.randint(0, 128, dtype="uint8", size=self.shape)
        for z in range(self.data.shape[0]):
            name = "z%03i.tiff" % z
            path = os.path.join(self.tmp_dir, name)
            imageio.imwrite(path, self.data[z])

    def test_open_image_stack(self):
        from elf.io import open_file
        from elf.io.image_stack_wrapper import ImageStackDataset, ImageStackFile

        f = open_file(self.tmp_dir)
        self.assertIsInstance(f, ImageStackFile)

        ds = f[self.pattern]
        self.assertIsInstance(ds, ImageStackDataset)


# TODO:
# test loading N5 files using zarr-python
# test loading knossos file

if __name__ == "__main__":
    unittest.main()
