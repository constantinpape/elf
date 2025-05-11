import os
import unittest
from shutil import rmtree

import numpy as np
try:
    import zarr
    zarr_major_version = int(zarr.__version__.split(".", maxsplit=1)[0])
except ImportError:
    zarr = None
    zarr_major_version = None

try:
    import z5py
except ImportError:
    z5py = None

from elf.io import open_file


class ZarrWrapperTest(unittest.TestCase):
    tmp_dir = "./tmp"
    test_file = os.path.join(tmp_dir, "test-file.zarr")

    def setUp(self):
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_dir)
        except OSError:
            pass

    def test_zarr_wrapper(self):
        """Test that the zarr wrapper supports the legacy functions create_dataset and require_dataset.
        """
        f = open_file(self.test_file, mode="w")

        # Test create dataset with default options.
        ds_name = "test1"
        test_data = np.random.rand(128, 128)
        chunks = (64, 64)
        ds = f.create_dataset(ds_name, data=test_data, chunks=chunks, dtype="float64")
        self.assertEqual(chunks, ds.chunks)
        self.assertTrue(np.allclose(test_data, ds[:]))

        # Test require dataset with already existing data.
        ds = f.require_dataset(ds_name)
        self.assertEqual(chunks, ds.chunks)
        self.assertTrue(np.allclose(test_data, ds[:]))

        # Test require dataset with new data.
        ds = f.require_dataset(ds_name)
        self.assertEqual(chunks, ds.chunks)
        self.assertTrue(np.allclose(test_data, ds[:]))

        # TODO test with compression in the argument.

    @unittest.skipIf(z5py is None, "Test requires z5py to create data.")
    def test_zarr_v2(self):
        """Test that the zarr wrapper can read data in zarr v2 format created with z5py.
        """
        ds_name = "test"
        test_data = np.random.rand(128, 128)
        chunks = (64, 64)
        with z5py.File(self.test_file, "w") as f:
            f.create_dataset(ds_name, data=test_data, chunks=chunks)

        f = open_file(self.test_file, mode="r")
        ds = f[ds_name]
        self.assertEqual(chunks, ds.chunks)
        self.assertTrue(np.allclose(test_data, ds[:]))

    @unittest.skipUnless(zarr_major_version == 3, "Test only possible for zarr v3")
    def test_zarr_v3(self):
        """Test that the zarr wrapper supports the functions of zarr v3.
        """


if __name__ == "__main__":
    unittest.main()
