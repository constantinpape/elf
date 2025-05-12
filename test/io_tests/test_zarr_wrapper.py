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

    def _test_create_dataset_impl(self, method, prefix):
        from elf.io.zarr_wrapper import SUPPORTED_CODECS

        chunks = (64, 64)
        test_data = np.random.rand(128, 128)
        shape, dtype = test_data.shape, test_data.dtype

        # Test creating an empty dataset with only shape and dtype.
        ds_name = f"{prefix}-1"
        ds = method(ds_name, shape=shape, dtype=dtype, chunks=chunks)
        self.assertEqual(chunks, ds.chunks)
        self.assertTrue(shape, ds.shape)
        self.assertTrue(dtype, ds.dtype)

        # Test create dataset when data and shape are passed.
        ds_name = f"{prefix}-2"
        ds = method(ds_name, data=test_data, shape=shape, chunks=chunks)
        self.assertEqual(chunks, ds.chunks)
        self.assertTrue(np.allclose(test_data, ds[:]))

        # Test create dataset when data, dtype and shape are passed.
        ds_name = f"{prefix}-3"
        ds = method(ds_name, data=test_data, shape=shape, dtype=dtype, chunks=chunks)
        self.assertEqual(chunks, ds.chunks)
        self.assertTrue(np.allclose(test_data, ds[:]))

        # Tests with compression arguments.
        for i, codec in enumerate(SUPPORTED_CODECS, 4):
            ds_name = f"{prefix}-{i}"
            ds = method(ds_name, data=test_data, shape=shape, chunks=chunks, compression=codec)
            self.assertEqual(chunks, ds.chunks)
            self.assertTrue(np.allclose(test_data, ds[:]))

        return ds_name, test_data, chunks

    def test_zarr_wrapper(self):
        """Test that the zarr wrapper supports the legacy functions create_dataset and require_dataset.
        """
        f = open_file(self.test_file, mode="w")
        self.assertTrue(isinstance(f, zarr.Group))
        ds_name, test_data, chunks = self._test_create_dataset_impl(f.create_dataset, "test-create-dataset")

        # Test require dataset with already existing data.
        ds = f.require_dataset(ds_name, shape=test_data.shape)
        self.assertEqual(chunks, ds.chunks)
        self.assertTrue(np.allclose(test_data, ds[:]))

        # Test that require dataset works as expected with new data
        self._test_create_dataset_impl(f.require_dataset, "test-require-dataset")

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
        chunks = (64, 64)
        test_data = np.random.rand(128, 128)
        with open_file(self.test_file, mode="w") as f:
            array = f.create_array("test", shape=test_data.shape, dtype=test_data.dtype, chunks=chunks)
            array[:] = test_data

            self.assertEqual(chunks, array.chunks)
            self.assertTrue(np.allclose(test_data, array[:]))


if __name__ == "__main__":
    unittest.main()
