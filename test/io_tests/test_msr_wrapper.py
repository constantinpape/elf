import os
import unittest
from pathlib import Path
from shutil import copyfile, copyfileobj, rmtree
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

try:
    from msr_reader import OBFFile
except ImportError:
    OBFFile = None


@unittest.skipIf(OBFFile is None, "Needs msr_reader")
class TestMSRWrapper(unittest.TestCase):
    sample_url = "https://owncloud.gwdg.de/index.php/s/3WZ5DDBCQqyU2Jj/download"
    tmp_dir = "./tmp"
    sample_path = Path(tmp_dir) / "sample.msr"
    sample_path_1 = Path(tmp_dir) / "sample_1.msr"
    sample_path_2 = Path(tmp_dir) / "sample_2.msr"

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.tmp_dir, exist_ok=True)
        if not cls.sample_path.exists():
            try:
                with urlopen(cls.sample_url, timeout=120) as src, open(cls.sample_path, "wb") as dst:
                    copyfileobj(src, dst)
            except URLError as exc:
                raise unittest.SkipTest(f"Could not download MSR test data: {exc}")

        copyfile(cls.sample_path, cls.sample_path_1)
        copyfile(cls.sample_path, cls.sample_path_2)

        with OBFFile(os.fspath(cls.sample_path)) as msr:
            cls.overview_name = msr.stack_names[0]
            cls.pre_name = msr.stack_names[6]
            cls.post_name = msr.stack_names[7]
            cls.overview = msr.read_stack(0)
            cls.pre = msr.read_stack(6)
            cls.post = msr.read_stack(7)
            cls.pre_post = np.stack([cls.pre, cls.post], axis=0)

    @classmethod
    def tearDownClass(cls):
        try:
            rmtree(cls.tmp_dir)
        except OSError:
            pass

    def test_dataset_single_stack(self):
        from elf.io import MSRDataset

        ds = MSRDataset(self.sample_path, self.pre_name)
        self.assertEqual(ds.shape, self.pre.shape)
        self.assertEqual(ds.dtype, self.pre.dtype)
        self.assertEqual(ds.size, self.pre.size)
        self.assertEqual(ds.ndim, self.pre.ndim)

        for bb in (np.s_[:], np.s_[10:32, 15:48]):
            self.assertTrue(np.array_equal(ds[bb], self.pre[bb]))

    def test_dataset_multiple_stacks(self):
        from elf.io import MSRDataset

        ds = MSRDataset(self.sample_path, (self.pre_name, self.post_name))
        self.assertEqual(ds.shape, self.pre_post.shape)
        self.assertEqual(ds.dtype, self.pre_post.dtype)
        self.assertEqual(ds.size, self.pre_post.size)
        self.assertEqual(ds.ndim, self.pre_post.ndim)

        for bb in (np.s_[:], np.s_[:, 10:32, 15:48], 1):
            self.assertTrue(np.array_equal(ds[bb], self.pre_post[bb]))

    def test_sample_collection(self):
        from elf.io import MSRSampleCollection

        collection = MSRSampleCollection(
            [self.sample_path_1, self.sample_path_2],
            stack_names=(self.pre_name, self.post_name),
        )
        self.assertEqual(collection.shape, self.pre_post.shape)
        self.assertEqual(collection.dtype, self.pre_post.dtype)
        self.assertEqual(collection.size, self.pre_post.size)
        self.assertEqual(collection.ndim, self.pre_post.ndim)
        self.assertIsNone(collection.chunks)
        self.assertEqual(collection.attrs, {})

        self.assertTrue(np.array_equal(collection.read_sample(0), self.pre_post))
        self.assertTrue(np.array_equal(collection.read_sample(1), self.pre_post))

    def test_file(self):
        from elf.io import MSRFile

        with MSRFile(self.sample_path) as f:
            self.assertIn("0", f)
            self.assertIn("6", f)
            self.assertIn(self.pre_name, f)
            self.assertIn("data", f)
            self.assertEqual(len(f), 9)

            self.assertTrue(np.array_equal(f["0"][:], self.overview))
            self.assertTrue(np.array_equal(f["6"][:], self.pre))
            self.assertTrue(np.array_equal(f[self.pre_name][:], self.pre))
            self.assertTrue(np.array_equal(f[self.post_name][:], self.post))

    def test_open_file(self):
        from elf.io import open_file, MSRFile

        with open_file(self.sample_path) as f:
            self.assertIsInstance(f, MSRFile)
            self.assertTrue(np.array_equal(f[self.pre_name][:], self.pre))


if __name__ == "__main__":
    unittest.main()
