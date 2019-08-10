import os
import unittest
from shutil import rmtree
import numpy as np
try:
    import z5py
except ImportError:
    z5py = None


class TestCachedVolume(unittest.TestCase):
    tmp_dir = './tmp'

    def setUp(self):
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_dir)
        except OSError:
            pass

    # @unittest.skipUnless(z5py, "Need z5py")
    @unittest.skip
    def test_cached_volume(self):
        from elf.wrapper.cached_volume import CachedVolume
        shape = (256,) * 3
        data = np.random.rand(*shape)
        f = z5py.File(os.path.join(self.tmp_dir, 'data.n5'))
        ds = f.create_dataset('data', data=data,
                              compression='gzip', chunks=(16, 128, 128))
        cached = CachedVolume(ds, chunks=(4, 256, 256))

        n_reps = 5
        indices = [np.s_[:], np.s_[:128, 128:, :64], np.s_[:200],
                   np.s_[33:95, 57:211], np.s_[111:222, 47:223, 37:198]]
        for _ in n_reps:
            for index in indices:
                out1 = ds[index]
                out2 = cached[index]
                self.assertEqual(out1.shape, out2.shape)
                self.assertTrue(np.allclose(out1, out2))


if __name__ == '__main__':
    unittest.main()
