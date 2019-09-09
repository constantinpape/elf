import unittest
import numpy as np

try:
    import nifty.tools as nt
except ImportError:
    nt = None


class TestUtil(unittest.TestCase):
    def test_normalize_index(self):
        from elf.util import normalize_index, squeeze_singletons
        shape = (128,) * 3
        x = np.random.rand(*shape)

        # is something important missing?
        indices = (np.s_[10:25, 30:60, 100:103],  # full index
                   np.s_[:],  # everything
                   np.s_[..., 10:25],  # ellipsis
                   np.s_[0, :, 10])  # singletons
        for index in indices:
            out1 = x[index]
            index, to_squeeze = normalize_index(index, shape)
            out2 = squeeze_singletons(x[index], to_squeeze)
            self.assertEqual(out1.shape, out2.shape)
            self.assertTrue(np.allclose(out1, out2))

    @unittest.skipUnless(nt, "Test needs nifty tools")
    def test_downscale_shape(self):
        from elf.util import downscale_shape
        n = 10
        max_len = 1024
        max_scale = 6
        for ndim in (2, 3, 4):
            origin = [0] * ndim
            for _ in range(n):
                shape = tuple(np.random.randint(1, max_len, size=ndim))
                scale = tuple(np.random.randint(1, max_scale, size=ndim))
                ds_shape = downscale_shape(shape, scale)
                exp_shape = tuple(nt.blocking(origin, list(shape), list(scale)).blocksPerAxis)
                self.assertEqual(ds_shape, exp_shape)


if __name__ == '__main__':
    unittest.main()
