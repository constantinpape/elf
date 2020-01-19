import unittest
import numpy as np

try:
    import nifty
except ImportError:
    nifty = None
try:
    import vigra
except ImportError:
    vigra = None


@unittest.skipUnless(nifty and vigra, "Need nifty and vigra")
class TestFilters(unittest.TestCase):
    def _test_filter(self, filt, filt_exp, sigma, inplace):
        shape = 3 * (64,)
        block_shape = 3 * (16,)
        x = np.random.rand(*shape)

        exp = filt_exp(x, sigma)
        if inplace:
            filt(x, sigma, block_shape=block_shape)
            self.assertTrue(np.allclose(exp, x))
        else:
            x_cpy = x.copy()
            res = np.zeros_like(x)
            res = filt(x, sigma, out=res, block_shape=block_shape)
            self.assertTrue(np.allclose(exp, res))
            # make sure x is unchaged
            self.assertTrue(np.allclose(x, x_cpy))

    def test_gaussian_smoothing(self):
        from elf.parallel.filters import gaussian_smoothing
        for sigma in (1.6, 3.5, 5.):
            self._test_filter(gaussian_smoothing, vigra.filters.gaussianSmoothing,
                              sigma=sigma, inplace=False)
            self._test_filter(gaussian_smoothing, vigra.filters.gaussianSmoothing,
                              sigma=sigma, inplace=True)

    # TODO test the other filters


if __name__ == '__main__':
    unittest.main()
