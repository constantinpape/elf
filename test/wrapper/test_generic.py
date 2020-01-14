import unittest
import numpy as np


class TestGenericWrapper(unittest.TestCase):
    shape = (32, 256, 256)

    def _test_generic(self, wrapper, trafo, **kwargs):
        x = np.random.randint(0, 255, size=self.shape)
        wrapped = wrapper(x, **kwargs)
        bbs = (np.s_[:], np.s_[:4, :9, : 135], np.s_[0, :-4, :88])
        for bb in bbs:
            out = wrapped[bb]
            exp = trafo(x[bb])
            self.assertTrue(np.array_equal(out, exp))

    def test_normalize_wrapper(self):
        from elf.wrapper import NormalizeWrapper

        def normalize(x):
            x = x.astype('float32')
            x -= x.min()
            x /= x.max()
            return x

        self._test_generic(NormalizeWrapper, normalize)

    def test_threshold_wrapper(self):
        from elf.wrapper import ThresholdWrapper
        self._test_generic(ThresholdWrapper, lambda x: np.greater(x, .5), threshold=.5)


if __name__ == '__main__':
    unittest.main()
