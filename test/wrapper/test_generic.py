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
            x = x.astype("float32")
            x -= x.min()
            x /= x.max()
            return x

        self._test_generic(NormalizeWrapper, normalize)

    def test_threshold_wrapper(self):
        from elf.wrapper import ThresholdWrapper
        self._test_generic(ThresholdWrapper, lambda x: np.greater(x, .5), threshold=.5)

    def test_roi_wrapper(self):
        from elf.wrapper import RoiWrapper

        x = np.random.rand(*self.shape)
        bbs = [np.s_[:], np.s_[:16, :128, :128], np.s_[0], np.s_[1:29, 3:17, 137:211], np.s_[:, 5:47, 77:73]]
        for bb in bbs:
            res = RoiWrapper(x, bb)[:]
            exp = x[bb]
            self.assertTrue(np.allclose(res, exp))

    def test_roi_wrapper_squeeze(self):
        from elf.wrapper import RoiWrapper

        x = np.random.rand(*self.shape)
        # Each roi introduces one or more singleton axes via integer indexing.
        bbs = [np.s_[0], np.s_[5, 10], np.s_[3, 4, 5], np.s_[2, 10:40, :]]
        for bb in bbs:
            wrapped = RoiWrapper(x, bb, squeeze=True)
            exp = x[bb]
            self.assertEqual(wrapped.shape, np.shape(exp))
            # A fully-reduced (scalar) wrapper must be indexed with `()`, just like numpy.
            res = wrapped[()] if wrapped.shape == () else wrapped[:]
            self.assertEqual(np.shape(res), np.shape(exp))
            self.assertTrue(np.array_equal(res, exp))

        # Sub-indexing a wrapper with squeezed roi axes maps to the correct volume region.
        wrapped = RoiWrapper(x, np.s_[2, 10:40, :], squeeze=True)
        self.assertTrue(np.array_equal(wrapped[:5, :5], x[2, 10:15, 0:5]))

        # Round-trip setitem writes into the correct volume region.
        y = np.zeros(self.shape)
        wrapped = RoiWrapper(y, np.s_[2, 10:40, :], squeeze=True)
        val = np.random.rand(*wrapped.shape)
        wrapped[:] = val
        self.assertTrue(np.array_equal(y[2, 10:40, :], val))

        # squeeze=False (default) keeps the singleton axes.
        wrapped = RoiWrapper(x, np.s_[0], squeeze=False)
        self.assertEqual(wrapped.shape, (1,) + self.shape[1:])


if __name__ == '__main__':
    unittest.main()
