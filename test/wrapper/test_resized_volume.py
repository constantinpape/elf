import unittest
import numpy as np
try:
    import vigra
except ImportError:
    vigra = None


# TODO investigate failures
class TestResizedVolume(unittest.TestCase):

    def _check_index(self, out1, out2, index,
                     check_close=True, halo=8):
        o1 = out1[index]
        o2 = out2[index]
        self.assertEqual(o1.shape, o2.shape)
        if check_close:
            bb = tuple(slice(halo, sh - halo) for sh in o1.shape)
            self.assertTrue(np.allclose(o1[bb], o2[bb]))

    @unittest.expectedFailure
    @unittest.skipUnless(vigra, "resize functionality needs vigra")
    def test_downscale(self):
        from elf.wrapper.resized_volume import ResizedVolume
        shape = 3 * (256,)
        x = np.random.rand(*shape).astype('float32')

        out_shape = 3 * (128,)
        orders = (0, 3)
        indices = (np.s_[:], np.s_[:64, :, 64:], np.s_[:64, :48, 40:95])
        halo = 8
        for order in orders:
            out1 = vigra.sampling.resize(x, shape=out_shape,
                                         order=order)
            out2 = ResizedVolume(x, shape=out_shape, order=order)
            self.assertEqual(out1.shape, out2.shape)
            for index in indices:
                self._check_index(out1, out2, index,
                                  check_close=True, halo=halo)
            index = np.s_[32:96, 33:55, 70]
            self._check_index(out1, out2, index, check_close=False)

    @unittest.expectedFailure
    @unittest.skipUnless(vigra, "resize functionality needs vigra")
    def test_upscale(self):
        from elf.wrapper.resized_volume import ResizedVolume
        shape = 3 * (128,)
        x = np.random.rand(*shape).astype('float32')

        out_shape = 3 * (256,)
        orders = (0, 3)
        indices = (np.s_[:], np.s_[:128, :, 128:], np.s_[:128, :97, 123:250])
        halo = 8
        for order in orders:
            out1 = vigra.sampling.resize(x, shape=out_shape,
                                         order=order)
            out2 = ResizedVolume(x, shape=out_shape, order=order)
            self.assertEqual(out1.shape, out2.shape)
            for index in indices:
                self._check_index(out1, out2, index,
                                  check_close=True, halo=halo)
            index = np.s_[64:107, 153:179, 93]
            self._check_index(out1, out2, index, check_close=False)


if __name__ == '__main__':
    unittest.main()
