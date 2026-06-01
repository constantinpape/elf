import unittest

import numpy as np
from scipy.ndimage import affine_transform


class TestResizedVolume(unittest.TestCase):
    def _reference(self, data, out_shape, order):
        # A resize is a coordinate scaling, which corresponds to an affine transform
        # with a diagonal scale matrix mapping output to input coordinates.
        scale = [sh / float(osh) for sh, osh in zip(data.shape, out_shape)]
        matrix = np.diag(scale + [1.0])
        return affine_transform(data.astype("float64"), matrix, order=order, output_shape=out_shape)

    def _check_index(self, wrapper, reference, index, halo=4):
        out = wrapper[index]
        ref = reference[index]
        self.assertEqual(out.shape, ref.shape)
        # Cut away the halo so that we don't compare boundary artefacts.
        if halo > 0 and all(sh > 2 * halo for sh in out.shape):
            bb = tuple(slice(halo, sh - halo) for sh in out.shape)
            out, ref = out[bb], ref[bb]
        self.assertTrue(np.allclose(out, ref, atol=1e-3))

    def _test_resize(self, shape, out_shape, indices, orders=(0, 1)):
        from elf.wrapper.resized_volume import ResizedVolume

        data = np.random.rand(*shape).astype("float32")
        for order in orders:
            wrapper = ResizedVolume(data, shape=out_shape, order=order)
            self.assertEqual(wrapper.shape, out_shape)
            reference = self._reference(data, out_shape, order)
            for index in indices:
                self._check_index(wrapper, reference, index)

    def test_downscale_3d(self):
        shape = 3 * (64,)
        out_shape = 3 * (32,)
        indices = (np.s_[:], np.s_[1:-1, 2:-2, 3:-3], np.s_[:16, 16:, :], np.s_[3:29, 7:24, 8:31])
        self._test_resize(shape, out_shape, indices)

    def test_upscale_3d(self):
        shape = 3 * (32,)
        out_shape = 3 * (64,)
        indices = (np.s_[:], np.s_[1:-1, 2:-2, 3:-3], np.s_[:32, 32:, :], np.s_[12:53, 7:54, 8:33])
        self._test_resize(shape, out_shape, indices)

    def test_anisotropic_3d(self):
        shape = (32, 64, 48)
        out_shape = (64, 32, 96)
        indices = (np.s_[:], np.s_[1:-1, 2:-2, 3:-3], np.s_[:32, 16:, :])
        self._test_resize(shape, out_shape, indices)

    def test_downscale_2d(self):
        shape = (128, 128)
        out_shape = (64, 100)
        indices = (np.s_[:], np.s_[1:-1, 2:-2], np.s_[:32, 32:], np.s_[12:53, 27:99])
        self._test_resize(shape, out_shape, indices)

    def test_upscale_2d(self):
        shape = (64, 100)
        out_shape = (128, 128)
        indices = (np.s_[:], np.s_[1:-1, 2:-2], np.s_[:64, 64:], np.s_[12:53, 27:111])
        self._test_resize(shape, out_shape, indices)

    def test_singleton_index(self):
        from elf.wrapper.resized_volume import ResizedVolume

        data = np.random.rand(32, 32, 32).astype("float32")
        out_shape = (64, 64, 64)
        wrapper = ResizedVolume(data, shape=out_shape, order=1)
        reference = self._reference(data, out_shape, order=1)
        # Indexing that squeezes out singleton dimensions must keep working.
        for index in (np.s_[4, 5, 6], np.s_[10, :, :], np.s_[:, 17, 3:40]):
            out = wrapper[index]
            ref = reference[index]
            self.assertEqual(out.shape, ref.shape)

    def test_bool_mask(self):
        from elf.wrapper.resized_volume import ResizedVolume

        # bool is not supported by the affine transform, so the wrapper round-trips through uint8.
        mask = np.zeros((64, 64, 64), dtype="bool")
        mask[16:48, 16:48, 16:48] = True
        out_shape = (32, 32, 32)
        wrapper = ResizedVolume(mask, shape=out_shape, order=0)
        out = wrapper[:]
        self.assertEqual(out.dtype, np.dtype("bool"))
        self.assertEqual(out.shape, out_shape)
        # The down-sampled center block should still be set, the corners should be empty.
        self.assertTrue(out[12:20, 12:20, 12:20].all())
        self.assertFalse(out[:4, :4, :4].any())

    def test_label_dtype(self):
        from elf.wrapper.resized_volume import ResizedVolume

        labels = np.random.randint(0, 100, size=(64, 64, 64)).astype("uint16")
        out_shape = (32, 32, 32)
        wrapper = ResizedVolume(labels, shape=out_shape, order=0)
        out = wrapper[:]
        self.assertEqual(out.dtype, np.dtype("uint16"))
        self.assertEqual(out.shape, out_shape)
        # Nearest-neighbor resizing must not introduce labels outside the input range.
        self.assertTrue(out.max() <= labels.max())

    def test_empty_and_constant_blocks(self):
        from elf.wrapper.resized_volume import ResizedVolume

        out_shape = (32, 32, 32)
        zeros = np.zeros((64, 64, 64), dtype="float32")
        self.assertEqual(ResizedVolume(zeros, shape=out_shape, order=1)[:].sum(), 0)

        ones = np.ones((64, 64, 64), dtype="float32")
        self.assertTrue(np.allclose(ResizedVolume(ones, shape=out_shape, order=1)[:], 1))

    def test_invalid_input(self):
        from elf.wrapper.resized_volume import ResizedVolume

        with self.assertRaises(ValueError):
            ResizedVolume(np.zeros((32, 32, 32)), shape=(16, 16))
        with self.assertRaises(ValueError):
            ResizedVolume(np.zeros((4, 4, 4, 4)), shape=(2, 2, 2, 2))


if __name__ == "__main__":
    unittest.main()
