import unittest
import numpy as np
try:
    import vigra
except ImportError:
    vigra = None


@unittest.skipUnless(vigra, "resize functionality needs vigra")
class TestResizedVolume(unittest.TestCase):

    def _check_index(self, out1, out2, index, halo):
        o1, o2 = out1[index], out2[index]
        self.assertEqual(o1.shape, o2.shape)

        # cut away halo, so that we don't compare boundary artefacts
        if halo > 0:
            bb = tuple(slice(halo, sh - halo) for sh in o1.shape)
        else:
            bb = np.s_[:]
        o1, o2 = o1[bb], o2[bb]
        self.assertTrue(np.allclose(o1[bb], o2[bb]))

    def _test_toy(self):
        from elf.wrapper.resized_volume import ResizedVolume
        from skimage.transform import resize
        from skimage.data import astronaut
        x = astronaut()[..., 0].astype('float32')
        x = vigra.sampling.resize(x, (256, 256))
        x = np.concatenate(256 * [x[None]], axis=0)

        out_shape = 3 * (128,)
        order = 0
        out1 = vigra.sampling.resize(x, shape=out_shape, order=order)
        out2 = ResizedVolume(x, shape=out_shape, order=order)
        out3 = resize(x, out_shape, order=0, preserve_range=True, anti_aliasing=False)
        assert out1.shape == out2.shape == out_shape
        # bb = np.s_[:64, :, 64:]
        bb = np.s_[:]
        o1 = out1[bb]
        o2 = out2[bb]
        o3 = out3[bb]
        import napari
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(o1, name='elf')
            viewer.add_image(o2, name='vigra')
            viewer.add_image(o3, name='skimage')
            # viewer.add_labels(diff, name='pix-diff')

    def _test_resize(self, shape, out_shape, indices):
        from elf.wrapper.resized_volume import ResizedVolume
        x = np.random.rand(*shape).astype('float32')

        halo = 8
        orders = [0, 3]
        for order in orders:
            out1 = vigra.sampling.resize(x, shape=out_shape, order=order)
            out2 = ResizedVolume(x, shape=out_shape, order=order)
            self.assertEqual(out1.shape, out2.shape)
            self.assertEqual(out1.shape, out_shape)
            for index in indices:
                self._check_index(out1, out2, index, halo=halo)

    def test_downscale_full_volume(self):
        shape = 3 * (256,)
        out_shape = 3 * (128,)
        self._test_resize(shape, out_shape, [np.s_[:]])

    @unittest.expectedFailure
    def test_downscale(self):
        shape = 3 * (256,)
        out_shape = 3 * (128,)
        indices = (np.s_[1:-1, 2:-2, 3:-3], np.s_[:64, :, 64:],
                   np.s_[:64, :48, 40:95], np.s_[:])
        self._test_resize(shape, out_shape, indices)

    def test_upscale_full_volume(self):
        shape = 3 * (128,)
        out_shape = 3 * (256,)
        self._test_resize(shape, out_shape, [np.s_[:]])

    @unittest.expectedFailure
    def test_upscale(self):
        shape = 3 * (128,)
        out_shape = 3 * (256,)
        indices = (np.s_[1:-1, 2:-2, 3:-3], np.s_[:128, :, 128:],
                   np.s_[:128, :97, 123:250], np.s_[:])
        self._test_resize(shape, out_shape, indices)


if __name__ == '__main__':
    unittest.main()
