import unittest
from functools import partial

import numpy as np


class TestBaseWrapper(unittest.TestCase):
    shape = (32, 256, 256)

    def test_simple_transform_wrapper_with_halo(self):
        from elf.wrapper.base import SimpleTransformationWrapperWithHalo
        from skimage.filters import gaussian

        data = np.random.rand(*self.shape)
        transformed = gaussian(data, sigma=1)
        wrapper = SimpleTransformationWrapperWithHalo(data, transformation=partial(gaussian, sigma=1), halo=(8, 8, 8))

        rois = (np.s_[5:24, 3:100, 4:123], np.s_[8:31, 2:200, :], np.s_[:], np.s_[4, 5, 6], np.s_[:, 0, 200])
        for roi in rois:
            x1, x2 = transformed[roi], wrapper[roi]
            self.assertEqual(x1.shape, x2.shape)
            self.assertTrue(np.allclose(x1, x2))

    def test_simple_transform_wrapper_with_channels(self):
        from elf.wrapper.base import SimpleTransformationWrapper

        shape_with_channels = (3,) + self.shape
        vol = np.random.rand(*shape_with_channels)
        wrapper = SimpleTransformationWrapper(
            vol, lambda x: 2 * x, with_channels=True, shape=self.shape
        )

        bb = np.s_[12:31, 10:37, 59:257]
        x = wrapper[bb]
        self.assertTrue(np.allclose(x, 2 * vol[(slice(None),) + bb]))

        bb = np.s_[4, 10:37, 59:257]
        x = wrapper[bb]
        self.assertTrue(np.allclose(x, 2 * vol[(slice(None),) + bb]))

        bb = np.s_[4, 10:37, 9]
        x = wrapper[bb]
        self.assertTrue(np.allclose(x, 2 * vol[(slice(None),) + bb]))

    def test_multi_transform_wrapper(self):
        from elf.wrapper.base import MultiTransformationWrapper

        # Test regions.
        rois = (np.s_[5:24, 3:100, 4:123], np.s_[8:31, 2:200, :], np.s_[:], np.s_[4, 5, 6], np.s_[:, 0, 200])

        # Test the use-case for 2 inputs.
        vol1 = np.random.rand(*self.shape) > 0.5
        vol2 = np.random.rand(*self.shape) > 0.5
        wrapped = MultiTransformationWrapper(np.logical_and, vol1, vol2)
        expected = np.logical_and(vol1, vol2)
        for bb in rois:
            self.assertTrue((wrapped[bb] == expected[bb]).all())

        # Test the use-case for 3 inputs.
        vol3 = np.random.rand(*self.shape) > 0.5
        wrapped = MultiTransformationWrapper(np.logical_and.reduce, vol1, vol2, vol3, apply_to_list=True)
        expected = np.logical_and.reduce([vol1, vol2, vol3])
        for bb in rois:
            self.assertTrue((wrapped[bb] == expected[bb]).all())


if __name__ == "__main__":
    unittest.main()
