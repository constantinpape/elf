import unittest
import numpy as np


class TestBaseWrapper(unittest.TestCase):
    shape = (32, 256, 256)

    def test_with_channels(self):
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


if __name__ == '__main__':
    unittest.main()
