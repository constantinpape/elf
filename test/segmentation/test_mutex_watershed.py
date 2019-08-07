import unittest
import numpy as np
try:
    import affogato
except ImportError:
    affogato = None


class TestMutexWatershed(unittest.TestCase):

    @unittest.skipUnless(affogato, "Need affogato for mutex watershed functionality")
    def test_mutex_watershed(self):
        from elf.segmentation.mutex_watershed import mutex_watershed
        shape = (10, 256, 256)
        aff_shape = (9,) + shape
        affs = np.random.rand(*aff_shape).astype('float32')

        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-3, 0, 0], [0, -3, 0], [0, 0, -3],
                   [-9, 0, 0], [0, -9, 0], [0, 0, -9]]
        strides = [4, 4, 4]
        seg = mutex_watershed(affs, offsets, strides, True)
        self.assertEqual(seg.shape, shape)
        # make sure the segmentation is not trivial
        self.assertGreater(len(np.unique(seg)), 10)

    @unittest.skipUnless(affogato, "Need affogato for mutex watershed functionality")
    def test_mutex_watershed_with_seeds(self):
        from elf.segmentation.mutex_watershed import mutex_watershed_with_seeds
        shape = (10, 256, 256)
        aff_shape = (9,) + shape
        affs = np.random.rand(*aff_shape).astype('float32')
        # TODO non-trivial seeds
        seeds = np.zeros(shape, dtype='uint64')

        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-3, 0, 0], [0, -3, 0], [0, 0, -3],
                   [-9, 0, 0], [0, -9, 0], [0, 0, -9]]
        strides = [4, 4, 4]
        seg = mutex_watershed_with_seeds(affs, offsets, seeds, strides, True)
        self.assertEqual(seg.shape, shape)
        # make sure the segmentation is not trivial
        self.assertGreater(len(np.unique(seg)), 10)
        # TODO check that seeds were actually taken into account


if __name__ == '__main__':
    unittest.main()
