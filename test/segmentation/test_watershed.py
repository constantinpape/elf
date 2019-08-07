import unittest
import numpy as np
try:
    import vigra
except ImportError:
    vigra = None


class TestWatershed(unittest.TestCase):

    @unittest.skipUnless(vigra, "Need vigra for watershed functionality")
    def test_size_filter(self):
        from elf.segmentation.watershed import apply_size_filter
        shape = (10, 256, 256)
        seg = np.random.randint(0, 1000, size=shape, dtype='uint32')
        inp = np.random.rand(*shape).astype('float32')
        size_filter = 5000
        seg_filtered, max_id = apply_size_filter(seg, inp, size_filter)
        self.assertEqual(max_id, seg_filtered.max())
        _, counts = np.unique(seg_filtered, return_counts=True)
        self.assertGreaterEqual(size_filter, counts.min())

    @unittest.skipUnless(vigra, "Need vigra for watershed functionality")
    def test_watershed(self):
        from elf.segmentation.watershed import watershed

        shape = (10, 256, 256)
        inp = np.random.rand(*shape).astype('float32')
        seeds = np.zeros(shape, dtype='uint32')

        seed_ids = list(range(1, 100))
        seed_points = np.random.choice(seeds.size, size=(len(seed_ids),), replace=False)
        coords = np.unravel_index(seed_points, shape)
        seeds[coords] = seed_ids

        seg, max_id = watershed(inp, seeds)
        self.assertEqual(max_id, max(seed_ids))
        self.assertEqual(max_id, seg.max())

        seg_ids = set(np.unique(seg))
        self.assertEqual(seg_ids, set(seed_ids))


if __name__ == '__main__':
    unittest.main()
