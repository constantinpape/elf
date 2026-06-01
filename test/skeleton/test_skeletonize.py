import unittest
import numpy as np
import skimage.data

try:
    import skan
except ImportError:
    skan = None


class TestSkeletonize(unittest.TestCase):
    def _get_object(self):
        x = skimage.data.horse()
        x = 1 - x
        x = np.repeat(x[None], 16, axis=0)
        return x

    @unittest.skipUnless(skan, "Needs skan")
    def test_skeletonize(self):
        from elf.skeleton import skeletonize
        x = self._get_object()
        nodes, edges = skeletonize(x)
        # make sure the results are non-trivial
        self.assertGreater(len(nodes), 10)
        self.assertGreater(len(edges), 10)
        # nodes are coordinates (one per dimension), edges are node-id pairs
        self.assertEqual(nodes.shape[1], x.ndim)
        self.assertEqual(edges.shape[1], 2)

    @unittest.skipUnless(skan, "Needs skan")
    def test_skeletonize_resolution(self):
        from elf.skeleton import skeletonize
        x = self._get_object()
        # a scalar and a per-axis resolution should both be accepted
        nodes_scalar, _ = skeletonize(x, resolution=2)
        nodes_tuple, _ = skeletonize(x, resolution=(2, 2, 2))
        self.assertGreater(len(nodes_scalar), 10)
        self.assertTrue(np.array_equal(nodes_scalar, nodes_tuple))

    def test_skeletonize_invalid_method(self):
        from elf.skeleton import skeletonize
        x = np.zeros((8, 8, 8), dtype="uint8")
        with self.assertRaises(ValueError):
            skeletonize(x, method="does-not-exist")


if __name__ == "__main__":
    unittest.main()
