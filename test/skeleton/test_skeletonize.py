import unittest
import numpy as np
import skimage.data

try:
    import skan
except ImportError:
    skan = None


class TestSkeletonize(unittest.TestCase):
    @unittest.skipUnless(skan, "Needs skan")
    def test_skeletonize(self):
        from elf.skeleton import skeletonize
        x = skimage.data.horse()
        x = 1 - x
        x = np.repeat(x[None], 16, axis=0)
        nodes, edges = skeletonize(x)
        # make sure the results are non-trivial
        self.assertGreater(len(nodes), 10)
        self.assertGreater(len(edges), 10)


if __name__ == '__main__':
    unittest.main()
