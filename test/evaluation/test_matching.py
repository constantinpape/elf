import unittest
import numpy as np

try:
    import nifty
except ImportError:
    nifty = None

# TODO try importing from dsb
# try:
#     from cremi_tools.metrics import adapted_rand, voi
# except ImportError:
#     adapted_rand, voi = None, None


class TestMatching(unittest.TestCase):
    # TODO implement download of example data in setUpClass
    url = ''

    # need to implement download first
    # @unittest.skipUnless(nifty and stardist, "Need nifty and stardist")
    @unittest.skip
    def test_matching(self):
        pass

    @unittest.skipUnless(nifty, "Need nifty")
    def test_matching_random_data(self):
        from elf.evaluation import matching
        shape = (256, 256)
        x = np.random.randint(0, 100, size=shape)
        y = np.random.randint(0, 100, size=shape)
        matching_scores = matching(x, y)
        self.assertGreaterEqual(matching_scores['precision'], 0.)
        self.assertGreaterEqual(matching_scores['recall'], 0.)
        self.assertGreaterEqual(matching_scores['accuracy'], 0.)
        self.assertGreaterEqual(matching_scores['f1'], 0.)


if __name__ == '__main__':
    unittest.main()
