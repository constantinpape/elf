import unittest
import numpy as np


class TestMatching(unittest.TestCase):
    def test_matching(self):
        from elf.evaluation import matching
        shape = (256, 256)

        def _check_scores(scores, func, exp):
            func(scores['precision'], exp)
            func(scores['recall'], exp)
            func(scores['accuracy'], exp)
            func(scores['f1'], exp)

        # random data
        x = np.random.randint(0, 10, size=shape)
        y = np.random.randint(0, 10, size=shape)
        scores = matching(x, y)
        _check_scores(scores, self.assertGreaterEqual, 0.)

        # same data
        scores = matching(x, x)
        _check_scores(scores, self.assertEqual, 1.)

    def test_map(self):
        from elf.evaluation import mean_average_precision
        shape = (256, 256)

        def _check_scores(scores, func, exp):
            func(scores['precision'], exp)
            func(scores['recall'], exp)
            func(scores['accuracy'], exp)
            func(scores['f1'], exp)

        # random data
        x = np.random.randint(0, 10, size=shape)
        y = np.random.randint(0, 10, size=shape)
        score = mean_average_precision(x, y)
        self.assertGreaterEqual(score, 0.)

        # same data
        score = mean_average_precision(x, x)
        self.assertEqual(score, 1.)


if __name__ == '__main__':
    unittest.main()
