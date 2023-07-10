import unittest
import numpy as np


def check_scores(scores, func, exp):
    func(scores['precision'], exp)
    func(scores['recall'], exp)
    func(scores['segmentation_accuracy'], exp)
    func(scores['f1'], exp)


class TestMatching(unittest.TestCase):
    def test_matching(self):
        from elf.evaluation import matching
        shape = (256, 256)

        # random data
        x = np.random.randint(0, 10, size=shape)
        y = np.random.randint(0, 10, size=shape)
        scores = matching(x, y)

        check_scores(scores, self.assertGreaterEqual, 0.)
        check_scores(scores, self.assertLess, 1.0)

        # same data
        scores = matching(x, x)
        check_scores(scores, self.assertEqual, 1.)

    def test_matching_threshold(self):
        from elf.evaluation import matching

        scores = matching([0, 0, 1, 1, 1, 2], [0, 0, 1, 1, 2, 2], threshold=0.7)
        check_scores(scores, self.assertEqual, 0.)
        scores = matching([0, 0, 1, 1, 1, 2], [0, 0, 1, 1, 2, 2], threshold=0.5)
        check_scores(scores, self.assertEqual, 1.)

    def test_matching_non_continuous_labels(self):
        from elf.evaluation import matching

        scores = matching([0, 0, 500, 500, 500, 2], [0, 0, 1, 1, 2, 2])
        check_scores(scores, self.assertEqual, 1.)

    def test_matching_ignore_zero_label_default(self):
        from elf.evaluation import matching

        scores = matching([0, 0, 1], [0, 1, 1], threshold=0.9)
        check_scores(scores, self.assertEqual, 0.)
        scores = matching([0, 0, 1], [0, 1, 1], threshold=0.5)
        check_scores(scores, self.assertEqual, 1.)

    def test_matching_example(self):
        from elf.evaluation import matching

        scores = matching([0, 1, 2, 3, 4], [0, 1, 0, 0, 0])
        expected = {'precision': 0.25, 'recall': 1.0, 'segmentation_accuracy': 0.25, 'f1': 0.4}
        self.assertEqual(scores, expected)

    def test_ignore_label_none(self):
        from elf.evaluation import matching

        scores = matching([0, 1], [1, 2], ignore_label=0)
        self.assertEqual(scores['precision'], 1.0)
        self.assertEqual(scores['recall'], 0.5)
        self.assertEqual(scores['segmentation_accuracy'], 0.5)
        self.assertAlmostEqual(scores['f1'], 2.0 / 3.0)

        scores = matching([0, 1], [1, 2], ignore_label=None)
        check_scores(scores, self.assertEqual, 1.)

    def test_mean_segmentation_accuracy(self):
        from elf.evaluation import mean_segmentation_accuracy
        shape = (256, 256)

        # random data
        x = np.random.randint(0, 10, size=shape)
        y = np.random.randint(0, 10, size=shape)
        score = mean_segmentation_accuracy(x, y)
        self.assertGreaterEqual(score, 0.)

        # same data
        score = mean_segmentation_accuracy(x, x)
        self.assertEqual(score, 1.)


if __name__ == '__main__':
    unittest.main()
