import unittest
import numpy as np


class TestDice(unittest.TestCase):

    def _get_seg(self, shape, with_mask, return_mask=False):
        seg = np.random.randint(1, 200, size=shape)
        if with_mask:
            mask = np.random.rand(*shape) > 0.5
            seg[mask] = 0
        if return_mask:
            return seg, mask
        else:
            return seg

    def test_dice_score(self):
        from elf.evaluation import dice_score

        shape = (64, 64)

        # test with identical data
        seg = self._get_seg(shape, True)
        gt = seg.copy()
        score = dice_score(seg, gt)
        self.assertAlmostEqual(score, 1.)

        # test with maximal dissimilar data
        seg, mask = self._get_seg(shape, True, True)
        gt = np.random.randint(1, 100, size=shape)
        gt[~mask] = 0
        score = dice_score(seg, gt)
        self.assertAlmostEqual(score, 0.)

        # test with random data
        seg = self._get_seg(shape, True)
        gt = self._get_seg(shape, True)
        score = dice_score(seg, gt)
        self.assertGreater(score, 0.)
        self.assertLess(score, 1.)

    def _test_symmetric_best_dice_score(self, impl):
        from elf.evaluation import symmetric_best_dice_score

        shape = (64, 64)

        # test with identical data
        seg = self._get_seg(shape, False)
        gt = seg.copy()
        score = symmetric_best_dice_score(seg, gt, impl=impl)
        self.assertAlmostEqual(score, 1.)

        # test with random data
        seg = self._get_seg(shape, True)
        gt = self._get_seg(shape, True)
        score = symmetric_best_dice_score(seg, gt, impl=impl)
        self.assertGreater(score, 0.)
        self.assertLess(score, 1.)

    def test_symmetric_best_dice_score_numpy(self):
        self._test_symmetric_best_dice_score('numpy')

    def test_symmetric_best_dice_score_nifty(self):
        self._test_symmetric_best_dice_score('nifty')

    def test_compare_implementations(self):
        from elf.evaluation import symmetric_best_dice_score

        N = 3
        shape = (64, 64)
        for _ in range(N):
            seg = self._get_seg(shape, True)
            gt = self._get_seg(shape, True)
            score1 = symmetric_best_dice_score(seg, gt, impl='numpy')
            score2 = symmetric_best_dice_score(seg, gt, impl='numpy')
            self.assertAlmostEqual(score1, score2)


if __name__ == '__main__':
    unittest.main()
