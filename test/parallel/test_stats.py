import unittest
import numpy as np


# TODO tests with mask and roi
class TestStats(unittest.TestCase):
    def _test_stat(self, stat_impl, np_stats):
        shape = 3 * (64,)
        block_shape = 3 * (16,)
        x = np.random.rand(*shape)

        res = stat_impl(x, block_shape=block_shape)
        if isinstance(np_stats, list):
            exp = [stat(x) for stat in np_stats]
            self.assertEqual(len(res), len(exp))
            for re, ex in zip(res, exp):
                self.assertAlmostEqual(re, ex)
        else:
            exp = np_stats(x)
            self.assertAlmostEqual(res, exp)

    def test_mean(self):
        from elf.parallel import mean
        self._test_stat(mean, np.mean)

    def test_std(self):
        from elf.parallel import std
        self._test_stat(std, np.std)

    def test_min(self):
        from elf.parallel import min
        self._test_stat(min, np.min)

    def test_max(self):
        from elf.parallel import max
        self._test_stat(max, np.max)

    def test_mean_and_std(self):
        from elf.parallel import mean_and_std
        self._test_stat(mean_and_std, [np.mean, np.std])

    def test_min_and_max(self):
        from elf.parallel import min_and_max
        self._test_stat(min_and_max, [np.min, np.max])


if __name__ == "__main__":
    unittest.main()
