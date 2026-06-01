import unittest
import numpy as np


class TestStats(unittest.TestCase):
    def _test_stat(self, stat_impl, np_stats, mask=None, roi=None, data=None):
        shape = 3 * (64,)
        block_shape = 3 * (16,)
        x = np.random.rand(*shape) if data is None else data

        kwargs = {"block_shape": block_shape}
        if mask is not None:
            kwargs["mask"] = mask
        if roi is not None:
            kwargs["roi"] = roi

        res = stat_impl(x, **kwargs)

        if mask is not None:
            reference = x[mask.astype("bool")]
        elif roi is not None:
            reference = x[roi]
        else:
            reference = x

        if isinstance(np_stats, list):
            exp = [stat(reference) for stat in np_stats]
            self.assertEqual(len(res), len(exp))
            for re, ex in zip(res, exp):
                self.assertAlmostEqual(re, ex)
        else:
            exp = np_stats(reference)
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

    def _make_mask(self, shape):
        rng = np.random.default_rng(seed=42)
        mask = rng.random(shape) > 0.5
        return mask

    def test_mean_with_mask(self):
        from elf.parallel import mean
        shape = 3 * (64,)
        mask = self._make_mask(shape)
        self._test_stat(mean, np.mean, mask=mask)

    def test_std_with_mask(self):
        from elf.parallel import std
        shape = 3 * (64,)
        mask = self._make_mask(shape)
        self._test_stat(std, np.std, mask=mask)

    def test_min_with_mask(self):
        from elf.parallel import min
        shape = 3 * (64,)
        mask = self._make_mask(shape)
        self._test_stat(min, np.min, mask=mask)

    def test_max_with_mask(self):
        from elf.parallel import max
        shape = 3 * (64,)
        mask = self._make_mask(shape)
        self._test_stat(max, np.max, mask=mask)

    def test_mean_and_std_with_mask(self):
        from elf.parallel import mean_and_std
        shape = 3 * (64,)
        mask = self._make_mask(shape)
        self._test_stat(mean_and_std, [np.mean, np.std], mask=mask)

    def test_min_and_max_with_mask(self):
        from elf.parallel import min_and_max
        shape = 3 * (64,)
        mask = self._make_mask(shape)
        self._test_stat(min_and_max, [np.min, np.max], mask=mask)

    def test_mean_with_roi(self):
        from elf.parallel import mean
        roi = np.s_[16:48, 16:48, 16:48]
        self._test_stat(mean, np.mean, roi=roi)

    def test_std_with_roi(self):
        from elf.parallel import std
        roi = np.s_[16:48, 16:48, 16:48]
        self._test_stat(std, np.std, roi=roi)

    def test_min_with_roi(self):
        from elf.parallel import min
        roi = np.s_[16:48, 16:48, 16:48]
        self._test_stat(min, np.min, roi=roi)

    def test_max_with_roi(self):
        from elf.parallel import max
        roi = np.s_[16:48, 16:48, 16:48]
        self._test_stat(max, np.max, roi=roi)

    def test_mean_and_std_with_roi(self):
        from elf.parallel import mean_and_std
        roi = np.s_[16:48, 16:48, 16:48]
        self._test_stat(mean_and_std, [np.mean, np.std], roi=roi)

    def test_min_and_max_with_roi(self):
        from elf.parallel import min_and_max
        roi = np.s_[16:48, 16:48, 16:48]
        self._test_stat(min_and_max, [np.min, np.max], roi=roi)


if __name__ == "__main__":
    unittest.main()
