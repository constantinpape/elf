import unittest
import numpy as np


class TestParallel(unittest.TestCase):
    def test_mean(self):
        from elf.parallel import mean
        shape = 3 * (64,)
        block_shape = 3 * (16,)
        x = np.random.rand(*shape)

        m1 = mean(x, block_shape=block_shape)
        m2 = x.mean()
        self.assertAlmostEqual(m1, m2)

    def test_mean_and_std(self):
        from elf.parallel import mean_and_std
        shape = 3 * (64,)
        block_shape = 3 * (16,)
        x = np.random.rand(*shape)

        m1, s1 = mean_and_std(x, block_shape=block_shape)
        m2, s2 = x.mean(), x.std()
        self.assertAlmostEqual(m1, m2)
        self.assertAlmostEqual(s1, s2)

    def test_std(self):
        from elf.parallel import std
        shape = 3 * (64,)
        block_shape = 3 * (16,)
        x = np.random.rand(*shape)

        s1 = std(x, block_shape=block_shape)
        s2 = x.std()
        self.assertAlmostEqual(s1, s2)

    def test_unique(self):
        from elf.parallel import unique

        shape = 3 * (32,)
        block_shape = 3 * (16,)
        x = np.random.randint(0, 2 * np.prod(shape), size=shape)

        u1 = unique(x, block_shape=block_shape)
        u2 = np.unique(x)
        self.assertTrue(np.array_equal(u1, u2))

        u1, c1 = unique(x, block_shape=block_shape, return_counts=True)
        u2, c2 = np.unique(x, return_counts=True)
        self.assertTrue(np.array_equal(u1, u2))
        self.assertTrue(np.array_equal(c1, c2))


if __name__ == '__main__':
    unittest.main()
