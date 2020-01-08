import unittest
import numpy as np

try:
    import nifty
except ImportError:
    nifty = None

try:
    import vigra
except ImportError:
    vigra = None


# TODO tests with mask
@unittest.skipUnless(nifty, "Need nifty")
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

    def test_relabel(self):
        from elf.parallel import relabel_consecutive
        shape = 3 * (32,)
        block_shape = 3 * (16,)
        x = np.random.randint(0, 2 * np.prod(shape), size=shape).astype('uint32')

        xx, max_id, mapping = relabel_consecutive(x, block_shape=block_shape)
        unx = np.unique(xx)

        # make sure that the result is consecutive and that max_id, mapping and
        # result are consistent
        self.assertEqual(max_id, unx.max())
        vals = np.array(list(mapping.values()))
        self.assertTrue(np.array_equal(unx, vals))

        # check against vigra result
        if vigra is not None:
            exp = vigra.analysis.relabelConsecutive(x)[0]
            unexp = np.unique(exp)
            self.assertTrue(np.array_equal(unx, unexp))

    def _test_op_array(self, op, op_exp, inplace):
        shape = 3 * (64,)
        block_shape = 3 * (16,)
        x = np.random.rand(*shape)
        y = np.random.rand(*shape)

        exp = op_exp(x, y)
        if inplace:
            op(x, y, block_shape=block_shape)
            self.assertTrue(np.array_equal(exp, x))
        else:
            res = np.zeros_like(x)
            res = op(x, y, out=res, block_shape=block_shape)
            self.assertTrue(np.array_equal(exp, res))

    def _test_op_scalar(self, op, op_exp, inplace):
        shape = 3 * (64,)
        block_shape = 3 * (16,)
        x = np.random.rand(*shape)
        y = np.random.rand()

        exp = op_exp(x, y)
        if inplace:
            op(x, y, block_shape=block_shape)
            self.assertTrue(np.array_equal(exp, x))
        else:
            res = np.zeros_like(x)
            res = op(x, y, out=res, block_shape=block_shape)
            self.assertTrue(np.array_equal(exp, res))

    def test_add(self):
        from elf.parallel import add
        self._test_op_array(add, np.add, True)
        self._test_op_array(add, np.add, False)
        self._test_op_scalar(add, np.add, True)
        self._test_op_scalar(add, np.add, False)

    def test_subtract(self):
        from elf.parallel import subtract
        self._test_op_array(subtract, np.subtract, True)
        self._test_op_array(subtract, np.subtract, False)
        self._test_op_scalar(subtract, np.subtract, True)
        self._test_op_scalar(subtract, np.subtract, False)

    def test_multiply(self):
        from elf.parallel import multiply
        self._test_op_array(multiply, np.multiply, True)
        self._test_op_array(multiply, np.multiply, False)
        self._test_op_scalar(multiply, np.multiply, True)
        self._test_op_scalar(multiply, np.multiply, False)

    def test_divide(self):
        from elf.parallel import divide
        self._test_op_array(divide, np.divide, True)
        self._test_op_array(divide, np.divide, False)
        self._test_op_scalar(divide, np.divide, True)
        self._test_op_scalar(divide, np.divide, False)


if __name__ == '__main__':
    unittest.main()
