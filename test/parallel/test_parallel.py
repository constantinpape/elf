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
            self.assertTrue(np.allclose(exp, x))
        else:
            x_cpy = x.copy()
            res = np.zeros_like(x)
            res = op(x, y, out=res, block_shape=block_shape)
            self.assertTrue(np.allclose(exp, res))
            # make sure x is unchaged
            self.assertTrue(np.allclose(x, x_cpy))

    def _test_op_scalar(self, op, op_exp, inplace):
        shape = 3 * (64,)
        block_shape = 3 * (16,)
        x = np.random.rand(*shape)
        y = np.random.rand()

        exp = op_exp(x, y)
        if inplace:
            op(x, y, block_shape=block_shape)
            self.assertTrue(np.allclose(exp, x))
        else:
            x_cpy = x.copy()
            res = np.zeros_like(exp)
            res = op(x, y, out=res, block_shape=block_shape)
            self.assertTrue(np.allclose(exp, res))
            # make sure x is unchaged
            self.assertTrue(np.allclose(x, x_cpy))

    def _test_op(self, op1, op2):
        self._test_op_array(op1, op2, True)
        self._test_op_array(op1, op2, False)
        self._test_op_scalar(op1, op2, True)
        self._test_op_scalar(op1, op2, False)

    def test_add(self):
        from elf.parallel import add
        self._test_op(add, np.add)

    def test_subtract(self):
        from elf.parallel import subtract
        self._test_op(subtract, np.subtract)

    def test_multiply(self):
        from elf.parallel import multiply
        self._test_op(multiply, np.multiply)

    def test_divide(self):
        from elf.parallel import divide
        self._test_op(divide, np.divide)

    def test_greater(self):
        from elf.parallel import greater
        self._test_op(greater, np.greater)

    def test_greater_equal(self):
        from elf.parallel import greater_equal
        self._test_op(greater_equal, np.greater_equal)

    def test_less(self):
        from elf.parallel import less
        self._test_op(less, np.less)

    def test_less_equal(self):
        from elf.parallel import less_equal
        self._test_op(less_equal, np.less_equal)


if __name__ == '__main__':
    unittest.main()
