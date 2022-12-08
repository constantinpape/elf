import unittest
import numpy as np


class TestOperations(unittest.TestCase):
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

    def _test_op_broadcast(self, op, op_exp):
        shapex = 3 * (64,)
        shapey = (1, 64, 64)
        block_shape = 3 * (16,)
        x = np.random.rand(*shapex)
        y = np.random.rand(*shapey)

        exp = op_exp(x, y)
        op(x, y, block_shape=block_shape)
        self.assertTrue(np.allclose(exp, x))

    def _test_op_masked(self, op, op_exp):
        shape = 3 * (64,)
        block_shape = 3 * (16,)
        x = np.random.rand(*shape)
        y = np.random.rand()
        mask = np.random.rand(*shape) > .5

        exp = x.copy()
        exp[mask] = op_exp(x[mask], y)
        op(x, y, block_shape=block_shape, mask=mask)
        self.assertTrue(np.allclose(exp, x))

    def _test_op_roi(self, op, op_exp):
        shape = 3 * (64,)
        block_shape = 3 * (16,)

        rois = [np.s_[:], np.s_[:32, 32:], np.s_[:47, :], np.s_[:, 13:], np.s_[2:31, 5:59]]
        for roi in rois:
            x = np.random.rand(*shape)
            y = np.random.rand()

            res = x.copy()
            op(res, y, block_shape=block_shape, roi=roi)
            exp = op_exp(x[roi], y)
            self.assertTrue(np.allclose(exp, res[roi]))

            not_roi = np.ones(res.shape, dtype='bool')
            not_roi[roi] = False
            self.assertTrue(np.allclose(res[not_roi], x[not_roi]))

    def _test_op(self, op1, op2):
        self._test_op_array(op1, op2, True)
        self._test_op_array(op1, op2, False)
        self._test_op_scalar(op1, op2, True)
        self._test_op_scalar(op1, op2, False)
        self._test_op_broadcast(op1, op2)
        self._test_op_masked(op1, op2)
        self._test_op_roi(op1, op2)

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

    def test_minimum(self):
        from elf.parallel import minimum
        self._test_op(minimum, np.minimum)

    def test_maximum(self):
        from elf.parallel import maximum
        self._test_op(maximum, np.maximum)


if __name__ == '__main__':
    unittest.main()
