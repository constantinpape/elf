import unittest
import numpy as np
import vigra


class TestParallel(unittest.TestCase):
    def test_unique(self):
        from elf.parallel import unique

        shape = 3 * (128,)
        block_shape = 3 * (32,)
        x = np.random.randint(0, 2 * np.prod(shape), size=shape)

        u1 = unique(x, block_shape=block_shape)
        u2 = np.unique(x)
        self.assertTrue(np.array_equal(u1, u2))

        u1, c1 = unique(x, block_shape=block_shape, return_counts=True)
        u2, c2 = np.unique(x, return_counts=True)
        self.assertTrue(np.array_equal(u1, u2))
        self.assertTrue(np.array_equal(c1, c2))

    def test_unique_with_mask(self):
        from elf.parallel import unique

        shape = 3 * (128,)
        block_shape = 3 * (32,)
        x = np.random.randint(0, 2 * np.prod(shape), size=shape)
        mask = np.random.rand(*shape) > 0.8

        u1 = unique(x, block_shape=block_shape, mask=mask)
        u2 = np.unique(x[mask])
        self.assertTrue(np.array_equal(u1, u2))

        u1, c1 = unique(x, block_shape=block_shape, return_counts=True, mask=mask)
        u2, c2 = np.unique(x[mask], return_counts=True)
        self.assertTrue(np.array_equal(u1, u2))
        self.assertTrue(np.array_equal(c1, c2))

    def test_relabel(self):
        from elf.parallel import relabel_consecutive
        shape = 3 * (128,)
        block_shape = 3 * (32,)
        x = np.random.randint(0, 2 * np.prod(shape), size=shape).astype("uint32")

        xx, max_id, mapping = relabel_consecutive(x, block_shape=block_shape)
        unx = np.unique(xx)

        # make sure that the result is consecutive and that max_id, mapping and
        # result are consistent
        self.assertEqual(max_id, unx.max())
        vals = np.array(list(mapping.values()))
        self.assertTrue(np.array_equal(unx, vals))

        # check against vigra result
        exp = vigra.analysis.relabelConsecutive(x)[0]
        unexp = np.unique(exp)
        self.assertTrue(np.array_equal(unx, unexp))

    # FIXME
    @unittest.expectedFailure
    def test_relabel_with_mask(self):
        from elf.parallel import relabel_consecutive
        shape = 3 * (128,)
        block_shape = 3 * (32,)
        x = np.random.randint(0, 2 * np.prod(shape), size=shape).astype("uint32")
        mask = np.random.rand(*shape) > 0.8

        xx, max_id, mapping = relabel_consecutive(x, block_shape=block_shape, mask=mask)
        unx = np.unique(xx[mask])

        # make sure that the result is consecutive and that max_id, mapping and
        # result are consistent
        self.assertEqual(max_id, unx.max())
        vals = np.array(list(mapping.values()))
        self.assertTrue(np.array_equal(unx, vals))


if __name__ == "__main__":
    unittest.main()
