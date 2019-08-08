import unittest
import numpy as np


class TestUtil(unittest.TestCase):
    def test_normalize_index(self):
        from elf.util import normalize_index, squeeze_singletons
        shape = (128,) * 3
        x = np.random.rand(*shape)

        # is something important missing?
        indices = (np.s_[10:25, 30:60, 100:103],  # full index
                   np.s_[:],  # everything
                   np.s_[..., 10:25],  # ellipsis
                   np.s_[0, :, 10])  # singletons
        for index in indices:
            out1 = x[index]
            index, to_squeeze = normalize_index(index, shape)
            out2 = squeeze_singletons(x[index], to_squeeze)
            self.assertEqual(out1.shape, out2.shape)
            self.assertTrue(np.allclose(out1, out2))


if __name__ == '__main__':
    unittest.main()
