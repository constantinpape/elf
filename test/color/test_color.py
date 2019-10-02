import unittest
import numpy as np
from tqdm import tqdm
try:
    import glasbey as gb_impl
except ImportError:
    gb_impl = None


class TestColor(unittest.TestCase):

    @unittest.skipUnless(gb_impl, "Need glasbey module")
    def test_color(self):
        from elf.color import glasbey
        id_list = [5, 12, 15, 25, 50]
        for nids in tqdm(id_list):
            color_table = glasbey(nids)
            self.assertEqual(color_table.dtype, np.dtype('uint8'))
            exp_shape = (nids, 3)
            self.assertEqual(exp_shape, color_table.shape)
            self.assertGreaterEqual(color_table.min(), 0)
            self.assertLessEqual(color_table.max(), 255)


if __name__ == '__main__':
    unittest.main()
