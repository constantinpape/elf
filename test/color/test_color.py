import os
import glob
import unittest
from unittest import mock

import numpy as np
from tqdm import tqdm
try:
    import glasbey as gb_impl
except ImportError:
    gb_impl = None


class TestColor(unittest.TestCase):

    def check_lut(self, color_table, nids):
        self.assertEqual(color_table.dtype, np.dtype('uint8'))
        exp_shape = (nids, 3)
        self.assertEqual(exp_shape, color_table.shape)
        if nids > 0:
            self.assertGreaterEqual(color_table.min(), 0)
            self.assertLessEqual(color_table.max(), 255)

    @unittest.skipUnless(gb_impl is not None and hasattr(gb_impl, "Glasbey"), "Need taketwo/glasbey module")
    def test_glasbey(self):
        from elf.color import glasbey
        id_list = [5, 12, 15, 25, 50]
        for nids in tqdm(id_list):
            color_table = glasbey(nids, "dark2")
            self.check_lut(color_table, nids)

    @unittest.skipUnless(gb_impl is not None and hasattr(gb_impl, "Glasbey"), "Need taketwo/glasbey module")
    def test_glasbey_invalid_palette(self):
        from elf.color import glasbey
        with self.assertRaises(ValueError):
            glasbey(5, "not_a_palette")

    def test_glasbey_missing_module(self):
        from elf.color import glasbey
        with mock.patch("elf.color.palette.glasey_impl", None):
            with self.assertRaises(ImportError):
                glasbey(5, "dark2")

    @unittest.skipUnless(gb_impl is not None and hasattr(gb_impl, "Glasbey"), "Need taketwo/glasbey module")
    def test_glasbey_palettes(self):
        from elf.color import glasbey
        palette_folder = os.path.join(os.path.split(gb_impl.__file__)[0], "palettes")
        names = sorted(
            os.path.splitext(os.path.split(p)[1])[0]
            for p in glob.glob(os.path.join(palette_folder, "*.txt"))
        )
        self.assertGreater(len(names), 0)
        for name in names[:3]:
            color_table = glasbey(10, name)
            self.check_lut(color_table, 10)

    @unittest.skipUnless(gb_impl is not None and hasattr(gb_impl, "Glasbey"), "Need taketwo/glasbey module")
    def test_glasbey_optional_args(self):
        from elf.color import glasbey
        color_table = glasbey(10, "dark2", no_black=False, lightness_range=(20, 80))
        self.check_lut(color_table, 10)

    def test_random_colors(self):
        from elf.color import random_colors
        id_list = [100, 255, 1000]
        for nids in id_list:
            color_table = random_colors(nids)
            self.check_lut(color_table, nids)

    def test_random_colors_edge_cases(self):
        from elf.color import random_colors
        for nids in (0, 1):
            color_table = random_colors(nids)
            self.check_lut(color_table, nids)


if __name__ == '__main__':
    unittest.main()
