import unittest


class TestParseWells(unittest.TestCase):

    def test_zero_based(self):
        from elf.htm.visualization import parse_wells
        positions, start, stop = parse_wells(["A0", "B0", "A1"], zero_based=True)
        self.assertEqual(positions, {"A0": (0, 0), "B0": (1, 0), "A1": (0, 1)})
        self.assertEqual(start, (0, 0))
        self.assertEqual(stop, (2, 2))

    def test_one_based(self):
        from elf.htm.visualization import parse_wells
        positions, start, stop = parse_wells(["A1", "B1", "A2"], zero_based=False)
        self.assertEqual(positions, {"A1": (0, 0), "B1": (1, 0), "A2": (0, 1)})
        self.assertEqual(start, (0, 0))
        self.assertEqual(stop, (2, 2))

    def test_one_based_rejects_zero_y(self):
        from elf.htm.visualization import parse_wells
        with self.assertRaises(AssertionError):
            parse_wells(["A0"], zero_based=False)


class TestGetWorldPosition(unittest.TestCase):

    def test_2d_origin(self):
        from elf.htm.visualization import get_world_position
        wp = get_world_position(
            well_x=0, well_y=0, pos=0,
            well_shape=(2, 2), well_spacing=16, site_spacing=4,
            shape=(64, 64),
        )
        self.assertEqual(wp, [0, 0])

    def test_2d_offset(self):
        # Well (1, 1), bottom-right site (pos=3 in a 2x2 well).
        # i = 1*2 + 1 = 3, j = 1*2 + 1 = 3
        # x = 64*3 + 3*4 + 1*16 = 220
        from elf.htm.visualization import get_world_position
        wp = get_world_position(
            well_x=1, well_y=1, pos=3,
            well_shape=(2, 2), well_spacing=16, site_spacing=4,
            shape=(64, 64),
        )
        self.assertEqual(wp, [220, 220])

    def test_leading_non_spatial_axes(self):
        from elf.htm.visualization import get_world_position
        wp = get_world_position(
            well_x=1, well_y=1, pos=3,
            well_shape=(2, 2), well_spacing=16, site_spacing=4,
            shape=(2, 64, 64),
        )
        # One leading non-spatial axis -> one prepended zero, spatial coords unchanged.
        self.assertEqual(wp, [0, 220, 220])


if __name__ == "__main__":
    unittest.main()
