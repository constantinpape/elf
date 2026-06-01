import unittest


class TestGridViews(unittest.TestCase):
    def test_get_position(self):
        from elf.visualisation.grid_views import get_position

        grid_shape = (2, 2)
        image_shape = (10, 10)
        spacing = 4
        # The first image sits at the origin.
        self.assertEqual(get_position(grid_shape, image_shape, 0, spacing), (0, 0))
        # Subsequent images are offset by image size + spacing.
        self.assertEqual(get_position(grid_shape, image_shape, 1, spacing), (0, 14))
        self.assertEqual(get_position(grid_shape, image_shape, 2, spacing), (14, 0))
        self.assertEqual(get_position(grid_shape, image_shape, 3, spacing), (14, 14))

    def test_get_position_rgb_shape(self):
        from elf.visualisation.grid_views import get_position

        # For rgb the channel axis is stripped before calling get_position; verify the
        # spatial dimensions (last two) drive the layout.
        grid_shape = (1, 2)
        image_shape = (8, 8)  # spatial shape only
        self.assertEqual(get_position(grid_shape, image_shape, 1, 2), (0, 10))

    def test_resolve_is_rgb(self):
        from elf.visualisation.grid_views import _resolve_is_rgb

        # Plain boolean is applied to every source.
        self.assertTrue(_resolve_is_rgb(True, "anything"))
        self.assertFalse(_resolve_is_rgb(False, "anything"))

        # Dict resolves per source name; missing entries default to False.
        is_rgb = {"raw": True, "mask": False}
        self.assertTrue(_resolve_is_rgb(is_rgb, "raw"))
        self.assertFalse(_resolve_is_rgb(is_rgb, "mask"))
        self.assertFalse(_resolve_is_rgb(is_rgb, "unknown"))


if __name__ == "__main__":
    unittest.main()
