import unittest

from elf.evaluation import rand_index
from skimage.data import binary_blobs
from skimage.measure import label


class TestStitching(unittest.TestCase):
    def get_data(self, size=1024, ndim=2):
        data = binary_blobs(size, blob_size_fraction=0.1, volume_fraction=0.2, n_dim=ndim)
        return data

    def test_stitch_segmentation(self):
        from elf.segmentation.stitching import stitch_segmentation

        def _segment(input_, block_id=None):
            segmentation = label(input_)
            return segmentation.astype("uint32")

        tile_overlap = (32, 32)
        tile_shapes = [(128, 128), (256, 256), (128, 256)]
        for tile_shape in tile_shapes:
            for _ in range(3):  # test 3 times with different data
                data = self.get_data()
                expected_segmentation = _segment(data)
                segmentation = stitch_segmentation(data, _segment, tile_shape, tile_overlap, verbose=False)
                are, _ = rand_index(segmentation, expected_segmentation)
                self.assertTrue(are < 0.05)

    def test_stitch_segmentation_3d(self):
        from elf.segmentation.stitching import stitch_segmentation

        def _segment(input_, block_id=None):
            segmentation = label(input_)
            return segmentation.astype("uint32")

        tile_overlap = (16, 16, 16)
        tile_shapes = [(32, 32, 32), (64, 64, 64), (32, 64, 24)]
        for tile_shape in tile_shapes:
            data = self.get_data(256, ndim=3)
            expected_segmentation = _segment(data)
            segmentation = stitch_segmentation(data, _segment, tile_shape, tile_overlap, verbose=False)
            are, _ = rand_index(segmentation, expected_segmentation)
            self.assertTrue(are < 0.05)


if __name__ == "__main__":
    unittest.main()
