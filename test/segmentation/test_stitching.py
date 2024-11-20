import unittest

import numpy as np
from skimage.data import binary_blobs
from skimage.measure import label

from elf.evaluation import rand_index


class TestStitching(unittest.TestCase):
    def get_data(self, size=1024, ndim=2):
        data = binary_blobs(size, blob_size_fraction=0.1, volume_fraction=0.2, n_dim=ndim)
        return data

    def get_tiled_data(self, size=1024, ndim=2, tile_shape=(512, 512)):
        data = self.get_data(size=size, ndim=ndim)
        data = label(data)  # Ensure all inputs are instances (the blobs are semantic labels)

        # Create tiles out of the data for testing label stitching.
        # Ensure offset for objects per tile to get individual ids per object per tile.
        # And finally stitch back the tiles.
        import nifty.tools as nt
        blocking = nt.blocking([0] * ndim, data.shape, tile_shape)
        n_blocks = blocking.numberOfBlocks

        labels = np.zeros(data.shape)
        offset = 0
        for tile_id in range(n_blocks):
            block = blocking.getBlock(tile_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

            tile = data[bb]
            tile = label(tile)
            tile[tile != 0] += offset
            offset = tile.max()

            labels[bb] = tile

        return labels, data  # returns the stitched labels and original labels

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

    def test_stitch_tiled_segmentation(self):
        from elf.segmentation.stitching import stitch_tiled_segmentation

        tile_shapes = [(224, 224), (256, 256), (512, 512)]
        for tile_shape in tile_shapes:
            # Get the tiled segmentation with unmerged instances at tile interfaces.
            labels, original_labels = self.get_tiled_data()
            stitched_labels = stitch_tiled_segmentation(segmentation=labels, tile_shape=tile_shape)
            self.assertEqual(labels.shape, stitched_labels.shape)
            # self.assertEqual(len(np.unique(original_labels)), len(np.unique(stitched_labels)))
            print(len(np.unique(original_labels)), len(np.unique(stitched_labels)))


if __name__ == "__main__":
    unittest.main()
