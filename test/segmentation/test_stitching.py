import unittest

import numpy as np
import nifty.tools as nt
from skimage.data import binary_blobs
from skimage.measure import label

from elf.evaluation import rand_index


class TestStitching(unittest.TestCase):
    def get_data(self, size=1024, ndim=2):
        data = binary_blobs(size, blob_size_fraction=0.1, volume_fraction=0.25, n_dim=ndim)
        return data

    def get_tiled_data(self, tile_shape, size=1024, ndim=2):
        data = self.get_data(size=size, ndim=ndim)
        original_data = label(data)  # Ensure all inputs are instances (the blobs are semantic labels)

        # Create tiles out of the data for testing label stitching.
        # Ensure offset for objects per tile to get individual ids per object per tile.
        # And finally stitch back the tiles.
        blocking = nt.blocking([0] * ndim, data.shape, tile_shape)
        n_blocks = blocking.numberOfBlocks

        labels = np.zeros(data.shape)
        offset = 0
        for tile_id in range(n_blocks):
            block = blocking.getBlock(tile_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

            tile = data[bb]
            tile = label(tile)
            tile_mask = tile != 0
            if tile_mask.sum() > 0:
                tile[tile_mask] += offset
                offset = tile.max()

            labels[bb] = tile

        return labels, original_data  # returns the stitched labels and original labels

    def _check_result(self, segmentation, expected_segmentation, rtol=1e-2, atol=1e-2):
        self.assertEqual(segmentation.shape, expected_segmentation.shape)

        # We remove small segments before evaluation, since these may get stitched wrongly.
        ids, sizes = np.unique(segmentation, return_counts=True)
        filter_ids = ids[sizes < 250]
        mask = np.isin(segmentation, filter_ids)
        segmentation[mask] = 0
        expected_segmentation[mask] = 0

        # We allow for some tolerance, because small objects might get stitched incorrectly.
        are, _ = rand_index(segmentation, expected_segmentation)
        self.assertTrue(np.isclose(are, 0, rtol=rtol, atol=atol))

    def test_stitch_segmentation(self):
        from elf.segmentation.stitching import stitch_segmentation

        def _segment(input_, block_id=None):
            segmentation = label(input_)
            return segmentation.astype("uint32")

        tile_overlap = (32, 32)
        tile_shapes = [(128, 128), (256, 256), (128, 256), (224, 224)]
        for tile_shape in tile_shapes:
            for _ in range(3):  # test 3 times with different data
                data = self.get_data()
                expected_segmentation = _segment(data)
                segmentation = stitch_segmentation(data, _segment, tile_shape, tile_overlap, verbose=False)
                self._check_result(segmentation, expected_segmentation)

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
            self._check_result(segmentation, expected_segmentation, rtol=0.1, atol=0.1)

    def test_stitch_tiled_segmentation(self):
        from elf.segmentation.stitching import stitch_tiled_segmentation

        tile_shapes = [(224, 224), (256, 256), (512, 512)]
        for tile_shape in tile_shapes:
            # Get the tiled segmentation with unmerged instances at tile interfaces.
            labels, original_labels = self.get_tiled_data(tile_shape=tile_shape, size=1000)
            stitched_labels = stitch_tiled_segmentation(segmentation=labels, tile_shape=tile_shape, verbose=False)
            self._check_result(stitched_labels, original_labels)


if __name__ == "__main__":
    unittest.main()
