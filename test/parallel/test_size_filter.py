import unittest
import numpy as np
from skimage.data import binary_blobs
from skimage.measure import label


def _size_filter(seg, min_size=None, max_size=None):
    ids, counts = np.unique(seg, return_counts=True)

    filter_ids = []
    if min_size is not None:
        filter_ids.extend(ids[counts < min_size])
    if max_size is not None:
        filter_ids.extend(ids[counts > max_size])
    filter_ids = np.array(filter_ids)

    res = seg.copy()
    res[np.isin(seg, filter_ids)] = 0
    return res


class TestSizeFilter(unittest.TestCase):
    def test_size_filter(self):
        from elf.parallel.size_filter import size_filter

        seg = binary_blobs(length=512, n_dim=2, volume_fraction=0.1)
        seg = label(seg, connectivity=1)

        _, sizes = np.unique(seg, return_counts=True)
        min_size = np.percentile(sizes, 25)
        max_size = np.percentile(sizes, 75)

        block_shape = (128, 128)

        filtered_res = np.zeros_like(seg)
        filtered_res = size_filter(seg, filtered_res, min_size=min_size, block_shape=block_shape, relabel=False)
        _, filtered_sizes = np.unique(filtered_res, return_counts=True)
        self.assertTrue((filtered_sizes >= min_size).all())
        filtered_exp = _size_filter(seg, min_size=min_size)
        self.assertTrue(np.array_equal(filtered_res, filtered_exp))

        filtered_res = np.zeros_like(seg)
        filtered_res = size_filter(seg, filtered_res, max_size=max_size, block_shape=block_shape, relabel=False)
        _, filtered_sizes = np.unique(filtered_res, return_counts=True)
        filtered_sizes = filtered_sizes[1:]
        self.assertTrue((filtered_sizes <= max_size).all())
        filtered_exp = _size_filter(seg, max_size=max_size)
        self.assertTrue(np.array_equal(filtered_res, filtered_exp))

        filtered_res = np.zeros_like(seg)
        filtered_res = size_filter(
            seg, filtered_res, min_size=min_size, max_size=max_size, block_shape=block_shape, relabel=False
        )
        _, filtered_sizes = np.unique(filtered_res, return_counts=True)
        self.assertTrue((filtered_sizes >= min_size).all())
        filtered_sizes = filtered_sizes[1:]
        self.assertTrue((filtered_sizes <= max_size).all())
        filtered_exp = _size_filter(seg, min_size=min_size, max_size=max_size)
        self.assertTrue(np.array_equal(filtered_res, filtered_exp))

    def test_size_filter_with_relabeling(self):
        from elf.parallel.size_filter import size_filter

        seg = binary_blobs(length=512, n_dim=2, volume_fraction=0.1)
        seg = label(seg, connectivity=1)

        _, sizes = np.unique(seg, return_counts=True)
        min_size = np.percentile(sizes, 25)

        block_shape = (128, 128)
        filtered_res = np.zeros_like(seg)
        filtered_res = size_filter(seg, filtered_res, min_size=min_size, block_shape=block_shape, relabel=True)
        filtered_ids, filtered_sizes = np.unique(filtered_res, return_counts=True)
        self.assertTrue((filtered_sizes >= min_size).all())

        # check that the array is consecutive
        diff = np.ediff1d(filtered_ids)
        self.assertTrue((diff == 1).all())


if __name__ == "__main__":
    unittest.main()
