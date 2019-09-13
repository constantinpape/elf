import unittest
import numpy as np

try:
    import nifty
except ImportError:
    nifty = None


class TestLabelMultiset(unittest.TestCase):

    def check_multisets(self, l1, l2, allow_equivalent=False):
        # check number of sets and entries
        self.assertEqual(l1.size, l2.size)
        self.assertEqual(l1.n_entries, l2.n_entries)
        # check amax vector
        self.assertTrue(np.array_equal(l1.argmax, l2.argmax))
        # multisets can equivalent without equality of offsets, ids, counts
        # but sizes need to agree
        if allow_equivalent:
            self.assertEqual(l1.offsets.shape, l2.offsets.shape)
            self.assertEqual(l1.ids.shape, l2.ids.shape)
            self.assertEqual(l1.counts.shape, l2.counts.shape)
        else:
            # check offset vector
            self.assertTrue(np.array_equal(l1.offsets, l2.offsets))
            # check ids and counts
            self.assertTrue(np.array_equal(l1.ids, l2.ids))
            self.assertTrue(np.array_equal(l1.counts, l2.counts))

    @unittest.skipUnless(nifty, "Need nifty")
    def test_serialization(self):
        from elf.label_multiset import (create_multiset_from_labels,
                                        serialize_multiset, deserialize_multiset)
        shape = (32, 32, 32)
        x = np.random.randint(0, 2000, size=shape, dtype='uint64')
        l1 = create_multiset_from_labels(x)
        ser = serialize_multiset(l1)
        l2 = deserialize_multiset(ser, shape)
        self.check_multisets(l1, l2)

    @unittest.skipUnless(nifty, "Need nifty")
    def test_multiset(self):
        from elf.label_multiset import create_multiset_from_labels
        shape = (32, 32, 32)
        x = np.random.randint(0, 2000, size=shape, dtype='uint64')
        multiset = create_multiset_from_labels(x)
        self.assertEqual(multiset.shape, x.shape)

        slices_to_check = [np.s_[:2, :2, :2], np.s_[:2, :4, :8], np.s_[:1, :16, :1], np.s_[:]]
        for slice_ in slices_to_check:
            ids, counts = multiset[slice_]
            ids_exp, counts_exp = np.unique(x[slice_], return_counts=True)
            self.assertTrue(np.array_equal(ids, ids_exp))
            self.assertTrue(np.array_equal(counts, counts_exp))

    @unittest.skipUnless(nifty, "Need nifty")
    def test_multiset_ds(self):
        from elf.label_multiset import (create_multiset_from_labels,
                                        downsample_multiset)
        shape = (32, 32, 32)
        x = np.random.randint(0, 2000, size=shape, dtype='uint64')
        multiset = create_multiset_from_labels(x)
        ds = [2, 2, 2]
        multiset = downsample_multiset(multiset, ds)
        self.assertEqual(multiset.shape, tuple(sh // dd for sh, dd in zip(x.shape, ds)))

        slices_to_check = [(np.s_[:1, :1, :1], np.s_[:2, :2, :2]),
                           (np.s_[:1, :2, :4], np.s_[:2, :4, :8]),
                           (np.s_[:1, :8, :1], np.s_[:2, :16, :2]),
                           (np.s_[:], np.s_[:])]
        for slice_a, slice_b in slices_to_check:
            # print("Checking", slice_a, slice_b)
            ids, counts = multiset[slice_a]
            # print(ids, counts)
            ids_exp, counts_exp = np.unique(x[slice_b], return_counts=True)
            # print(ids_exp, counts_exp)
            self.assertTrue(np.array_equal(ids, ids_exp))
            self.assertTrue(np.array_equal(counts, counts_exp))

    @unittest.skipUnless(nifty, "Need nifty")
    def test_merge_multisets(self):
        from elf.label_multiset import (create_multiset_from_labels,
                                        merge_multisets)
        shape = (32, 32, 32)
        chunks = (16, 16, 16)
        x = np.random.randint(0, 2000, size=shape, dtype='uint64')

        multisets = []
        grid_positions = []

        for ii in (0, 1):
            for jj in (0, 1):
                for kk in (0, 1):
                    grid_pos = (ii, jj, kk)
                    slice_ = tuple(slice(p * ch, (p + 1) * ch)
                                   for p, ch in zip(grid_pos, chunks))
                    multiset = create_multiset_from_labels(x[slice_])
                    multisets.append(multiset)
                    grid_positions.append(grid_pos)

        multiset = merge_multisets(multisets, grid_positions, shape, chunks)
        self.assertEqual(multiset.shape, shape)

        # NOTE we can't expect the two multi-sets to be identical because of permutation
        # invariance of the offsets. But we can expect the same number of entries, ids etc
        multiset_expected = create_multiset_from_labels(x)
        self.check_multisets(multiset, multiset_expected, allow_equivalent=True)

        slices_to_check = [np.s_[:1, :1, :1], np.s_[:2, :2, :2],
                           np.s_[:2, :4, :8], np.s_[:3, :16, :19], np.s_[:]]
        for slice_ in slices_to_check:
            ids, counts = multiset[slice_]
            ids_exp, counts_exp = np.unique(x[slice_], return_counts=True)
            self.assertTrue(np.array_equal(ids, ids_exp))
            self.assertTrue(np.array_equal(counts, counts_exp))


if __name__ == '__main__':
    unittest.main()
