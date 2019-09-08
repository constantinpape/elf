import os
import unittest
import numpy as np

try:
    import nifty
except ImportError:
    nifty = None

try:
    import z5py
except ImportError:
    z5py = None

DATA_FOLDER = os.path.join(os.path.split(__file__)[0],
                           '../../data/label_multiset')


class TestLabelMultisetPaintera(unittest.TestCase):
    path = os.path.join(DATA_FOLDER, 'test.n5')
    expected_path = os.path.join(DATA_FOLDER, 'test_paintera.n5')

    def check_multisets(self, l1, l2):
        # check number of sets and entries
        self.assertEqual(l1.size, l2.size)
        self.assertEqual(l1.n_entries, l2.n_entries)
        # check amax vector
        self.assertTrue(np.array_equal(l1.argmax, l2.argmax))
        # check offset vector
        self.assertTrue(np.array_equal(l1.offsets, l2.offsets))
        # check ids and counts
        self.assertTrue(np.array_equal(l1.ids, l2.ids))
        self.assertTrue(np.array_equal(l1.counts, l2.counts))

    def check_serializations(self, result, expected, shape):
        from elf.label_multiset import deserialize_multiset

        # 1.) deserialize to multisets and check they agree
        # we do this in addition to check 2.)
        # to pinpoint where things might go wrong
        l1 = deserialize_multiset(result, shape)
        l2 = deserialize_multiset(expected, shape)
        self.check_multisets(l1, l2)

        # 2.) check overal agreement of the serialization
        self.assertEqual(len(result), len(expected))
        self.assertTrue(np.array_equal(result, expected))

    def check_expected(self, key, key_expected):
        from elf.label_multiset import (create_multiset_from_labels,
                                        serialize_multiset)
        f = z5py.File(self.path)
        x = f[key][:]
        multiset = create_multiset_from_labels(x)
        self.assertEqual(multiset.shape, x.shape)
        ser = serialize_multiset(multiset)

        f = z5py.File(self.expected_path)
        expected_ser = f[key_expected].read_chunk((0, 0, 0))
        self.check_serializations(ser, expected_ser, x.shape)

    def check_downscale(self, key, key_expected):
        from elf.label_multiset import (create_multiset_from_labels,
                                        downsample_multiset,
                                        serialize_multiset)
        f = z5py.File(self.path)
        x = f[key][:]
        multiset = create_multiset_from_labels(x)
        self.assertEqual(multiset.shape, x.shape)
        multiset = downsample_multiset(multiset, [2, 2, 2], -1)
        self.assertEqual(multiset.shape, tuple(sh // 2 for sh in x.shape))
        ser = serialize_multiset(multiset)

        f = z5py.File(self.expected_path)
        expected_ser = f[key_expected].read_chunk((0, 0, 0))
        self.check_serializations(ser, expected_ser, x.shape)

    @unittest.skipUnless(nifty and z5py, "Need nifty and z5py")
    def test_create_from_labels_uniform(self):
        self.check_expected('uniform', 'uniform/data/s0')

    @unittest.skipUnless(nifty and z5py, "Need nifty and z5py")
    def test_create_from_labels_range(self):
        self.check_expected('range', 'range/data/s0')

    @unittest.skipUnless(nifty and z5py, "Need nifty and z5py")
    def test_create_from_multiset_range(self):
        self.check_downscale('range', 'range/data/s1')

    @unittest.skipUnless(nifty and z5py, "Need nifty and z5py")
    def test_create_from_multiset_uniform(self):
        self.check_downscale('uniform', 'uniform/data/s1')


if __name__ == '__main__':
    unittest.main()
