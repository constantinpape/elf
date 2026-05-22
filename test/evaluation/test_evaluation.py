import unittest
import numpy as np

# reference implementations from skimage
from skimage.metrics import variation_of_information as voi_ref


class TestVariationOfInformation(unittest.TestCase):
    """Tests for elf.evaluation.variation_of_information."""

    def test_identical(self):
        from elf.evaluation import variation_of_information
        seg = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        vis, vim = variation_of_information(seg, seg)
        self.assertAlmostEqual(vis, 0.0)
        self.assertAlmostEqual(vim, 0.0)

    def test_pure_split(self):
        # gt has one big object that is split into two in seg → only split error.
        from elf.evaluation import variation_of_information
        seg = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        gt_merge = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [3, 3, 4, 4], [3, 3, 4, 4]])
        vis, vim = variation_of_information(seg, gt_merge)
        self.assertGreater(vis, 0.0)
        self.assertAlmostEqual(vim, 0.0)

    def test_pure_merge(self):
        # seg merges two gt objects → only merge error.
        from elf.evaluation import variation_of_information
        seg_merge = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [3, 3, 4, 4], [3, 3, 4, 4]])
        gt = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        vis, vim = variation_of_information(seg_merge, gt)
        self.assertAlmostEqual(vis, 0.0)
        self.assertGreater(vim, 0.0)

    def test_against_skimage(self):
        from elf.evaluation import variation_of_information
        rng = np.random.default_rng(42)
        x = rng.integers(1, 20, size=(64, 64))
        y = rng.integers(1, 20, size=(64, 64))
        vis, vim = variation_of_information(x, y)
        # skimage returns [h(image1|image0), h(image0|image1)],
        # elf returns (split, merge) = (h(seg|gt), h(gt|seg)).
        skim = voi_ref(x, y)
        self.assertAlmostEqual(vis, skim[1])
        self.assertAlmostEqual(vim, skim[0])

    def test_ignore_label(self):
        from elf.evaluation import variation_of_information
        # Padding both segmentations with the same background should not change the result.
        seg = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        gt = np.array([[1, 1, 1, 1], [3, 3, 4, 4]])
        vis_full, vim_full = variation_of_information(seg, gt)

        bg_seg = np.zeros((2, 4), dtype=seg.dtype)
        seg_padded = np.concatenate([seg, bg_seg], axis=0)
        gt_padded = np.concatenate([gt, bg_seg], axis=0)
        vis, vim = variation_of_information(seg_padded, gt_padded, ignore_seg=[0], ignore_gt=[0])
        self.assertAlmostEqual(vis, vis_full)
        self.assertAlmostEqual(vim, vim_full)

    def test_use_log2(self):
        from elf.evaluation import variation_of_information
        rng = np.random.default_rng(7)
        x = rng.integers(1, 10, size=(32, 32))
        y = rng.integers(1, 10, size=(32, 32))
        vis2, vim2 = variation_of_information(x, y, use_log2=True)
        vise, vime = variation_of_information(x, y, use_log2=False)
        # log_e and log_2 differ by a constant factor of log(2).
        self.assertAlmostEqual(vis2, vise / np.log(2))
        self.assertAlmostEqual(vim2, vime / np.log(2))


class TestRandIndex(unittest.TestCase):
    """Tests for elf.evaluation.rand_index."""

    def test_identical(self):
        from elf.evaluation import rand_index
        seg = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        ari, ri = rand_index(seg, seg)
        self.assertAlmostEqual(ari, 0.0)
        self.assertAlmostEqual(ri, 1.0)

    def test_increasing_error(self):
        from elf.evaluation import rand_index
        seg = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        gt_merge = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [3, 3, 4, 4], [3, 3, 4, 4]])
        gt_worse = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        ari_partial, _ = rand_index(seg, gt_merge)
        ari_worse, _ = rand_index(seg, gt_worse)
        self.assertGreater(ari_partial, 0.0)
        self.assertGreater(ari_worse, ari_partial)

    def test_ignore_label(self):
        from elf.evaluation import rand_index
        # ARI computed only on a non-background sub-region should equal
        # the ARI of the full image when background pixels are ignored.
        seg = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        gt = np.array([[1, 1, 1, 1], [3, 3, 4, 4]])
        ari_full, _ = rand_index(seg, gt)

        bg = np.zeros((2, 4), dtype=seg.dtype)
        seg_padded = np.concatenate([seg, bg], axis=0)
        gt_padded = np.concatenate([gt, bg], axis=0)
        ari, _ = rand_index(seg_padded, gt_padded, ignore_seg=[0], ignore_gt=[0])
        self.assertAlmostEqual(ari, ari_full)


class TestCremiScore(unittest.TestCase):
    """Tests for elf.evaluation.cremi_score."""

    def test_identical(self):
        from elf.evaluation import cremi_score
        seg = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        vis, vim, ari, cs = cremi_score(seg, seg)
        self.assertAlmostEqual(vis, 0.0)
        self.assertAlmostEqual(vim, 0.0)
        self.assertAlmostEqual(ari, 0.0)
        self.assertAlmostEqual(cs, 0.0)

    def test_geometric_mean(self):
        from elf.evaluation import cremi_score, rand_index, variation_of_information
        rng = np.random.default_rng(123)
        seg = rng.integers(1, 20, size=(64, 64))
        gt = rng.integers(1, 20, size=(64, 64))
        vis, vim, ari, cs = cremi_score(seg, gt)
        vis_exp, vim_exp = variation_of_information(seg, gt)
        ari_exp, _ = rand_index(seg, gt)
        self.assertAlmostEqual(vis, vis_exp)
        self.assertAlmostEqual(vim, vim_exp)
        self.assertAlmostEqual(ari, ari_exp)
        self.assertAlmostEqual(cs, float(np.sqrt(ari_exp * (vis_exp + vim_exp))))


class TestObjectVI(unittest.TestCase):
    """Tests for elf.evaluation.object_vi."""

    def test_identical(self):
        from elf.evaluation import object_vi
        seg = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        scores = object_vi(seg, seg)
        self.assertEqual(set(scores.keys()), {1, 2, 3, 4})
        for vis, vim in scores.values():
            self.assertAlmostEqual(vis, 0.0)
            self.assertAlmostEqual(vim, 0.0)

    def test_split_object(self):
        # In gt, label "1" covers 4 pixels but in seg it's split into 1 and 2.
        # So object 1 in gt should have non-zero merge VI (gt object is split in seg).
        from elf.evaluation import object_vi
        seg = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        gt = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [3, 3, 4, 4], [3, 3, 4, 4]])
        scores = object_vi(seg, gt)
        self.assertIn(1, scores)
        vis, vim = scores[1]
        self.assertGreater(vim, 0.0)
        # objects 3 and 4 are clean
        self.assertAlmostEqual(scores[3][1], 0.0)
        self.assertAlmostEqual(scores[4][1], 0.0)

    def test_ignore_label(self):
        from elf.evaluation import object_vi
        seg = np.array([[0, 0, 2, 2], [1, 1, 2, 2]])
        gt = np.array([[0, 0, 1, 1], [1, 1, 1, 1]])
        scores = object_vi(seg, gt, ignore_gt=[0])
        # background should not appear as a gt object after ignoring.
        self.assertNotIn(0, scores)


class TestContigencyTable(unittest.TestCase):
    """Tests for the elf.evaluation.util.contigency_table helper."""

    def test_simple(self):
        from elf.evaluation.util import contigency_table
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([0, 1, 1, 2, 2, 2])
        a_dict, b_dict, p_ids, p_counts = contigency_table(a, b)
        self.assertEqual(a_dict, {0: 2.0, 1: 2.0, 2: 2.0})
        self.assertEqual(b_dict, {0: 1.0, 1: 2.0, 2: 3.0})
        # Reassemble pairs into a dict for order-insensitive comparison.
        observed = {(int(ia), int(ib)): float(c) for (ia, ib), c in zip(p_ids, p_counts)}
        expected = {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (1, 2): 1.0, (2, 2): 2.0}
        self.assertEqual(observed, expected)

    def test_counts_sum(self):
        from elf.evaluation.util import contigency_table
        rng = np.random.default_rng(0)
        a = rng.integers(0, 8, size=(20, 20))
        b = rng.integers(0, 8, size=(20, 20))
        a_dict, b_dict, p_ids, p_counts = contigency_table(a, b)
        # totals should match.
        self.assertAlmostEqual(sum(a_dict.values()), a.size)
        self.assertAlmostEqual(sum(b_dict.values()), b.size)
        self.assertAlmostEqual(p_counts.sum(), a.size)
        # contigency table should have a row for every present label in a.
        self.assertEqual(set(p_ids[:, 0].tolist()), set(a_dict.keys()))
        self.assertEqual(set(p_ids[:, 1].tolist()), set(b_dict.keys()))


class TestLabelOverlap(unittest.TestCase):
    """Tests for elf.evaluation.matching.label_overlap."""

    def test_simple(self):
        from elf.evaluation.matching import label_overlap
        a = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        b = np.array([[1, 1, 1, 2], [1, 1, 1, 2]])
        overlap, ignore_idx = label_overlap(a, b, ignore_label=None)
        # 2 labels x 2 labels matrix
        self.assertEqual(overlap.shape, (2, 2))
        # Per-row sums should equal the per-label counts in a.
        self.assertEqual(overlap.sum(), a.size)
        self.assertEqual(ignore_idx, (None, None))

    def test_ignore_label(self):
        from elf.evaluation.matching import label_overlap
        a = np.array([[0, 1, 2], [1, 1, 2]])
        b = np.array([[0, 1, 2], [1, 2, 2]])
        overlap, ignore_idx = label_overlap(a, b, ignore_label=0)
        # ignore_label indices should be valid for the returned matrix.
        self.assertIsNotNone(ignore_idx[0])
        self.assertIsNotNone(ignore_idx[1])


if __name__ == "__main__":
    unittest.main()
