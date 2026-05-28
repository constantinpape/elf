import unittest
import numpy as np


def _make_seg(shape, change_prob=0.99):
    size = int(np.prod(shape))
    seg = np.zeros(size, dtype="uint32")
    current = 1
    for i in range(size):
        seg[i] = current
        if np.random.rand() > change_prob:
            current += 1
    return seg.reshape(shape)


class TestLearning(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)

    def test_compute_edge_labels(self):
        from elf.segmentation.features import compute_rag
        from elf.segmentation.learning import compute_edge_labels
        shape = (32, 64, 64)
        seg = _make_seg(shape)
        gt = _make_seg(shape, change_prob=0.95)
        rag = compute_rag(seg)
        edge_labels = compute_edge_labels(rag, seg, gt)
        self.assertEqual(len(edge_labels), rag.number_of_edges)
        self.assertEqual(edge_labels.dtype, np.uint8)
        # mix of cut and merge edges expected
        self.assertGreater(int(edge_labels.sum()), 0)
        self.assertLess(int(edge_labels.sum()), rag.number_of_edges)

    def test_compute_edge_labels_with_ignore(self):
        from elf.segmentation.features import compute_rag
        from elf.segmentation.learning import compute_edge_labels
        shape = (16, 32, 32)
        seg = _make_seg(shape)
        gt = _make_seg(shape, change_prob=0.97)
        # zero out a portion of gt so it becomes the ignore label
        gt[:4] = 0
        rag = compute_rag(seg)
        edge_labels, edge_mask = compute_edge_labels(rag, seg, gt, ignore_label=0)
        self.assertEqual(len(edge_labels), rag.number_of_edges)
        self.assertEqual(len(edge_mask), rag.number_of_edges)
        self.assertLess(int(edge_mask.sum()), len(edge_mask))

    def test_learn_edge_random_forest(self):
        from elf.segmentation.learning import learn_edge_random_forest, predict_edge_random_forest
        rng = np.random.default_rng(0)
        n_edges = 200
        feats = rng.normal(size=(n_edges, 4)).astype("float32")
        # decision rule: positive sum → cut
        labels = (feats.sum(axis=1) > 0).astype("uint8")
        rf = learn_edge_random_forest(feats, labels, n_estimators=10, max_depth=3)
        probs = predict_edge_random_forest(rf, feats)
        self.assertEqual(len(probs), n_edges)
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())
        # the RF should fit reasonably well on its own training data
        preds = (probs > 0.5).astype("uint8")
        acc = float((preds == labels).mean())
        self.assertGreater(acc, 0.8)

    def test_learn_edge_random_forest_with_mask(self):
        from elf.segmentation.learning import learn_edge_random_forest
        rng = np.random.default_rng(1)
        n_edges = 100
        feats = rng.normal(size=(n_edges, 3)).astype("float32")
        labels = (feats[:, 0] > 0).astype("uint8")
        mask = np.zeros(n_edges, dtype=bool)
        mask[:80] = True
        rf = learn_edge_random_forest(feats, labels, edge_mask=mask,
                                      n_estimators=10, max_depth=3)
        # RF should fit on the masked subset
        self.assertEqual(rf.n_classes_, 2)

    def test_learn_and_predict_xyz_edges(self):
        from elf.segmentation.learning import (
            learn_random_forests_for_xyz_edges,
            predict_edge_random_forests_for_xyz_edges,
        )
        rng = np.random.default_rng(2)
        n_edges = 200
        feats = rng.normal(size=(n_edges, 3)).astype("float32")
        labels = (feats.sum(axis=1) > 0).astype("uint8")
        z_edges = np.zeros(n_edges, dtype=bool)
        z_edges[::3] = True
        rf_xy, rf_z = learn_random_forests_for_xyz_edges(
            feats, labels, z_edges=z_edges, n_estimators=10, max_depth=3,
        )
        probs = predict_edge_random_forests_for_xyz_edges(rf_xy, rf_z, feats, z_edges)
        self.assertEqual(len(probs), n_edges)
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())


if __name__ == "__main__":
    unittest.main()
