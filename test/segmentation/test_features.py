import unittest
import numpy as np
try:
    import vigra
except ImportError:
    vigra = None
try:
    import nifty
except ImportError:
    nifty = None


@unittest.skipUnless(vigra and nifty, "Need vigra and nifty")
class TestFeatures(unittest.TestCase):

    def make_seg(self, shape):
        size = np.prod([sh for sh in shape])
        seg = np.zeros(size, dtype='uint32')
        current_id = 1
        change_prob = .99
        for i in range(seg.size):
            seg[i] = current_id
            if np.random.rand() > change_prob:
                current_id += 1
        return seg.reshape(shape)

    def test_region_features(self):
        from elf.segmentation.features import compute_rag, compute_region_features

        shape = (32, 128, 128)
        inp = np.random.rand(*shape).astype('float32')
        seg = self.make_seg(shape)
        rag = compute_rag(seg)
        uv_ids = rag.uvIds()

        feats = compute_region_features(uv_ids, inp, seg)
        self.assertEqual(len(uv_ids), len(feats))
        self.assertFalse(np.allclose(feats, 0))

    def test_boundary_features_with_filters(self):
        from elf.segmentation.features import compute_rag, compute_boundary_features_with_filters

        shape = (64, 128, 128)
        inp = np.random.rand(*shape).astype('float32')
        seg = self.make_seg(shape)
        rag = compute_rag(seg)

        feats = compute_boundary_features_with_filters(rag, inp)
        self.assertEqual(rag.numberOfEdges, len(feats))
        self.assertFalse(np.allclose(feats, 0))

        feats = compute_boundary_features_with_filters(rag, inp, apply_2d=True)
        self.assertEqual(rag.numberOfEdges, len(feats))
        self.assertFalse(np.allclose(feats, 0))

    def test_lifted_problem_from_probabilities(self):
        from elf.segmentation.features import (compute_rag,
                                               lifted_problem_from_probabilities)
        shape = (32, 128, 128)
        seg = self.make_seg(shape)
        rag = compute_rag(seg)

        n_classes = 3
        input_maps = [np.random.rand(*shape).astype('float32') for _ in range(n_classes)]

        assignment_threshold = .5
        graph_depth = 4
        lifted_uvs, lifted_costs = lifted_problem_from_probabilities(rag, seg, input_maps,
                                                                     assignment_threshold, graph_depth)
        self.assertEqual(len(lifted_uvs), len(lifted_costs))

    def test_lifted_problem_from_segmentation(self):
        from elf.segmentation.features import (compute_rag,
                                               lifted_problem_from_segmentation)
        shape = (32, 128, 128)
        N = 5
        for _ in range(N):
            ws = self.make_seg(shape)
            rag = compute_rag(ws)
            seg = self.make_seg(shape)

            overlap_threshold = .5
            graph_depth = 4
            lifted_uvs, lifted_costs = lifted_problem_from_segmentation(rag, ws, seg,
                                                                        overlap_threshold,
                                                                        graph_depth,
                                                                        same_segment_cost=1.,
                                                                        different_segment_cost=-1)
            self.assertEqual(len(lifted_uvs), len(lifted_costs))


if __name__ == '__main__':
    unittest.main()
