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

    @unittest.skipUnless(vigra and nifty, "Need vigra and nifty")
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

    # FIXME this sometimes fails with out of bounds,
    # apparently because node ids in lifted edges can be 1 to big
    @unittest.skipUnless(vigra and nifty, "Need vigra and nifty")
    def test_lifted_problem_from_segmentation(self):
        from elf.segmentation.features import (compute_rag,
                                               lifted_problem_from_segmentation)
        shape = (32, 128, 128)
        ws = self.make_seg(shape)
        rag = compute_rag(ws)
        seg = self.make_seg(shape)

        overlap_threshold = .5
        graph_depth = 4
        lifted_uvs, lifted_costs = lifted_problem_from_segmentation(rag, ws, seg,
                                                                    overlap_threshold, graph_depth,
                                                                    same_segment_cost=1., different_segment_cost=-1)
        self.assertEqual(len(lifted_uvs), len(lifted_costs))


if __name__ == '__main__':
    unittest.main()
