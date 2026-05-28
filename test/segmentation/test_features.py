import unittest
import numpy as np


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

    def test_compute_rag(self):
        from elf.segmentation.features import compute_rag
        shape = (16, 32, 32)
        seg = self.make_seg(shape)
        rag = compute_rag(seg)
        # bic RAG exposes both camelCase and snake_case aliases
        self.assertEqual(rag.numberOfNodes, int(seg.max()) + 1)
        self.assertEqual(rag.uvIds().shape[1], 2)
        self.assertGreater(rag.numberOfEdges, 0)

    def test_compute_boundary_features(self):
        from elf.segmentation.features import compute_rag, compute_boundary_features
        shape = (16, 32, 32)
        seg = self.make_seg(shape)
        rag = compute_rag(seg)
        bmap = np.random.rand(*shape).astype('float32')
        feats = compute_boundary_features(rag, seg, bmap)
        self.assertEqual(len(feats), rag.numberOfEdges)
        self.assertGreater(feats.shape[1], 1)

    def test_compute_affinity_features(self):
        from elf.segmentation.features import compute_rag, compute_affinity_features
        shape = (16, 32, 32)
        seg = self.make_seg(shape)
        rag = compute_rag(seg)
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        affs = np.random.rand(len(offsets), *shape).astype('float32')
        feats = compute_affinity_features(rag, seg, affs, offsets)
        self.assertEqual(len(feats), rag.numberOfEdges)
        self.assertGreater(feats.shape[1], 1)

    def test_compute_boundary_mean_and_length(self):
        from elf.segmentation.features import compute_rag, compute_boundary_mean_and_length
        shape = (16, 32, 32)
        seg = self.make_seg(shape)
        rag = compute_rag(seg)
        bmap = np.random.rand(*shape).astype('float32')
        feats = compute_boundary_mean_and_length(rag, seg, bmap)
        self.assertEqual(feats.shape, (rag.numberOfEdges, 2))
        # column 1 (size) must be > 0 for valid edges
        self.assertTrue((feats[:, 1] > 0).all())

    def test_project_node_labels_to_pixels(self):
        from elf.segmentation.features import compute_rag, project_node_labels_to_pixels
        shape = (8, 16, 16)
        seg = self.make_seg(shape)
        rag = compute_rag(seg)
        # collapse to a 2-class labelling
        node_labels = (np.arange(rag.numberOfNodes) % 2).astype('uint64')
        out = project_node_labels_to_pixels(rag, seg, node_labels)
        self.assertEqual(out.shape, shape)
        self.assertTrue((np.unique(out) <= [0, 1]).all())

    def test_get_stitch_edges(self):
        from elf.segmentation.features import compute_rag, get_stitch_edges
        shape = (32, 32)
        seg = self.make_seg(shape)
        rag = compute_rag(seg)
        mask = get_stitch_edges(rag, seg, block_shape=(16, 16))
        self.assertEqual(len(mask), rag.numberOfEdges)
        self.assertEqual(mask.dtype, np.bool_)

    def test_compute_z_edge_mask(self):
        from elf.segmentation.features import compute_rag, compute_z_edge_mask
        shape = (4, 8, 8)
        seg = np.zeros(shape, dtype='uint32')
        # flat 2D superpixels per z slice with unique ids per slice
        nz_per_slice = 4
        for z in range(shape[0]):
            slice_ids = np.arange(nz_per_slice).repeat(shape[1] * shape[2] // nz_per_slice)
            seg[z] = z * nz_per_slice + slice_ids.reshape(shape[1], shape[2])
        rag = compute_rag(seg)
        mask = compute_z_edge_mask(rag, seg)
        self.assertEqual(len(mask), rag.numberOfEdges)
        # at least some edges should cross slices
        self.assertGreater(int(mask.sum()), 0)

    def test_lifted_edges_from_graph_neighborhood(self):
        from elf.segmentation.features import compute_rag, lifted_edges_from_graph_neighborhood
        shape = (8, 16, 16)
        seg = self.make_seg(shape)
        rag = compute_rag(seg)
        lifted_uvs = lifted_edges_from_graph_neighborhood(rag, max_graph_distance=3)
        self.assertGreater(len(lifted_uvs), 0)
        self.assertEqual(lifted_uvs.shape[1], 2)

    def test_region_features(self):
        from elf.segmentation.features import compute_rag, compute_region_features

        shape = (32, 128, 128)
        ndim = len(shape)
        inp = np.random.rand(*shape).astype('float32')
        seg = self.make_seg(shape)
        rag = compute_rag(seg)
        uv_ids = rag.uvIds()

        feats = compute_region_features(uv_ids, inp, seg)
        self.assertEqual(len(uv_ids), len(feats))
        self.assertFalse(np.allclose(feats, 0))
        # The feature layout matches the previous vigra implementation: the per-node statistics
        # (Count, Kurtosis, Maximum, Minimum, Quantiles[7], RegionRadii[ndim], Skewness, Sum,
        # Variance -> 14 + ndim cols) combined via [min, max, absdiff, sum], plus the coordinate
        # features (weighted + geometric centroid -> 2 * ndim cols) combined via squared distance.
        self.assertEqual(feats.shape[1], 4 * (14 + ndim) + 2 * ndim)
        self.assertTrue(np.isfinite(feats).all())

    def test_region_features_values(self):
        from elf.segmentation.features import _region_features

        seg = np.array([[0, 0, 1], [2, 2, 1], [2, 4, 4]], dtype="uint32")  # labels 0, 1, 2, 4 (gap at 3)
        inp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
        feats = _region_features(inp, seg, ["Count", "mean", "RegionCenter"])

        # Count is dense over 0..max(label), with the gap (label 3) staying zero.
        np.testing.assert_array_equal(feats["Count"], np.bincount(seg.ravel(), minlength=5))
        for label in (0, 1, 2, 4):
            self.assertAlmostEqual(feats["mean"][label], inp[seg == label].mean(), places=5)
        self.assertEqual(feats["mean"][3], 0.0)
        # geometric centroid (row, col) of label 4 at (2, 1) and (2, 2)
        np.testing.assert_allclose(feats["RegionCenter"][4], [2.0, 1.5], atol=1e-5)

    def test_boundary_features_with_filters(self):
        from elf.segmentation.features import compute_rag, compute_boundary_features_with_filters

        shape = (64, 128, 128)
        inp = np.random.rand(*shape).astype('float32')
        seg = self.make_seg(shape)
        rag = compute_rag(seg)

        feats = compute_boundary_features_with_filters(rag, seg, inp)
        self.assertEqual(rag.numberOfEdges, len(feats))
        self.assertFalse(np.allclose(feats, 0))

        feats = compute_boundary_features_with_filters(rag, seg, inp, apply_2d=True)
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
        self.assertEqual(lifted_costs.ndim, 1)

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
            self.assertEqual(lifted_costs.ndim, 1)

    def test_grid_graph(self):
        from elf.segmentation.features import compute_grid_graph

        # 2d test
        shape = (64, 64)
        g = compute_grid_graph(shape)
        self.assertEqual(g.numberOfNodes, shape[0] * shape[1])

        # 3d test
        shape = (32, 64, 64)
        g = compute_grid_graph(shape)
        self.assertEqual(g.numberOfNodes, shape[0] * shape[1] * shape[2])

    def _test_grid_graph_affinity_features(self, shape, offsets, strides):
        from elf.segmentation.features import compute_grid_graph, compute_grid_graph_affinity_features

        def _check_feats(feats):
            self.assertGreater(len(feats), 0)
            self.assertFalse(np.allclose(feats, 0))

        ndim = len(shape)
        # test
        g = compute_grid_graph(shape)
        aff_shape = (ndim,) + shape
        affs = np.random.rand(*aff_shape).astype('float32')

        # for the case without offsets, the edges returned must correspond
        # to the edges of the grid graph
        edges, feats = compute_grid_graph_affinity_features(g, affs)
        self.assertEqual(len(edges), g.numberOfEdges)
        self.assertEqual(len(feats), g.numberOfEdges)
        self.assertTrue(np.array_equal(edges, g.uvIds()))
        _check_feats(feats)

        # test - with offsets
        aff_shape = (len(offsets),) + shape
        affs = np.random.rand(*aff_shape).astype('float32')
        edges, feats = compute_grid_graph_affinity_features(g, affs, offsets=offsets)
        self.assertEqual(len(edges), len(feats))
        self.assertEqual(edges.shape[1], 2)
        _check_feats(feats)
        n_edges_full = len(edges)

        n_edges_exp = int(n_edges_full / np.prod(strides))
        # test - with offsets and strides
        edges, feats = compute_grid_graph_affinity_features(g, affs,
                                                            offsets=offsets, strides=strides)
        self.assertEqual(len(edges), len(feats))
        self.assertEqual(edges.shape[1], 2)
        _check_feats(feats)
        self.assertLess(np.abs(n_edges_exp - len(edges)), 0.01 * n_edges_full)

        # test - with offsets and randomized strides
        edges, feats = compute_grid_graph_affinity_features(g, affs,
                                                            offsets=offsets, strides=strides,
                                                            randomize_strides=True)
        self.assertEqual(len(edges), len(feats))
        self.assertEqual(edges.shape[1], 2)
        _check_feats(feats)
        self.assertLess(np.abs(n_edges_exp - len(edges)), 0.01 * n_edges_full)

    def test_grid_graph_affinity_features_2d(self):
        self._test_grid_graph_affinity_features(shape=(64, 64),
                                                offsets=[[-3, 0], [0, -3], [-9, 2], [-12, -7], [3, 3]],
                                                strides=[4, 4])

    def test_grid_graph_affinity_features_3d(self):
        self._test_grid_graph_affinity_features(shape=(32, 64, 64),
                                                offsets=[[0, -1, 1], [3, 0, 4],
                                                         [-7, 9, 32], [11, 7, 9]],
                                                strides=[2, 4, 4])

    def _test_grid_graph_image_features(self, shape, offsets, strides):
        from elf.segmentation.features import compute_grid_graph, compute_grid_graph_image_features

        def _check_dists(feats, dist):
            self.assertGreater(len(feats), 0)
            self.assertFalse(np.allclose(feats, 0))
            # all distances must be greater than 0
            self.assertTrue((feats >= 0).all())
            # cosine distance is bounded at 1
            if dist == 'cosine':
                self.assertTrue((feats <= 1).all())

        # test
        g = compute_grid_graph(shape)
        im_shape = (6,) + shape
        im = np.random.rand(*im_shape).astype('float32')
        for dist in ('l1', 'l2', 'cosine'):
            edges, feats = compute_grid_graph_image_features(g, im, dist)
            # for the case without offsets, the edges returned must correspond
            # to the edges of the grid graph
            self.assertEqual(len(edges), g.numberOfEdges)
            self.assertEqual(len(feats), g.numberOfEdges)
            self.assertTrue(np.array_equal(edges, g.uvIds()))
            _check_dists(feats, dist)

            # with offsets
            edges, feats = compute_grid_graph_image_features(g, im, dist,
                                                             offsets=offsets)
            self.assertEqual(len(feats), len(edges))
            _check_dists(feats, dist)
            n_edges_full = len(edges)

            n_edges_exp = int(n_edges_full / np.prod(strides))
            # with strides
            edges, feats = compute_grid_graph_image_features(g, im, dist,
                                                             offsets=offsets,
                                                             strides=strides)
            self.assertEqual(len(feats), len(edges))
            _check_dists(feats, dist)
            self.assertLess(np.abs(n_edges_exp - len(edges)), 0.01 * n_edges_full)

            # with randomized strides
            edges, feats = compute_grid_graph_image_features(g, im, dist,
                                                             offsets=offsets,
                                                             strides=strides,
                                                             randomize_strides=True)
            self.assertEqual(len(feats), len(edges))
            _check_dists(feats, dist)
            self.assertLess(len(edges), n_edges_full)
            self.assertLess(np.abs(n_edges_exp - len(edges)), 0.01 * n_edges_full)

    def test_grid_graph_image_features_2d(self):
        self._test_grid_graph_image_features(shape=(64, 64),
                                             offsets=[[-3, 0], [0, -3], [-9, 2], [-12, -7], [3, 3]],
                                             strides=[4, 4])

    def test_grid_graph_image_features_3d(self):
        self._test_grid_graph_image_features(shape=(32, 64, 64),
                                             offsets=[[0, -1, 1], [3, 0, 4],
                                                      [-7, 9, 32], [11, 7, 9]],
                                             strides=[2, 4, 4])

    def test_apply_mask_to_grid_graph_weights(self):
        from elf.segmentation.features import (apply_mask_to_grid_graph_weights,
                                               compute_grid_graph,
                                               compute_grid_graph_affinity_features)
        shape = (256, 256)
        aff_shape = (2,) + shape
        affs = np.random.rand(*aff_shape).astype("float32")
        g = compute_grid_graph(shape)
        _, weights = compute_grid_graph_affinity_features(g, affs)
        mask = np.random.rand(*shape) > 0.5
        weights = apply_mask_to_grid_graph_weights(g, mask, weights)
        self.assertEqual(len(weights), g.numberOfEdges)

    def test_apply_mask_to_grid_graph_edges_and_weights(self):
        from elf.segmentation.features import (apply_mask_to_grid_graph_edges_and_weights,
                                               compute_grid_graph,
                                               compute_grid_graph_affinity_features)
        shape = (256, 256)
        offsets = [[-3, 0], [0, -3], [3, -3], [3, 9]]
        aff_shape = (len(offsets),) + shape
        affs = np.random.rand(*aff_shape).astype("float32")
        g = compute_grid_graph(shape)
        edges, weights = compute_grid_graph_affinity_features(g, affs, offsets=offsets)
        n_edges_prev = len(edges)
        mask = np.random.rand(*shape) > 0.5
        edges, weights = apply_mask_to_grid_graph_edges_and_weights(g, mask, edges, weights)
        self.assertEqual(len(weights), len(edges))
        self.assertGreater(n_edges_prev, len(edges))


if __name__ == '__main__':
    unittest.main()
