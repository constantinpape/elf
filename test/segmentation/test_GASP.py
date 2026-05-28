import unittest

import numpy as np

import bioimage_cpp as bic

from elf.segmentation import GaspFromAffinities
from elf.segmentation.gasp import run_GASP
from elf.segmentation.watershed import WatershedOnDistanceTransformFromAffinities


class TestGASP(unittest.TestCase):
    def setUp(self):
        edges = np.array([
            [0, 1], [0, 2], [0, 3],
            [1, 3],
            [2, 3]
        ], dtype="uint64")
        self.g = bic.graph.UndirectedGraph.from_edges(4, edges)
        self.edgeIndicators = np.array([-10, -2, 6, 3, 11], dtype="float32")

    def test_gasp_average(self):
        seg, _ = run_GASP(self.g, self.edgeIndicators, linkage_criteria="mean")
        self.assertTrue(seg[0] != seg[1] and seg[1] == seg[2] and seg[2] == seg[3])

    def test_gasp_abs_max(self):
        seg, _ = run_GASP(self.g, self.edgeIndicators, linkage_criteria="abs_max")
        self.assertTrue(seg[0] != seg[1] and seg[0] == seg[2] and seg[0] == seg[3])

    def test_gasp_single_linkage(self):
        # Merge edges (0,1) and (2,3); mutex on (0,2) and (1,3). Expected: {0,1}, {2,3}.
        edges = np.array([[0, 1], [2, 3], [0, 2], [1, 3]], dtype="uint64")
        graph = bic.graph.UndirectedGraph.from_edges(4, edges)
        # Positive => attractive (merge), negative => repulsive (mutex).
        weights = np.array([1.0, 1.0, -1.0, -1.0], dtype="float32")
        node_labels, _ = run_GASP(graph, weights, linkage_criteria="max")
        self.assertTrue(
            node_labels[0] == node_labels[1] and node_labels[2] == node_labels[3]
            and node_labels[0] != node_labels[3]
        )

    def test_gasp_sum(self):
        seg, _ = run_GASP(self.g, self.edgeIndicators, linkage_criteria="sum")
        seg = seg.tolist()
        self.assertTrue(seg[0] != seg[1] and seg[0] == seg[2] and seg[0] == seg[3])

    def test_GASP_from_random_affinities_from_WSDT_superpixels(self):
        IMAGE_SHAPE = (10, 40, 40)
        START_AGGLO_FROM_WSDT_SUPERPIXELS = True

        offsets = [
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [-1, -1, -1]]

        random_affinities = np.random.uniform(size=(len(offsets),) + IMAGE_SHAPE).astype("float32")

        if START_AGGLO_FROM_WSDT_SUPERPIXELS:
            superpixel_gen = WatershedOnDistanceTransformFromAffinities(offsets,
                                                                        threshold=0.4,
                                                                        stacked_2d=True,
                                                                        sigma_seeds=0.1,
                                                                        min_size=20,
                                                                        )
        else:
            superpixel_gen = None

        run_GASP_kwargs = {"linkage_criteria": "mutex_watershed"}

        gasp_instance = GaspFromAffinities(offsets,
                                           superpixel_generator=superpixel_gen,
                                           run_GASP_kwargs=run_GASP_kwargs)
        final_segmentation, runtime = gasp_instance(random_affinities)
        self.assertEqual(final_segmentation.shape, IMAGE_SHAPE)

    def test_GASP_from_random_affinities_from_pixels(self):
        IMAGE_SHAPE = (10, 40, 40)

        offsets = [
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [-1, -1, -1]]

        random_affinities = np.random.uniform(size=(len(offsets),) + IMAGE_SHAPE).astype("float32")
        run_GASP_kwargs = {"linkage_criteria": "average"}

        gasp_instance = GaspFromAffinities(offsets,
                                           superpixel_generator=None,
                                           run_GASP_kwargs=run_GASP_kwargs)

        random_foreground_mask = np.random.randint(1, size=IMAGE_SHAPE).astype("bool")
        final_segmentation, runtime = gasp_instance(random_affinities, foreground_mask=random_foreground_mask)
        self.assertEqual(final_segmentation.shape, IMAGE_SHAPE)


if __name__ == "__main__":
    unittest.main()
