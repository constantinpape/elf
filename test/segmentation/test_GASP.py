import unittest

import numpy as np

import nifty
import nifty.graph
import nifty.graph.agglo as nagglo
from nifty.graph import components

from elf.segmentation import GaspFromAffinities
from elf.segmentation.watershed import WatershedOnDistanceTransformFromAffinities


class TestGASP(unittest.TestCase):
    def setUp(self):
        # Create a small graph:
        self.g = g = nifty.graph.UndirectedGraph(4)
        edges = np.array([
            [0, 1], [0, 2], [0, 3],
            [1, 3],
            [2, 3]
        ], dtype='uint64')
        g.insertEdges(edges)

        self.edgeIndicators = np.array([-10, -2, 6, 3, 11], dtype='float32')

    def test_gasp_average(self):
        clusterPolicy = nagglo.get_GASP_policy(
            graph=self.g,
            signed_edge_weights=self.edgeIndicators,
            linkage_criteria='mean',
            add_cannot_link_constraints=False)

        agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy)
        agglomerativeClustering.run()
        seg = agglomerativeClustering.result()
        self.assertTrue(seg[0] != seg[1] and seg[1] == seg[2] and seg[2] == seg[3])

    def test_gasp_abs_max(self):
        clusterPolicy = nagglo.get_GASP_policy(
            graph=self.g,
            signed_edge_weights=self.edgeIndicators,
            linkage_criteria='abs_max')

        agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy)
        agglomerativeClustering.run()
        seg = agglomerativeClustering.result()
        self.assertTrue(seg[0] != seg[1] and seg[0] == seg[2] and seg[0] == seg[3])

    def test_gasp_single_linkage(self):
        import nifty.graph as ngraph
        rag = ngraph.undirectedGridGraph((2, 2))
        connections = np.array([0, 0, 1, 1])
        graph_components = components(rag)
        graph_components.buildFromEdgeLabels(connections)
        node_labels = graph_components.componentLabels()
        self.assertTrue(
            node_labels[0] == node_labels[1] and node_labels[2] == node_labels[3] and node_labels[0] != node_labels[3])

    def test_gasp_sum(self):
        clusterPolicy = nagglo.get_GASP_policy(
            graph=self.g,
            signed_edge_weights=self.edgeIndicators,
            linkage_criteria='sum',
            add_cannot_link_constraints=False)

        agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy)
        agglomerativeClustering.run()
        seg = agglomerativeClustering.result().tolist()
        self.assertTrue(seg[0] != seg[1] and seg[0] == seg[2] and seg[0] == seg[3])

    def test_GASP_from_random_affinities_from_WSDT_superpixels(self):
        IMAGE_SHAPE = (10, 40, 40)
        START_AGGLO_FROM_WSDT_SUPERPIXELS = True

        offsets = [
            # Direct 3D neighborhood:
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            # Long-range connections:
            [-1, -1, -1]]

        # Generate some random affinities:
        random_affinities = np.random.uniform(size=(len(offsets),) + IMAGE_SHAPE).astype('float32')

        # Run GASP:
        if START_AGGLO_FROM_WSDT_SUPERPIXELS:
            # In this case the agglomeration is initialized with superpixels:
            # use additional option 'intersect_with_boundary_pixels' to break the SP along the boundaries
            # (see CREMI-experiments script for an example)
            superpixel_gen = WatershedOnDistanceTransformFromAffinities(offsets,
                                                                        threshold=0.4,
                                                                        stacked_2d=True,
                                                                        sigma_seeds=0.1,
                                                                        min_size=20,
                                                                        )
        else:
            superpixel_gen = None

        run_GASP_kwargs = {'linkage_criteria': 'mutex_watershed',
                           'add_cannot_link_constraints': False}

        gasp_instance = GaspFromAffinities(offsets,
                                           superpixel_generator=superpixel_gen,
                                           run_GASP_kwargs=run_GASP_kwargs)
        final_segmentation, runtime = gasp_instance(random_affinities)
        self.assertEqual(final_segmentation.shape, IMAGE_SHAPE)

    def test_GASP_from_random_affinities_from_pixels(self):
        IMAGE_SHAPE = (10, 40, 40)
        START_AGGLO_FROM_WSDT_SUPERPIXELS = False

        offsets = [
            # Direct 3D neighborhood:
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            # Long-range connections:
            [-1, -1, -1]]

        # Generate some random affinities:
        random_affinities = np.random.uniform(size=(len(offsets),) + IMAGE_SHAPE).astype('float32')

        # Run GASP:
        if START_AGGLO_FROM_WSDT_SUPERPIXELS:
            # In this case the agglomeration is initialized with superpixels:
            # use additional option 'intersect_with_boundary_pixels' to break the SP along the boundaries
            # (see CREMI-experiments script for an example)
            superpixel_gen = WatershedOnDistanceTransformFromAffinities(offsets,
                                                                        threshold=0.4,
                                                                        stacked_2d=True,
                                                                        sigma_seeds=0.1,
                                                                        min_size=20,
                                                                        )
        else:
            superpixel_gen = None

        run_GASP_kwargs = {'linkage_criteria': 'average',
                           'add_cannot_link_constraints': False}

        gasp_instance = GaspFromAffinities(offsets,
                                           superpixel_generator=superpixel_gen,
                                           run_GASP_kwargs=run_GASP_kwargs)

        random_foreground_mask = np.random.randint(1, size=IMAGE_SHAPE).astype("bool")
        final_segmentation, runtime = gasp_instance(random_affinities, foreground_mask=random_foreground_mask)
        self.assertEqual(final_segmentation.shape, IMAGE_SHAPE)


if __name__ == '__main__':
    unittest.main()
