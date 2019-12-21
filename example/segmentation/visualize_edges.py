import numpy as np
import napari

from elf.io import open_file
import elf.segmentation.features as feats
import elf.segmentation.watershed as ws
import elf.segmentation.multicut as mc

from elf.visualisation import visualise_attractive_and_repulsive_edges
from elf.visualisation import visualise_edges


def visualize_edges_isbi():
    """ Example for edge weight visualization, which is very useful to debug
    multicut (or other agglomeration) problems.
    """

    data_path = '/home/pape/Work/data/isbi/isbi_test_volume.h5'  # adjust this path
    with open_file(data_path, 'r') as f:
        # load the raw data
        raw = f['raw'][:]
        # load the affinities, we only need the first 3 channels
        affs = f['affinities'][:3, :]

    boundaries = np.mean(affs, axis=0)
    watershed, max_id = ws.stacked_watershed(boundaries, threshold=.5, sigma_seeds=2.)
    # compute the region adjacency graph
    rag = feats.compute_rag(watershed, n_labels=max_id + 1)

    # compute the edge weights
    edge_weights = feats.compute_boundary_features(rag, boundaries)[:, 0]
    z_edges = feats.compute_z_edge_mask(rag, watershed)
    xy_edges = ~z_edges
    xy_vals = visualise_edges(rag, edge_weights, ignore_edges=z_edges, edge_direction=0)
    z_vals = visualise_edges(rag, edge_weights, ignore_edges=xy_edges, edge_direction=2)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='raw')
        viewer.add_image(boundaries, name='boundaries')

        viewer.add_image(xy_vals, name='xy-edges')
        viewer.add_image(z_vals, name='z-edges')


def visualize_attractive_and_repulsive_edges_isbi(view_costs=False, weighting_scheme=None):
    """ Example for edge weight visualization, which is very useful to debug
    multicut (or other agglomeration) problems.
    """
    assert weighting_scheme in (None, 'z', 'xyz')

    data_path = '/home/pape/Work/data/isbi/isbi_test_volume.h5'  # adjust this path
    with open_file(data_path, 'r') as f:
        # load the raw data
        raw = f['raw'][:]
        # load the affinities, we only need the first 3 channels
        affs = f['affinities'][:3, :]

    boundaries = np.mean(affs, axis=0)
    watershed, max_id = ws.stacked_watershed(boundaries, threshold=.5, sigma_seeds=2.)
    # compute the region adjacency graph
    rag = feats.compute_rag(watershed, n_labels=max_id + 1)

    # compute edge features and xy/z edges
    edge_feats = feats.compute_boundary_mean_and_length(rag, boundaries)
    edge_feats, edge_sizes = edge_feats[:, 0], edge_feats[:, 1]

    z_edges = feats.compute_z_edge_mask(rag, watershed)
    xy_edges = ~z_edges

    if view_costs:
        if weighting_scheme is None:
            edge_populations = None
        elif weighting_scheme == 'z':
            edge_populations = [z_edges]
        elif weighting_scheme == 'xyz':
            edge_populations = [xy_edges, z_edges]
        costs = mc.transform_probabilities_to_costs(edge_feats, edge_sizes=edge_sizes,
                                                    edge_populations=edge_populations)
        att1xy, rep1xy = visualise_attractive_and_repulsive_edges(rag, costs,
                                                                  ignore_edges=z_edges,
                                                                  threshold=0.,
                                                                  large_values_are_attractive=True,
                                                                  edge_direction=0)
        att1z, rep1z = visualise_attractive_and_repulsive_edges(rag, costs,
                                                                ignore_edges=xy_edges,
                                                                threshold=0.,
                                                                large_values_are_attractive=True,
                                                                edge_direction=2)
    else:
        att1xy, rep1xy = visualise_attractive_and_repulsive_edges(rag, edge_feats,
                                                                  ignore_edges=z_edges,
                                                                  threshold=.5,
                                                                  large_values_are_attractive=False,
                                                                  edge_direction=0)
        att1z, rep1z = visualise_attractive_and_repulsive_edges(rag, edge_feats,
                                                                ignore_edges=xy_edges,
                                                                threshold=.5,
                                                                large_values_are_attractive=False,
                                                                edge_direction=2)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='raw')
        viewer.add_image(boundaries, name='boundaries')

        viewer.add_image(att1xy, name='attractive-xy')
        viewer.add_image(rep1xy, name='repuslive-xy')
        viewer.add_image(att1z, name='attractive-z')
        viewer.add_image(rep1z, name='repulsive-z')


if __name__ == '__main__':
    # visualize_edges_isbi()
    visualize_attractive_and_repulsive_edges_isbi(False)
