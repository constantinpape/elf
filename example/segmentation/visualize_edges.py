import numpy as np
import napari
import elf.segmentation.features as feats
import elf.segmentation.watershed as ws
import elf.segmentation.multicut as mc
# from elf.visualisation import visualise_attractive_and_repulsive_edges
from elf.visualisation import visualise_edges
from elf.io import open_file


# TODO we should parallelize this and put it into elf.seg.watershed as well
def make_2d_watershed(affs):
    # first, we have to make a single channel input map for the watershed,
    # which we obtain by averaging the afifnities
    boundary_input = np.mean(affs, axis=0)

    # next, we run the distance transform watershed.
    # the data is very anisotropic, so we apply the watershed in 2d
    # and stack the watershed results along z, with the appropriate id offset
    watershed = np.zeros_like(boundary_input, dtype='uint64')
    offset = 0
    for z in range(watershed.shape[0]):
        # the threshold parameter determines at which value the input map is thresholded before applying
        # the distance transform.
        # the parameter sigma_seeds determines how strong the seed map is smoothed before seeds are
        # computed via local minima. This controls the degree of over-segmentation
        wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=.5, sigma_seeds=2.)
        wsz += offset
        offset += max_id
        watershed[z] = wsz
    return boundary_input, watershed


def visualize_edges_isbi():
    data_path = '/home/pape/Work/data/isbi/isbi_test_volume.h5'  # adjust this path
    with open_file(data_path, 'r') as f:
        # load the raw data
        raw = f['raw'][:]
        # load the affinities, we only need the first 3 channels
        affs = f['affinities'][:3, :]

    boundaries, watershed = make_2d_watershed(affs)
    # compute the region adjacency graph
    rag = feats.compute_rag(watershed)

    # compute the edge costs
    # the offsets encode the pixel transition encoded by the
    # individual affinity channels. Here, we only have nearest neighbor transitions
    # offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    # costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
    costs = feats.compute_boundary_features(rag, boundaries)[:, 0]
    z_edges = feats.compute_z_edge_mask(rag, watershed)
    xy_edges = ~z_edges
    # att1xy, rep1xy = visualise_attractive_and_repulsive_edges(rag, costs,
    #                                                           ignore_edges=z_edges,
    #                                                           threshold=.5,
    #                                                           edge_direction=0)
    # att1z, rep1z = visualise_attractive_and_repulsive_edges(rag, costs,
    #                                                         ignore_edges=xy_edges,
    #                                                         threshold=.5,
    #                                                         edge_direction=2)
    xy_vals = visualise_edges(rag, costs, ignore_edges=z_edges, edge_direction=0)
    z_vals = visualise_edges(rag, costs, ignore_edges=xy_edges, edge_direction=2)

    # # z_edges = feats.compute_z_edge_mask(rag, watershed)
    # # xy_edges = np.logical_not(z_edges)
    # # edge_populations = [z_edges]
    # edge_populations = None
    # edge_sizes = feats.compute_boundary_mean_and_length(rag, boundaries)[:, 1]
    # costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes,
    #                                             edge_populations=edge_populations)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='raw')
        viewer.add_image(boundaries, name='boundaries')
        viewer.add_image(xy_vals, name='xy-edges')
        viewer.add_image(z_vals, name='z-edges')
        # viewer.add_image(att1xy, name='attractive-xy')
        # viewer.add_image(rep1xy, name='repuslive-xy')
        # viewer.add_image(att1z, name='attractive-z')
        # viewer.add_image(rep1z, name='repulsive-z')


if __name__ == '__main__':
    visualize_edges_isbi()
