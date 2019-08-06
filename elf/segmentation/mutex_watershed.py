import numpy as np
from vigra.analysis import relabelConsecutive
from affogato.segmentation import compute_mws_segmentation
from affogato.segmentation import MWSGridGraph, compute_mws_clustering


def mutex_watershed(affs, offsets, strides,
                    randomize_strides=False, mask=None,
                    noise_level=0):
    assert compute_mws_segmentation is not None, "Need affogato for mutex watershed"
    ndim = len(offsets[0])
    if noise_level > 0:
        affs += noise_level * np.random.rand(*affs.shape)
    affs[:ndim] *= -1
    affs[:ndim] += 1
    seg = compute_mws_segmentation(affs, offsets,
                                   number_of_attractive_channels=ndim,
                                   strides=strides, mask=mask,
                                   randomize_strides=randomize_strides)
    relabelConsecutive(seg, out=seg, start_label=1, keep_zeros=mask is not None)
    return seg


def compute_grid_graph(shape, mask=None, seeds=None):
    grid_graph = MWSGridGraph(shape)
    if mask is not None:
        grid_graph.set_mask(mask)
    if seeds is not None:
        grid_graph.set_seeds(seeds)
    return grid_graph


def mutex_watershed_with_seeds(affs, offsets, seeds, strides,
                               randomize_strides=False, mask=None,
                               noise_level=0, return_graph=False,
                               seed_state=None):
    assert compute_mws_segmentation is not None, "Need affogato for mutex watershed"
    ndim = len(offsets[0])
    if noise_level > 0:
        affs += noise_level * np.random.rand(*affs.shape)
    affs[:ndim] *= -1
    affs[:ndim] += 1

    # compute grid graph with seeds and optional mask
    shape = affs.shape[1:]
    grid_graph = compute_grid_graph(shape, mask, seeds)

    # compute nn and mutex nh
    grid_graph.intra_seed_weight = 1  # set intra-seed weight to maximal attractive
    if seed_state is not None:
        attractive_edges, attractive_weights = seed_state['attractive']
        grid_graph.set_seed_state(attractive_edges, attractive_weights)
    uvs, weights = grid_graph.compute_nh_and_weights(np.require(affs[:ndim], requirements='C'),
                                                     offsets[:ndim])

    grid_graph.intra_seed_weight = 0  # set intral-seed weight to minimal repulsive
    if seed_state is not None:
        repulsive_edges, repulsive_weights = seed_state['repulsive']
        grid_graph.clear_seed_state()
        grid_graph.set_seed_state(repulsive_edges, repulsive_weights)
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(np.require(affs[ndim:],
                                                                            requirements='C'),
                                                                 offsets[ndim:], strides, randomize_strides)

    # compute the segmentation
    n_nodes = grid_graph.n_nodes
    seg = compute_mws_clustering(n_nodes, uvs, mutex_uvs, weights, mutex_weights)
    relabelConsecutive(seg, out=seg, start_label=1, keep_zeros=mask is not None)
    seg = seg.reshape(shape)
    if mask is not None:
        seg[np.logical_not(mask)] = 0

    if return_graph:
        return seg, grid_graph
    else:
        return seg
