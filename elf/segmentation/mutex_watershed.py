import numpy as np
from vigra.analysis import relabelConsecutive
from affogato.segmentation import compute_mws_segmentation
from affogato.segmentation import MWSGridGraph, compute_mws_clustering

try:
    from .blockwise_mws_impl import blockwise_mws_impl
except ImportError:
    blockwise_mc_impl = None


def mutex_watershed(affs, offsets, strides,
                    randomize_strides=False, mask=None,
                    noise_level=0):
    """ Compute mutex watershed segmentation.

    Introduced in "The Mutex Watershed and its Objective: Efficient, Parameter-Free Image Partitioning":
    https://arxiv.org/pdf/1904.12654.pdf
    This function changes the affinities inplace. To avoid this, pass a copy of the affinities.

    Arguments:
        affs [np.ndarray] - input affinity map
        offsets [list[list[int]]] - pixel offsets corresponding to affinity channels
        strides [list[int]] - strides used to sub-sample long range edges
        randomize_strides [bool] - randomize the strides? (default: False)
        mask [np.ndarray] - mask to exclude from segmentation (default: None)
        noise_level [float] - sigma of noise added to affinities (default: 0)
    """
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
    """ Compute MWS grid graph.
    """
    grid_graph = MWSGridGraph(shape)
    if mask is not None:
        grid_graph.set_mask(mask)
    if seeds is not None:
        grid_graph.update_seeds(seeds)
    return grid_graph


def mutex_watershed_with_seeds(affs, offsets, seeds, strides,
                               randomize_strides=False, mask=None,
                               noise_level=0, return_graph=False,
                               seed_state=None):
    """ Compute mutex watershed segmentation with seeds.

    Introduced in "The Mutex Watershed and its Objective: Efficient, Parameter-Free Image Partitioning":
    https://arxiv.org/pdf/1904.12654.pdf
    This function changes the affinities inplace. To avoid this, pass a copy of the affinities.

    Arguments:
        affs [np.ndarray] - input affinity map
        offsets [list[list[int]]] - pixel offsets corresponding to affinity channels
        seeds [np.ndarray] - array with seed points
        strides [list[int]] - strides used to sub-sample long range edges
        randomize_strides [bool] - randomize the strides? (default: False)
        mask [np.ndarray] - mask to exclude from segmentation (default: None)
        noise_level [float] - sigma of noise added to affinities (default: 0)
        seed_state [dict] - seed state (default: None)
    """
    ndim = len(offsets[0])
    if noise_level > 0:
        affs += noise_level * np.random.rand(*affs.shape)
    affs[:ndim] *= -1
    affs[:ndim] += 1

    # compute grid graph with seeds and optional mask
    shape = affs.shape[1:]
    grid_graph = compute_grid_graph(shape, mask, seeds)

    # compute nn and mutex nh
    if seed_state is not None:
        attractive_edges, attractive_weights = seed_state['attractive']
        grid_graph.set_seed_state(attractive_edges, attractive_weights)
    grid_graph.add_attractive_seed_edges = True
    uvs, weights = grid_graph.compute_nh_and_weights(1. - np.require(affs[:ndim], requirements='C'),
                                                     offsets[:ndim])

    if seed_state is not None:
        repulsive_edges, repulsive_weights = seed_state['repulsive']
        grid_graph.clear_seed_state()
        grid_graph.set_seed_state(repulsive_edges, repulsive_weights)
    grid_graph.add_attractive_seed_edges = False
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(np.require(affs[ndim:],
                                                                            requirements='C'),
                                                                 offsets[ndim:], strides,
                                                                 randomize_strides)

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


def blockwise_mutex_watershed(affs, offsets, strides, block_shape,
                              randomize_strides=False, mask=None,
                              noise_level=0, beta0=.75, beta1=.5,
                              n_threads=None):
    """ Block-wise mutex watershed implementation.

    Solves mutex watershed in parallel for blocking of the input volume
    and then stitches block-wise segmentation with biased multicut.

    Arguments:
        affs [np.ndarray] - input affinity map
        offsets [list[list[int]]] - pixel offsets corresponding to affinity channels
        strides [list[int]] - strides used to sub-sample long range edges
        block_shape [list[int]] - block shape used for parallelizing the mws
        randomize_strides [bool] - randomize the strides? (default: False)
        mask [np.ndarray] - mask to exclude from segmentation (default: None)
        noise_level [float] - sigma of noise added to affinities (default: 0)
        beta0 [float] - boundary bias for the inner block edges (default: 0.75)
        beta1 [float] - boundary bias for the between block edges (default: 0.5)
        n_threads [int] - number of threads (default: None)
    """
    if blockwise_mws_impl is None:
        raise RuntimeError("Cannot run blockwise mutex watershed, probably nifty is misssing.")
    assert len(affs) == len(offsets)
    return blockwise_mws_impl(affs, offsets, strides, block_shape,
                              randomize_strides, mask=mask,
                              beta0=beta0, beta1=beta1,
                              noise_level=noise_level, n_threads=n_threads)
