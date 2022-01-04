import numpy as np
from vigra.analysis import relabelConsecutive
from affogato.segmentation import (compute_mws_clustering,
                                   compute_mws_segmentation,
                                   compute_semantic_mws_clustering,
                                   MWSGridGraph)

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

    This function changes the affinities inplace. To avoid this, pass a copy.

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


def mutex_watershed_clustering(uvs, mutex_uvs,
                               weights, mutex_weights,
                               n_nodes=None):
    """ Compute mutex watershed clustering.

    Introduced in "The Mutex Watershed and its Objective: Efficient, Parameter-Free Image Partitioning":
    https://arxiv.org/pdf/1904.12654.pdf

    Arguments:
        uvs [np.ndarray] - the uv ids for regular edges
        mutex_uvs [np.ndarray] - the uv ids for mutex edges
        weights [np.ndarray] - the weights for regular edges
        mutex_weights [np.ndarray] - the weights for mutex edges
        n_nodes [int] - the number of nodes. Will be computed from edges if not given (default: None)
    """
    if n_nodes is None:
        n_nodes = int(uvs.max()) + 1
    node_labels = compute_mws_clustering(n_nodes, uvs, mutex_uvs,
                                         weights.max() - weights,
                                         mutex_weights)
    relabelConsecutive(node_labels, out=node_labels, start_label=0, keep_zeros=False)
    return node_labels


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
        attractive_edges, attractive_weights = seed_state["attractive"]
        grid_graph.set_seed_state(attractive_edges, attractive_weights)
    grid_graph.add_attractive_seed_edges = True
    uvs, weights = grid_graph.compute_nh_and_weights(1. - np.require(affs[:ndim], requirements="C"),
                                                     offsets[:ndim])

    if seed_state is not None:
        repulsive_edges, repulsive_weights = seed_state["repulsive"]
        grid_graph.clear_seed_state()
        grid_graph.set_seed_state(repulsive_edges, repulsive_weights)
    grid_graph.add_attractive_seed_edges = False
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(np.require(affs[ndim:],
                                                                            requirements="C"),
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


def semantic_mutex_watershed_clustering(uvs, mutex_uvs, weights, mutex_weights,
                                        semantic_uts, semantic_weights,
                                        n_nodes=None, kappa=1.0):
    assert mutex_uvs.ndim == uvs.ndim == semantic_uts.ndim == 2
    assert mutex_uvs.shape[1] == uvs.shape[1] == semantic_uts.shape[1] == 2
    if n_nodes is None:
        n_nodes = int(uvs.max()) + 1
    instance_labels, semantic_labels = compute_semantic_mws_clustering(
        n_nodes, uvs, mutex_uvs, semantic_uts, weights, mutex_weights, kappa * semantic_weights
    )
    return instance_labels, semantic_labels


def _affs_to_graph(affs, offsets, strides, randomize_strides):
    shape = affs.shape[1:]
    n_nodes = np.prod(shape)
    grid_graph = MWSGridGraph(shape)

    # we set the number of attractive channels to the number of dims
    n_attr = len(shape)

    # nn uvs and weights
    nn_affs = affs[:n_attr].copy()
    nn_affs *= -1
    nn_affs += 1
    uvs, weights = grid_graph.compute_nh_and_weights(nn_affs, offsets[:n_attr])

    # mutex uvs and weights
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(np.require(affs[n_attr:], requirements="C"),
                                                                 offsets[n_attr:],
                                                                 strides=strides,
                                                                 randomize_strides=randomize_strides)
    return n_nodes, uvs, mutex_uvs, weights, mutex_weights


def _semantic_to_graph(semantic):
    shape = semantic.shape[1:]
    n_nodes = np.prod(shape)

    # semantic uts and weights
    semantic_argmax = np.argmax(semantic, axis=0)
    nodes = np.arange(n_nodes).reshape(shape)
    semantic_uts = np.stack((nodes.ravel(), semantic_argmax.ravel()), axis=1)
    semantic_weights = np.max(semantic, axis=0).flatten()

    return semantic_uts, semantic_weights


def semantic_mutex_watershed(affs, semantic_preds, offsets, strides,
                             randomize_strides=False, mask=None, kappa=1.0):
    """ Compute semantic mutex watershed segmentation. Computes instance and node labels.

    Introduced in "The Semantic Mutex Watershed for Efficient Bottom-Up Semantic Instance Segmentation":
    https://arxiv.org/pdf/1912.12717.pdf

    This function changes the affinities inplace. To avoid this, pass a copy.

    Arguments:
        affs [np.ndarray] - input affinity map
        semantic_preds [np.ndarray] - input semantic predictions
        offsets [list[list[int]]] - pixel offsets corresponding to affinity channels
        strides [list[int]] - strides used to sub-sample long range edges
        randomize_strides [bool] - randomize the strides? (default: False)
        mask [np.ndarray] - mask to exclude from segmentation (default: None)
        kappa [float] - weight factor for affinity and semantic weights (default: 1.0)
    """
    assert affs.shape[1:] == semantic_preds.shape[1:]
    shape = affs.shape[1:]

    (n_nodes, uvs, mutex_uvs,
     weights, mutex_weights) = _affs_to_graph(affs, offsets, strides, randomize_strides)
    semantic_uts, semantic_weights = _semantic_to_graph(semantic_preds)

    seg, sem = semantic_mutex_watershed_clustering(
        uvs, mutex_uvs, weights, mutex_weights,
        semantic_uts, semantic_weights,
        kappa=kappa, n_nodes=n_nodes
    )

    seg = seg.reshape(shape)
    sem = sem.reshape(shape)
    return seg, sem


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
