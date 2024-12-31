from typing import Dict, List, Optional, Tuple

import numpy as np
from vigra.analysis import relabelConsecutive
from affogato.segmentation import (compute_mws_clustering,
                                   compute_mws_segmentation,
                                   compute_semantic_mws_clustering,
                                   MWSGridGraph)
from .blockwise_mws_impl import blockwise_mws_impl


def mutex_watershed(
    affs: np.ndarray,
    offsets: List[List[int]],
    strides: List[int],
    randomize_strides: bool = False,
    mask: Optional[np.ndarray] = None,
    noise_level: float = 0.0
) -> np.ndarray:
    """Compute mutex watershed segmentation.

    Introduced in "The Mutex Watershed and its Objective: Efficient, Parameter-Free Image Partitioning":
    https://arxiv.org/pdf/1904.12654.pdf

    This function changes the affinities inplace. To avoid this, pass a copy.

    Args:
        affs: The input affinity map.
        offsets: The pixel offsets corresponding to the affinity channels.
        strides: The strides used to sub-sample long range edges.
        randomize_strides: Whether to randomize the strides.
        mask: Mask to exclude from segmentation.
        noise_level: Sigma of noise added to affinities.

    Returns:
        The segmentation.
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


def mutex_watershed_clustering(
    uvs: np.ndarray,
    mutex_uvs: np.ndarray,
    weights: np.ndarray,
    mutex_weights: np.ndarray,
    n_nodes: Optional[int] = None
) -> np.ndarray:
    """Compute mutex watershed clustering.

    Introduced in "The Mutex Watershed and its Objective: Efficient, Parameter-Free Image Partitioning":
    https://arxiv.org/pdf/1904.12654.pdf

    Args:
        uvs: The uv ids for regular edges.
        mutex_uvs: The uv ids for mutex edges.
        weights: The weights for regular edges.
        mutex_weights: The weights for mutex edges.
        n_nodes: The number of nodes. Will be computed from uvs if not given.

    Returns:
        The node labeling.
    """
    if n_nodes is None:
        n_nodes = int(uvs.max()) + 1
    node_labels = compute_mws_clustering(n_nodes, uvs, mutex_uvs, weights.max() - weights, mutex_weights)
    relabelConsecutive(node_labels, out=node_labels, start_label=0, keep_zeros=False)
    return node_labels


def compute_grid_graph(shape, mask=None, seeds=None):
    """@private
    """
    grid_graph = MWSGridGraph(shape)
    if mask is not None:
        grid_graph.set_mask(mask)
    if seeds is not None:
        grid_graph.update_seeds(seeds)
    return grid_graph


def mutex_watershed_with_seeds(
    affs: np.ndarray,
    offsets: List[List[int]],
    seeds: np.ndarray,
    strides: List[int],
    randomize_strides: bool = False,
    mask: Optional[np.ndarray] = None,
    noise_level: float = 0.0,
    return_graph: bool = False,
    seed_state: Optional[Dict] = None,
) -> np.ndarray:
    """Compute mutex watershed segmentation with seeds.

    Introduced in "The Mutex Watershed and its Objective: Efficient, Parameter-Free Image Partitioning":
    https://arxiv.org/pdf/1904.12654.pdf

    This function changes the affinities inplace. To avoid this, pass a copy of the affinities.

    Args:
        affs: The input affinity map.
        offsets: The pixel offsets corresponding to affinity channels.
        seeds: The array with seed points.
        strides: The strides used to sub-sample long range edges.
        randomize_strides: Whether to randomize the strides.
        mask: The mask to exclude from segmentation.
        noise_level: The sigma of noise added to affinities.
        seed_state: The seed state.

    Returns:
        The segmentation.
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
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(np.require(affs[ndim:], requirements="C"),
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


def semantic_mutex_watershed_clustering(
    uvs: np.ndarray,
    mutex_uvs: np.ndarray,
    weights: np.ndarray,
    mutex_weights: np.ndarray,
    semantic_uts: np.ndarray,
    semantic_weights: np.ndarray,
    n_nodes: Optional[int] = None,
    kappa: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute semantic mutex watershed clustering.

    Introduced in "The semantic mutex watershed for efficient bottom-up semantic instance segmentation":
    https://arxiv.org/pdf/1912.12717.pdf

    Args:
        uvs: The uv ids for regular edges.
        mutex_uvs: The uv ids for mutex edges.
        weights: The weights for regular edges.
        mutex_weights: The weights for mutex edges.
        semantic_uts: The semantic labels for the nodes.
        semantic_weights: The semantic weights for the nodes.
        n_nodes: The number of nodes. Will be computed from uvs if not given.
        kappa: The strength of the semantic weights compared to the mutex weights.

    Returns:
        The instance node labeling.
        The semantic node labeling.
    """
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


def semantic_mutex_watershed(
    affs: np.ndarray,
    semantic_preds: np.ndarray,
    offsets: List[List[int]],
    strides: List[int],
    randomize_strides: bool = False,
    mask: Optional[np.ndarray] = None,
    kappa: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute semantic mutex watershed segmentation. Computes instance and node labels.

    Introduced in "The Semantic Mutex Watershed for Efficient Bottom-Up Semantic Instance Segmentation":
    https://arxiv.org/pdf/1912.12717.pdf

    This function changes the affinities inplace. To avoid this, pass a copy.

    Args:
        affs: The innput affinity map.
        semantic_preds: The input semantic predictions.
        offsets: The pixel offsets corresponding to affinity channels.
        strides: The strides used to sub-sample long range edges.
        randomize_strides: Whether to randomize the strides.
        mask: Mask to exclude from segmentation.
        kappa: Weight factor for affinity and semantic weights.

    Returns:
        The instance segmentation.
        The semantic segmentation.
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


def blockwise_mutex_watershed(
    affs: np.ndarray,
    offsets: List[List[int]],
    strides: List[int],
    block_shape: Tuple[int, ...],
    randomize_strides: bool = False,
    mask: Optional[np.ndarray] = None,
    noise_level: float = 0.0,
    beta0: float = 0.75,
    beta1: float = 0.5,
    n_threads: Optional[int] = None
) -> np.ndarray:
    """Compute block-wise mutex watershed segmentation.

    Solves mutex watershed in parallel for blocking of the input volume
    and then stitches block-wise segmentation with biased multicut.

    Args:
        affs: The input affinity map.
        offsets: The pixel offsets corresponding to affinity channels.
        strides: The strides used to sub-sample long range edges.
        block_shape: The block shape used for parallelizing the MWS.
        randomize_strides: Whether to randomize the strides.
        mask: The mask to exclude from segmentation.
        noise_level: The sigma of noise added to affinities.
        beta0: The boundary bias for the inner block edges.
        beta1: The boundary bias for the in-between block edges.
        n_threads: The number of threads for parallelization.

    Returns:
        The instance segmentation.
    """
    if blockwise_mws_impl is None:
        raise RuntimeError("Cannot run blockwise mutex watershed, probably nifty is misssing.")
    assert len(affs) == len(offsets)
    return blockwise_mws_impl(affs, offsets, strides, block_shape,
                              randomize_strides, mask=mask,
                              beta0=beta0, beta1=beta1,
                              noise_level=noise_level, n_threads=n_threads)
