from typing import Dict, List, Optional, Tuple

import bioimage_cpp as bic
import numpy as np

from .blockwise_mws_impl import blockwise_mws_impl


# Note: bioimage-cpp does not provide an MWSGridGraph replacement that
# supports seed-edge injection. The helper class below reimplements the parts of
# affogato.segmentation.MWSGridGraph that elf.segmentation needs (seeded NN /
# long-range affinity emission + mask + explicit seed-state injection) on top of
# bic.graph.GridGraph2D/3D + bic.graph.features.grid_affinity_features_with_lifted.


class _MWSGridGraph:
    """@private

    Lightweight replacement for affogato's `MWSGridGraph`. Lazy-builds a
    bioimage-cpp grid graph, applies mask and seed semantics in Python, and
    emits (uvs, weights) for nearest-neighbour or long-range affinity passes.
    """

    def __init__(self, shape):
        self.shape = tuple(int(s) for s in shape)
        self.n_nodes = int(np.prod(self.shape))
        self._grid = bic.graph.grid_graph(self.shape)
        self.mask = None
        self.seeds = None
        self.add_attractive_seed_edges = False
        self._explicit_seed_edges = None  # (edges (M,2), weights (M,))

    def set_mask(self, mask: np.ndarray):
        assert mask.shape == self.shape
        self.mask = mask.astype(bool)

    def update_seeds(self, seeds: np.ndarray):
        assert seeds.shape == self.shape
        self.seeds = seeds.astype("uint64")

    def set_seed_state(self, edges: np.ndarray, weights: np.ndarray):
        edges = np.asarray(edges, dtype="uint64").reshape(-1, 2)
        weights = np.asarray(weights, dtype="float32").reshape(-1)
        assert len(edges) == len(weights)
        self._explicit_seed_edges = (edges, weights)

    def clear_seed_state(self):
        self._explicit_seed_edges = None

    def _seed_pair_edges(self, max_weight, attractive: bool):
        """Seed-induced edges.

        If ``attractive`` is True, emit a star of attractive edges between every
        pair of nodes sharing the same non-zero seed id (force them to merge).

        If ``attractive`` is False, emit one mutex (repulsive) edge between every
        pair of representative seed nodes for distinct seed ids (block them from
        merging).
        """
        if self.seeds is None:
            return np.empty((0, 2), dtype="uint64"), np.empty((0,), dtype="float32")
        flat = self.seeds.ravel()
        seed_ids = [int(s) for s in np.unique(flat) if s != 0]
        if not seed_ids:
            return np.empty((0, 2), dtype="uint64"), np.empty((0,), dtype="float32")

        edges = []
        if attractive:
            for sid in seed_ids:
                nodes = np.flatnonzero(flat == sid)
                if len(nodes) < 2:
                    continue
                anchor = nodes[0]
                for n in nodes[1:]:
                    edges.append([anchor, n])
        else:
            anchors = {sid: int(np.flatnonzero(flat == sid)[0]) for sid in seed_ids}
            for i, sid_a in enumerate(seed_ids):
                for sid_b in seed_ids[i + 1:]:
                    edges.append([anchors[sid_a], anchors[sid_b]])

        if not edges:
            return np.empty((0, 2), dtype="uint64"), np.empty((0,), dtype="float32")
        edges = np.array(edges, dtype="uint64")
        weights = np.full(len(edges), max_weight + 1.0, dtype="float32")
        return edges, weights

    def compute_nh_and_weights(
        self,
        affs: np.ndarray,
        offsets,
        strides: Optional[List[int]] = None,
        randomize_strides: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        affs = np.ascontiguousarray(affs, dtype="float32")
        local_w, local_valid, lifted_uvs, lifted_w, _ = bic.graph.features.grid_affinity_features_with_lifted(
            self._grid, affs, list(offsets),
        )
        # Combine local + lifted edges into single arrays of (uv, weight).
        local_uvs = self._grid.uv_ids()[local_valid]
        local_weights = local_w[local_valid]
        uvs = np.concatenate([local_uvs, lifted_uvs], axis=0).astype("uint64", copy=False)
        weights = np.concatenate([local_weights, lifted_w], axis=0).astype("float32", copy=False)

        # Apply mask: drop edges where both endpoints are masked, set transition
        # edges to a very repulsive weight.
        if self.mask is not None:
            node_ids = np.arange(self.n_nodes, dtype="uint64").reshape(self.shape)
            masked_ids = node_ids[~self.mask].ravel()
            in_mask = np.isin(uvs, masked_ids)
            u_masked, v_masked = in_mask[:, 0], in_mask[:, 1]
            both_masked = u_masked & v_masked
            uvs = uvs[~both_masked]
            weights = weights[~both_masked]
            one_masked = (u_masked | v_masked) & ~both_masked
            one_masked = one_masked[~both_masked]
            # Use 0 as a very repulsive transition weight on attractive passes.
            weights[one_masked] = 0.0

        # Apply strides if requested: subsample edges. We apply it to the combined
        # (local + lifted) set; NN edges with very high attractive weights still win in MWS.
        if strides is not None and np.prod(strides) > 1:
            keep = int(np.prod(strides))
            n = len(uvs)
            if randomize_strides:
                idx = np.random.choice(n, size=max(1, n // keep), replace=False)
                idx.sort()
            else:
                idx = np.arange(0, n, keep)
            uvs = uvs[idx]
            weights = weights[idx]

        # Seed handling: attractive pass emits same-seed edges, mutex pass emits
        # different-seed edges with very high weights so MWS treats them as fixed.
        max_w = float(weights.max()) if len(weights) else 1.0
        seed_edges, seed_weights = self._seed_pair_edges(max_w, attractive=self.add_attractive_seed_edges)
        if len(seed_edges) > 0:
            uvs = np.concatenate([uvs, seed_edges], axis=0)
            weights = np.concatenate([weights, seed_weights], axis=0)

        if self._explicit_seed_edges is not None:
            extra_edges, extra_weights = self._explicit_seed_edges
            uvs = np.concatenate([uvs, extra_edges], axis=0)
            weights = np.concatenate([weights, extra_weights.astype(weights.dtype, copy=False)], axis=0)

        return uvs, weights


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
    seg = bic.segmentation.mutex_watershed(
        np.ascontiguousarray(affs, dtype="float32"), list(offsets),
        number_of_attractive_channels=ndim,
        strides=list(strides), mask=mask, randomized_strides=randomize_strides,
    )
    return seg.astype("uint64", copy=False)


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

    # bic.graph.UndirectedGraph treats (u, v) and (v, u) as the same undirected edge and
    # deduplicates them. We must hand it canonical, deduplicated (u, v) pairs and align
    # the weights so len(weights) == graph.number_of_edges.
    uvs_u64 = uvs.astype("uint64", copy=False)
    canon = np.sort(uvs_u64, axis=1)
    keys = canon[:, 0].astype(np.uint64) * np.uint64(n_nodes) + canon[:, 1].astype(np.uint64)
    unique_keys, first_idx = np.unique(keys, return_index=True)
    canon_uvs = canon[first_idx]
    canon_weights = weights[first_idx]

    graph = bic.graph.UndirectedGraph.from_edges(int(n_nodes), canon_uvs)
    # The original used `weights.max() - weights` to flip the sense; bic's mutex_watershed_clustering
    # also picks higher weights first, so we do the same flip here.
    flipped = (canon_weights.max() - canon_weights).astype("float32", copy=False)
    mutex_weights = mutex_weights.astype("float32", copy=False)
    node_labels = bic.graph.mutex_watershed.mutex_watershed_clustering(
        graph, flipped, mutex_uvs.astype("uint64", copy=False), mutex_weights,
    )
    return node_labels


def compute_grid_graph(shape, mask=None, seeds=None):
    """@private
    """
    grid_graph = _MWSGridGraph(shape)
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

    See https://arxiv.org/pdf/1904.12654.pdf.
    This function changes the affinities inplace. To avoid this, pass a copy.

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
    uvs, weights = grid_graph.compute_nh_and_weights(
        1. - np.require(affs[:ndim], requirements="C"), offsets[:ndim],
    )

    if seed_state is not None:
        repulsive_edges, repulsive_weights = seed_state["repulsive"]
        grid_graph.clear_seed_state()
        grid_graph.set_seed_state(repulsive_edges, repulsive_weights)
    grid_graph.add_attractive_seed_edges = False
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(
        np.require(affs[ndim:], requirements="C"), offsets[ndim:], strides, randomize_strides,
    )

    n_nodes = grid_graph.n_nodes
    seg = mutex_watershed_clustering(uvs, mutex_uvs, weights, mutex_weights, n_nodes=n_nodes)
    # Foreground starts at 1, mask pixels are 0.
    seg = seg.astype("uint64", copy=False) + 1
    seg = seg.reshape(shape)
    if mask is not None:
        seg[np.logical_not(mask)] = 0

    if return_graph:
        return seg, grid_graph
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
        semantic_uts: The semantic labels for the nodes, shape (n_semantic, 2): (node_id, class_id).
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
    graph = bic.graph.UndirectedGraph.from_edges(int(n_nodes), uvs.astype("uint64", copy=False))
    instance_labels, semantic_labels = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph,
        weights.astype("float32", copy=False),
        mutex_uvs.astype("uint64", copy=False),
        mutex_weights.astype("float32", copy=False),
        semantic_uts.astype("uint64", copy=False),
        (kappa * semantic_weights).astype("float32", copy=False),
    )
    return instance_labels, semantic_labels


def _affs_to_graph(affs, offsets, strides, randomize_strides):
    shape = affs.shape[1:]
    n_attr = len(shape)

    grid_graph = _MWSGridGraph(shape)

    # nn uvs and weights (attractive: flip 1-x)
    nn_affs = affs[:n_attr].copy()
    nn_affs *= -1
    nn_affs += 1
    uvs, weights = grid_graph.compute_nh_and_weights(nn_affs, offsets[:n_attr])

    # mutex uvs and weights (long-range)
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(
        np.require(affs[n_attr:], requirements="C"),
        offsets[n_attr:],
        strides=strides,
        randomize_strides=randomize_strides,
    )
    return grid_graph.n_nodes, uvs, mutex_uvs, weights, mutex_weights


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

    See https://arxiv.org/pdf/1912.12717.pdf.
    This function changes the affinities inplace. To avoid this, pass a copy.

    Args:
        affs: The input affinity map.
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
