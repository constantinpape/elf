# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits

import numpy as np
import vigra
try:
    import hdbscan
except ImportError:
    hdbscan = None

from scipy.ndimage import shift
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA

from .features import (compute_grid_graph,
                       compute_grid_graph_affinity_features,
                       compute_grid_graph_image_features)
from .multicut import compute_edge_costs
from .mutex_watershed import mutex_watershed_clustering

#
# utils
#


def embedding_pca(embeddings, n_components=3, as_rgb=True):
    """
    """

    if as_rgb and n_components != 3:
        raise ValueError("")

    pca = PCA(n_components=n_components)
    embed_dim = embeddings.shape[0]
    shape = embeddings.shape[1:]

    embed_flat = embeddings.reshape(embed_dim, -1).T
    embed_flat = pca.fit_transform(embed_flat).T
    embed_flat = embed_flat.reshape((n_components,) + shape)

    if as_rgb:
        embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
        embed_flat = embed_flat.astype("uint8")

    return embed_flat


def _embeddings_to_probabilities(embed1, embed2, delta, embedding_axis):
    probs = (2 * delta - np.linalg.norm(embed1 - embed2, axis=embedding_axis)) / (2 * delta)
    probs = np.maximum(probs, 0) ** 2
    return probs


def edge_probabilities_from_embeddings(embeddings, segmentation, rag, delta):
    n_nodes = rag.numberOfNodes
    embed_dim = embeddings.shape[0]

    segmentation = segmentation.astype("uint32")
    mean_embeddings = np.zeros((n_nodes, embed_dim), dtype="float32")
    for cid in range(embed_dim):
        mean_embed = vigra.analysis.extractRegionFeatures(embeddings[cid],
                                                          segmentation, features=["mean"])["mean"]
        mean_embeddings[:, cid] = mean_embed

    uv_ids = rag.uvIds()
    embed_u = mean_embeddings[uv_ids[:, 0]]
    embed_v = mean_embeddings[uv_ids[:, 1]]
    edge_probabilities = 1. - _embeddings_to_probabilities(embed_u, embed_v, delta, embedding_axis=1)
    return edge_probabilities


# could probably be implemented more efficiently with shift kernels
# instead of explicit call to shift
# (or implement in C++ to save memory)
def embeddings_to_affinities(embeddings, offsets, delta, invert=False):
    """ Convert embeddings to affinities.

    Computes the affinity according to the formula
    a_ij = max((2 * delta - ||x_i - x_j||) / 2 * delta, 0) ** 2,
    where delta is the push force used in training the embeddings.
    Introduced in "Learning Dense Voxel Embeddings for 3D Neuron Reconstruction":
    https://arxiv.org/pdf/1909.09872.pdf

    Arguments:
        embeddings [np.ndarray] - the array with embeddings
        offsets [list] - the offset vectors for which to compute affinities
        delta [float] - the delta factor used in the push force when training the embeddings
        invert [bool] - whether to invert the affinites (default=False)
    """
    ndim = embeddings.ndim - 1
    if not all(len(off) == ndim for off in offsets):
        raise ValueError("Incosistent dimension of offsets and embeddings")

    n_channels = len(offsets)
    shape = embeddings.shape[1:]
    affinities = np.zeros((n_channels,) + shape, dtype="float32")

    for cid, off in enumerate(offsets):
        # we need to  shift in the other direction in order to
        # get the correct offset
        # also, we need to add a zero shift in the first axis
        shift_off = [0] + [-o for o in off]
        # we could also shift via np.pad and slicing
        shifted = shift(embeddings, shift_off, order=0, prefilter=False)
        affs = _embeddings_to_probabilities(embeddings, shifted, delta, embedding_axis=0)
        affinities[cid] = affs

    if invert:
        affinities = 1. - affinities

    return affinities


#
# density based segmentation
#


def _cluster(embeddings, clustering_alg, semantic_mask=None, remove_largest=False):
    output_shape = embeddings.shape[1:]
    # reshape (E, D, H, W) -> (E, D * H * W) and transpose -> (D * H * W, E)
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()

    result = np.zeros(flattened_embeddings.shape[0])

    if semantic_mask is not None:
        flattened_mask = semantic_mask.reshape(-1)
        assert flattened_mask.shape[0] == flattened_embeddings.shape[0]
    else:
        flattened_mask = np.ones(flattened_embeddings.shape[0])

    if flattened_mask.sum() == 0:
        # return zeros for empty masks
        return result.reshape(output_shape)

    # cluster only within the foreground mask
    clusters = clustering_alg.fit_predict(flattened_embeddings[flattened_mask == 1])
    # always increase the labels by 1 cause clustering results start from 0 and we may loose one object
    result[flattened_mask == 1] = clusters + 1

    if remove_largest:
        # set largest object to 0-label
        ids, counts = np.unique(result, return_counts=True)
        result[ids[np.argmax(counts)] == result] = 0

    return result.reshape(output_shape)


def segment_hdbscan(embeddings, min_size, eps, remove_largest, n_jobs=1):
    assert hdbscan is not None, "Needs hdbscan library"
    with threadpool_limits(limits=n_jobs):
        clustering = hdbscan.HDBSCAN(
            min_cluster_size=min_size, cluster_selection_epsilon=eps, core_dist_n_jobs=n_jobs
        )
        result = _cluster(embeddings, clustering, remove_largest=remove_largest).astype("uint64")
    return result


def segment_mean_shift(embeddings, bandwidth, n_jobs=1):
    with threadpool_limits(limits=n_jobs):
        clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=n_jobs)
        result = _cluster(embeddings, clustering).astype("uint64")
    return result


def segment_consistency(embeddings1, embeddings2, bandwidth, iou_threshold, num_anchors, skip_zero=True, n_jobs=1):
    def _iou(gt, seg):
        epsilon = 1e-5
        inter = (gt & seg).sum()
        union = (gt | seg).sum()

        iou = (inter + epsilon) / (union + epsilon)
        return iou

    with threadpool_limits(limits=n_jobs):
        clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=n_jobs)
        clusters = _cluster(embeddings1, clustering)

    for label_id in np.unique(clusters):
        if label_id == 0 and skip_zero:
            continue

        mask = clusters == label_id
        iou_table = []
        # FIXME: make it work for 3d
        y, x = np.nonzero(mask)
        for _ in range(num_anchors):
            ind = np.random.randint(len(y))
            # get random embedding anchor from emb-g
            anchor_emb = embeddings2[:, y[ind], x[ind]]
            # add necessary singleton dims
            anchor_emb = anchor_emb[:, None, None]
            # compute the instance mask from emb2
            inst_mask = np.linalg.norm(embeddings2 - anchor_emb, axis=0) < bandwidth
            iou_table.append(_iou(mask, inst_mask))
        # choose final IoU as a median
        final_iou = np.median(iou_table)

        if final_iou < iou_threshold:
            clusters[mask] = 0

    return clusters.astype("uint64")


#
# affinity based segmentation
#


def _ensure_mask_is_zero(seg, mask):
    inv_mask = ~mask
    mask_id = seg[inv_mask][0]
    if mask_id == 0:
        return seg

    seg_ids = np.unique(seg[mask])
    if 0 in seg_ids:
        seg[seg == 0] = mask_id
    seg[inv_mask] = 0

    return seg


def _get_lr_offsets(offsets):
    lr_offsets = [
        off for off in offsets if np.sum(np.abs(off)) > 1
    ]
    return lr_offsets


def _apply_mask(mask, g, weights, lr_edges, lr_weights):
    assert np.dtype(mask.dtype) == np.dtype("bool")
    node_ids = g.projectNodeIdsToPixels()
    assert node_ids.shape == mask.shape == tuple(g.shape), f"{node_ids.shape}, {mask.shape}, {g.shape}"
    masked_ids = node_ids[~mask]

    # local edges:
    # - set edges that connect masked nodes to max attractive
    # - set edges that connect masked and non-masked nodes to max repulsive
    local_edge_state = np.isin(g.uvIds(), masked_ids).sum(axis=1)
    local_masked_edges = local_edge_state == 2
    local_transition_edges = local_edge_state == 1
    weights[local_masked_edges] = 0.0
    weights[local_transition_edges] = 1.0

    # lr edges:
    # - remove edges that connect masked nodes
    # - set all edges that connect masked and non-masked nodes to max repulsive
    lr_edge_state = np.isin(lr_edges, masked_ids).sum(axis=1)
    lr_keep_edges = lr_edge_state != 2

    lr_edges, lr_weights, lr_edge_state = (lr_edges[lr_keep_edges],
                                           lr_weights[lr_keep_edges],
                                           lr_edge_state[lr_keep_edges])
    lr_transition_edges = lr_edge_state == 1
    lr_weights[lr_transition_edges] = 1.0

    return weights, lr_edges, lr_weights


# weight functions may normalize the weight values based on some statistics
# calculated for all weights. It's important to apply this weighting on a per offset channel
# basis, because long-range weights may be much larger than the short range weights.
def _process_weights(g, edges, weights, weight_function, beta,
                     offsets=None, strides=None, randomize_strides=None):

    def apply_weight_function():
        nonlocal weights
        edge_ids = g.projectEdgeIdsToPixels()
        invalid_edges = edge_ids == -1
        edge_ids[invalid_edges] = 0
        weights = weights[edge_ids]
        weights[invalid_edges] = 0
        for chan_id, weightc in enumerate(weights):
            weights[chan_id] = weight_function(weightc)
        edges, weights = compute_grid_graph_affinity_features(
            g, weights
        )
        assert len(weights) == g.numberOfEdges
        return edges, weights

    def apply_weight_function_lr():
        nonlocal weights
        edge_ids = g.projectEdgeIdsToPixelsWithOffsets(offsets)
        invalid_edges = edge_ids == -1
        edge_ids[invalid_edges] = 0
        weights = weights[edge_ids]
        weights[invalid_edges] = 0
        for chan_id, weightc in enumerate(weights):
            weights[chan_id] = weight_function(weightc)
        edges, weights = compute_grid_graph_affinity_features(
            g, weights, offsets=offsets,
            strides=strides, randomize_strides=randomize_strides
        )
        return edges, weights

    apply_weight = weight_function is not None
    if apply_weight and offsets is None:
        edges, weights = apply_weight_function()
    elif apply_weight and offsets is not None:
        edges, weights = apply_weight_function_lr()

    if beta is not None:
        weights = compute_edge_costs(weights, beta=beta)

    return edges, weights


def _embeddings_to_problem(embed, distance_type, beta=None,
                           offsets=None, strides=None, weight_function=None,
                           mask=None):
    im_shape = embed.shape[1:]
    g = compute_grid_graph(im_shape)
    _, weights = compute_grid_graph_image_features(g, embed, distance_type)
    _, weights = _process_weights(g, None, weights, weight_function, beta)
    if offsets is None:
        return g, weights

    lr_offsets = _get_lr_offsets(offsets)

    # we only compute with strides if we are not applying a weight function, otherwise
    # strides are applied later!
    strides_, randomize_ = (strides, True) if weight_function is None else (None, False)

    lr_edges, lr_weights = compute_grid_graph_image_features(
        g, embed, distance_type, offsets=lr_offsets, strides=strides_, randomize_strides=randomize_
    )

    if mask is not None:
        weights, lr_edges, lr_weights = _apply_mask(mask, g, weights, lr_edges, lr_weights)

    lr_edges, lr_weights = _process_weights(g, lr_edges, lr_weights, weight_function, beta, offsets=lr_offsets,
                                            strides=strides, randomize_strides=randomize_)
    return g, weights, lr_edges, lr_weights


# weight function based on the seung paper, using the push delta
# of the discriminative loss term.
def discriminative_loss_weight(dist, delta):
    dist = (2 * delta - dist) / (2 * delta)
    dist = 1. - np.maximum(dist, 0) ** 2
    return dist


def segment_embeddings_mws(embeddings, distance_type, offsets, bias=0.0,
                           strides=None, weight_function=None, mask=None):
    g, costs, mutex_uvs, mutex_costs = _embeddings_to_problem(
        embeddings, distance_type, beta=None,
        offsets=offsets, strides=strides,
        weight_function=weight_function,
        mask=mask
    )
    if bias > 0:
        mutex_costs += bias
    uvs = g.uvIds()
    seg = mutex_watershed_clustering(
        uvs, mutex_uvs, costs, mutex_costs
    ).reshape(embeddings.shape[1:])
    if mask is not None:
        seg = _ensure_mask_is_zero(seg, mask)
    return seg
