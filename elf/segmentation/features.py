import multiprocessing
from concurrent import futures
from typing import Dict, List, Optional, Tuple

import numpy as np
import vigra
import nifty
import nifty.graph.rag as nrag
import nifty.ground_truth as ngt
try:
    import nifty.distributed as ndist
except ImportError:
    ndist = None

try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff

from tqdm import tqdm
from .multicut import transform_probabilities_to_costs


#
# Region Adjacency Graph and Features
#

def compute_rag(segmentation: np.ndarray, n_labels: Optional[int] = None, n_threads: Optional[int] = None):
    """Compute region adjacency graph of segmentation.

    Args:
        segmentation: The segmentation.
        n_labels: The number of labels in the segmentation. If None, will be computed from the data.
        n_threads: The number of threads used, set to cpu count by default.

    Returns:
        The region adjacency graph.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    n_labels = int(segmentation.max()) + 1 if n_labels is None else n_labels
    rag = nrag.gridRag(segmentation, numberOfLabels=n_labels, numberOfThreads=n_threads)
    return rag


def compute_boundary_features(
    rag, boundary_map: np.ndarray, min_value: float = 0.0, max_value: float = 1.0, n_threads: Optional[int] = None
) -> np.ndarray:
    """Compute edge features from boundary map.

    Args:
        rag: The region adjacency graph.
        boundary_map:The boundary map.
        min_value: The minimum value used in accumulation.
        max_value: The maximum value used in accumulation.
        n_threads: The number of threads used, set to cpu count by default.

    Returns:
        The edge features.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if tuple(rag.shape) != boundary_map.shape:
        raise ValueError("Incompatible shapes: %s, %s" % (str(rag.shape), str(boundary_map.shape)))
    features = nrag.accumulateEdgeStandartFeatures(
        rag, boundary_map, min_value, max_value, numberOfThreads=n_threads
    )
    return features


def compute_affinity_features(
    rag,
    affinity_map: np.ndarray,
    offsets: List[List[int]],
    min_value: float = 0.0,
    max_value: float = 1.0,
    n_threads: Optional[int] = None
) -> np.ndarray:
    """Compute edge features from affinity map.

    Args:
        rag: The region adjacency graph.
        affinity_map: The affinity map.
        min_value: The minimum value used in accumulation.
        max_value: The maximum value used in accumulation.
        n_threads: The umber of threads used, set to cpu count by default.

    Returns:
        The edge features.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if tuple(rag.shape) != affinity_map.shape[1:]:
        raise ValueError("Incompatible shapes: %s, %s" % (str(rag.shape), str(affinity_map.shape[1:])))
    if len(offsets) != affinity_map.shape[0]:
        raise ValueError("Incompatible number of channels and offsets: %i, %i" % (len(offsets),
                                                                                  affinity_map.shape[0]))
    features = nrag.accumulateAffinityStandartFeatures(
        rag, affinity_map, offsets, min_value, max_value, numberOfThreads=n_threads
    )
    return features


def compute_boundary_mean_and_length(rag, input_: np.ndarray, n_threads: Optional[int] = None) -> np.ndarray:
    """Compute mean value and length of boundaries.

    Args:
        rag: The region adjacency graph.
        input_: The input map.
        n_threads: The number of threads used, set to cpu count by default.

    Returns:
        The edge features.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if tuple(rag.shape) != input_.shape:
        raise ValueError("Incompatible shapes: %s, %s" % (str(rag.shape), str(input_.shape)))
    features = nrag.accumulateEdgeMeanAndLength(rag, input_, numberOfThreads=n_threads)
    return features


# TODO generalize and move to elf.features.parallel
def _filter_2d(input_, filter_name, sigma, n_threads):
    filter_fu = getattr(ff, filter_name)

    def _fz(inp):
        response = filter_fu(inp, sigma)
        # we add a channel last axis for 2d filter responses
        if response.ndim == 2:
            response = response[None, ..., None]
        elif response.ndim == 3:
            response = response[None]
        else:
            raise RuntimeError("Invalid filter response")
        return response

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_fz, input_[z]) for z in range(input_.shape[0])]
        response = [t.result() for t in tasks]

    response = np.concatenate(response, axis=0)
    return response


def compute_boundary_features_with_filters(
    rag,
    input_: np.ndarray,
    apply_2d: bool = False,
    n_threads: Optional[int] = None,
    filters: Dict[str, List[float]] = {"gaussianSmoothing": [1.6, 4.2, 8.3],
                                       "laplacianOfGaussian": [1.6, 4.2, 8.3],
                                       "hessianOfGaussianEigenvalues": [1.6, 4.2, 8.3]}
) -> np.ndarray:
    """Compute boundary features accumulated over filter responses on input.

    Args:
        rag: The region adjacency graph.
        input_: The input data.
        apply_2d: Whether to apply the filters in 2d for 3d input data.
        n_threads: The number of threads.
        filters: The filters to apply, expects a dictionary mapping filter names to sigma values.

    Returns:
        The edge filters.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    features = []

    # apply 2d: we compute filters and derived features in parallel per filter
    if apply_2d:

        def _compute_2d(filter_name, sigma):
            response = _filter_2d(input_, filter_name, sigma, n_threads)
            assert response.ndim == 4
            n_channels = response.shape[-1]
            features = []
            for chan in range(n_channels):
                chan_data = response[..., chan]
                feats = compute_boundary_features(rag, chan_data,
                                                  chan_data.min(), chan_data.max(), n_threads)
                features.append(feats)

            features = np.concatenate(features, axis=1)
            assert len(features) == rag.numberOfEdges
            return features

        features = [_compute_2d(filter_name, sigma)
                    for filter_name, sigmas in filters.items() for sigma in sigmas]

    # apply 3d: we parallelize over the whole filter + feature computation
    # this can be very memory intensive, and it would be better to parallelize inside
    # of the loop, but 3d parallel filters in elf.parallel.filters are not working properly yet
    else:

        def _compute_3d(filter_name, sigma):
            filter_fu = getattr(ff, filter_name)
            response = filter_fu(input_, sigma)
            if response.ndim == 3:
                response = response[..., None]

            n_channels = response.shape[-1]
            features = []

            for chan in range(n_channels):
                chan_data = response[..., chan]
                feats = compute_boundary_features(rag, chan_data,
                                                  chan_data.min(), chan_data.max(),
                                                  n_threads=1)
                features.append(feats)
            features = np.concatenate(features, axis=1)
            assert len(features) == rag.numberOfEdges, f"{len(features), {rag.numberOfEdges}}"
            return features

        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(_compute_3d, filter_name, sigma)
                     for filter_name, sigmas in filters.items() for sigma in sigmas]
            features = [t.result() for t in tasks]

    features = np.concatenate(features, axis=1)
    assert len(features) == rag.numberOfEdges
    return features


def compute_region_features(
    uv_ids: np.ndarray,
    input_map: np.ndarray,
    segmentation: np.ndarray,
    n_threads: Optional[int] = None
) -> np.ndarray:
    """Compute edge features from an input map accumulated over segmentation and mapped to edges.

    Args:
        uv_ids: The edge uv ids.
        input_: The input data.
        segmentation: The segmentation.
        n_threads: The number of threads used, set to cpu count by default.

    Returns:
        The edge features.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads

    # compute the node features
    stat_feature_names = ["Count", "Kurtosis", "Maximum", "Minimum", "Quantiles",
                          "RegionRadii", "Skewness", "Sum", "Variance"]
    coord_feature_names = ["Weighted<RegionCenter>", "RegionCenter"]
    feature_names = stat_feature_names + coord_feature_names
    node_features = vigra.analysis.extractRegionFeatures(input_map, segmentation,
                                                         features=feature_names)

    # get the image statistics based features, that are combined via [min, max, sum, absdiff]
    stat_features = [node_features[fname] for fname in stat_feature_names]
    stat_features = np.concatenate([feat[:, None] if feat.ndim == 1 else feat
                                    for feat in stat_features], axis=1)

    # get the coordinate based features, that are combined via euclidean distance
    coord_features = [node_features[fname] for fname in coord_feature_names]
    coord_features = np.concatenate([feat[:, None] if feat.ndim == 1 else feat
                                     for feat in coord_features], axis=1)

    u, v = uv_ids[:, 0], uv_ids[:, 1]

    # combine the stat features for all edges
    feats_u, feats_v = stat_features[u], stat_features[v]
    features = [np.minimum(feats_u, feats_v), np.maximum(feats_u, feats_v),
                np.abs(feats_u - feats_v), feats_u + feats_v]

    # combine the coord features for all edges
    feats_u, feats_v = coord_features[u], coord_features[v]
    features.append((feats_u - feats_v) ** 2)

    features = np.nan_to_num(np.concatenate(features, axis=1))
    assert len(features) == len(uv_ids)
    return features


#
# Grid Graph and Features
#

def compute_grid_graph(shape: Tuple[int, ...]):
    """Compute grid graph for the given shape.

    Args:
        shape: The shape of the data.

    Returns:
        The grid graph.
    """
    grid_graph = nifty.graph.undirectedGridGraph(shape)
    return grid_graph


def compute_grid_graph_image_features(
    grid_graph,
    image: np.ndarray,
    mode: str,
    offsets: Optional[List[List[int]]] = None,
    strides: Optional[List[int]] = None,
    randomize_strides: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute edge features for image for the given grid_graph.

    Args:
        grid_graph: The grid graph
        image: The image, from which the features will be derived.
        mode: Feature accumulation method.
        offsets: The offsets, which correspond to the affinity channels.
            If none are given, the affinites for the nearest neighbor transitions are used.
        strides: The strides used to subsample edges that are computed from offsets.
        randomize_strides: Whether to subsample randomly instead of using regular strides.

    Returns:
        The uv ids of the edges.
        The edge features.
    """
    gndim = len(grid_graph.shape)

    if image.ndim == gndim:
        if offsets is not None:
            raise NotImplementedError
        modes = ("l1", "l2", "min", "max", "sum", "prod", "interpixel")
        if mode not in modes:
            raise ValueError(f"Invalid feature mode {mode}, expect one of {modes}")
        features = grid_graph.imageToEdgeMap(image, mode)
        edges = grid_graph.uvIds()

    elif image.ndim == gndim + 1:
        modes = ("l1", "l2", "cosine")
        if mode not in modes:
            raise ValueError(f"Invalid feature mode {mode}, expect one of {modes}")

        if offsets is None:
            features = grid_graph.imageWithChannelsToEdgeMap(image, mode)
            edges = grid_graph.uvIds()
        else:
            (n_edges,
             edges,
             features) = grid_graph.imageWithChannelsToEdgeMapWithOffsets(image, mode,
                                                                          offsets=offsets,
                                                                          strides=strides,
                                                                          randomize_strides=randomize_strides)
            edges, features = edges[:n_edges], features[:n_edges]

    else:
        msg = f"Invalid image dimension {image.ndim}, expect one of {gndim} or {gndim + 1}"
        raise ValueError(msg)

    return edges, features


def compute_grid_graph_affinity_features(
    grid_graph,
    affinities: np.ndarray,
    offsets: Optional[List[List[int]]] = None,
    strides: Optional[List[int]] = None,
    mask: Optional[np.ndarray] = None,
    randomize_strides: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute edge features from affinities for the given grid graph.

    Args:
        grid_graph: The grid graph
        affinities: The affinity map.
        offsets: The offsets, which correspond to the affinity channels.
            If none are given, the affinites for the nearest neighbor transitions are used.
        strides: The strides used to subsample edges that are computed from offsets.
        mask: Mask to exclude from the edge and feature computation.
        randomize_strides: Whether to subsample randomly instead of using regular strides.

    Returns:
        The uv ids of the edges.
        The edge features.
    """
    gndim = len(grid_graph.shape)
    if affinities.ndim != gndim + 1:
        raise ValueError

    if offsets is None:
        assert affinities.shape[0] == gndim
        assert strides is None
        assert mask is None
        features = grid_graph.affinitiesToEdgeMap(affinities)
        edges = grid_graph.uvIds()
    elif mask is not None:
        assert strides is None and not randomize_strides, "Strides and mask cannot be used at the same time"
        n_edges, edges, features = grid_graph.affinitiesToEdgeMapWithMask(affinities,
                                                                          offsets=offsets,
                                                                          mask=mask)
        edges, features = edges[:n_edges], features[:n_edges]
    else:
        n_edges, edges, features = grid_graph.affinitiesToEdgeMapWithOffsets(affinities,
                                                                             offsets=offsets,
                                                                             strides=strides,
                                                                             randomize_strides=randomize_strides)
        edges, features = edges[:n_edges], features[:n_edges]

    return edges, features


def apply_mask_to_grid_graph_weights(
    grid_graph,
    mask: np.ndarray,
    weights: np.ndarray,
    masked_edge_weight: float = 0.0,
    transition_edge_weight: float = 1.0,
) -> np.ndarray:
    """Mask edges in grid graph.

    Set the weights derived from a grid graph to a fixed value, for edges that connect masked nodes
    and edges that connect masked and unmasked nodes.

    Args:
        grid_graph: The grid graph.
        mask: The binary mask, foreground (=non-masked) is True.
        weights: The edge weights.
        masked_edge_weight: The value for edges that connect two masked nodes.
        transition_edge_weight: The value for edges that connect a masked with a non-masked node.

    Returns:
        The masked edge weights.
    """
    assert np.dtype(mask.dtype) == np.dtype("bool")
    node_ids = grid_graph.projectNodeIdsToPixels()
    assert node_ids.shape == mask.shape == tuple(grid_graph.shape), \
        f"{node_ids.shape}, {mask.shape}, {grid_graph.shape}"
    masked_ids = node_ids[~mask]

    edges = grid_graph.uvIds()
    assert len(edges) == len(weights)
    edge_state = np.isin(edges, masked_ids).sum(axis=1)
    masked_edges = edge_state == 2
    transition_edges = edge_state == 1
    weights[masked_edges] = masked_edge_weight
    weights[transition_edges] = transition_edge_weight
    return weights


def apply_mask_to_grid_graph_edges_and_weights(
    grid_graph, mask: np.ndarray, edges: np.ndarray, weights: np.ndarray, transition_edge_weight: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove uv ids that connect masked nodes and set weights that connect masked to non-masked nodes to a fixed value.

    Args:
        grid_graph: The grid graph.
        mask: The binary mask, foreground (=non-masked) is True.
        edges: The edges (uv-ids).
        weights: The edge weights.
        transition_edge_weight: The value for edges that connect a masked with a non-masked node.

    Returns:
        The edge uv-ids.
        The edge weights.
    """
    assert np.dtype(mask.dtype) == np.dtype("bool")
    node_ids = grid_graph.projectNodeIdsToPixels()
    assert node_ids.shape == mask.shape == tuple(grid_graph.shape), \
        f"{node_ids.shape}, {mask.shape}, {grid_graph.shape}"
    masked_ids = node_ids[~mask]

    edge_state = np.isin(edges, masked_ids).sum(axis=1)
    keep_edges = edge_state != 2

    edges, weights, edge_state = edges[keep_edges], weights[keep_edges], edge_state[keep_edges]
    transition_edges = edge_state == 1
    weights[transition_edges] = transition_edge_weight

    return edges, weights


#
# Lifted Features
#

def lifted_edges_from_graph_neighborhood(graph, max_graph_distance):
    """@private
    """
    if max_graph_distance < 2:
        raise ValueError(f"Graph distance must be greater equal 2, got {max_graph_distance}")
    if isinstance(graph, nifty.graph.UndirectedGraph):
        objective = nifty.graph.opt.lifted_multicut.liftedMulticutObjective(graph)
    else:
        graph_ = nifty.graph.undirectedGraph(graph.numberOfNodes)
        graph_.insertEdges(graph.uvIds())
        objective = nifty.graph.opt.lifted_multicut.liftedMulticutObjective(graph_)
    objective.insertLiftedEdgesBfs(max_graph_distance)
    lifted_uvs = objective.liftedUvIds()
    return lifted_uvs


def feats_to_costs_default(lifted_labels, lifted_features):
    """@private
    """
    # we assume that we only have different classes for a given lifted
    # edge here (mode = "different") and then set all edges to be repulsive

    # the higher the class probability, the more repulsive the edges should be,
    # so we just multiply both probabilities
    lifted_costs = lifted_features[:, 0] * lifted_features[:, 1]
    lifted_costs = transform_probabilities_to_costs(lifted_costs)
    return lifted_costs


def lifted_problem_from_probabilities(
    rag,
    watershed: np.ndarray,
    input_maps: List[np.ndarray],
    assignment_threshold: float,
    graph_depth: int,
    feats_to_costs: callable = feats_to_costs_default,
    mode: str = "different",
    n_threads: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lifted problem from probability maps by mapping them to superpixels.

    Example: compute a lifted problem from two attributions (axon, dendrite) that induce
    repulsive edges between different attributions. The construction of lifted eges and
    features can be customized using the `feats_to_costs` and `mode` arguments.
    ```
    lifted_uvs, lifted_costs = lifted_problem_from_probabilties(
       rag, superpixels,
       input_maps=[
         axon_probabilities,  # probabilty map for axon attribution
         dendrite_probabilities  # probability map for dendrite attributtion
       ],
       assignment_threshold=0.6,  # probability threshold to assign superpixels to a class
       graph_depth=10,  # the max. graph depth along which lifted edges are introduced
    )
    ```

    Args:
        rag: The region adjacency graph.
        watershed: The watershed over-segmentation.
        input_maps: List of probability maps. Each map must have the same shape as the watersheds
            and each map is treated as the probability to correspond to a different class.
        assignment_threshold: Minimal expression level to assign a class to a graph node (= watershed segment).
        graph_depth: Maximal graph depth up to which lifted edges will be included.
        feats_to_costs: Function to calculate the lifted costs from the class assignment probabilities.
            The input to the function are `lifted_labels`, which stores the two classes assigned to a lifted edge,
            and `lifted_features`, which stores the two assignment probabilities.
        mode: The mode for insertion of lifted edges. One of:
            "all" - lifted edges will be inserted in between all nodes with attribution.
            "different" - lifted edges will only be inserted in between nodes attributed to different classes.
            "same" - lifted edges will only be inserted in between nodes attribted to the same class.
        n_threads: The number of threads used for the calculation.

    Returns:
        The lifted uv ids (= superpixel ids connected by the lifted edge).
        The lifted costs (= cost associated with each lifted edge).
    """
    assert ndist is not None, "Need nifty.distributed package"

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    # validate inputs
    assert isinstance(input_maps, (list, tuple))
    assert all(isinstance(inp, np.ndarray) for inp in input_maps)
    shape = watershed.shape
    assert all(inp.shape == shape for inp in input_maps)

    # map the probability maps to superpixels - we only map to superpixels which
    # have a larger mean expression than `assignment_threshold`

    # TODO handle the dtype conversion for vigra gracefully somehow ...
    # think about supporting uint8 input and normalizing

    # TODO how do we handle cases where the same superpixel is mapped to
    # more than one class ?

    n_nodes = int(watershed.max()) + 1
    node_labels = np.zeros(n_nodes, dtype="uint64")
    node_features = np.zeros(n_nodes, dtype="float32")
    # TODO we could allow for more features that could then be used for the cost estimation
    for class_id, inp in enumerate(input_maps):
        mean_prob = vigra.analysis.extractRegionFeatures(inp, watershed, features=["mean"])["mean"]
        # we can in principle map multiple classes here, and right now will just override
        class_mask = mean_prob > assignment_threshold
        node_labels[class_mask] = class_id
        node_features[class_mask] = mean_prob[class_mask]

    # find all lifted edges up to the graph depth between mapped nodes
    # NOTE we need to convert to the different graph type for now, but
    # it would be nice to support all nifty graphs at some type
    uv_ids = rag.uvIds()
    g_temp = ndist.Graph(uv_ids)

    lifted_uvs = ndist.liftedNeighborhoodFromNodeLabels(g_temp, node_labels, graph_depth, mode=mode,
                                                        numberOfThreads=n_threads, ignoreLabel=0)
    lifted_labels = node_labels[lifted_uvs]
    lifted_features = node_features[lifted_uvs]

    lifted_costs = feats_to_costs(lifted_labels, lifted_features)
    return lifted_uvs, lifted_costs


# TODO support setting costs proportional to overlaps
def lifted_problem_from_segmentation(
    rag,
    watershed: np.ndarray,
    input_segmentation: np.ndarray,
    overlap_threshold: float,
    graph_depth: int,
    same_segment_cost: float,
    different_segment_cost: float,
    mode: str = "all",
    n_threads: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lifted problem from segmentation by mapping segments to superpixels.

    Args:
        rag: The region adjacency graph.
        watershed: The watershed over-segmentation.
        input_segmentation: The segmentation used to determine node attribution.
        overlap_threshold: The minimal overlap to assign a segment id to node.
        graph_depth: The maximal graph depth up to which lifted edges will be included
        same_segment_cost: The cost for edges between nodes with same segment id attribution.
        different_segment_cost: The cost for edges between nodes with different segment id attribution.
        mode: The mode for insertion of lifted edges. One of:
            "all" - lifted edges will be inserted in between all nodes with attribution.
            "different" - lifted edges will only be inserted in between nodes attributed to different classes.
            "same" - lifted edges will only be inserted in between nodes attribted to the same class.
        n_threads: The number of threads used for the calculation.

    Returns:
        The lifted uv ids (= superpixel ids connected by the lifted edge).
        The lifted costs (= cost associated with each lifted edge).
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    assert input_segmentation.shape == watershed.shape

    # compute the overlaps
    ovlp_comp = ngt.overlap(watershed, input_segmentation)
    ws_ids = np.unique(watershed)
    n_labels = int(ws_ids[-1]) + 1
    assert n_labels == rag.numberOfNodes, "%i, %i" % (n_labels, rag.numberOfNodes)

    # initialise the arrays for node labels, to be
    # dense in the watershed id space (even if some ws-ids are not present)
    node_labels = np.zeros(n_labels, dtype="uint64")

    # extract the overlap values and node labels from the overlap
    # computation results
    overlaps = [ovlp_comp.overlapArraysNormalized(ws_id, sorted=False)
                for ws_id in ws_ids]
    node_label_vals = np.array([ovlp[0][0] for ovlp in overlaps])
    overlap_values = np.array([ovlp[1][0] for ovlp in overlaps])
    node_label_vals[overlap_values < overlap_threshold] = 0
    assert len(node_label_vals) == len(ws_ids)
    node_labels[ws_ids] = node_label_vals

    # find all lifted edges up to the graph depth between mapped nodes
    # NOTE we need to convert to the different graph type for now, but
    # it would be nice to support all nifty graphs at some type
    uv_ids = rag.uvIds()
    g_temp = ndist.Graph(uv_ids)

    lifted_uvs = ndist.liftedNeighborhoodFromNodeLabels(g_temp, node_labels, graph_depth, mode=mode,
                                                        numberOfThreads=n_threads, ignoreLabel=0)
    # make sure that the lifted uv ids are in range of the node labels
    assert lifted_uvs.max() < rag.numberOfNodes, "%i, %i" % (int(lifted_uvs.max()),
                                                             rag.numberOfNodes)
    lifted_labels = node_labels[lifted_uvs]
    lifted_costs = np.zeros(len(lifted_labels), dtype="float64")

    same_mask = lifted_labels[:, 0] == lifted_labels[:, 1]
    lifted_costs[same_mask] = same_segment_cost
    lifted_costs[~same_mask] = different_segment_cost

    return lifted_uvs, lifted_costs


#
# Misc
#

def get_stitch_edges(
    rag,
    seg: np.ndarray,
    block_shape: Tuple[int, ...],
    n_threads: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    """Get the edges between blocks.

    Args:
        rag: The region adjacency graph.
        seg: The segmentation underlying the rag.
        block_shape: The shape of the blocking.
        n_threads: The number of threads used for the calculation.
        verbose: Whether to be verbose.

    Returns:
        The edge mask indicating edges between blocks.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    ndim = seg.ndim
    blocking = nifty.tools.blocking([0] * ndim, seg.shape, block_shape)

    def find_stitch_edges(block_id):
        stitch_edges = []
        block = blocking.getBlock(block_id)
        for axis in range(ndim):
            if blocking.getNeighborId(block_id, axis, True) == -1:
                continue
            face_a = tuple(
                beg if d == axis else slice(beg, end)
                for d, beg, end in zip(range(ndim), block.begin, block.end)
            )
            face_b = tuple(
                beg - 1 if d == axis else slice(beg, end)
                for d, beg, end in zip(range(ndim), block.begin, block.end)
            )

            labels_a = seg[face_a].ravel()
            labels_b = seg[face_b].ravel()

            uv_ids = np.concatenate(
                [labels_a[:, None], labels_b[:, None]],
                axis=1
            )
            uv_ids = np.unique(uv_ids, axis=0)

            edge_ids = rag.findEdges(uv_ids)
            edge_ids = edge_ids[edge_ids != -1]
            stitch_edges.append(edge_ids)

        if stitch_edges:
            stitch_edges = np.concatenate(stitch_edges)
            stitch_edges = np.unique(stitch_edges)
        else:
            stitch_edges = None
        return stitch_edges

    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            stitch_edges = list(tqdm(
                tp.map(find_stitch_edges, range(blocking.numberOfBlocks)),
                total=blocking.numberOfBlocks
            ))
        else:
            stitch_edges = tp.map(find_stitch_edges, range(blocking.numberOfBlocks))

    stitch_edges = np.concatenate([st for st in stitch_edges if st is not None])
    stitch_edges = np.unique(stitch_edges)
    full_edges = np.zeros(rag.numberOfEdges, dtype="bool")
    full_edges[stitch_edges] = 1
    return full_edges


def project_node_labels_to_pixels(rag, node_labels: np.ndarray, n_threads: Optional[int] = None) -> np.ndarray:
    """Project label values for graph nodes back to pixels to obtain segmentation.

    Args:
        rag: The region adjacency graph.
        node_labels: The array with node labels.
        n_threads: The number of threads used, set to cpu count by default.

    Returns:
        The segmentation.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if len(node_labels) != rag.numberOfNodes:
        raise ValueError("Incompatible number of node labels: %i, %i" % (len(node_labels), rag.numberOfNodes))
    seg = nrag.projectScalarNodeDataToPixels(rag, node_labels, numberOfThreads=n_threads)
    return seg


def compute_z_edge_mask(rag, watershed: np.ndarray) -> np.ndarray:
    """Compute edge mask of in-between plane edges for flat superpixels.

    Flat superpixels are volumetric superpixels that are independent across slices.
    This function does not check wether the input watersheds are actually flat.

    Args:
        rag: The region adjacency graph.
        watershed: The underlying watershed over-segmentation (superpixels).

    Returns:
        The edge mask indicating in-between slice edges.
    """
    node_z_coords = np.zeros(rag.numberOfNodes, dtype="uint32")
    for z in range(watershed.shape[0]):
        node_z_coords[watershed[z]] = z
    uv_ids = rag.uvIds()
    z_edge_mask = node_z_coords[uv_ids[:, 0]] != node_z_coords[uv_ids[:, 1]]
    return z_edge_mask
