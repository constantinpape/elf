import multiprocessing
from concurrent import futures
from typing import Dict, List, Optional, Tuple

import bioimage_cpp as bic
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.measure import regionprops_table

from tqdm import tqdm
from .multicut import transform_probabilities_to_costs


# Map fastfilters/vigra filter names to bic.filters callables.
_BIC_FILTERS = {
    "gaussianSmoothing": bic.filters.gaussian_smoothing,
    "gaussianGradientMagnitude": bic.filters.gaussian_gradient_magnitude,
    "laplacianOfGaussian": bic.filters.laplacian_of_gaussian,
    "hessianOfGaussianEigenvalues": bic.filters.hessian_of_gaussian_eigenvalues,
    "structureTensorEigenvalues": bic.filters.structure_tensor_eigenvalues,
    "gaussianDerivative": bic.filters.gaussian_derivative,
}


def _apply_filter(filter_name, image, sigma):
    """@private"""
    fu = _BIC_FILTERS[filter_name]
    if image.dtype not in (np.float32, np.float64, np.uint8, np.uint16):
        image = image.astype("float32")
    return fu(image, sigma)


#
# Region Adjacency Graph and Features
#

def compute_rag(segmentation: np.ndarray, n_labels: Optional[int] = None, n_threads: Optional[int] = None):
    """Compute region adjacency graph of segmentation.

    Args:
        segmentation: The segmentation.
        n_labels: Deprecated; ignored. Kept for backwards-compatibility.
        n_threads: The number of threads used, set to cpu count by default.

    Returns:
        The region adjacency graph (`bioimage_cpp.graph.RegionAdjacencyGraph`).
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if segmentation.dtype not in (np.uint32, np.uint64, np.int32, np.int64):
        segmentation = segmentation.astype("uint32")
    rag = bic.graph.region_adjacency_graph(segmentation, number_of_threads=n_threads)
    return rag


def compute_boundary_features(
    rag,
    segmentation: np.ndarray,
    boundary_map: np.ndarray,
    min_value: float = 0.0,  # noqa: ARG001 — deprecated, ignored
    max_value: float = 1.0,  # noqa: ARG001 — deprecated, ignored
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """Compute edge features from boundary map.

    Args:
        rag: The region adjacency graph.
        segmentation: The over-segmentation used to construct the RAG.
        boundary_map: The boundary map.
        min_value: Deprecated; ignored.
        max_value: Deprecated; ignored.
        n_threads: The number of threads used, set to cpu count by default.

    Returns:
        The edge features. Output has 12 columns
        (mean, median, std, min, max, p5, p10, p25, p75, p90, p95, size).
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if segmentation.shape != boundary_map.shape:
        raise ValueError("Incompatible shapes: %s, %s" % (str(segmentation.shape), str(boundary_map.shape)))
    features = bic.graph.features.edge_map_features_complex(
        rag, segmentation, boundary_map, number_of_threads=n_threads,
    )
    return features


def compute_affinity_features(
    rag,
    segmentation: np.ndarray,
    affinity_map: np.ndarray,
    offsets: List[List[int]],
    min_value: float = 0.0,  # noqa: ARG001 — deprecated, ignored
    max_value: float = 1.0,  # noqa: ARG001 — deprecated, ignored
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """Compute edge features from affinity map.

    Args:
        rag: The region adjacency graph.
        segmentation: The over-segmentation used to construct the RAG.
        affinity_map: The affinity map.
        offsets: The offsets corresponding to the affinity channels.
        min_value: Deprecated; ignored.
        max_value: Deprecated; ignored.
        n_threads: The number of threads used, set to cpu count by default.

    Returns:
        The edge features. Output has 12 columns
        (mean, median, std, min, max, p5, p10, p25, p75, p90, p95, size).
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if segmentation.shape != affinity_map.shape[1:]:
        raise ValueError("Incompatible shapes: %s, %s" % (str(segmentation.shape), str(affinity_map.shape[1:])))
    if len(offsets) != affinity_map.shape[0]:
        raise ValueError("Incompatible number of channels and offsets: %i, %i" % (len(offsets),
                                                                                  affinity_map.shape[0]))
    features = bic.graph.features.affinity_features_complex(
        rag, segmentation, affinity_map, offsets, number_of_threads=n_threads,
    )
    return features


def compute_boundary_mean_and_length(
    rag, segmentation: np.ndarray, input_: np.ndarray, n_threads: Optional[int] = None,
) -> np.ndarray:
    """Compute mean value and length of boundaries.

    Args:
        rag: The region adjacency graph.
        segmentation: The over-segmentation used to construct the RAG.
        input_: The input map.
        n_threads: The number of threads used, set to cpu count by default.

    Returns:
        The edge features with two columns (mean, size).
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if segmentation.shape != input_.shape:
        raise ValueError("Incompatible shapes: %s, %s" % (str(segmentation.shape), str(input_.shape)))
    features = bic.graph.features.edge_map_features(
        rag, segmentation, input_, number_of_threads=n_threads,
    )
    return features


# TODO generalize and move to elf.features.parallel
def _filter_2d(input_, filter_name, sigma, n_threads):
    def _fz(inp):
        response = _apply_filter(filter_name, inp, sigma)
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
    segmentation: np.ndarray,
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
        segmentation: The over-segmentation used to construct the RAG.
        input_: The input data.
        apply_2d: Whether to apply the filters in 2d for 3d input data.
        n_threads: The number of threads.
        filters: The filters to apply, expects a dictionary mapping filter names to sigma values.

    Returns:
        The edge features.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    features = []

    # apply 2d: we compute filters and derived features in parallel per filter
    if apply_2d:

        def _compute_2d(filter_name, sigma):
            response = _filter_2d(input_, filter_name, sigma, n_threads)
            assert response.ndim == 4
            n_channels = response.shape[-1]
            feats = []
            for chan in range(n_channels):
                chan_data = response[..., chan]
                feats.append(compute_boundary_features(rag, segmentation, chan_data, n_threads=n_threads))

            out = np.concatenate(feats, axis=1)
            assert len(out) == rag.number_of_edges
            return out

        features = [_compute_2d(filter_name, sigma)
                    for filter_name, sigmas in filters.items() for sigma in sigmas]

    # apply 3d: we parallelize over the whole filter + feature computation
    # this can be very memory intensive, and it would be better to parallelize inside
    # of the loop, but 3d parallel filters in elf.parallel.filters are not working properly yet
    else:

        def _compute_3d(filter_name, sigma):
            response = _apply_filter(filter_name, input_, sigma)
            if response.ndim == input_.ndim:
                response = response[..., None]

            n_channels = response.shape[-1]
            feats = []

            for chan in range(n_channels):
                chan_data = response[..., chan]
                feats.append(compute_boundary_features(rag, segmentation, chan_data, n_threads=1))
            out = np.concatenate(feats, axis=1)
            assert len(out) == rag.number_of_edges, f"{len(out), {rag.number_of_edges}}"
            return out

        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(_compute_3d, filter_name, sigma)
                     for filter_name, sigmas in filters.items() for sigma in sigmas]
            features = [t.result() for t in tasks]

    features = np.concatenate(features, axis=1)
    assert len(features) == rag.number_of_edges
    return features


# Intensity statistics that skimage.measure.regionprops does not provide natively.
# Each callback receives the region's cropped (regionmask, intensity_image); see
# `_region_features`. The function names double as the keys in the regionprops table.
def _quantiles(regionmask, intensity_image):
    """@private"""
    return np.percentile(intensity_image[regionmask], [0, 10, 25, 50, 75, 90, 100])


def _kurtosis(regionmask, intensity_image):
    """@private"""
    values = intensity_image[regionmask]
    if values.size < 2 or values.min() == values.max():
        return 0.0
    return kurtosis(values)


def _skewness(regionmask, intensity_image):
    """@private"""
    values = intensity_image[regionmask]
    if values.size < 2 or values.min() == values.max():
        return 0.0
    return skew(values)


def _variance(regionmask, intensity_image):
    """@private"""
    return np.var(intensity_image[regionmask])


def _sum(regionmask, intensity_image):
    """@private"""
    return intensity_image[regionmask].sum()


# Map vigra `extractRegionFeatures` names to their source in a skimage regionprops table.
# Names starting with "_" are computed via the extra-property callbacks above; the rest are
# native regionprops properties (array-valued ones are expanded into "<name>-<i>" columns).
_REGION_FEATURE_KEYS = {
    "Count": "num_pixels",
    "Maximum": "intensity_max",
    "Minimum": "intensity_min",
    "mean": "intensity_mean",
    "RegionCenter": "centroid",
    "Weighted<RegionCenter>": "centroid_weighted",
    "RegionRadii": "inertia_tensor_eigvals",
    "Quantiles": "_quantiles",
    "Kurtosis": "_kurtosis",
    "Skewness": "_skewness",
    "Variance": "_variance",
    "Sum": "_sum",
}
_REGION_FEATURE_EXTRA = {
    "_quantiles": _quantiles,
    "_kurtosis": _kurtosis,
    "_skewness": _skewness,
    "_variance": _variance,
    "_sum": _sum,
}


def _region_features(input_map: np.ndarray, segmentation: np.ndarray, feature_names: List[str]) -> Dict:
    """@private

    Replacement for ``vigra.analysis.extractRegionFeatures`` based on
    ``skimage.measure.regionprops``. Returns a dict mapping each requested feature name to a
    dense array indexed by label id (``0 .. segmentation.max()``); scalar features are 1D and
    coordinate/quantile/radii features are 2D, matching the vigra layout. Missing label ids
    (gaps) stay zero.
    """
    if segmentation.dtype.kind not in "iu":
        segmentation = segmentation.astype("int64")
    keys = [_REGION_FEATURE_KEYS[name] for name in feature_names]
    native = tuple(dict.fromkeys(key for key in keys if not key.startswith("_")))
    extra = tuple(dict.fromkeys(_REGION_FEATURE_EXTRA[key] for key in keys if key.startswith("_")))

    # skimage treats label 0 as background; shift by 1 so the original label 0 is included.
    table = regionprops_table(
        segmentation + 1, intensity_image=input_map.astype("float32", copy=False),
        properties=("label",) + native, extra_properties=(extra or None),
    )
    labels = np.asarray(table["label"]) - 1
    n_nodes = int(segmentation.max()) + 1

    def _gather(base):
        if base in table:
            return np.asarray(table[base], dtype="float32")[:, None]
        cols, i = [], 0
        while f"{base}-{i}" in table:
            cols.append(np.asarray(table[f"{base}-{i}"], dtype="float32"))
            i += 1
        return np.stack(cols, axis=1)

    result = {}
    for name, base in zip(feature_names, keys):
        cols = _gather(base)
        if name == "RegionRadii":  # vigra returns radii = sqrt of the coordinate-covariance eigenvalues
            cols = np.sqrt(np.maximum(cols, 0.0))
        dense = np.zeros((n_nodes, cols.shape[1]), dtype="float32")
        dense[labels] = cols
        result[name] = dense[:, 0] if dense.shape[1] == 1 else dense
    return result


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
    node_features = _region_features(input_map, segmentation, feature_names)

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
    return bic.graph.grid_graph(shape)


def _nn_offsets(ndim):
    return [[-1 if i == d else 0 for i in range(ndim)] for d in range(ndim)]


def _apply_strides(edges, weights, strides, randomize_strides):
    """Subsample (edges, weights) along the spatial periodicity defined by `strides`.

    Mirrors the behaviour of nifty's strides/randomize_strides parameter without
    spatial information: we simply keep one out of every `prod(strides)` entries
    (or a random subset of the same size if `randomize_strides` is True).
    """
    if strides is None:
        return edges, weights
    keep = int(np.prod(strides))
    if keep <= 1:
        return edges, weights
    n = len(edges)
    if randomize_strides:
        idx = np.random.choice(n, size=max(1, n // keep), replace=False)
        idx.sort()
    else:
        idx = np.arange(0, n, keep)
    return edges[idx], weights[idx]


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
        grid_graph: The grid graph.
        image: The image, from which the features will be derived.
        mode: Feature accumulation method. For multi-channel images, one of
            "l1", "l2", "cosine". For scalar images (without channels) only
            grid-boundary averaging is supported (any mode value is accepted).
        offsets: The offsets, which correspond to the affinity channels.
        strides: The strides used to subsample edges that are computed from offsets.
        randomize_strides: Whether to subsample randomly instead of using regular strides.

    Returns:
        The uv ids of the edges.
        The edge features.
    """
    gndim = len(grid_graph.shape)

    if image.ndim == gndim:
        if offsets is not None:
            raise NotImplementedError("Offsets with scalar images are not supported.")
        weights = bic.graph.features.grid_boundary_features(grid_graph, image.astype("float32"))
        edges = grid_graph.uv_ids()
        return edges, weights

    if image.ndim != gndim + 1:
        raise ValueError(f"Invalid image dimension {image.ndim}, expected {gndim} or {gndim + 1}")

    modes = ("l1", "l2", "cosine")
    if mode not in modes:
        raise ValueError(f"Invalid feature mode {mode}, expect one of {modes}")

    if offsets is None:
        # Compute affinities between adjacent pixels using nearest-neighbor offsets.
        nn_offs = _nn_offsets(gndim)
        affs = bic.affinities.compute_embedding_distances(
            image.astype("float32"), nn_offs, norm=mode,
        )
        weights, _valid = bic.graph.features.grid_affinity_features(grid_graph, affs, nn_offs)
        edges = grid_graph.uv_ids()
        return edges, weights

    # General path with arbitrary offsets: compute affinities then use _with_lifted.
    affs = bic.affinities.compute_embedding_distances(
        image.astype("float32"), offsets, norm=mode,
    )
    local_w, local_valid, lifted_uvs, lifted_w, _ = bic.graph.features.grid_affinity_features_with_lifted(
        grid_graph, affs, offsets,
    )
    edges = np.concatenate([grid_graph.uv_ids()[local_valid], lifted_uvs], axis=0)
    weights = np.concatenate([local_w[local_valid], lifted_w], axis=0)
    return _apply_strides(edges, weights, strides, randomize_strides)


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
        grid_graph: The grid graph.
        affinities: The affinity map.
        offsets: The offsets, which correspond to the affinity channels.
        strides: The strides used to subsample edges that are computed from offsets.
        mask: Mask to exclude from the edge and feature computation.
        randomize_strides: Whether to subsample randomly instead of using regular strides.

    Returns:
        The uv ids of the edges.
        The edge features.
    """
    gndim = len(grid_graph.shape)
    if affinities.ndim != gndim + 1:
        raise ValueError("affinities must have shape (channels, *grid_graph.shape)")

    if offsets is None:
        assert affinities.shape[0] == gndim
        assert strides is None
        assert mask is None
        nn_offs = _nn_offsets(gndim)
        weights, _valid = bic.graph.features.grid_affinity_features(grid_graph, affinities, nn_offs)
        edges = grid_graph.uv_ids()
        return edges, weights

    local_w, local_valid, lifted_uvs, lifted_w, _ = bic.graph.features.grid_affinity_features_with_lifted(
        grid_graph, affinities, offsets,
    )
    edges = np.concatenate([grid_graph.uv_ids()[local_valid], lifted_uvs], axis=0)
    weights = np.concatenate([local_w[local_valid], lifted_w], axis=0)

    if mask is not None:
        assert strides is None and not randomize_strides, "Strides and mask cannot be used at the same time"
        shape = tuple(grid_graph.shape)
        assert mask.shape == shape, (
            "compute_grid_graph_affinity_features with a per-pixel mask expects mask.shape == grid_graph.shape; "
            "per-channel edge masks are only supported on legacy nifty grid graphs."
        )
        node_ids = np.arange(np.prod(shape), dtype="uint64").reshape(shape)
        masked_ids = node_ids[~mask]
        edge_state = np.isin(edges, masked_ids).sum(axis=1)
        keep = edge_state != 2
        edges, weights = edges[keep], weights[keep]
        return edges, weights

    return _apply_strides(edges, weights, strides, randomize_strides)


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
    shape = tuple(grid_graph.shape)
    assert mask.shape == shape, f"{mask.shape}, {shape}"
    node_ids = np.arange(np.prod(shape), dtype="uint64").reshape(shape)
    masked_ids = node_ids[~mask]

    edges = grid_graph.uv_ids()
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
    shape = tuple(grid_graph.shape)
    assert mask.shape == shape, f"{mask.shape}, {shape}"
    node_ids = np.arange(np.prod(shape), dtype="uint64").reshape(shape)
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
    # With all-zero node_labels and mode='all', every node pair within the BFS hop window
    # [2, max_graph_distance] is returned (base-graph edges excluded).
    node_labels = np.zeros(graph.number_of_nodes, dtype="uint64")
    lifted_uvs = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, node_labels, graph_depth=max_graph_distance, mode="all",
    )
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

    Args:
        rag: The region adjacency graph.
        watershed: The watershed over-segmentation.
        input_maps: List of probability maps. Each map must have the same shape as the watersheds.
        assignment_threshold: Minimal expression level to assign a class to a graph node.
        graph_depth: Maximal graph depth up to which lifted edges will be included.
        feats_to_costs: Function to calculate the lifted costs from the class assignment probabilities.
        mode: The mode for insertion of lifted edges. One of "all", "different", "same".
        n_threads: The number of threads used for the calculation.

    Returns:
        The lifted uv ids.
        The lifted costs.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    assert isinstance(input_maps, (list, tuple))
    assert all(isinstance(inp, np.ndarray) for inp in input_maps)
    shape = watershed.shape
    assert all(inp.shape == shape for inp in input_maps)

    n_nodes = int(watershed.max()) + 1
    node_labels = np.zeros(n_nodes, dtype="uint64")
    node_features = np.zeros(n_nodes, dtype="float32")
    for class_id, inp in enumerate(input_maps):
        mean_prob = _region_features(inp, watershed, ["mean"])["mean"]
        class_mask = mean_prob > assignment_threshold
        node_labels[class_mask] = class_id
        node_features[class_mask] = mean_prob[class_mask]

    lifted_uvs = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        rag, node_labels, graph_depth=graph_depth, mode=mode,
        ignore_label=0, number_of_threads=n_threads,
    )
    lifted_labels = node_labels[lifted_uvs]
    lifted_features = node_features[lifted_uvs]

    lifted_costs = feats_to_costs(lifted_labels, lifted_features)
    return lifted_uvs, lifted_costs


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
        graph_depth: The maximal graph depth up to which lifted edges will be included.
        same_segment_cost: The cost for edges between nodes with same segment id attribution.
        different_segment_cost: The cost for edges between nodes with different segment id attribution.
        mode: The mode for insertion of lifted edges. One of "all", "different", "same".
        n_threads: The number of threads used for the calculation.

    Returns:
        The lifted uv ids.
        The lifted costs.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    assert input_segmentation.shape == watershed.shape

    ovlp = bic.utils.segmentation_overlap(watershed, input_segmentation)
    ws_ids = np.unique(watershed)
    n_labels = int(ws_ids[-1]) + 1
    assert n_labels == rag.number_of_nodes, "%i, %i" % (n_labels, rag.number_of_nodes)

    node_labels = np.zeros(n_labels, dtype="uint64")
    node_label_vals = np.zeros(len(ws_ids), dtype="uint64")
    overlap_values = np.zeros(len(ws_ids), dtype="float64")
    for i, ws_id in enumerate(ws_ids):
        best = ovlp.best_overlap_for_label_a(int(ws_id), ignore_zero=False)
        node_label_vals[i] = best.label
        overlap_values[i] = best.fraction
    node_label_vals[overlap_values < overlap_threshold] = 0
    node_labels[ws_ids] = node_label_vals

    lifted_uvs = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        rag, node_labels, graph_depth=graph_depth, mode=mode,
        ignore_label=0, number_of_threads=n_threads,
    )
    assert lifted_uvs.max() < rag.number_of_nodes, "%i, %i" % (int(lifted_uvs.max()), rag.number_of_nodes)
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
    blocking = bic.utils.Blocking([0] * ndim, list(seg.shape), list(block_shape))

    def find_stitch_edges(block_id):
        stitch_edges = []
        block = blocking.get_block(block_id)
        for axis in range(ndim):
            if blocking.get_neighbor_id(block_id, axis, True) == -1:
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

            edge_ids = rag.find_edges(uv_ids)
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
                tp.map(find_stitch_edges, range(blocking.number_of_blocks)),
                total=blocking.number_of_blocks
            ))
        else:
            stitch_edges = tp.map(find_stitch_edges, range(blocking.number_of_blocks))

    stitch_edges = np.concatenate([st for st in stitch_edges if st is not None])
    stitch_edges = np.unique(stitch_edges)
    full_edges = np.zeros(rag.number_of_edges, dtype="bool")
    full_edges[stitch_edges] = 1
    return full_edges


def project_node_labels_to_pixels(
    rag, segmentation: np.ndarray, node_labels: np.ndarray, n_threads: Optional[int] = None,
) -> np.ndarray:
    """Project label values for graph nodes back to pixels to obtain segmentation.

    Args:
        rag: The region adjacency graph.
        segmentation: The over-segmentation used to construct the RAG.
        node_labels: The array with node labels.
        n_threads: The number of threads used, set to cpu count by default.

    Returns:
        The segmentation.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if len(node_labels) != rag.number_of_nodes:
        raise ValueError("Incompatible number of node labels: %i, %i" % (len(node_labels), rag.number_of_nodes))
    # bic.graph.project_node_labels_to_pixels requires integer dtypes for both arrays.
    if segmentation.dtype not in (np.uint32, np.uint64, np.int32, np.int64):
        segmentation = segmentation.astype("uint64")
    if node_labels.dtype not in (np.uint32, np.uint64, np.int32, np.int64):
        node_labels = node_labels.astype("uint64")
    seg = bic.graph.project_node_labels_to_pixels(rag, segmentation, node_labels, number_of_threads=n_threads)
    return seg


def compute_z_edge_mask(rag, watershed: np.ndarray) -> np.ndarray:
    """Compute edge mask of in-between plane edges for flat superpixels.

    Args:
        rag: The region adjacency graph.
        watershed: The underlying watershed over-segmentation (superpixels).

    Returns:
        The edge mask indicating in-between slice edges.
    """
    node_z_coords = np.zeros(rag.number_of_nodes, dtype="uint32")
    for z in range(watershed.shape[0]):
        node_z_coords[watershed[z]] = z
    uv_ids = rag.uv_ids()
    z_edge_mask = node_z_coords[uv_ids[:, 0]] != node_z_coords[uv_ids[:, 1]]
    return z_edge_mask
