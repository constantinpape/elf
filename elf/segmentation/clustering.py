from typing import List, Optional, Union

import bioimage_cpp as bic
import numpy as np

from .features import (compute_rag, compute_affinity_features,
                       compute_boundary_mean_and_length, project_node_labels_to_pixels)
from .watershed import apply_size_filter


def mala_clustering(
    graph,
    edge_features: np.ndarray,
    edge_sizes: np.ndarray,
    threshold: float,
    return_object: bool = False,
) -> np.ndarray:
    """Compute agglomerative clustering with the MALA algorithm.

    From "Large Scale Image Segmentation with Structured Loss based Deep Learning for Connectome Reconstruction":
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8364622

    Args:
        graph: The graph to cluster.
        edge_features: The edge features for clustering.
        edge_sizes: The edge sizes. Currently ignored; kept for backwards-compatibility.
        threshold: The threshold to stop clustering.
        return_object: Not supported on the bioimage-cpp backend.

    Returns:
        The node labels obtained from clustering.
    """
    if return_object:
        raise NotImplementedError(
            "return_object=True is not supported on the bioimage-cpp backend "
            "(no UCM policy variant)."
        )
    del edge_sizes  # MALA policy in bic does not consume edge sizes.
    policy = bic.graph.agglomeration.MalaClusterPolicy(
        threshold=float(threshold), num_clusters_stop=0,
    )
    return policy.optimize(graph, np.asarray(edge_features))


def agglomerative_clustering(
    graph,
    edge_features: np.ndarray,
    node_sizes: np.ndarray,
    edge_sizes: np.ndarray,
    n_stop: int,
    size_regularizer: float,
    return_object: bool = False,
) -> np.ndarray:
    """Compute agglomerative clustering with size regularizer.

    Args:
        graph: The graph to cluster.
        edge_features: The edge features for clustering.
        node_sizes: The node sizes.
        edge_sizes: The edge sizes.
        n_stop: The target number of clusters.
        size_regularizer: The strength of the size regularizer.
        return_object: Not supported on the bioimage-cpp backend.

    Returns:
        The node labels obtained from clustering.
    """
    if return_object:
        raise NotImplementedError(
            "return_object=True is not supported on the bioimage-cpp backend "
            "(no UCM policy variant)."
        )
    policy = bic.graph.agglomeration.EdgeWeightedClusterPolicy(
        num_clusters_stop=int(n_stop), size_regularizer=float(size_regularizer),
    )
    return policy.optimize(
        graph,
        np.asarray(edge_features),
        edge_sizes=np.asarray(edge_sizes, dtype="float64"),
        node_sizes=np.asarray(node_sizes, dtype="float64"),
    )


def compute_graph_and_features(segmentation, input_map, offsets=None, n_threads=None):
    """@private
    """
    rag = compute_rag(segmentation, n_threads=n_threads)
    if offsets is None:
        if segmentation.shape != input_map.shape:
            raise ValueError("The shape of the boundary map and the segmentation needs to be the same")
        edge_weights = compute_boundary_mean_and_length(rag, segmentation, input_map, n_threads=n_threads)
        edge_weights, edge_sizes = edge_weights[:, 0], edge_weights[:, 1]
    else:
        n_offsets, spatial_shape = input_map.shape[0], input_map.shape[1:]
        if segmentation.shape != spatial_shape:
            raise ValueError("The shape of the boundary map and the segmentation needs to be the same")
        if len(offsets) != n_offsets:
            raise ValueError("The number of channels in the affinity map and the number of offsets need to be the same")
        edge_weights = compute_affinity_features(rag, segmentation, input_map, offsets, n_threads=n_threads)[:, 0]
        edge_sizes = compute_boundary_mean_and_length(rag, segmentation, input_map[0], n_threads=n_threads)[:, 1]
    return rag, edge_weights, edge_sizes, rag


def _cluster_segmentation_impl(segmentation, input_map, cluster_function,
                               offsets=None, n_threads=None, min_segment_size=0,
                               **cluster_kwargs):
    graph, edge_weights, edge_sizes, rag = compute_graph_and_features(
        segmentation, input_map, offsets=offsets, n_threads=n_threads,
    )
    clusters = cluster_function(graph=graph,
                                edge_features=edge_weights,
                                edge_sizes=edge_sizes,
                                **cluster_kwargs)
    # Make labels consecutive starting at 1 (relabel_sequential preserves 0; shift by 1 to relabel zero too).
    clusters = bic.segmentation.relabel_sequential(clusters.astype("uint64") + 1, offset=1)[0].astype("uint32")
    seg = project_node_labels_to_pixels(rag, segmentation, clusters)
    if min_segment_size > 0:
        inp = input_map if offsets is None else input_map[0]
        seg = apply_size_filter(seg, inp, min_segment_size)[0]
        seg = bic.segmentation.relabel_sequential(seg.astype("uint64") + 1, offset=1)[0]
    return seg


def cluster_segmentation(
    segmentation: np.ndarray,
    input_map: np.ndarray,
    n_stop: Union[int, float],
    size_regularizer: float = 1.0,
    offsets: Optional[List[List[int]]] = None,
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """Run agglomerative clustering to merge segments based on boundary probabilities or a affinity map.

    Computes a graph and edge weights, derived from the boundary probabilities or affinity map,
    and then apply agglomerative clustering to the graph to merge segments.
    Clustering stops when the number of clusters is below n_stop.

    Args:
        segmentation: The input segmentation.
        input_map: The input used to derive the edge weigths. Can either be boundary probabilities or affinities.
        n_stop: The number (or fraction) of clusters used as stopping criterion.
        size_regularizer: The strength of the size regularizer for agglomerative clustering.
        offsets: The offsets corresponding to the affinity channels.
        n_threads: Number of threads used, is set to the cpu count by default.

    Returns:
        The segmentation after clustering.
    """
    _, node_sizes = np.unique(segmentation, return_counts=True)
    if n_stop < 1:
        assert isinstance(n_stop, float)
        n_stop = int(n_stop * len(node_sizes))
    seg = _cluster_segmentation_impl(segmentation, input_map,
                                     cluster_function=agglomerative_clustering,
                                     offsets=offsets,
                                     n_threads=n_threads,
                                     min_segment_size=0,
                                     node_sizes=node_sizes,
                                     n_stop=n_stop,
                                     size_regularizer=size_regularizer)
    return seg


def cluster_segmentation_mala(
    segmentation: np.ndarray,
    input_map: np.ndarray,
    threshold: float,
    min_segment_size: int = 0,
    offsets: Optional[List[List[int]]] = None,
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """Run MALA agglomerative clustering to merge segments based on boundary probabilties or an affinity map.

    Computes a graph and edge weights, derived from the boundary probabilities or affinity map,
    and then apply agglomerative clustering to merge segments.
    The accumulated edge weights of clusters are used as stopping criterion
    and clustering stops when all edge weights are above or equal the threshold.

    Args:
        segmentation: The input segmentation.
        input_map: The input used to derive the edge weigths. Can either be boundary probabilities or affinities.
        threshold: Threshold used as stopping criterion.
        min_segment_size: Minimal size of segments in the segmentation result.
        offsets: The offsets corresponding to the affinity channels.
        n_threads: Number of threads used, is set to the cpu count by default.

    Returns:
        The segmentation after clustering.
    """
    seg = _cluster_segmentation_impl(segmentation, input_map,
                                     cluster_function=mala_clustering,
                                     offsets=offsets,
                                     n_threads=n_threads,
                                     min_segment_size=min_segment_size,
                                     threshold=threshold)
    return seg


def compute_linkage_matrix(clustering, normalize_distances=False):
    """@private
    """
    raise NotImplementedError(
        "Linkage matrix extraction is not available with the bioimage-cpp backend."
    )


def clusters_from_tree(tree, n_clusters):
    """@private
    """
    raise NotImplementedError(
        "Linkage tree extraction is not available with the bioimage-cpp backend."
    )
