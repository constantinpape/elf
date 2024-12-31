from typing import List, Optional, Union

import numpy as np
import nifty.graph.agglo as nagglo
from nifty.graph import UndirectedGraph
from vigra.analysis import relabelConsecutive

from .features import (compute_rag, compute_affinity_features,
                       compute_boundary_mean_and_length, project_node_labels_to_pixels)
from .watershed import apply_size_filter


def mala_clustering(
    graph: UndirectedGraph,
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
        edge_sizes: The edge sizes.
        threshold: The threshold to stop clustering.
        return_object: Whether to return the clustering object instead of the result.

    Returns:
        The node labels obtained from clustering.
    """
    n_nodes = graph.numberOfNodes
    policy_class = nagglo.malaClusterPolicyWithUcm if return_object else nagglo.malaClusterPolicy
    policy = policy_class(graph=graph,
                          edgeIndicators=edge_features,
                          nodeSizes=np.zeros(n_nodes, dtype="float"),
                          edgeSizes=edge_sizes,
                          threshold=threshold)
    clustering = nagglo.agglomerativeClustering(policy)
    if return_object:
        return clustering
    clustering.run()
    return clustering.result()


def agglomerative_clustering(
    graph: UndirectedGraph,
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
        return_object: Whether to return the clustering object instead of the result.

    Returns:
        The node labels obtained from clustering.
    """
    policy_class = nagglo.edgeWeightedClusterPolicyWithUcm if return_object else nagglo.edgeWeightedClusterPolicy
    policy = policy_class(graph=graph,
                          edgeIndicators=edge_features,
                          nodeSizes=node_sizes.astype("float"),
                          edgeSizes=edge_sizes.astype("float"),
                          numberOfNodesStop=n_stop,
                          sizeRegularizer=size_regularizer)
    clustering = nagglo.agglomerativeClustering(policy)
    if return_object:
        return clustering
    clustering.run()
    return clustering.result()


def compute_graph_and_features(segmentation, input_map, offsets=None, n_threads=None):
    """@private
    """
    # compute the graph and edge weihts / edge lens
    graph = compute_rag(segmentation, n_threads=n_threads)
    if offsets is None:
        if segmentation.shape != input_map.shape:
            raise ValueError("The shape of the boundary map and the segmentation needs to be the same")
        edge_weights = compute_boundary_mean_and_length(graph, input_map, n_threads=n_threads)
        edge_weights, edge_sizes = edge_weights[:, 0], edge_weights[:, 1]
    else:
        n_offsets, spatial_shape = input_map[0], input_map[1:]
        if segmentation.shape != spatial_shape:
            raise ValueError("The shape of the boundary map and the segmentation needs to be the same")
        if len(offsets) != n_offsets:
            raise ValueError("The number of channels in the affinity map and the number of offsets need to be the same")
        edge_weights = compute_affinity_features(graph, input_map, offsets, n_threads=n_threads)[:, 0]
        edge_sizes = compute_boundary_mean_and_length(graph, input_map[0], n_threads=n_threads)[:, 1]
    return graph, edge_weights, edge_sizes


def _cluster_segmentation_impl(segmentation, input_map, cluster_function,
                               offsets=None, n_threads=None, min_segment_size=0,
                               **cluster_kwargs):
    graph, edge_weights, edge_sizes = compute_graph_and_features(segmentation, input_map,
                                                                 offsets=offsets, n_threads=n_threads)
    clusters = cluster_function(graph=graph,
                                edge_features=edge_weights,
                                edge_sizes=edge_sizes,
                                **cluster_kwargs)
    clusters = relabelConsecutive(clusters, start_label=1, keep_zeros=False)[0].astype('uint32')
    seg = project_node_labels_to_pixels(graph, clusters)
    if min_segment_size > 0:
        inp = input_map if offsets is None else input_map[0]
        seg = apply_size_filter(seg, inp, min_segment_size)[0]
        seg = relabelConsecutive(seg, start_label=1, keep_zeros=False)[0]
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
    us, vs, dist, sizes = clustering.runAndGetLinkageMatrix()
    lm = [us, vs, dist, sizes]
    lm = list(map(np.array, lm))
    lm = np.concatenate([xx[:, None] for xx in lm], axis=1)
    if normalize_distances:
        lm[:, 2] -= lm[:, 2].min()
        lm[:, 2] /= (lm[:, 2].max() + 1e-6)
    return lm


def clusters_from_tree(tree, n_clusters):
    """@private
    """
    idx = len(tree) - n_clusters
    clusters = tree[:, idx]
    return clusters
