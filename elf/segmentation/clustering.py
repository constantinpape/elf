import numpy as np
import nifty.graph.agglo as nagglo
from .features import (compute_rag, compute_affinity_features,
                       compute_boundary_mean_and_length, project_node_labels_to_pixels)


def mala_clustering(graph, edge_features, edge_sizes, threshold):
    """ Compute segmentation with mala-style clustering.

    In "Large Scale Image Segmentation with Structured Loss based Deep Learning for Connectome Reconstruction":
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8364622

    Arguments:
        graph [nifty.graph] - graph to cluster
        edge_features [np.ndarray] - features used for clustering
        edge_sizes [np.ndarray] - sizes of edges
        threshold [float] - threshold to stop clustering
    """
    n_nodes = graph.numberOfNodes
    policy = nagglo.malaClusterPolicy(graph=graph,
                                      edgeIndicators=edge_features,
                                      nodeSizes=np.zeros(n_nodes, dtype='float'),
                                      edgeSizes=edge_sizes,
                                      threshold=threshold)
    clustering = nagglo.agglomerativeClustering(policy)
    clustering.run()
    return clustering.result()


def agglomerative_clustering(graph, edge_features,
                             node_sizes, edge_sizes,
                             n_stop, size_regularizer):
    """ Compute segmentation with agglomerative clustering with optional size regularizer.

    Arguments:
        graph [nifty.graph] - graph to cluster
        edge_features [np.ndarray] - features used for clustering
        node_sizes [np.ndarray] - sizes of nodes
        edge_sizes [np.ndarray] - sizes of edges
        n_stop [int] - target number of clusters
        size_regularizer [float] - strength of size regularizer
    """
    policy = nagglo.edgeWeightedClusterPolicy(graph=graph,
                                              edgeIndicators=edge_features,
                                              nodeSizes=node_sizes.astype('float'),
                                              edgeSizes=edge_sizes.astype('float'),
                                              numberOfNodesStop=n_stop,
                                              sizeRegularizer=size_regularizer)
    clustering = nagglo.agglomerativeClustering(policy)
    clustering.run()
    return clustering.result()


def cluster_segmentation(segmentation, input_map,
                         threshold=None, n_stop=None, size_regularizer=1.,
                         offsets=None, n_threads=None):
    """ Run clustering to merge segments provided a heightmap or affinity map.

    Computes a graph and edge weights (derived from the height map)
    and then agglomeratively clusters the graph to merge segments.
    Exactly one of the parameters threshold or n_stop needs to be given.
    If threshold is given, the accumulated edge weights of clusters is used as stopping criterion
    and clustering stops when all edge weights are above the threshold.
    If n_stop is given, clustering stops when the number of clusters is below n_stop.

    Arguments:
        segmentation [np.ndarray] - the input segmentation
        input_map [np.ndarray] - the input used to derive the edge weigths.
            Can either be boundary probabilities or affinities.
        threshold [float] - threshold used as stopping criterion (default: None)
        n_stop [int or float] - number (or fraction) of clusters used as stopping criterion (default: None)
        size_regularizer [float] - size regularizer for agglomerative clustering (default: 1.)
        offsets [list[list[int]]] - (default: None)
        n_threads [int] - number of threads used, set to cpu count by default. (default: None)
    """
    if (threshold is None) == (n_stop is None):
        raise ValueError("Exactly one of the parameters 'threshold' or 'n_stop' needs to be given")

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
        edge_weigths = compute_affinity_features(graph, input_map, offsets, n_threads=n_threads)[:, 0]
        edge_sizes = compute_boundary_mean_and_length(graph, input_map[0], n_threads=n_threads)[:, 1]

    # run clustering
    if n_stop is None:  # mala clustering
        clusters = mala_clustering(graph, edge_weigths, edge_sizes, threshold)

    else:  # agglomerative clustering
        _, node_sizes = np.unique(segmentation, return_counts=True)
        if n_stop < 1:
            assert isinstance(n_stop, float)
            n_stop = int(n_stop * len(node_sizes))
        clusters = agglomerative_clustering(graph, edge_weights,
                                            node_sizes, edge_sizes,
                                            n_stop, size_regularizer)

    return project_node_labels_to_pixels(graph, clusters)
