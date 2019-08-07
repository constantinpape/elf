import numpy as np
import nifty.graph.agglo as nagglo

# TODO add citations in doc-strings


def mala_clustering(graph, edge_features, edge_sizes, threshold):
    """ Compute segmentation with mala-style clustering.

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
    """ Compute segmentation with agglomerative clustering.

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
