import numpy as np
import nifty.graph.agglo as nagglo


def mala_clustering(graph, edge_features, edge_sizes, threshold):
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
    policy = nagglo.edgeWeightedClusterPolicy(graph=graph,
                                              edgeIndicators=edge_features,
                                              nodeSizes=node_sizes.astype('float'),
                                              edgeSizes=edge_sizes.astype('float'),
                                              numberOfNodesStop=n_stop,
                                              sizeRegularizer=size_regularizer)
    clustering = nagglo.agglomerativeClustering(policy)
    clustering.run()
    return clustering.result()
