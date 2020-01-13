import numpy as np
import nifty
import vigra


def graph_watershed(graph, edge_weigths, seed_nodes):
    """
    """
    # TODO do we need to cast to 'nifty.graph.undirectedGraph' if we get a rag?
    assert len(edge_weigths) == graph.numberOfEdges
    assert len(seed_nodes) == graph.numberOfNodes

    # run graph watershed
    node_labels = nifty.graph.edgeWeightedWatershedsSegmentation(graph, seed_nodes, edge_weigths)
    return node_labels


def graph_size_filter(graph, edge_weigths, node_sizes, min_size,
                      node_labels=None, relabel=False):
    """
    """
    n_nodes = graph.numberOfNodes

    if node_labels is None:
        seeds = np.zeros(n_nodes, dtype='uint64')
        assert n_nodes == len(node_sizes)
        keep_nodes = node_sizes >= min_size
        seeds[keep_nodes] = np.arange(0, n_nodes)[keep_nodes]
    else:
        assert n_nodes == len(node_labels)
        n_labels = int(node_labels.max() + 1)
        assert n_labels == len(node_sizes)
        discard_labels = np.where(node_sizes < min_size)[0]
        seeds = node_labels.copy()
        seeds[discard_labels] = 0

    if relabel:
        vigra.analysis.relabelConsecutive(seeds, start_label=1, keep_zeros=True, out=seeds)

    return graph_watershed(graph, edge_weigths, seeds)
