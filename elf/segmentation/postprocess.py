from typing import Optional

import numpy as np
import nifty
import vigra


def graph_watershed(graph: nifty.graph.UndirectedGraph, edge_weigths: np.ndarray, seed_nodes: np.ndarray) -> np.ndarray:
    """Compute graph watershed.

    Args:
        graph: The input graph.
        edge_weights: The edge weights used as heightmap for the watershed.
        seed_nodes: The seed nodes for the watershed. Nodes without a seed must be set to zero.

    Returns:
        The node labeling results of the graph watershed.
    """
    assert len(edge_weigths) == graph.numberOfEdges
    assert len(seed_nodes) == graph.numberOfNodes
    node_labels = nifty.graph.edgeWeightedWatershedsSegmentation(graph, seed_nodes, edge_weigths)
    return node_labels


def graph_size_filter(
    graph: nifty.graph.UndirectedGraph,
    edge_weigths: np.ndarray,
    node_sizes: np.ndarray,
    min_size: int,
    node_labels: Optional[np.ndarray] = None,
    relabel: bool = False,
) -> np.ndarray:
    """Size filter a graph via seeded edge watershed.

    Args:
        graph: The input graph.
        edge_weights: The edge weights.
        node_sizes: The node sizes.
        min_size: The minimal node sizes. Nodes with a smaller size will be filtered.
        node_labels: Optional initial node labeling.
        relabel: Whether to relabel the node labeling after the watershed.

    Returns:
        The size filtered node labeling.
    """
    n_nodes = graph.numberOfNodes

    if node_labels is None:
        seeds = np.zeros(n_nodes, dtype="uint64")
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
