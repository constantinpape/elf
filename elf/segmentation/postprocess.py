from typing import Optional

import bioimage_cpp as bic
import numpy as np


def graph_watershed(graph, edge_weigths: np.ndarray, seed_nodes: np.ndarray) -> np.ndarray:
    """Compute graph watershed.

    Args:
        graph: The input graph.
        edge_weights: The edge weights used as heightmap for the watershed.
        seed_nodes: The seed nodes for the watershed. Nodes without a seed must be set to zero.

    Returns:
        The node labeling results of the graph watershed.
    """
    assert len(edge_weigths) == graph.number_of_edges
    assert len(seed_nodes) == graph.number_of_nodes
    node_labels = bic.graph.edge_weighted_watershed(graph, edge_weigths, seed_nodes)
    return node_labels


def graph_size_filter(
    graph,
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
    n_nodes = graph.number_of_nodes

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
        seeds = node_labels.astype("uint64", copy=True)
        seeds[np.isin(seeds, discard_labels)] = 0

    if relabel:
        seeds, _, _ = bic.segmentation.relabel_sequential(seeds, offset=1)

    return graph_watershed(graph, edge_weigths, seeds)
