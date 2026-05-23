from typing import Optional

import bioimage_cpp as bic
import numpy as np


def _to_bic_graph(graph):
    """Accept a nifty graph or a bic graph; return a bic graph."""
    if isinstance(graph, (bic.graph.UndirectedGraph, bic.graph.RegionAdjacencyGraph,
                          bic.graph.GridGraph2D, bic.graph.GridGraph3D)):
        return graph
    # Fall back: convert via uv ids (handles nifty graphs from clustering.py).
    return bic.graph.UndirectedGraph.from_edges(graph.numberOfNodes, graph.uvIds())


def graph_watershed(graph, edge_weigths: np.ndarray, seed_nodes: np.ndarray) -> np.ndarray:
    """Compute graph watershed.

    Args:
        graph: The input graph.
        edge_weights: The edge weights used as heightmap for the watershed.
        seed_nodes: The seed nodes for the watershed. Nodes without a seed must be set to zero.

    Returns:
        The node labeling results of the graph watershed.
    """
    g = _to_bic_graph(graph)
    assert len(edge_weigths) == g.number_of_edges
    assert len(seed_nodes) == g.number_of_nodes
    node_labels = bic.graph.edge_weighted_watershed(g, edge_weigths, seed_nodes)
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
    g = _to_bic_graph(graph)
    n_nodes = g.number_of_nodes

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

    return graph_watershed(g, edge_weigths, seeds)
