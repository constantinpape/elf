import multiprocessing
import numpy as np
import nifty.graph.rag as nrag


def visualise_edges(rag, edge_values,
                    edge_direction=2, ignore_edges=None,
                    n_threads=None):
    """ Visualize values mapped to the edges of a rag as volume.

    Arguments:
        rag [nifty.rag] - region adjacency graph
        edge_values [np.ndarray] - values mapped to rag edges
        edge_direction [int] - direction into which the edges will be drawn:
            0 - drawn in both directions
            1 - drawn in negative direction
            2 - drawn in positive direction
        ignore_edges [np.ndarray]: mask or indices of edges that should not be drawn
        n_threads [int] - number of threads (default: None)

    Returns:
        np.ndarray - edge volume
    """
    assert rag.numberOfEdges == len(edge_values), "%i, %i" % (rag.numberOfEdges,
                                                              len(edge_values))
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads

    edge_builder = nrag.ragCoordinates(rag, numberOfThreads=n_threads)
    if ignore_edges is None:
        edge_values_ = edge_values
    else:
        edge_values_ = edge_values.copy()
        edge_values_[ignore_edges] = 0
    edge_vol = edge_builder.edgesToVolume(edge_values_, edgeDirection=edge_direction)
    return edge_vol


def _scale_values(values, threshold, invert):
    if invert:
        values = threshold - values
    values -= values.min()
    values /= values.max()
    return values


def visualise_attractive_and_repulsive_edges(rag, edge_values, threshold,
                                             large_values_are_attractive=True, edge_direction=2,
                                             ignore_edges=None, n_threads=None):
    """ Visualize values mapped to the edges of a rag that are attractive and repulsive.

    Arguments:
        rag [nifty.rag] - region adjacency graph
        edge_values [np.ndarray] - values mapped to rag edges
        threshold [float] - values below this threhold are repulsive, above repulsive
        large_values_are_attractive [bool] - are large values or small values attractive? (default: True)
        edge_direction [int] - direction into which the edges will be drawn: (default: 2)
            0 - drawn in both directions
            1 - drawn in negative direction
            2 - drawn in positive direction
        ignore_edges [np.ndarray]: mask or indices of edges that should not be drawn
        n_threads [int] - number of threads (default: None)

    Returns:
        np.ndarray - volume of attractive edges
        np.ndarray - volume of repulsive edges
    """
    assert rag.numberOfEdges == len(edge_values), "%i, %i" % (rag.numberOfEdges,
                                                              len(edge_values))
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    edge_builder = nrag.ragCoordinates(rag, numberOfThreads=n_threads)

    if ignore_edges is None:
        edge_values_ = edge_values
    else:
        edge_values_ = edge_values.copy()
        edge_values_[ignore_edges] = threshold

    # find and normalize the attractive edge values
    attractive_edge_values = np.zeros_like(edge_values)
    if large_values_are_attractive:
        attractive_edges = edge_values_ > threshold
        attractive_edge_values[attractive_edges] = _scale_values(edge_values_[attractive_edges],
                                                                 threshold, False)
    else:
        attractive_edges = edge_values_ < threshold
        attractive_edge_values[attractive_edges] = _scale_values(edge_values_[attractive_edges],
                                                                 threshold, True)
    edge_vol_attractive = edge_builder.edgesToVolume(attractive_edge_values,
                                                     edgeDirection=edge_direction)

    # find and normalize the repulsive edge values
    repulsive_edge_values = np.zeros_like(edge_values)
    if large_values_are_attractive:
        repulsive_edges = edge_values_ < threshold
        repulsive_edge_values[repulsive_edges] = _scale_values(edge_values_[repulsive_edges],
                                                               threshold, True)
    else:
        repulsive_edges = edge_values_ > threshold
        repulsive_edge_values[repulsive_edges] = _scale_values(edge_values_[repulsive_edges],
                                                               threshold, False)
    edge_vol_repulsive = edge_builder.edgesToVolume(repulsive_edge_values,
                                                    edgeDirection=edge_direction)

    return edge_vol_attractive, edge_vol_repulsive
