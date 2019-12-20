import nifty.graph.rag as nrag


def visualise_edges(rag, edge_values,
                    edge_direction=2, ignore_edges=None):
    """ Visualize values mapped to the edges of a rag as volume.

    Arguments:
        rag [nifty.rag] - region adjacency graph
        edge_values [np.ndarray] - values mapped to rag edges
        edge_direction [int] - direction into which the edges will be drawn:
            0 - drawn in both directions
            1 - drawn in negative direction
            2 - drawn in positive direction
        ignore_edges [np.ndarray]: mask or indices of edges that should not be drawn

    Returns:
        np.ndarray - volume of attractive edges
        np.ndarray - volume of repulsive edges
    """
    assert rag.numberOfEdges == len(edge_values), "%i, %i" % (rag.numberOfEdges,
                                                              len(edge_values))
    edge_builder = nrag.ragCoordinates(rag)
    if ignore_edges is None:
        edge_values_ = edge_values
    else:
        edge_values_ = edge_values.copy()
        edge_values_[ignore_edges] = 0
    edge_vol = edge_builder.edgesToVolume(edge_values_, edgeDirection=edge_direction)
    return edge_vol


def visualise_attractive_and_repulsive_edges(rag, edge_values, threshold,
                                             edge_direction=2, ignore_edges=None):
    """ Visualize values mapped to the edges of a rag that are attractive and repulsive.

    Arguments:
        rag [nifty.rag] - region adjacency graph
        edge_values [np.ndarray] - values mapped to rag edges
        threshold [float] - values below this threhold are repulsive, above repulsive
        edge_direction [int] - direction into which the edges will be drawn:
            0 - drawn in both directions
            1 - drawn in negative direction
            2 - drawn in positive direction
        ignore_edges [np.ndarray]: mask or indices of edges that should not be drawn

    Returns:
        np.ndarray - volume of attractive edges
        np.ndarray - volume of repulsive edges
    """
    assert rag.numberOfEdges == len(edge_values), "%i, %i" % (rag.numberOfEdges,
                                                              len(edge_values))
    edge_builder = nrag.ragCoordinates(rag)

    if ignore_edges is None:
        edge_values_ = edge_values
    else:
        edge_values_ = edge_values.copy()
        edge_values_[ignore_edges] = threshold

    # find and normalize the attractive edge values
    attractive_edges = edge_values_ > threshold
    attractive_edge_values = edge_values_.copy()
    attractive_edge_values[~attractive_edges] = 0
    attractive_edge_values /= attractive_edge_values.max()
    edge_vol_attractive = edge_builder.edgesToVolume(attractive_edge_values,
                                                     edgeDirection=edge_direction)

    # find and normalize the repulsive edge values
    repulsive_edges = edge_values_ < threshold
    repulsive_edge_values = edge_values_.copy()
    repulsive_edge_values = threshold - repulsive_edge_values
    repulsive_edge_values[~repulsive_edges] = 0
    repulsive_edge_values /= repulsive_edge_values.max()
    edge_vol_repulsive = edge_builder.edgesToVolume(repulsive_edge_values,
                                                    edgeDirection=edge_direction)
    return edge_vol_attractive, edge_vol_repulsive
