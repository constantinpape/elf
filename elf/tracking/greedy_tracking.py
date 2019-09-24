import numpy as np


def find_track_edges(edges):
    """
    """
    track_edges = []

    return track_edges


def greedy_tracking(segmentation, edges, edge_weights, threshold):
    """ Greedy lineage tracking in a volume with 2d segmentations.

    Based on https://github.com/funkey/flywing/03_process/greedy_track.py

    Arguments:
        segmentation [np.ndarray] - volume with slice-wise (2d) segmentations.
            segmentation ids need to be independent between slices.
        edges [list[np.ndarray]] - list with edges per time frame transition.
        edge_weights [list[np.ndarray]] - list with edge weights per timeframe transition.
        threshold [float] - threshold for edge weights to be considered.

    Returns:
    """
    filtered_edges = [[(w, u, v) for (u, v), w in zip(ee, ww) if w > threshold]
                      for ee, ww in zip(edges, edge_weights)]
    track_edges = [find_track_edges(e) for e in filtered_edges]

    graph = ''
    segmentation = ''

    return segmentation, graph
