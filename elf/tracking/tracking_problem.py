import multiprocessing
from concurrent import futures

import numpy as np
import nifty.graph.rag as nrag

from ..segmentation.multicut import transform_probabilities_to_costs


def tracking_problem(segmentation, affinities,
                     affinity_direction=-1, to_costs=False,
                     weight_edges=False, n_threads=None):
    """ Compute edges and weights between time-frames.

    Arguments:
        segmentation [np.ndarray] - volume with 2d segmentation corresponding to time-frames.
        affinities [np.ndarray] - time affinities encoding the transition probabilities.
        affinity_direction [int] - direction of affinities, 1 or -1. (default: -1)
        to_costs [bool] - transform probabilities to costs for multicut. (default: False)
        weight_edges [bool] - weight costs by edge sizes (only relevant if to_costs). (default: False)
        n_threads [int] - number of threads used for computation. (default: None)
    """
    if affinity_direction not in (-1, 1):
        raise ValueError("Invalid affinity direction %i" % affinity_direction)
    if segmentation.shape != affinities.shape:
        raise ValueError("Incompatible shapes: %s, %s" % (str(segmentation.shape),
                                                          str(affinities.shape)))
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads

    def compute_time_edges(t):
        # get the segmentation of timesteps t and t + 1
        seg0, seg1 = segmentation[t], segmentation[t + 1]

        # make sure that seg ids are consistent
        max0, min1 = seg0.max(), seg1.min()
        if max0 >= min1:
            raise RuntimeError("Inconsistent seg-ids in timesteps %i,%i: %i,%i" % (t, t + 1,
                                                                                   max0, min1))

        # get the corresponding affinities
        t_aff = t if affinity_direction == 1 else t + 1
        aff = affinities[t_aff]

        seg = np.stack([seg0, seg1])
        aff = np.stack([aff, aff])

        # compute rag and features
        rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1,
                           numberOfThreads=1)
        features = nrag.accumulateEdgeMeanAndLength(rag, aff,
                                                    numberOfThreads=1)

        # only keep edges between slices
        edges = rag.uvIds()
        edge_mask = (edges > max0).sum(axis=1) == 1

        edges = edges[edge_mask]
        features = features[edge_mask]
        assert len(edges) == len(features)
        edge_weights = features[:, 0]

        if to_costs and not weight_edges:
            edge_weights = transform_probabilities_to_costs(edge_weights)
        elif to_costs and weight_edges:
            edge_sizes = features[:, 1]
            edge_weights = transform_probabilities_to_costs(edge_weights,
                                                            edge_sizes=edge_sizes)
        return edges, edge_weights

    n_timesteps = segmentation.shape[0]
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(t) for t in range(n_timesteps - 1)]
        results = [t.result() for t in tasks]

    edges = [res[0] for res in results]
    edge_weights = [res[1] for res in results]
    return edges, edge_weights
