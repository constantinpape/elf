import multiprocessing
import nifty.graph.rag as nrag


def compute_rag(segmentation, n_threads=None):
    """ Compute region adjacency graph of segmentation.

    Arguments:
        segmentation [np.ndarray] - the segmentation
        n_threads [int] - number of threads used, set to cpu count by default. (default: None)
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    n_labels = int(segmentation.max()) + 1
    rag = nrag.gridRag(segmentation, numberOfLabels=n_labels,
                       numberOfThreads=n_threads)
    return rag


def compute_boundary_features(rag, boundary_map,
                              min_value=0, max_value=1, n_threads=None):
    """ Compute edge features from boundary map.

    Arguments:
        rag [RegionAdjacencyGraph] - region adjacency graph
        boundary_map [np.ndarray] - boundary map.
        min_value [float] - minimum value used in accumulation (default: 0)
        max_value [float] - maximum value used in accumulation (default: 1)
        n_threads [int] - number of threads used, set to cpu count by default. (default: None)
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if tuple(rag.shape) != boundary_map.shape:
        raise ValueError("Incompatible shapes: %s, %s" % (str(rag.shape),
                                                          str(boundary_map.shape)))
    features = nrag.accumulateEdgeStandartFeatures(rag, boundary_map,
                                                   min_value, max_value,
                                                   numberOfThreads=n_threads)
    return features


def compute_affinity_features(rag, affinity_map, offsets,
                              min_value=0, max_value=1, n_threads=None):
    """ Compute edge features from affinity map.

    Arguments:
        rag [RegionAdjacencyGraph] - region adjacency graph
        boundary_map [np.ndarray] - boundary map.
        min_value [float] - minimum value used in accumulation (default: 0)
        max_value [float] - maximum value used in accumulation (default: 1)
        n_threads [int] - number of threads used, set to cpu count by default. (default: None)
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if tuple(rag.shape) != affinity_map.shape[1:]:
        raise ValueError("Incompatible shapes: %s, %s" % (str(rag.shape),
                                                          str(affinity_map.shape[1:])))
    if len(offsets) != affinity_map.shape[0]:
        raise ValueError("Incompatible number of channels and offsets: %i, %i" % (len(offsets),
                                                                                  affinity_map.shape[0]))
    features = nrag.accumulateAffinityStandartFeatures(rag, affinity_map, offsets,
                                                       min_value, max_value,
                                                       numberOfThreads=n_threads)
    return features


def compute_boundary_mean_and_length(rag, input_, n_threads=None):
    """ Compute mean value and length of boundaries.

    Arguments:
        rag [RegionAdjacencyGraph] - region adjacency graph
        input_ [np.ndarray] - input map.
        n_threads [int] - number of threads used, set to cpu count by default. (default: None)
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if tuple(rag.shape) != input_.shape:
        raise ValueError("Incompatible shapes: %s, %s" % (str(rag.shape),
                                                          str(input_.shape)))
    features = nrag.accumulateEdgeMeanAndLength(rag, input_, numberOfThreads=n_threads)
    return features


# TODO
def compute_region_features(rag, input_map, segmentation, n_threads=None):
    """ Compute edge features from input accumulated over segments.

    Arguments:
        rag [RegionAdjacencyGraph] - region adjacency graph
        input_map [np.ndarray] - boundary map.
        segmentation [np.ndarray] - segmentation.
        n_threads [int] - number of threads used, set to cpu count by default. (default: None)
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads


def project_node_labels_to_pixels(rag, node_labels, n_threads=None):
    """ Project label values for graph nodes back to pixels to obtain segmentation.

    Arguments:
        rag [RegionAdjacencyGraph] - region adjacency graph
        node_labels [np.ndarray] - array with node labels
        n_threads [int] - number of threads used, set to cpu count by default. (default: None)
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if len(node_labels) != rag.numberOfNodes:
        raise ValueError("Incompatible number of node labels: %i, %i" % (len(node_labels),
                                                                         rag.numberOfNodes))
    seg = nrag.projectScalarNodeDataToPixels(rag, node_labels,
                                             numberOfThreads=n_threads)
    return seg


def feats_to_costs_default():
    pass


def lifted_problem_from_probabilities(rag, watershed, input_maps,
                                      assignment_threshold, graph_depth=None,
                                      problem_to_cost=feats_to_costs_default,
                                      n_threads=None):
    """
    """


def lifted_problem_from_segmentation(rag, watershed, input_segmentation,
                                     overlap_threshold, same_segment_cost,
                                     different_segment_cost, graph_depth=None):
    """
    """
