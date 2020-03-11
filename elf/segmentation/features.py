import multiprocessing

import numpy as np
import vigra
import nifty.graph.rag as nrag
import nifty.distributed as ndist
import nifty.ground_truth as ngt

try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff

from .multicut import transform_probabilities_to_costs


#
# Region Adjacency Graph and Features
#

def compute_rag(segmentation, n_labels=None, n_threads=None):
    """ Compute region adjacency graph of segmentation.

    Arguments:
        segmentation [np.ndarray] - the segmentation
        n_labels [int] - number of  labels in segmentation.
            If None is give, will be computed from the data. (default: None)
        n_threads [int] - number of threads used, set to cpu count by default. (default: None)
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    n_labels = int(segmentation.max()) + 1 if n_labels is None else n_labels
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
def compute_boundary_features_with_filters(rag, input_, apply_2d=False, n_threads=None,
                                           filters={'gaussianSmoothing': [],
                                                    'laplacianOfGaussian': [],
                                                    'hessianOfGaussianEigenvalues': []}):
    """ Compute boundary features accumulated over filter responses on input.

    Arguments:
        rag [RegionAdjacencyGraph] - region adjacency graph
        input_ [np.ndarray] - input data
        apply_2d [bool] - whether to apply the filters in 2d for 3d input data (default: bool)
        n_threads [int] - number of threads (default: None)
        filters [dict] - the filters to apply, expects a
            dictionary mapping filter names to sigma values (default: default_Filters)
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads


# TODO
def compute_region_features(rag, input_map, segmentation, n_threads=None):
    """ Compute edge features from input accumulated over segments.

    Arguments:
        rag [RegionAdjacencyGraph] - region adjacency graph
        input_ [np.ndarray] - input data.
        segmentation [np.ndarray] - segmentation.
        n_threads [int] - number of threads used, set to cpu count by default. (default: None)
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads


#
# Lifted Features
#

def feats_to_costs_default(lifted_labels, lifted_features):
    # we assume that we only have different classes for a given lifted
    # edge here (mode = 'different') and then set all edges to be repulsive

    # the higher the class probability, the more repulsive the edges should be,
    # so we just multiply both probabilities
    lifted_costs = lifted_features[:, 0] * lifted_features[:, 1]
    lifted_costs = transform_probabilities_to_costs(lifted_costs)
    return lifted_costs


def lifted_problem_from_probabilities(rag, watershed, input_maps,
                                      assignment_threshold, graph_depth,
                                      feats_to_costs=feats_to_costs_default,
                                      mode='different', n_threads=None):
    """ Compute lifted problem from probability maps by mapping them to superpixels.

    Arguments:
        rag [RegionAdjacencyGraph] - the region adjacency graph
        watershed [np.ndarray] - the watershed over segmentation
        input_maps [list[np.ndarray]] - list of probability maps. Each
            map must have the same shape as the watersheds and each map is
            treated as the probability to correspond to a different class.
        assignment_threshold [float] - minimal expression level to assign a
            class to a graph node (= watershed segment)
        graph_depth [int] - maximal graph depth up to which
            lifted edges will be included
        feats_to_costs [callable] - function to calculate the lifted costs from the
            class assignment probabilities. This becomes as inputs 'lifted_labels',
            which stores the two classes assigned to a lifted edge, and `lifted_features`,
            which stores the two assignment probabilities. (default: feats_to_costs_default).
        mode [str] - mode for insertion of lifted edges. Can be
            "all" - lifted edges will be inserted in between all nodes with attribution
            "different" - lifted edges will only be inserted in between nodes attributed to different classes
            "same" - lifted edges will only be inserted in between nodes attribted to the same class
            (default: "different")
        n_threads [int] - number of threads used for the calculation (default: None)
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    # validate inputs
    assert isinstance(input_maps, (list, tuple))
    assert all(isinstance(inp, np.ndarray) for inp in input_maps)
    shape = watershed.shape
    assert all(inp.shape == shape for inp in input_maps)

    # map the probability maps to superpixels - we only map to superpixels which
    # have a larger mean expression than `assignment_threshold`

    # TODO handle the dtype conversion for vigra gracefully somehow ...
    # think about supporting uint8 input and normalizing

    # TODO how do we handle cases where the same superpixel is mapped to
    # more than one class ?

    n_nodes = int(watershed.max()) + 1
    node_labels = np.zeros(n_nodes, dtype='uint64')
    node_features = np.zeros(n_nodes, dtype='float32')
    # TODO we could allow for more features that could then be used for the cost estimation
    for class_id, inp in enumerate(input_maps):
        mean_prob = vigra.analysis.extractRegionFeatures(inp, watershed, features=['mean'])['mean']
        # we can in principle map multiple classes here, and right now will just override
        class_mask = mean_prob > assignment_threshold
        node_labels[class_mask] = class_id
        node_features[class_mask] = mean_prob[class_mask]

    # find all lifted edges up to the graph depth between mapped nodes
    # NOTE we need to convert to the different graph type for now, but
    # it would be nice to support all nifty graphs at some type
    uv_ids = rag.uvIds()
    g_temp = ndist.Graph(uv_ids)

    lifted_uvs = ndist.liftedNeighborhoodFromNodeLabels(g_temp, node_labels, graph_depth, mode=mode,
                                                        numberOfThreads=n_threads, ignoreLabel=0)
    lifted_labels = node_labels[lifted_uvs]
    lifted_features = node_features[lifted_uvs]

    lifted_costs = feats_to_costs(lifted_labels, lifted_features)
    return lifted_uvs, lifted_costs


# TODO support setting costs proportional to overlaps
def lifted_problem_from_segmentation(rag, watershed, input_segmentation,
                                     overlap_threshold, graph_depth, same_segment_cost,
                                     different_segment_cost,
                                     mode='all', n_threads=None):
    """ Compute lifted problem from segmentation by mapping segments to
        watershed superpixels.

    Arguments:
        rag [RegionAdjacencyGraph] - the region adjacency graph
        watershed [np.ndarray] - the watershed over segmentation
        input_segmentation [np.ndarray] - Segmentation used to determine node attribution.
        overlap_threshold [float] - minimal overlap to assign a segment id to node
        graph_depth [int] - maximal graph depth up to which
            lifted edges will be included
        same_segment_cost [float] - costs for edges between nodes with same segment id attribution
        different_segment_cost [float] - costs for edges between nodes with different segment id attribution
        mode [str] - mode for insertion of lifted edges. Can be
            "all" - lifted edges will be inserted in between all nodes with attribution
            "different" - lifted edges will only be inserted in between nodes attributed to different classes
            "same" - lifted edges will only be inserted in between nodes attribted to the same class
            (default: "different")
        n_threads [int] - number of threads used for the calculation (default: None)
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    assert input_segmentation.shape == watershed.shape

    # compute the overlaps
    ovlp_comp = ngt.overlap(watershed, input_segmentation)
    ws_ids = np.unique(watershed)
    n_labels = ws_ids[-1] + 1
    assert n_labels == rag.numberOfNodes, "%i, %i" % (n_labels, rag.numberOfNodes)

    # initialise the arrays for node labels, to be
    # dense in the watershed id space (even if some ws-ids are not present)
    node_labels = np.zeros(n_labels, dtype='uint64')

    # extract the overlap values and node labels from the overlap
    # computation results
    overlaps = [ovlp_comp.overlapArraysNormalized(ws_id, sorted=False)
                for ws_id in ws_ids]
    node_label_vals = np.array([ovlp[0][0] for ovlp in overlaps])
    overlap_values = np.array([ovlp[1][0] for ovlp in overlaps])
    node_label_vals[overlap_values < overlap_threshold] = 0
    assert len(node_label_vals) == len(ws_ids)
    node_labels[ws_ids] = node_label_vals

    # find all lifted edges up to the graph depth between mapped nodes
    # NOTE we need to convert to the different graph type for now, but
    # it would be nice to support all nifty graphs at some type
    uv_ids = rag.uvIds()
    g_temp = ndist.Graph(uv_ids)

    lifted_uvs = ndist.liftedNeighborhoodFromNodeLabels(g_temp, node_labels, graph_depth, mode=mode,
                                                        numberOfThreads=n_threads, ignoreLabel=0)
    # make sure that the lifted uv ids are in range of the node labels
    assert lifted_uvs.max() < rag.numberOfNodes, "%i, %i" % (int(lifted_uvs.max()),
                                                             rag.numberOfNodes)
    lifted_labels = node_labels[lifted_uvs]
    lifted_costs = np.zeros_like(lifted_labels, dtype='float32')

    same_mask = lifted_labels[:, 0] == lifted_labels[:, 1]
    lifted_costs[same_mask] = same_segment_cost
    lifted_costs[~same_mask] = different_segment_cost

    return lifted_uvs, lifted_costs


#
# Misc
#

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


def compute_z_edge_mask(rag, watershed):
    """ Compute edge mask of in-between plane edges for flat superpixels.

    This function does not check wether the input watersheds are
    actually flat!
    """
    node_z_coords = np.zeros(rag.numberOfNodes, dtype='uint32')
    for z in range(watershed.shape[0]):
        node_z_coords[watershed[z]] = z
    uv_ids = rag.uvIds()
    z_edge_mask = node_z_coords[uv_ids[:, 0]] != node_z_coords[uv_ids[:, 1]]
    return z_edge_mask


# TODO
#
# Learning
#
