import os
import urllib.request

import numpy as np
import nifty
import nifty.ground_truth as ngt
import nifty.ufd as nufd
import pandas as pd
from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt
from vigra.analysis import relabelConsecutive


#
# multicut utils
#


# add large problems?
def load_multicut_problem(sample, size, path=None):
    """ Load example multicut problems.

    These problems were introduced in
    "Solving large multicut problems for connectomics via domain decomposition":
    https://openaccess.thecvf.com/content_ICCV_2017_workshops/w1/html/Pape_Solving_Large_Multicut_ICCV_2017_paper.html

    Arguments:
        sample [str] - the sample for this problem, 'A' 'B' or 'C'
        size [str] - the size for this problem, 'small' or 'medium'
        path [str] - where to save the problem file (default: None)
    """
    problems = {
        "A": {
            "small": "https://oc.embl.de/index.php/s/yVKwyQ8VoPXYkft/download",
            "medium": "https://oc.embl.de/index.php/s/ztnwjmv0bmd3mnS/download"
        },
        "B": {
            "small": "https://oc.embl.de/index.php/s/QKYA2EoMXqxQuO4/download",
            "medium": "https://oc.embl.de/index.php/s/yuk7VwCvgZC017q/download"
        },
        "C": {
            "small": "https://oc.embl.de/index.php/s/eDZprDwT2cXFAe0/download",
            "medium": "https://oc.embl.de/index.php/s/hGyqlkenHfsq5P4/download"
        }
    }
    assert sample in problems
    assert size in problems[sample]
    url = problems[sample][size]
    path = f'{size}_problem_sample{sample}' if path is None else path
    if not os.path.exists(path):
        with urllib.request.urlopen(url) as f:
            problem = f.read().decode('utf-8')
        with open(path, 'w') as f:
            f.write(problem)

    problem = np.genfromtxt(path)
    uv_ids = problem[:, :2].astype('uint64')
    n_nodes = int(uv_ids.max()) + 1
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)
    costs = problem[:, -1].astype('float32')

    return graph, costs


def analyse_multicut_problem(graph, costs, verbose=True, cost_threshold=0, topk=5):
    # problem size and cost summary
    n_nodes, n_edges = graph.numberOfNodes, graph.numberOfEdges
    min_cost, max_cost = costs.min(), costs.max()
    mean_cost, std_cost = costs.mean(), costs.std()

    # component analysis
    merge_edges = costs > cost_threshold
    ufd = nufd.ufd(n_nodes)
    uv_ids = graph.uvIds()
    ufd.merge(uv_ids[merge_edges])
    cc_labels = ufd.elementLabeling()
    cc_labels, max_id, _ = relabelConsecutive(cc_labels, start_label=0,
                                              keep_zeros=False)
    n_components = max_id + 1
    _, component_sizes = np.unique(cc_labels, return_counts=True)
    component_sizes = np.sort(component_sizes)[::-1]
    topk_rel_sizes = component_sizes[:topk].astype('float32') / n_nodes

    # TODO add partial optimality analysis from
    # http://proceedings.mlr.press/v80/lange18a.html
    if verbose:
        print("Analysis of multicut problem:")
        print("The graph has", n_nodes, "nodes and", n_edges, "edges")
        print("The costs are in range", min_cost, "to", max_cost)
        print("The mean cost is", mean_cost, "+-", std_cost)
        print("The problem decomposes into", n_components, "components at threshold", cost_threshold)
        print("The 5 largest components have the following sizes:", topk_rel_sizes)

    data = [n_nodes, n_edges,
            max_cost, min_cost, mean_cost, std_cost,
            n_components, cost_threshold]
    data.extend(topk_rel_sizes.tolist())
    columns = ['n_nodes', 'n_edges',
               'max_cost', 'min_cost', 'mean_cost', 'std_cost',
               'n_components', 'cost_threshold']
    columns.extend([f'relative_size_top{i+1}_component' for i in range(len(topk_rel_sizes))])
    df = pd.DataFrame(data=[data], columns=columns)
    return df


# TODO
def analyse_lifted_multicut_problem(graph, costs, lifted_uvs, lifted_costs):
    pass


def parse_visitor_output(output):
    if os.path.isfile(output):
        with open(output) as f:
            output = f.read()
    data = []
    for line in output.split('\n'):
        if not line.startswith('E:'):
            continue
        line = line.split()
        data.append([float(line[1]), float(line[3]), float(line[5])])
    columns = ['energy', 'runtime solver [s]', 'runtime total [s]']
    return pd.DataFrame(data=data, columns=columns)


#
# misc utils
#


def compute_maximum_label_overlap(seg_a, seg_b, ignore_zeros=False):
    """ For each node in seg_a, compute the node in seg_b with
    the biggest overalp.
    """
    ids_a = np.unique(seg_a)
    overlaps = np.zeros(int(ids_a.max() + 1), dtype=seg_b.dtype)

    ovlp = ngt.overlap(seg_a, seg_b)
    ovlp = [ovlp.overlapArrays(id_a, True)[0] for id_a in ids_a]
    if ignore_zeros:
        ovlp = np.array([ovl[1] if (ovl[0] == 0 and len(ovl) > 1) else ovl[0] for ovl in ovlp])
    else:
        ovlp = np.array([ovl[0] for ovl in ovlp])

    overlaps[ids_a] = ovlp
    return overlaps


def normalize_input(input_, eps=1e-7):
    """ Cast input to float and normalize to range [0, 1]

    Arguments:
        input_ [np.ndarray] - input tensor to be normalized
        eps [float] - epsilon for numerical stability (default: 1e-7)
    """
    input_ = input_.astype('float32')
    input_ -= input_.min()
    input_ /= (input_.max() + eps)
    return input_


#
# segmentation boundary utils
#


def smooth_edges(edges, gain=1.):
    """ Smooth edges, e.g. from 'seg_to_edges'
    by applying negative exponential distance transform.

    Arguments:
        edges [np.ndarray] - tensor with edges; expects edges to be 1
        gain [float] - gain factor in the exponent (default: 1.)
    """
    distances = distance_transform_edt(1 - edges)
    return np.exp(-gain * distances)


def seg_to_edges(segmentation, only_in_plane_edges=False):
    """ Compute edges from segmentation.

    Arguments:
        segmentation [np.ndarray] - input segmentation
        only_in_plane_edges [bool] - compute only in-plane edges
            for 3d segmentation (default: False)
    """
    if segmentation.ndim == 2:
        return make_2d_edges(segmentation)
    elif segmentation.ndim == 3:
        return make_3d2d_edges(segmentation) if only_in_plane_edges else\
            make_3d_edges(segmentation)
    else:
        raise ValueError("Invalid input dimension %i" % segmentation.ndim)


def make_3d_edges(segmentation):
    """ Make 3d edge volume from 3d segmentation
    """
    # NOTE we add one here to make sure that we don't have zero in the segmentation
    gz = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1, 1))
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3, 1))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 1, 3))
    return ((gx ** 2 + gy ** 2 + gz ** 2) > 0)


def make_3d2d_edges(segmentation):
    """ Make 3d edge volume from 3d segmentation
        but only compute the edges in xy-plane.
        This may be more appropriate for anisotropic volumes."""
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3, 1))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 1, 3))
    return ((gx ** 2 + gy ** 2) > 0)


def make_2d_edges(segmentation):
    """ Make 2d edges from 2d segmentation
    """
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1))
    return ((gx ** 2 + gy ** 2) > 0)
