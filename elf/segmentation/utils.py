import os
import urllib.request
from shutil import move
from typing import Optional, Tuple
from zipfile import ZipFile

import h5py
import numpy as np
import nifty
import nifty.ground_truth as ngt
import nifty.ufd as nufd
import pandas as pd
from scipy.ndimage import convolve, distance_transform_edt
from vigra.analysis import relabelConsecutive


#
# multicut and mutex watershed utils
#

def load_mutex_watershed_problem(split="test", prefix=None):
    """@private
    """
    assert split in ("test", "train")
    url = "https://oc.embl.de/index.php/s/sXJzYVK0xEgowOz/download"
    if prefix is None:
        prefix = "mutex_example_"

    train_path, test_path = f"{prefix}train.h5", f"{prefix}test.h5"
    if not (os.path.exists(train_path) and os.path.exists(test_path)):

        zip_path = "mutex_example.zip"
        print("Download mutex example data")
        with urllib.request.urlopen(url) as f:
            problem = f.read()
        with open(zip_path, "wb") as f:
            f.write(problem)

        for name_in_zip, out_path in zip(
            ["isbi_train_volume.h5", "isbi_test_volume.h5"], [train_path, test_path]
        ):
            with ZipFile(zip_path, "r") as f:
                f.extract(name_in_zip)
            move(name_in_zip, out_path)

        os.remove(zip_path)

    path = test_path if split == "test" else train_path
    with h5py.File(path, "r") as f:
        affs = f["affinities"][:]
    offsets = [
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
        [0, -9, 0], [0, 0, -9],
        [0, -9, -9], [0, 9, -9], [0, -9, -4],
        [0, -4, -9], [0, 4, -9], [0, 9, -4],
        [0, -27, 0], [0, 0, -27]
    ]

    return affs, offsets


def load_multicut_problem(
    sample: str, size: str, path: Optional[str] = None
) -> Tuple[nifty.graph.UndirectedGraph, np.ndarray]:
    """Load example multicut problems.

    These problems were introduced in
    "Solving large multicut problems for connectomics via domain decomposition":
    https://openaccess.thecvf.com/content_ICCV_2017_workshops/w1/html/Pape_Solving_Large_Multicut_ICCV_2017_paper.html

    Args:
        sample: The sample for this problem, one of "A" "B" or "C".
        size: The size of this problem, one of "small" or "medium".
        path: Where to save the problem file.

    Returns:
        The graph of the problem.
        The costs of the problem.
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
    path = f"{size}_problem_sample{sample}" if path is None else path
    if not os.path.exists(path):
        with urllib.request.urlopen(url) as f:
            problem = f.read().decode("utf-8")
        with open(path, "w") as f:
            f.write(problem)

    problem = np.genfromtxt(path)
    uv_ids = problem[:, :2].astype("uint64")
    n_nodes = int(uv_ids.max()) + 1
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)
    costs = problem[:, -1].astype("float32")

    return graph, costs


def analyse_multicut_problem(graph, costs, verbose=True, cost_threshold=0, topk=5):
    """@private
    """
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
    topk_rel_sizes = component_sizes[:topk].astype("float32") / n_nodes

    # Add partial optimality analysis from http://proceedings.mlr.press/v80/lange18a.html ?
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
    columns = ["n_nodes", "n_edges",
               "max_cost", "min_cost", "mean_cost", "std_cost",
               "n_components", "cost_threshold"]
    columns.extend([f"relative_size_top{i+1}_component" for i in range(len(topk_rel_sizes))])
    df = pd.DataFrame(data=[data], columns=columns)
    return df


def analyse_lifted_multicut_problem(graph, costs, lifted_uvs, lifted_costs):
    """@private
    """
    pass


def parse_visitor_output(output):
    """@private
    """
    if os.path.isfile(output):
        with open(output) as f:
            output = f.read()
    data = []
    for line in output.split("\n"):
        if not line.startswith("E:"):
            continue
        line = line.split()
        data.append([float(line[1]), float(line[3]), float(line[5])])
    columns = ["energy", "runtime solver [s]", "runtime total [s]"]
    return pd.DataFrame(data=data, columns=columns)


#
# misc utils
#


def compute_maximum_label_overlap(seg_a: np.ndarray, seg_b: np.ndarray, ignore_zeros: bool = False) -> np.ndarray:
    """Compute the node overlaps between objects in two segmentation.

    For each node in seg_a, compute the node in seg_b with the biggest overalp.

    Args:
        seg_a: The first segmentation.
        seg_b: The second segmentation.
        ignore_zeros: Whether to ignore overlap with zero (background id).

    Returns:
        The overlaps.
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


def normalize_input(input_: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Cast input to float and normalize to range [0, 1]

    Args:
        input_: The input data to be normalized.
        eps: Epsilon for numerical stability.

    Returns:
        The normalized input.
    """
    input_ = input_.astype("float32")
    input_ -= input_.min()
    input_ /= (input_.max() + eps)
    return input_


def map_background_to_zero(seg, background_label=None):
    """@private
    """
    if background_label is None:
        ids, sizes = np.unique(seg, return_counts=True)
        background_label = ids[np.argmax(sizes)]
    zero_mask = seg == 0
    seg[seg == background_label] = 0
    if zero_mask.sum():
        seg[zero_mask] = background_label
    return seg


#
# segmentation boundary utils
#


def sharpen_edges(edges: np.ndarray, percentile: int = 95, clip: bool = True) -> np.ndarray:
    """Sharpen the edges by dividing with a high percentile.

    Args:
        edges: The edge map, edges should have the value 1.
        percentile: The percentile for computing the normalization factor.
        clip: Wheter to clip the result to the range [0, 1].

    Returns:
        The sharpened edge map.
    """
    edges /= np.percentile(edges, percentile)
    if clip:
        edges = np.clip(edges, 0.0, 1.0)
    return edges


def smooth_edges(edges: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Smooth edges, e.g. from "seg_to_edges", by applying negative exponential distance transform.

    Args:
        edges: The edge map, edges should have the value 1.
        gain: Gain factor in the smoothing exponent.

    Returns:
        The smoothed edge map.
    """
    distances = distance_transform_edt(1 - edges)
    return np.exp(-gain * distances)


def seg_to_edges(segmentation: np.ndarray, only_in_plane_edges: bool = False) -> np.ndarray:
    """Compute edges from segmentation.

    Args:
        segmentation: The input segmentation.
        only_in_plane_edges: Whether to compute only in-plane edges for 3d segmentation.

    Returns:
        The edges of the segmentation.
    """
    if segmentation.ndim == 2:
        return make_2d_edges(segmentation)
    elif segmentation.ndim == 3:
        return make_3d2d_edges(segmentation) if only_in_plane_edges else make_3d_edges(segmentation)
    else:
        raise ValueError("Invalid input dimension %i" % segmentation.ndim)


def make_3d_edges(segmentation):
    """@private
    """
    # NOTE we add one here to make sure that we don't have zero in the segmentation
    gz = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1, 1))
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3, 1))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 1, 3))
    return ((gx ** 2 + gy ** 2 + gz ** 2) > 0)


def make_3d2d_edges(segmentation):
    """@private
    """
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3, 1))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 1, 3))
    return ((gx ** 2 + gy ** 2) > 0)


def make_2d_edges(segmentation):
    """@private
    """
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1))
    return ((gx ** 2 + gy ** 2) > 0)
