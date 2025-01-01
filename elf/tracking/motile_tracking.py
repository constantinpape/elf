"""Functionality for tracking microscopy data with [motile](https://github.com/funkelab/motile).
"""
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

try:
    import motile
    from motile import costs, constraints
except ImportError:
    motile, costs, constraints = None, None, None

from nifty.tools import takeDict
from skimage.measure import regionprops

from . import tracking_utils as utils


#
# Utility functionality for motile: Solution parsing and visualization.
#

def parse_result(solver, graph):
    """@private
    """
    lineage_graph = nx.DiGraph()

    node_indicators = solver.get_variables(motile.variables.NodeSelected)
    edge_indicators = solver.get_variables(motile.variables.EdgeSelected)

    # Build new graphs that contain the selected nodes and tracking / lineage results.
    for node, index in node_indicators.items():
        if solver.solution[index] > 0.5:
            lineage_graph.add_node(node, **graph.nodes[node])

    for edge, index in edge_indicators.items():
        if solver.solution[index] > 0.5:
            lineage_graph.add_edge(*edge, **graph.edges[edge])

    # Use connected components to find the lineages.
    lineages = nx.weakly_connected_components(lineage_graph)
    lineages = {lineage_id: list(lineage) for lineage_id, lineage in enumerate(lineages, 1)}
    return lineage_graph, lineages


def lineage_graph_to_track_graph(lineage_graph, lineages):
    """@private
    """
    # Create a new graph that only contains the tracks by not connecting nodes with a degree of 2.
    track_graph = nx.DiGraph()
    track_graph.add_nodes_from(lineage_graph.nodes)

    # Iterate over the edges to find splits and end tracks there.
    for (u, v), features in lineage_graph.edges.items():
        out_edges = lineage_graph.out_edges(u)
        # Normal track continuation
        if len(out_edges) == 1:
            track_graph.add_edge(u, v)
        # Otherwise track ends at division and we don't continue.

    # Use connected components to find the tracks.
    tracks = nx.weakly_connected_components(track_graph)
    tracks = {track_id: list(track) for track_id, track in enumerate(tracks, 1)}

    return track_graph, tracks


def get_node_assignment(node_ids, assignments):
    """@private
    """
    # Generate a dictionary that maps each node id (= segment id) to its assignment.
    node_assignment = {
        node_id: assignment_id for assignment_id, nodes in assignments.items() for node_id in nodes
    }

    # Everything that was not selected gets mapped to 0.
    not_selected = list(set(node_ids) - set(node_assignment.keys()))
    node_assignment.update({not_select: 0 for not_select in not_selected})

    return node_assignment


def recolor_segmentation(segmentation, node_to_assignment):
    """@private
    """
    # We need to add a value for mapping 0, otherwise the function fails.
    node_to_assignment_ = deepcopy(node_to_assignment)
    node_to_assignment_[0] = 0
    recolored_segmentation = takeDict(node_to_assignment_, segmentation)
    return recolored_segmentation


def create_data_for_track_layer(segmentation, lineage_graph, node_to_track, skip_zero=True):
    """@private
    """
    # Compute regionpros and extract centroids.
    props = regionprops(segmentation)
    centroids = {prop.label: prop.centroid for prop in props}

    # Create the track data representation for napari, which expects:
    # track_id, timepoint, (z), y, x
    track_data = np.array([
        [node_to_track[node_id]] + list(centroid) for node_id, centroid in centroids.items()
        if node_id in node_to_track
    ])
    if skip_zero:
        track_data = track_data[track_data[:, 0] != 0]

    # Order the track data by track_id and timepoint.
    sorted_indices = np.lexsort((track_data[:, 1], track_data[:, 0]))
    track_data = track_data[sorted_indices]

    # Create the parent graph for the tracks.
    parent_graph = {}
    for (u, v), features in lineage_graph.edges.items():
        out_edges = lineage_graph.out_edges(u)
        if len(out_edges) == 2:
            track_u, track_v = node_to_track[u], node_to_track[v]
            if skip_zero and (track_u == 0 or track_v == 0):
                continue
            parent_graph[track_v] = parent_graph.get(track_v, []) + [track_u]

    return track_data, parent_graph


#
# Utility functions for constructing motile tracking problems.
#


# We could expose further relevant weights and constants.
def construct_problem(
    segmentation: np.ndarray,
    node_costs: np.ndarray,
    edges_and_costs: List[Dict[str, Union[int, float]]],
    max_parents: int = 1,
    max_children: int = 2,
) -> Tuple["motile.solver.Solver", "motile.track_graph.TrackGraph"]:
    """Construct a motile tracking problem from a segmentation timeseries.

    Args:
        segmentation: The segmentation timeseries.
        node_costs: The node selection costs.
        edges_and_costs: The edge selection costs.
        max_parents: The maximal number of parents.
            Corresponding to the maximal number of edges to the previous time point.
        max_children: The maximal number of children.
            Corresponding to the maximal number of edges to the next time point.

    Returns:
        The motile solver.
        The motile tracking graph.
    """
    node_ids, indexes = np.unique(segmentation, return_index=True)
    indexes = np.unravel_index(indexes, shape=segmentation.shape)
    timeframes = indexes[0]

    # Get rid of 0.
    if node_ids[0] == 0:
        node_ids, timeframes = node_ids[1:], timeframes[1:]
    assert len(node_ids) == len(timeframes)

    graph = nx.DiGraph()
    # If the node function is not passed then we assume that all nodes should be selected.
    assert len(node_costs) == len(node_ids)
    nodes = [
        {"id": node_id, "score": score, "t": t} for node_id, score, t in zip(node_ids, node_costs, timeframes)
    ]

    graph.add_nodes_from([(node["id"], node) for node in nodes])
    graph.add_edges_from([(edge["source"], edge["target"], edge) for edge in edges_and_costs])

    # Get the tracking graph and the motile solver.
    graph = motile.TrackGraph(graph)
    solver = motile.Solver(graph)

    # We can do linear reweighting of the costs: a * x + b, where: a=weight, b=constant.
    solver.add_cost(costs.NodeSelection(weight=-1.0, attribute="score", constant=0))
    solver.add_cost(costs.EdgeSelection(weight=-1.0, attribute="score", constant=0))

    # Add the constraints: we allow for divisions (max childeren = 2).
    solver.add_constraint(constraints.MaxParents(max_parents))
    solver.add_constraint(constraints.MaxChildren(max_children))

    # Add costs for appearance and divisions.
    solver.add_cost(costs.Appear(constant=1.0))
    solver.add_cost(costs.Split(constant=1.0))

    return solver, graph


#
# Motile based tracking.
#


def _track_with_motile_impl(
    segmentation,
    relabel_segmentation=True,
    node_cost_function=None,
    edge_cost_function=None,
    node_selection_cost=0.95,
    **problem_kwargs,
):
    # Relabel the segmentation so that the ids are unique across time.
    # If `relabel_segmentation is False` the segmentation has to be in the correct format already.
    if relabel_segmentation:
        segmentation = utils.relabel_segmentation_across_time(segmentation)

    # Compute the node selection costs.
    # If `node_cost_function` is passed it is used to compute the costs.
    # Otherwise we set a fixed node selection cost.
    if node_cost_function is None:
        n_nodes = int(segmentation.max())
        node_costs = np.full(n_nodes, node_selection_cost)
    else:
        node_costs = node_cost_function(segmentation)

    # Compute the edges and edge selection cost.
    # If `edge_cost_function` is not given we use the default approach, based on overlap of adjacent slices.
    if edge_cost_function is None:
        edge_cost_function = utils.compute_edges_from_overlap
    edges_and_costs = edge_cost_function(segmentation)

    # Construct and solve the tracking problem.
    solver, graph = construct_problem(segmentation, node_costs, edges_and_costs, **problem_kwargs)
    solver.solve()

    return solver, graph, segmentation


def track_with_motile(
    segmentation: np.ndarray,
    relabel_segmentation: bool = True,
    node_cost_function: Optional[callable] = None,
    edge_cost_function: Optional[callable] = None,
    node_selection_cost: float = 0.95,
    **problem_kwargs,
) -> Tuple[np.ndarray, nx.DiGraph, Dict[int, List[int]], nx.DiGraph, Dict[int, List[int]]]:
    """Track segmented objects across time with motile.

    Args:
        segmentation: The input segmentation.
        relabel_segmentation: Whether to relabel the segmentation so that ids are unique across time.
            If set to False, then unique ids across time have to be ensured in the input.
        node_cost_function: Function for computing costs for node selection.
            If not given, then the constant factor `node_selection_cost` is used.
        edge_cost_function: Function for computing costs for edge selection.
            If not given, then the function `utils.compute_edges_from_overlap` is used.
        node_selection_cost: Node selection cost.
        problem_kwargs: Additional keyword arguments for constructing the tracking problem.

    Returns:
        The input segmentation after relabeling.
        The lineage graph, a directed graph that connects track ids across divisions or fusions.
        Map of lineage ids to track ids.
        The track graph, a directed graph that connects segmentation ids across time points.
        Map of track ids to segmentation ids.
    """
    if motile is None:
        raise RuntimeError("You have to install motile to use track_with_motile")

    solver, graph, segmentation = _track_with_motile_impl(
        segmentation, relabel_segmentation, node_cost_function, edge_cost_function,
        node_selection_cost, **problem_kwargs,
    )

    lineage_graph, lineages = parse_result(solver, graph)
    track_graph, tracks = lineage_graph_to_track_graph(lineage_graph, lineages)

    return segmentation, lineage_graph, lineages, track_graph, tracks


def get_representation_for_napari(
    segmentation: np.ndarray,
    lineage_graph: nx.DiGraph,
    lineages: Dict[int, List[int]],
    tracks: Dict[int, List[int]],
    color_by_lineage: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]:
    """Convert tracking result from motile into representation for napari.

    The output of this function can be passed to `napari.add_tracks` like this:
    ```
    tracking_result, track_data, parent_graph = get_representation_for_napari(...)
    viewer = napari.Viewer()
    viewer.add_labels(tracking_result)
    viewer.add_tracks(track_data, graph=parent_graph)
    napari.run()
    ```

    Args:
        segmentation: The input segmentation after relabeling.
        lineage_graph: The lineage graph result from tracking.
        lineages: The lineage assignment result from tracking.
        tracks: The track assignment result from tracking.
        color_by_lineage: Whether to color the tracking result by lineage id or by track id.

    Returns:
        The relabeled segmentation, where each segment id is either colored by the lineage id or track id.
        The track data for the napari tracks layer, which is a table containing track_id, timepoint, (z), y, x.
        The parent graph, which maps each track id to its parent id, if it exists.
    """
    node_ids = np.unique(segmentation)[1:]
    node_to_track = get_node_assignment(node_ids, tracks)
    node_to_lineage = get_node_assignment(node_ids, lineages)

    # Create label layer and track data for visualization in napari.
    tracking_result = recolor_segmentation(
        segmentation, node_to_lineage if color_by_lineage else node_to_track
    )

    # Create the track data and corresponding parent graph.
    track_data, parent_graph = create_data_for_track_layer(
        segmentation, lineage_graph, node_to_track, skip_zero=True
    )

    return tracking_result, track_data, parent_graph
