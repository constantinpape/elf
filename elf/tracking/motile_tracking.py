"""Functionality for tracking microscopy data with [motile](https://github.com/funkelab/motile).
"""
from copy import deepcopy

import motile
import networkx as nx
import numpy as np

from motile import costs, constraints
from nifty.tools import takeDict
from skimage.measure import regionprops

from . import tracking_utils as utils


#
# Utility functionality for motile:
# Solution parsing and visualization.
#

def parse_result(solver, graph):
    lineage_graph = nx.DiGraph()

    node_indicators = solver.get_variables(motile.variables.NodeSelected)
    edge_indicators = solver.get_variables(motile.variables.EdgeSelected)

    # build new graphs that contain the selected nodes and tracking / lineage results
    for node, index in node_indicators.items():
        if solver.solution[index] > 0.5:
            lineage_graph.add_node(node, **graph.nodes[node])

    for edge, index in edge_indicators.items():
        if solver.solution[index] > 0.5:
            lineage_graph.add_edge(*edge, **graph.edges[edge])

    # use connected components to find the lineages
    lineages = nx.weakly_connected_components(lineage_graph)
    lineages = {lineage_id: list(lineage) for lineage_id, lineage in enumerate(lineages, 1)}
    return lineage_graph, lineages


def lineage_graph_to_track_graph(lineage_graph, lineages):
    # create a new graph that only contains the tracks by not connecting nodes with a degree of 2
    track_graph = nx.DiGraph()
    track_graph.add_nodes_from(lineage_graph.nodes)

    # iterate over the edges to find splits and end tracks there
    for (u, v), features in lineage_graph.edges.items():
        out_edges = lineage_graph.out_edges(u)
        # normal track continuation
        if len(out_edges) == 1:
            track_graph.add_edge(u, v)
        # otherwise track ends at division and we don't continue

    # use connected components to find the tracks
    tracks = nx.weakly_connected_components(track_graph)
    tracks = {track_id: list(track) for track_id, track in enumerate(tracks, 1)}

    return track_graph, tracks


def get_node_assignment(node_ids, assignments):
    # generate a dictionary that maps each node id (= segment id) to its assignment
    node_assignment = {
        node_id: assignment_id for assignment_id, nodes in assignments.items() for node_id in nodes
    }

    # everything that was not selected gets mapped to 0
    not_selected = list(set(node_ids) - set(node_assignment.keys()))
    node_assignment.update({not_select: 0 for not_select in not_selected})

    return node_assignment


def recolor_segmentation(segmentation, node_to_assignment):
    # we need to add a value for mapping 0, otherwise the function fails
    node_to_assignment_ = deepcopy(node_to_assignment)
    node_to_assignment_[0] = 0
    recolored_segmentation = takeDict(node_to_assignment_, segmentation)
    return recolored_segmentation


def create_data_for_track_layer(segmentation, lineage_graph, node_to_track):
    # compute regionpros and extract centroids
    props = regionprops(segmentation)
    centroids = {prop.label: prop.centroid for prop in props}

    # create the track data representation for napari
    track_data = [
        [node_to_track[node_id]] + list(centroid) for node_id, centroid in centroids.items()
        if node_id in node_to_track
    ]

    # create the parent graph for tracks
    parent_graph = {}
    for (u, v), features in lineage_graph.edges.items():
        out_edges = lineage_graph.out_edges(u)
        if len(out_edges) == 2:
            track_u, track_v = node_to_track[u], node_to_track[v]
            parent_graph[track_v] = parent_graph.get(track_v, []) + [track_u]

    return track_data, parent_graph


#
# Utility functions for constructing motile tracking problems
#


# TODO exppose the relevant weights and constants!
def construct_problem(
    segmentation,
    node_costs,
    edges_and_costs,
    max_parents=1,
    max_children=2,
):
    node_ids, indexes = np.unique(segmentation, return_index=True)
    indexes = np.unravel_index(indexes, shape=segmentation.shape)
    timeframes = indexes[0]

    # get rid of 0
    if node_ids[0] == 0:
        node_ids, timeframes = node_ids[1:], timeframes[1:]
    assert len(node_ids) == len(timeframes)

    graph = nx.DiGraph()
    # if the node function is not passed then we assume that all nodes should be selected
    assert len(node_costs) == len(node_ids)
    nodes = [
        {"id": node_id, "score": score, "t": t} for node_id, score, t in zip(node_ids, node_costs, timeframes)
    ]

    graph.add_nodes_from([(node["id"], node) for node in nodes])
    graph.add_edges_from([(edge["source"], edge["target"], edge) for edge in edges_and_costs])

    # construct da graph
    graph = motile.TrackGraph(graph)
    solver = motile.Solver(graph)

    # we can do linear reweighting of the costs: a * x + b
    # where: a=weight, b=constant
    solver.add_costs(costs.NodeSelection(weight=-1.0, attribute="score", constant=0))
    solver.add_costs(costs.EdgeSelection(weight=-1.0, attribute="score", constant=0))

    # add the constraints: we allow for divisions (max childeren = 2)
    solver.add_constraints(constraints.MaxParents(max_parents))
    solver.add_constraints(constraints.MaxChildren(max_children))

    # add costs for appearance and divisions
    solver.add_costs(costs.Appear(constant=1.0))
    solver.add_costs(costs.Split(constant=1.0))

    return solver, graph


#
# Motile based tracking
#


def track_with_motile(
    segmentation,
    relabel_segmentation=True,
    node_cost_function=None,
    edge_cost_function=None,
    node_selection_cost=0.95,
    **problem_kwargs,
):
    """Track segmented objects across time with motile.

    Note: this will relabel the segmentation unless `relabel_segmentation=False`
    """
    # relabel sthe segmentation so that the ids are unique across time.
    # if `relabel_segmentation is False` the segmentation has to be in the correct format already
    if relabel_segmentation:
        segmentation = utils.relabel_segmentation_across_time(segmentation)

    # compute the node selection costs.
    # if `node_cost_function` is passed it is used to compute the costs.
    # otherwise we set a fixed node selection cost.
    if node_cost_function is None:
        n_nodes = int(segmentation.max())
        node_costs = np.full(n_nodes, node_selection_cost)
    else:
        node_costs = node_cost_function(segmentation)

    # compute the edges and edge selection cost.
    # if `edge_cost_function` is not given we use the default approach.
    # (currently based on overlap of adjacent slices)
    if edge_cost_function is None:
        edge_cost_function = utils.compute_edges_from_overlap
    edges_and_costs = edge_cost_function(segmentation)

    # construct the problem
    solver, graph = construct_problem(segmentation, node_costs, edges_and_costs, **problem_kwargs)

    # solver the problem
    solver.solve()

    # parse solution
    lineage_graph, lineages = parse_result(solver, graph)
    track_graph, tracks = lineage_graph_to_track_graph(lineage_graph, lineages)

    return segmentation, lineage_graph, lineages, track_graph, tracks


def get_representation_for_napari(segmentation, lineage_graph, lineages, tracks, color_by_lineage=True):

    node_ids = np.unique(segmentation)[1:]
    node_to_track = get_node_assignment(node_ids, tracks)
    node_to_lineage = get_node_assignment(node_ids, lineages)

    # create label layer and track data for visualization in napari
    tracking_result = recolor_segmentation(
        segmentation, node_to_lineage if color_by_lineage else node_to_track
    )

    # create the track data and corresponding parent graph
    track_data, parent_graph = create_data_for_track_layer(
        segmentation, lineage_graph, node_to_track
    )

    return tracking_result, track_data, parent_graph
