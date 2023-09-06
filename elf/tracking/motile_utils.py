"""Utility functionality for tracking with [motile](https://github.com/funkelab/motile).
"""
from copy import deepcopy

import motile
import networkx as nx
import nifty.ground_truth as ngt
import numpy as np

from motile import costs, constraints
from nifty.tools import takeDict
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from tqdm import trange


#
# Simple functionality for computing edges and costs
#

def compute_edges_from_overlap(segmentation, verbose=True):
    def compute_overlap_between_frames(frame_a, frame_b):
        overlap_function = ngt.overlap(frame_a, frame_b)

        node_ids = np.unique(frame_a)[1:]
        overlaps = [overlap_function.overlapArraysNormalized(node_id) for node_id in node_ids]

        source_ids = [src for node_id, ovlp in zip(node_ids, overlaps) for src in [node_id] * len(ovlp[0])]
        target_ids = [ov for ovlp in overlaps for ov in ovlp[0]]
        overlap_values = [ov for ovlp in overlaps for ov in ovlp[1]]
        assert len(source_ids) == len(target_ids) == len(overlap_values),\
            f"{len(source_ids)}, {len(target_ids)}, {len(overlap_values)}"

        edges = [
            {"source": source_id, "target": target_id, "score": ovlp}
            for source_id, target_id, ovlp in zip(source_ids, target_ids, overlap_values)
        ]

        # filter out zeros
        edges = [edge for edge in edges if edge["target"] != 0]
        return edges

    edges = []
    for t in trange(segmentation.shape[0] - 1, disable=not verbose, desc="Compute edges via overlap"):
        this_frame = segmentation[t]
        next_frame = segmentation[t + 1]
        frame_edges = compute_overlap_between_frames(this_frame, next_frame)
        edges.extend(frame_edges)
    return edges


def compute_edges_from_centroid_distance(segmentation, max_distance, normalize_distances=True, verbose=True):
    nt = segmentation.shape[0]
    props = regionprops(segmentation)
    centroids_and_labels = [[prop.centroid[0], prop.centroid[1:], prop.label] for prop in props]

    centroids, labels = {}, {}
    for t, centroid, label in centroids_and_labels:
        centroids[t] = centroids.get(t, []) + [centroid]
        labels[t] = labels.get(t, []) + [label]
    centroids = {t: np.stack(np.array(val)) for t, val in centroids.items()}
    labels = {t: np.array(val) for t, val in labels.items()}

    def compute_dist_between_frames(t):
        centers_a, centers_b = centroids[t], centroids[t + 1]
        labels_a, labels_b = labels[t], labels[t + 1]
        assert len(centers_a) == len(labels_a)
        assert len(centers_b) == len(labels_b)

        distances = cdist(centers_a, centers_b)
        edge_mask = distances <= max_distance
        distance_values = distances[edge_mask]

        idx_a, idx_b = np.where(edge_mask)
        source_ids, target_ids = labels_a[idx_a], labels_b[idx_b]
        assert len(distance_values) == len(source_ids) == len(target_ids)

        return source_ids, target_ids, distance_values
        # return edges

    source_ids, target_ids, distances = [], [], []
    for t in trange(nt - 1, disable=not verbose, desc="Compute edges via centroid distance"):
        this_src, this_tgt, this_dist = compute_dist_between_frames(t)
        source_ids.extend(this_src), target_ids.extend(this_tgt), distances.extend(this_dist)

    if normalize_distances:
        distances = np.array(distances)
        max_dist = distances.max()
        distances = 1.0 - distances / max_dist

    edges = [
        {"source": source_id, "target": target_id, "score": distance}
        for source_id, target_id, distance in zip(source_ids, target_ids, distances)
    ]
    return edges


# TODO does this work for 4d data (time + 3d)? if no we need to iterate over the time axis
def compute_node_costs_from_foreground_probabilities(segmentation, probabilities, cost_attribute="mean_intensity"):
    props = regionprops(segmentation, probabilities)
    costs = [getattr(prop, cost_attribute) for prop in props]
    return costs

#
# Utility functions for constructing tracking problems
#


def relabel_segmentation_across_time(segmentation):
    offset = 0
    relabeled = []
    for frame in segmentation:
        frame, _, _ = relabel_sequential(frame)
        frame[frame != 0] += offset
        offset = frame.max()
        relabeled.append(frame)
    return np.stack(relabeled)


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

    # TODO
    # - expose all the weights and constants

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
# Solution parsing and visualization
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
