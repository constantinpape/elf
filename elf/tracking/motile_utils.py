"""Utility functionality for tracking with [motile](ttps://github.com/funkelab/motile).
"""

import motile
import networkx as nx
import nifty.gt as ngt
import numpy as np

from motile import costs, constraints
from nifty.tools import takeDict
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from tqdm import trange


#
# Simple functionality for computing edges and costs
#

def compute_edges_from_overlap(segmentation, verbose=True):
    """
    """
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
        frame_edges = compute_edges_from_overlap(this_frame, next_frame)
        edges.extend(frame_edges)
    return edges


# TODO
def compute_edges_from_centroid_distance(segmentation, max_distance, verbose=True):
    pass


# TODO does this work for 4d data (time + 3d)? if no we need to iterate over the time axis
def compute_node_costs_from_foreground_probabilities(segmentation, probabilities, cost_attribute="mean_intensity"):
    """
    """
    props = regionprops(segmentation, probabilities)
    costs = [getattr(prop, cost_attribute) for prop in props]
    return costs

#
# Utility functions for constructing tracking problems
#


def relabel_segmentation_across_time(segmentation):
    """
    """
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
    """
    """
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

    return graph, solver


#
# Solution parsing and visualization
#


def parse_result(solver, graph):
    """
    """
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


def recolor_segmentation(segmentation, lineages):
    # generate a dictionary that maps each node id (= segment id) to its lineage
    node_to_lineage = {
        node_id: lineage_id for lineage_id, nodes in lineages.items() for node_id in nodes
    }

    # everything that was not selected gets mapped to 0
    seg_ids = np.unique(segmentation)
    not_selected = list(set(seg_ids) - set(node_to_lineage.keys()))
    node_to_lineage.update({not_select: 0 for not_select in not_selected})

    # relabel based on the dict
    # (this can also be with a numpy function, but I only know my convenience function by heart...)
    recolored_segmentation = takeDict(node_to_lineage, segmentation)
    return recolored_segmentation, node_to_lineage


def lineage_graph_to_track_graph(lineage_graph, lineages, node_to_lineage):
    # create a new graph that only contains the tracks by not connecting nodes with a degree of 2
    track_graph = nx.DiGraph()
    track_graph.add_nodes_from(lineage_graph.nodes)

    # iterate over the edges to find splits and end tracks there
    for (u, v), features in lineage_graph.edges.items():
        out_edges = lineage_graph.out_edges(u)
        # normal track continuation
        if len(out_edges) == 1:
            track_graph.add_edge(u, v)
        # otherwise track ends at division and we don't contnue

    # use connected components to find the tracks
    tracks = nx.weakly_connected_components(track_graph)

    # find the mapping of nodes to tracks and tracks to lineage
    track_to_nodes = {track_id: list(track) for track_id, track in enumerate(tracks, 1)}
    node_to_track = {
        node_id: track_id for track_id, nodes in track_to_nodes.items() for node_id in nodes
    }
    track_to_lineage = {lineage_id: node_to_track[lineage[0]] for lineage_id, lineage in lineages.items()}

    return node_to_track, track_to_lineage


def create_napari_track_layer(segmentation, lineage_graph, lineages, node_to_lineage):
    # extract the graph with only tracks (without divisions)
    # and the graph of tracks into lineages
    node_to_track, track_to_lineage = lineage_graph_to_track_graph(lineage_graph, lineages, node_to_lineage)

    # get the regionproperties for centroids
    props = regionprops(segmentation)
    centroids = {
        prop.label: prop.centroid for prop in props
    }
    track_data = [
        [node_to_track[node_id]] + list(centroid) for node_id, centroid in centroids.items()
    ]

    lineage_to_tracks = {}
    for track, lineage in track_to_lineage.items():
        lineage_to_tracks[lineage] = lineage_to_tracks.get(lineage, []) + [track]

    return track_data, lineage_to_tracks
