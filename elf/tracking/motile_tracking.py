import numpy as np
from . import motile_utils as utils


def track_with_motile(
    segmentation,
    relabel_segmentation=True,
    node_cost_function=None,
    edge_cost_function=None,
    node_selection_cost=0.95,
    **problem_kwargs,
):
    """Yadda yadda

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
    solver, graph = utils.construct_problem(segmentation, node_costs, edges_and_costs, **problem_kwargs)

    # solver the problem
    solver.solve()

    # parse solution
    lineage_graph, lineages = utils.parse_result(solver, graph)
    track_graph, tracks = utils.lineage_graph_to_track_graph(lineage_graph, lineages)

    return segmentation, lineage_graph, lineages, track_graph, tracks


def get_representation_for_napari(segmentation, lineage_graph, lineages, tracks, color_by_lineage=True):

    node_ids = np.unique(segmentation)[1:]
    node_to_track = utils.get_node_assignment(node_ids, tracks)
    node_to_lineage = utils.get_node_assignment(node_ids, lineages)

    # create label layer and track data for visualization in napari
    tracking_result = utils.recolor_segmentation(
        segmentation, node_to_lineage if color_by_lineage else node_to_track
    )

    # create the track data and corresponding parent graph
    track_data, parent_graph = utils.create_data_for_track_layer(
        segmentation, lineage_graph, node_to_track
    )

    return tracking_result, track_data, parent_graph
