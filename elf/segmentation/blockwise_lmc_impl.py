from concurrent import futures

import bioimage_cpp as bic
import numpy as np

from .blockwise_mc_impl import _relabel_from_zero, _remap_edges_sum_costs


def find_inner_lifted_edges(lifted_uv_ids, node_list):
    """@private
    """
    lifted_indices = np.arange(len(lifted_uv_ids), dtype="uint64")
    # find overlap of node_list with u-edges
    inner_us = np.isin(lifted_uv_ids[:, 0], node_list)
    inner_indices = lifted_indices[inner_us]
    inner_uvs = lifted_uv_ids[inner_us]
    # find overlap of node_list with v-edges
    inner_vs = np.isin(inner_uvs[:, 1], node_list)
    return inner_indices[inner_vs]


def solve_subproblems(graph, costs, lifted_uv_ids, lifted_costs,
                      segmentation, solver, blocking, halo, n_threads):
    """@private
    """

    uv_ids = graph.uv_ids()

    # solve sub-problem from one block
    def solve_subproblem(block_id):

        # extract nodes from this block
        block = blocking.get_block(block_id) if halo is None else\
            blocking.get_block_with_halo(block_id, list(halo)).outer_block
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        node_ids = np.unique(segmentation[bb]).astype("uint64")

        # get the sub-graph corresponding to the nodes
        inner_edges, outer_edges = graph.extract_subgraph_from_nodes(node_ids)
        sub_uvs = uv_ids[inner_edges]

        # relabel the sub-nodes and associated uv-ids for more efficient processing
        nodes_relabeled, max_id, mapping = _relabel_from_zero(node_ids)
        sub_uvs = bic.utils.take_dict(mapping, np.ascontiguousarray(sub_uvs, dtype="uint64"))
        n_local_nodes = max_id + 1
        sub_graph = bic.graph.UndirectedGraph.from_edges(n_local_nodes, sub_uvs)

        sub_costs = costs[inner_edges]
        assert len(sub_costs) == sub_graph.number_of_edges

        # get the inner lifted edges and costs
        inner_lifted_edges = find_inner_lifted_edges(lifted_uv_ids, node_ids)
        sub_lifted_uvs = bic.utils.take_dict(
            mapping, np.ascontiguousarray(lifted_uv_ids[inner_lifted_edges], dtype="uint64"),
        )
        sub_lifted_costs = lifted_costs[inner_lifted_edges]

        # solve multicut for the sub-graph
        sub_result = solver(sub_graph, sub_costs, sub_lifted_uvs, sub_lifted_costs)
        assert len(sub_result) == len(node_ids), "%i, %i" % (len(sub_result), len(node_ids))

        sub_edgeresult = sub_result[sub_uvs[:, 0]] != sub_result[sub_uvs[:, 1]]
        assert len(sub_edgeresult) == len(inner_edges)
        cut_edge_ids = inner_edges[sub_edgeresult]
        cut_edge_ids = np.concatenate([cut_edge_ids, outer_edges])
        return cut_edge_ids

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(solve_subproblem, block_id)
                 for block_id in range(blocking.number_of_blocks)]
        results = [t.result() for t in tasks]

    # merge the edge results to get all merge edges
    cut_edges = np.zeros(graph.number_of_edges, dtype="uint16")
    for res in results:
        cut_edges[res] += 1
    return cut_edges == 0


def update_edges(uv_ids, costs, labels, n_threads):
    """@private
    """
    return _remap_edges_sum_costs(uv_ids, labels, costs)


def reduce_problem(graph, costs, lifted_uv_ids, lifted_costs, merge_edges, n_threads):
    """@private
    """
    # merge node pairs with ufd
    n_nodes = graph.number_of_nodes
    nodes = np.arange(n_nodes, dtype="uint64")
    uv_ids = np.asarray(graph.uv_ids(), dtype="uint64")
    ufd = bic.utils.UnionFind(n_nodes)
    ufd.merge(np.ascontiguousarray(uv_ids[merge_edges], dtype="uint64"))

    # get then new node labels
    new_labels = ufd.find(nodes)

    # merge the edges and costs
    new_uv_ids, new_costs = update_edges(uv_ids, costs, new_labels, n_threads)
    new_lifted_uvs, new_lifted_costs = update_edges(
        np.asarray(lifted_uv_ids, dtype="uint64"), lifted_costs, new_labels, n_threads,
    )
    # build the new graph
    n_new_nodes = int(new_uv_ids.max()) + 1
    new_graph = bic.graph.UndirectedGraph.from_edges(
        n_new_nodes, np.ascontiguousarray(new_uv_ids, dtype="uint64"),
    )

    return new_graph, new_costs, new_lifted_uvs, new_lifted_costs, new_labels


def hierarchy_level(graph, costs, lifted_uv_ids, lifted_costs,
                    labels, segmentation, blocking,
                    internal_solver, n_threads, halo):
    """@private
    """
    merge_edges = solve_subproblems(graph, costs, lifted_uv_ids, lifted_costs,
                                    segmentation, internal_solver,
                                    blocking, halo, n_threads)
    graph, costs, lifted_uv_ids, lifted_costs, new_labels = reduce_problem(graph, costs,
                                                                           lifted_uv_ids, lifted_costs,
                                                                           merge_edges, n_threads)

    if labels is None:
        labels = new_labels
    else:
        labels = new_labels[labels]

    return graph, costs, lifted_uv_ids, lifted_costs, labels


def blockwise_lmc_impl(graph, costs, lifted_uv_ids, lifted_costs,
                       segmentation, internal_solver,
                       block_shape, n_threads, n_levels=1, halo=None):
    """@private
    """
    shape = segmentation.shape
    graph_, costs_ = graph, costs
    lifted_uv_ids_, lifted_costs_ = lifted_uv_ids, lifted_costs
    block_shape_ = block_shape
    labels = None

    for level in range(n_levels):
        blocking = bic.utils.Blocking(
            roi_begin=[0] * len(shape),
            roi_end=list(shape),
            block_shape=list(block_shape_),
        )
        graph_, costs_, lifted_uv_ids_, lifted_costs_, labels = hierarchy_level(graph_, costs_,
                                                                                lifted_uv_ids_, lifted_costs_,
                                                                                labels, segmentation, blocking,
                                                                                internal_solver, n_threads, halo)
        block_shape_ = [bs * 2 for bs in block_shape]

    # solve the final reduced problem
    final_labels = internal_solver(graph_, costs_, lifted_uv_ids_, lifted_costs_)
    # bring reduced problem back to the initial graph
    labels = final_labels[labels]
    return labels
