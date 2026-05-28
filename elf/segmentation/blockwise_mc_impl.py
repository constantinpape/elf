from concurrent import futures

import bioimage_cpp as bic
import numpy as np


def _relabel_from_zero(node_ids):
    """@private

    Map ``node_ids`` to consecutive integers starting at 0 (no special
    treatment of label 0). Returns (relabeled, max_id, {old: new}).
    """
    uniq, inverse = np.unique(node_ids, return_inverse=True)
    relabeled = inverse.astype(node_ids.dtype, copy=False)
    mapping = {int(old): int(new) for new, old in enumerate(uniq)}
    return relabeled, int(uniq.size - 1), mapping


def _relabel_keep_zero(labels):
    """@private

    Map labels to consecutive integers starting at 1 while keeping 0 fixed at 0.
    """
    out = np.zeros_like(labels)
    nonzero_mask = labels != 0
    if nonzero_mask.any():
        uniq, inverse = np.unique(labels[nonzero_mask], return_inverse=True)
        out[nonzero_mask] = (inverse + 1).astype(labels.dtype, copy=False)
        del uniq
    return out


def _remap_edges_sum_costs(uv_ids, new_labels, costs):
    """@private

    After contracting nodes to ``new_labels``, build the new edge list (one row per
    unique pair of distinct supernodes) and sum the costs of all original edges that
    collapse onto the same pair. Replaces nifty's EdgeMapping.
    """
    remapped = new_labels[uv_ids]
    keep = remapped[:, 0] != remapped[:, 1]
    remapped = remapped[keep]
    sub_costs = np.asarray(costs)[keep]

    # canonicalize each row so the same undirected edge always maps to the same key
    u = np.minimum(remapped[:, 0], remapped[:, 1]).astype(np.int64)
    v = np.maximum(remapped[:, 0], remapped[:, 1]).astype(np.int64)
    n_new = int(new_labels.max()) + 1
    key = u * n_new + v

    unique_key, inverse = np.unique(key, return_inverse=True)
    new_costs = np.zeros(unique_key.size, dtype=sub_costs.dtype)
    np.add.at(new_costs, inverse, sub_costs)

    new_u = (unique_key // n_new).astype(uv_ids.dtype, copy=False)
    new_v = (unique_key % n_new).astype(uv_ids.dtype, copy=False)
    new_uv_ids = np.stack([new_u, new_v], axis=1)
    return new_uv_ids, new_costs


def solve_subproblems(graph, costs, segmentation,
                      solver, blocking, halo, n_threads):
    """@private
    """
    uv_ids = graph.uv_ids() if hasattr(graph, "uv_ids") else graph.uvIds()

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
        assert len(sub_costs) == sub_graph.numberOfEdges

        # solve multicut for the sub-graph; only cut the outer edges if we don't have edges in this block
        if len(sub_costs) > 0:
            sub_result = solver(sub_graph, sub_costs)
            assert len(sub_result) == len(node_ids), "%i, %i" % (len(sub_result), len(node_ids))

            sub_edgeresult = sub_result[sub_uvs[:, 0]] != sub_result[sub_uvs[:, 1]]
            assert len(sub_edgeresult) == len(inner_edges)
            cut_edge_ids = inner_edges[sub_edgeresult]
            cut_edge_ids = np.concatenate([cut_edge_ids, outer_edges])
        else:
            cut_edge_ids = outer_edges

        return cut_edge_ids

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(solve_subproblem, block_id)
                 for block_id in range(blocking.number_of_blocks)]
        results = [t.result() for t in tasks]

    # merge the edge results to get all merge edges
    cut_edges = np.zeros(graph.numberOfEdges, dtype="uint16")
    for res in results:
        cut_edges[res] += 1
    return cut_edges == 0


def reduce_problem(graph, costs, merge_edges, n_threads):
    """@private
    """
    # merge node pairs with ufd
    n_nodes = graph.numberOfNodes
    nodes = np.arange(n_nodes, dtype="uint64")
    uv_ids = np.asarray(graph.uv_ids() if hasattr(graph, "uv_ids") else graph.uvIds(), dtype="uint64")
    ufd = bic.utils.UnionFind(n_nodes)
    ufd.merge(np.ascontiguousarray(uv_ids[merge_edges], dtype="uint64"))

    # get then new node labels
    new_labels = ufd.find(nodes)
    new_labels = _relabel_keep_zero(new_labels)

    # merge the costs
    new_uv_ids, new_costs = _remap_edges_sum_costs(uv_ids, new_labels, costs)
    assert len(new_uv_ids) == len(new_costs)

    # build the new graph
    n_new_nodes = int(new_uv_ids.max()) + 1
    new_graph = bic.graph.UndirectedGraph.from_edges(
        n_new_nodes, np.ascontiguousarray(new_uv_ids, dtype="uint64"),
    )

    return new_graph, new_costs, new_labels


def hierarchy_level(graph, costs, labels,
                    segmentation, blocking,
                    internal_solver, n_threads, halo):
    """@private
    """
    merge_edges = solve_subproblems(graph, costs, segmentation,
                                    internal_solver, blocking, halo, n_threads)
    graph, costs, new_labels = reduce_problem(graph, costs, merge_edges, n_threads)

    if labels is None:
        labels = new_labels
    else:
        labels = new_labels[labels]

    return graph, costs, labels


def blockwise_mc_impl(graph, costs, segmentation, internal_solver,
                      block_shape, n_threads, n_levels=1, halo=None):
    """@private
    """
    shape = segmentation.shape
    graph_, costs_ = graph, costs
    block_shape_ = block_shape
    labels = None

    for level in range(n_levels):
        blocking = bic.utils.Blocking(
            roi_begin=[0] * len(shape),
            roi_end=list(shape),
            block_shape=list(block_shape_),
        )
        graph_, costs_, labels = hierarchy_level(graph_, costs_, labels,
                                                 segmentation, blocking,
                                                 internal_solver, n_threads, halo)
        block_shape_ = [bs * 2 for bs in block_shape]

    # solve the final reduced problem
    final_labels = internal_solver(graph_, costs_)
    # bring reduced problem back to the initial graph
    labels = final_labels[labels]
    return labels
