from concurrent import futures
import numpy as np
import nifty
from vigra.analysis import relabelConsecutive


def solve_subproblems(graph, costs, segmentation,
                      solver, blocking, halo, n_threads):
    """@private
    """
    uv_ids = graph.uvIds()

    # solve sub-problem from one block
    def solve_subproblem(block_id):

        # extract nodes from this block
        block = blocking.getBlock(block_id) if halo is None else\
            blocking.getBlockWithHalo(block_id, halo).outerBlock
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        node_ids = np.unique(segmentation[bb])

        # get the sub-graph corresponding to the nodes
        inner_edges, outer_edges = graph.extractSubgraphFromNodes(node_ids)
        sub_uvs = uv_ids[inner_edges]

        # relabel the sub-nodes and associated uv-ids for more efficient processing
        nodes_relabeled, max_id, mapping = relabelConsecutive(node_ids,
                                                              start_label=0,
                                                              keep_zeros=False)
        sub_uvs = nifty.tools.takeDict(mapping, sub_uvs)
        n_local_nodes = max_id + 1
        sub_graph = nifty.graph.undirectedGraph(n_local_nodes)
        sub_graph.insertEdges(sub_uvs)

        sub_costs = costs[inner_edges]
        assert len(sub_costs) == sub_graph.numberOfEdges

        # solve multicut for the sub-graph; onlt cut the outer edges if we don"t have edges in this block
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
                 for block_id in range(blocking.numberOfBlocks)]
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
    nodes = np.arange(graph.numberOfNodes, dtype="uint64")
    uv_ids = graph.uvIds()
    ufd = nifty.ufd.ufd(graph.numberOfNodes)
    ufd.merge(uv_ids[merge_edges])

    # get then new node labels
    new_labels = ufd.find(nodes)
    new_labels = relabelConsecutive(new_labels, start_label=1, keep_zeros=True)[0]

    # merge the costs
    edge_mapping = nifty.tools.EdgeMapping(uv_ids, new_labels, numberOfThreads=n_threads)
    new_uv_ids = edge_mapping.newUvIds()

    new_costs = edge_mapping.mapEdgeValues(costs, "sum", numberOfThreads=n_threads)
    assert len(new_uv_ids) == len(new_costs)

    # build the new graph
    n_new_nodes = int(new_uv_ids.max()) + 1
    new_graph = nifty.graph.undirectedGraph(n_new_nodes)
    new_graph.insertEdges(new_uv_ids)

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
        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape_)
        graph_, costs_, labels = hierarchy_level(graph_, costs_, labels,
                                                 segmentation, blocking,
                                                 internal_solver, n_threads, halo)
        block_shape_ = [bs * 2 for bs in block_shape]

    # solve the final reduced problem
    final_labels = internal_solver(graph_, costs_)
    # bring reduced problem back to the initial graph
    labels = final_labels[labels]
    return labels
