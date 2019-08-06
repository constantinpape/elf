from concurrent import futures
from functools import partial

import numpy as np
import nifty
import nifty.ufd as nufd
import nifty.graph.opt.multicut as nmc
from vigra.analysis import relabelConsecutive

#
# TODO
# - support setting logging visitors
# - expose more parameters
#


def key_to_agglomerator(key):
    agglo_dict = {'kernighan-lin': multicut_kernighan_lin,
                  'greedy-additive': multicut_gaec,
                  'decomposition': multicut_decomposition,
                  'decomposition-gaec': partial(multicut_decomposition,
                                                solver='greedy-additive'),
                  'fusion-moves': multicut_fusion_moves}
    assert key in agglo_dict, key
    return agglo_dict[key]


def multicut_kernighan_lin(graph, costs, warmstart=True, time_limit=None, n_threads=1):
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.kernighanLinFactory(warmStartGreedy=warmstart).create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def multicut_gaec(graph, costs, time_limit=None, n_threads=1):
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.greedyAdditiveFactory().create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def multicut_decomposition(graph, costs, time_limit=None, n_threads=1,
                           solver='kernighan-lin'):

    # get the agglomerator
    agglomerator = key_to_agglomerator(solver)

    # merge attractive edges with ufd to
    # obtain natural connected components
    merge_edges = costs > 0
    ufd = nufd.ufd(graph.numberOfNodes)
    uv_ids = graph.uvIds()
    ufd.merge(uv_ids[merge_edges])
    cc_labels = ufd.elementLabeling()

    # relabel component ids consecutively
    cc_labels, max_id, _ = relabelConsecutive(cc_labels, start_label=0,
                                              keep_zeros=False)

    # TODO use c++ (Thorsten already has impl ?!)
    # TODO check that relabelConsecutive lifts gil ....
    # solve a component sub-problem
    def solve_component(component_id):

        # extract the nodes in this component
        sub_nodes = np.where(cc_labels == component_id)[0].astype('uint64')
        # if we only have a single node, return trivial labeling
        if len(sub_nodes) == 1:
            return sub_nodes, np.array([0], dtype='uint64'), 1

        # extract the subgraph corresponding to this component
        inner_edges, _ = graph.extractSubgraphFromNodes(sub_nodes)
        sub_uvs = uv_ids[inner_edges]
        assert len(inner_edges) == len(sub_uvs), "%i, %i" % (len(inner_edges), len(sub_uvs))

        # relabel sub-nodes and associated uv-ids
        sub_nodes_relabeled, max_local, node_mapping = relabelConsecutive(sub_nodes,
                                                                          start_label=0,
                                                                          keep_zeros=False)
        sub_uvs = nifty.tools.takeDict(node_mapping, sub_uvs)

        # build the graph
        sub_graph = nifty.graph.undirectedGraph(max_local + 1)
        sub_graph.insertEdges(sub_uvs)

        # solve local multicut
        sub_costs = costs[inner_edges]
        assert len(sub_costs) == sub_graph.numberOfEdges, "%i, %i" % (len(sub_costs),
                                                                      sub_graph.numberOfEdges)
        sub_labels = agglomerator(sub_graph, sub_costs, time_limit=time_limit)
        # relabel the solution
        sub_labels, max_seg_local, _ = relabelConsecutive(sub_labels, start_label=0,
                                                          keep_zeros=False)
        assert len(sub_labels) == len(sub_nodes), "%i, %i" % (len(sub_labels), len(sub_nodes))
        return sub_nodes, sub_labels, max_seg_local + 1

    # solve all components in parallel
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(solve_component, component_id)
                 for component_id in range(max_id + 1)]
        results = [t.result() for t in tasks]

    sub_nodes = [res[0] for res in results]
    sub_results = [res[1] for res in results]
    offsets = np.array([res[2] for res in results], dtype='uint64')

    # make proper offsets for the component results
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)

    # insert sub-results into the components
    node_labels = np.zeros_like(cc_labels, dtype='uint64')

    def insert_solution(component_id):
        nodes = sub_nodes[component_id]
        node_labels[nodes] = (sub_results[component_id] + offsets[component_id])

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(insert_solution, component_id)
                 for component_id in range(max_id + 1)]
        [t.result() for t in tasks]

    return node_labels


# TODO warmstart with gaec / kl
def multicut_fusion_moves(graph, costs, time_limit=None, n_threads=1,
                          internal_solver_name='kernighan-lin'):
    assert internal_solver_name in ('kernighan-lin', 'greedy-additive')
    objective = nmc.multicutObjective(graph, costs)

    if internal_solver_name == 'kernighan-lin':
        internal_solver = objective.greedyAdditiveFactory()
    else:
        internal_solver = objective.kernighanLinFactory(warmStartGreedy=True)

    # TODO expose
    seed_fraction = .05
    num_it_stop = 10
    num_it = 1000

    internal_solver = objective.fusionMoveSettings(mcFactory=internal_solver)
    proposal_gen = objective.watershedProposals(sigma=10,
                                                seedFraction=seed_fraction)

    solver = objective.fusionMoveBasedFactory(fusionMove=internal_solver,
                                              verbose=1, fuseN=2,
                                              proposalGen=proposal_gen,
                                              numberOfIterations=num_it,
                                              numberOfParallelProposals=2*n_threads,
                                              numberOfThreads=n_threads,
                                              stopIfNoImprovement=num_it_stop).create(objective)

    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)
