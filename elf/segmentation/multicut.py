from concurrent import futures
from functools import partial

import numpy as np
import nifty
import nifty.ufd as nufd
import nifty.graph.opt.multicut as nmc
from vigra.analysis import relabelConsecutive


def transform_probabilities_to_costs(probs, beta=.5, edge_sizes=None,
                                     weighting_exponent=1.):
    """ Transform probabilities to costs via negative log likelihood.

    Arguments:
        probs [np.ndarray] - Input probabilities.
        beta [float] - boundary bias (default: .5)
        edge_sizes [np.ndarray] - sizes of edges for weighting (default: None)
        weighting_exponent [float] - exponent used for weighting (default: 1.)
    """
    p_min = 0.001
    p_max = 1. - p_min
    costs = (p_max - p_min) * probs + p_min
    # probabilities to costs, second term is boundary bias
    costs = np.log((1. - costs) / costs) + np.log((1. - beta) / beta)
    # weight the costs with edge sizes, if they are given
    if edge_sizes is not None:
        assert len(edge_sizes) == len(costs)
        w = edge_sizes / edge_sizes.max()
        if weighting_exponent != 1.:
            w = w**weighting_exponent
        costs *= w
    return costs

#
# TODO
# - support setting logging visitors
# - expose more parameters
# - add citiations to doc strings
#


def get_multicut_solver(name, **kwargs):
    """ Get multicut solver by name.
    """
    solvers = {'kernighan-lin': partial(multicut_kernighan_lin, **kwargs),
               'greedy-additive': partial(multicut_gaec, **kwargs),
               'decomposition': partial(multicut_decomposition, **kwargs),
               'decomposition-gaec': partial(multicut_decomposition,
                                             internal_solver='greedy-additive', **kwargs),
               'fusion-moves': partial(multicut_fusion_moves, **kwargs)}
    try:
        solver = solvers[name]
    except KeyError:
        raise KeyError("Solver %s is not supported" % name)
    return solver


def multicut_kernighan_lin(graph, costs, time_limit=None, warmstart=True, **kwargs):
    """ Solve multicut problem with kernighan lin solver.

    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference (default: None)
        warmstart [bool] - whether to warmstart with gaec solution (default: True)
    """
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.kernighanLinFactory(warmStartGreedy=warmstart).create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def multicut_gaec(graph, costs, time_limit=None, **kwargs):
    """ Solve multicut problem with greedy-addtive edge contraction solver.

    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference (default: None)
    """
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.greedyAdditiveFactory().create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


# TODO move impl to nifty (Thorsten already has impl ?!)
def multicut_decomposition(graph, costs, time_limit=None,
                           n_threads=1, internal_solver='kernighan-lin',
                           **kwargs):
    """ Solve multicut problem with decomposition solver.

    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference (default: None)
        n_threads [int] - number of threads (default: 1)
        internal_solver [str] - name of solver used for connected components
            (default: 'kernighan-lin')
    """

    # get the agglomerator
    solver = get_multicut_solver(internal_solver)

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
        sub_labels = solver(sub_graph, sub_costs, time_limit=time_limit)
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


# TODO enable warmstart with gaec / kl
def multicut_fusion_moves(graph, costs, time_limit=None, n_threads=1,
                          internal_solver='kernighan-lin', seed_fraction=.05,
                          num_it=1000, num_it_stop=10):
    """ Solve multicut problem with fusion moves solver.

    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference (default: None)
        n_threasd [int] - number of threads (default: 1)
        internal_solver [str] - name of solver used for connected components
            (default: 'kernighan-lin')
        seed_fraction [float] - fraction of nodes used as seeds for proposal generation
            (default: .05)
        num_it [int] - maximal number of iterations (default: 1000)
        num_it_stop [int] - stop if no improvement after num_it_stop (default: 1000)
    """
    assert internal_solver in ('kernighan-lin', 'greedy-additive')
    objective = nmc.multicutObjective(graph, costs)

    if internal_solver == 'kernighan-lin':
        sub_solver = objective.greedyAdditiveFactory()
    else:
        sub_solver = objective.kernighanLinFactory(warmStartGreedy=True)

    sub_solver = objective.fusionMoveSettings(mcFactory=sub_solver)
    proposal_gen = objective.watershedProposals(sigma=10,
                                                seedFraction=seed_fraction)

    solver = objective.fusionMoveBasedFactory(fusionMove=sub_solver,
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
