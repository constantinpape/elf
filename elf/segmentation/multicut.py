from concurrent import futures
from functools import partial

import numpy as np
import nifty
import nifty.ufd as nufd
import nifty.graph.opt.multicut as nmc
from vigra.analysis import relabelConsecutive

from .blockwise_mc_impl import blockwise_mc_impl


def _to_objective(graph, costs):
    if isinstance(graph, nifty.graph.UndirectedGraph):
        graph_ = graph
    else:
        graph_ = nifty.graph.undirectedGraph(graph.numberOfNodes)
        graph_.insertEdges(graph.uvIds())
    objective = nmc.multicutObjective(graph_, costs)
    return objective


def _weight_edges(costs, edge_sizes, weighting_exponent):
    w = edge_sizes / float(edge_sizes.max())
    if weighting_exponent != 1.:
        w = w**weighting_exponent
    costs *= w
    return costs


def _weight_populations(costs, edge_sizes, edge_populations, weighting_exponent):
    # check that the population indices cover each edge at most once
    covered = np.zeros(len(costs), dtype='uint8')
    for edge_pop in edge_populations:
        covered[edge_pop] += 1
    assert (covered <= 1).all()

    for edge_pop in edge_populations:
        costs[edge_pop] = _weight_edges(costs[edge_pop], edge_sizes[edge_pop],
                                        weighting_exponent)

    return costs


def transform_probabilities_to_costs(probs, beta=.5, edge_sizes=None,
                                     edge_populations=None, weighting_exponent=1.):
    """ Transform probabilities to costs via negative log likelihood.

    Arguments:
        probs [np.ndarray] - Input probabilities.
        beta [float] - boundary bias (default: .5)
        edge_sizes [np.ndarray] - sizes of edges for weighting (default: None)
        edge_populations [list[np.ndarray]] - different edge populations that will be
            size weighted independently passed as list of masks or index arrays.
            This can e.g. be useful if we have flat superpixels in a 3d problem. (default: None)
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
        if edge_populations is None:
            costs = _weight_edges(costs, edge_sizes, weighting_exponent)
        else:
            costs = _weight_populations(costs, edge_sizes, edge_populations, weighting_exponent)
    return costs

#
# TODO
# - support setting logging visitors
# - expose more parameters
#


def get_multicut_solver(name, **kwargs):
    """ Get multicut solver by name.
    """
    solvers = {'kernighan-lin': partial(multicut_kernighan_lin, **kwargs),
               'greedy-additive': partial(multicut_gaec, **kwargs),
               'decomposition': partial(multicut_decomposition, **kwargs),
               'decomposition-gaec': partial(multicut_decomposition,
                                             internal_solver='greedy-additive', **kwargs),
               'fusion-moves': partial(multicut_fusion_moves, **kwargs),
               'blockwise-multicut': partial(blockwise_multicut, **kwargs)}
    try:
        solver = solvers[name]
    except KeyError:
        raise KeyError("Solver %s is not supported" % name)
    return solver


def blockwise_multicut(graph, costs, segmentation,
                       internal_solver, block_shape,
                       n_threads, n_levels=1, halo=None, **kwargs):
    """ Solve multicut with block-wise hierarchical solver.

    Introduced in "Solving large Multicut problems for connectomics via domain decomposition":
    http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w1/Pape_Solving_Large_Multicut_ICCV_2017_paper.pdf

    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        segmentation [np.ndarray] - segmentation underlying multicut problem
        internal_solver [str or callable] - internal solver
        block_shape [listlike] - shape of blocks used to extract sub-problems
        n_threads [int] - number of threads used to solve sub-problems in parallel
        n_levels [int] - number of hierarchy levels (default: 1)
        halo [listlike] - halo used to enlarge block shape (default: None)
    """
    solver = get_multicut_solver(internal_solver) if isinstance(internal_solver, str)\
        else internal_solver
    if not callable(solver):
        raise ValueError("Invalid argument for internal_solver.")
    return blockwise_mc_impl(graph, costs, segmentation, solver,
                             block_shape, n_threads, n_levels, halo)


def multicut_kernighan_lin(graph, costs, time_limit=None, warmstart=True, **kwargs):
    """ Solve multicut problem with kernighan lin solver.

    Introduced in "An efficient heuristic procedure for partitioning graphs":
    http://xilinx.asia/_hdl/4/eda.ee.ucla.edu/EE201A-04Spring/kl.pdf

    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference (default: None)
        warmstart [bool] - whether to warmstart with gaec solution (default: True)
    """
    objective = _to_objective(graph, costs)
    solver = objective.kernighanLinFactory(warmStartGreedy=warmstart).create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def multicut_gaec(graph, costs, time_limit=None, **kwargs):
    """ Solve multicut problem with greedy-addtive edge contraction solver.

    Introduced in "Fusion moves for correlation clustering":
    http://openaccess.thecvf.com/content_cvpr_2015/papers/Beier_Fusion_Moves_for_2015_CVPR_paper.pdf

    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference (default: None)
    """
    objective = _to_objective(graph, costs)
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
                          num_it=1000, num_it_stop=25):
    """ Solve multicut problem with fusion moves solver.

    Introduced in "Fusion moves for correlation clustering":
    http://openaccess.thecvf.com/content_cvpr_2015/papers/Beier_Fusion_Moves_for_2015_CVPR_paper.pdf

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
    objective = _to_objective(graph, costs)

    if internal_solver == 'kernighan-lin':
        sub_solver = objective.greedyAdditiveFactory()
    else:
        sub_solver = objective.kernighanLinFactory(warmStartGreedy=True)

    sub_solver = objective.fusionMoveSettings(mcFactory=sub_solver)
    proposal_gen = objective.watershedCcProposals(sigma=2., numberOfSeeds=seed_fraction)

    solver = objective.ccFusionMoveBasedFactory(fusionMove=sub_solver,
                                                proposalGenerator=proposal_gen,
                                                numberOfThreads=n_threads,
                                                numberOfIterations=num_it,
                                                stopIfNoImprovement=num_it_stop).create(objective)

    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)
