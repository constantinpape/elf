from functools import partial

import numpy as np
import nifty
import nifty.graph.opt.multicut as nmc

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


def compute_edge_costs(probs, edge_sizes=None, z_edge_mask=None,
                       beta=.5, weighting_scheme=None, weighting_exponent=1.):
    """ Compute edge costs from probabilities with a pre-defined weighting scheme.

    Arguments:
        probs [np.ndarray] - Input probabilities.
        edge_sizes [np.ndarray] - sizes of edges for weighting (default: None)
        z_edge_mask [np.ndarray] - edge mask for inter-slice edges,
            only necessary for weighting schemes z or xyz (default: None)
        beta [float] - boundary bias (default: .5)
        weighting_scheme [str] - scheme for weighting the edge costs based on size
            of the edges (default: NOne)
        weighting_exponent [float] - exponent used for weighting (default: 1.)
    """
    schemes = (None, 'all', 'none', 'xyz', 'z')
    if weighting_scheme not in schemes:
        schemes_str = ', '.join([str(scheme) for scheme in schemes])
        raise ValueError("Weighting scheme must be one of %s, got %s" % (schemes_str,
                                                                         str(weighting_scheme)))

    if weighting_scheme is None or weighting_scheme == 'none':
        edge_pop = edge_sizes_ = None

    elif weighting_scheme == 'all':
        if edge_sizes is None:
            raise ValueError("Need edge sizes for weighting scheme all")
        if len(edge_sizes) != len(probs):
            raise ValueError("Invalid edge sizes")
        edge_sizes_ = edge_sizes
        edge_pop = None

    elif weighting_scheme == 'xyz':
        if edge_sizes is None or z_edge_mask is None:
            raise ValueError("Need edge sizes and z edge mask for weighting scheme xyz")
        if len(edge_sizes) != len(probs) or len(z_edge_mask) != len(probs):
            raise ValueError("Invalid edge sizes or z edge mask")
        edge_pop = [z_edge_mask, np.logical_not(z_edge_mask)]
        edge_sizes_ = edge_sizes

    elif weighting_scheme == 'z':
        edge_pop = [z_edge_mask, np.logical_not(z_edge_mask)]
        edge_sizes_ = edge_sizes.copy()
        edge_sizes_[edge_pop[1]] = 1.
        if len(edge_sizes) != len(probs) or len(z_edge_mask) != len(probs):
            raise ValueError("Invalid edge sizes or z edge mask")
        if edge_sizes is None or z_edge_mask is None:
            raise ValueError("Need edge sizes and z edge mask for weighting scheme z")

    return transform_probabilities_to_costs(probs, beta=beta, edge_sizes=edge_sizes_,
                                            edge_populations=edge_pop,
                                            weighting_exponent=weighting_exponent)


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
               'fusion-moves': partial(multicut_fusion_moves, **kwargs),
               'blockwise-multicut': partial(blockwise_multicut, **kwargs),
               'greedy-fixation': partial(multicut_greedy_fixation, **kwargs),
               'cut-glue-cut': partial(multicut_cgc, **kwargs)}
    try:
        solver = solvers[name]
    except KeyError:
        raise KeyError("Solver %s is not supported" % name)
    return solver


# TODO cgc?
def _get_solver_factory(objective, internal_solver, warmstart=True):
    if internal_solver == 'kernighan-lin':
        sub_solver = objective.kernighanLinFactory(warmStartGreedy=warmstart)
    elif internal_solver == 'greedy-additive':
        sub_solver = objective.greedyAdditiveFactory()
    elif internal_solver == 'greedy-fixation':
        sub_solver = objective.greedyFixationFactory()
    elif internal_solver in ('fusion-move', 'decomposition'):
        raise NotImplementedError(f"Using {internal_solver} as internal solver is currently not supported.")
    else:
        raise ValueError(f"{internal_solver} cannot be used as internal solver.")
    return sub_solver


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
        time_limit [float] - time limit for inference in seconds (default: None)
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
        time_limit [float] - time limit for inference in seconds (default: None)
    """
    objective = _to_objective(graph, costs)
    solver = objective.greedyAdditiveFactory().create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def multicut_greedy_fixation(graph, costs, time_limit=None, **kwargs):
    """ Solve multicut problem with greedy fixation solver.

    Introduced in "A Comparative Study of Local Search Algorithms for Correlation Clustering":
    https://link.springer.com/chapter/10.1007/978-3-319-66709-6_9

    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference in seconds (default: None)
    """
    objective = _to_objective(graph, costs)
    solver = objective.greedyFixationFactory().create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def multicut_cgc(graph, costs, time_limit=None, **kwargs):
    pass


def multicut_decomposition(graph, costs, time_limit=None,
                           n_threads=1, internal_solver='kernighan-lin',
                           **kwargs):
    """ Solve multicut problem with decomposition solver.

    Introduced in "Break and Conquer: Efficient Correlation Clustering for Image Segmentation":
    https://link.springer.com/chapter/10.1007/978-3-642-39140-8_9

    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference in seconds (default: None)
        n_threads [int] - number of threads (default: 1)
        internal_solver [str] - name of solver used for connected components
            (default: 'kernighan-lin')
    """
    objective = _to_objective(graph, costs)
    solver_factory = _get_solver_factory(objective, internal_solver)
    solver = objective.multicutDecomposerFactory(
        submodelFactory=solver_factory,
        fallthroughFactory=solver_factory,
        numberOfThreads=n_threads
    ).create(objective)
    return solver.optimize()


def multicut_fusion_moves(graph, costs, time_limit=None, n_threads=1,
                          internal_solver='kernighan-lin', warmstart=True,
                          seed_fraction=.05,
                          num_it=1000, num_it_stop=25, sigma=2.):
    """ Solve multicut problem with fusion moves solver.

    Introduced in "Fusion moves for correlation clustering":
    http://openaccess.thecvf.com/content_cvpr_2015/papers/Beier_Fusion_Moves_for_2015_CVPR_paper.pdf

    Arguments:
        graph [nifty.graph] - graph of multicut problem
        costs [np.ndarray] - edge costs of multicut problem
        time_limit [float] - time limit for inference in seconds (default: None)
        n_threasd [int] - number of threads (default: 1)
        internal_solver [str] - name of solver used for connected components
            (default: 'kernighan-lin')
        warmstart [bool] - whether to warmstart with gaec solution (default: True)
        seed_fraction [float] - fraction of nodes used as seeds for proposal generation
            (default: .05)
        num_it [int] - maximal number of iterations (default: 1000)
        num_it_stop [int] - stop if no improvement after num_it_stop (default: 1000)
        sigma [float] - smoothing factor for weights in proposal generator (default: 2.)
    """
    objective = _to_objective(graph, costs)
    sub_solver = _get_solver_factory(objective, internal_solver)
    sub_solver = objective.fusionMoveSettings(mcFactory=sub_solver)
    proposal_gen = objective.watershedCcProposals(sigma=sigma, numberOfSeeds=seed_fraction)

    solver = objective.ccFusionMoveBasedFactory(fusionMove=sub_solver,
                                                warmStartGreedy=warmstart,
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
