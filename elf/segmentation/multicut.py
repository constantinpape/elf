from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import nifty
import nifty.graph.opt.multicut as nmc

from .blockwise_mc_impl import blockwise_mc_impl

try:
    import rama_py
except ImportError:
    rama_py = None


#
# cost functionality
#


def _weight_edges(costs, edge_sizes, weighting_exponent):
    w = edge_sizes / float(edge_sizes.max())
    if weighting_exponent != 1.:
        w = w**weighting_exponent
    costs *= w
    return costs


def _weight_populations(costs, edge_sizes, edge_populations, weighting_exponent):
    # check that the population indices cover each edge at most once
    covered = np.zeros(len(costs), dtype="uint8")
    for edge_pop in edge_populations:
        covered[edge_pop] += 1
    assert (covered <= 1).all()

    for edge_pop in edge_populations:
        costs[edge_pop] = _weight_edges(costs[edge_pop], edge_sizes[edge_pop],
                                        weighting_exponent)

    return costs


def transform_probabilities_to_costs(
    probs: np.ndarray,
    beta: float = 0.5,
    edge_sizes: Optional[np.ndarray] = None,
    edge_populations: Optional[List[np.ndarray]] = None,
    weighting_exponent: float = 1.0
) -> np.ndarray:
    """Transform probabilities to costs via negative log likelihood.

    Args:
        probs: The input probabilities.
        beta: The boundary bias term.
        edge_sizes: The izes of edges for weighting.
        edge_populations: Different edge populations that will be weighted by size independently.
            Have to be passed as list of masks or index arrays.
            This can for example be used for flat superpixels in a 3d problem.
        weighting_exponent: The exponent used for weighting.

    Returns:
        The costs.
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


def compute_edge_costs(
    probs: np.ndarray,
    edge_sizes: Optional[np.ndarray] = None,
    z_edge_mask: Optional[np.ndarray] = None,
    beta: float = 0.5,
    weighting_scheme: Optional[str] = None,
    weighting_exponent: float = 1.0,
) -> np.ndarray:
    """Compute edge costs from probabilities with a pre-defined weighting scheme.

    Args:
        probs: The input probabilities.
        edge_sizes: The sizes of edges for weighting.
        z_edge_mask: The edge mask for inter-slice edges, only necessary for weighting schemes 'z' or 'xyz'.
        beta: The boundary bias.
        weighting_scheme: The scheme for weighting the edge costs based on size of the edges.
        weighting_exponent: The exponent used for weighting.

    Returns:
        The costs.
    """
    schemes = (None, "all", "none", "xyz", "z")
    if weighting_scheme not in schemes:
        schemes_str = ", ".join([str(scheme) for scheme in schemes])
        raise ValueError("Weighting scheme must be one of %s, got %s" % (schemes_str, str(weighting_scheme)))

    if weighting_scheme is None or weighting_scheme == "none":
        edge_pop = edge_sizes_ = None

    elif weighting_scheme == "all":
        if edge_sizes is None:
            raise ValueError("Need edge sizes for weighting scheme all")
        if len(edge_sizes) != len(probs):
            raise ValueError("Invalid edge sizes")
        edge_sizes_ = edge_sizes
        edge_pop = None

    elif weighting_scheme == "xyz":
        if edge_sizes is None or z_edge_mask is None:
            raise ValueError("Need edge sizes and z edge mask for weighting scheme xyz")
        if len(edge_sizes) != len(probs) or len(z_edge_mask) != len(probs):
            raise ValueError("Invalid edge sizes or z edge mask")
        edge_pop = [z_edge_mask, np.logical_not(z_edge_mask)]
        edge_sizes_ = edge_sizes

    elif weighting_scheme == "z":
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
# multicut solvers
#


def _to_objective(graph, costs):
    if isinstance(graph, nifty.graph.UndirectedGraph):
        graph_ = graph
    else:
        graph_ = nifty.graph.undirectedGraph(graph.numberOfNodes)
        graph_.insertEdges(graph.uvIds())
    objective = nmc.multicutObjective(graph_, costs)
    return objective


def _get_solver_factory(objective, internal_solver, warmstart=True, warmstart_kl=False):
    if internal_solver == "kernighan-lin":
        sub_solver = objective.kernighanLinFactory(warmStartGreedy=warmstart)
    elif internal_solver == "greedy-additive":
        sub_solver = objective.greedyAdditiveFactory()
    elif internal_solver == "greedy-fixation":
        sub_solver = objective.greedyFixationFactory()
    elif internal_solver == "cut-glue-cut":
        if not nifty.Configuration.WITH_QPBO:
            raise RuntimeError("multicut_cgc requires nifty built with QPBO")
        sub_solver = objective.cgcFactory(warmStartGreedy=warmstart, warmStartKl=warmstart_kl)
    elif internal_solver == "ilp":
        if not any((nifty.Configuration.WITH_CPLEX, nifty.Configuration.WITH_GLPK, nifty.Configuration.WITH_GUROBI)):
            raise RuntimeError("multicut_ilp requires nifty built with at least one of CPLEX, GLPK or GUROBI")
        sub_solver = objective.multicutIlpFactory()
    elif internal_solver in ("fusion-move", "decomposition"):
        raise NotImplementedError(f"Using {internal_solver} as internal solver is currently not supported.")
    else:
        raise ValueError(f"{internal_solver} cannot be used as internal solver.")
    return sub_solver


def _get_visitor(objective, time_limit=None, **kwargs):
    logging_interval = kwargs.pop("logging_interval", None)
    log_level = kwargs.pop("log_level", "INFO")
    if time_limit is not None or logging_interval is not None:
        logging_interval = int(np.iinfo("int32").max) if logging_interval is None else logging_interval
        time_limit = float("inf") if time_limit is None else time_limit
        log_level = getattr(nifty.LogLevel, log_level, nifty.LogLevel.INFO)

        # I can't see a real difference between loggingVisitor and verboseVisitor.
        # Use loggingVisitor for now.

        # visitor = objective.verboseVisitor(visitNth=logging_interval,
        #                                    timeLimitTotal=time_limit,
        #                                    logLevel=log_level)

        visitor = objective.loggingVisitor(visitNth=logging_interval,
                                           timeLimitTotal=time_limit,
                                           logLevel=log_level)

        return visitor
    else:
        return None


def blockwise_multicut(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    segmentation: np.ndarray,
    internal_solver: Union[str, callable],
    block_shape: Tuple[int, ...],
    n_threads: int,
    n_levels: int = 1,
    halo: Tuple[int, ...] = None,
    **kwargs
) -> np.ndarray:
    """Solve multicut with block-wise hierarchical solver.

    Introduced in "Solving large Multicut problems for connectomics via domain decomposition":
    http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w1/Pape_Solving_Large_Multicut_ICCV_2017_paper.pdf

    Args:
        graph: The graph of the multicut problem.
        costs: The edge costs of the multicut problem.
        segmentation: The segmentation underlying the multicut problem.
        internal_solver: The internal solver.
        block_shape: The shape of blocks used to extract sub-problems.
        n_threads: The umber of threads used to solve sub-problems in parallel.
        n_levels: The number of hierarchy levels.
        halo: The halo used to enlarge the block shape.

    Returns:
        The node label solution to the multicut problem.
    """
    solver = get_multicut_solver(internal_solver) if isinstance(internal_solver, str) else internal_solver
    if not callable(solver):
        raise ValueError("Invalid argument for internal_solver.")
    return blockwise_mc_impl(graph, costs, segmentation, solver, block_shape, n_threads, n_levels, halo)


def multicut_kernighan_lin(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    time_limit: Optional[float] = None,
    warmstart: bool = True,
    **kwargs
) -> np.ndarray:
    """Solve multicut problem with kernighan lin solver.

    Introduced in "An efficient heuristic procedure for partitioning graphs":
    http://xilinx.asia/_hdl/4/eda.ee.ucla.edu/EE201A-04Spring/kl.pdf

    Args:
        graph: The graph of the multicut problem.
        costs: The edge costs of multicut problem.
        time_limit: The time limit for inference in seconds.
        warmstart: Whether to warmstart with GAEC solution.
        kwargs: Keyword arguments for the visitor.

    Returns:
        The node label solution to the multicut problem.
    """
    objective = _to_objective(graph, costs)
    solver = objective.kernighanLinFactory(warmStartGreedy=warmstart).create(objective)
    visitor = _get_visitor(objective, time_limit, **kwargs)
    return solver.optimize() if visitor is None else solver.optimize(visitor=visitor)


def multicut_gaec(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    time_limit: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """Solve multicut problem with greedy-addtive edge contraction solver.

    Introduced in "Fusion moves for correlation clustering":
    http://openaccess.thecvf.com/content_cvpr_2015/papers/Beier_Fusion_Moves_for_2015_CVPR_paper.pdf

    Args:
        graph: The graph of the multicut problem.
        costs: The edge costs of the multicut problem.
        time_limit: The time limit for inference in seconds.
        kwargs: Keyword arguments for the visitor.

    Returns:
        The node label solution to the multicut problem.
    """
    objective = _to_objective(graph, costs)
    solver = objective.greedyAdditiveFactory().create(objective)
    visitor = _get_visitor(objective, time_limit, **kwargs)
    return solver.optimize() if visitor is None else solver.optimize(visitor=visitor)


def multicut_greedy_fixation(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    time_limit: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """Solve multicut problem with greedy fixation solver.

    Introduced in "A Comparative Study of Local Search Algorithms for Correlation Clustering":
    https://link.springer.com/chapter/10.1007/978-3-319-66709-6_9

    Args:
        graph: The graph of the multicut problem.
        costs: The edge costs of the multicut problem.
        time_limit: The time limit for inference in seconds.
        kwargs: Keyword arguments for the visitor.

    Returns:
        The node label solution to the multicut problem.
    """
    objective = _to_objective(graph, costs)
    solver = objective.greedyFixationFactory().create(objective)
    visitor = _get_visitor(objective, time_limit, **kwargs)
    return solver.optimize() if visitor is None else solver.optimize(visitor=visitor)


def multicut_cgc(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    time_limit: Optional[float] = None,
    warmstart: bool = True,
    warmstart_kl: bool = True,
    **kwargs
) -> np.ndarray:
    """Solve multicut problem with cut, glue & cut solver.

    Introduced in "Cut, Glue & Cut: A Fast, Approximate Solver for Multicut Partitioning":
    https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Beier_Cut_Glue__2014_CVPR_paper.html

    Requires nifty build with QPBO.

    Args:
        graph: The graph of the multicut problem.
        costs: The edge costs of the multicut problem.
        time_limit: The time limit for inference in seconds.
        warmstart: Whether to warmstart with GAEC solution.
        warmstart_kl: Also use kernighan lin to warmstart.
        kwargs: Keyword arguments for the visitor.

    Returns:
        The node label solution to the multicut problem.
    """
    if not nifty.Configuration.WITH_QPBO:
        raise RuntimeError("multicut_cgc requires nifty built with QPBO")
    objective = _to_objective(graph, costs)
    solver = objective.cgcFactory(warmStartGreedy=warmstart, warmStartKl=warmstart_kl).create(objective)
    visitor = _get_visitor(objective, time_limit, **kwargs)
    return solver.optimize() if visitor is None else solver.optimize(visitor=visitor)


def multicut_decomposition(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    time_limit: Optional[float] = None,
    n_threads: int = 1,
    internal_solver: str = "kernighan-lin",
    **kwargs,
) -> np.ndarray:
    """Solve multicut problem with decomposition solver.

    Introduced in "Break and Conquer: Efficient Correlation Clustering for Image Segmentation":
    https://link.springer.com/chapter/10.1007/978-3-642-39140-8_9

    Args:
        graph: The graph of the multicut problem.
        costs: The dge costs of the multicut problem.
        time_limit: The time limit for inference in seconds.
        n_threads: The number of threads.
        internal_solver: The name of the solver to use for connected components.
        kwargs: Keyword arguments for the visitor.

    Returns:
        The node label solution to the multicut problem.
    """
    objective = _to_objective(graph, costs)
    solver_factory = _get_solver_factory(objective, internal_solver)
    solver = objective.multicutDecomposerFactory(
        submodelFactory=solver_factory,
        fallthroughFactory=solver_factory,
        numberOfThreads=n_threads
    ).create(objective)
    visitor = _get_visitor(objective, time_limit, **kwargs)
    return solver.optimize() if visitor is None else solver.optimize(visitor=visitor)


def multicut_fusion_moves(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    time_limit: Optional[float] = None,
    n_threads: int = 1,
    internal_solver: str = "kernighan-lin",
    warmstart: bool = True,
    warmstart_kl: bool = True,
    seed_fraction: float = 0.05,
    num_it: int = 1000,
    num_it_stop: int = 25,
    sigma: float = 2.0,
    **kwargs,
) -> np.ndarray:
    """Solve multicut problem with fusion moves solver.

    Introduced in "Fusion moves for correlation clustering":
    http://openaccess.thecvf.com/content_cvpr_2015/papers/Beier_Fusion_Moves_for_2015_CVPR_paper.pdf

    Args:
        graph: The graph of the multicut problem.
        costs: The dge costs of the multicut problem.
        time_limit: The time limit for inference in seconds.
        n_threads: The number of threads.
        internal_solver: The name of solver used for fusion moves.
        warmstart: Whether to warmstart with GAEC solution.
        warmstart_kl: Also use kernighan lin to warmstart.
        seed_fraction: The fraction of nodes used as seeds for proposal generation.
        num_it: The maximal number of iterations.
        num_it_stop: Stop if no improvement after num_it_stop.
        sigma: The smoothing factor for weights in proposal generator.
        kwargs: Keyword arguments for the visitor.

    Returns:
        The node label solution to the multicut problem.
    """
    objective = _to_objective(graph, costs)
    sub_solver = _get_solver_factory(objective, internal_solver)
    sub_solver = objective.fusionMoveSettings(mcFactory=sub_solver)
    proposal_gen = objective.watershedCcProposals(sigma=sigma, numberOfSeeds=seed_fraction)
    solver = objective.ccFusionMoveBasedFactory(fusionMove=sub_solver,
                                                warmStartGreedy=warmstart,
                                                warmStartKl=warmstart_kl,
                                                proposalGenerator=proposal_gen,
                                                numberOfThreads=n_threads,
                                                numberOfIterations=num_it,
                                                stopIfNoImprovement=num_it_stop).create(objective)
    visitor = _get_visitor(objective, time_limit, **kwargs)
    return solver.optimize() if visitor is None else solver.optimize(visitor=visitor)


def multicut_ilp(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    time_limit: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """Solve multicut problem with ilp solver.

    Introduced in "Globally Optimal Closed-surface Segmentation for Connectomics":
    https://link.springer.com/chapter/10.1007/978-3-642-33712-3_56

    Requires nifty build with CPLEX, GUROBI or GLPK.

    Args:
        graph: The graph of the multicut problem.
        costs: The dge costs of the multicut problem.
        time_limit: The time limit for inference in seconds.
        kwargs: Keyword arguments for the visitor.

    Returns:
        The node label solution to the multicut problem.
    """
    if not any((nifty.Configuration.WITH_CPLEX, nifty.Configuration.WITH_GLPK, nifty.Configuration.WITH_GUROBI)):
        raise RuntimeError("multicut_ilp requires nifty built with at least one of CPLEX, GLPK or GUROBI")
    objective = _to_objective(graph, costs)
    solver = objective.multicutIlpFactory().create(objective)
    visitor = _get_visitor(objective, time_limit, **kwargs)
    return solver.optimize() if visitor is None else solver.optimize(visitor=visitor)


def multicut_rama(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    time_limit: Optional[float] = None,
    mode: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """Solve multicut problem with RAMA solver.

    Introduced in "RAMA: A Rapid Multicut Algorithm on GPU":
    https://arxiv.org/abs/2109.01838

    Requires the rama_py package, see https://github.com/pawelswoboda/RAMA.

    Args:
        graph: Graph of the multicut problem.
        costs: The edge costs of the multicut problem.
        time_limit: The time limit for inference in seconds.
        mode: The RAMA mode.

    Returns:
        The node label solution to the multicut problem.
    """
    if rama_py is None:
        raise RuntimeError("Need rama_py to use multicut_rama function")
    uv_ids = graph.uvIds()
    if mode is None:
        opts = rama_py.multicut_solver_options()
    else:
        assert mode in ("P", "PD", "PD+", "D")
        opts = rama_py.multicut_solver_options(mode)
    opts.verbose = False
    node_labels = rama_py.rama_cuda(uv_ids[:, 0], uv_ids[:, 1], costs, opts)[0]
    assert len(node_labels) == graph.numberOfNodes, f"{len(node_labels)}, {graph.numberOfNodes}"
    return node_labels


_solvers = {"kernighan-lin": multicut_kernighan_lin,
            "greedy-additive": multicut_gaec,
            "decomposition": multicut_decomposition,
            "fusion-moves": multicut_fusion_moves,
            "blockwise-multicut": blockwise_multicut,
            "greedy-fixation": multicut_greedy_fixation,
            "cut-glue-cut": multicut_cgc,
            "ilp": multicut_ilp,
            "rama": multicut_rama}


def get_available_solver_names() -> List[str]:
    """Get available multicut solver names

    Returns:
        The solver names.
    """
    return list(_solvers.keys())


def get_multicut_solver(name: str, **kwargs):
    """Get multicut solver by name.

    Args:
        name: The solver name.
        kwargs: Keyword arguments for the solver.

    Returns:
        The solver.
    """
    try:
        solver = partial(_solvers[name], **kwargs)
    except KeyError:
        raise KeyError("Solver %s is not supported" % name)
    return solver
