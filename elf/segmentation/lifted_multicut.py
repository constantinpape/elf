import time
from functools import partial
from typing import Optional, Tuple, Union

import nifty
import nifty.graph.opt.lifted_multicut as nlmc
import numpy as np

from .blockwise_lmc_impl import blockwise_lmc_impl


def get_lifted_multicut_solver(name: str, **kwargs):
    """Get lifted multicut solver by name.

    Args:
        name: Name of the lifted multicut solver.

    Returns:
        The lifted multicut solver.
    """
    solvers = {"kernighan-lin": partial(lifted_multicut_kernighan_lin, **kwargs),
               "greedy-additive": partial(lifted_multicut_gaec, **kwargs),
               "fusion-moves": partial(lifted_multicut_fusion_moves, **kwargs)}
    try:
        solver = solvers[name]
    except KeyError:
        raise KeyError("Solver %s is not supported" % name)
    return solver


def blockwise_lifted_multicut(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    lifted_uv_ids: np.ndarray,
    lifted_costs: np.ndarray,
    segmentation: np.ndarray,
    internal_solver: Union[str, callable],
    block_shape: Tuple[int, ...],
    n_threads: int,
    n_levels: int = 1,
    halo: Optional[Tuple[int, ...]] = None,
    **kwargs
) -> np.ndarray:
    """Solve lifted multicut with block-wise hierarchical solver.

    Introduced in "Leveraging Domain Knowledge to improve EM image segmentation with Lifted Multicuts":
    https://arxiv.org/pdf/1905.10535.pdf

    Args:
        graph: Graph of the lifted multicut problem.
        costs : Edge costs of lifted multicut problem.
        lifted_uv_ids: The lifted edges.
        lifted_costs: The lifted edge costs.
        segmentation: The segmentation underlying the multicut problem.
        internal_solver: The internal solver.
        block_shape: The shape of blocks used to extract sub-problems.
        n_threads: The umber of threads used to solve sub-problems in parallel.
        n_levels: The number of hierarchy levels.
        halo: The halo used to enlarge the block shape.

    Returns:
        The node label solution to the lifted multicut problem.
    """
    solver = get_lifted_multicut_solver(internal_solver) if isinstance(internal_solver, str)\
        else internal_solver
    if not callable(solver):
        raise ValueError("Invalid argument for internal_solver.")
    return blockwise_lmc_impl(graph, costs, lifted_uv_ids, lifted_costs,
                              segmentation, solver, block_shape,
                              n_threads, n_levels, halo)


def _to_objective(graph, costs, lifted_uv_ids, lifted_costs):
    if isinstance(graph, nifty.graph.UndirectedGraph):
        graph_ = graph
    else:
        graph_ = nifty.graph.undirectedGraph(graph.numberOfNodes)
        graph_.insertEdges(graph.uvIds())
    objective = nlmc.liftedMulticutObjective(graph_)
    objective.setGraphEdgesCosts(costs)
    objective.setCosts(lifted_uv_ids, lifted_costs)
    return objective


def lifted_multicut_kernighan_lin(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    lifted_uv_ids: np.ndarray,
    lifted_costs: np.ndarray,
    time_limit: Optional[float] = None,
    warmstart: bool = True,
    **kwargs
) -> np.ndarray:
    """Solve lifted multicut problem with kernighan lin solver.

    Introduced in "Efficient decomposition of image and mesh graphs by lifted multicuts":
    http://openaccess.thecvf.com/content_iccv_2015/papers/Keuper_Efficient_Decomposition_of_ICCV_2015_paper.pdf

    Args:
        graph: Graph of the lifted multicut problem.
        costs: Edge costs of the lifted multicut problem.
        lifted_uv_ids: The lifted edges.
        lifted_costs: The lifted edge costs.
        time_limit: The time limit for inference.
        warmstart: Whether to warmstart with GAEC solution.

    Returns:
        The node label solution to the lifted multicut problem.
    """
    objective = _to_objective(graph, costs, lifted_uv_ids, lifted_costs)
    solver_kl = objective.liftedMulticutKernighanLinFactory().create(objective)
    if time_limit is None:
        if warmstart:
            solver_gaec = objective.liftedMulticutGreedyAdditiveFactory().create(objective)
            res = solver_gaec.optimize()
            return solver_kl.optimize(nodeLabels=res)
        else:
            return solver_kl.optimize()
    else:
        if warmstart:
            solver_gaec = objective.liftedMulticutGreedyAdditiveFactory().create(objective)
            visitor1 = objective.verboseVisitor(visitNth=1000000, timeLimitTotal=time_limit)
            t0 = time.time()
            res = solver_gaec.optimize(visitor=visitor1)
            t0 = time.time() - t0
            # Time limit is not hard, so t0 might actually be bigger than our time limit already.
            if t0 > time_limit:
                return res
            visitor2 = objective.verboseVisitor(visitNth=1000000, timeLimitTotal=time_limit - t0)
            return solver_kl.optimize(nodeLabels=res, visitor=visitor2)

        else:
            visitor = objective.verboseVisitor(visitNth=1000000, timeLimitTotal=time_limit)
            return solver_kl.optimize(visitor=visitor)


def lifted_multicut_gaec(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    lifted_uv_ids: np.ndarray,
    lifted_costs: np.ndarray,
    time_limit: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """Solve lifted multicut problem with greedy additive edge contraction solver.

    Introduced in "An efficient fusion move algorithm for the minimum cost lifted multicut problem":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/1939997197/beier_16_efficient.pdf

    Args:
        graph: The graph of the lifted multicut problem.
        costs: The edge costs of lifted multicut problem.
        lifted_uv_ids: The lifted edges.
        lifted_costs: The lifted edge costs.
        time_limit: The time limit for inference.

    Returns:
        The node label solution to the lifted multicut problem.
    """
    objective = _to_objective(graph, costs, lifted_uv_ids, lifted_costs)
    solver = objective.liftedMulticutGreedyAdditiveFactory().create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000, timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def lifted_multicut_fusion_moves(
    graph: nifty.graph.UndirectedGraph,
    costs: np.ndarray,
    lifted_uv_ids: np.ndarray,
    lifted_costs: np.ndarray,
    time_limit: Optional[float] = None,
    warmstart_gaec: bool = True,
    warmstart_kl: bool = True,
    **kwargs
) -> np.ndarray:
    """Solve lifted multicut problem with greedy additive edge contraction solver.

    Introduced in "An efficient fusion move algorithm for the minimum cost lifted multicut problem":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/1939997197/beier_16_efficient.pdf

    Args:
        graph: The graph of the lifted multicut problem.
        costs: The edge costs of the lifted multicut problem.
        lifted_uv_ids: The lifted edges.
        lifted_costs: The lifted edge costs.
        time_limit: The time limit for inference.
        warmstart_gaec: Whether to warmstart with GAEC solution.
        warmstart_kl: Whether to warmstart with KL solution.

    Returns:
        The node label solution to the lifted multicut problem.
    """
    objective = _to_objective(graph, costs, lifted_uv_ids, lifted_costs)

    # TODO keep track of time limits when warmstarting
    # perform warmstarts
    node_labels = None
    if warmstart_gaec:
        solver_gaec = objective.liftedMulticutGreedyAdditiveFactory().create(objective)
        node_labels = solver_gaec.optimize()
    if warmstart_kl:
        solver_kl = objective.liftedMulticutKernighanLinFactory().create(objective)
        node_labels = solver_kl.optimize(node_labels)

    solver = objective.fusionMoveBasedFactory(numberOfThreads=1).create(objective)
    if time_limit is None:
        return solver.optimize() if node_labels is None else solver.optimize(node_labels)
    else:
        visitor = objective.verboseVisitor(visitNth=1000000, timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor) if node_labels is None else\
            solver.optimize(nodeLabels=node_labels, visitor=visitor)
