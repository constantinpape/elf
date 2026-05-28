from functools import partial
from typing import Optional, Tuple, Union

import bioimage_cpp as bic
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
    graph,
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
    if isinstance(graph, bic.graph.UndirectedGraph):
        graph_ = graph
    else:
        uv = graph.uv_ids() if hasattr(graph, "uv_ids") else graph.uvIds()
        graph_ = bic.graph.UndirectedGraph.from_edges(graph.numberOfNodes, np.asarray(uv, dtype="uint64"))
    return bic.graph.lifted_multicut.LiftedMulticutObjective(
        graph_, costs,
        lifted_uvs=np.asarray(lifted_uv_ids, dtype="uint64"),
        lifted_costs=lifted_costs,
    )


def lifted_multicut_kernighan_lin(
    graph,
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
        time_limit: Ignored on the bioimage-cpp backend (no visitor support).
        warmstart: Whether to warmstart with GAEC solution.

    Returns:
        The node label solution to the lifted multicut problem.
    """
    objective = _to_objective(graph, costs, lifted_uv_ids, lifted_costs)
    if warmstart:
        solver = bic.graph.lifted_multicut.LiftedChainedSolvers([
            bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut(),
            bic.graph.lifted_multicut.LiftedKernighanLinMulticut(),
        ])
    else:
        solver = bic.graph.lifted_multicut.LiftedKernighanLinMulticut()
    return solver.optimize(objective)


def lifted_multicut_gaec(
    graph,
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
        time_limit: Ignored on the bioimage-cpp backend (no visitor support).

    Returns:
        The node label solution to the lifted multicut problem.
    """
    objective = _to_objective(graph, costs, lifted_uv_ids, lifted_costs)
    return bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut().optimize(objective)


def lifted_multicut_fusion_moves(
    graph,
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
        time_limit: Ignored on the bioimage-cpp backend (no visitor support).
        warmstart_gaec: Whether to warmstart with GAEC solution.
        warmstart_kl: Whether to warmstart with KL solution.

    Returns:
        The node label solution to the lifted multicut problem.
    """
    objective = _to_objective(graph, costs, lifted_uv_ids, lifted_costs)
    fusion = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(),
        sub_solver=bic.graph.lifted_multicut.LiftedKernighanLinMulticut(),
        number_of_threads=1,
    )
    chain = []
    if warmstart_gaec:
        chain.append(bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut())
    if warmstart_kl:
        chain.append(bic.graph.lifted_multicut.LiftedKernighanLinMulticut())
    chain.append(fusion)
    if len(chain) == 1:
        return fusion.optimize(objective)
    return bic.graph.lifted_multicut.LiftedChainedSolvers(chain).optimize(objective)
