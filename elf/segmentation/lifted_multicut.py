import time
from functools import partial
import nifty
import nifty.graph.opt.lifted_multicut as nlmc

from .blockwise_lmc_impl import blockwise_lmc_impl


def get_lifted_multicut_solver(name, **kwargs):
    """ Get lifted multicut solver by name.
    """
    solvers = {'kernighan-lin': partial(lifted_multicut_kernighan_lin, **kwargs),
               'greedy-additive': partial(lifted_multicut_gaec, **kwargs),
               'fusion-moves': partial(lifted_multicut_fusion_moves, **kwargs)}
    try:
        solver = solvers[name]
    except KeyError:
        raise KeyError("Solver %s is not supported" % name)
    return solver


# TODO
# - support logging visitor


def blockwise_lifted_multicut(graph, costs, lifted_uv_ids, lifted_costs,
                              segmentation, internal_solver, block_shape,
                              n_threads, n_levels=1, halo=None, **kwargs):
    """ Solve lifted multicut with block-wise hierarchical solver.

    Introduced in "Leveraging Domain Knowledge to improve EM image segmentation with Lifted Multicuts":
    https://arxiv.org/pdf/1905.10535.pdf

    Arguments:
        graph [nifty.graph] - graph of lifted multicut problem
        costs [np.ndarray] - edge costs of lifted multicut problem
        lifted_uv_ids [np.ndarray] - lifted edges
        lifted_costs [np.ndarray] - lifted edge costs
        segmentation [np.ndarray] - segmentation underlying multicut problem
        internal_solver [str or callable] - internal solver
        block_shape [listlike] - shape of blocks used to extract sub-problems
        n_threads [int] - number of threads used to solve sub-problems in parallel
        n_levels [int] - number of hierarchy levels (default: 1)
        halo [listlike] - halo used to enlarge block shape (default: None)
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


def lifted_multicut_kernighan_lin(graph, costs, lifted_uv_ids, lifted_costs,
                                  time_limit=None, warmstart=True,
                                  **kwargs):
    """ Solve lifted multicut problem with kernighan lin solver.

    Introduced in "Efficient decomposition of image and mesh graphs by lifted multicuts":
    http://openaccess.thecvf.com/content_iccv_2015/papers/Keuper_Efficient_Decomposition_of_ICCV_2015_paper.pdf

    Arguments:
        graph [nifty.graph] - graph of lifted multicut problem
        costs [np.ndarray] - edge costs of lifted multicut problem
        lifted_uv_ids [np.ndarray] - lifted edges
        lifted_costs [np.ndarray] - lifted edge costs
        time_limit [float] - time limit for inference (default: None)
        warmstart [bool] - whether to warmstart with gaec solution (default: True)
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
            visitor1 = objective.verboseVisitor(visitNth=1000000,
                                                timeLimitTotal=time_limit)
            t0 = time.time()
            res = solver_gaec.optimize(visitor=visitor1)
            t0 = time.time() - t0
            # time limit is not hard, so t0 might actually be bigger than
            # our time limit already
            if t0 > time_limit:
                return res
            visitor2 = objective.verboseVisitor(visitNth=1000000,
                                                timeLimitTotal=time_limit - t0)
            return solver_kl.optimize(nodeLabels=res,
                                      visitor=visitor2)

        else:
            visitor = objective.verboseVisitor(visitNth=1000000,
                                               timeLimitTotal=time_limit)
            return solver_kl.optimize(visitor=visitor)


def lifted_multicut_gaec(graph, costs, lifted_uv_ids, lifted_costs,
                         time_limit=None, **kwargs):
    """ Solve lifted multicut problem with greedy additive edge contraction solver.

    Introduced in "An efficient fusion move algorithm for the minimum cost lifted multicut problem":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/1939997197/beier_16_efficient.pdf

    Arguments:
        graph [nifty.graph] - graph of lifted multicut problem
        costs [np.ndarray] - edge costs of lifted multicut problem
        lifted_uv_ids [np.ndarray] - lifted edges
        lifted_costs [np.ndarray] - lifted edge costs
        time_limit [float] - time limit for inference (default: None)
    """
    objective = _to_objective(graph, costs, lifted_uv_ids, lifted_costs)
    solver = objective.liftedMulticutGreedyAdditiveFactory().create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def lifted_multicut_fusion_moves(graph, costs, lifted_uv_ids, lifted_costs,
                                 time_limit=None, warmstart_gaec=True, warmstart_kl=True,
                                 **kwargs):
    """ Solve lifted multicut problem with greedy additive edge contraction solver.

    Introduced in "An efficient fusion move algorithm for the minimum cost lifted multicut problem":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/1939997197/beier_16_efficient.pdf

    Arguments:
        graph [nifty.graph] - graph of lifted multicut problem
        costs [np.ndarray] - edge costs of lifted multicut problem
        lifted_uv_ids [np.ndarray] - lifted edges
        lifted_costs [np.ndarray] - lifted edge costs
        time_limit [float] - time limit for inference (default: None)
        warmstart_gaec [bool] - whether to warmstart with gaec solution (default: True)
        warmstart_kl [bool] - whether to warmstart with kl solution (default: True)
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

    # nifty only supports numberOfThreads=1
    # why doesn't nifty expose the internal solver here ?
    solver = objective.fusionMoveBasedFactory(numberOfThreads=1).create(objective)
    if time_limit is None:
        return solver.optimize() if node_labels is None else solver.optimize(node_labels)
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor) if node_labels is None else\
            solver.optimize(nodeLabels=node_labels, visitor=visitor)
