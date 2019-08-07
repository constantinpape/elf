import time
import partial
import nifty.graph.opt.lifted_multicut as nlmc


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
# - suppoer logging
# - add citations to doc-strings


def lifted_multicut_kernighan_lin(graph, costs, lifted_uv_ids, lifted_costs,
                                  time_limit=None, warmstart=True,
                                  **kwargs):
    """ Solve lifted multicut problem with kernighan lin solver.

    Arguments:
        graph [nifty.graph] - graph of lifted multicut problem
        costs [np.ndarray] - edge costs of lifted multicut problem
        lifted_uv_ids [np.ndarray] - lifted edges
        lifted_costs [np.ndarray] - lifted edge costs
        time_limit [float] - time limit for inference (default: None)
        warmstart [bool] - whether to warmstart with gaec solution (default: True)
    """
    objective = nlmc.liftedMulticutObjective(graph)
    objective.setGraphEdgesCosts(costs)
    objective.setCosts(lifted_uv_ids, lifted_costs)
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

    Arguments:
        graph [nifty.graph] - graph of lifted multicut problem
        costs [np.ndarray] - edge costs of lifted multicut problem
        lifted_uv_ids [np.ndarray] - lifted edges
        lifted_costs [np.ndarray] - lifted edge costs
        time_limit [float] - time limit for inference (default: None)
    """
    objective = nlmc.liftedMulticutObjective(graph)
    objective.setGraphEdgesCosts(costs)
    objective.setCosts(lifted_uv_ids, lifted_costs)
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

    Arguments:
        graph [nifty.graph] - graph of lifted multicut problem
        costs [np.ndarray] - edge costs of lifted multicut problem
        lifted_uv_ids [np.ndarray] - lifted edges
        lifted_costs [np.ndarray] - lifted edge costs
        time_limit [float] - time limit for inference (default: None)
        warmstart_gaec [bool] - whether to warmstart with gaec solution (default: True)
        warmstart_kl [bool] - whether to warmstart with kl solution (default: True)
    """
    objective = nlmc.liftedMulticutObjective(graph)
    objective.setGraphEdgesCosts(costs)
    objective.setCosts(lifted_uv_ids, lifted_costs)

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
