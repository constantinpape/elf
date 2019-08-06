import time
import nifty.graph.opt.lifted_multicut as nlmc


def key_to_lifted_agglomerator(key):
    agglo_dict = {'kernighan-lin': lifted_multicut_kernighan_lin,
                  'greedy-additive': lifted_multicut_gaec,
                  'fusion-moves': lifted_multicut_fusion_moves}
    assert key in agglo_dict, key
    return agglo_dict[key]


def lifted_multicut_kernighan_lin(graph, costs, lifted_uv_ids, lifted_costs,
                                  warmstart=True, time_limit=None, n_threads=1):
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
                         time_limit=None, n_threads=1):
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


# TODO why doesn't nifty expose the internal solver here ?
def lifted_multicut_fusion_moves(graph, costs, lifted_uv_ids, lifted_costs,
                                 warmstart_gaec=True, warmstart_kl=True,
                                 time_limit=None, n_threads=1):
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
    solver = objective.fusionMoveBasedFactory(numberOfThreads=1).create(objective)
    if time_limit is None:
        return solver.optimize(node_labels)
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(nodeLabels=node_labels, visitor=visitor)
