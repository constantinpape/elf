import argparse
import json
import time

import nifty
import numpy as np
from elf.segmentation.features import compute_grid_graph, compute_grid_graph_affinity_features
from elf.segmentation.multicut import get_multicut_solver, compute_edge_costs, _to_objective
from elf.segmentation.mutex_watershed import mutex_watershed_clustering
from elf.segmentation.utils import load_mutex_watershed_problem


# TODO include lifted multicut
def pixelwise_performance_experiments(solvers):
    affs, offsets = load_mutex_watershed_problem()
    shape = affs.shape[1:]
    grid_graph = compute_grid_graph(shape)

    local_uvs, local_weights = compute_grid_graph_affinity_features(grid_graph, affs[:3], offsets[:3])
    lr_uvs, lr_weights = compute_grid_graph_affinity_features(grid_graph, affs[3:], offsets[3:],
                                                              strides=[1, 4, 4])

    # NOTE comparing multicut and mutex on the same objective doesn't make so much sense,
    # but it's easier than using a segmentation metric for now

    # compute the multicut problem by concatenating local and lr edges
    edges = np.concatenate([local_uvs, lr_uvs], axis=0)
    graph = nifty.graph.undirectedGraph(grid_graph.numberOfNodes)
    graph.insertEdges(edges)
    costs = np.concatenate([local_weights, lr_weights], axis=0)
    costs = compute_edge_costs(costs)
    objective = _to_objective(graph, costs)

    print("Pixelwise performance experiments")
    results = {}
    for solver_name in solvers:
        # get the mode for RAMA solvers
        if solver_name.startswith("rama"):
            _, mode = solver_name.split("_")
            solver = get_multicut_solver("rama")
            t0 = time.time()
            node_labels = solver(graph, costs, mode=mode)
            t0 = time.time() - t0

        elif solver_name == "mutex-watershed":
            t0 = time.time()
            node_labels = mutex_watershed_clustering(local_uvs, lr_uvs,
                                                     local_weights, lr_weights)
            t0 = time.time() - t0

        else:
            solver = get_multicut_solver(solver_name)
            t0 = time.time()
            node_labels = solver(graph, costs)
            t0 = time.time() - t0

        energy = objective.evalNodeLabels(node_labels)
        print("Solver", solver_name, "runtime:", t0, "s, energy:", energy)
        results[solver_name] = (energy, t0)
    return results


def main():
    parser = argparse.ArgumentParser()

    default_solvers = ["mutex-watershed", "greedy-additive", "rama_P", "rama_PD"]
    parser.add_argument("--solvers", "-s", default=default_solvers)

    args = parser.parse_args()
    res = pixelwise_performance_experiments(args.solvers)

    res_path = "./results_pixelwise.json"
    with open(res_path, "w") as f:
        json.dump(res, f, sort_keys=True, indent=2)


if __name__ == "__main__":
    main()
