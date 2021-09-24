import argparse
import json
import os
import time

from elf.segmentation.multicut import get_multicut_solver, _to_objective
from elf.segmentation.utils import load_multicut_problem


def simple_performance_experiments(problem, solvers):
    os.makedirs("problems", exist_ok=True)
    path = f"./problems/{problem}"
    sample, size = problem.split("_")
    graph, costs = load_multicut_problem(sample, size, path)
    objective = _to_objective(graph, costs)

    results = {}
    print("Measure performance for sample:", problem)
    for solver_name in solvers:
        # get the mode for RAMA solvers
        if solver_name.startswith("rama"):
            _, mode = solver_name.split("_")
            solver = get_multicut_solver("rama")
            kwargs = {"mode": mode}
        else:
            solver = get_multicut_solver(solver_name)
            kwargs = {}
        t0 = time.time()
        node_labels = solver(graph, costs, **kwargs)
        t0 = time.time() - t0
        energy = objective.evalNodeLabels(node_labels)
        print("Solver", solver_name, "runtime:", t0, "s, energy:", energy)
        results[solver_name] = (energy, t0)

    return results


# TODO add large problems! where decomp should shine...
def main():
    parser = argparse.ArgumentParser()

    # default_solvers = ["decomposition", "kernighan-lin", "greedy-additive", "greedy-fixation"]
    default_solvers = ["decomposition", "kernighan-lin", "greedy-additive", "greedy-fixation",
                       "rama_P", "rama_PD+"]
    parser.add_argument("--solvers", "-s", default=default_solvers)

    default_problems = ["A_small", "B_small", "C_small",
                        "A_medium", "B_medium", "C_medium"]
    parser.add_argument("--problems", "-p", default=default_problems)

    # TODO save as a single csv instead
    print("Simple multicut performance experiments:")
    args = parser.parse_args()
    for problem in args.problems:
        res = simple_performance_experiments(problem, args.solvers)
        res_path = f"./results_{problem}.json"
        with open(res_path, "w") as f:
            json.dump(res, f, sort_keys=True, indent=2)


if __name__ == "__main__":
    main()
