"""Run GA / SA / QGA / Q2GA on the Calgary case-study instance and save:
  - results/convergence.csv : per-generation best-fitness history
  - results/summary.csv     : final best fitness, runtime, objective components
  - results/best_routes.json: decoded routes of the best Q2GA solution found
                               (used by make_maps.py for the route map)

The Calgary instance uses real road-network driving distances (via
calgary_instance.dist_matrix), not Euclidean distance.
"""

import os
import sys
import csv
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_AARP = os.path.join(os.path.dirname(ROOT), "Q2GA_AARP")
sys.path.insert(0, ROOT_AARP)
sys.path.insert(0, ROOT)

from src.aarp_model import evaluate_full, decode  # noqa: E402
from src.algorithms import ga, sa, qga, q2ga  # noqa: E402
from src.calgary_instance import build_instance  # noqa: E402

ALGORITHMS = {
    "GA": ga.run,
    "SA": sa.run,
    "QGA": qga.run,
    "Q2GA": q2ga.run,
}

# Tuned configuration from the main study (medium-instance config, since the
# Calgary case study has N=25 patients, between the medium (30) and small (15)
# instances): see Q2GA_AARP/src/run_experiments.py.
TUNED_KWARGS = {
    "GA": dict(pc=0.85, pm=0.10),
    "SA": dict(T0=10.0, alpha=0.97),
    "QGA": dict(delta_theta=0.05 * np.pi),
    "Q2GA": dict(epsilon_start=0.30, base_delta_theta=0.05 * np.pi),
}

N_SEEDS = 10
POP_SIZE = 20
N_GENERATIONS = 50

RESULTS_DIR = os.path.join(ROOT, "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    inst = build_instance(seed=42)

    conv_rows = []
    summ_rows = []

    best_q2ga = None  # (best_fitness, chromosome)

    for algo_name, algo_fn in ALGORITHMS.items():
        kwargs = TUNED_KWARGS[algo_name]
        for seed in range(N_SEEDS):
            res = algo_fn(inst, pop_size=POP_SIZE, n_generations=N_GENERATIONS, seed=seed, **kwargs)

            for gen, val in enumerate(res["history"]):
                conv_rows.append([algo_name, seed, gen, val])

            comp = evaluate_full(res["best"], inst)
            summ_rows.append([
                algo_name, seed, res["best_fitness"], res["runtime"],
                comp["C1"], comp["C2"], comp["C3"],
                comp["penalty1"], comp["penalty2"],
                comp["infeasibility"], comp["n_unserved"],
            ])
            print(f"{algo_name:6s} seed={seed:2d}  best={res['best_fitness']:.3f}  "
                  f"time={res['runtime']:.2f}s")

            if algo_name == "Q2GA":
                if best_q2ga is None or res["best_fitness"] < best_q2ga[0]:
                    best_q2ga = (res["best_fitness"], res["best"])

    conv_path = os.path.join(RESULTS_DIR, "convergence.csv")
    summ_path = os.path.join(RESULTS_DIR, "summary.csv")

    with open(conv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "seed", "generation", "best_fitness"])
        w.writerows(conv_rows)

    with open(summ_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "seed", "best_fitness", "runtime_s",
                     "C1", "C2", "C3", "penalty1", "penalty2", "infeasibility", "n_unserved"])
        w.writerows(summ_rows)

    # Decode and save the best Q2GA solution's routes for the route map
    best_fitness, best_chrom = best_q2ga
    routes = decode(best_chrom, inst)
    routes_out = {str(v): [int(p) for p in seq] for v, seq in routes.items()}
    comp = evaluate_full(best_chrom, inst)

    with open(os.path.join(RESULTS_DIR, "best_routes.json"), "w") as f:
        json.dump({
            "algorithm": "Q2GA",
            "best_fitness": best_fitness,
            "routes": routes_out,
            "objective_components": {k: (v if not isinstance(v, (np.floating,)) else float(v))
                                      for k, v in comp.items()},
        }, f, indent=2)

    print(f"\nSaved: {conv_path}")
    print(f"Saved: {summ_path}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'best_routes.json')}")
    print(f"Best Q2GA fitness: {best_fitness:.3f}")


if __name__ == "__main__":
    main()
