"""Sensitivity analysis on the uncertainty / disruption parameters
(gamma_budget, mttr) -- interpreted as a proxy for cyber-attack-induced
communication / control disruptions on the autonomous-ambulance fleet.

Model link (see aarp_model.generate_instance, Eq. 4-5 of the math model):
  - `gamma_budget` (Gamma): number of patients (out of N) whose service is
    affected by a disruption event. A larger Gamma represents a more
    widespread cyber/communication-disruption campaign (more vehicles or
    links knocked out simultaneously).
  - `mttr`: mean time to repair / recover from a disruption. A larger MTTR
    represents a slower incident-response / recovery capability (e.g. a more
    severe or harder-to-mitigate cyber-attack).

Both parameters feed into `theta_i`, the realised extra service-time buffer
for affected patients (Eq. 4-5), which directly increases C1/C2/C3 and the
delay penalties (Eq. 1-2).

For each setting we run all four algorithms (GA, SA, QGA, Q2GA) on the
*medium* instance with N_SEEDS repetitions and record:
  - mean best fitness, std
  - mean C1 (critical-patient makespan), penalty1, penalty2
  - mean infeasibility / unserved patients

Outputs:
  - results/sensitivity_gamma.csv
  - results/sensitivity_mttr.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aarp_model import generate_instance, evaluate_full
from src.algorithms import ga, sa, qga, q2ga

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

POP_SIZE = 20
N_GENERATIONS = 50
N_SEEDS = 5
SIZE = "medium"
BASE_SEED = 101

ALGORITHMS = {"GA": ga.run, "SA": sa.run, "QGA": qga.run, "Q2GA": q2ga.run}

# Gamma budget as a fraction of n_patients (default is 0.20)
GAMMA_FRACTIONS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
# MTTR values in time units (default is 8.0)
MTTR_VALUES = [2.0, 4.0, 8.0, 12.0, 16.0, 24.0]


def _run_setting(inst, label, value):
    rows = []
    for algo_name, algo_fn in ALGORITHMS.items():
        for seed in range(N_SEEDS):
            res = algo_fn(inst, pop_size=POP_SIZE, n_generations=N_GENERATIONS, seed=seed)
            comp = evaluate_full(res["best"], inst)
            rows.append({
                "param": label, "value": value, "algorithm": algo_name, "seed": seed,
                "best_fitness": res["best_fitness"], "C1": comp["C1"], "C2": comp["C2"],
                "C3": comp["C3"], "penalty1": comp["penalty1"], "penalty2": comp["penalty2"],
                "infeasibility": comp["infeasibility"], "n_unserved": comp["n_unserved"],
            })
    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Sweep gamma_budget (fraction of patients affected by disruptions) ---
    rows = []
    for frac in GAMMA_FRACTIONS:
        base = generate_instance(SIZE, seed=BASE_SEED)
        gb = int(round(frac * base.n_patients))
        inst = generate_instance(SIZE, seed=BASE_SEED, gamma_budget=gb)
        print(f"[gamma] frac={frac:.2f} -> gamma_budget={gb}")
        rows += _run_setting(inst, "gamma_frac", frac)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "sensitivity_gamma.csv"), index=False)

    # --- Sweep mttr (mean time to repair after a disruption) ---
    rows = []
    for mttr in MTTR_VALUES:
        inst = generate_instance(SIZE, seed=BASE_SEED, mttr=mttr)
        print(f"[mttr] mttr={mttr}")
        rows += _run_setting(inst, "mttr", mttr)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "sensitivity_mttr.csv"), index=False)

    print(f"\nSaved sensitivity results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
