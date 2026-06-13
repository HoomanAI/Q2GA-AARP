"""Cyber-risk sensitivity sweep for the Calgary case study.

Sweeps the disruption budget (Gamma, fraction of patients affected) and the
mean time to repair (MTTR), running Q2GA (the recommended algorithm from the
main study) with a reduced budget (5 seeds, 30 generations) at each setting,
to characterise how quality of service degrades under cyber-risk conditions
for this specific instance.

Outputs:
  - results/sensitivity_gamma.csv
  - results/sensitivity_mttr.csv
"""

import os
import sys
import csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_AARP = os.path.join(os.path.dirname(ROOT), "Q2GA_AARP")
sys.path.insert(0, ROOT_AARP)
sys.path.insert(0, ROOT)

from src.aarp_model import evaluate_full  # noqa: E402
from src.algorithms import q2ga  # noqa: E402
from src.calgary_instance import build_instance, N_PATIENTS  # noqa: E402

Q2GA_KWARGS = dict(epsilon_start=0.30, base_delta_theta=0.05 * np.pi)

N_SEEDS = 5
POP_SIZE = 20
N_GENERATIONS = 30

RESULTS_DIR = os.path.join(ROOT, "results")

GAMMA_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MTTR_VALUES = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0]


def _run_sweep(param, values, build_kwargs_fn, out_name):
    rows = []
    for val in values:
        inst = build_instance(seed=42, **build_kwargs_fn(val))
        for seed in range(N_SEEDS):
            res = q2ga.run(inst, pop_size=POP_SIZE, n_generations=N_GENERATIONS, seed=seed,
                            **Q2GA_KWARGS)
            comp = evaluate_full(res["best"], inst)
            rows.append([param, val, "Q2GA", seed, res["best_fitness"],
                         comp["C1"], comp["C2"], comp["C3"],
                         comp["penalty1"], comp["penalty2"],
                         comp["infeasibility"], comp["n_unserved"]])
        print(f"{param}={val}: done ({N_SEEDS} seeds)")

    out_path = os.path.join(RESULTS_DIR, out_name)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "value", "algorithm", "seed", "best_fitness",
                     "C1", "C2", "C3", "penalty1", "penalty2", "infeasibility", "n_unserved"])
        w.writerows(rows)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Gamma (disruption budget) sweep ===")
    _run_sweep("gamma_frac", GAMMA_FRACS,
               lambda v: dict(gamma_budget=max(0, round(v * N_PATIENTS))),
               "sensitivity_gamma.csv")

    print("\n=== MTTR sweep ===")
    _run_sweep("mttr", MTTR_VALUES,
               lambda v: dict(mttr=v),
               "sensitivity_mttr.csv")


if __name__ == "__main__":
    main()
