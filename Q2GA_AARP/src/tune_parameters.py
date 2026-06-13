"""Hyper-parameter tuning for GA, SA, QGA, Q2GA via grid search + MCDM ranking.

For each algorithm a small grid of hyper-parameter combinations is evaluated
on the *medium* AARP instance (N_SEEDS repetitions, POP_SIZE, N_GENERATIONS).
For every combination we record:
  - mean_fitness  : mean final best fitness (lower = better quality)
  - std_fitness   : standard deviation of final best fitness (lower = more
                    robust/consistent)
  - mean_runtime  : mean wall-clock runtime in seconds (lower = cheaper)

A simple weighted-sum MCDM (multi-criteria decision making) score is then
computed after min-max normalising each criterion within the algorithm's
grid:

    score = w_fit * norm(mean_fitness) + w_std * norm(std_fitness)
            + w_time * norm(mean_runtime)

with (w_fit, w_std, w_time) = (0.6, 0.2, 0.2). The lowest-scoring combination
is selected as the "tuned" configuration for that algorithm.

Outputs:
  - results/tuning_<algo>.csv   one row per grid combination, with score/rank
  - results/tuning_best.csv     the selected (best) configuration per algo
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aarp_model import generate_instance
from src.algorithms import ga, sa, qga, q2ga

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

POP_SIZE = 20
N_GENERATIONS = 50
N_SEEDS = 5
TUNE_SIZE = "medium"
TUNE_SEED = 101

W_FIT, W_STD, W_TIME = 0.6, 0.2, 0.2

GRIDS = {
    "GA": {
        "pc": [0.70, 0.85, 0.95],
        "pm": [0.02, 0.05, 0.10],
    },
    "SA": {
        "T0": [2.0, 5.0, 10.0],
        "alpha": [0.95, 0.97, 0.99],
    },
    "QGA": {
        "delta_theta_mult": [0.02, 0.05, 0.08],   # x pi
    },
    "Q2GA": {
        "epsilon_start": [0.10, 0.20, 0.30],
        "base_delta_theta_mult": [0.03, 0.05, 0.07],   # x pi
    },
}

RUNNERS = {"GA": ga.run, "SA": sa.run, "QGA": qga.run, "Q2GA": q2ga.run}


def _build_kwargs(algo, combo):
    if algo == "GA":
        return dict(pc=combo["pc"], pm=combo["pm"])
    if algo == "SA":
        return dict(T0=combo["T0"], alpha=combo["alpha"])
    if algo == "QGA":
        return dict(delta_theta=combo["delta_theta_mult"] * np.pi)
    if algo == "Q2GA":
        return dict(epsilon_start=combo["epsilon_start"],
                     base_delta_theta=combo["base_delta_theta_mult"] * np.pi)
    raise ValueError(algo)


def _grid_combos(grid):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def _minmax_norm(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def tune_algorithm(algo, inst):
    runner = RUNNERS[algo]
    rows = []
    for combo in _grid_combos(GRIDS[algo]):
        kwargs = _build_kwargs(algo, combo)
        fits, times = [], []
        for seed in range(N_SEEDS):
            res = runner(inst, pop_size=POP_SIZE, n_generations=N_GENERATIONS,
                          seed=seed, **kwargs)
            fits.append(res["best_fitness"])
            times.append(res["runtime"])
        row = dict(combo)
        row["mean_fitness"] = float(np.mean(fits))
        row["std_fitness"] = float(np.std(fits))
        row["mean_runtime"] = float(np.mean(times))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["norm_fitness"] = _minmax_norm(df["mean_fitness"])
    df["norm_std"] = _minmax_norm(df["std_fitness"])
    df["norm_runtime"] = _minmax_norm(df["mean_runtime"])
    df["mcdm_score"] = (W_FIT * df["norm_fitness"] + W_STD * df["norm_std"]
                         + W_TIME * df["norm_runtime"])
    df["rank"] = df["mcdm_score"].rank(method="min").astype(int)
    df = df.sort_values("mcdm_score").reset_index(drop=True)
    return df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    inst = generate_instance(TUNE_SIZE, seed=TUNE_SEED)
    print(f"Tuning on '{TUNE_SIZE}' instance "
          f"(N={inst.n_patients}, V={inst.n_vehicles}, H={inst.n_hospitals}), "
          f"pop={POP_SIZE}, gens={N_GENERATIONS}, seeds={N_SEEDS}")

    best_rows = []
    for algo in ["GA", "SA", "QGA", "Q2GA"]:
        print(f"\n=== Tuning {algo} ===")
        df = tune_algorithm(algo, inst)
        out_path = os.path.join(RESULTS_DIR, f"tuning_{algo}.csv")
        df.to_csv(out_path, index=False)
        print(df.to_string(index=False))
        best = df.iloc[0].to_dict()
        best_row = {"algorithm": algo}
        best_row.update(best)
        best_rows.append(best_row)
        print(f"--> Selected: { {k: v for k, v in best.items() if k in GRIDS[algo]} } "
              f"(mcdm_score={best['mcdm_score']:.4f})")

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(os.path.join(RESULTS_DIR, "tuning_best.csv"), index=False)
    print(f"\nSaved tuning results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
