"""Simulated Annealing (SA) for the AARP.

To allow a fair convergence comparison against the population-based
algorithms (GA, QGA, Q2GA), SA performs `pop_size` perturbation/acceptance
trials per "generation" and records the best fitness found so far at the
end of each generation.
"""

import time
import numpy as np
from src.aarp_model import evaluate


def run(inst, pop_size=40, n_generations=100, seed=0,
        T0=5.0, alpha=0.97, sigma=0.15):
    rng = np.random.default_rng(seed)
    D = inst.n_real_genes

    current = rng.uniform(0, 1, size=D)
    current_f = evaluate(current, inst)
    best, best_f = current.copy(), current_f

    history = []
    T = T0
    t0 = time.perf_counter()

    for gen in range(n_generations):
        for _ in range(pop_size):
            candidate = current.copy()
            n_flip = rng.integers(1, max(2, D // 5))
            idx = rng.choice(D, size=n_flip, replace=False)
            candidate[idx] += rng.normal(0, sigma, size=n_flip)
            candidate = np.clip(candidate, 0, 1)

            cand_f = evaluate(candidate, inst)
            delta = cand_f - current_f
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-6)):
                current, current_f = candidate, cand_f
                if current_f < best_f:
                    best, best_f = current.copy(), current_f

        T *= alpha
        history.append(float(best_f))

    runtime = time.perf_counter() - t0
    return dict(history=history, best=best, best_fitness=float(best_f), runtime=runtime)
