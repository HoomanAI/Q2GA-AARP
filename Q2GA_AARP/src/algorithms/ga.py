"""Real-coded Genetic Algorithm (GA) for the AARP."""

import time
import numpy as np
from src.aarp_model import evaluate


def run(inst, pop_size=40, n_generations=100, seed=0,
        pc=0.85, pm=0.05, tournament_k=3, elitism=2):
    rng = np.random.default_rng(seed)
    D = inst.n_real_genes

    pop = rng.uniform(0, 1, size=(pop_size, D))
    fitness = np.array([evaluate(ind, inst) for ind in pop])

    history = []
    t0 = time.perf_counter()

    for gen in range(n_generations):
        order = np.argsort(fitness)
        elites = pop[order[:elitism]].copy()

        new_pop = [e.copy() for e in elites]
        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fitness, tournament_k, rng)
            p2 = _tournament(pop, fitness, tournament_k, rng)
            if rng.random() < pc:
                c1, c2 = _sbx_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = _mutate(c1, pm, rng)
            c2 = _mutate(c2, pm, rng)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.array(new_pop)
        fitness = np.array([evaluate(ind, inst) for ind in pop])

        history.append(float(fitness.min()))

    runtime = time.perf_counter() - t0
    best_idx = int(np.argmin(fitness))
    return dict(history=history, best=pop[best_idx], best_fitness=float(fitness[best_idx]),
                 runtime=runtime)


def _tournament(pop, fitness, k, rng):
    idx = rng.integers(0, len(pop), size=k)
    best = idx[np.argmin(fitness[idx])]
    return pop[best]


def _sbx_crossover(p1, p2, rng, eta=15):
    u = rng.random(len(p1))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)),
                     (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def _mutate(c, pm, rng, sigma=0.1):
    mask = rng.random(len(c)) < pm
    c = c.copy()
    c[mask] += rng.normal(0, sigma, size=mask.sum())
    return np.clip(c, 0, 1)
