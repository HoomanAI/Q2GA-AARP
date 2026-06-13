"""Quantum-inspired Genetic Algorithm (QGA) for the AARP.

Each individual is a Q-chromosome of n_qubits qubits, each represented by
amplitude pair (alpha, beta) with alpha^2 + beta^2 = 1. In every generation
the population is observed (collapsed) into binary strings, decoded into
real-valued AARP chromosomes (see aarp_model.bits_to_real) and evaluated.
A fixed-angle rotation gate then moves every qubit toward the bit value of
the best-so-far solution (Han & Kim, 2002 style rotation table).
"""

import time
import numpy as np
from src.aarp_model import evaluate, bits_to_real, GENE_BITS

DELTA_THETA = 0.05 * np.pi   # fixed rotation magnitude


def _init_population(pop_size, n_qubits):
    alpha = np.full((pop_size, n_qubits), 1 / np.sqrt(2))
    beta = np.full((pop_size, n_qubits), 1 / np.sqrt(2))
    return alpha, beta


def _observe(alpha, beta, rng):
    p1 = beta ** 2
    return (rng.random(alpha.shape) < p1).astype(int)


def run(inst, pop_size=40, n_generations=100, seed=0, delta_theta=DELTA_THETA):
    rng = np.random.default_rng(seed)
    n_genes = inst.n_real_genes
    n_qubits = inst.n_qubits

    alpha, beta = _init_population(pop_size, n_qubits)

    best_bits = (rng.random(n_qubits) < 0.5).astype(int)
    best_real = bits_to_real(best_bits, n_genes)
    best_fitness = evaluate(best_real, inst)

    history = []
    t0 = time.perf_counter()

    for gen in range(n_generations):
        bits = _observe(alpha, beta, rng)
        fitnesses = np.empty(pop_size)
        for i in range(pop_size):
            real = bits_to_real(bits[i], n_genes)
            fitnesses[i] = evaluate(real, inst)

        gen_best_idx = int(np.argmin(fitnesses))
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = float(fitnesses[gen_best_idx])
            best_bits = bits[gen_best_idx].copy()

        ab = alpha * beta
        sign = np.sign(ab)
        zero_mask = sign == 0
        if zero_mask.any():
            sign[zero_mask] = rng.choice([-1.0, 1.0], size=zero_mask.sum())

        diff_mask = bits != best_bits[None, :]
        case01 = diff_mask & (bits == 0) & (best_bits[None, :] == 1)
        case10 = diff_mask & (bits == 1) & (best_bits[None, :] == 0)

        theta = np.zeros_like(alpha)
        theta[case01] = delta_theta * sign[case01]
        theta[case10] = -delta_theta * sign[case10]

        new_alpha = alpha * np.cos(theta) - beta * np.sin(theta)
        new_beta = alpha * np.sin(theta) + beta * np.cos(theta)
        norm = np.sqrt(new_alpha ** 2 + new_beta ** 2)
        alpha, beta = new_alpha / norm, new_beta / norm

        history.append(best_fitness)

    runtime = time.perf_counter() - t0
    best_real = bits_to_real(best_bits, n_genes)
    return dict(history=history, best=best_real, best_fitness=best_fitness, runtime=runtime)
