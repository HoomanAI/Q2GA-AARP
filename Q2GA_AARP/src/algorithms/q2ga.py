"""Reinforcement-Learning-assisted Quantum-inspired Genetic Algorithm (Q2GA).

Implements the Q-learning-based adaptive rotation-gate mechanism described
in Q2GA_Methodology.pdf:

  - Observation space O_t,i = (g, h, f) for each chromosome i in
    generation t:
        g : 1 if the population's mean fitness improved relative to the
            mean of the last 20 generations, else 0   (2 values)
        h : quartile bin of the average Hamming distance of chromosome i
            to the rest of the population                (4 values)
        f : quartile bin of chromosome i's fitness        (4 values)

  - Action space a_t,i in {1,...,5}: controls (1) how many genes are
    rotated, and (2) the integer multiplier in [-a, a] applied to the base
    rotation angle.

  - Reward: reward_t,i = (old_fitness - new_fitness) * 100 / old_fitness

  - Q-table update (running average):
        Q(O,a) <- Q(O,a) + (reward - Q(O,a)) / count(O,a)
"""

import time
import numpy as np
from src.aarp_model import evaluate, bits_to_real, GENE_BITS

BASE_DELTA_THETA = 0.05 * np.pi
N_ACTIONS = 5
HAM_BINS = 4
FIT_BINS = 4
GEN_WINDOW = 20


def _observe_population(alpha, beta, rng):
    p1 = beta ** 2
    return (rng.random(alpha.shape) < p1).astype(int)


def _quartile_bins(values, n_bins=4):
    """Assign each value to one of n_bins quantile-based bins (0..n_bins-1)."""
    if np.allclose(values, values[0]):
        return np.zeros(len(values), dtype=int)
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(values, edges)


def _rotate_qubits(alpha_i, beta_i, gene_idx, bits_i, best_bits, multiplier, rng,
                    random_dir_frac=0.20, base_delta_theta=BASE_DELTA_THETA):
    """Rotate all qubits belonging to genes in `gene_idx` for individual i."""
    qubit_idx = np.concatenate([np.arange(g * GENE_BITS, (g + 1) * GENE_BITS)
                                 for g in gene_idx])

    a, b = alpha_i[qubit_idx], beta_i[qubit_idx]
    x = bits_i[qubit_idx]
    bb = best_bits[qubit_idx]

    ab = a * b
    sign = np.sign(ab)
    zero = sign == 0
    if zero.any():
        sign[zero] = rng.choice([-1.0, 1.0], size=zero.sum())

    diff = x != bb
    case01 = diff & (x == 0) & (bb == 1)
    case10 = diff & (x == 1) & (bb == 0)

    base_sign = np.zeros_like(a)
    base_sign[case01] = sign[case01]
    base_sign[case10] = -sign[case10]

    # mostly aligned with best-genome direction, small fraction random direction
    rand_mask = rng.random(len(qubit_idx)) < random_dir_frac
    rand_sign = rng.choice([-1.0, 1.0], size=len(qubit_idx))
    final_sign = np.where(rand_mask, rand_sign, base_sign)

    theta = base_delta_theta * multiplier * final_sign
    new_a = a * np.cos(theta) - b * np.sin(theta)
    new_b = a * np.sin(theta) + b * np.cos(theta)
    norm = np.sqrt(new_a ** 2 + new_b ** 2)
    norm[norm == 0] = 1.0

    alpha_i = alpha_i.copy()
    beta_i = beta_i.copy()
    alpha_i[qubit_idx] = new_a / norm
    beta_i[qubit_idx] = new_b / norm
    return alpha_i, beta_i


def run(inst, pop_size=40, n_generations=100, seed=0,
        epsilon_start=0.20, epsilon_end=0.01, gen_window=5,
        base_delta_theta=0.05 * np.pi, random_dir_frac=0.02,
        learning_rate=0.5, min_coverage_frac=0.0):
    rng = np.random.default_rng(seed)
    n_genes = inst.n_real_genes
    n_qubits = inst.n_qubits

    alpha = np.full((pop_size, n_qubits), 1 / np.sqrt(2))
    beta = np.full((pop_size, n_qubits), 1 / np.sqrt(2))

    q_table = np.zeros((2, HAM_BINS, FIT_BINS, N_ACTIONS))
    counts = np.zeros((2, HAM_BINS, FIT_BINS, N_ACTIONS), dtype=int)

    best_bits = (rng.random(n_qubits) < 0.5).astype(int)
    best_real = bits_to_real(best_bits, n_genes)
    best_fitness = evaluate(best_real, inst)

    history = []
    mean_fitness_history = []
    t0 = time.perf_counter()

    for gen in range(n_generations):
        if n_generations > 1:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * gen / (n_generations - 1)
        else:
            epsilon = epsilon_start
        bits = _observe_population(alpha, beta, rng)
        fitnesses = np.empty(pop_size)
        for i in range(pop_size):
            fitnesses[i] = evaluate(bits_to_real(bits[i], n_genes), inst)

        gen_best_idx = int(np.argmin(fitnesses))
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = float(fitnesses[gen_best_idx])
            best_bits = bits[gen_best_idx].copy()

        # --- Observation space ---
        mean_fit_now = float(fitnesses.mean())
        if len(mean_fitness_history) > 0:
            window = mean_fitness_history[-gen_window:]
            g_obs = 1 if mean_fit_now < np.mean(window) else 0
        else:
            g_obs = 0
        mean_fitness_history.append(mean_fit_now)

        ham = np.zeros(pop_size)
        for i in range(pop_size):
            d = np.sum(bits[i][None, :] != bits, axis=1)
            ham[i] = (d.sum()) / max(1, pop_size - 1)
        h_bins = _quartile_bins(ham, HAM_BINS)
        f_bins = _quartile_bins(fitnesses, FIT_BINS)

        # --- Action selection (epsilon-greedy) and rotation ---
        for i in range(pop_size):
            obs = (g_obs, h_bins[i], f_bins[i])
            if rng.random() < epsilon:
                a_idx = rng.integers(0, N_ACTIONS)
            else:
                a_idx = int(np.argmax(q_table[obs]))
            action = a_idx + 1   # action value in {1,...,5}

            n_rotate = max(1, round(action / N_ACTIONS * n_genes),
                           round(min_coverage_frac * n_genes))
            n_rotate = min(n_rotate, n_genes)
            gene_idx = rng.choice(n_genes, size=n_rotate, replace=False)

            # Action always biases rotation toward the best-so-far genome;
            # the RL agent only tunes the magnitude (1..5 x base angle) and
            # coverage (n_rotate above), not the direction.
            multiplier = action

            new_alpha_i, new_beta_i = _rotate_qubits(
                alpha[i], beta[i], gene_idx, bits[i], best_bits, multiplier, rng,
                random_dir_frac=random_dir_frac, base_delta_theta=base_delta_theta)

            new_bits_i = (rng.random(n_qubits) < new_beta_i ** 2).astype(int)
            new_fitness_i = evaluate(bits_to_real(new_bits_i, n_genes), inst)

            old_fitness_i = fitnesses[i]
            if old_fitness_i != 0:
                reward = (old_fitness_i - new_fitness_i) * 100.0 / abs(old_fitness_i)
            else:
                reward = 0.0

            counts[obs][a_idx] += 1
            if learning_rate is None:
                lr = 1.0 / counts[obs][a_idx]
            else:
                lr = learning_rate
            q_table[obs][a_idx] += lr * (reward - q_table[obs][a_idx])

            alpha[i], beta[i] = new_alpha_i, new_beta_i

            if new_fitness_i < best_fitness:
                best_fitness = float(new_fitness_i)
                best_bits = new_bits_i.copy()

        history.append(best_fitness)

    runtime = time.perf_counter() - t0
    best_real = bits_to_real(best_bits, n_genes)
    return dict(history=history, best=best_real, best_fitness=best_fitness,
                 runtime=runtime, q_table=q_table)
