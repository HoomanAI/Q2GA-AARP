"""Generate all figures (PNG, white background) and companion .mat files
for regenerating MATLAB .fig figures via figures/matlab/make_all_figs.m.

Figures produced (per instance size where applicable):
  - conv_<size>.png        : shaded mean +/- std convergence curves
  - box_<size>.png         : boxplot of final best fitness
  - objectives_<size>.png  : grouped bar chart of objective components
  - runtime_bar.png        : mean runtime per algorithm/size
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import savemat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIG_PNG = os.path.join(ROOT, "figures", "png")
FIG_MAT = os.path.join(ROOT, "figures", "mat")

ALGOS = ["GA", "SA", "QGA", "Q2GA"]
COLORS = {"GA": "#1f77b4", "SA": "#ff7f0e", "QGA": "#2ca02c", "Q2GA": "#d62728"}
SIZES = ["small", "medium", "large"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def ensure_dirs():
    os.makedirs(FIG_PNG, exist_ok=True)
    os.makedirs(FIG_MAT, exist_ok=True)


def plot_convergence(conv_df):
    for size in SIZES:
        fig, ax = plt.subplots(figsize=(7, 5))
        mat_data = {}
        for algo in ALGOS:
            sub = conv_df[(conv_df["size"] == size) & (conv_df["algorithm"] == algo)]
            piv = sub.pivot(index="seed", columns="generation", values="best_fitness")
            mean = piv.mean(axis=0).values
            std = piv.std(axis=0).values
            gens = piv.columns.values

            ax.plot(gens, mean, label=algo, color=COLORS[algo], linewidth=2)
            ax.fill_between(gens, mean - std, mean + std, color=COLORS[algo], alpha=0.18)

            mat_data[f"{algo}_mean"] = mean
            mat_data[f"{algo}_std"] = std
            mat_data[f"{algo}_gens"] = gens

        ax.set_xlabel("Generation")
        ax.set_ylabel("Best fitness (objective value)")
        ax.set_title(f"Convergence comparison — {size} instance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_PNG, f"conv_{size}.png"), dpi=200)
        plt.close(fig)

        savemat(os.path.join(FIG_MAT, f"conv_{size}.mat"), mat_data)


def plot_boxplots(summ_df):
    for size in SIZES:
        fig, ax = plt.subplots(figsize=(7, 5))
        data = []
        mat_data = {}
        for algo in ALGOS:
            vals = summ_df[(summ_df["size"] == size) & (summ_df["algorithm"] == algo)]["best_fitness"].values
            data.append(vals)
            mat_data[algo] = vals

        bp = ax.boxplot(data, labels=ALGOS, patch_artist=True)
        for patch, algo in zip(bp["boxes"], ALGOS):
            patch.set_facecolor(COLORS[algo])
            patch.set_alpha(0.5)

        ax.set_ylabel("Final best fitness")
        ax.set_title(f"Final solution quality — {size} instance ({len(data[0])} runs)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_PNG, f"box_{size}.png"), dpi=200)
        plt.close(fig)

        savemat(os.path.join(FIG_MAT, f"box_{size}.mat"), mat_data)


def plot_objectives(summ_df):
    components = ["C1", "C2", "C3", "penalty1", "penalty2"]
    for size in SIZES:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(components))
        width = 0.2
        mat_data = {"components": np.array(components, dtype=object)}
        for j, algo in enumerate(ALGOS):
            sub = summ_df[(summ_df["size"] == size) & (summ_df["algorithm"] == algo)]
            means = [sub[c].mean() for c in components]
            ax.bar(x + (j - 1.5) * width, means, width, label=algo, color=COLORS[algo])
            mat_data[f"{algo}_means"] = np.array(means)

        ax.set_xticks(x)
        ax.set_xticklabels(["C1 (critical)", "C2 (moderate)", "C3 (minor)",
                             "Penalty-1", "Penalty-2"])
        ax.set_ylabel("Mean value")
        ax.set_title(f"Objective components of best solutions — {size} instance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_PNG, f"objectives_{size}.png"), dpi=200)
        plt.close(fig)

        savemat(os.path.join(FIG_MAT, f"objectives_{size}.mat"), mat_data)


def plot_runtime(summ_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(SIZES))
    width = 0.2
    mat_data = {}
    for j, algo in enumerate(ALGOS):
        means = [summ_df[(summ_df["size"] == s) & (summ_df["algorithm"] == algo)]["runtime_s"].mean()
                 for s in SIZES]
        ax.bar(x + (j - 1.5) * width, means, width, label=algo, color=COLORS[algo])
        mat_data[f"{algo}_means"] = np.array(means)

    ax.set_xticks(x)
    ax.set_xticklabels(SIZES)
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Mean runtime per algorithm and instance size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PNG, "runtime_bar.png"), dpi=200)
    plt.close(fig)

    savemat(os.path.join(FIG_MAT, "runtime_bar.mat"), mat_data)


def main():
    ensure_dirs()
    conv_df = pd.read_csv(os.path.join(RESULTS, "convergence.csv"))
    summ_df = pd.read_csv(os.path.join(RESULTS, "summary.csv"))

    plot_convergence(conv_df)
    plot_boxplots(summ_df)
    plot_objectives(summ_df)
    plot_runtime(summ_df)

    print("Figures written to", FIG_PNG)
    print(".mat data written to", FIG_MAT)


if __name__ == "__main__":
    main()
