"""Generate algorithm-comparison figures (PNG, white background) and
companion .mat files for the Ottawa case study:

  - convergence.png    : shaded mean +/- std convergence curves
  - boxplot.png        : boxplot of final best fitness per algorithm
  - objectives.png     : grouped bar chart of objective components
  - runtime_bar.png    : mean runtime per algorithm
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
    fig, ax = plt.subplots(figsize=(8, 5.5))
    mat_data = {}
    for algo in ALGOS:
        sub = conv_df[conv_df["algorithm"] == algo]
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
    ax.set_title("Convergence comparison -- Ottawa case study (N=25 patients)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PNG, "convergence.png"), dpi=200)
    plt.close(fig)
    savemat(os.path.join(FIG_MAT, "convergence.mat"), mat_data)


def plot_boxplot(summ_df):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    data = [summ_df[summ_df["algorithm"] == algo]["best_fitness"].values for algo in ALGOS]
    bp = ax.boxplot(data, tick_labels=ALGOS, patch_artist=True)
    for patch, algo in zip(bp["boxes"], ALGOS):
        patch.set_facecolor(COLORS[algo])
        patch.set_alpha(0.6)
    ax.set_ylabel("Final best fitness")
    ax.set_title("Final-fitness distribution -- Ottawa case study (10 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PNG, "boxplot.png"), dpi=200)
    plt.close(fig)

    savemat(os.path.join(FIG_MAT, "boxplot.mat"), {algo: data[i] for i, algo in enumerate(ALGOS)})


def plot_objectives(summ_df):
    components = ["C1", "C2", "C3", "penalty1", "penalty2"]
    labels = ["C1 (critical\ncompletion)", "C2 (moderate\ncompletion)",
              "C3 (minor\ncompletion)", "Penalty-1\n(critical delay)",
              "Penalty-2\n(moderate delay)"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(components))
    width = 0.2
    mat_data = {}
    for j, algo in enumerate(ALGOS):
        sub = summ_df[summ_df["algorithm"] == algo]
        means = [sub[c].mean() for c in components]
        ax.bar(x + (j - 1.5) * width, means, width, label=algo, color=COLORS[algo])
        mat_data[f"{algo}_means"] = np.array(means)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean value")
    ax.set_title("Objective-component breakdown -- Ottawa case study (mean over 10 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PNG, "objectives.png"), dpi=200)
    plt.close(fig)

    mat_data["labels"] = np.array(labels, dtype=object)
    savemat(os.path.join(FIG_MAT, "objectives.mat"), mat_data)


def plot_runtime(summ_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [summ_df[summ_df["algorithm"] == algo]["runtime_s"].mean() for algo in ALGOS]
    stds = [summ_df[summ_df["algorithm"] == algo]["runtime_s"].std() for algo in ALGOS]
    ax.bar(ALGOS, means, yerr=stds, color=[COLORS[a] for a in ALGOS], capsize=5)
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Runtime comparison -- Ottawa case study (10 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PNG, "runtime_bar.png"), dpi=200)
    plt.close(fig)

    savemat(os.path.join(FIG_MAT, "runtime_bar.mat"), {
        "algorithms": np.array(ALGOS, dtype=object),
        "mean_runtime": np.array(means), "std_runtime": np.array(stds),
    })


def main():
    ensure_dirs()
    conv_df = pd.read_csv(os.path.join(RESULTS, "convergence.csv"))
    summ_df = pd.read_csv(os.path.join(RESULTS, "summary.csv"))

    plot_convergence(conv_df)
    plot_boxplot(summ_df)
    plot_objectives(summ_df)
    plot_runtime(summ_df)

    print("Algorithm-comparison figures written to", FIG_PNG)
    print(".mat data written to", FIG_MAT)


if __name__ == "__main__":
    main()
