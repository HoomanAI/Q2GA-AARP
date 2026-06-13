"""Additional figures: parameter-tuning (MCDM) and cyber-risk sensitivity
analysis. Companion to make_figures.py -- white background PNGs plus .mat
exports for MATLAB .fig regeneration.

Figures produced:
  - tuning_<algo>.png      : MCDM score vs. hyper-parameter combination
  - sensitivity_gamma.png  : fitness / penalties vs. gamma_budget (cyber risk)
  - sensitivity_mttr.png   : fitness / penalties vs. MTTR (cyber risk)
  - sensitivity_gamma_infeasibility.png
  - sensitivity_mttr_infeasibility.png
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


def plot_tuning():
    for algo in ALGOS:
        df = pd.read_csv(os.path.join(RESULTS, f"tuning_{algo}.csv"))
        param_cols = [c for c in df.columns if c not in
                       ("mean_fitness", "std_fitness", "mean_runtime",
                        "norm_fitness", "norm_std", "norm_runtime", "mcdm_score", "rank")]
        labels = [", ".join(f"{c}={row[c]}" for c in param_cols) for _, row in df.iterrows()]

        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = np.arange(len(df))
        ax1.bar(x, df["mcdm_score"], color=COLORS[algo], alpha=0.7, label="MCDM score")
        ax1.set_ylabel("MCDM score (lower = better)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax1.set_title(f"{algo}: hyper-parameter tuning (MCDM ranking, medium instance)")

        ax2 = ax1.twinx()
        ax2.plot(x, df["mean_fitness"], "k-o", label="Mean final fitness")
        ax2.set_ylabel("Mean final fitness")
        ax2.grid(False)

        best_idx = int(df["mcdm_score"].idxmin())
        ax1.bar(best_idx, df["mcdm_score"].iloc[best_idx], color="gold",
                edgecolor="k", linewidth=1.5, label="Selected")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")

        fig.tight_layout()
        fig.savefig(os.path.join(FIG_PNG, f"tuning_{algo}.png"), dpi=200)
        plt.close(fig)

        savemat(os.path.join(FIG_MAT, f"tuning_{algo}.mat"), {
            "labels": np.array(labels, dtype=object),
            "mcdm_score": df["mcdm_score"].values,
            "mean_fitness": df["mean_fitness"].values,
            "best_idx": best_idx,
        })


def _plot_sensitivity(csv_name, xlabel, out_prefix, title_suffix):
    df = pd.read_csv(os.path.join(RESULTS, csv_name))

    # 1. Best fitness vs parameter
    fig, ax = plt.subplots(figsize=(8, 5))
    mat_data = {}
    for algo in ALGOS:
        sub = df[df["algorithm"] == algo]
        g = sub.groupby("value")["best_fitness"].agg(["mean", "std"]).reset_index()
        ax.plot(g["value"], g["mean"], "-o", color=COLORS[algo], label=algo)
        ax.fill_between(g["value"], g["mean"] - g["std"], g["mean"] + g["std"],
                         color=COLORS[algo], alpha=0.15)
        mat_data[f"{algo}_x"] = g["value"].values
        mat_data[f"{algo}_mean"] = g["mean"].values
        mat_data[f"{algo}_std"] = g["std"].values
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best fitness (objective value)")
    ax.set_title(f"Sensitivity of solution quality to {title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PNG, f"{out_prefix}_fitness.png"), dpi=200)
    plt.close(fig)
    savemat(os.path.join(FIG_MAT, f"{out_prefix}_fitness.mat"), mat_data)

    # 2. Penalty1 (critical-patient delay penalty) vs parameter
    fig, ax = plt.subplots(figsize=(8, 5))
    mat_data = {}
    for algo in ALGOS:
        sub = df[df["algorithm"] == algo]
        g = sub.groupby("value")["penalty1"].mean().reset_index()
        ax.plot(g["value"], g["penalty1"], "-o", color=COLORS[algo], label=algo)
        mat_data[f"{algo}_x"] = g["value"].values
        mat_data[f"{algo}_penalty1"] = g["penalty1"].values
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean critical-patient delay penalty (Penalty-1)")
    ax.set_title(f"Critical-patient delay penalty vs. {title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PNG, f"{out_prefix}_penalty1.png"), dpi=200)
    plt.close(fig)
    savemat(os.path.join(FIG_MAT, f"{out_prefix}_penalty1.mat"), mat_data)

    # 3. Infeasibility / unserved patients vs parameter
    fig, ax = plt.subplots(figsize=(8, 5))
    mat_data = {}
    for algo in ALGOS:
        sub = df[df["algorithm"] == algo]
        g = sub.groupby("value")["n_unserved"].mean().reset_index()
        ax.plot(g["value"], g["n_unserved"], "-o", color=COLORS[algo], label=algo)
        mat_data[f"{algo}_x"] = g["value"].values
        mat_data[f"{algo}_n_unserved"] = g["n_unserved"].values
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean number of unserved patients")
    ax.set_title(f"Service-disruption impact (unserved patients) vs. {title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PNG, f"{out_prefix}_unserved.png"), dpi=200)
    plt.close(fig)
    savemat(os.path.join(FIG_MAT, f"{out_prefix}_unserved.mat"), mat_data)


def main():
    ensure_dirs()
    plot_tuning()
    _plot_sensitivity("sensitivity_gamma.csv",
                       "Disruption budget  $\\Gamma$  (fraction of patients affected)",
                       "sens_gamma", "cyber-disruption budget $\\Gamma$ (gamma_budget)")
    _plot_sensitivity("sensitivity_mttr.csv",
                       "Mean time to repair, MTTR (time units)",
                       "sens_mttr", "disruption recovery time (MTTR)")
    print("Tuning + sensitivity figures written to", FIG_PNG)
    print(".mat data written to", FIG_MAT)


if __name__ == "__main__":
    main()
