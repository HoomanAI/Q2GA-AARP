"""Generate algorithm-comparison figures (PNG, white background) and
companion .mat files for the LA wildfire case study, comparing the
"baseline" (normal operations) and "wildfire" (Palisades Fire active)
scenarios:

  - convergence.png          : shaded mean +/- std convergence curves, baseline vs wildfire
  - boxplot.png               : boxplot of final best fitness per algorithm, baseline vs wildfire
  - objectives.png             : grouped bar chart of objective components, baseline vs wildfire (Q2GA)
  - runtime_bar.png           : mean runtime per algorithm (baseline)
  - wildfire_impact_summary.png : %-change in fitness/penalty/unserved, baseline -> wildfire, per algorithm
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
SCENARIOS = ["baseline", "wildfire"]
SCEN_LABELS = {"baseline": "Baseline (normal operations)",
               "wildfire": "Wildfire (Palisades Fire active)"}

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


def _load(scenario):
    conv = pd.read_csv(os.path.join(RESULTS, f"convergence_{scenario}.csv"))
    summ = pd.read_csv(os.path.join(RESULTS, f"summary_{scenario}.csv"))
    return conv, summ


def plot_convergence(conv_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    mat_data = {}
    for ax, scenario in zip(axes, SCENARIOS):
        conv_df = conv_data[scenario]
        for algo in ALGOS:
            sub = conv_df[conv_df["algorithm"] == algo]
            piv = sub.pivot(index="seed", columns="generation", values="best_fitness")
            mean = piv.mean(axis=0).values
            std = piv.std(axis=0).values
            gens = piv.columns.values

            ax.plot(gens, mean, label=algo, color=COLORS[algo], linewidth=2)
            ax.fill_between(gens, mean - std, mean + std, color=COLORS[algo], alpha=0.18)

            mat_data[f"{scenario}_{algo}_mean"] = mean
            mat_data[f"{scenario}_{algo}_std"] = std
            mat_data[f"{scenario}_{algo}_gens"] = gens

        ax.set_xlabel("Generation")
        ax.set_ylabel("Best fitness (objective value)")
        ax.set_title(SCEN_LABELS[scenario])
        ax.legend()

    fig.suptitle("Convergence comparison -- LA wildfire case study (N=25 patients)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG_PNG, "convergence.png"), dpi=200)
    plt.close(fig)
    savemat(os.path.join(FIG_MAT, "convergence.mat"), mat_data)


def plot_boxplot(summ_data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    mat_data = {}
    for ax, scenario in zip(axes, SCENARIOS):
        summ_df = summ_data[scenario]
        data = [summ_df[summ_df["algorithm"] == algo]["best_fitness"].values for algo in ALGOS]
        bp = ax.boxplot(data, tick_labels=ALGOS, patch_artist=True)
        for patch, algo in zip(bp["boxes"], ALGOS):
            patch.set_facecolor(COLORS[algo])
            patch.set_alpha(0.6)
        ax.set_ylabel("Final best fitness")
        ax.set_title(SCEN_LABELS[scenario])
        for i, algo in enumerate(ALGOS):
            mat_data[f"{scenario}_{algo}"] = data[i]

    fig.suptitle("Final-fitness distribution -- LA wildfire case study (10 seeds)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG_PNG, "boxplot.png"), dpi=200)
    plt.close(fig)
    savemat(os.path.join(FIG_MAT, "boxplot.mat"), mat_data)


def plot_objectives(summ_data):
    components = ["C1", "C2", "C3", "penalty1", "penalty2"]
    labels = ["C1 (critical\ncompletion)", "C2 (moderate\ncompletion)",
              "C3 (minor\ncompletion)", "Penalty-1\n(critical delay)",
              "Penalty-2\n(moderate delay)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    mat_data = {}
    for ax, scenario in zip(axes, SCENARIOS):
        summ_df = summ_data[scenario]
        x = np.arange(len(components))
        width = 0.2
        for j, algo in enumerate(ALGOS):
            sub = summ_df[summ_df["algorithm"] == algo]
            means = [sub[c].mean() for c in components]
            ax.bar(x + (j - 1.5) * width, means, width, label=algo, color=COLORS[algo])
            mat_data[f"{scenario}_{algo}_means"] = np.array(means)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean value")
        ax.set_title(SCEN_LABELS[scenario])
        ax.legend()

    fig.suptitle("Objective-component breakdown -- LA wildfire case study (mean over 10 seeds)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG_PNG, "objectives.png"), dpi=200)
    plt.close(fig)

    mat_data["labels"] = np.array(labels, dtype=object)
    savemat(os.path.join(FIG_MAT, "objectives.mat"), mat_data)


def plot_runtime(summ_data):
    summ_df = summ_data["baseline"]
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [summ_df[summ_df["algorithm"] == algo]["runtime_s"].mean() for algo in ALGOS]
    stds = [summ_df[summ_df["algorithm"] == algo]["runtime_s"].std() for algo in ALGOS]
    ax.bar(ALGOS, means, yerr=stds, color=[COLORS[a] for a in ALGOS], capsize=5)
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Runtime comparison -- LA wildfire case study (10 seeds, baseline scenario)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PNG, "runtime_bar.png"), dpi=200)
    plt.close(fig)

    savemat(os.path.join(FIG_MAT, "runtime_bar.mat"), {
        "algorithms": np.array(ALGOS, dtype=object),
        "mean_runtime": np.array(means), "std_runtime": np.array(stds),
    })


def plot_wildfire_impact(summ_data):
    """Percentage change baseline -> wildfire for fitness, total delay penalty
    and number of unserved patients, per algorithm -- the key "impact of the
    wildfire on the AARP/Q2GA problem setting" figure."""
    base = summ_data["baseline"]
    wild = summ_data["wildfire"]

    def pct(b, w):
        return 100.0 * (w - b) / b if b != 0 else (0.0 if w == 0 else 100.0 * w)

    fitness_pct, penalty_pct, unserved_delta = [], [], []
    for algo in ALGOS:
        b = base[base["algorithm"] == algo]
        w = wild[wild["algorithm"] == algo]
        b_fit, w_fit = b["best_fitness"].mean(), w["best_fitness"].mean()
        b_pen = (b["penalty1"] + b["penalty2"]).mean()
        w_pen = (w["penalty1"] + w["penalty2"]).mean()
        b_uns, w_uns = b["n_unserved"].mean(), w["n_unserved"].mean()

        fitness_pct.append(pct(b_fit, w_fit))
        penalty_pct.append(pct(b_pen, w_pen))
        unserved_delta.append(w_uns - b_uns)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(ALGOS))
    width = 0.35

    axes[0].bar(x - width / 2, fitness_pct, width, label="Fitness increase (%)", color="#1f77b4")
    axes[0].bar(x + width / 2, penalty_pct, width, label="Total delay-penalty increase (%)", color="#d62728")
    axes[0].axhline(0, color="k", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ALGOS)
    axes[0].set_ylabel("Relative change, baseline -> wildfire (%)")
    axes[0].set_title("Fitness and delay-penalty degradation")
    axes[0].legend()
    for i, (f, p) in enumerate(zip(fitness_pct, penalty_pct)):
        axes[0].text(i - width / 2, f, f"{f:.1f}%", ha="center", va="bottom", fontsize=9)
        axes[0].text(i + width / 2, p, f"{p:.1f}%", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, unserved_delta, color=[COLORS[a] for a in ALGOS])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ALGOS)
    axes[1].set_ylabel("Increase in mean # unserved patients")
    axes[1].set_title("Additional unserved patients due to wildfire")
    for i, d in enumerate(unserved_delta):
        axes[1].text(i, d, f"{d:+.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Impact of the Palisades Fire network/road outage on quality of service\n"
                  "LA wildfire case study, 10 seeds")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(FIG_PNG, "wildfire_impact_summary.png"), dpi=200)
    plt.close(fig)

    savemat(os.path.join(FIG_MAT, "wildfire_impact_summary.mat"), {
        "algorithms": np.array(ALGOS, dtype=object),
        "fitness_pct": np.array(fitness_pct),
        "penalty_pct": np.array(penalty_pct),
        "unserved_delta": np.array(unserved_delta),
    })


def main():
    ensure_dirs()
    conv_data = {s: _load(s)[0] for s in SCENARIOS}
    summ_data = {s: _load(s)[1] for s in SCENARIOS}

    plot_convergence(conv_data)
    plot_boxplot(summ_data)
    plot_objectives(summ_data)
    plot_runtime(summ_data)
    plot_wildfire_impact(summ_data)

    print("Algorithm-comparison figures written to", FIG_PNG)
    print(".mat data written to", FIG_MAT)


if __name__ == "__main__":
    main()
