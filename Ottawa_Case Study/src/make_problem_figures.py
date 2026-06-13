"""Generate problem/instance characteristic figures (PNG, white background)
and companion .mat files for the Ottawa case study.

Figures produced:
  - instance_layout_xy.png         : patient/hospital layout (local km frame)
  - patient_demographics.png       : patient-type counts
  - service_dropoff_times.png      : service-time / drop-off-time histograms
  - time_window_thresholds.png     : a1/a2 time-window thresholds by type
  - penalty_function_shape.png     : penalty-vs-arrival-time curves (Eq. 1-2)
  - goal_function_weights.png      : w1/w2/w3 weighted-sum structure
  - disruption_scenario.png        : realised theta_i per patient
  - qos_vs_gamma.png               : QoS metrics vs disruption budget Gamma
  - qos_vs_mttr.png                : QoS metrics vs MTTR (+availability axis)
  - availability_surface.png       : availability(MTTR, MTBF) contour map
  - cyber_risk_degradation_summary.png : QoS degradation, baseline vs worst-case
"""

import os
import sys
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
sys.path.insert(0, ROOT)

from src.ottawa_instance import build_instance, HOSPITALS, PATIENTS  # noqa: E402

TYPE_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c"}
TYPE_LABELS = {1: "Critical (Type 1)", 2: "Moderate (Type 2)", 3: "Minor (Type 3)"}

DEFAULT_MTBF = 120.0
DEFAULT_MTTR = 10.0
DEFAULT_GAMMA_FRAC = 0.20

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


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PNG, f"{name}.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_instance_layout(inst):
    fig, ax = plt.subplots(figsize=(8, 7))
    for t in (1, 2, 3):
        mask = inst.patient_type == t
        ax.scatter(inst.coords_patients[mask, 0], inst.coords_patients[mask, 1],
                   color=TYPE_COLORS[t], label=TYPE_LABELS[t], s=60, alpha=0.85,
                   edgecolor="k", linewidth=0.5)
    ax.scatter(inst.coords_hospitals[:, 0], inst.coords_hospitals[:, 1],
               marker="*", s=400, color="gold", edgecolor="k", linewidth=1,
               label="Hospitals", zorder=5)
    for i, p in enumerate(PATIENTS):
        ax.annotate(str(i), (inst.coords_patients[i, 0], inst.coords_patients[i, 1]),
                    fontsize=7, ha="center", va="center")
    ax.set_xlabel("x (km, east of Ottawa city centre)")
    ax.set_ylabel("y (km, north of Ottawa city centre)")
    ax.set_title("Ottawa case-study instance layout\n"
                  f"(N={inst.n_patients} patients, H={inst.n_hospitals} hospitals, "
                  f"M={inst.n_vehicles} vehicles)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    _save(fig, "instance_layout_xy")

    savemat(os.path.join(FIG_MAT, "instance_layout_xy.mat"), {
        "patient_x": inst.coords_patients[:, 0], "patient_y": inst.coords_patients[:, 1],
        "patient_type": inst.patient_type,
        "hospital_x": inst.coords_hospitals[:, 0], "hospital_y": inst.coords_hospitals[:, 1],
    })


# ---------------------------------------------------------------------------
def fig_patient_demographics(inst):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = [int((inst.patient_type == t).sum()) for t in (1, 2, 3)]
    ax.bar([TYPE_LABELS[t] for t in (1, 2, 3)], counts,
           color=[TYPE_COLORS[t] for t in (1, 2, 3)])
    for i, c in enumerate(counts):
        ax.text(i, c + 0.1, str(c), ha="center", fontsize=11)
    ax.set_ylabel("Number of patients")
    ax.set_title("Patient-type composition -- Ottawa case study")
    _save(fig, "patient_demographics")
    savemat(os.path.join(FIG_MAT, "patient_demographics.mat"), {"counts": np.array(counts)})


# ---------------------------------------------------------------------------
def fig_service_dropoff_times(inst):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(inst.service_time, bins=10, color="#1f77b4", alpha=0.8)
    axes[0].set_xlabel("Service time $s_i$ (minutes)")
    axes[0].set_ylabel("Number of patients")
    axes[0].set_title("On-scene service-time distribution")

    dropoff = inst.dropoff_time[inst.dropoff_time > 0]
    axes[1].hist(dropoff, bins=10, color="#ff7f0e", alpha=0.8)
    axes[1].set_xlabel("Hospital drop-off time (minutes)")
    axes[1].set_ylabel("Number of patients")
    axes[1].set_title("Drop-off-time distribution (Type 1/2 only)")

    _save(fig, "service_dropoff_times")
    savemat(os.path.join(FIG_MAT, "service_dropoff_times.mat"), {
        "service_time": inst.service_time, "dropoff_time": dropoff,
    })


# ---------------------------------------------------------------------------
def fig_time_window_thresholds(inst):
    fig, ax = plt.subplots(figsize=(7, 5))
    a1_t1 = inst.a1[inst.patient_type == 1]
    a2_t1 = inst.a2[inst.patient_type == 1]
    a1_t2 = inst.a1[inst.patient_type == 2]
    a2_t2 = inst.a2[inst.patient_type == 2]

    bp1 = ax.boxplot([a1_t1, a1_t2], positions=[1, 3], widths=0.6, patch_artist=True,
                      tick_labels=["", ""])
    bp2 = ax.boxplot([a2_t1, a2_t2], positions=[1.7, 3.7], widths=0.6, patch_artist=True,
                      tick_labels=["", ""])
    for box in bp1["boxes"]:
        box.set_facecolor("#1f77b4")
        box.set_alpha(0.6)
    for box in bp2["boxes"]:
        box.set_facecolor("#d62728")
        box.set_alpha(0.6)
    ax.set_xticks([1.35, 4.05])
    ax.set_xticklabels(["Type 1 (critical)", "Type 2 (moderate)"])
    ax.set_ylabel("Threshold (minutes from dispatch)")
    ax.set_title("Semi-soft time-window thresholds -- Ottawa case study")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              ["a1 (no-penalty threshold)", "a2 (critical threshold)"], loc="best")
    _save(fig, "time_window_thresholds")

    savemat(os.path.join(FIG_MAT, "time_window_thresholds.mat"), {
        "a1_type1": a1_t1, "a2_type1": a2_t1, "a1_type2": a1_t2, "a2_type2": a2_t2,
    })


# ---------------------------------------------------------------------------
def _penalty(arrival, a1, a2, lam):
    pen = np.zeros_like(arrival)
    mid = (arrival > a1) & (arrival <= a2)
    high = arrival > a2
    pen[mid] = (arrival[mid] - a1)
    pen[high] = (a2 - a1) + 2.0 * (arrival[high] - a2)
    return lam * pen


def fig_penalty_function(inst):
    a1_t1 = float(inst.a1[inst.patient_type == 1].mean())
    a2_t1 = float(inst.a2[inst.patient_type == 1].mean())
    a1_t2 = float(inst.a1[inst.patient_type == 2].mean())
    a2_t2 = float(inst.a2[inst.patient_type == 2].mean())

    arrival = np.linspace(0, max(a2_t1, a2_t2) * 1.4, 400)
    pen1 = _penalty(arrival, a1_t1, a2_t1, inst.lambda1)
    pen2 = _penalty(arrival, a1_t2, a2_t2, inst.lambda2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(arrival, pen1, color=TYPE_COLORS[1], linewidth=2,
            label=f"Type 1 (critical), lambda1={inst.lambda1}, a1={a1_t1:.1f}, a2={a2_t1:.1f}")
    ax.plot(arrival, pen2, color=TYPE_COLORS[2], linewidth=2,
            label=f"Type 2 (moderate), lambda2={inst.lambda2}, a1={a1_t2:.1f}, a2={a2_t2:.1f}")
    for a1, a2, c in ((a1_t1, a2_t1, TYPE_COLORS[1]), (a1_t2, a2_t2, TYPE_COLORS[2])):
        ax.axvline(a1, color=c, linestyle="--", alpha=0.5)
        ax.axvline(a2, color=c, linestyle=":", alpha=0.5)
    ax.set_xlabel("Patient arrival time $a_i$ (minutes from dispatch)")
    ax.set_ylabel("Delay penalty")
    ax.set_title("Semi-soft time-window delay-penalty function -- Ottawa case study\n"
                  "(dashed = a1 no-penalty threshold, dotted = a2 critical threshold)")
    ax.legend(fontsize=9)
    _save(fig, "penalty_function_shape")

    savemat(os.path.join(FIG_MAT, "penalty_function_shape.mat"), {
        "arrival": arrival, "penalty_type1": pen1, "penalty_type2": pen2,
        "a1_type1": a1_t1, "a2_type1": a2_t1, "a1_type2": a1_t2, "a2_type2": a2_t2,
    })


# ---------------------------------------------------------------------------
def fig_goal_function_weights(inst):
    weights = [inst.w1, inst.w2, inst.w3]
    labels = ["w1 . C1\n(critical completion)", "w2 . C2\n(moderate completion)",
              "w3 . C3\n(minor completion)"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, weights, color=["#d62728", "#ff7f0e", "#2ca02c"])
    for i, w in enumerate(weights):
        ax.text(i, w + 0.01, f"{w:.2f}", ha="center", fontsize=11)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 0.6)
    ax.set_title("Goal-function weighted-sum structure (Eq. 6-8)\n"
                  "fitness = w1.C1 + w2.C2 + w3.C3 + penalty1 + penalty2 + infeasibility")
    _save(fig, "goal_function_weights")
    savemat(os.path.join(FIG_MAT, "goal_function_weights.mat"),
            {"w1": inst.w1, "w2": inst.w2, "w3": inst.w3})


# ---------------------------------------------------------------------------
def fig_disruption_scenario(inst):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = np.where(inst.theta > 0, "#d62728", "#cccccc")
    ax.bar(np.arange(inst.n_patients), inst.theta, color=colors)
    ax.set_xticks(np.arange(inst.n_patients))
    ax.set_xlabel("Patient index")
    ax.set_ylabel("Realised disruption buffer $\\theta_i$ (minutes)")
    ax.set_title(f"Realised disruption scenario -- Ottawa case study "
                  f"(Gamma={inst.gamma_budget} of {inst.n_patients} patients affected, "
                  f"MTTR={inst.mttr})")
    _save(fig, "disruption_scenario")
    savemat(os.path.join(FIG_MAT, "disruption_scenario.mat"), {
        "theta": inst.theta, "gamma_budget": inst.gamma_budget, "mttr": inst.mttr,
    })


# ---------------------------------------------------------------------------
def _aggregate_sensitivity(csv_name):
    df = pd.read_csv(os.path.join(RESULTS, csv_name))
    df["penalty_total"] = df["penalty1"] + df["penalty2"]
    g = df.groupby("value").agg(
        fitness_mean=("best_fitness", "mean"),
        fitness_std=("best_fitness", "std"),
        penalty_mean=("penalty_total", "mean"),
        penalty_std=("penalty_total", "std"),
        unserved_mean=("n_unserved", "mean"),
    ).reset_index()
    return g.sort_values("value")


def fig_qos_vs_gamma():
    g = _aggregate_sensitivity("sensitivity_gamma.csv")
    gamma = g["value"].values
    fit_m, fit_s = g["fitness_mean"].values, g["fitness_std"].values
    pen_m, pen_s = g["penalty_mean"].values, g["penalty_std"].values
    uns_m = g["unserved_mean"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(gamma, fit_m, "-o", color="#1f77b4", linewidth=2)
    axes[0].fill_between(gamma, fit_m - fit_s, fit_m + fit_s, color="#1f77b4", alpha=0.15)
    axes[0].set_ylabel("Mean best fitness (Q2GA)")
    axes[0].set_title("Overall fitness vs. Gamma")

    axes[1].plot(gamma, pen_m, "-o", color="#d62728", linewidth=2)
    axes[1].fill_between(gamma, pen_m - pen_s, pen_m + pen_s, color="#d62728", alpha=0.15)
    axes[1].set_ylabel("Mean total delay penalty (Penalty-1 + Penalty-2)")
    axes[1].set_title("Service-quality penalty vs. Gamma")

    axes[2].plot(gamma, uns_m, "-o", color="#9467bd", linewidth=2)
    axes[2].set_ylabel("Mean number of unserved patients")
    axes[2].set_title("Coverage failure vs. Gamma")

    for ax in axes:
        ax.set_xlabel("Disruption budget Gamma (fraction of patients affected)")
        ax.axvline(DEFAULT_GAMMA_FRAC, color="gray", linestyle="--", alpha=0.6)

    fig.suptitle("Impact of cyber-disruption breadth (Gamma) on quality of service\n"
                  "Ottawa case study, Q2GA, 5 seeds; dashed line = default Gamma=0.20")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG_PNG, "qos_vs_gamma.png"), dpi=200)
    plt.close(fig)

    savemat(os.path.join(FIG_MAT, "qos_vs_gamma.mat"), {
        "gamma_frac": gamma, "fitness_mean": fit_m, "fitness_std": fit_s,
        "penalty_mean": pen_m, "penalty_std": pen_s, "unserved_mean": uns_m,
    })


def fig_qos_vs_mttr():
    g = _aggregate_sensitivity("sensitivity_mttr.csv")
    mttr = g["value"].values
    fit_m, fit_s = g["fitness_mean"].values, g["fitness_std"].values
    pen_m, pen_s = g["penalty_mean"].values, g["penalty_std"].values

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(mttr, fit_m, "-o", color="#1f77b4", linewidth=2, label="Mean best fitness (Q2GA)")
    ax1.fill_between(mttr, fit_m - fit_s, fit_m + fit_s, color="#1f77b4", alpha=0.15)
    ax1.set_xlabel("MTTR (mean time to repair, minutes)")
    ax1.set_ylabel("Mean best fitness", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(mttr, pen_m, "-s", color="#d62728", linewidth=2, label="Mean total delay penalty")
    ax2.fill_between(mttr, pen_m - pen_s, pen_m + pen_s, color="#d62728", alpha=0.12)
    ax2.set_ylabel("Mean total delay penalty (Penalty-1 + Penalty-2)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)

    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ticks = ax1.get_xticks()
    avail_ticks = DEFAULT_MTBF / (DEFAULT_MTBF + np.array(ticks))
    ax3.set_xticks(ticks)
    ax3.set_xticklabels([f"{a:.2f}" for a in avail_ticks])
    ax3.set_xlabel(f"System availability = MTBF / (MTBF + MTTR)  (MTBF={DEFAULT_MTBF:.0f})")

    ax1.set_title("Recovery time (MTTR) and resulting availability vs. quality of service\n"
                   "Ottawa case study, Q2GA, 5 seeds")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PNG, "qos_vs_mttr.png"), dpi=200)
    plt.close(fig)

    savemat(os.path.join(FIG_MAT, "qos_vs_mttr.mat"), {
        "mttr": mttr, "fitness_mean": fit_m, "fitness_std": fit_s,
        "penalty_mean": pen_m, "penalty_std": pen_s,
        "availability": DEFAULT_MTBF / (DEFAULT_MTBF + mttr),
    })


def fig_availability_surface():
    mttr_range = np.linspace(1, 30, 60)
    mtbf_range = np.linspace(20, 200, 60)
    MTTR, MTBF = np.meshgrid(mttr_range, mtbf_range)
    AVAIL = MTBF / (MTBF + MTTR)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cs = ax.contourf(MTTR, MTBF, AVAIL, levels=20, cmap="RdYlGn")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("System availability = MTBF / (MTBF + MTTR)")
    cl = ax.contour(MTTR, MTBF, AVAIL, levels=[0.7, 0.8, 0.9, 0.95, 0.98], colors="k", linewidths=0.8)
    ax.clabel(cl, inline=True, fontsize=8, fmt="%.2f")

    ax.scatter([DEFAULT_MTTR], [DEFAULT_MTBF], color="black", marker="x", s=120,
               linewidths=3, label=f"Ottawa case-study operating point\n(MTTR={DEFAULT_MTTR:.0f}, MTBF={DEFAULT_MTBF:.0f})")
    ax.legend(loc="lower right")
    ax.set_xlabel("MTTR (minutes)")
    ax.set_ylabel("MTBF (minutes)")
    ax.set_title("System availability as a function of MTTR and MTBF")
    _save(fig, "availability_surface")

    savemat(os.path.join(FIG_MAT, "availability_surface.mat"), {
        "mttr_range": mttr_range, "mtbf_range": mtbf_range, "availability": AVAIL,
        "default_mttr": DEFAULT_MTTR, "default_mtbf": DEFAULT_MTBF,
    })


def fig_cyber_risk_degradation_summary():
    g_gamma = _aggregate_sensitivity("sensitivity_gamma.csv")
    g_mttr = _aggregate_sensitivity("sensitivity_mttr.csv")

    base_g = g_gamma[g_gamma["value"] == 0.20].iloc[0]
    worst_g = g_gamma[g_gamma["value"] == 0.50].iloc[0]
    base_m = g_mttr[g_mttr["value"] == 10.0].iloc[0]
    worst_m = g_mttr[g_mttr["value"] == 30.0].iloc[0]

    def pct(base, worst):
        return 100.0 * (worst - base) / base if base != 0 else 0.0

    scenarios = ["Gamma: 0.20 -> 0.50 N\n(wider attack surface)",
                  "MTTR: 10 -> 30\n(slower recovery)"]
    fitness_pct = [pct(base_g["fitness_mean"], worst_g["fitness_mean"]),
                   pct(base_m["fitness_mean"], worst_m["fitness_mean"])]
    penalty_pct = [pct(base_g["penalty_mean"], worst_g["penalty_mean"]),
                   pct(base_m["penalty_mean"], worst_m["penalty_mean"])]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width / 2, fitness_pct, width, label="Fitness degradation (%)", color="#1f77b4")
    ax.bar(x + width / 2, penalty_pct, width, label="Total delay-penalty increase (%)", color="#d62728")
    ymin = min(0, min(fitness_pct + penalty_pct))
    ymax = max(fitness_pct + penalty_pct)
    span = ymax - ymin
    for i, (f, p) in enumerate(zip(fitness_pct, penalty_pct)):
        ax.text(i - width / 2, f + 0.05 * span * np.sign(f if f else 1), f"{f:.1f}%", ha="center", fontsize=10)
        ax.text(i + width / 2, p + 0.05 * span * np.sign(p if p else 1), f"{p:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(ymin - 0.15 * span, ymax + 0.2 * span)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Relative change vs. baseline (%)")
    ax.set_title("Quality-of-service degradation under worsening cyber-risk conditions\n"
                  "Ottawa case study, Q2GA, 5 seeds")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.legend()
    _save(fig, "cyber_risk_degradation_summary")

    savemat(os.path.join(FIG_MAT, "cyber_risk_degradation_summary.mat"), {
        "scenarios": np.array(scenarios, dtype=object),
        "fitness_pct": np.array(fitness_pct), "penalty_pct": np.array(penalty_pct),
    })


# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    inst = build_instance(seed=42)

    fig_instance_layout(inst)
    fig_patient_demographics(inst)
    fig_service_dropoff_times(inst)
    fig_time_window_thresholds(inst)
    fig_penalty_function(inst)
    fig_goal_function_weights(inst)
    fig_disruption_scenario(inst)
    fig_qos_vs_gamma()
    fig_qos_vs_mttr()
    fig_availability_surface()
    fig_cyber_risk_degradation_summary()

    print("Problem-level figures written to", FIG_PNG)
    print(".mat data written to", FIG_MAT)


if __name__ == "__main__":
    main()
