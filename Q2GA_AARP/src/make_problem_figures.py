"""Generate "problem-level" figures (PNG, white background) and companion
.mat files for regenerating MATLAB .fig figures via
problem_figures/matlab/make_all_problem_figs.m.

Unlike make_figures.py / make_figures2.py, these figures characterise the
AARP *problem instances and mathematical model itself* -- not the
performance of any optimisation algorithm. Figures produced:

  - instance_layout_<size>.png      : spatial layout of patients/hospitals
  - patient_demographics.png        : patient-type counts by instance size
  - vehicle_fleet_composition.png   : vehicle-class counts by instance size
  - service_dropoff_times.png       : service-time / drop-off-time distributions
  - time_window_thresholds.png      : a1/a2 semi-soft time-window thresholds
  - penalty_function_shape.png      : penalty-vs-arrival-time curves (Eq. 1-2)
  - mttr_mtbf_availability.png      : system availability vs MTTR/MTBF
  - mttr_theta_sensitivity.png      : disruption buffer theta vs MTTR
  - gamma_budget_sensitivity.png    : disruption coverage/severity vs Gamma
  - disruption_scenario_<size>.png  : realised theta_i per patient
  - goal_function_weights.png       : w1/w2/w3 weighted-sum structure
  - objective_components_<size>.png : C1/C2/C3/penalty/infeasibility breakdown

  - qos_vs_availability.png   : QoS metrics vs system availability (MTTR sweep)
  - qos_vs_gamma.png          : QoS metrics vs disruption budget Gamma
  - qos_vs_mttr.png           : QoS metrics vs MTTR (with availability twin axis)
  - availability_surface.png  : availability(MTTR, MTBF) contour map
  - cyber_risk_degradation_summary.png : QoS degradation, baseline vs worst-case

The QoS figures aggregate the existing per-algorithm sensitivity results
(results/sensitivity_gamma.csv, results/sensitivity_mttr.csv) across all four
algorithms and seeds, to characterise how the *problem's* quality-of-service
outcomes (fitness, delay penalties, unserved patients) respond to the
cyber-risk/availability parameters Gamma and MTTR -- independent of any one
algorithm's performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import savemat

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aarp_model import generate_instance, decode, simulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "problem_figures")
FIG_PNG = os.path.join(OUT, "png")
FIG_MAT = os.path.join(OUT, "mat")

DEFAULT_MTBF = 60.0
DEFAULT_MTTR = 8.0
DEFAULT_GAMMA_FRAC = 0.20

SIZES = ["small", "medium", "large"]
SEEDS = {"small": 100, "medium": 101, "large": 102}
SIZE_COLORS = {"small": "#1f77b4", "medium": "#ff7f0e", "large": "#2ca02c"}
TYPE_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c"}
TYPE_LABELS = {1: "Critical (Type 1)", 2: "Moderate (Type 2)", 3: "Minor (Type 3)"}

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
# 1. Instance layout (patients / hospitals)
# ---------------------------------------------------------------------------

def fig_instance_layout():
    for size in SIZES:
        inst = generate_instance(size, seed=SEEDS[size])
        fig, ax = plt.subplots(figsize=(7, 7))
        for t in (1, 2, 3):
            mask = inst.patient_type == t
            ax.scatter(inst.coords_patients[mask, 0], inst.coords_patients[mask, 1],
                       color=TYPE_COLORS[t], label=TYPE_LABELS[t], s=50, alpha=0.85,
                       edgecolor="k", linewidth=0.5)
        ax.scatter(inst.coords_hospitals[:, 0], inst.coords_hospitals[:, 1],
                   marker="*", s=350, color="gold", edgecolor="k", linewidth=1,
                   label="Hospitals", zorder=5)
        ax.set_xlabel("x coordinate")
        ax.set_ylabel("y coordinate")
        ax.set_title(f"Problem instance layout -- {size}\n"
                      f"(N={inst.n_patients} patients, H={inst.n_hospitals} hospitals, "
                      f"M={inst.n_vehicles} vehicles)")
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=9)
        _save(fig, f"instance_layout_{size}")

        savemat(os.path.join(FIG_MAT, f"instance_layout_{size}.mat"), {
            "patient_x": inst.coords_patients[:, 0],
            "patient_y": inst.coords_patients[:, 1],
            "patient_type": inst.patient_type,
            "hospital_x": inst.coords_hospitals[:, 0],
            "hospital_y": inst.coords_hospitals[:, 1],
        })


# ---------------------------------------------------------------------------
# 2. Patient demographics (type counts per instance size)
# ---------------------------------------------------------------------------

def fig_patient_demographics():
    rows = []
    for size in SIZES:
        inst = generate_instance(size, seed=SEEDS[size])
        for t in (1, 2, 3):
            rows.append({"size": size, "type": TYPE_LABELS[t],
                          "count": int((inst.patient_type == t).sum())})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SIZES))
    width = 0.25
    mat_data = {}
    for j, t in enumerate((1, 2, 3)):
        vals = [df[(df["size"] == s) & (df["type"] == TYPE_LABELS[t])]["count"].iloc[0]
                for s in SIZES]
        ax.bar(x + (j - 1) * width, vals, width, label=TYPE_LABELS[t], color=TYPE_COLORS[t])
        mat_data[f"type{t}_counts"] = np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(SIZES)
    ax.set_xlabel("Instance size")
    ax.set_ylabel("Number of patients")
    ax.set_title("Patient-type composition by instance size")
    ax.legend()
    _save(fig, "patient_demographics")
    savemat(os.path.join(FIG_MAT, "patient_demographics.mat"), mat_data)


# ---------------------------------------------------------------------------
# 3. Vehicle fleet composition
# ---------------------------------------------------------------------------

def fig_vehicle_fleet():
    rows = []
    for size in SIZES:
        inst = generate_instance(size, seed=SEEDS[size])
        for c, label in zip((0, 1, 2), ("Class A (critical/moderate)",
                                          "Class B (critical/moderate)",
                                          "Class C (minor)")):
            rows.append({"size": size, "class": label,
                          "count": int((inst.vehicle_class == c).sum())})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SIZES))
    width = 0.25
    mat_data = {}
    classes = ["Class A (critical/moderate)", "Class B (critical/moderate)", "Class C (minor)"]
    colors = ["#1f77b4", "#9467bd", "#2ca02c"]
    for j, cls in enumerate(classes):
        vals = [df[(df["size"] == s) & (df["class"] == cls)]["count"].iloc[0] for s in SIZES]
        ax.bar(x + (j - 1) * width, vals, width, label=cls, color=colors[j])
        mat_data[f"class_{j}_counts"] = np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(SIZES)
    ax.set_xlabel("Instance size")
    ax.set_ylabel("Number of vehicles")
    ax.set_title("Autonomous-ambulance fleet composition by instance size")
    ax.legend(fontsize=9)
    _save(fig, "vehicle_fleet_composition")
    savemat(os.path.join(FIG_MAT, "vehicle_fleet_composition.mat"), mat_data)


# ---------------------------------------------------------------------------
# 4. Service / drop-off time distributions
# ---------------------------------------------------------------------------

def fig_service_dropoff_times():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    mat_data = {}
    for size in SIZES:
        inst = generate_instance(size, seed=SEEDS[size])
        axes[0].hist(inst.service_time, bins=10, alpha=0.5, label=size, color=SIZE_COLORS[size])
        mat_data[f"service_time_{size}"] = inst.service_time

        dropoff = inst.dropoff_time[inst.dropoff_time > 0]
        axes[1].hist(dropoff, bins=10, alpha=0.5, label=size, color=SIZE_COLORS[size])
        mat_data[f"dropoff_time_{size}"] = dropoff

    axes[0].set_xlabel("Service time $s_i$ (time units)")
    axes[0].set_ylabel("Number of patients")
    axes[0].set_title("On-scene service-time distribution")
    axes[0].legend()

    axes[1].set_xlabel("Hospital drop-off time (time units)")
    axes[1].set_ylabel("Number of patients")
    axes[1].set_title("Drop-off-time distribution (Type 1/2 only)")
    axes[1].legend()

    _save(fig, "service_dropoff_times")
    savemat(os.path.join(FIG_MAT, "service_dropoff_times.mat"), mat_data)


# ---------------------------------------------------------------------------
# 5. Semi-soft time-window thresholds (a1, a2)
# ---------------------------------------------------------------------------

def fig_time_window_thresholds():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    mat_data = {}
    for ax, ptype, label in zip(axes, (1, 2), ("Type 1 (critical)", "Type 2 (moderate)")):
        a1_data, a2_data, labels_x = [], [], []
        for size in SIZES:
            inst = generate_instance(size, seed=SEEDS[size])
            mask = inst.patient_type == ptype
            a1_data.append(inst.a1[mask])
            a2_data.append(inst.a2[mask])
            labels_x.append(size)
            mat_data[f"a1_type{ptype}_{size}"] = inst.a1[mask]
            mat_data[f"a2_type{ptype}_{size}"] = inst.a2[mask]

        positions1 = np.arange(len(SIZES)) * 2 - 0.35
        positions2 = np.arange(len(SIZES)) * 2 + 0.35
        bp1 = ax.boxplot(a1_data, positions=positions1, widths=0.6, patch_artist=True,
                          labels=[""] * len(SIZES))
        bp2 = ax.boxplot(a2_data, positions=positions2, widths=0.6, patch_artist=True,
                          labels=[""] * len(SIZES))
        for box in bp1["boxes"]:
            box.set_facecolor("#1f77b4")
            box.set_alpha(0.6)
        for box in bp2["boxes"]:
            box.set_facecolor("#d62728")
            box.set_alpha(0.6)
        ax.set_xticks(np.arange(len(SIZES)) * 2)
        ax.set_xticklabels(labels_x)
        ax.set_ylabel("Threshold value (time units)")
        ax.set_title(f"Time-window thresholds -- {label}")
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["a1 (no-penalty threshold)",
                                                          "a2 (critical threshold)"],
                  loc="best", fontsize=9)

    _save(fig, "time_window_thresholds")
    savemat(os.path.join(FIG_MAT, "time_window_thresholds.mat"), mat_data)


# ---------------------------------------------------------------------------
# 6. Penalty function shape (Eq. 1-2)
# ---------------------------------------------------------------------------

def _penalty(arrival, a1, a2, lam):
    pen = np.zeros_like(arrival)
    mid = (arrival > a1) & (arrival <= a2)
    high = arrival > a2
    pen[mid] = (arrival[mid] - a1)
    pen[high] = (a2 - a1) + 2.0 * (arrival[high] - a2)
    return lam * pen


def fig_penalty_function():
    inst = generate_instance("medium", seed=SEEDS["medium"])
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
    ax.set_xlabel("Patient arrival time $a_i$ (time units)")
    ax.set_ylabel("Delay penalty")
    ax.set_title("Semi-soft time-window delay-penalty function (Eq. 1-2)\n"
                  "(dashed = a1 no-penalty threshold, dotted = a2 critical threshold)")
    ax.legend(fontsize=9)
    _save(fig, "penalty_function_shape")

    savemat(os.path.join(FIG_MAT, "penalty_function_shape.mat"), {
        "arrival": arrival, "penalty_type1": pen1, "penalty_type2": pen2,
        "a1_type1": a1_t1, "a2_type1": a2_t1, "a1_type2": a1_t2, "a2_type2": a2_t2,
    })


# ---------------------------------------------------------------------------
# 7. MTTR / MTBF availability
# ---------------------------------------------------------------------------

def fig_mttr_mtbf_availability():
    mttr_range = np.linspace(1, 30, 60)
    mtbf_values = [30, 60, 90, 120]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    mat_data = {}
    for mtbf in mtbf_values:
        avail = mtbf / (mtbf + mttr_range)
        axes[0].plot(mttr_range, avail, linewidth=2, label=f"MTBF={mtbf}")
        mat_data[f"avail_vs_mttr_mtbf{mtbf}"] = avail
    mat_data["mttr_range"] = mttr_range
    axes[0].set_xlabel("MTTR (time units)")
    axes[0].set_ylabel("Availability = MTBF / (MTBF + MTTR)")
    axes[0].set_title("System availability vs. MTTR")
    axes[0].legend()

    mtbf_range = np.linspace(10, 150, 60)
    mttr_values = [4, 8, 16, 24]
    for mttr in mttr_values:
        avail = mtbf_range / (mtbf_range + mttr)
        axes[1].plot(mtbf_range, avail, linewidth=2, label=f"MTTR={mttr}")
        mat_data[f"avail_vs_mtbf_mttr{mttr}"] = avail
    mat_data["mtbf_range"] = mtbf_range
    axes[1].set_xlabel("MTBF (time units)")
    axes[1].set_ylabel("Availability = MTBF / (MTBF + MTTR)")
    axes[1].set_title("System availability vs. MTBF")
    axes[1].legend()

    _save(fig, "mttr_mtbf_availability")
    savemat(os.path.join(FIG_MAT, "mttr_mtbf_availability.mat"), mat_data)


# ---------------------------------------------------------------------------
# 8. Disruption buffer theta vs MTTR
# ---------------------------------------------------------------------------

def fig_mttr_theta_sensitivity():
    mttr_values = [2, 4, 8, 12, 16, 20, 24]
    means, stds = [], []
    for mttr in mttr_values:
        inst = generate_instance("medium", seed=SEEDS["medium"], mttr=float(mttr))
        affected = inst.theta[inst.theta > 0]
        means.append(affected.mean() if len(affected) else 0.0)
        stds.append(affected.std() if len(affected) else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    means, stds = np.array(means), np.array(stds)
    ax.plot(mttr_values, means, "-o", color="#d62728", linewidth=2, label="Mean theta (affected patients)")
    ax.fill_between(mttr_values, means - stds, means + stds, color="#d62728", alpha=0.15)
    ax.set_xlabel("MTTR (time units)")
    ax.set_ylabel("Realised disruption buffer $\\theta_i$ (time units)")
    ax.set_title("Disruption-buffer magnitude vs. MTTR (medium instance, Gamma fixed at default)")
    ax.legend()
    _save(fig, "mttr_theta_sensitivity")

    savemat(os.path.join(FIG_MAT, "mttr_theta_sensitivity.mat"), {
        "mttr_values": np.array(mttr_values), "theta_mean": means, "theta_std": stds,
    })


# ---------------------------------------------------------------------------
# 9. Gamma (disruption budget) sensitivity
# ---------------------------------------------------------------------------

def fig_gamma_budget_sensitivity():
    inst0 = generate_instance("medium", seed=SEEDS["medium"])
    n = inst0.n_patients
    fracs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    n_affected, total_theta, mean_theta = [], [], []
    for frac in fracs:
        budget = max(0, round(frac * n))
        inst = generate_instance("medium", seed=SEEDS["medium"], gamma_budget=budget)
        affected = inst.theta[inst.theta > 0]
        n_affected.append(int((inst.theta > 0).sum()))
        total_theta.append(float(inst.theta.sum()))
        mean_theta.append(float(affected.mean()) if len(affected) else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(fracs, n_affected, "-o", color="#1f77b4", linewidth=2)
    axes[0].set_xlabel("Disruption budget Gamma (fraction of patients)")
    axes[0].set_ylabel("Number of affected patients")
    axes[0].set_title("Disruption coverage vs. Gamma")

    axes[1].plot(fracs, total_theta, "-o", color="#d62728", linewidth=2, label="Total theta")
    axes[1].plot(fracs, mean_theta, "-s", color="#ff7f0e", linewidth=2, label="Mean theta (affected)")
    axes[1].set_xlabel("Disruption budget Gamma (fraction of patients)")
    axes[1].set_ylabel("Disruption buffer theta (time units)")
    axes[1].set_title("Disruption severity vs. Gamma")
    axes[1].legend()

    _save(fig, "gamma_budget_sensitivity")
    savemat(os.path.join(FIG_MAT, "gamma_budget_sensitivity.mat"), {
        "gamma_frac": fracs, "n_affected": np.array(n_affected),
        "total_theta": np.array(total_theta), "mean_theta": np.array(mean_theta),
    })


# ---------------------------------------------------------------------------
# 10. Realised disruption scenario per instance (theta_i)
# ---------------------------------------------------------------------------

def fig_disruption_scenario():
    for size in SIZES:
        inst = generate_instance(size, seed=SEEDS[size])
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = np.where(inst.theta > 0, "#d62728", "#cccccc")
        ax.bar(np.arange(inst.n_patients), inst.theta, color=colors)
        ax.set_xlabel("Patient index")
        ax.set_ylabel("Realised disruption buffer $\\theta_i$ (time units)")
        ax.set_title(f"Realised disruption scenario -- {size} "
                      f"(Gamma={inst.gamma_budget} of {inst.n_patients} patients affected, "
                      f"MTTR={inst.mttr})")
        _save(fig, f"disruption_scenario_{size}")

        savemat(os.path.join(FIG_MAT, f"disruption_scenario_{size}.mat"), {
            "theta": inst.theta, "gamma_budget": inst.gamma_budget, "mttr": inst.mttr,
        })


# ---------------------------------------------------------------------------
# 11. Goal-function (weighted-sum) structure
# ---------------------------------------------------------------------------

def fig_goal_function_weights():
    inst = generate_instance("medium", seed=SEEDS["medium"])
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

    savemat(os.path.join(FIG_MAT, "goal_function_weights.mat"), {
        "w1": inst.w1, "w2": inst.w2, "w3": inst.w3,
    })


# ---------------------------------------------------------------------------
# 12. Objective-component breakdown for a representative solution
# ---------------------------------------------------------------------------

def fig_objective_components():
    for size in SIZES:
        inst = generate_instance(size, seed=SEEDS[size])
        rng = np.random.default_rng(0)
        # Search a few random chromosomes for a feasible (zero-infeasibility) solution
        best = None
        for trial in range(50):
            chrom = rng.random(inst.n_real_genes)
            routes = decode(chrom, inst)
            res = simulate(routes, inst)
            if res["infeasibility"] == 0:
                best = res
                break
            if best is None or res["infeasibility"] < best["infeasibility"]:
                best = res

        components = {
            "w1*C1": inst.w1 * best["C1"],
            "w2*C2": inst.w2 * best["C2"],
            "w3*C3": inst.w3 * best["C3"],
            "penalty1": best["penalty1"],
            "penalty2": best["penalty2"],
            "infeasibility": best["infeasibility"],
        }
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(components.keys())
        vals = list(components.values())
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
        ax.bar(names, vals, color=colors)
        ax.set_ylabel("Contribution to fitness")
        ax.set_title(f"Goal-function component breakdown -- {size}\n"
                      f"(random feasible-search solution; total fitness = {best['fitness']:.2f}, "
                      f"unserved = {best['n_unserved']})")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        _save(fig, f"objective_components_{size}")

        savemat(os.path.join(FIG_MAT, f"objective_components_{size}.mat"), {
            "names": np.array(names, dtype=object),
            "values": np.array(vals),
            "fitness": best["fitness"],
            "n_unserved": best["n_unserved"],
        })


# ---------------------------------------------------------------------------
# 13. QoS vs. system availability (derived from MTTR sweep, MTBF fixed)
# ---------------------------------------------------------------------------

def _aggregate_sensitivity(csv_name):
    """Aggregate a sensitivity CSV across all algorithms/seeds, per `value`."""
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


def fig_qos_vs_availability():
    g = _aggregate_sensitivity("sensitivity_mttr.csv")
    mttr = g["value"].values
    availability = DEFAULT_MTBF / (DEFAULT_MTBF + mttr)

    order = np.argsort(availability)
    avail_s = availability[order]
    fit_m, fit_s = g["fitness_mean"].values[order], g["fitness_std"].values[order]
    pen_m, pen_s = g["penalty_mean"].values[order], g["penalty_std"].values[order]
    uns_m = g["unserved_mean"].values[order]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(avail_s, fit_m, "-o", color="#1f77b4", linewidth=2)
    axes[0].fill_between(avail_s, fit_m - fit_s, fit_m + fit_s, color="#1f77b4", alpha=0.15)
    axes[0].set_ylabel("Mean best fitness")
    axes[0].set_title("Overall fitness vs. availability")

    axes[1].plot(avail_s, pen_m, "-o", color="#d62728", linewidth=2)
    axes[1].fill_between(avail_s, pen_m - pen_s, pen_m + pen_s, color="#d62728", alpha=0.15)
    axes[1].set_ylabel("Mean total delay penalty (Penalty-1 + Penalty-2)")
    axes[1].set_title("Service-quality penalty vs. availability")

    axes[2].plot(avail_s, uns_m, "-o", color="#9467bd", linewidth=2)
    axes[2].set_ylabel("Mean number of unserved patients")
    axes[2].set_title("Coverage failure vs. availability")

    for ax in axes:
        ax.set_xlabel("System availability = MTBF / (MTBF + MTTR)")
        ax.invert_xaxis()  # decreasing availability -> right side (worse)

    fig.suptitle("Impact of system availability (MTTR sweep, MTBF=60 fixed) on quality of service\n"
                  "(aggregated across GA/SA/QGA/Q2GA, medium instance, 5 seeds)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG_PNG, "qos_vs_availability.png"), dpi=200)
    plt.close(fig)

    savemat(os.path.join(FIG_MAT, "qos_vs_availability.mat"), {
        "mttr": mttr[order], "availability": avail_s,
        "fitness_mean": fit_m, "fitness_std": fit_s,
        "penalty_mean": pen_m, "penalty_std": pen_s,
        "unserved_mean": uns_m,
    })


# ---------------------------------------------------------------------------
# 14. QoS vs. cyber-disruption budget Gamma
# ---------------------------------------------------------------------------

def fig_qos_vs_gamma():
    g = _aggregate_sensitivity("sensitivity_gamma.csv")
    gamma = g["value"].values
    fit_m, fit_s = g["fitness_mean"].values, g["fitness_std"].values
    pen_m, pen_s = g["penalty_mean"].values, g["penalty_std"].values
    uns_m = g["unserved_mean"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(gamma, fit_m, "-o", color="#1f77b4", linewidth=2)
    axes[0].fill_between(gamma, fit_m - fit_s, fit_m + fit_s, color="#1f77b4", alpha=0.15)
    axes[0].set_ylabel("Mean best fitness")
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
                  "(aggregated across GA/SA/QGA/Q2GA, medium instance, 5 seeds; "
                  "dashed line = default Gamma=0.20)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG_PNG, "qos_vs_gamma.png"), dpi=200)
    plt.close(fig)

    savemat(os.path.join(FIG_MAT, "qos_vs_gamma.mat"), {
        "gamma_frac": gamma, "fitness_mean": fit_m, "fitness_std": fit_s,
        "penalty_mean": pen_m, "penalty_std": pen_s, "unserved_mean": uns_m,
    })


# ---------------------------------------------------------------------------
# 15. QoS vs. MTTR with availability twin axis
# ---------------------------------------------------------------------------

def fig_qos_vs_mttr():
    g = _aggregate_sensitivity("sensitivity_mttr.csv")
    mttr = g["value"].values
    fit_m, fit_s = g["fitness_mean"].values, g["fitness_std"].values
    pen_m, pen_s = g["penalty_mean"].values, g["penalty_std"].values

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(mttr, fit_m, "-o", color="#1f77b4", linewidth=2, label="Mean best fitness")
    ax1.fill_between(mttr, fit_m - fit_s, fit_m + fit_s, color="#1f77b4", alpha=0.15)
    ax1.set_xlabel("MTTR (mean time to repair, time units)")
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
    avail_ticks = DEFAULT_MTBF / (DEFAULT_MTBF + np.array(ax1.get_xticks()))
    ax3.set_xticks(ax1.get_xticks())
    ax3.set_xticklabels([f"{a:.2f}" for a in avail_ticks])
    ax3.set_xlabel("System availability = MTBF / (MTBF + MTTR)  (MTBF=60)")

    ax1.set_title("Recovery time (MTTR) and resulting availability vs. quality of service\n"
                   "(aggregated across GA/SA/QGA/Q2GA, medium instance, 5 seeds)")
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


# ---------------------------------------------------------------------------
# 16. Availability(MTTR, MTBF) contour surface
# ---------------------------------------------------------------------------

def fig_availability_surface():
    mttr_range = np.linspace(1, 30, 60)
    mtbf_range = np.linspace(10, 150, 60)
    MTTR, MTBF = np.meshgrid(mttr_range, mtbf_range)
    AVAIL = MTBF / (MTBF + MTTR)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cs = ax.contourf(MTTR, MTBF, AVAIL, levels=20, cmap="RdYlGn")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("System availability = MTBF / (MTBF + MTTR)")
    cl = ax.contour(MTTR, MTBF, AVAIL, levels=[0.7, 0.8, 0.9, 0.95, 0.98], colors="k", linewidths=0.8)
    ax.clabel(cl, inline=True, fontsize=8, fmt="%.2f")

    ax.scatter([DEFAULT_MTTR], [DEFAULT_MTBF], color="black", marker="x", s=120,
               linewidths=3, label=f"Default operating point\n(MTTR={DEFAULT_MTTR:.0f}, MTBF={DEFAULT_MTBF:.0f})")
    ax.legend(loc="upper right")
    ax.set_xlabel("MTTR (time units)")
    ax.set_ylabel("MTBF (time units)")
    ax.set_title("System availability as a function of MTTR and MTBF")
    _save(fig, "availability_surface")

    savemat(os.path.join(FIG_MAT, "availability_surface.mat"), {
        "mttr_range": mttr_range, "mtbf_range": mtbf_range, "availability": AVAIL,
        "default_mttr": DEFAULT_MTTR, "default_mtbf": DEFAULT_MTBF,
    })


# ---------------------------------------------------------------------------
# 17. Cyber-risk QoS degradation summary (baseline vs. worst case)
# ---------------------------------------------------------------------------

def fig_cyber_risk_degradation_summary():
    g_gamma = _aggregate_sensitivity("sensitivity_gamma.csv")
    g_mttr = _aggregate_sensitivity("sensitivity_mttr.csv")

    base_g = g_gamma[g_gamma["value"] == 0.20].iloc[0]
    worst_g = g_gamma[g_gamma["value"] == 0.50].iloc[0]
    base_m = g_mttr[g_mttr["value"] == 8.0].iloc[0]
    worst_m = g_mttr[g_mttr["value"] == 24.0].iloc[0]

    def pct(base, worst):
        return 100.0 * (worst - base) / base if base != 0 else 0.0

    scenarios = ["Gamma: 0.20 -> 0.50 N\n(wider attack surface)",
                  "MTTR: 8 -> 24\n(slower recovery)"]
    fitness_pct = [pct(base_g["fitness_mean"], worst_g["fitness_mean"]),
                   pct(base_m["fitness_mean"], worst_m["fitness_mean"])]
    penalty_pct = [pct(base_g["penalty_mean"], worst_g["penalty_mean"]),
                   pct(base_m["penalty_mean"], worst_m["penalty_mean"])]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width / 2, fitness_pct, width, label="Fitness degradation (%)", color="#1f77b4")
    ax.bar(x + width / 2, penalty_pct, width, label="Total delay-penalty increase (%)", color="#d62728")
    ymax = max(fitness_pct + penalty_pct)
    for i, (f, p) in enumerate(zip(fitness_pct, penalty_pct)):
        ax.text(i - width / 2, f + 0.05 * ymax, f"{f:.1f}%", ha="center", fontsize=10)
        ax.text(i + width / 2, p + 0.05 * ymax, f"{p:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, ymax * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Relative change vs. baseline (%)")
    ax.set_title("Quality-of-service degradation under worsening cyber-risk conditions\n"
                  "(aggregated across GA/SA/QGA/Q2GA, medium instance, 5 seeds)")
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
    fig_instance_layout()
    fig_patient_demographics()
    fig_vehicle_fleet()
    fig_service_dropoff_times()
    fig_time_window_thresholds()
    fig_penalty_function()
    fig_mttr_mtbf_availability()
    fig_mttr_theta_sensitivity()
    fig_gamma_budget_sensitivity()
    fig_disruption_scenario()
    fig_goal_function_weights()
    fig_objective_components()
    fig_qos_vs_availability()
    fig_qos_vs_gamma()
    fig_qos_vs_mttr()
    fig_availability_surface()
    fig_cyber_risk_degradation_summary()
    print("Problem-level figures written to", FIG_PNG)
    print(".mat data written to", FIG_MAT)


if __name__ == "__main__":
    main()
