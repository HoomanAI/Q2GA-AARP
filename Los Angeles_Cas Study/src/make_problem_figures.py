"""Generate problem/instance characteristic figures (PNG, white background)
and companion .mat files for the LA wildfire case study.

Figures produced:
  - instance_layout_xy.png      : patient/hospital layout with Palisades Fire zone overlay
  - patient_demographics.png    : patient-type counts
  - service_dropoff_times.png   : service-time / drop-off-time histograms
  - time_window_thresholds.png  : a1/a2 time-window thresholds by type
  - penalty_function_shape.png  : penalty-vs-arrival-time curves (Eq. 1-2)
  - goal_function_weights.png   : w1/w2/w3 weighted-sum structure
  - disruption_scenario.png     : realised theta_i per patient, baseline vs wildfire
  - availability_by_zone.png    : MTTR / MTBF / availability, in-zone vs out-of-zone vs baseline
  - distance_impact.png         : road-distance change (wildfire - baseline) per patient, from nearest hospital
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import savemat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_PNG = os.path.join(ROOT, "figures", "png")
FIG_MAT = os.path.join(ROOT, "figures", "mat")
sys.path.insert(0, ROOT)

from src.la_instance import (build_instance, HOSPITALS, PATIENTS,  # noqa: E402
                              WILDFIRE_ZONE_PATIENT_IDX, N_HOSPITALS)

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
def fig_instance_layout(inst_base):
    fig, ax = plt.subplots(figsize=(8, 7))

    zone_mask = np.zeros(inst_base.n_patients, dtype=bool)
    zone_mask[WILDFIRE_ZONE_PATIENT_IDX] = True

    for t in (1, 2, 3):
        mask = (inst_base.patient_type == t) & ~zone_mask
        ax.scatter(inst_base.coords_patients[mask, 0], inst_base.coords_patients[mask, 1],
                   color=TYPE_COLORS[t], label=TYPE_LABELS[t], s=60, alpha=0.85,
                   edgecolor="k", linewidth=0.5)
    for t in (1, 2, 3):
        mask = (inst_base.patient_type == t) & zone_mask
        if mask.sum():
            ax.scatter(inst_base.coords_patients[mask, 0], inst_base.coords_patients[mask, 1],
                       color=TYPE_COLORS[t], s=110, alpha=0.95,
                       edgecolor="black", linewidth=1.6, marker="s",
                       label=(f"{TYPE_LABELS[t]} (in fire zone)" if t == 1 else None))

    ax.scatter(inst_base.coords_hospitals[:, 0], inst_base.coords_hospitals[:, 1],
               marker="*", s=400, color="gold", edgecolor="k", linewidth=1,
               label="Hospitals", zorder=5)
    for i, p in enumerate(PATIENTS):
        ax.annotate(str(i), (inst_base.coords_patients[i, 0], inst_base.coords_patients[i, 1]),
                    fontsize=7, ha="center", va="center")

    # Palisades Fire zone overlay (approximate burn-zone circle used for road
    # closures and degraded MTTR/MTBF)
    from src.la_instance import WILDFIRE_ZONE_CENTER, WILDFIRE_ZONE_RADIUS_KM, latlon_to_xy
    cx, cy = latlon_to_xy(*WILDFIRE_ZONE_CENTER)
    circle = plt.Circle((cx, cy), WILDFIRE_ZONE_RADIUS_KM, color="orangered",
                         fill=True, alpha=0.10, edgecolor="orangered", linewidth=2,
                         linestyle="--", label="Palisades Fire zone (approx.)")
    ax.add_patch(circle)

    ax.set_xlabel("x (km, east of reference point)")
    ax.set_ylabel("y (km, north of reference point)")
    ax.set_title("LA wildfire case-study instance layout\n"
                  f"(N={inst_base.n_patients} patients, H={inst_base.n_hospitals} hospitals, "
                  f"M={inst_base.n_vehicles} vehicles; squares = Palisades Fire zone)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    _save(fig, "instance_layout_xy")

    savemat(os.path.join(FIG_MAT, "instance_layout_xy.mat"), {
        "patient_x": inst_base.coords_patients[:, 0], "patient_y": inst_base.coords_patients[:, 1],
        "patient_type": inst_base.patient_type,
        "hospital_x": inst_base.coords_hospitals[:, 0], "hospital_y": inst_base.coords_hospitals[:, 1],
        "wildfire_zone_idx": np.array(WILDFIRE_ZONE_PATIENT_IDX),
        "wildfire_zone_center_xy": np.array([cx, cy]),
        "wildfire_zone_radius_km": WILDFIRE_ZONE_RADIUS_KM,
    })


# ---------------------------------------------------------------------------
def fig_patient_demographics(inst_base):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = [int((inst_base.patient_type == t).sum()) for t in (1, 2, 3)]
    ax.bar([TYPE_LABELS[t] for t in (1, 2, 3)], counts,
           color=[TYPE_COLORS[t] for t in (1, 2, 3)])
    for i, c in enumerate(counts):
        ax.text(i, c + 0.1, str(c), ha="center", fontsize=11)
    ax.set_ylabel("Number of patients")
    ax.set_title("Patient-type composition -- LA wildfire case study")
    _save(fig, "patient_demographics")
    savemat(os.path.join(FIG_MAT, "patient_demographics.mat"), {"counts": np.array(counts)})


# ---------------------------------------------------------------------------
def fig_service_dropoff_times(inst_base):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(inst_base.service_time, bins=10, color="#1f77b4", alpha=0.8)
    axes[0].set_xlabel("Service time $s_i$ (minutes)")
    axes[0].set_ylabel("Number of patients")
    axes[0].set_title("On-scene service-time distribution")

    dropoff = inst_base.dropoff_time[inst_base.dropoff_time > 0]
    axes[1].hist(dropoff, bins=10, color="#ff7f0e", alpha=0.8)
    axes[1].set_xlabel("Hospital drop-off time (minutes)")
    axes[1].set_ylabel("Number of patients")
    axes[1].set_title("Drop-off-time distribution (Type 1/2 only)")

    _save(fig, "service_dropoff_times")
    savemat(os.path.join(FIG_MAT, "service_dropoff_times.mat"), {
        "service_time": inst_base.service_time, "dropoff_time": dropoff,
    })


# ---------------------------------------------------------------------------
def fig_time_window_thresholds(inst_base):
    fig, ax = plt.subplots(figsize=(7, 5))
    a1_t1 = inst_base.a1[inst_base.patient_type == 1]
    a2_t1 = inst_base.a2[inst_base.patient_type == 1]
    a1_t2 = inst_base.a1[inst_base.patient_type == 2]
    a2_t2 = inst_base.a2[inst_base.patient_type == 2]

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
    ax.set_title("Semi-soft time-window thresholds -- LA wildfire case study")
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


def fig_penalty_function(inst_base):
    a1_t1 = float(inst_base.a1[inst_base.patient_type == 1].mean())
    a2_t1 = float(inst_base.a2[inst_base.patient_type == 1].mean())
    a1_t2 = float(inst_base.a1[inst_base.patient_type == 2].mean())
    a2_t2 = float(inst_base.a2[inst_base.patient_type == 2].mean())

    arrival = np.linspace(0, max(a2_t1, a2_t2) * 1.4, 400)
    pen1 = _penalty(arrival, a1_t1, a2_t1, inst_base.lambda1)
    pen2 = _penalty(arrival, a1_t2, a2_t2, inst_base.lambda2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(arrival, pen1, color=TYPE_COLORS[1], linewidth=2,
            label=f"Type 1 (critical), lambda1={inst_base.lambda1}, a1={a1_t1:.1f}, a2={a2_t1:.1f}")
    ax.plot(arrival, pen2, color=TYPE_COLORS[2], linewidth=2,
            label=f"Type 2 (moderate), lambda2={inst_base.lambda2}, a1={a1_t2:.1f}, a2={a2_t2:.1f}")
    for a1, a2, c in ((a1_t1, a2_t1, TYPE_COLORS[1]), (a1_t2, a2_t2, TYPE_COLORS[2])):
        ax.axvline(a1, color=c, linestyle="--", alpha=0.5)
        ax.axvline(a2, color=c, linestyle=":", alpha=0.5)
    ax.set_xlabel("Patient arrival time $a_i$ (minutes from dispatch)")
    ax.set_ylabel("Delay penalty")
    ax.set_title("Semi-soft time-window delay-penalty function -- LA wildfire case study\n"
                  "(dashed = a1 no-penalty threshold, dotted = a2 critical threshold)")
    ax.legend(fontsize=9)
    _save(fig, "penalty_function_shape")

    savemat(os.path.join(FIG_MAT, "penalty_function_shape.mat"), {
        "arrival": arrival, "penalty_type1": pen1, "penalty_type2": pen2,
        "a1_type1": a1_t1, "a2_type1": a2_t1, "a1_type2": a1_t2, "a2_type2": a2_t2,
    })


# ---------------------------------------------------------------------------
def fig_goal_function_weights(inst_base):
    weights = [inst_base.w1, inst_base.w2, inst_base.w3]
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
            {"w1": inst_base.w1, "w2": inst_base.w2, "w3": inst_base.w3})


# ---------------------------------------------------------------------------
def fig_disruption_scenario(inst_base, inst_wild):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, inst, label in zip(axes, (inst_base, inst_wild), ("Baseline", "Wildfire")):
        colors = np.where(inst.theta > 0, "#d62728", "#cccccc")
        zone_mask = np.zeros(inst.n_patients, dtype=bool)
        zone_mask[WILDFIRE_ZONE_PATIENT_IDX] = True
        edge = np.where(zone_mask, "black", "none")
        lw = np.where(zone_mask, 2.0, 0.0)
        ax.bar(np.arange(inst.n_patients), inst.theta, color=colors,
               edgecolor=edge.tolist(), linewidth=lw.tolist())
        ax.set_xticks(np.arange(inst.n_patients))
        ax.set_xlabel("Patient index (black outline = Palisades Fire zone)")
        ax.set_ylabel("Realised disruption buffer $\\theta_i$ (minutes)")
        ax.set_title(f"{label}: Gamma={inst.gamma_budget}/{inst.n_patients}, "
                      f"MTTR={inst.mttr:.0f}, MTBF={inst.mtbf:.0f}, "
                      f"avail={inst.mtbf/(inst.mtbf+inst.mttr):.2f}")

    fig.suptitle("Realised disruption scenario, baseline vs wildfire -- LA case study")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_PNG, "disruption_scenario.png"), dpi=200)
    plt.close(fig)

    savemat(os.path.join(FIG_MAT, "disruption_scenario.mat"), {
        "theta_baseline": inst_base.theta, "theta_wildfire": inst_wild.theta,
        "gamma_baseline": inst_base.gamma_budget, "gamma_wildfire": inst_wild.gamma_budget,
        "mttr_baseline": inst_base.mttr, "mttr_wildfire": inst_wild.mttr,
        "mtbf_baseline": inst_base.mtbf, "mtbf_wildfire": inst_wild.mtbf,
        "wildfire_zone_idx": np.array(WILDFIRE_ZONE_PATIENT_IDX),
    })


# ---------------------------------------------------------------------------
def fig_availability_by_zone():
    MTTR_ZONE, MTBF_ZONE = 60.0, 30.0
    MTTR_OUT, MTBF_OUT = 12.0, 100.0
    MTTR_BASE, MTBF_BASE = 10.0, 120.0

    rows = [
        ("Baseline\n(citywide)", MTTR_BASE, MTBF_BASE),
        ("Wildfire:\noutside fire zone", MTTR_OUT, MTBF_OUT),
        ("Wildfire:\ninside fire zone", MTTR_ZONE, MTBF_ZONE),
    ]
    labels = [r[0] for r in rows]
    mttrs = [r[1] for r in rows]
    mtbfs = [r[2] for r in rows]
    avails = [mtbf / (mtbf + mttr) for mttr, mtbf in zip(mttrs, mtbfs)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, mttrs, width, label="MTTR (min)", color="#d62728")
    axes[0].bar(x + width / 2, mtbfs, width, label="MTBF (min)", color="#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Minutes")
    axes[0].set_title("MTTR / MTBF by zone")
    axes[0].legend()
    for i, (mr, mb) in enumerate(zip(mttrs, mtbfs)):
        axes[0].text(i - width / 2, mr, f"{mr:.0f}", ha="center", va="bottom", fontsize=9)
        axes[0].text(i + width / 2, mb, f"{mb:.0f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, avails, color=["#2ca02c", "#ff7f0e", "#d62728"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("System availability = MTBF / (MTBF + MTTR)")
    axes[1].set_title("Resulting availability by zone")
    for i, a in enumerate(avails):
        axes[1].text(i, a, f"{a:.2f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Wildfire-induced network-outage impact on MTTR / MTBF / availability\n"
                  "LA wildfire case study (Palisades Fire, Jan 2025)")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(FIG_PNG, "availability_by_zone.png"), dpi=200)
    plt.close(fig)

    savemat(os.path.join(FIG_MAT, "availability_by_zone.mat"), {
        "labels": np.array(labels, dtype=object),
        "mttr": np.array(mttrs), "mtbf": np.array(mtbfs), "availability": np.array(avails),
    })


# ---------------------------------------------------------------------------
def fig_distance_impact(inst_base, inst_wild):
    """Change in shortest road distance from each patient to its nearest
    hospital, baseline vs wildfire (illustrates the effect of burn-zone road
    closures)."""
    D_base = inst_base.dist_matrix
    D_wild = inst_wild.dist_matrix
    n_h = N_HOSPITALS

    base_min = np.array([D_base[n_h + p, :n_h].min() for p in range(inst_base.n_patients)])
    wild_min = np.array([D_wild[n_h + p, :n_h].min() for p in range(inst_wild.n_patients)])
    delta = wild_min - base_min

    zone_mask = np.zeros(inst_base.n_patients, dtype=bool)
    zone_mask[WILDFIRE_ZONE_PATIENT_IDX] = True
    colors = np.where(zone_mask, "#d62728", "#1f77b4")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(np.arange(inst_base.n_patients), delta, color=colors)
    ax.set_xticks(np.arange(inst_base.n_patients))
    ax.set_xlabel("Patient index (red = inside Palisades Fire zone)")
    ax.set_ylabel("Change in shortest distance to nearest hospital (km)\nwildfire - baseline")
    ax.set_title("Road-network impact of burn-zone closures on ambulance access\n"
                  "LA wildfire case study")
    ax.axhline(0, color="k", linewidth=0.8)
    _save(fig, "distance_impact")

    savemat(os.path.join(FIG_MAT, "distance_impact.mat"), {
        "nearest_hosp_dist_baseline": base_min,
        "nearest_hosp_dist_wildfire": wild_min,
        "delta": delta,
        "wildfire_zone_idx": np.array(WILDFIRE_ZONE_PATIENT_IDX),
    })


# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    inst_base = build_instance(scenario="baseline", seed=42)
    inst_wild = build_instance(scenario="wildfire", seed=42)

    fig_instance_layout(inst_base)
    fig_patient_demographics(inst_base)
    fig_service_dropoff_times(inst_base)
    fig_time_window_thresholds(inst_base)
    fig_penalty_function(inst_base)
    fig_goal_function_weights(inst_base)
    fig_disruption_scenario(inst_base, inst_wild)
    fig_availability_by_zone()
    fig_distance_impact(inst_base, inst_wild)

    print("Problem-level figures written to", FIG_PNG)
    print(".mat data written to", FIG_MAT)


if __name__ == "__main__":
    main()
