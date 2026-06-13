"""Shared helper functions that build pandas DataFrames for the report
tables (problem sizes, performance comparison, MCDM tuning, sensitivity /
cyber-risk analysis). Used by make_report.py and make_report2.py.
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

ALGOS = ["GA", "SA", "QGA", "Q2GA"]


def problem_size_table():
    from src.aarp_model import generate_instance
    rows = []
    for size, seed in [("small", 100), ("medium", 101), ("large", 102)]:
        inst = generate_instance(size, seed=seed)
        n_crit = int((inst.patient_type == 1).sum())
        n_mod = int((inst.patient_type == 2).sum())
        n_minor = int((inst.patient_type == 3).sum())
        rows.append({
            "Instance": size,
            "Patients (N)": inst.n_patients,
            "Type-1 (critical)": n_crit,
            "Type-2 (moderate)": n_mod,
            "Type-3 (minor)": n_minor,
            "Hospitals (H)": inst.n_hospitals,
            "Vehicles (M)": inst.n_vehicles,
            "Class A": int((inst.vehicle_class == 0).sum()),
            "Class B": int((inst.vehicle_class == 1).sum()),
            "Class C": int((inst.vehicle_class == 2).sum()),
            "Decision vars (2N)": inst.n_real_genes,
            "Qubits (QGA/Q2GA)": inst.n_qubits,
            "Gamma (uncertainty budget)": inst.gamma_budget,
            "MTTR": inst.mttr,
        })
    return pd.DataFrame(rows)


def performance_comparison_table(summ_df):
    rows = []
    for size in ["small", "medium", "large"]:
        sub = summ_df[summ_df["size"] == size]
        means = sub.groupby("algorithm")["best_fitness"].mean()
        ranks = means.rank(method="min")
        for algo in ALGOS:
            g = sub[sub["algorithm"] == algo]
            rows.append({
                "Instance": size,
                "Algorithm": algo,
                "Mean fitness": g["best_fitness"].mean(),
                "Std fitness": g["best_fitness"].std(),
                "Best (min)": g["best_fitness"].min(),
                "Worst (max)": g["best_fitness"].max(),
                "Mean runtime (s)": g["runtime_s"].mean(),
                "Mean unserved": g["n_unserved"].mean(),
                "Rank": int(ranks[algo]),
            })
    return pd.DataFrame(rows)


def overall_rank_table(summ_df):
    rows = []
    for algo in ALGOS:
        ranks = []
        for size in ["small", "medium", "large"]:
            sub = summ_df[summ_df["size"] == size]
            means = sub.groupby("algorithm")["best_fitness"].mean()
            ranks.append(means.rank(method="min")[algo])
        rows.append({"Algorithm": algo, "Small rank": ranks[0], "Medium rank": ranks[1],
                      "Large rank": ranks[2], "Mean rank": np.mean(ranks)})
    df = pd.DataFrame(rows).sort_values("Mean rank").reset_index(drop=True)
    return df


def tuning_table(algo):
    df = pd.read_csv(os.path.join(RESULTS, f"tuning_{algo}.csv"))
    return df


def sensitivity_table(csv_name, value_label):
    df = pd.read_csv(os.path.join(RESULTS, csv_name))
    g = df.groupby(["value", "algorithm"]).agg(
        mean_fitness=("best_fitness", "mean"),
        std_fitness=("best_fitness", "std"),
        mean_penalty1=("penalty1", "mean"),
        mean_unserved=("n_unserved", "mean"),
    ).reset_index()
    g = g.rename(columns={"value": value_label})
    return g


def cyber_risk_insight_table():
    """Compares the worst-case (highest gamma / mttr) setting against the
    baseline (gamma_frac=0.2 / mttr=8) for each algorithm, expressed as a
    percentage degradation -- a simple robustness/cyber-resilience score."""
    rows = []

    g = pd.read_csv(os.path.join(RESULTS, "sensitivity_gamma.csv"))
    base = g[g["value"] == 0.20].groupby("algorithm")["best_fitness"].mean()
    worst = g[g["value"] == 0.50].groupby("algorithm")["best_fitness"].mean()
    base_pen = g[g["value"] == 0.20].groupby("algorithm")["penalty1"].mean()
    worst_pen = g[g["value"] == 0.50].groupby("algorithm")["penalty1"].mean()
    base_uns = g[g["value"] == 0.20].groupby("algorithm")["n_unserved"].mean()
    worst_uns = g[g["value"] == 0.50].groupby("algorithm")["n_unserved"].mean()

    for algo in ALGOS:
        rows.append({
            "Algorithm": algo,
            "Risk factor": "Disruption budget Gamma: 0.20 -> 0.50 N",
            "Baseline fitness": base[algo],
            "Worst-case fitness": worst[algo],
            "Fitness degradation (%)": 100.0 * (worst[algo] - base[algo]) / base[algo],
            "Penalty-1 increase": worst_pen[algo] - base_pen[algo],
            "Extra unserved patients": worst_uns[algo] - base_uns[algo],
        })

    m = pd.read_csv(os.path.join(RESULTS, "sensitivity_mttr.csv"))
    base = m[m["value"] == 8.0].groupby("algorithm")["best_fitness"].mean()
    worst = m[m["value"] == 24.0].groupby("algorithm")["best_fitness"].mean()
    base_pen = m[m["value"] == 8.0].groupby("algorithm")["penalty1"].mean()
    worst_pen = m[m["value"] == 24.0].groupby("algorithm")["penalty1"].mean()
    base_uns = m[m["value"] == 8.0].groupby("algorithm")["n_unserved"].mean()
    worst_uns = m[m["value"] == 24.0].groupby("algorithm")["n_unserved"].mean()

    for algo in ALGOS:
        rows.append({
            "Algorithm": algo,
            "Risk factor": "MTTR: 8 -> 24 time units",
            "Baseline fitness": base[algo],
            "Worst-case fitness": worst[algo],
            "Fitness degradation (%)": 100.0 * (worst[algo] - base[algo]) / base[algo],
            "Penalty-1 increase": worst_pen[algo] - base_pen[algo],
            "Extra unserved patients": worst_uns[algo] - base_uns[algo],
        })

    return pd.DataFrame(rows)
