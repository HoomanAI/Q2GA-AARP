"""
Mathematical model for the Autonomous Ambulance Routing Problem (AARP).

This module implements:
  - Instance generation (small / medium / large) with patients, hospitals,
    autonomous-ambulance (AV) fleet, battery parameters, semi-soft time
    windows based on a survival function, and MTTR/MTBF-based service
    interruption (uncertainty buffer) scenarios.
  - A real-valued ([0,1]^D) chromosome decoder that maps a chromosome to
    vehicle routes (assignment + sequencing random keys).
  - A route simulator that evaluates the multi-objective formulation
    (Eq. 6-8 of the mathematical model) as a single weighted-sum fitness
    value, suitable for GA / SA / QGA / Q2GA.

References: "Mathematical Modelling.docx" (AARP formulation) and
"Q2GA_Methodology.pdf" (RL-assisted Quantum-inspired GA).
"""

import numpy as np
from dataclasses import dataclass, field

GENE_BITS = 6                     # bits per real-valued gene for QGA/Q2GA
BIG_M = 3000.0                    # infeasibility penalty


@dataclass
class Instance:
    name: str
    n_patients: int
    n_hospitals: int
    n_vehicles: int

    coords_patients: np.ndarray       # (n_patients, 2)
    coords_hospitals: np.ndarray      # (n_hospitals, 2)

    patient_type: np.ndarray          # values in {1,2,3}
    service_time: np.ndarray          # base service time s_i
    dropoff_time: np.ndarray          # hospital drop-off time (type 1/2 only)

    vehicle_class: np.ndarray         # 0=A, 1=B, 2=C
    vehicle_home: np.ndarray          # index of home hospital

    Q: float                          # battery capacity
    r: float                          # consumption rate per unit distance
    speed: float                      # travel speed (distance/time)

    a1: np.ndarray                    # no-penalty threshold (type 1/2)
    a2: np.ndarray                    # critical threshold (type 1/2)
    lambda1: float                    # penalty coefficient (type 1)
    lambda2: float                    # penalty coefficient (type 2)

    hosp_cap_critical: np.ndarray     # per-hospital capacity for type 1
    hosp_cap_moderate: np.ndarray     # per-hospital capacity for type 2

    mttr: float
    mtbf: float
    gamma_budget: int                 # uncertainty budget Gamma
    e_max: np.ndarray                 # max uncertainty per patient
    theta: np.ndarray                 # realised uncertainty buffer (fixed scenario)

    w1: float = 0.5
    w2: float = 0.3
    w3: float = 0.2

    # Optional precomputed travel-distance matrix between all locations,
    # indexed 0..n_hospitals-1 for hospitals followed by n_hospitals..
    # n_hospitals+n_patients-1 for patients (e.g. road-network distances
    # in km from a routing service). If None, Euclidean distance on
    # coords_hospitals/coords_patients is used instead.
    dist_matrix: np.ndarray = None

    @property
    def n_real_genes(self):
        return 2 * self.n_patients

    @property
    def n_qubits(self):
        return self.n_real_genes * GENE_BITS

    def _coord(self, idx):
        if idx < self.n_hospitals:
            return self.coords_hospitals[idx]
        return self.coords_patients[idx - self.n_hospitals]

    def loc_dist(self, i, j):
        """Travel distance between location indices i and j (hospitals are
        indices 0..n_hospitals-1, patients are n_hospitals..
        n_hospitals+n_patients-1). Uses `dist_matrix` if available, else
        falls back to Euclidean distance on coordinates."""
        if self.dist_matrix is not None:
            return float(self.dist_matrix[i, j])
        return _euclid(self._coord(i), self._coord(j))


def _euclid(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def generate_instance(size: str, seed: int = 0,
                       gamma_budget: int = None, mttr: float = None) -> Instance:
    """Generate a synthetic AARP instance.

    size in {'small','medium','large'}

    `gamma_budget` and `mttr` may be supplied to override the default
    uncertainty-budget / mean-time-to-repair values (used for the cyber-risk
    sensitivity analysis): a larger `gamma_budget` means more patients are
    affected by a communication/control disruption, and a larger `mttr`
    means each disruption lasts longer on average.
    """
    sizes = {
        "small": dict(n_patients=15, n_hospitals=2, n_vehicles=5),
        "medium": dict(n_patients=30, n_hospitals=3, n_vehicles=9),
        "large": dict(n_patients=50, n_hospitals=4, n_vehicles=14),
    }
    if size not in sizes:
        raise ValueError(f"unknown size '{size}'")
    cfg = sizes[size]
    rng = np.random.default_rng(seed)

    n_patients = cfg["n_patients"]
    n_hospitals = cfg["n_hospitals"]
    n_vehicles = cfg["n_vehicles"]

    AREA = 60.0
    coords_patients = rng.uniform(0, AREA, size=(n_patients, 2))
    coords_hospitals = rng.uniform(0, AREA, size=(n_hospitals, 2))

    # Patient triage classification: 1=critical, 2=moderate, 3=minor
    patient_type = rng.choice([1, 2, 3], size=n_patients, p=[0.25, 0.35, 0.40])

    service_time = rng.uniform(5, 15, size=n_patients)
    dropoff_time = np.where(np.isin(patient_type, [1, 2]),
                             rng.uniform(5, 12, size=n_patients), 0.0)

    # Vehicle classes: 0=Class A (critical), 1=Class B (moderate),
    # 2=Class C (minor). Roughly proportional to demand mix.
    n_a = max(1, round(n_vehicles * 0.30))
    n_b = max(1, round(n_vehicles * 0.35))
    n_c = n_vehicles - n_a - n_b
    vehicle_class = np.array([0] * n_a + [1] * n_b + [2] * n_c)
    rng.shuffle(vehicle_class)
    vehicle_home = rng.integers(0, n_hospitals, size=n_vehicles)

    # Battery parameters
    Q = 350.0
    r = 1.0          # battery units consumed per unit distance
    speed = 1.0      # distance units per time unit -> travel_time = distance

    # Survival-function based semi-soft time-window thresholds
    a1 = np.full(n_patients, np.inf)
    a2 = np.full(n_patients, np.inf)
    crit_mask = patient_type == 1
    mod_mask = patient_type == 2
    a1[crit_mask] = rng.uniform(40, 80, size=crit_mask.sum())
    a2[crit_mask] = a1[crit_mask] + rng.uniform(30, 60, size=crit_mask.sum())
    a1[mod_mask] = rng.uniform(60, 110, size=mod_mask.sum())
    a2[mod_mask] = a1[mod_mask] + rng.uniform(40, 70, size=mod_mask.sum())

    lambda1 = 5.0    # penalty coefficient for type-1 (critical)
    lambda2 = 2.0    # penalty coefficient for type-2 (moderate)

    n_crit = int(crit_mask.sum())
    n_mod = int(mod_mask.sum())
    hosp_cap_critical = rng.integers(max(2, n_crit // n_hospitals),
                                      max(3, n_crit), size=n_hospitals)
    hosp_cap_moderate = rng.integers(max(2, n_mod // n_hospitals),
                                      max(3, n_mod), size=n_hospitals)

    # MTTR / MTBF based interruption modelling
    if mttr is None:
        mttr = 8.0       # mean time to repair (time units)
    mtbf = 60.0      # mean time between failures
    if gamma_budget is None:
        gamma_budget = max(1, round(0.2 * n_patients))   # budget of uncertainty
    gamma_budget = int(min(max(gamma_budget, 0), n_patients))
    e_max = rng.uniform(5, 15, size=n_patients)

    # Pre-sample a fixed disruption scenario (Eq. 4-5): exponential
    # interruption times with rate 1/MTTR, capped by e_max, applied to
    # at most `gamma_budget` patients (z_i indicator).
    e_i = rng.exponential(scale=mttr, size=n_patients)
    affected_idx = rng.choice(n_patients, size=gamma_budget, replace=False)
    z = np.zeros(n_patients, dtype=int)
    z[affected_idx] = 1
    theta = z * np.minimum(e_i, e_max)

    return Instance(
        name=size, n_patients=n_patients, n_hospitals=n_hospitals,
        n_vehicles=n_vehicles, coords_patients=coords_patients,
        coords_hospitals=coords_hospitals, patient_type=patient_type,
        service_time=service_time, dropoff_time=dropoff_time,
        vehicle_class=vehicle_class, vehicle_home=vehicle_home,
        Q=Q, r=r, speed=speed, a1=a1, a2=a2, lambda1=lambda1, lambda2=lambda2,
        hosp_cap_critical=hosp_cap_critical, hosp_cap_moderate=hosp_cap_moderate,
        mttr=mttr, mtbf=mtbf, gamma_budget=gamma_budget, e_max=e_max, theta=theta,
    )


# ---------------------------------------------------------------------------
# Decoding and simulation
# ---------------------------------------------------------------------------

def compatible_vehicles(inst: Instance, ptype: int):
    """Return indices of vehicles allowed to serve a patient of given type.

    Per the model assumptions: Class A and Class B AVs may serve Type-1
    (critical) and Type-2 (moderate) patients; Class C AVs serve Type-3
    (minor) patients only.
    """
    if ptype in (1, 2):
        return np.where(np.isin(inst.vehicle_class, [0, 1]))[0]
    return np.where(inst.vehicle_class == 2)[0]


def decode(chromosome: np.ndarray, inst: Instance):
    """Decode a real-valued chromosome in [0,1]^(2N) into per-vehicle routes.

    chromosome[0:N]  -> assignment keys
    chromosome[N:2N] -> sequencing keys
    """
    n = inst.n_patients
    assign_key = chromosome[:n]
    seq_key = chromosome[n:]

    routes = {v: [] for v in range(inst.n_vehicles)}
    for i in range(n):
        comp = compatible_vehicles(inst, inst.patient_type[i])
        idx = int(assign_key[i] * len(comp))
        idx = min(idx, len(comp) - 1)
        v = comp[idx]
        routes[v].append(i)

    for v in routes:
        routes[v].sort(key=lambda i: seq_key[i])
    return routes


def simulate(routes, inst: Instance):
    """Simulate routes and compute objective components.

    Returns a dict with C1, C2, C3 (per-type latest completion times),
    penalty1, penalty2 (delay penalties), infeasibility (penalty terms for
    battery / capacity / unserved violations), and `fitness` (weighted sum).
    """
    n = inst.n_patients
    completion = np.full(n, np.nan)
    arrival = np.full(n, np.nan)
    served = np.zeros(n, dtype=bool)

    cap_crit = inst.hosp_cap_critical.copy()
    cap_mod = inst.hosp_cap_moderate.copy()

    infeasibility = 0.0

    H = inst.n_hospitals

    for v, seq in routes.items():
        if not seq:
            continue
        loc_idx = int(inst.vehicle_home[v])
        time = 0.0
        battery = inst.Q

        for p in seq:
            p_idx = H + p
            d = inst.loc_dist(loc_idx, p_idx)

            # Battery feasibility check (Eq. 3): recharge at nearest hospital
            # if needed before proceeding.
            if inst.r * d > battery:
                hosp_d = np.array([inst.loc_dist(loc_idx, h) for h in range(H)])
                nearest = int(np.argmin(hosp_d))
                if inst.r * hosp_d[nearest] <= battery:
                    time += hosp_d[nearest] / inst.speed
                    loc_idx = nearest
                    battery = inst.Q
                    d = inst.loc_dist(loc_idx, p_idx)
                if inst.r * d > battery:
                    # patient cannot be reached -> unserved, heavy penalty
                    infeasibility += BIG_M
                    continue

            travel = d / inst.speed
            arr = time + travel
            battery -= inst.r * d

            service = inst.service_time[p] + inst.theta[p]
            comp = arr + service
            new_loc_idx = p_idx

            if inst.patient_type[p] in (1, 2):
                hosp_d = np.array([inst.loc_dist(p_idx, h) for h in range(H)])
                nearest = int(np.argmin(hosp_d))
                if inst.r * hosp_d[nearest] > battery:
                    infeasibility += BIG_M
                else:
                    comp += hosp_d[nearest] / inst.speed + inst.dropoff_time[p]
                    battery -= inst.r * hosp_d[nearest]
                    new_loc_idx = nearest
                    if inst.patient_type[p] == 1:
                        cap_crit[nearest] -= 1
                        if cap_crit[nearest] < 0:
                            infeasibility += BIG_M
                    else:
                        cap_mod[nearest] -= 1
                        if cap_mod[nearest] < 0:
                            infeasibility += BIG_M

            if battery < 0:
                infeasibility += BIG_M * abs(battery)

            arrival[p] = arr
            completion[p] = comp
            served[p] = True
            loc_idx = new_loc_idx
            time = comp

    # Unserved patients
    n_unserved = n - served.sum()
    infeasibility += BIG_M * n_unserved

    def _safe_max(vals):
        vals = vals[~np.isnan(vals)]
        return float(vals.max()) if len(vals) else BIG_M

    C1 = _safe_max(completion[inst.patient_type == 1])
    C2 = _safe_max(completion[inst.patient_type == 2])
    C3 = _safe_max(completion[inst.patient_type == 3])

    # Delay penalties via survival-function thresholds (Eq. 1-2)
    penalty1 = 0.0
    penalty2 = 0.0
    for i in range(n):
        if not served[i] or inst.patient_type[i] not in (1, 2):
            continue
        a_i = arrival[i]
        a1, a2 = inst.a1[i], inst.a2[i]
        if a_i <= a1:
            pen = 0.0
        elif a_i <= a2:
            pen = (a_i - a1)
        else:
            pen = (a2 - a1) + 2.0 * (a_i - a2)   # steep penalty beyond a2
        if inst.patient_type[i] == 1:
            penalty1 += inst.lambda1 * pen
        else:
            penalty2 += inst.lambda2 * pen

    fitness = (inst.w1 * C1 + inst.w2 * C2 + inst.w3 * C3
               + penalty1 + penalty2 + infeasibility)

    return dict(C1=C1, C2=C2, C3=C3, penalty1=penalty1, penalty2=penalty2,
                 infeasibility=infeasibility, fitness=fitness,
                 n_unserved=int(n_unserved))


def evaluate(chromosome: np.ndarray, inst: Instance) -> float:
    routes = decode(chromosome, inst)
    res = simulate(routes, inst)
    return res["fitness"]


def evaluate_full(chromosome: np.ndarray, inst: Instance) -> dict:
    routes = decode(chromosome, inst)
    return simulate(routes, inst)


# ---------------------------------------------------------------------------
# Binary <-> real helpers (used by QGA / Q2GA)
# ---------------------------------------------------------------------------

_POW2 = 2 ** np.arange(GENE_BITS - 1, -1, -1)


def bits_to_real(bits: np.ndarray, n_genes: int) -> np.ndarray:
    """bits: array of length n_genes*GENE_BITS of {0,1} -> real vector in [0,1]."""
    bits = bits.reshape(n_genes, GENE_BITS)
    ints = bits @ _POW2
    return ints / (2 ** GENE_BITS - 1)
