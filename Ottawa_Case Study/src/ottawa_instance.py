"""Ottawa case-study instance for the Autonomous Ambulance Routing Problem
(AARP) model implemented in ../../Q2GA_AARP/src/aarp_model.py.

Real-world data used:
  - 3 Ottawa hospitals (approximate public addresses / coordinates):
      H0: The Ottawa Hospital -- Civic Campus,    1053 Carling Ave
      H1: The Ottawa Hospital -- General Campus,  501 Smyth Rd
      H2: Queensway Carleton Hospital,            3045 Baseline Rd
  - 25 patient locations spread across Ottawa residential
    neighbourhoods (Barrhaven, Kanata, Stittsville, Nepean, Westboro,
    Hintonburg, Centretown, Sandy Hill, Vanier, Overbrook, Rockcliffe Park,
    Orleans, Gloucester, Alta Vista, Riverside South, Manotick, Greely,
    Carlington, Carleton Heights, the Glebe, Old Ottawa South).

Latitude/longitude are converted to a local Cartesian (km) frame via an
equirectangular projection centred on Ottawa (45.4215 N, -75.6972 W), so the
existing AARP simulator (which assumes Euclidean distances) can be applied
directly. 1 distance unit = 1 km; 1 time unit = 1 minute; vehicle speed is
set to a representative urban average of 30 km/h (0.5 km/min).
"""

import os
import numpy as np
import sys

ROOT_AARP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "Q2GA_AARP")
sys.path.insert(0, ROOT_AARP)

from src.aarp_model import Instance, GENE_BITS  # noqa: E402

# Reference point for the local projection (approx. centre of Ottawa)
LAT0 = 45.4215
LON0 = -75.6972
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320 * np.cos(np.radians(LAT0))


def latlon_to_xy(lat, lon):
    x = (lon - LON0) * KM_PER_DEG_LON
    y = (lat - LAT0) * KM_PER_DEG_LAT
    return x, y


# ---------------------------------------------------------------------------
# Real-world location data
# ---------------------------------------------------------------------------

HOSPITALS = [
    dict(name="The Ottawa Hospital -- Civic Campus", lat=45.3947, lon=-75.7173),
    dict(name="The Ottawa Hospital -- General Campus", lat=45.3879, lon=-75.6917),
    dict(name="Queensway Carleton Hospital", lat=45.3344, lon=-75.7765),
]

PATIENTS = [
    dict(name="Barrhaven (Half Moon Bay)", lat=45.2750, lon=-75.7430),
    dict(name="Barrhaven (Stonebridge)", lat=45.2820, lon=-75.7330),
    dict(name="Kanata (Bridlewood)", lat=45.2950, lon=-75.9050),
    dict(name="Kanata (Beaverbrook)", lat=45.3200, lon=-75.9100),
    dict(name="Stittsville", lat=45.2540, lon=-75.9200),
    dict(name="Nepean (Centrepointe)", lat=45.3470, lon=-75.7530),
    dict(name="Nepean (Bells Corners)", lat=45.3300, lon=-75.8200),
    dict(name="Westboro", lat=45.3890, lon=-75.7560),
    dict(name="Hintonburg", lat=45.4040, lon=-75.7260),
    dict(name="Centretown", lat=45.4150, lon=-75.6950),
    dict(name="Sandy Hill", lat=45.4230, lon=-75.6800),
    dict(name="Vanier", lat=45.4370, lon=-75.6650),
    dict(name="Overbrook", lat=45.4290, lon=-75.6580),
    dict(name="Rockcliffe Park", lat=45.4430, lon=-75.6720),
    dict(name="Orleans (Convent Glen)", lat=45.4570, lon=-75.5500),
    dict(name="Orleans (Chapel Hill)", lat=45.4720, lon=-75.5180),
    dict(name="Gloucester (Blackburn Hamlet)", lat=45.4280, lon=-75.5800),
    dict(name="Alta Vista", lat=45.3850, lon=-75.6650),
    dict(name="Riverside South", lat=45.3210, lon=-75.6630),
    dict(name="Manotick", lat=45.2280, lon=-75.6730),
    dict(name="Greely", lat=45.2620, lon=-75.6080),
    dict(name="Carlington", lat=45.3780, lon=-75.7280),
    dict(name="Carleton Heights", lat=45.3700, lon=-75.7330),
    dict(name="The Glebe", lat=45.3990, lon=-75.6890),
    dict(name="Old Ottawa South", lat=45.3920, lon=-75.6840),
]

N_PATIENTS = len(PATIENTS)
N_HOSPITALS = len(HOSPITALS)
N_VEHICLES = 9


def build_instance(seed: int = 42, gamma_budget: int = None, mttr: float = None) -> Instance:
    """Build an AARP `Instance` for the Ottawa case study.

    `gamma_budget` and `mttr` may be supplied to override the default
    cyber-risk uncertainty-buffer settings (used for sensitivity analysis).
    """
    rng = np.random.default_rng(seed)

    coords_patients = np.array([latlon_to_xy(p["lat"], p["lon"]) for p in PATIENTS])
    coords_hospitals = np.array([latlon_to_xy(h["lat"], h["lon"]) for h in HOSPITALS])

    # Patient triage classification: 1=critical, 2=moderate, 3=minor
    patient_type = rng.choice([1, 2, 3], size=N_PATIENTS, p=[0.25, 0.35, 0.40])

    service_time = rng.uniform(8, 20, size=N_PATIENTS)        # minutes on scene
    dropoff_time = np.where(np.isin(patient_type, [1, 2]),
                             rng.uniform(10, 20, size=N_PATIENTS), 0.0)

    # Vehicle classes: 0=Class A (critical/moderate), 1=Class B (critical/moderate),
    # 2=Class C (minor). Roughly proportional to demand mix.
    n_a = max(1, round(N_VEHICLES * 0.30))
    n_b = max(1, round(N_VEHICLES * 0.35))
    n_c = N_VEHICLES - n_a - n_b
    vehicle_class = np.array([0] * n_a + [1] * n_b + [2] * n_c)
    rng.shuffle(vehicle_class)
    # 3 vehicles based at each of the 3 hospitals
    vehicle_home = np.array([i % N_HOSPITALS for i in range(N_VEHICLES)])
    rng.shuffle(vehicle_home)

    # Battery parameters (electric ambulance: ~300 km range)
    Q = 300.0
    r = 1.0           # battery units consumed per km
    speed = 0.5       # km per minute (30 km/h average urban speed)

    # Semi-soft time-window thresholds (minutes from dispatch)
    a1 = np.full(N_PATIENTS, np.inf)
    a2 = np.full(N_PATIENTS, np.inf)
    crit_mask = patient_type == 1
    mod_mask = patient_type == 2
    a1[crit_mask] = rng.uniform(20, 40, size=crit_mask.sum())
    a2[crit_mask] = a1[crit_mask] + rng.uniform(15, 30, size=crit_mask.sum())
    a1[mod_mask] = rng.uniform(40, 70, size=mod_mask.sum())
    a2[mod_mask] = a1[mod_mask] + rng.uniform(25, 45, size=mod_mask.sum())

    lambda1 = 5.0
    lambda2 = 2.0

    n_crit = int(crit_mask.sum())
    n_mod = int(mod_mask.sum())
    hosp_cap_critical = rng.integers(max(2, n_crit // N_HOSPITALS), max(3, n_crit), size=N_HOSPITALS)
    hosp_cap_moderate = rng.integers(max(2, n_mod // N_HOSPITALS), max(3, n_mod), size=N_HOSPITALS)

    # MTTR / MTBF based interruption modelling (cyber-risk uncertainty buffer)
    if mttr is None:
        mttr = 10.0       # mean time to repair / recover (minutes)
    mtbf = 120.0      # mean time between failures (minutes)
    if gamma_budget is None:
        gamma_budget = max(1, round(0.2 * N_PATIENTS))  # 20% of patients affected
    gamma_budget = int(min(max(gamma_budget, 0), N_PATIENTS))
    e_max = rng.uniform(10, 25, size=N_PATIENTS)

    e_i = rng.exponential(scale=mttr, size=N_PATIENTS)
    affected_idx = rng.choice(N_PATIENTS, size=gamma_budget, replace=False)
    z = np.zeros(N_PATIENTS, dtype=int)
    z[affected_idx] = 1
    theta = z * np.minimum(e_i, e_max)

    return Instance(
        name="ottawa_case_study", n_patients=N_PATIENTS, n_hospitals=N_HOSPITALS,
        n_vehicles=N_VEHICLES, coords_patients=coords_patients,
        coords_hospitals=coords_hospitals, patient_type=patient_type,
        service_time=service_time, dropoff_time=dropoff_time,
        vehicle_class=vehicle_class, vehicle_home=vehicle_home,
        Q=Q, r=r, speed=speed, a1=a1, a2=a2, lambda1=lambda1, lambda2=lambda2,
        hosp_cap_critical=hosp_cap_critical, hosp_cap_moderate=hosp_cap_moderate,
        mttr=mttr, mtbf=mtbf, gamma_budget=gamma_budget, e_max=e_max, theta=theta,
    )


if __name__ == "__main__":
    inst = build_instance()
    print(f"Ottawa case study: N={inst.n_patients} patients, "
          f"H={inst.n_hospitals} hospitals, M={inst.n_vehicles} vehicles")
    print("Patient type counts:", {t: int((inst.patient_type == t).sum()) for t in (1, 2, 3)})
    print("Coordinate bounding box (km):",
          inst.coords_patients.min(axis=0), inst.coords_patients.max(axis=0))
