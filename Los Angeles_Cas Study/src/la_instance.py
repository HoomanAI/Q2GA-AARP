"""Los Angeles wildfire case-study instance for the Autonomous Ambulance
Routing Problem (AARP) model implemented in
../../Q2GA_AARP/src/aarp_model.py.

Real-world basis:
  The January 2025 Palisades Fire burned through Pacific Palisades, the
  Topanga area, and parts of Malibu in west Los Angeles, destroying homes
  and infrastructure and knocking out cell towers, fibre, and power across
  that area for an extended period (a real "network outage in part of the
  city"). This case study models that event:

    Wildfire (Palisades Fire, Jan 2025)
      -> physical road-network damage / closures inside the burn zone
      -> communications-infrastructure outage inside the burn zone
         (modelled as elevated MTTR and reduced MTBF, i.e. lower
          availability = MTBF / (MTBF + MTTR), for patients located there)
      -> impact on the AARP/Q2GA problem (longer realised disruption
         buffers theta_i, longer detours, more unserved/late patients).

Real-world data used:
  - 3 hospitals (approximate public locations):
      H0: UCLA Ronald Reagan Medical Center, Westwood
      H1: Providence Saint John's Health Center, Santa Monica
      H2: Providence Cedars-Sinai Tarzana Medical Center, Tarzana
  - 25 patient locations spanning west LA / the Palisades Fire area
    (Pacific Palisades, Topanga, Malibu, Castellammare, etc.) and
    unaffected west-side / Valley neighbourhoods (Santa Monica, Venice,
    Brentwood, Westwood, Bel Air, Beverly Hills, Century City, Culver City,
    Encino, Tarzana, Woodland Hills, Calabasas, Sherman Oaks, ...).

As with the Calgary case study, real road-network driving distances (via
`osmnx`/`networkx` on the OSM drive network) are used instead of Euclidean
distance -- here in TWO variants:
  - a "baseline" road graph (normal conditions), and
  - a "wildfire" road graph in which roads inside the Palisades Fire burn
    zone are removed (representing closures / destruction), so routes that
    would otherwise pass through or terminate inside the zone must detour
    or fall back to a slow straight-line estimate.

Two `build_instance(scenario=...)` variants are provided:
  - scenario="baseline": normal MTTR=10 / MTBF=120 for all patients,
    20% of patients randomly affected by minor disruptions (as in the
    Ottawa/Calgary case studies), baseline road network.
  - scenario="wildfire": patients inside the Palisades Fire zone get
    degraded communications (MTTR=60, MTBF=30 -> availability ~0.33 vs
    ~0.92 baseline) and are all marked as disrupted (z_i=1); the wildfire
    road network (with burn-zone roads removed) is used for travel
    distances.

Latitude/longitude are converted to a local Cartesian (km) frame via an
equirectangular projection centred near the Palisades Fire zone
(34.06 N, -118.50 W) for plotting purposes only. 1 distance unit = 1 km;
1 time unit = 1 minute; vehicle speed = 0.5 km/min (30 km/h average urban
speed).
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_AARP = os.path.join(os.path.dirname(ROOT), "Q2GA_AARP")
sys.path.insert(0, ROOT_AARP)

from src.aarp_model import Instance, GENE_BITS  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")

# Reference point for the local projection (near the Palisades Fire zone)
LAT0 = 34.06
LON0 = -118.50
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.320 * np.cos(np.radians(LAT0))


def latlon_to_xy(lat, lon):
    x = (lon - LON0) * KM_PER_DEG_LON
    y = (lat - LAT0) * KM_PER_DEG_LAT
    return x, y


# ---------------------------------------------------------------------------
# Real-world location data
# ---------------------------------------------------------------------------

HOSPITALS = [
    dict(name="UCLA Ronald Reagan Medical Center (Westwood)", lat=34.0658, lon=-118.4452),
    dict(name="Providence Saint John's Health Center (Santa Monica)", lat=34.0259, lon=-118.4781),
    dict(name="Providence Cedars-Sinai Tarzana Medical Center (Tarzana)", lat=34.1733, lon=-118.5384),
]

# Patients 0-7 lie inside the Palisades Fire burn zone (Jan 2025); the
# remainder are elsewhere on the LA west side / San Fernando Valley.
PATIENTS = [
    # --- Palisades Fire zone (wildfire-affected) ---
    dict(name="Pacific Palisades Village", lat=34.0480, lon=-118.5265),
    dict(name="Palisades Highlands", lat=34.0633, lon=-118.5363),
    dict(name="Marquez Knolls", lat=34.0376, lon=-118.5235),
    dict(name="Castellammare", lat=34.0367, lon=-118.5440),
    dict(name="Sunset Mesa", lat=34.0387, lon=-118.5337),
    dict(name="Rustic Canyon", lat=34.0500, lon=-118.5400),
    dict(name="Topanga", lat=34.0942, lon=-118.6004),
    dict(name="Malibu (Carbon Beach / PCH)", lat=34.0367, lon=-118.5650),
    # --- Unaffected west-side / Valley neighbourhoods ---
    dict(name="Santa Monica", lat=34.0195, lon=-118.4912),
    dict(name="Venice", lat=33.9850, lon=-118.4695),
    dict(name="Mar Vista", lat=34.0008, lon=-118.4351),
    dict(name="Brentwood", lat=34.0524, lon=-118.4738),
    dict(name="Westwood", lat=34.0633, lon=-118.4453),
    dict(name="Bel Air", lat=34.0928, lon=-118.4598),
    dict(name="Beverly Hills", lat=34.0736, lon=-118.4004),
    dict(name="Century City", lat=34.0586, lon=-118.4176),
    dict(name="Culver City", lat=34.0211, lon=-118.3965),
    dict(name="Encino", lat=34.1592, lon=-118.5012),
    dict(name="Tarzana", lat=34.1808, lon=-118.5396),
    dict(name="Woodland Hills", lat=34.1683, lon=-118.6059),
    dict(name="Calabasas", lat=34.1378, lon=-118.6603),
    dict(name="Sherman Oaks", lat=34.1509, lon=-118.4490),
    dict(name="West Los Angeles", lat=34.0364, lon=-118.4486),
    dict(name="Pacific Palisades Drive (East)", lat=34.0700, lon=-118.5230),
    dict(name="Malibu Canyon / Las Virgenes", lat=34.0840, lon=-118.6776),
]

N_PATIENTS = len(PATIENTS)
N_HOSPITALS = len(HOSPITALS)
N_VEHICLES = 9

# Indices (within PATIENTS) of patients located inside the Palisades Fire
# burn zone -- used both for the road-closure graph and the degraded
# communications (MTTR/MTBF) modelling in the "wildfire" scenario.
WILDFIRE_ZONE_PATIENT_IDX = list(range(8))  # the first 8 patients above

# Approximate Palisades Fire perimeter, used (a) to decide which OSM road
# edges to remove for the "wildfire" road graph, and (b) for map overlays.
WILDFIRE_ZONE_CENTER = (34.058, -118.545)  # (lat, lon)
WILDFIRE_ZONE_RADIUS_KM = 7.0


def _dist_km(lat1, lon1, lat2, lon2):
    x1, y1 = latlon_to_xy(lat1, lon1)
    x2, y2 = latlon_to_xy(lat2, lon2)
    return float(np.hypot(x1 - x2, y1 - y2))


def in_wildfire_zone(lat, lon):
    return _dist_km(lat, lon, *WILDFIRE_ZONE_CENTER) <= WILDFIRE_ZONE_RADIUS_KM


# ---------------------------------------------------------------------------
# Road-network distance matrices (OpenStreetMap drive network)
# ---------------------------------------------------------------------------

def _all_locations():
    """Hospitals first (indices 0..N_HOSPITALS-1), then patients
    (indices N_HOSPITALS..N_HOSPITALS+N_PATIENTS-1) -- matches the index
    convention used by Instance.loc_dist()."""
    return [(h["lat"], h["lon"]) for h in HOSPITALS] + \
           [(p["lat"], p["lon"]) for p in PATIENTS]


def _bbox(locations, pad=0.03):
    lats = [lat for lat, lon in locations]
    lons = [lon for lat, lon in locations]
    west, east = min(lons) - pad, max(lons) + pad
    south, north = min(lats) - pad, max(lats) + pad
    return (west, south, east, north)


def _download_graph(locations):
    import osmnx as ox
    graphml_path = os.path.join(RESULTS_DIR, "road_network.graphml")
    if os.path.exists(graphml_path):
        return ox.load_graphml(graphml_path)
    bbox = _bbox(locations)
    G = ox.graph_from_bbox(bbox=bbox, network_type="drive", simplify=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ox.save_graphml(G, graphml_path)
    return G


def _wildfire_graph(G):
    """Returns a copy of G with all edges that lie inside the Palisades Fire
    burn zone removed (representing road closures / destruction)."""
    import osmnx as ox
    graphml_path = os.path.join(RESULTS_DIR, "road_network_wildfire.graphml")
    if os.path.exists(graphml_path):
        return ox.load_graphml(graphml_path)

    Gw = G.copy()
    nodes_in_zone = set()
    for n, data in Gw.nodes(data=True):
        lat, lon = data.get("y"), data.get("x")
        if lat is not None and lon is not None and in_wildfire_zone(lat, lon):
            nodes_in_zone.add(n)

    edges_to_remove = []
    for u, v, k in Gw.edges(keys=True):
        if u in nodes_in_zone and v in nodes_in_zone:
            edges_to_remove.append((u, v, k))
    Gw.remove_edges_from(edges_to_remove)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ox.save_graphml(Gw, graphml_path)
    return Gw


def _compute_dist_matrix_for_graph(G, locations):
    """Returns an (n x n) matrix of shortest road-network driving distances
    (km) between all locations, using the supplied graph G."""
    import osmnx as ox
    import networkx as nx

    lats = [lat for lat, lon in locations]
    lons = [lon for lat, lon in locations]
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)

    n = len(locations)
    D = np.zeros((n, n))
    for i, src in enumerate(nodes):
        try:
            lengths = nx.single_source_dijkstra_path_length(G, src, weight="length")
        except Exception:
            lengths = {}
        for j, dst in enumerate(nodes):
            if i == j:
                continue
            d_m = lengths.get(dst, None)
            if d_m is None:
                # disconnected fallback: straight-line distance * detour factor
                # (representing a road closure forcing a long, slow detour)
                lat_i, lon_i = locations[i]
                lat_j, lon_j = locations[j]
                xi, yi = latlon_to_xy(lat_i, lon_i)
                xj, yj = latlon_to_xy(lat_j, lon_j)
                d_m = 1000.0 * 1.8 * np.hypot(xi - xj, yi - yj)
            D[i, j] = d_m / 1000.0  # metres -> km
    return D


def _get_dist_matrix(scenario):
    cache_name = "dist_matrix_wildfire.npy" if scenario == "wildfire" else "dist_matrix.npy"
    cache_path = os.path.join(RESULTS_DIR, cache_name)
    if os.path.exists(cache_path):
        return np.load(cache_path)

    locations = _all_locations()
    G = _download_graph(locations)
    if scenario == "wildfire":
        G = _wildfire_graph(G)
    D = _compute_dist_matrix_for_graph(G, locations)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(cache_path, D)
    return D


# ---------------------------------------------------------------------------
def build_instance(scenario: str = "baseline", seed: int = 42,
                    gamma_budget: int = None, mttr: float = None) -> Instance:
    """Build an AARP `Instance` for the LA wildfire case study.

    scenario: "baseline" (normal operations, no wildfire) or
              "wildfire" (Palisades Fire active -- road closures inside the
              burn zone AND degraded communications/MTTR-MTBF for patients
              located there).

    `gamma_budget` and `mttr` may be supplied to override the default
    cyber-risk uncertainty-buffer settings (used for sensitivity analysis);
    they are only applied in the "baseline" scenario -- the "wildfire"
    scenario always uses the zone-based disruption model described above.
    """
    rng = np.random.default_rng(seed)

    coords_patients = np.array([latlon_to_xy(p["lat"], p["lon"]) for p in PATIENTS])
    coords_hospitals = np.array([latlon_to_xy(h["lat"], h["lon"]) for h in HOSPITALS])

    dist_matrix = _get_dist_matrix(scenario)

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

    # --- MTTR / MTBF based interruption modelling (cyber-risk uncertainty buffer) ---
    if scenario == "wildfire":
        # Patients inside the Palisades Fire burn zone: communications
        # infrastructure (cell towers, fibre, power) is degraded/destroyed
        # -> much longer mean time to repair, much shorter mean time
        # between failures -> low availability.
        MTTR_ZONE, MTBF_ZONE = 60.0, 30.0          # availability ~ 0.333
        MTTR_OUT, MTBF_OUT = 12.0, 100.0           # availability ~ 0.893 (slightly degraded region-wide)

        zone_mask = np.zeros(N_PATIENTS, dtype=bool)
        zone_mask[WILDFIRE_ZONE_PATIENT_IDX] = True

        mttr_i = np.where(zone_mask, MTTR_ZONE, MTTR_OUT)
        mtbf_i = np.where(zone_mask, MTBF_ZONE, MTBF_OUT)

        e_max = np.where(zone_mask,
                          rng.uniform(25, 40, size=N_PATIENTS),
                          rng.uniform(10, 25, size=N_PATIENTS))

        # All patients in the burn zone are disrupted (z=1); outside the
        # zone, a small number of additional patients are affected by
        # region-wide network congestion during the disaster.
        z = zone_mask.astype(int).copy()
        n_extra = max(0, round(0.10 * (N_PATIENTS - zone_mask.sum())))
        outside_idx = np.where(~zone_mask)[0]
        extra_idx = rng.choice(outside_idx, size=n_extra, replace=False)
        z[extra_idx] = 1

        e_i = rng.exponential(scale=mttr_i, size=N_PATIENTS)
        theta = z * np.minimum(e_i, e_max)

        gamma_budget_val = int(z.sum())
        mttr_report = MTTR_ZONE
        mtbf_report = MTBF_ZONE
    else:
        if mttr is None:
            mttr = 10.0       # mean time to repair / recover (minutes)
        mtbf = 120.0          # mean time between failures (minutes)
        if gamma_budget is None:
            gamma_budget = max(1, round(0.2 * N_PATIENTS))  # 20% of patients affected
        gamma_budget = int(min(max(gamma_budget, 0), N_PATIENTS))
        e_max = rng.uniform(10, 25, size=N_PATIENTS)

        e_i = rng.exponential(scale=mttr, size=N_PATIENTS)
        affected_idx = rng.choice(N_PATIENTS, size=gamma_budget, replace=False)
        z = np.zeros(N_PATIENTS, dtype=int)
        z[affected_idx] = 1
        theta = z * np.minimum(e_i, e_max)

        gamma_budget_val = gamma_budget
        mttr_report = mttr
        mtbf_report = mtbf

    return Instance(
        name=f"la_wildfire_case_study_{scenario}", n_patients=N_PATIENTS, n_hospitals=N_HOSPITALS,
        n_vehicles=N_VEHICLES, coords_patients=coords_patients,
        coords_hospitals=coords_hospitals, patient_type=patient_type,
        service_time=service_time, dropoff_time=dropoff_time,
        vehicle_class=vehicle_class, vehicle_home=vehicle_home,
        Q=Q, r=r, speed=speed, a1=a1, a2=a2, lambda1=lambda1, lambda2=lambda2,
        hosp_cap_critical=hosp_cap_critical, hosp_cap_moderate=hosp_cap_moderate,
        mttr=mttr_report, mtbf=mtbf_report, gamma_budget=gamma_budget_val,
        e_max=e_max, theta=theta,
        dist_matrix=dist_matrix,
    )


if __name__ == "__main__":
    for scenario in ("baseline", "wildfire"):
        inst = build_instance(scenario=scenario)
        print(f"\n=== LA case study ({scenario}) ===")
        print(f"N={inst.n_patients} patients, H={inst.n_hospitals} hospitals, M={inst.n_vehicles} vehicles")
        print("Patient type counts:", {t: int((inst.patient_type == t).sum()) for t in (1, 2, 3)})
        print("gamma_budget:", inst.gamma_budget, " mttr:", inst.mttr, " mtbf:", inst.mtbf,
              " availability:", inst.mtbf / (inst.mtbf + inst.mttr))
        print("theta (disruption buffer) stats: mean=%.3f max=%.3f n_nonzero=%d" %
              (inst.theta.mean(), inst.theta.max(), int((inst.theta > 0).sum())))
        print("Road-distance matrix shape:", inst.dist_matrix.shape)
        print("Max road distance (km):", inst.dist_matrix.max())
