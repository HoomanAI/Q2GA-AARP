"""Calgary case-study instance for the Autonomous Ambulance Routing Problem
(AARP) model implemented in ../../Q2GA_AARP/src/aarp_model.py.

Real-world data used:
  - 3 Calgary hospitals (approximate public addresses / coordinates):
      H0: Foothills Medical Centre,   1403 29 St NW
      H1: Peter Lougheed Centre,      3500 26 Ave NE
      H2: Rockyview General Hospital, 7007 14 St SW
  - 25 patient locations spread across Calgary residential neighbourhoods
    (Tuscany, Royal Oak, Evanston, Panorama Hills, Saddle Ridge, Taradale,
    Marlborough, Forest Lawn, Ogden, Riverbend, McKenzie Towne, Mahogany,
    Auburn Bay, Cranston, Sundance, New Brighton, Copperfield, Signal Hill,
    Aspen Woods, Springbank Hill, West Springs, Bowness, Montgomery,
    Killarney, Inglewood).

Unlike the Ottawa case study (which uses Euclidean distance on a local
Cartesian projection), this Calgary instance computes a real **road-network**
travel-distance matrix between every pair of hospitals/patients using the
OpenStreetMap drive network (via the `osmnx` / `networkx` packages) and the
shortest-path distance along that network (in km). This matrix is passed to
the AARP `Instance` as `dist_matrix`, so the route simulator
(`aarp_model.simulate`) uses real driving distances instead of
straight-line (Euclidean) distances.

The road network and the resulting distance matrix are cached to
`results/road_network.graphml` and `results/dist_matrix.npy` respectively,
since downloading/solving the network is comparatively slow.

Latitude/longitude are still converted to a local Cartesian (km) frame via an
equirectangular projection centred on Calgary (51.0447 N, -114.0719 W) for
plotting purposes (instance-layout figures, etc.) -- only `dist_matrix`
(road-network distances) is used by the simulator for travel distances.
1 distance unit = 1 km; 1 time unit = 1 minute; vehicle speed is set to a
representative urban average of 30 km/h (0.5 km/min).
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_AARP = os.path.join(os.path.dirname(ROOT), "Q2GA_AARP")
sys.path.insert(0, ROOT_AARP)

from src.aarp_model import Instance, GENE_BITS  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")

# Reference point for the local projection (approx. centre of Calgary)
LAT0 = 51.0447
LON0 = -114.0719
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
    dict(name="Foothills Medical Centre", lat=51.0644, lon=-114.1306),
    dict(name="Peter Lougheed Centre", lat=51.0625, lon=-114.0086),
    dict(name="Rockyview General Hospital", lat=50.9847, lon=-114.1115),
]

PATIENTS = [
    dict(name="Tuscany", lat=51.1244, lon=-114.2400),
    dict(name="Royal Oak", lat=51.1280, lon=-114.2150),
    dict(name="Evanston", lat=51.1480, lon=-114.1100),
    dict(name="Panorama Hills", lat=51.1450, lon=-114.0700),
    dict(name="Saddle Ridge", lat=51.1250, lon=-113.9700),
    dict(name="Taradale", lat=51.1100, lon=-113.9650),
    dict(name="Marlborough", lat=51.0580, lon=-113.9900),
    dict(name="Forest Lawn", lat=51.0430, lon=-113.9850),
    dict(name="Ogden", lat=50.9950, lon=-114.0050),
    dict(name="Riverbend", lat=50.9750, lon=-113.9900),
    dict(name="McKenzie Towne", lat=50.9300, lon=-113.9700),
    dict(name="Mahogany", lat=50.9100, lon=-113.9450),
    dict(name="Auburn Bay", lat=50.9150, lon=-113.9600),
    dict(name="Cranston", lat=50.8950, lon=-114.0000),
    dict(name="Sundance", lat=50.9450, lon=-114.0200),
    dict(name="New Brighton", lat=50.9270, lon=-113.9800),
    dict(name="Copperfield", lat=50.9100, lon=-113.9250),
    dict(name="Signal Hill", lat=51.0250, lon=-114.1700),
    dict(name="Aspen Woods", lat=51.0450, lon=-114.1950),
    dict(name="Springbank Hill", lat=51.0350, lon=-114.2100),
    dict(name="West Springs", lat=51.0550, lon=-114.2150),
    dict(name="Bowness", lat=51.0850, lon=-114.1750),
    dict(name="Montgomery", lat=51.0750, lon=-114.1450),
    dict(name="Killarney", lat=51.0250, lon=-114.1250),
    dict(name="Inglewood", lat=51.0350, lon=-114.0250),
]

N_PATIENTS = len(PATIENTS)
N_HOSPITALS = len(HOSPITALS)
N_VEHICLES = 9


# ---------------------------------------------------------------------------
# Road-network distance matrix (OpenStreetMap drive network)
# ---------------------------------------------------------------------------

def _all_locations():
    """Hospitals first (indices 0..N_HOSPITALS-1), then patients
    (indices N_HOSPITALS..N_HOSPITALS+N_PATIENTS-1) -- matches the index
    convention used by Instance.loc_dist()."""
    return [(h["lat"], h["lon"]) for h in HOSPITALS] + \
           [(p["lat"], p["lon"]) for p in PATIENTS]


def _bbox(locations, pad=0.05):
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


def _compute_dist_matrix(locations):
    """Returns an (n x n) matrix of shortest road-network driving distances
    (km) between all locations, using OSM data via osmnx/networkx."""
    import osmnx as ox
    import networkx as nx

    G = _download_graph(locations)
    lats = [lat for lat, lon in locations]
    lons = [lon for lat, lon in locations]
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)

    n = len(locations)
    D = np.zeros((n, n))
    for i, src in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, src, weight="length")
        for j, dst in enumerate(nodes):
            if i == j:
                continue
            d_m = lengths.get(dst, None)
            if d_m is None:
                # disconnected fallback: straight-line distance * detour factor
                lat_i, lon_i = locations[i]
                lat_j, lon_j = locations[j]
                xi, yi = latlon_to_xy(lat_i, lon_i)
                xj, yj = latlon_to_xy(lat_j, lon_j)
                d_m = 1000.0 * 1.3 * np.hypot(xi - xj, yi - yj)
            D[i, j] = d_m / 1000.0  # metres -> km
    return D


def _get_dist_matrix():
    cache_path = os.path.join(RESULTS_DIR, "dist_matrix.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    locations = _all_locations()
    D = _compute_dist_matrix(locations)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(cache_path, D)
    return D


# ---------------------------------------------------------------------------
def build_instance(seed: int = 42, gamma_budget: int = None, mttr: float = None) -> Instance:
    """Build an AARP `Instance` for the Calgary case study, using real
    road-network driving distances (`dist_matrix`) instead of Euclidean
    distance.

    `gamma_budget` and `mttr` may be supplied to override the default
    cyber-risk uncertainty-buffer settings (used for sensitivity analysis).
    """
    rng = np.random.default_rng(seed)

    coords_patients = np.array([latlon_to_xy(p["lat"], p["lon"]) for p in PATIENTS])
    coords_hospitals = np.array([latlon_to_xy(h["lat"], h["lon"]) for h in HOSPITALS])

    dist_matrix = _get_dist_matrix()

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
        name="calgary_case_study", n_patients=N_PATIENTS, n_hospitals=N_HOSPITALS,
        n_vehicles=N_VEHICLES, coords_patients=coords_patients,
        coords_hospitals=coords_hospitals, patient_type=patient_type,
        service_time=service_time, dropoff_time=dropoff_time,
        vehicle_class=vehicle_class, vehicle_home=vehicle_home,
        Q=Q, r=r, speed=speed, a1=a1, a2=a2, lambda1=lambda1, lambda2=lambda2,
        hosp_cap_critical=hosp_cap_critical, hosp_cap_moderate=hosp_cap_moderate,
        mttr=mttr, mtbf=mtbf, gamma_budget=gamma_budget, e_max=e_max, theta=theta,
        dist_matrix=dist_matrix,
    )


if __name__ == "__main__":
    inst = build_instance()
    print(f"Calgary case study: N={inst.n_patients} patients, "
          f"H={inst.n_hospitals} hospitals, M={inst.n_vehicles} vehicles")
    print("Patient type counts:", {t: int((inst.patient_type == t).sum()) for t in (1, 2, 3)})
    print("Coordinate bounding box (km):",
          inst.coords_patients.min(axis=0), inst.coords_patients.max(axis=0))
    print("Road-distance matrix shape:", inst.dist_matrix.shape)
    print("Max road distance (km):", inst.dist_matrix.max())
