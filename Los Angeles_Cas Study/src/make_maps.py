"""Generate OpenStreetMap-based maps for the LA wildfire case study.

Outputs (in maps/):
  - static_map.png         : hospitals + patient locations + Palisades Fire zone, OSM basemap
  - interactive_map.html    : folium interactive map (hospitals + patients + fire zone, popups)
  - route_map_baseline.png  : best Q2GA solution's vehicle routes, baseline scenario
  - route_map_wildfire.png  : best Q2GA solution's vehicle routes, wildfire scenario
  - route_map_baseline.html / route_map_wildfire.html : interactive folium versions
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import contextily as cx
import folium

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_AARP = os.path.join(os.path.dirname(ROOT), "Q2GA_AARP")
sys.path.insert(0, ROOT_AARP)
sys.path.insert(0, ROOT)

from src.la_instance import (build_instance, HOSPITALS, PATIENTS,  # noqa: E402
                              WILDFIRE_ZONE_PATIENT_IDX, WILDFIRE_ZONE_CENTER,
                              WILDFIRE_ZONE_RADIUS_KM)

MAPS_DIR = os.path.join(ROOT, "maps")
RESULTS_DIR = os.path.join(ROOT, "results")

TYPE_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c"}
TYPE_LABELS = {1: "Critical", 2: "Moderate", 3: "Minor"}

ROUTE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

MAP_CENTER = [34.06, -118.50]

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
})


def ensure_dirs():
    os.makedirs(MAPS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
def lonlat_to_webmercator(lon, lat):
    k = 6378137.0
    x = lon * (np.pi / 180) * k
    y = np.log(np.tan((90 + lat) * np.pi / 360)) * k
    return x, y


def _zone_circle_webmercator():
    """Approximate the Palisades Fire zone circle (defined in local km
    coordinates) as a polygon in Web Mercator for plotting."""
    lat0, lon0 = WILDFIRE_ZONE_CENTER
    KM_PER_DEG_LAT = 111.0
    KM_PER_DEG_LON = 111.320 * np.cos(np.radians(lat0))
    angles = np.linspace(0, 2 * np.pi, 100)
    pts = []
    for a in angles:
        dlat = (WILDFIRE_ZONE_RADIUS_KM * np.sin(a)) / KM_PER_DEG_LAT
        dlon = (WILDFIRE_ZONE_RADIUS_KM * np.cos(a)) / KM_PER_DEG_LON
        x, y = lonlat_to_webmercator(lon0 + dlon, lat0 + dlat)
        pts.append((x, y))
    return pts


def _add_zone_overlay(ax):
    pts = _zone_circle_webmercator()
    xs, ys = zip(*pts)
    ax.fill(xs, ys, color="orangered", alpha=0.12, zorder=2)
    ax.plot(xs, ys, color="orangered", linewidth=2, linestyle="--", zorder=3,
            label="Palisades Fire zone (approx.)")


# ---------------------------------------------------------------------------
def fig_static_map(inst):
    fig, ax = plt.subplots(figsize=(9, 9))

    _add_zone_overlay(ax)

    for t in (1, 2, 3):
        pts = [p for i, p in enumerate(PATIENTS) if inst.patient_type[i] == t]
        if not pts:
            continue
        xs, ys = zip(*[lonlat_to_webmercator(p["lon"], p["lat"]) for p in pts])
        ax.scatter(xs, ys, color=TYPE_COLORS[t], s=70, alpha=0.9, edgecolor="k",
                   linewidth=0.6, label=f"{TYPE_LABELS[t]} patients", zorder=4)

    hx, hy = zip(*[lonlat_to_webmercator(h["lon"], h["lat"]) for h in HOSPITALS])
    ax.scatter(hx, hy, marker="*", s=500, color="gold", edgecolor="k", linewidth=1.2,
               label="Hospitals", zorder=5)
    for h, x, y in zip(HOSPITALS, hx, hy):
        ax.annotate(h["name"], (x, y), fontsize=7, xytext=(6, 6),
                    textcoords="offset points")

    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, crs="EPSG:3857")
    ax.set_title("LA wildfire case study: hospitals, patient locations,\n"
                  "and Palisades Fire zone (Jan 2025) -- OpenStreetMap basemap")
    ax.set_axis_off()
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(MAPS_DIR, "static_map.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_interactive_map(inst):
    m = folium.Map(location=MAP_CENTER, zoom_start=11, tiles="OpenStreetMap")

    folium.Circle(
        location=list(WILDFIRE_ZONE_CENTER),
        radius=WILDFIRE_ZONE_RADIUS_KM * 1000.0,
        color="orangered", fill=True, fill_opacity=0.12, weight=2, dash_array="6",
        popup="Palisades Fire zone (approx.)",
    ).add_to(m)

    for h in HOSPITALS:
        folium.Marker(
            location=[h["lat"], h["lon"]],
            popup=h["name"],
            icon=folium.Icon(color="orange", icon="plus-sign"),
        ).add_to(m)

    color_map = {1: "red", 2: "orange", 3: "green"}
    for i, p in enumerate(PATIENTS):
        t = int(inst.patient_type[i])
        in_zone = i in WILDFIRE_ZONE_PATIENT_IDX
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=9 if in_zone else 7,
            color="black" if in_zone else color_map[t],
            weight=2 if in_zone else 1,
            fill=True,
            fill_color=color_map[t],
            fill_opacity=0.85,
            popup=f"P{i}: {p['name']} ({TYPE_LABELS[t]})"
                  + (" -- Palisades Fire zone" if in_zone else ""),
        ).add_to(m)

    m.save(os.path.join(MAPS_DIR, "interactive_map.html"))


# ---------------------------------------------------------------------------
def _load_routes(scenario):
    path = os.path.join(RESULTS_DIR, f"best_routes_{scenario}.json")
    with open(path) as f:
        return json.load(f)


def fig_route_map(inst, scenario):
    data = _load_routes(scenario)
    routes = data["routes"]

    fig, ax = plt.subplots(figsize=(9, 9))

    _add_zone_overlay(ax)

    hx, hy = zip(*[lonlat_to_webmercator(h["lon"], h["lat"]) for h in HOSPITALS])
    px, py = zip(*[lonlat_to_webmercator(p["lon"], p["lat"]) for p in PATIENTS])

    for v_idx_str, seq in routes.items():
        v_idx = int(v_idx_str)
        color = ROUTE_COLORS[v_idx % len(ROUTE_COLORS)]
        home = int(inst.vehicle_home[v_idx])
        path_x = [hx[home]]
        path_y = [hy[home]]
        for p_idx in seq:
            path_x.append(px[p_idx])
            path_y.append(py[p_idx])
        path_x.append(hx[home])
        path_y.append(hy[home])
        if len(seq) > 0:
            ax.plot(path_x, path_y, color=color, linewidth=2, alpha=0.8,
                    label=f"Vehicle {v_idx} (home H{home}, {len(seq)} stops)", zorder=4)
            ax.annotate("", xy=(path_x[1], path_y[1]), xytext=(path_x[0], path_y[0]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=4)

    for t in (1, 2, 3):
        idxs = [i for i in range(inst.n_patients) if inst.patient_type[i] == t]
        if not idxs:
            continue
        ax.scatter([px[i] for i in idxs], [py[i] for i in idxs], color=TYPE_COLORS[t],
                   s=60, edgecolor="k", linewidth=0.6, zorder=5,
                   label=f"{TYPE_LABELS[t]} patients")
    for i in range(inst.n_patients):
        ax.annotate(str(i), (px[i], py[i]), fontsize=7, ha="center", va="center", zorder=6)

    ax.scatter(hx, hy, marker="*", s=500, color="gold", edgecolor="k", linewidth=1.2,
               label="Hospitals", zorder=6)

    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, crs="EPSG:3857")
    scen_label = "Baseline (normal operations)" if scenario == "baseline" else "Wildfire (Palisades Fire active)"
    ax.set_title(f"LA wildfire case study -- {scen_label}\nbest Q2GA route plan "
                 f"(fitness = {data['best_fitness']:.2f}, "
                 f"{data['objective_components']['n_unserved']:.0f} unserved patients)")
    ax.set_axis_off()
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=1)

    fig.tight_layout()
    fig.savefig(os.path.join(MAPS_DIR, f"route_map_{scenario}.png"), dpi=200)
    plt.close(fig)


def fig_route_map_interactive(inst, scenario):
    data = _load_routes(scenario)
    routes = data["routes"]

    m = folium.Map(location=MAP_CENTER, zoom_start=11, tiles="OpenStreetMap")

    folium.Circle(
        location=list(WILDFIRE_ZONE_CENTER),
        radius=WILDFIRE_ZONE_RADIUS_KM * 1000.0,
        color="orangered", fill=True, fill_opacity=0.12, weight=2, dash_array="6",
        popup="Palisades Fire zone (approx.)",
    ).add_to(m)

    for h in HOSPITALS:
        folium.Marker(
            location=[h["lat"], h["lon"]],
            popup=h["name"],
            icon=folium.Icon(color="orange", icon="plus-sign"),
        ).add_to(m)

    color_map = {1: "red", 2: "orange", 3: "green"}
    for i, p in enumerate(PATIENTS):
        t = int(inst.patient_type[i])
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=7,
            color=color_map[t],
            fill=True,
            fill_color=color_map[t],
            fill_opacity=0.85,
            popup=f"P{i}: {p['name']} ({TYPE_LABELS[t]})",
        ).add_to(m)

    for v_idx_str, seq in routes.items():
        v_idx = int(v_idx_str)
        if not seq:
            continue
        color = ROUTE_COLORS[v_idx % len(ROUTE_COLORS)]
        home = int(inst.vehicle_home[v_idx])
        coords = [(HOSPITALS[home]["lat"], HOSPITALS[home]["lon"])]
        for p_idx in seq:
            coords.append((PATIENTS[p_idx]["lat"], PATIENTS[p_idx]["lon"]))
        coords.append((HOSPITALS[home]["lat"], HOSPITALS[home]["lon"]))
        folium.PolyLine(coords, color=color, weight=3, opacity=0.8,
                        popup=f"Vehicle {v_idx} (home H{home}, {len(seq)} stops)").add_to(m)

    m.save(os.path.join(MAPS_DIR, f"route_map_{scenario}.html"))


# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    inst_base = build_instance(scenario="baseline", seed=42)
    inst_wild = build_instance(scenario="wildfire", seed=42)

    fig_static_map(inst_base)
    fig_interactive_map(inst_base)
    fig_route_map(inst_base, "baseline")
    fig_route_map(inst_wild, "wildfire")
    fig_route_map_interactive(inst_base, "baseline")
    fig_route_map_interactive(inst_wild, "wildfire")

    print("Maps written to", MAPS_DIR)


if __name__ == "__main__":
    main()
