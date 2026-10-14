#!/usr/bin/env python
"""
Regenerate Figures 3 and 4 of the manuscript
"Fuzzy-PSO Optimized DBSCAN for Non-Spatial Clustering" (Seid Mehammed Abdu).

Figure 2  - Clustering results produced by the proposed Fuzzy-PSO DBSCAN
            algorithm: study-area map with the collected data (a) before
            clustering and (b) after the proposed algorithm (GIS view,
            Results and Discussion section).

Figure 3  - Collected data categorized by service type over the study area
            (Kirkos & Bole sub-cities: Meskel Flower and Chechenia areas,
            Addis Ababa). Original base map: QGIS + OpenStreetMap; this script
            fetches OSM tiles with contextily when available.

Figure 4a - The given data after it has been grouped by the proposed
            Fuzzy-PSO DBSCAN algorithm (normalized year of establishment vs
            numeric service type).

Figure 4b - Silhouette and Davies-Bouldin comparison of the four methods,
            using the values reported in Table 4 of the manuscript.

Usage
-----
    python scripts/generate_figures.py                      # synthetic demo data
    python scripts/generate_figures.py --data your.csv      # your dataset
    python scripts/generate_figures.py --no-tiles           # skip OSM tiles

Expected CSV columns: facility_type, year_established, latitude, longitude
Outputs PNG (300 dpi) + PDF into figures/.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing import preprocess_data, z_score_normalize
from algorithms.pso import PSO
from algorithms.fuzzy_dbscan import FuzzyDBSCAN

# ----------------------------------------------------------------------
# Study area (manuscript, Dataset preparation / Results sections)
# ----------------------------------------------------------------------
ADDIS_EXTENT = dict(lat_min=8.90, lat_max=9.15, lon_min=38.65, lon_max=38.85)
STUDY_EXTENT = dict(lat_min=8.995, lat_max=9.045, lon_min=38.730, lon_max=38.775)

SERVICE_COLORS = {
    "education": "#1f77b4",
    "health": "#d62728",
    "entertainment": "#2ca02c",
}

# Table 4 of the manuscript (existing-work comparison)
TABLE4 = pd.DataFrame({
    "Method": ["DBSCAN", "Fuzzy DBSCAN", "PSO-DBSCAN", "Proposed Fuzzy-PSO"],
    "Silhouette": [0.60, 0.69, 0.74, 0.80],
    "Davies-Bouldin": [0.94, 0.82, 0.71, 0.59],
})


def style_publication(rc=True):
    if not rc:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def synthetic_data(n=150, seed=42):
    """
    Purposive-sample-like demo data around Meskel Flower and Chechenia
    (Kirkos) and Bole, Addis Ababa. Replace with the real CSV via --data.
    """
    rng = np.random.default_rng(seed)
    areas = [
        # (lat, lon, lat_sd, lon_sd, n, year_range)
        (9.0153, 38.7440, 0.006, 0.008, n // 3, (2000, 2012)),   # Meskel Flower
        (9.0089, 38.7580, 0.006, 0.008, n // 3, (2008, 2024)),   # Chechenia
        (9.0250, 38.7650, 0.008, 0.010, n - 2 * (n // 3), (2000, 2024)),  # Bole
    ]
    rows = []
    for lat, lon, lat_sd, lon_sd, count, (y0, y1) in areas:
        rows.append({
            "facility_type": rng.choice(list(SERVICE_COLORS), count,
                                        p=[0.45, 0.35, 0.20]),
            "year_established": rng.integers(y0, y1 + 1, count),
            "latitude": rng.normal(lat, lat_sd, count),
            "longitude": rng.normal(lon, lon_sd, count),
        })
    df = pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_data(path):
    if path is None:
        print("No --data given: using synthetic demo data (seeded).")
        return synthetic_data()
    df = pd.read_csv(path)
    required = {"facility_type", "year_established", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    print(f"Loaded {len(df)} records from {path}")
    return df


# ----------------------------------------------------------------------
# Shared geo helpers
# ----------------------------------------------------------------------
def merc(lon, lat):
    """Web-Mercator (EPSG:3857) transform without geopandas."""
    x = np.radians(np.asarray(lon, dtype=float)) * 6378137.0
    y = np.log(np.tan(np.pi / 4 + np.asarray(lat, dtype=float) * np.pi / 360)) \
        * 6378137.0
    return x, y


def draw_basemap(ax, use_tiles=True):
    """
    Try to add a street basemap. OSM volunteer servers block programmatic
    tile use; CARTO requires an API key; Esri raster tiles work without a
    key. Returns (plotted: bool, attribution: str).
    """
    if not use_tiles:
        return False, ""
    try:
        import contextily as ctx
    except Exception as exc:
        print(f"contextily unavailable ({exc}); drawing plain basemap.")
        return False, ""

    providers = [
        getattr(ctx.providers, "Esri").WorldStreetMap,
        getattr(ctx.providers, "Esri").WorldImagery,
        getattr(ctx.providers, "CartoDB").Positron,
    ]
    for provider in providers:
        try:
            ctx.add_basemap(ax, source=provider, attribution=False)
            attribution = provider.get("attribution", "") \
                if isinstance(provider, dict) else ""
            print(f"basemap: {getattr(provider, 'name', provider)}")
            return True, attribution
        except Exception as exc:
            print(f"tile provider failed ({exc}); trying next.")
    print("No tile provider worked; falling back to plain map.")
    return False, ""


def scatter_geo(ax, lon, lat, **kwargs):
    """Scatter lon/lat on a basemap-aware axes (auto Web-Mercator)."""
    if getattr(ax, "_has_basemap", False):
        x, y = merc(lon, lat)
    else:
        x, y = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    return ax.scatter(x, y, **kwargs)


def _style_map_axes(ax, plotted, attribution, title):
    ax.set_title(title)
    if plotted:
        ax.set_xticks([])
        ax.set_yticks([])
        short_attr = attribution.split("--")[0].split("—")[0].strip(" ,")
        ax.text(0.0, -0.04, f"Basemap: {short_attr or 'Esri'}",
                transform=ax.transAxes, fontsize=7, color="gray")
    else:
        ax.set_xlim(STUDY_EXTENT["lon_min"], STUDY_EXTENT["lon_max"])
        ax.set_ylim(STUDY_EXTENT["lat_min"], STUDY_EXTENT["lat_max"])
        ax.set_aspect(1 / np.cos(np.deg2rad(9.03)))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")


def run_proposed(data, n_particles=20, n_iterations=50, seed=42):
    """Z-score normalize non-spatial features and run the proposed pipeline."""
    proc = preprocess_data(data).reset_index(drop=True)
    feat = pd.DataFrame({
        "year_established": proc["year_established"].astype(float),
        "facility_type_encoded":
            proc["facility_type"].astype("category").cat.codes.astype(float),
    })
    norm = z_score_normalize(feat)
    best = PSO(n_particles=n_particles, n_iterations=n_iterations,
               mode="fuzzy", random_state=seed).optimize(norm)
    model = FuzzyDBSCAN(eps=best["eps"], min_pts_min=best["min_pts_min"],
                        min_pts_max=best["min_pts_max"])
    labels, memberships = model.fit_predict(norm)
    return proc, norm, best, labels, memberships


# ----------------------------------------------------------------------
# Figure 2: GIS clustering results of the proposed algorithm
# (Results and Discussion: data before clustering / after clustering)
# ----------------------------------------------------------------------
def figure2(data, out_prefix, use_tiles=True, n_particles=20,
            n_iterations=50, seed=42):
    proc, norm, best, labels, memberships = run_proposed(
        data, n_particles=n_particles, n_iterations=n_iterations, seed=seed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

    for ax, title in ((ax1, "(a) Collected data before clustering"),
                      (ax2, "(b) After the proposed Fuzzy–PSO DBSCAN")):
        # Set the study-area extent BEFORE fetching tiles: contextily infers
        # the requested tile box from the current axes limits.
        if use_tiles:
            x0, y0 = merc([STUDY_EXTENT["lon_min"]], [STUDY_EXTENT["lat_min"]])
            x1, y1 = merc([STUDY_EXTENT["lon_max"]], [STUDY_EXTENT["lat_max"]])
            ax.set_xlim(x0.item(), x1.item())
            ax.set_ylim(y0.item(), y1.item())
            plotted, attribution = draw_basemap(ax, True)
            if not plotted:
                ax.set_xlim(STUDY_EXTENT["lon_min"], STUDY_EXTENT["lon_max"])
                ax.set_ylim(STUDY_EXTENT["lat_min"], STUDY_EXTENT["lat_max"])
        else:
            plotted, attribution = draw_basemap(ax, False)
        ax._has_basemap = plotted
        _style_map_axes(ax, plotted, attribution, title)

    # Panel A: raw points, no clusters yet
    scatter_geo(ax1, proc["longitude"], proc["latitude"],
                s=16, c="#555555", alpha=0.85,
                edgecolors="white", linewidths=0.3, zorder=3)

    # Panel B: same points colored by the proposed clustering
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    cmap = plt.get_cmap("tab10")
    for cid in range(n_clusters):
        mask = labels == cid
        scatter_geo(ax2, proc.loc[mask, "longitude"], proc.loc[mask, "latitude"],
                    s=16, c=[cmap(cid % 10)] * int(mask.sum()), alpha=0.9,
                    edgecolors="white", linewidths=0.3, zorder=3,
                    label=f"Cluster {cid}")
    noise = labels == -1
    if noise.any():
        scatter_geo(ax2, proc.loc[noise, "longitude"], proc.loc[noise, "latitude"],
                    s=20, marker="x", c="black", linewidths=0.8, zorder=2,
                    label="Noise")

    ax2.legend(loc="upper right", framealpha=0.9, title="Proposed clusters")

    fig.suptitle(
        "Figure 2. Clustering results produced by the proposed "
        "Fuzzy–PSO DBSCAN algorithm\n"
        f"(ε={best['eps']:.3f}, MinPtsMin={best['min_pts_min']}, "
        f"MinPtsMax={best['min_pts_max']}, Silhouette={best['fitness']:.3f})",
        fontsize=11, y=0.99)
    fig.tight_layout()
    save(fig, out_prefix + "figure2_clustering_results")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3: map of collected data by service type
# ----------------------------------------------------------------------
def figure3(data, out_prefix, use_tiles=True):
    has_ctx = False
    if use_tiles:
        try:
            import contextily as ctx
            has_ctx = True
        except Exception as exc:
            print(f"contextily unavailable ({exc}); drawing plain basemap.")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5.0), gridspec_kw={"width_ratios": [1, 1.6]})

    # --- Panel A: overall Addis Ababa with study-area rectangle ---------
    for stype, color in SERVICE_COLORS.items():
        sub = data[data["facility_type"] == stype]
        if len(sub):
            ax1.scatter(sub["longitude"], sub["latitude"], s=4, c=color,
                        alpha=0.7, zorder=2)
    ax1.add_patch(Rectangle(
        (STUDY_EXTENT["lon_min"], STUDY_EXTENT["lat_min"]),
        STUDY_EXTENT["lon_max"] - STUDY_EXTENT["lon_min"],
        STUDY_EXTENT["lat_max"] - STUDY_EXTENT["lat_min"],
        fill=False, edgecolor="black", lw=1.4))
    ax1.set_xlim(ADDIS_EXTENT["lon_min"], ADDIS_EXTENT["lon_max"])
    ax1.set_ylim(ADDIS_EXTENT["lat_min"], ADDIS_EXTENT["lat_max"])
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("(a) Addis Ababa — study area")
    ax1.set_aspect(1 / np.cos(np.deg2rad(9.03)))
    ax1.text(0.5, 0.02, "Study area", transform=ax1.transAxes,
             ha="center", fontsize=8, style="italic")

    # --- Panel B: zoomed study area with points by service type --------
    plotted = False
    if has_ctx:
        # OSM volunteer servers block programmatic tile use; CARTO now
        # requires an API key (watermarked tiles). Esri raster tiles work
        # without a key and look clean in print.
        providers = [
            getattr(ctx.providers, "Esri").WorldStreetMap,
            getattr(ctx.providers, "Esri").WorldImagery,
            getattr(ctx.providers, "CartoDB").Positron,
        ]
        attribution = ""
        for provider in providers:
            try:
                # Web-Mercator transform (EPSG:3857) without geopandas
                def merc(lon, lat):
                    x = np.radians(lon) * 6378137.0
                    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * 6378137.0
                    return x, y

                for stype, color in SERVICE_COLORS.items():
                    sub = data[data["facility_type"] == stype]
                    if len(sub):
                        x, y = merc(sub["longitude"].values,
                                    sub["latitude"].values)
                        ax2.scatter(x, y, s=18, c=color, alpha=0.9,
                                    edgecolors="white", linewidths=0.3, zorder=3)
                ctx.add_basemap(ax2, source=provider, attribution=False)
                plotted = True
                attribution = provider.get("attribution", "") \
                    if isinstance(provider, dict) else ""
                print(f"basemap: {provider}")
                break
            except Exception as exc:
                print(f"tile provider {provider} failed ({exc}); trying next.")
                ax2.cla()
        if not plotted:
            print("No tile provider worked; falling back to plain map.")
            has_ctx = False

    if not plotted:
        for stype, color in SERVICE_COLORS.items():
            sub = data[data["facility_type"] == stype]
            if len(sub):
                ax2.scatter(sub["longitude"], sub["latitude"], s=18, c=color,
                            alpha=0.85, edgecolors="white", linewidths=0.3,
                            zorder=3)
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")

    if has_ctx and plotted:
        # axes are in Web-Mercator metres; hide graticule, add attribution
        ax2.set_title("(b) Study area (Meskel Flower & Chechenia, Kirkos/Bole)")
        ax2.set_xticks([])
        ax2.set_yticks([])
        # keep only the lead source of the attribution string
        short_attr = attribution.split("--")[0].split("—")[0].strip(" ,")
        ax2.text(0.0, -0.04, f"Basemap: {short_attr or 'Esri'}",
                 transform=ax2.transAxes, fontsize=7, color="gray")
    else:
        ax2.set_xlim(STUDY_EXTENT["lon_min"], STUDY_EXTENT["lon_max"])
        ax2.set_ylim(STUDY_EXTENT["lat_min"], STUDY_EXTENT["lat_max"])
        ax2.set_aspect(1 / np.cos(np.deg2rad(9.03)))
        ax2.set_title("(b) Study area (Meskel Flower & Chechenia, Kirkos/Bole)")

    handles = [Line2D([], [], marker="o", linestyle="", markersize=6,
                      color=c, label=s.title()) for s, c in SERVICE_COLORS.items()]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=6,
                          color="gray", label="Noise / other"))
    ax2.legend(handles=handles, title="Service type", loc="upper right",
               framealpha=0.9)

    fig.suptitle("Figure 3. Collected data categorized by service type",
                 fontsize=12, y=0.99)
    fig.tight_layout()
    save(fig, out_prefix + "figure3_map_by_service_type")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 4a: data grouped by the proposed Fuzzy-PSO DBSCAN
# ----------------------------------------------------------------------
def figure4a(data, out_prefix, n_particles=20, n_iterations=50, seed=42):
    proc, norm, best, labels, memberships = run_proposed(
        data, n_particles=n_particles, n_iterations=n_iterations, seed=seed)
    feat_types = proc["facility_type"].astype("category")
    type_names = list(feat_types.cat.categories)
    codes = feat_types.cat.codes.astype(float).values

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    cmap = plt.get_cmap("tab10")
    for cid in range(n_clusters):
        mask = labels == cid
        ax.scatter(norm[mask, 0], norm[mask, 1], s=28, color=cmap(cid % 10),
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"Cluster {cid}")
    noise = labels == -1
    if noise.any():
        ax.scatter(norm[noise, 0], norm[noise, 1], s=28, marker="x",
                   color="black", linewidths=0.8, zorder=2, label="Noise")

    # y-axis ticks back at the service-type categories
    uniq = np.unique(codes)
    ax.set_yticks([norm[codes == c, 1][0] for c in uniq])
    ax.set_yticklabels([type_names[int(c)].title() for c in uniq])

    ax.set_xlabel("Normalized year of establishment")
    ax.set_ylabel("Service type (normalized)")
    ax.set_title(
        f"Figure 4a. Data grouped by the proposed Fuzzy–PSO DBSCAN\n"
        f"(ε={best['eps']:.3f}, MinPtsMin={best['min_pts_min']}, "
        f"MinPtsMax={best['min_pts_max']}, Silhouette={best['fitness']:.3f})")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    save(fig, out_prefix + "figure4a_grouped_by_proposed")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 4b: metric comparison bars (Table 4)
# ----------------------------------------------------------------------
def figure4b(out_prefix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    colors = ["#9ecae1", "#6baed6", "#4292c6", "#08519c"]

    ax1.bar(TABLE4["Method"], TABLE4["Silhouette"], color=colors)
    for i, v in enumerate(TABLE4["Silhouette"]):
        ax1.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax1.set_ylim(0, 0.95)
    ax1.set_ylabel("Silhouette coefficient  (higher is better →)")
    ax1.set_title("(a) Silhouette coefficient")

    ax2.bar(TABLE4["Method"], TABLE4["Davies-Bouldin"], color=colors)
    for i, v in enumerate(TABLE4["Davies-Bouldin"]):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Davies–Bouldin index  (← lower is better)")
    ax2.set_title("(b) Davies–Bouldin index")

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Figure 4b. Clustering quality comparison (Table 4 values)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, out_prefix + "figure4b_metrics_comparison")
    plt.close(fig)


def save(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(OUTDIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None,
                        help="CSV with facility_type, year_established, "
                             "latitude, longitude (default: synthetic demo)")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--no-tiles", action="store_true",
                        help="skip basemap tile download for Figures 2 and 3")
    parser.add_argument("--figs", default="2,3,4a,4b",
                        help="comma-separated figure ids to generate "
                             "(default: 2,3,4a,4b)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global OUTDIR
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    style_publication()
    data = load_data(args.data)
    want = {f.strip() for f in args.figs.split(",")}

    if "2" in want:
        figure2(data, out_prefix="", use_tiles=not args.no_tiles, seed=args.seed)
    if "3" in want:
        figure3(data, out_prefix="", use_tiles=not args.no_tiles)
    if "4a" in want:
        figure4a(data, out_prefix="", seed=args.seed)
    if "4b" in want:
        figure4b(out_prefix="")
    print("Done.")


if __name__ == "__main__":
    main()
