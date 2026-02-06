# -*- coding: utf-8 -*-
"""
SHAKEpropagate: Spatio-temporal wave propagation and arrival-time modeling on ShakeMap grids
==========================================================================================

`SHAKEpropagate` provides a lightweight, research-oriented framework to model **wavefront
propagation** and **arrival times** over a ShakeMap spatial domain. It is designed to support
EEW-style lead-time analysis, time-dependent SHAKEmaps workflows, and validation against
observed station arrivals.

Core capabilities
-----------------
1) **Input parsing**
   - Reads a USGS ShakeMap XML grid (e.g., `grid.xml`) and an optional rupture GeoJSON.
   - Extracts event metadata (including origin time, if present) and the spatial grid definition.

2) **Speed / slowness field construction**
   - Builds a 2-D effective wave-speed map using:
     - uniform speed,
     - VS30-based mappings (linear, logarithmic, piecewise classes),
     - two-layer effective slowness mixing,
     - dispersion-like baseline × class-factor mapping,
     - or user-specified custom mappings.

3) **Travel-time computation**
   - Computes travel-time fields T(x, y) using:
     - Dijkstra (8-connected graph search)
     - FMM (Fast Marching Method; 4-neighbor upwind, optional)

4) **Outputs**
   - Exports figures and artifacts such as:
     - travel-time maps and contours,
     - MMI + travel-time contour overlays,
     - time-lapse frames,
     - CSV tables and NumPy arrays,
     - (optional) augmented grid products for downstream workflows.

5) **Output organization**
   - Outputs are written under:
       `export/SHAKEpropagate/<event_id>/<scenario_name>/`

Notes
-----
- Folder naming follows the project convention: **"SHAKEpropagate"**.
- Cartopy is optional. If unavailable, plotting falls back to plain Matplotlib.
- Speed models are intentionally flexible to support calibration studies
  (e.g., global scaling λ, rupture seeding vs point seeding, class-speed tuning).

Quick single-scenario example (copy/paste)
------------------------------------------
from modules.SHAKEpropagate import SHAKEpropagate, Inputs

XML = "./event_data/SHAKEfetch/usgs-shakemap-versions/us7000pn9s/us7000pn9s_us_022_grid.xml"
RUP = "./event_data/SHAKEfetch/usgs-rupture-versions/us7000pn9s/us7000pn9s_us_022_rupture.json"

sim = SHAKEpropagate(Inputs(shakemap_xml=XML, rupture_file=RUP))

# Scenario name becomes the folder under export/SHAKEpropagate/<eventid>/...
CASE = "dijkstra_surface_vs30_piecewise_fault"

result = sim.run_scenario(
    case_name=CASE,
    seed_from="rupture",                 # "epicenter" | "rupture" | "auto"
    make_frames=False,                   # True to render time-lapse frames
    export_all=True,                     # save ALL artifacts (figures + frames + tables + arrays)
    probe_points=[(96.10, 21.98, "Mandalay"), (100.50, 13.75, "Bangkok")],
    overrides={
        # Solver
        "mode": "dijkstra",              # "dijkstra" | "fmm" | "ml"(NotImplemented)

        # Speed model family
        "speed_model": "vs30_piecewise", # "uniform"|"vs30_linear"|"vs30_log"|
                                         # "vs30_piecewise"|"two_layer"|"dispersion_period"|"custom"

        # Wave type (used by some mappings to pick bands)
        "wave_type": "Surface",          # "S" | "Surface"

        # Uniform (only if speed_model="uniform")
        "uniform_c_km_s": 3.2,

        # Global λ_c scale (applied after model mapping)
        "speed_scale_lambda": 1.0,

        # VS30 linear / log bands (only if "vs30_linear"/"vs30_log")
        "vs30_min": 150.0, "vs30_max": 1000.0,
        "c_min_s": 3.2, "c_max_s": 4.0,
        "c_min_surface": 2.4, "c_max_surface": 3.4,

        # Piecewise (only if "vs30_piecewise")
        "piecewise_breaks": (180.0, 360.0, 760.0),
        "surface_class_speeds": (2.4, 2.8, 3.1, 3.4),
        "s_class_speeds": (2.2, 3.0, 3.6, 4.1),

        # Two-layer (only if "two_layer")
        "two_layer_w_surface": 0.10,
        "two_layer_c_crust": 3.5,
        "shallow_v_floor": 0.6,

        # Dispersion-like (only if "dispersion_period")
        "dispersion_period_s": 5.0,
        "dispersion_baseline_c_km_s": 3.0,
        "dispersion_class_factors": (0.92, 0.97, 1.00, 1.04),

        # Rupture & timing
        "source_mode": "rupture_edges",  # "point"|"rupture_edges"|"rupture_fill"
        "seed_radius_km": 5.0,
        "densify_factor": 1.0,
        "use_fault_timing": False,       # if True: t0 = along-rupture distance / Vr
        "rupture_velocity_km_s": 2.8,

        # Magnitude-dependent Vr (optional)
        "use_vr_from_mag": False,        # if True, override Vr via M-dependent formula
        "vr_a": 2.7, "vr_b": 0.2, "vr_min": 2.4, "vr_max": 3.4,

        # Figure controls
        "dpi": 160,
        "frame_figsize": (9.5, 7.6),
        "tmap_figsize": (10.0, 8.0),
        "mmi_contours_figsize": (10.5, 8.0),

        # Logging
        "log_level": "INFO",
    }
)

print("Outputs →", sim.outputs.out_dir)
print("Summary:", result["summary"])

Date:
    January, 2026
Version:
    26.4

    
"""




from __future__ import annotations
import os, math, json, warnings, logging, csv
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Literal, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Optional GIS stack
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.patches as mpatches
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False
    warnings.warn("Cartopy not available, figures will use plain Matplotlib.")

# XML
import xml.etree.ElementTree as ET

# ------------------------------ logging setup --------------------------------
logger = logging.getLogger("SHAKEpropagate")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ----------------------------- small utilities -------------------------------
def _ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist; return path."""
    os.makedirs(path, exist_ok=True)
    return path

def _haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Vectorized haversine distance (km). Supports array broadcasting."""
    R = 6371.0
    lon1r = np.radians(np.asarray(lon1))
    lat1r = np.radians(np.asarray(lat1))
    lon2r = np.radians(np.asarray(lon2))
    lat2r = np.radians(np.asarray(lat2))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Public alias to the module's haversine (km)."""
    return _haversine_km(lon1, lat1, lon2, lat2)

def _nearest_index(lon_grid, lat_grid, lon, lat) -> Tuple[int, int]:
    """Return (i,j) index of the grid cell whose center is closest to (lon,lat)."""
    dist = _haversine_km(lon_grid, lat_grid, lon, lat)
    return np.unravel_index(np.nanargmin(dist), lon_grid.shape)

def _lonlat_to_xy_km(lon_grid, lat_grid, lon0, lat0) -> Tuple[np.ndarray, np.ndarray]:
    """Local equirectangular metric (km) about (lon0,lat0) for step lengths."""
    lat0r = np.radians(lat0)
    x = (lon_grid - lon0) * np.cos(lat0r) * (np.pi/180) * 6371.0
    y = (lat_grid - lat0) * (np.pi/180) * 6371.0
    return x, y

# ------------------------------- USGS MMI styling ----------------------------
def _usgs_mmi_cmap_norm():
    """USGS MMI discrete palette and class bounds (cmap, norm, ticks, label)."""
    usgs_colors = [
        (1.00, 1.00, 1.00, 1.0),
        (0.75, 0.80, 1.00, 1.0),
        (0.63, 0.90, 1.00, 1.0),
        (0.50, 1.00, 1.00, 1.0),
        (0.48, 1.00, 0.58, 1.0),
        (1.00, 1.00, 0.00, 1.0),
        (1.00, 0.78, 0.00, 1.0),
        (1.00, 0.57, 0.00, 1.0),
        (1.00, 0.00, 0.00, 1.0),
        (0.78, 0.00, 0.00, 1.0),
        (0.50, 0.00, 0.00, 1.0)
    ]
    bounds = [0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10]
    cmap = mpl.colors.ListedColormap(usgs_colors)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = bounds
    label = "Intensity (MMI)"
    return cmap, norm, ticks, label

def _usgs_basemap(ax, extent, label_size=12):
    """Cartopy basemap similar to SHAKEmapper."""
    if not _HAS_CARTOPY:
        return
    ax.coastlines(zorder=10)
    ax.add_feature(cfeature.BORDERS, zorder=10, linestyle='-')
    ax.add_feature(cfeature.OCEAN, zorder=9, facecolor='skyblue')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1.2, color='gray', alpha=0.6, linestyle='--', zorder=999)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": label_size}
    gl.ylabel_style = {"size": label_size}
    frame = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               fill=False, edgecolor='black', linewidth=0.8, zorder=100)
    ax.add_patch(frame)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

# --------------------------------- dataclasses -------------------------------
@dataclass
class EventMeta:
    """Metadata for the earthquake event parsed from XML."""
    event_id: str
    epicenter_lon: float
    epicenter_lat: float
    event_time_str: str
    depth_km: float
    magnitude: float
    description: str

@dataclass
class Inputs:
    """File inputs and optional externally supplied VS30 grid."""
    shakemap_xml: str
    rupture_file: Optional[str] = None
    vs30_grid: Optional[np.ndarray] = None  # overrides VS30/SVEL from XML if provided

@dataclass
class Settings:
    """
    Tunable settings for solvers, speed models, plotting, and rupture seeding.
    Use `run_scenario(overrides={...})` to set any field at runtime.
    """
    # Solver
    mode: Literal["dijkstra","fmm","ml"] = "dijkstra"

    # Speed models
    speed_model: Literal[
        "uniform", "vs30_linear", "vs30_log", "vs30_piecewise",
        "two_layer", "dispersion_period", "custom"
    ] = "vs30_linear"
    uniform_c_km_s: float = 3.2

    # Global velocity scale factor λc (applied after building c and then clamped)
    speed_scale_lambda: float = 1.0

    # VS30 banded mapping
    wave_type: Literal["S", "Surface"] = "S"
    vs30_min: float = 150.0
    vs30_max: float = 1000.0
    c_min_s: float = 3.2
    c_max_s: float = 4.0
    c_min_surface: float = 2.4
    c_max_surface: float = 3.4

    # Piecewise class breaks & speeds
    piecewise_breaks: Tuple[float,float,float] = (180.0, 360.0, 800)  # E/D/C/B+
    surface_class_speeds: Tuple[float,float,float,float] = (2.4, 2.8, 3.1, 3.4)
    s_class_speeds: Tuple[float,float,float,float] = (2.2, 3.0, 3.6, 4.1)

    # Two-layer effective slowness
    two_layer_w_surface: float = 0.10
    two_layer_c_crust: float = 3.5
    shallow_v_floor: float = 0.6

    # Dispersion-like
    dispersion_period_s: float = 5.0
    dispersion_baseline_c_km_s: float = 3.0
    dispersion_class_factors: Tuple[float,float,float,float] = (0.92, 0.97, 1.00, 1.04)

    # Solver / frames
    soften_band_s: float = 1.0
    frame_dt_s: float = 1.0
    frame_times_s: Optional[List[float]] = None

    # Plotting & figure control (user-overridable externally)
    im_label: str = "MMI"
    title_prefix: str = "ShakeMap propagation"
    dpi: int = 140
    use_cartopy: bool = True
    coastline: bool = True
    borders: bool = True
    gridlines: bool = False
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # User-controllable figure sizes (inches). If None, use defaults.
    frame_figsize: Optional[Tuple[float, float]] = None
    tmap_figsize: Optional[Tuple[float, float]] = None
    mmi_contours_figsize: Optional[Tuple[float, float]] = None

    # Rupture seeding & timing
    source_mode: Literal["point", "rupture_edges", "rupture_fill"] = "rupture_edges"
    seed_radius_km: float = 5.0
    densify_factor: float = 1.0
    use_fault_timing: bool = False
    rupture_velocity_km_s: float = 2.8

    # Magnitude-dependent rupture velocity parameters
    use_vr_from_mag: bool = False
    vr_a: float = 2.7
    vr_b: float = 0.2
    vr_min: float = 2.4
    vr_max: float = 3.4

    # Diagnostics
    log_level: Literal["WARNING","INFO","DEBUG"] = "INFO"

@dataclass
class Outputs:
    """Output directories and export toggles for arrays (npy/csv)."""
    out_dir: str = "./export"
    frames_dir: str = field(default_factory=lambda: "./export/SHAKEpropagate")
    save_tmap_npy: bool = True
    save_tmap_csv: bool = False
    save_speed_npy: bool = False
    save_speed_csv: bool = False




# ============================ STATION/VALIDATION HELPERS ======================
# (Module-scope functions; safe to import from notebooks.)

# Column name guesses for station CSV parsing
_COL_GUESS = {
    "lon": ["lon","longitude","station_lon","x","Long","LONG","Lon","Longitude"],
    "lat": ["lat","latitude","station_lat","y","Lat","LAT","Latitude"],
    "id":  ["id","station","code","sta","name","Station","STATION","net_sta","NET_STA"],
    "t_obs": [
        "delta_s","Delta_s","delta","Delta",
        "travel_time_s","arrival_seconds","time_to_arrival_s","t_obs","Tobs",
        "Delta_of_Arrival_Time_s","Delta_of_arrival_time_s","delta_of_arrival_time_s"
    ],
    "arrival_ts": ["arrival_utc","arrival_time_utc","arrival_time","t_arrival_utc","ArrivalUTC","ARRIVAL_UTC"],
    "origin_ts":  ["origin_utc","origin_time_utc","t0_utc","OriginUTC","ORIGIN_UTC","event_time_utc"]
}

def _pick_col(df, explicit, candidates):
    """Pick a column from a DataFrame by explicit name or case-insensitive candidates."""
    if explicit and explicit in df.columns:
        return explicit
    if explicit:
        for c in df.columns:
            if c.lower() == str(explicit).lower():
                return c
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

def _parse_iso_any(s):
    """Robust ISO/timestamp parser → datetime (timezone-aware if possible) or None."""
    import pandas as pd, datetime as dt, numpy as _np
    if s is None or (isinstance(s, float) and _np.isnan(s)):
        return None
    txt = str(s).strip().replace("Z","+00:00")
    try:
        return dt.datetime.fromisoformat(txt)
    except Exception:
        try:
            return pd.to_datetime(txt, utc=True).to_pydatetime()
        except Exception:
            return None

def _read_origin_from_xml(xml_path: str):
    """Load event origin time from ShakeMap XML via the module loader."""
    sim_tmp = SHAKEpropagate(Inputs(shakemap_xml=xml_path))
    sim_tmp.load_from_xml()
    tstr = sim_tmp.event.event_time_str
    T0 = _parse_iso_any(tstr)
    if T0 is None:
        raise ValueError(f"Could not parse event_time_str from XML: {tstr}")
    return T0

def load_station_table(path: str,
                       lon_col: str = None,
                       lat_col: str = None,
                       id_col: str = None,
                       phase_col: str = None,
                       phase_filter: str = None,
                       t_obs_col: str = None,
                       origin_time_utc=None,
                       xml_path_for_origin: str = None):
    """
    Read a station CSV and return a DataFrame with columns:
        ['station_id','lon','lat','t_obs_s'].

    Parameters
    ----------
    path : str
        CSV file path.
    lon_col, lat_col, id_col : str or None
        Explicit column names (case-insensitive). If None, auto-detect.
    phase_col : str or None
        Column that contains phase names (e.g., P/S). Used if phase_filter is set.
    phase_filter : str or None
        If provided, keep only rows whose phase matches this (case-insensitive).
    t_obs_col : str or None
        If provided, interpreted directly as observed travel time (seconds).
        Otherwise derived from arrival and origin timestamps.
    origin_time_utc : None | datetime | str
        - None: expect an origin time column per-row or supply xml_path_for_origin.
        - "from_xml": read origin from xml_path_for_origin (required).
        - str/datetime: a fixed origin time applied to all rows.
    xml_path_for_origin : str or None
        Used only when origin_time_utc == "from_xml".

    Returns
    -------
    pandas.DataFrame
        Columns: station_id (str), lon (float), lat (float), t_obs_s (float)
    """
    import pandas as pd
    import numpy as _np

    df = pd.read_csv(path)

    # Optional phase filter
    if phase_filter is not None:
        guess_phase = phase_col
        if guess_phase is None:
            for c in ["phase","Phase","PHASE","pick_phase","PickPhase","PH"]:
                if c in df.columns:
                    guess_phase = c
                    break
        if guess_phase and guess_phase in df.columns:
            df = df[df[guess_phase].astype(str).str.upper() == str(phase_filter).upper()].copy()

    col_lon = _pick_col(df, lon_col, _COL_GUESS["lon"])
    col_lat = _pick_col(df, lat_col, _COL_GUESS["lat"])
    col_id  = _pick_col(df, id_col,  _COL_GUESS["id"])
    col_tob = _pick_col(df, t_obs_col, _COL_GUESS["t_obs"])
    col_arr = _pick_col(df, None, _COL_GUESS["arrival_ts"])
    col_org = _pick_col(df, None, _COL_GUESS["origin_ts"])

    if not col_lon or not col_lat:
        raise ValueError(f"Could not find lon/lat columns. Columns={list(df.columns)}")
    if not col_id:
        df["__id__"] = [f"sta_{i+1}" for i in range(len(df))]
        col_id = "__id__"

    # Resolve travel time column or compute from timestamps
    if col_tob is not None:
        t_obs_s = pd.to_numeric(df[col_tob], errors="coerce")
    else:
        if col_arr is None:
            raise ValueError("No travel-time column and no arrival timestamp column found.")
        arr_ts = df[col_arr].apply(_parse_iso_any)

        if col_org is not None:
            org_ts = df[col_org].apply(_parse_iso_any)
        else:
            # Single origin time for all rows
            if isinstance(origin_time_utc, str) and origin_time_utc.strip().lower() == "from_xml":
                if not xml_path_for_origin:
                    raise ValueError("origin_time_utc='from_xml' requires xml_path_for_origin.")
                T0 = _read_origin_from_xml(xml_path_for_origin)
            elif hasattr(origin_time_utc, "isoformat"):
                T0 = origin_time_utc
            else:
                T0 = _parse_iso_any(origin_time_utc)
            if T0 is None:
                raise ValueError("No per-row origin and ORIGIN_TIME_UTC not provided/parsable.")
            org_ts = pd.Series([T0]*len(df))

        t_obs_s = (arr_ts - org_ts).dt.total_seconds()

    out = pd.DataFrame({
        "station_id": df[col_id].astype(str).values,
        "lon": pd.to_numeric(df[col_lon], errors="coerce").values,
        "lat": pd.to_numeric(df[col_lat], errors="coerce").values,
        "t_obs_s": pd.to_numeric(t_obs_s, errors="coerce").values
    }).replace([_np.inf,-_np.inf], _np.nan).dropna(subset=["lon","lat","t_obs_s"])
    out = out[out["t_obs_s"] >= 0].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("After cleaning, no valid station rows remain.")

    # Friendly print for debugging (module logger)
    try:
        logger.info(f"[station] detected columns: "
                    f"{{'id': '{col_id}', 'lon': '{col_lon}', 'lat': '{col_lat}', "
                    f"'t_obs': '{col_tob}', 'arrival_ts': '{col_arr}', 'origin_ts': '{col_org}'}}")
        logger.info(f"[station] loaded rows: {len(out)}")
    except Exception:
        pass

    return out

def plot_mmi_contours_with_stations(sim: "SHAKEpropagate",
                                    stations_df,
                                    fname,
                                    dpi=160,
                                    figsize=(10.5, 8.0),
                                    annotate=True,
                                    label_mode="time",  # "id_time" | "time"
                                    label_fmt_id_time="{id} ({t:.0f}s)",
                                    label_fmt_time="{t:.0f}s",
                                    contour_interval_s=10.0):
    """
    Draw source MMI with arrival-time contours and overlay station markers/labels.

    Parameters
    ----------
    sim : SHAKEpropagate
        A simulation instance that already has T_map_s computed.
    stations_df : pandas.DataFrame
        Must contain columns: station_id, lon, lat, t_obs_s
    label_mode : {"id_time","time"}
        "id_time" shows Station ID + observed time; "time" shows only observed time.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    if sim.T_map_s is None:
        raise RuntimeError("sim.T_map_s is None. Run compute_travel_time_field() first.")

    # USGS palette from module helper
    cmap, norm, _ticks, _label = _usgs_mmi_cmap_norm()

    # contour levels from T
    T = sim.T_map_s
    tmax = float(np.nanpercentile(T, 98))
    step = float(contour_interval_s)
    levels_s = list(np.arange(0, max(step, tmax + 0.5*step), step))

    # Extract station arrays
    xs = stations_df["lon"].to_numpy()
    ys = stations_df["lat"].to_numpy()
    ids = stations_df["station_id"].astype(str).to_numpy()
    ts  = stations_df["t_obs_s"].to_numpy()

    # Cartopy optional
    use_cartopy = False
    if getattr(sim.settings, "use_cartopy", True) and _HAS_CARTOPY:
        use_cartopy = True

    if use_cartopy:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=proj)

        extent = [float(sim.lon_grid.min()), float(sim.lon_grid.max()),
                  float(sim.lat_grid.min()), float(sim.lat_grid.max())]
        _usgs_basemap(ax, extent, label_size=12)

        im = ax.pcolormesh(sim.lon_grid, sim.lat_grid, sim.im_grid, transform=proj,
                           cmap=cmap, norm=norm, shading="auto", zorder=5)
        cs = ax.contour(sim.lon_grid, sim.lat_grid, T, levels=levels_s,
                        colors="k", linewidths=0.9, alpha=0.95, transform=proj, zorder=15)
        ax.clabel(cs, fmt="%d s", inline=True, inline_spacing=1)

        ax.plot(sim.event.epicenter_lon, sim.event.epicenter_lat,
                marker="*", color="white", markersize=9, transform=proj, zorder=20)

        ax.scatter(xs, ys, s=46, marker="^", c="red", edgecolors="black",
                   linewidths=0.6, transform=proj, zorder=30)

        if annotate:
            if label_mode == "id_time":
                for x, y, sid, t in zip(xs, ys, ids, ts):
                    ax.text(x, y, " " + label_fmt_id_time.format(id=sid, t=t),
                            transform=proj, fontsize=8, color="k", zorder=35)
            else:
                for x, y, t in zip(xs, ys, ts):
                    ax.text(x, y, " " + label_fmt_time.format(t=t),
                            transform=proj, fontsize=8, color="k", zorder=35)

        ax.set_title("MMI with arrival-time contours (station overlays)")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Intensity (MMI)")
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.pcolormesh(sim.lon_grid, sim.lat_grid, sim.im_grid, cmap=cmap, norm=norm, shading="auto", zorder=5)
        cs = ax.contour(sim.lon_grid, sim.lat_grid, T, levels=levels_s, colors="k",
                        linewidths=0.9, alpha=0.95, zorder=15)
        ax.clabel(cs, fmt="%d s", inline=True, inline_spacing=1)
        ax.plot(sim.event.epicenter_lon, sim.event.epicenter_lat, marker="*", color="white", markersize=9, zorder=20)
        ax.scatter(xs, ys, s=46, marker="^", c="red", edgecolors="black", linewidths=0.6, zorder=30)

        if annotate:
            if label_mode == "id_time":
                for x, y, sid, t in zip(xs, ys, ids, ts):
                    ax.text(x, y, " " + label_fmt_id_time.format(id=sid, t=t),
                            fontsize=8, color="k", zorder=35)
            else:
                for x, y, t in zip(xs, ys, ts):
                    ax.text(x, y, " " + label_fmt_time.format(t=t),
                            fontsize=8, color="k", zorder=35)

        ax.set_title("MMI with arrival-time contours (station overlays)")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Intensity (MMI)")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    fig.savefig(fname, bbox_inches="tight"); plt.close(fig)
    return fname

# Public export list (optional)
__all__ = [
    "EventMeta", "Inputs", "Settings", "Outputs", "SHAKEpropagate",
    "load_station_table", "plot_mmi_contours_with_stations",
    "haversine_km"
]





# --------------------------------- main class --------------------------------
class SHAKEpropagate:
    """
    SHAKEpropagate main API.

    Workflow:
        - load_from_xml(): parse event & grids (MMI, lon/lat, VS30/SVEL)
        - set_source_from_rupture() or _build_point_source(): seed origin cells
        - compute_speed_map(): build c(x,y)
        - compute_travel_time_field(): Dijkstra or FMM
        - export_all_artifacts(): figures, frames, CSVs/XML, arrays

    High-level helper:
        - run_scenario(): one-shot orchestrator that also rebases exports to
          export/SHAKEpropagate/<eventid>/<scenario_name>/
    """

    def __init__(self, inputs: Inputs, settings: Settings = Settings(), outputs: Outputs = Outputs()):
        self.inputs = inputs
        self.settings = settings
        self.outputs = outputs

        logger.setLevel(getattr(logging, self.settings.log_level, logging.INFO))
        _ensure_dir(self.outputs.out_dir)
        _ensure_dir(self.outputs.frames_dir)

        # Filled later
        self.event: Optional[EventMeta] = None
        self.lon_grid: Optional[np.ndarray] = None
        self.lat_grid: Optional[np.ndarray] = None
        self.im_grid: Optional[np.ndarray]  = None
        self.vs30_grid: Optional[np.ndarray] = None
        self.c_map: Optional[np.ndarray] = None
        self.T_map_s: Optional[np.ndarray] = None
        self.source_mask: Optional[np.ndarray] = None
        self.T0_seed: Optional[np.ndarray] = None

        # Metric caches
        self._xkm: Optional[np.ndarray] = None
        self._ykm: Optional[np.ndarray] = None

    # ------------------------------ XML parser -------------------------------
    def load_from_xml(self):
        """Parse ShakeMap XML into event metadata and lon/lat/MMI/VS30 grids."""
        xml_path = self.inputs.shakemap_xml
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"ShakeMap XML not found: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()
        if "}" in root.tag:
            ns_uri = root.tag.split("}")[0].strip("{")
            ns = {"sm": ns_uri}
        else:
            ns = {"sm": ""}

        # Event
        ev = root.find("sm:event", ns)
        if ev is None:
            raise ValueError("Could not find <event> in ShakeMap XML.")

        self.event = EventMeta(
            event_id      = ev.get("event_id", ""),
            epicenter_lon = float(ev.get("lon")),
            epicenter_lat = float(ev.get("lat")),
            event_time_str= ev.get("event_timestamp", ""),
            depth_km      = float(ev.get("depth")),
            magnitude     = float(ev.get("magnitude")),
            description   = ev.get("event_description", "")
        )
        logger.info(f"Loaded event {self.event.event_id} M{self.event.magnitude:.1f} "
                    f"({self.event.epicenter_lat:.3f},{self.event.epicenter_lon:.3f})")

        # Grid spec
        gs = root.find("sm:grid_specification", ns)
        if gs is None:
            raise ValueError("Could not find <grid_specification> in ShakeMap XML.")
        nlon = int(float(gs.get("nlon"))); nlat = int(float(gs.get("nlat")))

        # Fields
        fields = root.findall("sm:grid_field", ns)
        if not fields:
            raise ValueError("Could not find any <grid_field> in ShakeMap XML.")
        field_by_index = {int(f.get("index")): f.get("name").upper() for f in fields}

        # Grid data
        gd = root.find("sm:grid_data", ns)
        if gd is None or gd.text is None:
            raise ValueError("Could not find <grid_data> text in ShakeMap XML.")
        flat = np.fromstring(gd.text.strip(), sep=" ")
        nfields = len(field_by_index)
        if flat.size % nfields != 0:
            raise ValueError(f"grid_data size {flat.size} not divisible by number of fields {nfields}.")
        npts = flat.size // nfields
        arr = flat.reshape((npts, nfields))

        name_to_col = {field_by_index[i+1].upper(): i for i in range(nfields)}
        def _col(cands: List[str]) -> Optional[int]:
            for cand in cands:
                if cand.upper() in name_to_col:
                    return name_to_col[cand.upper()]
            return None

        ix_lon  = _col(["LON","LONGITUDE","X"])
        ix_lat  = _col(["LAT","LATITUDE","Y"])
        ix_mmi  = _col(["MMI","INTENSITY"])
        ix_vs30 = _col(["VS30","SVEL"])  # many ShakeMaps publish SVEL (m/s)

        if None in (ix_lon, ix_lat, ix_mmi):
            raise ValueError(f"Required fields not found. Present: {list(name_to_col.keys())}")

        lon_grid = arr[:, ix_lon].reshape(nlat, nlon)
        lat_grid = arr[:, ix_lat].reshape(nlat, nlon)
        mmi_grid = arr[:, ix_mmi].reshape(nlat, nlon)

        self.lon_grid = lon_grid
        self.lat_grid = lat_grid
        self.im_grid  = mmi_grid

        if ix_vs30 is not None:
            self.vs30_grid = arr[:, ix_vs30].reshape(nlat, nlon)

        # Metric cache
        lon0 = float(np.mean(self.lon_grid)); lat0 = float(np.mean(self.lat_grid))
        self._xkm, self._ykm = _lonlat_to_xy_km(self.lon_grid, self.lat_grid, lon0, lat0)

    # -------------------------- rupture seeding helpers -----------------------
    def _seed_mask_from_points(self, points_lonlat: List[Tuple[float,float]], seed_radius_km: float) -> np.ndarray:
        """Build a boolean mask by painting a disk around each seed point."""
        H, W = self.im_grid.shape
        mask = np.zeros((H, W), dtype=bool)
        # approximate grid cell size in km
        cell_km = float(np.median([
            np.nanmedian(np.abs(np.diff(self._xkm, axis=1))),
            np.nanmedian(np.abs(np.diff(self._ykm, axis=0)))
        ]))
        cell_km = max(cell_km, 1e-3)
        rad_cells = max(1, int(np.ceil(seed_radius_km / cell_km)))

        for lon, lat in points_lonlat:
            i0, j0 = _nearest_index(self.lon_grid, self.lat_grid, lon, lat)
            i1, i2 = max(0, i0 - rad_cells), min(H - 1, i0 + rad_cells)
            j1, j2 = max(0, j0 - rad_cells), min(W - 1, j0 + rad_cells)
            di = np.arange(i1, i2 + 1)[:, None] - i0
            dj = np.arange(j1, j2 + 1)[None, :] - j0
            r = np.sqrt((di * cell_km)**2 + (dj * cell_km)**2)
            disk = r <= seed_radius_km + 1e-9
            sub = mask[i1:i2+1, j1:j2+1]
            sub[:] = sub | disk

        return mask

    # -------------------------- rupture GeoJSON parser ------------------------
    def set_source_from_rupture(self,
                                rupture_file: Optional[str] = None,
                                densify_factor: Optional[float] = None,
                                seed_radius_km: Optional[float] = None):
        """
        Seed the source from rupture geometry:
            - "rupture_edges": seed along lines or polygon rings.
            - "rupture_fill" : additionally seed the polygon interior.
        Optional: finite-fault timing if settings.use_fault_timing is True.

        Magnitude-dependent rupture velocity:
            if self.settings.use_vr_from_mag and self.event present:
                Vr = clip(vr_a + vr_b * (M-6), vr_min, vr_max)
            else:
                Vr = rupture_velocity_km_s
        """
        if rupture_file is None:
            rupture_file = self.inputs.rupture_file
        if densify_factor is None:
            densify_factor = self.settings.densify_factor
        if seed_radius_km is None:
            seed_radius_km = self.settings.seed_radius_km

        if rupture_file is None or not os.path.isfile(rupture_file):
            logger.warning("No rupture file provided/found; using epicentral point source.")
            self._build_point_source()
            return

        logger.info(f"Reading rupture: {rupture_file}")
        with open(rupture_file, "r", encoding="utf-8") as f:
            gj = json.load(f)

        line_lists: List[List[Tuple[float,float]]] = []
        polygon_rings: List[List[Tuple[float,float]]] = []

        def _add_line(coord_seq):
            # densify relative to cell size for smoother coverage
            cell_km = float(np.median([
                np.nanmedian(np.abs(np.diff(self._xkm, axis=1))),
                np.nanmedian(np.abs(np.diff(self._ykm, axis=0)))
            ]))
            cell_km = max(cell_km, 1e-3)
            pts: List[Tuple[float,float]] = []
            for a, b in zip(coord_seq[:-1], coord_seq[1:]):
                lon1, lat1 = float(a[0]), float(a[1])
                lon2, lat2 = float(b[0]), float(b[1])
                seg_km = float(_haversine_km(lon1, lat1, lon2, lat2))
                nstep = max(1, int(math.ceil(densify_factor * seg_km / cell_km)))
                for s in range(nstep):
                    t = s / nstep
                    pts.append((lon1 + t * (lon2 - lon1), lat1 + t * (lat2 - lat1)))
            pts.append((float(coord_seq[-1][0]), float(coord_seq[-1][1])))
            line_lists.append(pts)

        def _walk_geom(geom: Dict):
            gtype = geom.get("type", "")
            if gtype == "LineString":
                _add_line(geom["coordinates"])
            elif gtype == "MultiLineString":
                for seg in geom["coordinates"]:
                    _add_line(seg)
            elif gtype == "Polygon":
                for ring in geom["coordinates"]:
                    polygon_rings.append([(float(x[0]), float(x[1])) for x in ring])
                    _add_line(ring)
            elif gtype == "MultiPolygon":
                for poly in geom["coordinates"]:
                    for ring in poly:
                        polygon_rings.append([(float(x[0]), float(x[1])) for x in ring])
                        _add_line(ring)

        if "features" in gj:
            for feat in gj["features"]:
                _walk_geom(feat.get("geometry", {}))
        elif "geometry" in gj:
            _walk_geom(gj["geometry"])

        n_lines = len(line_lists); n_polyrings = len(polygon_rings)
        logger.info(f"Rupture geometries: lines={n_lines}, polygon_rings={n_polyrings}")
        total_pts = sum(len(seg) for seg in line_lists)
        logger.info(f"Densified rupture sample points: {total_pts} (densify_factor={densify_factor})")

        # Seed points
        seed_points: List[Tuple[float,float]] = [pt for seg in line_lists for pt in seg]

        # Optional polygon interior fill
        if self.settings.source_mode == "rupture_fill" and polygon_rings:
            try:
                from matplotlib.path import Path
                XY = np.column_stack([self.lon_grid.ravel(order="C"), self.lat_grid.ravel(order="C")])
                inside_any = np.zeros(XY.shape[0], dtype=bool)
                for ring in polygon_rings:
                    path = Path(ring, closed=True)
                    inside_any |= path.contains_points(XY)
                inside_mask = inside_any.reshape(self.im_grid.shape)
                I, J = np.where(inside_mask)
                for i, j in zip(I, J):
                    seed_points.append((float(self.lon_grid[i, j]), float(self.lat_grid[i, j])))
                logger.info(f"Polygon-fill seeding added {len(I)} interior grid cells.")
            except Exception:
                warnings.warn("Polygon fill requested but matplotlib.path not available; seeding edges only.")

        if not seed_points:
            logger.warning("No rupture-derived seed points; falling back to point source.")
            self._build_point_source()
            return

        # Paint buffered seeds on grid
        mask = self._seed_mask_from_points(seed_points, seed_radius_km=seed_radius_km)
        self.source_mask = mask

        # Basic stats
        seeded_cells = int(self.source_mask.sum())
        logger.info(f"Seeded grid cells (buffer {seed_radius_km} km): {seeded_cells}")

        # Optional finite-fault timing t0 along rupture
        self.T0_seed = None
        if self.settings.use_fault_timing and line_lists:
            all_pts = [pt for seg in line_lists for pt in seg]
            d2epi = np.array([_haversine_km(lon, lat, self.event.epicenter_lon, self.event.epicenter_lat)
                              for (lon, lat) in all_pts])
            hypo_idx = int(np.argmin(d2epi))
            hypo_lon, hypo_lat = all_pts[hypo_idx]

            t0_map = np.full(self.im_grid.shape, np.inf, dtype=float)

            # --- magnitude-dependent Vr (or fixed) ---
            if self.settings.use_vr_from_mag and self.event is not None:
                M = float(self.event.magnitude)
                Vr = np.clip(self.settings.vr_a + self.settings.vr_b * (M - 6.0),
                             self.settings.vr_min, self.settings.vr_max)
            else:
                Vr = float(self.settings.rupture_velocity_km_s)
            Vr = max(0.5, float(Vr))  # numerical lower bound

            for pts in line_lists:
                # cumulative distances along segment
                d = [0.0]
                for a, b in zip(pts[:-1], pts[1:]):
                    d.append(d[-1] + float(_haversine_km(a[0], a[1], b[0], b[1])))
                d = np.array(d)
                idx_h = int(np.argmin([_haversine_km(p[0], p[1], hypo_lon, hypo_lat) for p in pts]))
                for k, (lon, lat) in enumerate(pts):
                    i, j = _nearest_index(self.lon_grid, self.lat_grid, lon, lat)
                    t0 = abs(d[k] - d[idx_h]) / Vr
                    if t0 < t0_map[i, j]:
                        t0_map[i, j] = t0

            t0_map[~mask] = np.inf
            if np.isfinite(t0_map).any():
                self.T0_seed = t0_map
                t0 = self.T0_seed[np.isfinite(self.T0_seed)]
                logger.info(f"Finite-fault t0 stats (s): min={t0.min():.2f}, p50={np.median(t0):.2f}, max={t0.max():.2f}")
            else:
                logger.warning("Fault timing enabled but no finite t0 assigned on source mask.")

    def _build_point_source(self):
        """Seed a single cell at the epicenter (t0=0)."""
        i, j = _nearest_index(self.lon_grid, self.lat_grid,
                              self.event.epicenter_lon, self.event.epicenter_lat)
        mask = np.zeros_like(self.im_grid, dtype=bool)
        mask[i, j] = True
        self.source_mask = mask
        self.T0_seed = None
        logger.info("Point source seeded at epicenter.")

    # --------------------------- speed map construction -----------------------
    def compute_speed_map(self, custom_c_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Build the speed map c(x,y) based on `settings.speed_model`.
        If speed_model='custom', you must pass `custom_c_map`.

        After the primary model builds c, we apply a global scale λc and
        clamp to a realistic crustal range, per settings.speed_scale_lambda.
        """
        s = self.settings
        if self.im_grid is None:
            raise RuntimeError("Load ShakeMap first (call load_from_xml or run_scenario).")
        v = (self.inputs.vs30_grid if getattr(self.inputs, "vs30_grid", None) is not None else self.vs30_grid)

        if s.speed_model == "uniform":
            c = np.full_like(self.im_grid, float(s.uniform_c_km_s), dtype=float)

        elif s.speed_model == "vs30_linear":
            if v is None: raise ValueError("VS30/SVEL required for vs30_linear.")
            c = self._map_vs30_linear(v)

        elif s.speed_model == "vs30_log":
            if v is None: raise ValueError("VS30/SVEL required for vs30_log.")
            c = self._map_vs30_log(v)

        elif s.speed_model == "vs30_piecewise":
            if v is None: raise ValueError("VS30/SVEL required for vs30_piecewise.")
            c = self._map_vs30_piecewise(v)

        elif s.speed_model == "two_layer":
            if v is None: raise ValueError("VS30/SVEL required for two_layer.")
            c = self._map_two_layer(v)

        elif s.speed_model == "dispersion_period":
            if v is None: raise ValueError("VS30/SVEL required for dispersion_period.")
            c = self._map_dispersion_period(v)

        elif s.speed_model == "custom":
            if custom_c_map is None: raise ValueError("Provide custom_c_map for speed_model='custom'.")
            c = custom_c_map.astype(float)

        else:
            raise ValueError(f"Unknown speed_model {s.speed_model}")

        # --- Global λc scaling and crustal clamp ---
        c = c * float(self.settings.speed_scale_lambda)
        c = np.clip(c, 1.8, 4.5)

        self.c_map = c

        # Optional arrays export
        if self.outputs.save_speed_npy:
            np.save(os.path.join(self.outputs.out_dir, "speed_map_km_s.npy"), self.c_map)
        if self.outputs.save_speed_csv:
            np.savetxt(os.path.join(self.outputs.out_dir, "speed_map_km_s.csv"), self.c_map, delimiter=",")

        # Report
        cfinite = self.c_map[np.isfinite(self.c_map)]
        if cfinite.size:
            logger.info(f"[speed] wave_type={s.wave_type}, model={s.speed_model}, "
                        f"min={np.nanmin(cfinite):.2f}, med={np.nanmedian(cfinite):.2f}, "
                        f"max={np.nanmax(cfinite):.2f} km/s")
            if np.nanmin(cfinite) < 1.5 or np.nanmax(cfinite) > 6.0:
                warnings.warn("Speed extremes look unusual for crustal waves. Check settings or vs30 input.")
        return self.c_map

    # ----- VS30 mapping kernels
    def _map_vs30_linear(self, v: np.ndarray) -> np.ndarray:
        s = self.settings
        vclip = np.clip(v.astype(float), s.vs30_min, s.vs30_max)
        if s.wave_type == "S": cmin, cmax = s.c_min_s, s.c_max_s
        else:                  cmin, cmax = s.c_min_surface, s.c_max_surface
        return cmin + (cmax - cmin) * (vclip - s.vs30_min) / max(1e-6, (s.vs30_max - s.vs30_min))

    def _map_vs30_log(self, v: np.ndarray) -> np.ndarray:
        s = self.settings
        vclip = np.maximum(v.astype(float), 100.0)  # avoid log(0)
        logv = np.log10(vclip)
        if s.wave_type == "S": target_min, target_max = s.c_min_s, s.c_max_s
        else:                  target_min, target_max = s.c_min_surface, s.c_max_surface
        a = np.log10(max(s.vs30_min, 100.0)); b = np.log10(max(s.vs30_max, s.vs30_min+1))
        w = (logv - a) / max(1e-6, (b - a)); w = np.clip(w, 0, 1)
        return target_min + (target_max - target_min) * w

    def _map_vs30_piecewise(self, v: np.ndarray) -> np.ndarray:
        s = self.settings
        brk1, brk2, brk3 = s.piecewise_breaks  # 180, 360, 760
        if s.wave_type == "S": speeds = s.s_class_speeds
        else:                  speeds = s.surface_class_speeds
        c = np.empty_like(v, dtype=float)
        c[v <  brk1] = speeds[0]   # E
        c[(v >= brk1) & (v < brk2)] = speeds[1]  # D
        c[(v >= brk2) & (v < brk3)] = speeds[2]  # C
        c[v >= brk3] = speeds[3]   # B/A
        return c

    def _map_two_layer(self, v: np.ndarray) -> np.ndarray:
        s = self.settings
        vclip = np.clip(v.astype(float), 100.0, 1500.0)
        c_shallow = np.maximum(0.001*vclip, s.shallow_v_floor)  # VS30 m/s -> ~km/s
        c_crust = np.full_like(vclip, s.two_layer_c_crust, dtype=float)
        w = np.clip(float(s.two_layer_w_surface), 0.0, 1.0)
        s_eff = w*(1.0/np.maximum(c_shallow, 1e-6)) + (1.0-w)*(1.0/np.maximum(c_crust, 1e-6))
        return 1.0/np.maximum(s_eff, 1e-6)

    def _map_dispersion_period(self, v: np.ndarray) -> np.ndarray:
        s = self.settings
        base = float(s.dispersion_baseline_c_km_s)
        brk1, brk2, brk3 = s.piecewise_breaks
        fE, fD, fC, fB = s.dispersion_class_factors
        c = np.full_like(v, base, dtype=float)
        c[v <  brk1] *= fE
        c[(v >= brk1) & (v < brk2)] *= fD
        c[(v >= brk2) & (v < brk3)] *= fC
        c[v >= brk3] *= fB
        return c

    # ----------------------------- travel times -------------------------------
    def _dijkstra_travel_time(self) -> np.ndarray:
        """Eight-connected grid Dijkstra using local metric step lengths."""
        if self.c_map is None:
            raise RuntimeError("Compute speed map first.")
        if self.source_mask is None:
            self._build_point_source()

        H, W = self.im_grid.shape
        T = np.full((H, W), np.inf, dtype=float)
        visited = np.zeros((H, W), dtype=bool)

        import heapq
        pq = []

        # initialize sources
        src_i, src_j = np.where(self.source_mask)
        for i, j in zip(src_i, src_j):
            t0 = 0.0
            if getattr(self, "T0_seed", None) is not None:
                val = float(self.T0_seed[i, j])
                t0 = val if np.isfinite(val) else 0.0
            T[i, j] = t0
            heapq.heappush(pq, (t0, int(i), int(j)))

        NEI = [(-1,0),(1,0),(0,-1),(0,1),( -1,-1),(-1,1),(1,-1),(1,1)]

        if self._xkm is None or self._ykm is None:
            lon0 = float(np.mean(self.lon_grid)); lat0 = float(np.mean(self.lat_grid))
            self._xkm, self._ykm = _lonlat_to_xy_km(self.lon_grid, self.lat_grid, lon0, lat0)

        def step_len(i, j, i2, j2) -> float:
            dx = self._xkm[i2, j2] - self._xkm[i, j]
            dy = self._ykm[i2, j2] - self._ykm[i, j]
            return math.hypot(dx, dy)

        while pq:
            t, i, j = heapq.heappop(pq)
            if visited[i, j]:
                continue
            visited[i, j] = True
            cij = self.c_map[i, j]
            for di, dj in NEI:
                i2, j2 = i + di, j + dj
                if i2 < 0 or j2 < 0 or i2 >= H or j2 >= W:  # bounds
                    continue
                if visited[i2, j2]:
                    continue
                L = step_len(i, j, i2, j2)
                # --- Mid-edge speed interpolation (smoother than harmonic mean) ---
                # Uses current cell, neighbor cell, and vertical neighbors at column j.
                c_edge = 0.25 * (cij + self.c_map[i2, j2] +
                                 self.c_map[max(0, i-1), j] + self.c_map[min(H-1, i+1), j])
                c_edge = np.clip(c_edge, 1.0, 6.0)  # numerical/physical guard
                dt = L / max(c_edge, 1e-6)
                t_new = t + dt
                if t_new < T[i2, j2]:
                    T[i2, j2] = t_new
                    heapq.heappush(pq, (t_new, i2, j2))

        self.T_map_s = T
        if self.outputs.save_tmap_npy:
            np.save(os.path.join(self.outputs.out_dir, "travel_time_s.npy"), self.T_map_s)
        if self.outputs.save_tmap_csv:
            np.savetxt(os.path.join(self.outputs.out_dir, "travel_time_s.csv"), self.T_map_s, delimiter=",")
        return T

    # ----------- Fast Marching Method (FMM) eikonal solver (4-neighbor) -------
    def _fmm_travel_time(self) -> np.ndarray:
        """4-neighbor Fast Marching (upwind quadratic update) on the metric grid."""
        import heapq
        if self.c_map is None: raise RuntimeError("Compute speed map first.")
        if self.source_mask is None: self._build_point_source()
        H, W = self.im_grid.shape
        T = np.full((H,W), np.inf, dtype=float)
        alive = np.zeros((H,W), dtype=bool)   # frozen nodes
        band = np.zeros((H,W), dtype=bool)    # in narrow band
        sfield = 1.0 / np.maximum(self.c_map, 1e-6)  # slowness

        if self._xkm is None or self._ykm is None:
            lon0 = float(np.mean(self.lon_grid)); lat0 = float(np.mean(self.lat_grid))
            self._xkm, self._ykm = _lonlat_to_xy_km(self.lon_grid, self.lat_grid, lon0, lat0)

        pq = []
        I, J = np.where(self.source_mask)
        # initialize band from sources
        for i, j in zip(I, J):
            t0 = 0.0
            if getattr(self, "T0_seed", None) is not None and np.isfinite(self.T0_seed[i,j]):
                t0 = float(self.T0_seed[i,j])
            T[i,j] = t0
            alive[i,j] = True
            for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                i2, j2 = i+di, j+dj
                if 0 <= i2 < H and 0 <= j2 < W and not alive[i2,j2] and not band[i2,j2]:
                    T[i2,j2] = self._fmm_update(i2, j2, T, alive, sfield)
                    heapq.heappush(pq, (T[i2,j2], i2, j2))
                    band[i2,j2] = True

        while pq:
            t, i, j = heapq.heappop(pq)
            if alive[i,j]: continue
            alive[i,j] = True
            for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                i2, j2 = i+di, j+dj
                if 0 <= i2 < H and 0 <= j2 < W and not alive[i2,j2]:
                    Told = T[i2,j2]
                    Tnew = self._fmm_update(i2, j2, T, alive, sfield)
                    if Tnew < Told - 1e-12:
                        T[i2,j2] = Tnew
                        heapq.heappush(pq, (Tnew, i2, j2))
                        band[i2,j2] = True

        self.T_map_s = T
        if self.outputs.save_tmap_npy:
            np.save(os.path.join(self.outputs.out_dir, "travel_time_s.npy"), self.T_map_s)
        if self.outputs.save_tmap_csv:
            np.savetxt(os.path.join(self.outputs.out_dir, "travel_time_s.csv"), self.T_map_s, delimiter=",")
        return T

    def _fmm_update(self, i: int, j: int, T: np.ndarray, alive: np.ndarray, sfield: np.ndarray) -> float:
        """Upwind quadratic update for a single node using local metric spacings."""
        def local_step(axis: np.ndarray, ii: int, jj: int) -> float:
            v = []
            if jj-1 >= 0: v.append(abs(axis[ii,jj] - axis[ii,jj-1]))
            if jj+1 < axis.shape[1]: v.append(abs(axis[ii,jj+1] - axis[ii,jj]))
            if not v: v = [np.nanmedian(np.abs(np.diff(axis, axis=1)))]
            return max(np.nanmedian(v), 1e-6)

        dx = local_step(self._xkm, i, j)
        dy = local_step(self._ykm, i, j)

        Tx = []
        if j-1 >= 0 and np.isfinite(T[i, j-1]) and alive[i, j-1]: Tx.append(T[i, j-1])
        if j+1 < T.shape[1] and np.isfinite(T[i, j+1]) and alive[i, j+1]: Tx.append(T[i, j+1])
        Ty = []
        if i-1 >= 0 and np.isfinite(T[i-1, j]) and alive[i-1, j]: Ty.append(T[i-1, j])
        if i+1 < T.shape[0] and np.isfinite(T[i+1, j]) and alive[i+1, j]: Ty.append(T[i+1, j])

        a = min(Tx) if Tx else np.inf
        b = min(Ty) if Ty else np.inf
        s = float(sfield[i, j])

        if not np.isfinite(a) and np.isfinite(b):  # one-sided
            return b + dy / (1.0/s)
        if np.isfinite(a) and not np.isfinite(b):
            return a + dx / (1.0/s)
        if not np.isfinite(a) and not np.isfinite(b):
            return np.inf

        # Solve (T - a)^2/dx^2 + (T - b)^2/dy^2 = 1/s^2
        Ta, Tb = sorted([a, b])
        A = (1.0/dx**2 + 1.0/dy**2)
        B = -2.0*(Ta/dx**2 + Tb/dy**2)
        C = (Ta**2/dx**2 + Tb**2/dy**2) - 1.0/s**2
        disc = B*B - 4*A*C
        if disc < 0:
            return max(Ta + dx/s, Tb + dy/s)
        Tnew = (-B + math.sqrt(disc)) / (2*A)
        return max(Tnew, max(Ta, Tb))

    def compute_travel_time_field(self) -> np.ndarray:
        """Dispatch to the selected solver, report summary stats, and return T(x,y)."""
        if self.settings.mode == "dijkstra":
            T = self._dijkstra_travel_time()
        elif self.settings.mode == "fmm":
            T = self._fmm_travel_time()
        elif self.settings.mode == "ml":
            raise NotImplementedError("Machine learning mode is a future extension.")
        else:
            raise ValueError(f"Unknown mode {self.settings.mode}")

        finite_T = T[np.isfinite(T)]
        if finite_T.size:
            logger.info(f"[times] arrival s: min={np.min(finite_T):.2f}, "
                        f"p50={np.median(finite_T):.2f}, "
                        f"p90={np.percentile(finite_T,90):.2f}, "
                        f"max={np.max(finite_T):.2f}")
        return T

    # ------------------------------- exports ----------------------------------
    def export_csv(self, path: str = None) -> str:
        """Export grid table (lon,lat,MMI,VS30?,speed,arrival) to CSV."""
        if path is None:
            path = os.path.join(self.outputs.out_dir, "shake_propagation.csv")
        if self.lon_grid is None or self.T_map_s is None or self.c_map is None:
            raise RuntimeError("Need lon_grid, T_map_s, c_map to export.")

        flat = lambda a: a.ravel(order="C")
        cols = {
            "lon": flat(self.lon_grid),
            "lat": flat(self.lat_grid),
            "mmi": flat(self.im_grid),
            "speed_km_s": flat(self.c_map),
            "arrival_s": flat(self.T_map_s),
        }
        if self.vs30_grid is not None:
            cols["vs30_m_s"] = flat(self.vs30_grid)

        keys = list(cols.keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for i in range(cols["lon"].size):
                w.writerow([cols[k][i] for k in keys])
        logger.info(f"[export] CSV written: {path}")
        return path

    def export_augmented_xml(self, path: str = None) -> str:
        """Write an augmented ShakeMap-style XML including SPEED_KM_S and ARRIVAL_S."""
        if path is None:
            path = os.path.join(self.outputs.out_dir, "shake_propagation_augmented.xml")
        if self.event is None or self.lon_grid is None or self.T_map_s is None or self.c_map is None:
            raise RuntimeError("Need event, grids, c_map, T_map_s before exporting XML.")

        nlat, nlon = self.im_grid.shape
        lon_axis = self.lon_grid[0, :]
        lat_axis = self.lat_grid[:, 0]
        lon_min, lon_max = float(lon_axis.min()), float(lon_axis.max())
        lat_min, lat_max = float(lat_axis.min()), float(lat_axis.max())
        dlon = float(np.abs(np.diff(lon_axis)).mean()) if lon_axis.size > 1 else 0.0
        dlat = float(np.abs(np.diff(lat_axis)).mean()) if lat_axis.size > 1 else 0.0

        flat = lambda a: a.ravel(order="C")
        fields = [("LON","dd"), ("LAT","dd"), ("MMI","intensity")]
        if self.vs30_grid is not None:
            fields.append(("VS30","m/s"))
        fields.append(("SPEED_KM_S","km/s"))
        fields.append(("ARRIVAL_S","s"))

        lonF = flat(self.lon_grid); latF = flat(self.lat_grid); mmiF = flat(self.im_grid)
        data_cols = [lonF, latF, mmiF]
        if self.vs30_grid is not None:
            data_cols.append(flat(self.vs30_grid))
        data_cols.append(flat(self.c_map))
        data_cols.append(flat(self.T_map_s))

        root = ET.Element("shakemap_grid",
                          attrib={
                              "event_id": self.event.event_id,
                              "shakemap_id": self.event.event_id,
                              "shakemap_version": "1",
                              "code_version": "SHAKEpropagate-0.6",
                              "process_timestamp": self.event.event_time_str,
                              "shakemap_originator": "custom",
                              "map_status": "custom",
                              "shakemap_event_type": "ACTUAL"
                          })
        ET.SubElement(root, "event", attrib={
            "event_id": self.event.event_id,
            "magnitude": f"{self.event.magnitude}",
            "depth": f"{self.event.depth_km}",
            "lat": f"{self.event.epicenter_lat}",
            "lon": f"{self.event.epicenter_lon}",
            "event_timestamp": self.event.event_time_str,
            "event_network": "custom",
            "event_description": self.event.description
        })
        ET.SubElement(root, "grid_specification", attrib={
            "lon_min": f"{lon_min:.4f}",
            "lat_min": f"{lat_min:.4f}",
            "lon_max": f"{lon_max:.4f}",
            "lat_max": f"{lat_max:.4f}",
            "nominal_lon_spacing": f"{dlon:.4f}",
            "nominal_lat_spacing": f"{dlat:.4f}",
            "nlon": f"{nlon}",
            "nlat": f"{nlat}"
        })
        for idx, (name, units) in enumerate(fields, start=1):
            ET.SubElement(root, "grid_field", attrib={"index": str(idx), "name": name, "units": units})

        rows = []
        col_idx_base = 3
        for i in range(lonF.size):
            row = [f"{lonF[i]:.4f}", f"{latF[i]:.4f}", f"{mmiF[i]:.3f}"]
            col_idx = col_idx_base
            if self.vs30_grid is not None:
                row.append(f"{data_cols[col_idx][i]:.1f}"); col_idx += 1
            row.append(f"{data_cols[col_idx][i]:.3f}"); col_idx += 1  # SPEED_KM_S
            row.append(f"{data_cols[col_idx][i]:.2f}")                 # ARRIVAL_S
            rows.append(" ".join(row))
        gd = ET.SubElement(root, "grid_data")
        gd.text = " " + " ".join(rows)

        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        logger.info(f"[export] Augmented XML written: {path}")
        return path

    # ------------------------------- visualization ---------------------------
    def _frame_values(self, t_s: float) -> np.ndarray:
        """Return the time-masked intensity grid for a given time, with soft band."""
        if self.T_map_s is None:
            raise RuntimeError("Run compute_travel_time_field first.")
        if self.settings.soften_band_s and self.settings.soften_band_s > 0:
            alpha = np.clip((t_s - self.T_map_s) / self.settings.soften_band_s, 0.0, 1.0)
            return np.where(self.T_map_s <= t_s, self.im_grid * alpha, np.nan)
        else:
            return np.where(self.T_map_s <= t_s, self.im_grid, np.nan)

    def _plot_basemap(self, ax, extent):
        """Wrapper to draw Cartopy basemap if configured."""
        if not _HAS_CARTOPY or not self.settings.use_cartopy:
            return
        _usgs_basemap(ax, extent, label_size=12)

    def render_frame(self, t_s: float, fname: Optional[str] = None):
        """Render a single time slice (e.g., for animation)."""
        vals = self._frame_values(t_s)
        title = f"{self.settings.title_prefix}, t = {t_s:.1f} s, {self.settings.im_label}"

        use_usgs_mmi = self.settings.im_label.upper() == "MMI"
        if use_usgs_mmi:
            cmap, norm, ticks, label = _usgs_mmi_cmap_norm()
        else:
            cmap, norm, ticks, label = plt.get_cmap("turbo"), None, None, self.settings.im_label

        default_size = (7.0, 5.6) if (self.settings.use_cartopy and _HAS_CARTOPY) else (6.8, 5.2)
        fs = self.settings.frame_figsize or default_size

        if self.settings.use_cartopy and _HAS_CARTOPY:
            proj = ccrs.PlateCarree()
            fig = plt.figure(figsize=fs, dpi=self.settings.dpi)
            ax = plt.axes(projection=proj)
            extent = [float(self.lon_grid.min()), float(self.lon_grid.max()),
                      float(self.lat_grid.min()), float(self.lat_grid.max())]
            self._plot_basemap(ax, extent)
            im = ax.pcolormesh(self.lon_grid, self.lat_grid, vals, transform=proj,
                               cmap=cmap, norm=norm, shading="auto",
                               vmin=self.settings.vmin, vmax=self.settings.vmax)
            ax.plot(self.event.epicenter_lon, self.event.epicenter_lat, marker="*", color="red",
                    markersize=9, transform=proj, zorder=999)
            ax.set_title(title)
            cb = plt.colorbar(im, ax=ax, shrink=0.8, label=label)
            if ticks is not None:
                cb.set_ticks(ticks)
        else:
            fig, ax = plt.subplots(figsize=fs, dpi=self.settings.dpi)
            im = ax.pcolormesh(self.lon_grid, self.lat_grid, vals, cmap=cmap,
                               norm=norm, shading="auto",
                               vmin=self.settings.vmin, vmax=self.settings.vmax)
            ax.plot(self.event.epicenter_lon, self.event.epicenter_lat, marker="*", color="k", markersize=9, zorder=999)
            ax.set_title(title)
            cb = plt.colorbar(im, ax=ax, shrink=0.8, label=label)
            if ticks is not None:
                cb.set_ticks(ticks)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

        if fname:
            fig.savefig(fname, bbox_inches="tight"); plt.close(fig)
        else:
            plt.show()

    def render_frames(self, times_s: Optional[List[float]] = None):
        """Render a sequence of frames. If times_s is None, auto-generate at frame_dt_s."""
        if self.T_map_s is None:
            raise RuntimeError("Run compute_travel_time_field first.")
        if times_s is None:
            if self.settings.frame_times_s is not None:
                times_s = self.settings.frame_times_s
            else:
                tmax = float(np.nanpercentile(self.T_map_s, 98))
                times_s = list(np.arange(0, tmax + self.settings.frame_dt_s, self.settings.frame_dt_s))
        for t in times_s:
            fname = os.path.join(self.outputs.frames_dir, f"frame_{int(round(t))}s.png")
            self.render_frame(t, fname=fname)

    def plot_travel_time_map(self, fname: Optional[str] = None,
                             interval_s: float = 10.0,
                             levels_s: Optional[List[float]] = None):
        """Plot the travel-time map with labeled contours."""
        if self.T_map_s is None:
            raise RuntimeError("Run compute_travel_time_field first.")
        T = self.T_map_s
        tmax = float(np.nanpercentile(T, 98))
        if levels_s is None:
            levels_s = list(np.arange(0, max(1.0, tmax), interval_s))

        default_size = (7.6, 6.0) if (self.settings.use_cartopy and _HAS_CARTOPY) else (7.0, 5.8)
        fs = self.settings.tmap_figsize or default_size

        if self.settings.use_cartopy and _HAS_CARTOPY:
            proj = ccrs.PlateCarree()
            fig = plt.figure(figsize=fs, dpi=self.settings.dpi)
            ax = plt.axes(projection=proj)
            extent = [float(self.lon_grid.min()), float(self.lon_grid.max()),
                      float(self.lat_grid.min()), float(self.lat_grid.max())]
            _usgs_basemap(ax, extent, label_size=12)
            im = ax.pcolormesh(self.lon_grid, self.lat_grid, T, transform=proj,
                               cmap="viridis", shading="auto")
            cs = ax.contour(self.lon_grid, self.lat_grid, T, levels=levels_s, colors="k",
                            linewidths=0.7, alpha=0.9, transform=proj, zorder=1000)
            ax.clabel(cs, fmt="%d s")
            ax.plot(self.event.epicenter_lon, self.event.epicenter_lat, marker="*", color="red",
                    markersize=9, transform=proj, zorder=999)
            ax.set_title("Arrival time map with contours")
            plt.colorbar(im, ax=ax, shrink=0.8, label="Arrival time, s")
        else:
            fig, ax = plt.subplots(figsize=fs, dpi=self.settings.dpi)
            im = ax.pcolormesh(self.lon_grid, self.lat_grid, T, cmap="viridis", shading="auto")
            cs = ax.contour(self.lon_grid, self.lat_grid, T, levels=levels_s, colors="k",
                            linewidths=0.7, alpha=0.9,zorder=999)
            ax.clabel(cs, fmt="%d s")
            ax.plot(self.event.epicenter_lon, self.event.epicenter_lat, marker="*", color="red", markersize=9, zorder=999)
            ax.set_title("Arrival time map with contours")
            plt.colorbar(im, ax=ax, shrink=0.8, label="Arrival time, s")
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

        if fname:
            fig.savefig(fname, bbox_inches="tight"); plt.close(fig)
        else:
            plt.show()

    def plot_mmi_with_travel_time_contours(self, fname: Optional[str] = None,
                                           interval_s: float = 10.0,
                                           levels_s: Optional[List[float]] = None):
        """Plot source MMI with overlaid travel-time contours."""
        if self.T_map_s is None:
            raise RuntimeError("Run compute_travel_time_field first.")
        T = self.T_map_s
        tmax = float(np.nanpercentile(T, 98))
        if levels_s is None:
            levels_s = list(np.arange(0, max(1.0, tmax), interval_s))

        cmap, norm, ticks, label = _usgs_mmi_cmap_norm()
        default_size = (7.6, 6.0) if (self.settings.use_cartopy and _HAS_CARTOPY) else (7.0, 5.8)
        fs = self.settings.mmi_contours_figsize or default_size

        if self.settings.use_cartopy and _HAS_CARTOPY:
            proj = ccrs.PlateCarree()
            fig = plt.figure(figsize=fs, dpi=self.settings.dpi)
            ax = plt.axes(projection=proj)
            extent = [float(self.lon_grid.min()), float(self.lon_grid.max()),
                      float(self.lat_grid.min()), float(self.lat_grid.max())]
            _usgs_basemap(ax, extent, label_size=12)
            im = ax.pcolormesh(self.lon_grid, self.lat_grid, self.im_grid, transform=proj,
                               cmap=cmap, norm=norm, shading="auto")
            cs = ax.contour(self.lon_grid, self.lat_grid, T, levels=levels_s, colors="k",
                            linewidths=0.7, alpha=0.9, transform=proj, zorder=1000)
            ax.clabel(cs, fmt="%d s")
            ax.plot(self.event.epicenter_lon, self.event.epicenter_lat, marker="*", color="red",
                    markersize=9, transform=proj, zorder=998)
            ax.set_title(f"{self.settings.im_label} with arrival-time contours")
            cb = plt.colorbar(im, ax=ax, shrink=0.8, label=label)
            cb.set_ticks(ticks)
        else:
            fig, ax = plt.subplots(figsize=fs, dpi=self.settings.dpi)
            im = ax.pcolormesh(self.lon_grid, self.lat_grid, self.im_grid, cmap=cmap, norm=norm, shading="auto")
            cs = ax.contour(self.lon_grid, self.lat_grid, T, levels=levels_s, colors="k", linewidths=0.7, alpha=0.9, zorder=999)
            ax.clabel(cs, fmt="%d s")
            ax.plot(self.event.epicenter_lon, self.event.epicenter_lat, marker="*", color="red", markersize=9, zorder=999)
            ax.set_title(f"{self.settings.im_label} with arrival-time contours")
            cb = plt.colorbar(im, ax=ax, shrink=0.8, label=label)
            cb.set_ticks(ticks)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

        if fname:
            fig.savefig(fname, bbox_inches="tight"); plt.close(fig)
        else:
            plt.show()

    # ------------------------------- arrivals --------------------------------
    def arrival_times_at_points(self,
                                lons: List[float],
                                lats: List[float],
                                names: Optional[List[str]] = None,
                                issue_time_s: Optional[float] = None) -> List[Dict]:
        """
        Return arrival-time info at named points (lon,lat).
        If issue_time_s is given, also compute lead times (arrival - issue).
        """
        if self.T_map_s is None:
            raise RuntimeError("Run compute_travel_time_field first.")
        out = []
        for k, (LON, LAT) in enumerate(zip(lons, lats)):
            i, j = _nearest_index(self.lon_grid, self.lat_grid, LON, LAT)
            t_arr = float(self.T_map_s[i, j])
            lead = None if issue_time_s is None else t_arr - float(issue_time_s)
            name = names[k] if names and k < len(names) else f"site_{k+1}"
            dist_km = float(_haversine_km(LON, LAT, self.event.epicenter_lon, self.event.epicenter_lat))
            out.append({
                "name": name,
                "lon": float(LON),
                "lat": float(LAT),
                "epicentral_distance_km": dist_km,
                "arrival_time_s": t_arr,
                "lead_time_s": lead
            })
        return out

    # ------------------------------- debug helpers ---------------------------
    def debug_source_summary(self):
        """Print basic summary for the current source mask."""
        if self.source_mask is None:
            print("[debug] source_mask is None")
            return
        n = int(self.source_mask.sum())
        print(f"[debug] seeded grid cells: {n}")
        if n > 0:
            I, J = np.where(self.source_mask)
            lonS = self.lon_grid[I, J]; latS = self.lat_grid[I, J]
            print(f"[debug] source lon range: {lonS.min():.3f} .. {lonS.max():.3f}")
            print(f"[debug] source lat range: {latS.min():.3f} .. {latS.max():.3f}")

    def plot_source_mask(self, fname: Optional[str] = None):
        """Plot a quick binary map of seeded cells."""
        if self.source_mask is None:
            raise RuntimeError("No source_mask yet. Call set_source_from_rupture() or _build_point_source().")
        plt.figure(figsize=self.settings.frame_figsize or (6.0, 5.0), dpi=self.settings.dpi)
        plt.pcolormesh(self.lon_grid, self.lat_grid, self.source_mask, shading="nearest")
        plt.scatter(self.event.epicenter_lon, self.event.epicenter_lat, marker="*", c="r", zorder=999)
        plt.title("Seeded source cells")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        if fname:
            plt.savefig(fname, bbox_inches="tight"); plt.close()
        else:
            plt.show()

    # ------------------------------- helpers ---------------------------------
    def sanity_report(self):
        """Print basic statistics of speeds and arrival times (if available)."""
        if self.c_map is not None:
            c = self.c_map[np.isfinite(self.c_map)]
            if c.size:
                print(f"[sanity] speed km/s: min={np.min(c):.2f}, p50={np.median(c):.2f}, "
                      f"p90={np.percentile(c,90):.2f}, max={np.max(c):.2f}")
        if self.T_map_s is not None:
            T = self.T_map_s[np.isfinite(self.T_map_s)]
            if T.size:
                print(f"[sanity] arrival s: min={np.min(T):.2f}, p50={np.median(T):.2f}, "
                      f"p90={np.percentile(T,90):.2f}, max={np.max(T):.2f}")

    def summary(self) -> Dict:
        """Return a small dictionary of key run settings & grid shape."""
        return {
            "event_id": None if self.event is None else self.event.event_id,
            "grid_shape": None if self.im_grid is None else list(self.im_grid.shape),
            "speed_model": self.settings.speed_model,
            "mode": self.settings.mode,
            "wave_type": self.settings.wave_type,
            "source_mode": self.settings.source_mode
        }

    # ----------------------------- export helpers (ALL) -----------------------
    def _export_arrivals_csv(self, arrivals, path: str):
        """Write a small arrivals table for probe points."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name","lon","lat","epicentral_distance_km","arrival_time_s","lead_time_s"])
            for r in arrivals:
                w.writerow([r["name"], f"{r['lon']:.5f}", f"{r['lat']:.5f}",
                            f"{r['epicentral_distance_km']:.3f}",
                            f"{r['arrival_time_s']:.3f}",
                            "" if r["lead_time_s"] is None else f"{r['lead_time_s']:.3f}"])
        logger.info(f"[export] Arrivals CSV written: {path}")

    def export_all_artifacts(self, case_name: str, include_frames: bool = True, arrivals=None):
        """
        Save every artifact we can: figures, frames, tables, arrays.
        Requires load+seed+compute to have been run already.
        """
        # Figures
        self.plot_mmi_with_travel_time_contours(
            fname=os.path.join(self.outputs.out_dir, f"{case_name}_mmi_tcontours.png"))
        self.plot_travel_time_map(
            fname=os.path.join(self.outputs.out_dir, f"{case_name}_tmap.png"))
        try:
            self.plot_source_mask(
                fname=os.path.join(self.outputs.out_dir, f"{case_name}_sourcemask.png"))
        except Exception as e:
            logger.warning(f"plot_source_mask skipped: {e}")

        # Frames
        if include_frames:
            self.render_frames()

        # Tables
        self.export_csv(os.path.join(self.outputs.out_dir, f"{case_name}.csv"))
        self.export_augmented_xml(os.path.join(self.outputs.out_dir, f"{case_name}.xml"))

        # Arrays (NPY/CSV)
        np.save(os.path.join(self.outputs.out_dir, "speed_map_km_s.npy"), self.c_map)
        np.savetxt(os.path.join(self.outputs.out_dir, "speed_map_km_s.csv"), self.c_map, delimiter=",")
        np.save(os.path.join(self.outputs.out_dir, "travel_time_s.npy"), self.T_map_s)
        np.savetxt(os.path.join(self.outputs.out_dir, "travel_time_s.csv"), self.T_map_s, delimiter=",")

        # Arrivals table
        if arrivals:
            self._export_arrivals_csv(arrivals, os.path.join(self.outputs.out_dir, f"{case_name}_arrivals.csv"))

    # ------------------------------- scenario API ----------------------------
    def _rebase_output_dirs(self, case_name: str):
        """Set outputs to export/SHAKEpropagate/<eventid>/<case_name>/..."""
        if self.event is None:
            raise RuntimeError("Event not loaded; cannot set export directories.")
        base = os.path.join("export", "SHAKEpropagate", self.event.event_id, case_name)
        self.outputs.out_dir = _ensure_dir(base)
        self.outputs.frames_dir = _ensure_dir(os.path.join(base, "frames"))
        logger.info(f"[export-base] {self.outputs.out_dir}")

    def run_scenario(self,
                     case_name: str = "default",
                     overrides: Optional[Dict] = None,
                     seed_from: Literal["auto","epicenter","rupture"]="auto",
                     make_frames: bool = False,
                     plot_contours: bool = True,   # kept for compatibility; ignored if export_all=True
                     export: bool = True,          # kept for compatibility; ignored if export_all=True
                     export_all: bool = True,      # write figures + frames + tables + arrays
                     probe_points: Optional[List[Tuple[float,float,str]]] = None) -> Dict:
        """
        High-level convenience runner that performs:
          load -> set outputs -> seed -> build speed -> compute times -> export

        Returns
        -------
        dict with:
          summary: run summary dict
          arrivals: list of per-point arrival info (if probe_points provided)
        """
        # Apply overrides to settings
        if overrides:
            for k, v in overrides.items():
                if hasattr(self.settings, k):
                    setattr(self.settings, k, v)
                else:
                    logger.warning(f"Unknown setting '{k}' in overrides; ignored.")

        # Parse XML if needed
        if self.lon_grid is None or self.event is None:
            self.load_from_xml()

        # Create export dirs for this scenario
        self._rebase_output_dirs(case_name)

        # Seed selection
        if seed_from == "epicenter":
            self._build_point_source()
        elif seed_from == "rupture" or (seed_from == "auto" and self.inputs.rupture_file):
            self.set_source_from_rupture()
        else:
            self._build_point_source()

        # Speed + times
        self.compute_speed_map()
        self.compute_travel_time_field()

        # Optional per-point arrivals
        arrivals = None
        if probe_points:
            lons = [p[0] for p in probe_points]; lats = [p[1] for p in probe_points]
            names = [p[2] for p in probe_points]
            arrivals = self.arrival_times_at_points(lons, lats, names)

        # Exports
        if export_all:
            self.export_all_artifacts(case_name, include_frames=make_frames, arrivals=arrivals)
        else:
            if plot_contours:
                self.plot_mmi_with_travel_time_contours(
                    fname=os.path.join(self.outputs.out_dir, f"{case_name}_mmi_tcontours.png"))
            if make_frames:
                self.render_frames()
            if export:
                self.export_csv(os.path.join(self.outputs.out_dir, f"{case_name}.csv"))
                self.export_augmented_xml(os.path.join(self.outputs.out_dir, f"{case_name}.xml"))

        return {"summary": self.summary(), "arrivals": arrivals}








    # =============================================================================
    #
    #
    #
    #
    # PATCH: WILBER WAVEFORMS → ARRIVAL PICKS → CARTOPY MAP OVERLAY
    # 
    #
    #
    # =============================================================================

    # -------------------------------------------------------------------------
    # A) Small helpers
    # -------------------------------------------------------------------------
    def _wilber__log(self, msg):
        try:
            logger.info(msg)
        except Exception:
            print(msg)

    def _wilber__as_path(self, p):
        import os
        return os.path.normpath(str(p))


    # override 
    def _wilber__parse_iso(self, s):
        import datetime as dt
        if s is None:
            return None
        txt = str(s).strip().replace("Z", "+00:00")
        try:
            return dt.datetime.fromisoformat(txt)
        except Exception:
            try:
                import pandas as pd
                return pd.to_datetime(txt, utc=False).to_pydatetime()
            except Exception:
                return None



    def _wilber__parse_iso(self, s):
        """
        Parse ISO-ish datetime strings robustly and return a **naive UTC datetime**.

        Why: ShakeMap XML often uses 'Z' / '+00:00' (timezone-aware),
        while Wilber waveform timestamps are usually naive. Mixing aware/naive
        silently breaks subtraction and collapses the picking pipeline.

        Rules:
          - If timezone-aware -> convert to UTC and drop tzinfo (naive UTC).
          - If naive -> keep as-is (assumed UTC-like, consistent with Wilber files).
        """
        import datetime as dt

        if s is None:
            return None

        txt = str(s).strip()
        if not txt:
            return None

        # normalize common Z suffix
        txt = txt.replace("Z", "+00:00")

        def _to_naive_utc(d):
            try:
                if getattr(d, "tzinfo", None) is not None:
                    # convert to UTC then drop tzinfo
                    return d.astimezone(dt.timezone.utc).replace(tzinfo=None)
                return d
            except Exception:
                return d

        # try stdlib
        try:
            d = dt.datetime.fromisoformat(txt)
            return _to_naive_utc(d)
        except Exception:
            pass

        # try pandas
        try:
            import pandas as pd
            d = pd.to_datetime(txt, utc=True, errors="raise").to_pydatetime()
            return _to_naive_utc(d)
        except Exception:
            return None


    def _wilber__require_grids(self):
        """Ensure lon_grid/lat_grid exist."""
        if getattr(self, "lon_grid", None) is None or getattr(self, "lat_grid", None) is None:
            self.load_from_xml()
        if getattr(self, "lon_grid", None) is None or getattr(self, "lat_grid", None) is None:
            raise RuntimeError("lon_grid/lat_grid not available after load_from_xml().")

    def _wilber__ensure_dir(self, p):
        import os
        os.makedirs(p, exist_ok=True)
        return p

    #overriden
    def _wilber__read_origin_from_xml(self, xml_path):
        """
        Read origin time UTC from ShakeMap grid.xml (best effort).
        Falls back to self.event.origin_time_utc if already loaded.
        """
        import os
        import xml.etree.ElementTree as ET

        # If event already loaded, prefer it
        try:
            ot = getattr(self.event, "origin_time_utc", None)
            if ot is not None:
                return ot
        except Exception:
            pass

        xp = self._wilber__as_path(xml_path)
        if not os.path.isfile(xp):
            raise FileNotFoundError(f"XML not found for origin time: {xp}")

        tree = ET.parse(xp)
        root = tree.getroot()

        ev = root.find(".//event")
        if ev is not None:
            t = ev.attrib.get("time") or ev.attrib.get("origin_time") or ev.attrib.get("event_time")
            if t:
                dtv = self._wilber__parse_iso(t)
                if dtv:
                    return dtv

        raise ValueError("Could not find origin time in XML. Provide origin_time_utc explicitly.")




    def _wilber__read_origin_from_xml(self, xml_path):
        """
        Read origin time UTC from ShakeMap grid.xml (best effort, namespace-tolerant).

        ShakeMap grid.xml typically stores origin time on the <event ...> element as:
          - event_timestamp="YYYY-MM-DDTHH:MM:SS[.sss]Z"
        Some variants may also include: time, origin_time, event_time.

        Returns: naive UTC datetime (see _wilber__parse_iso).
        """
        import os
        import xml.etree.ElementTree as ET

        xp = self._wilber__as_path(xml_path)
        if not os.path.isfile(xp):
            raise FileNotFoundError(f"XML not found for origin time: {xp}")

        # If self already loaded event metadata, prefer it (fast + consistent)
        try:
            if getattr(self, "event", None) is not None:
                for attr in ("origin_time_utc", "event_time_utc"):
                    ot = getattr(self.event, attr, None)
                    if hasattr(ot, "isoformat"):
                        return self._wilber__parse_iso(ot)

                # most likely present in your codebase after load_from_xml()
                t = getattr(self.event, "event_time_str", None)
                if t:
                    dtv = self._wilber__parse_iso(t)
                    if dtv:
                        return dtv
        except Exception:
            pass

        # Parse XML directly (namespace tolerant)
        try:
            tree = ET.parse(xp)
            root = tree.getroot()

            ev = None
            for node in root.iter():
                try:
                    if str(node.tag).lower().endswith("event"):
                        ev = node
                        break
                except Exception:
                    continue

            if ev is not None:
                for key in ("event_timestamp", "time", "origin_time", "event_time"):
                    t = ev.attrib.get(key)
                    if t:
                        dtv = self._wilber__parse_iso(t)
                        if dtv:
                            return dtv
        except Exception:
            pass

        # Last resort: let SHAKEpropagate's own XML loader populate event_time_str
        try:
            if hasattr(self, "load_from_xml"):
                self.load_from_xml()
                if getattr(self, "event", None) is not None:
                    t = getattr(self.event, "event_time_str", None)
                    if t:
                        dtv = self._wilber__parse_iso(t)
                        if dtv:
                            return dtv
        except Exception:
            pass

        raise ValueError(
            "Could not find origin time in XML. Expected <event ... event_timestamp='...'> "
            "(or time/origin_time/event_time). Provide origin_time_utc explicitly if needed."
        )



        
    # override 
    def _wilber__get_origin_time(self, origin_time_utc, xml_path_for_origin):
        """
        Resolve origin time:
          - datetime -> return
          - "from_xml" -> parse from xml_path_for_origin
          - ISO string -> parse
        """
        if hasattr(origin_time_utc, "isoformat"):
            return origin_time_utc
        if isinstance(origin_time_utc, str) and origin_time_utc.strip().lower() == "from_xml":
            if not xml_path_for_origin:
                raise ValueError("origin_time_utc='from_xml' requires xml_path_for_origin.")
            return self._wilber__read_origin_from_xml(xml_path_for_origin)
        out = self._wilber__parse_iso(origin_time_utc)
        if out is None:
            raise ValueError("Could not resolve origin_time_utc. Provide datetime/ISO string or 'from_xml'.")
        return out




    def _wilber__get_origin_time(self, origin_time_utc, xml_path_for_origin):
        """
        Resolve origin time:
          - datetime -> return (normalized to naive UTC)
          - None / 'from_xml' -> parse from xml_path_for_origin (or self.inputs.shakemap_xml)
          - ISO string -> parse (normalized to naive UTC)
        """
        # datetime-like
        if hasattr(origin_time_utc, "isoformat"):
            return self._wilber__parse_iso(origin_time_utc)

        # treat None as from_xml (prevents "broken pipeline" when user forgets)
        if origin_time_utc is None:
            origin_time_utc = "from_xml"

        # from_xml mode
        if isinstance(origin_time_utc, str) and origin_time_utc.strip().lower() == "from_xml":
            xp = xml_path_for_origin
            if not xp:
                try:
                    xp = getattr(getattr(self, "inputs", None), "shakemap_xml", None)
                except Exception:
                    xp = None
            if not xp:
                raise ValueError(
                    "origin_time_utc='from_xml' requires xml_path_for_origin "
                    "(or self.inputs.shakemap_xml)."
                )
            return self._wilber__read_origin_from_xml(xp)

        # ISO string mode
        out = self._wilber__parse_iso(origin_time_utc)
        if out is None:
            raise ValueError("Could not resolve origin_time_utc. Provide datetime/ISO string, None, or 'from_xml'.")
        return out




    def _wilber__pick_window_from_origin(self, trace_start_utc, origin_utc, t_obs_window_s):
        """
        Convert desired t_obs window relative to origin (e.g. [0,+300]) into
        a pick window relative to trace start.
        """
        dt0 = (origin_utc - trace_start_utc).total_seconds()
        a = float(t_obs_window_s[0])
        b = float(t_obs_window_s[1])
        min_pick_s = max(0.0, dt0 + a)
        max_pick_s = max(min_pick_s, dt0 + b)
        return float(min_pick_s), float(max_pick_s)

    def _wilber__interp_T_at(self, T_map, lons, lats, method="nearest"):
        """
        Interpolate T_map (seconds) at lon/lat points.
        Uses nearest (robust) by default.
        """
        import numpy as np
        self._wilber__require_grids()
        lons = np.asarray(lons, float)
        lats = np.asarray(lats, float)

        if method == "nearest":
            out = np.full(len(lons), np.nan, float)
            for i, (lo, la) in enumerate(zip(lons, lats)):
                jj = int(np.argmin(np.abs(self.lon_grid[0, :] - lo)))
                ii = int(np.argmin(np.abs(self.lat_grid[:, 0] - la)))
                out[i] = float(T_map[ii, jj])
            return out

        return self._wilber__interp_T_at(T_map, lons, lats, method="nearest")

    def _wilber__get_shakemap_field(self, name="MMI"):
        """
        Get a ShakeMap raster field for background plotting.
        Prefers existing self.im_grid if available; else tries internal dict-like stores.
        """
        self._wilber__require_grids()

        if hasattr(self, "im_grid") and getattr(self, "im_grid") is not None:
            return getattr(self, "im_grid")

        for attr in ["fields", "grid_fields", "data_fields", "_fields", "field_grids"]:
            if hasattr(self, attr):
                obj = getattr(self, attr)
                try:
                    if isinstance(obj, dict) and name in obj:
                        return obj[name]
                except Exception:
                    pass

        return None

    # -------------------------------------------------------------------------
    # A1) Soft filter helper: multi-channel clustering at a station
    # -------------------------------------------------------------------------
    def _wilber__soft_cluster_channel_picks(self, ok_df, *, soft_cluster_tol_s=3.0):
        """
        Given ok_df (per-channel picks) for ONE station, cluster by t_obs_s.
        Rule:
          - If at least 2 channels cluster within tol -> select that cluster
          - Else -> no cluster (return best single pick; caller may mark inconsistent)
        Returns: (df_cluster, meta_dict)
        meta_dict keys:
          soft_cluster_n, soft_cluster_spread_s, soft_cluster_used_chans
        """
        import numpy as np

        df = ok_df.copy()
        if df.empty:
            return df, {"soft_cluster_n": 0, "soft_cluster_spread_s": np.nan, "soft_cluster_used_chans": ""}

        tol = float(soft_cluster_tol_s)
        tt = df["t_obs_s"].astype(float).values
        ch = df["chan"].astype(str).values

        # sort by time
        order = np.argsort(tt)
        tt = tt[order]
        df = df.iloc[order].reset_index(drop=True)

        # simple 1D clustering (single-link) by consecutive gaps <= tol
        clusters = []
        start = 0
        for i in range(1, len(tt)):
            if (tt[i] - tt[i - 1]) > tol:
                clusters.append((start, i - 1))
                start = i
        clusters.append((start, len(tt) - 1))

        # compute sizes; choose max size; tie-break: smallest spread, then earliest mean
        cand = []
        for (a, b) in clusters:
            n = (b - a + 1)
            if n <= 0:
                continue
            tseg = tt[a:b + 1]
            spread = float(np.nanmax(tseg) - np.nanmin(tseg)) if n > 1 else 0.0
            mean = float(np.nanmean(tseg))
            cand.append((n, spread, mean, a, b))

        if len(cand) == 0:
            return df.iloc[:1].copy(), {"soft_cluster_n": 1, "soft_cluster_spread_s": 0.0, "soft_cluster_used_chans": str(df.iloc[0]["chan"])}

        cand.sort(key=lambda x: (-x[0], x[1], x[2]))  # max n, min spread, earliest mean
        n_best, spread_best, mean_best, a, b = cand[0]

        if int(n_best) < 2:
            # no usable cluster
            one = df.iloc[:1].copy()
            return one, {"soft_cluster_n": 1, "soft_cluster_spread_s": 0.0, "soft_cluster_used_chans": str(one.iloc[0]["chan"])}

        dfc = df.iloc[a:b + 1].copy()
        used = ",".join(dfc["chan"].astype(str).tolist())
        return dfc, {"soft_cluster_n": int(n_best), "soft_cluster_spread_s": float(spread_best), "soft_cluster_used_chans": used}

    # -------------------------------------------------------------------------
    # A2) OPTIONAL: ObsPy utilities (safe fallback if missing/errors)
    # -------------------------------------------------------------------------
    def _wilber__obspy_available(self):
        try:
            import obspy  # noqa: F401
            return True, ""
        except Exception as e:
            return False, str(e)

    def _wilber__obspy_trace_from_arrays(self, x, sps, t0_utc, *, net="", sta="", loc="", chan=""):
        """
        Build an ObsPy Trace from numpy array + metadata.
        Raises ImportError if obspy missing.
        """
        import numpy as np
        ok, err = self._wilber__obspy_available()
        if not ok:
            raise ImportError(err)

        from obspy import Trace, UTCDateTime
        tr = Trace(data=np.asarray(x, dtype=np.float32))
        tr.stats.sampling_rate = float(sps)
        tr.stats.starttime = UTCDateTime(t0_utc.isoformat())
        tr.stats.network = str(net or "")
        tr.stats.station = str(sta or "")
        tr.stats.location = str(loc or "")
        tr.stats.channel = str(chan or "")
        return tr

    def _wilber__obspy_preprocess(self, tr, preprocess):
        """
        preprocess:
          - None: no-op
          - "default": detrend + demean + taper + bandpass(0.5, min(20, 0.45*fs))
          - dict: keys {detrend:bool, demean:bool, taper:dict|bool, filter:{type:..., ...}}
        """
        t = tr.copy()
        if preprocess is None:
            return t

        if isinstance(preprocess, str) and preprocess.lower() == "default":
            try: t.detrend("linear")
            except Exception: pass
            try: t.detrend("demean")
            except Exception: pass
            try: t.taper(max_percentage=0.05)
            except Exception: pass
            try:
                fs = float(t.stats.sampling_rate)
                t.filter("bandpass", freqmin=0.5, freqmax=min(20.0, 0.45 * fs))
            except Exception:
                pass
            return t

        if isinstance(preprocess, dict):
            try:
                if preprocess.get("detrend", False):
                    t.detrend("linear")
            except Exception:
                pass
            try:
                if preprocess.get("demean", False):
                    t.detrend("demean")
            except Exception:
                pass
            try:
                tp = preprocess.get("taper", None)
                if tp:
                    if isinstance(tp, dict):
                        t.taper(**tp)
                    else:
                        t.taper(max_percentage=0.05)
            except Exception:
                pass
            try:
                ff = preprocess.get("filter", None)
                if isinstance(ff, dict) and ff.get("type", None):
                    ftype = ff.get("type")
                    kwargs = {k: v for k, v in ff.items() if k != "type"}
                    t.filter(ftype, **kwargs)
            except Exception:
                pass
            return t

        return t

    def wilber_pick_arrival_obspy_stalta(
        self,
        t_s,
        x,
        sps,
        *,
        min_pick_s=0.0,
        max_pick_s=None,
        sta_s=0.5,
        lta_s=10.0,
        trig_on=3.5,
        trig_off=1.0,
        preprocess="default",   # None | "default" | dict
        pick_mode="first_on",   # "first_on" | "max_ratio"
    ):
        """
        ObsPy classic STA/LTA + trigger_onset.
        If ObsPy missing/errors -> raises; caller should fallback.
        Returns dict compatible with wilber_pick_arrival_stalta().
        """
        import numpy as np
        from obspy.signal.trigger import classic_sta_lta, trigger_onset

        # Build trace just for preprocessing convenience
        tr = self._wilber__obspy_trace_from_arrays(x, sps, t0_utc=self._wilber__parse_iso("1970-01-01T00:00:00"))
        tr = self._wilber__obspy_preprocess(tr, preprocess)

        data = np.asarray(tr.data, float)
        nsta = max(1, int(round(float(sta_s) * float(sps))))
        nlta = max(nsta + 1, int(round(float(lta_s) * float(sps))))

        cft = classic_sta_lta(data, nsta, nlta)

        t_s = np.asarray(t_s, float)
        lo = float(min_pick_s) if min_pick_s is not None else 0.0
        hi = float(max_pick_s) if max_pick_s is not None else float(t_s[-1])

        i0 = int(max(0, np.floor(lo * float(sps))))
        i1 = int(min(len(cft) - 1, np.ceil(hi * float(sps))))
        if i1 < i0:
            i1 = i0

        cft_w = np.full_like(cft, np.nan, dtype=float)
        cft_w[i0:i1 + 1] = cft[i0:i1 + 1]

        finite = np.isfinite(cft_w)
        if not np.any(finite):
            return {"picked": False, "t_pick_rel_s": np.nan, "idx": -1, "ratio_max": np.nan}

        ratio_max = float(np.nanmax(cft_w))

        mode = str(pick_mode).lower().strip()
        if mode == "max_ratio":
            idx = int(np.nanargmax(cft_w))
            return {"picked": True, "t_pick_rel_s": float(t_s[idx]), "idx": idx, "ratio_max": ratio_max}

        # first trigger onset window
        on_off = trigger_onset(np.nan_to_num(cft_w, nan=-1e9), float(trig_on), float(trig_off))
        if on_off is None or len(on_off) == 0:
            return {"picked": False, "t_pick_rel_s": np.nan, "idx": -1, "ratio_max": ratio_max}

        idx = int(on_off[0][0])
        return {"picked": True, "t_pick_rel_s": float(t_s[idx]), "idx": idx, "ratio_max": ratio_max}

    def wilber_pick_arrival_obspy_ar(
        self,
        t_s,
        x,
        sps,
        *,
        min_pick_s=0.0,
        max_pick_s=None,
        preprocess="default",
        # ar_pick knobs
        f1=1.0, f2=20.0,
        lta_p=10.0, sta_p=1.0,
        lta_s=10.0, sta_s=1.0,
        m_p=2, m_s=8,
        l_p=0.5, l_s=0.5,
    ):
        """
        ObsPy ar_pick picker (returns P and S picks; we use earliest within window).
        If ObsPy missing/errors -> raises; caller should fallback.
        """
        import numpy as np
        from obspy.signal.trigger import ar_pick

        # build trace for preprocessing
        tr = self._wilber__obspy_trace_from_arrays(x, sps, t0_utc=self._wilber__parse_iso("1970-01-01T00:00:00"))
        tr = self._wilber__obspy_preprocess(tr, preprocess)
        data = np.asarray(tr.data, float)

        # ar_pick wants 3 components often; for single channel we pass same three
        p_pick, s_pick = ar_pick(
            data, data, data, float(sps),
            float(f1), float(f2),
            float(lta_p), float(sta_p),
            float(lta_s), float(sta_s),
            int(m_p), int(m_s),
            float(l_p), float(l_s),
        )

        # picks are seconds from start (float) or None
        candidates = []
        if p_pick is not None and np.isfinite(p_pick):
            candidates.append(float(p_pick))
        if s_pick is not None and np.isfinite(s_pick):
            candidates.append(float(s_pick))

        if len(candidates) == 0:
            return {"picked": False, "t_pick_rel_s": np.nan, "idx": -1, "ratio_max": np.nan}

        # enforce bounds
        lo = float(min_pick_s) if min_pick_s is not None else 0.0
        hi = float(max_pick_s) if max_pick_s is not None else float(t_s[-1])
        cand_in = [c for c in candidates if lo <= c <= hi]
        if len(cand_in) == 0:
            return {"picked": False, "t_pick_rel_s": np.nan, "idx": -1, "ratio_max": np.nan}

        t_pick = min(cand_in)
        idx = int(np.argmin(np.abs(np.asarray(t_s, float) - float(t_pick))))
        return {"picked": True, "t_pick_rel_s": float(t_s[idx]), "idx": idx, "ratio_max": np.nan}

    # -------------------------------------------------------------------------
    # B) Station list reader (Wilber)
    # -------------------------------------------------------------------------
    def wilber_read_station_list(self, station_txt_path, *, return_pandas=True, log=True):
        """
        Read Wilber station list (pipe-separated) into DataFrame:
          net, sta, lat, lon, elev_m, distance_deg, azimuth_deg, name, station_id
        """
        import numpy as np
        p = self._wilber__as_path(station_txt_path)

        rows = []
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip()
            cols = [c.strip() for c in header.split("|")]
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [x.strip() for x in line.split("|")]
                while len(parts) < len(cols):
                    parts.append("")
                row = dict(zip(cols, parts))

                net = str(row.get("Net", "")).strip()
                sta = str(row.get("Station", "")).strip()
                lat = float(row.get("Latitude"))
                lon = float(row.get("Longitude"))
                elev = row.get("Elevation", "")
                dist = row.get("Distance", "")
                az = row.get("Azimuth", "")
                name = str(row.get("Station Name", "")).strip()

                rows.append({
                    "net": net,
                    "sta": sta,
                    "station_id": f"{net}.{sta}",
                    "lat": lat,
                    "lon": lon,
                    "elev_m": float(elev) if str(elev).strip() else np.nan,
                    "distance_deg": float(dist) if str(dist).strip() else np.nan,
                    "azimuth_deg": float(az) if str(az).strip() else np.nan,
                    "name": name,
                })

        if return_pandas:
            import pandas as pd
            df = pd.DataFrame(rows)
            if log:
                self._wilber__log(f"[wilber] station list loaded: {len(df)} stations from {p}")
            return df
        if log:
            self._wilber__log(f"[wilber] station list loaded: {len(rows)} stations from {p}")
        return rows

    # -------------------------------------------------------------------------
    # C) Waveform folder indexer (TIMESERIES headers)
    # -------------------------------------------------------------------------
    def _wilber_read_timeseries_header(self, filepath):
        """
        Read first line (TIMESERIES ...) and parse:
          id, nsamp, sps, starttime
        Returns dict {valid, net, sta, loc, chan, nsamp, sps, t0_utc}
        """
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()

        if not first.upper().startswith("TIMESERIES"):
            return {"valid": False, "raw": first}

        bits = [b.strip() for b in first.split(",")]
        if len(bits) < 4:
            return {"valid": False, "raw": first}

        id_part = bits[0].split(None, 1)
        ts_id = id_part[1].strip() if len(id_part) > 1 else ""
        parts = ts_id.split("_")
        net = parts[0] if len(parts) > 0 else ""
        sta = parts[1] if len(parts) > 1 else ""
        loc = parts[2] if len(parts) > 2 else ""
        chan = parts[3] if len(parts) > 3 else ""

        try:
            nsamp = int(bits[1].split()[0])
        except Exception:
            nsamp = None
        try:
            sps = float(bits[2].split()[0])
        except Exception:
            sps = None

        t0 = self._wilber__parse_iso(bits[3])

        return {
            "valid": True,
            "net": str(net).strip(),
            "sta": str(sta).strip(),
            "loc": str(loc).strip(),
            "chan": str(chan).strip(),
            "nsamp": nsamp,
            "sps": sps,
            "t0_utc": t0,
            "header_raw": first,
        }

    def wilber_index_waveform_folder(
        self,
        waveform_dir,
        *,
        extensions=(".txt", ".asc", ".ascii", ".dat", ""),
        log=True,
        return_pandas=True,
    ):
        """
        Scan folder and parse TIMESERIES header for each file.
        Returns DataFrame with:
          filepath, net, sta, loc, chan, t0_utc, sps, nsamp, valid_header, header_raw
        """
        import os
        d = self._wilber__as_path(waveform_dir)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Waveform directory not found: {d}")

        exts = set([e.lower() for e in extensions])
        rows = []
        for fn in sorted(os.listdir(d)):
            fp = os.path.join(d, fn)
            if not os.path.isfile(fp):
                continue
            _, ext = os.path.splitext(fn)
            ext = ext.lower()
            if ext not in exts and not (ext == "" and "" in exts):
                continue

            hdr = self._wilber_read_timeseries_header(fp)
            rows.append({
                "filepath": fp,
                "net": hdr.get("net", ""),
                "sta": hdr.get("sta", ""),
                "loc": hdr.get("loc", ""),
                "chan": hdr.get("chan", ""),
                "t0_utc": hdr.get("t0_utc", None),
                "sps": hdr.get("sps", None),
                "nsamp": hdr.get("nsamp", None),
                "valid_header": bool(hdr.get("valid", False)),
                "header_raw": hdr.get("header_raw", ""),
            })

        if return_pandas:
            import pandas as pd
            df = pd.DataFrame(rows)
            if log:
                self._wilber__log(f"[wilber] waveform index: {len(df)} files scanned in {d}")
                self._wilber__log(f"[wilber] valid TIMESERIES headers: {int(df['valid_header'].sum()) if 'valid_header' in df else len(df)}")
            return df
        if log:
            self._wilber__log(f"[wilber] waveform index: {len(rows)} files scanned in {d}")
        return rows

    # -------------------------------------------------------------------------
    # D) Inventory report (station list vs folder)
    # -------------------------------------------------------------------------
    def wilber_inventory_report(
        self,
        station_list_path,
        waveform_dir,
        *,
        channel_family_filter=None,
        prefer_chans=("HHZ", "BHZ", "EHZ", "HNZ"),
        export_dir=None,
        export_prefix="wilber_inventory",
        log=True,
    ):
        """
        Cross-check stations listed vs stations with waveforms.
        Returns: stations_df, wave_index_df, inv_dict
        """
        import os
        import pandas as pd

        stations_df = self.wilber_read_station_list(station_list_path, return_pandas=True, log=log)
        wave_index_df = self.wilber_index_waveform_folder(waveform_dir, return_pandas=True, log=log)

        wi = wave_index_df.copy()
        wi = wi[wi["valid_header"] == True].copy()
        wi["net"] = wi["net"].astype(str).str.strip()
        wi["sta"] = wi["sta"].astype(str).str.strip()
        wi["chan"] = wi["chan"].astype(str).str.strip()

        if channel_family_filter:
            fam = str(channel_family_filter).upper().strip()
            wi = wi[wi["chan"].str.upper().str.startswith(fam)].copy()

        st_keys = set((net, sta) for (net, sta) in stations_df[["net", "sta"]].itertuples(index=False, name=None))
        wf_keys = set((net, sta) for (net, sta) in wi[["net", "sta"]].drop_duplicates().itertuples(index=False, name=None))

        in_list_not_folder = sorted(st_keys - wf_keys)
        in_folder_not_list = sorted(wf_keys - st_keys)

        prefer_set = set([c.upper() for c in (prefer_chans or ())])
        wi["_cU"] = wi["chan"].str.upper()
        wf_with_pref = wi[wi["_cU"].isin(prefer_set)][["net", "sta"]].drop_duplicates()
        n_with_pref = int(len(wf_with_pref))

        inv = {
            "N_stations_listed": int(len(stations_df)),
            "N_unique_waveform_stations": int(len(wi[["net", "sta"]].drop_duplicates())),
            "N_waveform_files_valid": int(len(wi)),
            "N_list_not_folder": int(len(in_list_not_folder)),
            "N_folder_not_list": int(len(in_folder_not_list)),
            "N_waveform_stations_with_preferred_channel": int(n_with_pref),
            "stations_in_list_not_in_folder": in_list_not_folder,
            "stations_in_folder_not_in_list": in_folder_not_list,
        }

        if log:
            self._wilber__log(f"[wilber-qc] listed stations: {inv['N_stations_listed']}")
            self._wilber__log(f"[wilber-qc] waveform stations: {inv['N_unique_waveform_stations']}")
            self._wilber__log(f"[wilber-qc] list NOT in folder: {inv['N_list_not_folder']}")
            self._wilber__log(f"[wilber-qc] folder NOT in list: {inv['N_folder_not_list']}")
            self._wilber__log(f"[wilber-qc] stations with preferred channel: {inv['N_waveform_stations_with_preferred_channel']}")

        if export_dir:
            out = self._wilber__ensure_dir(self._wilber__as_path(export_dir))
            stations_df.to_csv(os.path.join(out, f"{export_prefix}_stations.csv"), index=False)
            wave_index_df.to_csv(os.path.join(out, f"{export_prefix}_wave_index.csv"), index=False)
            pd.DataFrame(in_list_not_folder, columns=["net", "sta"]).to_csv(os.path.join(out, f"{export_prefix}_list_not_in_folder.csv"), index=False)
            pd.DataFrame(in_folder_not_list, columns=["net", "sta"]).to_csv(os.path.join(out, f"{export_prefix}_folder_not_in_list.csv"), index=False)

        return stations_df, wave_index_df, inv

    # -------------------------------------------------------------------------
    # E) Waveform reader (robust) + internal STA/LTA picker
    # -------------------------------------------------------------------------
    def wilber_read_timeseries(self, filepath, *, max_samples=None, dtype=float):
        """
        Read Wilber TIMESERIES ASCII file.

        Robustness:
          - Reads amplitude column always.
          - Detects which column looks like a timestamp (tries first two columns).
          - Lightly parses timestamps (first + last) to validate header t0 and infer sps if needed.
          - Falls back safely to header SPS indexing if timestamps unparseable.
        """
        import numpy as np

        hdr = self._wilber_read_timeseries_header(filepath)
        if not hdr.get("valid", False):
            raise ValueError(f"Not a valid Wilber TIMESERIES file: {filepath}")

        net = hdr["net"]; sta = hdr["sta"]; loc = hdr["loc"]; chan = hdr["chan"]
        t0 = hdr["t0_utc"]
        sps = float(hdr["sps"]) if hdr.get("sps") else None
        if t0 is None or sps is None or sps <= 0:
            raise ValueError(f"Invalid header (t0 or sps) for {filepath}")

        vals = []
        t_first = None
        t_last = None
        timestamp_col = None
        amp_col = None

        def _try_parse_time(token):
            return self._wilber__parse_iso(token)

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            _ = f.readline()  # header
            for k, line in enumerate(f):
                if max_samples is not None and k >= int(max_samples):
                    break
                line = line.strip()
                if not line:
                    continue
                pp = line.split()
                if len(pp) < 2:
                    continue

                # detect timestamp column early (best effort)
                if timestamp_col is None:
                    # try pp[0] then pp[1]
                    tA = _try_parse_time(pp[0])
                    tB = _try_parse_time(pp[1]) if len(pp) > 1 else None
                    if tA is not None:
                        timestamp_col = 0
                        amp_col = 1
                    elif tB is not None and len(pp) > 2:
                        timestamp_col = 1
                        amp_col = 2
                    else:
                        # no obvious timestamp; assume amp in pp[1]
                        timestamp_col = -1
                        amp_col = 1

                if timestamp_col >= 0:
                    if t_first is None:
                        t_first = _try_parse_time(pp[timestamp_col])
                    t_last = _try_parse_time(pp[timestamp_col]) or t_last

                try:
                    vals.append(float(pp[amp_col]))
                except Exception:
                    continue

        x = np.asarray(vals, dtype=dtype)

        # timestamp consistency checks (only if timestamps were actually parsed)
        if t_first is not None:
            dt_err = abs((t_first - t0).total_seconds())
            if dt_err > 2.0:
                # prefer the actual first sample timestamp if header seems off
                t0 = t_first

            if t_last is not None and len(x) > 10:
                dur = (t_last - t0).total_seconds()
                if dur > 0:
                    sps_eff = (len(x) - 1) / dur
                    if np.isfinite(sps_eff) and 0.5 * sps <= sps_eff <= 2.0 * sps:
                        sps = float(sps_eff)

        t_s = np.arange(len(x), dtype=float) / float(sps)

        return {
            "filepath": filepath,
            "net": net, "sta": sta, "loc": loc, "chan": chan,
            "t0_utc": t0, "sps": float(sps),
            "t_s": t_s, "x": x,
        }

    def wilber_pick_arrival_stalta(
        self,
        t_s,
        x,
        sps,
        *,
        sta_s=0.5,
        lta_s=10.0,
        trig_on=3.5,
        min_pick_s=0.0,
        max_pick_s=None,
        detrend=True,
        envelope="rms",      # "abs" or "rms"
        rms_win_s=0.2,
        pick_mode="first_on",  # "first_on" | "max_ratio"
    ):
        """
        Internal STA/LTA picker on envelope within [min_pick_s, max_pick_s].
        """
        import numpy as np

        xx = np.asarray(x, float)
        if detrend:
            xx = xx - np.nanmedian(xx)

        if str(envelope).lower() == "rms":
            w = max(1, int(round(float(rms_win_s) * float(sps))))
            ker = np.ones(w, float) / float(w)
            env = np.sqrt(np.convolve(xx * xx, ker, mode="same"))
        else:
            env = np.abs(xx)

        n_sta = max(1, int(round(float(sta_s) * float(sps))))
        n_lta = max(n_sta + 1, int(round(float(lta_s) * float(sps))))

        c = np.cumsum(env, dtype=float)
        c = np.insert(c, 0, 0.0)

        def movmean(n):
            out = (c[n:] - c[:-n]) / float(n)
            return np.r_[np.full(n - 1, np.nan), out]

        sta = movmean(n_sta)
        lta = movmean(n_lta)
        ratio = sta / (lta + 1e-12)

        t_s = np.asarray(t_s, float)
        lo = float(min_pick_s) if min_pick_s is not None else 0.0
        hi = float(max_pick_s) if max_pick_s is not None else float(t_s[-1])

        mask = (t_s >= lo) & (t_s <= hi)
        ratio_w = np.where(mask, ratio, np.nan)

        finite = np.isfinite(ratio_w)
        if not np.any(finite):
            return {"picked": False, "t_pick_rel_s": np.nan, "idx": -1, "ratio_max": np.nan}

        ratio_max = float(np.nanmax(ratio_w))

        mode = str(pick_mode).lower().strip()
        if mode == "max_ratio":
            i0 = int(np.nanargmax(ratio_w))
            return {"picked": True, "t_pick_rel_s": float(t_s[i0]), "idx": i0, "ratio_max": ratio_max}

        idx_on = np.where(ratio_w >= float(trig_on))[0]
        if idx_on.size == 0:
            return {"picked": False, "t_pick_rel_s": np.nan, "idx": -1, "ratio_max": ratio_max}

        i0 = int(idx_on[0])
        return {"picked": True, "t_pick_rel_s": float(t_s[i0]), "idx": i0, "ratio_max": ratio_max}

    # -------------------------------------------------------------------------
    # F0) OPTIONAL: station outlier filtering (distance vs arrival-time)
    # -------------------------------------------------------------------------
    def _wilber__station_outlier_filter(
        self,
        arrivals_df,
        *,
        distance_km=None,
        use_only_picked=True,
        robust_iter=2,
        z_mad=3.5,
        hard_filter=True,
        hard_invalidate=None,   # 
        plot=False,
        plot_path=None,
        title="Arrival-time vs distance outlier filter",
        log=True,
    ):
        """
        Flags outliers based on robust linear fit of t_obs_s vs distance_km, using MAD of residuals.
        - Does NOT drop stations.
        - Always keeps flags in dataframe.
        - If hard_invalidate (or hard_filter) is True: sets picked=False and clears t_obs_s for outliers.
        Adds/keeps columns:
          outlier_flag, outlier_score, residual_s, t_pred_fit_s
        Returns (arrivals_df_out, info_dict).
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        df = arrivals_df.copy()

        # override hard_filter if hard_invalidate explicitly provided
        if hard_invalidate is not None:
            hard_filter = bool(hard_invalidate)

        if distance_km is None:
            # compute epicentral distance
            try:
                d_km = haversine_km(
                    df["lon"].astype(float).values,
                    df["lat"].astype(float).values,
                    float(self.event.epicenter_lon),
                    float(self.event.epicenter_lat),
                )
            except Exception:
                d_km = np.full(len(df), np.nan, float)
            df["epicentral_distance_km"] = d_km
        else:
            df["epicentral_distance_km"] = np.asarray(distance_km, float)

        # initialize outputs (do not remove existing values if present)
        if "outlier_flag" not in df.columns:
            df["outlier_flag"] = False
        else:
            df["outlier_flag"] = df["outlier_flag"].astype(bool)

        if "outlier_score" not in df.columns:
            df["outlier_score"] = np.nan

        if "residual_s" not in df.columns:
            df["residual_s"] = np.nan

        if "t_pred_fit_s" not in df.columns:
            df["t_pred_fit_s"] = np.nan

        mask_base = np.isfinite(df["epicentral_distance_km"].astype(float).values) & np.isfinite(df["t_obs_s"].astype(float).values)
        if use_only_picked and "picked" in df.columns:
            mask_base = mask_base & df["picked"].astype(bool).values

        x = df.loc[mask_base, "epicentral_distance_km"].astype(float).values
        y = df.loc[mask_base, "t_obs_s"].astype(float).values

        info = {
            "used_points": int(len(x)),
            "z_mad": float(z_mad),
            "hard_filter": bool(hard_filter),
            "n_flagged": 0,
            "slope": np.nan,
            "intercept": np.nan,
            "mad_s": np.nan,
        }

        if len(x) < 8:
            if log:
                self._wilber__log(f"[wilber-outlier] Not enough points for outlier fit (N={len(x)}). Skipping.")
            return df, info

        # iterative robust fit
        keep = np.ones(len(x), bool)
        a = b = np.nan
        for _ in range(int(max(1, robust_iter))):
            xx = x[keep]; yy = y[keep]
            if len(xx) < 6:
                break
            # simple linear fit
            b, a = np.polyfit(xx, yy, 1)  # yy ~ b*xx + a
            yhat = b * xx + a
            res = yy - yhat
            med = np.nanmedian(res)
            mad = np.nanmedian(np.abs(res - med))
            scale = 1.4826 * mad if mad > 0 else np.nanstd(res)
            if not np.isfinite(scale) or scale <= 0:
                break
            z = np.abs(res - med) / scale
            keep = z <= float(z_mad)

        # compute final residual scores for all used points
        b, a = np.polyfit(x[keep] if np.any(keep) else x, y[keep] if np.any(keep) else y, 1)
        yhat_all = b * x + a
        res_all = y - yhat_all
        med = np.nanmedian(res_all)
        mad = np.nanmedian(np.abs(res_all - med))
        scale = 1.4826 * mad if mad > 0 else np.nanstd(res_all)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        z_all = np.abs(res_all - med) / scale
        flagged = z_all > float(z_mad)

        # write outputs back to df in the original row order for mask_base
        base_index = df.loc[mask_base].index
        df.loc[base_index, "t_pred_fit_s"] = yhat_all
        df.loc[base_index, "residual_s"] = res_all
        df.loc[base_index, "outlier_score"] = z_all
        df.loc[base_index, "outlier_flag"] = flagged

        info["n_flagged"] = int(np.sum(flagged))
        info["slope"] = float(b)
        info["intercept"] = float(a)
        info["mad_s"] = float(scale)

        # Invalidate if requested (never drop)
        if info["n_flagged"] > 0 and hard_filter:
            mflag = df["outlier_flag"].astype(bool).values
            if "picked" in df.columns:
                df.loc[mflag, "picked"] = False
            df.loc[mflag, "t_obs_s"] = np.nan
            if "t_pick_utc" in df.columns:
                df.loc[mflag, "t_pick_utc"] = pd.NaT
            # append note
            if "notes" in df.columns:
                df.loc[mflag, "notes"] = df.loc[mflag, "notes"].astype(str).str.cat(["OUTLIER_FILTERED"], sep="; ")
            else:
                df["notes"] = ""
                df.loc[mflag, "notes"] = "OUTLIER_FILTERED"

        if plot:
            try:
                fig = plt.figure(figsize=(9.5, 6.5), dpi=200)
                ax = plt.gca()
                ax.scatter(x, y, s=18, alpha=0.55, label="picked stations")
                ax.scatter(x[flagged], y[flagged], s=30, alpha=0.95, marker="x", label="outliers")
                xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                ax.plot(xx, b * xx + a, linewidth=1.2, label="robust fit")
                ax.set_xlabel("Epicentral distance (km)")
                ax.set_ylabel("Observed arrival time t_obs (s)")
                ax.set_title(title)
                ax.legend(loc="best")
                ax.grid(True, alpha=0.25)
                if plot_path:
                    self._wilber__ensure_dir(__import__("os").path.dirname(self._wilber__as_path(plot_path)))
                    fig.savefig(self._wilber__as_path(plot_path), bbox_inches="tight")
                    plt.close(fig)
                else:
                    plt.show()
            except Exception as e:
                if log:
                    self._wilber__log(f"[wilber-outlier] Plot failed: {e}")

        if log:
            self._wilber__log(f"[wilber-outlier] fit: t = {info['slope']:.4f}*d + {info['intercept']:.2f} (MAD scale~{info['mad_s']:.2f}s), flagged={info['n_flagged']} invalidate={bool(hard_filter)}")

        return df, info

    # -------------------------------------------------------------------------
    # F0b) PUBLIC WRAPPER: keep notebooks from breaking
    # -------------------------------------------------------------------------
    def wilber_filter_station_outliers_distance_time(
        self,
        arrivals_df,
        *,
        distance_km=None,
        use_only_picked=True,
        robust_iter=2,
        z_mad=3.5,
        hard_filter=True,
        hard_invalidate=None,
        plot=False,
        plot_path=None,
        title="Arrival-time vs distance outlier filter",
        log=True,
    ):
        """
        Public wrapper for distance–time outlier filtering (backward-compatible name).
        Calls _wilber__station_outlier_filter().
        """
        return self._wilber__station_outlier_filter(
            arrivals_df,
            distance_km=distance_km,
            use_only_picked=use_only_picked,
            robust_iter=robust_iter,
            z_mad=z_mad,
            hard_filter=hard_filter,
            hard_invalidate=hard_invalidate,
            plot=plot,
            plot_path=plot_path,
            title=title,
            log=log,
        )



    
    # -------------------------------------------------------------------------
    # F) Build arrivals dataset (ALL stations kept; mismatches flagged)
    # -------------------------------------------------------------------------
    # override 
    def wilber_build_arrivals_dataset(
        self,
        station_list_path,
        waveform_dir,
        *,
        origin_time_utc="from_xml",
        xml_path_for_origin=None,
        phase="S",
        t_obs_window_s=(-300.0, 300.0),
        channel_family_filter=None,
        include_low_rate_channels=False,
        min_sps=20.0,
        picker="stalta",
        picker_kwargs=None,
        min_ratio_max=2.0,
        validate_other_channels=True,
        channel_consistency_metric="spread",
        channel_consistency_warn_s=3.0,
        require_min_channels_for_consistency=2,
        aggregation="best",
        export_dir=None,
        export_prefix="wilber_arrivals",
        log=True,

        # NEW: Two-stage filtering selection
        filter_mode="off",              # "off" | "soft" | "hard" | "both"
        soft_cluster_tol_s=3.0,         # seconds
        hard_invalidate=None,           # True/False; if None uses outlier_hard_filter

        # OPTIONAL (existing): distance-time outlier filtering
        outlier_filter=False,
        outlier_hard_filter=True,
        outlier_z_mad=3.5,
        outlier_iter=2,
        outlier_plot=False,
        outlier_plot_path=None,

        # NEW: pick_ok acceptance window controls (avoid hidden rejection)
        t_obs_accept_window_s=None,     # None -> uses t_obs_window_s
        t_obs_accept_margin_s=0.0,      # expands accept window by +/- margin
    ):
        """
        Returns:
          arrivals_df (ALL stations from station list, picked or not)
          picks_df (per channel file)
          qc dict
        """
        import os
        import numpy as np
        import pandas as pd
        import datetime as dt
        import inspect

        picker_kwargs = dict(picker_kwargs or {})

        # ---- Robust kwarg filtering (KEY FIX for your current failure) ----
        def _filter_kwargs(func, kw, drop=("min_pick_s", "max_pick_s")):
            """
            Keep only kwargs accepted by func signature.
            Prevents passing 'preprocess' into internal stalta, etc.
            """
            kw2 = dict(kw or {})
            for d in drop:
                kw2.pop(d, None)
            try:
                sig = inspect.signature(func)
                allowed = set(sig.parameters.keys())
                return {k: v for k, v in kw2.items() if k in allowed}
            except Exception:
                # fallback: at least drop known window keys
                return kw2

        # normalize picker name once
        picker_name = str(picker).lower().strip()

        # normalize filter_mode
        mode = str(filter_mode).lower().strip()
        if mode not in ("off", "soft", "hard", "both"):
            mode = "off"

        soft_on = mode in ("soft", "both")
        hard_on = mode in ("hard", "both")

        # backward compatibility: if user used outlier_filter=True but left filter_mode="off"
        if bool(outlier_filter) and mode == "off":
            hard_on = True

        stations_df = self.wilber_read_station_list(station_list_path, return_pandas=True, log=log)
        wave_index_df = self.wilber_index_waveform_folder(waveform_dir, return_pandas=True, log=log)
        wi = wave_index_df[wave_index_df["valid_header"] == True].copy()

        wi["net"] = wi["net"].astype(str).str.strip()
        wi["sta"] = wi["sta"].astype(str).str.strip()
        wi["chan"] = wi["chan"].astype(str).str.strip()
        wi["loc"] = wi["loc"].astype(str).fillna("").str.strip()

        if channel_family_filter:
            fam = str(channel_family_filter).upper().strip()
            wi = wi[wi["chan"].str.upper().str.startswith(fam)].copy()

        if not include_low_rate_channels:
            wi = wi[pd.to_numeric(wi["sps"], errors="coerce").fillna(0.0) >= float(min_sps)].copy()

        origin_utc = self._wilber__get_origin_time(origin_time_utc, xml_path_for_origin)

        st_keys = set((net, sta) for (net, sta) in stations_df[["net", "sta"]].itertuples(index=False, name=None))
        wi = wi[wi.apply(lambda r: (r["net"], r["sta"]) in st_keys, axis=1)].copy()

        phaseU = str(phase).upper().strip()
        if phaseU == "P":
            prefer_suffix = ("Z",)
        else:
            prefer_suffix = ("E", "N", "Z")

        # acceptance window for pick_ok (relative to origin)
        if t_obs_accept_window_s is None:
            accept0, accept1 = float(t_obs_window_s[0]), float(t_obs_window_s[1])
        else:
            accept0, accept1 = float(t_obs_accept_window_s[0]), float(t_obs_accept_window_s[1])
        mrg = float(t_obs_accept_margin_s)
        accept0 -= mrg
        accept1 += mrg

        # per-file picking
        pick_rows = []
        for r in wi.itertuples(index=False):
            fp = str(r.filepath)
            try:
                tr = self.wilber_read_timeseries(fp)

                min_pick_s, max_pick_s = self._wilber__pick_window_from_origin(
                    tr["t0_utc"], origin_utc, t_obs_window_s
                )

                mp = float(picker_kwargs.get("min_pick_s", min_pick_s))
                xp = picker_kwargs.get("max_pick_s", max_pick_s)
                xp = float(xp) if xp is not None else float(max_pick_s)

                picker_used = picker_name

                # ---- picker switch (ObsPy optional fallback) ----
                if picker_name == "stalta":
                    pk = self.wilber_pick_arrival_stalta(
                        tr["t_s"], tr["x"], tr["sps"],
                        min_pick_s=mp, max_pick_s=xp,
                        **_filter_kwargs(self.wilber_pick_arrival_stalta, picker_kwargs)
                    )

                elif picker_name == "obspy_stalta":
                    try:
                        ok_obspy, err_obspy = self._wilber__obspy_available()
                        if not ok_obspy:
                            raise ImportError(err_obspy)
                        pk = self.wilber_pick_arrival_obspy_stalta(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **_filter_kwargs(self.wilber_pick_arrival_obspy_stalta, picker_kwargs)
                        )
                    except Exception as e:
                        # fallback to internal
                        picker_used = "stalta_fallback"
                        if log:
                            self._wilber__log(f"[wilber][obspy_stalta] failed for {fp}: {e} -> falling back to internal stalta.")
                        pk = self.wilber_pick_arrival_stalta(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **_filter_kwargs(self.wilber_pick_arrival_stalta, picker_kwargs)
                        )

                elif picker_name == "obspy_ar":
                    try:
                        ok_obspy, err_obspy = self._wilber__obspy_available()
                        if not ok_obspy:
                            raise ImportError(err_obspy)
                        pk = self.wilber_pick_arrival_obspy_ar(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **_filter_kwargs(self.wilber_pick_arrival_obspy_ar, picker_kwargs)
                        )
                    except Exception as e:
                        picker_used = "stalta_fallback"
                        if log:
                            self._wilber__log(f"[wilber][obspy_ar] failed for {fp}: {e} -> falling back to internal stalta.")
                        pk = self.wilber_pick_arrival_stalta(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **_filter_kwargs(self.wilber_pick_arrival_stalta, picker_kwargs)
                        )
                else:
                    raise ValueError("picker must be one of: 'stalta','obspy_stalta','obspy_ar'")

                picked = bool(pk.get("picked", False))
                if picked:
                    t_pick_utc = tr["t0_utc"] + dt.timedelta(seconds=float(pk["t_pick_rel_s"]))
                    t_obs_s = (t_pick_utc - origin_utc).total_seconds()
                else:
                    t_pick_utc = pd.NaT
                    t_obs_s = np.nan

                ratio_max = float(pk.get("ratio_max", np.nan))

                # pick_ok uses explicit acceptance window
                time_ok = picked and np.isfinite(t_obs_s) and (accept0 <= float(t_obs_s) <= accept1)

                ratio_ok = True
                if (str(picker_used).startswith("stalta") or picker_name == "obspy_stalta"):
                    if np.isfinite(ratio_max):
                        ratio_ok = (ratio_max >= float(min_ratio_max))
                    else:
                        ratio_ok = True

                pick_ok = bool(picked and np.isfinite(t_obs_s) and time_ok and ratio_ok)

                pick_rows.append({
                    "net": tr["net"], "sta": tr["sta"], "loc": tr["loc"], "chan": tr["chan"],
                    "filepath": fp,
                    "t0_utc": tr["t0_utc"], "sps": tr["sps"],
                    "picked": picked,
                    "t_pick_rel_s": float(pk.get("t_pick_rel_s", np.nan)),
                    "t_pick_utc": t_pick_utc,
                    "t_obs_s": float(t_obs_s) if np.isfinite(t_obs_s) else np.nan,
                    "ratio_max": ratio_max,
                    "min_pick_s_used": float(mp),
                    "max_pick_s_used": float(xp),
                    "pick_ok": bool(pick_ok),
                    "picker_used": picker_used,
                    "error": "",
                })

            except Exception as e:
                pick_rows.append({
                    "net": str(r.net), "sta": str(r.sta), "loc": str(r.loc), "chan": str(r.chan),
                    "filepath": fp,
                    "t0_utc": r.t0_utc, "sps": r.sps,
                    "picked": False, "t_pick_rel_s": np.nan, "t_pick_utc": pd.NaT, "t_obs_s": np.nan,
                    "ratio_max": np.nan,
                    "min_pick_s_used": np.nan, "max_pick_s_used": np.nan,
                    "pick_ok": False,
                    "picker_used": "error",
                    "error": str(e),
                })

        picks_df = pd.DataFrame(pick_rows)

        # build arrivals_df with ALL stations
        arrivals_df = stations_df.copy()
        arrivals_df["picked"] = False
        arrivals_df["t_pick_utc"] = pd.NaT
        arrivals_df["t_obs_s"] = np.nan
        arrivals_df["chan_used"] = ""
        arrivals_df["loc_used"] = ""
        arrivals_df["ratio_max_used"] = np.nan
        arrivals_df["waveform_file_used"] = ""
        arrivals_df["n_ok_channels"] = 0
        arrivals_df["channel_spread_s"] = np.nan
        arrivals_df["channel_std_s"] = np.nan
        arrivals_df["channels_ok_list"] = ""
        arrivals_df["channel_consistent"] = True
        arrivals_df["notes"] = ""
        arrivals_df["origin_utc"] = origin_utc
        arrivals_df["phase"] = phaseU

        # soft-filter output columns (station-level)
        arrivals_df["soft_cluster_n"] = 0
        arrivals_df["soft_cluster_spread_s"] = np.nan
        arrivals_df["soft_cluster_used_chans"] = ""
        arrivals_df["soft_inconsistent_flag"] = False

        # hard-filter output columns (station-level)
        if "outlier_flag" not in arrivals_df.columns:
            arrivals_df["outlier_flag"] = False
        if "outlier_score" not in arrivals_df.columns:
            arrivals_df["outlier_score"] = np.nan
        if "residual_s" not in arrivals_df.columns:
            arrivals_df["residual_s"] = np.nan
        if "t_pred_fit_s" not in arrivals_df.columns:
            arrivals_df["t_pred_fit_s"] = np.nan

        warn_count = 0
        soft_inconsistent_count = 0

        for i, st in arrivals_df.iterrows():
            net = str(st["net"]); sta = str(st["sta"])
            sub = picks_df[(picks_df["net"] == net) & (picks_df["sta"] == sta)].copy()
            if sub.empty:
                arrivals_df.at[i, "notes"] = "NO_WAVEFORMS_FOR_STATION"
                continue

            ok = sub[sub["pick_ok"] == True].copy()
            arrivals_df.at[i, "n_ok_channels"] = int(len(ok))
            if ok.empty:
                arrivals_df.at[i, "notes"] = "NO_VALID_PICK_IN_WINDOW"
                continue

            # channel consistency stats (DO NOT drop station)
            if validate_other_channels and len(ok) >= int(require_min_channels_for_consistency):
                vals = ok["t_obs_s"].astype(float).values
                spread = float(np.nanmax(vals) - np.nanmin(vals))
                stdv = float(np.nanstd(vals))
                arrivals_df.at[i, "channel_spread_s"] = spread
                arrivals_df.at[i, "channel_std_s"] = stdv
                arrivals_df.at[i, "channels_ok_list"] = ",".join(ok["chan"].astype(str).tolist())

                metric = spread if str(channel_consistency_metric).lower() == "spread" else stdv
                if np.isfinite(metric) and metric > float(channel_consistency_warn_s):
                    arrivals_df.at[i, "channel_consistent"] = False
                    arrivals_df.at[i, "notes"] = f"WARNING: channel_{channel_consistency_metric}={metric:.2f}s exceeds {float(channel_consistency_warn_s):.2f}s"
                    warn_count += 1
            else:
                arrivals_df.at[i, "channels_ok_list"] = ",".join(ok["chan"].astype(str).tolist())

            # ---- Soft filter (multi-channel clustering) ----
            ok_use = ok.copy()
            if soft_on and len(ok_use) >= 2:
                dfc, meta = self._wilber__soft_cluster_channel_picks(ok_use, soft_cluster_tol_s=float(soft_cluster_tol_s))
                arrivals_df.at[i, "soft_cluster_n"] = int(meta.get("soft_cluster_n", 0))
                arrivals_df.at[i, "soft_cluster_spread_s"] = float(meta.get("soft_cluster_spread_s", np.nan)) if meta.get("soft_cluster_spread_s", None) is not None else np.nan
                arrivals_df.at[i, "soft_cluster_used_chans"] = str(meta.get("soft_cluster_used_chans", ""))

                all_spread = float(np.nanmax(ok_use["t_obs_s"].astype(float).values) - np.nanmin(ok_use["t_obs_s"].astype(float).values)) if len(ok_use) > 1 else 0.0
                if int(meta.get("soft_cluster_n", 0)) < 2 and len(ok_use) >= 2 and np.isfinite(all_spread) and all_spread > float(soft_cluster_tol_s):
                    arrivals_df.at[i, "soft_inconsistent_flag"] = True
                    soft_inconsistent_count += 1
                else:
                    arrivals_df.at[i, "soft_inconsistent_flag"] = False

                if int(meta.get("soft_cluster_n", 0)) >= 2:
                    ok_use = dfc.copy()
            else:
                arrivals_df.at[i, "soft_cluster_n"] = int(len(ok_use))
                if len(ok_use) >= 2:
                    vals = ok_use["t_obs_s"].astype(float).values
                    arrivals_df.at[i, "soft_cluster_spread_s"] = float(np.nanmax(vals) - np.nanmin(vals))
                    arrivals_df.at[i, "soft_cluster_used_chans"] = ",".join(ok_use["chan"].astype(str).tolist())
                elif len(ok_use) == 1:
                    arrivals_df.at[i, "soft_cluster_spread_s"] = 0.0
                    arrivals_df.at[i, "soft_cluster_used_chans"] = str(ok_use.iloc[0]["chan"])
                arrivals_df.at[i, "soft_inconsistent_flag"] = False

            # choose representative pick (from ok_use)
            ok_use["_suffix"] = ok_use["chan"].astype(str).str.upper().str[-1:]
            ok_pref = ok_use[ok_use["_suffix"].isin(prefer_suffix)].copy()
            if ok_pref.empty:
                ok_pref = ok_use.copy()

            if str(aggregation).lower() == "earliest":
                chosen = ok_pref.sort_values("t_obs_s", ascending=True).iloc[0]
            else:
                chosen = ok_pref.sort_values(["ratio_max", "t_obs_s"], ascending=[False, True]).iloc[0]

            arrivals_df.at[i, "picked"] = True
            arrivals_df.at[i, "t_pick_utc"] = chosen["t_pick_utc"]
            arrivals_df.at[i, "t_obs_s"] = float(chosen["t_obs_s"])
            arrivals_df.at[i, "chan_used"] = str(chosen["chan"])
            arrivals_df.at[i, "loc_used"] = str(chosen["loc"]) if pd.notna(chosen["loc"]) else ""
            arrivals_df.at[i, "ratio_max_used"] = float(chosen["ratio_max"]) if pd.notna(chosen["ratio_max"]) else np.nan
            arrivals_df.at[i, "waveform_file_used"] = str(chosen["filepath"])

        # ---- Hard filter (distance vs time) ----
        outlier_info = None
        if hard_on or bool(outlier_filter):
            if hard_invalidate is None:
                hard_invalidate_use = bool(outlier_hard_filter)
            else:
                hard_invalidate_use = bool(hard_invalidate)

            try:
                arrivals_df, outlier_info = self._wilber__station_outlier_filter(
                    arrivals_df,
                    use_only_picked=True,
                    robust_iter=int(outlier_iter),
                    z_mad=float(outlier_z_mad),
                    hard_invalidate=bool(hard_invalidate_use),   # CONSISTENT NOW
                    plot=bool(outlier_plot),
                    plot_path=outlier_plot_path,
                    log=log,
                )
            except Exception as e:
                if log:
                    self._wilber__log(f"[wilber] could not run hard filter: {e}")
                outlier_info = {"error": str(e)}

        # QC: include errors summary (helps instantly when something breaks again)
        n_errors = int((picks_df["picker_used"] == "error").sum()) if "picker_used" in picks_df.columns else 0
        top_error = ""
        try:
            bad = picks_df[picks_df["error"].astype(str).str.len() > 0]
            if len(bad):
                top_error = str(bad["error"].value_counts().index[0])
        except Exception:
            pass

        qc = {
            "N_picks_total": int(len(picks_df)),
            "N_picks_ok": int((picks_df["pick_ok"] == True).sum()),
            "N_station_arrivals": int((arrivals_df["picked"] == True).sum()),
            "N_channel_consistency_warnings": int(warn_count),
            "N_soft_inconsistent": int(soft_inconsistent_count),
            "N_picks_error": int(n_errors),
            "top_error": top_error,
            "t_obs_window_s": (float(t_obs_window_s[0]), float(t_obs_window_s[1])),
            "t_obs_accept_window_s": (float(accept0), float(accept1)),
            "phase": phaseU,
            "filter_mode": mode,
            "soft_cluster_tol_s": float(soft_cluster_tol_s),
            "hard_filter": bool(hard_on or bool(outlier_filter)),
            "hard_invalidate": bool(hard_invalidate if hard_invalidate is not None else outlier_hard_filter),
            "outlier_info": outlier_info if outlier_info is not None else {},
        }

        if log:
            self._wilber__log(f"[wilber] picks total={qc['N_picks_total']}, ok={qc['N_picks_ok']}, station arrivals={qc['N_station_arrivals']}")
            self._wilber__log(f"[wilber] pick_ok accept window: [{qc['t_obs_accept_window_s'][0]:.1f}, {qc['t_obs_accept_window_s'][1]:.1f}] s (mode={mode})")
            self._wilber__log(f"[wilber] channel-consistency warnings: {qc['N_channel_consistency_warnings']} (threshold={float(channel_consistency_warn_s):.2f}s, metric={channel_consistency_metric})")
            if soft_on:
                self._wilber__log(f"[wilber] soft filter: tol={float(soft_cluster_tol_s):.2f}s inconsistent={qc['N_soft_inconsistent']}")
            if qc.get("N_picks_error", 0) > 0:
                self._wilber__log(f"[wilber] picker errors: N={qc['N_picks_error']} top='{qc.get('top_error','')}'")
            if outlier_info is not None and isinstance(outlier_info, dict) and len(outlier_info) > 0:
                self._wilber__log(f"[wilber] hard filter: flagged={outlier_info.get('n_flagged', 0)} invalidate={outlier_info.get('hard_invalidate', False)} z={outlier_info.get('z_mad', np.nan)}")

        if export_dir:
            out = self._wilber__ensure_dir(self._wilber__as_path(export_dir))
            arrivals_df.to_csv(os.path.join(out, f"{export_prefix}_arrivals_station.csv"), index=False)
            picks_df.to_csv(os.path.join(out, f"{export_prefix}_picks_channel.csv"), index=False)

        return arrivals_df, picks_df, qc



            
    # override 
    def wilber_build_arrivals_dataset(
        self,
        station_list_path,
        waveform_dir,
        *,
        origin_time_utc="from_xml",
        xml_path_for_origin=None,
        phase="S",
        t_obs_window_s=(-300.0, 300.0),
        channel_family_filter=None,
        include_low_rate_channels=False,
        min_sps=20.0,
        picker="stalta",
        picker_kwargs=None,
        min_ratio_max=2.0,
        validate_other_channels=True,
        channel_consistency_metric="spread",
        channel_consistency_warn_s=3.0,
        require_min_channels_for_consistency=2,
        aggregation="best",
        export_dir=None,
        export_prefix="wilber_arrivals",
        log=True,

        # NEW: Two-stage filtering selection
        filter_mode="off",              # "off" | "soft" | "hard" | "both"
        soft_cluster_tol_s=3.0,         # seconds
        hard_invalidate=None,           # True/False; if None uses outlier_hard_filter

        # OPTIONAL (existing): distance-time outlier filtering
        outlier_filter=False,
        outlier_hard_filter=True,
        outlier_z_mad=3.5,
        outlier_iter=2,
        outlier_plot=False,
        outlier_plot_path=None,

        # NEW: pick_ok acceptance window controls (avoid hidden rejection)
        t_obs_accept_window_s=None,     # None -> uses t_obs_window_s
        t_obs_accept_margin_s=0.0,      # expands accept window by +/- margin
    ):
        """
        Returns:
          arrivals_df (ALL stations from station list, picked or not)
          picks_df (per channel file)
          qc dict
        """
        import os
        import numpy as np
        import pandas as pd
        import datetime as dt

        picker_kwargs = dict(picker_kwargs or {})

        # normalize filter_mode
        mode = str(filter_mode).lower().strip()
        if mode not in ("off", "soft", "hard", "both"):
            mode = "off"

        soft_on = mode in ("soft", "both")
        hard_on = mode in ("hard", "both")

        # backward compatibility: if user used outlier_filter=True but left filter_mode="off"
        if bool(outlier_filter) and mode == "off":
            hard_on = True

        stations_df = self.wilber_read_station_list(station_list_path, return_pandas=True, log=log)
        wave_index_df = self.wilber_index_waveform_folder(waveform_dir, return_pandas=True, log=log)
        wi = wave_index_df[wave_index_df["valid_header"] == True].copy()

        wi["net"] = wi["net"].astype(str).str.strip()
        wi["sta"] = wi["sta"].astype(str).str.strip()
        wi["chan"] = wi["chan"].astype(str).str.strip()
        wi["loc"] = wi["loc"].astype(str).fillna("").str.strip()

        if channel_family_filter:
            fam = str(channel_family_filter).upper().strip()
            wi = wi[wi["chan"].str.upper().str.startswith(fam)].copy()

        if not include_low_rate_channels:
            wi = wi[pd.to_numeric(wi["sps"], errors="coerce").fillna(0.0) >= float(min_sps)].copy()

        origin_utc = self._wilber__get_origin_time(origin_time_utc, xml_path_for_origin)

        st_keys = set((net, sta) for (net, sta) in stations_df[["net", "sta"]].itertuples(index=False, name=None))
        wi = wi[wi.apply(lambda r: (r["net"], r["sta"]) in st_keys, axis=1)].copy()

        phaseU = str(phase).upper().strip()
        if phaseU == "P":
            prefer_suffix = ("Z",)
        else:
            prefer_suffix = ("E", "N", "Z")

        # acceptance window for pick_ok (relative to origin)
        if t_obs_accept_window_s is None:
            accept0, accept1 = float(t_obs_window_s[0]), float(t_obs_window_s[1])
        else:
            accept0, accept1 = float(t_obs_accept_window_s[0]), float(t_obs_accept_window_s[1])
        mrg = float(t_obs_accept_margin_s)
        accept0 -= mrg
        accept1 += mrg

        # per-file picking
        pick_rows = []
        for r in wi.itertuples(index=False):
            fp = str(r.filepath)
            try:
                tr = self.wilber_read_timeseries(fp)

                min_pick_s, max_pick_s = self._wilber__pick_window_from_origin(
                    tr["t0_utc"], origin_utc, t_obs_window_s
                )

                mp = float(picker_kwargs.get("min_pick_s", min_pick_s))
                xp = picker_kwargs.get("max_pick_s", max_pick_s)
                xp = float(xp) if xp is not None else float(max_pick_s)

                picker_used = str(picker)

                # ---- picker switch (ObsPy optional fallback) ----
                if picker == "stalta":
                    pk = self.wilber_pick_arrival_stalta(
                        tr["t_s"], tr["x"], tr["sps"],
                        min_pick_s=mp, max_pick_s=xp,
                        **{k: v for k, v in picker_kwargs.items() if k not in ("min_pick_s", "max_pick_s")}
                    )

                elif picker == "obspy_stalta":
                    try:
                        ok_obspy, err_obspy = self._wilber__obspy_available()
                        if not ok_obspy:
                            raise ImportError(err_obspy)
                        pk = self.wilber_pick_arrival_obspy_stalta(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **{k: v for k, v in picker_kwargs.items() if k not in ("min_pick_s", "max_pick_s")}
                        )
                    except Exception as e:
                        # fallback to internal
                        picker_used = "stalta_fallback"
                        if log:
                            self._wilber__log(f"[wilber][obspy_stalta] failed for {fp}: {e} -> falling back to internal stalta.")
                        pk = self.wilber_pick_arrival_stalta(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **{k: v for k, v in picker_kwargs.items() if k not in ("min_pick_s", "max_pick_s")}
                        )

                elif picker == "obspy_ar":
                    try:
                        ok_obspy, err_obspy = self._wilber__obspy_available()
                        if not ok_obspy:
                            raise ImportError(err_obspy)
                        pk = self.wilber_pick_arrival_obspy_ar(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **{k: v for k, v in picker_kwargs.items() if k not in ("min_pick_s", "max_pick_s")}
                        )
                    except Exception as e:
                        picker_used = "stalta_fallback"
                        if log:
                            self._wilber__log(f"[wilber][obspy_ar] failed for {fp}: {e} -> falling back to internal stalta.")
                        pk = self.wilber_pick_arrival_stalta(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **{k: v for k, v in picker_kwargs.items() if k not in ("min_pick_s", "max_pick_s")}
                        )

                else:
                    raise ValueError("picker must be one of: 'stalta','obspy_stalta','obspy_ar'")

                picked = bool(pk.get("picked", False))
                if picked:
                    t_pick_utc = tr["t0_utc"] + dt.timedelta(seconds=float(pk["t_pick_rel_s"]))
                    t_obs_s = (t_pick_utc - origin_utc).total_seconds()
                else:
                    t_pick_utc = pd.NaT
                    t_obs_s = np.nan

                ratio_max = float(pk.get("ratio_max", np.nan))

                # pick_ok uses an explicit acceptance window (defaults to t_obs_window_s)
                # and can be widened with t_obs_accept_margin_s to avoid accidental rejection.
                time_ok = picked and np.isfinite(t_obs_s) and (accept0 <= float(t_obs_s) <= accept1)

                ratio_ok = True
                if (picker_used.startswith("stalta") or picker == "obspy_stalta"):
                    # for stalta-based pickers, ratio_max is meaningful; for others it can be nan
                    if np.isfinite(ratio_max):
                        ratio_ok = (ratio_max >= float(min_ratio_max))
                    else:
                        ratio_ok = True

                pick_ok = bool(picked and np.isfinite(t_obs_s) and time_ok and ratio_ok)

                pick_rows.append({
                    "net": tr["net"], "sta": tr["sta"], "loc": tr["loc"], "chan": tr["chan"],
                    "filepath": fp,
                    "t0_utc": tr["t0_utc"], "sps": tr["sps"],
                    "picked": picked,
                    "t_pick_rel_s": float(pk.get("t_pick_rel_s", np.nan)),
                    "t_pick_utc": t_pick_utc,
                    "t_obs_s": float(t_obs_s) if np.isfinite(t_obs_s) else np.nan,
                    "ratio_max": ratio_max,
                    "min_pick_s_used": float(mp),
                    "max_pick_s_used": float(xp),
                    "pick_ok": bool(pick_ok),
                    "picker_used": picker_used,
                    "error": "",
                })

            except Exception as e:
                pick_rows.append({
                    "net": str(r.net), "sta": str(r.sta), "loc": str(r.loc), "chan": str(r.chan),
                    "filepath": fp,
                    "t0_utc": r.t0_utc, "sps": r.sps,
                    "picked": False, "t_pick_rel_s": np.nan, "t_pick_utc": pd.NaT, "t_obs_s": np.nan,
                    "ratio_max": np.nan,
                    "min_pick_s_used": np.nan, "max_pick_s_used": np.nan,
                    "pick_ok": False,
                    "picker_used": "error",
                    "error": str(e),
                })

        picks_df = pd.DataFrame(pick_rows)

        # build arrivals_df with ALL stations
        arrivals_df = stations_df.copy()
        arrivals_df["picked"] = False
        arrivals_df["t_pick_utc"] = pd.NaT
        arrivals_df["t_obs_s"] = np.nan
        arrivals_df["chan_used"] = ""
        arrivals_df["loc_used"] = ""
        arrivals_df["ratio_max_used"] = np.nan
        arrivals_df["waveform_file_used"] = ""
        arrivals_df["n_ok_channels"] = 0
        arrivals_df["channel_spread_s"] = np.nan
        arrivals_df["channel_std_s"] = np.nan
        arrivals_df["channels_ok_list"] = ""
        arrivals_df["channel_consistent"] = True
        arrivals_df["notes"] = ""
        arrivals_df["origin_utc"] = origin_utc
        arrivals_df["phase"] = phaseU

        # NEW: soft-filter output columns (station-level)
        arrivals_df["soft_cluster_n"] = 0
        arrivals_df["soft_cluster_spread_s"] = np.nan
        arrivals_df["soft_cluster_used_chans"] = ""
        arrivals_df["soft_inconsistent_flag"] = False

        # NEW: hard-filter output columns (station-level)
        if "outlier_flag" not in arrivals_df.columns:
            arrivals_df["outlier_flag"] = False
        if "outlier_score" not in arrivals_df.columns:
            arrivals_df["outlier_score"] = np.nan
        if "residual_s" not in arrivals_df.columns:
            arrivals_df["residual_s"] = np.nan
        if "t_pred_fit_s" not in arrivals_df.columns:
            arrivals_df["t_pred_fit_s"] = np.nan

        warn_count = 0
        soft_inconsistent_count = 0

        for i, st in arrivals_df.iterrows():
            net = str(st["net"]); sta = str(st["sta"])
            sub = picks_df[(picks_df["net"] == net) & (picks_df["sta"] == sta)].copy()
            if sub.empty:
                arrivals_df.at[i, "notes"] = "NO_WAVEFORMS_FOR_STATION"
                continue

            ok = sub[sub["pick_ok"] == True].copy()
            arrivals_df.at[i, "n_ok_channels"] = int(len(ok))
            if ok.empty:
                arrivals_df.at[i, "notes"] = "NO_VALID_PICK_IN_WINDOW"
                continue

            # channel consistency stats (DO NOT drop station)
            if validate_other_channels and len(ok) >= int(require_min_channels_for_consistency):
                vals = ok["t_obs_s"].astype(float).values
                spread = float(np.nanmax(vals) - np.nanmin(vals))
                stdv = float(np.nanstd(vals))
                arrivals_df.at[i, "channel_spread_s"] = spread
                arrivals_df.at[i, "channel_std_s"] = stdv
                arrivals_df.at[i, "channels_ok_list"] = ",".join(ok["chan"].astype(str).tolist())

                metric = spread if str(channel_consistency_metric).lower() == "spread" else stdv
                if np.isfinite(metric) and metric > float(channel_consistency_warn_s):
                    arrivals_df.at[i, "channel_consistent"] = False
                    arrivals_df.at[i, "notes"] = f"WARNING: channel_{channel_consistency_metric}={metric:.2f}s exceeds {float(channel_consistency_warn_s):.2f}s"
                    warn_count += 1
            else:
                # still record ok channel list if available
                arrivals_df.at[i, "channels_ok_list"] = ",".join(ok["chan"].astype(str).tolist())

            # ---- Soft filter (multi-channel clustering) ----
            ok_use = ok.copy()
            if soft_on and len(ok_use) >= 2:
                dfc, meta = self._wilber__soft_cluster_channel_picks(ok_use, soft_cluster_tol_s=float(soft_cluster_tol_s))
                arrivals_df.at[i, "soft_cluster_n"] = int(meta.get("soft_cluster_n", 0))
                arrivals_df.at[i, "soft_cluster_spread_s"] = float(meta.get("soft_cluster_spread_s", np.nan)) if meta.get("soft_cluster_spread_s", None) is not None else np.nan
                arrivals_df.at[i, "soft_cluster_used_chans"] = str(meta.get("soft_cluster_used_chans", ""))

                # mark inconsistent only when we could NOT form a >=2 cluster but we have multiple ok channels
                all_spread = float(np.nanmax(ok_use["t_obs_s"].astype(float).values) - np.nanmin(ok_use["t_obs_s"].astype(float).values)) if len(ok_use) > 1 else 0.0
                if int(meta.get("soft_cluster_n", 0)) < 2 and len(ok_use) >= 2 and np.isfinite(all_spread) and all_spread > float(soft_cluster_tol_s):
                    arrivals_df.at[i, "soft_inconsistent_flag"] = True
                    soft_inconsistent_count += 1
                else:
                    arrivals_df.at[i, "soft_inconsistent_flag"] = False

                # If cluster found (>=2), restrict selection to that cluster; else keep all ok channels
                if int(meta.get("soft_cluster_n", 0)) >= 2:
                    ok_use = dfc.copy()
            else:
                # no soft filter: still populate fields minimally
                arrivals_df.at[i, "soft_cluster_n"] = int(len(ok_use))
                if len(ok_use) >= 2:
                    vals = ok_use["t_obs_s"].astype(float).values
                    arrivals_df.at[i, "soft_cluster_spread_s"] = float(np.nanmax(vals) - np.nanmin(vals))
                    arrivals_df.at[i, "soft_cluster_used_chans"] = ",".join(ok_use["chan"].astype(str).tolist())
                elif len(ok_use) == 1:
                    arrivals_df.at[i, "soft_cluster_spread_s"] = 0.0
                    arrivals_df.at[i, "soft_cluster_used_chans"] = str(ok_use.iloc[0]["chan"])
                arrivals_df.at[i, "soft_inconsistent_flag"] = False

            # choose representative pick (from ok_use)
            ok_use["_suffix"] = ok_use["chan"].astype(str).str.upper().str[-1:]
            ok_pref = ok_use[ok_use["_suffix"].isin(prefer_suffix)].copy()
            if ok_pref.empty:
                ok_pref = ok_use.copy()

            if str(aggregation).lower() == "earliest":
                chosen = ok_pref.sort_values("t_obs_s", ascending=True).iloc[0]
            else:
                # keep your existing core behavior
                chosen = ok_pref.sort_values(["ratio_max", "t_obs_s"], ascending=[False, True]).iloc[0]

            arrivals_df.at[i, "picked"] = True
            arrivals_df.at[i, "t_pick_utc"] = chosen["t_pick_utc"]
            arrivals_df.at[i, "t_obs_s"] = float(chosen["t_obs_s"])
            arrivals_df.at[i, "chan_used"] = str(chosen["chan"])
            arrivals_df.at[i, "loc_used"] = str(chosen["loc"]) if pd.notna(chosen["loc"]) else ""
            arrivals_df.at[i, "ratio_max_used"] = float(chosen["ratio_max"]) if pd.notna(chosen["ratio_max"]) else np.nan
            arrivals_df.at[i, "waveform_file_used"] = str(chosen["filepath"])

        # ---- Hard filter (distance vs time) ----
        outlier_info = None
        if hard_on or bool(outlier_filter):
            # determine invalidation behavior
            if hard_invalidate is None:
                hard_invalidate_use = bool(outlier_hard_filter)
            else:
                hard_invalidate_use = bool(hard_invalidate)

            try:
                arrivals_df, outlier_info = self._wilber__station_outlier_filter(
                    arrivals_df,
                    use_only_picked=True,
                    robust_iter=int(outlier_iter),
                    z_mad=float(outlier_z_mad),
                    hard_invalidate=bool(hard_invalidate_use),
                    plot=bool(outlier_plot),
                    plot_path=outlier_plot_path,
                    log=log,
                )
            except Exception as e:
                if log:
                    self._wilber__log(f"[wilber] could not run hard filter: {e}")
                outlier_info = {"error": str(e)}

        qc = {
            "N_picks_total": int(len(picks_df)),
            "N_picks_ok": int((picks_df["pick_ok"] == True).sum()),
            "N_station_arrivals": int((arrivals_df["picked"] == True).sum()),
            "N_channel_consistency_warnings": int(warn_count),
            "N_soft_inconsistent": int(soft_inconsistent_count),
            "t_obs_window_s": (float(t_obs_window_s[0]), float(t_obs_window_s[1])),
            "t_obs_accept_window_s": (float(accept0), float(accept1)),
            "phase": phaseU,
            "filter_mode": mode,
            "soft_cluster_tol_s": float(soft_cluster_tol_s),
            "hard_filter": bool(hard_on or bool(outlier_filter)),
            "hard_invalidate": bool(hard_invalidate if hard_invalidate is not None else outlier_hard_filter),
            "outlier_info": outlier_info if outlier_info is not None else {},
        }

        if log:
            self._wilber__log(f"[wilber] picks total={qc['N_picks_total']}, ok={qc['N_picks_ok']}, station arrivals={qc['N_station_arrivals']}")
            self._wilber__log(f"[wilber] pick_ok accept window: [{qc['t_obs_accept_window_s'][0]:.1f}, {qc['t_obs_accept_window_s'][1]:.1f}] s (mode={mode})")
            self._wilber__log(f"[wilber] channel-consistency warnings: {qc['N_channel_consistency_warnings']} (threshold={float(channel_consistency_warn_s):.2f}s, metric={channel_consistency_metric})")
            if soft_on:
                self._wilber__log(f"[wilber] soft filter: tol={float(soft_cluster_tol_s):.2f}s inconsistent={qc['N_soft_inconsistent']}")
            if outlier_info is not None and isinstance(outlier_info, dict) and len(outlier_info) > 0:
                self._wilber__log(f"[wilber] hard filter: flagged={outlier_info.get('n_flagged', 0)} invalidate={outlier_info.get('hard_filter', False)} z={outlier_info.get('z_mad', np.nan)}")

        if export_dir:
            out = self._wilber__ensure_dir(self._wilber__as_path(export_dir))
            arrivals_df.to_csv(os.path.join(out, f"{export_prefix}_arrivals_station.csv"), index=False)
            picks_df.to_csv(os.path.join(out, f"{export_prefix}_picks_channel.csv"), index=False)

        return arrivals_df, picks_df, qc



    # -------------------------------------------------------------------------
    # F) Build arrivals dataset (ALL stations kept; mismatches flagged)
    # -------------------------------------------------------------------------
    def wilber_build_arrivals_dataset(
        self,
        station_list_path,
        waveform_dir,
        *,
        origin_time_utc="from_xml",
        xml_path_for_origin=None,
        phase="S",
        t_obs_window_s=(-300.0, 300.0),
        channel_family_filter=None,
        include_low_rate_channels=False,
        min_sps=20.0,
        picker="stalta",
        picker_kwargs=None,
        min_ratio_max=2.0,
        validate_other_channels=True,
        channel_consistency_metric="spread",
        channel_consistency_warn_s=3.0,
        require_min_channels_for_consistency=2,
        aggregation="best",
        export_dir=None,
        export_prefix="wilber_arrivals",
        log=True,
    
        # NEW: Two-stage filtering selection
        filter_mode="off",              # "off" | "soft" | "hard" | "local" | "soft+local" | "both" | "all"
        soft_cluster_tol_s=3.0,         # seconds
        hard_invalidate=None,           # True/False; if None uses outlier_hard_filter
    
        # OPTIONAL (existing): distance-time outlier filtering
        outlier_filter=False,
        outlier_hard_filter=True,
        outlier_z_mad=3.5,
        outlier_iter=2,
        outlier_plot=False,
        outlier_plot_path=None,
    
        # NEW: local spatial outlier filtering (flagging near-neighbor jumps)
        local_filter_kwargs=None,       # dict, see defaults below
        local_invalidate=False,         # flag-only default; True unpicks local outliers
    
        # NEW: pick_ok acceptance window controls (avoid hidden rejection)
        t_obs_accept_window_s=None,     # None -> uses t_obs_window_s
        t_obs_accept_margin_s=0.0,      # expands accept window by +/- margin
    ):
        """
        Returns:
          arrivals_df (ALL stations from station list, picked or not)
          picks_df (per channel file)
          qc dict
        """
        import os
        import numpy as np
        import pandas as pd
        import datetime as dt
        import inspect
    
        picker_kwargs = dict(picker_kwargs or {})
    
        # ---- Robust kwarg filtering (KEY FIX for your current failure) ----
        def _filter_kwargs(func, kw, drop=("min_pick_s", "max_pick_s")):
            """
            Keep only kwargs accepted by func signature.
            Prevents passing 'preprocess' into internal stalta, etc.
            """
            kw2 = dict(kw or {})
            for d in drop:
                kw2.pop(d, None)
            try:
                sig = inspect.signature(func)
                allowed = set(sig.parameters.keys())
                return {k: v for k, v in kw2.items() if k in allowed}
            except Exception:
                # fallback: at least drop known window keys
                return kw2
    
        # normalize picker name once
        picker_name = str(picker).lower().strip()
    
        # normalize filter_mode (now supports local variants)
        mode = str(filter_mode).lower().strip()
        allowed = ("off", "soft", "hard", "local", "soft+local", "both", "all")
        if mode not in allowed:
            mode = "off"
    
        # define which stages are on
        soft_on = mode in ("soft", "both", "soft+local", "all")
        local_on = mode in ("local", "soft+local", "all")
        hard_on = mode in ("hard", "both", "all")
    
        # backward compatibility: if user used outlier_filter=True but left filter_mode="off"
        if bool(outlier_filter) and mode == "off":
            hard_on = True
    
        stations_df = self.wilber_read_station_list(station_list_path, return_pandas=True, log=log)
        wave_index_df = self.wilber_index_waveform_folder(waveform_dir, return_pandas=True, log=log)
        wi = wave_index_df[wave_index_df["valid_header"] == True].copy()
    
        wi["net"] = wi["net"].astype(str).str.strip()
        wi["sta"] = wi["sta"].astype(str).str.strip()
        wi["chan"] = wi["chan"].astype(str).str.strip()
        wi["loc"] = wi["loc"].astype(str).fillna("").str.strip()
    
        if channel_family_filter:
            fam = str(channel_family_filter).upper().strip()
            wi = wi[wi["chan"].str.upper().str.startswith(fam)].copy()
    
        if not include_low_rate_channels:
            wi = wi[pd.to_numeric(wi["sps"], errors="coerce").fillna(0.0) >= float(min_sps)].copy()
    
        origin_utc = self._wilber__get_origin_time(origin_time_utc, xml_path_for_origin)
    
        st_keys = set((net, sta) for (net, sta) in stations_df[["net", "sta"]].itertuples(index=False, name=None))
        wi = wi[wi.apply(lambda r: (r["net"], r["sta"]) in st_keys, axis=1)].copy()
    
        phaseU = str(phase).upper().strip()
        if phaseU == "P":
            prefer_suffix = ("Z",)
        else:
            prefer_suffix = ("E", "N", "Z")
    
        # acceptance window for pick_ok (relative to origin)
        if t_obs_accept_window_s is None:
            accept0, accept1 = float(t_obs_window_s[0]), float(t_obs_window_s[1])
        else:
            accept0, accept1 = float(t_obs_accept_window_s[0]), float(t_obs_accept_window_s[1])
        mrg = float(t_obs_accept_margin_s)
        accept0 -= mrg
        accept1 += mrg
    
        # per-file picking
        pick_rows = []
        for r in wi.itertuples(index=False):
            fp = str(r.filepath)
            try:
                tr = self.wilber_read_timeseries(fp)
    
                min_pick_s, max_pick_s = self._wilber__pick_window_from_origin(
                    tr["t0_utc"], origin_utc, t_obs_window_s
                )
    
                mp = float(picker_kwargs.get("min_pick_s", min_pick_s))
                xp = picker_kwargs.get("max_pick_s", max_pick_s)
                xp = float(xp) if xp is not None else float(max_pick_s)
    
                picker_used = picker_name
    
                # ---- picker switch (ObsPy optional fallback) ----
                if picker_name == "stalta":
                    pk = self.wilber_pick_arrival_stalta(
                        tr["t_s"], tr["x"], tr["sps"],
                        min_pick_s=mp, max_pick_s=xp,
                        **_filter_kwargs(self.wilber_pick_arrival_stalta, picker_kwargs)
                    )
    
                elif picker_name == "obspy_stalta":
                    try:
                        ok_obspy, err_obspy = self._wilber__obspy_available()
                        if not ok_obspy:
                            raise ImportError(err_obspy)
                        pk = self.wilber_pick_arrival_obspy_stalta(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **_filter_kwargs(self.wilber_pick_arrival_obspy_stalta, picker_kwargs)
                        )
                    except Exception as e:
                        # fallback to internal
                        picker_used = "stalta_fallback"
                        if log:
                            self._wilber__log(f"[wilber][obspy_stalta] failed for {fp}: {e} -> falling back to internal stalta.")
                        pk = self.wilber_pick_arrival_stalta(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **_filter_kwargs(self.wilber_pick_arrival_stalta, picker_kwargs)
                        )
    
                elif picker_name == "obspy_ar":
                    try:
                        ok_obspy, err_obspy = self._wilber__obspy_available()
                        if not ok_obspy:
                            raise ImportError(err_obspy)
                        pk = self.wilber_pick_arrival_obspy_ar(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **_filter_kwargs(self.wilber_pick_arrival_obspy_ar, picker_kwargs)
                        )
                    except Exception as e:
                        picker_used = "stalta_fallback"
                        if log:
                            self._wilber__log(f"[wilber][obspy_ar] failed for {fp}: {e} -> falling back to internal stalta.")
                        pk = self.wilber_pick_arrival_stalta(
                            tr["t_s"], tr["x"], tr["sps"],
                            min_pick_s=mp, max_pick_s=xp,
                            **_filter_kwargs(self.wilber_pick_arrival_stalta, picker_kwargs)
                        )
                else:
                    raise ValueError("picker must be one of: 'stalta','obspy_stalta','obspy_ar'")
    
                picked = bool(pk.get("picked", False))
                if picked:
                    t_pick_utc = tr["t0_utc"] + dt.timedelta(seconds=float(pk["t_pick_rel_s"]))
                    t_obs_s = (t_pick_utc - origin_utc).total_seconds()
                else:
                    t_pick_utc = pd.NaT
                    t_obs_s = np.nan
    
                ratio_max = float(pk.get("ratio_max", np.nan))
    
                # pick_ok uses explicit acceptance window
                time_ok = picked and np.isfinite(t_obs_s) and (accept0 <= float(t_obs_s) <= accept1)
    
                ratio_ok = True
                if (str(picker_used).startswith("stalta") or picker_name == "obspy_stalta"):
                    if np.isfinite(ratio_max):
                        ratio_ok = (ratio_max >= float(min_ratio_max))
                    else:
                        ratio_ok = True
    
                pick_ok = bool(picked and np.isfinite(t_obs_s) and time_ok and ratio_ok)
    
                pick_rows.append({
                    "net": tr["net"], "sta": tr["sta"], "loc": tr["loc"], "chan": tr["chan"],
                    "filepath": fp,
                    "t0_utc": tr["t0_utc"], "sps": tr["sps"],
                    "picked": picked,
                    "t_pick_rel_s": float(pk.get("t_pick_rel_s", np.nan)),
                    "t_pick_utc": t_pick_utc,
                    "t_obs_s": float(t_obs_s) if np.isfinite(t_obs_s) else np.nan,
                    "ratio_max": ratio_max,
                    "min_pick_s_used": float(mp),
                    "max_pick_s_used": float(xp),
                    "pick_ok": bool(pick_ok),
                    "picker_used": picker_used,
                    "error": "",
                })
    
            except Exception as e:
                pick_rows.append({
                    "net": str(r.net), "sta": str(r.sta), "loc": str(r.loc), "chan": str(r.chan),
                    "filepath": fp,
                    "t0_utc": r.t0_utc, "sps": r.sps,
                    "picked": False, "t_pick_rel_s": np.nan, "t_pick_utc": pd.NaT, "t_obs_s": np.nan,
                    "ratio_max": np.nan,
                    "min_pick_s_used": np.nan, "max_pick_s_used": np.nan,
                    "pick_ok": False,
                    "picker_used": "error",
                    "error": str(e),
                })
    
        picks_df = pd.DataFrame(pick_rows)
    
        # build arrivals_df with ALL stations
        arrivals_df = stations_df.copy()
        arrivals_df["picked"] = False
        arrivals_df["t_pick_utc"] = pd.NaT
        arrivals_df["t_obs_s"] = np.nan
        arrivals_df["chan_used"] = ""
        arrivals_df["loc_used"] = ""
        arrivals_df["ratio_max_used"] = np.nan
        arrivals_df["waveform_file_used"] = ""
        arrivals_df["n_ok_channels"] = 0
        arrivals_df["channel_spread_s"] = np.nan
        arrivals_df["channel_std_s"] = np.nan
        arrivals_df["channels_ok_list"] = ""
        arrivals_df["channel_consistent"] = True
        arrivals_df["notes"] = ""
        arrivals_df["origin_utc"] = origin_utc
        arrivals_df["phase"] = phaseU
    
        # soft-filter output columns (station-level)
        arrivals_df["soft_cluster_n"] = 0
        arrivals_df["soft_cluster_spread_s"] = np.nan
        arrivals_df["soft_cluster_used_chans"] = ""
        arrivals_df["soft_inconsistent_flag"] = False
    
        # hard-filter output columns (station-level)
        if "outlier_flag" not in arrivals_df.columns:
            arrivals_df["outlier_flag"] = False
        if "outlier_score" not in arrivals_df.columns:
            arrivals_df["outlier_score"] = np.nan
        if "residual_s" not in arrivals_df.columns:
            arrivals_df["residual_s"] = np.nan
        if "t_pred_fit_s" not in arrivals_df.columns:
            arrivals_df["t_pred_fit_s"] = np.nan
    
        # local-filter output columns (station-level)
        if "local_outlier_flag" not in arrivals_df.columns:
            arrivals_df["local_outlier_flag"] = False
        if "local_outlier_score" not in arrivals_df.columns:
            arrivals_df["local_outlier_score"] = np.nan
        if "local_n_neighbors" not in arrivals_df.columns:
            arrivals_df["local_n_neighbors"] = 0
        if "local_median" not in arrivals_df.columns:
            arrivals_df["local_median"] = np.nan
        if "local_mad" not in arrivals_df.columns:
            arrivals_df["local_mad"] = np.nan
    
        warn_count = 0
        soft_inconsistent_count = 0
    
        for i, st in arrivals_df.iterrows():
            net = str(st["net"]); sta = str(st["sta"])
            sub = picks_df[(picks_df["net"] == net) & (picks_df["sta"] == sta)].copy()
            if sub.empty:
                arrivals_df.at[i, "notes"] = "NO_WAVEFORMS_FOR_STATION"
                continue
    
            ok = sub[sub["pick_ok"] == True].copy()
            arrivals_df.at[i, "n_ok_channels"] = int(len(ok))
            if ok.empty:
                arrivals_df.at[i, "notes"] = "NO_VALID_PICK_IN_WINDOW"
                continue
    
            # channel consistency stats (DO NOT drop station)
            if validate_other_channels and len(ok) >= int(require_min_channels_for_consistency):
                vals = ok["t_obs_s"].astype(float).values
                spread = float(np.nanmax(vals) - np.nanmin(vals))
                stdv = float(np.nanstd(vals))
                arrivals_df.at[i, "channel_spread_s"] = spread
                arrivals_df.at[i, "channel_std_s"] = stdv
                arrivals_df.at[i, "channels_ok_list"] = ",".join(ok["chan"].astype(str).tolist())
    
                metric = spread if str(channel_consistency_metric).lower() == "spread" else stdv
                if np.isfinite(metric) and metric > float(channel_consistency_warn_s):
                    arrivals_df.at[i, "channel_consistent"] = False
                    arrivals_df.at[i, "notes"] = f"WARNING: channel_{channel_consistency_metric}={metric:.2f}s exceeds {float(channel_consistency_warn_s):.2f}s"
                    warn_count += 1
            else:
                arrivals_df.at[i, "channels_ok_list"] = ",".join(ok["chan"].astype(str).tolist())
    
            # ---- Soft filter (multi-channel clustering) ----
            ok_use = ok.copy()
            if soft_on and len(ok_use) >= 2:
                dfc, meta = self._wilber__soft_cluster_channel_picks(ok_use, soft_cluster_tol_s=float(soft_cluster_tol_s))
                arrivals_df.at[i, "soft_cluster_n"] = int(meta.get("soft_cluster_n", 0))
                arrivals_df.at[i, "soft_cluster_spread_s"] = float(meta.get("soft_cluster_spread_s", np.nan)) if meta.get("soft_cluster_spread_s", None) is not None else np.nan
                arrivals_df.at[i, "soft_cluster_used_chans"] = str(meta.get("soft_cluster_used_chans", ""))
    
                all_spread = float(np.nanmax(ok_use["t_obs_s"].astype(float).values) - np.nanmin(ok_use["t_obs_s"].astype(float).values)) if len(ok_use) > 1 else 0.0
                if int(meta.get("soft_cluster_n", 0)) < 2 and len(ok_use) >= 2 and np.isfinite(all_spread) and all_spread > float(soft_cluster_tol_s):
                    arrivals_df.at[i, "soft_inconsistent_flag"] = True
                    soft_inconsistent_count += 1
                else:
                    arrivals_df.at[i, "soft_inconsistent_flag"] = False
    
                if int(meta.get("soft_cluster_n", 0)) >= 2:
                    ok_use = dfc.copy()
            else:
                arrivals_df.at[i, "soft_cluster_n"] = int(len(ok_use))
                if len(ok_use) >= 2:
                    vals = ok_use["t_obs_s"].astype(float).values
                    arrivals_df.at[i, "soft_cluster_spread_s"] = float(np.nanmax(vals) - np.nanmin(vals))
                    arrivals_df.at[i, "soft_cluster_used_chans"] = ",".join(ok_use["chan"].astype(str).tolist())
                elif len(ok_use) == 1:
                    arrivals_df.at[i, "soft_cluster_spread_s"] = 0.0
                    arrivals_df.at[i, "soft_cluster_used_chans"] = str(ok_use.iloc[0]["chan"])
                arrivals_df.at[i, "soft_inconsistent_flag"] = False
    
            # choose representative pick (from ok_use)
            ok_use["_suffix"] = ok_use["chan"].astype(str).str.upper().str[-1:]
            ok_pref = ok_use[ok_use["_suffix"].isin(prefer_suffix)].copy()
            if ok_pref.empty:
                ok_pref = ok_use.copy()
    
            if str(aggregation).lower() == "earliest":
                chosen = ok_pref.sort_values("t_obs_s", ascending=True).iloc[0]
            else:
                chosen = ok_pref.sort_values(["ratio_max", "t_obs_s"], ascending=[False, True]).iloc[0]
    
            arrivals_df.at[i, "picked"] = True
            arrivals_df.at[i, "t_pick_utc"] = chosen["t_pick_utc"]
            arrivals_df.at[i, "t_obs_s"] = float(chosen["t_obs_s"])
            arrivals_df.at[i, "chan_used"] = str(chosen["chan"])
            arrivals_df.at[i, "loc_used"] = str(chosen["loc"]) if pd.notna(chosen["loc"]) else ""
            arrivals_df.at[i, "ratio_max_used"] = float(chosen["ratio_max"]) if pd.notna(chosen["ratio_max"]) else np.nan
            arrivals_df.at[i, "waveform_file_used"] = str(chosen["filepath"])
    
        # ---- Hard filter (distance vs time) ----  (UNCHANGED)
        outlier_info = None
        if hard_on or bool(outlier_filter):
            if hard_invalidate is None:
                hard_invalidate_use = bool(outlier_hard_filter)
            else:
                hard_invalidate_use = bool(hard_invalidate)
    
            try:
                arrivals_df, outlier_info = self._wilber__station_outlier_filter(
                    arrivals_df,
                    use_only_picked=True,
                    robust_iter=int(outlier_iter),
                    z_mad=float(outlier_z_mad),
                    hard_invalidate=bool(hard_invalidate_use),   # CONSISTENT NOW
                    plot=bool(outlier_plot),
                    plot_path=outlier_plot_path,
                    log=log,
                )
            except Exception as e:
                if log:
                    self._wilber__log(f"[wilber] could not run hard filter: {e}")
                outlier_info = {"error": str(e)}
    
        # ---- Local filter (NEW) ----
        local_info = None
        if local_on:
            # defaults tuned to your “nearby jump” observation
            lk = dict(
                value_col="t_obs_s",
                radius_km=50.0,
                k_min=5,
                z_mad=4.0,
                fallback="use_k_nearest",
                k_nearest=8,
                hard_invalidate=bool(local_invalidate),
                max_flag_frac=0.35,
            )
            if isinstance(local_filter_kwargs, dict):
                lk.update(local_filter_kwargs)
    
            try:
                arrivals_df, local_info = self._wilber__station_local_outlier_filter(
                    arrivals_df,
                    use_only_picked=True,
                    log=log,
                    **lk
                )
            except Exception as e:
                if log:
                    self._wilber__log(f"[wilber] could not run local filter: {e}")
                local_info = {"error": str(e)}
    
        # QC: include errors summary (helps instantly when something breaks again)
        n_errors = int((picks_df["picker_used"] == "error").sum()) if "picker_used" in picks_df.columns else 0
        top_error = ""
        try:
            bad = picks_df[picks_df["error"].astype(str).str.len() > 0]
            if len(bad):
                top_error = str(bad["error"].value_counts().index[0])
        except Exception:
            pass
    
        qc = {
            "N_picks_total": int(len(picks_df)),
            "N_picks_ok": int((picks_df["pick_ok"] == True).sum()),
            "N_station_arrivals": int((arrivals_df["picked"] == True).sum()),
            "N_channel_consistency_warnings": int(warn_count),
            "N_soft_inconsistent": int(soft_inconsistent_count),
            "N_picks_error": int(n_errors),
            "top_error": top_error,
            "t_obs_window_s": (float(t_obs_window_s[0]), float(t_obs_window_s[1])),
            "t_obs_accept_window_s": (float(accept0), float(accept1)),
            "phase": phaseU,
            "filter_mode": mode,
            "soft_cluster_tol_s": float(soft_cluster_tol_s),
            "hard_filter": bool(hard_on or bool(outlier_filter)),
            "hard_invalidate": bool(hard_invalidate if hard_invalidate is not None else outlier_hard_filter),
            "outlier_info": outlier_info if outlier_info is not None else {},
            "local_filter": bool(local_on),
            "local_invalidate": bool(local_invalidate),
            "local_info": local_info if local_info is not None else {},
            "N_local_flagged": int(arrivals_df["local_outlier_flag"].sum()) if "local_outlier_flag" in arrivals_df.columns else 0,
        }
    
        if log:
            self._wilber__log(f"[wilber] picks total={qc['N_picks_total']}, ok={qc['N_picks_ok']}, station arrivals={qc['N_station_arrivals']}")
            self._wilber__log(f"[wilber] pick_ok accept window: [{qc['t_obs_accept_window_s'][0]:.1f}, {qc['t_obs_accept_window_s'][1]:.1f}] s (mode={mode})")
            self._wilber__log(f"[wilber] channel-consistency warnings: {qc['N_channel_consistency_warnings']} (threshold={float(channel_consistency_warn_s):.2f}s, metric={channel_consistency_metric})")
            if soft_on:
                self._wilber__log(f"[wilber] soft filter: tol={float(soft_cluster_tol_s):.2f}s inconsistent={qc['N_soft_inconsistent']}")
            if local_on:
                li = qc.get("local_info", {}) or {}
                self._wilber__log(f"[wilber] local filter: flagged={qc['N_local_flagged']} R={li.get('radius_km', np.nan)}km kmin={li.get('k_min', np.nan)} z={li.get('z_mad', np.nan)}")
            if qc.get("N_picks_error", 0) > 0:
                self._wilber__log(f"[wilber] picker errors: N={qc['N_picks_error']} top='{qc.get('top_error','')}'")
            if outlier_info is not None and isinstance(outlier_info, dict) and len(outlier_info) > 0:
                self._wilber__log(f"[wilber] hard filter: flagged={outlier_info.get('n_flagged', 0)} invalidate={outlier_info.get('hard_invalidate', False)} z={outlier_info.get('z_mad', np.nan)}")
    
        if export_dir:
            out = self._wilber__ensure_dir(self._wilber__as_path(export_dir))
            arrivals_df.to_csv(os.path.join(out, f"{export_prefix}_arrivals_station.csv"), index=False)
            picks_df.to_csv(os.path.join(out, f"{export_prefix}_picks_channel.csv"), index=False)
    
        return arrivals_df, picks_df, qc


    
    # -------------------------------------------------------------------------
    # G) Compute T_map for a model (reuses existing SHAKEpropagate pipeline)
    # -------------------------------------------------------------------------
    # override 
    def wilber_compute_T_map_for_model_impl(
        self,
        *,
        speed_model="vs30_piecewise",
        seed_from="epicenter",
        overrides=None,
    ):
        """
        Compute and return a travel-time map (seconds) for a given model.
        Restores settings after run.
        """
        import numpy as np

        self._wilber__require_grids()
        overrides = dict(overrides or {})

        snap = {}
        for k in vars(self.settings).keys():
            snap[k] = getattr(self.settings, k)

        overrides["speed_model"] = speed_model
        for k, v in overrides.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)

        if seed_from == "epicenter":
            try:
                self._build_point_source()
            except Exception:
                self.set_source_point()
        elif seed_from in ("rupture", "auto"):
            if getattr(self.inputs, "rupture_file", None):
                self.set_source_from_rupture()
            else:
                try:
                    self._build_point_source()
                except Exception:
                    self.set_source_point()
        else:
            try:
                self._build_point_source()
            except Exception:
                self.set_source_point()

        self.compute_speed_map()
        self.compute_travel_time_field()

        T = np.array(self.T_map_s, float, copy=True)

        for k, v in snap.items():
            try:
                setattr(self.settings, k, v)
            except Exception:
                pass

        return T, snap

    # -------------------------------------------------------------------------
    # H) Cartopy scientific map plot (with USGS MMI styling)
    # -------------------------------------------------------------------------
    # override 
    def wilber_plot_cartopy_shakemap_time_contours_and_stations(
        self,
        *,
        T_map,
        arrivals_df,
        outpath=None,
        title=None,
        base_imt="MMI",
        base_alpha=0.70,
        base_cmap=None,
        show_shakemap_colorbar=True,
        contour_levels=None,
        contour_label_fmt="{:.0f}s",
        contour_kwargs=None,
        station_marker="^",
        station_color="red",
        station_edge="k",
        station_size=40,
        show_station_labels=True,
        station_label_fmt="{t:.0f}s",
        station_label_offset=(0.03, 0.03),
        show_unpicked_stations=True,
        unpicked_alpha=0.35,
        picked_alpha=0.95,
        figsize=(11.0, 8.5),
        dpi=300,
        extent_pad_deg=0.7,
        add_ocean=True,
        add_land=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=False,
        gridline_kwargs=None,
        add_legend=True,
        show=True,

        # NEW: flagged plot controls
        show_flagged_labels=True,
        flagged_alpha=0.20,
        flagged_label_alpha=0.35,
        flagged_label_fmt_outlier="{res:+.0f}s",
        flagged_label_fmt_soft="spread={spr:.0f}s",
    ):
        """
        Cartopy plot:
          - basemap
          - ShakeMap raster (MMI uses USGS discrete palette)
          - modeled travel-time contours
          - stations and arrival labels
          - flagged stations plotted (soft/hard flags) with lower alpha and optional labels
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if not _HAS_CARTOPY:
            raise RuntimeError("Cartopy is not available (_HAS_CARTOPY=False). Install cartopy or disable this plot.")

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        self._wilber__require_grids()
        contour_kwargs = dict(contour_kwargs or {})
        gridline_kwargs = dict(gridline_kwargs or {})

        lons = np.asarray(arrivals_df["lon"], float)
        lats = np.asarray(arrivals_df["lat"], float)
        lon_min = np.nanmin(lons); lon_max = np.nanmax(lons)
        lat_min = np.nanmin(lats); lat_max = np.nanmax(lats)
        if not np.isfinite(lon_min):
            lon_min = float(np.nanmin(self.lon_grid)); lon_max = float(np.nanmax(self.lon_grid))
            lat_min = float(np.nanmin(self.lat_grid)); lat_max = float(np.nanmax(self.lat_grid))

        pad = float(extent_pad_deg)
        extent = [lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad]

        Z = self._wilber__get_shakemap_field(base_imt)
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=proj)
        ax.set_extent(extent, crs=proj)

        # Prefer SHAKEmapper-like basemap if available
        try:
            _usgs_basemap(ax, extent)
        except Exception:
            if add_ocean:
                ax.add_feature(cfeature.OCEAN, zorder=0)
            if add_land:
                ax.add_feature(cfeature.LAND, zorder=0)
            if add_borders:
                ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=1)
            if add_coastlines:
                ax.coastlines(resolution="110m", linewidth=0.8, zorder=1)
            if add_gridlines:
                gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5, **gridline_kwargs)
                gl.top_labels = False
                gl.right_labels = False

        # --- ShakeMap raster with USGS MMI styling ---
        im = None
        mmi_mode = str(base_imt).upper().strip() == "MMI"
        if Z is not None:
            if mmi_mode and base_cmap is None:
                cmap, norm, ticks, label = _usgs_mmi_cmap_norm()
                im = ax.pcolormesh(
                    self.lon_grid, self.lat_grid, Z,
                    transform=proj,
                    shading="auto",
                    alpha=float(base_alpha),
                    zorder=2,
                    cmap=cmap,
                    norm=norm
                )
            else:
                im = ax.pcolormesh(
                    self.lon_grid, self.lat_grid, Z,
                    transform=proj,
                    shading="auto",
                    alpha=float(base_alpha),
                    zorder=2,
                    cmap=base_cmap
                )

        # Travel-time contours
        T = np.asarray(T_map, float)
        if contour_levels is None:
            tmax = float(np.nanmax(T))
            step = 20.0 if tmax > 200 else 10.0
            contour_levels = np.arange(0.0, tmax + 0.1 * step, step)

        cs = ax.contour(
            self.lon_grid, self.lat_grid, T,
            levels=contour_levels,
            transform=proj,
            zorder=4,
            **{"colors": "k", "linewidths": 0.9, "alpha": 0.95, **contour_kwargs}
        )
        ax.clabel(cs, inline=True, fmt=lambda v: contour_label_fmt.format(v))

        # Epicenter marker
        try:
            ax.plot(self.event.epicenter_lon, self.event.epicenter_lat,
                    marker="*", markersize=10, color="yellow", markeredgecolor="k",
                    transform=proj, zorder=6)
        except Exception:
            pass

        # Stations
        df = arrivals_df.copy()
        if "station_id" not in df.columns:
            df["station_id"] = df["net"].astype(str) + "." + df["sta"].astype(str)

        picked_mask = df["picked"].astype(bool) if "picked" in df.columns else np.isfinite(df["t_obs_s"].astype(float))
        df_plot = df.copy() if show_unpicked_stations else df[picked_mask].copy()

        # flagged mask (picked stations only)
        soft_flag = df_plot["soft_inconsistent_flag"].astype(bool) if "soft_inconsistent_flag" in df_plot.columns else False
        hard_flag = df_plot["outlier_flag"].astype(bool) if "outlier_flag" in df_plot.columns else False
        flagged_mask = picked_mask & (soft_flag | hard_flag)

        if show_unpicked_stations:
            d0 = df_plot[~picked_mask].copy()
            if len(d0):
                ax.scatter(
                    d0["lon"].astype(float), d0["lat"].astype(float),
                    s=float(station_size),
                    marker=station_marker,
                    c=station_color,
                    edgecolors=station_edge,
                    linewidths=0.4,
                    alpha=float(unpicked_alpha),
                    transform=proj,
                    zorder=5,
                    label="Stations (no valid pick)"
                )

        # unflagged picked
        d1 = df_plot[picked_mask & (~flagged_mask)].copy()
        if len(d1):
            ax.scatter(
                d1["lon"].astype(float), d1["lat"].astype(float),
                s=float(station_size),
                marker=station_marker,
                c=station_color,
                edgecolors=station_edge,
                linewidths=0.4,
                alpha=float(picked_alpha),
                transform=proj,
                zorder=7,
                label="Stations (picked)"
            )

        # flagged picked (still plotted)
        d1f = df_plot[flagged_mask].copy()
        if len(d1f):
            ax.scatter(
                d1f["lon"].astype(float), d1f["lat"].astype(float),
                s=float(station_size),
                marker=station_marker,
                c=station_color,
                edgecolors=station_edge,
                linewidths=0.4,
                alpha=float(flagged_alpha),
                transform=proj,
                zorder=7,
                label="Stations (flagged)"
            )

        dx, dy = float(station_label_offset[0]), float(station_label_offset[1])

        # standard station labels (unflagged by default to keep plot readable)
        if show_station_labels and len(d1):
            for r in d1.itertuples(index=False):
                try:
                    t = float(getattr(r, "t_obs_s"))
                    if not np.isfinite(t):
                        continue
                    ax.text(
                        float(getattr(r, "lon")) + dx,
                        float(getattr(r, "lat")) + dy,
                        station_label_fmt.format(t=t),
                        transform=proj,
                        fontsize=8,
                        color="k",
                        zorder=8,
                        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2)
                    )
                except Exception:
                    continue

        # flagged labels (optional)
        if bool(show_flagged_labels) and len(d1f):
            for r in d1f.itertuples(index=False):
                try:
                    lon = float(getattr(r, "lon"))
                    lat = float(getattr(r, "lat"))
                    lab = None

                    # hard outlier residual label (preferred if available)
                    if hasattr(r, "outlier_flag") and bool(getattr(r, "outlier_flag")) and hasattr(r, "residual_s"):
                        res = float(getattr(r, "residual_s"))
                        if np.isfinite(res):
                            lab = str(flagged_label_fmt_outlier).format(res=res)

                    # soft inconsistency spread label (fallback / or if only soft flag)
                    if lab is None and hasattr(r, "soft_inconsistent_flag") and bool(getattr(r, "soft_inconsistent_flag")):
                        if hasattr(r, "soft_cluster_spread_s"):
                            spr = float(getattr(r, "soft_cluster_spread_s"))
                            if np.isfinite(spr):
                                lab = str(flagged_label_fmt_soft).format(spr=spr)

                    if lab is None:
                        continue

                    ax.text(
                        lon + dx,
                        lat + dy,
                        lab,
                        transform=proj,
                        fontsize=7,
                        color="k",
                        alpha=float(flagged_label_alpha),
                        zorder=9,
                        bbox=dict(facecolor="white", alpha=0.45, edgecolor="none", pad=1.0)
                    )
                except Exception:
                    continue

        # Colorbar(s)
        if show_shakemap_colorbar and im is not None:
            cbar = plt.colorbar(im, ax=ax, shrink=0.80, pad=0.02)
            if mmi_mode:
                _, _, ticks, label = _usgs_mmi_cmap_norm()
                cbar.set_ticks(ticks)
                cbar.set_label(label)
            else:
                cbar.set_label(base_imt)

        ax.set_title(title or "SHAKEmap background + modeled travel-time contours + station arrivals")

        if add_legend:
            try:
                ax.legend(loc="lower left", framealpha=0.85)
            except Exception:
                pass

        if outpath:
            outpath = self._wilber__as_path(outpath)
            self._wilber__ensure_dir(__import__("os").path.dirname(outpath))
            fig.savefig(outpath, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig



    def wilber_plot_cartopy_shakemap_time_contours_and_stations(
        self,
        *,
        T_map,
        arrivals_df,
        outpath=None,
        title=None,
        base_imt="MMI",
        base_alpha=0.70,
        base_cmap=None,
        show_shakemap_colorbar=True,
        contour_levels=None,
        contour_label_fmt="{:.0f}s",
        contour_kwargs=None,
        station_marker="^",
        station_color="red",
        station_edge="k",
        station_size=40,
        show_station_labels=True,
        station_label_fmt="{t:.0f}s",
        station_label_offset=(0.03, 0.03),
        show_unpicked_stations=True,
        unpicked_alpha=0.35,
        picked_alpha=0.95,
        figsize=(11.0, 8.5),
        dpi=300,
        extent_pad_deg=0.7,
        add_ocean=True,
        add_land=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=False,
        gridline_kwargs=None,
        add_legend=True,
        show=True,
    
        # NEW: extent control (DEFAULT: shakemap)
        extent_mode="shakemap",      # "shakemap" | "stations" | "grid" | "manual"
        extent=None,                 # [lonmin, lonmax, latmin, latmax] if extent_mode="manual"
        clip_stations_to_extent=True,  # <- do not plot anything outside extent (your request)
    
        # NEW: flagged plot controls
        show_flagged_labels=True,
        flagged_alpha=0.20,
        flagged_label_alpha=0.35,
        flagged_label_fmt_outlier="{res:+.0f}s",
        flagged_label_fmt_soft="spread={spr:.0f}s",
    ):
        """
        Cartopy plot:
          - basemap
          - ShakeMap raster (MMI uses USGS discrete palette)
          - modeled travel-time contours
          - stations and arrival labels
          - flagged stations plotted (soft/hard flags) with lower alpha and optional labels
    
        Extent behavior (matches your residual/rate maps philosophy):
          - DEFAULT: extent_mode="shakemap" (ShakeMap grid bounds + pad)
          - extent_mode="stations" uses station bounds + pad
          - extent_mode="grid" uses lon_grid/lat_grid bounds + pad (same as shakemap here)
          - extent_mode="manual" uses user `extent` exactly (no pad)
        Clipping:
          - clip_stations_to_extent=True removes stations outside map extent
            (so nothing is plotted outside the map).
        """
        import numpy as np
        import matplotlib.pyplot as plt
    
        if not _HAS_CARTOPY:
            raise RuntimeError("Cartopy is not available (_HAS_CARTOPY=False). Install cartopy or disable this plot.")
    
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    
        self._wilber__require_grids()
        contour_kwargs = dict(contour_kwargs or {})
        gridline_kwargs = dict(gridline_kwargs or {})
    
        # -------------------------
        # extent resolution (like your residual function)
        # -------------------------
        pad = float(extent_pad_deg)
    
        def _finite_bounds(x):
            x = np.asarray(x, float).ravel()
            x = x[np.isfinite(x)]
            if x.size == 0:
                return None
            return float(np.min(x)), float(np.max(x))
    
        def _extent_from_stations(df):
            lons = np.asarray(df["lon"], float)
            lats = np.asarray(df["lat"], float)
            lb = _finite_bounds(lons)
            ab = _finite_bounds(lats)
            if lb is None or ab is None:
                return None
            return [lb[0] - pad, lb[1] + pad, ab[0] - pad, ab[1] + pad]
    
        def _extent_from_shakemap_grid():
            lonG = getattr(self, "lon_grid", None)
            latG = getattr(self, "lat_grid", None)
            if lonG is None or latG is None:
                lonG = getattr(self, "_wilber_last_lon_grid", None)
                latG = getattr(self, "_wilber_last_lat_grid", None)
            if lonG is None or latG is None:
                raise ValueError("extent_mode='shakemap' requested but no lon/lat grid found on self.")
            lonG = np.asarray(lonG, dtype=float)
            latG = np.asarray(latG, dtype=float)
            return [
                float(np.nanmin(lonG) - pad),
                float(np.nanmax(lonG) + pad),
                float(np.nanmin(latG) - pad),
                float(np.nanmax(latG) + pad),
            ]
    
        if extent is not None:
            extent_mode_use = "manual"
        else:
            extent_mode_use = str(extent_mode).strip().lower()
    
        if extent_mode_use == "manual":
            if extent is None or len(extent) != 4:
                raise ValueError("extent_mode='manual' requires extent=[lonmin, lonmax, latmin, latmax].")
            extent_use = list(map(float, extent))
        elif extent_mode_use in ("shakemap", "grid"):
            extent_use = _extent_from_shakemap_grid()
        elif extent_mode_use == "stations":
            # if stations extent can't be computed, fallback to shakemap
            extent_use = _extent_from_stations(arrivals_df)
            if extent_use is None:
                extent_use = _extent_from_shakemap_grid()
        else:
            # safe fallback
            extent_use = _extent_from_shakemap_grid()
    
        # -------------------------
        # setup figure/axes
        # -------------------------
        Z = self._wilber__get_shakemap_field(base_imt)
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=proj)
        ax.set_extent(extent_use, crs=proj)
    
        # Prefer SHAKEmapper-like basemap if available
        try:
            _usgs_basemap(ax, extent_use)
        except Exception:
            if add_ocean:
                ax.add_feature(cfeature.OCEAN, zorder=0)
            if add_land:
                ax.add_feature(cfeature.LAND, zorder=0)
            if add_borders:
                ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=1)
            if add_coastlines:
                ax.coastlines(resolution="110m", linewidth=0.8, zorder=1)
            if add_gridlines:
                gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5, **gridline_kwargs)
                gl.top_labels = False
                gl.right_labels = False
    
        # --- ShakeMap raster with USGS MMI styling ---
        im = None
        mmi_mode = str(base_imt).upper().strip() == "MMI"
        if Z is not None:
            if mmi_mode and base_cmap is None:
                cmap, norm, ticks, label = _usgs_mmi_cmap_norm()
                im = ax.pcolormesh(
                    self.lon_grid, self.lat_grid, Z,
                    transform=proj,
                    shading="auto",
                    alpha=float(base_alpha),
                    zorder=2,
                    cmap=cmap,
                    norm=norm
                )
            else:
                im = ax.pcolormesh(
                    self.lon_grid, self.lat_grid, Z,
                    transform=proj,
                    shading="auto",
                    alpha=float(base_alpha),
                    zorder=2,
                    cmap=base_cmap
                )
    
        # Travel-time contours
        T = np.asarray(T_map, float)
        if contour_levels is None:
            tmax = float(np.nanmax(T))
            step = 20.0 if tmax > 200 else 10.0
            contour_levels = np.arange(0.0, tmax + 0.1 * step, step)
    
        cs = ax.contour(
            self.lon_grid, self.lat_grid, T,
            levels=contour_levels,
            transform=proj,
            zorder=4,
            **{"colors": "k", "linewidths": 0.9, "alpha": 0.95, **contour_kwargs}
        )
        ax.clabel(cs, inline=True, fmt=lambda v: contour_label_fmt.format(v))
    
        # Epicenter marker
        try:
            ax.plot(
                self.event.epicenter_lon, self.event.epicenter_lat,
                marker="*", markersize=10, color="yellow", markeredgecolor="k",
                transform=proj, zorder=6
            )
        except Exception:
            pass
    
        # -------------------------
        # Stations (clip to extent if requested)
        # -------------------------
        df = arrivals_df.copy()
        if "station_id" not in df.columns:
            df["station_id"] = df["net"].astype(str) + "." + df["sta"].astype(str)
    
        # Ensure numeric lon/lat
        df["lon"] = np.asarray(df["lon"], float)
        df["lat"] = np.asarray(df["lat"], float)
    
        if bool(clip_stations_to_extent):
            eps = 1e-9
            in_ext = (
                (df["lon"] >= extent_use[0] - eps) &
                (df["lon"] <= extent_use[1] + eps) &
                (df["lat"] >= extent_use[2] - eps) &
                (df["lat"] <= extent_use[3] + eps)
            )
            df = df.loc[in_ext].copy()
    
        picked_mask = df["picked"].astype(bool) if "picked" in df.columns else np.isfinite(df["t_obs_s"].astype(float))
        df_plot = df.copy() if show_unpicked_stations else df.loc[picked_mask].copy()
    
        # flagged mask (picked stations only)
        soft_flag = df_plot["soft_inconsistent_flag"].astype(bool) if "soft_inconsistent_flag" in df_plot.columns else False
        hard_flag = df_plot["outlier_flag"].astype(bool) if "outlier_flag" in df_plot.columns else False
        flagged_mask = picked_mask & (soft_flag | hard_flag)
    
        if show_unpicked_stations:
            d0 = df_plot.loc[~picked_mask].copy()
            if len(d0):
                ax.scatter(
                    d0["lon"].astype(float), d0["lat"].astype(float),
                    s=float(station_size),
                    marker=station_marker,
                    c=station_color,
                    edgecolors=station_edge,
                    linewidths=0.4,
                    alpha=float(unpicked_alpha),
                    transform=proj,
                    zorder=5,
                    label="Stations (no valid pick)"
                )
    
        # unflagged picked
        d1 = df_plot.loc[picked_mask & (~flagged_mask)].copy()
        if len(d1):
            ax.scatter(
                d1["lon"].astype(float), d1["lat"].astype(float),
                s=float(station_size),
                marker=station_marker,
                c=station_color,
                edgecolors=station_edge,
                linewidths=0.4,
                alpha=float(picked_alpha),
                transform=proj,
                zorder=7,
                label="Stations (picked)"
            )
    
        # flagged picked (still plotted)
        d1f = df_plot.loc[flagged_mask].copy()
        if len(d1f):
            ax.scatter(
                d1f["lon"].astype(float), d1f["lat"].astype(float),
                s=float(station_size),
                marker=station_marker,
                c=station_color,
                edgecolors=station_edge,
                linewidths=0.4,
                alpha=float(flagged_alpha),
                transform=proj,
                zorder=7,
                label="Stations (flagged)"
            )
    
        dx, dy = float(station_label_offset[0]), float(station_label_offset[1])
    
        # standard station labels (unflagged by default to keep plot readable)
        if show_station_labels and len(d1):
            for r in d1.itertuples(index=False):
                try:
                    t = float(getattr(r, "t_obs_s"))
                    if not np.isfinite(t):
                        continue
                    ax.text(
                        float(getattr(r, "lon")) + dx,
                        float(getattr(r, "lat")) + dy,
                        station_label_fmt.format(t=t),
                        transform=proj,
                        fontsize=8,
                        color="k",
                        zorder=8,
                        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2)
                    )
                except Exception:
                    continue
    
        # flagged labels (optional)
        if bool(show_flagged_labels) and len(d1f):
            for r in d1f.itertuples(index=False):
                try:
                    lon = float(getattr(r, "lon"))
                    lat = float(getattr(r, "lat"))
                    lab = None
    
                    # hard outlier residual label (preferred if available)
                    if hasattr(r, "outlier_flag") and bool(getattr(r, "outlier_flag")) and hasattr(r, "residual_s"):
                        res = float(getattr(r, "residual_s"))
                        if np.isfinite(res):
                            lab = str(flagged_label_fmt_outlier).format(res=res)
    
                    # soft inconsistency spread label (fallback)
                    if lab is None and hasattr(r, "soft_inconsistent_flag") and bool(getattr(r, "soft_inconsistent_flag")):
                        if hasattr(r, "soft_cluster_spread_s"):
                            spr = float(getattr(r, "soft_cluster_spread_s"))
                            if np.isfinite(spr):
                                lab = str(flagged_label_fmt_soft).format(spr=spr)
    
                    if lab is None:
                        continue
    
                    ax.text(
                        lon + dx,
                        lat + dy,
                        lab,
                        transform=proj,
                        fontsize=7,
                        color="k",
                        alpha=float(flagged_label_alpha),
                        zorder=9,
                        bbox=dict(facecolor="white", alpha=0.45, edgecolor="none", pad=1.0)
                    )
                except Exception:
                    continue
    
        # Colorbar(s)
        if show_shakemap_colorbar and im is not None:
            cbar = plt.colorbar(im, ax=ax, shrink=0.80, pad=0.02)
            if mmi_mode:
                _, _, ticks, label = _usgs_mmi_cmap_norm()
                cbar.set_ticks(ticks)
                cbar.set_label(label)
            else:
                cbar.set_label(base_imt)
    
        ax.set_title(title or "SHAKEmap background + modeled travel-time contours + station arrivals")
    
        # Legend FIX: force legend above ocean/land (and _usgs_basemap layers)
        if add_legend:
            try:
                leg = ax.legend(loc="lower left", framealpha=0.85)
                if leg is not None:
                    leg.set_zorder(2000)
            except Exception:
                pass
    
        if outpath:
            outpath = self._wilber__as_path(outpath)
            self._wilber__ensure_dir(__import__("os").path.dirname(outpath))
            fig.savefig(outpath, bbox_inches="tight")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig
    
    
    










    # -------------------------------------------------------------------------
    # W) DEBUG / VISUAL QC TOOLS — waveform visualization + internal picking
    # -------------------------------------------------------------------------
    def _wilber__haversine_km(self, lon1, lat1, lon2, lat2):
        """
        Local haversine (km). Uses global haversine_km if present, else internal.
        Accepts scalars; returns float.
        """
        try:
            return float(haversine_km(lon1, lat1, lon2, lat2))  # noqa: F821
        except Exception:
            import numpy as np
            R = 6371.0
            lon1 = np.deg2rad(float(lon1)); lat1 = np.deg2rad(float(lat1))
            lon2 = np.deg2rad(float(lon2)); lat2 = np.deg2rad(float(lat2))
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
            c = 2.0*np.arcsin(np.sqrt(a))
            return float(R*c)

    def _wilber__standardize_station_id(self, station_id):
        """
        Normalize station id to "NET.STA" or "NET.STA.LOC.CHAN" depending on input.
        """
        s = str(station_id).strip()
        s = s.replace("_", ".")
        parts = [p for p in s.split(".") if p != ""]
        # allow "NET.STA" or "NET.STA.LOC.CHAN"
        if len(parts) >= 2:
            return ".".join(parts[:4]) if len(parts) >= 4 else ".".join(parts[:2])
        return s

    def wilber_list_station_waveforms(
        self,
        wave_index_df,
        station_id,
        *,
        include_loc_chan=True,
        sort_by=("chan", "t0_utc"),
        log=True,
        return_pandas=True,
    ):
        """
        List available waveforms for a given station_id (NET.STA or NET.STA.LOC.CHAN).
        Input wave_index_df should come from wilber_index_waveform_folder().

        Returns subset DataFrame with at least:
          filepath, net, sta, loc, chan, t0_utc, sps, nsamp, valid_header
        """
        import pandas as pd

        sid = self._wilber__standardize_station_id(station_id)
        parts = [p for p in sid.split(".") if p != ""]
        net = parts[0] if len(parts) > 0 else ""
        sta = parts[1] if len(parts) > 1 else ""
        loc = parts[2] if len(parts) > 2 else None
        chan = parts[3] if len(parts) > 3 else None

        wi = wave_index_df.copy()
        if "valid_header" in wi.columns:
            wi = wi[wi["valid_header"] == True].copy()

        wi["net"] = wi["net"].astype(str).str.strip()
        wi["sta"] = wi["sta"].astype(str).str.strip()
        wi["loc"] = wi["loc"].astype(str).fillna("").str.strip()
        wi["chan"] = wi["chan"].astype(str).str.strip()

        sub = wi[(wi["net"] == net) & (wi["sta"] == sta)].copy()
        if include_loc_chan and loc is not None:
            sub = sub[sub["loc"] == str(loc)].copy()
        if include_loc_chan and chan is not None:
            sub = sub[sub["chan"] == str(chan)].copy()

        # sorting
        try:
            if sort_by:
                sub = sub.sort_values(list(sort_by), ascending=True)
        except Exception:
            pass

        if log:
            self._wilber__log(f"[wilber-viz] station {net}.{sta}: {len(sub)} waveform files")

        return sub if return_pandas else sub.to_dict("records")

    def _wilber__choose_waveform_file(
        self,
        wave_index_df,
        station_id,
        *,
        prefer_chans=("HHZ", "BHZ", "EHZ", "HNZ", "HHN", "HHE", "BHN", "BHE", "EHN", "EHE"),
        channel_family_filter=None,
        longest_preferred=True,
        log=True,
    ):
        """
        Pick a single representative waveform file for station_id from wave_index_df.
        Preference:
          1) prefer_chans exact match
          2) otherwise any channel family filter match (e.g., 'HH')
          3) otherwise longest nsamp
        """
        import numpy as np
        import pandas as pd

        sub = self.wilber_list_station_waveforms(wave_index_df, station_id, log=False, return_pandas=True)
        if sub.empty:
            if log:
                self._wilber__log(f"[wilber-viz] no waveforms found for {station_id}")
            return None

        # optional channel family filter
        if channel_family_filter:
            fam = str(channel_family_filter).upper().strip()
            sub = sub[sub["chan"].astype(str).str.upper().str.startswith(fam)].copy()
            if sub.empty:
                return None

        sub["_chanU"] = sub["chan"].astype(str).str.upper()
        prefer_set = [str(c).upper() for c in (prefer_chans or ())]
        sub["_is_pref"] = sub["_chanU"].isin(prefer_set)

        # ensure numeric nsamp
        sub["_nsamp"] = pd.to_numeric(sub.get("nsamp", np.nan), errors="coerce")
        sub["_nsamp"] = sub["_nsamp"].fillna(-1)

        if sub["_is_pref"].any():
            cand = sub[sub["_is_pref"] == True].copy()
            if longest_preferred:
                cand = cand.sort_values(["_nsamp", "t0_utc"], ascending=[False, True])
            else:
                cand = cand.sort_values(["t0_utc"], ascending=True)
            return str(cand.iloc[0]["filepath"])

        # fallback: longest file
        sub = sub.sort_values(["_nsamp", "t0_utc"], ascending=[False, True])
        return str(sub.iloc[0]["filepath"])

    def _wilber__simple_predicted_arrivals(
        self,
        station_lon,
        station_lat,
        origin_utc,
        *,
        vP_km_s=6.0,
        vS_km_s=3.5,
        extra_delay_s=0.0,
    ):
        """
        Simple predicted arrivals from epicentral distance using constant velocities.
        Returns dict with tP_s, tS_s (seconds after origin). If event coords missing -> NaN.
        """
        import numpy as np

        try:
            evlon = float(self.event.epicenter_lon)
            evlat = float(self.event.epicenter_lat)
        except Exception:
            return {"tP_s": np.nan, "tS_s": np.nan, "dist_km": np.nan}

        d_km = self._wilber__haversine_km(float(station_lon), float(station_lat), evlon, evlat)
        tP = (d_km / float(vP_km_s)) + float(extra_delay_s) if vP_km_s and vP_km_s > 0 else np.nan
        tS = (d_km / float(vS_km_s)) + float(extra_delay_s) if vS_km_s and vS_km_s > 0 else np.nan
        return {"tP_s": float(tP), "tS_s": float(tS), "dist_km": float(d_km)}

    def _wilber__prep_trace_for_plot(
        self,
        x,
        sps,
        *,
        detrend=True,
        normalize="none",   # "none" | "maxabs" | "std"
        clip_percentile=None,
        decimate_to_hz=None,
    ):
        """
        Lightweight preprocessing for plotting only.
        Returns (x_plot, sps_plot, scale_info)
        """
        import numpy as np

        xx = np.asarray(x, float).copy()
        fs = float(sps)

        if detrend:
            try:
                xx = xx - np.nanmedian(xx)
            except Exception:
                pass

        # decimate (simple stride) for speed
        if decimate_to_hz is not None:
            target = float(decimate_to_hz)
            if target > 0 and fs > target:
                step = int(max(1, round(fs / target)))
                xx = xx[::step]
                fs = fs / step

        # optional clipping for visualization
        if clip_percentile is not None:
            p = float(clip_percentile)
            p = min(max(p, 50.0), 100.0)
            lim = np.nanpercentile(np.abs(xx), p)
            if np.isfinite(lim) and lim > 0:
                xx = np.clip(xx, -lim, lim)

        scale_info = {"normalize": str(normalize)}
        nmode = str(normalize).lower().strip()
        if nmode == "maxabs":
            m = float(np.nanmax(np.abs(xx))) if np.any(np.isfinite(xx)) else 0.0
            if m > 0:
                xx = xx / m
                scale_info["scale"] = m
        elif nmode == "std":
            sd = float(np.nanstd(xx)) if np.any(np.isfinite(xx)) else 0.0
            if sd > 0:
                xx = xx / sd
                scale_info["scale"] = sd

        return xx, fs, scale_info

    def wilber_debug_plot_waveform_pick(
        self,
        *,
        filepath=None,
        wave_index_df=None,
        station_id=None,
        # time control
        origin_time_utc="from_xml",
        xml_path_for_origin=None,
        t_obs_window_s=(0.0, 500.0),
        extra_window_pad_s=30.0,   # extra pad around zoom window for context
        # picker control
        picker="stalta",           # "stalta" | "obspy_stalta" | "obspy_ar"
        picker_kwargs=None,
        min_ratio_max=2.0,
        # predicted arrivals (simple)
        show_predicted=True,
        vP_km_s=6.0,
        vS_km_s=3.5,
        extra_delay_s=0.0,
        # plot formatting
        detrend=True,
        normalize="none",          # "none" | "maxabs" | "std"
        clip_percentile=None,
        decimate_to_hz=50.0,
        figsize=(12.0, 6.5),
        dpi=200,
        title=None,
        show=True,
        save=False,
        outpath=None,
        # styling knobs
        alpha_full=0.85,
        alpha_zoom=0.95,
        lw_full=0.8,
        lw_zoom=0.9,
        show_grid=True,
        show_legend=True,
        log=True,
    ):
        """
        Visual QC plot for ONE waveform:
          - Top: full trace
          - Bottom: zoomed window around [origin+t_obs_window_s] mapped to trace time
          - Overlays: origin, predicted P/S (optional), picked time (if any)

        Selection:
          - If filepath provided: uses it.
          - Else requires (wave_index_df and station_id) to choose a representative file.

        Notes:
          - Uses internal picker by default (stalta). ObsPy pickers fall back safely
            only if you call them here via the wrapper logic below.
          - This function does NOT write into arrivals tables; it’s purely for inspection.
        """
        import os
        import numpy as np
        import pandas as pd
        import datetime as dt
        import matplotlib.pyplot as plt

        picker_kwargs = dict(picker_kwargs or {})

        # resolve filepath if not provided
        if filepath is None:
            if wave_index_df is None or station_id is None:
                raise ValueError("Provide either filepath OR (wave_index_df and station_id).")
            filepath = self._wilber__choose_waveform_file(
                wave_index_df,
                station_id,
                prefer_chans=picker_kwargs.pop("prefer_chans", ("HHZ","BHZ","EHZ","HNZ","HHN","HHE")),
                channel_family_filter=picker_kwargs.pop("channel_family_filter", None),
                longest_preferred=True,
                log=log,
            )
            if filepath is None:
                raise ValueError(f"No waveform file found for station_id={station_id}.")

        # read waveform
        tr = self.wilber_read_timeseries(filepath)
        x = tr["x"]
        sps = float(tr["sps"])
        t0 = tr["t0_utc"]

        # origin time
        origin_utc = self._wilber__get_origin_time(origin_time_utc, xml_path_for_origin)

        # window mapping: origin-relative window -> trace-relative seconds
        min_pick_s, max_pick_s = self._wilber__pick_window_from_origin(t0, origin_utc, t_obs_window_s)
        # clamp window to trace bounds (important for short traces)
        dur_s = (len(x) - 1) / sps if len(x) > 1 else 0.0
        min_pick_s_cl = float(max(0.0, min(min_pick_s, dur_s)))
        max_pick_s_cl = float(max(min_pick_s_cl, min(max_pick_s, dur_s)))

        # build time axis for plotting
        t_s = tr["t_s"]

        # pick using requested picker (with safe ObsPy fallback)
        picker_used = str(picker)
        pk = {"picked": False, "t_pick_rel_s": np.nan, "idx": -1, "ratio_max": np.nan}

        mp = float(picker_kwargs.pop("min_pick_s", min_pick_s_cl))
        xp = picker_kwargs.pop("max_pick_s", max_pick_s_cl)
        xp = float(xp) if xp is not None else float(max_pick_s_cl)

        try:
            if picker == "stalta":
                pk = self.wilber_pick_arrival_stalta(t_s, x, sps, min_pick_s=mp, max_pick_s=xp, **picker_kwargs)
            elif picker == "obspy_stalta":
                try:
                    pk = self.wilber_pick_arrival_obspy_stalta(t_s, x, sps, min_pick_s=mp, max_pick_s=xp, **picker_kwargs)
                except Exception as e:
                    picker_used = "stalta_fallback"
                    if log:
                        self._wilber__log(f"[wilber-viz][obspy_stalta] failed: {e} -> fallback to internal stalta")
                    pk = self.wilber_pick_arrival_stalta(t_s, x, sps, min_pick_s=mp, max_pick_s=xp, **picker_kwargs)
            elif picker == "obspy_ar":
                try:
                    pk = self.wilber_pick_arrival_obspy_ar(t_s, x, sps, min_pick_s=mp, max_pick_s=xp, **picker_kwargs)
                except Exception as e:
                    picker_used = "stalta_fallback"
                    if log:
                        self._wilber__log(f"[wilber-viz][obspy_ar] failed: {e} -> fallback to internal stalta")
                    pk = self.wilber_pick_arrival_stalta(t_s, x, sps, min_pick_s=mp, max_pick_s=xp, **picker_kwargs)
            else:
                raise ValueError("picker must be one of: 'stalta','obspy_stalta','obspy_ar'")
        except Exception as e:
            if log:
                self._wilber__log(f"[wilber-viz] picker error for {filepath}: {e}")

        picked = bool(pk.get("picked", False))
        t_pick_rel_s = float(pk.get("t_pick_rel_s", np.nan)) if picked else np.nan
        ratio_max = float(pk.get("ratio_max", np.nan))

        t_pick_utc = (t0 + dt.timedelta(seconds=t_pick_rel_s)) if picked and np.isfinite(t_pick_rel_s) else pd.NaT
        t_obs_s = (t_pick_utc - origin_utc).total_seconds() if picked and t_pick_utc is not pd.NaT else np.nan

        pick_ok = (
            picked
            and np.isfinite(t_obs_s)
            and (float(t_obs_window_s[0]) <= float(t_obs_s) <= float(t_obs_window_s[1]))
            and ((ratio_max >= float(min_ratio_max)) if np.isfinite(ratio_max) else True)
        )

        # preprocess for plotting
        x_full, fs_plot, scale_info = self._wilber__prep_trace_for_plot(
            x, sps,
            detrend=detrend,
            normalize=normalize,
            clip_percentile=clip_percentile,
            decimate_to_hz=decimate_to_hz,
        )
        t_full = np.arange(len(x_full), dtype=float) / float(fs_plot)

        # window bounds in plot sampling
        min_plot_s = float(min_pick_s_cl)
        max_plot_s = float(max_pick_s_cl)
        pad = float(extra_window_pad_s)
        z0 = max(0.0, min_plot_s - pad)
        z1 = min(float(t_full[-1]) if len(t_full) else 0.0, max_plot_s + pad)

        # predicted arrivals (simple, based on epicentral distance)
        pred = {"tP_s": np.nan, "tS_s": np.nan, "dist_km": np.nan}
        if show_predicted:
            try:
                pred = self._wilber__simple_predicted_arrivals(
                    station_lon=float(tr.get("lon", np.nan)) if "lon" in tr else np.nan,
                    station_lat=float(tr.get("lat", np.nan)) if "lat" in tr else np.nan,
                    origin_utc=origin_utc,
                    vP_km_s=vP_km_s, vS_km_s=vS_km_s,
                    extra_delay_s=extra_delay_s,
                )
            except Exception:
                # try to infer station coords if station list stored somewhere else (optional)
                pred = {"tP_s": np.nan, "tS_s": np.nan, "dist_km": np.nan}

        # plot
        fig = plt.figure(figsize=figsize, dpi=dpi)

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(t_full, x_full, linewidth=lw_full, alpha=alpha_full,color="0.25", label="Full trace")
        ax1.axvline(min_plot_s, linewidth=1.1,color="#1f77b4", alpha=0.7, label="Pick window start")
        ax1.axvline(max_plot_s, linewidth=1.1, color="#ff7f0e",alpha=0.7, label="Pick window end")

        # origin line in trace-relative seconds
        # origin relative to trace start: dt0 = origin - t0
        dt0 = (origin_utc - t0).total_seconds()
        if np.isfinite(dt0):
            ax1.axvline(max(0.0, float(dt0)), color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8, label="Origin (mapped)")

        # picked line
        if picked and np.isfinite(t_pick_rel_s):
            ax1.axvline(float(t_pick_rel_s), linestyle="-", color="#d62728",  linewidth=1.6, alpha=0.9, label="Picked")

        if show_grid:
            ax1.grid(True, alpha=0.25)

        ax1.set_ylabel("Amplitude (arb.)")

        # Zoom panel
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        m = (t_full >= z0) & (t_full <= z1)
        if np.any(m):
            ax2.plot(t_full[m], x_full[m], linewidth=lw_zoom, alpha=alpha_zoom, label="Zoomed window")
        ax2.axvline(min_plot_s, linewidth=1.1, alpha=0.7)
        ax2.axvline(max_plot_s, linewidth=1.1, alpha=0.7)
        if np.isfinite(dt0):
            ax2.axvline(max(0.0, float(dt0)), linestyle="--", linewidth=1.2, alpha=0.8)
        if picked and np.isfinite(t_pick_rel_s):
            ax2.axvline(float(t_pick_rel_s), linestyle="-", linewidth=1.6, alpha=0.9)

        if show_grid:
            ax2.grid(True, alpha=0.25)

        ax2.set_xlabel("Seconds since trace start")
        ax2.set_ylabel("Amplitude (arb.)")

        # title + annotation
        base = os.path.basename(str(filepath))
        lab = title or f"Wilber waveform pick QC — {base}"
        fig.suptitle(lab, fontsize=12)

        info_lines = [
            f"picker={picker_used}  picked={picked}  ok={pick_ok}",
            f"origin_utc={origin_utc}  trace_t0={t0}",
            f"t_obs_window_s={tuple(map(float, t_obs_window_s))}  mapped_win=[{min_plot_s:.1f},{max_plot_s:.1f}]s",
        ]
        if picked:
            info_lines.append(f"t_pick_rel={t_pick_rel_s:.2f}s  t_obs={t_obs_s:.2f}s  ratio_max={ratio_max if np.isfinite(ratio_max) else np.nan:.2f}")
        else:
            info_lines.append("no pick")
        if show_predicted and np.isfinite(pred.get("dist_km", np.nan)):
            info_lines.append(f"dist~{pred['dist_km']:.1f}km  tP~{pred['tP_s']:.1f}s  tS~{pred['tS_s']:.1f}s  (const vP/vS)")

        ax2.text(
            0.01, 0.02,
            "\n".join(info_lines),
            transform=ax2.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=6)
        )

        if show_legend:
            try:
                ax1.legend(loc="upper right", framealpha=0.85, fontsize=8)
            except Exception:
                pass

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save:
            if not outpath:
                raise ValueError("save=True requires outpath.")
            outpath = self._wilber__as_path(outpath)
            self._wilber__ensure_dir(os.path.dirname(outpath))
            fig.savefig(outpath, bbox_inches="tight")
            if log:
                self._wilber__log(f"[wilber-viz] saved: {outpath}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return {
            "filepath": str(filepath),
            "picker_used": picker_used,
            "picked": bool(picked),
            "pick_ok": bool(pick_ok),
            "t_pick_rel_s": float(t_pick_rel_s) if np.isfinite(t_pick_rel_s) else np.nan,
            "t_obs_s": float(t_obs_s) if np.isfinite(t_obs_s) else np.nan,
            "ratio_max": float(ratio_max) if np.isfinite(ratio_max) else np.nan,
            "origin_utc": origin_utc,
            "trace_t0_utc": t0,
            "min_pick_s_used": float(mp),
            "max_pick_s_used": float(xp),
        }

    def wilber_debug_plot_random_station_files(
        self,
        wave_index_df,
        *,
        n=10,
        seed=123,
        origin_time_utc="from_xml",
        xml_path_for_origin=None,
        t_obs_window_s=(0.0, 500.0),
        picker="stalta",
        picker_kwargs=None,
        min_ratio_max=2.0,
        station_subset=None,   # optional iterable of "NET.STA"
        prefer_chans=("HHZ","BHZ","EHZ","HNZ"),
        channel_family_filter=None,
        decimate_to_hz=50.0,
        normalize="none",
        clip_percentile=None,
        figsize=(12.0, 6.5),
        dpi=200,
        show=True,
        save=False,
        outdir=None,
        fname_prefix="wilber_pick_qc",
        log=True,
    ):
        """
        Plot QC figures for random stations (one representative file per station).

        - Picks a station at random from wave_index_df (valid headers only),
          optionally restricted to station_subset.
        - Chooses a representative file per station (prefer_chans then longest).
        - Calls wilber_debug_plot_waveform_pick().

        If save=True: writes PNGs into outdir.
        """
        import os
        import numpy as np
        import pandas as pd

        picker_kwargs = dict(picker_kwargs or {})

        wi = wave_index_df.copy()
        if "valid_header" in wi.columns:
            wi = wi[wi["valid_header"] == True].copy()

        wi["net"] = wi["net"].astype(str).str.strip()
        wi["sta"] = wi["sta"].astype(str).str.strip()
        wi["station_id"] = wi["net"] + "." + wi["sta"]

        if station_subset is not None:
            ss = set([self._wilber__standardize_station_id(s).split(".")[0] + "." +
                      self._wilber__standardize_station_id(s).split(".")[1] for s in station_subset])
            wi = wi[wi["station_id"].isin(ss)].copy()

        st = wi["station_id"].dropna().drop_duplicates().tolist()
        if len(st) == 0:
            raise ValueError("No stations available for random selection (check wave_index_df / station_subset).")

        rng = np.random.default_rng(int(seed))
        chosen = list(rng.choice(st, size=int(min(n, len(st))), replace=False))

        results = []
        for k, sid in enumerate(chosen, start=1):
            fp = self._wilber__choose_waveform_file(
                wi,
                sid,
                prefer_chans=prefer_chans,
                channel_family_filter=channel_family_filter,
                longest_preferred=True,
                log=False,
            )
            if fp is None:
                continue

            outpath = None
            if save:
                if not outdir:
                    raise ValueError("save=True requires outdir.")
                outdirN = self._wilber__ensure_dir(self._wilber__as_path(outdir))
                outpath = os.path.join(outdirN, f"{fname_prefix}_{k:03d}_{sid.replace('.','_')}.png")

            if log:
                self._wilber__log(f"[wilber-viz] ({k}/{len(chosen)}) station {sid} -> {os.path.basename(fp)}")

            res = self.wilber_debug_plot_waveform_pick(
                filepath=fp,
                origin_time_utc=origin_time_utc,
                xml_path_for_origin=xml_path_for_origin,
                t_obs_window_s=t_obs_window_s,
                picker=picker,
                picker_kwargs=picker_kwargs,
                min_ratio_max=min_ratio_max,
                decimate_to_hz=decimate_to_hz,
                normalize=normalize,
                clip_percentile=clip_percentile,
                figsize=figsize,
                dpi=dpi,
                show=show,
                save=save,
                outpath=outpath,
                log=log,
            )
            results.append(res)

        return results

    def wilber_debug_plot_from_picks_df(
        self,
        picks_df,
        *,
        wave_index_df=None,
        mode="random",            # "random" | "picked" | "pick_ok"
        n=10,
        seed=123,
        origin_time_utc="from_xml",
        xml_path_for_origin=None,
        t_obs_window_s=(0.0, 500.0),
        picker="stalta",
        picker_kwargs=None,
        min_ratio_max=2.0,
        decimate_to_hz=50.0,
        normalize="none",
        clip_percentile=None,
        figsize=(12.0, 6.5),
        dpi=200,
        show=True,
        save=False,
        outdir=None,
        fname_prefix="wilber_pick_qc_from_picksdf",
        log=True,
    ):
        """
        Plot QC figures for files referenced in picks_df (channel-level table).

        mode:
          - "random": any file rows
          - "picked": only rows where picked==True
          - "pick_ok": only rows where pick_ok==True

        If picks_df doesn't include filepath, you can supply wave_index_df + station_id,
        but best is to use picks_df produced by wilber_build_arrivals_dataset().
        """
        import os
        import numpy as np
        import pandas as pd

        if "filepath" not in picks_df.columns:
            raise ValueError("picks_df must contain a 'filepath' column.")

        df = picks_df.copy()
        m = str(mode).lower().strip()
        if m == "picked" and "picked" in df.columns:
            df = df[df["picked"] == True].copy()
        if m == "pick_ok" and "pick_ok" in df.columns:
            df = df[df["pick_ok"] == True].copy()

        files = df["filepath"].dropna().astype(str).drop_duplicates().tolist()
        if len(files) == 0:
            raise ValueError(f"No files available for mode='{mode}'.")

        rng = np.random.default_rng(int(seed))
        chosen = list(rng.choice(files, size=int(min(n, len(files))), replace=False))

        results = []
        for k, fp in enumerate(chosen, start=1):
            outpath = None
            if save:
                if not outdir:
                    raise ValueError("save=True requires outdir.")
                outdirN = self._wilber__ensure_dir(self._wilber__as_path(outdir))
                outpath = os.path.join(outdirN, f"{fname_prefix}_{k:03d}.png")

            if log:
                self._wilber__log(f"[wilber-viz] ({k}/{len(chosen)}) file -> {os.path.basename(fp)}")

            res = self.wilber_debug_plot_waveform_pick(
                filepath=fp,
                origin_time_utc=origin_time_utc,
                xml_path_for_origin=xml_path_for_origin,
                t_obs_window_s=t_obs_window_s,
                picker=picker,
                picker_kwargs=picker_kwargs,
                min_ratio_max=min_ratio_max,
                decimate_to_hz=decimate_to_hz,
                normalize=normalize,
                clip_percentile=clip_percentile,
                figsize=figsize,
                dpi=dpi,
                show=show,
                save=save,
                outpath=outpath,
                log=log,
            )
            results.append(res)

        return results


    # -------------------------------------------------------------------------
    # Update override functions
    # -------------------------------------------------------------------------
    def _wilber__station_outlier_filter(
        self,
        arrivals_df,
        *,
        distance_km=None,
        use_only_picked=True,
        robust_iter=2,
        z_mad=3.5,
        hard_filter=True,
        hard_invalidate=None,   # NEW: alias (preferred by your pipeline)
        plot=False,
        plot_path=None,
        title="Arrival-time vs distance outlier filter",
        log=True,
    ):
        """
        Flags outliers based on robust linear fit of t_obs_s vs distance_km, using MAD of residuals.

        Behavior:
          - NEVER drops stations.
          - Always adds/updates columns:
              outlier_flag, outlier_score, residual_s, t_pred_fit_s
          - If invalidation is ON (hard_invalidate=True OR hard_filter=True):
              sets picked=False and clears t_obs_s/t_pick_utc for outliers.

        Returns (df_out, info_dict).
        """
        import numpy as np
        import pandas as pd

        # Resolve invalidation behavior consistently
        if hard_invalidate is None:
            hard_invalidate = bool(hard_filter)
        else:
            hard_invalidate = bool(hard_invalidate)

        df = arrivals_df.copy()

        # Ensure required columns exist
        if "outlier_flag" not in df.columns:
            df["outlier_flag"] = False
        if "outlier_score" not in df.columns:
            df["outlier_score"] = np.nan
        if "residual_s" not in df.columns:
            df["residual_s"] = np.nan
        if "t_pred_fit_s" not in df.columns:
            df["t_pred_fit_s"] = np.nan

        # Distance
        if distance_km is None:
            try:
                d_km = haversine_km(
                    df["lon"].astype(float).values,
                    df["lat"].astype(float).values,
                    float(self.event.epicenter_lon),
                    float(self.event.epicenter_lat),
                )
            except Exception:
                d_km = np.full(len(df), np.nan, float)
            df["epicentral_distance_km"] = d_km
        else:
            df["epicentral_distance_km"] = np.asarray(distance_km, float)

        # Base mask
        mask_base = (
            np.isfinite(df["epicentral_distance_km"].astype(float).values)
            & np.isfinite(df["t_obs_s"].astype(float).values)
        )
        if use_only_picked and "picked" in df.columns:
            mask_base = mask_base & df["picked"].astype(bool).values

        x = df.loc[mask_base, "epicentral_distance_km"].astype(float).values
        y = df.loc[mask_base, "t_obs_s"].astype(float).values

        info = {
            "used_points": int(len(x)),
            "z_mad": float(z_mad),
            "hard_invalidate": bool(hard_invalidate),
            "n_flagged": 0,
            "slope": np.nan,
            "intercept": np.nan,
            "mad_s": np.nan,
        }

        if len(x) < 8:
            if log:
                self._wilber__log(f"[wilber-outlier] Not enough points for outlier fit (N={len(x)}). Skipping.")
            return df, info

        # Robust iterative fit
        keep = np.ones(len(x), bool)
        b = a = np.nan
        for _ in range(int(max(1, robust_iter))):
            xx = x[keep]
            yy = y[keep]
            if len(xx) < 6:
                break

            b, a = np.polyfit(xx, yy, 1)  # yy ~ b*xx + a
            yhat = b * xx + a
            res = yy - yhat

            med = np.nanmedian(res)
            mad = np.nanmedian(np.abs(res - med))
            scale = 1.4826 * mad if mad > 0 else np.nanstd(res)
            if not np.isfinite(scale) or scale <= 0:
                break

            z = np.abs(res - med) / scale
            keep = z <= float(z_mad)

        # Final fit (using keep if available)
        xx_fit = x[keep] if np.any(keep) else x
        yy_fit = y[keep] if np.any(keep) else y
        b, a = np.polyfit(xx_fit, yy_fit, 1)

        yhat_all = b * x + a
        res_all = y - yhat_all

        med = np.nanmedian(res_all)
        mad = np.nanmedian(np.abs(res_all - med))
        scale = 1.4826 * mad if mad > 0 else np.nanstd(res_all)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        z_all = np.abs(res_all - med) / scale
        flagged = z_all > float(z_mad)

        # Write back to df
        idx_used = df.loc[mask_base].index
        df.loc[idx_used, "t_pred_fit_s"] = yhat_all
        df.loc[idx_used, "residual_s"] = res_all
        df.loc[idx_used, "outlier_score"] = z_all
        df.loc[idx_used, "outlier_flag"] = flagged

        info["n_flagged"] = int(np.sum(flagged))
        info["slope"] = float(b)
        info["intercept"] = float(a)
        info["mad_s"] = float(scale)

        # Invalidate (but never drop)
        if info["n_flagged"] > 0 and hard_invalidate:
            mflag = df["outlier_flag"].astype(bool).values
            if "picked" in df.columns:
                df.loc[mflag, "picked"] = False
            df.loc[mflag, "t_obs_s"] = np.nan
            if "t_pick_utc" in df.columns:
                df.loc[mflag, "t_pick_utc"] = pd.NaT

            if "notes" in df.columns:
                df.loc[mflag, "notes"] = df.loc[mflag, "notes"].astype(str).str.cat(["OUTLIER_FLAGGED"], sep="; ")
            else:
                df["notes"] = ""
                df.loc[mflag, "notes"] = "OUTLIER_FLAGGED"

        # Optional plot
        if plot:
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(9.5, 6.5), dpi=200)
                ax = plt.gca()
                ax.scatter(x, y, s=18, alpha=0.55, label="used picks")
                ax.scatter(x[flagged], y[flagged], s=30, alpha=0.95, marker="x", label="outliers")
                xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                ax.plot(xx, b * xx + a, linewidth=1.2, label="robust fit")
                ax.set_xlabel("Epicentral distance (km)")
                ax.set_ylabel("Observed arrival time t_obs (s)")
                ax.set_title(title)
                ax.legend(loc="best")
                ax.grid(True, alpha=0.25)
                if plot_path:
                    self._wilber__ensure_dir(__import__("os").path.dirname(self._wilber__as_path(plot_path)))
                    fig.savefig(self._wilber__as_path(plot_path), bbox_inches="tight")
                    plt.close(fig)
                else:
                    plt.show()
            except Exception as e:
                if log:
                    self._wilber__log(f"[wilber-outlier] Plot failed: {e}")

        if log:
            self._wilber__log(
                f"[wilber-outlier] fit: t = {info['slope']:.4f}*d + {info['intercept']:.2f} "
                f"(MAD scale~{info['mad_s']:.2f}s), flagged={info['n_flagged']} invalidate={info['hard_invalidate']}"
            )

        return df, info






    # =============================================================================
    # W) MODEL SKILL + BAYESIAN UPDATE TOOLS 
    # 
    #
    # 
    #   1) wilber_station_model_skill(...)
    #        - Computes residuals: r = t_obs - t_model at picked stations
    #        - Returns metrics dict + resid_df (ready for plotting/exports)
    #
    #   2) wilber_plot_station_residual_basemap(...)
    #        - Cartopy basemap: stations colored by residuals (or any value column)
    #        - Optional light "binned residual raster" behind points (NO kriging)
    #
    #   3) wilber_fit_speed_scale_lambda(...)
    #        - Sweeps / fits the global speed_scale_lambda using station data
    #        - Optionally recomputes T_map for each lambda (accurate) OR
    #          approximates by T_scaled = T_prior / lambda (fast)
    #
    #   4) wilber_update_T_map_bayes_kernel(...)
    #        - Bayesian-style spatial update (kernel residual correction):
    #              T_post(x) = T_prior(x) + R_hat(x)
    #        - R_hat uses Gaussian weights with a "prior strength" stabilizer
    #        - Returns T_post + meta (including correction field and diagnostics)
    #
    #   5) wilber_plot_T_map_difference(...)
    #        - Cartopy plot of delta field: (T_post - T_prior)
    #
    # Notes / philosophy:
    #   - NEVER breaks your working arrival/picking pipeline.
    #   - Treats T_prior as the prior mean field.
    #   - Produces posterior fields as new arrays (you decide what to use).
    #   - No kriging dependency; binned raster + Bayesian kernel update instead.
    # =============================================================================
    
    def _wilber__haversine_km(self, lon1, lat1, lon2, lat2):
        """Vectorized-ish haversine distance in km."""
        import numpy as np
        R = 6371.0
        lon1 = np.asarray(lon1, dtype=float); lat1 = np.asarray(lat1, dtype=float)
        lon2 = np.asarray(lon2, dtype=float); lat2 = np.asarray(lat2, dtype=float)
        dlon = np.deg2rad(lon2 - lon1)
        dlat = np.deg2rad(lat2 - lat1)
        a = np.sin(dlat / 2.0) ** 2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        return R * c
    
    
    def _wilber__grid_lonlat_from_snap(self, snap=None, *, lon_grid=None, lat_grid=None):
        """
        Robustly obtain lon/lat grids for a T_map.
        Accepts:
          - snap dict from wilber_compute_T_map_for_model() (preferred)
          - lon_grid/lat_grid explicitly
        Returns: (LON, LAT) as 2D arrays.
        """
        import numpy as np
    
        if lon_grid is not None and lat_grid is not None:
            return np.asarray(lon_grid, dtype=float), np.asarray(lat_grid, dtype=float)
    
        if snap is None:
            raise ValueError("Need snap (from wilber_compute_T_map_for_model) or lon_grid/lat_grid.")
    
        # Common patterns
        if isinstance(snap, dict):
            if ("lon_grid" in snap) and ("lat_grid" in snap):
                return np.asarray(snap["lon_grid"], dtype=float), np.asarray(snap["lat_grid"], dtype=float)
    
            # lon_vec/lat_vec -> mesh
            if ("lon_vec" in snap) and ("lat_vec" in snap):
                lonv = np.asarray(snap["lon_vec"], dtype=float)
                latv = np.asarray(snap["lat_vec"], dtype=float)
                LON, LAT = np.meshgrid(lonv, latv)
                return LON, LAT
    
            # sometimes "lon" / "lat"
            if ("lon" in snap) and ("lat" in snap):
                lonv = np.asarray(snap["lon"], dtype=float)
                latv = np.asarray(snap["lat"], dtype=float)
                if lonv.ndim == 1 and latv.ndim == 1:
                    LON, LAT = np.meshgrid(lonv, latv)
                    return LON, LAT
                return np.asarray(lonv, dtype=float), np.asarray(latv, dtype=float)
    
        raise ValueError("Could not infer lon/lat grids from snap. Provide lon_grid and lat_grid explicitly.")
    
    
    def wilber_station_model_skill(
        self,
        arrivals_df,
        T_map,
        *,
        # core selection
        use_only_picked=True,
        exclude_flagged=True,
        flagged_col="outlier_flag",
        picked_col="picked",
        # interpolation
        interp_method="nearest",  # "nearest" | "bilinear" (if your interp supports it)
        # columns (keep flexible)
        lon_col="lon",
        lat_col="lat",
        t_obs_col="t_obs_s",
        # optional distance diagnostics
        add_distance_km=True,
        event_lon=None,
        event_lat=None,
        # summary metrics options
        metric_mode="all",  # "all" | "finite_only"
        robust=True,        # adds MAD/median stats
        # exports
        return_df=True,
        log=True,
    ):
        """
        Compute station-level residuals between observed arrivals and a modeled travel-time field.
    
        Residual definition:
            residual_s = t_obs_s - t_model_s
          - residual > 0  => model is late (too slow locally)
          - residual < 0  => model is early (too fast locally)
    
        Returns:
          metrics dict, resid_df (if return_df=True else None)
        """
        import numpy as np
        import pandas as pd
    
        df = arrivals_df.copy()
    
        # subset
        if use_only_picked and picked_col in df.columns:
            df = df[df[picked_col].astype(bool) == True].copy()
    
        if exclude_flagged and flagged_col in df.columns:
            df = df[df[flagged_col].astype(bool) == False].copy()
    
        if df.empty:
            metrics = {
                "N": 0,
                "note": "No stations after filtering (picked/exclude_flagged).",
            }
            return metrics, (df if return_df else None)
    
        # interpolate modeled time at stations
        t_pred = self._wilber__interp_T_at(
            T_map,
            df[lon_col].to_numpy(float),
            df[lat_col].to_numpy(float),
            method=str(interp_method).lower().strip(),
        )
    
        df["t_model_s"] = t_pred
        df["residual_s"] = df[t_obs_col].to_numpy(float) - df["t_model_s"].to_numpy(float)
    
        # optional distance diagnostic (useful to spot distance-dependent bias)
        if add_distance_km:
            if event_lon is None or event_lat is None:
                # try to find from self if available
                try:
                    ev = getattr(self, "event", None)
                    if ev is not None:
                        event_lon = float(getattr(ev, "longitude", None) or getattr(ev, "lon", None))
                        event_lat = float(getattr(ev, "latitude", None) or getattr(ev, "lat", None))
                except Exception:
                    pass
            if event_lon is not None and event_lat is not None:
                df["dist_km"] = self._wilber__haversine_km(event_lon, event_lat, df[lon_col].to_numpy(float), df[lat_col].to_numpy(float))
            else:
                df["dist_km"] = np.nan
    
        # finite subset for metrics
        finite = df[np.isfinite(df["residual_s"].astype(float))].copy()
        if metric_mode == "finite_only":
            base = finite
        else:
            base = df
    
        N_all = int(len(df))
        N_fin = int(len(finite))
    
        metrics = {
            "N": N_all,
            "N_finite": N_fin,
        }
    
        if N_fin == 0:
            metrics["note"] = "No finite residuals."
            if log:
                self._wilber__log("[wilber-skill] No finite residuals to score.")
            return metrics, (df if return_df else None)
    
        r = finite["residual_s"].to_numpy(float)
    
        # standard metrics
        metrics["bias_mean_s"] = float(np.nanmean(r))
        metrics["mae_s"] = float(np.nanmean(np.abs(r)))
        metrics["rmse_s"] = float(np.sqrt(np.nanmean(r ** 2)))
        metrics["median_s"] = float(np.nanmedian(r))
        metrics["p10_s"] = float(np.nanpercentile(r, 10))
        metrics["p90_s"] = float(np.nanpercentile(r, 90))
    
        # robust extras
        if robust:
            med = float(np.nanmedian(r))
            mad = float(np.nanmedian(np.abs(r - med)))
            metrics["mad_s"] = mad
            # "robust sigma" ~ 1.4826*MAD (normal-equivalent)
            metrics["robust_sigma_s"] = float(1.4826 * mad)
    
        if log:
            self._wilber__log(
                f"[wilber-skill] N={N_all} (finite={N_fin}) "
                f"bias={metrics['bias_mean_s']:+.2f}s mae={metrics['mae_s']:.2f}s rmse={metrics['rmse_s']:.2f}s "
                f"median={metrics['median_s']:+.2f}s p10/p90=({metrics['p10_s']:+.2f},{metrics['p90_s']:+.2f})"
            )
    
        return metrics, (df if return_df else None)

    
    # override 
    def wilber_plot_station_residual_basemap(
        self,
        df,
        *,
        # what to plot
        value_col="residual_s",
        value_label="Residual (t_obs - t_model) [s]",
        # station selection
        use_only_picked=True,
        picked_col="picked",
        exclude_flagged=False,
        flagged_col="outlier_flag",
        # station point styling
        marker="o",
        size=40,
        edgecolor="k",
        linewidth=0.6,
        alpha=0.95,
        # labels
        show_labels=False,
        label_col=None,             # e.g. "residual_s" or custom
        label_fmt="{:+.0f}",        # format for numeric label_col
        label_size=8,
        label_alpha=0.85,
        label_offset=(0.03, 0.03),  # degrees offset in lon/lat
        # optional "binned raster" behind points (NO kriging)
        add_binned_raster=False,
        bin_deg=0.25,
        bin_stat="median",          # "median" | "mean"
        raster_alpha=0.30,
        raster_vmin=None,
        raster_vmax=None,
        # color scaling
        vmin=None,
        vmax=None,
        cmap="coolwarm",            # diverging default (residuals)
        show_colorbar=True,
        colorbar_shrink=0.86,
        colorbar_pad=0.02,
        # cartopy basemap styling (match your style)
        extent=None,                # [lonmin, lonmax, latmin, latmax]
        extent_pad_deg=0.7,
        add_ocean=True,
        add_land=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=False,
        # figure
        title=None,
        figsize=(11.0, 8.5),
        dpi=250,
        # output
        show=True,
        save=False,
        outpath=None,
        log=True,
    ):
        """
        Cartopy basemap of station values (e.g. residuals) with optional binned raster overlay.
    
        Tip:
          - For residuals: use diverging colormap and symmetric vmin/vmax.
          - add_binned_raster=True helps you SEE local structure without kriging.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
    
        # Cartopy imports
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    
        d = df.copy()
    
        if use_only_picked and picked_col in d.columns:
            d = d[d[picked_col].astype(bool) == True].copy()
    
        if exclude_flagged and flagged_col in d.columns:
            d = d[d[flagged_col].astype(bool) == False].copy()
    
        # must have lon/lat
        if "lon" not in d.columns or "lat" not in d.columns:
            raise ValueError("df must contain 'lon' and 'lat' columns (station coordinates).")
    
        # values
        vals = pd.to_numeric(d[value_col], errors="coerce").to_numpy(float)
        ok = np.isfinite(vals)
        d = d.loc[ok].copy()
        vals = vals[ok]
    
        if len(d) == 0:
            raise ValueError("No finite station values to plot.")
    
        lons = d["lon"].to_numpy(float)
        lats = d["lat"].to_numpy(float)
    
        # extent
        if extent is None:
            lonmin = float(np.nanmin(lons) - float(extent_pad_deg))
            lonmax = float(np.nanmax(lons) + float(extent_pad_deg))
            latmin = float(np.nanmin(lats) - float(extent_pad_deg))
            latmax = float(np.nanmax(lats) + float(extent_pad_deg))
            extent = [lonmin, lonmax, latmin, latmax]
    
        # auto symmetric scaling for residual-like values
        if (vmin is None) and (vmax is None):
            m = float(np.nanmax(np.abs(vals)))
            vmin, vmax = -m, +m
    
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
        if add_ocean:
            ax.add_feature(cfeature.OCEAN, zorder=0)
        if add_land:
            ax.add_feature(cfeature.LAND, zorder=0)
        if add_borders:
            ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=1)
        if add_coastlines:
            ax.coastlines(resolution="110m", linewidth=0.6, zorder=1)
    
        if add_gridlines:
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
    
        # optional binned raster overlay
        if add_binned_raster:
            # build bin edges
            b = float(bin_deg)
            lon_edges = np.arange(extent[0], extent[1] + b, b)
            lat_edges = np.arange(extent[2], extent[3] + b, b)
    
            # assign bins
            lon_i = np.digitize(lons, lon_edges) - 1
            lat_i = np.digitize(lats, lat_edges) - 1
    
            # grid to fill
            grid = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan, dtype=float)
    
            # aggregate by cell
            for ii in range(len(lons)):
                i = lat_i[ii]; j = lon_i[ii]
                if i < 0 or j < 0 or i >= grid.shape[0] or j >= grid.shape[1]:
                    continue
                if not np.isfinite(vals[ii]):
                    continue
                # accumulate in a list per cell (simple approach)
                if np.isnan(grid[i, j]):
                    grid[i, j] = vals[ii]
                else:
                    # store sum in-place is not median-ready; do a light workaround:
                    # We'll do mean in-place; for median we fall back to a second pass.
                    pass
    
            if str(bin_stat).lower() == "mean":
                # mean per cell using sums/counts
                sums = np.zeros_like(grid)
                cnts = np.zeros_like(grid)
                for ii in range(len(lons)):
                    i = lat_i[ii]; j = lon_i[ii]
                    if i < 0 or j < 0 or i >= grid.shape[0] or j >= grid.shape[1]:
                        continue
                    if not np.isfinite(vals[ii]):
                        continue
                    sums[i, j] += vals[ii]
                    cnts[i, j] += 1.0
                with np.errstate(invalid="ignore", divide="ignore"):
                    grid = np.where(cnts > 0, sums / cnts, np.nan)
            else:
                # median per cell: collect lists (still fine for typical station counts)
                cells = {}
                for ii in range(len(lons)):
                    i = lat_i[ii]; j = lon_i[ii]
                    if i < 0 or j < 0 or i >= (len(lat_edges)-1) or j >= (len(lon_edges)-1):
                        continue
                    if not np.isfinite(vals[ii]):
                        continue
                    cells.setdefault((i, j), []).append(float(vals[ii]))
                for (i, j), vv in cells.items():
                    grid[i, j] = float(np.nanmedian(np.asarray(vv, dtype=float)))
    
            # plot raster
            LonE, LatE = np.meshgrid(lon_edges, lat_edges)
            rvmin = vmin if raster_vmin is None else raster_vmin
            rvmax = vmax if raster_vmax is None else raster_vmax
            ax.pcolormesh(LonE, LatE, grid, cmap=cmap, vmin=rvmin, vmax=rvmax, alpha=float(raster_alpha), zorder=2)
    
        # station scatter
        sc = ax.scatter(
            lons, lats,
            c=vals,
            s=float(size),
            marker=marker,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            edgecolors=edgecolor,
            linewidths=float(linewidth),
            alpha=float(alpha),
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
    
        # labels (optional)
        if show_labels:
            if label_col is None:
                label_col = value_col
            labv = d[label_col].to_numpy()
            dx, dy = float(label_offset[0]), float(label_offset[1])
            for lon, lat, v in zip(lons, lats, labv):
                try:
                    if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
                        txt = str(label_fmt).format(float(v))
                    else:
                        txt = str(v)
                except Exception:
                    txt = str(v)
                ax.text(
                    lon + dx, lat + dy, txt,
                    fontsize=float(label_size),
                    alpha=float(label_alpha),
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )
    
        if show_colorbar:
            cb = plt.colorbar(sc, ax=ax, shrink=float(colorbar_shrink), pad=float(colorbar_pad))
            cb.set_label(str(value_label))
    
        if title:
            ax.set_title(str(title))
    
        if save:
            if not outpath:
                raise ValueError("save=True requires outpath.")
            fig.savefig(outpath, bbox_inches="tight")
            if log:
                self._wilber__log(f"[wilber-resid-map] wrote: {outpath}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return {"fig": fig, "ax": ax, "N": int(len(d)), "extent": extent}
    


    # override 
    def wilber_plot_station_residual_basemap(
        self,
        df,
        *,
        # what to plot
        value_col="residual_s",
        value_label="Residual (t_obs - t_model) [s]",
        # station selection
        use_only_picked=True,
        picked_col="picked",
        exclude_flagged=False,
        flagged_col="outlier_flag",
        # station point styling
        marker="o",
        size=40,
        edgecolor="k",
        linewidth=0.6,
        alpha=0.95,
        # labels
        show_labels=False,
        label_col=None,             # e.g. "residual_s" or custom
        label_fmt="{:+.0f}",        # format for numeric label_col
        label_size=8,
        label_alpha=0.85,
        label_offset=(0.03, 0.03),  # degrees offset in lon/lat
        # optional "binned raster" behind points (NO kriging)
        add_binned_raster=False,
        bin_deg=0.25,
        bin_stat="median",          # "median" | "mean"
        raster_alpha=0.30,
        raster_vmin=None,
        raster_vmax=None,
        # color scaling
        vmin=None,
        vmax=None,
        cmap="coolwarm",            # diverging default (residuals)
        show_colorbar=True,
        colorbar_shrink=0.86,
        colorbar_pad=0.02,
        # cartopy basemap styling
        extent=None,                # [lonmin, lonmax, latmin, latmax] (manual override)
        extent_mode="stations",     # "stations" | "shakemap" | "grid" | "manual"
        extent_pad_deg=0.7,
        basemap="simple",           # "simple" (current behavior) | "usgs" (calls _usgs_basemap)
        basemap_kwargs=None,        # passed to _usgs_basemap when basemap="usgs"
        add_ocean=True,
        add_land=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=False,
        # zorder modularity
        zorder=None,                # dict override, e.g. {"stations": 10, "legend": 2000}
        # optional legend support (only used if you pass handles/labels)
        legend=False,
        legend_kwargs=None,
        legend_handles=None,
        legend_labels=None,
        # figure
        title=None,
        figsize=(11.0, 8.5),
        dpi=250,
        # output
        show=True,
        save=False,
        outpath=None,
        log=True,
    ):
        """
        Cartopy basemap of station values (e.g. residuals) with optional binned raster overlay.
    
        Modularity adds:
          - extent_mode: "stations" (default/current), "shakemap", "grid", "manual" (or pass extent=...)
          - zorder dict to control layer ordering (legend forced on top)
          - optional legend (only if legend=True and handles/labels provided)
          - basemap="usgs" optionally calls your module-level _usgs_basemap(ax, extent, ...)
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
    
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    
        # -------------------------
        # zorder defaults (overrideable)
        # -------------------------
        zdef = {
            "basemap": 0,
            "land_ocean": 0,
            "borders_coast": 1,
            "gridlines": 2,
            "raster": 2,
            "stations": 3,
            "labels": 4,
            "legend": 2000,    # keep above everything (incl. _usgs_basemap gridlines zorder=999)
            "colorbar": 2100,
        }
        if isinstance(zorder, dict):
            zdef.update(zorder)
    
        d = df.copy()
    
        if use_only_picked and picked_col in d.columns:
            d = d[d[picked_col].astype(bool) == True].copy()
    
        if exclude_flagged and flagged_col in d.columns:
            d = d[d[flagged_col].astype(bool) == False].copy()
    
        if "lon" not in d.columns or "lat" not in d.columns:
            raise ValueError("df must contain 'lon' and 'lat' columns (station coordinates).")
    
        vals = pd.to_numeric(d[value_col], errors="coerce").to_numpy(float)
        ok = np.isfinite(vals)
        d = d.loc[ok].copy()
        vals = vals[ok]
    
        if len(d) == 0:
            raise ValueError("No finite station values to plot.")
    
        lons = d["lon"].to_numpy(float)
        lats = d["lat"].to_numpy(float)
    
        # -------------------------
        # extent resolution
        # -------------------------
        if extent is not None:
            extent_mode = "manual"
    
        def _extent_from_stations():
            return [
                float(np.nanmin(lons) - float(extent_pad_deg)),
                float(np.nanmax(lons) + float(extent_pad_deg)),
                float(np.nanmin(lats) - float(extent_pad_deg)),
                float(np.nanmax(lats) + float(extent_pad_deg)),
            ]
    
        def _extent_from_shakemap_grid():
            # try common attribute names
            lonG = getattr(self, "lon_grid", None)
            latG = getattr(self, "lat_grid", None)
            if lonG is None or latG is None:
                # fallback to cached wilber grids if present
                lonG = getattr(self, "_wilber_last_lon_grid", None)
                latG = getattr(self, "_wilber_last_lat_grid", None)
            if lonG is None or latG is None:
                raise ValueError("extent_mode='shakemap' requested but no lon/lat grid found on self.")
            lonG = np.asarray(lonG, dtype=float)
            latG = np.asarray(latG, dtype=float)
            return [
                float(np.nanmin(lonG) - float(extent_pad_deg)),
                float(np.nanmax(lonG) + float(extent_pad_deg)),
                float(np.nanmin(latG) - float(extent_pad_deg)),
                float(np.nanmax(latG) + float(extent_pad_deg)),
            ]
    
        def _extent_from_grid():
            # for this plot, "grid" means same as stations unless you later expand it
            return _extent_from_stations()
    
        if extent_mode == "manual":
            if extent is None:
                raise ValueError("extent_mode='manual' requires extent=[lonmin, lonmax, latmin, latmax].")
            extent_use = list(map(float, extent))
        elif extent_mode == "shakemap":
            extent_use = _extent_from_shakemap_grid()
        elif extent_mode == "grid":
            extent_use = _extent_from_grid()
        else:
            # default/current
            extent_use = _extent_from_stations()
    
        # -------------------------
        # auto symmetric scaling for residual-like values
        # -------------------------
        if (vmin is None) and (vmax is None):
            m = float(np.nanmax(np.abs(vals)))
            vmin, vmax = -m, +m
    
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent_use, crs=ccrs.PlateCarree())
    
        # -------------------------
        # basemap
        # -------------------------
        if basemap_kwargs is None:
            basemap_kwargs = {}
        if str(basemap).lower() == "usgs":
            # module-level helper in this file
            try:
                _usgs_basemap(ax, extent_use, **basemap_kwargs)
            except TypeError:
                # if helper signature differs in your local edits
                _usgs_basemap(ax, extent_use)
        else:
            if add_ocean:
                ax.add_feature(cfeature.OCEAN, zorder=zdef["land_ocean"])
            if add_land:
                ax.add_feature(cfeature.LAND, zorder=zdef["land_ocean"])
            if add_borders:
                ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=zdef["borders_coast"])
            if add_coastlines:
                ax.coastlines(resolution="110m", linewidth=0.6, zorder=zdef["borders_coast"])
    
            if add_gridlines:
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle="--")
                gl.top_labels = False
                gl.right_labels = False
                # cartopy gridliner doesn't expose a clean zorder setter; keep legend well above.
    
        # -------------------------
        # optional binned raster overlay
        # -------------------------
        if add_binned_raster:
            b = float(bin_deg)
            lon_edges = np.arange(extent_use[0], extent_use[1] + b, b)
            lat_edges = np.arange(extent_use[2], extent_use[3] + b, b)
    
            lon_i = np.digitize(lons, lon_edges) - 1
            lat_i = np.digitize(lats, lat_edges) - 1
    
            grid = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan, dtype=float)
    
            if str(bin_stat).lower() == "mean":
                sums = np.zeros_like(grid)
                cnts = np.zeros_like(grid)
                for ii in range(len(lons)):
                    i = lat_i[ii]; j = lon_i[ii]
                    if i < 0 or j < 0 or i >= grid.shape[0] or j >= grid.shape[1]:
                        continue
                    if not np.isfinite(vals[ii]):
                        continue
                    sums[i, j] += vals[ii]
                    cnts[i, j] += 1.0
                with np.errstate(invalid="ignore", divide="ignore"):
                    grid = np.where(cnts > 0, sums / cnts, np.nan)
            else:
                cells = {}
                for ii in range(len(lons)):
                    i = lat_i[ii]; j = lon_i[ii]
                    if i < 0 or j < 0 or i >= (len(lat_edges)-1) or j >= (len(lon_edges)-1):
                        continue
                    if not np.isfinite(vals[ii]):
                        continue
                    cells.setdefault((i, j), []).append(float(vals[ii]))
                for (i, j), vv in cells.items():
                    grid[i, j] = float(np.nanmedian(np.asarray(vv, dtype=float)))
    
            LonE, LatE = np.meshgrid(lon_edges, lat_edges)
            rvmin = vmin if raster_vmin is None else raster_vmin
            rvmax = vmax if raster_vmax is None else raster_vmax
            ax.pcolormesh(
                LonE, LatE, grid,
                cmap=cmap, vmin=rvmin, vmax=rvmax,
                alpha=float(raster_alpha),
                zorder=zdef["raster"],
                transform=ccrs.PlateCarree(),
            )
    
        # -------------------------
        # station scatter
        # -------------------------
        sc = ax.scatter(
            lons, lats,
            c=vals,
            s=float(size),
            marker=marker,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            edgecolors=edgecolor,
            linewidths=float(linewidth),
            alpha=float(alpha),
            transform=ccrs.PlateCarree(),
            zorder=zdef["stations"],
        )
    
        # -------------------------
        # labels (optional)
        # -------------------------
        if show_labels:
            if label_col is None:
                label_col = value_col
            labv = d[label_col].to_numpy()
            dx, dy = float(label_offset[0]), float(label_offset[1])
            for lon, lat, v in zip(lons, lats, labv):
                try:
                    if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
                        txt = str(label_fmt).format(float(v))
                    else:
                        txt = str(v)
                except Exception:
                    txt = str(v)
                ax.text(
                    lon + dx, lat + dy, txt,
                    fontsize=float(label_size),
                    alpha=float(label_alpha),
                    transform=ccrs.PlateCarree(),
                    zorder=zdef["labels"],
                )
    
        # -------------------------
        # legend (optional, fully controlled + always on top)
        # -------------------------
        if legend:
            lk = dict(legend_kwargs or {})
            # If user passes handles/labels, use them; otherwise try default legend() (won't show unless artists have labels)
            if legend_handles is not None and legend_labels is not None:
                leg = ax.legend(legend_handles, legend_labels, **lk)
            else:
                leg = ax.legend(**lk)
            if leg is not None:
                leg.set_zorder(zdef["legend"])
    
        # -------------------------
        # colorbar
        # -------------------------
        if show_colorbar:
            cb = plt.colorbar(sc, ax=ax, shrink=float(colorbar_shrink), pad=float(colorbar_pad))
            cb.set_label(str(value_label))
            # ensure above plot artists; colorbar is separate axes, but keep semantic zorder in dict if needed.
    
        if title:
            ax.set_title(str(title))
    
        if save:
            if not outpath:
                raise ValueError("save=True requires outpath.")
            fig.savefig(outpath, bbox_inches="tight")
            if log:
                self._wilber__log(f"[wilber-resid-map] wrote: {outpath}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return {"fig": fig, "ax": ax, "N": int(len(d)), "extent": extent_use}
    
    

    def wilber_plot_station_residual_basemap(
        self,
        df,
        *,
        # what to plot
        value_col="residual_s",
        value_label="Residual (t_obs - t_model) [s]",
        # station selection
        use_only_picked=True,
        picked_col="picked",
        exclude_flagged=False,
        flagged_col="outlier_flag",
        # station point styling
        marker="o",
        size=40,
        edgecolor="k",
        linewidth=0.6,
        alpha=0.95,
        # labels
        show_labels=False,
        label_col=None,             # e.g. "residual_s" or custom
        label_fmt="{:+.0f}",        # format for numeric label_col
        label_size=8,
        label_alpha=0.85,
        label_offset=(0.03, 0.03),  # degrees offset in lon/lat
        # optional "binned raster" behind points (NO kriging)
        add_binned_raster=False,
        bin_deg=0.25,
        bin_stat="median",          # "median" | "mean"
        raster_alpha=0.30,
        raster_vmin=None,
        raster_vmax=None,
        # color scaling
        vmin=None,
        vmax=None,
        cmap="coolwarm",            # diverging default (residuals)
        show_colorbar=True,
        colorbar_shrink=0.86,
        colorbar_pad=0.02,
        # cartopy basemap styling
        extent=None,                # [lonmin, lonmax, latmin, latmax] (manual override)
        extent_mode="stations",     # "stations" | "shakemap" | "grid" | "manual"
        extent_pad_deg=0.7,
        basemap="simple",           # "simple" (current behavior) | "usgs" (calls _usgs_basemap)
        basemap_kwargs=None,        # passed to _usgs_basemap when basemap="usgs"
        add_ocean=True,
        add_land=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=False,
    
        # -------------------------
        # NEW: clip + ShakeMap background + wave-time contours (defaults ON)
        # -------------------------
        clip_stations_to_extent=True,
    
        show_shakemap=True,
        base_imt="MMI",
        base_alpha=0.35,
        base_cmap=None,
        show_shakemap_colorbar=False,     # default off to avoid double colorbars
        shakemap_colorbar_shrink=0.80,
        shakemap_colorbar_pad=0.02,
    
        show_time_contours=True,
        # Provide any/all of these; function will plot whichever are available + requested
        T_prior0=None,        # "first model wave with no scaling"
        T_post_lambda=None,   # "alpha update after rescaling" (your naming)
        T_model_update=None,  # "bayes updated model"
        contour_which=("post_lambda",),   # ("prior0",) | ("post_lambda",) | ("model_update",) | ("prior0","post_lambda","model_update")
        contour_levels=None,
        contour_label_fmt="{:.0f}s",
        clabel=False,
        clabel_kwargs=None,
        contour_kwargs_prior0=None,
        contour_kwargs_post_lambda=None,
        contour_kwargs_model_update=None,
    
        # zorder modularity
        zorder=None,                # dict override, e.g. {"stations": 10, "legend": 2000}
        # optional legend support (only used if you pass handles/labels)
        legend=False,
        legend_kwargs=None,
        legend_handles=None,
        legend_labels=None,
        # figure
        title=None,
        figsize=(11.0, 8.5),
        dpi=250,
        # output
        show=True,
        save=False,
        outpath=None,
        log=True,
    ):
        """
        Cartopy basemap of station values (e.g. residuals) with optional binned raster overlay.
    
        Added (non-breaking):
          - clip_stations_to_extent: prevents plotting stations outside the chosen extent.
          - show_shakemap: overlay ShakeMap raster (base_imt) under residual points.
          - show_time_contours: overlay wave/travel-time contour lines for any of:
                T_prior0, T_post_lambda, T_model_update
            Choose which ones via contour_which. Default: ("post_lambda",)
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
    
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    
        # -------------------------
        # zorder defaults (overrideable)
        # -------------------------
        zdef = {
            "basemap": 0,
            "land_ocean": 0,
            "borders_coast": 1,
            "gridlines": 2,
            "shakemap": 2,        # below raster/stations by default
            "raster": 2,
            "contours": 3,
            "stations": 4,
            "labels": 5,
            "legend": 2000,       # keep above everything
            "colorbar": 2100,
        }
        if isinstance(zorder, dict):
            zdef.update(zorder)
    
        d = df.copy()
    
        if use_only_picked and picked_col in d.columns:
            d = d[d[picked_col].astype(bool) == True].copy()
    
        if exclude_flagged and flagged_col in d.columns:
            d = d[d[flagged_col].astype(bool) == False].copy()
    
        if "lon" not in d.columns or "lat" not in d.columns:
            raise ValueError("df must contain 'lon' and 'lat' columns (station coordinates).")
    
        vals = pd.to_numeric(d[value_col], errors="coerce").to_numpy(float)
        ok = np.isfinite(vals)
        d = d.loc[ok].copy()
        vals = vals[ok]
    
        if len(d) == 0:
            raise ValueError("No finite station values to plot.")
    
        lons = d["lon"].to_numpy(float)
        lats = d["lat"].to_numpy(float)
    
        # -------------------------
        # extent resolution
        # -------------------------
        if extent is not None:
            extent_mode = "manual"
    
        def _extent_from_stations():
            return [
                float(np.nanmin(lons) - float(extent_pad_deg)),
                float(np.nanmax(lons) + float(extent_pad_deg)),
                float(np.nanmin(lats) - float(extent_pad_deg)),
                float(np.nanmax(lats) + float(extent_pad_deg)),
            ]
    
        def _extent_from_shakemap_grid():
            lonG = getattr(self, "lon_grid", None)
            latG = getattr(self, "lat_grid", None)
            if lonG is None or latG is None:
                lonG = getattr(self, "_wilber_last_lon_grid", None)
                latG = getattr(self, "_wilber_last_lat_grid", None)
            if lonG is None or latG is None:
                raise ValueError("extent_mode='shakemap' requested but no lon/lat grid found on self.")
            lonG = np.asarray(lonG, dtype=float)
            latG = np.asarray(latG, dtype=float)
            return [
                float(np.nanmin(lonG) - float(extent_pad_deg)),
                float(np.nanmax(lonG) + float(extent_pad_deg)),
                float(np.nanmin(latG) - float(extent_pad_deg)),
                float(np.nanmax(latG) + float(extent_pad_deg)),
            ]
    
        def _extent_from_grid():
            # for this plot, keep legacy behavior (same as stations unless you later expand it)
            return _extent_from_stations()
    
        if extent_mode == "manual":
            if extent is None:
                raise ValueError("extent_mode='manual' requires extent=[lonmin, lonmax, latmin, latmax].")
            extent_use = list(map(float, extent))
        elif extent_mode == "shakemap":
            extent_use = _extent_from_shakemap_grid()
        elif extent_mode == "grid":
            extent_use = _extent_from_grid()
        else:
            extent_use = _extent_from_stations()
    
        # -------------------------
        # clip stations to extent (prevents points outside map)
        # -------------------------
        if bool(clip_stations_to_extent):
            eps = 1e-9
            in_ext = (
                (d["lon"].astype(float) >= extent_use[0] - eps) &
                (d["lon"].astype(float) <= extent_use[1] + eps) &
                (d["lat"].astype(float) >= extent_use[2] - eps) &
                (d["lat"].astype(float) <= extent_use[3] + eps)
            )
            d = d.loc[in_ext].copy()
            vals = pd.to_numeric(d[value_col], errors="coerce").to_numpy(float)
            ok = np.isfinite(vals)
            d = d.loc[ok].copy()
            vals = vals[ok]
            if len(d) == 0:
                raise ValueError("After clipping to extent, no stations remain to plot.")
            lons = d["lon"].to_numpy(float)
            lats = d["lat"].to_numpy(float)
    
        # -------------------------
        # auto symmetric scaling for residual-like values
        # -------------------------
        if (vmin is None) and (vmax is None):
            m = float(np.nanmax(np.abs(vals)))
            vmin, vmax = -m, +m
    
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent_use, crs=ccrs.PlateCarree())
    
        # -------------------------
        # basemap
        # -------------------------
        if basemap_kwargs is None:
            basemap_kwargs = {}
        if str(basemap).lower() == "usgs":
            try:
                _usgs_basemap(ax, extent_use, **basemap_kwargs)
            except TypeError:
                _usgs_basemap(ax, extent_use)
        else:
            if add_ocean:
                ax.add_feature(cfeature.OCEAN, zorder=zdef["land_ocean"])
            if add_land:
                ax.add_feature(cfeature.LAND, zorder=zdef["land_ocean"])
            if add_borders:
                ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=zdef["borders_coast"])
            if add_coastlines:
                ax.coastlines(resolution="110m", linewidth=0.6, zorder=zdef["borders_coast"])
    
            if add_gridlines:
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle="--")
                gl.top_labels = False
                gl.right_labels = False
    
        # -------------------------
        # ShakeMap raster background (optional)
        # -------------------------
        shakemap_im = None
        if bool(show_shakemap):
            # require lon_grid/lat_grid + ShakeMap field helper
            try:
                self._wilber__require_grids()
                Z = self._wilber__get_shakemap_field(base_imt)
            except Exception:
                Z = None
    
            if Z is not None:
                mmi_mode = str(base_imt).upper().strip() == "MMI"
                if mmi_mode and (base_cmap is None):
                    try:
                        cmap_mmi, norm_mmi, ticks_mmi, label_mmi = _usgs_mmi_cmap_norm()
                        shakemap_im = ax.pcolormesh(
                            self.lon_grid, self.lat_grid, Z,
                            transform=ccrs.PlateCarree(),
                            shading="auto",
                            alpha=float(base_alpha),
                            zorder=zdef["shakemap"],
                            cmap=cmap_mmi,
                            norm=norm_mmi,
                        )
                    except Exception:
                        shakemap_im = ax.pcolormesh(
                            self.lon_grid, self.lat_grid, Z,
                            transform=ccrs.PlateCarree(),
                            shading="auto",
                            alpha=float(base_alpha),
                            zorder=zdef["shakemap"],
                        )
                else:
                    shakemap_im = ax.pcolormesh(
                        self.lon_grid, self.lat_grid, Z,
                        transform=ccrs.PlateCarree(),
                        shading="auto",
                        alpha=float(base_alpha),
                        zorder=zdef["shakemap"],
                        cmap=base_cmap,
                    )
    
                if bool(show_shakemap_colorbar) and (shakemap_im is not None):
                    cb2 = plt.colorbar(
                        shakemap_im, ax=ax,
                        shrink=float(shakemap_colorbar_shrink),
                        pad=float(shakemap_colorbar_pad),
                    )
                    if str(base_imt).upper().strip() == "MMI":
                        try:
                            _, _, ticks_mmi, label_mmi = _usgs_mmi_cmap_norm()
                            cb2.set_ticks(ticks_mmi)
                            cb2.set_label(label_mmi)
                        except Exception:
                            cb2.set_label("MMI")
                    else:
                        cb2.set_label(str(base_imt))
    
        # -------------------------
        # optional binned raster overlay (behind points)
        # -------------------------
        if add_binned_raster:
            b = float(bin_deg)
            lon_edges = np.arange(extent_use[0], extent_use[1] + b, b)
            lat_edges = np.arange(extent_use[2], extent_use[3] + b, b)
    
            lon_i = np.digitize(lons, lon_edges) - 1
            lat_i = np.digitize(lats, lat_edges) - 1
    
            grid = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan, dtype=float)
    
            if str(bin_stat).lower() == "mean":
                sums = np.zeros_like(grid)
                cnts = np.zeros_like(grid)
                for ii in range(len(lons)):
                    i = lat_i[ii]; j = lon_i[ii]
                    if i < 0 or j < 0 or i >= grid.shape[0] or j >= grid.shape[1]:
                        continue
                    if not np.isfinite(vals[ii]):
                        continue
                    sums[i, j] += vals[ii]
                    cnts[i, j] += 1.0
                with np.errstate(invalid="ignore", divide="ignore"):
                    grid = np.where(cnts > 0, sums / cnts, np.nan)
            else:
                cells = {}
                for ii in range(len(lons)):
                    i = lat_i[ii]; j = lon_i[ii]
                    if i < 0 or j < 0 or i >= (len(lat_edges)-1) or j >= (len(lon_edges)-1):
                        continue
                    if not np.isfinite(vals[ii]):
                        continue
                    cells.setdefault((i, j), []).append(float(vals[ii]))
                for (i, j), vv in cells.items():
                    grid[i, j] = float(np.nanmedian(np.asarray(vv, dtype=float)))
    
            LonE, LatE = np.meshgrid(lon_edges, lat_edges)
            rvmin = vmin if raster_vmin is None else raster_vmin
            rvmax = vmax if raster_vmax is None else raster_vmax
            ax.pcolormesh(
                LonE, LatE, grid,
                cmap=cmap, vmin=rvmin, vmax=rvmax,
                alpha=float(raster_alpha),
                zorder=zdef["raster"],
                transform=ccrs.PlateCarree(),
            )
    
        # -------------------------
        # wave/time contours (optional)
        # -------------------------
        if bool(show_time_contours):
            # only attempt if we have a lon/lat grid
            try:
                self._wilber__require_grids()
                LON = np.asarray(self.lon_grid, float)
                LAT = np.asarray(self.lat_grid, float)
            except Exception:
                LON = None
                LAT = None
    
            if (LON is not None) and (LAT is not None):
                # decide which stages exist
                stage_map = {
                    "prior0": T_prior0,
                    "post_lambda": T_post_lambda,
                    "model_update": T_model_update,
                }
    
                which = contour_which
                if isinstance(which, str):
                    which = (which,)
                which = tuple([str(w).strip().lower() for w in which])
    
                # if user left default ("post_lambda",) but didn't pass it, fall back to first available
                if all(stage_map.get(w) is None for w in which):
                    for cand in ("post_lambda", "model_update", "prior0"):
                        if stage_map.get(cand) is not None:
                            which = (cand,)
                            break
    
                # default styles (overrideable)
                kw0 = {"colors": "k", "linewidths": 0.9, "alpha": 0.85, "linestyles": "-"}
                kw1 = {"colors": "k", "linewidths": 1.0, "alpha": 0.95, "linestyles": "--"}
                kw2 = {"colors": "k", "linewidths": 1.1, "alpha": 0.95, "linestyles": "-."}
                if isinstance(contour_kwargs_prior0, dict):
                    kw0.update(contour_kwargs_prior0)
                if isinstance(contour_kwargs_post_lambda, dict):
                    kw1.update(contour_kwargs_post_lambda)
                if isinstance(contour_kwargs_model_update, dict):
                    kw2.update(contour_kwargs_model_update)
    
                stage_style = {
                    "prior0": kw0,
                    "post_lambda": kw1,
                    "model_update": kw2,
                }
    
                for st in which:
                    Tst = stage_map.get(st)
                    if Tst is None:
                        continue
                    Tst = np.asarray(Tst, float)
    
                    ck = dict(stage_style.get(st, {}))
                    ck.update({
                        "transform": ccrs.PlateCarree(),
                        "zorder": zdef["contours"],
                    })
                    if contour_levels is not None:
                        ck["levels"] = contour_levels
    
                    cs = ax.contour(LON, LAT, Tst, **ck)
    
                    if bool(clabel):
                        lk = {"fontsize": 8, "inline": True, "fmt": (lambda v: contour_label_fmt.format(v))}
                        if isinstance(clabel_kwargs, dict):
                            lk.update(clabel_kwargs)
                        ax.clabel(cs, **lk)
    
        # -------------------------
        # station scatter (residuals)
        # -------------------------
        sc = ax.scatter(
            lons, lats,
            c=vals,
            s=float(size),
            marker=marker,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            edgecolors=edgecolor,
            linewidths=float(linewidth),
            alpha=float(alpha),
            transform=ccrs.PlateCarree(),
            zorder=zdef["stations"],
        )
    
        # -------------------------
        # labels (optional)
        # -------------------------
        if show_labels:
            if label_col is None:
                label_col = value_col
            labv = d[label_col].to_numpy()
            dx, dy = float(label_offset[0]), float(label_offset[1])
            for lon, lat, v in zip(lons, lats, labv):
                try:
                    if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
                        txt = str(label_fmt).format(float(v))
                    else:
                        txt = str(v)
                except Exception:
                    txt = str(v)
                ax.text(
                    lon + dx, lat + dy, txt,
                    fontsize=float(label_size),
                    alpha=float(label_alpha),
                    transform=ccrs.PlateCarree(),
                    zorder=zdef["labels"],
                )
    
        # -------------------------
        # legend (optional, always on top)
        # -------------------------
        if legend:
            lk = dict(legend_kwargs or {})
            if legend_handles is not None and legend_labels is not None:
                leg = ax.legend(legend_handles, legend_labels, **lk)
            else:
                leg = ax.legend(**lk)
            if leg is not None:
                leg.set_zorder(zdef["legend"])
    
        # -------------------------
        # colorbar (residuals)
        # -------------------------
        if show_colorbar:
            cb = plt.colorbar(sc, ax=ax, shrink=float(colorbar_shrink), pad=float(colorbar_pad))
            cb.set_label(str(value_label))
    
        if title:
            ax.set_title(str(title))
    
        if save:
            if not outpath:
                raise ValueError("save=True requires outpath.")
            fig.savefig(outpath, bbox_inches="tight")
            if log:
                self._wilber__log(f"[wilber-resid-map] wrote: {outpath}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return {"fig": fig, "ax": ax, "N": int(len(d)), "extent": extent_use}
    
    
    







    
    def wilber_fit_speed_scale_lambda(
        self,
        arrivals_df,
        T_prior,
        *,
        # model recomputation (accurate) vs time-scaling (fast)
        recompute_each=False,
        speed_model=None,
        seed_from="epicenter",
        overrides_template=None,
        # sweep controls
        lambda_list=None,                 # list/np.array; if None uses sensible defaults
        refine=True,                      # second pass around best
        refine_span=0.35,                 # +/- span around best
        refine_n=9,                       # number of points in refine pass
        # scoring
        score="mae",                      # "mae" | "rmse" | "median_abs"
        use_only_picked=True,
        exclude_flagged=True,
        flagged_col="outlier_flag",
        interp_method="nearest",
        # outputs
        return_table=True,
        log=True,
    ):
        """
        Fit / sweep global speed_scale_lambda against station observations.
    
        Two modes:
          A) recompute_each=False (FAST):
             uses approximation T_lambda ~ T_prior / lambda
          B) recompute_each=True (ACCURATE):
             recomputes T_map using wilber_compute_T_map_for_model with overrides_template updated.
    
        Returns:
          best_lambda, sweep_df (if return_table=True), best_T (T_map using best_lambda)
        """
        import numpy as np
        import pandas as pd
    
        if lambda_list is None:
            # broad "safe" range (you can override)
            lambda_list = np.array([0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2], dtype=float)
        else:
            lambda_list = np.asarray(lambda_list, dtype=float)
    
        def _score_from_resid(r):
            r = np.asarray(r, dtype=float)
            r = r[np.isfinite(r)]
            if r.size == 0:
                return np.inf
            if score == "rmse":
                return float(np.sqrt(np.mean(r ** 2)))
            if score == "median_abs":
                return float(np.median(np.abs(r)))
            # default mae
            return float(np.mean(np.abs(r)))
    
        # prepare station subset once
        df0 = arrivals_df.copy()
        if use_only_picked and "picked" in df0.columns:
            df0 = df0[df0["picked"].astype(bool) == True].copy()
        if exclude_flagged and flagged_col in df0.columns:
            df0 = df0[df0[flagged_col].astype(bool) == False].copy()
    
        if df0.empty:
            raise ValueError("No stations available for fitting after filtering (picked/flagged).")
    
        # observed times
        t_obs = df0["t_obs_s"].to_numpy(float)
    
        rows = []
        best = {"lam": None, "score": np.inf, "T": None, "snap": None}
    
        # helper to get T for lambda
        def _get_T_for_lambda(lam):
            if not recompute_each:
                # fast approximation: if c -> lam*c then t -> t/lam
                return (T_prior / float(lam)), None
    
            # accurate: recompute using your solver
            ov = dict(overrides_template or {})
            ov["speed_scale_lambda"] = float(lam)
            Tm, snap = self.wilber_compute_T_map_for_model(
                speed_model=(speed_model if speed_model is not None else ov.get("speed_model", None)),
                seed_from=seed_from,
                overrides=ov,
            )
            return Tm, snap
    
        for lam in lambda_list:
            Tm, _snap = _get_T_for_lambda(lam)
            t_mod = self._wilber__interp_T_at(Tm, df0["lon"].to_numpy(float), df0["lat"].to_numpy(float), method=str(interp_method))
            resid = t_obs - t_mod
            sc = _score_from_resid(resid)
    
            rows.append({"lambda": float(lam), "score": float(sc)})
            if sc < best["score"]:
                best["lam"] = float(lam)
                best["score"] = float(sc)
                best["T"] = Tm
                best["snap"] = _snap
    
        # refine around best (optional)
        if refine and best["lam"] is not None and np.isfinite(best["lam"]):
            lam0 = float(best["lam"])
            lo = max(0.05, lam0 - float(refine_span))
            hi = lam0 + float(refine_span)
            lams2 = np.linspace(lo, hi, int(refine_n)).astype(float)
    
            for lam in lams2:
                Tm, _snap = _get_T_for_lambda(lam)
                t_mod = self._wilber__interp_T_at(Tm, df0["lon"].to_numpy(float), df0["lat"].to_numpy(float), method=str(interp_method))
                resid = t_obs - t_mod
                sc = _score_from_resid(resid)
    
                rows.append({"lambda": float(lam), "score": float(sc), "refine": True})
                if sc < best["score"]:
                    best["lam"] = float(lam)
                    best["score"] = float(sc)
                    best["T"] = Tm
                    best["snap"] = _snap
    
        sweep_df = pd.DataFrame(rows)
        sweep_df = sweep_df.sort_values("score", ascending=True).reset_index(drop=True)
    
        if log:
            self._wilber__log(f"[wilber-fit-lambda] best lambda={best['lam']:.3f} score({score})={best['score']:.3f} recompute_each={recompute_each}")
    
        if return_table:
            return best["lam"], sweep_df, best["T"], best["snap"]
        return best["lam"], None, best["T"], best["snap"]
    
    
    def wilber_update_T_map_bayes_kernel(
        self,
        arrivals_df,
        T_prior,
        *,
        snap=None,
        lon_grid=None,
        lat_grid=None,
        # station selection
        use_only_picked=True,
        exclude_flagged=True,
        flagged_col="outlier_flag",
        # interpolation for residuals at stations
        interp_method="nearest",
        # Bayesian kernel update knobs
        radius_km=60.0,            # influence length scale
        kernel="gaussian",         # "gaussian" | "exp"
        prior_strength=1.0,        # stabilizer alpha: larger => smaller corrections (stronger prior)
        min_stations=3,            # if fewer stations overall -> no update (return prior)
        min_weight_sum=1e-6,       # numerical safety
        # clipping / robustness
        residual_clip_s=None,      # e.g., 120.0 to clip extreme residuals
        # optional: apply to a masked area only (future)
        # outputs
        return_fields=True,        # include correction field R and weight sum W
        log=True,
    ):
        """
        Bayesian-style spatial update to travel-time field using station residuals.
    
        Prior mean:
          T_prior(x)
    
        Observations at stations:
          t_obs_i
    
        Residuals at stations:
          r_i = t_obs_i - T_prior(x_i)
    
        Posterior mean correction:
          R_hat(x) = sum_i w_i(x) * r_i / (sum_i w_i(x) + prior_strength)
    
        Posterior:
          T_post(x) = T_prior(x) + R_hat(x)
    
        Returns:
          T_post, meta dict
            meta includes: correction_field, weight_sum_field, station_residuals_df, settings
        """
        import numpy as np
        import pandas as pd
    
        df = arrivals_df.copy()
        if use_only_picked and "picked" in df.columns:
            df = df[df["picked"].astype(bool) == True].copy()
        if exclude_flagged and flagged_col in df.columns:
            df = df[df[flagged_col].astype(bool) == False].copy()
    
        if len(df) < int(min_stations):
            if log:
                self._wilber__log(f"[wilber-bayes] Not enough stations for update (N={len(df)} < {min_stations}). Returning prior.")
            meta = {
                "note": "insufficient_stations",
                "N_stations": int(len(df)),
                "settings": {
                    "radius_km": float(radius_km),
                    "kernel": str(kernel),
                    "prior_strength": float(prior_strength),
                },
            }
            return T_prior.copy(), meta
    
        # get lon/lat grids
        LON, LAT = self._wilber__grid_lonlat_from_snap(snap, lon_grid=lon_grid, lat_grid=lat_grid)
        LON = np.asarray(LON, dtype=float)
        LAT = np.asarray(LAT, dtype=float)
    
        # modeled times at stations from T_prior
        t_mod = self._wilber__interp_T_at(
            T_prior,
            df["lon"].to_numpy(float),
            df["lat"].to_numpy(float),
            method=str(interp_method).lower().strip(),
        )
        df["t_model_s"] = t_mod
        df["residual_s"] = df["t_obs_s"].to_numpy(float) - df["t_model_s"].to_numpy(float)
    
        # optional clip
        if residual_clip_s is not None:
            rc = float(residual_clip_s)
            df["residual_s"] = np.clip(df["residual_s"].to_numpy(float), -rc, +rc)
    
        # prepare update fields
        R_num = np.zeros_like(T_prior, dtype=float)   # sum w*r
        W_sum = np.zeros_like(T_prior, dtype=float)   # sum w
    
        # fast-ish local distance approximation:
        # use equirectangular distance in km (good for moderate extents)
        # dx_km ~ 111*cos(lat0)*(dlon), dy_km ~ 111*(dlat)
        kernelU = str(kernel).lower().strip()
        rad = float(radius_km)
        rad2 = rad * rad
    
        for lon0, lat0, r0 in zip(df["lon"].to_numpy(float), df["lat"].to_numpy(float), df["residual_s"].to_numpy(float)):
            # compute dx,dy in km for every grid cell
            coslat = np.cos(np.deg2rad(float(lat0)))
            dx = (LON - lon0) * 111.0 * coslat
            dy = (LAT - lat0) * 111.0
            d2 = dx * dx + dy * dy
    
            if kernelU == "exp":
                # w = exp(-d/r)
                d = np.sqrt(d2)
                w = np.exp(-d / max(1e-6, rad))
            else:
                # gaussian: w = exp(-0.5*(d/r)^2)
                w = np.exp(-0.5 * (d2 / max(1e-12, rad2)))
    
            # accumulate
            W_sum += w
            R_num += w * float(r0)
    
        # posterior correction
        alpha = float(prior_strength)
        denom = (W_sum + alpha)
        denom = np.where(denom > float(min_weight_sum), denom, np.nan)
    
        R_hat = R_num / denom
        R_hat = np.where(np.isfinite(R_hat), R_hat, 0.0)
    
        T_post = T_prior + R_hat
    
        meta = {
            "N_stations": int(len(df)),
            "settings": {
                "radius_km": float(radius_km),
                "kernel": str(kernelU),
                "prior_strength": float(alpha),
                "interp_method": str(interp_method),
                "residual_clip_s": (None if residual_clip_s is None else float(residual_clip_s)),
            },
            "station_residuals_df": df,
        }
    
        if return_fields:
            meta["correction_field_s"] = R_hat
            meta["weight_sum_field"] = W_sum
    
        if log:
            # simple summary of residuals used
            rr = df["residual_s"].to_numpy(float)
            self._wilber__log(
                f"[wilber-bayes] Updated field using N={len(df)} stations. "
                f"Residuals: median={float(np.nanmedian(rr)):+.2f}s p10/p90=({float(np.nanpercentile(rr,10)):+.2f},{float(np.nanpercentile(rr,90)):+.2f}) "
                f"radius={rad:.1f}km prior_strength={alpha:.2f}"
            )
    
        return T_post, meta
    
    # override 
    def wilber_plot_T_map_difference(
        self,
        T_new,
        T_ref,
        *,
        snap=None,
        lon_grid=None,
        lat_grid=None,
        # field controls
        delta_label="ΔT (new - ref) [s]",
        cmap="coolwarm",
        vmin=None,
        vmax=None,
        alpha=0.85,
        # cartopy styling
        extent=None,
        extent_pad_deg=0.7,
        add_ocean=True,
        add_land=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=False,
        # figure
        title=None,
        figsize=(11.0, 8.5),
        dpi=250,
        show_colorbar=True,
        colorbar_shrink=0.86,
        colorbar_pad=0.02,
        # output
        show=True,
        save=False,
        outpath=None,
        log=True,
    ):
        """
        Cartopy basemap of delta field: (T_new - T_ref).
        Useful to visualize what the Bayesian update changed spatially.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    
        LON, LAT = self._wilber__grid_lonlat_from_snap(snap, lon_grid=lon_grid, lat_grid=lat_grid)
    
        D = np.asarray(T_new, dtype=float) - np.asarray(T_ref, dtype=float)
    
        # extent
        if extent is None:
            lonmin = float(np.nanmin(LON) - float(extent_pad_deg))
            lonmax = float(np.nanmax(LON) + float(extent_pad_deg))
            latmin = float(np.nanmin(LAT) - float(extent_pad_deg))
            latmax = float(np.nanmax(LAT) + float(extent_pad_deg))
            extent = [lonmin, lonmax, latmin, latmax]
    
        # symmetric scaling
        if vmin is None and vmax is None:
            m = float(np.nanmax(np.abs(D)))
            vmin, vmax = -m, +m
    
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
        if add_ocean:
            ax.add_feature(cfeature.OCEAN, zorder=0)
        if add_land:
            ax.add_feature(cfeature.LAND, zorder=0)
        if add_borders:
            ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=1)
        if add_coastlines:
            ax.coastlines(resolution="110m", linewidth=0.6, zorder=1)
    
        if add_gridlines:
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
    
        im = ax.pcolormesh(LON, LAT, D, cmap=cmap, vmin=vmin, vmax=vmax, alpha=float(alpha), transform=ccrs.PlateCarree(), zorder=2)
    
        if show_colorbar:
            cb = plt.colorbar(im, ax=ax, shrink=float(colorbar_shrink), pad=float(colorbar_pad))
            cb.set_label(str(delta_label))
    
        if title:
            ax.set_title(str(title))
    
        if save:
            if not outpath:
                raise ValueError("save=True requires outpath.")
            fig.savefig(outpath, bbox_inches="tight")
            if log:
                self._wilber__log(f"[wilber-delta-map] wrote: {outpath}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return {"fig": fig, "ax": ax, "extent": extent}


    
    # override
    def wilber_plot_T_map_difference(
        self,
        T_new,
        T_ref,
        *,
        snap=None,
        lon_grid=None,
        lat_grid=None,
        # field controls
        delta_label="ΔT (new - ref) [s]",
        cmap="coolwarm",
        vmin=None,
        vmax=None,
        alpha=0.85,
        # extent controls
        extent=None,                 # manual override
        extent_mode="grid",          # "grid" | "shakemap" | "manual"
        extent_pad_deg=0.7,
        # basemap styling
        basemap="simple",            # "simple" (current) | "usgs" (calls _usgs_basemap)
        basemap_kwargs=None,         # passed to _usgs_basemap when basemap="usgs"
        add_ocean=True,
        add_land=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=False,
        # contours of time (requested)
        show_contours=False,
        contour_source="ref",        # "ref" | "new"
        contour_levels=None,         # int or list/array; None -> matplotlib default
        contour_style="solid",       # "solid" | "dashed" | "dashdot" | "dotted"
        contour_color="k",
        contour_linewidth=0.8,
        contour_alpha=0.85,
        contour_kwargs=None,         # overrides for plt.contour
        clabel=False,
        clabel_kwargs=None,
        # zorder modularity
        zorder=None,                 # dict override
        # figure
        title=None,
        figsize=(11.0, 8.5),
        dpi=250,
        show_colorbar=True,
        colorbar_shrink=0.86,
        colorbar_pad=0.02,
        # output
        show=True,
        save=False,
        outpath=None,
        log=True,
    ):
        """
        Cartopy basemap of delta field: (T_new - T_ref).
    
        Modularity adds:
          - extent_mode: "grid" (default/current), "shakemap", "manual" (or pass extent=...)
          - show_contours + style/color/kwargs to plot time contours (ref or new) over delta
          - zorder dict to control layer ordering (contours/legend etc.)
          - basemap="usgs" optionally calls your module-level _usgs_basemap(ax, extent, ...)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    
        # -------------------------
        # zorder defaults (overrideable)
        # -------------------------
        zdef = {
            "basemap": 0,
            "land_ocean": 0,
            "borders_coast": 1,
            "gridlines": 2,
            "field": 2,
            "contours": 3,
            "labels": 4,
            "legend": 2000,
            "colorbar": 2100,
        }
        if isinstance(zorder, dict):
            zdef.update(zorder)
    
        # grids
        LON, LAT = self._wilber__grid_lonlat_from_snap(snap, lon_grid=lon_grid, lat_grid=lat_grid)
    
        Tn = np.asarray(T_new, dtype=float)
        Tr = np.asarray(T_ref, dtype=float)
        D = Tn - Tr
    
        # -------------------------
        # extent resolution
        # -------------------------
        if extent is not None:
            extent_mode = "manual"
    
        def _extent_from_grid():
            return [
                float(np.nanmin(LON) - float(extent_pad_deg)),
                float(np.nanmax(LON) + float(extent_pad_deg)),
                float(np.nanmin(LAT) - float(extent_pad_deg)),
                float(np.nanmax(LAT) + float(extent_pad_deg)),
            ]
    
        def _extent_from_shakemap_grid():
            lonG = getattr(self, "lon_grid", None)
            latG = getattr(self, "lat_grid", None)
            if lonG is None or latG is None:
                lonG = getattr(self, "_wilber_last_lon_grid", None)
                latG = getattr(self, "_wilber_last_lat_grid", None)
            if lonG is None or latG is None:
                raise ValueError("extent_mode='shakemap' requested but no lon/lat grid found on self.")
            lonG = np.asarray(lonG, dtype=float)
            latG = np.asarray(latG, dtype=float)
            return [
                float(np.nanmin(lonG) - float(extent_pad_deg)),
                float(np.nanmax(lonG) + float(extent_pad_deg)),
                float(np.nanmin(latG) - float(extent_pad_deg)),
                float(np.nanmax(latG) + float(extent_pad_deg)),
            ]
    
        if extent_mode == "manual":
            if extent is None:
                raise ValueError("extent_mode='manual' requires extent=[lonmin, lonmax, latmin, latmax].")
            extent_use = list(map(float, extent))
        elif extent_mode == "shakemap":
            extent_use = _extent_from_shakemap_grid()
        else:
            # default/current
            extent_use = _extent_from_grid()
    
        # -------------------------
        # symmetric scaling
        # -------------------------
        if vmin is None and vmax is None:
            m = float(np.nanmax(np.abs(D)))
            vmin, vmax = -m, +m
    
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent_use, crs=ccrs.PlateCarree())
    
        # -------------------------
        # basemap
        # -------------------------
        if basemap_kwargs is None:
            basemap_kwargs = {}
        if str(basemap).lower() == "usgs":
            try:
                _usgs_basemap(ax, extent_use, **basemap_kwargs)
            except TypeError:
                _usgs_basemap(ax, extent_use)
        else:
            if add_ocean:
                ax.add_feature(cfeature.OCEAN, zorder=zdef["land_ocean"])
            if add_land:
                ax.add_feature(cfeature.LAND, zorder=zdef["land_ocean"])
            if add_borders:
                ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=zdef["borders_coast"])
            if add_coastlines:
                ax.coastlines(resolution="110m", linewidth=0.6, zorder=zdef["borders_coast"])
    
            if add_gridlines:
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle="--")
                gl.top_labels = False
                gl.right_labels = False
    
        # -------------------------
        # delta field
        # -------------------------
        im = ax.pcolormesh(
            LON, LAT, D,
            cmap=cmap, vmin=vmin, vmax=vmax,
            alpha=float(alpha),
            transform=ccrs.PlateCarree(),
            zorder=zdef["field"],
        )
    
        # -------------------------
        # contours (time)
        # -------------------------
        if show_contours:
            src = str(contour_source).lower().strip()
            Tsrc = Tr if src == "ref" else Tn
    
            ls_map = {"solid": "-", "dashed": "--", "dashdot": "-.", "dotted": ":"}
            linestyle = ls_map.get(str(contour_style).lower().strip(), "-")
    
            ck = {
                "colors": contour_color,
                "linewidths": float(contour_linewidth),
                "linestyles": linestyle,
                "alpha": float(contour_alpha),
                "transform": ccrs.PlateCarree(),
                "zorder": zdef["contours"],
            }
            # handle levels
            if contour_levels is not None:
                ck["levels"] = contour_levels
    
            if isinstance(contour_kwargs, dict):
                ck.update(contour_kwargs)
    
            cs = ax.contour(LON, LAT, Tsrc, **ck)
    
            if clabel:
                lk = {"fontsize": 8, "inline": True}
                if isinstance(clabel_kwargs, dict):
                    lk.update(clabel_kwargs)
                ax.clabel(cs, **lk)
    
        # -------------------------
        # colorbar
        # -------------------------
        if show_colorbar:
            cb = plt.colorbar(im, ax=ax, shrink=float(colorbar_shrink), pad=float(colorbar_pad))
            cb.set_label(str(delta_label))
    
        if title:
            ax.set_title(str(title))
    
        if save:
            if not outpath:
                raise ValueError("save=True requires outpath.")
            fig.savefig(outpath, bbox_inches="tight")
            if log:
                self._wilber__log(f"[wilber-delta-map] wrote: {outpath}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return {"fig": fig, "ax": ax, "extent": extent_use}
    
        



    def wilber_plot_T_map_difference(
        self,
        T_new,
        T_ref,
        *,
        snap=None,
        lon_grid=None,
        lat_grid=None,
        # -------------------------
        # NEW: ShakeMap background (DEFAULT ON)
        # -------------------------
        show_shakemap=True,
        base_imt="MMI",
        base_alpha=0.55,
        base_cmap=None,
        show_shakemap_colorbar=False,   # default OFF for delta map (usually noisy)
        shakemap_colorbar_shrink=0.80,
        shakemap_colorbar_pad=0.02,
        # field controls
        delta_label="ΔT (new - ref) [s]",
        cmap="coolwarm",
        vmin=None,
        vmax=None,
        alpha=0.85,
        # extent controls
        extent=None,                 # manual override
        extent_mode="grid",          # "grid" | "shakemap" | "manual"
        extent_pad_deg=0.7,
        # basemap styling
        basemap="simple",            # "simple" | "usgs"
        basemap_kwargs=None,
        add_ocean=True,
        add_land=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=False,
        gridline_kwargs=None,
        # contours of time (DEFAULT ON now)
        show_contours=True,
        contour_source="ref",        # "ref" | "new" | "both"
        contour_levels=None,
        contour_style="solid",
        contour_color="k",
        contour_linewidth=0.8,
        contour_alpha=0.85,
        contour_kwargs=None,
        clabel=False,
        clabel_kwargs=None,
    
        # -------------------------
        # station overlay (DEFAULT ON)
        # -------------------------
        add_station_overlay=True,
        arrivals_df=None,            # dataframe with lon/lat + picked/flags + t_obs_s/residual_s
        use_only_picked=True,
        picked_col="picked",
        # flags to consider "flagged" (any present will be used)
        flag_cols=("outlier_flag", "soft_inconsistent_flag", "local_outlier_flag"),
        # clip station overlay to map extent
        clip_stations_to_extent=True,
    
        # station styling
        station_marker="^",
        station_color="red",
        station_edge="k",
        station_size=40,
        unpicked_alpha=0.30,
        picked_alpha=0.95,
        flagged_alpha=0.20,
    
        # station labels (used + flagged)
        show_station_labels=True,
        station_label_fmt="{t:.0f}s",
        station_label_offset=(0.03, 0.03),
    
        show_flagged_labels=True,
        flagged_label_alpha=0.35,
        flagged_label_fmt_outlier="{res:+.0f}s",     # uses residual_s if available
        flagged_label_fmt_soft="spread={spr:.0f}s",  # uses soft_cluster_spread_s if available
        flagged_label_fmt_local="local z={z:.1f}",   # uses local_outlier_score if available
    
        # legend control for overlay
        add_station_legend=True,
        station_legend_kwargs=None,
    
        # zorder modularity
        zorder=None,
        # figure
        title=None,
        figsize=(11.0, 8.5),
        dpi=250,
        show_colorbar=True,
        colorbar_shrink=0.86,
        colorbar_pad=0.02,
        # output
        show=True,
        save=False,
        outpath=None,
        log=True,
    ):
        """
        Cartopy basemap of delta field: (T_new - T_ref).
    
        Adds (optional, default ON):
          - ShakeMap raster underlay (MMI uses USGS discrete palette if available)
          - time contours from ref/new/both over delta
          - station overlay (picked/unpicked/flagged) + labels
    
        Notes:
          - If `arrivals_df` is None, station overlay is skipped automatically.
          - Station overlay can be clipped to extent to avoid “outside map” artifacts.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    
        # -------------------------
        # zorder defaults (overrideable)
        # -------------------------
        zdef = {
            "basemap": 0,
            "land_ocean": 0,
            "borders_coast": 1,
            "gridlines": 2,
            "shakemap": 2,          # underlay
            "field": 3,             # delta
            "contours": 4,
            "stations_unpicked": 5,
            "stations_picked": 7,
            "stations_flagged": 7,
            "labels": 8,
            "legend": 2000,
            "colorbar": 2100,
            "shakemap_colorbar": 2050,
        }
        if isinstance(zorder, dict):
            zdef.update(zorder)
    
        # grids
        LON, LAT = self._wilber__grid_lonlat_from_snap(snap, lon_grid=lon_grid, lat_grid=lat_grid)
    
        Tn = np.asarray(T_new, dtype=float)
        Tr = np.asarray(T_ref, dtype=float)
        D = Tn - Tr
    
        # -------------------------
        # extent resolution
        # -------------------------
        if extent is not None:
            extent_mode = "manual"
    
        def _extent_from_grid():
            return [
                float(np.nanmin(LON) - float(extent_pad_deg)),
                float(np.nanmax(LON) + float(extent_pad_deg)),
                float(np.nanmin(LAT) - float(extent_pad_deg)),
                float(np.nanmax(LAT) + float(extent_pad_deg)),
            ]
    
        def _extent_from_shakemap_grid():
            lonG = getattr(self, "lon_grid", None)
            latG = getattr(self, "lat_grid", None)
            if lonG is None or latG is None:
                lonG = getattr(self, "_wilber_last_lon_grid", None)
                latG = getattr(self, "_wilber_last_lat_grid", None)
            if lonG is None or latG is None:
                raise ValueError("extent_mode='shakemap' requested but no lon/lat grid found on self.")
            lonG = np.asarray(lonG, dtype=float)
            latG = np.asarray(latG, dtype=float)
            return [
                float(np.nanmin(lonG) - float(extent_pad_deg)),
                float(np.nanmax(lonG) + float(extent_pad_deg)),
                float(np.nanmin(latG) - float(extent_pad_deg)),
                float(np.nanmax(latG) + float(extent_pad_deg)),
            ]
    
        if str(extent_mode).lower().strip() == "manual":
            if extent is None:
                raise ValueError("extent_mode='manual' requires extent=[lonmin, lonmax, latmin, latmax].")
            extent_use = list(map(float, extent))
        elif str(extent_mode).lower().strip() == "shakemap":
            extent_use = _extent_from_shakemap_grid()
        else:
            extent_use = _extent_from_grid()
    
        # -------------------------
        # symmetric scaling
        # -------------------------
        if vmin is None and vmax is None:
            m = float(np.nanmax(np.abs(D)))
            vmin, vmax = -m, +m
    
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent_use, crs=ccrs.PlateCarree())
    
        # -------------------------
        # basemap
        # -------------------------
        if basemap_kwargs is None:
            basemap_kwargs = {}
        if gridline_kwargs is None:
            gridline_kwargs = {}
    
        if str(basemap).lower() == "usgs":
            try:
                _usgs_basemap(ax, extent_use, **basemap_kwargs)
            except TypeError:
                _usgs_basemap(ax, extent_use)
        else:
            if add_ocean:
                ax.add_feature(cfeature.OCEAN, zorder=zdef["land_ocean"])
            if add_land:
                ax.add_feature(cfeature.LAND, zorder=zdef["land_ocean"])
            if add_borders:
                ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=zdef["borders_coast"])
            if add_coastlines:
                ax.coastlines(resolution="110m", linewidth=0.6, zorder=zdef["borders_coast"])
    
            if add_gridlines:
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle="--", **gridline_kwargs)
                gl.top_labels = False
                gl.right_labels = False
    
        # -------------------------
        # ShakeMap underlay (optional)
        # -------------------------
        im_sm = None
        mmi_mode = str(base_imt).upper().strip() == "MMI"
        if bool(show_shakemap):
            Z = None
            try:
                Z = self._wilber__get_shakemap_field(base_imt)
            except Exception:
                Z = None
    
            if Z is not None:
                proj = ccrs.PlateCarree()
                if mmi_mode and (base_cmap is None):
                    # use USGS MMI discrete palette if available in your module
                    try:
                        cmap_sm, norm_sm, ticks_sm, label_sm = _usgs_mmi_cmap_norm()
                        im_sm = ax.pcolormesh(
                            self.lon_grid, self.lat_grid, Z,
                            transform=proj,
                            shading="auto",
                            alpha=float(base_alpha),
                            zorder=zdef["shakemap"],
                            cmap=cmap_sm,
                            norm=norm_sm,
                        )
                    except Exception:
                        im_sm = ax.pcolormesh(
                            self.lon_grid, self.lat_grid, Z,
                            transform=proj,
                            shading="auto",
                            alpha=float(base_alpha),
                            zorder=zdef["shakemap"],
                        )
                else:
                    im_sm = ax.pcolormesh(
                        self.lon_grid, self.lat_grid, Z,
                        transform=ccrs.PlateCarree(),
                        shading="auto",
                        alpha=float(base_alpha),
                        zorder=zdef["shakemap"],
                        cmap=base_cmap,
                    )
    
                if bool(show_shakemap_colorbar) and (im_sm is not None):
                    cbar_sm = plt.colorbar(im_sm, ax=ax, shrink=float(shakemap_colorbar_shrink), pad=float(shakemap_colorbar_pad))
                    if mmi_mode:
                        try:
                            _, _, ticks_sm, label_sm = _usgs_mmi_cmap_norm()
                            cbar_sm.set_ticks(ticks_sm)
                            cbar_sm.set_label(label_sm)
                        except Exception:
                            cbar_sm.set_label(str(base_imt))
                    else:
                        cbar_sm.set_label(str(base_imt))
    
        # -------------------------
        # delta field
        # -------------------------
        im = ax.pcolormesh(
            LON, LAT, D,
            cmap=cmap, vmin=vmin, vmax=vmax,
            alpha=float(alpha),
            transform=ccrs.PlateCarree(),
            zorder=zdef["field"],
        )
    
        # -------------------------
        # contours (time) — supports "ref", "new", "both"
        # -------------------------
        if bool(show_contours):
            src = str(contour_source).lower().strip()
            if src not in ("ref", "new", "both"):
                src = "ref"
    
            ls_map = {"solid": "-", "dashed": "--", "dashdot": "-.", "dotted": ":"}
            linestyle = ls_map.get(str(contour_style).lower().strip(), "-")
    
            def _plot_contours(Tsrc):
                ck = {
                    "colors": contour_color,
                    "linewidths": float(contour_linewidth),
                    "linestyles": linestyle,
                    "alpha": float(contour_alpha),
                    "transform": ccrs.PlateCarree(),
                    "zorder": zdef["contours"],
                }
                if contour_levels is not None:
                    ck["levels"] = contour_levels
                if isinstance(contour_kwargs, dict):
                    ck.update(contour_kwargs)
    
                cs = ax.contour(LON, LAT, np.asarray(Tsrc, float), **ck)
                if clabel:
                    lk = {"fontsize": 8, "inline": True}
                    if isinstance(clabel_kwargs, dict):
                        lk.update(clabel_kwargs)
                    ax.clabel(cs, **lk)
                return cs
    
            if src in ("ref", "both"):
                _plot_contours(Tr)
            if src in ("new", "both"):
                _plot_contours(Tn)
    
        # -------------------------
        # station overlay (picked/unpicked/flagged) + labels
        # -------------------------
        station_handles = []
        station_labels = []
    
        if bool(add_station_overlay) and (arrivals_df is not None):
            df = arrivals_df.copy()
    
            if ("lon" not in df.columns) or ("lat" not in df.columns):
                raise ValueError("arrivals_df must contain 'lon' and 'lat' columns for station overlay.")
    
            df["lon"] = np.asarray(df["lon"], float)
            df["lat"] = np.asarray(df["lat"], float)
    
            if bool(clip_stations_to_extent):
                eps = 1e-9
                in_ext = (
                    (df["lon"] >= extent_use[0] - eps) &
                    (df["lon"] <= extent_use[1] + eps) &
                    (df["lat"] >= extent_use[2] - eps) &
                    (df["lat"] <= extent_use[3] + eps)
                )
                df = df.loc[in_ext].copy()
    
            if picked_col in df.columns:
                picked_mask = df[picked_col].astype(bool).to_numpy()
            else:
                picked_mask = np.isfinite(np.asarray(df["t_obs_s"], float)) if "t_obs_s" in df.columns else np.ones(len(df), dtype=bool)
    
            flag_any = np.zeros(len(df), dtype=bool)
            for c in (flag_cols or ()):
                if c in df.columns:
                    flag_any |= df[c].astype(bool).to_numpy()
    
            if bool(use_only_picked):
                df = df.loc[picked_mask].copy()
                # recompute masks after trim
                if picked_col in df.columns:
                    picked_mask = df[picked_col].astype(bool).to_numpy()
                else:
                    picked_mask = np.isfinite(np.asarray(df["t_obs_s"], float)) if "t_obs_s" in df.columns else np.ones(len(df), dtype=bool)
    
                flag_any = np.zeros(len(df), dtype=bool)
                for c in (flag_cols or ()):
                    if c in df.columns:
                        flag_any |= df[c].astype(bool).to_numpy()
    
            d_unpicked = df.loc[~picked_mask].copy()
            d_picked = df.loc[picked_mask & (~flag_any)].copy()
            d_flagged = df.loc[picked_mask & (flag_any)].copy()
    
            if len(d_unpicked) and (not bool(use_only_picked)):
                h = ax.scatter(
                    d_unpicked["lon"], d_unpicked["lat"],
                    s=float(station_size),
                    marker=station_marker,
                    c=station_color,
                    edgecolors=station_edge,
                    linewidths=0.4,
                    alpha=float(unpicked_alpha),
                    transform=ccrs.PlateCarree(),
                    zorder=zdef["stations_unpicked"],
                    label="Stations (no pick)",
                )
                station_handles.append(h); station_labels.append("Stations (no pick)")
    
            if len(d_picked):
                h = ax.scatter(
                    d_picked["lon"], d_picked["lat"],
                    s=float(station_size),
                    marker=station_marker,
                    c=station_color,
                    edgecolors=station_edge,
                    linewidths=0.4,
                    alpha=float(picked_alpha),
                    transform=ccrs.PlateCarree(),
                    zorder=zdef["stations_picked"],
                    label="Stations (used)",
                )
                station_handles.append(h); station_labels.append("Stations (used)")
    
            if len(d_flagged):
                h = ax.scatter(
                    d_flagged["lon"], d_flagged["lat"],
                    s=float(station_size),
                    marker=station_marker,
                    c=station_color,
                    edgecolors=station_edge,
                    linewidths=0.4,
                    alpha=float(flagged_alpha),
                    transform=ccrs.PlateCarree(),
                    zorder=zdef["stations_flagged"],
                    label="Stations (flagged)",
                )
                station_handles.append(h); station_labels.append("Stations (flagged)")
    
            dx, dy = float(station_label_offset[0]), float(station_label_offset[1])
    
            if bool(show_station_labels) and len(d_picked) and ("t_obs_s" in d_picked.columns):
                for r in d_picked.itertuples(index=False):
                    try:
                        t = float(getattr(r, "t_obs_s"))
                        if not np.isfinite(t):
                            continue
                        ax.text(
                            float(getattr(r, "lon")) + dx,
                            float(getattr(r, "lat")) + dy,
                            station_label_fmt.format(t=t),
                            transform=ccrs.PlateCarree(),
                            fontsize=8,
                            color="k",
                            zorder=zdef["labels"],
                            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2),
                        )
                    except Exception:
                        continue
    
            if bool(show_flagged_labels) and len(d_flagged):
                for r in d_flagged.itertuples(index=False):
                    try:
                        lon = float(getattr(r, "lon"))
                        lat = float(getattr(r, "lat"))
                        lab = None
    
                        if hasattr(r, "outlier_flag") and bool(getattr(r, "outlier_flag")):
                            if hasattr(r, "residual_s"):
                                res = float(getattr(r, "residual_s"))
                                if np.isfinite(res):
                                    lab = str(flagged_label_fmt_outlier).format(res=res)
                            if lab is None:
                                lab = "hard outlier"
    
                        if lab is None and hasattr(r, "soft_inconsistent_flag") and bool(getattr(r, "soft_inconsistent_flag")):
                            if hasattr(r, "soft_cluster_spread_s"):
                                spr = float(getattr(r, "soft_cluster_spread_s"))
                                if np.isfinite(spr):
                                    lab = str(flagged_label_fmt_soft).format(spr=spr)
                            if lab is None:
                                lab = "soft inconsistent"
    
                        if lab is None and hasattr(r, "local_outlier_flag") and bool(getattr(r, "local_outlier_flag")):
                            if hasattr(r, "local_outlier_score"):
                                z = float(getattr(r, "local_outlier_score"))
                                if np.isfinite(z):
                                    lab = str(flagged_label_fmt_local).format(z=z)
                            if lab is None:
                                lab = "local outlier"
    
                        if lab is None:
                            continue
    
                        ax.text(
                            lon + dx,
                            lat + dy,
                            lab,
                            transform=ccrs.PlateCarree(),
                            fontsize=7,
                            color="k",
                            alpha=float(flagged_label_alpha),
                            zorder=zdef["labels"] + 1,
                            bbox=dict(facecolor="white", alpha=0.45, edgecolor="none", pad=1.0),
                        )
                    except Exception:
                        continue
    
            if bool(add_station_legend) and len(station_handles):
                lk = dict(station_legend_kwargs or {})
                if "loc" not in lk:
                    lk["loc"] = "lower left"
                if "framealpha" not in lk:
                    lk["framealpha"] = 0.85
    
                try:
                    leg = ax.legend(station_handles, station_labels, **lk)
                    if leg is not None:
                        leg.set_zorder(zdef["legend"])
                except Exception:
                    pass
    
        # -------------------------
        # colorbars (delta always, shakemap optional)
        # -------------------------
        if show_colorbar:
            cb = plt.colorbar(im, ax=ax, shrink=float(colorbar_shrink), pad=float(colorbar_pad))
            cb.set_label(str(delta_label))
    
        if title:
            ax.set_title(str(title))
    
        if save:
            if not outpath:
                raise ValueError("save=True requires outpath.")
            fig.savefig(outpath, bbox_inches="tight")
            if log:
                self._wilber__log(f"[wilber-delta-map] wrote: {outpath}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return {
            "fig": fig,
            "ax": ax,
            "extent": extent_use,
            "station_overlay": bool(add_station_overlay and (arrivals_df is not None)),
            "shakemap_underlay": bool(show_shakemap),
            "time_contours": bool(show_contours),
        }
    
    



    # =============================================================================
    # PATCH: GRID SNAP CACHING + SAFE BAYES UPDATE (no snap required)
    # =============================================================================
    
    def wilber_compute_T_map_for_model(
        self,
        *,
        speed_model="uniform",
        seed_from="epicenter",
        overrides=None,
    ):
        """
        Compute a travel-time map T_map for a given speed model.
    
        Patch addition:
          - caches the returned 'snap' (grid metadata) on the instance so other
            downstream steps (e.g., Bayesian update) can reuse the SAME grid
            even if user forgets to pass snap around.
    
        Returns:
          T_map, snap
        """
        # Call your existing implementation (original body)
        # -------------------------------------------------
        # NOTE: This assumes you already have a working method with this name.
        # We re-run the original via super-pattern is not available; so we must
        # inline the prior body here in your codebase. Since you are pasting at
        # the end of the class, we need to *re-implement* by calling the original
        # method name stored before override. If you don't have that, then:
        #
        # -> EASIEST: rename your current method to _wilber__compute_T_map_for_model_impl
        #    once, then this wrapper will call it.
        #
        # But to avoid requiring that refactor, we do a safe approach:
        # - If your class already has an internal helper (common in your patches)
        #   called _wilber__compute_T_map_for_model_impl, we use it.
        # - Otherwise we raise with an instructive message.
        #
        impl = getattr(self, "_wilber__compute_T_map_for_model_impl", None)
        if impl is None:
            raise RuntimeError(
                "This patch expects your original wilber_compute_T_map_for_model logic to be "
                "moved into self._wilber__compute_T_map_for_model_impl(...) once. "
                "Do this: rename your current wilber_compute_T_map_for_model -> _wilber__compute_T_map_for_model_impl, "
                "then paste this patch again."
            )
    
        T_map, snap = impl(speed_model=speed_model, seed_from=seed_from, overrides=overrides)
    
        # Cache snap/grid for downstream steps (Bayes update, diagnostics, etc.)
        try:
            self._wilber_last_snap = snap
            # If snap contains lon/lat grids, cache them too (optional but useful)
            lonlat = None
            if isinstance(snap, dict):
                for k_lon, k_lat in (("lon", "lat"), ("LON", "LAT"), ("lon_grid", "lat_grid"), ("LON_grid", "LAT_grid")):
                    if k_lon in snap and k_lat in snap:
                        lonlat = (snap[k_lon], snap[k_lat])
                        break
            if lonlat is not None:
                self._wilber_last_lon_grid = lonlat[0]
                self._wilber_last_lat_grid = lonlat[1]
            else:
                # leave as-is; _wilber__grid_lonlat_from_snap can derive from snap later
                pass
        except Exception:
            # Never break compute because caching failed
            pass
    
        return T_map, snap
    
    
    def _wilber__grid_lonlat_from_snap(self, snap, lon_grid=None, lat_grid=None):
        """
        Resolve the canonical (lon_grid, lat_grid) for a T_map.
    
        Priority:
          1) explicit lon_grid/lat_grid args (if both provided)
          2) snap (if provided)
          3) cached last-grid from self._wilber_last_snap / self._wilber_last_lon_grid/_lat_grid
    
        This prevents pipeline breaks when 'snap' is accidentally not passed around.
        """
        import numpy as np
    
        # 1) explicit grids override everything
        if lon_grid is not None and lat_grid is not None:
            return np.asarray(lon_grid, dtype=float), np.asarray(lat_grid, dtype=float)
    
        # 2) try snap if provided
        if snap is not None:
            if isinstance(snap, dict):
                # common patterns in your codebase
                for k_lon, k_lat in (
                    ("lon", "lat"),
                    ("LON", "LAT"),
                    ("lon_grid", "lat_grid"),
                    ("LON_grid", "LAT_grid"),
                ):
                    if k_lon in snap and k_lat in snap:
                        return np.asarray(snap[k_lon], dtype=float), np.asarray(snap[k_lat], dtype=float)
    
                # another common pattern: snap["grid"]={"lon":..,"lat":..}
                g = snap.get("grid", None)
                if isinstance(g, dict) and ("lon" in g and "lat" in g):
                    return np.asarray(g["lon"], dtype=float), np.asarray(g["lat"], dtype=float)
    
        # 3) fallback to cached grid
        lon_c = getattr(self, "_wilber_last_lon_grid", None)
        lat_c = getattr(self, "_wilber_last_lat_grid", None)
        if lon_c is not None and lat_c is not None:
            return np.asarray(lon_c, dtype=float), np.asarray(lat_c, dtype=float)
    
        snap_c = getattr(self, "_wilber_last_snap", None)
        if snap_c is not None and isinstance(snap_c, dict):
            for k_lon, k_lat in (
                ("lon", "lat"),
                ("LON", "LAT"),
                ("lon_grid", "lat_grid"),
                ("LON_grid", "LAT_grid"),
            ):
                if k_lon in snap_c and k_lat in snap_c:
                    return np.asarray(snap_c[k_lon], dtype=float), np.asarray(snap_c[k_lat], dtype=float)
            g = snap_c.get("grid", None)
            if isinstance(g, dict) and ("lon" in g and "lat" in g):
                return np.asarray(g["lon"], dtype=float), np.asarray(g["lat"], dtype=float)
    
        raise ValueError("Need snap (from wilber_compute_T_map_for_model) or lon_grid/lat_grid.")
    
    
    def wilber_update_T_map_bayes_kernel(
        self,
        arrivals_df,
        T_prior,
        *,
        snap=None,
        lon_grid=None,
        lat_grid=None,
        use_only_picked=True,
        exclude_flagged=True,
        flagged_col="outlier_flag",
        interp_method="nearest",
        # Bayesian / kernel knobs
        radius_km=50.0,
        kernel="gaussian",          # "gaussian" | "inv" | "inv2"
        prior_strength=1.0,         # larger -> less correction (stronger prior)
        min_stations=8,
        min_weight_sum=1e-6,
        residual_clip_s=None,       # e.g., 300.0 to prevent crazy corrections
        return_fields=False,
        log=True,
    ):
        """
        Bayesian-style residual update:
          T_post(x) = T_prior(x) + R_hat(x)
    
        where R_hat(x) is a kernel-weighted average of station residuals with a
        prior_strength term that shrinks corrections toward 0 away from stations.
    
        PATCH FIX:
          - snap is OPTIONAL now. If snap=None, we fall back to cached last-grid
            from wilber_compute_T_map_for_model, or user-provided lon_grid/lat_grid.
        """
        import numpy as np
    
        # ------------------------------------------------------------
        # 0) pick station residuals r_i = t_obs - t_pred(prior)
        # ------------------------------------------------------------
        df = arrivals_df.copy()
        if use_only_picked:
            df = df[df["picked"] == True].copy()
    
        if exclude_flagged and (flagged_col in df.columns):
            df = df[df[flagged_col].astype(bool) == False].copy()
    
        # must have finite obs + coords
        for c in ("lon", "lat", "t_obs_s"):
            if c not in df.columns:
                raise ValueError(f"arrivals_df missing required column '{c}'")
    
        df = df[np.isfinite(df["lon"].astype(float)) & np.isfinite(df["lat"].astype(float))].copy()
        df = df[np.isfinite(df["t_obs_s"].astype(float))].copy()
    
        if len(df) < int(min_stations):
            if log:
                self._wilber__log(f"[wilber-bayes] not enough stations for update: N={len(df)} < {int(min_stations)} -> returning prior.")
            meta = {"n_used": int(len(df)), "status": "not_enough_stations"}
            return (T_prior.copy(), meta) if not return_fields else (T_prior.copy(), meta, None, None)
    
        # station predicted times from prior
        t_pred_i = self._wilber__interp_T_at(
            T_prior,
            df["lon"].to_numpy(float),
            df["lat"].to_numpy(float),
            method=str(interp_method),
        )
        r_i = df["t_obs_s"].to_numpy(float) - np.asarray(t_pred_i, dtype=float)
    
        if residual_clip_s is not None and np.isfinite(float(residual_clip_s)):
            clip = float(residual_clip_s)
            r_i = np.clip(r_i, -clip, +clip)
    
        # ------------------------------------------------------------
        # 1) get lon/lat grids (snap OR explicit OR cached)
        # ------------------------------------------------------------
        LON, LAT = self._wilber__grid_lonlat_from_snap(snap, lon_grid=lon_grid, lat_grid=lat_grid)
        LON = np.asarray(LON, dtype=float)
        LAT = np.asarray(LAT, dtype=float)
    
        # allow both meshgrid-like (2D) or vectors (1D)
        if LON.ndim == 1 and LAT.ndim == 1:
            LON2, LAT2 = np.meshgrid(LON, LAT)
        else:
            LON2, LAT2 = LON, LAT
    
        # ------------------------------------------------------------
        # 2) compute kernel weights and correction field
        # ------------------------------------------------------------
        # fast approximate km distance (good enough for local/regional)
        # km per degree lat ~111; lon scaled by cos(lat)
        km_per_deg = 111.0
        lat0 = float(np.nanmean(df["lat"].to_numpy(float)))
        coslat = float(np.cos(np.deg2rad(lat0)))
        coslat = max(0.1, min(1.0, coslat))
    
        sx = df["lon"].to_numpy(float)
        sy = df["lat"].to_numpy(float)
    
        # flatten grid for vectorized compute
        gx = LON2.ravel()
        gy = LAT2.ravel()
    
        dx = (gx[:, None] - sx[None, :]) * (km_per_deg * coslat)
        dy = (gy[:, None] - sy[None, :]) * km_per_deg
        d2 = dx * dx + dy * dy
    
        R = float(radius_km)
        R = max(1e-3, R)
    
        ker = str(kernel).lower().strip()
        if ker == "gaussian":
            w = np.exp(-0.5 * d2 / (R * R))
        elif ker == "inv":
            w = 1.0 / (np.sqrt(d2) + 1e-6)
        elif ker == "inv2":
            w = 1.0 / (d2 + 1e-6)
        else:
            w = np.exp(-0.5 * d2 / (R * R))
    
        wsum = np.sum(w, axis=1)
    
        # Bayesian shrinkage: add prior_strength to denominator (prevents overfit)
        alpha = max(0.0, float(prior_strength))
        denom = (wsum + alpha)
    
        # avoid divide-by-zero
        good = denom > float(min_weight_sum)
        corr = np.zeros_like(denom, dtype=float)
        if np.any(good):
            corr[good] = (w[good, :] @ r_i) / denom[good]
    
        Corr2D = corr.reshape(LON2.shape)
        T_post = np.asarray(T_prior, dtype=float) + Corr2D
    
        meta = {
            "n_used": int(len(df)),
            "kernel": ker,
            "radius_km": float(R),
            "prior_strength": float(alpha),
            "residual_clip_s": float(residual_clip_s) if residual_clip_s is not None else None,
            "status": "ok",
        }
    
        if log:
            self._wilber__log(f"[wilber-bayes] update ok: N={meta['n_used']} kernel={ker} R={R:.1f}km prior_strength={alpha:.2f}")
    
        if return_fields:
            # return correction + weight sum fields for diagnostics
            Wsum2D = wsum.reshape(LON2.shape)
            return T_post, meta, Corr2D, Wsum2D
    
        return T_post, meta



    # =============================================================================
    # PATCH: GRID SNAP CACHING + SAFE BAYES UPDATE (no snap required)
    # =============================================================================
    
    def wilber_compute_T_map_for_model(
        self,
        *,
        speed_model="vs30_piecewise",
        seed_from="epicenter",
        overrides=None,
    ):
        """
        Wrapper around your existing implementation that also caches the canonical
        grid (lon/lat) and the returned snap for downstream steps.
    
        Works with either of these implementations (no renaming required):
          - self.wilber_compute_T_map_for_model_impl   (your current name)
          - self._wilber__compute_T_map_for_model_impl (older patch name)
        """
        import numpy as np
    
        # ---------------------------------------------------------
        # 1) find the implementation
        # ---------------------------------------------------------
        impl = getattr(self, "wilber_compute_T_map_for_model_impl", None)
        if impl is None:
            impl = getattr(self, "_wilber__compute_T_map_for_model_impl", None)
    
        if impl is None:
            raise RuntimeError(
                "Could not find a compute implementation. Expected either:\n"
                "  - wilber_compute_T_map_for_model_impl(...)\n"
                "  - _wilber__compute_T_map_for_model_impl(...)\n"
                "Your class must define one of those."
            )
    
        # ---------------------------------------------------------
        # 2) run model compute
        # ---------------------------------------------------------
        T_map, snap = impl(speed_model=speed_model, seed_from=seed_from, overrides=overrides)
    
        # ---------------------------------------------------------
        # 3) cache snap and grids so downstream (Bayes update etc) never breaks
        # ---------------------------------------------------------
        try:
            self._wilber_last_snap = snap
    
            # prefer the instance canonical grids (these exist in your codebase)
            lon_g = getattr(self, "lon_grid", None)
            lat_g = getattr(self, "lat_grid", None)
    
            if lon_g is not None and lat_g is not None:
                self._wilber_last_lon_grid = np.asarray(lon_g, dtype=float)
                self._wilber_last_lat_grid = np.asarray(lat_g, dtype=float)
    
            # also try to cache from snap if present (optional)
            if isinstance(snap, dict):
                for k_lon, k_lat in (
                    ("lon_grid", "lat_grid"),
                    ("lon", "lat"),
                    ("LON", "LAT"),
                    ("grid_lon", "grid_lat"),
                ):
                    if k_lon in snap and k_lat in snap:
                        self._wilber_last_lon_grid = np.asarray(snap[k_lon], dtype=float)
                        self._wilber_last_lat_grid = np.asarray(snap[k_lat], dtype=float)
                        break
        except Exception:
            # never break compute because caching failed
            pass
    
        return T_map, snap
    
    
    def _wilber__grid_lonlat_from_snap(self, snap=None, lon_grid=None, lat_grid=None):
        """
        Resolve the canonical (lon_grid, lat_grid) for a T_map.
    
        Priority:
          1) explicit lon_grid/lat_grid args (if both provided)
          2) snap (if provided and contains grids)
          3) cached last-grid from self._wilber_last_lon_grid/_wilber_last_lat_grid
          4) instance grids self.lon_grid/self.lat_grid (loaded from XML)
          5) cached last snap if it contains grids
    
        This prevents pipeline breaks when 'snap' is accidentally not passed around.
        """
        import numpy as np
    
        # 1) explicit grids override everything
        if lon_grid is not None and lat_grid is not None:
            return np.asarray(lon_grid, dtype=float), np.asarray(lat_grid, dtype=float)
    
        # 2) try snap if provided
        if snap is not None:
            if isinstance(snap, dict):
                for k_lon, k_lat in (
                    ("lon_grid", "lat_grid"),
                    ("lon", "lat"),
                    ("LON", "LAT"),
                    ("grid_lon", "grid_lat"),
                ):
                    if k_lon in snap and k_lat in snap:
                        return np.asarray(snap[k_lon], dtype=float), np.asarray(snap[k_lat], dtype=float)
    
                g = snap.get("grid", None)
                if isinstance(g, dict) and ("lon" in g and "lat" in g):
                    return np.asarray(g["lon"], dtype=float), np.asarray(g["lat"], dtype=float)
    
        # 3) cached last-grid
        lon_c = getattr(self, "_wilber_last_lon_grid", None)
        lat_c = getattr(self, "_wilber_last_lat_grid", None)
        if lon_c is not None and lat_c is not None:
            return np.asarray(lon_c, dtype=float), np.asarray(lat_c, dtype=float)
    
        # 4) instance canonical grids (preferred)
        lon_g = getattr(self, "lon_grid", None)
        lat_g = getattr(self, "lat_grid", None)
        if lon_g is not None and lat_g is not None:
            return np.asarray(lon_g, dtype=float), np.asarray(lat_g, dtype=float)
    
        # 5) cached last snap if it has grids
        snap_c = getattr(self, "_wilber_last_snap", None)
        if snap_c is not None and isinstance(snap_c, dict):
            for k_lon, k_lat in (
                ("lon_grid", "lat_grid"),
                ("lon", "lat"),
                ("LON", "LAT"),
                ("grid_lon", "grid_lat"),
            ):
                if k_lon in snap_c and k_lat in snap_c:
                    return np.asarray(snap_c[k_lon], dtype=float), np.asarray(snap_c[k_lat], dtype=float)
    
            g = snap_c.get("grid", None)
            if isinstance(g, dict) and ("lon" in g and "lat" in g):
                return np.asarray(g["lon"], dtype=float), np.asarray(g["lat"], dtype=float)
    
        raise ValueError("Need snap (from wilber_compute_T_map_for_model) or lon_grid/lat_grid, and no cached/instance grid was found.")
    
    
    def wilber_update_T_map_bayes_kernel(
        self,
        arrivals_df,
        T_prior,
        *,
        snap=None,
        lon_grid=None,
        lat_grid=None,
        use_only_picked=True,
        exclude_flagged=True,
        flagged_col="outlier_flag",
        interp_method="nearest",
    
        # Bayesian / kernel knobs
        radius_km=50.0,
        kernel="gaussian",          # "gaussian" | "inv" | "inv2"
        prior_strength=1.0,         # larger -> less correction (stronger prior)
        min_stations=8,
        min_weight_sum=1e-6,
        residual_clip_s=None,       # e.g., 300.0 to prevent crazy corrections
        return_fields=False,
        log=True,
    ):
        """
        Bayesian-style residual update:
          T_post(x) = T_prior(x) + R_hat(x)
    
        PATCH FIX:
          - snap is OPTIONAL now. If snap=None, we fall back to:
              cached last-grid OR instance lon_grid/lat_grid OR explicit lon/lat.
        """
        import numpy as np
    
        # ------------------------------------------------------------
        # 0) select stations and compute residuals r_i = t_obs - t_pred(prior)
        # ------------------------------------------------------------
        df = arrivals_df.copy()
    
        if use_only_picked:
            df = df[df["picked"] == True].copy()
    
        if exclude_flagged and (flagged_col in df.columns):
            df = df[df[flagged_col].astype(bool) == False].copy()
    
        for c in ("lon", "lat", "t_obs_s"):
            if c not in df.columns:
                raise ValueError(f"arrivals_df missing required column '{c}'")
    
        df = df[np.isfinite(df["lon"].astype(float)) & np.isfinite(df["lat"].astype(float))].copy()
        df = df[np.isfinite(df["t_obs_s"].astype(float))].copy()
    
        if len(df) < int(min_stations):
            if log:
                self._wilber__log(f"[wilber-bayes] not enough stations for update: N={len(df)} < {int(min_stations)} -> returning prior.")
            meta = {"n_used": int(len(df)), "status": "not_enough_stations"}
            return (T_prior.copy(), meta) if not return_fields else (T_prior.copy(), meta, None, None)
    
        # station predicted times from prior
        t_pred_i = self._wilber__interp_T_at(
            T_prior,
            df["lon"].to_numpy(float),
            df["lat"].to_numpy(float),
            method=str(interp_method),
        )
        r_i = df["t_obs_s"].to_numpy(float) - np.asarray(t_pred_i, dtype=float)
    
        if residual_clip_s is not None and np.isfinite(float(residual_clip_s)):
            clip = float(residual_clip_s)
            r_i = np.clip(r_i, -clip, +clip)
    
        # ------------------------------------------------------------
        # 1) get lon/lat grids (snap OR explicit OR cached OR instance)
        # ------------------------------------------------------------
        LON, LAT = self._wilber__grid_lonlat_from_snap(snap=snap, lon_grid=lon_grid, lat_grid=lat_grid)
        LON = np.asarray(LON, dtype=float)
        LAT = np.asarray(LAT, dtype=float)
    
        # allow both meshgrid-like (2D) or vectors (1D)
        if LON.ndim == 1 and LAT.ndim == 1:
            LON2, LAT2 = np.meshgrid(LON, LAT)
        else:
            LON2, LAT2 = LON, LAT
    
        # ------------------------------------------------------------
        # 2) kernel correction field
        # ------------------------------------------------------------
        km_per_deg = 111.0
        lat0 = float(np.nanmean(df["lat"].to_numpy(float)))
        coslat = float(np.cos(np.deg2rad(lat0)))
        coslat = max(0.1, min(1.0, coslat))
    
        sx = df["lon"].to_numpy(float)
        sy = df["lat"].to_numpy(float)
    
        gx = LON2.ravel()
        gy = LAT2.ravel()
    
        dx = (gx[:, None] - sx[None, :]) * (km_per_deg * coslat)
        dy = (gy[:, None] - sy[None, :]) * km_per_deg
        d2 = dx * dx + dy * dy
    
        R = max(1e-3, float(radius_km))
        ker = str(kernel).lower().strip()
    
        if ker == "gaussian":
            w = np.exp(-0.5 * d2 / (R * R))
        elif ker == "inv":
            w = 1.0 / (np.sqrt(d2) + 1e-6)
        elif ker == "inv2":
            w = 1.0 / (d2 + 1e-6)
        else:
            w = np.exp(-0.5 * d2 / (R * R))
    
        wsum = np.sum(w, axis=1)
    
        alpha = max(0.0, float(prior_strength))
        denom = (wsum + alpha)
    
        good = denom > float(min_weight_sum)
        corr = np.zeros_like(denom, dtype=float)
        if np.any(good):
            corr[good] = (w[good, :] @ r_i) / denom[good]
    
        Corr2D = corr.reshape(LON2.shape)
        T_post = np.asarray(T_prior, dtype=float) + Corr2D
    
        meta = {
            "n_used": int(len(df)),
            "kernel": ker,
            "radius_km": float(R),
            "prior_strength": float(alpha),
            "residual_clip_s": float(residual_clip_s) if residual_clip_s is not None else None,
            "status": "ok",
        }
    
        if log:
            self._wilber__log(f"[wilber-bayes] update ok: N={meta['n_used']} kernel={ker} R={R:.1f}km prior_strength={alpha:.2f}")
    
        if return_fields:
            Wsum2D = wsum.reshape(LON2.shape)
            return T_post, meta, Corr2D, Wsum2D
    
        return T_post, meta





    # =============================================================================
    # PATCH: LOCAL spatial outlier filter
    #
    # Adds supported filter_mode strings:
    #   "off" | "soft" | "hard" | "local" | "soft+local" | "both" | "all"
    #
    # =============================================================================
    
    
    # ---------------------------------------------------------------------------
    # LOCAL spatial outlier filter
    # ---------------------------------------------------------------------------
    def _wilber__station_local_outlier_filter(
        self,
        arrivals_df,
        *,
        use_only_picked=True,
        value_col="t_obs_s",          # local consistency on observed pick time
        radius_km=50.0,               # neighborhood radius
        k_min=5,                      # minimum neighbors required
        z_mad=4.0,                    # robust threshold
        fallback="use_k_nearest",     # "skip" | "use_k_nearest"
        k_nearest=8,                  # used if fallback=="use_k_nearest"
        hard_invalidate=False,        # if True -> unpick station; else flag-only
        max_flag_frac=0.35,           # safety: don't let local flag dominate
        eps=1e-9,
        log=True,
    ):
        """
        Local spatial outlier filter (NO fitting):
        flags stations whose value deviates strongly from nearby stations.
    
        Adds/updates columns on arrivals_df:
          - local_outlier_flag (bool)
          - local_outlier_score (float)  # robust z
          - local_n_neighbors (int)
          - local_median (float)
          - local_mad (float)
    
        NOTE: uses lon/lat in arrivals_df; expects columns: lon, lat, and value_col.
        """
        import numpy as np
        import pandas as pd
    
        df = arrivals_df.copy()
    
        # ensure cols exist
        for c in ["local_outlier_flag", "local_outlier_score", "local_n_neighbors", "local_median", "local_mad"]:
            if c not in df.columns:
                df[c] = (False if c == "local_outlier_flag" else np.nan)
        if "notes" not in df.columns:
            df["notes"] = ""
    
        if "lon" not in df.columns or "lat" not in df.columns:
            if log:
                self._wilber__log("[wilber][local] arrivals_df missing lon/lat -> skipping local filter.")
            return df, {"error": "missing_lon_lat"}
    
        # subset used for neighborhood computations
        if use_only_picked and "picked" in df.columns:
            use = df[df["picked"] == True].copy()
        else:
            use = df.copy()
    
        # finite values only
        v = pd.to_numeric(use.get(value_col, np.nan), errors="coerce").to_numpy(float)
        lon = pd.to_numeric(use["lon"], errors="coerce").to_numpy(float)
        lat = pd.to_numeric(use["lat"], errors="coerce").to_numpy(float)
    
        ok = np.isfinite(v) & np.isfinite(lon) & np.isfinite(lat)
        use = use.loc[ok].copy()
        v = v[ok]
        lon = lon[ok]
        lat = lat[ok]
    
        n = int(len(use))
        if n < max(3, int(k_min) + 1):
            info = {"n_used": n, "n_flagged": 0, "reason": "too_few_stations"}
            if log:
                self._wilber__log(f"[wilber][local] skip: n_used={n} < {max(3, int(k_min)+1)}")
            return df, info
    
        # distance matrix (fast enough for typical Wilber station counts)
        # equirectangular approximation -> km
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        lat0 = float(np.nanmean(lat_rad))
        x = (lon_rad - lon_rad[:, None]) * np.cos(lat0)
        y = (lat_rad - lat_rad[:, None])
        dist_km = 6371.0 * np.sqrt(x * x + y * y)  # [n,n]
        np.fill_diagonal(dist_km, np.inf)
    
        # compute local robust z per station
        z = np.full(n, np.nan, dtype=float)
        nbr_n = np.zeros(n, dtype=int)
        lmed = np.full(n, np.nan, dtype=float)
        lmad = np.full(n, np.nan, dtype=float)
    
        R = float(radius_km)
        kmin = int(k_min)
        knear = int(k_nearest)
    
        for i in range(n):
            idx = np.where(dist_km[i, :] <= R)[0]
    
            # fallback if not enough neighbors
            if idx.size < kmin:
                if str(fallback).lower().strip() == "use_k_nearest":
                    # choose k nearest (excluding itself already inf)
                    order = np.argsort(dist_km[i, :])
                    idx = order[: max(kmin, knear)]
                    idx = idx[np.isfinite(dist_km[i, idx])]
                else:
                    idx = np.array([], dtype=int)
    
            nbr_n[i] = int(idx.size)
            if idx.size < kmin:
                continue
    
            vv = v[idx]
            m = float(np.nanmedian(vv))
            mad = float(np.nanmedian(np.abs(vv - m)))
            s = 1.4826 * mad
            s = max(s, float(eps))
    
            lmed[i] = m
            lmad[i] = s
            z[i] = abs(float(v[i]) - m) / s
    
        # flagging
        zthr = float(z_mad)
        flag = np.isfinite(z) & (z > zthr)
    
        # safety: avoid pathological over-flagging (e.g., if radius too small)
        max_frac = float(max_flag_frac)
        if max_frac > 0 and np.isfinite(max_frac):
            if flag.sum() / max(1, n) > max_frac:
                # keep only the worst offenders
                worst = np.argsort(np.where(np.isfinite(z), z, -np.inf))[::-1]
                keep_n = int(np.floor(max_frac * n))
                keep_n = max(0, keep_n)
                new_flag = np.zeros(n, dtype=bool)
                if keep_n > 0:
                    new_flag[worst[:keep_n]] = True
                flag = new_flag
    
        # write back into full df
        use_idx = use.index.to_numpy()
    
        df.loc[use_idx, "local_outlier_flag"] = flag
        df.loc[use_idx, "local_outlier_score"] = z
        df.loc[use_idx, "local_n_neighbors"] = nbr_n
        df.loc[use_idx, "local_median"] = lmed
        df.loc[use_idx, "local_mad"] = lmad
    
        if bool(hard_invalidate):
            # invalidate locally-flagged picks
            bad_idx = use_idx[flag]
            if len(bad_idx):
                df.loc[bad_idx, "picked"] = False
                df.loc[bad_idx, value_col] = np.nan
                df.loc[bad_idx, "notes"] = df.loc[bad_idx, "notes"].astype(str).str.strip() + "|LOCAL_OUTLIER_INVALIDATED"
    
        info = {
            "n_used": int(n),
            "n_flagged": int(flag.sum()),
            "radius_km": float(R),
            "k_min": int(kmin),
            "z_mad": float(zthr),
            "fallback": str(fallback),
            "k_nearest": int(knear),
            "hard_invalidate": bool(hard_invalidate),
            "max_flag_frac": float(max_frac),
            "value_col": str(value_col),
        }
    
        if log:
            self._wilber__log(
                f"[wilber][local] used={info['n_used']} flagged={info['n_flagged']} "
                f"R={info['radius_km']}km kmin={info['k_min']} z>{info['z_mad']}"
            )
    
        return df, info
    
    
