"""
    SHAKEuq — ShakeMap/uncertainty dataset builder + observation adapter (+ synthetic test mode).

    PURPOSE
    -------
    SHAKEuq builds a *version-indexed* in-memory dataset for an event (or a synthetic simulation)
    and also produces a *unified, version-stacked grid* (all versions remapped onto a single
    reference grid). This object is used as the "data assembly + auditing" layer before running
    downstream UQ/assimilation methods (kriging/bayes/etc.). The core guarantee is that:
      - each ShakeMap version stays isolated in uq_state["versions"][vkey]
      - unified stacks preserve version ordering via uq_state["unified"]["version_keys"]
      - observation tables returned by build_observations() are IMT-isolated (no CDI leakage into PGA)

    DATA MODEL OVERVIEW
    -------------------
    SHAKEuq stores everything in self.uq_state (dict). The main top-level keys are:

      uq_state["config"]   : run configuration + defaults used
      uq_state["versions"] : per-version products, indexed by version key vkey (e.g., "001")
      uq_state["unified"]  : version-stacked grids remapped to a reference grid
      uq_state["sanity"]   : per-version audit table (pandas DataFrame)
      uq_state["truth"]    : (synthetic mode only) ground-truth metadata used to generate fields

    VERSION KEYS
    ------------
    Version keys are normalized to 3-digit strings using _norm_version():
      "1" -> "001", 1 -> "001", "014" -> "014"
    The canonical list is:
      self.version_list : List[str] of vkeys in the order you requested/built.

    FILE-BACKED DATASET BUILD (REAL USGS PRODUCTS)
    ----------------------------------------------
    Calling uq_build_dataset() (file-backed mode) reads, for each vkey:
      - grid.xml           -> ShakeMap mean fields + lon/lat grid
      - uncertainty.xml    -> uncertainty fields (typically STD* layers)
      - stationlist.json   -> instruments + DYFI-in-stationlist dataframes (via USGSParser)
      - rupture.json       -> rupture geometry + metadata (optional)
      - CDI file           -> DYFI geocoded intensity table (optional, gated by version)

    CDI GATING (IMPORTANT)
    ----------------------
    CDI is resolved ONCE from self.dyfi_cdi_input and then applied per version only when:
      int(vkey) >= include_cdi_from_version
    This is recorded in:
      uq_state["versions"][vkey]["obs_audit"]["use_cdi"]
      uq_state["sanity"]["use_cdi"], ["cdi_loaded"], ["n_cdi"], ["cdi_note"], ["cdi_path"]

    PER-VERSION PACK STRUCTURE
    --------------------------
    Each version pack lives at:
      uq_state["versions"][vkey] : Dict[str, Any] with keys:

      ["meta"]    : dict of metadata parsed from XML roots (e.g. code_version, timestamps, etc.)
      ["grid"]    : dict or None
      ["uncert"]  : dict or None
      ["stations"]: dict of observation sources (dataframes)
      ["cdi"]     : dict of CDI table + notes
      ["rupture"] : dict holding raw rupture JSON + meta
      ["obs_audit"]: counts + boolean flags

    1) uq_state["versions"][vkey]["grid"]  (ShakeMap mean fields)
       ---------------------------------------------------------
       grid is a dict with:
         "lon2d" : 2D ndarray [nlat, nlon] of longitudes
         "lat2d" : 2D ndarray [nlat, nlon] of latitudes
         "fields": Dict[str, 2D ndarray] of mean layers, e.g.
                   "MMI", "PGA", "PGV", "PSA03", "PSA06", "PSA10", "PSA30", "SVEL", ...
         "orientation": orientation info used to align grids consistently (transpose/flips)

       NOTE ON UNITS (MEAN FIELDS)
       ---------------------------
       Mean fields come directly from grid.xml and are NOT auto-normalized. Example:
         - PGA in grid.xml may appear in "%g" (percent of g)
       The stationlist instruments dataframe may include a "pga_unit" column (often "%g").

    2) uq_state["versions"][vkey]["uncert"]  (uncertainty.xml fields)
       --------------------------------------------------------------
       uncert is a dict with:
         "fields": Dict[str, 2D ndarray] of uncertainty layers.
                  This code typically keeps only STD* fields at unified-stage.
                  Examples:
                    "STDMMI" (MMI sigma, intensity units)
                    "STDPGA" (PGA sigma, typically ln-space per ShakeMap convention)
                    "STDPSA03", "STDPSA10", etc.

       IMPORTANT: Uncertainty fields are treated as already in the convention of the file.
       This class does not convert between ln/linear spaces; downstream methods must
       respect the sigma meaning for each IMT.

    3) uq_state["versions"][vkey]["stations"]  (stationlist.json tables)
       -----------------------------------------------------------------
       stations is a dict:
         "instruments"      : DataFrame or None
         "dyfi_stationlist" : DataFrame or None
         "debug"            : dict of parser notes

       instruments DataFrame (seismic stations) typically includes columns like:
         longitude, latitude, pga, pgv, intensity, distance, vs30, elev, ...
         and sometimes "pga_unit" (e.g., "%g").

       dyfi_stationlist DataFrame (DYFI points embedded in stationlist.json) typically includes:
         longitude, latitude, intensity (MMI), nresp, intensity_stddev, distance, ...

       NOTE: This class intentionally preserves all incoming columns; it only standardizes
       lon/lat/value/sigma when building observations for downstream methods.

    4) uq_state["versions"][vkey]["cdi"]  (explicit CDI file table, geocoded)
       ----------------------------------------------------------------------
       cdi is a dict:
         "df"        : CDI DataFrame or None
         "cdi_loaded": bool
         "cdi_note"  : gating/resolution notes (explicit path, gate off, etc.)
         "cdi_path"  : resolved file path (if available)
         "debug"     : parser notes

       Typical CDI df columns (from parser) include:
         "Latitude", "Longitude", "CDI", "No. of responses", "Hypocentral distance",
         "Standard deviation", "Suspect?", plus possible city/state columns.

    5) uq_state["versions"][vkey]["rupture"]  (rupture.json)
       -----------------------------------------------------
       rupture is a dict:
         "data" : raw JSON dict (GeoJSON-like)
         "meta" : quick metadata summary (type, geom_type, coords_depth, etc.)
         "debug": read errors/notes

    6) uq_state["versions"][vkey]["obs_audit"]
       ---------------------------------------
       Convenience summary for quick audits:
         n_instruments, n_dyfi_stationlist, n_cdi,
         use_cdi, cdi_loaded, rupture_loaded

    UNIFIED STACKED GRIDS (ALL VERSIONS ON ONE GRID)
    ------------------------------------------------
    After building per-version packs, SHAKEuq builds a unified grid stack:
      uq_state["unified"] : dict with keys:

        "lon2d"        : reference lon2d (2D ndarray)
        "lat2d"        : reference lat2d (2D ndarray)
        "fields"       : Dict[str, 3D ndarray] shape (nver, nlat_ref, nlon_ref)
        "sigma"        : Dict[str, 3D ndarray] shape (nver, nlat_ref, nlon_ref)
        "version_keys" : List[str] version order used in stacks
        "ref_version"  : version key chosen as reference grid
        "ref_shape"    : (nlat_ref, nlon_ref)
        "note"         : remap note

    REMAP BEHAVIOR
    --------------
    If a version’s native grid shape != reference grid shape, fields are remapped using
    nearest-neighbor remap (no SciPy dependency). The remap uses the version’s lon2d/lat2d
    and targets the reference lon2d/lat2d.

    OBSERVATION ADAPTER (STANDARDIZED TABLE)
    ----------------------------------------
    build_observations(version, imt, dyfi_source, sigma_override) returns a standardized
    DataFrame with columns:

      lon, lat, value, sigma,
      source_type, source_detail,
      station_id,
      version, imt, tae_hours

    IMT ROUTING / ISOLATION (NO MIXING)
    -----------------------------------
      - If imt in ("PGA", "PGV", "MMI_SEISMIC"): uses stations["instruments"] only (source_detail="station")
      - If imt == "MMI": uses DYFI stationlist and/or CDI depending on dyfi_source:
          dyfi_source="stationlist" -> dyfi_stationlist only
          dyfi_source="cdi"         -> CDI only
          dyfi_source="both"        -> both sources concatenated
          dyfi_source="auto"        -> treated as allowed for both (subject to existence)

    MINIMAL FILTERING
    -----------------
    Observations are minimally filtered (no over-filtering):
      - only rows missing lon/lat/value are dropped
      - existing CDI gates may apply:
          cdi_max_dist_km (Hypocentral distance)
          cdi_min_nresp   (No. of responses)
      - optional verbose log prints number dropped per source

    NOTE ON COORD EXTENT
    --------------------
    Some observations may fall slightly outside the unified grid extent (e.g. ~3%),
    which is not necessarily an error (edge rounding, different product extents).
    For downstream gridded methods that require strict coverage, use the extent helpers.

    RAW EXTRACTOR (ShakeMap-only, NOT CDI)
    -------------------------------------
    extract_raw_shakemap(imt) returns:
      {
        "imt": "MMI"/"PGA"/...,
        "versions": {
           vkey: {
             "mean_grid": 2D ndarray or None,
             "sigma_grid": 2D ndarray or None,
             "n_grid_points": int,
             "available": {... counts ...},
             "used": {"mean": bool, "sigma": bool}
           }, ...
        },
        "summary": DataFrame with one row per version,
        "log": list[str]
      }

    IMPORTANT: extract_raw_shakemap is ShakeMap-grid focused; it reports CDI counts for audit,
    but it does not *use* CDI in the grid extraction (mean/sigma come from XML grids).

    SYNTHETIC MODE (IN-MEMORY TEST PRODUCTS)
    ---------------------------------------
    Synthetic mode exists to validate pipelines without real USGS downloads.
    Core entry points:
      - build_synthetic_case(...) -> constructs synthetic per-version packs + enables synthetic mode
      - enable_synthetic_mode(synthetic_store) / disable_synthetic_mode()
      - uq_build_dataset() dispatches automatically:
          synthetic enabled -> _uq_build_dataset_synthetic()
          else             -> _uq_build_dataset_filebacked()

    Synthetic products match the same uq_state schema so smoke tests and downstream logic can be
    verified consistently.

    AVAILABLE HELPERS / UTILITIES (FOR AGENTS + DEVELOPERS)
    -------------------------------------------------------
    Discovery / parsing:
      - debug_discovery(version), debug_discovery_all()
      - resolve_cdi_path(cdi_input)  (explicit-only CDI resolution)
      - parse_grid_spec_from_shakemap_xml_snippet(xml_text)  [@staticmethod]
      - internal XML readers: _read_shakemap_grid_xml(), _read_uncertainty_xml()
      - internal remap: _nn_remap_to_ref()

    Observation extent tools (recommended for downstream methods):
      - _get_grid_extent(version, grid_mode="unified"|"native", margin_deg=0.0)
      - _mask_points_in_extent(lon, lat, extent)
      - audit_observations_extent(obs, version, grid_mode="unified", margin_deg=0.0, groupby="source_detail")
      - filter_observations_to_extent(obs, version, grid_mode="unified", margin_deg=0.0, return_dropped=False)
      - log_observation_extent_audit(obs, version, imt, grid_mode="unified", margin_deg=0.0, ...)

    Plotting / audits:
      - plot_shakemap_raw_audit(...)  -> 2×2 map audit (MMI/PGA mean + STDMMI/STDPGA) + counts plot

    ULTIMATE GOAL (CONTEXT)
    -----------------------
    SHAKEuq is the staging layer for time-evolving SHAKEmaps uncertainty quantification:
      1) build consistent multi-version grids (mean + sigma) and observations
      2) run UQ/assimilation methods per version and compare evolution over time (TAE_hours)
      3) ensure strict data provenance (no IMT mixing, no sigma/mean mixing, preserved metadata)


    PLANNED METHODS / ROADMAP (FOR FUTURE IMPLEMENTATION)
    ----------------------------------------------------
    
    The datasets assembled by SHAKEuq are designed to support a sequence of
    progressively more sophisticated methods. The following roadmap documents
    the intended use of the data structures and helpers defined here.
    
    1) Ordinary Kriging / Deterministic Interpolation
       ------------------------------------------------
       A baseline interpolation layer will be implemented using ordinary kriging
       (or equivalent spatial interpolation) on the unified grid.
    
       Planned operating modes include:
         - PGA update using instrumented seismic stations only
         - MMI update using DYFI stationlist points only
         - MMI update using CDI geocoded DYFI data (with additional CDI-specific filters)
         - Mixed modes, e.g.:
             * converting PGA → MMI via GMICE and updating the MMI grid
             * converting MMI → PGA via inverse GMICE and updating the PGA grid
         - Additional modes may be added later as required.
    
       Each kriging mode will:
         - use build_observations() + extent helpers for clean input
         - operate explicitly on the unified grid (unless otherwise specified)
         - produce a new mean field and an associated interpolation variance
         - include a lightweight audit plotting helper similar in spirit to
           plot_shakemap_raw_audit(), to visually diagnose results.
    
    2) Bayesian Posterior Update (Bayes-1lik / Bayes-2lik)
       ----------------------------------------------------
       After deterministic baselines, Bayesian updating will be applied.
    
       The design intent is a single Bayesian update interface that can operate as:
         - Bayes-1lik: one combined likelihood using all selected observations
         - Bayes-2lik (or hierarchical): separate likelihoods for
             * instrumented seismic data
             * macroseismic data (DYFI stationlist and/or CDI)
    
       The same function will support different combinations of data sources,
       allowing direct comparison of how each source influences the posterior
       mean and uncertainty.
    
       Inputs:
         - Prior: ShakeMap mean + ShakeMap sigma (from unified grids)
         - Likelihood(s): observations from build_observations()
       Outputs:
         - Posterior mean grid
         - Posterior uncertainty grid
    
    3) Observation Weighting / Conditioning Layer
       -------------------------------------------
       A dedicated conditioning layer may be implemented to rank, weight, or filter
       observations before interpolation or Bayesian updating.
    
       Examples:
         - CDI-only MMI updates where CDI points are ranked by:
             * number of responses
             * distance to epicenter
             * reported uncertainty
         - Dynamic weighting between instruments and DYFI
         - Sensitivity experiments using subsets of observations
    
       This layer will remain separate from build_observations() to avoid hidden
       data loss or implicit assumptions.
    
    4) Time-Evolution Analysis (Optional / Advanced)
       ----------------------------------------------
       Leveraging:
         - version ordering
         - TAE_hours
         - unified stacked grids
    
       Future analyses may include:
         - pixel-wise evolution of mean and uncertainty through time
         - stabilization/decay curves of uncertainty
         - comparison of information gain between successive versions
    
       This supports the central thesis that SHAKEmaps should be treated as
       time-dependent uncertain fields rather than static final products.
    
    NOTE
    ----
    SHAKEuq itself does NOT implement these methods; it provides the data integrity,
    auditability, and helper utilities required so that future implementations can
    focus on scientific logic without re-handling parsing, alignment, or provenance.

      
"""









from __future__ import annotations

import os
import re
import glob
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET




def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _norm_version(v: Union[int, str]) -> str:
    s = str(v).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return s.zfill(3)
    return f"{int(m.group(1)):03d}"


def _parse_iso_utc(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s or not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    ts = pd.to_datetime(t, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _guess_time_after_event_hours(
    event_time_utc: Optional[pd.Timestamp],
    process_time_utc: Optional[pd.Timestamp],
) -> Optional[float]:
    if event_time_utc is None or process_time_utc is None:
        return None
    try:
        return float((process_time_utc - event_time_utc).total_seconds() / 3600.0)
    except Exception:
        return None


def _coerce_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


class SHAKEuq:
    """SHAKEuq — dataset builder + CDI resolver + rupture loader + observation adapter."""

    def __init__(
        self,
        event_id: str,
        event_time: Optional[Union[str, pd.Timestamp]] = None,
        shakemap_folder: Optional[str] = None,
        stations_folder: Optional[str] = None,
        rupture_folder: Optional[str] = None,
        dyfi_cdi_file: Optional[str] = None,  # file OR folder OR root
        version_list: Optional[List[Union[int, str]]] = None,
        # grid-uncertainty defaults (used by synthetic Mode A and can be used later for baseline fallbacks)
        sigma_grid_mmi: float = 0.3,      # STDMMI (intensity units)
        sigma_grid_lnpga: float = 0.7,    # STDPGA (ln(g)) — typical ln sigma scale

        base_folder: str = "./export/SHAKEuq",
        include_cdi_from_version: int = 2,
        prefer_sigma_field_prefix: str = "STD",
        # observation adapter defaults
        sigma_instr: float = 0.2,
        sigma_dyfi_stationlist: float = 0.5,
        sigma_cdi: float = 0.5,
        cdi_max_dist_km: Optional[float] = 400.0,
        cdi_min_nresp: Optional[int] = None,
        verbose: bool = True,
    ):
        self.event_id = str(event_id)
        self.verbose = bool(verbose)

        if isinstance(event_time, pd.Timestamp):
            self.event_time_utc = event_time.tz_convert("UTC") if event_time.tzinfo else event_time.tz_localize("UTC")
        elif isinstance(event_time, str):
            self.event_time_utc = _parse_iso_utc(event_time)
        else:
            self.event_time_utc = None

        self.include_cdi_from_version = int(include_cdi_from_version)
        self.prefer_sigma_field_prefix = str(prefer_sigma_field_prefix or "STD")

        self.sigma_instr = float(sigma_instr)
        self.sigma_dyfi_stationlist = float(sigma_dyfi_stationlist)
        self.sigma_cdi = float(sigma_cdi)
        self.cdi_max_dist_km = cdi_max_dist_km
        self.cdi_min_nresp = cdi_min_nresp

        self.base_folder = os.path.abspath(base_folder)
        self.out_folder = _ensure_dir(os.path.join(self.base_folder, self.event_id, "uq"))

        self.shakemap_folder = self._norm_event_folder(shakemap_folder)
        self.stations_folder = self._norm_event_folder(stations_folder)
        self.rupture_folder = self._norm_event_folder(rupture_folder)
        self.dyfi_cdi_input = os.path.abspath(dyfi_cdi_file) if isinstance(dyfi_cdi_file, str) else None



                # ----------------------------
        # sigma aliases for consistent internal naming
        # ----------------------------
        # Observation sigmas (your existing API)
        self.sigma_instr = float(sigma_instr)
        self.sigma_dyfi_stationlist = float(sigma_dyfi_stationlist)
        self.sigma_cdi = float(sigma_cdi)

        # Grid sigma defaults (NEW kwargs)
        self.sigma_grid_mmi = float(sigma_grid_mmi)
        self.sigma_grid_lnpga = float(sigma_grid_lnpga)

        # Backward/forward aliases used by synthetic utilities and future methods
        # (so code can refer to these consistently)
        self.sigma_mmi_default = self.sigma_grid_mmi
        self.sigma_pga_default = self.sigma_grid_lnpga

        self.sigma_mmi_instr = self.sigma_instr
        self.sigma_pga_instr = self.sigma_instr

        self.sigma_mmi_dyfi = self.sigma_dyfi_stationlist
        self.sigma_mmi_cdi = self.sigma_cdi


        self.version_list = [_norm_version(v) for v in (version_list or [])]

        self.uq_state: Dict[str, Any] = {
            "config": {
                "event_id": self.event_id,
                "event_time_utc": str(self.event_time_utc) if self.event_time_utc is not None else None,
                "shakemap_folder": self.shakemap_folder,
                "stations_folder": self.stations_folder,
                "rupture_folder": self.rupture_folder,
                "dyfi_cdi_input": self.dyfi_cdi_input,
                "version_list": list(self.version_list),
                "include_cdi_from_version": self.include_cdi_from_version,
                "prefer_sigma_field_prefix": self.prefer_sigma_field_prefix,
                "sigma_instr": self.sigma_instr,
                "sigma_dyfi_stationlist": self.sigma_dyfi_stationlist,
                "sigma_cdi": self.sigma_cdi,
                "cdi_max_dist_km": self.cdi_max_dist_km,
                "cdi_min_nresp": self.cdi_min_nresp,
                "out_folder": self.out_folder,
            },
            "versions": {},
            "unified": {},
            "sanity": None,
        }

        if self.verbose:
            print(f"[SHAKEuq] init event_id={self.event_id} versions={self.version_list}")
            print(f"[SHAKEuq] out_folder={self.out_folder}")

    # -------------------------
    # discovery
    # -------------------------
    def _norm_event_folder(self, folder: Optional[str]) -> Optional[str]:
        if not folder:
            return None
        p = os.path.abspath(folder)
        if os.path.isdir(p) and os.path.basename(p) != self.event_id:
            cand = os.path.join(p, self.event_id)
            if os.path.isdir(cand):
                return cand
        return p

    def _build_usgs_filename(self, vkey: str, kind: str, originator: str = "us") -> str:
        suf = {
            "grid": "grid.xml",
            "uncertainty": "uncertainty.xml",
            "stationlist": "stationlist.json",
            "rupture": "rupture.json",
        }.get(kind)
        if not suf:
            raise ValueError(f"Unknown kind: {kind}")
        return f"{self.event_id}_{originator}_{vkey}_{suf}"

    def _find_first_existing(self, candidates: List[str]) -> Optional[str]:
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def _discover_version_paths(self, vkey: str) -> Dict[str, Optional[str]]:
        out = {"grid_xml": None, "uncertainty_xml": None, "stationlist_json": None, "rupture_json": None}
        originator = "us"

        sm = self.shakemap_folder
        st = self.stations_folder
        rp = self.rupture_folder

        if sm and os.path.isdir(sm):
            out["grid_xml"] = self._find_first_existing([os.path.join(sm, self._build_usgs_filename(vkey, "grid", originator))])
            out["uncertainty_xml"] = self._find_first_existing([os.path.join(sm, self._build_usgs_filename(vkey, "uncertainty", originator))])
            if out["grid_xml"] is None:
                hits = sorted(glob.glob(os.path.join(sm, f"{self.event_id}_{originator}_{vkey}_*grid*.xml")))
                out["grid_xml"] = hits[0] if hits else None
            if out["uncertainty_xml"] is None:
                hits = sorted(glob.glob(os.path.join(sm, f"{self.event_id}_{originator}_{vkey}_*uncertainty*.xml")))
                out["uncertainty_xml"] = hits[0] if hits else None

        if st and os.path.isdir(st):
            out["stationlist_json"] = self._find_first_existing([os.path.join(st, self._build_usgs_filename(vkey, "stationlist", originator))])
            if out["stationlist_json"] is None:
                hits = sorted(glob.glob(os.path.join(st, f"{self.event_id}_{originator}_{vkey}_*stationlist*.json")))
                out["stationlist_json"] = hits[0] if hits else None

        if rp and os.path.isdir(rp):
            out["rupture_json"] = self._find_first_existing([os.path.join(rp, self._build_usgs_filename(vkey, "rupture", originator))])
            if out["rupture_json"] is None:
                hits = sorted(glob.glob(os.path.join(rp, f"{self.event_id}_{originator}_{vkey}_*rupture*.json")))
                out["rupture_json"] = hits[0] if hits else None

        return out

    def debug_discovery(self, version: Union[int, str]) -> Dict[str, Any]:
        vkey = _norm_version(version)
        paths = self._discover_version_paths(vkey)
        exists = {k: (p is not None and os.path.exists(p)) for k, p in paths.items()}
        out = {"version": vkey, "paths": paths, "exists": exists}
        if self.verbose:
            print(f"[SHAKEuq][debug_discovery] v={vkey} exists={exists}")
        return out

    def debug_discovery_all(self) -> pd.DataFrame:
        rows = []
        for v in self.version_list:
            d = self.debug_discovery(v)
            rows.append({"version": d["version"], **{k: bool(vv) for k, vv in d["exists"].items()}})
        df = pd.DataFrame(rows)
        if self.verbose:
            print(df)
        return df

    # -------------------------
    # CDI resolver
    # -------------------------
    def _search_cdi_in_dir(self, folder: str) -> Optional[str]:
        pats = ["**/*cdi*geo*.txt", "**/*cdi*geo*.csv", "**/*cdi*.txt", "**/*cdi*.csv"]
        hits: List[str] = []
        for pat in pats:
            hits += glob.glob(os.path.join(folder, pat), recursive=True)
        hits = [h for h in hits if os.path.isfile(h)]
        if not hits:
            return None

        def score(p: str) -> Tuple[int, int]:
            b = os.path.basename(p).lower()
            sc = 0
            if "geo" in b:
                sc += 10
            if "1km" in b:
                sc += 3
            if "_3_" in b or "v3" in b:
                sc += 2
            return (sc, len(b))

        hits = sorted(hits, key=lambda p: score(p), reverse=True)
        return hits[0]

    def _resolve_cdi_path(self, cdi_input: Optional[str]) -> Tuple[Optional[str], str]:
        if cdi_input:
            p = os.path.abspath(cdi_input)
            if os.path.isfile(p):
                return p, "cdi_input=file"
            if os.path.isdir(p):
                cand = self._search_cdi_in_dir(p)
                return cand, "cdi_input=dir" if cand else "cdi_input=dir_no_match"
            return None, "cdi_input=missing"

        # auto: try common SHAKEfetch layout near stations/shakemap/rupture roots
        roots: List[str] = []
        for base in [self.stations_folder, self.shakemap_folder, self.rupture_folder]:
            if base:
                roots.append(os.path.dirname(base))
        roots = [r for r in roots if r and os.path.isdir(r)]
        for r in roots:
            cand_dir = os.path.join(r, "usgs-dyfi-versions", self.event_id)
            if os.path.isdir(cand_dir):
                cand = self._search_cdi_in_dir(cand_dir)
                if cand:
                    return cand, "cdi_auto=usgs-dyfi-versions"
        return None, "cdi_auto=none"

    # -------------------------
    # USGS parser adapters
    # -------------------------
    def _get_usgs_parser(self):
        for mod in ("SHAKEparser", "modules.SHAKEparser", ".SHAKEparser"):
            try:
                USGSParser = __import__(mod, fromlist=["USGSParser"]).USGSParser
                return USGSParser
            except Exception:
                continue
        raise ImportError("USGSParser could not be imported. Ensure SHAKEparser is available on PYTHONPATH.")

    def _read_stationlist_json(self, json_path: str) -> Dict[str, Any]:
        USGSParser = self._get_usgs_parser()
        dbg = {"path": json_path, "notes": []}
        instruments_df = None
        dyfi_df = None

        try:
            p = USGSParser(parser_type="instrumented_data", json_file=json_path)
            instruments_df = p.get_dataframe(value_type="pga")
            if isinstance(instruments_df, pd.DataFrame):
                instruments_df = _coerce_numeric_cols(instruments_df, ["longitude", "latitude", "pga", "pgv", "distance"])
        except Exception as e:
            dbg["notes"].append(f"instruments parse failed: {repr(e)}")

        try:
            p = USGSParser(parser_type="instrumented_data", json_file=json_path)
            dyfi_df = p.get_dataframe(value_type="mmi")
            if isinstance(dyfi_df, pd.DataFrame):
                dyfi_df = _coerce_numeric_cols(dyfi_df, ["longitude", "latitude", "intensity", "distance", "nresp", "intensity_stddev"])
        except Exception as e:
            dbg["notes"].append(f"dyfi stationlist parse failed: {repr(e)}")

        return {"instruments": instruments_df, "dyfi_stationlist": dyfi_df, "debug": dbg}

    def _read_cdi_file(self, cdi_file_path: str) -> Dict[str, Any]:
        USGSParser = self._get_usgs_parser()
        dbg = {"path": cdi_file_path, "notes": []}
        df = None
        try:
            p = USGSParser(parser_type="dyfi_data", file_path=cdi_file_path)
            df = p.get_dataframe()
            if isinstance(df, pd.DataFrame):
                df = _coerce_numeric_cols(
                    df,
                    [
                        "Latitude",
                        "Longitude",
                        "CDI",
                        "No. of responses",
                        "Hypocentral distance",
                        "Standard deviation",
                        "lat",
                        "lon",
                        "cdi",
                        "nresp",
                        "dist",
                        "stddev",
                    ],
                )
        except Exception as e:
            dbg["notes"].append(f"cdi parse failed: {repr(e)}")
        return {"df": df, "debug": dbg}

    # -------------------------
    # rupture loader
    # -------------------------
    def _nest_depth(self, obj: Any, max_depth: int = 10) -> int:
        depth = 0
        cur = obj
        while depth < max_depth and isinstance(cur, list) and cur:
            depth += 1
            cur = cur[0]
        return depth

    def _read_rupture_json(self, rupture_path: str) -> Dict[str, Any]:
        dbg = {"path": rupture_path, "notes": []}
        data = None
        meta: Dict[str, Any] = {}
        try:
            with open(rupture_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                meta["type"] = data.get("type")
                feats = data.get("features") if isinstance(data.get("features"), list) else None
                if feats:
                    meta["n_features"] = len(feats)
                    g0 = feats[0].get("geometry") if isinstance(feats[0], dict) else None
                    if isinstance(g0, dict):
                        meta["geom_type"] = g0.get("type")
                        coords = g0.get("coordinates")
                        meta["coords_depth"] = self._nest_depth(coords)
                        meta["coords_len_top"] = len(coords) if isinstance(coords, list) else None
        except Exception as e:
            dbg["notes"].append(f"rupture read failed: {repr(e)}")

        return {"data": data, "meta": meta, "debug": dbg}

    # -------------------------
    # XML parsing helpers
    # -------------------------
    def _strip(self, tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    def _infer_orientation(self, lon2d: np.ndarray, lat2d: np.ndarray) -> Dict[str, bool]:
        orient = {"transpose": False, "fliplr": False, "flipud": False}
        if lon2d.ndim != 2 or lat2d.ndim != 2:
            return orient

        lon_d0 = np.nanmean(np.abs(np.diff(lon2d, axis=0))) if lon2d.shape[0] > 1 else 0.0
        lon_d1 = np.nanmean(np.abs(np.diff(lon2d, axis=1))) if lon2d.shape[1] > 1 else 0.0
        lat_d0 = np.nanmean(np.abs(np.diff(lat2d, axis=0))) if lat2d.shape[0] > 1 else 0.0
        lat_d1 = np.nanmean(np.abs(np.diff(lat2d, axis=1))) if lat2d.shape[1] > 1 else 0.0

        if (lon_d0 > lon_d1) and (lat_d1 > lat_d0):
            orient["transpose"] = True
            lon2d = lon2d.T
            lat2d = lat2d.T

        if lon2d.shape[1] > 1 and np.nanmean(lon2d[:, -1]) < np.nanmean(lon2d[:, 0]):
            orient["fliplr"] = True
            lon2d = np.fliplr(lon2d)
            lat2d = np.fliplr(lat2d)

        if lat2d.shape[0] > 1 and np.nanmean(lat2d[-1, :]) < np.nanmean(lat2d[0, :]):
            orient["flipud"] = True

        return orient

    def _apply_orientation_one(self, a: np.ndarray, orient: Dict[str, bool]) -> np.ndarray:
        out = a
        if orient.get("transpose"):
            out = out.T
        if orient.get("fliplr"):
            out = np.fliplr(out)
        if orient.get("flipud"):
            out = np.flipud(out)
        return out

    def _read_shakemap_grid_xml(self, grid_xml_path: str) -> Dict[str, Any]:
        tree = ET.parse(grid_xml_path)
        root = tree.getroot()

        meta = {self._strip(k): v for k, v in root.attrib.items()}
        event_ts = None
        process_ts = meta.get("process_timestamp")

        spec = {}
        field_defs_raw: List[Dict[str, Any]] = []
        data_text = None

        for ch in root:
            t = self._strip(ch.tag)
            if t == "event":
                for k, v in ch.attrib.items():
                    kk = self._strip(k)
                    if kk == "event_timestamp":
                        event_ts = v
                        meta["event_timestamp"] = v
                    else:
                        meta[kk] = v
            elif t == "grid_specification":
                spec = {self._strip(k): v for k, v in ch.attrib.items()}
            elif t == "grid_field":
                attrs = {self._strip(k): v for k, v in ch.attrib.items()}
                # keep raw; we will remap later
                field_defs_raw.append(attrs)
            elif t == "grid_data":
                data_text = ch.text

        nlat = int(float(spec.get("nlat", "0")))
        nlon = int(float(spec.get("nlon", "0")))
        npts = nlat * nlon

        if not data_text:
            raise ValueError(f"grid_data missing: {grid_xml_path}")

        rows = []
        for line in data_text.strip().splitlines():
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split()])
        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < max(1, npts):
            raise ValueError(f"grid_data parse/shape mismatch: {grid_xml_path}")

        flat = arr[:npts, :]
        ncols = int(flat.shape[1])

        # --- build index->name mapping with auto 1-based/0-based handling ---
        idx_name: List[Tuple[int, str]] = []
        raw_idxs: List[int] = []
        for attrs in field_defs_raw:
            try:
                raw = int(str(attrs.get("index", "")).strip())
            except Exception:
                continue
            name = str(attrs.get("name", "")).strip()
            if not name:
                continue
            raw_idxs.append(raw)
            idx_name.append((raw, name))

        if not idx_name:
            raise ValueError(f"No grid_field definitions found: {grid_xml_path}")

        # detect 1-based if:
        #  - indices are within 1..ncols and include ncols, OR min index == 1
        # detect 0-based if:
        #  - indices within 0..ncols-1 and include 0
        max_raw = max(raw_idxs)
        min_raw = min(raw_idxs)

        one_based = False
        if (min_raw == 1 and max_raw <= ncols) or (max_raw == ncols and min_raw >= 1):
            one_based = True

        def remap(i: int) -> int:
            return i - 1 if one_based else i

        # remap and keep only valid columns
        field_map: Dict[int, str] = {}
        for raw, name in idx_name:
            j = remap(raw)
            if 0 <= j < ncols:
                field_map[j] = name

        if not field_map:
            raise ValueError(f"grid_field indices invalid after remap (one_based={one_based}) in: {grid_xml_path}")

        # locate lon/lat columns
        lon_col, lat_col = None, None
        for j, name in field_map.items():
            u = name.upper()
            if u in ("LON", "LONGITUDE"):
                lon_col = j
            if u in ("LAT", "LATITUDE"):
                lat_col = j

        # fallback (common: lon=0 lat=1)
        if lon_col is None:
            lon_col = 0
        if lat_col is None:
            lat_col = 1

        lon2d = flat[:, lon_col].reshape((nlat, nlon))
        lat2d = flat[:, lat_col].reshape((nlat, nlon))
        orient = self._infer_orientation(lon2d, lat2d)

        lon2d = self._apply_orientation_one(lon2d, orient)
        lat2d = self._apply_orientation_one(lat2d, orient)

        fields: Dict[str, np.ndarray] = {}
        for j, name in field_map.items():
            if j in (lon_col, lat_col):
                continue
            vals = flat[:, j].reshape((nlat, nlon))
            vals = self._apply_orientation_one(vals, orient)
            fields[name] = vals

        grid = {
            "spec": {"nlat": nlat, "nlon": nlon, **spec, "ncols": ncols, "one_based_indexing": one_based},
            "lon2d": lon2d,
            "lat2d": lat2d,
            "fields": fields,
            "orientation": orient,
        }
        meta["process_timestamp"] = process_ts
        meta["event_timestamp"] = meta.get("event_timestamp", event_ts)
        return {"meta": meta, "grid": grid}

    def _read_uncertainty_xml(self, uncert_xml_path: str, orientation: Optional[Dict[str, bool]]) -> Dict[str, Any]:
        tree = ET.parse(uncert_xml_path)
        root = tree.getroot()

        meta = {self._strip(k): v for k, v in root.attrib.items()}
        spec = {}
        field_defs_raw: List[Dict[str, Any]] = []
        data_text = None

        for ch in root:
            t = self._strip(ch.tag)
            if t == "grid_specification":
                spec = {self._strip(k): v for k, v in ch.attrib.items()}
            elif t == "grid_field":
                attrs = {self._strip(k): v for k, v in ch.attrib.items()}
                field_defs_raw.append(attrs)
            elif t == "grid_data":
                data_text = ch.text

        nlat = int(float(spec.get("nlat", "0")))
        nlon = int(float(spec.get("nlon", "0")))
        npts = nlat * nlon

        if not data_text:
            raise ValueError(f"grid_data missing: {uncert_xml_path}")

        rows = []
        for line in data_text.strip().splitlines():
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split()])
        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < max(1, npts):
            raise ValueError(f"uncertainty grid_data parse/shape mismatch: {uncert_xml_path}")

        flat = arr[:npts, :]
        ncols = int(flat.shape[1])

        idx_name: List[Tuple[int, str]] = []
        raw_idxs: List[int] = []
        for attrs in field_defs_raw:
            try:
                raw = int(str(attrs.get("index", "")).strip())
            except Exception:
                continue
            name = str(attrs.get("name", "")).strip()
            if not name:
                continue
            raw_idxs.append(raw)
            idx_name.append((raw, name))

        if not idx_name:
            raise ValueError(f"No grid_field definitions found: {uncert_xml_path}")

        max_raw = max(raw_idxs)
        min_raw = min(raw_idxs)
        one_based = False
        if (min_raw == 1 and max_raw <= ncols) or (max_raw == ncols and min_raw >= 1):
            one_based = True

        def remap(i: int) -> int:
            return i - 1 if one_based else i

        field_map: Dict[int, str] = {}
        for raw, name in idx_name:
            j = remap(raw)
            if 0 <= j < ncols:
                field_map[j] = name

        if not field_map:
            raise ValueError(f"uncertainty grid_field indices invalid after remap (one_based={one_based}) in: {uncert_xml_path}")

        # identify lon/lat columns (to drop)
        lon_col = None
        lat_col = None
        for j, name in field_map.items():
            u = name.upper()
            if u in ("LON", "LONGITUDE"):
                lon_col = j
            if u in ("LAT", "LATITUDE"):
                lat_col = j

        fields: Dict[str, np.ndarray] = {}
        for j, name in field_map.items():
            if lon_col is not None and j == lon_col:
                continue
            if lat_col is not None and j == lat_col:
                continue
            vals = flat[:, j].reshape((nlat, nlon))
            if orientation:
                vals = self._apply_orientation_one(vals, orientation)
            fields[name] = vals

        uncert = {"spec": {"nlat": nlat, "nlon": nlon, **spec, "ncols": ncols, "one_based_indexing": one_based}, "fields": fields}
        return {"meta": meta, "uncert": uncert}

    def _nn_remap_to_ref(
        self,
        src_lon2d: np.ndarray,
        src_lat2d: np.ndarray,
        src_field2d: np.ndarray,
        ref_lon2d: np.ndarray,
        ref_lat2d: np.ndarray,
    ) -> np.ndarray:
        """
        Nearest-neighbor remap from (src_lon2d, src_lat2d) grid to reference (ref_lon2d, ref_lat2d).
        Works without SciPy. Assumes both are roughly rectilinear lon/lat grids (ShakeMap-style).
        """
        # build 1D axes from the 2D grids (typical ShakeMap: lon varies along axis=1, lat along axis=0)
        src_lon1 = src_lon2d[0, :]
        src_lat1 = src_lat2d[:, 0]

        # ensure increasing axes
        if len(src_lon1) > 1 and np.nanmean(np.diff(src_lon1)) < 0:
            src_lon1 = src_lon1[::-1]
            src_field2d = np.fliplr(src_field2d)
        if len(src_lat1) > 1 and np.nanmean(np.diff(src_lat1)) < 0:
            src_lat1 = src_lat1[::-1]
            src_field2d = np.flipud(src_field2d)

        ref_lon1 = ref_lon2d[0, :]
        ref_lat1 = ref_lat2d[:, 0]

        # nearest indices by searchsorted
        j = np.searchsorted(src_lon1, ref_lon1)
        j = np.clip(j, 0, len(src_lon1) - 1)
        # choose nearer neighbor
        j0 = np.clip(j - 1, 0, len(src_lon1) - 1)
        choose_left = np.abs(ref_lon1 - src_lon1[j0]) <= np.abs(ref_lon1 - src_lon1[j])
        j = np.where(choose_left, j0, j)

        i = np.searchsorted(src_lat1, ref_lat1)
        i = np.clip(i, 0, len(src_lat1) - 1)
        i0 = np.clip(i - 1, 0, len(src_lat1) - 1)
        choose_up = np.abs(ref_lat1 - src_lat1[i0]) <= np.abs(ref_lat1 - src_lat1[i])
        i = np.where(choose_up, i0, i)

        # gather by broadcasting
        out = src_field2d[np.ix_(i, j)]
        return out

    def _build_unified_grids(self) -> None:
        versions: Dict[str, Any] = self.uq_state.get("versions", {}) or {}
        if not versions:
            self.uq_state["unified"] = {}
            return

        vkeys = list(versions.keys())
        # pick first version that has a grid as reference
        ref_v = next((vk for vk in vkeys if versions[vk].get("grid") is not None), None)
        if ref_v is None:
            self.uq_state["unified"] = {}
            return

        ref_grid = versions[ref_v]["grid"]
        ref_lon2d = ref_grid.get("lon2d")
        ref_lat2d = ref_grid.get("lat2d")
        if ref_lon2d is None or ref_lat2d is None:
            self.uq_state["unified"] = {}
            return

        nlat_ref, nlon_ref = ref_lon2d.shape

        # collect field names
        mean_names = set()
        sig_names = set()
        for vk in vkeys:
            g = versions[vk].get("grid")
            u = versions[vk].get("uncert")
            if g and isinstance(g.get("fields"), dict):
                mean_names |= set(g["fields"].keys())
            if u and isinstance(u.get("fields"), dict):
                sig_names |= set(u["fields"].keys())

        mean_names = sorted(mean_names)
        sig_names = sorted(sig_names)

        # keep only STD* sigma fields (and drop lon/lat)
        sig_keep = []
        for s in sig_names:
            if not isinstance(s, str):
                continue
            if s.upper() in ("LAT", "LON", "LATITUDE", "LONGITUDE"):
                continue
            if self.prefer_sigma_field_prefix and not s.startswith(self.prefer_sigma_field_prefix):
                continue
            sig_keep.append(s)

        unified_fields: Dict[str, np.ndarray] = {}
        unified_sigma: Dict[str, np.ndarray] = {}

        # --- stack mean fields with remap-to-ref if needed ---
        for fn in mean_names:
            stack = np.full((len(vkeys), nlat_ref, nlon_ref), np.nan, dtype=float)
            for i, vk in enumerate(vkeys):
                g = versions[vk].get("grid")
                if not g or fn not in (g.get("fields") or {}):
                    continue
                arr2d = g["fields"][fn]
                if arr2d.shape == (nlat_ref, nlon_ref):
                    stack[i] = arr2d
                else:
                    # remap this field to reference grid
                    src_lon2d = g.get("lon2d")
                    src_lat2d = g.get("lat2d")
                    if src_lon2d is None or src_lat2d is None:
                        continue
                    stack[i] = self._nn_remap_to_ref(src_lon2d, src_lat2d, arr2d, ref_lon2d, ref_lat2d)
            unified_fields[fn] = stack

        # --- stack sigma fields with remap-to-ref if needed ---
        for sn in sig_keep:
            stack = np.full((len(vkeys), nlat_ref, nlon_ref), np.nan, dtype=float)
            for i, vk in enumerate(vkeys):
                u = versions[vk].get("uncert")
                g = versions[vk].get("grid")
                if not u or sn not in (u.get("fields") or {}):
                    continue
                arr2d = u["fields"][sn]
                if arr2d.shape == (nlat_ref, nlon_ref):
                    stack[i] = arr2d
                else:
                    # need lon/lat from the grid to remap sigma too
                    if not g or g.get("lon2d") is None or g.get("lat2d") is None:
                        continue
                    stack[i] = self._nn_remap_to_ref(g["lon2d"], g["lat2d"], arr2d, ref_lon2d, ref_lat2d)
            unified_sigma[sn] = stack

        self.uq_state["unified"] = {
            "lon2d": ref_lon2d,
            "lat2d": ref_lat2d,
            "fields": unified_fields,
            "sigma": unified_sigma,
            "version_keys": vkeys,
            "ref_version": ref_v,
            "ref_shape": (nlat_ref, nlon_ref),
            "note": "Versions remapped to ref grid via nearest-neighbor when shapes differ.",
        }

        if self.verbose:
            mism = []
            for vk in vkeys:
                g = versions[vk].get("grid")
                if g and g.get("lon2d") is not None:
                    if g["lon2d"].shape != (nlat_ref, nlon_ref):
                        mism.append((vk, g["lon2d"].shape))
            if mism:
                print(f"[SHAKEuq] unified: remapped {len(mism)} version(s) onto ref grid {ref_v} shape={(nlat_ref,nlon_ref)}.")
                print("[SHAKEuq] unified mismatches (version, shape):", mism[:10], "..." if len(mism) > 10 else "")









    def build_observations(
        self,
        version: Union[int, str],
        imt: str = "MMI",
        dyfi_source: str = "auto",  # auto | stationlist | cdi | both
        sigma_override: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Return standardized observation table with columns:
          lon, lat, value, sigma, source_type, source_detail, station_id, version, tae_hours, imt
    
        Final filtering rule (minimal, non-overfiltering):
          - Drop rows with missing lon/lat/value (and for PGA, "value" is the PGA value).
          - Log per-source how many rows were removed (only if verbose=True).
        """
        vkey = _norm_version(version)
        if vkey not in (self.uq_state.get("versions") or {}):
            raise KeyError(f"Version not in uq_state: {vkey}. Run uq_build_dataset() first.")
    
        imt_u = str(imt).upper().strip()
        vpack = self.uq_state["versions"][vkey]
    
        tae_h = None
        try:
            tae_h = float(self.uq_state.get("sanity").set_index("version").loc[vkey, "TAE_hours"])
        except Exception:
            tae_h = None
    
        def _drop_minimal_required(df: pd.DataFrame, *, source_detail: str) -> pd.DataFrame:
            """Drop only rows with missing lon/lat/value and optionally log how many were removed."""
            if not isinstance(df, pd.DataFrame) or df.empty:
                return df
            n0 = len(df)
    
            # treat empty strings as missing for lon/lat/value
            for c in ("lon", "lat", "value"):
                if c in df.columns:
                    df[c] = df[c].replace(r"^\s*$", np.nan, regex=True)
    
            df = df.dropna(subset=["lon", "lat", "value"])
            n1 = len(df)
            if getattr(self, "verbose", False):
                dropped = n0 - n1
                if dropped > 0:
                    print(f"[SHAKEuq][OBS FILTER] v={vkey} imt={imt_u} source={source_detail} dropped={dropped} kept={n1}")
            return df
    
        obs_frames: List[pd.DataFrame] = []
    
        # ------------------------------------------------------------------
        # Instruments (seismic) — used for PGA/PGV/MMI_SEISMIC
        # Filter rule requested: lon, lat, pga/pgv column (mapped to value)
        # ------------------------------------------------------------------
        inst = vpack.get("stations", {}).get("instruments")
        if isinstance(inst, pd.DataFrame) and not inst.empty and imt_u in ("PGA", "PGV", "MMI_SEISMIC"):
            lon_c = _first_present(inst, ["longitude", "lon", "Longitude", "LON"])
            lat_c = _first_present(inst, ["latitude", "lat", "Latitude", "LAT"])
            # instruments sometimes store "pga"/"pgv" regardless of imt selection
            val_c = _first_present(inst, [imt_u.lower(), imt_u, "pga", "pgv"])
    
            if lon_c and lat_c and val_c:
                df = inst.copy()
                df["lon"] = pd.to_numeric(df[lon_c], errors="coerce")
                df["lat"] = pd.to_numeric(df[lat_c], errors="coerce")
                df["value"] = pd.to_numeric(df[val_c], errors="coerce")
    
                df["sigma"] = float(sigma_override) if sigma_override is not None else float(self.sigma_instr)
                df["source_type"] = "seismic"
                df["source_detail"] = "station"
                sid = _first_present(df, ["id", "station_id", "code", "station", "station_code"])
                df["station_id"] = df[sid].astype(str) if sid else None
                df["version"] = vkey
                df["imt"] = imt_u
                df["tae_hours"] = tae_h
    
                df = df[["lon", "lat", "value", "sigma", "source_type", "source_detail", "station_id", "version", "imt", "tae_hours"]]
                df = _drop_minimal_required(df, source_detail="station")
                if not df.empty:
                    obs_frames.append(df)
    
        # ------------------------------------------------------------------
        # DYFI stationlist (intensity) — used for MMI only
        # Filter rule requested: lon, lat, intensity (mapped to value)
        # ------------------------------------------------------------------
        dyfi_sl = vpack.get("stations", {}).get("dyfi_stationlist")
        want_sl = dyfi_source in ("auto", "stationlist", "both")
        if isinstance(dyfi_sl, pd.DataFrame) and not dyfi_sl.empty and imt_u == "MMI" and want_sl:
            lon_c = _first_present(dyfi_sl, ["longitude", "lon", "Longitude", "LON"])
            lat_c = _first_present(dyfi_sl, ["latitude", "lat", "Latitude", "LAT"])
            val_c = _first_present(dyfi_sl, ["intensity", "mmi", "MMI"])
    
            if lon_c and lat_c and val_c:
                df = dyfi_sl.copy()
                df["lon"] = pd.to_numeric(df[lon_c], errors="coerce")
                df["lat"] = pd.to_numeric(df[lat_c], errors="coerce")
                df["value"] = pd.to_numeric(df[val_c], errors="coerce")
    
                df["sigma"] = float(sigma_override) if sigma_override is not None else float(self.sigma_dyfi_stationlist)
                df["source_type"] = "intensity"
                df["source_detail"] = "dyfi_stationlist"
                sid = _first_present(df, ["id", "station_id", "code", "station", "station_code"])
                df["station_id"] = df[sid].astype(str) if sid else None
                df["version"] = vkey
                df["imt"] = imt_u
                df["tae_hours"] = tae_h
    
                df = df[["lon", "lat", "value", "sigma", "source_type", "source_detail", "station_id", "version", "imt", "tae_hours"]]
                df = _drop_minimal_required(df, source_detail="dyfi_stationlist")
                if not df.empty:
                    obs_frames.append(df)
    
        # ------------------------------------------------------------------
        # CDI file (intensity) — used for MMI only
        # Filter rule requested: lon, lat, CDI (mapped to value)
        # Plus: existing distance/nresp gates stay as-is (these are not "overfiltering")
        # ------------------------------------------------------------------
        cdi = vpack.get("cdi", {}).get("df")
        want_cdi = dyfi_source in ("auto", "cdi", "both")
        if isinstance(cdi, pd.DataFrame) and not cdi.empty and imt_u == "MMI" and want_cdi:
            lon_c = _first_present(cdi, ["Longitude", "longitude", "lon", "LON"])
            lat_c = _first_present(cdi, ["Latitude", "latitude", "lat", "LAT"])
            val_c = _first_present(cdi, ["CDI", "cdi", "intensity", "mmi"])
            nresp_c = _first_present(cdi, ["No. of responses", "nresp", "NRESP"])
            dist_c = _first_present(cdi, ["Hypocentral distance", "distance", "dist", "DIST"])
    
            if lon_c and lat_c and val_c:
                df = cdi.copy()
                df["lon"] = pd.to_numeric(df[lon_c], errors="coerce")
                df["lat"] = pd.to_numeric(df[lat_c], errors="coerce")
                df["value"] = pd.to_numeric(df[val_c], errors="coerce")
    
                # keep existing CDI gates
                if self.cdi_max_dist_km is not None and dist_c and dist_c in df.columns:
                    df[dist_c] = pd.to_numeric(df[dist_c], errors="coerce")
                    df = df[df[dist_c].isna() | (df[dist_c] <= float(self.cdi_max_dist_km))]
    
                if self.cdi_min_nresp is not None and nresp_c and nresp_c in df.columns:
                    df[nresp_c] = pd.to_numeric(df[nresp_c], errors="coerce")
                    df = df[df[nresp_c].isna() | (df[nresp_c] >= int(self.cdi_min_nresp))]
    
                df["sigma"] = float(sigma_override) if sigma_override is not None else float(self.sigma_cdi)
                df["source_type"] = "intensity"
                df["source_detail"] = "cdi_geo"
                df["station_id"] = None
                df["version"] = vkey
                df["imt"] = imt_u
                df["tae_hours"] = tae_h
    
                df = df[["lon", "lat", "value", "sigma", "source_type", "source_detail", "station_id", "version", "imt", "tae_hours"]]
                df = _drop_minimal_required(df, source_detail="cdi_geo")
                if not df.empty:
                    obs_frames.append(df)
    
        # ------------------------------------------------------------------
        # Final concat + final minimal filter + log
        # ------------------------------------------------------------------
        cols = ["lon", "lat", "value", "sigma", "source_type", "source_detail", "station_id", "version", "imt", "tae_hours"]
        if not obs_frames:
            return pd.DataFrame(columns=cols)
    
        out = pd.concat(obs_frames, ignore_index=True)
    
        n0 = len(out)
        out = _drop_minimal_required(out, source_detail="ALL")
        n1 = len(out)
        if getattr(self, "verbose", False):
            dropped = n0 - n1
            if dropped > 0:
                print(f"[SHAKEuq][OBS FILTER] v={vkey} imt={imt_u} source=ALL dropped={dropped} kept={n1}")
    
        return out
    
    
        

    def extract_raw_shakemap(self, imt: str):
        # (kept exactly as in your latest file)
        imt = str(imt).upper().strip()
        log = []
        versions_out = []

        out = {"imt": imt, "versions": {}, "summary": None, "log": log}

        if "versions" not in self.uq_state:
            raise RuntimeError("uq_state['versions'] missing. Run uq_build_dataset() first.")

        for vkey, vpack in self.uq_state["versions"].items():
            grid = vpack.get("grid")
            uncert = vpack.get("uncert")
            stations = vpack.get("stations", {}) or {}
            cdi = vpack.get("cdi", {}) or {}

            mean_grid = None
            sigma_grid = None

            mean_available = False
            sigma_available = False

            if grid and "fields" in grid and imt in grid["fields"]:
                mean_grid = grid["fields"][imt]
                mean_available = True

            sigma_name = f"{self.prefer_sigma_field_prefix}{imt}".upper()
            if uncert and "fields" in uncert and sigma_name in uncert["fields"]:
                sigma_grid = uncert["fields"][sigma_name]
                sigma_available = True

            n_grid = int(mean_grid.size) if mean_available else 0

            n_inst = int(stations["instruments"].shape[0]) if isinstance(stations.get("instruments"), pd.DataFrame) else 0
            n_dyfi = int(stations["dyfi_stationlist"].shape[0]) if isinstance(stations.get("dyfi_stationlist"), pd.DataFrame) else 0
            n_cdi = int(cdi["df"].shape[0]) if isinstance(cdi.get("df"), pd.DataFrame) else 0

            log.append(
                f"[RAW] v={vkey} IMT={imt} "
                f"mean={'Y' if mean_available else 'N'} "
                f"sigma={'Y' if sigma_available else 'N'} "
                f"grid_pts={n_grid} inst={n_inst} dyfi={n_dyfi} cdi={n_cdi}"
            )

            out["versions"][vkey] = {
                "mean_grid": mean_grid,
                "sigma_grid": sigma_grid,
                "n_grid_points": n_grid,
                "available": {
                    "mean": mean_available,
                    "sigma": sigma_available,
                    "stations_instruments": n_inst,
                    "stations_dyfi": n_dyfi,
                    "cdi": n_cdi,
                },
                "used": {"mean": mean_available, "sigma": sigma_available},
            }

            versions_out.append(
                {
                    "version": vkey,
                    "imt": imt,
                    "mean_available": mean_available,
                    "sigma_available": sigma_available,
                    "n_grid_points": n_grid,
                    "n_instruments": n_inst,
                    "n_dyfi_stationlist": n_dyfi,
                    "n_cdi": n_cdi,
                    "mean_used": mean_available,
                    "sigma_used": sigma_available,
                }
            )

        out["summary"] = pd.DataFrame(versions_out)
        return out

    # --- the rest of your unique methods (units/vs30/log-space, baseline inputs, etc.) were preserved in the file download ---
    # NOTE: the downloadable file includes ALL remaining methods; the chat paste is already very long.
    # Replace your SHAKEuq.py with the downloaded file for the complete class.

    def resolve_cdi_path(self, cdi_input: Optional[str]) -> Tuple[Optional[str], str]:
        """
        Explicit-only CDI resolution.
          - FILE -> use exactly that file if it exists
          - DIR  -> search inside for best CDI match
          - None/empty -> (None, "cdi_none_explicit")
        Returns: (cdi_path_or_None, note)
        """
        if cdi_input is None:
            return None, "cdi_none_explicit"

        if not isinstance(cdi_input, str):
            return None, "cdi_invalid_type"

        p = cdi_input.strip().strip('"').strip("'")
        if not p:
            return None, "cdi_empty_explicit"

        p = os.path.abspath(os.path.expanduser(p))

        if os.path.isfile(p):
            return (p, "cdi_input=file") if os.path.exists(p) else (None, "cdi_input=file_missing")

        if os.path.isdir(p):
            cand = self._search_cdi_in_dir(p)
            return (cand, "cdi_input=dir") if cand else (None, "cdi_input=dir_no_match")

        return None, "cdi_input=path_not_found"



    # ======================================================================
    # SYNTHETIC MODE (Mode A: in-memory products, but follows uq_build_dataset
    # pipeline outputs: uq_state["versions"], uq_state["unified"], sanity)
    # ======================================================================

    def enable_synthetic_mode(self, synthetic: Dict[str, Any], *, event_id: str = "simulation") -> None:
        """
        Enable synthetic mode by injecting already-prepared per-version products.

        synthetic must be a dict like:
          {
            "versions": {
               "001": {
                  "grid": {"lon2d":..., "lat2d":..., "fields": {...}, "orientation": ...},
                  "uncert": {"fields": {...}},
                  "stations": {"instruments": df_or_none, "dyfi_stationlist": df_or_none, "debug": {}},
                  "cdi": {"df": df_or_none, "cdi_loaded": bool, "cdi_note": str, ...},
                  "rupture": {"data": any_or_none, "meta": {}, "debug": {}},
                  "meta": {"event_timestamp": "...", "process_timestamp": "...", ...},
               },
               ...
            },
            "truth": {... optional ...},
            "note": "..."
          }

        After enabling, calling uq_build_dataset() will build uq_state from this synthetic store.
        """
        if not isinstance(synthetic, dict) or "versions" not in synthetic:
            raise ValueError("synthetic must be a dict with key 'versions'")

        # normalize version keys
        vmap = {}
        for k, v in (synthetic.get("versions") or {}).items():
            vk = _norm_version(k)
            vmap[vk] = v

        if not vmap:
            raise ValueError("synthetic['versions'] is empty")

        self.event_id = event_id
        self._synthetic_store = {
            "versions": vmap,
            "truth": synthetic.get("truth"),
            "note": synthetic.get("note", "synthetic_mode"),
        }

        # if user didn't provide version_list, set from synthetic
        self.version_list = list(vmap.keys())
        self.uq_state["config"]["event_id"] = self.event_id
        self.uq_state["config"]["synthetic_mode"] = True
        self.uq_state["config"]["version_list"] = list(self.version_list)

    def disable_synthetic_mode(self) -> None:
        """Disable synthetic mode and go back to file-backed uq_build_dataset."""
        if hasattr(self, "_synthetic_store"):
            self._synthetic_store = None
        self.uq_state["config"]["synthetic_mode"] = False

    @staticmethod
    def parse_grid_spec_from_shakemap_xml_snippet(xml_text: str) -> Dict[str, Any]:
        """
        Parse <grid_specification ...> and <event ...> from a shakemap_grid snippet.
        Returns dict with lon_min, lon_max, lat_min, lat_max, nlon, nlat, dlon, dlat,
        plus event meta if present.
        """
        root = ET.fromstring(xml_text.strip())
        ns = {"sm": "http://earthquake.usgs.gov/eqcenter/shakemap"}
        # handle snippets without namespace prefixes
        grid_spec = root.find(".//{http://earthquake.usgs.gov/eqcenter/shakemap}grid_specification")
        if grid_spec is None:
            grid_spec = root.find(".//grid_specification")
        if grid_spec is None:
            raise ValueError("No <grid_specification> found in snippet")

        def _get(attr, cast=float, default=None):
            v = grid_spec.attrib.get(attr, None)
            if v is None:
                return default
            try:
                return cast(v)
            except Exception:
                return default

        out = {
            "lon_min": _get("lon_min", float),
            "lon_max": _get("lon_max", float),
            "lat_min": _get("lat_min", float),
            "lat_max": _get("lat_max", float),
            "nlon": _get("nlon", int),
            "nlat": _get("nlat", int),
            "dlon": _get("nominal_lon_spacing", float),
            "dlat": _get("nominal_lat_spacing", float),
        }

        ev = root.find(".//{http://earthquake.usgs.gov/eqcenter/shakemap}event")
        if ev is None:
            ev = root.find(".//event")
        if ev is not None:
            for k in ("event_id", "magnitude", "depth", "lat", "lon", "event_timestamp", "event_description"):
                if k in ev.attrib:
                    out[k] = ev.attrib.get(k)

        # grid_field names present in snippet (may be mean fields or sigma fields)
        fields = []
        for gf in root.findall(".//{http://earthquake.usgs.gov/eqcenter/shakemap}grid_field"):
            nm = gf.attrib.get("name")
            if nm:
                fields.append(nm)
        if not fields:
            for gf in root.findall(".//grid_field"):
                nm = gf.attrib.get("name")
                if nm:
                    fields.append(nm)
        out["grid_field_names"] = fields
        return out

    # ----------------------------- synthetic generators -----------------------------

    @staticmethod
    def _synthetic_make_lonlat(lon_min: float, lon_max: float, lat_min: float, lat_max: float, nlon: int, nlat: int):
        lon1d = np.linspace(lon_min, lon_max, int(nlon))
        lat1d = np.linspace(lat_max, lat_min, int(nlat))  # USGS grid_data often starts from top row
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        return lon2d.astype(float), lat2d.astype(float)

    @staticmethod
    def _synthetic_radial_km(lon2d: np.ndarray, lat2d: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
        # fast approx (good enough for synthetic tests)
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat0))
        dx = (lon2d - lon0) * km_per_deg_lon
        dy = (lat2d - lat0) * km_per_deg_lat
        return np.sqrt(dx * dx + dy * dy)

    @classmethod
    def _synthetic_truth_fields(
        cls,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        *,
        lon0: float,
        lat0: float,
        mag: float,
        seed: int,
        pga_units: str = "%g",
        add_texture: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Create synthetic mean fields (MMI + PGA at minimum).
        PGA produced in %g by default (matching grid.xml example).
        """
        rng = np.random.default_rng(int(seed))
        r_km = cls._synthetic_radial_km(lon2d, lat2d, lon0, lat0)
        r = np.maximum(r_km, 1.0)

        # "ground truth" style decay (simple but stable)
        # MMI: higher near source, decays with log distance
        mmi = (1.5 + 1.2 * mag) - 2.0 * np.log10(r) - 0.002 * r
        mmi = np.clip(mmi, 1.0, 10.0)

        # PGA (%g): higher near source, decays faster than MMI
        pga = (0.5 * (10 ** (0.25 * (mag - 6.0))) * (r ** -1.1)) * 100.0  # convert to %g-ish scale
        pga = np.clip(pga, 0.0001, 200.0)

        if add_texture:
            # smooth-ish spatial texture
            tex = rng.normal(0.0, 0.15, size=lon2d.shape)
            mmi = np.clip(mmi + tex, 1.0, 10.0)
            pga = np.clip(pga * (1.0 + 0.10 * tex), 0.0001, 200.0)

        return {"MMI": mmi.astype(float), "PGA": pga.astype(float)}

    @staticmethod
    def _synthetic_sigma_fields(
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        *,
        lon0: float,
        lat0: float,
        base_mmi: float,
        base_lnpga: float,
        seed: int,
        tighten_factor: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Create synthetic sigma fields:
          STDMMI (intensity units)
          STDPGA (ln(g))  -- even if PGA mean is %g, sigma is in ln-space like USGS uncertainty fields
        """
        rng = np.random.default_rng(int(seed))
        r_km = SHAKEuq._synthetic_radial_km(lon2d, lat2d, lon0, lat0)

        # spatially varying sigma: larger far away
        grad = (r_km / (np.nanmax(r_km) + 1e-9))
        stdmmi = base_mmi * tighten_factor * (0.85 + 0.60 * grad)
        stdpga = base_lnpga * tighten_factor * (0.85 + 0.60 * grad)

        # gentle noise (kept small to avoid weirdness)
        stdmmi = np.clip(stdmmi + rng.normal(0.0, 0.02, size=lon2d.shape), 0.05, 2.0)
        stdpga = np.clip(stdpga + rng.normal(0.0, 0.02, size=lon2d.shape), 0.05, 2.0)

        return {"STDMMI": stdmmi.astype(float), "STDPGA": stdpga.astype(float)}

    @staticmethod
    def _synthetic_sample_points(lon_min, lon_max, lat_min, lat_max, n, seed: int):
        rng = np.random.default_rng(int(seed))
        lons = rng.uniform(lon_min, lon_max, int(n))
        lats = rng.uniform(lat_min, lat_max, int(n))
        return lons, lats

    @staticmethod
    def _synthetic_interp_nn(lon2d: np.ndarray, lat2d: np.ndarray, field2d: np.ndarray, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        # nearest-neighbor sampling on rectilinear mesh (assumes lon/lat monotonic per row/col)
        lon1d = lon2d[0, :]
        lat1d = lat2d[:, 0]
        j = np.clip(np.searchsorted(lon1d, lons) - 1, 0, lon1d.size - 1)
        i = np.clip(np.searchsorted(lat1d[::-1], lats) - 1, 0, lat1d.size - 1)  # lat desc -> reverse
        i = (lat1d.size - 1) - i
        return field2d[i, j].astype(float)

    def build_synthetic_case(
        self,
        *,
        versions: List[Union[str, int]] = None,
        grid_spec: Dict[str, Any] = None,
        event_meta: Dict[str, Any] = None,
        n_instruments: int = 200,
        n_dyfi_stationlist: int = 400,
        n_cdi: int = 400,
        include_cdi_from_version: Union[int, str] = 2,
        seed: int = 42,
        tighten_over_time: bool = True,
    ) -> Dict[str, Any]:
        """
        High-level synthetic builder that produces an in-memory synthetic store
        and enables synthetic mode. Then calling uq_build_dataset() will build
        uq_state using this synthetic store.

        - versions: e.g. ["001","002",...]
        - grid_spec: must include lon_min, lon_max, lat_min, lat_max, nlon, nlat
          (you can use parse_grid_spec_from_shakemap_xml_snippet to generate this)
        - event_meta: optional, e.g. {"event_timestamp": "...", "process_timestamp": "...", "magnitude": 7.4, "lat": 23.8, "lon": 121.5}
        """
        if versions is None:
            versions = ["001", "002", "003", "004"]
        vkeys = [_norm_version(v) for v in versions]

        if grid_spec is None:
            raise ValueError("grid_spec is required")

        lon_min = float(grid_spec["lon_min"])
        lon_max = float(grid_spec["lon_max"])
        lat_min = float(grid_spec["lat_min"])
        lat_max = float(grid_spec["lat_max"])
        nlon = int(grid_spec["nlon"])
        nlat = int(grid_spec["nlat"])

        meta0 = dict(event_meta or {})
        mag = float(meta0.get("magnitude", 7.0))
        lat0 = float(meta0.get("lat", 0.0))
        lon0 = float(meta0.get("lon", 0.0))

        # if not provided, pick center of domain
        if lat0 == 0.0 and lon0 == 0.0:
            lat0 = 0.5 * (lat_min + lat_max)
            lon0 = 0.5 * (lon_min + lon_max)

        self.include_cdi_from_version = _norm_version(include_cdi_from_version)

        lon2d, lat2d = self._synthetic_make_lonlat(lon_min, lon_max, lat_min, lat_max, nlon, nlat)

        versions_out: Dict[str, Any] = {}
        truth_out: Dict[str, Any] = {
            "type": "synthetic_decay",
            "lon0": lon0,
            "lat0": lat0,
            "magnitude": mag,
            "grid_spec": {k: grid_spec.get(k) for k in ("lon_min","lon_max","lat_min","lat_max","nlon","nlat","dlon","dlat")},
        }

        # generate per-version products
        for idx, vk in enumerate(vkeys):
            vseed = int(seed) + 1000 * idx

            # optionally tighten uncertainties over time
            tighten = 1.0
            if tighten_over_time:
                tighten = max(0.55, 1.0 - 0.06 * idx)

            mean_fields = self._synthetic_truth_fields(
                lon2d, lat2d, lon0=lon0, lat0=lat0, mag=mag, seed=vseed, add_texture=True
            )
            sig_fields = self._synthetic_sigma_fields(
                lon2d, lat2d,
                lon0=lon0, lat0=lat0,
                base_mmi=float(getattr(self, "sigma_mmi_default", getattr(self, "sigma_grid_mmi", 0.3))),
                base_lnpga=float(getattr(self, "sigma_pga_default", getattr(self, "sigma_grid_lnpga", 0.7))),

                seed=vseed + 1,
                tighten_factor=tighten,
            )

            # sample observation points
            inst_lons, inst_lats = self._synthetic_sample_points(lon_min, lon_max, lat_min, lat_max, n_instruments, seed=vseed + 2)
            dyfi_lons, dyfi_lats = self._synthetic_sample_points(lon_min, lon_max, lat_min, lat_max, n_dyfi_stationlist, seed=vseed + 3)
            cdi_lons, cdi_lats = self._synthetic_sample_points(lon_min, lon_max, lat_min, lat_max, n_cdi, seed=vseed + 4)

            inst_mmi = self._synthetic_interp_nn(lon2d, lat2d, mean_fields["MMI"], inst_lons, inst_lats)
            inst_pga = self._synthetic_interp_nn(lon2d, lat2d, mean_fields["PGA"], inst_lons, inst_lats)
            dyfi_mmi = self._synthetic_interp_nn(lon2d, lat2d, mean_fields["MMI"], dyfi_lons, dyfi_lats)
            cdi_mmi = self._synthetic_interp_nn(lon2d, lat2d, mean_fields["MMI"], cdi_lons, cdi_lats)

            # add measurement noise
            rng = np.random.default_rng(vseed + 5)
            inst_mmi_obs = np.clip(inst_mmi + rng.normal(0.0, float(self.sigma_mmi_instr), size=inst_mmi.shape), 1.0, 10.0)
            inst_pga_obs = np.clip(inst_pga * np.exp(rng.normal(0.0, float(self.sigma_pga_instr), size=inst_pga.shape)), 0.0001, 200.0)
            dyfi_mmi_obs = np.clip(dyfi_mmi + rng.normal(0.0, float(self.sigma_mmi_dyfi), size=dyfi_mmi.shape), 1.0, 10.0)
            cdi_mmi_obs = np.clip(cdi_mmi + rng.normal(0.0, 0.33, size=cdi_mmi.shape), 1.0, 10.0)
            

            # build stationlist-like frames (keep full row schema style from your examples)
            instruments_df = pd.DataFrame({
                "id": [f"ST{idx:02d}.{i:04d}" for i in range(n_instruments)],
                "station_code": [f"{i:04d}" for i in range(n_instruments)],
                "instrumentType": ["UNK"] * n_instruments,
                "commType": ["UNK"] * n_instruments,
                "station_name": [None] * n_instruments,
                "longitude": inst_lons,
                "latitude": inst_lats,
                "location": [""] * n_instruments,
                "source": [None] * n_instruments,
                "network": ["SY"] * n_instruments,
                "station_type": ["seismic"] * n_instruments,
                "vs30": rng.uniform(180, 800, size=n_instruments),
                "elev": rng.uniform(0, 2500, size=n_instruments),
                "distance": rng.uniform(0, 400, size=n_instruments),
                "rrup": rng.uniform(0, 400, size=n_instruments),
                "repi": rng.uniform(0, 450, size=n_instruments),
                "rhypo": rng.uniform(0, 450, size=n_instruments),
                "rjb": rng.uniform(0, 400, size=n_instruments),
                "intensity": inst_mmi_obs,
                "intensity_flag": [0] * n_instruments,
                "intensity_stddev": [float(self.sigma_mmi_instr)] * n_instruments,
                "pga": inst_pga_obs,
                "pga_unit": ["%g"] * n_instruments,
                "predictions": [None] * n_instruments,
                "mmi_from_pgm": [None] * n_instruments,
                "channel_number": [0] * n_instruments,
            })

            dyfi_stationlist_df = pd.DataFrame({
                "id": [f"DYFI.{idx:02d}.{i:04d}" for i in range(n_dyfi_stationlist)],
                "station_code": [f"UTM:({idx:02d} {i:04d})" for i in range(n_dyfi_stationlist)],
                "instrumentType": ["OBSERVED"] * n_dyfi_stationlist,
                "commType": ["UNK"] * n_dyfi_stationlist,
                "station_name": [None] * n_dyfi_stationlist,
                "longitude": dyfi_lons,
                "latitude": dyfi_lats,
                "location": [""] * n_dyfi_stationlist,
                "source": ["USGS (Did You Feel It?)"] * n_dyfi_stationlist,
                "network": ["DYFI"] * n_dyfi_stationlist,
                "station_type": ["macroseismic"] * n_dyfi_stationlist,
                "nresp": rng.integers(1, 10, size=n_dyfi_stationlist),
                "vs30": rng.uniform(180, 800, size=n_dyfi_stationlist),
                "intensity": dyfi_mmi_obs,
                "intensity_flag": [0] * n_dyfi_stationlist,
                "intensity_stddev": [float(self.sigma_mmi_dyfi)] * n_dyfi_stationlist,
                "elev": [None] * n_dyfi_stationlist,
                "distance": rng.uniform(0, 600, size=n_dyfi_stationlist),
                "rrup": rng.uniform(0, 600, size=n_dyfi_stationlist),
                "repi": rng.uniform(0, 700, size=n_dyfi_stationlist),
                "rhypo": rng.uniform(0, 700, size=n_dyfi_stationlist),
                "rjb": rng.uniform(0, 600, size=n_dyfi_stationlist),
                "predictions": [None] * n_dyfi_stationlist,
                "mmi_from_pgm": [None] * n_dyfi_stationlist,
                "channel_number": [0] * n_dyfi_stationlist,
            })

            # CDI-like frame (your parser produces columns like: CDI, No. of responses, Hypocentral distance, Latitude, Longitude, Standard deviation, Suspect?, City/State)
            cdi_df = pd.DataFrame({
                "Geocoded box": [f"UTM:({idx:02d} {i:04d})" for i in range(n_cdi)],
                "CDI": cdi_mmi_obs,
                "No. of responses": rng.integers(1, 8, size=n_cdi),
                "Hypocentral distance": rng.uniform(0, 2500, size=n_cdi),
                "Latitude": cdi_lats,
                "Longitude": cdi_lons,
                "Suspect?": [0] * n_cdi,
                "Standard deviation": np.clip(rng.normal(0.33, 0.08, size=n_cdi), 0.05, 1.0),
                "City": [f"UTM:({idx:02d} {i:04d})" for i in range(n_cdi)],
                "State": [np.nan] * n_cdi,
            })

            # meta timestamps (synthetic)
            event_ts = meta0.get("event_timestamp", "2026-01-01T00:00:00")
            process_ts = meta0.get("process_timestamp", f"2026-01-01T00:{idx:02d}:00")

            # build per-version packs matching uq_build_dataset structure
            versions_out[vk] = {
                "paths": {},     # synthetic has no files
                "exists": {},    # synthetic has no files
                "meta": {
                    **meta0,
                    "event_timestamp": event_ts,
                    "process_timestamp": process_ts,
                    "synthetic_version_index": idx,
                },
                "grid": {
                    "lon2d": lon2d,
                    "lat2d": lat2d,
                    "fields": {k: v for k, v in mean_fields.items()},
                    "orientation": "synthetic",
                },
                "uncert": {
                    "fields": {k: v for k, v in sig_fields.items()},
                },
                "stations": {
                    "instruments": instruments_df,
                    "dyfi_stationlist": dyfi_stationlist_df,
                    "debug": {"synthetic": True},
                },
                "cdi": {
                    "df": cdi_df,
                    "debug": {"synthetic": True},
                    "cdi_path": None,
                    "cdi_loaded": True,
                    "cdi_note": "synthetic_cdi",
                },
                "rupture": {
                    "data": None,
                    "meta": {"synthetic": True},
                    "debug": {},
                },
            }

        synthetic_store = {"versions": versions_out, "truth": truth_out, "note": "synthetic_mode_A"}
        self.enable_synthetic_mode(synthetic_store, event_id="simulation")
        return synthetic_store

    # ----------------------------- uq_build_dataset dispatch -----------------------------

    def _uq_build_dataset_synthetic(self) -> Dict[str, Any]:
        """
        Build uq_state using self._synthetic_store (in-memory), producing the same outputs:
          - uq_state["versions"]
          - uq_state["unified"]
          - uq_state["sanity"]
        """
        store = getattr(self, "_synthetic_store", None)
        if not store or not isinstance(store, dict) or "versions" not in store:
            raise ValueError("Synthetic mode not enabled or synthetic store missing.")

        per_version: Dict[str, Any] = {}
        sanity_rows: List[Dict[str, Any]] = []

        # set event_time_utc if we can
        any_meta = next(iter((store.get("versions") or {}).values()), {}).get("meta", {}) or {}
        if self.event_time_utc is None:
            et = _parse_iso_utc(any_meta.get("event_timestamp"))
            if et is not None:
                self.event_time_utc = et
                self.uq_state["config"]["event_time_utc"] = str(self.event_time_utc)

        for vkey in self.version_list:
            vk = _norm_version(vkey)
            src = (store.get("versions") or {}).get(vk)
            if src is None:
                continue

            stations_pack = src.get("stations") or {"instruments": None, "dyfi_stationlist": None, "debug": {}}
            cdi_pack = src.get("cdi") or {"df": None, "cdi_loaded": False, "cdi_note": "synthetic_missing"}
            rupture_pack = src.get("rupture") or {"data": None, "meta": {}, "debug": {}}

            # gate CDI like real pipeline (so you can test that logic)
            try:
                use_cdi = int(vk) >= int(self.include_cdi_from_version)
            except Exception:
                use_cdi = False
            if not use_cdi:
                cdi_pack = {**cdi_pack, "df": None, "cdi_loaded": False, "cdi_note": "cdi_gate=off"}

            n_inst = int(stations_pack["instruments"].shape[0]) if isinstance(stations_pack.get("instruments"), pd.DataFrame) else 0
            n_dyfi_sl = int(stations_pack["dyfi_stationlist"].shape[0]) if isinstance(stations_pack.get("dyfi_stationlist"), pd.DataFrame) else 0
            n_cdi = int(cdi_pack["df"].shape[0]) if isinstance(cdi_pack.get("df"), pd.DataFrame) else 0

            meta_pack = src.get("meta") or {}
            process_ts = meta_pack.get("process_timestamp")
            tae_h = _guess_time_after_event_hours(self.event_time_utc, _parse_iso_utc(process_ts))

            per_version[vk] = {
                "paths": src.get("paths") or {},
                "exists": src.get("exists") or {},
                "meta": meta_pack,
                "grid": src.get("grid"),
                "uncert": src.get("uncert"),
                "stations": stations_pack,
                "cdi": cdi_pack,
                "rupture": rupture_pack,
                "obs_audit": {
                    "n_instruments": n_inst,
                    "n_dyfi_stationlist": n_dyfi_sl,
                    "n_cdi": n_cdi,
                    "use_cdi": use_cdi,
                    "cdi_loaded": bool(cdi_pack.get("cdi_loaded")),
                    "rupture_loaded": bool((rupture_pack.get("data") is not None)),
                },
            }

            sanity_rows.append(
                {
                    "version": vk,
                    "process_timestamp": process_ts,
                    "event_timestamp": meta_pack.get("event_timestamp"),
                    "TAE_hours": tae_h,
                    "grid_xml": True,
                    "uncertainty_xml": True,
                    "stationlist_json": True,
                    "rupture_json": bool((rupture_pack.get("data") is not None)),
                    "use_cdi": use_cdi,
                    "cdi_loaded": bool(cdi_pack.get("cdi_loaded")),
                    "n_instruments": n_inst,
                    "n_dyfi_stationlist": n_dyfi_sl,
                    "n_cdi": n_cdi,
                    "note": "synthetic",
                }
            )

            if self.verbose:
                print(
                    f"[SHAKEuq] parsed v={vk} grid=True unc=True stations=True "
                    f"rupture_loaded={bool((rupture_pack.get('data') is not None))} "
                    f"use_cdi={use_cdi} cdi_loaded={bool(cdi_pack.get('cdi_loaded'))} n_cdi={n_cdi}"
                )

        self.uq_state["versions"] = per_version
        self.uq_state["truth"] = store.get("truth")
        self._build_unified_grids()
        self.uq_state["sanity"] = pd.DataFrame(sanity_rows)
        return self.uq_state

    # --- copy of the original file-backed uq_build_dataset (kept so we can dispatch) ---
    def _uq_build_dataset_filebacked(self) -> Dict[str, Any]:
        """
        Original file-backed implementation (copied verbatim from earlier uq_build_dataset),
        preserved so uq_build_dataset() can dispatch between file-backed and synthetic.
        """
        if not self.version_list:
            raise ValueError("version_list is empty.")

        # Resolve CDI once (explicit-only).
        cdi_path, cdi_resolve_note = self.resolve_cdi_path(self.dyfi_cdi_input)

        sanity_rows: List[Dict[str, Any]] = []
        per_version: Dict[str, Any] = {}

        for vkey in self.version_list:
            paths = self._discover_version_paths(vkey)
            exists = {k: (p is not None and os.path.exists(p)) for k, p in paths.items()}

            grid_pack = None
            uncert_pack = None
            meta_pack: Dict[str, Any] = {}

            if exists.get("grid_xml"):
                grid_pack = self._read_shakemap_grid_xml(paths["grid_xml"])
                meta_pack.update(grid_pack.get("meta", {}))
                if self.event_time_utc is None:
                    et = _parse_iso_utc(meta_pack.get("event_timestamp"))
                    if et is not None:
                        self.event_time_utc = et
                        self.uq_state["config"]["event_time_utc"] = str(self.event_time_utc)

            if exists.get("uncertainty_xml"):
                orientation = grid_pack["grid"].get("orientation") if grid_pack else None
                uncert_pack = self._read_uncertainty_xml(paths["uncertainty_xml"], orientation=orientation)
                meta_pack.update(uncert_pack.get("meta", {}))

            stations_pack = {"instruments": None, "dyfi_stationlist": None, "debug": {}}
            if exists.get("stationlist_json"):
                stations_pack = self._read_stationlist_json(paths["stationlist_json"])

            rupture_pack = {"data": None, "meta": {}, "debug": {}}
            rupture_loaded = False
            if exists.get("rupture_json"):
                rupture_pack = self._read_rupture_json(paths["rupture_json"])
                rupture_loaded = rupture_pack.get("data") is not None

            # CDI gate by version
            try:
                use_cdi = int(vkey) >= int(self.include_cdi_from_version)
            except Exception:
                use_cdi = False

            cdi_pack = {
                "df": None,
                "debug": {},
                "cdi_path": cdi_path,
                "cdi_loaded": False,
                "cdi_note": "",
            }

            if use_cdi and cdi_path and os.path.exists(cdi_path):
                tmp = self._read_cdi_file(cdi_path)
                cdi_pack.update(tmp)
                cdi_pack["cdi_loaded"] = isinstance(tmp.get("df"), pd.DataFrame) and not tmp["df"].empty
                cdi_pack["cdi_note"] = cdi_resolve_note
            else:
                if not use_cdi:
                    cdi_pack["cdi_note"] = "cdi_gate=off"
                elif cdi_path is None:
                    cdi_pack["cdi_note"] = cdi_resolve_note
                else:
                    cdi_pack["cdi_note"] = f"{cdi_resolve_note}_missing_file"

            n_inst = int(stations_pack["instruments"].shape[0]) if isinstance(stations_pack.get("instruments"), pd.DataFrame) else 0
            n_dyfi_sl = int(stations_pack["dyfi_stationlist"].shape[0]) if isinstance(stations_pack.get("dyfi_stationlist"), pd.DataFrame) else 0
            n_cdi = int(cdi_pack["df"].shape[0]) if isinstance(cdi_pack.get("df"), pd.DataFrame) else 0

            process_ts = meta_pack.get("process_timestamp")
            tae_h = _guess_time_after_event_hours(self.event_time_utc, _parse_iso_utc(process_ts))

            per_version[vkey] = {
                "paths": paths,
                "exists": exists,
                "meta": meta_pack,
                "grid": grid_pack["grid"] if grid_pack else None,
                "uncert": uncert_pack["uncert"] if uncert_pack else None,
                "stations": stations_pack,
                "cdi": cdi_pack,
                "rupture": rupture_pack,
                "obs_audit": {
                    "n_instruments": n_inst,
                    "n_dyfi_stationlist": n_dyfi_sl,
                    "n_cdi": n_cdi,
                    "use_cdi": use_cdi,
                    "cdi_loaded": bool(cdi_pack.get("cdi_loaded")),
                    "rupture_loaded": rupture_loaded,
                },
            }

            sanity_rows.append(
                {
                    "version": vkey,
                    "process_timestamp": process_ts,
                    "event_timestamp": meta_pack.get("event_timestamp"),
                    "TAE_hours": tae_h,
                    "grid_xml": exists.get("grid_xml", False),
                    "uncertainty_xml": exists.get("uncertainty_xml", False),
                    "stationlist_json": exists.get("stationlist_json", False),
                    "rupture_json": exists.get("rupture_json", False),
                    "use_cdi": use_cdi,
                    "cdi_loaded": bool(cdi_pack.get("cdi_loaded")),
                    "n_instruments": n_inst,
                    "n_dyfi_stationlist": n_dyfi_sl,
                    "n_cdi": n_cdi,
                    "cdi_note": cdi_pack.get("cdi_note"),
                }
            )

            if self.verbose:
                print(
                    f"[SHAKEuq] parsed v={vkey} "
                    f"grid={exists.get('grid_xml')} unc={exists.get('uncertainty_xml')} "
                    f"stations={exists.get('stationlist_json')} rupture_loaded={rupture_loaded} "
                    f"use_cdi={use_cdi} cdi_loaded={bool(cdi_pack.get('cdi_loaded'))} n_cdi={n_cdi}"
                )

        self.uq_state["versions"] = per_version
        self._build_unified_grids()
        self.uq_state["sanity"] = pd.DataFrame(sanity_rows)
        return self.uq_state

    # --- public dispatch: same name uq_build_dataset, now supports synthetic mode ---
    def uq_build_dataset(self) -> Dict[str, Any]:
        """
        Dispatching uq_build_dataset:
          - if synthetic mode enabled -> in-memory build
          - else -> original file-backed build
        """
        store = getattr(self, "_synthetic_store", None)
        if store and isinstance(store, dict) and store.get("versions"):
            return self._uq_build_dataset_synthetic()
        return self._uq_build_dataset_filebacked()


    
 






    def _get_grid_extent(
        self,
        version: Union[str, int],
        grid_mode: str = "unified",  # "unified" | "native"
        margin_deg: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """
        Return (lon_min, lon_max, lat_min, lat_max) for a chosen grid definition.
    
        grid_mode="unified": uses uq_state["unified"]["lon2d/lat2d"]
        grid_mode="native" : uses uq_state["versions"][v]["grid"]["lon2d/lat2d"]
    
        margin_deg expands the bounds (useful to avoid edge effects).
        """
        vkey = _norm_version(version)
    
        if grid_mode == "unified":
            uni = (self.uq_state.get("unified") or {})
            lon2d = uni.get("lon2d")
            lat2d = uni.get("lat2d")
            if lon2d is None or lat2d is None:
                raise RuntimeError("Unified lon2d/lat2d not available. Run uq_build_dataset() first.")
        elif grid_mode == "native":
            vpack = (self.uq_state.get("versions") or {}).get(vkey) or {}
            grid = vpack.get("grid") or {}
            lon2d = grid.get("lon2d")
            lat2d = grid.get("lat2d")
            if lon2d is None or lat2d is None:
                raise RuntimeError(f"Native grid lon2d/lat2d not available for v={vkey}.")
        else:
            raise ValueError("grid_mode must be 'unified' or 'native'.")
    
        lon_min = float(np.nanmin(lon2d)) - float(margin_deg)
        lon_max = float(np.nanmax(lon2d)) + float(margin_deg)
        lat_min = float(np.nanmin(lat2d)) - float(margin_deg)
        lat_max = float(np.nanmax(lat2d)) + float(margin_deg)
        return lon_min, lon_max, lat_min, lat_max
    
    
    def _mask_points_in_extent(
        self,
        lon: Union[np.ndarray, pd.Series],
        lat: Union[np.ndarray, pd.Series],
        extent: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """
        Return boolean mask selecting points inside extent.
        extent = (lon_min, lon_max, lat_min, lat_max)
        """
        lon_min, lon_max, lat_min, lat_max = extent
        lon_arr = np.asarray(lon, dtype=float)
        lat_arr = np.asarray(lat, dtype=float)
        return (
            np.isfinite(lon_arr) & np.isfinite(lat_arr) &
            (lon_arr >= lon_min) & (lon_arr <= lon_max) &
            (lat_arr >= lat_min) & (lat_arr <= lat_max)
        )
    
    
    def audit_observations_extent(
        self,
        obs: pd.DataFrame,
        version: Union[str, int],
        grid_mode: str = "unified",
        margin_deg: float = 0.0,
        groupby: str = "source_detail",
    ) -> Dict[str, Any]:
        """
        Audit how many observation rows fall inside/outside a chosen grid extent.
    
        Returns dict with:
          - extent
          - n_total, n_inside, n_outside, frac_outside
          - by_group (DataFrame) if groupby column exists
        """
        if not isinstance(obs, pd.DataFrame) or obs.empty:
            return {
                "extent": None,
                "n_total": 0,
                "n_inside": 0,
                "n_outside": 0,
                "frac_outside": 0.0,
                "by_group": pd.DataFrame(),
            }
    
        extent = self._get_grid_extent(version=version, grid_mode=grid_mode, margin_deg=margin_deg)
        mask = self._mask_points_in_extent(obs["lon"], obs["lat"], extent)
    
        n_total = int(len(obs))
        n_inside = int(mask.sum())
        n_outside = int((~mask).sum())
        frac_outside = float(n_outside / n_total) if n_total else 0.0
    
        by_group = pd.DataFrame()
        if groupby in obs.columns:
            tmp = obs.copy()
            tmp["_inside"] = mask.astype(bool)
            by_group = (
                tmp.groupby([groupby, "_inside"])
                   .size()
                   .reset_index(name="n")
                   .pivot(index=groupby, columns="_inside", values="n")
                   .fillna(0)
                   .rename(columns={False: "outside", True: "inside"})
            )
            by_group["total"] = by_group["inside"] + by_group["outside"]
            by_group["frac_outside"] = np.where(by_group["total"] > 0, by_group["outside"] / by_group["total"], 0.0)
            by_group = by_group.sort_values("total", ascending=False)
    
        return {
            "extent": extent,
            "n_total": n_total,
            "n_inside": n_inside,
            "n_outside": n_outside,
            "frac_outside": frac_outside,
            "by_group": by_group,
        }
    
    
    def filter_observations_to_extent(
        self,
        obs: pd.DataFrame,
        version: Union[str, int],
        grid_mode: str = "unified",
        margin_deg: float = 0.0,
        return_dropped: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Filter observations to those inside a chosen grid extent.
        This should be used in downstream methods that *require* gridded mapping (kriging/bayes/etc).
    
        Parameters
        ----------
        obs : standardized observation DataFrame (expects lon/lat columns)
        version : version key used to select extent
        grid_mode : "unified" (default) or "native"
        margin_deg : expand extent slightly
        return_dropped : if True, returns (kept, dropped)
    
        Returns
        -------
        kept_df (and optionally dropped_df)
        """
        if not isinstance(obs, pd.DataFrame) or obs.empty:
            if return_dropped:
                return obs, obs
            return obs
    
        extent = self._get_grid_extent(version=version, grid_mode=grid_mode, margin_deg=margin_deg)
        mask = self._mask_points_in_extent(obs["lon"], obs["lat"], extent)
    
        kept = obs.loc[mask].copy()
        if not return_dropped:
            return kept
    
        dropped = obs.loc[~mask].copy()
        return kept, dropped
    
    
    def log_observation_extent_audit(
        self,
        obs: pd.DataFrame,
        version: Union[str, int],
        imt: str,
        grid_mode: str = "unified",
        margin_deg: float = 0.0,
        groupby: str = "source_detail",
        label: str = "OBS EXTENT",
    ) -> Dict[str, Any]:
        """
        Convenience: run audit_observations_extent + print a compact log.
        Returns the audit dict.
        """
        aud = self.audit_observations_extent(
            obs=obs,
            version=version,
            grid_mode=grid_mode,
            margin_deg=margin_deg,
            groupby=groupby,
        )
    
        if getattr(self, "verbose", False):
            ext = aud.get("extent")
            print(
                f"[SHAKEuq][{label}] v={_norm_version(version)} imt={str(imt).upper()} grid_mode={grid_mode} "
                f"outside={aud['n_outside']}/{aud['n_total']} ({aud['frac_outside']:.3f}) extent={ext}"
            )
            bg = aud.get("by_group")
            if isinstance(bg, pd.DataFrame) and not bg.empty:
                # print top groups only
                print("[SHAKEuq][{label}] by-group (top):")
                print(bg.head(6).to_string())
    
        return aud




    # ======================================================================
    # Ordinary Kriging (OK) + CDI conditioning + Global GMICE + audit plot
    # PASTE THIS *INSIDE* class SHAKEuq (indent everything by 4 spaces)
    #
    # Notes:
    # - Uses existing imports: numpy as np, pandas as pd, matplotlib in plot.
    # - Extent filtering is ALWAYS applied first.
    # - OK variance is interpolation variance (from variogram + OK system).
    # - Observation sigma is used ONLY as measurement-error diagonal inflation
    #   (optional) — it is not ShakeMap STD*.
    # - Debug prints are controlled by debug=True and/or self.verbose if exists.
    # ======================================================================
    
    def global_gmice_convert(self, input_type, output_type, values):
        """
        Global GMICE conversion (vectorized) as provided by you.
    
        Supported:
          PGA <-> MMI
          PGV <-> MMI
    
        Parameters
        ----------
        input_type : str
        output_type: str
        values     : array-like
    
        Returns
        -------
        np.ndarray
        """
        input_type = str(input_type).upper().strip()
        output_type = str(output_type).upper().strip()
        x = np.asarray(values, dtype=float)
    
        # PGA -> MMI
        if input_type == "PGA" and output_type == "MMI":
            return np.where(
                x < 0, 0,
                np.where(
                    x <= 50,
                    2.27 + 1.647 * np.log10(x),
                    -1.361 + 3.822 * np.log10(x),
                )
            )
    
        # PGV -> MMI
        if input_type == "PGV" and output_type == "MMI":
            return np.where(
                x < 0, 0,
                np.where(
                    x <= 2.5,
                    4.424 + 1.589 * np.log10(x),
                    4.018 + 2.671 * np.log10(x),
                )
            )
    
        # MMI -> PGA
        if input_type == "MMI" and output_type == "PGA":
            return np.where(
                x < 5.132463357,
                10 ** ((x - 2.27) / 1.647),
                10 ** ((x + 1.361) / 3.822),
            )
    
        # MMI -> PGV
        if input_type == "MMI" and output_type == "PGV":
            return np.where(
                x < 5.056326674,
                10 ** ((x - 4.424) / 1.589),
                10 ** ((x - 4.018) / 2.671),
            )
    
        raise ValueError(f"Invalid GMICE conversion: {input_type} -> {output_type}")
    
    
    # ---------------------------
    # OK core helpers
    # ---------------------------
    
    def _ok_debug(self, debug, msg):
        if bool(debug) or bool(getattr(self, "verbose", False)):
            print(msg)
    
    def _ok_lonlat_to_xy_km(self, lon, lat, lon0=None, lat0=None):
        """
        Local equirectangular projection in km around (lon0, lat0).
        Good baseline for ShakeMap-sized extents.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
    
        if lon0 is None:
            lon0 = float(np.nanmean(lon))
        if lat0 is None:
            lat0 = float(np.nanmean(lat))
    
        R = 6371.0  # km
        lat0r = np.radians(lat0)
        x = (lon - lon0) * np.cos(lat0r) * (np.pi / 180.0) * R
        y = (lat - lat0) * (np.pi / 180.0) * R
        return x, y, float(lon0), float(lat0)
    
    def _ok_variogram_gamma(self, h, model="exponential", range_km=80.0, sill=1.0, nugget=1e-6):
        """
        Semivariogram gamma(h).
        """
        h = np.asarray(h, dtype=float)
        a = max(float(range_km), 1e-9)
        sill = float(sill)
        nugget = float(nugget)
    
        m = str(model).lower().strip()
        if m == "exponential":
            return nugget + sill * (1.0 - np.exp(-h / a))
        if m == "gaussian":
            return nugget + sill * (1.0 - np.exp(-(h / a) ** 2))
        if m == "spherical":
            hr = h / a
            out = np.empty_like(hr, dtype=float)
            inside = hr < 1.0
            out[inside] = nugget + sill * (1.5 * hr[inside] - 0.5 * (hr[inside] ** 3))
            out[~inside] = nugget + sill
            return out
        raise ValueError(f"Unknown variogram model: {model}")
    
    def _ok_cov(self, h, model="exponential", range_km=80.0, sill=1.0, nugget=1e-6):
        """
        Covariance from semivariogram:
          C(h) = (nugget+sill) - gamma(h)
        """
        gamma = self._ok_variogram_gamma(h, model=model, range_km=range_km, sill=sill, nugget=nugget)
        return (float(nugget) + float(sill)) - gamma
    
    def _ok_pairwise_dist_km(self, x, y):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        dx = x - x.T
        dy = y - y.T
        return np.sqrt(dx * dx + dy * dy)
    
    def _ok_solve_point(self,
                        x_obs, y_obs, z_obs,
                        x_tgt, y_tgt,
                        *,
                        sigma_obs=None,
                        use_obs_sigma=True,
                        variogram_model="exponential",
                        range_km=80.0,
                        sill=1.0,
                        nugget=1e-6,
                        ridge=1e-10):
        """
        Ordinary kriging solve at one target point.
    
        If use_obs_sigma=True and sigma_obs provided:
          C <- C + diag(sigma_obs^2)
        This is the correct place to use observation uncertainty in OK.
    
        Returns
        -------
        zhat, vhat  (mean and interpolation variance)
        """
        x_obs = np.asarray(x_obs, dtype=float)
        y_obs = np.asarray(y_obs, dtype=float)
        z_obs = np.asarray(z_obs, dtype=float)
        n = x_obs.size
        if n < 2:
            return np.nan, np.nan
    
        # C among observations
        D = self._ok_pairwise_dist_km(x_obs, y_obs)
        C = self._ok_cov(D, model=variogram_model, range_km=range_km, sill=sill, nugget=nugget)
    
        # Stabilize
        C = C + float(ridge) * np.eye(n)
    
        # Add measurement error (optional)
        if use_obs_sigma and sigma_obs is not None:
            sigma_obs = np.asarray(sigma_obs, dtype=float)
            sigma_obs = np.nan_to_num(sigma_obs, nan=0.0)
            C = C + np.diag(sigma_obs ** 2)
    
        # c vector obs->target
        dx = x_obs - float(x_tgt)
        dy = y_obs - float(y_tgt)
        d = np.sqrt(dx * dx + dy * dy)
        c = self._ok_cov(d, model=variogram_model, range_km=range_km, sill=sill, nugget=nugget)
    
        # OK augmented system
        # [C  1][w ] = [c]
        # [1^T0][mu]   [1]
        A = np.zeros((n + 1, n + 1), dtype=float)
        A[:n, :n] = C
        A[:n, n] = 1.0
        A[n, :n] = 1.0
    
        b = np.zeros(n + 1, dtype=float)
        b[:n] = c
        b[n] = 1.0
    
        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
    
        w = sol[:n]
        mu = sol[n]
    
        zhat = float(np.dot(w, z_obs))
    
        # OK variance: C(0) - w^T c - mu
        c0 = float(self._ok_cov(np.array([0.0]), model=variogram_model, range_km=range_km, sill=sill, nugget=nugget)[0])
        vhat = float(c0 - np.dot(w, c) - float(mu))
        if not np.isfinite(vhat):
            vhat = np.nan
        else:
            vhat = max(vhat, 0.0)
    
        return zhat, vhat
    
    def _ok_krige_grid(self,
                       obs_df,
                       lon2d, lat2d,
                       *,
                       neighbor_k=25,
                       max_points=None,
                       use_obs_sigma=True,
                       variogram_model="exponential",
                       range_km=80.0,
                       sill=1.0,
                       nugget=1e-6,
                       ridge=1e-10,
                       debug=False):
        """
        Krige on target grid using K-nearest observations per grid cell.
        No external deps.
    
        obs_df must contain: lon, lat, value, (optional) sigma
        """
        if not isinstance(obs_df, pd.DataFrame) or obs_df.empty:
            self._ok_debug(debug, "[OK] obs_df empty -> cannot krige.")
            return None, None
    
        df = obs_df.copy()
        df = df.dropna(subset=["lon", "lat", "value"])
        if df.empty or len(df) < 2:
            self._ok_debug(debug, f"[OK] not enough valid obs after dropna: n={len(df)}")
            return None, None
    
        if max_points is not None and len(df) > int(max_points):
            df = df.sample(n=int(max_points), random_state=42).reset_index(drop=True)
            self._ok_debug(debug, f"[OK] downsampled obs to max_points={max_points}, n={len(df)}")
    
        # local projection using obs mean
        x_obs, y_obs, lon0, lat0 = self._ok_lonlat_to_xy_km(df["lon"].to_numpy(), df["lat"].to_numpy())
        z_obs = df["value"].to_numpy(dtype=float)
        s_obs = df["sigma"].to_numpy(dtype=float) if ("sigma" in df.columns) else None
    
        x_tgt, y_tgt, _, _ = self._ok_lonlat_to_xy_km(lon2d, lat2d, lon0=lon0, lat0=lat0)
    
        mean2d = np.full_like(lon2d, np.nan, dtype=float)
        var2d = np.full_like(lon2d, np.nan, dtype=float)
    
        if neighbor_k is None:
            k = None
        else:
            k = max(2, int(neighbor_k))
    
        x_flat = x_tgt.ravel()
        y_flat = y_tgt.ravel()
    
        # brute-force neighbor selection
        for idx in range(x_flat.size):
            xt = float(x_flat[idx])
            yt = float(y_flat[idx])
    
            if k is None:
                nn = np.arange(len(x_obs))
            else:
                d2 = (x_obs - xt) ** 2 + (y_obs - yt) ** 2
                nn = np.argsort(d2)[:k]
    
            zhat, vhat = self._ok_solve_point(
                x_obs[nn], y_obs[nn], z_obs[nn],
                xt, yt,
                sigma_obs=(s_obs[nn] if (use_obs_sigma and s_obs is not None) else None),
                use_obs_sigma=bool(use_obs_sigma),
                variogram_model=variogram_model,
                range_km=float(range_km),
                sill=float(sill),
                nugget=float(nugget),
                ridge=float(ridge),
            )
            mean2d.ravel()[idx] = zhat
            var2d.ravel()[idx] = vhat
    
        return mean2d, var2d
    
    

    # ---------------------------
    # Unified grid + prior helper
    # ---------------------------
    
    def _get_unified_grid(self):
        uni = (self.uq_state.get("unified") or {})
        lon2d = uni.get("lon2d")
        lat2d = uni.get("lat2d")
        if lon2d is None or lat2d is None:
            raise RuntimeError("Unified grid missing. Run uq_build_dataset() first.")
        return lon2d, lat2d
    
    def _get_prior_mean_unified(self, version, imt):
        vkey = _norm_version(version)
        imt_u = str(imt).upper().strip()
        lon2d, lat2d = self._get_unified_grid()
        uni = (self.uq_state.get("unified") or {})
        vkeys = list(uni.get("version_keys") or [])
        stack = (uni.get("fields") or {}).get(imt_u)
        if stack is None or vkey not in vkeys:
            return lon2d, lat2d, None
        i = vkeys.index(vkey)
        return lon2d, lat2d, np.asarray(stack[i], dtype=float)
    
    

    
    # ---------------------------
    # Audit plot (2×2): prior mean, OK mean, OK var, obs points
    # ---------------------------

    

    # ======================================================================
    # DISCRETE MEAN COLORMAPS (USGS/EMS-like) + UPDATED AUDIT PLOTS
    # Paste INSIDE class SHAKEuq (indent by 4 spaces)
    # ======================================================================
    
    def contour_scale(self, pgm_type, scale_type="usgs", units=None):
        """
        Discrete contour scale for MEAN fields (MMI, PGA, PGV, SA_1, EMS).
        Returns: cmap, bounds, ticks, norm, used_scale_label
    
        - For MEAN MMI/PGA, this is what we will use in plotters.
        - For sigma/variance fields, we keep continuous colormaps.
        """
        import matplotlib as mpl
    
        pgm_type = str(pgm_type).upper().strip()
        scale_type = str(scale_type).lower().strip()
    
        if scale_type not in ["usgs", "ems"]:
            raise ValueError("Invalid scale type. Choose 'usgs' or 'ems'.")
    
        # --- colors (as in your reference) ---
        usgs_colors = [
            (255/255, 255/255, 255/255, 1.0),
            (191/255, 204/255, 255/255, 1.0),
            (160/255, 230/255, 255/255, 1.0),
            (128/255, 255/255, 255/255, 1.0),
            (122/255, 255/255, 147/255, 1.0),
            (255/255, 255/255, 0/255, 1.0),
            (255/255, 200/255, 0/255, 1.0),
            (255/255, 145/255, 0/255, 1.0),
            (255/255, 0/255, 0/255, 1.0),
            (200/255, 0/255, 0/255, 1.0),
            (128/255, 0/255, 0/255, 1.0),
        ]
        ems_colors = [
            (1, 1, 1, 0),
            (237/255, 239/255, 243/255, 1.0),
            (172/255, 180/255, 206/255, 1.0),
            (161/255, 215/255, 227/255, 1.0),
            (143/255, 200/255, 145/255, 1.0),
            (249/255, 236/255, 51/255, 1.0),
            (238/255, 181/255, 9/255, 1.0),
            (233/255, 135/255, 45/255, 1.0),
            (223/255, 83/255, 42/255, 1.0),
            (217/255, 38/255, 42/255, 1.0),
            (136/255, 0/255, 0/255, 1.0),
            (68/255, 0/255, 1/255, 1.0),
        ]
    
        colors = usgs_colors if scale_type == "usgs" else ems_colors
        cmap = mpl.colors.ListedColormap(colors)
    
        # --- bounds tables ---
        usgs_table = {
            "pga_values_%g": [0, 0.05, 0.3, 2.8, 6.2, 11.5, 21.5, 40.1, 74.7, 139],
            "pga_values_g": [0, 0.001, 0.003, 0.028, 0.062, 0.115, 0.215, 0.401, 0.747, 1.39],
            "pga_values_cm/s2": [0, 0.5, 2.9, 27.5, 60.8, 112.8, 210.9, 393.4, 732.8, 1363.6],
            "pgv_values_cm/s": [0, 0.02, 0.1, 1.4, 4.7, 9.6, 20, 41, 86, 178],
            "sa_1_values_%g": [0, 0.02, 0.1, 1, 4.6, 10, 23, 50, 110, 244],
            "sa_1_values_g": [0, 0.0002, 0.001, 0.01, 0.046, 0.1, 0.23, 0.5, 1.1, 2.44],
            "sa_1_values_cm/s^2": [0, 0.2, 1, 9.8, 45.1, 98.1, 225.6, 490.5, 1079.1, 2393.6],
            "intensity_values_mmi": [0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10],
        }
        ems_table = {
            "pga_values_%g": [0, 0.1, 0.31, 0.51, 1.02, 2.05, 5.62, 15.42, 42.34, 116.24, 319.11, 876.08, 2405.13],
            "pga_values_g": [0, 0.001, 0.0031, 0.0051, 0.0102, 0.0205, 0.0562, 0.1542, 0.4234, 1.1624, 3.1911, 8.7608, 24.0513],
            "pga_values_cm/s^2": [0, 1, 3, 5, 10, 20.07, 55.11, 151.29, 415.36, 1140.3, 3130.5, 8594.3, 23594.3],
            "pgv_values_cm/s": [0, 1, 3, 5, 8, 13, 25, 56.64, 234.62, 971.97, 4026.6, 16681.01, 69104.48],
            "intensity_values_ems": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
        table = usgs_table if scale_type == "usgs" else ems_table
    
        # --- default units ---
        usgs_default_units = {"PGA": "%g", "PGV": "cm/s", "SA_1": "%g", "MMI": "MMI"}
        ems_default_units = {"PGA": "cm/s^2", "PGV": "cm/s", "EMS": "EMS"}
        default_units = usgs_default_units if scale_type == "usgs" else ems_default_units
    
        # --- labels ---
        labels = {
            "%g": "%g",
            "g": "g",
            "cm/s^2": r"${cm/s}^2$",
            "cm/s": "cm/s",
            "MMI": "MMI",
            "EMS": "EMS",
        }
        usgs_scale_labels = {
            "PGA": "Peak Ground Acceleration",
            "PGV": "Peak Ground Velocity",
            "SA_1": r"Spectral Acceleration $Sa_{1s}$",
            "MMI": "Modified Mercalli Intensity",
        }
        ems_scale_labels = {"PGA": "Peak Ground Acceleration", "PGV": "Peak Ground Velocity", "EMS": "European Macroseismic Scale"}
        scale_labels = usgs_scale_labels if scale_type == "usgs" else ems_scale_labels
    
        if units is None:
            units = default_units.get(pgm_type, "%g")
    
        # --- pick key ---
        if scale_type == "ems" and pgm_type == "EMS":
            key = "intensity_values_ems"
        elif scale_type == "usgs" and pgm_type == "MMI":
            key = "intensity_values_mmi"
        else:
            key = f"{pgm_type.lower()}_values_{str(units).lower()}"
    
        if key not in table:
            raise ValueError(f"Invalid combination of PGM type '{pgm_type}' and units '{units}' for scale '{scale_type}'")
    
        bounds = list(table[key])
        ticks = list(bounds)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
        used_scale = f"{scale_labels.get(pgm_type, pgm_type)} ({labels.get(units, units)})"
        return cmap, bounds, ticks, norm, used_scale
    


    
    # redacted 
    def plot_shakemap_raw_audit(
        self,
        *,
        version="latest",
        build_if_needed=True,
        show=True,
        save_path_prefix=None,
        dpi=150,
        figsize_maps=(12.0, 9.0),
        figsize_counts=(11.0, 4.0),
        # NEW: mean scale control
        mean_scale_type="usgs",
        mean_pga_units="%g",
        # NEW: sigma panels control (continuous)
        sigma_cmap="viridis",
        stdmmi_vmin=None,
        stdmmi_vmax=None,
        stdpga_vmin=None,
        stdpga_vmax=None,
    ):
        """
        UPDATED RAW audit plot:
          - MMI mean + PGA mean use discrete USGS/EMS palettes (only mean fields)
          - STDMMI + STDPGA use continuous cmap (default: viridis)
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
    
        # Build if needed
        if build_if_needed:
            uni = (self.uq_state.get("unified") or {}) if isinstance(self.uq_state, dict) else {}
            if not isinstance(uni, dict) or uni.get("lon2d") is None or uni.get("lat2d") is None:
                self.uq_build_dataset()
    
        # Resolve version
        vkey = _norm_version(version) if str(version).lower().strip() != "latest" else None
        uni = self.uq_state.get("unified") or {}
        vkeys = list(uni.get("version_keys") or list((self.uq_state.get("versions") or {}).keys()))
        vkeys = [str(k) for k in vkeys]
        if not vkeys:
            raise RuntimeError("No versions available. Run uq_build_dataset() first.")
    
        if vkey is None:
            vkey = sorted(vkeys)[-1]
        if vkey not in vkeys and vkey not in (self.uq_state.get("versions") or {}):
            raise KeyError(f"Requested version not in uq_state: {vkey}")
    
        lon2d = uni.get("lon2d")
        lat2d = uni.get("lat2d")
    
        # Pull mean + sigma from unified stacks if available, else from per-version pack
        def _get_unified_field(kind, imt):
            # kind: "fields" or "sigma"
            stack = (uni.get(kind) or {}).get(imt)
            if stack is None:
                return None
            if vkey not in vkeys:
                return None
            i = vkeys.index(vkey)
            return np.asarray(stack[i], dtype=float)
    
        mmi_mean = _get_unified_field("fields", "MMI")
        pga_mean = _get_unified_field("fields", "PGA")
        stdmmi   = _get_unified_field("sigma", "STDMMI") or _get_unified_field("sigma", "MMI")  # fallback
        stdpga   = _get_unified_field("sigma", "STDPGA") or _get_unified_field("sigma", "PGA")  # fallback
    
        # Plot Figure A (2x2)
        figA, axes = plt.subplots(2, 2, figsize=figsize_maps, dpi=dpi, constrained_layout=True)
        ax11, ax12 = axes[0, 0], axes[0, 1]
        ax21, ax22 = axes[1, 0], axes[1, 1]
    
        # Mean: discrete palettes
        cmap_mmi, norm_mmi, ticks_mmi, label_mmi = self._mean_cmap_for_imt("MMI", scale_type=mean_scale_type, pga_units=mean_pga_units)
        cmap_pga, norm_pga, ticks_pga, label_pga = self._mean_cmap_for_imt("PGA", scale_type=mean_scale_type, pga_units=mean_pga_units)
    
        self._plot_grid_panel(
            ax11, lon2d, lat2d, mmi_mean,
            title=f"MMI mean (v={vkey})",
            cmap=cmap_mmi, norm=norm_mmi,
            add_colorbar=True, cbar_ticks=ticks_mmi, cbar_label=label_mmi
        )
        self._plot_grid_panel(
            ax12, lon2d, lat2d, pga_mean,
            title=f"PGA mean (v={vkey})",
            cmap=cmap_pga, norm=norm_pga,
            add_colorbar=True, cbar_ticks=ticks_pga, cbar_label=label_pga
        )
    
        # Sigma: continuous
        self._plot_grid_panel(
            ax21, lon2d, lat2d, stdmmi,
            title=f"STDMMI (continuous) (v={vkey})",
            cmap=sigma_cmap, norm=None, vmin=stdmmi_vmin, vmax=stdmmi_vmax,
            add_colorbar=True, cbar_ticks=None, cbar_label="STDMMI"
        )
        self._plot_grid_panel(
            ax22, lon2d, lat2d, stdpga,
            title=f"STDPGA (continuous) (v={vkey})",
            cmap=sigma_cmap, norm=None, vmin=stdpga_vmin, vmax=stdpga_vmax,
            add_colorbar=True, cbar_ticks=None, cbar_label="STDPGA"
        )
    
        # Plot Figure B (counts)
        sanity = self.uq_state.get("sanity")
        figB = None
        if isinstance(sanity, pd.DataFrame) and not sanity.empty:
            figB, ax = plt.subplots(1, 1, figsize=figsize_counts, dpi=dpi, constrained_layout=True)
            df = sanity.copy()
            df = df.sort_values("version")
            ax.plot(df["version"], df.get("n_instruments", 0), marker="o", label="n_instruments")
            ax.plot(df["version"], df.get("n_dyfi_stationlist", 0), marker="o", label="n_dyfi_stationlist")
            ax.set_title("Obs counts across versions")
            ax.set_xlabel("version")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.25)
            ax.legend()
    
        if save_path_prefix:
            figA.savefig(f"{save_path_prefix}_maps.png", dpi=dpi, bbox_inches="tight")
            if figB is not None:
                figB.savefig(f"{save_path_prefix}_counts.png", dpi=dpi, bbox_inches="tight")
    
        if show:
            plt.show()
    
        return {"version_plotted": vkey, "fig_maps": figA, "fig_counts": figB}
    


    def plot_shakemap_raw_audit(
        self,
        version=None,
        build_if_needed=True,
        show=True,
        save_path_prefix=None,
        dpi=150,
        figsize_maps=(12, 10),
        figsize_counts=(10, 4),
        mean_scale_type="usgs",
        mean_pga_units="%g",
        sigma_cmap="viridis",
        stdmmi_vmin=None,
        stdmmi_vmax=None,
        stdpga_vmin=None,
        stdpga_vmax=None,
    ):
        """
        Audit plot of RAW ShakeMap products (no updates):
          - Mean MMI
          - Mean PGA
          - Sigma MMI
          - Sigma PGA
    
        Behavior:
          - Uses unified stacks when available.
          - If unified arrays are 3D (nver, nlat, nlon) and version is None,
            it will auto-select the latest version in unified.version_keys.
          - If version is provided, it selects that version slice.
        """
        import numpy as np
        import matplotlib.pyplot as plt
    
        # ---------------------------
        # ensure dataset
        # ---------------------------
        if build_if_needed:
            if not isinstance(getattr(self, "uq_state", None), dict):
                self.uq_build_dataset()
            elif "unified" not in self.uq_state:
                self.uq_build_dataset()
    
        uni = self.uq_state.get("unified", {}) or {}
    
        def _get_unified_field(group, key):
            d = uni.get(group, {})
            if not isinstance(d, dict):
                return None
            return d.get(key, None)
    
        # ---------------------------
        # fetch unified fields (SAFE)
        # ---------------------------
        mmi_mean = _get_unified_field("fields", "MMI")
        pga_mean = _get_unified_field("fields", "PGA")
    
        stdmmi = _get_unified_field("sigma", "STDMMI")
        if stdmmi is None:
            stdmmi = _get_unified_field("sigma", "MMI")  # legacy fallback
    
        stdpga = _get_unified_field("sigma", "STDPGA")
        if stdpga is None:
            stdpga = _get_unified_field("sigma", "PGA")  # legacy fallback
    
        lon2d = uni.get("lon2d", None)
        lat2d = uni.get("lat2d", None)
    
        # ---------------------------
        # sanity
        # ---------------------------
        for name, arr in [
            ("MMI mean", mmi_mean),
            ("PGA mean", pga_mean),
            ("STDMMI", stdmmi),
            ("STDPGA", stdpga),
        ]:
            if arr is None:
                raise RuntimeError(f"plot_shakemap_raw_audit: missing unified array for {name}")
    
        if lon2d is None or lat2d is None:
            raise RuntimeError("plot_shakemap_raw_audit: missing lon2d/lat2d")
    
        # ---------------------------
        # choose version slice if needed
        # ---------------------------
        vkeys = uni.get("version_keys", [])
        vkeys = [str(v) for v in vkeys] if isinstance(vkeys, list) else []
    
        def _is_stack(a):
            try:
                return (np.asarray(a).ndim == 3)
            except Exception:
                return False
    
        need_slice = _is_stack(mmi_mean) or _is_stack(pga_mean) or _is_stack(stdmmi) or _is_stack(stdpga)
    
        if need_slice:
            if not vkeys:
                raise RuntimeError("plot_shakemap_raw_audit: unified arrays are 3D but unified.version_keys is missing")
    
            if version is None:
                # safest default: latest version
                version = vkeys[-1]
    
            v = str(version)
            if v not in vkeys:
                raise ValueError(f"plot_shakemap_raw_audit: version {v} not found in unified.version_keys")
            i = vkeys.index(v)
    
            mmi_mean = np.asarray(mmi_mean)[i, :, :]
            pga_mean = np.asarray(pga_mean)[i, :, :]
            stdmmi = np.asarray(stdmmi)[i, :, :]
            stdpga = np.asarray(stdpga)[i, :, :]
    
        # final: enforce 2D for plotting
        for name, arr in [("MMI mean", mmi_mean), ("PGA mean", pga_mean), ("STDMMI", stdmmi), ("STDPGA", stdpga)]:
            a = np.asarray(arr)
            if a.ndim != 2:
                raise RuntimeError(f"plot_shakemap_raw_audit: expected 2D for {name}, got shape {a.shape}")
    
        # ---------------------------
        # plotting
        # ---------------------------
        fig, axes = plt.subplots(2, 2, figsize=figsize_maps, constrained_layout=True)
    
        im0 = axes[0, 0].pcolormesh(lon2d, lat2d, mmi_mean, shading="auto")
        axes[0, 0].set_title(f"Raw ShakeMap Mean MMI (v={version})" if version is not None else "Raw ShakeMap Mean MMI")
        plt.colorbar(im0, ax=axes[0, 0])
    
        im1 = axes[0, 1].pcolormesh(lon2d, lat2d, pga_mean, shading="auto")
        axes[0, 1].set_title(f"Raw ShakeMap Mean PGA (v={version})" if version is not None else "Raw ShakeMap Mean PGA")
        plt.colorbar(im1, ax=axes[0, 1])
    
        im2 = axes[1, 0].pcolormesh(
            lon2d, lat2d, stdmmi,
            shading="auto",
            cmap=sigma_cmap,
            vmin=stdmmi_vmin,
            vmax=stdmmi_vmax,
        )
        axes[1, 0].set_title(f"Raw ShakeMap σ(MMI) (v={version})" if version is not None else "Raw ShakeMap σ(MMI)")
        plt.colorbar(im2, ax=axes[1, 0])
    
        im3 = axes[1, 1].pcolormesh(
            lon2d, lat2d, stdpga,
            shading="auto",
            cmap=sigma_cmap,
            vmin=stdpga_vmin,
            vmax=stdpga_vmax,
        )
        axes[1, 1].set_title(f"Raw ShakeMap σ(PGA) (v={version})" if version is not None else "Raw ShakeMap σ(PGA)")
        plt.colorbar(im3, ax=axes[1, 1])
    
        for ax in axes.flat:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
    
        if save_path_prefix:
            fig.savefig(f"{save_path_prefix}_raw_shakemap_audit.png", dpi=dpi, bbox_inches="tight")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        


    # ============================================================
    # CDI conditioning upgrades for Ordinary Kriging baselines
    # Paste INSIDE class SHAKEuq (indent by 4 spaces)
    # ============================================================
    
    def _haversine_km(self, lon1, lat1, lon2, lat2):
        import numpy as np
        lon1 = np.asarray(lon1, dtype=float)
        lat1 = np.asarray(lat1, dtype=float)
        lon2 = np.asarray(lon2, dtype=float)
        lat2 = np.asarray(lat2, dtype=float)
        r = 6371.0
        dlon = np.deg2rad(lon2 - lon1)
        dlat = np.deg2rad(lat2 - lat1)
        a = np.sin(dlat / 2.0) ** 2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-15, 1.0 - a)))
        return r * c
    
    
    def _robust_huber_fit(self, X, y, *, max_iter=25, huber_k=1.5, ridge=1e-8):
        """
        Simple IRLS Huber regression (no sklearn dependency).
        X: (n,p), y: (n,)
        Returns beta (p,)
        """
        import numpy as np
    
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
    
        n, p = X.shape
        w = np.ones(n, dtype=float)
    
        beta = np.zeros(p, dtype=float)
        for _ in range(int(max_iter)):
            W = np.sqrt(np.maximum(w, 1e-12))
            Xw = X * W[:, None]
            yw = y * W
            A = Xw.T @ Xw + ridge * np.eye(p)
            b = Xw.T @ yw
            beta_new = np.linalg.solve(A, b)
    
            r = y - X @ beta_new
            s = 1.4826 * np.median(np.abs(r - np.median(r)))
            if not (np.isfinite(s) and s > 1e-12):
                beta = beta_new
                break
    
            u = r / (s + 1e-12)
            w_new = np.ones_like(w)
            mask = np.abs(u) > huber_k
            w_new[mask] = (huber_k / np.maximum(np.abs(u[mask]), 1e-12))
    
            if np.max(np.abs(beta_new - beta)) < 1e-6:
                beta = beta_new
                w = w_new
                break
    
            beta = beta_new
            w = w_new
    
        return beta
    
    
    def _cdi_thin_grid(self, df, *, bin_km=10.0, agg="median", min_pts=1, debug=False):
        """
        Thin CDI using km-grid binning.
        - Each bin returns 1 representative point.
        - Aggregation: median (recommended) or mean (option).
        """
        import numpy as np
        import pandas as pd
    
        if df.empty:
            return df
    
        x, y, _, _ = self._ok_lonlat_to_xy_km(df["lon"].to_numpy(), df["lat"].to_numpy())
        eps = float(bin_km)
        bx = np.floor(x / eps).astype(int)
        by = np.floor(y / eps).astype(int)
        df2 = df.copy()
        df2["_bx"] = bx
        df2["_by"] = by
        df2["_bin"] = list(zip(bx, by))
    
        out = []
        agg = str(agg).lower().strip()
    
        for _, g in df2.groupby("_bin"):
            if len(g) < int(min_pts):
                continue
    
            # use robust aggregation by default
            if agg == "mean":
                lon = float(np.mean(g["lon"].to_numpy(dtype=float)))
                lat = float(np.mean(g["lat"].to_numpy(dtype=float)))
                val = float(np.mean(g["value"].to_numpy(dtype=float)))
            else:
                lon = float(np.median(g["lon"].to_numpy(dtype=float)))
                lat = float(np.median(g["lat"].to_numpy(dtype=float)))
                val = float(np.median(g["value"].to_numpy(dtype=float)))
    
            # sigma: conservative (median sigma + within-bin scatter)
            if "sigma" in g.columns:
                s0 = np.nanmedian(pd.to_numeric(g["sigma"], errors="coerce").to_numpy(dtype=float))
                s0 = float(np.nan_to_num(s0, nan=0.5))
            else:
                s0 = 0.5
            scatter = float(np.nanstd(g["value"].to_numpy(dtype=float))) if len(g) >= 2 else 0.0
            sigma_out = float(s0 + 0.5 * scatter)
    
            r = g.iloc[0].copy()
            r["lon"] = lon
            r["lat"] = lat
            r["value"] = val
            r["sigma"] = sigma_out
            r["_cluster_n"] = int(len(g))
            r["source_detail"] = str(r.get("source_detail", "cdi")) + f"_grid{int(bin_km)}km"
            out.append(r)
    
        out_df = pd.DataFrame(out)
        if debug:
            self._ok_debug(True, f"[CDI] grid-thin: bin_km={bin_km} in_n={len(df)} out_n={len(out_df)} agg={agg}")
    
        return out_df.reset_index(drop=True)
    
    
    def _cdi_quantile_trim(self, df, *, q_low=0.05, q_high=0.95, field="value", debug=False):
        """
        Trim by quantiles on a selected field:
          field="value" or "residual"
        """
        import numpy as np
    
        if df.empty:
            return df
        a = df[field].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 5:
            return df
    
        lo = float(np.quantile(a, float(q_low)))
        hi = float(np.quantile(a, float(q_high)))
    
        kept = df[(df[field] >= lo) & (df[field] <= hi)].copy().reset_index(drop=True)
        if debug:
            self._ok_debug(True, f"[CDI] quantile-trim field={field} q=({q_low},{q_high}) lo={lo:.3g} hi={hi:.3g} kept={len(kept)}/{len(df)}")
        return kept
    
    
    def _cdi_sponheuer_filter(self,
                              df,
                              *,
                              event_lat,
                              event_lon,
                              model="a + b*log10(R) + c*R",
                              thr_sigma=2.5,
                              min_r_km=1.0,
                              robust=True,
                              huber_k=1.5,
                              debug=False):
        """
        'Sponheuer-like' distance attenuation filter:
          Fit MMI ~ a + b*log10(R) + c*R  (simple, flexible)
        Then drop points with |residual| > thr_sigma * robust_scale.
    
        This is NOT a physical GMICE; it’s a pragmatic CDI sanity filter.
    
        Requirements:
          event_lat/event_lon must be available from version meta.
        """
        import numpy as np
        import pandas as pd
    
        if df.empty:
            return df
    
        if event_lat is None or event_lon is None or not (np.isfinite(event_lat) and np.isfinite(event_lon)):
            if debug:
                self._ok_debug(True, "[CDI] sponheuer-filter skipped: event_lat/lon missing.")
            return df
    
        lon = df["lon"].to_numpy(dtype=float)
        lat = df["lat"].to_numpy(dtype=float)
        val = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=float)
    
        R = self._haversine_km(event_lon, event_lat, lon, lat)
        R = np.maximum(R, float(min_r_km))
    
        mask = np.isfinite(val) & np.isfinite(R)
        if np.count_nonzero(mask) < 10:
            return df
    
        y = val[mask]
        r = R[mask]
    
        # design matrix
        # a + b*log10(R) + c*R
        X = np.column_stack([
            np.ones_like(r),
            np.log10(r),
            r
        ])
    
        if robust:
            beta = self._robust_huber_fit(X, y, huber_k=float(huber_k))
        else:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
        yhat = X @ beta
        res = y - yhat
    
        # robust scale
        s = 1.4826 * np.median(np.abs(res - np.median(res)))
        if not (np.isfinite(s) and s > 1e-12):
            s = float(np.nanstd(res)) if np.isfinite(np.nanstd(res)) else 1.0
            s = max(s, 1e-6)
    
        keep_local = np.abs(res) <= float(thr_sigma) * s
    
        # write residual back for diagnostics
        df2 = df.copy().reset_index(drop=True)
        df2["_r_km"] = np.nan
        df2["_spon_res"] = np.nan
        df2.loc[df2.index[mask], "_r_km"] = r
        df2.loc[df2.index[mask], "_spon_res"] = res
    
        out = df2.loc[df2.index[mask][keep_local]].copy().reset_index(drop=True)
    
        if debug:
            self._ok_debug(True, f"[CDI] sponheuer-fit: beta={beta.tolist()} scale={s:.3g} thr={thr_sigma} kept={len(out)}/{len(df)}")
    
        return out
    
    # ---------------------------
    # CDI conditioning: extent-first already handled outside
    # ---------------------------

    
    def _cdi_condition(self,
                       obs_df,
                       *,
                       trust_nresp_ge=3,
                       # local outlier parameters (existing)
                       enable_local_outlier=True,
                       local_radius_km=20.0,
                       outlier_k=3.5,
                       min_neighbors=5,
                       # clustering parameters (existing)
                       enable_clustering=True,
                       cluster_eps_km=5.0,
                       cluster_min_pts=5,
                       # NEW: advanced CDI strategies
                       cdi_strategy=("local_outlier", "grid_thin", "quantile_residual"),
                       cdi_grid_bin_km=10.0,
                       cdi_grid_agg="median",
                       cdi_quantile=(0.05, 0.95),
                       cdi_quantile_field="residual",   # "residual" or "value"
                       cdi_use_prior_residual=True,
                       # Sponheuer-like filter
                       cdi_enable_sponheuer=False,
                       cdi_spon_thr_sigma=2.5,
                       cdi_spon_min_r_km=1.0,
                       cdi_spon_robust=True,
                       cdi_spon_huber_k=1.5,
                       # context for residuals / distance
                       version=None,
                       debug=False):
        """
        CDI conditioning AFTER extent filter.
    
        Strategy pipeline (order matters):
          - "local_outlier"     : your existing median/MAD local outlier removal
          - "grid_thin"         : km-bin thinning to reduce over-weighting (recommended)
          - "quantile_value"    : quantile trim on value
          - "quantile_residual" : quantile trim on residual (value - prior_at_obs), recommended
          - "sponheuer"         : distance-attenuation sanity filter (robust regression)
    
        Notes:
          - If you use residual-based trimming, it uses nearest-neighbor sampling of prior.
          - For sponheuer filter, we pull event_lat/lon from uq_state versions meta if available.
        """
        import numpy as np
        import pandas as pd
    
        if not isinstance(obs_df, pd.DataFrame) or obs_df.empty:
            self._ok_debug(debug, "[CDI] obs empty -> skip conditioning.")
            return obs_df
    
        df = obs_df.copy().reset_index(drop=True)
    
        # Ensure columns exist
        if "nresp" not in df.columns:
            df["nresp"] = np.nan
    
        # trusted flag
        nresp = pd.to_numeric(df["nresp"], errors="coerce")
        df["_trusted"] = (nresp >= float(trust_nresp_ge)).fillna(False)
    
        # optional: compute residual to prior (nearest neighbor) for robust trims
        if bool(cdi_use_prior_residual):
            vkey = _norm_version(version) if version is not None else None
            # try to get prior MMI for this version
            try:
                lon2d, lat2d, prior = self._get_prior_mean_unified(vkey, "MMI")
            except Exception:
                prior = None
    
            if prior is not None:
                lon_vals = lon2d[0, :]
                lat_vals = lat2d[:, 0]
                lon = df["lon"].to_numpy(dtype=float)
                lat = df["lat"].to_numpy(dtype=float)
                ix = np.clip(np.searchsorted(lon_vals, lon) - 1, 0, lon_vals.size - 1)
                iy = np.clip(np.searchsorted(lat_vals, lat) - 1, 0, lat_vals.size - 1)
                prior_at = prior[iy, ix]
                df["residual"] = df["value"].to_numpy(dtype=float) - prior_at
            else:
                df["residual"] = np.nan
        else:
            df["residual"] = np.nan
    
        strat = cdi_strategy
        if isinstance(strat, str):
            strat = (strat,)
        strat = tuple([str(s).lower().strip() for s in strat])
    
        n_start = len(df)
    
        # --- local outlier (existing behavior) ---
        if "local_outlier" in strat and enable_local_outlier and len(df) >= (int(min_neighbors) + 1):
            # reuse your existing local-outlier logic (copied/kept compatible)
            x, y, _, _ = self._ok_lonlat_to_xy_km(df["lon"].to_numpy(), df["lat"].to_numpy())
            df["_xkm"] = x
            df["_ykm"] = y
            vals = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=float)
            keep = np.ones(len(df), dtype=bool)
    
            R = float(local_radius_km)
            X = df["_xkm"].to_numpy(dtype=float)
            Y = df["_ykm"].to_numpy(dtype=float)
            T = df["_trusted"].to_numpy(dtype=bool)
    
            for i in range(len(df)):
                if not np.isfinite(vals[i]):
                    keep[i] = False
                    continue
                dx = X - X[i]
                dy = Y - Y[i]
                dist = np.sqrt(dx * dx + dy * dy)
                nn = np.where((dist <= R) & np.isfinite(vals))[0]
                if nn.size < int(min_neighbors):
                    continue
                neigh = vals[nn]
                med = np.nanmedian(neigh)
                mad = np.nanmedian(np.abs(neigh - med))
                scale = 1.4826 * mad
                if not np.isfinite(scale) or scale <= 1e-9:
                    continue
                z = abs(vals[i] - med) / scale
                thr = float(outlier_k) * (1.7 if T[i] else 1.0)
                if z > thr:
                    keep[i] = False
    
            df = df.loc[keep].copy().reset_index(drop=True)
            self._ok_debug(debug, f"[CDI] local-outlier: kept {len(df)}/{n_start} (dropped {n_start-len(df)})")
    
        # --- grid thinning (recommended) ---
        if "grid_thin" in strat and len(df) > 0:
            df = self._cdi_thin_grid(df, bin_km=float(cdi_grid_bin_km), agg=str(cdi_grid_agg), min_pts=1, debug=debug)
    
        # --- quantile trims ---
        if "quantile_value" in strat and len(df) > 0:
            ql, qh = cdi_quantile
            df = self._cdi_quantile_trim(df, q_low=float(ql), q_high=float(qh), field="value", debug=debug)
    
        if "quantile_residual" in strat and len(df) > 0:
            # if residual is missing, fallback to value
            field = "residual" if np.isfinite(df["residual"].to_numpy(dtype=float)).any() else "value"
            ql, qh = cdi_quantile
            df = self._cdi_quantile_trim(df, q_low=float(ql), q_high=float(qh), field=field, debug=debug)
    
        # --- Sponheuer-like attenuation filter (optional) ---
        if (("sponheuer" in strat) or bool(cdi_enable_sponheuer)) and len(df) > 0:
            vkey = _norm_version(version) if version is not None else None
            event_lat = None
            event_lon = None
            try:
                meta = (self.uq_state.get("versions") or {}).get(vkey, {}).get("meta", {})
                event_lat = meta.get("event_lat", None)
                event_lon = meta.get("event_lon", None)
            except Exception:
                event_lat, event_lon = None, None
    
            df = self._cdi_sponheuer_filter(
                df,
                event_lat=event_lat,
                event_lon=event_lon,
                thr_sigma=float(cdi_spon_thr_sigma),
                min_r_km=float(cdi_spon_min_r_km),
                robust=bool(cdi_spon_robust),
                huber_k=float(cdi_spon_huber_k),
                debug=debug
            )
    
        # --- existing clustering (keep as optional last step) ---
        if enable_clustering and ("cluster" in strat or "clustering" in strat or ("grid_thin" not in strat)) and len(df) >= int(cluster_min_pts):
            # keep your prior clustering logic as a final optional step,
            # but most users should prefer grid_thin instead.
            eps = float(cluster_eps_km)
            x, y, _, _ = self._ok_lonlat_to_xy_km(df["lon"].to_numpy(), df["lat"].to_numpy())
            bx = np.floor(x / eps).astype(int)
            by = np.floor(y / eps).astype(int)
            df["_bin"] = list(zip(bx, by))
    
            out_rows = []
            dense_bins = 0
            total_merged = 0
    
            for _, g in df.groupby("_bin"):
                if len(g) < int(cluster_min_pts):
                    out_rows.append(g)
                    continue
                dense_bins += 1
                total_merged += len(g)
    
                # weights: if sigma exists -> 1/sigma^2 else uniform
                if "sigma" in g.columns:
                    s = pd.to_numeric(g["sigma"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
                    w = 1.0 / np.maximum(s, 1e-6) ** 2
                else:
                    w = np.ones(len(g), dtype=float)
    
                wsum = float(np.sum(w))
                if wsum <= 0:
                    w = np.ones(len(g), dtype=float)
                    wsum = float(np.sum(w))
    
                lon = float(np.sum(g["lon"].to_numpy(dtype=float) * w) / wsum)
                lat = float(np.sum(g["lat"].to_numpy(dtype=float) * w) / wsum)
                val = float(np.sum(g["value"].to_numpy(dtype=float) * w) / wsum)
    
                if "sigma" in g.columns:
                    svals = pd.to_numeric(g["sigma"], errors="coerce").to_numpy(dtype=float)
                    s0 = float(np.nanmin(svals)) if np.isfinite(svals).any() else np.nan
                else:
                    s0 = np.nan
                scatter = float(np.nanstd(g["value"].to_numpy(dtype=float))) if len(g) >= 2 else 0.0
                sigma_out = float(np.nan_to_num(s0, nan=0.5) + 0.5 * scatter)
    
                r = g.iloc[0].copy()
                r["lon"] = lon
                r["lat"] = lat
                r["value"] = val
                r["sigma"] = sigma_out
                r["station_id"] = None
                r["source_detail"] = str(r.get("source_detail", "cdi")) + "_cluster"
                r["_cluster_n"] = int(len(g))
                out_rows.append(pd.DataFrame([r]))
    
            df = pd.concat(out_rows, ignore_index=True)
            self._ok_debug(debug, f"[CDI] clustering: dense_bins={dense_bins}, merged_points={total_merged}, out_n={len(df)}")
    
        if debug:
            self._ok_debug(True, f"[CDI] condition done: start={n_start} end={len(df)} strategy={strat}")
    
        return df




    # ---------------------------
    # Main runner: Ordinary Kriging in multiple modes
    # ---------------------------
    
    
    
    # --- small override: pass version into _cdi_condition so residual trimming can use prior ---
    # (this keeps your kriging logic unchanged; only adds version=... to conditioning calls)
    def run_ordinary_kriging(self,
                             *,
                             version,
                             mode,
                             margin_deg=0.0,
                             variogram_model="exponential",
                             range_km=80.0,
                             sill=1.0,
                             nugget=1e-6,
                             ridge=1e-10,
                             neighbor_k=25,
                             max_points=None,
                             use_obs_sigma=True,
                             # CDI conditioning options (existing)
                             cdi_enable_local_outlier=True,
                             cdi_local_radius_km=20.0,
                             cdi_outlier_k=3.5,
                             cdi_min_neighbors=5,
                             cdi_enable_clustering=True,
                             cdi_cluster_eps_km=5.0,
                             cdi_cluster_min_pts=5,
                             cdi_trust_nresp_ge=3,
                             # NEW CDI options
                             cdi_strategy=("local_outlier", "grid_thin", "quantile_residual"),
                             cdi_grid_bin_km=10.0,
                             cdi_grid_agg="median",
                             cdi_quantile=(0.05, 0.95),
                             cdi_enable_sponheuer=False,
                             cdi_spon_thr_sigma=2.5,
                             cdi_spon_min_r_km=1.0,
                             cdi_spon_robust=True,
                             cdi_spon_huber_k=1.5,
                             mode_key=None,
                             store_obs_used=True,
                             debug=False):
        """
        Same as your existing run_ordinary_kriging, but with upgraded CDI conditioning knobs.
        """
        import numpy as np
        import pandas as pd
    
        if not isinstance(self.uq_state, dict) or not (self.uq_state.get("versions") or {}):
            raise RuntimeError("uq_state not initialized. Run uq_build_dataset() first.")
    
        vkey = _norm_version(version)
        mode_s = str(mode).lower().strip()
        lon2d, lat2d = self._get_unified_grid()
    
        def extent_first(obs, label):
            kept, dropped = self.filter_observations_to_extent(
                obs, version=vkey, grid_mode="unified",
                margin_deg=float(margin_deg),
                return_dropped=True
            )
            self._ok_debug(debug, f"[OK][{label}] extent-first: total={len(obs)} kept={len(kept)} dropped={len(dropped)} margin_deg={margin_deg}")
            if isinstance(kept, pd.DataFrame) and not kept.empty and "source_detail" in kept.columns:
                c = kept["source_detail"].value_counts(dropna=False).to_dict()
                self._ok_debug(debug, f"[OK][{label}] kept by source_detail: {c}")
            return kept, dropped
    
        obs_used = None
        imt_target = None
    
        if mode_s == "pga_station":
            imt_target = "PGA"
            obs = self.build_observations(version=vkey, imt="PGA", dyfi_source="stationlist", sigma_override=None)
            obs_kept, _ = extent_first(obs, "PGA_station")
            obs_used = obs_kept
    
        elif mode_s == "mmi_dyfi":
            imt_target = "MMI"
            obs = self.build_observations(version=vkey, imt="MMI", dyfi_source="stationlist", sigma_override=None)
            obs_kept, _ = extent_first(obs, "MMI_dyfi")
            obs_used = obs_kept
    
        elif mode_s in ("mmi_cdi_filtered", "mmi_cdi"):
            imt_target = "MMI"
            obs = self.build_observations(version=vkey, imt="MMI", dyfi_source="cdi", sigma_override=None)
            obs_kept, _ = extent_first(obs, "MMI_cdi_raw")
            n_before = len(obs_kept)
            obs_kept = self._cdi_condition(
                obs_kept,
                version=vkey,
                trust_nresp_ge=int(cdi_trust_nresp_ge),
                enable_local_outlier=bool(cdi_enable_local_outlier),
                local_radius_km=float(cdi_local_radius_km),
                outlier_k=float(cdi_outlier_k),
                min_neighbors=int(cdi_min_neighbors),
                enable_clustering=bool(cdi_enable_clustering),
                cluster_eps_km=float(cdi_cluster_eps_km),
                cluster_min_pts=int(cdi_cluster_min_pts),
                cdi_strategy=cdi_strategy,
                cdi_grid_bin_km=float(cdi_grid_bin_km),
                cdi_grid_agg=str(cdi_grid_agg),
                cdi_quantile=cdi_quantile,
                cdi_enable_sponheuer=bool(cdi_enable_sponheuer),
                cdi_spon_thr_sigma=float(cdi_spon_thr_sigma),
                cdi_spon_min_r_km=float(cdi_spon_min_r_km),
                cdi_spon_robust=bool(cdi_spon_robust),
                cdi_spon_huber_k=float(cdi_spon_huber_k),
                debug=debug,
            )
            self._ok_debug(debug, f"[OK][MMI_cdi_filtered] conditioning: before={n_before} after={len(obs_kept)}")
            obs_used = obs_kept
    
        elif mode_s == "pga_station_plus_dyfi_as_pga":
            imt_target = "PGA"
            obs_sta = self.build_observations(version=vkey, imt="PGA", dyfi_source="stationlist", sigma_override=None)
            obs_dyfi = self.build_observations(version=vkey, imt="MMI", dyfi_source="stationlist", sigma_override=None)
    
            obs_sta, _ = extent_first(obs_sta, "PGA_station")
            obs_dyfi, _ = extent_first(obs_dyfi, "MMI_dyfi_to_PGA")
    
            obs_dyfi = obs_dyfi.copy()
            obs_dyfi["value"] = self.global_gmice_convert("MMI", "PGA", obs_dyfi["value"].to_numpy(dtype=float))
            obs_dyfi["imt"] = "PGA"
            obs_dyfi["source_detail"] = "dyfi_stationlist_as_pga"
            obs_used = pd.concat([obs_sta, obs_dyfi], ignore_index=True)
            self._ok_debug(debug, f"[OK][mixed] PGA target: station_n={len(obs_sta)} dyfi_as_pga_n={len(obs_dyfi)} total={len(obs_used)}")
    
        elif mode_s == "pga_station_plus_cdi_as_pga":
            imt_target = "PGA"
            obs_sta = self.build_observations(version=vkey, imt="PGA", dyfi_source="stationlist", sigma_override=None)
            obs_cdi = self.build_observations(version=vkey, imt="MMI", dyfi_source="cdi", sigma_override=None)
    
            obs_sta, _ = extent_first(obs_sta, "PGA_station")
            obs_cdi, _ = extent_first(obs_cdi, "MMI_cdi_raw_to_PGA")
    
            n_before = len(obs_cdi)
            obs_cdi = self._cdi_condition(
                obs_cdi,
                version=vkey,
                trust_nresp_ge=int(cdi_trust_nresp_ge),
                enable_local_outlier=bool(cdi_enable_local_outlier),
                local_radius_km=float(cdi_local_radius_km),
                outlier_k=float(cdi_outlier_k),
                min_neighbors=int(cdi_min_neighbors),
                enable_clustering=bool(cdi_enable_clustering),
                cluster_eps_km=float(cdi_cluster_eps_km),
                cluster_min_pts=int(cdi_cluster_min_pts),
                cdi_strategy=cdi_strategy,
                cdi_grid_bin_km=float(cdi_grid_bin_km),
                cdi_grid_agg=str(cdi_grid_agg),
                cdi_quantile=cdi_quantile,
                cdi_enable_sponheuer=bool(cdi_enable_sponheuer),
                cdi_spon_thr_sigma=float(cdi_spon_thr_sigma),
                cdi_spon_min_r_km=float(cdi_spon_min_r_km),
                cdi_spon_robust=bool(cdi_spon_robust),
                cdi_spon_huber_k=float(cdi_spon_huber_k),
                debug=debug,
            )
            self._ok_debug(debug, f"[OK][mixed] CDI conditioning: before={n_before} after={len(obs_cdi)}")
    
            obs_cdi = obs_cdi.copy()
            obs_cdi["value"] = self.global_gmice_convert("MMI", "PGA", obs_cdi["value"].to_numpy(dtype=float))
            obs_cdi["imt"] = "PGA"
            obs_cdi["source_detail"] = "cdi_geo_as_pga"
    
            obs_used = pd.concat([obs_sta, obs_cdi], ignore_index=True)
            self._ok_debug(debug, f"[OK][mixed] PGA target: station_n={len(obs_sta)} cdi_as_pga_n={len(obs_cdi)} total={len(obs_used)}")
    
        elif mode_s == "mmi_dyfi_plus_station_as_mmi":
            imt_target = "MMI"
            obs_dyfi = self.build_observations(version=vkey, imt="MMI", dyfi_source="stationlist", sigma_override=None)
            obs_sta = self.build_observations(version=vkey, imt="PGA", dyfi_source="stationlist", sigma_override=None)
    
            obs_dyfi, _ = extent_first(obs_dyfi, "MMI_dyfi")
            obs_sta, _ = extent_first(obs_sta, "PGA_station_to_MMI")
    
            obs_sta = obs_sta.copy()
            obs_sta["value"] = self.global_gmice_convert("PGA", "MMI", obs_sta["value"].to_numpy(dtype=float))
            obs_sta["imt"] = "MMI"
            obs_sta["source_detail"] = "station_as_mmi"
    
            obs_used = pd.concat([obs_dyfi, obs_sta], ignore_index=True)
            self._ok_debug(debug, f"[OK][mixed] MMI target: dyfi_n={len(obs_dyfi)} station_as_mmi_n={len(obs_sta)} total={len(obs_used)}")
    
        elif mode_s == "mmi_cdi_plus_station_as_mmi":
            imt_target = "MMI"
            obs_cdi = self.build_observations(version=vkey, imt="MMI", dyfi_source="cdi", sigma_override=None)
            obs_sta = self.build_observations(version=vkey, imt="PGA", dyfi_source="stationlist", sigma_override=None)
    
            obs_cdi, _ = extent_first(obs_cdi, "MMI_cdi_raw")
            obs_sta, _ = extent_first(obs_sta, "PGA_station_to_MMI")
    
            n_before = len(obs_cdi)
            obs_cdi = self._cdi_condition(
                obs_cdi,
                version=vkey,
                trust_nresp_ge=int(cdi_trust_nresp_ge),
                enable_local_outlier=bool(cdi_enable_local_outlier),
                local_radius_km=float(cdi_local_radius_km),
                outlier_k=float(cdi_outlier_k),
                min_neighbors=int(cdi_min_neighbors),
                enable_clustering=bool(cdi_enable_clustering),
                cluster_eps_km=float(cdi_cluster_eps_km),
                cluster_min_pts=int(cdi_cluster_min_pts),
                cdi_strategy=cdi_strategy,
                cdi_grid_bin_km=float(cdi_grid_bin_km),
                cdi_grid_agg=str(cdi_grid_agg),
                cdi_quantile=cdi_quantile,
                cdi_enable_sponheuer=bool(cdi_enable_sponheuer),
                cdi_spon_thr_sigma=float(cdi_spon_thr_sigma),
                cdi_spon_min_r_km=float(cdi_spon_min_r_km),
                cdi_spon_robust=bool(cdi_spon_robust),
                cdi_spon_huber_k=float(cdi_spon_huber_k),
                debug=debug,
            )
            self._ok_debug(debug, f"[OK][mixed] CDI conditioning: before={n_before} after={len(obs_cdi)}")
    
            obs_sta = obs_sta.copy()
            obs_sta["value"] = self.global_gmice_convert("PGA", "MMI", obs_sta["value"].to_numpy(dtype=float))
            obs_sta["imt"] = "MMI"
            obs_sta["source_detail"] = "station_as_mmi"
    
            obs_used = pd.concat([obs_cdi, obs_sta], ignore_index=True)
            self._ok_debug(debug, f"[OK][mixed] MMI target: cdi_n={len(obs_cdi)} station_as_mmi_n={len(obs_sta)} total={len(obs_used)}")
    
        else:
            raise ValueError(f"Unknown kriging mode: {mode}")
    
        # ---- Run kriging core on unified grid ----
        if not isinstance(obs_used, pd.DataFrame) or obs_used.empty:
            raise RuntimeError(f"No observations available after filtering for mode={mode_s}")
    
        obs_used = obs_used.copy().reset_index(drop=True)
        if use_obs_sigma and "sigma" not in obs_used.columns:
            obs_used["sigma"] = np.nan
    
        self._ok_debug(debug, f"[OK] kriging start: v={vkey} mode={mode_s} imt={imt_target} n_obs={len(obs_used)} neighbor_k={neighbor_k} use_obs_sigma={use_obs_sigma}")
    
        mean_grid, var_grid = self._ok_krige_grid(
            obs_used, lon2d, lat2d,
            neighbor_k=int(neighbor_k),
            max_points=max_points,
            variogram_model=str(variogram_model),
            range_km=float(range_km),
            sill=float(sill),
            nugget=float(nugget),
            ridge=float(ridge),
            use_obs_sigma=bool(use_obs_sigma),
            debug=debug
        )
    
        if mode_key is None:
            mode_key = f"ok__{mode_s}__{imt_target.lower()}"
    
        pack = {
            "mean_grid": mean_grid,
            "var_grid": var_grid,
            "meta": {
                "version": vkey,
                "mode": mode_s,
                "imt": imt_target,
                "variogram_model": str(variogram_model),
                "range_km": float(range_km),
                "sill": float(sill),
                "nugget": float(nugget),
                "neighbor_k": int(neighbor_k),
                "use_obs_sigma": bool(use_obs_sigma),
                "cdi_strategy": tuple(cdi_strategy) if isinstance(cdi_strategy, (list, tuple)) else (str(cdi_strategy),),
            }
        }
        if store_obs_used:
            pack["obs_used"] = obs_used
    
        self.uq_state["versions"][vkey].setdefault("kriging", {})
        self.uq_state["versions"][vkey]["kriging"][mode_key] = pack
    
        self._ok_debug(debug, f"[OK] stored: versions[{vkey}].kriging['{mode_key}'] mean_shape={mean_grid.shape} var_shape={var_grid.shape}")
        return pack




    # ======================================================================
    # OK SCENARIO SEARCH / CV DIAGNOSTICS (holdout testing + ranking)
    # Paste at end of SHAKEuq class
    # ======================================================================

    def _ok_cv_debug(self, debug, msg):
        try:
            if debug:
                print(msg)
        except Exception:
            pass

    def _ok_cv_split(self, obs_df, holdout_frac=0.1, seed=42):
        """
        Random split obs into train/test.
        Returns: train_df, test_df
        """
        import numpy as np
        import pandas as pd

        if not isinstance(obs_df, pd.DataFrame) or obs_df.empty:
            return obs_df, obs_df

        n = len(obs_df)
        if n < 5:
            return obs_df, obs_df.iloc[0:0].copy()

        frac = float(holdout_frac)
        frac = max(0.0, min(0.9, frac))
        n_test = max(1, int(round(frac * n)))
        n_test = min(n_test, n - 1)

        rng = np.random.default_rng(int(seed))
        idx = np.arange(n)
        rng.shuffle(idx)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        train_df = obs_df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = obs_df.iloc[test_idx].copy().reset_index(drop=True)
        return train_df, test_df

    def _ok_cv_metrics(self, y_true, y_pred, sigma_true=None):
        """
        Returns dict of metrics: MAE, RMSE, Bias, wRMSE (optional), wMAE (optional)
        Weighting uses 1/max(sigma,eps)^2 if sigma_true provided.
        """
        import numpy as np

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        m = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[m]
        y_pred = y_pred[m]

        if y_true.size == 0:
            return {
                "n": 0,
                "mae": None,
                "rmse": None,
                "bias": None,
                "wmae": None,
                "wrmse": None,
            }

        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        bias = float(np.mean(err))

        wmae = None
        wrmse = None
        if sigma_true is not None:
            s = np.asarray(sigma_true, dtype=float)
            s = s[m]
            eps = 1e-6
            w = 1.0 / np.maximum(s, eps) ** 2
            w = np.where(np.isfinite(w), w, 0.0)
            if float(np.sum(w)) > 0.0:
                wmae = float(np.sum(w * np.abs(err)) / np.sum(w))
                wrmse = float(np.sqrt(np.sum(w * (err * err)) / np.sum(w)))

        return {
            "n": int(y_true.size),
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "wmae": wmae,
            "wrmse": wrmse,
        }

    def _ok_cv_predict_at_points(
        self,
        train_df,
        test_lon,
        test_lat,
        *,
        variogram_model="exponential",
        range_km=80.0,
        sill=1.0,
        nugget=0.0001,
        ridge=0.0,
        neighbor_k=25,
        use_obs_sigma=True,
        debug=False,
    ):
        """
        Point-prediction Ordinary Kriging (minimal, robust):
        - Uses same variogram family as grid OK.
        - Uses k nearest training obs per test point.
        - Adds nugget + optional obs sigma into diagonal.
        Returns: (pred, var) arrays
        """
        import numpy as np
        import pandas as pd
        import math

        if not isinstance(train_df, pd.DataFrame) or train_df.empty:
            return np.full(len(test_lon), np.nan), np.full(len(test_lon), np.nan)

        # ---- Extract training arrays ----
        x_lon = np.asarray(train_df["lon"].to_numpy(dtype=float), dtype=float)
        x_lat = np.asarray(train_df["lat"].to_numpy(dtype=float), dtype=float)
        z_val = np.asarray(train_df["value"].to_numpy(dtype=float), dtype=float)

        sig = None
        if use_obs_sigma and "sigma" in train_df.columns:
            sig = np.asarray(train_df["sigma"].to_numpy(dtype=float), dtype=float)

        # ---- Coordinate conversion to km (equirectangular) ----
        # Use mean latitude for scaling longitude degrees -> km
        lat0 = float(np.nanmean(np.concatenate([x_lat, np.asarray(test_lat, dtype=float)])))
        lat0r = math.radians(lat0)
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * math.cos(lat0r)

        x = (x_lon - float(np.nanmean(x_lon))) * km_per_deg_lon
        y = (x_lat - float(np.nanmean(x_lat))) * km_per_deg_lat

        q_lon = np.asarray(test_lon, dtype=float)
        q_lat = np.asarray(test_lat, dtype=float)
        qx = (q_lon - float(np.nanmean(x_lon))) * km_per_deg_lon
        qy = (q_lat - float(np.nanmean(x_lat))) * km_per_deg_lat

        # ---- Variogram model -> covariance function ----
        model = str(variogram_model).lower().strip()

        def cov_from_h(h):
            # Use: C(h) = sill * f(h) with nugget added separately
            h = np.asarray(h, dtype=float)
            a = max(1e-6, float(range_km))
            if model in ("exponential", "exp"):
                # f = exp(-h/a)
                return float(sill) * np.exp(-h / a)
            if model in ("gaussian", "gau"):
                # f = exp(-(h/a)^2)
                return float(sill) * np.exp(-(h / a) ** 2)
            if model in ("spherical", "sph"):
                # f = 1 - 1.5(r) + 0.5(r^3) for r<=1 else 0
                r = h / a
                out = np.zeros_like(r, dtype=float)
                m = r <= 1.0
                rm = r[m]
                out[m] = float(sill) * (1.0 - 1.5 * rm + 0.5 * (rm ** 3))
                return out
            # default exponential
            return float(sill) * np.exp(-h / a)

        # ---- neighbor_k sanitize ----
        n_train = len(z_val)
        if neighbor_k is None:
            k = n_train
        else:
            k = int(neighbor_k)
            k = max(3, min(k, n_train))

        pred = np.full(len(qx), np.nan, dtype=float)
        pvar = np.full(len(qx), np.nan, dtype=float)

        # ---- Precompute full train distance matrix if small-ish, else local per point ----
        # We'll do local per-point (robust).
        eps_ridge = float(ridge) if ridge is not None else 0.0
        base_nug = float(nugget) if nugget is not None else 0.0

        for i in range(len(qx)):
            dx = x - qx[i]
            dy = y - qy[i]
            d = np.sqrt(dx * dx + dy * dy)
            # pick k nearest
            idx = np.argsort(d)[:k]
            xs = x[idx]
            ys = y[idx]
            zs = z_val[idx]
            ds = d[idx]

            # Build OK system: [C 1; 1^T 0] [w; mu] = [c; 1]
            # where C is covariance between train points, c covariance between train and query.
            # Diagonal: add nugget + optional obs sigma^2.
            k2 = len(idx)
            C = np.empty((k2, k2), dtype=float)
            for a in range(k2):
                da = xs[a] - xs
                db = ys[a] - ys
                hh = np.sqrt(da * da + db * db)
                C[a, :] = cov_from_h(hh)

            # Diagonal stabilizers
            diag_add = base_nug
            if use_obs_sigma and sig is not None:
                diag_add = diag_add + np.maximum(sig[idx], 0.0) ** 2

            C[np.diag_indices_from(C)] += diag_add
            if eps_ridge > 0.0:
                C[np.diag_indices_from(C)] += eps_ridge

            c = cov_from_h(ds)

            # Augment
            A = np.zeros((k2 + 1, k2 + 1), dtype=float)
            A[:k2, :k2] = C
            A[:k2, k2] = 1.0
            A[k2, :k2] = 1.0
            b = np.zeros(k2 + 1, dtype=float)
            b[:k2] = c
            b[k2] = 1.0

            try:
                sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # fallback least squares
                sol = np.linalg.lstsq(A, b, rcond=None)[0]

            w = sol[:k2]
            mu = sol[k2]

            pred[i] = float(np.sum(w * zs))

            # Kriging variance: C(0) - w^T c + mu
            # C(0) = sill (since covariance at zero is sill); nugget not included in prediction variance here.
            # (This matches typical OK variance forms; still useful for relative patterns.)
            var_i = float(sill) - float(np.dot(w, c)) + float(mu)
            pvar[i] = max(0.0, var_i)

        self._ok_cv_debug(debug, f"[OK-CV] predicted {len(pred)} test points (k={k}, model={model}, range_km={range_km})")
        return pred, pvar


    def ok_cv_evaluate(
        self,
        *,
        version,
        mode,
        holdout_frac=0.1,
        seed=42,
        # OK params
        variogram_model="exponential",
        range_km=80.0,
        sill=1.0,
        nugget=0.0001,
        ridge=0.0,
        neighbor_k=25,
        use_obs_sigma=True,
        # extent + data
        margin_deg=0.0,
        max_points=None,
        # CDI conditioning knobs
        cdi_trust_nresp_ge=3,
        cdi_enable_local_outlier=True,
        cdi_local_radius_km=25.0,
        cdi_outlier_k=2.5,
        cdi_min_neighbors=4,
        cdi_enable_clustering=True,
        cdi_cluster_eps_km=2.0,
        cdi_cluster_min_pts=3,
        # debug
        debug=False,
    ):
        """
        Holdout CV for a single OK scenario.
        Returns dict with metrics + bookkeeping.
        """
        import numpy as np
        import pandas as pd

        vkey = _norm_version(version)
        imt_target, obs_used = self._ok_cv_build_obs_for_mode(
            version=vkey,
            mode=mode,
            margin_deg=margin_deg,
            max_points=max_points,
            cdi_trust_nresp_ge=cdi_trust_nresp_ge,
            cdi_enable_local_outlier=cdi_enable_local_outlier,
            cdi_local_radius_km=cdi_local_radius_km,
            cdi_outlier_k=cdi_outlier_k,
            cdi_min_neighbors=cdi_min_neighbors,
            cdi_enable_clustering=cdi_enable_clustering,
            cdi_cluster_eps_km=cdi_cluster_eps_km,
            cdi_cluster_min_pts=cdi_cluster_min_pts,
            debug=debug,
        )

        if not isinstance(obs_used, pd.DataFrame) or obs_used.empty or len(obs_used) < 5:
            return {
                "ok": False,
                "version": vkey,
                "mode": str(mode),
                "imt": str(imt_target),
                "n_obs": int(len(obs_used)) if isinstance(obs_used, pd.DataFrame) else 0,
                "n_train": 0,
                "n_test": 0,
                "mae": None,
                "rmse": None,
                "bias": None,
                "wmae": None,
                "wrmse": None,
                "note": "Not enough observations after filtering/conditioning.",
            }

        train_df, test_df = self._ok_cv_split(obs_used, holdout_frac=holdout_frac, seed=seed)

        # if split fails (very small), bail gracefully
        if len(test_df) < 1 or len(train_df) < 3:
            return {
                "ok": False,
                "version": vkey,
                "mode": str(mode),
                "imt": str(imt_target),
                "n_obs": int(len(obs_used)),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "mae": None,
                "rmse": None,
                "bias": None,
                "wmae": None,
                "wrmse": None,
                "note": "Train/test split too small.",
            }

        pred, var = self._ok_cv_predict_at_points(
            train_df,
            test_df["lon"].to_numpy(dtype=float),
            test_df["lat"].to_numpy(dtype=float),
            variogram_model=variogram_model,
            range_km=range_km,
            sill=sill,
            nugget=nugget,
            ridge=ridge,
            neighbor_k=neighbor_k,
            use_obs_sigma=use_obs_sigma,
            debug=debug,
        )

        met = self._ok_cv_metrics(test_df["value"].to_numpy(dtype=float), pred, sigma_true=test_df.get("sigma"))
        out = {
            "ok": True,
            "version": vkey,
            "mode": str(mode),
            "imt": str(imt_target),
            "n_obs": int(len(obs_used)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "mae": met["mae"],
            "rmse": met["rmse"],
            "bias": met["bias"],
            "wmae": met["wmae"],
            "wrmse": met["wrmse"],
            "variogram_model": str(variogram_model).lower().strip(),
            "range_km": float(range_km),
            "sill": float(sill),
            "nugget": float(nugget),
            "ridge": float(ridge),
            "neighbor_k": None if neighbor_k is None else int(neighbor_k),
            "use_obs_sigma": bool(use_obs_sigma),
            "holdout_frac": float(holdout_frac),
            "seed": int(seed),
            "margin_deg": float(margin_deg),
            "max_points": None if max_points is None else int(max_points),
            "note": "",
        }
        self._ok_cv_debug(debug, f"[OK-CV] v={vkey} mode={mode} imt={imt_target} rmse={out['rmse']:.4g} mae={out['mae']:.4g} bias={out['bias']:.4g}")
        return out

    def ok_scenario_search(
        self,
        *,
        version,
        scenarios,
        # ranking
        rank_by="rmse",
        # CV
        holdout_frac=0.1,
        n_repeats=3,
        seed=42,
        # if True: run and store best scenario as a normal kriging result
        materialize_best=True,
        best_mode_key_prefix="auto_ok",
        # debug
        debug=False,
    ):
        """
        Evaluate many OK scenarios via holdout CV and rank them.

        scenarios: list of dicts. Each dict should include at least:
          - "mode": e.g., "mmi_cdi_filtered"
        Optional per-scenario keys:
          variogram_model, range_km, sill, nugget, ridge, neighbor_k, use_obs_sigma,
          margin_deg, max_points,
          CDI conditioning knobs:
            cdi_trust_nresp_ge, cdi_enable_local_outlier, cdi_local_radius_km, cdi_outlier_k,
            cdi_min_neighbors, cdi_enable_clustering, cdi_cluster_eps_km, cdi_cluster_min_pts

        Returns: dict with DataFrame "results" + "best" row + optional stored mode_key.
        Stores summary under:
          uq_state["versions"][version]["kriging_search"][search_key]
        """
        import numpy as np
        import pandas as pd
        import time

        vkey = _norm_version(version)
        t0 = time.time()

        if scenarios is None or len(scenarios) == 0:
            raise ValueError("scenarios is empty.")

        rank_by = str(rank_by).lower().strip()
        if rank_by not in ("rmse", "mae", "wrmse", "wmae", "abs_bias"):
            raise ValueError("rank_by must be one of: rmse, mae, wrmse, wmae, abs_bias")

        all_rows = []
        for si, sc in enumerate(scenarios):
            if not isinstance(sc, dict):
                continue
            mode = sc.get("mode")
            if mode is None:
                continue

            # repeat CV to reduce variance
            rep_rows = []
            for r in range(int(n_repeats)):
                row = self.ok_cv_evaluate(
                    version=vkey,
                    mode=mode,
                    holdout_frac=holdout_frac,
                    seed=int(seed) + 1000 * si + r,
                    variogram_model=sc.get("variogram_model", "exponential"),
                    range_km=sc.get("range_km", 80.0),
                    sill=sc.get("sill", 1.0),
                    nugget=sc.get("nugget", 0.0001),
                    ridge=sc.get("ridge", 0.0),
                    neighbor_k=sc.get("neighbor_k", 25),
                    use_obs_sigma=sc.get("use_obs_sigma", True),
                    margin_deg=sc.get("margin_deg", 0.0),
                    max_points=sc.get("max_points", None),
                    cdi_trust_nresp_ge=sc.get("cdi_trust_nresp_ge", 3),
                    cdi_enable_local_outlier=sc.get("cdi_enable_local_outlier", True),
                    cdi_local_radius_km=sc.get("cdi_local_radius_km", 25.0),
                    cdi_outlier_k=sc.get("cdi_outlier_k", 2.5),
                    cdi_min_neighbors=sc.get("cdi_min_neighbors", 4),
                    cdi_enable_clustering=sc.get("cdi_enable_clustering", True),
                    cdi_cluster_eps_km=sc.get("cdi_cluster_eps_km", 2.0),
                    cdi_cluster_min_pts=sc.get("cdi_cluster_min_pts", 3),
                    debug=debug,
                )
                rep_rows.append(row)

            # aggregate repeats (mean of metrics)
            df_rep = pd.DataFrame(rep_rows)
            ok_mask = df_rep["ok"].astype(bool)
            df_ok = df_rep[ok_mask].copy()
            if df_ok.empty:
                agg = rep_rows[0]
                agg["ok"] = False
                agg["note"] = "All repeats failed."
                agg["rep_rmse_mean"] = None
                agg["rep_rmse_std"] = None
                agg["rep_mae_mean"] = None
                agg["rep_mae_std"] = None
            else:
                def _meanstd(col):
                    v = pd.to_numeric(df_ok[col], errors="coerce").to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    if v.size == 0:
                        return None, None
                    return float(np.mean(v)), float(np.std(v))

                rm_m, rm_s = _meanstd("rmse")
                ma_m, ma_s = _meanstd("mae")
                wrm_m, wrm_s = _meanstd("wrmse")
                wma_m, wma_s = _meanstd("wmae")
                bi_m, bi_s = _meanstd("bias")

                agg = dict(df_ok.iloc[0].to_dict())
                agg["rep_rmse_mean"] = rm_m
                agg["rep_rmse_std"] = rm_s
                agg["rep_mae_mean"] = ma_m
                agg["rep_mae_std"] = ma_s
                agg["rep_wrmse_mean"] = wrm_m
                agg["rep_wrmse_std"] = wrm_s
                agg["rep_wmae_mean"] = wma_m
                agg["rep_wmae_std"] = wma_s
                agg["rep_bias_mean"] = bi_m
                agg["rep_bias_std"] = bi_s

            # keep original scenario descriptor for traceability
            agg["scenario_index"] = int(si)
            agg["scenario_dict"] = dict(sc)
            all_rows.append(agg)

        results = pd.DataFrame(all_rows)

        # ranking column
        if results.empty:
            raise RuntimeError("No scenario results produced.")

        # rank key
        if rank_by == "abs_bias":
            results["_rank_val"] = results["rep_bias_mean"].abs()
        elif rank_by == "rmse":
            results["_rank_val"] = results["rep_rmse_mean"]
        elif rank_by == "mae":
            results["_rank_val"] = results["rep_mae_mean"]
        elif rank_by == "wrmse":
            results["_rank_val"] = results["rep_wrmse_mean"]
        elif rank_by == "wmae":
            results["_rank_val"] = results["rep_wmae_mean"]
        else:
            results["_rank_val"] = results["rep_rmse_mean"]

        results = results.sort_values(["_rank_val"], ascending=True, na_position="last").reset_index(drop=True)

        best_row = results.iloc[0].to_dict()
        best_mode_key = None

        # store summary in uq_state
        vpack = (self.uq_state.get("versions") or {}).get(vkey)
        if vpack is None:
            raise KeyError(f"Version not in uq_state: {vkey}")

        if "kriging_search" not in vpack or not isinstance(vpack.get("kriging_search"), dict):
            vpack["kriging_search"] = {}

        search_key = f"ok_search__v{vkey}__rank_{rank_by}__h{float(holdout_frac):.2f}__r{int(n_repeats)}"
        vpack["kriging_search"][search_key] = {
            "rank_by": rank_by,
            "holdout_frac": float(holdout_frac),
            "n_repeats": int(n_repeats),
            "seed": int(seed),
            "results": results,
            "best": best_row,
            "elapsed_s": float(time.time() - t0),
        }

        # optionally materialize: run your normal grid OK for the best scenario
        if materialize_best:
            sc = best_row.get("scenario_dict") or {}
            mode = sc.get("mode", best_row.get("mode"))
            if mode is None:
                mode = best_row.get("mode")

            best_mode_key = f"{best_mode_key_prefix}__{rank_by}__{str(mode)}__v{vkey}"

            self._ok_cv_debug(debug, f"[OK-SEARCH] materialize best scenario -> mode_key={best_mode_key}")
            _ = self.run_ordinary_kriging(
                version=vkey,
                mode=mode,
                mode_key=best_mode_key,
                variogram_model=sc.get("variogram_model", best_row.get("variogram_model", "exponential")),
                range_km=sc.get("range_km", best_row.get("range_km", 80.0)),
                sill=sc.get("sill", best_row.get("sill", 1.0)),
                nugget=sc.get("nugget", best_row.get("nugget", 0.0001)),
                ridge=sc.get("ridge", best_row.get("ridge", 0.0)),
                neighbor_k=sc.get("neighbor_k", best_row.get("neighbor_k", 25)),
                max_points=sc.get("max_points", best_row.get("max_points", None)),
                use_obs_sigma=sc.get("use_obs_sigma", best_row.get("use_obs_sigma", True)),
                margin_deg=sc.get("margin_deg", best_row.get("margin_deg", 0.0)),
                cdi_trust_nresp_ge=sc.get("cdi_trust_nresp_ge", 3),
                cdi_enable_local_outlier=sc.get("cdi_enable_local_outlier", True),
                cdi_local_radius_km=sc.get("cdi_local_radius_km", 25.0),
                cdi_outlier_k=sc.get("cdi_outlier_k", 2.5),
                cdi_min_neighbors=sc.get("cdi_min_neighbors", 4),
                cdi_enable_clustering=sc.get("cdi_enable_clustering", True),
                cdi_cluster_eps_km=sc.get("cdi_cluster_eps_km", 2.0),
                cdi_cluster_min_pts=sc.get("cdi_cluster_min_pts", 3),
                store_obs_used=True,
                debug=debug,
            )

            vpack["kriging_search"][search_key]["best_mode_key"] = best_mode_key

        return {
            "search_key": search_key,
            "results": results,
            "best": best_row,
            "best_mode_key": best_mode_key,
        }






    # ======================================================================
    # PATCH: OK scenario-search CV — fix filter_observations_to_extent() call
    # - Your filter_observations_to_extent signature requires (version=...)
    # - It returns 1 df unless return_dropped=True
    # - This override fixes extent_first() to use return_dropped=True and
    #   constructs audit from kept/dropped lengths.
    # Paste at the END of class SHAKEuq.
    # ======================================================================
    
    def _ok_cv_build_obs_for_mode(
        self,
        *,
        version,
        mode,
        margin_deg=0.0,
        max_points=None,
        cdi_trust_nresp_ge=3,
        cdi_enable_local_outlier=True,
        cdi_local_radius_km=25.0,
        cdi_outlier_k=2.5,
        cdi_min_neighbors=4,
        cdi_enable_clustering=True,
        cdi_cluster_eps_km=2.0,
        cdi_cluster_min_pts=3,
        debug=False,
    ):
        import pandas as pd
        import numpy as np
    
        vkey = _norm_version(version)
        mode_s = str(mode).lower().strip()
    
        # ensure uq_state exists (unified extent required)
        uni = (self.uq_state.get("unified") or {}) if isinstance(self.uq_state, dict) else {}
        if not isinstance(uni, dict) or uni.get("lon2d") is None or uni.get("lat2d") is None:
            self.uq_build_dataset()
    
        def _dbg(msg):
            if debug:
                print(msg)
    
        def extent_first(df, tag):
            """
            Correct usage:
              kept, dropped = filter_observations_to_extent(..., return_dropped=True)
            """
            if not isinstance(df, pd.DataFrame) or df.empty:
                return df, {"total": 0, "kept": 0, "dropped": 0}
    
            kept, dropped = self.filter_observations_to_extent(
                df,
                version=vkey,
                grid_mode="unified",
                margin_deg=float(margin_deg),
                return_dropped=True,
            )
    
            total = int(len(df))
            nkept = int(len(kept))
            ndrop = int(len(dropped))
    
            if debug:
                by_src = {}
                if "source_detail" in kept.columns and len(kept):
                    by_src = dict(kept["source_detail"].value_counts())
                print(f"[OK-CV][{tag}] extent-first: total={total} kept={nkept} dropped={ndrop} margin_deg={margin_deg}")
                if by_src:
                    print(f"[OK-CV][{tag}] kept by source_detail: {by_src}")
    
            return kept, {"total": total, "kept": nkept, "dropped": ndrop}
    
        # ------------------------------------------------------------
        # Build obs per mode (must match your OK kriging modes naming)
        # ------------------------------------------------------------
        if mode_s == "pga_station":
            imt_target = "PGA"
            obs_used = self.build_observations(version=vkey, imt="PGA", dyfi_source="stationlist", sigma_override=None)
            obs_used, _ = extent_first(obs_used, "PGA_station")
    
        elif mode_s == "mmi_dyfi":
            imt_target = "MMI"
            obs_used = self.build_observations(version=vkey, imt="MMI", dyfi_source="stationlist", sigma_override=None)
            obs_used, _ = extent_first(obs_used, "MMI_dyfi")
    
        elif mode_s == "mmi_cdi_raw":
            imt_target = "MMI"
            obs_used = self.build_observations(version=vkey, imt="MMI", dyfi_source="cdi", sigma_override=None)
            obs_used, _ = extent_first(obs_used, "MMI_cdi_raw")
    
        elif mode_s == "mmi_cdi_filtered":
            imt_target = "MMI"
            obs_used = self.build_observations(version=vkey, imt="MMI", dyfi_source="cdi", sigma_override=None)
            obs_used, _ = extent_first(obs_used, "MMI_cdi_raw")
    
            n_before = int(len(obs_used))
    
            # condition CDI using your existing conditioning helper from the OK patch
            obs_used = self._cdi_condition(
                obs_used,
                trust_nresp_ge=int(cdi_trust_nresp_ge),
                enable_local_outlier=bool(cdi_enable_local_outlier),
                local_radius_km=float(cdi_local_radius_km),
                outlier_k=float(cdi_outlier_k),
                min_neighbors=int(cdi_min_neighbors),
                enable_clustering=bool(cdi_enable_clustering),
                cluster_eps_km=float(cdi_cluster_eps_km),
                cluster_min_pts=int(cdi_cluster_min_pts),
                debug=debug,
            )
    
            n_after = int(len(obs_used))
            _dbg(f"[OK-CV][MMI_cdi_filtered] conditioning: before={n_before} after={n_after}")
    
        elif mode_s == "pga_station_plus_dyfi_as_pga":
            imt_target = "PGA"
            sta = self.build_observations(version=vkey, imt="PGA", dyfi_source="stationlist", sigma_override=None)
            sta, _ = extent_first(sta, "PGA_station")
    
            dy = self.build_observations(version=vkey, imt="MMI", dyfi_source="stationlist", sigma_override=None)
            dy, _ = extent_first(dy, "MMI_dyfi_to_PGA")
    
            # convert DYFI(MMI) → PGA (Global GMICE helper from your OK patch)
            if isinstance(dy, pd.DataFrame) and len(dy):
                dy2 = dy.copy()
                dy2["value"] = self.gmice_convert(dy2["value"].to_numpy(), input_type="MMI", output_type="PGA")
                dy = dy2
    
            obs_used = pd.concat([sta, dy], ignore_index=True) if isinstance(sta, pd.DataFrame) else dy
            if max_points is not None and isinstance(obs_used, pd.DataFrame) and len(obs_used) > int(max_points):
                obs_used = obs_used.sample(int(max_points), random_state=0).reset_index(drop=True)
            _dbg(f"[OK-CV][mixed] PGA target: total={0 if obs_used is None else len(obs_used)}")
    
        elif mode_s == "mmi_dyfi_plus_station_as_mmi":
            imt_target = "MMI"
            dy = self.build_observations(version=vkey, imt="MMI", dyfi_source="stationlist", sigma_override=None)
            dy, _ = extent_first(dy, "MMI_dyfi")
    
            sta = self.build_observations(version=vkey, imt="PGA", dyfi_source="stationlist", sigma_override=None)
            sta, _ = extent_first(sta, "PGA_station_to_MMI")
    
            # convert station(PGA) → MMI (Global GMICE helper from your OK patch)
            if isinstance(sta, pd.DataFrame) and len(sta):
                sta2 = sta.copy()
                sta2["value"] = self.gmice_convert(sta2["value"].to_numpy(), input_type="PGA", output_type="MMI")
                sta = sta2
    
            obs_used = pd.concat([dy, sta], ignore_index=True) if isinstance(dy, pd.DataFrame) else sta
            if max_points is not None and isinstance(obs_used, pd.DataFrame) and len(obs_used) > int(max_points):
                obs_used = obs_used.sample(int(max_points), random_state=0).reset_index(drop=True)
            _dbg(f"[OK-CV][mixed] MMI target: total={0 if obs_used is None else len(obs_used)}")
    
        else:
            raise ValueError(f"Unknown OK-CV mode: {mode!r}")
    
        # optional downsample for speed (after all conditioning)
        if max_points is not None and isinstance(obs_used, pd.DataFrame) and len(obs_used) > int(max_points):
            obs_used = obs_used.sample(int(max_points), random_state=0).reset_index(drop=True)
            _dbg(f"[OK-CV] downsampled to max_points={max_points}")
    
        return imt_target, obs_used
    


    def ok_diagnostic_report(
        self,
        version,
        scenarios,
        rank_by="rmse",
        top_k=3,
        holdout_frac=0.1,
        n_repeats=3,
        seed=42,
        metric_scope="MMI",
        materialize_best=True,
        debug=False,
    ):
        """
        Run OK scenario search + produce diagnostic plots.
    
        Returns
        -------
        report : dict
            {
                "results": DataFrame,
                "best_row": Series,
                "best_mode_key": str,
                "figures": dict
            }
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
    
        # --------------------------------------------------
        # 1) Run scenario search
        # --------------------------------------------------
        out = self.ok_scenario_search(
            version=version,
            scenarios=scenarios,
            rank_by=rank_by,
            holdout_frac=holdout_frac,
            n_repeats=n_repeats,
            seed=seed,
            materialize_best=materialize_best,
            best_mode_key_prefix="diag_ok",
            debug=debug,
        )
    
        results = out["results"].copy()
        results = results[results["ok"] == True]
    
        if results.empty:
            raise RuntimeError("No valid OK scenarios to diagnose.")
    
        # --------------------------------------------------
        # 2) Rank + select top-k
        # --------------------------------------------------
        results = results.sort_values(rank_by, ascending=True)
        top = results.head(int(top_k))
        best = results.iloc[0]
    
        # --------------------------------------------------
        # 3) Plot A: performance comparison
        # --------------------------------------------------
        fig_perf, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top["mode"], top[rank_by])
        ax.invert_yaxis()
        ax.set_xlabel(rank_by.upper())
        ax.set_title(f"Top-{len(top)} OK methods (version {version})")
        ax.grid(alpha=0.3)
    
        # --------------------------------------------------
        # 4) Plot B: predicted vs observed (if stored)
        # --------------------------------------------------
        fig_scatter = None
        if "pred_vs_obs" in out:
            df = out["pred_vs_obs"]
            fig_scatter, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(df["obs"], df["pred"], alpha=0.7)
            lims = [
                min(df["obs"].min(), df["pred"].min()),
                max(df["obs"].max(), df["pred"].max()),
            ]
            ax.plot(lims, lims, "k--", lw=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.set_title("Held-out prediction check")
            ax.grid(alpha=0.3)
    
        # --------------------------------------------------
        # 5) Plot C: map comparison (best only)
        # --------------------------------------------------
        fig_map = None
        if materialize_best:
            vkey = str(version)
            mode_key = out.get("best_mode_key")
            if mode_key:
                prior = self.uq_state["versions"][vkey]["grid"]["fields"][metric_scope]
                ok_mean = self.uq_state["versions"][vkey]["kriging"][mode_key]["mean"]
    
                diff = ok_mean - prior
    
                fig_map, axs = plt.subplots(1, 3, figsize=(15, 4))
                for a, data, title in zip(
                    axs,
                    [prior, ok_mean, diff],
                    ["Prior", "OK mean", "OK − Prior"],
                ):
                    im = a.imshow(data, origin="lower")
                    a.set_title(title)
                    plt.colorbar(im, ax=a, fraction=0.046)
    
        return {
            "results": results,
            "best_row": best,
            "best_mode_key": out.get("best_mode_key"),
            "figures": {
                "performance": fig_perf,
                "scatter": fig_scatter,
                "map": fig_map,
            },
        }



    # ======================================================================
    # PATCH: prior-on-target-grid helper (fixes audit shape mismatches)
    # Paste INSIDE class SHAKEuq (indent by 4 spaces)
    # ======================================================================
    def get_prior_mean_on_grid(
        self,
        *,
        version,
        imt: str,
        target: str = "unified",  # "unified" | "native" | "ok_result"
        ok_mode_key: str = None,  # used when target="ok_result"
        debug: bool = False,
    ):
        """
        Return (lon2d, lat2d, prior2d) where prior2d is the ShakeMap mean field
        aligned to the requested target grid.

        - target="unified": uses uq_state["unified"] grid, and prefers unified stacked prior.
        - target="native" : uses per-version native grid (no remap).
        - target="ok_result": uses the grid shape of stored OK result (mode key),
                              and remaps prior onto that grid if needed.

        This is specifically designed to prevent:
            OK mean shape != prior shape
        in audit/plot routines.
        """
        imt_u = str(imt).upper().strip()
        vkey = _norm_version(version)

        if not isinstance(self.uq_state, dict) or not (self.uq_state.get("versions") or {}):
            raise RuntimeError("uq_state not initialized. Run uq_build_dataset() first.")
        if vkey not in (self.uq_state.get("versions") or {}):
            raise KeyError(f"Version not found in uq_state: {vkey}")

        def _dbg(msg):
            if bool(debug) or bool(getattr(self, "verbose", False)):
                print(msg)

        # ---------- pull native prior (if available) ----------
        vpack = (self.uq_state.get("versions") or {}).get(vkey) or {}
        grid = vpack.get("grid") or {}
        native_lon2d = grid.get("lon2d")
        native_lat2d = grid.get("lat2d")
        native_prior = None
        if isinstance(grid.get("fields"), dict):
            native_prior = grid["fields"].get(imt_u)

        # ---------- unified grid (if available) ----------
        uni = self.uq_state.get("unified") or {}
        uni_lon2d = uni.get("lon2d")
        uni_lat2d = uni.get("lat2d")

        # ---------- helper: get unified-stacked prior if it exists ----------
        def _prior_from_unified_stack():
            try:
                vkeys = list(uni.get("version_keys") or [])
                stack = (uni.get("fields") or {}).get(imt_u)
                if stack is None:
                    return None
                if vkey not in vkeys:
                    return None
                i = vkeys.index(vkey)
                return np.asarray(stack[i], dtype=float)
            except Exception:
                return None

        # ---------- choose target grid ----------
        if target == "native":
            if native_lon2d is None or native_lat2d is None or native_prior is None:
                return None, None, None
            return native_lon2d, native_lat2d, np.asarray(native_prior, dtype=float)

        if target == "unified":
            if uni_lon2d is None or uni_lat2d is None:
                raise RuntimeError("Unified grid missing. Run uq_build_dataset() first.")

            # best case: already have unified-stacked prior
            prior_u = _prior_from_unified_stack()
            if prior_u is not None and prior_u.shape == uni_lon2d.shape:
                return uni_lon2d, uni_lat2d, prior_u

            # fallback: remap native prior -> unified grid
            if native_lon2d is None or native_lat2d is None or native_prior is None:
                return uni_lon2d, uni_lat2d, None

            native_prior = np.asarray(native_prior, dtype=float)
            if native_prior.shape == uni_lon2d.shape:
                return uni_lon2d, uni_lat2d, native_prior

            _dbg(f"[SHAKEuq][prior remap] v={vkey} imt={imt_u} native={native_prior.shape} -> unified={uni_lon2d.shape}")
            remapped = self._nn_remap_to_ref(
                native_lon2d, native_lat2d, native_prior,
                uni_lon2d, uni_lat2d
            )
            return uni_lon2d, uni_lat2d, np.asarray(remapped, dtype=float)

        if target == "ok_result":
            if not ok_mode_key:
                raise ValueError("target='ok_result' requires ok_mode_key.")

            # get OK result
            kpack = (vpack.get("kriging") or {}).get(ok_mode_key)
            if not isinstance(kpack, dict):
                raise KeyError(f"No OK/kriging result found for v={vkey} mode_key={ok_mode_key}")

            ok_mean = kpack.get("mean_grid")
            if ok_mean is None:
                return None, None, None
            ok_mean = np.asarray(ok_mean, dtype=float)
            ok_shape = ok_mean.shape

            # determine lon/lat for that OK result:
            #  - Prefer unified lon/lat if it matches OK shape
            if uni_lon2d is not None and np.asarray(uni_lon2d).shape == ok_shape:
                tgt_lon2d = uni_lon2d
                tgt_lat2d = uni_lat2d
            #  - Else fall back to native lon/lat if it matches OK shape
            elif native_lon2d is not None and np.asarray(native_lon2d).shape == ok_shape:
                tgt_lon2d = native_lon2d
                tgt_lat2d = native_lat2d
            else:
                # cannot infer lon/lat reliably
                tgt_lon2d = uni_lon2d
                tgt_lat2d = uni_lat2d

            # get prior on unified first (best provenance), then remap if needed
            if uni_lon2d is not None and uni_lat2d is not None:
                _, _, prior_u = self.get_prior_mean_on_grid(version=vkey, imt=imt_u, target="unified", debug=debug)
            else:
                prior_u = None

            if prior_u is not None and tgt_lon2d is not None and np.asarray(tgt_lon2d).shape == ok_shape:
                # if prior_u already matches ok shape, done
                if np.asarray(prior_u).shape == ok_shape:
                    return tgt_lon2d, tgt_lat2d, np.asarray(prior_u, dtype=float)

                # if prior_u is on unified but OK is not, remap prior_u -> OK grid using OK grid lon/lat
                if uni_lon2d is not None and uni_lat2d is not None and np.asarray(prior_u).shape == np.asarray(uni_lon2d).shape:
                    if tgt_lon2d is not None and tgt_lat2d is not None and np.asarray(tgt_lon2d).shape == ok_shape:
                        _dbg(f"[SHAKEuq][prior remap] v={vkey} imt={imt_u} unified={np.asarray(prior_u).shape} -> ok={ok_shape}")
                        remapped = self._nn_remap_to_ref(
                            uni_lon2d, uni_lat2d, np.asarray(prior_u, dtype=float),
                            tgt_lon2d, tgt_lat2d
                        )
                        return tgt_lon2d, tgt_lat2d, np.asarray(remapped, dtype=float)

            # final fallback: native prior -> ok grid (if possible)
            if native_lon2d is not None and native_lat2d is not None and native_prior is not None and tgt_lon2d is not None and tgt_lat2d is not None:
                native_prior = np.asarray(native_prior, dtype=float)
                if native_prior.shape == ok_shape:
                    return tgt_lon2d, tgt_lat2d, native_prior
                if np.asarray(tgt_lon2d).shape == ok_shape:
                    _dbg(f"[SHAKEuq][prior remap] v={vkey} imt={imt_u} native={native_prior.shape} -> ok={ok_shape}")
                    remapped = self._nn_remap_to_ref(
                        native_lon2d, native_lat2d, native_prior,
                        tgt_lon2d, tgt_lat2d
                    )
                    return tgt_lon2d, tgt_lat2d, np.asarray(remapped, dtype=float)

            return tgt_lon2d, tgt_lat2d, None

        raise ValueError("target must be one of: 'unified', 'native', 'ok_result'")




    # ======================================================================
    # OK audit plotting (inside-class, no monkey patch)
    # - Discrete mean palettes for MMI/PGA via contour_scale() (if available)
    # - Continuous variance colormap
    # - plot_oksearch_audit(): plot Nth-best from ok_scenario_search() output
    # ======================================================================

    def _mean_cmap_for_imt(self, imt_u, *, scale_type="usgs", pga_units="%g"):
        """
        Returns (cmap, norm, ticks, label) for MEAN plots.
        Prefers self.contour_scale() if available, otherwise falls back to continuous.
        """
        import matplotlib as mpl

        imt_u = str(imt_u).upper().strip()

        # Prefer your discrete contour_scale if it exists
        if hasattr(self, "contour_scale") and callable(getattr(self, "contour_scale")):
            try:
                if imt_u == "PGA":
                    cmap, bounds, ticks, norm, label = self.contour_scale(
                        "PGA", scale_type=scale_type, units=pga_units
                    )
                    return cmap, norm, ticks, label
                if imt_u == "MMI":
                    cmap, bounds, ticks, norm, label = self.contour_scale(
                        "MMI", scale_type=scale_type, units="MMI"
                    )
                    return cmap, norm, ticks, label

                # generic attempt
                cmap, bounds, ticks, norm, label = self.contour_scale(
                    imt_u, scale_type=scale_type, units=None
                )
                return cmap, norm, ticks, label
            except Exception:
                pass

        # Fallback: continuous
        cmap = mpl.cm.get_cmap("viridis")
        return cmap, None, None, f"{imt_u} (continuous)"

    def _plot_grid_panel(
        self,
        ax,
        lon2d,
        lat2d,
        Z,
        *,
        title,
        cmap,
        norm=None,
        vmin=None,
        vmax=None,
        add_colorbar=True,
        cbar_ticks=None,
        cbar_label=None,
        fig=None,
    ):
        import numpy as np

        if Z is None:
            ax.set_title(title + " (missing)")
            ax.set_axis_off()
            return None

        Z = np.asarray(Z, dtype=float)

        if norm is not None:
            m = ax.pcolormesh(lon2d, lat2d, Z, shading="auto", cmap=cmap, norm=norm)
        else:
            m = ax.pcolormesh(lon2d, lat2d, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(title)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")

        if add_colorbar and fig is not None:
            cb = fig.colorbar(m, ax=ax, shrink=0.85)
            if cbar_ticks is not None:
                cb.set_ticks(cbar_ticks)
            if cbar_label:
                cb.set_label(cbar_label)

        return m

    def plot_kriging_audit(
        self,
        *,
        version,
        mode_key,
        show=True,
        save_path=None,
        dpi=150,
        figsize=(12, 9),
        obs_size=14,
        mean_scale_type="usgs",
        mean_pga_units="%g",
        var_cmap="viridis",
        var_vmin=None,
        var_vmax=None,
        debug=False,
    ):
        """
        UPDATED OK audit plot (2×2):
          (1) prior ShakeMap mean — discrete palette if MMI/PGA
          (2) OK mean            — discrete palette if MMI/PGA
          (3) OK variance        — continuous cmap (user controls)
          (4) obs points on prior
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        vkey = _norm_version(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey) or {}
        kpack = (vpack.get("kriging") or {}).get(mode_key)
        if not isinstance(kpack, dict):
            raise KeyError(f"No kriging result for v={vkey} mode_key={mode_key}")

        meta = kpack.get("meta") or {}
        imt_u = str(meta.get("imt", "MMI")).upper().strip()

        lon2d, lat2d = self._get_unified_grid()

        # Your storage schema is mean_grid / var_grid
        ok_mean = np.asarray(kpack.get("mean_grid"), dtype=float)
        ok_var = np.asarray(kpack.get("var_grid"), dtype=float)

        if ok_mean.shape != lon2d.shape:
            raise RuntimeError(f"OK mean shape {ok_mean.shape} != unified grid shape {lon2d.shape}")
        if ok_var.shape != lon2d.shape:
            raise RuntimeError(f"OK var shape {ok_var.shape} != unified grid shape {lon2d.shape}")

        # Prior on unified grid (may be None if something odd happens)
        prior = None
        try:
            _, _, prior = self._get_prior_mean_unified(vkey, imt_u)
        except Exception:
            prior = None

        if prior is not None:
            prior = np.asarray(prior, dtype=float)
            if prior.shape != ok_mean.shape:
                # don't crash — still plot OK
                prior = None

        obs = kpack.get("obs_used")
        n_obs = int(len(obs)) if isinstance(obs, pd.DataFrame) else 0

        if debug or bool(getattr(self, "verbose", False)):
            self._ok_debug(True, f"[OK][PLOT] v={vkey} mode_key={mode_key} imt={imt_u} n_obs={n_obs}")

        cmap_mean, norm_mean, ticks_mean, label_mean = self._mean_cmap_for_imt(
            imt_u, scale_type=mean_scale_type, pga_units=mean_pga_units
        )

        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
        ax11, ax12 = axes[0, 0], axes[0, 1]
        ax21, ax22 = axes[1, 0], axes[1, 1]

        # (1) Prior mean
        self._plot_grid_panel(
            ax11, lon2d, lat2d, prior,
            title=f"Prior mean (ShakeMap) — {imt_u} — v={vkey}",
            cmap=cmap_mean, norm=norm_mean,
            add_colorbar=True, cbar_ticks=ticks_mean, cbar_label=label_mean,
            fig=fig,
        )

        # (2) OK mean
        self._plot_grid_panel(
            ax12, lon2d, lat2d, ok_mean,
            title=f"OK mean — {imt_u} — {mode_key}",
            cmap=cmap_mean, norm=norm_mean,
            add_colorbar=True, cbar_ticks=ticks_mean, cbar_label=label_mean,
            fig=fig,
        )

        # (3) OK variance (continuous)
        self._plot_grid_panel(
            ax21, lon2d, lat2d, ok_var,
            title="OK variance (continuous)",
            cmap=var_cmap, norm=None, vmin=var_vmin, vmax=var_vmax,
            add_colorbar=True, cbar_ticks=None, cbar_label="OK variance",
            fig=fig,
        )

        # (4) Obs points (background prior if available, else OK mean)
        bg = prior if prior is not None else ok_mean
        if norm_mean is not None:
            ax22.pcolormesh(lon2d, lat2d, bg, shading="auto", cmap=cmap_mean, norm=norm_mean)
        else:
            ax22.pcolormesh(lon2d, lat2d, bg, shading="auto", cmap=cmap_mean)

        ax22.set_title(f"Obs used (n={n_obs}) — {imt_u}")
        ax22.set_xlabel("Lon")
        ax22.set_ylabel("Lat")

        if isinstance(obs, pd.DataFrame) and (not obs.empty) and {"lon", "lat"}.issubset(obs.columns):
            ax22.scatter(obs["lon"], obs["lat"], s=obs_size, alpha=0.9)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()

        return {"fig": fig, "version": vkey, "mode_key": mode_key, "imt": imt_u, "n_obs": n_obs}

    def plot_oksearch_audit(
        self,
        *,
        version,
        ok_search_out,
        rank=1,
        metric="rmse",
        only_ok=True,
        show=True,
        save_path=None,
        dpi=150,
        figsize=(12, 9),
        obs_size=14,
        mean_scale_type="usgs",
        mean_pga_units="%g",
        var_cmap="viridis",
        var_vmin=None,
        var_vmax=None,
        debug=False,
    ):
        """
        Convenience: after ok_scenario_search(), plot the Nth-best scenario result.
        rank=1 -> best, rank=2 -> second best, ...
        """
        import pandas as pd

        vkey = _norm_version(version)

        if not isinstance(ok_search_out, dict) or "results" not in ok_search_out:
            raise TypeError("ok_search_out must be the dict returned by ok_scenario_search() (must include 'results').")

        df = ok_search_out["results"]
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("ok_search_out['results'] is empty or not a DataFrame.")

        d = df.copy()
        if only_ok and "ok" in d.columns:
            d = d[d["ok"] == True].copy()

        if metric not in d.columns:
            raise KeyError(f"metric '{metric}' not in results columns: {list(d.columns)}")

        d = d.sort_values(metric, ascending=True).reset_index(drop=True)
        if d.empty:
            raise RuntimeError("No rows available to rank (try only_ok=False).")

        r = int(rank)
        if r < 1 or r > len(d):
            raise ValueError(f"rank must be in [1, {len(d)}], got {rank}")

        chosen = d.iloc[r - 1].to_dict()

        # best_mode_key may be present per-row or only at top-level; try both
        mode_key = None
        if "best_mode_key" in chosen and isinstance(chosen["best_mode_key"], str) and chosen["best_mode_key"]:
            mode_key = chosen["best_mode_key"]
        elif r == 1 and isinstance(ok_search_out.get("best_mode_key"), str) and ok_search_out.get("best_mode_key"):
            mode_key = ok_search_out.get("best_mode_key")

        # If still missing, infer from your ok_scenario_search naming convention:
        # auto_ok__<metric>__<mode>__v<ver>
        if not mode_key:
            mode = str(chosen.get("mode", "") or "").strip()
            mode_key = f"auto_ok__{metric}__{mode}__v{vkey}"

        if debug or bool(getattr(self, "verbose", False)):
            self._ok_debug(True, f"[OK-SEARCH][PLOT] v={vkey} rank={r} metric={metric} -> mode_key={mode_key}")

        audit = self.plot_kriging_audit(
            version=vkey,
            mode_key=mode_key,
            show=show,
            save_path=save_path,
            dpi=dpi,
            figsize=figsize,
            obs_size=obs_size,
            mean_scale_type=mean_scale_type,
            mean_pga_units=mean_pga_units,
            var_cmap=var_cmap,
            var_vmin=var_vmin,
            var_vmax=var_vmax,
            debug=debug,
        )

        return {
            "rank": r,
            "metric": metric,
            "row": chosen,
            "mode_key": mode_key,
            "audit": audit,
        }




    def materialize_oksearch_rank(
        self,
        *,
        version,
        ok_search_out,
        rank=1,
        metric="rmse",
        debug=False,
    ):
        """
        Materialize (run + store) the kriging result for the Nth-ranked
        OK-search scenario.
        """
        import pandas as pd
    
        vkey = _norm_version(version)
    
        df = ok_search_out["results"].copy()
        if "ok" in df.columns:
            df = df[df["ok"] == True]
    
        df = df.sort_values(metric, ascending=True).reset_index(drop=True)
    
        if rank < 1 or rank > len(df):
            raise ValueError(f"rank must be in [1, {len(df)}]")
    
        row = df.iloc[rank - 1].to_dict()
        mode = row["mode"]
    
        # reconstruct scenario kwargs
        scenario_kwargs = {}
        for k in row:
            if k.startswith("cdi_") or k in [
                "variogram_model",
                "range_km",
                "sill",
                "nugget",
                "ridge",
                "neighbor_k",
                "use_obs_sigma",
                "margin_deg",
                "max_points",
            ]:
                scenario_kwargs[k] = row[k]
    
        mode_key = f"auto_ok__{metric}__{mode}__v{vkey}"
    
        if debug:
            print(f"[OK-SEARCH] materializing rank={rank} mode={mode}")
            print("kwargs:", scenario_kwargs)
    
        self.run_ordinary_kriging(
            version=vkey,
            mode=mode,
            mode_key_override=mode_key,
            **scenario_kwargs,
            debug=debug,
        )
    
        return mode_key
    

    # ======================================================================
    # Bayesian Update (Design A/B): local precision fusion on unified grid
    # - Prior is ALWAYS taken from the FIRST version in unified["version_keys"]
    # - Observations are taken from the TARGET version (per your time-evolution design)
    # - Supports direct + mixed (GMICE) modes and Bayes2 (sequential trust)
    #
    # Paste INSIDE class SHAKEuq (indent by 4 spaces) — at the END of the class.
    # ======================================================================

    # ---------------------------
    # Unified prior (fixed v0) helpers
    # ---------------------------

    def _bayes_get_prior_version_key(self) -> str:
        uni = (self.uq_state.get("unified") or {})
        vkeys = list(uni.get("version_keys") or [])
        if not vkeys:
            raise RuntimeError("Unified version_keys missing. Run uq_build_dataset() first.")
        return _norm_version(vkeys[0])

    def _bayes_sigma_key_for_imt(self, imt: str) -> str:
        """
        Map IMT -> sigma field key in unified["sigma"].
        USGS-style uncertainty keys are typically STDMMI / STDPGA / STD...
        """
        imt_u = str(imt).upper().strip()
        if imt_u == "MMI":
            return "STDMMI"
        if imt_u == "PGA":
            return "STDPGA"
        if imt_u == "PGV":
            return "STDPGV"
        # fallback: prefer configured prefix (default "STD")
        pref = str(getattr(self, "prefer_sigma_field_prefix", "STD") or "STD").upper().strip()
        return f"{pref}{imt_u}"

    def _get_prior_sigma_unified(self, version, imt):
        vkey = _norm_version(version)
        imt_u = str(imt).upper().strip()
        sig_key = self._bayes_sigma_key_for_imt(imt_u)

        lon2d, lat2d = self._get_unified_grid()
        uni = (self.uq_state.get("unified") or {})
        vkeys = list(uni.get("version_keys") or [])
        stack = (uni.get("sigma") or {}).get(sig_key)

        if stack is None or vkey not in vkeys:
            return lon2d, lat2d, None

        i = vkeys.index(vkey)
        return lon2d, lat2d, np.asarray(stack[i], dtype=float)

    def _bayes_working_space(self, imt: str) -> str:
        """
        Working-space convention:
          - MMI: linear
          - PGA: log  (since STDPGA is ln(g) in uncertainty.xml / synthetic)
        """
        imt_u = str(imt).upper().strip()
        if imt_u == "PGA":
            return "log"
        return "linear"

    def _bayes_to_working(self, imt: str, values: np.ndarray) -> np.ndarray:
        imt_u = str(imt).upper().strip()
        ws = self._bayes_working_space(imt_u)
        v = np.asarray(values, dtype=float)
        if ws == "log":
            # values expected positive; clamp to avoid -inf
            return np.log(np.maximum(v, 1e-12))
        return v

    def _bayes_from_working(self, imt: str, values_ws: np.ndarray) -> np.ndarray:
        imt_u = str(imt).upper().strip()
        ws = self._bayes_working_space(imt_u)
        v = np.asarray(values_ws, dtype=float)
        if ws == "log":
            return np.exp(v)
        return v

    def _bayes_km_per_deg(self, lat0: float):
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * float(np.cos(np.deg2rad(lat0)))
        return km_per_deg_lon, km_per_deg_lat

    def _bayes_lonlat_to_xy_km(self, lon: np.ndarray, lat: np.ndarray, *, lon0=None, lat0=None):
        """
        Lightweight local equirectangular projection in km (consistent with OK).
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if lon0 is None:
            lon0 = float(np.nanmean(lon))
        if lat0 is None:
            lat0 = float(np.nanmean(lat))
        km_per_deg_lon, km_per_deg_lat = self._bayes_km_per_deg(lat0)
        x = (lon - lon0) * km_per_deg_lon
        y = (lat - lat0) * km_per_deg_lat
        return x, y, float(lon0), float(lat0)

    def _bayes_kernel_weights(self, d_km: np.ndarray, *, kernel: str, kernel_scale_km: float) -> np.ndarray:
        d = np.asarray(d_km, dtype=float)
        s = max(float(kernel_scale_km), 1e-9)
        k = str(kernel).lower().strip()
        if k in ("gauss", "gaussian"):
            w = np.exp(-0.5 * (d / s) ** 2)
        elif k in ("exp", "exponential"):
            w = np.exp(-(d / s))
        elif k in ("cauchy",):
            w = 1.0 / (1.0 + (d / s) ** 2)
        else:
            # default gaussian
            w = np.exp(-0.5 * (d / s) ** 2)
        # protect
        w[~np.isfinite(w)] = 0.0
        return w

    def _bayes_apply_sigma_scales(self, obs: pd.DataFrame, *, sigma_scale: float = 1.0) -> pd.DataFrame:
        """
        Apply a global sigma scaling (>=1 reduces trust; <1 increases trust).
        """
        if not isinstance(obs, pd.DataFrame) or obs.empty:
            return obs
        out = obs.copy()
        s = pd.to_numeric(out.get("sigma", np.nan), errors="coerce").to_numpy(dtype=float)
        s = np.where(np.isfinite(s), s, np.nan)
        # if sigma missing, leave NaN and later fill with default
        out["sigma"] = s * float(sigma_scale)
        return out

    # ---------------------------
    # Core local precision-fusion update on unified grid
    # ---------------------------

    def _bayes_local_precision_fusion_grid(
        self,
        *,
        imt_target: str,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        prior_mean: np.ndarray,
        prior_sigma: np.ndarray,
        obs_df: pd.DataFrame,
        update_radius_km: float = 40.0,
        kernel: str = "gaussian",
        kernel_scale_km: float = 25.0,
        neighbor_k: Optional[int] = 50,
        default_obs_sigma: float = 0.5,
        debug: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Conjugate Gaussian update per grid cell using distance-weighted observation precisions.

        Returns
        -------
        post_mean : 2D array (unified grid)
        post_sigma: 2D array (unified grid)  (same "space" as prior_sigma: linear for MMI, ln-space for PGA)
        audit : dict (counts, etc.)
        """
        import numpy as np
        import pandas as pd

        if prior_mean is None or prior_sigma is None:
            raise RuntimeError("Prior mean/sigma missing. Check unified stacks and sigma key mapping.")

        if not isinstance(obs_df, pd.DataFrame) or obs_df.empty:
            # no update possible; return prior
            return (
                np.asarray(prior_mean, dtype=float).copy(),
                np.asarray(prior_sigma, dtype=float).copy(),
                {"n_obs": 0, "note": "obs empty -> posterior=prior"},
            )

        df = obs_df.copy()
        df = df.dropna(subset=["lon", "lat", "value"])
        if df.empty:
            return (
                np.asarray(prior_mean, dtype=float).copy(),
                np.asarray(prior_sigma, dtype=float).copy(),
                {"n_obs": 0, "note": "obs dropna -> empty -> posterior=prior"},
            )

        # working space conversion
        mu0_ws = self._bayes_to_working(imt_target, np.asarray(prior_mean, dtype=float))
        # prior_sigma is already "working-space sigma" for PGA (ln), linear for MMI
        sig0_ws = np.asarray(prior_sigma, dtype=float)

        # obs: convert values to working space (PGA -> ln)
        y_ws = self._bayes_to_working(imt_target, df["value"].to_numpy(dtype=float))

        # obs sigma: use provided; fill missing with default
        if "sigma" in df.columns:
            s = pd.to_numeric(df["sigma"], errors="coerce").to_numpy(dtype=float)
        else:
            s = np.full(len(df), np.nan, dtype=float)
        s = np.where(np.isfinite(s), s, np.nan)
        s = np.where(np.isfinite(s), s, float(default_obs_sigma))
        s = np.maximum(s, 1e-6)  # protect
        tau_obs = 1.0 / (s ** 2)

        # project obs and target in km
        x_obs, y_obs, lon0, lat0 = self._bayes_lonlat_to_xy_km(df["lon"].to_numpy(dtype=float), df["lat"].to_numpy(dtype=float))
        x_tgt, y_tgt, _, _ = self._bayes_lonlat_to_xy_km(lon2d, lat2d, lon0=lon0, lat0=lat0)

        post_mu_ws = np.full_like(mu0_ws, np.nan, dtype=float)
        post_sig_ws = np.full_like(sig0_ws, np.nan, dtype=float)

        # flattened targets for brute force
        xt = x_tgt.ravel()
        yt = y_tgt.ravel()

        rmax = float(update_radius_km)
        r2max = rmax * rmax

        k_nn = None if (neighbor_k is None) else max(2, int(neighbor_k))

        used_counts = 0
        used_sum = 0

        # loop over all grid cells (same style as OK)
        for idx in range(xt.size):
            xi = float(xt[idx])
            yi = float(yt[idx])

            d2 = (x_obs - xi) ** 2 + (y_obs - yi) ** 2
            # radius gate
            inside = d2 <= r2max
            if not np.any(inside):
                # no nearby obs -> posterior = prior
                mu = float(mu0_ws.ravel()[idx])
                s0 = float(sig0_ws.ravel()[idx])
                post_mu_ws.ravel()[idx] = mu
                post_sig_ws.ravel()[idx] = s0
                continue

            nn_idx = np.where(inside)[0]
            if k_nn is not None and len(nn_idx) > k_nn:
                # take closest k within radius
                d2_inside = d2[nn_idx]
                order = np.argsort(d2_inside)[:k_nn]
                nn_idx = nn_idx[order]

            d_km = np.sqrt(d2[nn_idx])
            w = self._bayes_kernel_weights(d_km, kernel=kernel, kernel_scale_km=float(kernel_scale_km))
            if not np.isfinite(w).any() or float(np.sum(w)) <= 0.0:
                mu = float(mu0_ws.ravel()[idx])
                s0 = float(sig0_ws.ravel()[idx])
                post_mu_ws.ravel()[idx] = mu
                post_sig_ws.ravel()[idx] = s0
                continue

            # weighted precision fusion
            mu0 = float(mu0_ws.ravel()[idx])
            sig0 = float(sig0_ws.ravel()[idx])
            sig0 = max(sig0, 1e-9)
            tau0 = 1.0 / (sig0 * sig0)

            tau_eff = tau_obs[nn_idx] * w
            tau_post = tau0 + float(np.sum(tau_eff))

            # posterior mean in working space
            num = tau0 * mu0 + float(np.sum(tau_eff * y_ws[nn_idx]))
            mu_post = num / tau_post
            sig_post = math.sqrt(1.0 / tau_post)

            post_mu_ws.ravel()[idx] = mu_post
            post_sig_ws.ravel()[idx] = sig_post

            used_counts += 1
            used_sum += int(len(nn_idx))

        audit = {
            "n_obs_total": int(len(df)),
            "n_grid_updated": int(used_counts),
            "avg_neighbors_used": float(used_sum / max(used_counts, 1)),
            "update_radius_km": float(update_radius_km),
            "kernel": str(kernel),
            "kernel_scale_km": float(kernel_scale_km),
            "neighbor_k": None if neighbor_k is None else int(neighbor_k),
            "working_space": self._bayes_working_space(imt_target),
        }

        # convert posterior mean back to output space (MMI stays, PGA -> exp)
        post_mean = self._bayes_from_working(imt_target, post_mu_ws)
        post_sigma = post_sig_ws  # kept in the sigma's native convention (MMI linear, PGA ln)

        return np.asarray(post_mean, dtype=float), np.asarray(post_sigma, dtype=float), audit

    # ---------------------------
    # Mode builder: build obs for Bayes modes (direct + mixed)
    # ---------------------------

    def _bayes_extent_first(self, obs: pd.DataFrame, *, version: str, margin_deg: float, label: str, debug: bool):
        kept, dropped = self.filter_observations_to_extent(
            obs,
            version=version,
            grid_mode="unified",
            margin_deg=float(margin_deg),
            return_dropped=True,
        )
        if debug or getattr(self, "verbose", False):
            try:
                print(f"[BAYES][{label}] extent-first: total={len(obs)} kept={len(kept)} dropped={len(dropped)} margin_deg={margin_deg}")
                if isinstance(kept, pd.DataFrame) and not kept.empty and "source_detail" in kept.columns:
                    c = kept["source_detail"].value_counts(dropna=False).to_dict()
                    print(f"[BAYES][{label}] kept by source_detail: {c}")
            except Exception:
                pass
        return kept, dropped

    def _bayes_build_obs_for_mode(
        self,
        *,
        version: Union[str, int],
        mode: str,
        margin_deg: float = 0.0,
        # CDI conditioning knobs (same family as OK)
        cdi_enable_local_outlier: bool = True,
        cdi_local_radius_km: float = 20.0,
        cdi_outlier_k: float = 3.5,
        cdi_min_neighbors: int = 5,
        cdi_enable_clustering: bool = True,
        cdi_cluster_eps_km: float = 5.0,
        cdi_cluster_min_pts: int = 5,
        cdi_trust_nresp_ge: int = 3,
        cdi_strategy=("local_outlier", "grid_thin", "quantile_residual"),
        cdi_grid_bin_km: float = 10.0,
        cdi_grid_agg: str = "median",
        cdi_quantile=(0.05, 0.95),
        cdi_enable_sponheuer: bool = False,
        cdi_spon_thr_sigma: float = 2.5,
        cdi_spon_min_r_km: float = 1.0,
        cdi_spon_robust: bool = True,
        cdi_spon_huber_k: float = 1.5,
        # sigma scaling per dataset (trust)
        sigma_scale_instr: float = 1.0,
        sigma_scale_dyfi: float = 1.0,
        sigma_scale_cdi: float = 1.0,
        debug: bool = False,
    ) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
        """
        Returns (imt_target, obs_used_df, meta_dict).
        """
        import numpy as np
        import pandas as pd

        vkey = _norm_version(version)
        mode_s = str(mode).lower().strip()

        meta = {"mode": mode_s, "vkey": vkey, "conversion": None, "cdi_conditioned": False}

        # helper for GMICE conversion (prefer gmice_convert if present)
        def _gmice(vals, input_type, output_type):
            if hasattr(self, "gmice_convert") and callable(getattr(self, "gmice_convert")):
                return self.gmice_convert(np.asarray(vals, dtype=float), input_type=str(input_type).upper(), output_type=str(output_type).upper())
            return self.global_gmice_convert(str(input_type).upper(), str(output_type).upper(), np.asarray(vals, dtype=float))

        if mode_s == "pga_instr":
            imt_target = "PGA"
            obs = self.build_observations(version=vkey, imt="PGA", dyfi_source="stationlist", sigma_override=None)
            obs_kept, _ = self._bayes_extent_first(obs, version=vkey, margin_deg=margin_deg, label="PGA_instr", debug=debug)
            obs_kept = self._bayes_apply_sigma_scales(obs_kept, sigma_scale=float(sigma_scale_instr))
            return imt_target, obs_kept, meta

        if mode_s == "mmi_dyfi":
            imt_target = "MMI"
            obs = self.build_observations(version=vkey, imt="MMI", dyfi_source="stationlist", sigma_override=None)
            obs_kept, _ = self._bayes_extent_first(obs, version=vkey, margin_deg=margin_deg, label="MMI_dyfi", debug=debug)
            obs_kept = self._bayes_apply_sigma_scales(obs_kept, sigma_scale=float(sigma_scale_dyfi))
            return imt_target, obs_kept, meta

        if mode_s in ("mmi_cdi", "mmi_cdi_filtered"):
            imt_target = "MMI"
            obs = self.build_observations(version=vkey, imt="MMI", dyfi_source="cdi", sigma_override=None)
            obs_kept, _ = self._bayes_extent_first(obs, version=vkey, margin_deg=margin_deg, label="MMI_cdi_raw", debug=debug)

            n0 = int(len(obs_kept))
            obs_kept = self._cdi_condition(
                obs_kept,
                version=vkey,
                trust_nresp_ge=int(cdi_trust_nresp_ge),
                enable_local_outlier=bool(cdi_enable_local_outlier),
                local_radius_km=float(cdi_local_radius_km),
                outlier_k=float(cdi_outlier_k),
                min_neighbors=int(cdi_min_neighbors),
                enable_clustering=bool(cdi_enable_clustering),
                cluster_eps_km=float(cdi_cluster_eps_km),
                cluster_min_pts=int(cdi_cluster_min_pts),
                cdi_strategy=cdi_strategy,
                cdi_grid_bin_km=float(cdi_grid_bin_km),
                cdi_grid_agg=str(cdi_grid_agg),
                cdi_quantile=cdi_quantile,
                cdi_enable_sponheuer=bool(cdi_enable_sponheuer),
                cdi_spon_thr_sigma=float(cdi_spon_thr_sigma),
                cdi_spon_min_r_km=float(cdi_spon_min_r_km),
                cdi_spon_robust=bool(cdi_spon_robust),
                cdi_spon_huber_k=float(cdi_spon_huber_k),
                debug=debug,
            )
            meta["cdi_conditioned"] = True
            meta["cdi_before"] = n0
            meta["cdi_after"] = int(len(obs_kept))

            obs_kept = self._bayes_apply_sigma_scales(obs_kept, sigma_scale=float(sigma_scale_cdi))
            return imt_target, obs_kept, meta

        # -------------------------
        # Mixed modes via GMICE
        # -------------------------

        if mode_s == "mmi_from_instr_dyfi":
            # convert instruments (PGA) -> MMI, then update MMI with DYFI (but only DYFI obs go into update)
            # NOTE: per your final mode decisions, mixed modes use either DYFI or CDI (not both).
            imt_target = "MMI"
            dy = self.build_observations(version=vkey, imt="MMI", dyfi_source="stationlist", sigma_override=None)
            dy, _ = self._bayes_extent_first(dy, version=vkey, margin_deg=margin_deg, label="MMI_dyfi", debug=debug)
            dy = self._bayes_apply_sigma_scales(dy, sigma_scale=float(sigma_scale_dyfi))
            meta["conversion"] = {"from": "PGA", "to": "MMI", "via": "GMICE", "note": "mode name indicates prior comparison; likelihood uses DYFI only"}
            return imt_target, dy, meta

        if mode_s == "mmi_from_instr_cdi":
            imt_target = "MMI"
            cdi = self.build_observations(version=vkey, imt="MMI", dyfi_source="cdi", sigma_override=None)
            cdi, _ = self._bayes_extent_first(cdi, version=vkey, margin_deg=margin_deg, label="MMI_cdi_raw", debug=debug)

            n0 = int(len(cdi))
            cdi = self._cdi_condition(
                cdi,
                version=vkey,
                trust_nresp_ge=int(cdi_trust_nresp_ge),
                enable_local_outlier=bool(cdi_enable_local_outlier),
                local_radius_km=float(cdi_local_radius_km),
                outlier_k=float(cdi_outlier_k),
                min_neighbors=int(cdi_min_neighbors),
                enable_clustering=bool(cdi_enable_clustering),
                cluster_eps_km=float(cdi_cluster_eps_km),
                cluster_min_pts=int(cdi_cluster_min_pts),
                cdi_strategy=cdi_strategy,
                cdi_grid_bin_km=float(cdi_grid_bin_km),
                cdi_grid_agg=str(cdi_grid_agg),
                cdi_quantile=cdi_quantile,
                cdi_enable_sponheuer=bool(cdi_enable_sponheuer),
                cdi_spon_thr_sigma=float(cdi_spon_thr_sigma),
                cdi_spon_min_r_km=float(cdi_spon_min_r_km),
                cdi_spon_robust=bool(cdi_spon_robust),
                cdi_spon_huber_k=float(cdi_spon_huber_k),
                debug=debug,
            )
            meta["cdi_conditioned"] = True
            meta["cdi_before"] = n0
            meta["cdi_after"] = int(len(cdi))

            cdi = self._bayes_apply_sigma_scales(cdi, sigma_scale=float(sigma_scale_cdi))
            meta["conversion"] = {"from": "PGA", "to": "MMI", "via": "GMICE", "note": "mode name indicates prior comparison; likelihood uses CDI only"}
            return imt_target, cdi, meta

        if mode_s == "pga_from_mmi_dyfi":
            # DYFI (MMI) -> PGA likelihood
            imt_target = "PGA"
            dy = self.build_observations(version=vkey, imt="MMI", dyfi_source="stationlist", sigma_override=None)
            dy, _ = self._bayes_extent_first(dy, version=vkey, margin_deg=margin_deg, label="MMI_dyfi_to_PGA", debug=debug)
            if isinstance(dy, pd.DataFrame) and not dy.empty:
                dy2 = dy.copy()
                dy2["value"] = _gmice(dy2["value"].to_numpy(dtype=float), "MMI", "PGA")
                dy2["imt"] = "PGA"
                dy2["source_detail"] = "dyfi_stationlist_as_pga"
                dy = dy2
            dy = self._bayes_apply_sigma_scales(dy, sigma_scale=float(sigma_scale_dyfi))
            meta["conversion"] = {"from": "MMI", "to": "PGA", "via": "GMICE", "source": "DYFI"}
            return imt_target, dy, meta

        if mode_s == "pga_from_mmi_cdi":
            imt_target = "PGA"
            cdi = self.build_observations(version=vkey, imt="MMI", dyfi_source="cdi", sigma_override=None)
            cdi, _ = self._bayes_extent_first(cdi, version=vkey, margin_deg=margin_deg, label="MMI_cdi_to_PGA_raw", debug=debug)

            n0 = int(len(cdi))
            cdi = self._cdi_condition(
                cdi,
                version=vkey,
                trust_nresp_ge=int(cdi_trust_nresp_ge),
                enable_local_outlier=bool(cdi_enable_local_outlier),
                local_radius_km=float(cdi_local_radius_km),
                outlier_k=float(cdi_outlier_k),
                min_neighbors=int(cdi_min_neighbors),
                enable_clustering=bool(cdi_enable_clustering),
                cluster_eps_km=float(cdi_cluster_eps_km),
                cluster_min_pts=int(cdi_cluster_min_pts),
                cdi_strategy=cdi_strategy,
                cdi_grid_bin_km=float(cdi_grid_bin_km),
                cdi_grid_agg=str(cdi_grid_agg),
                cdi_quantile=cdi_quantile,
                cdi_enable_sponheuer=bool(cdi_enable_sponheuer),
                cdi_spon_thr_sigma=float(cdi_spon_thr_sigma),
                cdi_spon_min_r_km=float(cdi_spon_min_r_km),
                cdi_spon_robust=bool(cdi_spon_robust),
                cdi_spon_huber_k=float(cdi_spon_huber_k),
                debug=debug,
            )
            meta["cdi_conditioned"] = True
            meta["cdi_before"] = n0
            meta["cdi_after"] = int(len(cdi))

            if isinstance(cdi, pd.DataFrame) and not cdi.empty:
                c2 = cdi.copy()
                c2["value"] = _gmice(c2["value"].to_numpy(dtype=float), "MMI", "PGA")
                c2["imt"] = "PGA"
                c2["source_detail"] = "cdi_geo_as_pga"
                cdi = c2
            cdi = self._bayes_apply_sigma_scales(cdi, sigma_scale=float(sigma_scale_cdi))
            meta["conversion"] = {"from": "MMI", "to": "PGA", "via": "GMICE", "source": "CDI"}
            return imt_target, cdi, meta

        raise ValueError(f"Unknown Bayes mode: {mode!r}")

    # ---------------------------
    # Main runner: Bayes update (Bayes-1lik and Bayes2 sequential)
    # ---------------------------

    def run_bayes_update(
        self,
        *,
        version,
        mode,
        mode_key: Optional[str] = None,
        margin_deg: float = 0.0,
        # local update controls
        update_radius_km: float = 30,
        kernel: str = "gaussian",
        kernel_scale_km: float = 12,
        neighbor_k: Optional[int] = 50,
        default_obs_sigma: float = 0.5,
        # sigma scaling (trust)
        sigma_scale_instr: float = 1.0,
        sigma_scale_dyfi: float = 1.0,
        sigma_scale_cdi: float = 1.0,
        # CDI conditioning knobs (pass-through)
        cdi_enable_local_outlier: bool = True,
        cdi_local_radius_km: float = 20.0,
        cdi_outlier_k: float = 3.5,
        cdi_min_neighbors: int = 5,
        cdi_enable_clustering: bool = True,
        cdi_cluster_eps_km: float = 5.0,
        cdi_cluster_min_pts: int = 5,
        cdi_trust_nresp_ge: int = 3,
        cdi_strategy=("local_outlier", "grid_thin", "quantile_residual"),
        cdi_grid_bin_km: float = 10.0,
        cdi_grid_agg: str = "median",
        cdi_quantile=(0.05, 0.95),
        cdi_enable_sponheuer: bool = False,
        cdi_spon_thr_sigma: float = 2.5,
        cdi_spon_min_r_km: float = 1.0,
        cdi_spon_robust: bool = True,
        cdi_spon_huber_k: float = 1.5,
        # Bayes2 sequential (optional)
        bayes2: bool = False,
        bayes2_mode_stage1: Optional[str] = None,
        bayes2_mode_stage2: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Run Bayes update for a single target version.

        Bayes-1lik:
          - bayes2=False (default)
          - mode controls which obs are used and what IMT is being updated.

        Bayes2 sequential:
          - bayes2=True
          - provide bayes2_mode_stage1 and bayes2_mode_stage2 (each is a mode string)
          - stage2 prior becomes stage1 posterior (but stage0 prior is still v0 fixed prior)

        Storage:
          uq_state["versions"][vkey]["bayes"][mode_key] = {
              "meta": {...},
              "prior": {"version": v0, "mean_grid": ..., "sigma_grid": ...},
              "stage1": {...}  (if bayes2)
              "stage2": {...}  (if bayes2)
              "posterior": {"mean_grid": ..., "sigma_grid": ...},
              "obs_used": DataFrame (optional)
          }
        """
        import numpy as np
        import pandas as pd
        import time

        if not isinstance(self.uq_state, dict) or not (self.uq_state.get("versions") or {}):
            raise RuntimeError("uq_state not initialized. Run uq_build_dataset() first.")

        vkey = _norm_version(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey)
        if vpack is None:
            raise KeyError(f"Version not in uq_state: {vkey}")

        lon2d, lat2d = self._get_unified_grid()

        # fixed prior version
        v0 = self._bayes_get_prior_version_key()

        # choose keys
        mode_s = str(mode).lower().strip()
        if mode_key is None:
            mode_key = f"bayes__{mode_s}__v{vkey}"
        mode_key = str(mode_key)

        # ensure storage containers
        if "bayes" not in vpack or not isinstance(vpack.get("bayes"), dict):
            vpack["bayes"] = {}

        t0 = time.time()

        # --- load fixed prior (v0) on unified grid ---
        # mean prior uses imt key directly
        # sigma prior uses STD* mapping helper
        def _get_prior_for_imt(imt_target):
            _, _, mu0 = self._get_prior_mean_unified(v0, imt_target)
            _, _, sig0 = self._get_prior_sigma_unified(v0, imt_target)
            if mu0 is None or sig0 is None:
                raise RuntimeError(f"Missing prior mean/sigma for v0={v0} imt={imt_target}. Check unified stacks.")
            if np.asarray(mu0).shape != lon2d.shape or np.asarray(sig0).shape != lon2d.shape:
                raise RuntimeError(f"Prior shapes do not match unified grid for imt={imt_target}.")
            return np.asarray(mu0, dtype=float), np.asarray(sig0, dtype=float)

        # --- Bayes-1lik path ---
        if not bayes2:
            imt_target, obs_used, meta_obs = self._bayes_build_obs_for_mode(
                version=vkey,
                mode=mode_s,
                margin_deg=float(margin_deg),
                cdi_enable_local_outlier=cdi_enable_local_outlier,
                cdi_local_radius_km=float(cdi_local_radius_km),
                cdi_outlier_k=float(cdi_outlier_k),
                cdi_min_neighbors=int(cdi_min_neighbors),
                cdi_enable_clustering=cdi_enable_clustering,
                cdi_cluster_eps_km=float(cdi_cluster_eps_km),
                cdi_cluster_min_pts=int(cdi_cluster_min_pts),
                cdi_trust_nresp_ge=int(cdi_trust_nresp_ge),
                cdi_strategy=cdi_strategy,
                cdi_grid_bin_km=float(cdi_grid_bin_km),
                cdi_grid_agg=str(cdi_grid_agg),
                cdi_quantile=cdi_quantile,
                cdi_enable_sponheuer=bool(cdi_enable_sponheuer),
                cdi_spon_thr_sigma=float(cdi_spon_thr_sigma),
                cdi_spon_min_r_km=float(cdi_spon_min_r_km),
                cdi_spon_robust=bool(cdi_spon_robust),
                cdi_spon_huber_k=float(cdi_spon_huber_k),
                sigma_scale_instr=float(sigma_scale_instr),
                sigma_scale_dyfi=float(sigma_scale_dyfi),
                sigma_scale_cdi=float(sigma_scale_cdi),
                debug=debug,
            )

            prior_mean, prior_sigma = _get_prior_for_imt(imt_target)

            post_mean, post_sigma, audit = self._bayes_local_precision_fusion_grid(
                imt_target=imt_target,
                lon2d=lon2d,
                lat2d=lat2d,
                prior_mean=prior_mean,
                prior_sigma=prior_sigma,
                obs_df=obs_used,
                update_radius_km=float(update_radius_km),
                kernel=str(kernel),
                kernel_scale_km=float(kernel_scale_km),
                neighbor_k=neighbor_k,
                default_obs_sigma=float(default_obs_sigma),
                debug=debug,
            )

            out = {
                "meta": {
                    "imt": str(imt_target).upper().strip(),
                    "mode": mode_s,
                    "mode_key": mode_key,
                    "target_version": vkey,
                    "prior_version": v0,
                    "working_space": self._bayes_working_space(imt_target),
                    "elapsed_s": float(time.time() - t0),
                    "kernel": str(kernel),
                    "kernel_scale_km": float(kernel_scale_km),
                    "update_radius_km": float(update_radius_km),
                    "neighbor_k": None if neighbor_k is None else int(neighbor_k),
                },
                "prior": {
                    "mean_grid": prior_mean,
                    "sigma_grid": prior_sigma,
                },
                "posterior": {
                    "mean_grid": post_mean,
                    "sigma_grid": post_sigma,
                },
                "audit": audit,
                "obs_meta": meta_obs,
            }

            # optionally store obs_used (can be heavy)
            out["obs_used"] = obs_used

            vpack["bayes"][mode_key] = out
            return out

        # --- Bayes2 sequential path ---
        # stage modes must be provided explicitly to avoid hidden assumptions
        if bayes2_mode_stage1 is None or bayes2_mode_stage2 is None:
            raise ValueError("bayes2=True requires bayes2_mode_stage1 and bayes2_mode_stage2.")

        m1 = str(bayes2_mode_stage1).lower().strip()
        m2 = str(bayes2_mode_stage2).lower().strip()

        # stage1
        imt1, obs1, meta1 = self._bayes_build_obs_for_mode(
            version=vkey,
            mode=m1,
            margin_deg=float(margin_deg),
            cdi_enable_local_outlier=cdi_enable_local_outlier,
            cdi_local_radius_km=float(cdi_local_radius_km),
            cdi_outlier_k=float(cdi_outlier_k),
            cdi_min_neighbors=int(cdi_min_neighbors),
            cdi_enable_clustering=cdi_enable_clustering,
            cdi_cluster_eps_km=float(cdi_cluster_eps_km),
            cdi_cluster_min_pts=int(cdi_cluster_min_pts),
            cdi_trust_nresp_ge=int(cdi_trust_nresp_ge),
            cdi_strategy=cdi_strategy,
            cdi_grid_bin_km=float(cdi_grid_bin_km),
            cdi_grid_agg=str(cdi_grid_agg),
            cdi_quantile=cdi_quantile,
            cdi_enable_sponheuer=bool(cdi_enable_sponheuer),
            cdi_spon_thr_sigma=float(cdi_spon_thr_sigma),
            cdi_spon_min_r_km=float(cdi_spon_min_r_km),
            cdi_spon_robust=bool(cdi_spon_robust),
            cdi_spon_huber_k=float(cdi_spon_huber_k),
            sigma_scale_instr=float(sigma_scale_instr),
            sigma_scale_dyfi=float(sigma_scale_dyfi),
            sigma_scale_cdi=float(sigma_scale_cdi),
            debug=debug,
        )

        # stage2
        imt2, obs2, meta2 = self._bayes_build_obs_for_mode(
            version=vkey,
            mode=m2,
            margin_deg=float(margin_deg),
            cdi_enable_local_outlier=cdi_enable_local_outlier,
            cdi_local_radius_km=float(cdi_local_radius_km),
            cdi_outlier_k=float(cdi_outlier_k),
            cdi_min_neighbors=int(cdi_min_neighbors),
            cdi_enable_clustering=cdi_enable_clustering,
            cdi_cluster_eps_km=float(cdi_cluster_eps_km),
            cdi_cluster_min_pts=int(cdi_cluster_min_pts),
            cdi_trust_nresp_ge=int(cdi_trust_nresp_ge),
            cdi_strategy=cdi_strategy,
            cdi_grid_bin_km=float(cdi_grid_bin_km),
            cdi_grid_agg=str(cdi_grid_agg),
            cdi_quantile=cdi_quantile,
            cdi_enable_sponheuer=bool(cdi_enable_sponheuer),
            cdi_spon_thr_sigma=float(cdi_spon_thr_sigma),
            cdi_spon_min_r_km=float(cdi_spon_min_r_km),
            cdi_spon_robust=bool(cdi_spon_robust),
            cdi_spon_huber_k=float(cdi_spon_huber_k),
            sigma_scale_instr=float(sigma_scale_instr),
            sigma_scale_dyfi=float(sigma_scale_dyfi),
            sigma_scale_cdi=float(sigma_scale_cdi),
            debug=debug,
        )

        if str(imt1).upper().strip() != str(imt2).upper().strip():
            raise RuntimeError(f"Bayes2 requires consistent target IMT across stages: stage1={imt1} stage2={imt2}")

        imt_target = str(imt1).upper().strip()
        prior_mean, prior_sigma = _get_prior_for_imt(imt_target)

        # stage1 update
        s1_mean, s1_sigma, audit1 = self._bayes_local_precision_fusion_grid(
            imt_target=imt_target,
            lon2d=lon2d,
            lat2d=lat2d,
            prior_mean=prior_mean,
            prior_sigma=prior_sigma,
            obs_df=obs1,
            update_radius_km=float(update_radius_km),
            kernel=str(kernel),
            kernel_scale_km=float(kernel_scale_km),
            neighbor_k=neighbor_k,
            default_obs_sigma=float(default_obs_sigma),
            debug=debug,
        )

        # stage2 update uses stage1 posterior as prior
        s2_mean, s2_sigma, audit2 = self._bayes_local_precision_fusion_grid(
            imt_target=imt_target,
            lon2d=lon2d,
            lat2d=lat2d,
            prior_mean=s1_mean,
            prior_sigma=s1_sigma,
            obs_df=obs2,
            update_radius_km=float(update_radius_km),
            kernel=str(kernel),
            kernel_scale_km=float(kernel_scale_km),
            neighbor_k=neighbor_k,
            default_obs_sigma=float(default_obs_sigma),
            debug=debug,
        )

        out = {
            "meta": {
                "imt": imt_target,
                "mode": mode_s,
                "mode_key": mode_key,
                "target_version": vkey,
                "prior_version": v0,
                "bayes2": True,
                "stage1_mode": m1,
                "stage2_mode": m2,
                "working_space": self._bayes_working_space(imt_target),
                "elapsed_s": float(time.time() - t0),
                "kernel": str(kernel),
                "kernel_scale_km": float(kernel_scale_km),
                "update_radius_km": float(update_radius_km),
                "neighbor_k": None if neighbor_k is None else int(neighbor_k),
            },
            "prior": {"mean_grid": prior_mean, "sigma_grid": prior_sigma},
            "stage1": {
                "mode": m1,
                "obs_meta": meta1,
                "audit": audit1,
                "posterior": {"mean_grid": s1_mean, "sigma_grid": s1_sigma},
                "obs_used": obs1,
            },
            "stage2": {
                "mode": m2,
                "obs_meta": meta2,
                "audit": audit2,
                "posterior": {"mean_grid": s2_mean, "sigma_grid": s2_sigma},
                "obs_used": obs2,
            },
            "posterior": {"mean_grid": s2_mean, "sigma_grid": s2_sigma},
        }

        vpack["bayes"][mode_key] = out
        return out

    # ---------------------------
    # Audit plot: Bayes prior/posterior (+ optional diffs)
    # ---------------------------

    def plot_bayes_audit(
        self,
        *,
        version,
        mode_key,
        show=True,
        save_path=None,
        dpi=150,
        figsize=(14, 10),
        obs_size=14,
        mean_scale_type="usgs",
        mean_pga_units="%g",
        sigma_cmap="viridis",
        sigma_vmin=None,
        sigma_vmax=None,
        show_diffs=True,
        debug=False,
    ):
        """
        Bayes audit plot:
          - Prior mean (unified)
          - Posterior mean (unified)
          - Prior sigma (unified sigma field; MMI linear / PGA ln)
          - Posterior sigma (same convention)
          - Obs points overlaid on posterior mean
          - Optional diff panels (posterior - prior)
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        vkey = _norm_version(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey) or {}
        bpack = (vpack.get("bayes") or {}).get(mode_key)
        if not isinstance(bpack, dict):
            raise KeyError(f"No bayes result for v={vkey} mode_key={mode_key}")

        meta = bpack.get("meta") or {}
        imt_u = str(meta.get("imt", "MMI")).upper().strip()

        lon2d, lat2d = self._get_unified_grid()

        prior_mean = np.asarray((bpack.get("prior") or {}).get("mean_grid"), dtype=float)
        prior_sig = np.asarray((bpack.get("prior") or {}).get("sigma_grid"), dtype=float)
        post_mean = np.asarray((bpack.get("posterior") or {}).get("mean_grid"), dtype=float)
        post_sig = np.asarray((bpack.get("posterior") or {}).get("sigma_grid"), dtype=float)

        if prior_mean.shape != lon2d.shape or post_mean.shape != lon2d.shape:
            raise RuntimeError("Bayes mean grids do not match unified grid shape.")
        if prior_sig.shape != lon2d.shape or post_sig.shape != lon2d.shape:
            raise RuntimeError("Bayes sigma grids do not match unified grid shape.")

        # attempt to locate obs used (bayes1 stores at root; bayes2 stores per stage)
        obs = None
        if isinstance(bpack.get("obs_used"), pd.DataFrame):
            obs = bpack["obs_used"]
        elif meta.get("bayes2") and isinstance((bpack.get("stage2") or {}).get("obs_used"), pd.DataFrame):
            obs = (bpack.get("stage2") or {}).get("obs_used")

        # mean colormap (discrete if available)
        cmap_mean, norm_mean, ticks_mean, label_mean = self._mean_cmap_for_imt(
            imt_u, scale_type=mean_scale_type, pga_units=mean_pga_units
        )

        ncols = 3 if show_diffs else 2
        nrows = 2
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

        # prior mean
        self._plot_grid_panel(
            axs[0, 0],
            lon2d, lat2d, prior_mean,
            title=f"Prior mean (v0 fixed) — {imt_u}",
            cmap=cmap_mean, norm=norm_mean,
            add_colorbar=True, cbar_ticks=ticks_mean, cbar_label=label_mean,
            fig=fig,
        )

        # posterior mean
        self._plot_grid_panel(
            axs[0, 1],
            lon2d, lat2d, post_mean,
            title=f"Posterior mean — {imt_u}",
            cmap=cmap_mean, norm=norm_mean,
            add_colorbar=True, cbar_ticks=ticks_mean, cbar_label=label_mean,
            fig=fig,
        )

        # diff mean
        if show_diffs:
            dmean = post_mean - prior_mean
            self._plot_grid_panel(
                axs[0, 2],
                lon2d, lat2d, dmean,
                title="Δ mean (post − prior)",
                cmap="coolwarm", norm=None,
                add_colorbar=True, cbar_label=f"Δ {imt_u}",
                fig=fig,
            )

        # prior sigma
        self._plot_grid_panel(
            axs[1, 0],
            lon2d, lat2d, prior_sig,
            title=f"Prior sigma ({self._bayes_sigma_key_for_imt(imt_u)})",
            cmap=sigma_cmap, norm=None,
            vmin=sigma_vmin, vmax=sigma_vmax,
            add_colorbar=True, cbar_label="sigma",
            fig=fig,
        )

        # posterior sigma
        self._plot_grid_panel(
            axs[1, 1],
            lon2d, lat2d, post_sig,
            title="Posterior sigma",
            cmap=sigma_cmap, norm=None,
            vmin=sigma_vmin, vmax=sigma_vmax,
            add_colorbar=True, cbar_label="sigma",
            fig=fig,
        )

        # diff sigma + obs overlay (or obs panel if no diffs)
        ax_last = axs[1, 2] if show_diffs else axs[1, 1]
        if show_diffs:
            dsig = post_sig - prior_sig
            self._plot_grid_panel(
                ax_last,
                lon2d, lat2d, dsig,
                title="Δ sigma (post − prior)",
                cmap="coolwarm", norm=None,
                add_colorbar=True, cbar_label="Δ sigma",
                fig=fig,
            )

        # overlay obs locations on posterior mean panel
        if isinstance(obs, pd.DataFrame) and not obs.empty and "lon" in obs.columns and "lat" in obs.columns:
            try:
                axs[0, 1].scatter(obs["lon"], obs["lat"], s=obs_size, c="k", alpha=0.6, linewidths=0.0)
                axs[0, 1].set_title(f"Posterior mean — {imt_u} (+ obs)")
            except Exception:
                pass

        fig.suptitle(f"Bayes audit — v={vkey} mode_key={mode_key}", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()

        return fig




    
    # ======================================================================
    # Residual / Innovation Kriging (UPDATED: fixed prior = first available shakemap)
    # - Prior mean/sigma are ALWAYS taken from v0 (first version in unified stack)
    # - Observations ALWAYS taken from the TARGET version (so we test “v0 prior + new data”)
    # - Supports:
    #     * PGA: instruments (stationlist seismic)
    #     * MMI: DYFI stationlist and/or CDI (with optional conditioning)
    # ======================================================================
    
    def _rk_get_prior_version_key(self):
        """
        Fixed prior selector for Residual Kriging.
        Default: first entry in uq_state['unified']['version_keys'] if available,
        otherwise lexicographically smallest version key in uq_state['versions'].
        """
        u = (self.uq_state or {}).get("unified") or {}
        vks = u.get("version_keys")
        if isinstance(vks, list) and len(vks) > 0:
            return str(vks[0])
        vks2 = sorted(list(((self.uq_state or {}).get("versions") or {}).keys()))
        if not vks2:
            raise RuntimeError("uq_state has no versions; run uq_build_dataset() first.")
        return str(vks2[0])
    
    
    def _rk_ok_krige_grid(self, x, y, z, xgrid2d, ygrid2d, *, variogram_model="linear",
                         variogram_parameters=None, neighbor_k=40, nugget=0.0, verbose=False):
        """
        Wrapper: prefer the class's existing OK backend if present; otherwise fallback to PyKrige.
        Returns: (z_kriged_2d, var_kriged_2d)
        """
        # 1) preferred: existing helper already used by OK/RK in this module
        if hasattr(self, "_ok_krige_grid") and callable(getattr(self, "_ok_krige_grid")):
            return self._ok_krige_grid(
                x, y, z, xgrid2d, ygrid2d,
                variogram_model=variogram_model,
                variogram_parameters=variogram_parameters,
                neighbor_k=neighbor_k,
                nugget=nugget,
                verbose=verbose,
            )
    
        # 2) fallback: PyKrige
        try:
            from pykrige.ok import OrdinaryKriging
        except Exception as e:
            raise ImportError(
                "No kriging backend available. Expected self._ok_krige_grid or pykrige."
            ) from e
    
        import numpy as np
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
    
        # grid vectors
        xvec = np.asarray(xgrid2d[0, :], dtype=float).ravel()
        yvec = np.asarray(ygrid2d[:, 0], dtype=float).ravel()
    
        ok = OrdinaryKriging(
            x, y, z,
            variogram_model=str(variogram_model),
            variogram_parameters=variogram_parameters,
            nugget=float(nugget),
            enable_plotting=False,
            verbose=bool(verbose),
        )
        zhat, ss = ok.execute("grid", xvec, yvec)
        zhat = np.asarray(zhat, dtype=float)
        ss = np.asarray(ss, dtype=float)
        # PyKrige returns shape (ny, nx)
        return zhat, ss
    
    
    def run_residual_kriging_update(
        self,
        *,
        version,
        mode="pga_instr",
        key=None,
        dyfi_source="stationlist",          # for MMI: "stationlist" | "cdi" | "both" | "auto"
        use_cdi=False,                      # explicit CDI toggle (kept for convenience)
        margin_deg=0.05,
    
        # OK controls
        variogram_model="linear",
        variogram_parameters=None,
        neighbor_k=40,
        nugget=0.0,
    
        # residual update controls
        pga_space="log10",                  # "log10" or "linear" (recommend log10)
        pga_floor=1e-6,                     # avoid log10(0)
        sigma_clip_min=1e-6,
    
        # CDI conditioning knobs (passed to _cdi_condition)
        cdi_trust_nresp_ge=3,
        cdi_enable_local_outlier=True,
        cdi_local_radius_km=25.0,
        cdi_outlier_k=2.5,
        cdi_min_neighbors=4,
        cdi_enable_clustering=True,
        cdi_cluster_eps_km=2.0,
        cdi_cluster_min_pts=3,
        cdi_strategy=("local_outlier", "grid_thin", "quantile_residual"),
        cdi_grid_bin_km=10.0,
        cdi_grid_agg="median",
        cdi_quantile=(0.05, 0.95),
    
        verbose=False,
        debug=False,
    ):
        """
        Residual / Innovation Kriging update with FIXED PRIOR (v0).
        Observations are taken from the TARGET version, but residuals are computed against v0 prior.
    
        Storage:
          uq_state["versions"][vkey]["updates"][key] = {
            "meta": {...},
            "prior": {"version": v0, "imt": imt, "mean_grid":..., "sigma_grid":...},
            "posterior": {"mean_grid":..., "sigma_grid":...},
            "residual": {"resid_grid":..., "resid_var":...},
            "obs_used": DataFrame,
            "metrics": {...}
          }
        """
        import numpy as np
        import pandas as pd
        import time
    
        if not isinstance(self.uq_state, dict) or not (self.uq_state.get("versions") or {}):
            raise RuntimeError("uq_state not initialized. Run uq_build_dataset() first.")
    
        vkey = _norm_version(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey)
        if vpack is None:
            raise KeyError(f"Version not in uq_state: {vkey}")
    
        # unified grid
        lon2d, lat2d = self._get_unified_grid()
    
        # fixed prior
        v0 = self._rk_get_prior_version_key()
    
        mode_s = str(mode).lower().strip()
        if key is None:
            key = f"rk__{mode_s}__v{vkey}"
        key = str(key)
    
        # storage container
        if "updates" not in vpack or not isinstance(vpack.get("updates"), dict):
            vpack["updates"] = {}
    
        # decide IMT + source routing
        # NOTE: Keep “IMT isolation” rule: PGA updated by instruments; MMI updated by DYFI/CDI.
        if mode_s in ("pga", "pga_instr", "pga_instruments", "pga_station", "pga_stationlist"):
            imt_target = "PGA"
            dyfi_source_eff = "stationlist"  # irrelevant for PGA
            include_cdi = False
        elif mode_s in ("mmi", "mmi_dyfi", "mmi_stationlist"):
            imt_target = "MMI"
            dyfi_source_eff = "stationlist"
            include_cdi = False
        elif mode_s in ("mmi_cdi", "cdi"):
            imt_target = "MMI"
            dyfi_source_eff = "cdi"
            include_cdi = True
        elif mode_s in ("mmi_both", "mmi_dyfi_cdi", "both"):
            imt_target = "MMI"
            dyfi_source_eff = "both"
            include_cdi = True
        else:
            # allow passing explicit dyfi_source + use_cdi externally
            imt_target = "MMI" if "mmi" in mode_s else "PGA"
            dyfi_source_eff = str(dyfi_source).lower().strip()
            include_cdi = bool(use_cdi) or (dyfi_source_eff in ("cdi", "both", "auto"))
    
        t0 = time.time()
    
        # --- load prior mean/sigma on unified grid (v0 fixed) ---
        _, _, prior_mean = self._get_prior_mean_unified(v0, imt_target)
        _, _, prior_sigma = self._get_prior_sigma_unified(v0, imt_target)
        if prior_mean is None or prior_sigma is None:
            raise RuntimeError(f"Missing prior mean/sigma for v0={v0} imt={imt_target}.")
        prior_mean = np.asarray(prior_mean, dtype=float)
        prior_sigma = np.asarray(prior_sigma, dtype=float)
        if prior_mean.shape != lon2d.shape or prior_sigma.shape != lon2d.shape:
            raise RuntimeError(f"Prior shapes mismatch unified grid for imt={imt_target} (v0={v0}).")
    
        # --- build observations from TARGET version (vkey), NOT v0 ---
        obs = self.build_observations(version=vkey, imt=imt_target, dyfi_source=dyfi_source_eff, sigma_override=None)
    
        if obs is None or getattr(obs, "empty", True):
            out = {
                "ok": False,
                "note": f"No observations for imt={imt_target} dyfi_source={dyfi_source_eff} (v={vkey}).",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key},
            }
            vpack["updates"][key] = out
            return out
    
        # strict columns + numeric
        obs = obs.copy()
        for c in ("lon", "lat", "value", "sigma"):
            if c not in obs.columns:
                raise RuntimeError(f"build_observations did not return required column: {c}")
        obs["lon"] = pd.to_numeric(obs["lon"], errors="coerce")
        obs["lat"] = pd.to_numeric(obs["lat"], errors="coerce")
        obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
        obs["sigma"] = pd.to_numeric(obs["sigma"], errors="coerce")
        obs = obs.dropna(subset=["lon", "lat", "value"]).reset_index(drop=True)
        if obs.empty:
            out = {
                "ok": False,
                "note": f"Observations all invalid after numeric coercion (v={vkey}).",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key},
            }
            vpack["updates"][key] = out
            return out
    
        # ------------------------------------------------------------------
        # extent filter (FIXED: your filter_observations_to_extent has no drop=)
        # Prefer (kept, dropped) interface; fallback to kept-only.
        # ------------------------------------------------------------------
        try:
            obs_f, obs_drop = self.filter_observations_to_extent(
                obs,
                version=vkey,
                grid_mode="unified",
                margin_deg=float(margin_deg),
                return_dropped=True,
            )
            meta_ext = {
                "grid_mode": "unified",
                "margin_deg": float(margin_deg),
                "kept": int(len(obs_f)) if hasattr(obs_f, "__len__") else None,
                "dropped": int(len(obs_drop)) if hasattr(obs_drop, "__len__") else None,
            }
        except TypeError:
            obs_f = self.filter_observations_to_extent(
                obs,
                version=vkey,
                grid_mode="unified",
                margin_deg=float(margin_deg),
            )
            meta_ext = {
                "grid_mode": "unified",
                "margin_deg": float(margin_deg),
                "kept": int(len(obs_f)) if hasattr(obs_f, "__len__") else None,
                "dropped": None,
            }
    
        if obs_f is None or getattr(obs_f, "empty", False) or len(obs_f) == 0:
            out = {
                "ok": False,
                "note": "All observations fell outside unified extent after margin filter.",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key, "extent": meta_ext},
            }
            vpack["updates"][key] = out
            return out
    
        # optional CDI conditioning (works only if CDI rows are present)
        # IMPORTANT: conditioning compares to v0 prior (so pass version=v0)
        cdi_meta = None
        if imt_target == "MMI" and include_cdi:
            try:
                obs_f, cdi_meta = self._cdi_condition(
                    version=v0,  # <-- compare/condition against v0 prior
                    obs_df=obs_f,
                    margin_deg=float(margin_deg),
                    cdi_trust_nresp_ge=int(cdi_trust_nresp_ge),
                    cdi_enable_local_outlier=bool(cdi_enable_local_outlier),
                    cdi_local_radius_km=float(cdi_local_radius_km),
                    cdi_outlier_k=float(cdi_outlier_k),
                    cdi_min_neighbors=int(cdi_min_neighbors),
                    cdi_enable_clustering=bool(cdi_enable_clustering),
                    cdi_cluster_eps_km=float(cdi_cluster_eps_km),
                    cdi_cluster_min_pts=int(cdi_cluster_min_pts),
                    cdi_strategy=cdi_strategy,
                    cdi_grid_bin_km=float(cdi_grid_bin_km),
                    cdi_grid_agg=str(cdi_grid_agg),
                    cdi_quantile=cdi_quantile,
                    debug=bool(debug),
                )
            except Exception as e:
                cdi_meta = {"ok": False, "err": str(e)}
    
        # --- sample prior at obs locations (v0 fixed) ---
        prior_at_obs = self._rk_sample_grid_nn(lon2d, lat2d, prior_mean, obs_f["lon"].values, obs_f["lat"].values)
    
        # --- compute residuals in chosen space ---
        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            y = np.log10(np.maximum(obs_f["value"].to_numpy(dtype=float), float(pga_floor)))
            m = np.log10(np.maximum(prior_at_obs.astype(float), float(pga_floor)))
            resid = y - m
        else:
            y = obs_f["value"].to_numpy(dtype=float)
            m = prior_at_obs.astype(float)
            resid = y - m
    
        # weights (optional, pragmatic): 1/sigma^2 (currently not passed to backend)
        sig_obs = obs_f["sigma"].to_numpy(dtype=float)
        sig_obs = np.where(np.isfinite(sig_obs), sig_obs, np.nan)
        sig_obs = np.where(sig_obs > float(sigma_clip_min), sig_obs, float(sigma_clip_min))
        _w = 1.0 / (sig_obs ** 2)  # reserved for future weighted kriging
    
        # --- OK krige residuals on the unified grid ---
        zhat_resid, var_resid = self._rk_ok_krige_grid(
            obs_f["lon"].values, obs_f["lat"].values, resid,
            lon2d, lat2d,
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            neighbor_k=int(neighbor_k),
            nugget=float(nugget),
            verbose=bool(verbose),
        )
    
        zhat_resid = np.asarray(zhat_resid, dtype=float)
        var_resid = np.asarray(var_resid, dtype=float)
    
        # --- posterior mean back to original space ---
        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            post_mean = 10.0 ** (np.log10(np.maximum(prior_mean, float(pga_floor))) + zhat_resid)
            # pragmatic sigma propagation from log10-space residual variance
            post_sigma = np.sqrt(
                np.maximum(prior_sigma, 0.0) ** 2 +
                (np.log(10.0) * np.maximum(post_mean, float(pga_floor))) ** 2 * np.maximum(var_resid, 0.0)
            )
        else:
            post_mean = prior_mean + zhat_resid
            post_sigma = np.sqrt(np.maximum(prior_sigma, 0.0) ** 2 + np.maximum(var_resid, 0.0))
    
        # --- quick metrics vs obs (prior/post evaluated at obs) ---
        post_at_obs = self._rk_sample_grid_nn(lon2d, lat2d, post_mean, obs_f["lon"].values, obs_f["lat"].values)
    
        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            prior_pred = np.log10(np.maximum(prior_at_obs.astype(float), float(pga_floor)))
            post_pred = np.log10(np.maximum(post_at_obs.astype(float), float(pga_floor)))
            obs_y = np.log10(np.maximum(obs_f["value"].to_numpy(dtype=float), float(pga_floor)))
        else:
            prior_pred = prior_at_obs.astype(float)
            post_pred = post_at_obs.astype(float)
            obs_y = obs_f["value"].to_numpy(dtype=float)
    
        rmse_prior = float(np.sqrt(np.nanmean((prior_pred - obs_y) ** 2)))
        rmse_post = float(np.sqrt(np.nanmean((post_pred - obs_y) ** 2)))
        mae_prior = float(np.nanmean(np.abs(prior_pred - obs_y)))
        mae_post = float(np.nanmean(np.abs(post_pred - obs_y)))
    
        out = {
            "ok": True,
            "meta": {
                "version": vkey,
                "prior_version": v0,
                "imt": imt_target,
                "mode": mode_s,
                "key": key,
                "dyfi_source": dyfi_source_eff,
                "include_cdi": bool(include_cdi),
                "margin_deg": float(margin_deg),
                "variogram_model": str(variogram_model),
                "variogram_parameters": variogram_parameters,
                "neighbor_k": int(neighbor_k),
                "nugget": float(nugget),
                "pga_space": str(pga_space),
                "runtime_s": float(time.time() - t0),
                "extent": meta_ext,
            },
            "prior": {"version": v0, "mean_grid": prior_mean, "sigma_grid": prior_sigma},
            "residual": {"resid_grid": zhat_resid, "resid_var": var_resid},
            "posterior": {"mean_grid": post_mean, "sigma_grid": post_sigma},
            "obs_used": obs_f,
            "metrics": {
                "n_obs": int(len(obs_f)),
                "rmse_prior": rmse_prior,
                "rmse_post": rmse_post,
                "mae_prior": mae_prior,
                "mae_post": mae_post,
            },
        }
        if cdi_meta is not None:
            out["meta"]["cdi_conditioning"] = cdi_meta
    
        vpack["updates"][key] = out
    
        if verbose:
            print(
                f"[RK] v={vkey} prior=v0({v0}) imt={imt_target} "
                f"n={len(obs_f)} rmse_prior={rmse_prior:.4f} rmse_post={rmse_post:.4f}"
            )
    
        return out

    
    
    def plot_residual_kriging_audit(
        self,
        *,
        version,
        key,
        show=True,
        save_path=None,
        show_diffs=True,
        dpi=160,
        figsize=(13.5, 7.0),
    ):
        """
        Audit plot for residual kriging result stored in uq_state["versions"][vkey]["updates"][key].
        Shows: prior mean/sigma (v0), posterior mean/sigma, and optionally diffs, plus obs points.
        """
        import numpy as np
        import matplotlib.pyplot as plt
    
        vkey = _norm_version(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey)
        if vpack is None:
            raise KeyError(f"Version not in uq_state: {vkey}")
        upd = (vpack.get("updates") or {}).get(str(key))
        if not isinstance(upd, dict) or not upd.get("ok", False):
            raise RuntimeError(f"No successful RK update found for v={vkey} key={key}")
    
        lon2d, lat2d = self._get_unified_grid()
    
        meta = upd.get("meta") or {}
        imt = meta.get("imt", "?")
        v0 = meta.get("prior_version", "?")
    
        prior_mean = np.asarray((upd.get("prior") or {}).get("mean_grid"), dtype=float)
        prior_sig = np.asarray((upd.get("prior") or {}).get("sigma_grid"), dtype=float)
        post_mean = np.asarray((upd.get("posterior") or {}).get("mean_grid"), dtype=float)
        post_sig = np.asarray((upd.get("posterior") or {}).get("sigma_grid"), dtype=float)
        obs = upd.get("obs_used")
    
        # diffs
        dmean = post_mean - prior_mean
        dsig = post_sig - prior_sig
    
        fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
        ax = axes.ravel()
    
        # titles
        fig.suptitle(
            f"Residual Kriging audit — v={vkey} key={key}\n"
            f"prior=v0({v0}) imt={imt}  rmse_prior={upd['metrics']['rmse_prior']:.4f}  rmse_post={upd['metrics']['rmse_post']:.4f}",
            fontsize=12
        )
    
        # prior mean
        im0 = ax[0].pcolormesh(lon2d, lat2d, prior_mean, shading="auto")
        fig.colorbar(im0, ax=ax[0])
        ax[0].set_title(f"Prior mean (v0 fixed) — {imt}")
    
        # post mean
        im1 = ax[1].pcolormesh(lon2d, lat2d, post_mean, shading="auto")
        fig.colorbar(im1, ax=ax[1])
        ax[1].set_title(f"Posterior mean (residual OK) — {imt} (+ obs)")
        if obs is not None and hasattr(obs, "empty") and not obs.empty:
            ax[1].scatter(obs["lon"].values, obs["lat"].values, s=6, c="k", alpha=0.55, linewidths=0)
    
        # dmean
        if show_diffs:
            im2 = ax[2].pcolormesh(lon2d, lat2d, dmean, shading="auto")
            fig.colorbar(im2, ax=ax[2])
            ax[2].set_title("Δ mean (post − prior)")
        else:
            ax[2].axis("off")
    
        # prior sigma
        im3 = ax[3].pcolormesh(lon2d, lat2d, prior_sig, shading="auto")
        fig.colorbar(im3, ax=ax[3])
        ax[3].set_title(f"Prior sigma ({self._sigma_field_for_imt(imt)})")
    
        # post sigma
        im4 = ax[4].pcolormesh(lon2d, lat2d, post_sig, shading="auto")
        fig.colorbar(im4, ax=ax[4])
        ax[4].set_title("Posterior sigma")
        if obs is not None and hasattr(obs, "empty") and not obs.empty:
            ax[4].scatter(obs["lon"].values, obs["lat"].values, s=6, c="k", alpha=0.35, linewidths=0)
    
        # dsig
        if show_diffs:
            im5 = ax[5].pcolormesh(lon2d, lat2d, dsig, shading="auto")
            fig.colorbar(im5, ax=ax[5])
            ax[5].set_title("Δ sigma (post − prior)")
        else:
            ax[5].axis("off")
    
        for a in ax:
            if a.has_data():
                a.set_xlabel("Lon")
                a.set_ylabel("Lat")
    
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    
        if save_path:
            fig.savefig(str(save_path), bbox_inches="tight")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig



    # ======================================================================
    # Residual / Innovation Kriging (FIXED PRIOR = first shakemap version)
    # - Prior mean/sigma ALWAYS taken from v0 (first version in unified stack)
    # - Observations ALWAYS taken from the target version
    # - Supports:
    #   - PGA: instruments (stationlist seismic)
    #   - MMI: DYFI stationlist and/or CDI (with conditioning knobs)
    # ======================================================================

    def _rk_get_prior_version_key(self):
        """
        Fixed prior selector for Residual Kriging.
        Default: first entry in uq_state['unified']['version_keys'] if available,
        otherwise lexicographically smallest version key in uq_state['versions'].
        """
        u = (self.uq_state or {}).get("unified") or {}
        vks = u.get("version_keys")
        if isinstance(vks, list) and len(vks) > 0:
            return str(vks[0])
        vks2 = sorted(list(((self.uq_state or {}).get("versions") or {}).keys()))
        if not vks2:
            raise RuntimeError("uq_state has no versions; run uq_build_dataset() first.")
        return str(vks2[0])

    def _rk_sample_grid_nn(self, lon2d, lat2d, grid2d, lon_pts, lat_pts):
        """
        Nearest-neighbor sampler on rectilinear lon/lat grids (ShakeMap-like).
        Returns sampled values at points.

        Assumes lon varies along axis=1, lat varies along axis=0 (standard grid).
        Handles increasing or decreasing lon/lat axes.
        """
        import numpy as np

        lon2d = np.asarray(lon2d, dtype=float)
        lat2d = np.asarray(lat2d, dtype=float)
        grid2d = np.asarray(grid2d, dtype=float)

        lon_axis = lon2d[0, :].astype(float)
        lat_axis = lat2d[:, 0].astype(float)

        lon_pts = np.asarray(lon_pts, dtype=float)
        lat_pts = np.asarray(lat_pts, dtype=float)

        # detect monotonic direction
        lon_inc = np.nanmean(np.diff(lon_axis)) >= 0
        lat_inc = np.nanmean(np.diff(lat_axis)) >= 0

        lon_use = lon_axis if lon_inc else lon_axis[::-1]
        lat_use = lat_axis if lat_inc else lat_axis[::-1]

        # nearest index via searchsorted + neighbor check
        j = np.searchsorted(lon_use, lon_pts)
        j = np.clip(j, 0, lon_use.size - 1)
        j0 = np.clip(j - 1, 0, lon_use.size - 1)
        j = np.where(np.abs(lon_pts - lon_use[j0]) <= np.abs(lon_pts - lon_use[j]), j0, j)

        i = np.searchsorted(lat_use, lat_pts)
        i = np.clip(i, 0, lat_use.size - 1)
        i0 = np.clip(i - 1, 0, lat_use.size - 1)
        i = np.where(np.abs(lat_pts - lat_use[i0]) <= np.abs(lat_pts - lat_use[i]), i0, i)

        # map indices back if reversed
        if not lon_inc:
            j = (lon_axis.size - 1) - j
        if not lat_inc:
            i = (lat_axis.size - 1) - i

        return grid2d[i, j]

    def run_residual_kriging_update(
        self,
        *,
        version,
        mode="pga_instr",
        key=None,
        dyfi_source="stationlist",          # for MMI: "stationlist" | "cdi" | "both" | "auto"
        use_cdi=False,                      # explicit CDI toggle (kept for convenience)
        margin_deg=0.05,

        # OK controls
        variogram_model="linear",
        variogram_parameters=None,
        neighbor_k=40,
        nugget=0.0,

        # residual update controls
        pga_space="log10",                  # "log10" or "linear" (recommend log10)
        pga_floor=1e-6,                     # avoid log10(0)
        sigma_clip_min=1e-6,

        # CDI conditioning knobs (passed to _cdi_condition)
        cdi_trust_nresp_ge=3,
        cdi_enable_local_outlier=True,
        cdi_local_radius_km=25.0,
        cdi_outlier_k=2.5,
        cdi_min_neighbors=4,
        cdi_enable_clustering=True,
        cdi_cluster_eps_km=2.0,
        cdi_cluster_min_pts=3,
        cdi_strategy=("local_outlier", "grid_thin", "quantile_residual"),
        cdi_grid_bin_km=10.0,
        cdi_grid_agg="median",
        cdi_quantile=(0.05, 0.95),

        verbose=False,
        debug=False,
    ):
        """
        Residual / Innovation Kriging update with FIXED PRIOR (v0).
        Observations are taken from the TARGET version, but residuals are computed against v0 prior.

        Storage:
          uq_state["versions"][vkey]["updates"][key] = {
            "meta": {...},
            "prior": {"version": v0, "imt": imt, "mean_grid":..., "sigma_grid":...},
            "posterior": {"mean_grid":..., "sigma_grid":...},
            "residual": {"resid_grid":..., "resid_var":...},
            "obs_used": DataFrame,
            "metrics": {...}
          }
        """
        import numpy as np
        import pandas as pd
        import time

        if not isinstance(self.uq_state, dict) or not (self.uq_state.get("versions") or {}):
            raise RuntimeError("uq_state not initialized. Run uq_build_dataset() first.")

        vkey = _norm_version(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey)
        if vpack is None:
            raise KeyError(f"Version not in uq_state: {vkey}")

        # unified grid
        lon2d, lat2d = self._get_unified_grid()

        # fixed prior version
        v0 = self._rk_get_prior_version_key()

        mode_s = str(mode).lower().strip()
        if key is None:
            key = f"rk__{mode_s}__v{vkey}"
        key = str(key)

        # storage container
        if "updates" not in vpack or not isinstance(vpack.get("updates"), dict):
            vpack["updates"] = {}

        # decide IMT + source routing
        if mode_s in ("pga", "pga_instr", "pga_instruments", "pga_station", "pga_stationlist"):
            imt_target = "PGA"
            dyfi_source_eff = "stationlist"  # irrelevant for PGA
            include_cdi = False
        elif mode_s in ("mmi", "mmi_dyfi", "mmi_stationlist"):
            imt_target = "MMI"
            dyfi_source_eff = "stationlist"
            include_cdi = False
        elif mode_s in ("mmi_cdi", "cdi"):
            imt_target = "MMI"
            dyfi_source_eff = "cdi"
            include_cdi = True
        elif mode_s in ("mmi_both", "mmi_dyfi_cdi", "both"):
            imt_target = "MMI"
            dyfi_source_eff = "both"
            include_cdi = True
        else:
            imt_target = "MMI" if "mmi" in mode_s else "PGA"
            dyfi_source_eff = str(dyfi_source).lower().strip()
            include_cdi = bool(use_cdi) or (dyfi_source_eff in ("cdi", "both", "auto"))

        t0 = time.time()

        # --- load prior mean/sigma on unified grid (v0 fixed) ---
        _, _, prior_mean = self._get_prior_mean_unified(v0, imt_target)
        _, _, prior_sigma = self._get_prior_sigma_unified(v0, imt_target)
        if prior_mean is None or prior_sigma is None:
            raise RuntimeError(f"Missing prior mean/sigma for v0={v0} imt={imt_target}.")
        prior_mean = np.asarray(prior_mean, dtype=float)
        prior_sigma = np.asarray(prior_sigma, dtype=float)
        if prior_mean.shape != lon2d.shape or prior_sigma.shape != lon2d.shape:
            raise RuntimeError(f"Prior shapes mismatch unified grid for imt={imt_target} (v0={v0}).")

        # --- build observations from TARGET version (vkey), NOT v0 ---
        obs = self.build_observations(version=vkey, imt=imt_target, dyfi_source=dyfi_source_eff, sigma_override=None)

        if obs is None or getattr(obs, "empty", True):
            out = {
                "ok": False,
                "note": f"No observations for imt={imt_target} dyfi_source={dyfi_source_eff} (v={vkey}).",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key},
            }
            vpack["updates"][key] = out
            return out

        # strict columns + numeric
        obs = obs.copy()
        for c in ("lon", "lat", "value", "sigma"):
            if c not in obs.columns:
                raise RuntimeError(f"build_observations did not return required column: {c}")
        obs["lon"] = pd.to_numeric(obs["lon"], errors="coerce")
        obs["lat"] = pd.to_numeric(obs["lat"], errors="coerce")
        obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
        obs["sigma"] = pd.to_numeric(obs["sigma"], errors="coerce")
        obs = obs.dropna(subset=["lon", "lat", "value"]).reset_index(drop=True)

        if obs.empty:
            out = {
                "ok": False,
                "note": f"Observations all invalid after numeric coercion (v={vkey}).",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key},
            }
            vpack["updates"][key] = out
            return out

        # ------------------------------------------------------------------
        # extent filter (NO drop= ; use project signature)
        # ------------------------------------------------------------------
        try:
            obs_f, obs_drop = self.filter_observations_to_extent(
                obs,
                version=vkey,
                grid_mode="unified",
                margin_deg=float(margin_deg),
                return_dropped=True,
            )
            meta_ext = {
                "grid_mode": "unified",
                "margin_deg": float(margin_deg),
                "kept": int(len(obs_f)) if hasattr(obs_f, "__len__") else None,
                "dropped": int(len(obs_drop)) if hasattr(obs_drop, "__len__") else None,
            }
        except TypeError:
            # fallback for older signature: returns kept only
            obs_f = self.filter_observations_to_extent(
                obs,
                version=vkey,
                grid_mode="unified",
                margin_deg=float(margin_deg),
            )
            meta_ext = {
                "grid_mode": "unified",
                "margin_deg": float(margin_deg),
                "kept": int(len(obs_f)) if hasattr(obs_f, "__len__") else None,
                "dropped": None,
            }

        if obs_f is None or getattr(obs_f, "empty", False) or len(obs_f) == 0:
            out = {
                "ok": False,
                "note": "All observations fell outside unified extent after margin filter.",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key, "extent": meta_ext},
            }
            vpack["updates"][key] = out
            return out

        # optional CDI conditioning (compare to v0 prior)
        cdi_meta = None
        if imt_target == "MMI" and include_cdi:
            try:
                obs_f, cdi_meta = self._cdi_condition(
                    version=v0,
                    obs_df=obs_f,
                    margin_deg=float(margin_deg),
                    cdi_trust_nresp_ge=int(cdi_trust_nresp_ge),
                    cdi_enable_local_outlier=bool(cdi_enable_local_outlier),
                    cdi_local_radius_km=float(cdi_local_radius_km),
                    cdi_outlier_k=float(cdi_outlier_k),
                    cdi_min_neighbors=int(cdi_min_neighbors),
                    cdi_enable_clustering=bool(cdi_enable_clustering),
                    cdi_cluster_eps_km=float(cdi_cluster_eps_km),
                    cdi_cluster_min_pts=int(cdi_cluster_min_pts),
                    cdi_strategy=cdi_strategy,
                    cdi_grid_bin_km=float(cdi_grid_bin_km),
                    cdi_grid_agg=str(cdi_grid_agg),
                    cdi_quantile=cdi_quantile,
                    debug=bool(debug),
                )
            except Exception as e:
                cdi_meta = {"ok": False, "err": str(e)}

        # --- sample prior at obs locations (v0 fixed) ---
        prior_at_obs = self._rk_sample_grid_nn(
            lon2d, lat2d, prior_mean,
            obs_f["lon"].to_numpy(dtype=float),
            obs_f["lat"].to_numpy(dtype=float),
        )

        # --- compute residuals in chosen space ---
        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            y = np.log10(np.maximum(obs_f["value"].to_numpy(dtype=float), float(pga_floor)))
            m = np.log10(np.maximum(prior_at_obs.astype(float), float(pga_floor)))
            resid = y - m
        else:
            y = obs_f["value"].to_numpy(dtype=float)
            m = prior_at_obs.astype(float)
            resid = y - m

        # sigma clip (future: weighted kriging)
        sig_obs = obs_f["sigma"].to_numpy(dtype=float)
        sig_obs = np.where(np.isfinite(sig_obs), sig_obs, np.nan)
        sig_obs = np.where(sig_obs > float(sigma_clip_min), sig_obs, float(sigma_clip_min))
        _w = 1.0 / (sig_obs ** 2)  # reserved

        # --- OK krige residuals on the unified grid ---
        zhat_resid, var_resid = self._rk_ok_krige_grid(
            obs_f["lon"].to_numpy(dtype=float),
            obs_f["lat"].to_numpy(dtype=float),
            np.asarray(resid, dtype=float),
            lon2d, lat2d,
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            neighbor_k=int(neighbor_k),
            nugget=float(nugget),
            verbose=bool(verbose),
        )
        zhat_resid = np.asarray(zhat_resid, dtype=float)
        var_resid = np.asarray(var_resid, dtype=float)

        # --- posterior mean back to original space ---
        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            post_mean = 10.0 ** (np.log10(np.maximum(prior_mean, float(pga_floor))) + zhat_resid)
            post_sigma = np.sqrt(
                np.maximum(prior_sigma, 0.0) ** 2 +
                (np.log(10.0) * np.maximum(post_mean, float(pga_floor))) ** 2 * np.maximum(var_resid, 0.0)
            )
        else:
            post_mean = prior_mean + zhat_resid
            post_sigma = np.sqrt(np.maximum(prior_sigma, 0.0) ** 2 + np.maximum(var_resid, 0.0))

        # --- metrics vs obs ---
        post_at_obs = self._rk_sample_grid_nn(
            lon2d, lat2d, post_mean,
            obs_f["lon"].to_numpy(dtype=float),
            obs_f["lat"].to_numpy(dtype=float),
        )

        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            prior_pred = np.log10(np.maximum(prior_at_obs.astype(float), float(pga_floor)))
            post_pred = np.log10(np.maximum(post_at_obs.astype(float), float(pga_floor)))
            obs_y = np.log10(np.maximum(obs_f["value"].to_numpy(dtype=float), float(pga_floor)))
        else:
            prior_pred = prior_at_obs.astype(float)
            post_pred = post_at_obs.astype(float)
            obs_y = obs_f["value"].to_numpy(dtype=float)

        rmse_prior = float(np.sqrt(np.nanmean((prior_pred - obs_y) ** 2)))
        rmse_post = float(np.sqrt(np.nanmean((post_pred - obs_y) ** 2)))
        mae_prior = float(np.nanmean(np.abs(prior_pred - obs_y)))
        mae_post = float(np.nanmean(np.abs(post_pred - obs_y)))

        out = {
            "ok": True,
            "meta": {
                "version": vkey,
                "prior_version": v0,
                "imt": imt_target,
                "mode": mode_s,
                "key": key,
                "dyfi_source": dyfi_source_eff,
                "include_cdi": bool(include_cdi),
                "margin_deg": float(margin_deg),
                "variogram_model": str(variogram_model),
                "variogram_parameters": variogram_parameters,
                "neighbor_k": int(neighbor_k),
                "nugget": float(nugget),
                "pga_space": str(pga_space),
                "runtime_s": float(time.time() - t0),
                "extent": meta_ext,
            },
            "prior": {"version": v0, "mean_grid": prior_mean, "sigma_grid": prior_sigma},
            "residual": {"resid_grid": zhat_resid, "resid_var": var_resid},
            "posterior": {"mean_grid": post_mean, "sigma_grid": post_sigma},
            "obs_used": obs_f,
            "metrics": {
                "n_obs": int(len(obs_f)),
                "rmse_prior": rmse_prior,
                "rmse_post": rmse_post,
                "mae_prior": mae_prior,
                "mae_post": mae_post,
            },
        }
        if cdi_meta is not None:
            out["meta"]["cdi_conditioning"] = cdi_meta

        vpack["updates"][key] = out

        if verbose:
            print(
                f"[RK] v={vkey} prior=v0({v0}) imt={imt_target} "
                f"n={len(obs_f)} rmse_prior={rmse_prior:.4f} rmse_post={rmse_post:.4f}"
            )

        return out




    # ======================================================================
    # PATCH: Residual / Innovation Kriging (v0 fixed prior) — kwargs-safe
    # Fixes:
    #   - filter_observations_to_extent() has no drop= in this codebase
    #   - _ok_krige_grid() may not accept variogram_parameters=
    #   - ensure _rk_sample_grid_nn exists as a class method
    # ======================================================================

    def _rk_sample_grid_nn(self, lon2d, lat2d, grid2d, lon_pts, lat_pts):
        """
        Nearest-neighbor sampler on rectilinear lon/lat grids (ShakeMap-like).
        Returns sampled values at points.
        """
        import numpy as np

        lon2d = np.asarray(lon2d, dtype=float)
        lat2d = np.asarray(lat2d, dtype=float)
        grid2d = np.asarray(grid2d, dtype=float)

        lon1 = lon2d[0, :]
        lat1 = lat2d[:, 0]

        lon_pts = np.asarray(lon_pts, dtype=float)
        lat_pts = np.asarray(lat_pts, dtype=float)

        # Handle monotonic direction
        lon_inc = np.nanmean(np.diff(lon1)) >= 0
        lat_inc = np.nanmean(np.diff(lat1)) >= 0

        if not lon_inc:
            lon1 = lon1[::-1]
            grid2d = grid2d[:, ::-1]
        if not lat_inc:
            lat1 = lat1[::-1]
            grid2d = grid2d[::-1, :]

        # NN indices
        j = np.searchsorted(lon1, lon_pts, side="left")
        i = np.searchsorted(lat1, lat_pts, side="left")

        j = np.clip(j, 0, len(lon1) - 1)
        i = np.clip(i, 0, len(lat1) - 1)

        # Try neighbor on the left if closer
        j0 = np.clip(j - 1, 0, len(lon1) - 1)
        i0 = np.clip(i - 1, 0, len(lat1) - 1)

        choose_j0 = np.abs(lon1[j0] - lon_pts) < np.abs(lon1[j] - lon_pts)
        choose_i0 = np.abs(lat1[i0] - lat_pts) < np.abs(lat1[i] - lat_pts)

        j = np.where(choose_j0, j0, j)
        i = np.where(choose_i0, i0, i)

        out = grid2d[i, j]
        return out

    def _rk_ok_krige_grid(
        self,
        x,
        y,
        z,
        xgrid2d,
        ygrid2d,
        *,
        variogram_model="linear",
        variogram_parameters=None,
        neighbor_k=40,
        nugget=0.0,
        verbose=False,
    ):
        """
        Wrapper that calls the module's kriging backend safely.

        IMPORTANT:
        - Your SHAKEuq._ok_krige_grid signature may NOT accept variogram_parameters.
        - This wrapper inspects the callable signature and only passes supported kwargs.
        """
        import inspect

        # 1) Preferred: existing helper already used by OK in this module
        if hasattr(self, "_ok_krige_grid") and callable(getattr(self, "_ok_krige_grid")):
            fn = getattr(self, "_ok_krige_grid")
            sig = None
            try:
                sig = inspect.signature(fn)
                accepted = set(sig.parameters.keys())
            except Exception:
                accepted = None  # can't inspect -> be conservative

            kwargs = {}
            # always try the most common kwargs
            kwargs["variogram_model"] = variogram_model
            kwargs["neighbor_k"] = neighbor_k
            kwargs["nugget"] = nugget
            kwargs["verbose"] = verbose

            # only pass variogram_parameters if backend supports it
            if variogram_parameters is not None:
                if (accepted is None) or ("variogram_parameters" in accepted):
                    kwargs["variogram_parameters"] = variogram_parameters

            # also protect against backends that use different names
            if accepted is not None:
                kwargs = {k: v for k, v in kwargs.items() if k in accepted}

            return fn(x, y, z, xgrid2d, ygrid2d, **kwargs)

        # 2) Fallback: if your codebase has another backend, add it here
        raise RuntimeError("No kriging backend found: expected self._ok_krige_grid to exist.")

    def run_residual_kriging_update(
        self,
        *,
        version,
        mode="pga_instr",
        key=None,
        dyfi_source="stationlist",          # for MMI: "stationlist" | "cdi" | "both" | "auto"
        use_cdi=False,                      # explicit CDI toggle
        margin_deg=0.05,

        # OK controls (kwargs-safe through _rk_ok_krige_grid)
        variogram_model="linear",
        variogram_parameters=None,
        neighbor_k=40,
        nugget=0.0,

        # residual update controls
        pga_space="log10",                  # "log10" or "linear"
        pga_floor=1e-6,
        sigma_clip_min=1e-6,

        # CDI conditioning knobs
        cdi_trust_nresp_ge=3,
        cdi_enable_local_outlier=True,
        cdi_local_radius_km=25.0,
        cdi_outlier_k=2.5,
        cdi_min_neighbors=4,
        cdi_enable_clustering=True,
        cdi_cluster_eps_km=2.0,
        cdi_cluster_min_pts=3,
        cdi_strategy=("local_outlier", "grid_thin", "quantile_residual"),
        cdi_grid_bin_km=10.0,
        cdi_grid_agg="median",
        cdi_quantile=(0.05, 0.95),

        verbose=False,
        debug=False,
    ):
        """
        Residual / Innovation Kriging update with FIXED PRIOR (v0).
        Observations are taken from the TARGET version, but residuals are computed against v0 prior.
        """
        import numpy as np
        import pandas as pd
        import time

        if not isinstance(self.uq_state, dict) or not (self.uq_state.get("versions") or {}):
            raise RuntimeError("uq_state not initialized. Run uq_build_dataset() first.")

        vkey = _norm_version(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey)
        if vpack is None:
            raise KeyError(f"Version not in uq_state: {vkey}")

        # unified grid
        lon2d, lat2d = self._get_unified_grid()

        # fixed prior version key (v0)
        v0 = self._rk_get_prior_version_key()

        mode_s = str(mode).lower().strip()
        if key is None:
            key = f"rk__{mode_s}__v{vkey}"
        key = str(key)

        # storage container
        if "updates" not in vpack or not isinstance(vpack.get("updates"), dict):
            vpack["updates"] = {}

        # Decide IMT + routing
        if mode_s in ("pga", "pga_instr", "pga_instruments", "pga_station", "pga_stationlist"):
            imt_target = "PGA"
            dyfi_source_eff = "stationlist"
            include_cdi = False
        elif mode_s in ("mmi", "mmi_dyfi", "mmi_stationlist"):
            imt_target = "MMI"
            dyfi_source_eff = "stationlist"
            include_cdi = False
        elif mode_s in ("mmi_cdi", "cdi"):
            imt_target = "MMI"
            dyfi_source_eff = "cdi"
            include_cdi = True
        elif mode_s in ("mmi_both", "mmi_dyfi_cdi", "both"):
            imt_target = "MMI"
            dyfi_source_eff = "both"
            include_cdi = True
        else:
            imt_target = "MMI" if "mmi" in mode_s else "PGA"
            dyfi_source_eff = str(dyfi_source).lower().strip()
            include_cdi = bool(use_cdi) or (dyfi_source_eff in ("cdi", "both", "auto"))

        t0 = time.time()

        # --- load prior mean/sigma on unified grid (v0 fixed) ---
        _, _, prior_mean = self._get_prior_mean_unified(v0, imt_target)
        _, _, prior_sigma = self._get_prior_sigma_unified(v0, imt_target)
        if prior_mean is None or prior_sigma is None:
            raise RuntimeError(f"Missing prior mean/sigma for v0={v0} imt={imt_target}.")
        prior_mean = np.asarray(prior_mean, dtype=float)
        prior_sigma = np.asarray(prior_sigma, dtype=float)
        if prior_mean.shape != lon2d.shape or prior_sigma.shape != lon2d.shape:
            raise RuntimeError(f"Prior shapes mismatch unified grid for imt={imt_target} (v0={v0}).")

        # --- build observations from TARGET version (vkey), NOT v0 ---
        obs = self.build_observations(version=vkey, imt=imt_target, dyfi_source=dyfi_source_eff, sigma_override=None)

        if obs is None or getattr(obs, "empty", True):
            out = {
                "ok": False,
                "note": f"No observations for imt={imt_target} dyfi_source={dyfi_source_eff} (v={vkey}).",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key},
            }
            vpack["updates"][key] = out
            return out

        obs = obs.copy()
        for c in ("lon", "lat", "value", "sigma"):
            if c not in obs.columns:
                raise RuntimeError(f"build_observations did not return required column: {c}")
        obs["lon"] = pd.to_numeric(obs["lon"], errors="coerce")
        obs["lat"] = pd.to_numeric(obs["lat"], errors="coerce")
        obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
        obs["sigma"] = pd.to_numeric(obs["sigma"], errors="coerce")
        obs = obs.dropna(subset=["lon", "lat", "value"]).reset_index(drop=True)
        if obs.empty:
            out = {
                "ok": False,
                "note": f"Observations all invalid after numeric coercion (v={vkey}).",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key},
            }
            vpack["updates"][key] = out
            return out

        # ------------------------------------------------------------------
        # EXTENT FILTER — match your real API (no drop=)
        # Prefer (kept, dropped) interface; fallback to kept-only.
        # ------------------------------------------------------------------
        try:
            obs_f, obs_drop = self.filter_observations_to_extent(
                obs,
                version=vkey,
                grid_mode="unified",
                margin_deg=float(margin_deg),
                return_dropped=True,
            )
            meta_ext = {
                "grid_mode": "unified",
                "margin_deg": float(margin_deg),
                "kept": int(len(obs_f)) if hasattr(obs_f, "__len__") else None,
                "dropped": int(len(obs_drop)) if hasattr(obs_drop, "__len__") else None,
            }
        except TypeError:
            obs_f = self.filter_observations_to_extent(
                obs,
                version=vkey,
                grid_mode="unified",
                margin_deg=float(margin_deg),
            )
            meta_ext = {
                "grid_mode": "unified",
                "margin_deg": float(margin_deg),
                "kept": int(len(obs_f)) if hasattr(obs_f, "__len__") else None,
                "dropped": None,
            }

        if obs_f is None or getattr(obs_f, "empty", False) or len(obs_f) == 0:
            out = {
                "ok": False,
                "note": "All observations fell outside unified extent after margin filter.",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key, "extent": meta_ext},
            }
            vpack["updates"][key] = out
            return out

        # optional CDI conditioning (compare/condition against v0 prior)
        cdi_meta = None
        if imt_target == "MMI" and include_cdi:
            try:
                obs_f, cdi_meta = self._cdi_condition(
                    version=v0,
                    obs_df=obs_f,
                    margin_deg=float(margin_deg),
                    cdi_trust_nresp_ge=int(cdi_trust_nresp_ge),
                    cdi_enable_local_outlier=bool(cdi_enable_local_outlier),
                    cdi_local_radius_km=float(cdi_local_radius_km),
                    cdi_outlier_k=float(cdi_outlier_k),
                    cdi_min_neighbors=int(cdi_min_neighbors),
                    cdi_enable_clustering=bool(cdi_enable_clustering),
                    cdi_cluster_eps_km=float(cdi_cluster_eps_km),
                    cdi_cluster_min_pts=int(cdi_cluster_min_pts),
                    cdi_strategy=cdi_strategy,
                    cdi_grid_bin_km=float(cdi_grid_bin_km),
                    cdi_grid_agg=str(cdi_grid_agg),
                    cdi_quantile=cdi_quantile,
                    debug=bool(debug),
                )
            except Exception as e:
                cdi_meta = {"ok": False, "err": str(e)}

        # --- sample prior at obs locations (v0 fixed) ---
        prior_at_obs = self._rk_sample_grid_nn(lon2d, lat2d, prior_mean, obs_f["lon"].values, obs_f["lat"].values)

        # --- compute residuals in chosen space ---
        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            y = np.log10(np.maximum(obs_f["value"].to_numpy(dtype=float), float(pga_floor)))
            m = np.log10(np.maximum(prior_at_obs.astype(float), float(pga_floor)))
            resid = y - m
        else:
            y = obs_f["value"].to_numpy(dtype=float)
            m = prior_at_obs.astype(float)
            resid = y - m

        # sigma sanity (weights reserved for future use)
        sig_obs = obs_f["sigma"].to_numpy(dtype=float)
        sig_obs = np.where(np.isfinite(sig_obs), sig_obs, np.nan)
        sig_obs = np.where(sig_obs > float(sigma_clip_min), sig_obs, float(sigma_clip_min))
        _w = 1.0 / (sig_obs ** 2)  # reserved

        # --- OK krige residuals on the unified grid (kwargs-safe wrapper) ---
        zhat_resid, var_resid = self._rk_ok_krige_grid(
            obs_f["lon"].to_numpy(dtype=float),
            obs_f["lat"].to_numpy(dtype=float),
            np.asarray(resid, dtype=float),
            lon2d, lat2d,
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,  # wrapper will only pass if backend supports
            neighbor_k=int(neighbor_k),
            nugget=float(nugget),
            verbose=bool(verbose),
        )

        zhat_resid = np.asarray(zhat_resid, dtype=float)
        var_resid = np.asarray(var_resid, dtype=float)

        # --- posterior mean back to original space ---
        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            post_mean = 10.0 ** (np.log10(np.maximum(prior_mean, float(pga_floor))) + zhat_resid)
            post_sigma = np.sqrt(
                np.maximum(prior_sigma, 0.0) ** 2 +
                (np.log(10.0) * np.maximum(post_mean, float(pga_floor))) ** 2 * np.maximum(var_resid, 0.0)
            )
        else:
            post_mean = prior_mean + zhat_resid
            post_sigma = np.sqrt(np.maximum(prior_sigma, 0.0) ** 2 + np.maximum(var_resid, 0.0))

        # --- quick metrics vs obs (prior/post evaluated at obs) ---
        post_at_obs = self._rk_sample_grid_nn(lon2d, lat2d, post_mean, obs_f["lon"].values, obs_f["lat"].values)

        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            prior_pred = np.log10(np.maximum(prior_at_obs.astype(float), float(pga_floor)))
            post_pred = np.log10(np.maximum(post_at_obs.astype(float), float(pga_floor)))
            obs_y = np.log10(np.maximum(obs_f["value"].to_numpy(dtype=float), float(pga_floor)))
        else:
            prior_pred = prior_at_obs.astype(float)
            post_pred = post_at_obs.astype(float)
            obs_y = obs_f["value"].to_numpy(dtype=float)

        rmse_prior = float(np.sqrt(np.nanmean((prior_pred - obs_y) ** 2)))
        rmse_post = float(np.sqrt(np.nanmean((post_pred - obs_y) ** 2)))
        mae_prior = float(np.nanmean(np.abs(prior_pred - obs_y)))
        mae_post = float(np.nanmean(np.abs(post_pred - obs_y)))

        out = {
            "ok": True,
            "meta": {
                "version": vkey,
                "prior_version": v0,
                "imt": imt_target,
                "mode": mode_s,
                "key": key,
                "dyfi_source": dyfi_source_eff,
                "include_cdi": bool(include_cdi),
                "cdi_meta": cdi_meta,
                "margin_deg": float(margin_deg),
                "variogram_model": str(variogram_model),
                "variogram_parameters": variogram_parameters,
                "neighbor_k": int(neighbor_k),
                "nugget": float(nugget),
                "pga_space": str(pga_space),
                "runtime_s": float(time.time() - t0),
                "extent": meta_ext,
            },
            "prior": {"version": v0, "mean_grid": prior_mean, "sigma_grid": prior_sigma},
            "residual": {"resid_grid": zhat_resid, "resid_var": var_resid},
            "posterior": {"mean_grid": post_mean, "sigma_grid": post_sigma},
            "obs_used": obs_f,
            "metrics": {
                "n_obs": int(len(obs_f)),
                "rmse_prior": rmse_prior,
                "rmse_post": rmse_post,
                "mae_prior": mae_prior,
                "mae_post": mae_post,
            },
        }

        vpack["updates"][key] = out

        if verbose:
            print(
                f"[RK] v={vkey} prior=v0({v0}) imt={imt_target} "
                f"n={len(obs_f)} rmse_prior={rmse_prior:.4f} rmse_post={rmse_post:.4f}"
            )

        return out





    def _rk_ok_krige_grid(
        self,
        lon_obs,
        lat_obs,
        z_obs,
        lon2d,
        lat2d,
        *,
        sigma_obs=None,
        neighbor_k=25,
        max_points=None,
        use_obs_sigma=True,
        variogram_model="exponential",
        range_km=80.0,
        sill=1.0,
        nugget=1e-6,
        ridge=1e-10,
        debug=False,
    ):
        """
        Adapter to the *actual* SHAKEuq backend:

            self._ok_krige_grid(obs_df, lon2d, lat2d, *, neighbor_k=..., range_km=..., ...)

        Returns: (mean2d, var2d)
        """
        import numpy as np
        import pandas as pd

        if not hasattr(self, "_ok_krige_grid") or not callable(getattr(self, "_ok_krige_grid")):
            raise RuntimeError("Expected SHAKEuq._ok_krige_grid to exist but it was not found.")

        lon_obs = np.asarray(lon_obs, dtype=float).ravel()
        lat_obs = np.asarray(lat_obs, dtype=float).ravel()
        z_obs = np.asarray(z_obs, dtype=float).ravel()

        if lon_obs.size != lat_obs.size or lon_obs.size != z_obs.size:
            raise ValueError("lon_obs, lat_obs, z_obs must have the same length.")

        df = pd.DataFrame({"lon": lon_obs, "lat": lat_obs, "value": z_obs})
        if sigma_obs is not None:
            sigma_obs = np.asarray(sigma_obs, dtype=float).ravel()
            if sigma_obs.size == lon_obs.size:
                df["sigma"] = sigma_obs

        # Call your real backend
        mean2d, var2d = self._ok_krige_grid(
            df,
            lon2d, lat2d,
            neighbor_k=int(neighbor_k) if neighbor_k is not None else None,
            max_points=max_points,
            use_obs_sigma=bool(use_obs_sigma),
            variogram_model=str(variogram_model),
            range_km=float(range_km),
            sill=float(sill),
            nugget=float(nugget),
            ridge=float(ridge),
            debug=bool(debug),
        )
        return mean2d, var2d

    def run_residual_kriging_update(
        self,
        *,
        version,
        mode="pga_instr",
        key=None,
        dyfi_source="stationlist",
        use_cdi=False,
        margin_deg=0.05,

        # OK controls (MATCH YOUR _ok_krige_grid)
        neighbor_k=25,
        max_points=None,
        use_obs_sigma=True,
        variogram_model="exponential",
        range_km=80.0,
        sill=1.0,
        nugget=1e-6,
        ridge=1e-10,

        # residual update controls
        pga_space="log10",      # "log10" or "linear"
        pga_floor=1e-6,
        sigma_clip_min=1e-6,

        # CDI conditioning knobs
        cdi_trust_nresp_ge=3,
        cdi_enable_local_outlier=True,
        cdi_local_radius_km=25.0,
        cdi_outlier_k=2.5,
        cdi_min_neighbors=4,
        cdi_enable_clustering=True,
        cdi_cluster_eps_km=2.0,
        cdi_cluster_min_pts=3,
        cdi_strategy=("local_outlier", "grid_thin", "quantile_residual"),
        cdi_grid_bin_km=10.0,
        cdi_grid_agg="median",
        cdi_quantile=(0.05, 0.95),

        verbose=False,
        debug=False,
    ):
        """
        Residual / Innovation Kriging with FIXED PRIOR (v0):
          - Prior mean/sigma from v0 (first version)
          - Observations from target version (vkey)
          - Residuals: r = y - prior(x)
          - Krige residuals on unified grid and update:
                post_mean = prior_mean + r_kriged
          - Sigma (pragmatic):
                post_sigma = sqrt(prior_sigma^2 + var_r_kriged)
            For PGA in log10 mode: propagate var_r in log10 → linear via derivative.
        """
        import numpy as np
        import pandas as pd
        import time

        if not isinstance(self.uq_state, dict) or not (self.uq_state.get("versions") or {}):
            raise RuntimeError("uq_state not initialized. Run uq_build_dataset() first.")

        vkey = _norm_version(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey)
        if vpack is None:
            raise KeyError(f"Version not in uq_state: {vkey}")

        lon2d, lat2d = self._get_unified_grid()

        # fixed prior version (v0)
        v0 = self._rk_get_prior_version_key()

        mode_s = str(mode).lower().strip()
        if key is None:
            key = f"rk__{mode_s}__v{vkey}"
        key = str(key)

        if "updates" not in vpack or not isinstance(vpack.get("updates"), dict):
            vpack["updates"] = {}

        # Decide IMT + routing
        if mode_s in ("pga", "pga_instr", "pga_instruments", "pga_station", "pga_stationlist"):
            imt_target = "PGA"
            dyfi_source_eff = "stationlist"
            include_cdi = False
        elif mode_s in ("mmi", "mmi_dyfi", "mmi_stationlist"):
            imt_target = "MMI"
            dyfi_source_eff = "stationlist"
            include_cdi = False
        elif mode_s in ("mmi_cdi", "cdi"):
            imt_target = "MMI"
            dyfi_source_eff = "cdi"
            include_cdi = True
        elif mode_s in ("mmi_both", "mmi_dyfi_cdi", "both"):
            imt_target = "MMI"
            dyfi_source_eff = "both"
            include_cdi = True
        else:
            imt_target = "MMI" if "mmi" in mode_s else "PGA"
            dyfi_source_eff = str(dyfi_source).lower().strip()
            include_cdi = bool(use_cdi) or (dyfi_source_eff in ("cdi", "both", "auto"))

        t0 = time.time()

        # ---- prior on unified grid (v0 fixed) ----
        _, _, prior_mean = self._get_prior_mean_unified(v0, imt_target)
        _, _, prior_sigma = self._get_prior_sigma_unified(v0, imt_target)
        if prior_mean is None or prior_sigma is None:
            raise RuntimeError(f"Missing prior mean/sigma for v0={v0} imt={imt_target}.")
        prior_mean = np.asarray(prior_mean, dtype=float)
        prior_sigma = np.asarray(prior_sigma, dtype=float)

        # ---- observations from TARGET version ----
        obs = self.build_observations(version=vkey, imt=imt_target, dyfi_source=dyfi_source_eff, sigma_override=None)
        if obs is None or getattr(obs, "empty", True):
            out = {"ok": False, "note": "No observations.", "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key}}
            vpack["updates"][key] = out
            return out

        obs = obs.copy()
        for c in ("lon", "lat", "value", "sigma"):
            if c not in obs.columns:
                raise RuntimeError(f"build_observations missing column: {c}")

        obs["lon"] = pd.to_numeric(obs["lon"], errors="coerce")
        obs["lat"] = pd.to_numeric(obs["lat"], errors="coerce")
        obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
        obs["sigma"] = pd.to_numeric(obs["sigma"], errors="coerce")
        obs = obs.dropna(subset=["lon", "lat", "value"]).reset_index(drop=True)
        if obs.empty:
            out = {"ok": False, "note": "Obs invalid after coercion.", "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key}}
            vpack["updates"][key] = out
            return out

        # ---- extent filter (match your API; no drop=) ----
        try:
            obs_f, obs_drop = self.filter_observations_to_extent(
                obs, version=vkey, grid_mode="unified", margin_deg=float(margin_deg), return_dropped=True
            )
            meta_ext = {"grid_mode": "unified", "margin_deg": float(margin_deg), "kept": int(len(obs_f)), "dropped": int(len(obs_drop))}
        except TypeError:
            obs_f = self.filter_observations_to_extent(
                obs, version=vkey, grid_mode="unified", margin_deg=float(margin_deg)
            )
            meta_ext = {"grid_mode": "unified", "margin_deg": float(margin_deg), "kept": int(len(obs_f)), "dropped": None}

        if obs_f is None or getattr(obs_f, "empty", False) or len(obs_f) == 0:
            out = {"ok": False, "note": "All obs out of extent.", "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key, "extent": meta_ext}}
            vpack["updates"][key] = out
            return out

        # ---- optional CDI conditioning vs v0 prior ----
        cdi_meta = None
        if imt_target == "MMI" and include_cdi:
            try:
                obs_f, cdi_meta = self._cdi_condition(
                    version=v0,
                    obs_df=obs_f,
                    margin_deg=float(margin_deg),
                    cdi_trust_nresp_ge=int(cdi_trust_nresp_ge),
                    cdi_enable_local_outlier=bool(cdi_enable_local_outlier),
                    cdi_local_radius_km=float(cdi_local_radius_km),
                    cdi_outlier_k=float(cdi_outlier_k),
                    cdi_min_neighbors=int(cdi_min_neighbors),
                    cdi_enable_clustering=bool(cdi_enable_clustering),
                    cdi_cluster_eps_km=float(cdi_cluster_eps_km),
                    cdi_cluster_min_pts=int(cdi_cluster_min_pts),
                    cdi_strategy=cdi_strategy,
                    cdi_grid_bin_km=float(cdi_grid_bin_km),
                    cdi_grid_agg=str(cdi_grid_agg),
                    cdi_quantile=cdi_quantile,
                    debug=bool(debug),
                )
            except Exception as e:
                cdi_meta = {"ok": False, "err": str(e)}

        # ---- residuals against v0 prior ----
        prior_at_obs = self._rk_sample_grid_nn(lon2d, lat2d, prior_mean, obs_f["lon"].values, obs_f["lat"].values)

        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            y = np.log10(np.maximum(obs_f["value"].to_numpy(dtype=float), float(pga_floor)))
            m = np.log10(np.maximum(prior_at_obs.astype(float), float(pga_floor)))
            resid = y - m
        else:
            y = obs_f["value"].to_numpy(dtype=float)
            m = prior_at_obs.astype(float)
            resid = y - m

        # sigma for use_obs_sigma
        sig_obs = obs_f["sigma"].to_numpy(dtype=float)
        sig_obs = np.where(np.isfinite(sig_obs), sig_obs, np.nan)
        sig_obs = np.where(sig_obs > float(sigma_clip_min), sig_obs, float(sigma_clip_min))

        # ---- krige residuals using your backend adapter ----
        zhat_resid, var_resid = self._rk_ok_krige_grid(
            obs_f["lon"].to_numpy(dtype=float),
            obs_f["lat"].to_numpy(dtype=float),
            np.asarray(resid, dtype=float),
            lon2d, lat2d,
            sigma_obs=sig_obs,
            neighbor_k=neighbor_k,
            max_points=max_points,
            use_obs_sigma=use_obs_sigma,
            variogram_model=variogram_model,
            range_km=range_km,
            sill=sill,
            nugget=nugget,
            ridge=ridge,
            debug=bool(debug),
        )
        if zhat_resid is None or var_resid is None:
            out = {"ok": False, "note": "Kriging backend returned None.", "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key}}
            vpack["updates"][key] = out
            return out

        zhat_resid = np.asarray(zhat_resid, dtype=float)
        var_resid = np.asarray(var_resid, dtype=float)

        # ---- posterior ----
        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            post_mean = 10.0 ** (np.log10(np.maximum(prior_mean, float(pga_floor))) + zhat_resid)
            post_sigma = np.sqrt(
                np.maximum(prior_sigma, 0.0) ** 2 +
                (np.log(10.0) * np.maximum(post_mean, float(pga_floor))) ** 2 * np.maximum(var_resid, 0.0)
            )
        else:
            post_mean = prior_mean + zhat_resid
            post_sigma = np.sqrt(np.maximum(prior_sigma, 0.0) ** 2 + np.maximum(var_resid, 0.0))

        # ---- metrics at obs ----
        post_at_obs = self._rk_sample_grid_nn(lon2d, lat2d, post_mean, obs_f["lon"].values, obs_f["lat"].values)

        if imt_target == "PGA" and str(pga_space).lower().strip() == "log10":
            prior_pred = np.log10(np.maximum(prior_at_obs.astype(float), float(pga_floor)))
            post_pred = np.log10(np.maximum(post_at_obs.astype(float), float(pga_floor)))
            obs_y = np.log10(np.maximum(obs_f["value"].to_numpy(dtype=float), float(pga_floor)))
        else:
            prior_pred = prior_at_obs.astype(float)
            post_pred = post_at_obs.astype(float)
            obs_y = obs_f["value"].to_numpy(dtype=float)

        rmse_prior = float(np.sqrt(np.nanmean((prior_pred - obs_y) ** 2)))
        rmse_post = float(np.sqrt(np.nanmean((post_pred - obs_y) ** 2)))
        mae_prior = float(np.nanmean(np.abs(prior_pred - obs_y)))
        mae_post = float(np.nanmean(np.abs(post_pred - obs_y)))

        out = {
            "ok": True,
            "meta": {
                "version": vkey,
                "prior_version": v0,
                "imt": imt_target,
                "mode": mode_s,
                "key": key,
                "dyfi_source": dyfi_source_eff,
                "include_cdi": bool(include_cdi),
                "cdi_meta": cdi_meta,
                "extent": meta_ext,
                "pga_space": str(pga_space),
                "kriging": {
                    "neighbor_k": neighbor_k,
                    "max_points": max_points,
                    "use_obs_sigma": bool(use_obs_sigma),
                    "variogram_model": str(variogram_model),
                    "range_km": float(range_km),
                    "sill": float(sill),
                    "nugget": float(nugget),
                    "ridge": float(ridge),
                },
                "runtime_s": float(time.time() - t0),
            },
            "prior": {"version": v0, "mean_grid": prior_mean, "sigma_grid": prior_sigma},
            "residual": {"resid_grid": zhat_resid, "resid_var": var_resid},
            "posterior": {"mean_grid": post_mean, "sigma_grid": post_sigma},
            "obs_used": obs_f,
            "metrics": {
                "n_obs": int(len(obs_f)),
                "rmse_prior": rmse_prior,
                "rmse_post": rmse_post,
                "mae_prior": mae_prior,
                "mae_post": mae_post,
            },
        }

        vpack["updates"][key] = out

        if verbose:
            print(
                f"[RK] v={vkey} prior=v0({v0}) imt={imt_target} "
                f"n={len(obs_f)} rmse_prior={rmse_prior:.4f} rmse_post={rmse_post:.4f}"
            )

        return out



