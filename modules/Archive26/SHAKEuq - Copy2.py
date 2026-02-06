"""

SHAKEuq — ShakeMap uncertainty dataset builder, observation adapter,
and staging layer for uncertainty quantification and data assimilation.

PURPOSE
-------
SHAKEuq builds a *version-indexed* in-memory dataset for a seismic event
(or a synthetic simulation) and produces a *unified, version-stacked grid*
in which all ShakeMap versions are remapped onto a single reference grid.

This class is intentionally NOT a solver. It is the data integrity,
alignment, and auditing layer that sits *before* uncertainty quantification
(UQ) and assimilation methods (Bayesian update, kriging, residual kriging).

Core guarantees:
  - Each ShakeMap version remains isolated in uq_state["versions"][vkey]
  - Unified stacks preserve version ordering via uq_state["unified"]["version_keys"]
  - Mean and uncertainty fields are never mixed implicitly
  - Observations are strictly IMT-isolated (e.g., no CDI leakage into PGA)

This design supports the central thesis that SHAKEmaps must be treated as
*time-dependent uncertain fields*, not static final products.

DATA MODEL OVERVIEW
-------------------
All data are stored in self.uq_state (dict). Top-level keys:

  uq_state["config"]   : run configuration + defaults used
  uq_state["versions"] : per-version products, indexed by version key vkey (e.g. "001")
  uq_state["unified"]  : version-stacked mean/sigma grids on a reference grid
  uq_state["sanity"]   : per-version audit table (pandas DataFrame)
  uq_state["truth"]    : (synthetic mode only) ground-truth metadata

VERSION KEYS
------------
Version keys are normalized to 3-digit strings using _norm_version():
  "1" -> "001", 1 -> "001", "014" -> "014"

The canonical order is preserved in:
  self.version_list : List[str] of version keys in build order

FILE-BACKED DATASET BUILD (REAL USGS PRODUCTS)
----------------------------------------------
Calling uq_build_dataset() in file-backed mode reads, for each version vkey:

  - grid.xml           -> ShakeMap mean fields + lon/lat grid
  - uncertainty.xml    -> ShakeMap uncertainty fields
  - stationlist.json   -> instruments + DYFI-in-stationlist (via USGSParser)
  - rupture.json       -> rupture geometry + metadata (optional)
  - CDI file           -> geocoded DYFI intensity table (optional)

CDI GATING (IMPORTANT)
----------------------
CDI is resolved once from self.dyfi_cdi_input and applied per version only when:

  int(vkey) >= include_cdi_from_version

This gating is recorded in:
  uq_state["versions"][vkey]["obs_audit"]["use_cdi"]
  uq_state["sanity"]["use_cdi"], ["cdi_loaded"], ["n_cdi"], ["cdi_note"], ["cdi_path"]

PER-VERSION PACK STRUCTURE
--------------------------
Each version pack lives at:

  uq_state["versions"][vkey] : Dict[str, Any]

with keys:

  ["meta"]      : metadata parsed from ShakeMap XML
  ["grid"]      : mean fields + native lon/lat grid
  ["uncert"]    : uncertainty fields from uncertainty.xml
  ["stations"]  : observation tables from stationlist.json
  ["cdi"]       : explicit CDI table (if available)
  ["rupture"]   : rupture geometry and metadata
  ["obs_audit"] : counts + boolean flags
  ["updates"]   : results from downstream UQ / assimilation methods

MEAN FIELDS (ShakeMap grids)
----------------------------
uq_state["versions"][vkey]["grid"] contains:

  "lon2d", "lat2d" : 2D ndarrays
  "fields"         : Dict[str, 2D ndarray] (e.g., MMI, PGA, PGV, PSA*)
  "orientation"    : grid alignment metadata

NOTE ON UNITS:
  Mean fields are taken directly from grid.xml and are NOT auto-normalized.
  For example, PGA may be expressed in "%g". This class preserves the original
  convention; downstream methods must respect the implied space.

UNCERTAINTY FIELDS
------------------
uq_state["versions"][vkey]["uncert"]["fields"] contains STD* layers such as:

  "STDMMI"  : MMI sigma (intensity units)
  "STDPGA"  : PGA sigma (ShakeMap convention, often log/ln-based)

SHAKEuq does not convert uncertainty between spaces.
All downstream methods must treat sigma consistently with the chosen workspace.

OBSERVATION TABLES
------------------
Observations are parsed and stored without destructive filtering.

Stationlist sources:
  - instruments          : seismic stations (PGA, PGV, etc.)
  - dyfi_stationlist     : DYFI points embedded in stationlist.json

Explicit CDI source:
  - geocoded CDI table from text file

build_observations(version, imt, dyfi_source, ...) returns a standardized
DataFrame with columns:

  lon, lat, value, sigma,
  source_type, source_detail,
  station_id,
  version, imt, tae_hours

IMT ROUTING / ISOLATION
----------------------
No IMT mixing is allowed.

  - PGA / PGV / seismic IMTs:
      instruments only

  - MMI:
      DYFI stationlist and/or CDI depending on dyfi_source:
        "stationlist", "cdi", "both", "auto"

UNIFIED STACKED GRIDS
--------------------
After per-version parsing, all versions are remapped to a single reference grid:

  uq_state["unified"] contains:
    "lon2d", "lat2d"
    "fields"       : Dict[str, 3D ndarray] (nver, nlat, nlon)
    "sigma"        : Dict[str, 3D ndarray]
    "version_keys" : ordered list of versions
    "ref_version"  : chosen reference grid
    "note"         : remap notes

Remapping uses nearest-neighbor logic (no SciPy dependency).

CURRENTLY IMPLEMENTED UPDATE METHODS
------------------------------------
The following update families are currently implemented and stored under:

  uq_state["versions"][vkey]["updates"][update_key]

1) Raw ShakeMap extraction
   - extract_raw_shakemap(imt)
   - Mean and sigma directly from ShakeMap grids
   - Used as baseline and control

2) Bayesian posterior update
   - Likelihood-based conditioning of ShakeMap mean + sigma
   - Supports different observation combinations
   - Produces posterior mean and uncertainty grids

3) Ordinary Kriging (baseline spatial interpolation)
   - Deterministic interpolation of observations on unified grid
   - Produces interpolated mean and kriging variance

4) Residual / Innovation Kriging
   - Fixed prior (v0 ShakeMap)
   - Observations from target version
   - Residuals: r = obs − prior(x_obs)
   - Kriging of residuals, then:
       posterior_mean = prior_mean + kriged_residual
   - Posterior uncertainty combines prior sigma and kriging variance
   - Supports:
       * PGA (instrument stations)
       * MMI (DYFI stationlist, optional CDI with conditioning)

Audit plotting helpers exist to visualize:
  - prior vs posterior mean
  - prior vs posterior sigma
  - delta fields
  - observation footprints

NOTE ON WORKSPACES
------------------
Some methods (e.g., PGA residual kriging) may operate in linear or log10 space.
SHAKEuq does not enforce a workspace; the chosen method must document and
consistently apply its space and sigma interpretation.

ADDING NEW METHODS (GUIDE FOR FUTURE WORK)
------------------------------------------
When adding new update methods, the following rules MUST be respected:

  - Do not modify uq_state["versions"][vkey]["grid"] or ["uncert"]
  - Store all outputs under uq_state["versions"][vkey]["updates"][key]
  - Always record:
      * prior source (version, method)
      * posterior mean grid
      * posterior uncertainty grid
      * observations used
      * minimal metrics (RMSE/MAE at obs)
  - Never mix IMTs implicitly
  - Never reinterpret sigma without explicitly documenting the space
  - Use build_observations() and extent helpers for input consistency
  - Keep conditioning / weighting logic separate from parsing

SHAKEuq itself remains a staging and auditing layer.
Scientific logic belongs in clearly named update methods that consume
the datasets assembled here.

SYNTHETIC MODE
--------------
Synthetic mode exists to validate pipelines without real USGS products.
Synthetic datasets follow the same uq_state schema and allow controlled
testing of uncertainty propagation and update behavior.

ULTIMATE CONTEXT
----------------
SHAKEuq enables systematic comparison of how mean and uncertainty evolve
through time, data availability, and update philosophy.

It is designed to support rapid-response risk decisions by making
uncertainty evolution explicit, auditable, and reproducible.



      
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


        
    # redacted 
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



    def _build_unified_grids(self) -> None:
        """
        Build a unified (stacked) grid across versions.
    
        Reference grid:
          - Uses the first version that has a valid grid as the reference.
          - All other versions are remapped onto that reference grid (nearest-neighbor) if shapes differ.
    
        Optional grid downsampling (VERY optional, default off):
          - If user sets `self.unified_grid_stride` (int >= 1, or (sy, sx)),
            the reference grid is decimated *after* detection, and all fields are stacked/remapped onto
            the decimated reference grid.
          - Default behavior (no attribute, or stride=1) is unchanged.
    
        Notes:
          - We keep this method conservative to avoid breaking downstream pipelines:
            no schema changes, only adds optional metadata keys in uq_state["unified"].
        """
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
    
        ref_lon2d = np.asarray(ref_lon2d)
        ref_lat2d = np.asarray(ref_lat2d)
    
        # -------------------------------
        # Optional decimation of ref grid
        # -------------------------------
        # User can set after init: self.unified_grid_stride = 2  OR  (2, 3)
        stride = getattr(self, "unified_grid_stride", None)
        sy = sx = 1
        if stride is None:
            sy = sx = 1
        elif isinstance(stride, (int, np.integer)):
            sy = sx = int(stride)
        elif isinstance(stride, (list, tuple)) and len(stride) == 2:
            sy = int(stride[0])
            sx = int(stride[1])
        else:
            # invalid user value -> ignore safely
            sy = sx = 1
    
        # sanitize
        if sy < 1:
            sy = 1
        if sx < 1:
            sx = 1
    
        # Target (unified) grid is either the full ref grid, or a decimated version of it.
        if sy == 1 and sx == 1:
            tgt_lon2d = ref_lon2d
            tgt_lat2d = ref_lat2d
            downsample_note = None
            downsample_stride = (1, 1)
        else:
            tgt_lon2d = ref_lon2d[::sy, ::sx]
            tgt_lat2d = ref_lat2d[::sy, ::sx]
            downsample_stride = (sy, sx)
            downsample_note = f"Reference grid decimated by stride (sy={sy}, sx={sx})."
    
        nlat_tgt, nlon_tgt = tgt_lon2d.shape
    
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
    
        # --- stack mean fields with remap-to-target if needed ---
        for fn in mean_names:
            stack = np.full((len(vkeys), nlat_tgt, nlon_tgt), np.nan, dtype=float)
            for i, vk in enumerate(vkeys):
                g = versions[vk].get("grid")
                if not g or fn not in (g.get("fields") or {}):
                    continue
    
                arr2d = np.asarray(g["fields"][fn], dtype=float)
    
                # Fast path: if this version already matches the *full* ref shape and we are decimating,
                # slice directly to avoid an unnecessary remap.
                if (sy != 1 or sx != 1) and arr2d.shape == ref_lon2d.shape:
                    try:
                        stack[i] = arr2d[::sy, ::sx]
                        continue
                    except Exception:
                        # fall back to remap if slicing fails for any reason
                        pass
    
                # If this version already matches the target shape, use directly.
                if arr2d.shape == (nlat_tgt, nlon_tgt):
                    stack[i] = arr2d
                    continue
    
                # Otherwise remap to target grid.
                src_lon2d = g.get("lon2d")
                src_lat2d = g.get("lat2d")
                if src_lon2d is None or src_lat2d is None:
                    continue
    
                stack[i] = self._nn_remap_to_ref(
                    np.asarray(src_lon2d),
                    np.asarray(src_lat2d),
                    arr2d,
                    tgt_lon2d,
                    tgt_lat2d,
                )
    
            unified_fields[fn] = stack
    
        # --- stack sigma fields with remap-to-target if needed ---
        for sn in sig_keep:
            stack = np.full((len(vkeys), nlat_tgt, nlon_tgt), np.nan, dtype=float)
            for i, vk in enumerate(vkeys):
                u = versions[vk].get("uncert")
                g = versions[vk].get("grid")
                if not u or sn not in (u.get("fields") or {}):
                    continue
    
                arr2d = np.asarray(u["fields"][sn], dtype=float)
    
                # Fast path: if sigma matches full ref shape and we are decimating, slice directly.
                if (sy != 1 or sx != 1) and arr2d.shape == ref_lon2d.shape:
                    try:
                        stack[i] = arr2d[::sy, ::sx]
                        continue
                    except Exception:
                        pass
    
                # If already matches target shape, use directly.
                if arr2d.shape == (nlat_tgt, nlon_tgt):
                    stack[i] = arr2d
                    continue
    
                # Otherwise remap sigma using the version's grid lon/lat
                if not g or g.get("lon2d") is None or g.get("lat2d") is None:
                    continue
    
                stack[i] = self._nn_remap_to_ref(
                    np.asarray(g["lon2d"]),
                    np.asarray(g["lat2d"]),
                    arr2d,
                    tgt_lon2d,
                    tgt_lat2d,
                )
    
            unified_sigma[sn] = stack
    
        note = "Versions remapped to ref grid via nearest-neighbor when shapes differ."
        if downsample_note:
            note = note + " " + downsample_note
    
        self.uq_state["unified"] = {
            "lon2d": tgt_lon2d,
            "lat2d": tgt_lat2d,
            "fields": unified_fields,
            "sigma": unified_sigma,
            "version_keys": vkeys,
            "ref_version": ref_v,
            "ref_shape": (int(ref_lon2d.shape[0]), int(ref_lon2d.shape[1])),
            "shape": (int(nlat_tgt), int(nlon_tgt)),
            "downsample_stride": tuple(downsample_stride),
            "note": note,
        }
    
        if self.verbose:
            mism = []
            for vk in vkeys:
                g = versions[vk].get("grid")
                if g and g.get("lon2d") is not None:
                    sh = np.asarray(g["lon2d"]).shape
                    # Report mismatch relative to the *reference full* shape (same behavior as before),
                    # but also keep it informative if downsampling is active.
                    if sh != tuple(ref_lon2d.shape):
                        mism.append((vk, sh))
            if mism:
                msg = f"[SHAKEuq] unified: remapped {len(mism)} version(s) onto ref grid {ref_v} ref_shape={tuple(ref_lon2d.shape)}"
                if (sy != 1 or sx != 1):
                    msg += f" -> unified_shape={(nlat_tgt, nlon_tgt)} stride={(sy, sx)}"
                msg += "."
                print(msg)
                print("[SHAKEuq] unified mismatches (version, native_shape):", mism[:10], "..." if len(mism) > 10 else "")
    
    




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
    #redacted
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




    def uq_build_dataset(self, unified_grid_stride: Any = None) -> Dict[str, Any]:
        """
        Dispatching uq_build_dataset:
          - if synthetic mode enabled -> in-memory build
          - else -> original file-backed build
    
        Optional:
          unified_grid_stride:
            - None (default): keep current behavior (no decimation unless self.unified_grid_stride is already set)
            - int >= 1: use same stride in (y,x)
            - (sy, sx): tuple/list of two ints >= 1
        """
        # Preserve any existing setting so we never break notebooks/pipelines that set it elsewhere.
        _prev_stride = getattr(self, "unified_grid_stride", None)
        _has_prev = hasattr(self, "unified_grid_stride")
    
        if unified_grid_stride is not None:
            # Accept int or (sy, sx). Invalid values will be handled safely inside _build_unified_grids().
            self.unified_grid_stride = unified_grid_stride
    
        try:
            store = getattr(self, "_synthetic_store", None)
            if store and isinstance(store, dict) and store.get("versions"):
                return self._uq_build_dataset_synthetic()
            return self._uq_build_dataset_filebacked()
        finally:
            # Restore previous state exactly to avoid side effects across runs.
            if unified_grid_stride is not None:
                if _has_prev:
                    self.unified_grid_stride = _prev_stride
                else:
                    try:
                        delattr(self, "unified_grid_stride")
                    except Exception:
                        # ultra-safe fallback; leaving attribute is better than crashing a pipeline
                        self.unified_grid_stride = _prev_stride





    



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

    def _rk_get_prior_version_key(self) -> str:
        """
        Fixed prior selector for Residual Kriging:
        - Prefer uq_state["unified"]["version_keys"][0] when present.
        - Else fall back to lexicographically smallest uq_state["versions"] key.
        """
        u = (self.uq_state or {}).get("unified") or {}
        vks = u.get("version_keys")
        if isinstance(vks, list) and len(vks) > 0:
            return str(vks[0])

        vks2 = sorted(list(((self.uq_state or {}).get("versions") or {}).keys()))
        if not vks2:
            raise RuntimeError("uq_state has no versions; run uq_build_dataset() first.")
        return str(vks2[0])

    def _rk_sample_grid_nn(self, lon2d, lat2d, grid2d, lon_obs, lat_obs):
        """
        Nearest-neighbor sample of a 2D grid at observation lon/lat locations.

        Assumes lon2d/lat2d define a rectilinear grid (as in unified grid stacks).
        """
        import numpy as np

        lon2d = np.asarray(lon2d, dtype=float)
        lat2d = np.asarray(lat2d, dtype=float)
        Z = np.asarray(grid2d, dtype=float)

        lon_obs = np.asarray(lon_obs, dtype=float).ravel()
        lat_obs = np.asarray(lat_obs, dtype=float).ravel()

        if lon2d.ndim != 2 or lat2d.ndim != 2 or Z.ndim != 2:
            raise ValueError("lon2d, lat2d, grid2d must be 2D arrays.")

        # derive 1D axes from the rectilinear grid
        lon1 = lon2d[0, :]
        lat1 = lat2d[:, 0]

        # handle possibly decreasing axes
        lon_inc = bool(lon1[-1] >= lon1[0])
        lat_inc = bool(lat1[-1] >= lat1[0])

        lon_axis = lon1 if lon_inc else lon1[::-1]
        lat_axis = lat1 if lat_inc else lat1[::-1]
        Z_work = Z.copy()
        if not lat_inc:
            Z_work = Z_work[::-1, :]
        if not lon_inc:
            Z_work = Z_work[:, ::-1]

        # nearest indices
        j = np.searchsorted(lon_axis, lon_obs, side="left")
        j = np.clip(j, 0, lon_axis.size - 1)
        j0 = np.clip(j - 1, 0, lon_axis.size - 1)
        choose_left = np.abs(lon_obs - lon_axis[j0]) <= np.abs(lon_obs - lon_axis[j])
        j = np.where(choose_left, j0, j)

        i = np.searchsorted(lat_axis, lat_obs, side="left")
        i = np.clip(i, 0, lat_axis.size - 1)
        i0 = np.clip(i - 1, 0, lat_axis.size - 1)
        choose_up = np.abs(lat_obs - lat_axis[i0]) <= np.abs(lat_obs - lat_axis[i])
        i = np.where(choose_up, i0, i)

        return Z_work[i, j].astype(float)

    def _rk_to_working_space(self, imt_u: str, x_lin, *, pga_space="log10", pga_floor=1e-6):
        """
        Convert values to the working space used for residual kriging.

        - MMI: linear (identity)
        - PGA:
            * "linear": identity
            * "log10": log10(max(x, pga_floor))
        """
        import numpy as np

        imt_u = str(imt_u).upper().strip()
        x = np.asarray(x_lin, dtype=float)

        if imt_u != "PGA":
            return x

        ps = str(pga_space).lower().strip()
        if ps == "linear":
            return x
        if ps == "log10":
            xclip = np.maximum(x, float(pga_floor))
            return np.log10(xclip)

        raise ValueError(f"Invalid pga_space={pga_space!r}; expected 'linear' or 'log10'.")

    def _rk_sigma_to_working_space(self, imt_u: str, mu_lin, sig_lin, *, pga_space="log10", pga_floor=1e-6, sigma_clip_min=1e-6):
        """
        Pragmatic sigma mapping into working space.

        - MMI: sigma unchanged
        - PGA:
            * "linear": sigma unchanged
            * "log10": delta-method: sigma_log10 ≈ sigma_lin / (ln(10) * max(mu_lin, pga_floor))
        """
        import numpy as np

        imt_u = str(imt_u).upper().strip()
        mu = np.asarray(mu_lin, dtype=float)
        sig = np.asarray(sig_lin, dtype=float)

        if imt_u != "PGA":
            return np.maximum(sig, float(sigma_clip_min))

        ps = str(pga_space).lower().strip()
        if ps == "linear":
            return np.maximum(sig, float(sigma_clip_min))

        if ps == "log10":
            denom = np.log(10.0) * np.maximum(mu, float(pga_floor))
            out = sig / np.maximum(denom, 1e-12)
            return np.maximum(out, float(sigma_clip_min))

        raise ValueError(f"Invalid pga_space={pga_space!r}; expected 'linear' or 'log10'.")

    def _rk_from_working_space(self, imt_u: str, x_work):
        """
        Convert working-space values back to native space (for grids / reporting).

        Note: for PGA we only invert if we *know* we used log10; in this RK workflow
        we keep track via meta["pga_space"] and only call this when pga_space=="log10".
        """
        import numpy as np

        imt_u = str(imt_u).upper().strip()
        x = np.asarray(x_work, dtype=float)

        if imt_u != "PGA":
            return x
        # inversion handled explicitly by caller depending on pga_space
        return x

    def _rk_make_residual_obs_df(
        self,
        *,
        obs_df,
        imt_u: str,
        prior_at_obs_lin,
        pga_space="log10",
        pga_floor=1e-6,
        sigma_clip_min=1e-6,
    ):
        """
        Build the exact obs DataFrame expected by _ok_krige_grid:
          columns: lon, lat, value [, sigma]

        Here, 'value' is the residual in working space.
        Sigma (if present in obs_df) is mapped into working space consistently.
        """
        import numpy as np
        import pandas as pd

        if not isinstance(obs_df, pd.DataFrame) or obs_df.empty:
            return pd.DataFrame(columns=["lon", "lat", "value", "sigma"])

        if not {"lon", "lat", "value"}.issubset(obs_df.columns):
            raise RuntimeError("Observations must include columns: lon, lat, value.")

        imt_u = str(imt_u).upper().strip()

        y_lin = pd.to_numeric(obs_df["value"], errors="coerce").to_numpy(dtype=float)
        mu0_lin = np.asarray(prior_at_obs_lin, dtype=float)

        # residual in working space
        y_work = self._rk_to_working_space(imt_u, y_lin, pga_space=pga_space, pga_floor=pga_floor)
        mu0_work = self._rk_to_working_space(imt_u, mu0_lin, pga_space=pga_space, pga_floor=pga_floor)
        r_work = y_work - mu0_work

        out = pd.DataFrame(
            {
                "lon": pd.to_numeric(obs_df["lon"], errors="coerce").to_numpy(dtype=float),
                "lat": pd.to_numeric(obs_df["lat"], errors="coerce").to_numpy(dtype=float),
                "value": r_work.astype(float),
            }
        )

        if "sigma" in obs_df.columns:
            sig_lin = pd.to_numeric(obs_df["sigma"], errors="coerce").to_numpy(dtype=float)
            # map obs sigma using obs mean as the linearization point
            sig_work = self._rk_sigma_to_working_space(
                imt_u, y_lin, sig_lin,
                pga_space=pga_space, pga_floor=pga_floor, sigma_clip_min=sigma_clip_min
            )
            out["sigma"] = sig_work.astype(float)

        # drop invalid coords / residuals
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=["lon", "lat", "value"]).reset_index(drop=True)
        return out

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
                post_mean = prior_mean + r_kriged   (in chosen working space)
          - Sigma (pragmatic, consistent):
                post_sigma = sqrt(prior_sigma^2 + var_r_kriged)
            where both prior_sigma and var_r_kriged are expressed in the SAME working space.
            If pga_space=="log10", we convert posterior mean back to linear (10**),
            and map posterior sigma back via delta method:
                sigma_lin ≈ ln(10) * mu_lin * sigma_log10
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

        # ensure storage container
        if "updates" not in vpack or not isinstance(vpack.get("updates"), dict):
            vpack["updates"] = {}

        # fixed prior version (v0)
        v0 = self._rk_get_prior_version_key()

        # choose key
        mode_s = str(mode).lower().strip()
        if key is None:
            key = f"rk__{mode_s}__v{vkey}"
        key = str(key)

        t0 = time.time()

        lon2d, lat2d = self._get_unified_grid()

        # ---- select IMT + dyfi routing (IMT isolation) ----
        # PGA updated ONLY with instruments (stationlist seismic)
        # MMI updated with DYFI and optionally CDI
        if mode_s.startswith("pga"):
            imt_target = "PGA"
            dyfi_source_eff = "stationlist"
            include_cdi = False
        else:
            imt_target = "MMI"
            dyfi_source_eff = str(dyfi_source).lower().strip()
            include_cdi = bool(use_cdi)

        # ---- load fixed prior mean/sigma on unified grid ----
        _, _, prior_mean_lin = self._get_prior_mean_unified(v0, imt_target)
        _, _, prior_sigma_lin = self._get_prior_sigma_unified(v0, imt_target)

        if prior_mean_lin is None or prior_sigma_lin is None:
            raise RuntimeError(f"Missing prior mean/sigma for v0={v0} imt={imt_target}. Check unified stacks.")

        prior_mean_lin = np.asarray(prior_mean_lin, dtype=float)
        prior_sigma_lin = np.asarray(prior_sigma_lin, dtype=float)

        if prior_mean_lin.shape != lon2d.shape or prior_sigma_lin.shape != lon2d.shape:
            raise RuntimeError(f"Prior shapes do not match unified grid for imt={imt_target} (v0={v0}).")

        # ---- observations from TARGET version (vkey) ----
        obs = self.build_observations(version=vkey, imt=imt_target, dyfi_source=dyfi_source_eff)

        # validate minimal columns
        if not isinstance(obs, pd.DataFrame) or obs.empty:
            out = {
                "ok": False,
                "note": f"No observations returned by build_observations (v={vkey}, imt={imt_target}).",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key},
            }
            vpack["updates"][key] = out
            return out

        for c in ("lon", "lat", "value"):
            if c not in obs.columns:
                raise RuntimeError(f"build_observations did not return required column: {c}")

        # numeric coercion
        obs = obs.copy()
        obs["lon"] = pd.to_numeric(obs["lon"], errors="coerce")
        obs["lat"] = pd.to_numeric(obs["lat"], errors="coerce")
        obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
        if "sigma" in obs.columns:
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
        # extent filter (project signature)
        # ------------------------------------------------------------------
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

        if obs_f is None or getattr(obs_f, "empty", False) or len(obs_f) == 0:
            out = {
                "ok": False,
                "note": "All observations fell outside unified extent after margin filter.",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key, "extent": meta_ext},
            }
            vpack["updates"][key] = out
            return out

        # ------------------------------------------------------------------
        # optional CDI conditioning (compare against v0 prior, not target map)
        # only applies when MMI and caller requested CDI inclusion
        # ------------------------------------------------------------------
        cdi_meta = None
        if imt_target == "MMI" and include_cdi:
            try:
                obs_f, cdi_meta = self._cdi_condition(
                    version=v0,  # compare/condition vs v0 prior (FIXED)
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

        if obs_f is None or getattr(obs_f, "empty", False) or len(obs_f) == 0:
            out = {
                "ok": False,
                "note": "No observations remain after optional CDI conditioning.",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key, "extent": meta_ext, "cdi_meta": cdi_meta},
            }
            vpack["updates"][key] = out
            return out

        # ------------------------------------------------------------------
        # Sample FIXED prior (v0) at obs locations for residual construction
        # ------------------------------------------------------------------
        prior_at_obs_lin = self._rk_sample_grid_nn(
            lon2d, lat2d, prior_mean_lin,
            obs_f["lon"].to_numpy(dtype=float),
            obs_f["lat"].to_numpy(dtype=float),
        )

        # build residual obs_df in EXACT format expected by _ok_krige_grid
        resid_obs = self._rk_make_residual_obs_df(
            obs_df=obs_f,
            imt_u=imt_target,
            prior_at_obs_lin=prior_at_obs_lin,
            pga_space=pga_space,
            pga_floor=pga_floor,
            sigma_clip_min=sigma_clip_min,
        )

        if resid_obs is None or getattr(resid_obs, "empty", False) or len(resid_obs) == 0:
            out = {
                "ok": False,
                "note": "Residual observations invalid/empty after construction.",
                "meta": {"version": vkey, "prior_version": v0, "imt": imt_target, "mode": mode_s, "key": key, "extent": meta_ext},
            }
            vpack["updates"][key] = out
            return out

        # ------------------------------------------------------------------
        # Krige residuals on unified grid using YOUR _ok_krige_grid signature
        # ------------------------------------------------------------------
        zhat_resid, var_resid = self._ok_krige_grid(
            resid_obs,
            lon2d, lat2d,
            neighbor_k=int(neighbor_k) if neighbor_k is not None else 25,
            max_points=max_points,
            use_obs_sigma=bool(use_obs_sigma),
            variogram_model=str(variogram_model),
            range_km=float(range_km),
            sill=float(sill),
            nugget=float(nugget),
            ridge=float(ridge),
            debug=bool(debug),
        )

        zhat_resid = np.asarray(zhat_resid, dtype=float)
        var_resid = np.asarray(var_resid, dtype=float)

        # ------------------------------------------------------------------
        # Update mean/sigma in working space, then map back if needed
        # ------------------------------------------------------------------
        prior_mean_work = self._rk_to_working_space(imt_target, prior_mean_lin, pga_space=pga_space, pga_floor=pga_floor)
        prior_sig_work = self._rk_sigma_to_working_space(
            imt_target, prior_mean_lin, prior_sigma_lin,
            pga_space=pga_space, pga_floor=pga_floor, sigma_clip_min=sigma_clip_min
        )

        post_mean_work = prior_mean_work + zhat_resid
        post_sig_work = np.sqrt(np.maximum(0.0, prior_sig_work ** 2 + np.maximum(var_resid, 0.0)))
        post_sig_work = np.maximum(post_sig_work, float(sigma_clip_min))

        # default outputs in native space
        post_mean = post_mean_work.copy()
        post_sigma = post_sig_work.copy()
        prior_mean = prior_mean_lin.copy()
        prior_sigma = prior_sigma_lin.copy()

        if str(imt_target).upper() == "PGA" and str(pga_space).lower().strip() == "log10":
            # invert mean: linear PGA = 10**(log10-mean)
            post_mean = np.power(10.0, post_mean_work)

            # map sigma back with delta method: sigma_lin ≈ ln(10) * mu_lin * sigma_log10
            post_sigma = (np.log(10.0) * np.maximum(post_mean, float(pga_floor)) * post_sig_work)

            # keep prior in native for storage; residual fields remain in working space
            post_sigma = np.maximum(post_sigma, float(sigma_clip_min))

        # ------------------------------------------------------------------
        # Metrics at observation points (native observation space)
        # ------------------------------------------------------------------
        # sample posterior at obs locations (using same NN sampler)
        post_at_obs = self._rk_sample_grid_nn(
            lon2d, lat2d, post_mean,
            obs_f["lon"].to_numpy(dtype=float),
            obs_f["lat"].to_numpy(dtype=float),
        )

        # prior prediction at obs in native space
        prior_pred = np.asarray(prior_at_obs_lin, dtype=float)
        post_pred = np.asarray(post_at_obs, dtype=float)
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
                    "neighbor_k": int(neighbor_k) if neighbor_k is not None else None,
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
            "residual": {
                "resid_grid": zhat_resid,  # working space
                "resid_var": var_resid,    # working space variance
            },
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





    def _sigma_field_for_imt(self, imt):
        """
        Small helper used by audit plots for labeling sigma panels.
        We keep this intentionally simple and stable (no schema assumptions).

        Notes:
        - For PGA, sigma stored by residual-kriging is in *native PGA space* (linear),
          even if the residual kriging was done in log10 space (the method converts back).
        - For MMI, sigma is in MMI units.
        """
        imt_u = str(imt).upper().strip()
        if imt_u == "PGA":
            return "σ(PGA)"
        if imt_u == "MMI":
            return "σ(MMI)"
        return f"σ({imt_u})"

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
        mean_scale_type="usgs",
        mean_pga_units="%g",
        sigma_cmap="viridis",
        sigma_vmin=None,
        sigma_vmax=None,
    ):
        """
        Audit plot for residual kriging result stored in:
            uq_state["versions"][vkey]["updates"][key]

        Panels (2×3):
          - Prior mean (v0 fixed)
          - Posterior mean (+ obs)
          - Δ mean (post - prior)      [optional]
          - Prior sigma
          - Posterior sigma (+ obs)
          - Δ sigma (post - prior)     [optional]
        """
        import numpy as np
        import pandas as pd
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
        imt_u = str(meta.get("imt", "?")).upper().strip()
        v0 = str(meta.get("prior_version", "?"))

        prior_mean = np.asarray((upd.get("prior") or {}).get("mean_grid"), dtype=float)
        prior_sig = np.asarray((upd.get("prior") or {}).get("sigma_grid"), dtype=float)
        post_mean = np.asarray((upd.get("posterior") or {}).get("mean_grid"), dtype=float)
        post_sig = np.asarray((upd.get("posterior") or {}).get("sigma_grid"), dtype=float)
        obs = upd.get("obs_used")

        # basic shape checks (fail early)
        for name, arr in [
            ("prior_mean", prior_mean),
            ("prior_sig", prior_sig),
            ("post_mean", post_mean),
            ("post_sig", post_sig),
        ]:
            if arr.shape != lon2d.shape:
                raise RuntimeError(f"{name} shape {arr.shape} != unified grid shape {lon2d.shape}")

        # diffs
        dmean = post_mean - prior_mean
        dsig = post_sig - prior_sig

        # mean colormap (discrete for MMI/PGA, aligned with your other audit plots)
        cmap_mean, norm_mean, ticks_mean, label_mean = self._mean_cmap_for_imt(
            imt_u, scale_type=mean_scale_type, pga_units=mean_pga_units
        )

        rmse_prior = None
        rmse_post = None
        try:
            rmse_prior = float((upd.get("metrics") or {}).get("rmse_prior"))
            rmse_post = float((upd.get("metrics") or {}).get("rmse_post"))
        except Exception:
            pass

        title_line = (
            f"Residual Kriging audit — v={vkey} key={key}\n"
            f"prior=v0({v0}) imt={imt_u}"
        )
        if rmse_prior is not None and rmse_post is not None:
            title_line += f"  rmse_prior={rmse_prior:.4f}  rmse_post={rmse_post:.4f}"

        fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
        ax = axes.ravel()
        fig.suptitle(title_line, fontsize=12)

        # (1) Prior mean
        self._plot_grid_panel(
            ax[0], lon2d, lat2d, prior_mean,
            title=f"Prior mean (v0 fixed) — {imt_u}",
            cmap=cmap_mean, norm=norm_mean,
            add_colorbar=True, cbar_ticks=ticks_mean, cbar_label=label_mean,
            fig=fig,
        )

        # (2) Posterior mean (+ obs)
        self._plot_grid_panel(
            ax[1], lon2d, lat2d, post_mean,
            title=f"Posterior mean (residual OK) — {imt_u} (+ obs)",
            cmap=cmap_mean, norm=norm_mean,
            add_colorbar=True, cbar_ticks=ticks_mean, cbar_label=label_mean,
            fig=fig,
        )
        if isinstance(obs, pd.DataFrame) and (not obs.empty) and {"lon", "lat"}.issubset(obs.columns):
            ax[1].scatter(obs["lon"].values, obs["lat"].values, s=6, c="k", alpha=0.55, linewidths=0)

        # (3) Δ mean
        if show_diffs:
            self._plot_grid_panel(
                ax[2], lon2d, lat2d, dmean,
                title="Δ mean (post − prior)",
                cmap="coolwarm", norm=None,
                add_colorbar=True, cbar_ticks=None, cbar_label="Δ mean",
                fig=fig,
            )
        else:
            ax[2].axis("off")

        # (4) Prior sigma
        self._plot_grid_panel(
            ax[3], lon2d, lat2d, prior_sig,
            title=f"Prior sigma ({self._sigma_field_for_imt(imt_u)})",
            cmap=sigma_cmap, norm=None, vmin=sigma_vmin, vmax=sigma_vmax,
            add_colorbar=True, cbar_ticks=None, cbar_label="sigma",
            fig=fig,
        )

        # (5) Posterior sigma (+ obs)
        self._plot_grid_panel(
            ax[4], lon2d, lat2d, post_sig,
            title="Posterior sigma (+ obs)",
            cmap=sigma_cmap, norm=None, vmin=sigma_vmin, vmax=sigma_vmax,
            add_colorbar=True, cbar_ticks=None, cbar_label="sigma",
            fig=fig,
        )
        if isinstance(obs, pd.DataFrame) and (not obs.empty) and {"lon", "lat"}.issubset(obs.columns):
            ax[4].scatter(obs["lon"].values, obs["lat"].values, s=6, c="k", alpha=0.35, linewidths=0)

        # (6) Δ sigma
        if show_diffs:
            self._plot_grid_panel(
                ax[5], lon2d, lat2d, dsig,
                title="Δ sigma (post − prior)",
                cmap="coolwarm", norm=None,
                add_colorbar=True, cbar_ticks=None, cbar_label="Δ sigma",
                fig=fig,
            )
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
# CDI TOOLKIT (CLEANING + WEIGHTING) + FILTER TEST HARNESS
# + Innovation Kriging wrapper for MMI using FILTERED CDI (fixed prior v0)
# ----------------------------------------------------------------------
# Design goals:
#   - Do NOT change build_observations() behavior (keep minimal filtering).
#   - Keep CDI filtering/weighting separate from kriging for auditability.
#   - Provide a scientific tester:
#         "Does filtered CDI agree better with a later reference shakemap?"
#   - Then feed the chosen filter profile into innovation kriging (MMI only).
# ======================================================================

    def _cdi_profile_to_kwargs_cdi_pick_nresp_column(self, profile: dict, obs_df=None) -> dict:
        """
        Convert a user 'profile' dict into kwargs compatible with _cdi_condition(),
        AND (optionally) auto-pick a responses column name for weighting.

        This is intentionally defensive:
          - Accepts native names (trust_nresp_ge, enable_local_outlier, ...)
          - Accepts legacy aliases (cdi_trust_nresp_ge, cdi_enable_local_outlier, ...)
          - Returns ONLY keys that _cdi_condition() in THIS file is expected to accept
            (based on how run_residual_kriging_update() calls _cdi_condition()).

        Returns:
          {
            "cdi_kwargs": {... safe kwargs for _cdi_condition ...},
            "nresp_col": "..." or None,
            "profile_used": {.. normalized defaults applied ..}
          }
        """
        if profile is None:
            profile = {}
        if not isinstance(profile, dict):
            raise TypeError("profile must be a dict or None")

        # Native -> alias list (first found wins)
        def pick(*names, default=None):
            for n in names:
                if n in profile:
                    return profile[n]
            return default

        # ---- normalized profile (with defaults) ----
        prof_used = {}

        # core gating / trust
        prof_used["trust_nresp_ge"] = int(pick("trust_nresp_ge", "cdi_trust_nresp_ge", default=3))

        # local outlier filter
        prof_used["enable_local_outlier"] = bool(pick("enable_local_outlier", "cdi_enable_local_outlier", default=True))
        prof_used["local_radius_km"] = float(pick("local_radius_km", "cdi_local_radius_km", default=25.0))
        prof_used["outlier_k"] = float(pick("outlier_k", "cdi_outlier_k", default=2.5))
        prof_used["min_neighbors"] = int(pick("min_neighbors", "cdi_min_neighbors", default=4))

        # clustering
        prof_used["enable_clustering"] = bool(pick("enable_clustering", "cdi_enable_clustering", default=True))
        prof_used["cluster_eps_km"] = float(pick("cluster_eps_km", "cdi_cluster_eps_km", default=2.0))
        prof_used["cluster_min_pts"] = int(pick("cluster_min_pts", "cdi_cluster_min_pts", default=3))

        # strategy chain
        prof_used["cdi_strategy"] = pick(
            "cdi_strategy",
            "strategy",
            default=("local_outlier", "grid_thin", "quantile_residual"),
        )
        prof_used["cdi_grid_bin_km"] = float(pick("cdi_grid_bin_km", "grid_bin_km", default=10.0))
        prof_used["cdi_grid_agg"] = str(pick("cdi_grid_agg", "grid_agg", default="median"))
        prof_used["cdi_quantile"] = pick("cdi_quantile", "quantile", default=(0.05, 0.95))

        # Optional: user can add a *pre-binning* stage before _cdi_condition
        # (kept OUTSIDE _cdi_condition so we don't touch its API).
        prof_used["enable_prebin_grid"] = bool(pick("enable_prebin_grid", default=False))
        prof_used["prebin_km"] = float(pick("prebin_km", default=20.0))
        prof_used["prebin_agg"] = str(pick("prebin_agg", default="median"))

        # Optional: radial binning (rings) around epicenter (also outside _cdi_condition).
        prof_used["enable_radial_bin"] = bool(pick("enable_radial_bin", default=False))
        prof_used["radial_bin_km"] = float(pick("radial_bin_km", default=25.0))
        prof_used["radial_agg"] = str(pick("radial_agg", default="median"))

        # ---- safe kwargs for _cdi_condition (match call style used elsewhere in THIS file) ----
        cdi_kwargs = {
            "cdi_trust_nresp_ge": int(prof_used["trust_nresp_ge"]),
            "cdi_enable_local_outlier": bool(prof_used["enable_local_outlier"]),
            "cdi_local_radius_km": float(prof_used["local_radius_km"]),
            "cdi_outlier_k": float(prof_used["outlier_k"]),
            "cdi_min_neighbors": int(prof_used["min_neighbors"]),
            "cdi_enable_clustering": bool(prof_used["enable_clustering"]),
            "cdi_cluster_eps_km": float(prof_used["cluster_eps_km"]),
            "cdi_cluster_min_pts": int(prof_used["cluster_min_pts"]),
            "cdi_strategy": prof_used["cdi_strategy"],
            "cdi_grid_bin_km": float(prof_used["cdi_grid_bin_km"]),
            "cdi_grid_agg": str(prof_used["cdi_grid_agg"]),
            "cdi_quantile": prof_used["cdi_quantile"],
        }

        nresp_col = None
        if obs_df is not None:
            nresp_col = self._cdi_pick_nresp_column(obs_df)

        return {"cdi_kwargs": cdi_kwargs, "nresp_col": nresp_col, "profile_used": prof_used}

    # Backward-compatible alias (older notebooks might call this)
    def _cdi_profile_to_kwargs(self, profile: dict) -> dict:
        """Compatibility wrapper: returns ONLY cdi_kwargs."""
        out = self._cdi_profile_to_kwargs_cdi_pick_nresp_column(profile=profile, obs_df=None)
        return out["cdi_kwargs"]

    def _cdi_pick_nresp_column(self, df):
        """
        Best-effort pick of an n-responses column (for weighting), without assuming schema.
        Returns column name or None.
        """
        if df is None or not hasattr(df, "columns"):
            return None

        candidates = [
            "nresp",
            "nResp",
            "n_responses",
            "num_responses",
            "No. of responses",
            "No. of responses ",
            "No. of Responses",
            "responses",
            "nresponses",
        ]
        for c in candidates:
            if c in df.columns:
                return c

        # fallback: any column containing 'resp'
        for c in df.columns:
            try:
                if isinstance(c, str) and ("resp" in c.lower()):
                    return c
            except Exception:
                continue
        return None

    def _cdi_grid_prebin(self, obs_df, *, bin_km=20.0, agg="median"):
        """
        Optional pre-binning in approximate km grid cells to reduce dense CDI clusters.
        This happens BEFORE _cdi_condition() (so it's outside that API).

        obs_df must include: lon, lat, value, sigma (and may include others).
        Returns a new DataFrame with same standardized columns retained.
        """
        import numpy as np
        import pandas as pd

        if obs_df is None or getattr(obs_df, "empty", False):
            return obs_df

        df = obs_df.copy()

        # crude km/deg conversion near mid-lat; good enough for binning
        lat0 = float(np.nanmedian(df["lat"].to_numpy(dtype=float)))
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * max(np.cos(np.deg2rad(lat0)), 1e-6)

        dx = float(bin_km) / km_per_deg_lon
        dy = float(bin_km) / km_per_deg_lat
        if dx <= 0 or dy <= 0:
            return df

        gx = np.floor(df["lon"].to_numpy(dtype=float) / dx).astype(int)
        gy = np.floor(df["lat"].to_numpy(dtype=float) / dy).astype(int)
        df["_gx"] = gx
        df["_gy"] = gy

        # choose aggregator
        agg = str(agg).lower().strip()
        if agg not in ("median", "mean"):
            agg = "median"

        grp = df.groupby(["_gx", "_gy"], dropna=False)

        def _agg_series(s):
            s = pd.to_numeric(s, errors="coerce")
            if agg == "mean":
                return float(np.nanmean(s.to_numpy(dtype=float)))
            return float(np.nanmedian(s.to_numpy(dtype=float)))

        out = grp.agg(
            lon=("lon", _agg_series),
            lat=("lat", _agg_series),
            value=("value", _agg_series),
            sigma=("sigma", _agg_series),
        ).reset_index(drop=True)

        # preserve required standardized metadata columns if present
        keep_meta_cols = ["source_type", "source_detail", "station_id", "version", "imt", "tae_hours"]
        for c in keep_meta_cols:
            if c in df.columns and c not in out.columns:
                # take the first representative
                out[c] = grp[c].first().reset_index(drop=True)

        return out

    def _cdi_radial_bin(self, obs_df, *, version, bin_km=25.0, agg="median"):
        """
        Optional radial binning (rings) around epicenter: aggregates CDI by distance bands.
        This is a *research tool* for robustness testing (outside _cdi_condition()).

        Requires event lon/lat from the version meta if available.
        If not available, returns input unchanged.
        """
        import numpy as np
        import pandas as pd

        if obs_df is None or getattr(obs_df, "empty", False):
            return obs_df

        vkey = _norm_version(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey) or {}
        meta = vpack.get("meta") or {}

        try:
            evlon = float(meta.get("lon"))
            evlat = float(meta.get("lat"))
        except Exception:
            return obs_df

        df = obs_df.copy()

        # haversine distance (km)
        lon = df["lon"].to_numpy(dtype=float)
        lat = df["lat"].to_numpy(dtype=float)

        R = 6371.0
        phi1 = np.deg2rad(evlat)
        phi2 = np.deg2rad(lat)
        dphi = phi2 - phi1
        dl = np.deg2rad(lon - evlon)
        a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(dl / 2.0) ** 2)
        d_km = 2.0 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))

        b = float(bin_km)
        if not np.isfinite(b) or b <= 0:
            return df

        ring = np.floor(d_km / b).astype(int)
        df["_ring"] = ring

        agg = str(agg).lower().strip()
        if agg not in ("median", "mean"):
            agg = "median"

        def _agg_series(s):
            s = pd.to_numeric(s, errors="coerce")
            if agg == "mean":
                return float(np.nanmean(s.to_numpy(dtype=float)))
            return float(np.nanmedian(s.to_numpy(dtype=float)))

        grp = df.groupby(["_ring"], dropna=False)
        out = grp.agg(
            lon=("lon", _agg_series),
            lat=("lat", _agg_series),
            value=("value", _agg_series),
            sigma=("sigma", _agg_series),
        ).reset_index(drop=True)

        keep_meta_cols = ["source_type", "source_detail", "station_id", "version", "imt", "tae_hours"]
        for c in keep_meta_cols:
            if c in df.columns and c not in out.columns:
                out[c] = grp[c].first().reset_index(drop=True)

        return out

    def cdi_prepare_filtered(
        self,
        *,
        version,
        margin_deg=0.05,
        profile=None,
        apply_conditioning=True,
        apply_weighting=False,
        weight_mode="nresp_sqrt",   # "nresp", "nresp_sqrt", "nresp_log", "none"
        weight_clip=(1.0, 50.0),
        debug=False,
    ):
        """
        Prepare CDI-only MMI observations for a target version, with:
          1) build_observations(..., imt="MMI", dyfi_source="cdi")
          2) filter_observations_to_extent(..., grid_mode="unified")
          3) OPTIONAL pre-binning (grid and/or radial) from profile (outside _cdi_condition)
          4) optional _cdi_condition(...) using FIXED prior version v0 (via version=prior_v0)
          5) optional weighting -> sigma adjustment (so kriging can use use_obs_sigma=True)

        Conditioning (step 4) MUST compare against v0 prior, not the target map.

        Returns dict:
          ok, obs_raw, obs_extent, obs_filtered, meta
        """
        import numpy as np
        import pandas as pd

        if not isinstance(self.uq_state, dict) or not (self.uq_state.get("versions") or {}):
            raise RuntimeError("uq_state not initialized. Run uq_build_dataset() first.")

        vkey = _norm_version(version)
        prior_v0 = self._rk_get_prior_version_key()

        # --- build CDI obs (CDI-only) ---
        obs_raw = self.build_observations(vkey, imt="MMI", dyfi_source="cdi", sigma_override=None)
        if obs_raw is None or getattr(obs_raw, "empty", False) or len(obs_raw) == 0:
            return {
                "ok": False,
                "note": "No CDI observations returned by build_observations(dyfi_source='cdi').",
                "meta": {"version": vkey, "prior_version": prior_v0},
                "obs_raw": obs_raw,
                "obs_extent": None,
                "obs_filtered": None,
            }

        # --- extent filter (strict for gridded ops) ---
        obs_extent, obs_drop = self.filter_observations_to_extent(
            obs_raw,
            version=vkey,
            grid_mode="unified",
            margin_deg=float(margin_deg),
            return_dropped=True,
        )
        if obs_extent is None or getattr(obs_extent, "empty", False) or len(obs_extent) == 0:
            return {
                "ok": False,
                "note": "All CDI observations outside extent after filter_observations_to_extent().",
                "meta": {
                    "version": vkey,
                    "prior_version": prior_v0,
                    "margin_deg": float(margin_deg),
                    "n_raw": int(len(obs_raw)),
                    "n_kept": 0,
                    "n_dropped": int(len(obs_drop)) if hasattr(obs_drop, "__len__") else None,
                },
                "obs_raw": obs_raw,
                "obs_extent": obs_extent,
                "obs_filtered": None,
            }

        obs_f = obs_extent.copy()

        # --- profile mapping + optional pre-binning knobs ---
        mapped = self._cdi_profile_to_kwargs_cdi_pick_nresp_column(profile=(profile or {}), obs_df=obs_f)
        prof_used = mapped["profile_used"]
        cdi_kwargs = mapped["cdi_kwargs"]

        prebin_meta = {"applied": False, "n_before": int(len(obs_f)), "n_after": int(len(obs_f))}
        if prof_used.get("enable_prebin_grid", False):
            n0 = int(len(obs_f))
            obs_f = self._cdi_grid_prebin(obs_f, bin_km=float(prof_used.get("prebin_km", 20.0)), agg=str(prof_used.get("prebin_agg", "median")))
            prebin_meta = {
                "applied": True,
                "type": "grid",
                "bin_km": float(prof_used.get("prebin_km", 20.0)),
                "agg": str(prof_used.get("prebin_agg", "median")),
                "n_before": n0,
                "n_after": int(len(obs_f)) if obs_f is not None else 0,
            }

        radial_meta = {"applied": False, "n_before": int(len(obs_f)), "n_after": int(len(obs_f))}
        if prof_used.get("enable_radial_bin", False) and obs_f is not None and not obs_f.empty:
            n0 = int(len(obs_f))
            obs_f = self._cdi_radial_bin(
                obs_f,
                version=vkey,
                bin_km=float(prof_used.get("radial_bin_km", 25.0)),
                agg=str(prof_used.get("radial_agg", "median")),
            )
            radial_meta = {
                "applied": True,
                "type": "radial",
                "bin_km": float(prof_used.get("radial_bin_km", 25.0)),
                "agg": str(prof_used.get("radial_agg", "median")),
                "n_before": n0,
                "n_after": int(len(obs_f)) if obs_f is not None else 0,
            }

        # --- optional conditioning: compare vs FIXED prior v0 ---
        cdi_meta = None
        if apply_conditioning and obs_f is not None and not obs_f.empty:
            try:
                obs_f, cdi_meta = self._cdi_condition(
                    version=prior_v0,          # compare residuals against FIXED v0 prior
                    obs_df=obs_f,
                    margin_deg=float(margin_deg),
                    debug=bool(debug),
                    **cdi_kwargs,
                )
            except Exception as e:
                cdi_meta = {"ok": False, "err": str(e)}

        if obs_f is None or getattr(obs_f, "empty", False) or len(obs_f) == 0:
            return {
                "ok": False,
                "note": "No CDI observations remain after conditioning.",
                "meta": {
                    "version": vkey,
                    "prior_version": prior_v0,
                    "margin_deg": float(margin_deg),
                    "n_raw": int(len(obs_raw)),
                    "n_extent": int(len(obs_extent)),
                    "n_filtered": 0,
                    "profile_used": prof_used,
                    "prebin": prebin_meta,
                    "radial_bin": radial_meta,
                    "cdi_meta": cdi_meta,
                },
                "obs_raw": obs_raw,
                "obs_extent": obs_extent,
                "obs_filtered": obs_f,
            }

        # --- optional weighting: convert weight -> sigma_eff (so kriging can respect it) ---
        weight_meta = {"applied": False}
        if apply_weighting and ("sigma" in obs_f.columns):
            nresp_col = self._cdi_pick_nresp_column(obs_f)
            w = np.ones(len(obs_f), dtype=float)

            wm = str(weight_mode).lower().strip()
            if wm in ("nresp", "nresp_sqrt", "nresp_log") and nresp_col is not None:
                vals = pd.to_numeric(obs_f[nresp_col], errors="coerce").to_numpy(dtype=float)
                vals = np.where(np.isfinite(vals), vals, 1.0)
                vals = np.maximum(vals, 1.0)
                if wm == "nresp_sqrt":
                    vals = np.sqrt(vals)
                elif wm == "nresp_log":
                    vals = np.log10(vals + 1.0)
                    vals = np.maximum(vals, 1e-6)
                w = vals

            wmin, wmax = float(weight_clip[0]), float(weight_clip[1])
            w = np.clip(w, wmin, wmax)

            sigma = pd.to_numeric(obs_f["sigma"], errors="coerce").to_numpy(dtype=float)
            sigma = np.where(np.isfinite(sigma), sigma, np.nan)

            sigma_eff = sigma / np.sqrt(w)

            obs_f = obs_f.copy()
            obs_f["weight"] = w
            obs_f["sigma_raw"] = obs_f["sigma"]
            obs_f["sigma"] = sigma_eff  # overwrite so _rk_ok_krige_grid uses it

            weight_meta = {
                "applied": True,
                "mode": wm,
                "nresp_col": nresp_col,
                "clip": (wmin, wmax),
                "w_min": float(np.nanmin(w)) if len(w) else None,
                "w_max": float(np.nanmax(w)) if len(w) else None,
            }

        meta = {
            "version": vkey,
            "prior_version": prior_v0,
            "margin_deg": float(margin_deg),
            "n_raw": int(len(obs_raw)),
            "n_extent": int(len(obs_extent)),
            "n_filtered": int(len(obs_f)),
            "profile_used": prof_used,
            "prebin": prebin_meta,
            "radial_bin": radial_meta,
            "conditioning": {"applied": bool(apply_conditioning), "cdi_meta": cdi_meta},
            "weighting": weight_meta,
        }

        return {
            "ok": True,
            "obs_raw": obs_raw,
            "obs_extent": obs_extent,
            "obs_filtered": obs_f,
            "meta": meta,
        }

    def cdi_filtering_diagnostic(
        self,
        *,
        version,
        reference_version,
        profiles,
        margin_deg=0.05,
        apply_conditioning=True,
        apply_weighting=False,
        weight_mode="nresp_sqrt",
        weight_clip=(1.0, 50.0),
        save_csv_path=None,
        make_plots=False,
        plot_out_dir=None,
        dpi=160,
        debug=False,
    ):
        """
        Scientific tester:
          Q1: "Does filtered CDI agree better with a later reference shakemap?"
              (where reference_version is typically the most data-rich/latest version)

        For each profile in `profiles`:
          - build CDI obs (target version)
          - extent filter
          - optional pre-binning (profile)
          - optional conditioning vs v0 prior (NOT vs reference)
          - compute misfit of filtered CDI against reference shakemap mean at those points

        Returns:
          {
            "ok": True/False,
            "table": DataFrame,
            "best": dict(profile=..., idx=..., row=...),
            "meta": {...}
          }
        """
        import numpy as np
        import pandas as pd
        import os

        vkey = _norm_version(version)
        ref_v = _norm_version(reference_version)
        v0 = self._rk_get_prior_version_key()

        if profiles is None or not isinstance(profiles, (list, tuple)) or len(profiles) == 0:
            raise ValueError("profiles must be a non-empty list of dict profiles")

        # reference mean grid (unified) for sampling
        lon2d, lat2d = self._get_unified_grid()
        _, _, ref_mean = self._get_prior_mean_unified(ref_v, "MMI")
        if ref_mean is None:
            return {"ok": False, "note": f"Reference mean missing for ref_v={ref_v} (MMI).", "table": None}

        ref_mean = np.asarray(ref_mean, dtype=float)

        rows = []
        for i, prof in enumerate(profiles):
            prep = self.cdi_prepare_filtered(
                version=vkey,
                margin_deg=margin_deg,
                profile=prof,
                apply_conditioning=apply_conditioning,
                apply_weighting=apply_weighting,
                weight_mode=weight_mode,
                weight_clip=weight_clip,
                debug=debug,
            )

            if not prep.get("ok", False):
                rows.append(
                    {
                        "idx": i,
                        "ok": False,
                        "n_kept": 0,
                        "rmse_ref": np.nan,
                        "mae_ref": np.nan,
                        "bias_ref": np.nan,
                        "std_ref": np.nan,
                        "profile": prof,
                        "note": prep.get("note", "prep failed"),
                    }
                )
                continue

            obs = prep["obs_filtered"]
            if obs is None or obs.empty:
                rows.append(
                    {
                        "idx": i,
                        "ok": False,
                        "n_kept": 0,
                        "rmse_ref": np.nan,
                        "mae_ref": np.nan,
                        "bias_ref": np.nan,
                        "std_ref": np.nan,
                        "profile": prof,
                        "note": "empty after prep",
                    }
                )
                continue

            # compare CDI values to reference mean at obs locations
            ref_at_obs = self._rk_sample_grid_nn(lon2d, lat2d, ref_mean, obs["lon"].values, obs["lat"].values)
            y = obs["value"].to_numpy(dtype=float)
            r = y - ref_at_obs.astype(float)

            rmse = float(np.sqrt(np.nanmean(r**2))) if len(r) else np.nan
            mae = float(np.nanmean(np.abs(r))) if len(r) else np.nan
            bias = float(np.nanmean(r)) if len(r) else np.nan
            std = float(np.nanstd(r)) if len(r) else np.nan

            rows.append(
                {
                    "idx": i,
                    "ok": True,
                    "n_kept": int(len(obs)),
                    "rmse_ref": rmse,
                    "mae_ref": mae,
                    "bias_ref": bias,
                    "std_ref": std,
                    "profile": prep.get("meta", {}).get("profile_used", prof),
                    "note": "ok",
                }
            )

        tab = pd.DataFrame(rows)
        # best = minimal rmse_ref among ok
        tab_ok = tab[tab["ok"] == True].copy()
        best = None
        if not tab_ok.empty:
            j = int(tab_ok["rmse_ref"].astype(float).idxmin())
            best = {"idx": int(tab.loc[j, "idx"]), "profile": tab.loc[j, "profile"], "row": tab.loc[j].to_dict()}

        # save CSV
        if save_csv_path:
            try:
                os.makedirs(os.path.dirname(str(save_csv_path)), exist_ok=True)
            except Exception:
                pass
            tab.to_csv(str(save_csv_path), index=False)

        # optional plots: RMSE vs n_kept + residual hist for best
        if make_plots and plot_out_dir:
            try:
                import matplotlib.pyplot as plt

                os.makedirs(str(plot_out_dir), exist_ok=True)

                fig = plt.figure(figsize=(10.5, 4.0), dpi=dpi)
                ax = fig.add_subplot(1, 1, 1)
                ax.scatter(tab["n_kept"].to_numpy(dtype=float), tab["rmse_ref"].to_numpy(dtype=float), s=24)
                ax.set_xlabel("CDI points kept (n)")
                ax.set_ylabel("RMSE vs reference MMI")
                ax.set_title(f"CDI filtering diagnostic — v={vkey} vs ref={ref_v} (prior=v0 {v0})")
                fig.tight_layout()
                fig.savefig(os.path.join(str(plot_out_dir), f"cdi_diag_scatter_v{vkey}_ref{ref_v}.png"), bbox_inches="tight")
                plt.close(fig)

                if best is not None:
                    # recompute residuals for best to plot histogram
                    prep_best = self.cdi_prepare_filtered(
                        version=vkey,
                        margin_deg=margin_deg,
                        profile=best["profile"],
                        apply_conditioning=apply_conditioning,
                        apply_weighting=apply_weighting,
                        weight_mode=weight_mode,
                        weight_clip=weight_clip,
                        debug=debug,
                    )
                    if prep_best.get("ok", False):
                        obs = prep_best["obs_filtered"]
                        ref_at_obs = self._rk_sample_grid_nn(lon2d, lat2d, ref_mean, obs["lon"].values, obs["lat"].values)
                        r = obs["value"].to_numpy(dtype=float) - ref_at_obs.astype(float)

                        fig = plt.figure(figsize=(10.5, 4.0), dpi=dpi)
                        ax = fig.add_subplot(1, 1, 1)
                        ax.hist(r[np.isfinite(r)], bins=40)
                        ax.set_xlabel("CDI − Reference MMI residual")
                        ax.set_ylabel("count")
                        ax.set_title(f"Best profile residuals (idx={best['idx']}) — n={len(obs)}")
                        fig.tight_layout()
                        fig.savefig(os.path.join(str(plot_out_dir), f"cdi_diag_hist_best_v{vkey}_ref{ref_v}.png"), bbox_inches="tight")
                        plt.close(fig)
            except Exception:
                # plots are optional; do not fail the diagnostic
                pass

        return {
            "ok": True,
            "table": tab,
            "best": best,
            "meta": {"version": vkey, "reference_version": ref_v, "prior_version": v0, "n_profiles": int(len(profiles))},
        }


    
    def run_innovation_kriging_mmi_cdi_filtered(
        self,
        *,
        version,
        key=None,
        profile=None,
        margin_deg=0.05,
        apply_conditioning=True,
        apply_weighting=False,
        weight_mode="nresp_sqrt",
        weight_clip=(1.0, 50.0),
    
        # OK controls (MUST match _ok_krige_grid)
        neighbor_k=25,
        max_points=None,
        use_obs_sigma=True,
        variogram_model="exponential",
        range_km=80.0,
        sill=1.0,
        nugget=1e-6,
        ridge=1e-10,
    
        verbose=False,
        debug=False,
    ):
        """
        Innovation Kriging wrapper specifically for:
          - IMT = MMI
          - Observations = CDI ONLY, after CDI toolkit filtering/weighting
          - FIXED prior = v0 (first version in unified stack)
    
        Stores results under:
          uq_state["versions"][vkey]["updates"][key]
        (same layout as run_residual_kriging_update).
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
    
        if "updates" not in vpack or not isinstance(vpack.get("updates"), dict):
            vpack["updates"] = {}
    
        lon2d, lat2d = self._get_unified_grid()
        v0 = self._rk_get_prior_version_key()
    
        if key is None:
            key = f"rk__mmi_cdi_filtered__v{vkey}"
        key = str(key)
    
        t0 = time.time()
    
        # ---- prior on unified grid (v0 fixed) ----
        _, _, prior_mean = self._get_prior_mean_unified(v0, "MMI")
        _, _, prior_sigma = self._get_prior_sigma_unified(v0, "MMI")
        if prior_mean is None or prior_sigma is None:
            raise RuntimeError(f"Missing prior mean/sigma for v0={v0} imt=MMI.")
        prior_mean = np.asarray(prior_mean, dtype=float)
        prior_sigma = np.asarray(prior_sigma, dtype=float)
    
        # ---- filtered CDI observations (target version) ----
        prep = self.cdi_prepare_filtered(
            version=vkey,
            margin_deg=margin_deg,
            profile=profile,
            apply_conditioning=apply_conditioning,
            apply_weighting=apply_weighting,
            weight_mode=weight_mode,
            weight_clip=weight_clip,
            debug=debug,
        )
        if not prep.get("ok", False):
            out = {
                "ok": False,
                "note": prep.get("note", "cdi_prepare_filtered failed"),
                "meta": {"version": vkey, "prior_version": v0, "imt": "MMI", "mode": "mmi_cdi_filtered", "key": key},
                "prep": prep,
            }
            vpack["updates"][key] = out
            return out
    
        obs_f = prep["obs_filtered"]
        if obs_f is None or obs_f.empty:
            out = {
                "ok": False,
                "note": "No CDI obs after filtering.",
                "meta": {"version": vkey, "prior_version": v0, "imt": "MMI", "mode": "mmi_cdi_filtered", "key": key},
                "prep": prep,
            }
            vpack["updates"][key] = out
            return out
    
        # ---- residuals against v0 prior ----
        prior_at_obs = self._rk_sample_grid_nn(
            lon2d, lat2d, prior_mean,
            obs_f["lon"].values, obs_f["lat"].values
        )
        y = obs_f["value"].to_numpy(dtype=float)
        resid = y - prior_at_obs.astype(float)
    
        # ---- Build residual obs_df for _ok_krige_grid ----
        # _ok_krige_grid expects obs_df with lon/lat/value and optional sigma.
        obs_r = obs_f.copy()
        obs_r["value"] = np.asarray(resid, dtype=float)
    
        # Ensure sigma exists and is finite if we want to use it
        if use_obs_sigma:
            if "sigma" not in obs_r.columns:
                # fallback: no sigma column => kriging proceeds unweighted
                pass
            else:
                s = pd.to_numeric(obs_r["sigma"], errors="coerce").to_numpy(dtype=float)
                s = np.where(np.isfinite(s), s, np.nan)
                # avoid zeros/negatives
                s = np.where(np.isfinite(s), np.maximum(s, 1e-6), np.nan)
                obs_r["sigma"] = s
    
        # ---- krige residuals (DF-based API) ----
        zhat_resid, var_resid = self._ok_krige_grid(
            obs_r,
            lon2d, lat2d,
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
            out = {
                "ok": False,
                "note": "Kriging backend returned None.",
                "meta": {"version": vkey, "prior_version": v0, "imt": "MMI", "mode": "mmi_cdi_filtered", "key": key},
                "prep": prep,
            }
            vpack["updates"][key] = out
            return out
    
        zhat_resid = np.asarray(zhat_resid, dtype=float)
        var_resid = np.asarray(var_resid, dtype=float)
    
        # ---- posterior ----
        post_mean = prior_mean + zhat_resid
        post_sigma = np.sqrt(np.maximum(prior_sigma, 0.0) ** 2 + np.maximum(var_resid, 0.0))
    
        # ---- metrics at obs ----
        post_at_obs = self._rk_sample_grid_nn(
            lon2d, lat2d, post_mean,
            obs_f["lon"].values, obs_f["lat"].values
        )
        prior_at_obs2 = prior_at_obs.astype(float)
    
        e_prior = y - prior_at_obs2
        e_post = y - post_at_obs.astype(float)
    
        rmse_prior = float(np.sqrt(np.nanmean(e_prior**2))) if len(e_prior) else np.nan
        rmse_post = float(np.sqrt(np.nanmean(e_post**2))) if len(e_post) else np.nan
        mae_prior = float(np.nanmean(np.abs(e_prior))) if len(e_prior) else np.nan
        mae_post = float(np.nanmean(np.abs(e_post))) if len(e_post) else np.nan
    
        meta = {
            "imt": "MMI",
            "mode": "mmi_cdi_filtered",
            "key": key,
            "version": vkey,
            "prior_version": v0,
            "t_seconds": float(time.time() - t0),
            "prep_meta": prep.get("meta"),
            "krige": {
                "neighbor_k": neighbor_k,
                "max_points": max_points,
                "use_obs_sigma": bool(use_obs_sigma),
                "variogram_model": variogram_model,
                "range_km": float(range_km),
                "sill": float(sill),
                "nugget": float(nugget),
                "ridge": float(ridge),
            },
        }
    
        out = {
            "ok": True,
            "meta": meta,
            "prior": {"mean_grid": prior_mean, "sigma_grid": prior_sigma},
            "posterior": {"mean_grid": post_mean, "sigma_grid": post_sigma},
            "residual": {
                "resid_at_obs": np.asarray(resid, dtype=float),
                "kriged_residual": zhat_resid,
                "kriged_var": var_resid,
            },
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
            print(f"[RK-CDI] v={vkey} prior=v0({v0}) imt=MMI n={len(obs_f)} rmse_prior={rmse_prior:.4f} rmse_post={rmse_post:.4f}")
    
        return out




# ======================================================================
# CDI 5-SET TEST HARNESS + PLOTTING (thesis-ready, simple lon/lat maps)
# ----------------------------------------------------------------------
# Adds:
#   - cdi_profiles_5sets(): A–E fixed filter sets
#   - cdi_filtering_diagnostic_5sets(): run sets + compute metrics vs reference MMI
#   - uq_plot_cdi_rmse_vs_nkept(): scatter (A–E)
#   - uq_plot_cdi_residual_hist(): hist overlay or small multiples (A–E)
#   - uq_plot_cdi_residual_map(): spatial residual scatter map (simple lon/lat)
# ======================================================================

    def cdi_profiles_5sets(
        self,
        *,
        nresp_trust_only: int = 10,
        nresp_mild: int = 3,
        local_radius_km: float = 25.0,
        outlier_k: float = 2.5,
        min_neighbors: int = 4,
        cluster_eps_km: float = 2.0,
        cluster_min_pts: int = 3,
        grid_bin_km_C: float = 10.0,
        grid_bin_km_E: float = 20.0,
        quantile_D=(0.05, 0.95),
        quantile_off=(0.0, 1.0),
    ):
        """
        Return the finalized 5 CDI filter sets (A–E) as named dicts of kwargs
        compatible with _cdi_condition(...).

        Sets:
          A Raw (control): Stage-1 only (extent + validity done elsewhere) => disable all conditioning by strategy
          B Trust-only: nresp gate only
          C Spatial-consistency: local outlier + decluster (grid_thin)
          D Physics-informed: residual trimming vs v0 (quantile_residual) + mild trust
          E Areal/binned: stronger declustering/binning (grid_thin with larger bin) + mild trust

        Notes:
          - We do NOT assume any extra preprocessing (no radial binning here).
          - We keep the API strictly aligned to existing _cdi_condition signature in this file.
        """
        # Helper: safe strategy tuples
        # (we assume _cdi_condition understands these labels because run_residual_kriging_update uses them)
        STRAT_A = tuple()  # no conditioning beyond trust gate (which we set very permissive)
        STRAT_B = tuple()  # trust gate only
        STRAT_C = ("local_outlier", "grid_thin")
        STRAT_D = ("quantile_residual",)
        STRAT_E = ("grid_thin",)

        # A: Raw control (we keep trust gate extremely permissive; no other stages)
        A = dict(
            cdi_trust_nresp_ge=1,
            cdi_enable_local_outlier=False,
            cdi_local_radius_km=float(local_radius_km),
            cdi_outlier_k=float(outlier_k),
            cdi_min_neighbors=int(min_neighbors),
            cdi_enable_clustering=False,
            cdi_cluster_eps_km=float(cluster_eps_km),
            cdi_cluster_min_pts=int(cluster_min_pts),
            cdi_strategy=STRAT_A,
            cdi_grid_bin_km=float(grid_bin_km_C),
            cdi_grid_agg="median",
            cdi_quantile=quantile_off,
        )

        # B: Trust-only
        B = dict(
            cdi_trust_nresp_ge=int(nresp_trust_only),
            cdi_enable_local_outlier=False,
            cdi_local_radius_km=float(local_radius_km),
            cdi_outlier_k=float(outlier_k),
            cdi_min_neighbors=int(min_neighbors),
            cdi_enable_clustering=False,
            cdi_cluster_eps_km=float(cluster_eps_km),
            cdi_cluster_min_pts=int(cluster_min_pts),
            cdi_strategy=STRAT_B,
            cdi_grid_bin_km=float(grid_bin_km_C),
            cdi_grid_agg="median",
            cdi_quantile=quantile_off,
        )

        # C: Spatial-consistency (local outlier + decluster)
        C = dict(
            cdi_trust_nresp_ge=int(nresp_mild),
            cdi_enable_local_outlier=True,
            cdi_local_radius_km=float(local_radius_km),
            cdi_outlier_k=float(outlier_k),
            cdi_min_neighbors=int(min_neighbors),
            cdi_enable_clustering=False,
            cdi_cluster_eps_km=float(cluster_eps_km),
            cdi_cluster_min_pts=int(cluster_min_pts),
            cdi_strategy=STRAT_C,
            cdi_grid_bin_km=float(grid_bin_km_C),
            cdi_grid_agg="median",
            cdi_quantile=quantile_off,
        )

        # D: Physics-informed (residual trimming vs v0)
        D = dict(
            cdi_trust_nresp_ge=int(nresp_mild),
            cdi_enable_local_outlier=False,  # keep D focused on residual-based physics filtering
            cdi_local_radius_km=float(local_radius_km),
            cdi_outlier_k=float(outlier_k),
            cdi_min_neighbors=int(min_neighbors),
            cdi_enable_clustering=False,
            cdi_cluster_eps_km=float(cluster_eps_km),
            cdi_cluster_min_pts=int(cluster_min_pts),
            cdi_strategy=STRAT_D,
            cdi_grid_bin_km=float(grid_bin_km_C),
            cdi_grid_agg="median",
            cdi_quantile=quantile_D,
        )

        # E: Areal/binned (stronger binning/declustering)
        E = dict(
            cdi_trust_nresp_ge=int(nresp_mild),
            cdi_enable_local_outlier=False,
            cdi_local_radius_km=float(local_radius_km),
            cdi_outlier_k=float(outlier_k),
            cdi_min_neighbors=int(min_neighbors),
            cdi_enable_clustering=False,
            cdi_cluster_eps_km=float(cluster_eps_km),
            cdi_cluster_min_pts=int(cluster_min_pts),
            cdi_strategy=STRAT_E,
            cdi_grid_bin_km=float(grid_bin_km_E),
            cdi_grid_agg="median",
            cdi_quantile=quantile_off,
        )

        return {
            "A_raw": A,
            "B_trust_only": B,
            "C_spatial_consistency": C,
            "D_physics_residual": D,
            "E_areal_binned": E,
        }

    def cdi_filtering_diagnostic_5sets(
        self,
        *,
        version,
        reference_version,
        margin_deg: float = 0.05,
        profiles_5sets: dict = None,
        debug: bool = False,
    ):
        """
        Run the 5 CDI sets (A–E) for one target version and evaluate against
        a later reference MMI shakemap version.

        Uses only existing helpers in this file:
          - build_observations(..., imt="MMI", dyfi_source="cdi")
          - filter_observations_to_extent(...)
          - _cdi_condition(version=v0, obs_df=...)  [IMPORTANT: compare vs fixed prior v0]
          - _get_unified_grid()
          - _get_prior_mean_unified(reference_version, "MMI")
          - _rk_sample_grid_nn()

        Returns dict with:
          - results_df: per-set metrics
          - residuals: dict[name] -> residual array (CDI - ref MMI at obs)
          - obs_used: dict[name] -> obs dataframe used (post-filter)
          - meta
        """
        import numpy as np
        import pandas as pd

        vkey = _norm_version(version)
        ref_v = _norm_version(reference_version)
        v0 = self._rk_get_prior_version_key()

        if profiles_5sets is None:
            profiles_5sets = self.cdi_profiles_5sets()

        # 1) build CDI obs (minimal filtering inside build_observations)
        obs_raw = self.build_observations(vkey, imt="MMI", dyfi_source="cdi", sigma_override=None)
        if obs_raw is None or getattr(obs_raw, "empty", False):
            return {
                "ok": False,
                "note": "No CDI observations returned by build_observations(dyfi_source='cdi').",
                "results_df": None,
                "residuals": {},
                "obs_used": {},
                "meta": {"version": vkey, "reference_version": ref_v, "prior_version": v0},
            }

        # 2) extent filter (unified grid)
        obs_ext, obs_drop = self.filter_observations_to_extent(
            obs_raw,
            version=vkey,
            grid_mode="unified",
            margin_deg=float(margin_deg),
            return_dropped=True,
        )
        if obs_ext is None or getattr(obs_ext, "empty", False):
            return {
                "ok": False,
                "note": "All CDI observations outside unified extent after filter_observations_to_extent().",
                "results_df": None,
                "residuals": {},
                "obs_used": {},
                "meta": {"version": vkey, "reference_version": ref_v, "prior_version": v0, "n_raw": int(len(obs_raw))},
            }

        # 3) reference MMI mean grid for sampling
        lon2d, lat2d = self._get_unified_grid()
        _, _, ref_mean = self._get_prior_mean_unified(ref_v, "MMI")
        if ref_mean is None:
            return {
                "ok": False,
                "note": f"Reference mean missing for reference_version={ref_v} imt=MMI.",
                "results_df": None,
                "residuals": {},
                "obs_used": {},
                "meta": {"version": vkey, "reference_version": ref_v, "prior_version": v0},
            }
        ref_mean = np.asarray(ref_mean, dtype=float)

        rows = []
        residuals = {}
        obs_used = {}

        n_raw = int(len(obs_raw))
        n_ext = int(len(obs_ext))

        # 4) run each set through _cdi_condition (vs v0) then compare to ref at obs
        for name, prof in profiles_5sets.items():
            df = obs_ext.copy()

            # Apply CDI conditioning vs FIXED prior v0
            try:
                df2, meta2 = self._cdi_condition(
                    version=v0,
                    obs_df=df,
                    margin_deg=float(margin_deg),
                    debug=bool(debug),
                    **prof,
                )
            except TypeError as e:
                # signature mismatch or unexpected kwargs -> fail this profile cleanly
                df2, meta2 = None, {"ok": False, "err": f"TypeError: {e}"}
            except Exception as e:
                df2, meta2 = None, {"ok": False, "err": str(e)}

            if df2 is None or getattr(df2, "empty", False):
                rows.append(
                    dict(
                        set_name=name,
                        ok=False,
                        n_kept=0,
                        retained_frac=np.nan,
                        rmse_ref=np.nan,
                        mae_ref=np.nan,
                        bias_ref=np.nan,
                        std_ref=np.nan,
                        note=(meta2.get("err") if isinstance(meta2, dict) else "empty after conditioning"),
                    )
                )
                continue

            # Compare CDI values to reference MMI at obs locations
            ref_at_obs = self._rk_sample_grid_nn(lon2d, lat2d, ref_mean, df2["lon"].values, df2["lat"].values)
            y = df2["value"].to_numpy(dtype=float)
            r = y - ref_at_obs.astype(float)

            rmse = float(np.sqrt(np.nanmean(r**2))) if len(r) else np.nan
            mae = float(np.nanmean(np.abs(r))) if len(r) else np.nan
            bias = float(np.nanmean(r)) if len(r) else np.nan
            std = float(np.nanstd(r)) if len(r) else np.nan

            residuals[name] = r
            obs_used[name] = df2

            nk = int(len(df2))
            rows.append(
                dict(
                    set_name=name,
                    ok=True,
                    n_kept=nk,
                    retained_frac=(nk / max(n_raw, 1)),
                    rmse_ref=rmse,
                    mae_ref=mae,
                    bias_ref=bias,
                    std_ref=std,
                    note="ok",
                )
            )

        results_df = pd.DataFrame(rows)
        # Best = min rmse_ref among ok
        best_name = None
        if not results_df.empty and (results_df["ok"] == True).any():
            best_row = results_df[results_df["ok"] == True].sort_values("rmse_ref", ascending=True).iloc[0]
            best_name = str(best_row["set_name"])

        return {
            "ok": True,
            "results_df": results_df,
            "residuals": residuals,
            "obs_used": obs_used,
            "best_set": best_name,
            "meta": {
                "version": vkey,
                "reference_version": ref_v,
                "prior_version": v0,
                "margin_deg": float(margin_deg),
                "n_raw": n_raw,
                "n_extent": n_ext,
                "n_dropped_extent": int(len(obs_drop)) if obs_drop is not None else None,
            },
        }

    # ----------------------------
    # Plot 1: RMSE vs n_kept scatter
    # ----------------------------
    def uq_plot_cdi_rmse_vs_nkept(
        self,
        diag: dict,
        *,
        title: str = None,
        x_key: str = "n_kept",
        y_key: str = "rmse_ref",
    
        # labeling
        annotate: bool = True,
        annotate_kwargs: dict = None,
    
        # figure
        figsize=(10, 4.5),
        font_sizes: dict = None,
    
        # axes / grid
        grid: bool = True,
        grid_kwargs: dict = None,
    
        # legend (NOW DEFAULT TRUE)
        legend: bool = True,
        legend_kwargs: dict = None,
    
        xlim=None,
        ylim=None,
        show_title: bool = True,
    
        # saving
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        dpi: int = 300,
        show: bool = True,
    ):
        """
        Plot RMSE vs number of CDI points kept for 5-set diagnostic (single event).
    
        - Each CDI set is plotted as its own scatter point (enables legend).
        - `annotate=True` places text next to points.
        - `legend=True` shows set names in a legend (default).
        """
    
        import os
        import numpy as np
        import matplotlib.pyplot as plt
    
        # ----------------------------
        # defaults
        # ----------------------------
        if font_sizes is None:
            font_sizes = dict(
                title=12,
                label=11,
                tick=10,
                annot=10,
                legend=10,
            )
    
        if grid_kwargs is None:
            grid_kwargs = {"alpha": 0.25}
    
        if legend_kwargs is None:
            legend_kwargs = {"loc": "best", "frameon": True}
    
        if annotate_kwargs is None:
            annotate_kwargs = {"xytext": (6, 6), "textcoords": "offset points"}
    
        # ----------------------------
        # input validation
        # ----------------------------
        df = diag.get("results_df")
        meta = diag.get("meta", {}) or {}
    
        if df is None or df.empty:
            raise RuntimeError("diag['results_df'] missing or empty.")
    
        # ----------------------------
        # figure
        # ----------------------------
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
    
        # ----------------------------
        # scatter: ONE POINT PER SET
        # ----------------------------
        for _, row in df.iterrows():
            name = str(row.get("set_name", ""))
            xi = float(row.get(x_key, np.nan))
            yi = float(row.get(y_key, np.nan))
    
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
    
            ax.scatter(
                xi,
                yi,
                s=60,
                label=name if legend else None,
                zorder=3,
            )
    
            if annotate:
                ax.annotate(
                    name,
                    (xi, yi),
                    fontsize=font_sizes.get("annot", 10),
                    **annotate_kwargs,
                )
    
        # ----------------------------
        # axes
        # ----------------------------
        ax.set_xlabel(x_key, fontsize=font_sizes.get("label", 11))
        ax.set_ylabel(y_key, fontsize=font_sizes.get("label", 11))
        ax.tick_params(labelsize=font_sizes.get("tick", 10))
    
        if show_title:
            if title is None:
                title = (
                    f"CDI 5-set diagnostic — RMSE vs n_kept "
                    f"(v{meta.get('version')} ref{meta.get('reference_version')})"
                )
            ax.set_title(title, fontsize=font_sizes.get("title", 12))
    
        if grid:
            ax.grid(True, **grid_kwargs)
    
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    
        # ----------------------------
        # legend
        # ----------------------------
        if legend:
            ax.legend(
                fontsize=font_sizes.get("legend", 10),
                **legend_kwargs,
            )
    
        fig.tight_layout()
    
        # ----------------------------
        # save
        # ----------------------------
        if output_path and save:
            os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
            base, ext = os.path.splitext(str(output_path))
            if ext.lower() in (".png", ".pdf", ".svg"):
                fig.savefig(str(output_path), bbox_inches="tight")
            else:
                for fmt in save_formats:
                    fig.savefig(f"{base}.{fmt}", bbox_inches="tight")
    
        # ----------------------------
        # show / close
        # ----------------------------
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig
    

    # ----------------------------
    # Plot 2: residual histogram (overlay or small multiples)
    # ----------------------------
    def uq_plot_cdi_residual_hist(
        self,
        diag: dict,
        *,
        sets=None,
        mode: str = "overlay",   # "overlay" or "small_multiples"
        bins: int = 40,
        density: bool = False,
        figsize=(11, 4.5),
        font_sizes: dict = None,
        grid: bool = True,
        grid_kwargs: dict = None,
        legend: bool = True,
        legend_kwargs: dict = None,
        xlim=None,
        ylim=None,
        show_title: bool = True,
        title: str = None,
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        dpi: int = 300,
        show: bool = True,
    ):
        """
        Plot residual distributions for selected sets.
        Residual defined as (CDI - reference MMI at obs).

        diag must include diag["residuals"] dict.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        if font_sizes is None:
            font_sizes = {"title": 12, "label": 11, "tick": 10, "legend": 10}
        if grid_kwargs is None:
            grid_kwargs = {"alpha": 0.25}
        if legend_kwargs is None:
            legend_kwargs = {"loc": "best", "frameon": True, "fontsize": font_sizes.get("legend", 10)}

        res = diag.get("residuals") or {}
        meta = diag.get("meta", {}) or {}

        if sets is None:
            sets = list(res.keys())
        sets = [s for s in sets if s in res]

        if len(sets) == 0:
            raise RuntimeError("No residuals found for requested sets.")

        if show_title and title is None:
            title = f"CDI residuals vs reference — v={meta.get('version')} ref={meta.get('reference_version')}"

        if str(mode).lower().strip() == "small_multiples":
            n = len(sets)
            fig = plt.figure(figsize=figsize, dpi=dpi)
            # simple layout: 1 row if <=3 else 2 rows
            ncols = 3 if n > 3 else n
            nrows = int(np.ceil(n / ncols))
            for i, s in enumerate(sets):
                ax = fig.add_subplot(nrows, ncols, i + 1)
                r = np.asarray(res[s], dtype=float)
                r = r[np.isfinite(r)]
                ax.hist(r, bins=bins, density=density)
                ax.set_title(s, fontsize=font_sizes.get("label", 11))
                ax.tick_params(labelsize=font_sizes.get("tick", 10))
                if grid:
                    ax.grid(True, **grid_kwargs)
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if i % ncols == 0:
                    ax.set_ylabel("density" if density else "count", fontsize=font_sizes.get("label", 11))
                ax.set_xlabel("CDI − Reference MMI", fontsize=font_sizes.get("label", 11))

            if show_title:
                fig.suptitle(title, fontsize=font_sizes.get("title", 12))
                fig.tight_layout(rect=[0, 0, 1, 0.92])
            else:
                fig.tight_layout()

        else:
            # overlay
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(1, 1, 1)
            for s in sets:
                r = np.asarray(res[s], dtype=float)
                r = r[np.isfinite(r)]
                ax.hist(r, bins=bins, density=density, alpha=0.5, label=s)

            ax.set_xlabel("CDI − Reference MMI", fontsize=font_sizes.get("label", 11))
            ax.set_ylabel("density" if density else "count", fontsize=font_sizes.get("label", 11))
            ax.tick_params(labelsize=font_sizes.get("tick", 10))
            if show_title:
                ax.set_title(title, fontsize=font_sizes.get("title", 12))
            if grid:
                ax.grid(True, **grid_kwargs)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if legend:
                ax.legend(**legend_kwargs)

            fig.tight_layout()

        if output_path and save:
            os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
            base, ext = os.path.splitext(str(output_path))
            if ext.lower() in (".png", ".pdf", ".svg"):
                fig.savefig(str(output_path), bbox_inches="tight")
            else:
                for fmt in save_formats:
                    fig.savefig(f"{base}.{fmt}", bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    # ----------------------------
    # Plot 3: spatial residual map (simple lon/lat scatter)
    # ----------------------------
    def uq_plot_cdi_residual_map(
        self,
        diag: dict,
        *,
        set_name: str = None,
    
        # figure
        figsize=(6.0, 6.0),
        dpi: int = 300,
    
        # points (residuals)
        s: float = 16.0,
        alpha: float = 0.75,
        marker: str = "o",
        edgecolor=None,
        linewidths: float = 0.0,
    
        # residual color scaling
        cmap: str = "RdBu_r",
        vmin=None,
        vmax=None,
        symmetric: bool = True,
        robust: bool = True,
        robust_q=(2.0, 98.0),
        center: float = 0.0,
    
        # colorbar (residual)
        show_colorbar: bool = True,
        cbar_label: str = "CDI − Reference MMI",
        cbar_label_kwargs: dict = None,
        cbar_tick_params: dict = None,
        colorbar_kwargs: dict = None,
    
        # cartopy / map
        use_cartopy: bool = True,
        projection: str = "platecarree",  # "platecarree" | "mercator" | "utm"
        coastlines: bool = True,
        coastlines_kwargs: dict = None,
        borders: bool = True,
        borders_kwargs: dict = None,
        land: bool = True,
        land_kwargs: dict = None,
        ocean: bool = False,
        ocean_kwargs: dict = None,
        lakes: bool = False,
        lakes_kwargs: dict = None,
        rivers: bool = False,
        rivers_kwargs: dict = None,
    
        # gridlines
        gridlines: bool = True,
        gridlines_kwargs: dict = None,
    
        # extent
        xlim=None,
        ylim=None,
        extent_pad_deg: float = 0.25,
    
        # ShakeMap background (MMI mean)
        show_shakemap: bool = False,
        shakemap_source: str = "reference",    # "reference" | "target" | "001"/"014"/...
        shakemap_imt: str = "MMI",
        shakemap_scale_type: str = "usgs",
        shakemap_units=None,                  # keep None for MMI
        shakemap_alpha: float = 0.55,
        shakemap_shading: str = "auto",
        shakemap_rasterized: bool = True,
        shakemap_kwargs: dict = None,
    
        # titles/fonts
        show_title: bool = True,
        title: str = None,
        font_sizes: dict = None,
    
        # zorder control (ALL parts)
        zorders: dict = None,
    
        # saving
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        show: bool = True, extent=None,

    ):
        """
        Spatial residual map: CDI residual points (CDI − reference MMI) on a Cartopy basemap.
        Optional ShakeMap background layer (MMI mean grid) with SHAKEmap discrete scale
        via self.contour_scale(...).
    
        diag: output from cdi_filtering_diagnostic_5sets(), expected keys:
          - diag["obs_used"][set_name] DataFrame with lon, lat
          - diag["residuals"][set_name] residual array
          - diag["meta"] includes: version, reference_version, prior_version
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
    
        # ---- defaults ----
        if font_sizes is None:
            font_sizes = {"title": 12, "label": 11, "tick": 10}
    
        if colorbar_kwargs is None:
            colorbar_kwargs = {"shrink": 0.86, "pad": 0.02}
    
        if cbar_label_kwargs is None:
            cbar_label_kwargs = {"fontsize": font_sizes.get("label", 11)}
    
        if cbar_tick_params is None:
            cbar_tick_params = {"labelsize": font_sizes.get("tick", 10)}
    
        if coastlines_kwargs is None:
            coastlines_kwargs = {"linewidth": 0.7}
    
        if borders_kwargs is None:
            borders_kwargs = {"linewidth": 0.6, "linestyle": "-"}
    
        if land_kwargs is None:
            land_kwargs = {"alpha": 0.25}
    
        if ocean_kwargs is None:
            ocean_kwargs = {"alpha": 0.15}
    
        if lakes_kwargs is None:
            lakes_kwargs = {"alpha": 0.15}
    
        if rivers_kwargs is None:
            rivers_kwargs = {"alpha": 0.25}
    
        if gridlines_kwargs is None:
            gridlines_kwargs = {"draw_labels": True, "linewidth": 0.4, "alpha": 0.35, "linestyle": "--"}
    
        if shakemap_kwargs is None:
            shakemap_kwargs = {}
    
        # zorder defaults (user can override via dict)
        zdef = {
            "ocean": 0,
            "land": 1,
            "shakemap": 2,
            "lakes": 3,
            "rivers": 3,
            "borders": 4,
            "coastlines": 5,
            "gridlines": 6,
            "points": 10,
        }
        if isinstance(zorders, dict):
            zdef.update(zorders)
    
        meta = diag.get("meta", {}) or {}
        obs_used = diag.get("obs_used") or {}
        residuals = diag.get("residuals") or {}
    
        if set_name is None:
            set_name = diag.get("best_set")
        if set_name is None:
            raise ValueError("set_name not provided and diag['best_set'] is None.")
        if set_name not in obs_used or set_name not in residuals:
            raise KeyError(f"set_name={set_name} not found in diag outputs.")
    
        df = obs_used[set_name]
        r = np.asarray(residuals[set_name], dtype=float)
    
        lon = df["lon"].to_numpy(dtype=float)
        lat = df["lat"].to_numpy(dtype=float)
    
        # drop non-finite
        m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(r)
        lonp = lon[m]
        latp = lat[m]
        rp = r[m]


        # ---- determine extent ----
        # Priority:
        #   1) user-provided box extent (xlim+ylim)
        #   2) if ShakeMap background requested -> use unified grid extent (stable, reproducible)
        #   3) else -> derive from CDI points (with padding)
        if extent is not None:
            extent = [float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])]
        elif xlim is not None and ylim is not None:
            extent = [float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1])]
        else:
            if bool(show_shakemap):
                # Use unified grid extent (preferred when background grid is plotted)
                try:
                    v_for_extent = None
                    src = str(shakemap_source).lower().strip()
                    if src in ("ref", "reference"):
                        v_for_extent = _norm_version(meta.get("reference_version"))
                    elif src in ("tgt", "target"):
                        v_for_extent = _norm_version(meta.get("version"))
                    else:
                        v_for_extent = _norm_version(shakemap_source)
        
                    ext = self._get_grid_extent(v_for_extent, grid_mode="unified", margin_deg=0.0)
                    extent = [float(ext[0]), float(ext[1]), float(ext[2]), float(ext[3])]
                except Exception:
                    # fallback to point-driven extent if something unexpected happens
                    if lonp.size > 0 and latp.size > 0:
                        extent = [
                            float(np.nanmin(lonp) - extent_pad_deg),
                            float(np.nanmax(lonp) + extent_pad_deg),
                            float(np.nanmin(latp) - extent_pad_deg),
                            float(np.nanmax(latp) + extent_pad_deg),
                        ]
                    else:
                        extent = None
            else:
                # Point-driven extent (default behavior)
                if lonp.size > 0 and latp.size > 0:
                    extent = [
                        float(np.nanmin(lonp) - extent_pad_deg),
                        float(np.nanmax(lonp) + extent_pad_deg),
                        float(np.nanmin(latp) - extent_pad_deg),
                        float(np.nanmax(latp) + extent_pad_deg),
                    ]
                else:
                    extent = None

    
        # ---- residual color scaling ----
        vmin_use = vmin
        vmax_use = vmax
        if vmin_use is None or vmax_use is None:
            if rp.size == 0:
                vmin_use, vmax_use = (-1.0, 1.0)
            else:
                if robust:
                    qlo, qhi = float(robust_q[0]), float(robust_q[1])
                    lo = float(np.nanpercentile(rp, qlo))
                    hi = float(np.nanpercentile(rp, qhi))
                else:
                    lo = float(np.nanmin(rp))
                    hi = float(np.nanmax(rp))
    
                if symmetric:
                    mabs = float(np.nanmax(np.abs([lo - center, hi - center])))
                    vmin_use = center - mabs
                    vmax_use = center + mabs
                else:
                    vmin_use = lo if vmin_use is None else vmin_use
                    vmax_use = hi if vmax_use is None else vmax_use
    
        # ---- build fig/axes ----
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = None
        used_cartopy = False
    
        if use_cartopy:
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
    
                proj_name = str(projection).lower().strip()
                if proj_name == "mercator":
                    proj = ccrs.Mercator()
                elif proj_name == "utm":
                    lon0 = float(np.nanmean(lonp)) if lonp.size else 0.0
                    lat0 = float(np.nanmean(latp)) if latp.size else 0.0
                    zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
                    south = bool(lat0 < 0)
                    proj = ccrs.UTM(zone=zone, southern_hemisphere=south)
                else:
                    proj = ccrs.PlateCarree()
    
                ax = fig.add_subplot(1, 1, 1, projection=proj)
                used_cartopy = True
    
                if extent is not None:
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
                # Background features (assign zorder everywhere)
                if ocean:
                    ax.add_feature(cfeature.OCEAN, zorder=zdef["ocean"], **ocean_kwargs)
                if land:
                    ax.add_feature(cfeature.LAND, zorder=zdef["land"], **land_kwargs)
    
                # Optional ShakeMap background
                if show_shakemap:
                    # Determine which version to plot
                    src = str(shakemap_source).lower().strip()
                    if src in ("ref", "reference"):
                        v_bg = _norm_version(meta.get("reference_version"))
                    elif src in ("tgt", "target"):
                        v_bg = _norm_version(meta.get("version"))
                    else:
                        v_bg = _norm_version(shakemap_source)
    
                    _, _, bg_mean = self._get_prior_mean_unified(v_bg, str(shakemap_imt))
                    if bg_mean is not None:
                        # get unified grid for background (same lon/lat used in UQ)
                        lon2d, lat2d = self._get_unified_grid()
    
                        # use existing discrete ShakeMap scale
                        try:
                            cmap_bg, _bounds, _ticks, norm_bg, _label = self.contour_scale(
                                str(shakemap_imt), scale_type=str(shakemap_scale_type), units=shakemap_units
                            )
                        except Exception:
                            cmap_bg, norm_bg = None, None
    
                        pm = ax.pcolormesh(
                            lon2d,
                            lat2d,
                            np.asarray(bg_mean, dtype=float),
                            transform=ccrs.PlateCarree(),
                            shading=shakemap_shading,
                            alpha=float(shakemap_alpha),
                            cmap=cmap_bg,
                            norm=norm_bg,
                            rasterized=bool(shakemap_rasterized),
                            zorder=zdef["shakemap"],
                            **(shakemap_kwargs or {}),
                        )
                    # else: silently skip background if missing
    
                if lakes:
                    ax.add_feature(cfeature.LAKES, zorder=zdef["lakes"], **lakes_kwargs)
                if rivers:
                    ax.add_feature(cfeature.RIVERS, zorder=zdef["rivers"], **rivers_kwargs)
    
                if borders:
                    ax.add_feature(cfeature.BORDERS, zorder=zdef["borders"], **borders_kwargs)
                if coastlines:
                    ax.coastlines(zorder=zdef["coastlines"], **coastlines_kwargs)
    
                if gridlines:
                    gl = ax.gridlines(crs=ccrs.PlateCarree(), zorder=zdef["gridlines"], **gridlines_kwargs)
                    try:
                        gl.top_labels = False
                        gl.right_labels = False
                    except Exception:
                        pass
    
                sc = ax.scatter(
                    lonp, latp,
                    c=rp,
                    s=s,
                    alpha=alpha,
                    marker=marker,
                    cmap=cmap,
                    vmin=vmin_use,
                    vmax=vmax_use,
                    transform=ccrs.PlateCarree(),
                    edgecolors=edgecolor,
                    linewidths=linewidths,
                    zorder=zdef["points"],
                )
    
            except Exception:
                used_cartopy = False
                ax = None
    
        if not used_cartopy:
            # fallback (no cartopy)
            ax = fig.add_subplot(1, 1, 1)
    
            # Optional ShakeMap background in fallback mode
            if show_shakemap:
                src = str(shakemap_source).lower().strip()
                if src in ("ref", "reference"):
                    v_bg = _norm_version(meta.get("reference_version"))
                elif src in ("tgt", "target"):
                    v_bg = _norm_version(meta.get("version"))
                else:
                    v_bg = _norm_version(shakemap_source)
    
                _, _, bg_mean = self._get_prior_mean_unified(v_bg, str(shakemap_imt))
                if bg_mean is not None:
                    lon2d, lat2d = self._get_unified_grid()
                    try:
                        cmap_bg, _bounds, _ticks, norm_bg, _label = self.contour_scale(
                            str(shakemap_imt), scale_type=str(shakemap_scale_type), units=shakemap_units
                        )
                    except Exception:
                        cmap_bg, norm_bg = None, None
    
                    ax.pcolormesh(
                        lon2d, lat2d, np.asarray(bg_mean, dtype=float),
                        shading=shakemap_shading,
                        alpha=float(shakemap_alpha),
                        cmap=cmap_bg,
                        norm=norm_bg,
                        rasterized=bool(shakemap_rasterized),
                        zorder=zdef["shakemap"],
                        **(shakemap_kwargs or {}),
                    )
    
            sc = ax.scatter(
                lonp, latp,
                c=rp,
                s=s,
                alpha=alpha,
                marker=marker,
                cmap=cmap,
                vmin=vmin_use,
                vmax=vmax_use,
                edgecolors=edgecolor,
                linewidths=linewidths,
                zorder=zdef["points"],
            )
    
            if extent is not None:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
    
            ax.set_xlabel("Lon", fontsize=font_sizes.get("label", 11))
            ax.set_ylabel("Lat", fontsize=font_sizes.get("label", 11))
            ax.tick_params(labelsize=font_sizes.get("tick", 10))
            ax.grid(True, alpha=0.25, zorder=zdef["gridlines"])
    
        # ---- title ----
        if show_title:
            if title is None:
                title = f"CDI residuals ({set_name}) — v={meta.get('version')} ref={meta.get('reference_version')}"
            ax.set_title(title, fontsize=font_sizes.get("title", 12))
    
        # ---- residual colorbar ----
        if show_colorbar:
            cb = fig.colorbar(sc, ax=ax, **colorbar_kwargs)
            cb.set_label(cbar_label, **cbar_label_kwargs)
            cb.ax.tick_params(**cbar_tick_params)
    
        fig.tight_layout()
    
        # ---- save/show ----
        if output_path and save:
            os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
            base, ext = os.path.splitext(str(output_path))
            if ext.lower() in (".png", ".pdf", ".svg"):
                fig.savefig(str(output_path), bbox_inches="tight")
            else:
                for fmt in save_formats:
                    fig.savefig(f"{base}.{fmt}", bbox_inches="tight")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig
    

    
    
    
    def cdi_profiles_5sets(
        self,
        *,
        nresp_trust_only: int = 10,
        nresp_mild: int = 3,
        local_radius_km: float = 25.0,
        outlier_k: float = 2.5,
        min_neighbors: int = 4,
        cluster_eps_km: float = 2.0,
        cluster_min_pts: int = 3,
        grid_bin_km_C: float = 10.0,
        grid_bin_km_E: float = 20.0,
        quantile_D=(0.05, 0.95),
        quantile_off=(0.0, 1.0),
    ):
        """
        Return the finalized 5 CDI filter sets (A–E) as named dicts of kwargs
        compatible with _cdi_condition(...).
    
        IMPORTANT:
          These keys MUST match _cdi_condition signature exactly.
          Do NOT prefix keys with 'cdi_' except for parameters that are actually named that way.
        """
        # strategy tuples (must match _cdi_condition internal checks)
        STRAT_A = tuple()
        STRAT_B = tuple()
        STRAT_C = ("local_outlier", "grid_thin")
        STRAT_D = ("quantile_residual",)
        STRAT_E = ("grid_thin",)
    
        # A: Raw control
        A = dict(
            trust_nresp_ge=1,
            enable_local_outlier=False,
            local_radius_km=float(local_radius_km),
            outlier_k=float(outlier_k),
            min_neighbors=int(min_neighbors),
            enable_clustering=False,
            cluster_eps_km=float(cluster_eps_km),
            cluster_min_pts=int(cluster_min_pts),
            cdi_strategy=STRAT_A,
            cdi_grid_bin_km=float(grid_bin_km_C),
            cdi_grid_agg="median",
            cdi_quantile=quantile_off,
        )
    
        # B: Trust-only
        B = dict(
            trust_nresp_ge=int(nresp_trust_only),
            enable_local_outlier=False,
            local_radius_km=float(local_radius_km),
            outlier_k=float(outlier_k),
            min_neighbors=int(min_neighbors),
            enable_clustering=False,
            cluster_eps_km=float(cluster_eps_km),
            cluster_min_pts=int(cluster_min_pts),
            cdi_strategy=STRAT_B,
            cdi_grid_bin_km=float(grid_bin_km_C),
            cdi_grid_agg="median",
            cdi_quantile=quantile_off,
        )
    
        # C: Spatial-consistency (local outlier + grid thinning)
        C = dict(
            trust_nresp_ge=int(nresp_mild),
            enable_local_outlier=True,
            local_radius_km=float(local_radius_km),
            outlier_k=float(outlier_k),
            min_neighbors=int(min_neighbors),
            enable_clustering=False,
            cluster_eps_km=float(cluster_eps_km),
            cluster_min_pts=int(cluster_min_pts),
            cdi_strategy=STRAT_C,
            cdi_grid_bin_km=float(grid_bin_km_C),
            cdi_grid_agg="median",
            cdi_quantile=quantile_off,
        )
    
        # D: Physics-informed residual trimming vs v0
        D = dict(
            trust_nresp_ge=int(nresp_mild),
            enable_local_outlier=False,  # keep D focused on residual screening
            local_radius_km=float(local_radius_km),
            outlier_k=float(outlier_k),
            min_neighbors=int(min_neighbors),
            enable_clustering=False,
            cluster_eps_km=float(cluster_eps_km),
            cluster_min_pts=int(cluster_min_pts),
            cdi_strategy=STRAT_D,
            cdi_grid_bin_km=float(grid_bin_km_C),
            cdi_grid_agg="median",
            cdi_quantile=quantile_D,
        )
    
        # E: Areal/binned (stronger binning via grid thinning with larger bin)
        E = dict(
            trust_nresp_ge=int(nresp_mild),
            enable_local_outlier=False,
            local_radius_km=float(local_radius_km),
            outlier_k=float(outlier_k),
            min_neighbors=int(min_neighbors),
            enable_clustering=False,
            cluster_eps_km=float(cluster_eps_km),
            cluster_min_pts=int(cluster_min_pts),
            cdi_strategy=STRAT_E,
            cdi_grid_bin_km=float(grid_bin_km_E),
            cdi_grid_agg="median",
            cdi_quantile=quantile_off,
        )
    
        return {
            "A_raw": A,
            "B_trust_only": B,
            "C_spatial_consistency": C,
            "D_physics_residual": D,
            "E_areal_binned": E,
        }
    
    
    def cdi_filtering_diagnostic_5sets(
        self,
        *,
        version,
        reference_version,
        margin_deg: float = 0.05,
        profiles_5sets: dict = None,
        debug: bool = False,
    ):
        """
        Run the 5 CDI sets (A–E) for one target version and evaluate against
        a later reference MMI shakemap version.
    
        Uses only existing helpers in this file:
          - build_observations(..., imt="MMI", dyfi_source="cdi")
          - filter_observations_to_extent(...)
          - _cdi_condition(obs_df, version=v0, **profile_kwargs)  [IMPORTANT: compare vs fixed prior v0]
          - _get_unified_grid()
          - _get_prior_mean_unified(reference_version, "MMI")
          - _rk_sample_grid_nn()
    
        Returns dict with:
          - results_df: per-set metrics
          - residuals: dict[set_name] -> residual array (CDI - ref MMI at obs)
          - obs_used: dict[set_name] -> obs dataframe used (post-filter)
          - best_set
          - meta
        """
        import numpy as np
        import pandas as pd
    
        vkey = _norm_version(version)
        ref_v = _norm_version(reference_version)
        v0 = self._rk_get_prior_version_key()
    
        if profiles_5sets is None:
            profiles_5sets = self.cdi_profiles_5sets()
    
        # 1) build CDI obs (minimal filtering inside build_observations)
        obs_raw = self.build_observations(vkey, imt="MMI", dyfi_source="cdi", sigma_override=None)
        if obs_raw is None or getattr(obs_raw, "empty", False):
            return {
                "ok": False,
                "note": "No CDI observations returned by build_observations(dyfi_source='cdi').",
                "results_df": None,
                "residuals": {},
                "obs_used": {},
                "best_set": None,
                "meta": {"version": vkey, "reference_version": ref_v, "prior_version": v0},
            }
    
        # 2) extent filter (unified grid)
        obs_ext, obs_drop = self.filter_observations_to_extent(
            obs_raw,
            version=vkey,
            grid_mode="unified",
            margin_deg=float(margin_deg),
            return_dropped=True,
        )
        if obs_ext is None or getattr(obs_ext, "empty", False):
            return {
                "ok": False,
                "note": "All CDI observations outside unified extent after filter_observations_to_extent().",
                "results_df": None,
                "residuals": {},
                "obs_used": {},
                "best_set": None,
                "meta": {
                    "version": vkey,
                    "reference_version": ref_v,
                    "prior_version": v0,
                    "n_raw": int(len(obs_raw)),
                },
            }
    
        # 3) reference MMI mean grid for sampling
        lon2d, lat2d = self._get_unified_grid()
        _, _, ref_mean = self._get_prior_mean_unified(ref_v, "MMI")
        if ref_mean is None:
            return {
                "ok": False,
                "note": f"Reference mean missing for reference_version={ref_v} imt=MMI.",
                "results_df": None,
                "residuals": {},
                "obs_used": {},
                "best_set": None,
                "meta": {"version": vkey, "reference_version": ref_v, "prior_version": v0},
            }
        ref_mean = np.asarray(ref_mean, dtype=float)
    
        rows = []
        residuals = {}
        obs_used = {}
    
        n_raw = int(len(obs_raw))
        n_ext = int(len(obs_ext))
    
        # 4) run each set through _cdi_condition (vs v0) then compare to ref at obs
        for name, prof in profiles_5sets.items():
            df = obs_ext.copy()
    
            try:
                # IMPORTANT: _cdi_condition takes obs_df as positional
                # and does NOT accept margin_deg
                df2 = self._cdi_condition(
                    df,
                    version=v0,            # fixed prior for residual-based conditioning
                    debug=bool(debug),
                    **prof,
                )
            except TypeError as e:
                df2 = None
                err = f"TypeError: {e}"
            except Exception as e:
                df2 = None
                err = str(e)
    
            if df2 is None or getattr(df2, "empty", False):
                rows.append(
                    dict(
                        set_name=name,
                        ok=False,
                        n_kept=0,
                        retained_frac=np.nan,
                        rmse_ref=np.nan,
                        mae_ref=np.nan,
                        bias_ref=np.nan,
                        std_ref=np.nan,
                        note=(err if "err" in locals() else "empty after conditioning"),
                    )
                )
                # clear err for next loop iteration
                if "err" in locals():
                    del err
                continue
    
            # Compare CDI values to reference MMI at obs locations
            ref_at_obs = self._rk_sample_grid_nn(
                lon2d, lat2d, ref_mean,
                df2["lon"].values, df2["lat"].values
            )
            y = df2["value"].to_numpy(dtype=float)
            r = y - ref_at_obs.astype(float)
    
            rmse = float(np.sqrt(np.nanmean(r**2))) if len(r) else np.nan
            mae = float(np.nanmean(np.abs(r))) if len(r) else np.nan
            bias = float(np.nanmean(r)) if len(r) else np.nan
            std = float(np.nanstd(r)) if len(r) else np.nan
    
            residuals[name] = r
            obs_used[name] = df2
    
            nk = int(len(df2))
            rows.append(
                dict(
                    set_name=name,
                    ok=True,
                    n_kept=nk,
                    retained_frac=(nk / max(n_raw, 1)),
                    rmse_ref=rmse,
                    mae_ref=mae,
                    bias_ref=bias,
                    std_ref=std,
                    note="ok",
                )
            )
    
        results_df = pd.DataFrame(rows)
    
        best_name = None
        if results_df is not None and (not results_df.empty) and (results_df["ok"] == True).any():
            best_row = results_df[results_df["ok"] == True].sort_values("rmse_ref", ascending=True).iloc[0]
            best_name = str(best_row["set_name"])
    
        return {
            "ok": True,
            "results_df": results_df,
            "residuals": residuals,
            "obs_used": obs_used,
            "best_set": best_name,
            "meta": {
                "version": vkey,
                "reference_version": ref_v,
                "prior_version": v0,
                "margin_deg": float(margin_deg),
                "n_raw": n_raw,
                "n_extent": n_ext,
                "n_dropped_extent": int(len(obs_drop)) if obs_drop is not None else None,
            },
        }








    # ======================================================================================
    # PATCH: Ensemble / Particle-style update (moment-matched to Bayes posterior moments)
    # - Paste INSIDE the SHAKEuq class (near the end of the class body).
    #
    # Design goals:
    #   * Zero disruption to existing pipelines (no changes to existing methods/keys).
    #   * Reuse your existing, validated local Bayes fusion to get posterior mean/sigma.
    #   * Build an ensemble by sampling a smooth prior field and then "moment-matching"
    #     to the Bayes posterior (stable, fast, avoids huge matrix ops).
    #   * Store results under: uq_state["versions"][vkey]["updates"][update_key]
    #   * Provide an audit plot function for this method.
    #
    # Note:
    #   * This is an "ensemble wrapper" around your Bayes local precision fusion.
    #   * It produces stable posterior maps + quantiles and is ideal for later
    #     master tracking/comparison utilities.
    # ======================================================================================
    
    def run_ensemble_update(
        self,
        *,
        version,
        mode,
        update_key: str = None,
        margin_deg: float = 0.0,
        # Bayes-like local update controls (used to compute posterior mean/sigma moments)
        update_radius_km: float = 30.0,
        kernel: str = "gaussian",
        kernel_scale_km: float = 12.0,
        neighbor_k: Optional[int] = 50,
        default_obs_sigma: float = 0.5,
        # sigma scaling (trust) forwarded to obs builder (same as Bayes)
        sigma_scale_instr: float = 1.0,
        sigma_scale_dyfi: float = 1.0,
        sigma_scale_cdi: float = 1.0,
        # Ensemble controls
        n_ens: int = 100,
        seed: Optional[int] = 0,
        corr_model: str = "gaussian",     # "gaussian" | "exp"  (controls smoothness filter)
        corr_range_km: float = 50.0,      # approximate correlation range (km)
        corr_nugget: float = 0.05,        # 0..1 fraction of white noise mixed into smooth noise
        # Output controls
        quantiles: Tuple[float, float, float] = (0.05, 0.5, 0.95),
        store_obs_used: bool = True,
        debug: bool = False,
        # CDI conditioning knobs (pass-through to _bayes_build_obs_for_mode)
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
    ) -> Dict[str, Any]:
        """
        Ensemble update (particle-style) for one version/mode.
    
        Strategy (robust + cheap):
          1) Use your existing local Bayes fusion to compute posterior mean/sigma on the unified grid.
          2) Sample a smooth prior ensemble around the prior mean with spatial correlation.
          3) "Moment-match" the ensemble anomalies to the posterior sigma and recenter to posterior mean.
             This yields stable ensembles with correct first/second moments (approx conditioning).
    
        Stores a pack under:
          uq_state["versions"][vkey]["updates"][update_key]
        """
        import time
        import numpy as np
        import pandas as pd
    
        if not isinstance(self.uq_state, dict) or not (self.uq_state.get("versions") or {}):
            raise RuntimeError("uq_state not initialized. Run uq_build_dataset() first.")
    
        vkey = _norm_version(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey)
        if vpack is None:
            raise KeyError(f"Version not in uq_state: {vkey}")
    
        lon2d, lat2d = self._get_unified_grid()
    
        # fixed prior version (v0)
        v0 = self._bayes_get_prior_version_key()
    
        mode_s = str(mode).lower().strip()
        if update_key is None:
            update_key = f"ens__{mode_s}__v{vkey}"
        update_key = str(update_key)
    
        # ensure storage
        if "updates" not in vpack or not isinstance(vpack.get("updates"), dict):
            vpack["updates"] = {}
    
        t0 = time.time()
    
        # --- build obs using the same Bayes routing (IMT-isolated, CDI filtering if requested) ---
        imt_target, obs_used, obs_meta = self._bayes_build_obs_for_mode(
            version=vkey,
            mode=mode_s,
            margin_deg=float(margin_deg),
            cdi_enable_local_outlier=bool(cdi_enable_local_outlier),
            cdi_local_radius_km=float(cdi_local_radius_km),
            cdi_outlier_k=float(cdi_outlier_k),
            cdi_min_neighbors=int(cdi_min_neighbors),
            cdi_enable_clustering=bool(cdi_enable_clustering),
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
    
        # --- prior from unified stacks (already consistent with your Bayes implementation) ---
        _, _, prior_mean = self._get_prior_mean_unified(v0, imt_target)
        _, _, prior_sigma = self._get_prior_sigma_unified(v0, imt_target)
        if prior_mean is None or prior_sigma is None:
            raise RuntimeError(f"Missing prior mean/sigma for v0={v0} imt={imt_target}. Check unified stacks.")
        prior_mean = np.asarray(prior_mean, dtype=float)
        prior_sigma = np.asarray(prior_sigma, dtype=float)
        if prior_mean.shape != lon2d.shape or prior_sigma.shape != lon2d.shape:
            raise RuntimeError(f"Prior shapes do not match unified grid for imt={imt_target}.")
    
        # --- compute posterior moments using existing Bayes local fusion (stable reference) ---
        post_mean, post_sigma, bayes_audit = self._bayes_local_precision_fusion_grid(
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
        post_mean = np.asarray(post_mean, dtype=float)
        post_sigma = np.asarray(post_sigma, dtype=float)
        if post_mean.shape != lon2d.shape or post_sigma.shape != lon2d.shape:
            raise RuntimeError("Posterior mean/sigma returned by Bayes fusion do not match unified grid.")
    
        # --- sample prior ensemble (working space) ---
        ens_prior = self._ens_sample_prior_ensemble(
            prior_mean=prior_mean,
            prior_sigma=prior_sigma,
            lon2d=lon2d,
            lat2d=lat2d,
            n_ens=int(n_ens),
            seed=seed,
            corr_model=str(corr_model),
            corr_range_km=float(corr_range_km),
            corr_nugget=float(corr_nugget),
        )
    
        # --- condition ensemble by moment matching to posterior mean/sigma (robust, fast) ---
        ens_post = self._ens_moment_match_to_posterior(
            ens_prior=ens_prior,
            prior_mean=prior_mean,
            prior_sigma=prior_sigma,
            post_mean=post_mean,
            post_sigma=post_sigma,
        )
    
        # --- summarize ensemble in working space ---
        summ = self._ens_summarize_ensemble(ens_post, quantiles=quantiles)
    
        # --- metrics at obs points (posterior vs prior) in working space ---
        metrics = {}
        try:
            if obs_used is not None and isinstance(obs_used, pd.DataFrame) and len(obs_used) > 0:
                lon_obs = obs_used["lon"].to_numpy(dtype=float)
                lat_obs = obs_used["lat"].to_numpy(dtype=float)
                y_obs = obs_used["value"].to_numpy(dtype=float)
                # nearest-neighbor sampling on unified grid
                y_prior = self._rk_sample_grid_nn(lon2d, lat2d, prior_mean, lon_obs, lat_obs)
                y_post = self._rk_sample_grid_nn(lon2d, lat2d, summ["mean_grid"], lon_obs, lat_obs)
    
                def _rmse(a, b):
                    a = np.asarray(a, dtype=float)
                    b = np.asarray(b, dtype=float)
                    m = np.isfinite(a) & np.isfinite(b)
                    if not np.any(m):
                        return np.nan
                    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
    
                def _mae(a, b):
                    a = np.asarray(a, dtype=float)
                    b = np.asarray(b, dtype=float)
                    m = np.isfinite(a) & np.isfinite(b)
                    if not np.any(m):
                        return np.nan
                    return float(np.mean(np.abs(a[m] - b[m])))
    
                metrics = {
                    "n_obs": int(len(obs_used)),
                    "rmse_prior": _rmse(y_prior, y_obs),
                    "rmse_post": _rmse(y_post, y_obs),
                    "mae_prior": _mae(y_prior, y_obs),
                    "mae_post": _mae(y_post, y_obs),
                }
        except Exception as e:
            metrics = {"error": f"{type(e).__name__}: {e}"}
    
        pack = {
            "meta": {
                "method": "ensemble_moment_match",
                "version": vkey,
                "imt": imt_target,
                "mode": mode_s,
                "prior_version": v0,
                "n_ens": int(n_ens),
                "seed": seed,
                "corr_model": str(corr_model),
                "corr_range_km": float(corr_range_km),
                "corr_nugget": float(corr_nugget),
                "update_radius_km": float(update_radius_km),
                "kernel": str(kernel),
                "kernel_scale_km": float(kernel_scale_km),
                "neighbor_k": neighbor_k,
                "default_obs_sigma": float(default_obs_sigma),
                "sigma_scale_instr": float(sigma_scale_instr),
                "sigma_scale_dyfi": float(sigma_scale_dyfi),
                "sigma_scale_cdi": float(sigma_scale_cdi),
                "elapsed_s": float(time.time() - t0),
            },
            "prior": {
                "version": v0,
                "mean_grid": prior_mean,
                "sigma_grid": prior_sigma,
            },
            # reference posterior moments (from Bayes fusion)
            "posterior_moments": {
                "mean_grid": post_mean,
                "sigma_grid": post_sigma,
                "audit": bayes_audit,
            },
            # ensemble posterior (moment-matched)
            "posterior": {
                "mean_grid": summ["mean_grid"],
                "sigma_grid": summ["sigma_grid"],
                "quantiles": {k: v for k, v in summ.items() if k.startswith("q")},
            },
            "metrics": metrics,
            "obs_meta": obs_meta,
        }
    
        if store_obs_used:
            pack["obs_used"] = obs_used
    
        vpack["updates"][update_key] = pack
    
        if self.verbose:
            print(f"[SHAKEuq] ensemble update stored: updates['{update_key}']  imt={imt_target}  n_ens={n_ens}  n_obs={int(metrics.get('n_obs', 0) or 0)}")
    
        return pack
    
    
    def _ens_sample_prior_ensemble(
        self,
        *,
        prior_mean: np.ndarray,
        prior_sigma: np.ndarray,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        n_ens: int = 100,
        seed: Optional[int] = 0,
        corr_model: str = "gaussian",
        corr_range_km: float = 50.0,
        corr_nugget: float = 0.05,
    ) -> np.ndarray:
        """
        Sample a smooth prior ensemble around prior_mean with spatial correlation.
    
        Implementation:
          - Draw white noise.
          - Apply an FFT-domain low-pass filter (gaussian/exp-like) based on corr_range_km.
          - Mix in a nugget fraction of white noise.
          - Scale by prior_sigma and add to prior_mean.
    
        Returns:
          ens array (n_ens, nlat, nlon) in the SAME working space as prior_mean/prior_sigma.
        """
        import numpy as np
    
        mu = np.asarray(prior_mean, dtype=float)
        sig = np.asarray(prior_sigma, dtype=float)
        nlat, nlon = mu.shape
    
        rs = np.random.RandomState(None if seed is None else int(seed))
    
        # estimate grid spacing in km (robust median of neighbor diffs)
        def _nanmedian(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            return float(np.median(x)) if x.size else np.nan
    
        # lon spacing varies with latitude; use local median step converted to km
        # Use helper if available; otherwise approximate 111 km per degree and cos(lat) for lon.
        try:
            km_per_deg = float(self._bayes_km_per_deg())
        except Exception:
            km_per_deg = 111.32
    
        # dy from lat2d
        dlat = np.diff(np.asarray(lat2d, dtype=float), axis=0)
        dlat_med = _nanmedian(np.abs(dlat))
        dy_km = km_per_deg * dlat_med if np.isfinite(dlat_med) and dlat_med > 0 else np.nan
    
        # dx from lon2d (correct by cos(lat) with median latitude)
        dlon = np.diff(np.asarray(lon2d, dtype=float), axis=1)
        dlon_med = _nanmedian(np.abs(dlon))
        lat_med = _nanmedian(np.asarray(lat2d, dtype=float))
        coslat = float(np.cos(np.deg2rad(lat_med))) if np.isfinite(lat_med) else 1.0
        dx_km = km_per_deg * coslat * dlon_med if np.isfinite(dlon_med) and dlon_med > 0 else np.nan
    
        # fallback if spacing cannot be inferred
        if not np.isfinite(dx_km) or dx_km <= 0:
            dx_km = 1.0
        if not np.isfinite(dy_km) or dy_km <= 0:
            dy_km = 1.0
    
        # convert desired correlation range to smoothing scale in grid cells
        # keep it bounded to avoid degenerate filters
        s_x = max(1.0, float(corr_range_km) / float(dx_km))
        s_y = max(1.0, float(corr_range_km) / float(dy_km))
    
        # build frequency grids
        kx = np.fft.fftfreq(nlon)
        ky = np.fft.fftfreq(nlat)
        KX, KY = np.meshgrid(kx, ky)
    
        cm = str(corr_model).lower().strip()
        if cm in ("exp", "exponential"):
            # "exp-like" low-pass in frequency domain (heavier tails than gaussian)
            # use 1 / (1 + a*k^2) form as a stable approximation
            a = (2.0 * np.pi) ** 2
            filt = 1.0 / (1.0 + a * ((KX * s_x) ** 2 + (KY * s_y) ** 2))
        else:
            # gaussian low-pass
            filt = np.exp(-0.5 * ((2.0 * np.pi * KX * s_x) ** 2 + (2.0 * np.pi * KY * s_y) ** 2))
    
        # nugget mix fraction (clamped)
        nug = float(corr_nugget)
        if not np.isfinite(nug):
            nug = 0.0
        nug = min(max(nug, 0.0), 1.0)
        a_s = np.sqrt(max(0.0, 1.0 - nug))
        a_w = np.sqrt(max(0.0, nug))
    
        ens = np.empty((int(n_ens), nlat, nlon), dtype=float)
    
        for i in range(int(n_ens)):
            white = rs.normal(size=(nlat, nlon)).astype(float)
    
            F = np.fft.fft2(white)
            smooth = np.fft.ifft2(F * filt).real
    
            # standardize smooth to unit variance (robust)
            sstd = float(np.std(smooth)) if np.isfinite(np.std(smooth)) and np.std(smooth) > 0 else 1.0
            smooth = smooth / sstd
    
            # combine smooth + nugget white and scale by sigma
            z = a_s * smooth + a_w * white
            # standardize combined too
            zstd = float(np.std(z)) if np.isfinite(np.std(z)) and np.std(z) > 0 else 1.0
            z = z / zstd
    
            ens[i] = mu + sig * z
    
        return ens
    
    
    def _ens_moment_match_to_posterior(
        self,
        *,
        ens_prior: np.ndarray,
        prior_mean: np.ndarray,
        prior_sigma: np.ndarray,
        post_mean: np.ndarray,
        post_sigma: np.ndarray,
        sigma_eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Moment-match prior ensemble to posterior mean/sigma.
    
        X_post = post_mean + (post_sigma / prior_sigma) * (X_prior - prior_mean)
    
        This preserves the prior spatial anomaly structure while enforcing the posterior moments.
        """
        import numpy as np
    
        X = np.asarray(ens_prior, dtype=float)
        mu0 = np.asarray(prior_mean, dtype=float)
        s0 = np.asarray(prior_sigma, dtype=float)
        mu1 = np.asarray(post_mean, dtype=float)
        s1 = np.asarray(post_sigma, dtype=float)
    
        # avoid divide by zero
        denom = np.maximum(np.abs(s0), float(sigma_eps))
        scale = s1 / denom
    
        # broadcast scale to ensemble
        A = X - mu0[None, :, :]
        Xmm = mu1[None, :, :] + A * scale[None, :, :]
    
        # recenter exactly to posterior mean (numeric stability)
        mean_now = np.nanmean(Xmm, axis=0)
        Xmm = Xmm + (mu1 - mean_now)[None, :, :]
    
        return Xmm
    
    
    def _ens_summarize_ensemble(
        self,
        ens: np.ndarray,
        *,
        quantiles: Tuple[float, float, float] = (0.05, 0.5, 0.95),
    ) -> Dict[str, np.ndarray]:
        """
        Summarize an ensemble into mean/sigma and requested quantiles.
    
        Returns dict with:
          mean_grid, sigma_grid, q05_grid, q50_grid, q95_grid (names follow quantiles)
        """
        import numpy as np
    
        X = np.asarray(ens, dtype=float)
        out: Dict[str, np.ndarray] = {}
    
        out["mean_grid"] = np.nanmean(X, axis=0)
        out["sigma_grid"] = np.nanstd(X, axis=0)
    
        qs = list(quantiles or [])
        for q in qs:
            qf = float(q)
            qname = f"q{int(round(qf * 100)):02d}_grid"
            out[qname] = np.nanquantile(X, qf, axis=0)
    
        return out
    
    
    def plot_ensemble_audit(
        self,
        *,
        version,
        update_key: str,
        imt: Optional[str] = None,
        margin_deg: float = 0.0,
        show: bool = True,
        savepath: Optional[str] = None,
        dpi: int = 160,
        figsize: Tuple[float, float] = (13.0, 8.0),
        overlay_obs: bool = True,
        obs_marker_size: float = 10.0,
    ) -> Any:
        """
        Audit plot for an ensemble update pack stored in:
          uq_state["versions"][vkey]["updates"][update_key]
    
        Plots (simple, stable):
          - Published (unified) mean for target version
          - Prior mean
          - Posterior mean (ensemble mean)
          - Posterior sigma (ensemble std)
          - Difference: posterior - published
    
        Notes:
          - This is a lightweight audit (consistent with existing Bayes/RK style).
          - Uses unified grids, so shapes always align.
        """
        import numpy as np
        import matplotlib.pyplot as plt
    
        vkey = _norm_version(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey)
        if vpack is None:
            raise KeyError(f"Version not in uq_state: {vkey}")
        up = (vpack.get("updates") or {}).get(update_key)
        if up is None:
            raise KeyError(f"updates['{update_key}'] missing for version {vkey}")
    
        lon2d, lat2d = self._get_unified_grid()
    
        imt_target = str(imt).upper().strip() if imt is not None else str(up.get("meta", {}).get("imt", "")).upper().strip()
        if not imt_target:
            raise ValueError("imt not provided and not found in pack meta.")
    
        # Published mean on unified grid for this version (from unified stack)
        uni = self.uq_state.get("unified") or {}
        vkeys = list(uni.get("version_keys") or [])
        if vkey not in vkeys:
            raise KeyError(f"Version {vkey} not in unified.version_keys.")
        idx = vkeys.index(vkey)
        published_work = np.asarray((uni.get("fields") or {}).get(imt_target, [])[idx], dtype=float)
    
        prior_work = np.asarray((up.get("prior") or {}).get("mean_grid"), dtype=float)
        post_work = np.asarray((up.get("posterior") or {}).get("mean_grid"), dtype=float)
        post_sig_work = np.asarray((up.get("posterior") or {}).get("sigma_grid"), dtype=float)
    
        # Convert for plotting if needed (consistent with your Bayes conventions)
        # (If working space is linear, this is identity.)
        try:
            published_plot = self._bayes_from_working(imt_target, published_work)
            prior_plot = self._bayes_from_working(imt_target, prior_work)
            post_plot = self._bayes_from_working(imt_target, post_work)
        except Exception:
            published_plot = published_work
            prior_plot = prior_work
            post_plot = post_work
    
        # sigma: for log working space, the sigma maps are already in that working space
        # We plot sigma as-is (working sigma). This matches your existing Bayes audit style.
        post_sig_plot = post_sig_work
    
        extent = self._get_grid_extent(vkey, grid_mode="unified", margin_deg=float(margin_deg))
    
        fig, ax = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
        ax = np.asarray(ax)
    
        def _imshow(a, Z, title):
            im = a.imshow(Z, extent=extent, origin="lower", aspect="auto")
            a.set_title(title)
            a.set_xlabel("Lon")
            a.set_ylabel("Lat")
            plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    
        _imshow(ax[0, 0], published_plot, f"Published (unified) mean — v{vkey} — {imt_target}")
        _imshow(ax[0, 1], prior_plot, f"Prior mean (v0) — {imt_target}")
        _imshow(ax[0, 2], post_plot, f"Posterior mean (ensemble) — {imt_target}")
        _imshow(ax[1, 0], post_sig_plot, f"Posterior sigma (ensemble std) — {imt_target}")
    
        diff_pub = post_plot - published_plot
        _imshow(ax[1, 1], diff_pub, "Posterior − Published")
    
        diff_prior = post_plot - prior_plot
        _imshow(ax[1, 2], diff_prior, "Posterior − Prior")
    
        # overlay obs
        if overlay_obs and ("obs_used" in up) and up["obs_used"] is not None:
            try:
                odf = up["obs_used"]
                ax[0, 2].scatter(odf["lon"], odf["lat"], s=float(obs_marker_size), marker="o")
                ax[1, 1].scatter(odf["lon"], odf["lat"], s=float(obs_marker_size), marker="o")
                ax[1, 2].scatter(odf["lon"], odf["lat"], s=float(obs_marker_size), marker="o")
            except Exception:
                pass
    
        fig.tight_layout()
    
        if savepath:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    
        if show:
            plt.show()
    
        return fig






     # ==========================================================================================
    # Targets / decay diagnostics (versions or time-after-event axis) + point/area sampling
    # - Two plotting functions:
    #   1) uq_plot_targets_decay        : convenience (will compute missing method outputs)
    #   2) uq_plot_targets_decay_fast   : strict/fast (plots ONLY what already exists)
    # - Truth lines are ONLY drawn for what="mean" (never for sigma).
    #
    # SAFETY FIXES (this patch):
    # - Prevent kwarg collisions when auto-computing (e.g., passing mode twice).
    # - Refresh vpack after compute so newly stored results are visible.
    # - Robust grid extraction (fallback keys for mean/sigma in stored packs).
    # - Optional default mode inference if user forgot to provide it (MMI vs PGA).
    # ==========================================================================================

    def _uq__norm_version(self, v):
        """Normalize version to 3-digit string ('001', '012', ...)."""
        try:
            if isinstance(v, str):
                s = v.strip()
                if s.isdigit():
                    return f"{int(s):03d}"
                if len(s) == 3 and s[0].isdigit():
                    return s
                return s
            return f"{int(v):03d}"
        except Exception:
            return str(v)

    def _uq__get_unified_grid_safe(self):
        """Return (lon2d, lat2d) from existing helpers or uq_state."""
        try:
            if hasattr(self, "_get_unified_grid"):
                return self._get_unified_grid()
        except Exception:
            pass

        u = (self.uq_state or {}).get("unified") or {}
        lon2d = u.get("lon2d", None)
        lat2d = u.get("lat2d", None)
        if lon2d is None or lat2d is None:
            lon2d = u.get("lon", u.get("lons", None))
            lat2d = u.get("lat", u.get("lats", None))
        if lon2d is None or lat2d is None:
            raise RuntimeError(
                "Unified grid not found. Expected _get_unified_grid() or uq_state['unified']['lon2d'/'lat2d']."
            )
        return lon2d, lat2d

    def _uq__axis_versions_or_time(self, version_list, *, x_axis="auto"):
        """Return (vkeys, x, x_labels, tae_hours)."""
        import numpy as np

        if version_list is None:
            vkeys = list(getattr(self, "version_list", []) or [])
            vkeys = [self._uq__norm_version(v) for v in vkeys]
        else:
            vkeys = [self._uq__norm_version(v) for v in version_list]

        if not vkeys:
            return [], np.array([]), [], []

        tae = []
        try:
            sanity = (self.uq_state or {}).get("sanity")
            if sanity is not None:
                st = sanity.set_index("version")
                for vk in vkeys:
                    tae.append(float(st.loc[vk, "TAE_hours"]))
            else:
                tae = [np.nan] * len(vkeys)
        except Exception:
            tae = [np.nan] * len(vkeys)

        x_axis_s = str(x_axis).lower().strip()
        if x_axis_s == "auto":
            x_axis_s = "time" if all(np.isfinite(t) for t in tae) else "versions"

        if x_axis_s == "time":
            x = np.array([t if np.isfinite(t) else np.nan for t in tae], dtype=float)
            x_labels = [f"{t:.2g}" if np.isfinite(t) else "NA" for t in x]
        else:
            x = np.array([int(vk) for vk in vkeys], dtype=float)
            x_labels = [str(int(vk)) for vk in vkeys]

        return vkeys, x, x_labels, tae

    def _uq__nearest_grid_sample(self, lon2d, lat2d, field2d, *, lon, lat):
        """Nearest-neighbor sample of field2d at (lon, lat) on lon2d/lat2d."""
        import numpy as np
        if field2d is None:
            return np.nan
        try:
            d2 = (lon2d - float(lon)) ** 2 + (lat2d - float(lat)) ** 2
            j = int(np.nanargmin(d2))
            return float(np.ravel(field2d)[j])
        except Exception:
            return np.nan

    def _uq__haversine_km(self, lon1, lat1, lon2, lat2):
        """Vectorized haversine distance (km). lon/lat in degrees."""
        import numpy as np
        R = 6371.0
        lon1r = np.deg2rad(lon1)
        lat1r = np.deg2rad(lat1)
        lon2r = np.deg2rad(lon2)
        lat2r = np.deg2rad(lat2)
        dlon = lon2r - lon1r
        dlat = lat2r - lat1r
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return R * c

    def _uq__circle_stats(self, lon2d, lat2d, field2d, *, lon, lat, radius_km=30.0):
        """Return (mean, median, min, max, n) of field2d within radius_km around (lon, lat)."""
        import numpy as np
        if field2d is None:
            return np.nan, np.nan, np.nan, np.nan, 0
        try:
            d = self._uq__haversine_km(lon2d, lat2d, float(lon), float(lat))
            m = (d <= float(radius_km))
            if not np.any(m):
                return np.nan, np.nan, np.nan, np.nan, 0
            vals = np.asarray(field2d, dtype=float)[m]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return np.nan, np.nan, np.nan, np.nan, 0
            return float(np.mean(vals)), float(np.median(vals)), float(np.min(vals)), float(np.max(vals)), int(vals.size)
        except Exception:
            return np.nan, np.nan, np.nan, np.nan, 0

    def _uq__get_published_unified_grids(self, vkey, imt_u):
        """Return (mean2d, sigma2d) for the published (raw) ShakeMap on unified grid."""
        imt = str(imt_u).upper().strip()
        u = (self.uq_state or {}).get("unified") or {}
        vkeys = list(u.get("version_keys") or [])
        if vkey not in vkeys:
            return None, None
        i = vkeys.index(vkey)

        mean2d = None
        sigma2d = None

        try:
            mean_stack = (u.get("fields") or {}).get(imt)
            mean2d = None if mean_stack is None else mean_stack[i]
        except Exception:
            mean2d = None

        try:
            if hasattr(self, "_bayes_sigma_key_for_imt"):
                skey = self._bayes_sigma_key_for_imt(imt)
            else:
                skey = f"SIGMA_{imt}"
            sig_stack = (u.get("sigma") or {}).get(skey)
            sigma2d = None if sig_stack is None else sig_stack[i]
        except Exception:
            sigma2d = None

        return mean2d, sigma2d

    def _uq__resolve_bayes_key(self, vkey, *, mode=None, mode_key=None, imt_u=None):
        """Resolve bayes key for a target version. Returns key or None."""
        vpack = ((self.uq_state or {}).get("versions") or {}).get(vkey) or {}
        b = vpack.get("bayes") or {}
        if not isinstance(b, dict) or not b:
            return None

        if mode_key is not None:
            mk = str(mode_key)
            return mk if mk in b else None

        if mode is not None:
            mk = f"bayes__{str(mode).lower().strip()}__v{vkey}"
            return mk if mk in b else None

        for k, pack in b.items():
            try:
                if f"__v{vkey}" not in str(k):
                    continue
                if imt_u is not None:
                    imt = str(imt_u).upper().strip()
                    if str(((pack.get("meta") or {}).get("imt") or "")).upper().strip() != imt:
                        continue
                return str(k)
            except Exception:
                continue

        return str(next(iter(b.keys())))

    def _uq__resolve_update_key(self, vkey, *, prefix, key=None):
        """Resolve updates[prefix*] key for a version."""
        vpack = ((self.uq_state or {}).get("versions") or {}).get(vkey) or {}
        upd = vpack.get("updates") or {}
        if not isinstance(upd, dict) or not upd:
            return None

        if key is not None:
            kk = str(key)
            return kk if kk in upd else None

        pref = str(prefix)
        for k in upd.keys():
            ks = str(k)
            if ks.startswith(pref) and f"__v{vkey}" in ks:
                return ks
        for k in upd.keys():
            ks = str(k)
            if ks.startswith(pref):
                return ks
        return None

    def _uq__default_mode_for_imt(self, imt_u, *, family="bayes"):
        """If user forgot mode, infer a safe default by IMT."""
        imt = str(imt_u).upper().strip()
        fam = str(family).lower().strip()
        if imt == "PGA":
            return "pga_instr"
        # for MMI default to dyfi; user can override to CDI modes
        if fam in ("rk", "residual_kriging"):
            return "mmi_dyfi"
        return "mmi_dyfi"

    def _uq__strip_kwargs(self, d, strip_keys):
        """Return a copy of dict d without keys in strip_keys."""
        if not isinstance(d, dict) or not d:
            return {}
        return {k: v for k, v in d.items() if k not in set(strip_keys)}

    def _uq__extract_mean_sigma_from_pack(self, pack):
        """Robustly extract (mean2d, sigma2d) from different stored pack shapes."""
        if not isinstance(pack, dict) or not pack:
            return None, None

        # common patterns:
        # - bayes: pack["posterior"]["mean_grid"/"sigma_grid"]
        # - rk/ens: pack["mean_post"/"sigma_post"] OR pack["posterior"]["mean_grid"/"sigma_grid"]
        post = pack.get("posterior")
        if isinstance(post, dict):
            mu = post.get("mean_grid", post.get("mean_post", post.get("mean")))
            sig = post.get("sigma_grid", post.get("sigma_post", post.get("sigma")))
            if mu is not None or sig is not None:
                return mu, sig

        mu = pack.get("mean_post", pack.get("mean_grid", pack.get("mean")))
        sig = pack.get("sigma_post", pack.get("sigma_grid", pack.get("sigma")))
        return mu, sig

    def _uq__get_method_grids(
        self,
        *,
        vkey,
        method,
        imt,
        compute_if_missing=False,
        method_kwargs=None,
        debug=False,
    ):
        """Return (mean2d, sigma2d, meta_dict, status)."""
        imt_u = str(imt).upper().strip()
        m = str(method).lower().strip()
        mk = {"raw": "ShakeMap", "shakemap": "ShakeMap", "shakemap ": "ShakeMap", "ShakeMap": "ShakeMap",
              "bayes": "bayes",
              "rk": "rk", "residual_kriging": "rk",
              "ens": "ens", "ensemble": "ens"}.get(m, method)

        meta = {"method": mk, "imt": imt_u, "vkey": vkey}
        method_kwargs = method_kwargs or {}

        # published raw
        if str(mk).lower().strip() == "shakemap":
            mu, sig = self._uq__get_published_unified_grids(vkey, imt_u)
            return mu, sig, meta, ("ok" if (mu is not None or sig is not None) else "missing")

        if not isinstance(self.uq_state, dict):
            return None, None, meta, "missing"

        # ---------------------
        # Bayes
        # ---------------------
        if str(mk).lower().strip() == "bayes":
            # allow either: mode OR mode_key
            mode = method_kwargs.get("mode", None)
            mode_key = method_kwargs.get("mode_key", None)

            # safe defaults if user forgot mode (only used for compute)
            if mode is None and mode_key is None:
                mode = self._uq__default_mode_for_imt(imt_u, family="bayes")

            # resolve existing
            k = self._uq__resolve_bayes_key(vkey, mode=mode, mode_key=mode_key, imt_u=imt_u)

            # compute if missing
            if k is None and compute_if_missing and hasattr(self, "run_bayes_update"):
                try:
                    # critical: avoid passing mode twice
                    run_kwargs = self._uq__strip_kwargs(method_kwargs, ("mode", "mode_key", "mode_fallback"))
                    self.run_bayes_update(
                        version=vkey,
                        mode=str(mode).lower().strip() if mode is not None else self._uq__default_mode_for_imt(imt_u, family="bayes"),
                        mode_key=mode_key,
                        **run_kwargs,
                    )
                except Exception as e:
                    if debug:
                        print(f"[uq_plot_targets_decay] bayes compute failed (v={vkey}): {repr(e)}")

                # refresh and resolve again AFTER compute
                k = self._uq__resolve_bayes_key(vkey, mode=mode, mode_key=mode_key, imt_u=imt_u)

            if k is None:
                return None, None, meta, ("compute_fail" if compute_if_missing else "missing")

            vpack = ((self.uq_state.get("versions") or {}).get(vkey) or {})
            pack = (vpack.get("bayes") or {}).get(k) or {}
            meta.update({"key": k, "meta_pack": pack.get("meta") or {}})

            mu, sig = self._uq__extract_mean_sigma_from_pack(pack)
            return mu, sig, meta, ("ok" if (mu is not None or sig is not None) else "missing")

        # ---------------------
        # Residual Kriging
        # ---------------------
        if str(mk).lower().strip() == "rk":
            key = method_kwargs.get("key", None)

            # if key is not explicitly provided, resolve by prefix
            k = self._uq__resolve_update_key(vkey, prefix="rk__", key=key)

            if k is None and compute_if_missing and hasattr(self, "run_residual_kriging_update"):
                try:
                    # ensure mode exists for compute
                    if "mode" not in method_kwargs or method_kwargs.get("mode") is None:
                        method_kwargs = dict(method_kwargs)
                        method_kwargs["mode"] = self._uq__default_mode_for_imt(imt_u, family="rk")

                    run_kwargs = self._uq__strip_kwargs(method_kwargs, ("key",))
                    self.run_residual_kriging_update(version=vkey, key=key, **run_kwargs)
                except Exception as e:
                    if debug:
                        print(f"[uq_plot_targets_decay] rk compute failed (v={vkey}): {repr(e)}")

                # refresh and resolve again AFTER compute
                k = self._uq__resolve_update_key(vkey, prefix="rk__", key=key)

            if k is None:
                return None, None, meta, ("compute_fail" if compute_if_missing else "missing")

            vpack = ((self.uq_state.get("versions") or {}).get(vkey) or {})
            pack = (vpack.get("updates") or {}).get(k) or {}
            meta.update({"key": k, "meta_pack": pack.get("meta") or {}})

            mu, sig = self._uq__extract_mean_sigma_from_pack(pack)
            return mu, sig, meta, ("ok" if (mu is not None or sig is not None) else "missing")

        # ---------------------
        # Ensemble
        # ---------------------
        if str(mk).lower().strip() == "ens":
            update_key = method_kwargs.get("update_key", None)
            k = self._uq__resolve_update_key(vkey, prefix="ens__", key=update_key)

            if k is None and compute_if_missing and hasattr(self, "run_ensemble_update"):
                try:
                    run_kwargs = self._uq__strip_kwargs(method_kwargs, ("update_key",))
                    self.run_ensemble_update(version=vkey, update_key=update_key, **run_kwargs)
                except Exception as e:
                    if debug:
                        print(f"[uq_plot_targets_decay] ens compute failed (v={vkey}): {repr(e)}")

                k = self._uq__resolve_update_key(vkey, prefix="ens__", key=update_key)

            if k is None:
                return None, None, meta, ("compute_fail" if compute_if_missing else "missing")

            vpack = ((self.uq_state.get("versions") or {}).get(vkey) or {})
            pack = (vpack.get("updates") or {}).get(k) or {}
            meta.update({"key": k, "meta_pack": pack.get("meta") or {}})

            mu, sig = self._uq__extract_mean_sigma_from_pack(pack)
            return mu, sig, meta, ("ok" if (mu is not None or sig is not None) else "missing")

        return None, None, meta, "missing"


        
    # redacted 
    def _uq__plot_targets_decay_core(
        self,
        *,
        version_list=None,
        imt="MMI",
        points=None,
        global_targets=None,
        what="sigma",
        methods=("ShakeMap", "bayes"),
        method_kwargs=None,
        compute_if_missing=False,
        # axis controls
        x_axis="auto",  # "auto" | "versions" | "time"
        # point sampling mode
        point_mode="point",       # "point" | "area"
        area_radius_km=30.0,
        # plot style
        figsize=(10, 5),
        dpi=150,
        markers=None,
        linestyles=None,
        colors=None,
        linewidth=1.8,
        markersize=5.0,
        xrotation=0,
        title=None,
        # reference lines
        show_24h_line=True,
        line24_x=24.0,
        line24_kwargs=None,
        show_truth_line=True,
        truth_kwargs=None,
        # raw global/area summary overlays
        raw_show_summary=True,
        raw_fill_minmax=True,
        raw_fill_alpha=0.18,
        # saving + export
        save=False,
        save_path_prefix=None,
        save_formats=("png",),
        export_data=False,
        export_path=None,
        show=False,
        close_figs=True,
        debug=False,
        # strict reporting
        report_missing=True,
    ):
        """Core engine used by both convenience and fast variants."""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        if not isinstance(self.uq_state, dict) or not ((self.uq_state.get("versions") or {})):
            raise RuntimeError("uq_state not initialized. Run uq_build_dataset() first.")

        imt_u = str(imt).upper().strip()
        what_s = str(what).lower().strip()
        if what_s not in ("sigma", "mean"):
            raise ValueError("what must be 'sigma' or 'mean'.")

        vkeys, x, x_labels, tae_h = self._uq__axis_versions_or_time(version_list, x_axis=x_axis)
        if not vkeys:
            raise ValueError("No versions available to plot.")

        point_mode_s = str(point_mode).lower().strip()
        if point_mode_s not in ("point", "area"):
            raise ValueError("point_mode must be 'point' or 'area'.")

        # targets
        targets = []
        if points:
            for p in points:
                if not isinstance(p, dict):
                    continue
                pid = str(p.get("id", "P"))
                lat = p.get("lat", None)
                lon = p.get("lon", None)
                if lat is None or lon is None:
                    continue
                targets.append(("point", pid, float(lat), float(lon), dict(p)))

        if global_targets:
            for g in global_targets:
                targets.append(("global", str(g), None, None, {}))

        if not targets:
            targets = [("global", "GLOBAL", None, None, {})]

        # style maps
        methods = list(methods)
        markers = markers or {}
        linestyles = linestyles or {}
        colors = colors or {}
        method_kwargs = method_kwargs or {}
        line24_kwargs = line24_kwargs or {"color": "0.5", "linewidth": 1.5, "linestyle": "--", "alpha": 0.8, "label": "24h"}
        truth_kwargs = truth_kwargs or {"color": "r", "linewidth": 1.8, "linestyle": "-", "alpha": 0.9, "label": "Observed"}

        lon2d, lat2d = self._uq__get_unified_grid_safe()

        # 24h line placement
        x24 = None
        if show_24h_line:
            try:
                x_axis_s = str(x_axis).lower().strip()
                if x_axis_s == "time" or (x_axis_s == "auto" and all(np.isfinite(t) for t in tae_h)):
                    x24 = float(line24_x)
                else:
                    if any(np.isfinite(t) for t in tae_h):
                        tarr = np.array([t if np.isfinite(t) else np.nan for t in tae_h], dtype=float)
                        j = int(np.nanargmin(np.abs(tarr - float(line24_x))))
                        x24 = float(x[j])
            except Exception:
                x24 = None

        # missing report collector
        missing_map = {}  # (method)->list of versions

        rows = []
        figs = []

        for (ttype, tid, tlat, tlon, tmeta) in targets:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # raw summary caches for GLOBAL and POINT-AREA (ShakeMap only)
            raw_med_cache = []
            raw_min_cache = []
            raw_max_cache = []

            for meth in methods:
                yvals = []

                # robust retrieval of per-method kwargs (case-insensitive)
                mk = str(meth)
                mk_l = mk.lower().strip()
                mkw = (
                    method_kwargs.get(mk)
                    or method_kwargs.get(mk_l)
                    or method_kwargs.get(mk_l.replace(" ", ""))
                    or {}
                )

                for j, vk in enumerate(vkeys):
                    mu2d, sig2d, meta, status = self._uq__get_method_grids(
                        vkey=vk,
                        method=meth,
                        imt=imt_u,
                        compute_if_missing=compute_if_missing,
                        method_kwargs=mkw,
                        debug=debug,
                    )

                    if status != "ok" and str(meth).lower().strip() not in ("shakemap", "raw"):
                        missing_map.setdefault(str(meth), []).append(vk)

                    grid2d = sig2d if what_s == "sigma" else mu2d

                    # extraction
                    if ttype == "point":
                        if point_mode_s == "point":
                            val = self._uq__nearest_grid_sample(lon2d, lat2d, grid2d, lon=tlon, lat=tlat)
                        else:
                            vmean, vmed, vmin, vmax, n = self._uq__circle_stats(
                                lon2d, lat2d, grid2d, lon=tlon, lat=tlat, radius_km=area_radius_km
                            )
                            val = vmean

                            # cache raw-only envelope for POINT-AREA
                            if str(meth).lower().strip() in ("shakemap", "raw") and raw_show_summary:
                                raw_med_cache.append(vmed)
                                raw_min_cache.append(vmin)
                                raw_max_cache.append(vmax)
                    else:
                        # global
                        if grid2d is None:
                            val = np.nan
                        else:
                            vv = np.asarray(grid2d, dtype=float)
                            val = float(np.nanmean(vv))
                            if str(meth).lower().strip() in ("shakemap", "raw") and raw_show_summary:
                                raw_med_cache.append(float(np.nanmedian(vv)))
                                raw_min_cache.append(float(np.nanmin(vv)))
                                raw_max_cache.append(float(np.nanmax(vv)))

                    yvals.append(val)

                    rows.append({
                        "target_type": ttype,
                        "target_id": tid,
                        "point_mode": point_mode_s if ttype == "point" else None,
                        "area_radius_km": float(area_radius_km) if (ttype == "point" and point_mode_s == "area") else np.nan,
                        "imt": imt_u,
                        "method": str(meth),
                        "method_status": status,
                        "version": vk,
                        "x": float(x[j]) if np.isfinite(x[j]) else np.nan,
                        "x_label": x_labels[j],
                        "TAE_hours": float(tae_h[j]) if (j < len(tae_h) and np.isfinite(tae_h[j])) else np.nan,
                        "what": what_s,
                        "value": float(val) if np.isfinite(val) else np.nan,
                    })

                y = np.array(yvals, dtype=float)

                ax.plot(
                    x,
                    y,
                    marker=markers.get(mk, "o"),
                    linestyle=linestyles.get(mk, "-"),
                    color=colors.get(mk, None),
                    linewidth=float(linewidth),
                    markersize=float(markersize),
                    label=mk,
                )

            # raw summary overlays (GLOBAL or POINT-AREA) — ShakeMap only
            raw_needed = (ttype == "global") or (ttype == "point" and point_mode_s == "area")
            if raw_needed and raw_show_summary and raw_med_cache and len(raw_med_cache) == len(x):
                try:
                    ax.plot(x, np.array(raw_med_cache, dtype=float), marker="s", linestyle="--", linewidth=1.2, label="ShakeMap median")
                    if raw_fill_minmax and raw_min_cache and raw_max_cache and len(raw_min_cache) == len(raw_max_cache) == len(x):
                        ymin = np.array(raw_min_cache, dtype=float)
                        ymax = np.array(raw_max_cache, dtype=float)
                        ax.fill_between(x, ymin, ymax, alpha=float(raw_fill_alpha), label="ShakeMap min–max")
                except Exception:
                    pass

            # truth line: ONLY for MEAN plots (never for sigma)
            if (what_s == "mean") and (ttype == "point") and show_truth_line and isinstance(tmeta, dict):
                tv = None
                if imt_u == "MMI":
                    tv = tmeta.get("true_mmi", None)
                elif imt_u == "PGA":
                    tv = tmeta.get("true_pga", None)
                else:
                    tv = tmeta.get(f"true_{imt_u.lower()}", None)
                if tv is not None:
                    try:
                        ax.axhline(float(tv), **truth_kwargs)
                    except Exception:
                        pass

            # 24h line
            if show_24h_line and x24 is not None and np.isfinite(x24):
                try:
                    ax.axvline(float(x24), **line24_kwargs)
                except Exception:
                    pass

            x_axis_s = str(x_axis).lower().strip()
            xlab = "Time after event (hours)" if (x_axis_s == "time" or (x_axis_s == "auto" and all(np.isfinite(t) for t in tae_h))) else "ShakeMap version"

            if what_s == "sigma":
                try:
                    ylab = self._sigma_field_for_imt(imt_u) if hasattr(self, "_sigma_field_for_imt") else f"{imt_u} sigma"
                except Exception:
                    ylab = f"{imt_u} sigma"
            else:
                ylab = f"{imt_u} mean"

            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.tick_params(axis="x", rotation=float(xrotation))
            ax.grid(True, alpha=0.3)

            if title is None:
                if ttype == "point":
                    if point_mode_s == "area":
                        ttl = f"{imt_u} {what_s} decay @ {tid} (area mean, r={area_radius_km:g} km)"
                    else:
                        ttl = f"{imt_u} {what_s} decay @ {tid} (point)"
                else:
                    ttl = f"{imt_u} {what_s} decay ({tid})"
            else:
                ttl = title

            ax.set_title(ttl)
            ax.legend()
            fig.tight_layout()

            # save per target
            if save and save_path_prefix:
                stem = f"{save_path_prefix}-{imt_u}-{what_s}-{tid}"
                for ext in (save_formats or ("png",)):
                    p = f"{stem}.{str(ext).lstrip('.')}"
                    fig.savefig(p, dpi=dpi, bbox_inches="tight")

            if show:
                plt.show()

            if close_figs and not show:
                plt.close(fig)

            figs.append(fig)

        df = pd.DataFrame(rows)

        # export data
        if export_data and export_path:
            try:
                df.to_csv(str(export_path), index=False)
            except Exception as e:
                if debug:
                    print(f"[uq_plot_targets_decay] export failed: {repr(e)}")

        # missing report
        if report_missing and missing_map:
            print("\n[uq_plot_targets_decay] Missing method outputs:")
            for m, vs in missing_map.items():
                uniq = []
                for vv in vs:
                    if vv not in uniq:
                        uniq.append(vv)
                print(f"  - {m}: {len(uniq)} versions missing (e.g. {', '.join(uniq[:8])}{' ...' if len(uniq) > 8 else ''})")

            if not compute_if_missing:
                print("  (Tip) Use uq_plot_targets_decay() (convenience) to compute missing curves automatically,")
                print("        or pre-run updates (run_bayes_update / run_residual_kriging_update / run_ensemble_update) then re-plot.")

        return figs, df




    def _uq__plot_targets_decay_core(
        self,
        *,
        version_list=None,
        imt="MMI",
        points=None,
        global_targets=None,
        what="sigma",
        methods=("ShakeMap", "bayes"),
        method_kwargs=None,
        compute_if_missing=False,
        # axis controls
        x_axis="auto",  # "auto" | "versions" | "time"
        # point sampling mode
        point_mode="point",       # "point" | "area"
        area_radius_km=30.0,
        # plot style
        figsize=(10, 5),
        dpi=150,
        markers=None,
        linestyles=None,
        colors=None,
        linewidth=1.8,
        markersize=5.0,
        xrotation=0,
        title=None,
        # reference lines
        show_24h_line=True,
        line24_x=24.0,
        line24_kwargs=None,
        show_truth_line=True,
        truth_kwargs=None,
        # raw global/area summary overlays
        raw_show_summary=True,
        raw_fill_minmax=True,
        raw_fill_alpha=0.18,
        # saving + export
        save=False,
        save_path_prefix=None,
        save_formats=("png",),
        export_data=False,
        export_path=None,
        show=False,
        close_figs=True,
        debug=False,
        # strict reporting
        report_missing=True,
        # NEW: y-axis controls
        pga_ylog=False,          # if True, use log scale on y-axis, but ONLY when imt=="PGA"
        pga_ylog_base=10.0,      # log base (matplotlib uses base=10 by default; kept explicit)
        pga_ylog_min=1e-6,       # clip nonpositive values to NaN when plotting on log scale
    ):
        """Core engine used by both convenience and fast variants.
    
        Updates in this version:
        - ENS robustness: if method == 'ens' and no mode is provided, infer a default mode from IMT
          (MMI -> 'mmi_dyfi', else -> 'pga_instr') so compute_if_missing=True can actually compute ENS.
        - Style robustness: markers/linestyles/colors are resolved case-insensitively (e.g., 'ShakeMap' vs 'shakemap').
        - NEW: Optional log y-axis for PGA only (pga_ylog=True). Nonpositive values are masked.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
    
        if not isinstance(self.uq_state, dict) or not ((self.uq_state.get("versions") or {})):
            raise RuntimeError("uq_state not initialized. Run uq_build_dataset() first.")
    
        imt_u = str(imt).upper().strip()
        what_s = str(what).lower().strip()
        if what_s not in ("sigma", "mean"):
            raise ValueError("what must be 'sigma' or 'mean'.")
    
        vkeys, x, x_labels, tae_h = self._uq__axis_versions_or_time(version_list, x_axis=x_axis)
        if not vkeys:
            raise ValueError("No versions available to plot.")
    
        point_mode_s = str(point_mode).lower().strip()
        if point_mode_s not in ("point", "area"):
            raise ValueError("point_mode must be 'point' or 'area'.")
    
        # targets
        targets = []
        if points:
            for p in points:
                if not isinstance(p, dict):
                    continue
                pid = str(p.get("id", "P"))
                lat = p.get("lat", None)
                lon = p.get("lon", None)
                if lat is None or lon is None:
                    continue
                targets.append(("point", pid, float(lat), float(lon), dict(p)))
    
        if global_targets:
            for g in global_targets:
                targets.append(("global", str(g), None, None, {}))
    
        if not targets:
            targets = [("global", "GLOBAL", None, None, {})]
    
        # style maps
        methods = list(methods)
        markers = markers or {}
        linestyles = linestyles or {}
        colors = colors or {}
        method_kwargs = method_kwargs or {}
        line24_kwargs = line24_kwargs or {"color": "0.5", "linewidth": 1.5, "linestyle": "--", "alpha": 0.8, "label": "24h"}
        truth_kwargs = truth_kwargs or {"color": "r", "linewidth": 1.8, "linestyle": "-", "alpha": 0.9, "label": "Observed"}
    
        lon2d, lat2d = self._uq__get_unified_grid_safe()
    
        # 24h line placement
        x24 = None
        if show_24h_line:
            try:
                x_axis_s = str(x_axis).lower().strip()
                if x_axis_s == "time" or (x_axis_s == "auto" and all(np.isfinite(t) for t in tae_h)):
                    x24 = float(line24_x)
                else:
                    if any(np.isfinite(t) for t in tae_h):
                        tarr = np.array([t if np.isfinite(t) else np.nan for t in tae_h], dtype=float)
                        j = int(np.nanargmin(np.abs(tarr - float(line24_x))))
                        x24 = float(x[j])
            except Exception:
                x24 = None
    
        # missing report collector
        missing_map = {}  # (method)->list of versions
    
        rows = []
        figs = []
    
        # should we apply log scale?
        use_logy = bool(pga_ylog) and (imt_u == "PGA")
        if bool(pga_ylog) and (imt_u != "PGA") and debug:
            print(f"[decay_core] pga_ylog=True ignored for imt={imt_u} (log y-axis only applies to PGA).")
    
        for (ttype, tid, tlat, tlon, tmeta) in targets:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
            # raw summary caches for GLOBAL and POINT-AREA (ShakeMap only)
            raw_med_cache = []
            raw_min_cache = []
            raw_max_cache = []
    
            for meth in methods:
                yvals = []
    
                # robust retrieval of per-method kwargs (case-insensitive)
                mk = str(meth)
                mk_l = mk.lower().strip()
                mkw = (
                    method_kwargs.get(mk)
                    or method_kwargs.get(mk_l)
                    or method_kwargs.get(mk_l.replace(" ", ""))
                    or {}
                )
    
                # ENS needs mode to compute; infer if missing (do NOT mutate caller dict)
                if mk_l == "ens":
                    mode = mkw.get("mode", None) if isinstance(mkw, dict) else None
                    if mode is None or str(mode).strip() == "":
                        mkw = dict(mkw) if isinstance(mkw, dict) else {}
                        mkw["mode"] = "mmi_dyfi" if imt_u == "MMI" else "pga_instr"
                        if debug:
                            print(f"[decay_core] inferred ENS mode='{mkw['mode']}' for imt={imt_u}")
    
                for j, vk in enumerate(vkeys):
                    mu2d, sig2d, meta, status = self._uq__get_method_grids(
                        vkey=vk,
                        method=meth,
                        imt=imt_u,
                        compute_if_missing=compute_if_missing,
                        method_kwargs=mkw,
                        debug=debug,
                    )
    
                    if status != "ok" and str(meth).lower().strip() not in ("shakemap", "raw"):
                        missing_map.setdefault(str(meth), []).append(vk)
    
                    grid2d = sig2d if what_s == "sigma" else mu2d
    
                    # extraction
                    if ttype == "point":
                        if point_mode_s == "point":
                            val = self._uq__nearest_grid_sample(lon2d, lat2d, grid2d, lon=tlon, lat=tlat)
                        else:
                            vmean, vmed, vmin, vmax, n = self._uq__circle_stats(
                                lon2d, lat2d, grid2d, lon=tlon, lat=tlat, radius_km=area_radius_km
                            )
                            val = vmean
    
                            # cache raw-only envelope for POINT-AREA
                            if str(meth).lower().strip() in ("shakemap", "raw") and raw_show_summary:
                                raw_med_cache.append(vmed)
                                raw_min_cache.append(vmin)
                                raw_max_cache.append(vmax)
                    else:
                        # global
                        if grid2d is None:
                            val = np.nan
                        else:
                            vv = np.asarray(grid2d, dtype=float)
                            val = float(np.nanmean(vv))
                            if str(meth).lower().strip() in ("shakemap", "raw") and raw_show_summary:
                                raw_med_cache.append(float(np.nanmedian(vv)))
                                raw_min_cache.append(float(np.nanmin(vv)))
                                raw_max_cache.append(float(np.nanmax(vv)))
    
                    # NEW: if using log y-axis for PGA, mask/clamp invalid values
                    if use_logy:
                        try:
                            fv = float(val)
                            if not np.isfinite(fv) or fv <= 0.0:
                                val = np.nan
                            elif fv < float(pga_ylog_min):
                                # clip ultra-small values to a minimum positive threshold
                                val = float(pga_ylog_min)
                        except Exception:
                            val = np.nan
    
                    yvals.append(val)
    
                    rows.append({
                        "target_type": ttype,
                        "target_id": tid,
                        "point_mode": point_mode_s if ttype == "point" else None,
                        "area_radius_km": float(area_radius_km) if (ttype == "point" and point_mode_s == "area") else np.nan,
                        "imt": imt_u,
                        "method": str(meth),
                        "method_status": status,
                        "version": vk,
                        "x": float(x[j]) if np.isfinite(x[j]) else np.nan,
                        "x_label": x_labels[j],
                        "TAE_hours": float(tae_h[j]) if (j < len(tae_h) and np.isfinite(tae_h[j])) else np.nan,
                        "what": what_s,
                        "value": float(val) if np.isfinite(val) else np.nan,
                        "ylog": bool(use_logy),
                    })
    
                y = np.array(yvals, dtype=float)
    
                # style lookup (case-insensitive)
                marker = markers.get(mk, markers.get(mk_l, "o"))
                ls = linestyles.get(mk, linestyles.get(mk_l, "-"))
                col = colors.get(mk, colors.get(mk_l, None))
    
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linestyle=ls,
                    color=col,
                    linewidth=float(linewidth),
                    markersize=float(markersize),
                    label=mk,
                )
    
            # raw summary overlays (GLOBAL or POINT-AREA) — ShakeMap only
            raw_needed = (ttype == "global") or (ttype == "point" and point_mode_s == "area")
            if raw_needed and raw_show_summary and raw_med_cache and len(raw_med_cache) == len(x):
                try:
                    ymed = np.array(raw_med_cache, dtype=float)
                    ymin = np.array(raw_min_cache, dtype=float) if raw_min_cache else None
                    ymax = np.array(raw_max_cache, dtype=float) if raw_max_cache else None
    
                    # apply log masking to overlays too if needed
                    if use_logy:
                        ymed = np.where(np.isfinite(ymed) & (ymed > 0.0), ymed, np.nan)
                        if ymin is not None:
                            ymin = np.where(np.isfinite(ymin) & (ymin > 0.0), ymin, np.nan)
                        if ymax is not None:
                            ymax = np.where(np.isfinite(ymax) & (ymax > 0.0), ymax, np.nan)
    
                    ax.plot(x, ymed, marker="s", linestyle="--", linewidth=1.2, label="ShakeMap median")
                    if raw_fill_minmax and ymin is not None and ymax is not None and len(ymin) == len(ymax) == len(x):
                        ax.fill_between(x, ymin, ymax, alpha=float(raw_fill_alpha), label="ShakeMap min–max")
                except Exception:
                    pass
    
            # truth line: ONLY for MEAN plots (never for sigma)
            if (what_s == "mean") and (ttype == "point") and show_truth_line and isinstance(tmeta, dict):
                tv = None
                if imt_u == "MMI":
                    tv = tmeta.get("true_mmi", None)
                elif imt_u == "PGA":
                    tv = tmeta.get("true_pga", None)
                else:
                    tv = tmeta.get(f"true_{imt_u.lower()}", None)
                if tv is not None:
                    try:
                        tvf = float(tv)
                        if use_logy and tvf <= 0.0:
                            tvf = np.nan
                        if np.isfinite(tvf):
                            ax.axhline(tvf, **truth_kwargs)
                    except Exception:
                        pass
    
            # 24h line
            if show_24h_line and x24 is not None and np.isfinite(x24):
                try:
                    ax.axvline(float(x24), **line24_kwargs)
                except Exception:
                    pass
    
            x_axis_s = str(x_axis).lower().strip()
            xlab = "Time after event (hours)" if (x_axis_s == "time" or (x_axis_s == "auto" and all(np.isfinite(t) for t in tae_h))) else "ShakeMap version"
    
            if what_s == "sigma":
                try:
                    ylab = self._sigma_field_for_imt(imt_u) if hasattr(self, "_sigma_field_for_imt") else f"{imt_u} sigma"
                except Exception:
                    ylab = f"{imt_u} sigma"
            else:
                ylab = f"{imt_u} mean"
    
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.tick_params(axis="x", rotation=float(xrotation))
            ax.grid(True, alpha=0.3)
    
            # NEW: set y-axis to log scale (PGA only)
            if use_logy:
                try:
                    # matplotlib supports base=... in recent versions; older versions use basex/basey or no kw.
                    ax.set_yscale("log", base=float(pga_ylog_base))
                except TypeError:
                    ax.set_yscale("log")
    
            if title is None:
                if ttype == "point":
                    if point_mode_s == "area":
                        ttl = f"{imt_u} {what_s} decay @ {tid} (area mean, r={area_radius_km:g} km)"
                    else:
                        ttl = f"{imt_u} {what_s} decay @ {tid} (point)"
                else:
                    ttl = f"{imt_u} {what_s} decay ({tid})"
            else:
                ttl = title
    
            ax.set_title(ttl)
            ax.legend()
            fig.tight_layout()
    
            # save per target
            if save and save_path_prefix:
                stem = f"{save_path_prefix}-{imt_u}-{what_s}-{tid}"
                for ext in (save_formats or ("png",)):
                    p = f"{stem}.{str(ext).lstrip('.')}"
                    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    
            if show:
                plt.show()
    
            if close_figs and not show:
                plt.close(fig)
    
            figs.append(fig)
    
        df = pd.DataFrame(rows)
    
        # export data
        if export_data and export_path:
            try:
                df.to_csv(str(export_path), index=False)
            except Exception as e:
                if debug:
                    print(f"[uq_plot_targets_decay] export failed: {repr(e)}")
    
        # missing report
        if report_missing and missing_map:
            print("\n[uq_plot_targets_decay] Missing method outputs:")
            for m, vs in missing_map.items():
                uniq = []
                for vv in vs:
                    if vv not in uniq:
                        uniq.append(vv)
                print(f"  - {m}: {len(uniq)} versions missing (e.g. {', '.join(uniq[:8])}{' ...' if len(uniq) > 8 else ''})")
    
            if not compute_if_missing:
                print("  (Tip) Use uq_plot_targets_decay() (convenience) to compute missing curves automatically,")
                print("        or pre-run updates (run_bayes_update / run_residual_kriging_update / run_ensemble_update) then re-plot.")
    
        return figs, df
    
    
    
    
    # ----------------------------------------------------------------------
    # Convenience / legacy name: computes missing methods automatically
    # ----------------------------------------------------------------------
    def uq_plot_targets_decay(self, **kwargs):
        """Convenience target-decay plot (legacy name).

        This variant will compute missing requested method outputs (Bayes/RK/Ensemble) as needed,
        then plot from stored uq_state.
        """
        return self._uq__plot_targets_decay_core(compute_if_missing=True, report_missing=True, **kwargs)

    # ----------------------------------------------------------------------
    # Fast / strict: plots ONLY existing computed outputs (no state mutation)
    # ----------------------------------------------------------------------
    def uq_plot_targets_decay_fast(self, **kwargs):
        """Fast/strict target-decay plot.

        This variant does NOT compute anything. It only plots what already exists in uq_state.
        Missing methods are reported, and their series will appear as NaN (invisible).
        """
        return self._uq__plot_targets_decay_core(compute_if_missing=False, report_missing=True, **kwargs)







# ======================================================================================
# ADD-ON (NO monkey patch): Extend SHAKEuq in-module by subclassing with same name.
# Paste this at the VERY END of SHAKEuq.py (module level).
#
# This rebinds the name `SHAKEuq` to a subclass that inherits everything above and adds:
#   - uq_plot_method_map
#   - uq_plot_method_map_panel
#   - uq_plot_difference_map
#   - uq_plot_version_differences (batch helper)
# plus small internal helpers for rupture parsing + exporting.
#
# NOTE: After editing, restart kernel or reload the module so the new class definition is used.
# ======================================================================================


    # -------------------------
    # Small internal utilities
    # -------------------------
    def _uq__norm_version_safe(self, version):
        try:
            return self._uq__norm_version(version)  # if present
        except Exception:
            return self._norm_version(version)      # fallback

    def _uq__save_figure_simple(
        self,
        fig,
        *,
        fname_stem: str,
        subdir: str = "uq_plots/uq_maps",
        output_path: str = None,
        save_formats=("png", "pdf"),
        dpi: int = 300,
    ):
        import os
        base = output_path or getattr(self, "out_folder", None) or "."
        out_dir = os.path.join(base, subdir)
        os.makedirs(out_dir, exist_ok=True)

        paths = {}
        for fmt in (save_formats or ()):
            fmt = str(fmt).lstrip(".").lower()
            fpath = os.path.join(out_dir, f"{fname_stem}.{fmt}")
            fig.savefig(fpath, dpi=int(dpi), bbox_inches="tight")
            paths[fmt] = fpath
        return paths

    def _uq__export_grid_csv(
        self,
        *,
        lon2d,
        lat2d,
        Z2d,
        fname_stem: str,
        output_path: str = None,
        subdir: str = "uq_exports/uq_maps",
    ):
        import os
        import numpy as np
        import pandas as pd

        base = output_path or getattr(self, "out_folder", None) or "."
        out_dir = os.path.join(base, subdir)
        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(
            {
                "lon": np.asarray(lon2d, float).ravel(),
                "lat": np.asarray(lat2d, float).ravel(),
                "value": np.asarray(Z2d, float).ravel(),
            }
        )
        fpath = os.path.join(out_dir, f"{fname_stem}.csv")
        df.to_csv(fpath, index=False)
        return fpath

    def _uq__export_meta_json(
        self,
        *,
        meta: dict,
        fname_stem: str,
        output_path: str = None,
        subdir: str = "uq_exports/uq_maps",
    ):
        import os, json
        base = output_path or getattr(self, "out_folder", None) or "."
        out_dir = os.path.join(base, subdir)
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, f"{fname_stem}.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(meta or {}, f, indent=2, sort_keys=True)
        return fpath

    def _uq__extract_rupture_lines(self, version):
        """
        Extract rupture polylines (lon, lat) for the given version if present in uq_state.
        Returns list[(lon_arr, lat_arr)].
        """
        import numpy as np

        if not getattr(self, "uq_state", None):
            return []
        vkey = self._uq__norm_version_safe(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey, {}) or {}
        rup = vpack.get("rupture", None)
        if rup is None:
            return []

        # Some builders store rupture as {"data": <geojson>}
        if isinstance(rup, dict) and "data" in rup and isinstance(rup["data"], dict):
            rup = rup["data"]

        def _iter_geoms(obj):
            if obj is None:
                return
            if isinstance(obj, dict):
                t = obj.get("type")
                if t == "FeatureCollection":
                    for f in obj.get("features", []) or []:
                        yield from _iter_geoms(f)
                elif t == "Feature":
                    yield from _iter_geoms(obj.get("geometry"))
                elif t in ("LineString", "MultiLineString", "Polygon", "MultiPolygon", "GeometryCollection"):
                    yield obj
                else:
                    for v in obj.values():
                        yield from _iter_geoms(v)
            elif isinstance(obj, list):
                for it in obj:
                    yield from _iter_geoms(it)

        lines = []
        for g in _iter_geoms(rup):
            gtype = g.get("type")
            coords = g.get("coordinates", None)

            if gtype == "LineString" and coords:
                arr = np.asarray(coords, float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    lines.append((arr[:, 0], arr[:, 1]))

            elif gtype == "MultiLineString" and coords:
                for seg in coords:
                    arr = np.asarray(seg, float)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        lines.append((arr[:, 0], arr[:, 1]))

            elif gtype == "Polygon" and coords:
                for ring in coords:
                    arr = np.asarray(ring, float)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        lines.append((arr[:, 0], arr[:, 1]))

            elif gtype == "MultiPolygon" and coords:
                for poly in coords:
                    for ring in poly:
                        arr = np.asarray(ring, float)
                        if arr.ndim == 2 and arr.shape[1] >= 2:
                            lines.append((arr[:, 0], arr[:, 1]))

            elif gtype == "GeometryCollection":
                for sub in g.get("geometries", []) or []:
                    for _ in _iter_geoms(sub):
                        pass

        return lines

    def _uq__collect_version_obs_xy(self, version, imt: str):
        """
        Collect version-specific obs lon/lat using build_observations(version, imt).
        Returns (lats, lons, weights_or_None, df_or_None).
        """
        import numpy as np

        vkey = self._uq__norm_version_safe(version)
        try:
            obs = self.build_observations(version=vkey, imt=str(imt))
        except Exception:
            obs = None

        if obs is None:
            return np.asarray([]), np.asarray([]), None, None

        # permissive:
        if isinstance(obs, dict):
            df = obs.get("df") if "df" in obs else obs.get("obs_df")
            if df is None and "data" in obs:
                df = obs["data"]
        else:
            df = obs

        if df is None:
            return np.asarray([]), np.asarray([]), None, None

        cols = {c.lower(): c for c in df.columns}
        lonc = cols.get("lon") or cols.get("longitude") or cols.get("x") or cols.get("lon_deg")
        latc = cols.get("lat") or cols.get("latitude") or cols.get("y") or cols.get("lat_deg")
        if lonc is None or latc is None:
            return np.asarray([]), np.asarray([]), None, df

        lons = np.asarray(df[lonc], float)
        lats = np.asarray(df[latc], float)
        wc = cols.get("weight") or cols.get("w")
        w = np.asarray(df[wc], float) if wc is not None else None
        return lats, lons, w, df

    def _uq__get_shakemap_unified_fields(self, version, imt: str):
        """
        Return (mean2d, sigma2d) for published ShakeMap from unified stacks.
        """
        import numpy as np

        if not getattr(self, "uq_state", None):
            raise RuntimeError("uq_state missing. Run uq_build_dataset first.")
        vkey = self._uq__norm_version_safe(version)

        u = (self.uq_state.get("unified") or {})
        imt_u = str(imt).upper().strip()

        vkeys = u.get("version_keys", None) or u.get("versions", None) or self.uq_state.get("version_list", None)
        if vkeys is None:
            raise RuntimeError("Unified version_keys not found in uq_state['unified'].")

        vkeys_n = [self._uq__norm_version_safe(v) for v in list(vkeys)]
        if vkey not in vkeys_n:
            raise KeyError(f"Version {vkey} not found in unified version_keys.")
        vidx = int(vkeys_n.index(vkey))

        fields = (u.get("fields") or {})
        sigmas = (u.get("sigma") or u.get("sigmas") or {})

        if imt_u not in fields:
            raise KeyError(f"IMT {imt_u} not found in uq_state['unified']['fields'].")

        mean_stack = np.asarray(fields[imt_u], float)
        mean2d = mean_stack[vidx, :, :]

        sigma2d = None
        if isinstance(sigmas, dict) and imt_u in sigmas:
            sig_stack = np.asarray(sigmas[imt_u], float)
            if sig_stack.ndim == 3:
                sigma2d = sig_stack[vidx, :, :]
            else:
                sigma2d = sig_stack[vidx, ...]
        return mean2d, sigma2d

    def _uq__get_field_for_plot(
        self,
        *,
        version,
        imt: str,
        method: str,
        what: str,
        compute_if_missing: bool,
        method_kwargs: dict = None,
    ):
        """
        Canonical accessor for plotting:
          - method in {"ShakeMap","published","raw"} => unified stacks
          - else => self._uq__get_method_grids(...), optionally compute/store
        Returns (Z2d, meta, status, mu2d, sig2d).
        """
        what_u = str(what).lower().strip()
        method_u = str(method).strip()
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()

        if method_u.lower() in ("shakemap", "published", "raw"):
            mu2d, sig2d = self._uq__get_shakemap_unified_fields(vkey, imt_u)
            Z = mu2d if what_u == "mean" else sig2d
            status = "ok" if Z is not None else "missing"
            meta = {"method": "ShakeMap", "key": "published", "imt": imt_u, "version": vkey}
            return Z, meta, status, mu2d, sig2d

        mk = method_kwargs if isinstance(method_kwargs, dict) else {}
        mu2d, sig2d, meta, status = self._uq__get_method_grids(
            vkey=vkey,
            method=method_u,
            imt=imt_u,
            compute_if_missing=bool(compute_if_missing),
            method_kwargs=mk,
        )
        Z = mu2d if what_u == "mean" else sig2d
        return Z, (meta or {}), status, mu2d, sig2d

    def _uq__resolve_colorbar_spec(
        self,
        *,
        imt: str,
        what: str,
        mean_scale_type: str = "usgs",
        mean_pga_units: str = "%g",
        cmap=None,
        norm=None,
        ticks=None,
        bounds=None,
        label=None,
        custom_colorbar=None,
    ):
        """
        Resolve (cmap, norm, ticks, label, bounds).
        - MEAN MMI/PGA: uses self.contour_scale(...) by default
        - SIGMA: defaults to viridis unless overridden
        - custom_colorbar overrides defaults:
            dict with keys cmap/norm/ticks/label/bounds
            or tuple/list (cmap, norm, ticks, label) or (cmap, norm, ticks, label, bounds)
        """
        imt_u = str(imt).upper().strip()
        what_u = str(what).lower().strip()

        if custom_colorbar is not None:
            if isinstance(custom_colorbar, dict):
                return (
                    custom_colorbar.get("cmap", cmap),
                    custom_colorbar.get("norm", norm),
                    custom_colorbar.get("ticks", ticks),
                    custom_colorbar.get("label", label),
                    custom_colorbar.get("bounds", bounds),
                )
            if isinstance(custom_colorbar, (tuple, list)):
                if len(custom_colorbar) == 4:
                    cmap, norm, ticks, label = custom_colorbar
                    return cmap, norm, ticks, label, bounds
                if len(custom_colorbar) >= 5:
                    cmap, norm, ticks, label, bounds = custom_colorbar[:5]
                    return cmap, norm, ticks, label, bounds

        if what_u == "mean":
            try:
                cmap0, bounds0, ticks0, norm0, _used = self.contour_scale(
                    imt_u, scale_type=str(mean_scale_type), units=str(mean_pga_units)
                )
                cmap = cmap if cmap is not None else cmap0
                norm = norm if norm is not None else norm0
                ticks = ticks if ticks is not None else ticks0
                bounds = bounds if bounds is not None else bounds0
                if label is None:
                    label = f"{imt_u} (mean)"
                return cmap, norm, ticks, label, bounds
            except Exception:
                if cmap is None:
                    cmap = "viridis"
                if label is None:
                    label = f"{imt_u} (mean)"
                return cmap, norm, ticks, label, bounds

        if cmap is None:
            cmap = "viridis"
        if label is None:
            label = f"{imt_u} (σ)"
        return cmap, norm, ticks, label, bounds

    def _uq__autosym_limits(self, Z, pct=99.0):
        import numpy as np
        z = np.asarray(Z, float)
        z = z[np.isfinite(z)]
        if z.size == 0:
            return (-1.0, 1.0)
        hi = float(np.nanpercentile(np.abs(z), pct))
        if hi <= 0:
            hi = float(np.nanmax(np.abs(z))) if np.isfinite(np.nanmax(np.abs(z))) else 1.0
            if hi <= 0:
                hi = 1.0
        return (-hi, hi)

    def _uq__two_slope_norm(self, vmin, vmax, vcenter=0.0):
        try:
            from matplotlib.colors import TwoSlopeNorm
            return TwoSlopeNorm(vmin=float(vmin), vcenter=float(vcenter), vmax=float(vmax))
        except Exception:
            return None

    def _uq__plot_counts_box(self, ax, *, version, zorder=40, fontsize=10):
        vkey = self._uq__norm_version_safe(version)
        vpack = (self.uq_state.get("versions") or {}).get(vkey, {}) or {}
        counts = vpack.get("counts", {}) or {}

        n_inst = counts.get("n_instrumented", counts.get("n_inst", None))
        n_dyfi = counts.get("n_dyfi", counts.get("n_dyfi_stationlist", None))

        if n_inst is None and n_dyfi is None:
            return

        txt = f"v{vkey}\n"
        if n_inst is not None:
            txt += f"inst={int(n_inst)}\n"
        if n_dyfi is not None:
            txt += f"dyfi={int(n_dyfi)}\n"

        try:
            ax.text(
                0.01, 0.01, txt.strip(),
                transform=ax.transAxes,
                fontsize=float(fontsize),
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.8),
                zorder=float(zorder),
            )
        except Exception:
            pass

    def _uq__cartopy_axes(
        self,
        *,
        figsize=(10, 8),
        extent=None,
        use_utm=True,
        zorders=None,
        add_ocean=True,
        add_land=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=True,
        land_kwargs=None,
        ocean_kwargs=None,
        borders_kwargs=None,
        coast_kwargs=None,
        gridline_kwargs=None,
    ):
        """
        Returns (fig, ax, used_cartopy, ccrs, zdef).
        zdef is default zorder map which can be overridden by zorders dict.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        zdef = {
            "ocean": 0,
            "land": 1,
            "field": 2,
            "borders": 4,
            "coastlines": 5,
            "gridlines": 6,
            "contours": 25,
            "rupture": 20,
            "obs": 30,
            "text": 40,
            "legend": 50,
            "colorbar": 60,
        }
        if isinstance(zorders, dict):
            zdef.update({str(k): float(v) for k, v in zorders.items()})

        land_kwargs = land_kwargs or {}
        ocean_kwargs = ocean_kwargs or {}
        borders_kwargs = borders_kwargs or {"linewidth": 0.6}
        coast_kwargs = coast_kwargs or {"linewidth": 0.6}
        gridline_kwargs = gridline_kwargs or {"linewidth": 0.5, "alpha": 0.35, "linestyle": "-"}

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            has_cartopy = True
        except Exception:
            ccrs = None
            cfeature = None
            has_cartopy = False

        fig = plt.figure(figsize=figsize)
        if not has_cartopy:
            ax = fig.add_subplot(1, 1, 1)
            if extent is not None:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            return fig, ax, False, None, zdef

        proj = ccrs.PlateCarree()
        if use_utm and extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            lon0 = float(np.nanmean([lon_min, lon_max]))
            lat0 = float(np.nanmean([lat_min, lat_max]))
            zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
            south = bool(lat0 < 0)
            try:
                proj = ccrs.UTM(zone=zone, southern_hemisphere=south)
            except Exception:
                proj = ccrs.PlateCarree()

        ax = fig.add_subplot(1, 1, 1, projection=proj)
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        if add_ocean:
            ax.add_feature(cfeature.OCEAN, zorder=zdef["ocean"], **ocean_kwargs)
        if add_land:
            ax.add_feature(cfeature.LAND, zorder=zdef["land"], **land_kwargs)
        if add_borders:
            ax.add_feature(cfeature.BORDERS, zorder=zdef["borders"], **borders_kwargs)
        if add_coastlines:
            ax.coastlines(zorder=zdef["coastlines"], **coast_kwargs)

        if add_gridlines:
            gl = ax.gridlines(draw_labels=True, zorder=zdef["gridlines"], **gridline_kwargs)
            try:
                gl.top_labels = False
                gl.right_labels = False
            except Exception:
                pass

        return fig, ax, True, ccrs, zdef

    def _uq__overlay_obs_and_rupture(
        self,
        ax,
        *,
        version,
        imt: str,
        used_cartopy: bool,
        ccrs,
        zdef: dict,
        show_obs=True,
        obs_size=10.0,
        obs_kwargs=None,
        show_rupture=True,
        rupture_kwargs=None,
        legend=True,
        legend_kwargs=None,
    ):
        obs_kwargs = obs_kwargs or {}
        rupture_kwargs = rupture_kwargs or {"linewidth": 1.5}
        legend_kwargs = legend_kwargs or {"loc": "upper right", "fontsize": 10, "frameon": True}

        tr = ccrs.PlateCarree() if (used_cartopy and ccrs is not None) else None

        if show_obs:
            lats, lons, _w, _df = self._uq__collect_version_obs_xy(version, imt)
            if lats.size > 0:
                ax.scatter(
                    lons, lats,
                    s=float(obs_size),
                    transform=tr,
                    zorder=zdef.get("obs", 30),
                    label="Observations",
                    **obs_kwargs,
                )

        if show_rupture:
            lines = self._uq__extract_rupture_lines(version)
            if lines:
                first = True
                for x, y in lines:
                    ax.plot(
                        x, y,
                        transform=tr,
                        zorder=zdef.get("rupture", 20),
                        label="Rupture" if first else None,
                        **rupture_kwargs,
                    )
                    first = False

        if legend and (show_obs or show_rupture):
            try:
                ax.legend(**legend_kwargs)
            except Exception:
                pass

    def _uq__fname_stem(self, method, what, imt, version, key=None, tag=None):
        event_id = getattr(self, "event_id", "event")
        v = str(version)
        k = str(key) if key else "nokey"
        t = f"_{tag}" if tag else ""
        return f"map_{event_id}_v{v}_{str(imt).upper()}_{str(method)}_{str(what).lower()}_{k}{t}"

    def _uq__apply_colorbar(
        self,
        fig,
        ax,
        pm,
        *,
        ticks=None,
        label=None,
        shrink=0.70,
        pad=0.02,
        label_fs=10.0,
        tick_fs=9.0,
        colorbar_kwargs=None,
        zorder=None,
    ):
        colorbar_kwargs = colorbar_kwargs or {}
        cb = fig.colorbar(pm, ax=ax, shrink=float(shrink), pad=float(pad), **colorbar_kwargs)
        if ticks is not None:
            try:
                cb.set_ticks(ticks)
            except Exception:
                pass
        if label is not None:
            try:
                cb.set_label(str(label), fontsize=float(label_fs))
            except Exception:
                pass
        try:
            cb.ax.tick_params(labelsize=float(tick_fs))
        except Exception:
            pass
        if zorder is not None:
            try:
                cb.ax.set_zorder(float(zorder))
            except Exception:
                pass
        return cb

    # -------------------------
    # Public mapping functions
    # -------------------------
    def uq_plot_method_map(
        self,
        *,
        version,
        imt: str = "MMI",
        method: str = "ShakeMap",        # ShakeMap | bayes | rk | ens | ...
        what: str = "mean",              # mean | sigma
        compute_if_missing: bool = False,
        method_kwargs: dict = None,

        # overlays
        show_obs: bool = True,
        obs_size: float = 10.0,
        obs_kwargs: dict = None,
        show_rupture: bool = True,
        rupture_kwargs: dict = None,

        # optional station-radius footprint contour
        station_radius_km: float = 0.0,
        station_radius_kwargs: dict = None,

        # base features + zorders
        zorders: dict = None,
        add_ocean: bool = True,
        add_land: bool = True,
        add_borders: bool = True,
        add_coastlines: bool = True,
        add_gridlines: bool = True,
        use_utm: bool = True,

        # color / colorbar
        mean_scale_type: str = "usgs",
        mean_pga_units: str = "%g",
        cmap=None,
        norm=None,
        ticks=None,
        bounds=None,
        vmin=None,
        vmax=None,
        custom_colorbar=None,            # override spec
        colorbar: bool = True,
        colorbar_shrink: float = 0.70,
        colorbar_label: str = None,
        colorbar_kwargs: dict = None,

        # text sizes
        title: str = None,
        show_title: bool = True,
        title_fs: float = 14,
        label_fs: float = 11,
        tick_fs: float = 10,
        legend: bool = True,
        legend_kwargs: dict = None,
        counts_box: bool = True,
        counts_box_fs: float = 10,

        # figure
        figsize=(10, 8),
        dpi: int = 300,

        # saving + export
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        show: bool = True,
        export_grid_csv: bool = False,
        export_meta_json: bool = False,

        # extent
        margin_deg: float = 0.0,
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        if not getattr(self, "uq_state", None):
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")

        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
        what_u = str(what).lower().strip()
        method_u = str(method).strip()

        lon2d, lat2d = self._uq__get_unified_grid_safe()
        extent = self._get_grid_extent(vkey, grid_mode="unified", margin_deg=float(margin_deg))

        Z, meta, status, _mu2d, _sig2d = self._uq__get_field_for_plot(
            version=vkey,
            imt=imt_u,
            method=method_u,
            what=what_u,
            compute_if_missing=bool(compute_if_missing),
            method_kwargs=method_kwargs,
        )
        if Z is None or str(status).lower() != "ok":
            raise RuntimeError(
                f"Requested field missing: method={method_u}, what={what_u}, imt={imt_u}, v={vkey}; status={status}"
            )

        cmap_r, norm_r, ticks_r, label_r, _bounds_r = self._uq__resolve_colorbar_spec(
            imt=imt_u,
            what=what_u,
            mean_scale_type=mean_scale_type,
            mean_pga_units=mean_pga_units,
            cmap=cmap,
            norm=norm,
            ticks=ticks,
            bounds=bounds,
            label=colorbar_label,
            custom_colorbar=custom_colorbar,
        )

        fig, ax, used_cartopy, ccrs, zdef = self._uq__cartopy_axes(
            figsize=figsize,
            extent=extent,
            use_utm=bool(use_utm),
            zorders=zorders,
            add_ocean=bool(add_ocean),
            add_land=bool(add_land),
            add_borders=bool(add_borders),
            add_coastlines=bool(add_coastlines),
            add_gridlines=bool(add_gridlines),
        )

        tr = ccrs.PlateCarree() if (used_cartopy and ccrs is not None) else None

        pm = ax.pcolormesh(
            lon2d, lat2d, np.asarray(Z, float),
            transform=tr,
            cmap=cmap_r,
            norm=norm_r,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            zorder=zdef.get("field", 2),
        )

        if station_radius_km and float(station_radius_km) > 0:
            station_radius_kwargs = station_radius_kwargs or {"linewidths": 1.0}
            lats, lons, _w, _df = self._uq__collect_version_obs_xy(vkey, imt_u)
            if lats.size > 0:
                try:
                    m = self._uq_mask_within_km_of_points(lat2d, lon2d, lats, lons, float(station_radius_km))
                    ax.contour(
                        lon2d, lat2d, np.asarray(m, float),
                        levels=[0.5],
                        transform=tr,
                        zorder=zdef.get("contours", 25),
                        **station_radius_kwargs,
                    )
                except Exception:
                    pass

        self._uq__overlay_obs_and_rupture(
            ax,
            version=vkey,
            imt=imt_u,
            used_cartopy=used_cartopy,
            ccrs=ccrs,
            zdef=zdef,
            show_obs=bool(show_obs),
            obs_size=float(obs_size),
            obs_kwargs=obs_kwargs,
            show_rupture=bool(show_rupture),
            rupture_kwargs=rupture_kwargs,
            legend=bool(legend),
            legend_kwargs=legend_kwargs,
        )

        if counts_box:
            self._uq__plot_counts_box(ax, version=vkey, zorder=zdef.get("text", 40), fontsize=counts_box_fs)

        if colorbar:
            self._uq__apply_colorbar(
                fig, ax, pm,
                ticks=ticks_r,
                label=label_r,
                shrink=colorbar_shrink,
                pad=0.02,
                label_fs=label_fs,
                tick_fs=tick_fs,
                colorbar_kwargs=colorbar_kwargs,
                zorder=zdef.get("colorbar", None),
            )

        if show_title:
            if title is None:
                title = f"{method_u} {what_u} — {imt_u} — v{vkey}"
            ax.set_title(str(title), fontsize=float(title_fs), pad=10)

        fig.tight_layout()

        key = (meta or {}).get("key", None) or (meta or {}).get("update_key", None) or (meta or {}).get("bayes_key", None)
        fname_stem = self._uq__fname_stem(method_u, what_u, imt_u, vkey, key=key)

        saved_paths = {}
        grid_csv_path = None
        meta_json_path = None

        if save:
            saved_paths = self._uq__save_figure_simple(
                fig,
                fname_stem=fname_stem,
                subdir="uq_plots/uq_maps",
                output_path=output_path,
                save_formats=save_formats,
                dpi=dpi,
            )

        if export_grid_csv:
            grid_csv_path = self._uq__export_grid_csv(
                lon2d=lon2d, lat2d=lat2d, Z2d=Z,
                fname_stem=fname_stem,
                output_path=output_path,
                subdir="uq_exports/uq_maps",
            )

        if export_meta_json:
            meta_out = dict(meta or {})
            meta_out.update(
                dict(
                    event_id=getattr(self, "event_id", "event"),
                    version=vkey,
                    imt=imt_u,
                    method=method_u,
                    what=what_u,
                    status=status,
                    extent=extent,
                    saved_paths=saved_paths,
                    grid_csv_path=grid_csv_path,
                )
            )
            meta_json_path = self._uq__export_meta_json(
                meta=meta_out,
                fname_stem=fname_stem,
                output_path=output_path,
                subdir="uq_exports/uq_maps",
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

        record = dict(
            event_id=getattr(self, "event_id", "event"),
            version=vkey,
            imt=imt_u,
            method=method_u,
            what=what_u,
            status=status,
            key=key,
            saved_paths=saved_paths,
            grid_csv_path=grid_csv_path,
            meta_json_path=meta_json_path,
        )
        return fig, ax, record


# redacted 
    def uq_plot_method_map_panel(
        self,
        *,
        version,
        imt: str = "MMI",
        method: str = "bayes",
        method_kwargs: dict = None,
        compute_if_missing: bool = False,
        include_deltas: bool = True,

        # overlays (plotted from this version)
        show_obs: bool = True,
        obs_size: float = 10.0,
        obs_kwargs: dict = None,
        show_rupture: bool = True,
        rupture_kwargs: dict = None,

        # base features + zorders
        zorders: dict = None,
        use_utm: bool = True,
        add_ocean: bool = True,
        add_land: bool = True,
        add_borders: bool = True,
        add_coastlines: bool = True,
        add_gridlines: bool = True,

        # colorbar controls (mean uses contour_scale default; sigma uses continuous)
        mean_scale_type: str = "usgs",
        mean_pga_units: str = "%g",
        custom_colorbar_mean=None,
        custom_colorbar_sigma=None,
        colorbar_shrink: float = 0.70,
        colorbar_kwargs: dict = None,
        colorbar_label_mean: str = None,
        colorbar_label_sigma: str = None,
        colorbar_label_delta: str = "Δ (centered at 0)",
        colorbar_label_delta_sigma: str = "Δ (centered at 0)",

        # fonts
        title_fs: float = 13,
        label_fs: float = 10,
        tick_fs: float = 9,

        # figure
        figsize=(14, 10),
        dpi: int = 250,

        # saving
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        show: bool = True,
    ):
        """
        Panel: ShakeMap mean/sigma vs one method mean/sigma, optionally deltas.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if not getattr(self, "uq_state", None):
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")

        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
        method_u = str(method).strip()

        lon2d, lat2d = self._uq__get_unified_grid_safe()
        extent = self._get_grid_extent(vkey, grid_mode="unified", margin_deg=0.0)

        # fetch fields
        Zs_mean, _, _, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method="ShakeMap", what="mean",
            compute_if_missing=False, method_kwargs=None
        )
        Zs_sig, _, _, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method="ShakeMap", what="sigma",
            compute_if_missing=False, method_kwargs=None
        )
        Zm_mean, meta_m, _, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method=method_u, what="mean",
            compute_if_missing=bool(compute_if_missing), method_kwargs=method_kwargs
        )
        Zm_sig, meta_s, _, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method=method_u, what="sigma",
            compute_if_missing=bool(compute_if_missing), method_kwargs=method_kwargs
        )

        if Zs_mean is None or Zm_mean is None:
            raise RuntimeError("Panel requires mean fields for ShakeMap and method.")
        if Zs_sig is None or Zm_sig is None:
            include_deltas = False

        ncols = 3 if include_deltas else 2
        nrows = 2
        fig = plt.figure(figsize=figsize)

        # Decide projection ONCE
        ftmp, axtmp, used_cartopy, ccrs, zdef = self._uq__cartopy_axes(
            figsize=(1, 1),
            extent=extent,
            use_utm=bool(use_utm),
            zorders=zorders,
            add_ocean=False,
            add_land=False,
            add_borders=False,
            add_coastlines=False,
            add_gridlines=False,
        )
        proj = getattr(axtmp, "projection", None)
        plt.close(ftmp)

        def _add_base_features(ax):
            if not used_cartopy or ccrs is None:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                return
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            try:
                import cartopy.feature as cfeature
                if add_ocean:
                    ax.add_feature(cfeature.OCEAN, zorder=zdef["ocean"])
                if add_land:
                    ax.add_feature(cfeature.LAND, zorder=zdef["land"])
                if add_borders:
                    ax.add_feature(cfeature.BORDERS, zorder=zdef["borders"], linewidth=0.6)
                if add_coastlines:
                    ax.coastlines(zorder=zdef["coastlines"], linewidth=0.6)
                if add_gridlines:
                    ax.gridlines(draw_labels=False, zorder=zdef["gridlines"], linewidth=0.5, alpha=0.35)
            except Exception:
                pass

        # Build axes
        axs = []
        for i in range(nrows * ncols):
            if used_cartopy and ccrs is not None and proj is not None:
                ax = fig.add_subplot(nrows, ncols, i + 1, projection=proj)
            else:
                ax = fig.add_subplot(nrows, ncols, i + 1)
            _add_base_features(ax)
            axs.append(ax)

        tr = ccrs.PlateCarree() if (used_cartopy and ccrs is not None) else None
        colorbar_kwargs = colorbar_kwargs or {}

        def _plot_slot(ax, Z, *, cmap, norm, vmin=None, vmax=None, ttl=""):
            pm = ax.pcolormesh(
                lon2d, lat2d, np.asarray(Z, float),
                transform=tr,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
                zorder=zdef.get("field", 2),
            )
            ax.set_title(str(ttl), fontsize=float(title_fs))
            if show_obs or show_rupture:
                self._uq__overlay_obs_and_rupture(
                    ax,
                    version=vkey,
                    imt=imt_u,
                    used_cartopy=used_cartopy,
                    ccrs=ccrs,
                    zdef=zdef,
                    show_obs=bool(show_obs),
                    obs_size=float(obs_size),
                    obs_kwargs=obs_kwargs,
                    show_rupture=bool(show_rupture),
                    rupture_kwargs=rupture_kwargs,
                    legend=False,
                )
            return pm

        # Resolve colormaps/norms for mean and sigma
        cmap_sm_mean, norm_sm_mean, ticks_sm_mean, lbl_sm_mean, _ = self._uq__resolve_colorbar_spec(
            imt=imt_u, what="mean",
            mean_scale_type=mean_scale_type, mean_pga_units=mean_pga_units,
            custom_colorbar=custom_colorbar_mean,
            label=colorbar_label_mean,
        )
        cmap_m_mean, norm_m_mean, ticks_m_mean, lbl_m_mean, _ = self._uq__resolve_colorbar_spec(
            imt=imt_u, what="mean",
            mean_scale_type=mean_scale_type, mean_pga_units=mean_pga_units,
            custom_colorbar=custom_colorbar_mean,
            label=colorbar_label_mean,
        )

        cmap_sm_sig, norm_sm_sig, ticks_sm_sig, lbl_sm_sig, _ = self._uq__resolve_colorbar_spec(
            imt=imt_u, what="sigma",
            mean_scale_type=mean_scale_type, mean_pga_units=mean_pga_units,
            custom_colorbar=custom_colorbar_sigma,
            label=colorbar_label_sigma,
        )
        cmap_m_sig, norm_m_sig, ticks_m_sig, lbl_m_sig, _ = self._uq__resolve_colorbar_spec(
            imt=imt_u, what="sigma",
            mean_scale_type=mean_scale_type, mean_pga_units=mean_pga_units,
            custom_colorbar=custom_colorbar_sigma,
            label=colorbar_label_sigma,
        )

        # Row 0: mean
        pm0 = _plot_slot(axs[0], Zs_mean, cmap=cmap_sm_mean, norm=norm_sm_mean, ttl=f"ShakeMap mean (v{vkey})")
        pm1 = _plot_slot(axs[1], Zm_mean, cmap=cmap_m_mean, norm=norm_m_mean, ttl=f"{method_u} mean (v{vkey})")

        pm2 = None
        if include_deltas:
            dmean = np.asarray(Zm_mean, float) - np.asarray(Zs_mean, float)
            vmin_d, vmax_d = self._uq__autosym_limits(dmean)
            dn = self._uq__two_slope_norm(vmin_d, vmax_d, vcenter=0.0)
            pm2 = _plot_slot(axs[2], dmean, cmap="seismic", norm=dn, vmin=vmin_d, vmax=vmax_d, ttl=f"Δmean ({method_u} − ShakeMap)")

        # Row 1: sigma
        pm3 = _plot_slot(axs[ncols + 0], Zs_sig, cmap=cmap_sm_sig, norm=norm_sm_sig, ttl=f"ShakeMap σ (v{vkey})") if Zs_sig is not None else None
        pm4 = _plot_slot(axs[ncols + 1], Zm_sig, cmap=cmap_m_sig, norm=norm_m_sig, ttl=f"{method_u} σ (v{vkey})") if Zm_sig is not None else None

        pm5 = None
        if include_deltas and (Zs_sig is not None) and (Zm_sig is not None):
            dsig = np.asarray(Zm_sig, float) - np.asarray(Zs_sig, float)
            vmin_d2, vmax_d2 = self._uq__autosym_limits(dsig)
            dn2 = self._uq__two_slope_norm(vmin_d2, vmax_d2, vcenter=0.0)
            pm5 = _plot_slot(axs[ncols + 2], dsig, cmap="seismic", norm=dn2, vmin=vmin_d2, vmax=vmax_d2, ttl=f"Δσ ({method_u} − ShakeMap)")

        # Colorbars (per-axis, with full override control)
        self._uq__apply_colorbar(
            fig, axs[0], pm0,
            ticks=ticks_sm_mean,
            label=lbl_sm_mean if colorbar_label_mean is None else colorbar_label_mean,
            shrink=colorbar_shrink,
            pad=0.02,
            label_fs=label_fs,
            tick_fs=tick_fs,
            colorbar_kwargs=colorbar_kwargs,
            zorder=zdef.get("colorbar", None),
        )
        self._uq__apply_colorbar(
            fig, axs[1], pm1,
            ticks=ticks_m_mean,
            label=lbl_m_mean if colorbar_label_mean is None else colorbar_label_mean,
            shrink=colorbar_shrink,
            pad=0.02,
            label_fs=label_fs,
            tick_fs=tick_fs,
            colorbar_kwargs=colorbar_kwargs,
            zorder=zdef.get("colorbar", None),
        )
        if pm3 is not None:
            self._uq__apply_colorbar(
                fig, axs[ncols + 0], pm3,
                ticks=ticks_sm_sig,
                label=lbl_sm_sig if colorbar_label_sigma is None else colorbar_label_sigma,
                shrink=colorbar_shrink,
                pad=0.02,
                label_fs=label_fs,
                tick_fs=tick_fs,
                colorbar_kwargs=colorbar_kwargs,
                zorder=zdef.get("colorbar", None),
            )
        if pm4 is not None:
            self._uq__apply_colorbar(
                fig, axs[ncols + 1], pm4,
                ticks=ticks_m_sig,
                label=lbl_m_sig if colorbar_label_sigma is None else colorbar_label_sigma,
                shrink=colorbar_shrink,
                pad=0.02,
                label_fs=label_fs,
                tick_fs=tick_fs,
                colorbar_kwargs=colorbar_kwargs,
                zorder=zdef.get("colorbar", None),
            )
        if include_deltas and pm2 is not None:
            self._uq__apply_colorbar(
                fig, axs[2], pm2,
                ticks=None,
                label=colorbar_label_delta,
                shrink=colorbar_shrink,
                pad=0.02,
                label_fs=label_fs,
                tick_fs=tick_fs,
                colorbar_kwargs=colorbar_kwargs,
                zorder=zdef.get("colorbar", None),
            )
        if include_deltas and pm5 is not None:
            self._uq__apply_colorbar(
                fig, axs[ncols + 2], pm5,
                ticks=None,
                label=colorbar_label_delta_sigma,
                shrink=colorbar_shrink,
                pad=0.02,
                label_fs=label_fs,
                tick_fs=tick_fs,
                colorbar_kwargs=colorbar_kwargs,
                zorder=zdef.get("colorbar", None),
            )

        fig.tight_layout()

        saved_paths = {}
        if save:
            event_id = getattr(self, "event_id", "event")
            fname_stem = f"panel_{event_id}_v{vkey}_{imt_u}_{method_u}"
            saved_paths = self._uq__save_figure_simple(
                fig,
                fname_stem=fname_stem,
                subdir="uq_plots/uq_maps",
                output_path=output_path,
                save_formats=save_formats,
                dpi=dpi,
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

        record = dict(
            event_id=getattr(self, "event_id", "event"),
            version=vkey,
            imt=imt_u,
            method=method_u,
            saved_paths=saved_paths,
        )
        return fig, axs, record

    def uq_plot_difference_map(
        self,
        *,
        imt: str = "MMI",
        what: str = "mean",                  # mean | sigma
        method: str = "ShakeMap",            # ShakeMap or method name
        v_from=None,
        v_to=None,
        mode: str = "incremental",           # incremental | cumulative_from_first
        compute_if_missing: bool = False,
        method_kwargs: dict = None,

        # overlays from v_to by default
        show_obs: bool = True,
        obs_size: float = 10.0,
        obs_kwargs: dict = None,
        show_rupture: bool = True,
        rupture_kwargs: dict = None,

        # diverging colormap fixed at 0->white
        cmap: str = "seismic",
        vmin=None,
        vmax=None,

        # base features + zorders
        zorders: dict = None,
        use_utm: bool = True,
        add_ocean: bool = True,
        add_land: bool = True,
        add_borders: bool = True,
        add_coastlines: bool = True,
        add_gridlines: bool = True,

        # figure + styling
        figsize=(10, 8),
        dpi: int = 300,
        title: str = None,
        show_title: bool = True,
        title_fs: float = 14,
        label_fs: float = 11,
        tick_fs: float = 10,
        colorbar_shrink: float = 0.70,
        colorbar_label: str = None,
        colorbar_kwargs: dict = None,

        # saving/export
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        show: bool = True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        if not getattr(self, "uq_state", None):
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")

        imt_u = str(imt).upper().strip()
        what_u = str(what).lower().strip()
        method_u = str(method).strip()
        mode_u = str(mode).lower().strip()

        vlist = (self.uq_state.get("version_list") or [])
        if not vlist:
            raise RuntimeError("uq_state['version_list'] missing.")

        if v_to is None:
            v_to = vlist[-1]
        if mode_u == "cumulative_from_first":
            v_from = vlist[0]
        elif v_from is None:
            v_from = vlist[0]

        v_to_k = self._uq__norm_version_safe(v_to)
        v_from_k = self._uq__norm_version_safe(v_from)

        Zfrom, meta_f, _, *_ = self._uq__get_field_for_plot(
            version=v_from_k, imt=imt_u, method=method_u, what=what_u,
            compute_if_missing=bool(compute_if_missing), method_kwargs=method_kwargs
        )
        Zto, meta_t, _, *_ = self._uq__get_field_for_plot(
            version=v_to_k, imt=imt_u, method=method_u, what=what_u,
            compute_if_missing=bool(compute_if_missing), method_kwargs=method_kwargs
        )
        if Zfrom is None or Zto is None:
            raise RuntimeError("Difference fields missing (from/to).")

        d = np.asarray(Zto, float) - np.asarray(Zfrom, float)

        if vmin is None or vmax is None:
            vmin_, vmax_ = self._uq__autosym_limits(d)
            vmin = vmin if vmin is not None else vmin_
            vmax = vmax if vmax is not None else vmax_
        dn = self._uq__two_slope_norm(vmin, vmax, vcenter=0.0)

        lon2d, lat2d = self._uq__get_unified_grid_safe()
        extent = self._get_grid_extent(v_to_k, grid_mode="unified", margin_deg=0.0)

        fig, ax, used_cartopy, ccrs, zdef = self._uq__cartopy_axes(
            figsize=figsize,
            extent=extent,
            use_utm=bool(use_utm),
            zorders=zorders,
            add_ocean=bool(add_ocean),
            add_land=bool(add_land),
            add_borders=bool(add_borders),
            add_coastlines=bool(add_coastlines),
            add_gridlines=bool(add_gridlines),
        )
        tr = ccrs.PlateCarree() if (used_cartopy and ccrs is not None) else None

        pm = ax.pcolormesh(
            lon2d, lat2d, d,
            transform=tr,
            cmap=str(cmap),
            norm=dn,
            vmin=float(vmin),
            vmax=float(vmax),
            shading="auto",
            zorder=zdef.get("field", 2),
        )

        self._uq__overlay_obs_and_rupture(
            ax,
            version=v_to_k,
            imt=imt_u,
            used_cartopy=used_cartopy,
            ccrs=ccrs,
            zdef=zdef,
            show_obs=bool(show_obs),
            obs_size=float(obs_size),
            obs_kwargs=obs_kwargs,
            show_rupture=bool(show_rupture),
            rupture_kwargs=rupture_kwargs,
            legend=True,
        )
        self._uq__plot_counts_box(ax, version=v_to_k, zorder=zdef.get("text", 40), fontsize=10)

        if colorbar_label is None:
            colorbar_label = f"Δ{what_u} ({method_u}) = v{v_to_k} − v{v_from_k}"
        self._uq__apply_colorbar(
            fig, ax, pm,
            ticks=None,
            label=colorbar_label,
            shrink=colorbar_shrink,
            pad=0.02,
            label_fs=label_fs,
            tick_fs=tick_fs,
            colorbar_kwargs=colorbar_kwargs,
            zorder=zdef.get("colorbar", None),
        )

        if show_title:
            if title is None:
                title = f"Δ{what_u} ({method_u}) — {imt_u} — v{v_from_k}→v{v_to_k}"
            ax.set_title(str(title), fontsize=float(title_fs), pad=10)

        fig.tight_layout()

        key = (meta_t or {}).get("key", None) or (meta_t or {}).get("update_key", None) or (meta_t or {}).get("bayes_key", None)
        event_id = getattr(self, "event_id", "event")
        fname_stem = f"diff_{event_id}_{method_u}_{what_u}_{imt_u}_v{v_from_k}_to_v{v_to_k}_{key or 'nokey'}"

        saved_paths = {}
        if save:
            saved_paths = self._uq__save_figure_simple(
                fig,
                fname_stem=fname_stem,
                subdir="uq_plots/uq_maps",
                output_path=output_path,
                save_formats=save_formats,
                dpi=dpi,
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

        record = dict(
            event_id=event_id,
            method=method_u,
            what=what_u,
            imt=imt_u,
            v_from=v_from_k,
            v_to=v_to_k,
            key=key,
            saved_paths=saved_paths,
        )
        return fig, ax, record

    def uq_plot_version_differences(
        self,
        *,
        imt: str = "MMI",
        what: str = "mean",
        method: str = "ShakeMap",
        mode: str = "incremental",                # incremental | cumulative_from_first
        compute_if_missing: bool = False,
        method_kwargs: dict = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        output_path: str = None,
        show: bool = False,
        **diff_kwargs,
    ):
        """
        Batch helper: save individual difference figures.
        - incremental: v(i) - v(i-1)
        - cumulative_from_first: v(i) - v(0)
        Returns list of records.
        """
        vlist = (self.uq_state.get("version_list") or [])
        if not vlist:
            raise RuntimeError("uq_state['version_list'] missing.")

        mode_u = str(mode).lower().strip()
        records = []

        if mode_u == "incremental":
            pairs = [(vlist[i - 1], vlist[i]) for i in range(1, len(vlist))]
        elif mode_u == "cumulative_from_first":
            v0 = vlist[0]
            pairs = [(v0, vlist[i]) for i in range(1, len(vlist))]
        else:
            raise ValueError("mode must be 'incremental' or 'cumulative_from_first'.")

        for vf, vt in pairs:
            fig, ax, rec = self.uq_plot_difference_map(
                imt=imt,
                what=what,
                method=method,
                v_from=vf,
                v_to=vt,
                mode="incremental",
                compute_if_missing=compute_if_missing,
                method_kwargs=method_kwargs,
                save=save,
                save_formats=save_formats,
                output_path=output_path,
                show=show,
                **diff_kwargs,
            )
            records.append(rec)

        return records



    # --- PATCH REPLACEMENTS (copy/paste over the same methods inside your ADD-ON subclass) ---
    
    def _uq__get_shakemap_unified_fields(self, version, imt: str):
        """
        Return (mean2d, sigma2d) for published ShakeMap from unified stacks.
    
        NOTE:
        - Some events/IMTs do not have unified sigma stacks for MMI (or sigma may be missing).
        - In that case we fall back to the existing prior-sigma accessor (_uq_prior_sigma_fields),
          using TOTAL sigma by default (consistent with your legacy sigma maps).
        """
        import numpy as np
    
        if not getattr(self, "uq_state", None):
            raise RuntimeError("uq_state missing. Run uq_build_dataset first.")
        vkey = self._uq__norm_version_safe(version)
    
        u = (self.uq_state.get("unified") or {})
        imt_u = str(imt).upper().strip()
    
        vkeys = u.get("version_keys", None) or u.get("versions", None) or self.uq_state.get("version_list", None)
        if vkeys is None:
            raise RuntimeError("Unified version_keys not found in uq_state['unified'].")
    
        vkeys_n = [self._uq__norm_version_safe(v) for v in list(vkeys)]
        if vkey not in vkeys_n:
            raise KeyError(f"Version {vkey} not found in unified version_keys.")
        vidx = int(vkeys_n.index(vkey))
    
        fields = (u.get("fields") or {})
        sigmas = (u.get("sigma") or u.get("sigmas") or {})
    
        if imt_u not in fields:
            raise KeyError(f"IMT {imt_u} not found in uq_state['unified']['fields'].")
    
        mean_stack = np.asarray(fields[imt_u], float)
        mean2d = mean_stack[vidx, :, :]
    
        sigma2d = None
        # 1) Try unified sigma stacks (if present)
        try:
            if isinstance(sigmas, dict) and imt_u in sigmas:
                sig_stack = np.asarray(sigmas[imt_u], float)
                if sig_stack.ndim >= 3:
                    sigma2d = sig_stack[vidx, :, :]
                else:
                    sigma2d = sig_stack[vidx, ...]
        except Exception:
            sigma2d = None
    
        # 2) Fallback: prior/local sigma fields (TOTAL by default)
        if sigma2d is None:
            try:
                sig_ep, sig_total, _sig_a = self._uq_prior_sigma_fields(vkey, imt_u)
                if sig_total is not None:
                    sigma2d = np.asarray(sig_total, float)
                elif sig_ep is not None:
                    sigma2d = np.asarray(sig_ep, float)
            except Exception:
                sigma2d = None
    
        return mean2d, sigma2d
    
    
    def _uq__overlay_obs_and_rupture(
        self,
        ax,
        *,
        version,
        imt: str,
        used_cartopy: bool,
        ccrs,
        zdef: dict,
        show_obs=True,
        obs_size=10.0,
        obs_kwargs=None,
        show_rupture=True,
        rupture_kwargs=None,
        legend=True,
        legend_kwargs=None,
    ):
        obs_kwargs = obs_kwargs or {}
        rupture_kwargs = rupture_kwargs or {"linewidth": 1.5}
        legend_kwargs = legend_kwargs or {"loc": "upper right", "fontsize": 10, "frameon": True}
    
        tr = ccrs.PlateCarree() if (used_cartopy and ccrs is not None) else None
    
        if show_obs:
            lats, lons, _w, _df = self._uq__collect_version_obs_xy(version, imt)
            if lats.size > 0:
                ax.scatter(
                    lons, lats,
                    s=float(obs_size),
                    transform=tr,
                    zorder=zdef.get("obs", 30),
                    label="Observations",
                    **obs_kwargs,
                )
    
        if show_rupture:
            lines = self._uq__extract_rupture_lines(version)
            if lines:
                first = True
                for x, y in lines:
                    ax.plot(
                        x, y,
                        transform=tr,
                        zorder=zdef.get("rupture", 20),
                        label="Rupture" if first else None,
                        **rupture_kwargs,
                    )
                    first = False
    
        # Guard legend: only call if there is at least one labeled artist
        if legend and (show_obs or show_rupture):
            try:
                handles, labels = ax.get_legend_handles_labels()
                labels = [lb for lb in labels if lb and not str(lb).startswith("_")]
                if len(labels) > 0:
                    ax.legend(**legend_kwargs)
            except Exception:
                pass


    # --- PATCH REPLACEMENT (copy/paste over your existing _uq__get_field_for_plot in the ADD-ON subclass) ---
    
    def _uq__get_field_for_plot(
        self,
        *,
        version,
        imt: str,
        method: str,
        what: str,
        compute_if_missing: bool,
        method_kwargs: dict = None,
    ):
        """
        Canonical accessor for plotting:
          - method in {"ShakeMap","published","raw"} => unified stacks (mean) + fallback sigma
          - else => self._uq__get_method_grids(...), optionally compute/store
    
        Returns (Z2d, meta, status, mu2d, sig2d).
        """
        what_u = str(what).lower().strip()
        method_u = str(method).strip()
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
    
        if method_u.lower() in ("shakemap", "published", "raw"):
            mu2d, sig2d = self._uq__get_shakemap_unified_fields(vkey, imt_u)
    
            # EXTRA fallback: some builds keep sigma only via _uq_prior_sigma_fields(version_int, imt)
            if (what_u != "mean") and (sig2d is None):
                try:
                    v_int = int(vkey)
                except Exception:
                    v_int = vkey
                try:
                    sig_ep, sig_total, _sig_a = self._uq_prior_sigma_fields(v_int, imt_u)
                    if sig_total is not None:
                        sig2d = sig_total
                    elif sig_ep is not None:
                        sig2d = sig_ep
                except Exception:
                    pass
    
            Z = mu2d if what_u == "mean" else sig2d
            status = "ok" if Z is not None else "missing"
            meta = {"method": "ShakeMap", "key": "published", "imt": imt_u, "version": vkey}
            return Z, meta, status, mu2d, sig2d
    
        mk = method_kwargs if isinstance(method_kwargs, dict) else {}
        mu2d, sig2d, meta, status = self._uq__get_method_grids(
            vkey=vkey,
            method=method_u,
            imt=imt_u,
            compute_if_missing=bool(compute_if_missing),
            method_kwargs=mk,
        )
        Z = mu2d if what_u == "mean" else sig2d
        return Z, (meta or {}), status, mu2d, sig2d


    def _uq__get_shakemap_unified_fields(self, version, imt: str):
        """
        Return (mean2d, sigma2d) for published ShakeMap from unified stacks.
    
        Canonical storage in uq_state:
          - mean:  uq_state["unified"]["fields"][IMT]
          - sigma: uq_state["unified"]["sigma"][STD<IMT>]  e.g., STDMMI, STDPGA
    
        sigma2d is returned as None if:
          - the sigma stack is absent, OR
          - the requested version slice is all-NaN / non-finite.
    
        This function does NOT fabricate sigma and does NOT silently fill NaNs.
        """
        import numpy as np
    
        if not getattr(self, "uq_state", None):
            raise RuntimeError("uq_state missing. Run uq_build_dataset() first.")
    
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
    
        u = (self.uq_state.get("unified") or {})
    
        # Robust unified version key source
        vkeys = (
            u.get("version_keys")
            or u.get("versions")
            or self.uq_state.get("version_list")
            or self.uq_state.get("versions")
        )
        if not vkeys:
            raise RuntimeError("Unified version_keys not found in uq_state['unified'].")
    
        vkeys_n = [self._uq__norm_version_safe(v) for v in list(vkeys)]
        if vkey not in vkeys_n:
            raise KeyError(f"Version {vkey} not found in unified version_keys.")
        vidx = int(vkeys_n.index(vkey))
    
        # Mean stacks
        fields = (u.get("fields") or {})
        if imt_u not in fields:
            raise KeyError(f"IMT {imt_u} not found in uq_state['unified']['fields'].")





    def _uq__get_shakemap_unified_fields(self, version, imt: str):
        """
        Return (mean2d, sigma2d) for published ShakeMap from unified stacks.
    
        Canonical storage in uq_state:
          - mean:  uq_state["unified"]["fields"][IMT]
          - sigma: uq_state["unified"]["sigma"][STD<IMT>]  e.g., STDMMI, STDPGA
    
        sigma2d is returned as None if:
          - the sigma stack is absent, OR
          - the requested version slice is all-NaN / non-finite.
    
        This function does NOT fabricate sigma and does NOT silently fill NaNs.
        """
        import numpy as np
    
        if not getattr(self, "uq_state", None):
            raise RuntimeError("uq_state missing. Run uq_build_dataset() first.")
    
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
    
        u = (self.uq_state.get("unified") or {})
    
        # Robust unified version key source
        vkeys = (
            u.get("version_keys")
            or u.get("versions")
            or self.uq_state.get("version_list")
            or self.uq_state.get("versions")
        )
        if not vkeys:
            raise RuntimeError("Unified version_keys not found in uq_state['unified'].")
    
        vkeys_n = [self._uq__norm_version_safe(v) for v in list(vkeys)]
        if vkey not in vkeys_n:
            raise KeyError(f"Version {vkey} not found in unified version_keys.")
        vidx = int(vkeys_n.index(vkey))
    
        # Mean stacks
        fields = (u.get("fields") or {})
        if imt_u not in fields:
            raise KeyError(f"IMT {imt_u} not found in uq_state['unified']['fields'].")
    
        mean_stack = np.asarray(fields[imt_u], float)
        if mean_stack.ndim < 3:
            raise RuntimeError(f"Unified mean stack for {imt_u} has unexpected shape: {mean_stack.shape}")
        mean2d = mean_stack[vidx, :, :]
    
        # Sigma stacks (STD* keys)
        sigma2d = None
        sigmas = (u.get("sigma") or u.get("sigmas") or {})
        if isinstance(sigmas, dict) and sigmas:
            if hasattr(self, "_bayes_sigma_key_for_imt"):
                sig_key = self._bayes_sigma_key_for_imt(imt_u)
            else:
                sig_key = f"STD{imt_u}"
    
            if sig_key in sigmas:
                sig_stack = np.asarray(sigmas[sig_key], float)
    
                if sig_stack.ndim >= 3:
                    cand = sig_stack[vidx, :, :]
                elif sig_stack.ndim == 2:
                    # Edge case: [nver, npts]-like; still index by version.
                    cand = sig_stack[vidx, :]
                else:
                    cand = None
    
                if cand is not None and np.size(cand) > 0 and np.any(np.isfinite(cand)):
                    sigma2d = cand
                else:
                    sigma2d = None
    
        return mean2d, sigma2d
    
    
    def _uq__get_field_for_plot(
        self,
        *,
        version,
        imt,
        method,
        what,
        compute_if_missing=False,
        method_kwargs=None,
    ):
        """
        Canonical plot accessor.
    
        For published ShakeMap:
          - mean always from unified fields[IMT]
          - sigma from unified sigma[STD<IMT>]
          - if mean exists but sigma is unavailable -> status="missing_uncertainty" (explicit)
    
        For non-published methods:
          - defer to existing self._uq__get_method_grids() contract (compute/store if needed)
    
        Returns:
          Z, meta, status, mu2d, sig2d
        """
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
        method_u = str(method).strip()
        what_u = str(what).lower().strip()
    
        def _is_shakemap(m: str) -> bool:
            mk = str(m).lower().strip().replace(" ", "")
            return mk in ("shakemap", "published", "raw")
    
        # -------------------------
        # Published ShakeMap branch
        # -------------------------
        if _is_shakemap(method_u):
            mu2d, sig2d = self._uq__get_shakemap_unified_fields(vkey, imt_u)
    
            if what_u == "mean":
                Z = mu2d
            elif what_u == "sigma":
                Z = sig2d
            else:
                return (
                    None,
                    {"method": "ShakeMap", "imt": imt_u, "version": vkey, "what": what_u, "reason": "unknown_what"},
                    "missing",
                    mu2d,
                    sig2d,
                )
    
            if what_u == "sigma" and (mu2d is not None) and (Z is None):
                status = "missing_uncertainty"
                meta = {
                    "method": "ShakeMap",
                    "imt": imt_u,
                    "version": vkey,
                    "what": what_u,
                    "reason": "published_sigma_not_available",
                }
            else:
                status = "ok" if Z is not None else "missing"
                meta = {"method": "ShakeMap", "imt": imt_u, "version": vkey, "what": what_u}
    
            return Z, meta, status, mu2d, sig2d
    
        # -------------------------
        # Method branch (rk/bayes/ens/...)
        # -------------------------
        if not hasattr(self, "_uq__get_method_grids"):
            raise RuntimeError("Missing _uq__get_method_grids() implementation; cannot fetch non-ShakeMap method grids.")
    
        Z, meta, status, mu2d, sig2d = self._uq__get_method_grids(
            version=vkey,
            imt=imt_u,
            method=method_u,
            what=what_u,
            compute_if_missing=bool(compute_if_missing),
            method_kwargs=method_kwargs,
        )
        return Z, meta, status, mu2d, sig2d

    
        mean_stack = np.asarray(fields[imt_u], float)
        if mean_stack.ndim < 3:
            raise RuntimeError(f"Unified mean stack for {imt_u} has unexpected shape: {mean_stack.shape}")
        mean2d = mean_stack[vidx, :, :]
    
        # Sigma stacks (STD* keys)
        sigma2d = None
        sigmas = (u.get("sigma") or u.get("sigmas") or {})
        if isinstance(sigmas, dict) and sigmas:
            if hasattr(self, "_bayes_sigma_key_for_imt"):
                sig_key = self._bayes_sigma_key_for_imt(imt_u)
            else:
                sig_key = f"STD{imt_u}"
    
            if sig_key in sigmas:
                sig_stack = np.asarray(sigmas[sig_key], float)
    
                if sig_stack.ndim >= 3:
                    cand = sig_stack[vidx, :, :]
                elif sig_stack.ndim == 2:
                    # Edge case: [nver, npts]-like; still index by version.
                    cand = sig_stack[vidx, :]
                else:
                    cand = None
    
                if cand is not None and np.size(cand) > 0 and np.any(np.isfinite(cand)):
                    sigma2d = cand
                else:
                    sigma2d = None
    
        return mean2d, sigma2d
    
    
    def _uq__get_field_for_plot(
        self,
        *,
        version,
        imt,
        method,
        what,
        compute_if_missing=False,
        method_kwargs=None,
    ):
        """
        Canonical plot accessor.
    
        For published ShakeMap:
          - mean always from unified fields[IMT]
          - sigma from unified sigma[STD<IMT>]
          - if mean exists but sigma is unavailable -> status="missing_uncertainty" (explicit)
    
        For non-published methods:
          - defer to existing self._uq__get_method_grids() contract (compute/store if needed)
    
        Returns:
          Z, meta, status, mu2d, sig2d
        """
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
        method_u = str(method).strip()
        what_u = str(what).lower().strip()
    
        def _is_shakemap(m: str) -> bool:
            mk = str(m).lower().strip().replace(" ", "")
            return mk in ("shakemap", "published", "raw")
    
        # -------------------------
        # Published ShakeMap branch
        # -------------------------
        if _is_shakemap(method_u):
            mu2d, sig2d = self._uq__get_shakemap_unified_fields(vkey, imt_u)
    
            if what_u == "mean":
                Z = mu2d
            elif what_u == "sigma":
                Z = sig2d
            else:
                return (
                    None,
                    {"method": "ShakeMap", "imt": imt_u, "version": vkey, "what": what_u, "reason": "unknown_what"},
                    "missing",
                    mu2d,
                    sig2d,
                )
    
            if what_u == "sigma" and (mu2d is not None) and (Z is None):
                status = "missing_uncertainty"
                meta = {
                    "method": "ShakeMap",
                    "imt": imt_u,
                    "version": vkey,
                    "what": what_u,
                    "reason": "published_sigma_not_available",
                }
            else:
                status = "ok" if Z is not None else "missing"
                meta = {"method": "ShakeMap", "imt": imt_u, "version": vkey, "what": what_u}
    
            return Z, meta, status, mu2d, sig2d
    
        # -------------------------
        # Method branch (rk/bayes/ens/...)
        # -------------------------
        if not hasattr(self, "_uq__get_method_grids"):
            raise RuntimeError("Missing _uq__get_method_grids() implementation; cannot fetch non-ShakeMap method grids.")
    
        Z, meta, status, mu2d, sig2d = self._uq__get_method_grids(
            version=vkey,
            imt=imt_u,
            method=method_u,
            what=what_u,
            compute_if_missing=bool(compute_if_missing),
            method_kwargs=method_kwargs,
        )
        return Z, meta, status, mu2d, sig2d
    


    def _uq__get_field_for_plot(
        self,
        *,
        version,
        imt,
        method,
        what,
        compute_if_missing=False,
        method_kwargs=None,
    ):
        """
        Canonical accessor used by plotting helpers.
    
        IMPORTANT: This implementation matches the *actual* signature of
        self._uq__get_method_grids() in your codebase:
    
            _uq__get_method_grids(*, vkey, method, imt, compute_if_missing=False, method_kwargs=None, debug=False)
            -> (mean2d, sigma2d, meta, status)
    
        It then selects Z based on `what` ("mean" or "sigma") and returns:
    
            (Z, meta, status, mu2d, sig2d)
    
        Philosophy preserved:
          - If published ShakeMap mean exists but published sigma does not,
            we do NOT fabricate sigma and do NOT silently NaN-fill.
            We return status="missing_uncertainty" and a reason in meta.
        """
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
        method_u = str(method).strip()
        what_u = str(what).lower().strip()
    
        if not hasattr(self, "_uq__get_method_grids"):
            raise RuntimeError(
                "Missing _uq__get_method_grids() implementation; cannot fetch grids for plotting."
            )
    
        # NOTE: _uq__get_method_grids returns (mu2d, sig2d, meta, status)
        mu2d, sig2d, meta, status = self._uq__get_method_grids(
            vkey=vkey,
            method=method_u,
            imt=imt_u,
            compute_if_missing=bool(compute_if_missing),
            method_kwargs=(method_kwargs or None),
            debug=False,
        )
    
        # Choose requested field
        if what_u == "mean":
            Z = mu2d
        elif what_u == "sigma":
            Z = sig2d
        else:
            # Unknown requested field
            meta = dict(meta or {})
            meta["what"] = what_u
            meta["reason"] = "unknown_what"
            return None, meta, "missing", mu2d, sig2d
    
        # Make missing-uncertainty explicit for published ShakeMap
        mk = str((meta or {}).get("method", "")).lower().strip().replace(" ", "")
        is_published = mk in ("shakemap", "published", "raw")
    
        if is_published and what_u == "sigma" and (mu2d is not None) and (Z is None):
            meta = dict(meta or {})
            meta["what"] = what_u
            meta["reason"] = "published_sigma_not_available"
            return None, meta, "missing_uncertainty", mu2d, sig2d
    
        # Otherwise, keep the underlying status unless the requested Z is missing
        if Z is None:
            meta = dict(meta or {})
            meta["what"] = what_u
            meta.setdefault("reason", "requested_field_missing")
            return None, meta, "missing", mu2d, sig2d
    
        meta = dict(meta or {})
        meta["what"] = what_u
        return Z, meta, "ok", mu2d, sig2d



    def _plot_slot(ax, Z, *, cmap, norm=None, vmin=None, vmax=None, ttl=""):
        """
        Matplotlib >= 3.8: cannot pass (norm) together with (vmin/vmax).
        - If norm is provided, we do NOT pass vmin/vmax.
        - If norm is None, we pass vmin/vmax as usual.
        """
        Zarr = np.asarray(Z, float)
    
        pkm_kwargs = dict(
            transform=tr,
            cmap=cmap,
            shading="auto",
            zorder=zdef.get("field", 2),
        )
    
        if norm is not None:
            pkm_kwargs["norm"] = norm
            # IMPORTANT: do not pass vmin/vmax when norm is supplied
        else:
            if vmin is not None:
                pkm_kwargs["vmin"] = vmin
            if vmax is not None:
                pkm_kwargs["vmax"] = vmax
    
        pm = ax.pcolormesh(lon2d, lat2d, Zarr, **pkm_kwargs)
    
        ax.set_title(str(ttl), fontsize=float(title_fs))
        if show_obs or show_rupture:
            pass  # keep your existing overlay block below this unchanged
    
        return pm





    def _uq__get_field_for_plot(
        self,
        *,
        version,
        imt,
        method,
        what,
        compute_if_missing=False,
        method_kwargs=None,
    ):
        """
        Canonical plot accessor.
    
        Returns
        -------
        (Z, meta, status, mu2d, sig2d)
    
        Philosophy:
          - Missing uncertainty is information.
          - If the requested field does not exist, return Z=None with status != "ok".
          - Callers decide whether to raise, skip, or downgrade a panel.
        """
        if not getattr(self, "uq_state", None):
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")
    
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
        what_u = str(what).lower().strip()
        method_u = str(method).strip()
    
        # -----------------------------
        # 1) ShakeMap (published/raw)
        # -----------------------------
        mkey = str(method_u).lower().strip().replace(" ", "")
        if mkey in ("shakemap", "published", "raw"):
            # Preferred: per-version published fields
            pack = (self.uq_state.get("versions", {}) or {}).get(vkey, {}) or {}
            grids = pack.get("grids", {}) or {}
    
            # Mean should exist for MMI/PGA if parsing worked
            mu2d = grids.get("mean", {}).get(imt_u, None)
            sig2d = grids.get("sigma", {}).get(imt_u, None)
    
            # Fallback: unified stacks (already on canonical grid)
            if mu2d is None:
                try:
                    mu2d = (self.uq_state.get("unified", {}) or {}).get("mean", {}).get(imt_u, {}).get(vkey, None)
                except Exception:
                    mu2d = None
            if sig2d is None:
                try:
                    sig2d = (self.uq_state.get("unified", {}) or {}).get("sigma", {}).get(imt_u, {}).get(vkey, None)
                except Exception:
                    sig2d = None
    
            # Minimal meta
            meta = {
                "source": "ShakeMap",
                "version": vkey,
                "imt": imt_u,
                "what": what_u,
            }
    
            if what_u == "mean":
                if mu2d is None:
                    return None, meta, "missing", None, sig2d
                return mu2d, meta, "ok", mu2d, sig2d
    
            if what_u == "sigma":
                # critical: do NOT fabricate sigma
                if sig2d is None:
                    return None, meta, "missing_uncertainty", mu2d, None
                return sig2d, meta, "ok", mu2d, sig2d
    
            raise ValueError(f"Unknown what='{what}'. Use 'mean' or 'sigma'.")
    
        # -----------------------------
        # 2) Other methods (bayes/rk/ens/...)
        # -----------------------------
        if not hasattr(self, "_uq__get_method_grids"):
            raise RuntimeError("Missing _uq__get_method_grids() implementation; cannot fetch method grids.")
    
        # IMPORTANT: your canonical internal call uses vkey= (NOT version=)
        mu2d, sig2d, meta_m, status = self._uq__get_method_grids(
            vkey=vkey,
            method=method_u,
            imt=imt_u,
            compute_if_missing=bool(compute_if_missing),
            method_kwargs=method_kwargs,
            debug=False,
        )
    
        meta = meta_m if isinstance(meta_m, dict) else {}
        meta.setdefault("version", vkey)
        meta.setdefault("imt", imt_u)
        meta.setdefault("method", method_u)
    
        if str(status).lower().strip() != "ok":
            return None, meta, status, mu2d, sig2d
    
        if what_u == "mean":
            if mu2d is None:
                return None, meta, "missing", None, sig2d
            return mu2d, meta, "ok", mu2d, sig2d
    
        if what_u == "sigma":
            if sig2d is None:
                return None, meta, "missing_uncertainty", mu2d, None
            return sig2d, meta, "ok", mu2d, sig2d
    
        raise ValueError(f"Unknown what='{what}'. Use 'mean' or 'sigma'.")
    
    
    
    
    
    
    
    
    
    def uq_plot_method_map_panel(
        self,
        *,
        version,
        imt,
        method,
        method_kwargs=None,
        compute_if_missing=False,
        include_deltas=True,
        # overlays
        show_obs=True,
        obs_size=10.0,
        obs_kwargs=None,
        show_rupture=True,
        rupture_kwargs=None,
        # basemap
        zorders=None,
        use_utm=True,
        add_ocean=True,
        add_land=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=False,
        # colorbars
        mean_scale_type="usgs",
        mean_pga_units="%g",
        custom_colorbar_mean=False,
        custom_colorbar_sigma=False,
        colorbar_shrink=0.70,
        colorbar_kwargs=None,
        colorbar_label_mean=None,
        colorbar_label_sigma=None,
        colorbar_label_delta="Δmean",
        colorbar_label_delta_sigma="Δσ",
        # text/figure
        title_fs=12,
        label_fs=10,
        tick_fs=9,
        figsize=(12.0, 7.5),
        dpi=300,
        output_path=None,
        save=False,
        save_formats=("png",),
        show=True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
    
        if not getattr(self, "uq_state", None):
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")
    
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
        method_u = str(method).strip()
    
        lon2d, lat2d = self._uq__get_unified_grid_safe()
        extent = self._get_grid_extent(vkey, grid_mode="unified", margin_deg=0.0)
    
        # Fetch fields
        Zs_mean, _, status_sm_m, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method="ShakeMap", what="mean",
            compute_if_missing=False, method_kwargs=None
        )
        Zs_sig, _, status_sm_s, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method="ShakeMap", what="sigma",
            compute_if_missing=False, method_kwargs=None
        )
        Zm_mean, meta_m, status_m_m, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method=method_u, what="mean",
            compute_if_missing=bool(compute_if_missing), method_kwargs=method_kwargs
        )
        Zm_sig, meta_s, status_m_s, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method=method_u, what="sigma",
            compute_if_missing=bool(compute_if_missing), method_kwargs=method_kwargs
        )
    
        if Zs_mean is None or str(status_sm_m).lower() != "ok":
            raise RuntimeError(f"Panel requires ShakeMap mean, but got status={status_sm_m}.")
        if Zm_mean is None or str(status_m_m).lower() != "ok":
            raise RuntimeError(f"Panel requires method mean, but got status={status_m_m}.")
    
        # If any sigma missing, we downgrade to mean-only row and drop deltas row/col as needed.
        have_sigma = (Zs_sig is not None and str(status_sm_s).lower() == "ok" and
                      Zm_sig is not None and str(status_m_s).lower() == "ok")
        if not have_sigma:
            include_deltas = False
    
        ncols = 3 if include_deltas else 2
        nrows = 2 if have_sigma else 1
    
        # Decide projection once using existing helper
        ftmp, axtmp, used_cartopy, ccrs, zdef = self._uq__cartopy_axes(
            figsize=(1, 1),
            extent=extent,
            use_utm=bool(use_utm),
            zorders=zorders,
            add_ocean=False,
            add_land=False,
            add_borders=False,
            add_coastlines=False,
            add_gridlines=False,
        )
        proj = getattr(axtmp, "projection", None)
        plt.close(ftmp)
    
        fig = plt.figure(figsize=figsize, dpi=dpi)
        axs = []
        for r in range(nrows):
            for c in range(ncols):
                idx = r * ncols + c + 1
                if used_cartopy and proj is not None:
                    ax = fig.add_subplot(nrows, ncols, idx, projection=proj)
                else:
                    ax = fig.add_subplot(nrows, ncols, idx)
                axs.append(ax)
    
        tr = ccrs.PlateCarree() if (used_cartopy and ccrs is not None) else None
    
        def _add_base(ax):
            if not used_cartopy or ccrs is None:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_xlabel("Longitude", fontsize=float(label_fs))
                ax.set_ylabel("Latitude", fontsize=float(label_fs))
                return
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            try:
                import cartopy.feature as cfeature
                if add_ocean: ax.add_feature(cfeature.OCEAN, zorder=zdef.get("ocean", 0))
                if add_land: ax.add_feature(cfeature.LAND, zorder=zdef.get("land", 1))
                if add_borders: ax.add_feature(cfeature.BORDERS, zorder=zdef.get("borders", 4), linewidth=0.6)
                if add_coastlines: ax.coastlines(zorder=zdef.get("coastlines", 5), linewidth=0.6)
                if add_gridlines:
                    ax.gridlines(draw_labels=False, zorder=zdef.get("gridlines", 6), linewidth=0.5, alpha=0.35)
            except Exception:
                pass
    
        def _plot_slot(ax, Z, *, cmap, norm, ttl="", add_cb=True, cb_label=None):
            _add_base(ax)
    
            # IMPORTANT: if norm is provided, DO NOT pass vmin/vmax to pcolormesh
            pm = ax.pcolormesh(
                lon2d, lat2d, np.asarray(Z, float),
                transform=tr,
                cmap=cmap,
                norm=norm,
                shading="auto",
                zorder=zdef.get("field", 2),
            )
            ax.set_title(str(ttl), fontsize=float(title_fs))
    
            # overlays
            if show_rupture:
                try:
                    self._uq__overlay_rupture(ax=ax, version=vkey, used_cartopy=used_cartopy, ccrs=ccrs, **(rupture_kwargs or {}))
                except Exception:
                    pass
            if show_obs:
                try:
                    self._uq__overlay_observations(ax=ax, version=vkey, imt=imt_u, used_cartopy=used_cartopy, ccrs=ccrs,
                                                  size=float(obs_size), **(obs_kwargs or {}))
                except Exception:
                    pass
    
            if add_cb:
                try:
                    cbar = fig.colorbar(pm, ax=ax, shrink=float(colorbar_shrink), **(colorbar_kwargs or {}))
                    if cb_label:
                        cbar.set_label(str(cb_label), fontsize=float(label_fs))
                    cbar.ax.tick_params(labelsize=float(tick_fs))
                except Exception:
                    pass
            return pm
    
        # Resolve colorbars using your existing helper
        cmap_sm_m, norm_sm_m, *_ = self._uq__resolve_colorbar_spec(
            imt=imt_u, what="mean", scale_type=mean_scale_type, pga_units=mean_pga_units,
            cmap=None, norm=None, ticks=None, bounds=None, vmin=None, vmax=None,
            custom_colorbar=bool(custom_colorbar_mean),
        )
        cmap_sm_s, norm_sm_s, *_ = self._uq__resolve_colorbar_spec(
            imt=imt_u, what="sigma", scale_type=mean_scale_type, pga_units=mean_pga_units,
            cmap=None, norm=None, ticks=None, bounds=None, vmin=None, vmax=None,
            custom_colorbar=bool(custom_colorbar_sigma),
        )
    
        # Row 0: means
        _plot_slot(axs[0], Zs_mean, cmap=cmap_sm_m, norm=norm_sm_m,
                   ttl=f"ShakeMap mean (v{vkey})", cb_label=colorbar_label_mean)
        _plot_slot(axs[1], Zm_mean, cmap=cmap_sm_m, norm=norm_sm_m,
                   ttl=f"{method_u} mean (v{vkey})", cb_label=colorbar_label_mean)
    
        if include_deltas:
            dmean = np.asarray(Zm_mean, float) - np.asarray(Zs_mean, float)
            vmin_d, vmax_d = self._uq__autosym_limits(dmean)
            dn = self._uq__two_slope_norm(vmin_d, vmax_d, vcenter=0.0)
            _plot_slot(axs[2], dmean, cmap="seismic", norm=dn,
                       ttl=f"Δmean ({method_u} − ShakeMap)", cb_label=colorbar_label_delta)
    
        # Row 1: sigmas (only if both available)
        if have_sigma:
            base = ncols
            _plot_slot(axs[base + 0], Zs_sig, cmap=cmap_sm_s, norm=norm_sm_s,
                       ttl=f"ShakeMap σ (v{vkey})", cb_label=colorbar_label_sigma)
            _plot_slot(axs[base + 1], Zm_sig, cmap=cmap_sm_s, norm=norm_sm_s,
                       ttl=f"{method_u} σ (v{vkey})", cb_label=colorbar_label_sigma)
    
            if include_deltas:
                dsig = np.asarray(Zm_sig, float) - np.asarray(Zs_sig, float)
                vmin_d, vmax_d = self._uq__autosym_limits(dsig)
                dn = self._uq__two_slope_norm(vmin_d, vmax_d, vcenter=0.0)
                _plot_slot(axs[base + 2], dsig, cmap="seismic", norm=dn,
                           ttl=f"Δσ ({method_u} − ShakeMap)", cb_label=colorbar_label_delta_sigma)
    
        # Save/show hook (keep your existing conventions lightly)
        rec = {
            "version": vkey,
            "imt": imt_u,
            "method": method_u,
            "have_sigma": bool(have_sigma),
            "include_deltas": bool(include_deltas),
            "status_sm_mean": str(status_sm_m),
            "status_sm_sigma": str(status_sm_s),
            "status_m_mean": str(status_m_m),
            "status_m_sigma": str(status_m_s),
        }
    
        if save:
            try:
                self._uq__save_figure_bundle(
                    fig=fig,
                    output_path=output_path,
                    base_name=f"panel__v{vkey}__{imt_u}__{method_u}".replace(" ", ""),
                    save_formats=save_formats,
                )
            except Exception:
                pass
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig, axs, rec
    
    
    
    
    
    
    def uq_plot_difference_map(
        self,
        *,
        imt,
        what,
        method="ShakeMap",
        mode="incremental",
        v_from=None,
        v_to=None,
        compute_if_missing=False,
        method_kwargs=None,
        cmap="seismic",
        vmin=None,
        vmax=None,
        # basemap
        zorders=None,
        use_utm=True,
        add_ocean=True,
        add_land=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=False,
        # overlays
        show_obs=False,
        obs_size=10.0,
        obs_kwargs=None,
        show_rupture=True,
        rupture_kwargs=None,
        # fig/output
        title=None,
        figsize=(10.0, 7.5),
        dpi=300,
        output_path=None,
        save=False,
        save_formats=("png",),
        show=True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
    
        if not getattr(self, "uq_state", None):
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")
    
        imt_u = str(imt).upper().strip()
        what_u = str(what).lower().strip()
        method_u = str(method).strip()
        mode_u = str(mode).lower().strip()
    
        vlist = (self.uq_state.get("version_list") or [])
        if not vlist:
            raise RuntimeError("uq_state['version_list'] missing.")
    
        if v_to is None:
            v_to = vlist[-1]
        if mode_u == "cumulative_from_first":
            v_from = vlist[0]
        elif v_from is None:
            v_from = vlist[0]
    
        v_to_k = self._uq__norm_version_safe(v_to)
        v_from_k = self._uq__norm_version_safe(v_from)
    
        Zfrom, _, status_f, *_ = self._uq__get_field_for_plot(
            version=v_from_k, imt=imt_u, method=method_u, what=what_u,
            compute_if_missing=bool(compute_if_missing), method_kwargs=method_kwargs
        )
        Zto, _, status_t, *_ = self._uq__get_field_for_plot(
            version=v_to_k, imt=imt_u, method=method_u, what=what_u,
            compute_if_missing=bool(compute_if_missing), method_kwargs=method_kwargs
        )
        if Zfrom is None or str(status_f).lower() != "ok" or Zto is None or str(status_t).lower() != "ok":
            raise RuntimeError(f"Difference fields missing (from/to). status_from={status_f}, status_to={status_t}")
    
        d = np.asarray(Zto, float) - np.asarray(Zfrom, float)
    
        if vmin is None or vmax is None:
            vmin_, vmax_ = self._uq__autosym_limits(d)
            vmin = vmin if vmin is not None else vmin_
            vmax = vmax if vmax is not None else vmax_
        dn = self._uq__two_slope_norm(float(vmin), float(vmax), vcenter=0.0)
    
        lon2d, lat2d = self._uq__get_unified_grid_safe()
        extent = self._get_grid_extent(v_to_k, grid_mode="unified", margin_deg=0.0)
    
        fig, ax, used_cartopy, ccrs, zdef = self._uq__cartopy_axes(
            figsize=figsize,
            extent=extent,
            use_utm=bool(use_utm),
            zorders=zorders,
            add_ocean=bool(add_ocean),
            add_land=bool(add_land),
            add_borders=bool(add_borders),
            add_coastlines=bool(add_coastlines),
            add_gridlines=bool(add_gridlines),
        )
        tr = ccrs.PlateCarree() if (used_cartopy and ccrs is not None) else None
    
        # IMPORTANT: norm provided => do NOT pass vmin/vmax to pcolormesh
        pm = ax.pcolormesh(
            lon2d, lat2d, d,
            transform=tr,
            cmap=str(cmap),
            norm=dn,
            shading="auto",
            zorder=zdef.get("field", 2),
        )
    
        if title is None:
            title = f"Δ{what_u} ({method_u}) : v{v_to_k} − v{v_from_k}"
        ax.set_title(str(title))
    
        if show_rupture:
            try:
                self._uq__overlay_rupture(ax=ax, version=v_to_k, used_cartopy=used_cartopy, ccrs=ccrs, **(rupture_kwargs or {}))
            except Exception:
                pass
        if show_obs:
            try:
                self._uq__overlay_observations(ax=ax, version=v_to_k, imt=imt_u, used_cartopy=used_cartopy, ccrs=ccrs,
                                              size=float(obs_size), **(obs_kwargs or {}))
            except Exception:
                pass
    
        try:
            cbar = fig.colorbar(pm, ax=ax, shrink=0.86)
            cbar.set_label(f"Δ{what_u}")
        except Exception:
            pass
    
        if save:
            try:
                self._uq__save_figure_bundle(
                    fig=fig,
                    output_path=output_path,
                    base_name=f"diff__{method_u}__{what_u}__{imt_u}__v{v_to_k}-v{v_from_k}".replace(" ", ""),
                    save_formats=save_formats,
                )
            except Exception:
                pass
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        rec = dict(
            version_from=v_from_k,
            version_to=v_to_k,
            imt=imt_u,
            method=method_u,
            what=what_u,
            vmin=float(vmin),
            vmax=float(vmax),
        )
        return fig, ax, rec






    def _uq__get_shakemap_unified_fields(self, version, imt: str):
        """
        Published ShakeMap accessor on the unified grid.
    
        Returns
        -------
        (mean2d, sigma2d)
    
        Storage (authoritative):
          - mean stack:  uq_state["unified"]["fields"][IMT]     (e.g., "MMI", "PGA")
          - sigma stack: uq_state["unified"]["sigma"][STD<IMT>] (e.g., "STDMMI", "STDPGA")
          - version index: uq_state["unified"]["version_keys"]
        """
        import numpy as np
    
        if not getattr(self, "uq_state", None):
            raise RuntimeError("uq_state missing. Run uq_build_dataset() first.")
    
        u = (self.uq_state.get("unified") or {})
        if not u:
            raise RuntimeError("uq_state['unified'] missing/empty. Unified stacks not built.")
    
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
    
        vkeys = u.get("version_keys") or []
        if not vkeys:
            raise RuntimeError("uq_state['unified']['version_keys'] missing/empty.")
        vkeys_n = [self._uq__norm_version_safe(v) for v in list(vkeys)]
        if vkey not in vkeys_n:
            raise KeyError(f"Version {vkey} not found in uq_state['unified']['version_keys'].")
    
        vidx = int(vkeys_n.index(vkey))
    
        fields = (u.get("fields") or {})
        sigmas = (u.get("sigma") or u.get("sigmas") or {})
    
        # ---- mean ----
        if imt_u not in fields:
            # This is a true missing published mean for that IMT
            return None, None
    
        mean_stack = np.asarray(fields[imt_u], float)
        if mean_stack.ndim < 3:
            raise RuntimeError(f"Unified mean stack for {imt_u} has unexpected shape: {mean_stack.shape}")
        mean2d = mean_stack[vidx, :, :]
    
        # ---- sigma ----
        sigma2d = None
        if isinstance(sigmas, dict) and sigmas:
            # canonical sigma key: STD<IMT>
            sig_key = self._bayes_sigma_key_for_imt(imt_u) if hasattr(self, "_bayes_sigma_key_for_imt") else f"STD{imt_u}"
            if sig_key in sigmas:
                sig_stack = np.asarray(sigmas[sig_key], float)
                if sig_stack.ndim >= 3:
                    cand = sig_stack[vidx, :, :]
                else:
                    cand = sig_stack[vidx, ...]
                # treat all-NaN as "missing uncertainty"
                if np.size(cand) > 0 and np.any(np.isfinite(cand)):
                    sigma2d = cand
                else:
                    sigma2d = None
    
        # treat all-NaN mean as missing
        if mean2d is None or not (np.size(mean2d) > 0 and np.any(np.isfinite(mean2d))):
            mean2d = None
    
        return mean2d, sigma2d







    def _uq__get_field_for_plot(
        self,
        *,
        version,
        imt,
        method,
        what,
        compute_if_missing=False,
        method_kwargs=None,
    ):
        """
        Canonical plot accessor used by uq_plot_method_map / panel / diffs.
    
        Returns:
          (Z, meta, status, mu2d, sig2d)
    
        Key rules:
          - ShakeMap mean/sigma come from unified stacks via _uq__get_shakemap_unified_fields()
          - sigma missing is reported explicitly as "missing_uncertainty"
          - non-ShakeMap methods are fetched via _uq__get_method_grids(vkey=..., ...) which returns
            (mu2d, sig2d, meta, status)
        """
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
        what_u = str(what).lower().strip()
        method_u = str(method).strip()
    
        def _is_shakemap(m: str) -> bool:
            mk = str(m).lower().strip().replace(" ", "")
            return mk in ("shakemap", "published", "raw")
    
        # -------------------------
        # Published ShakeMap
        # -------------------------
        if _is_shakemap(method_u):
            mu2d, sig2d = self._uq__get_shakemap_unified_fields(vkey, imt_u)
    
            meta = {"method": "ShakeMap", "imt": imt_u, "version": vkey, "what": what_u}
    
            if what_u == "mean":
                if mu2d is None:
                    return None, meta, "missing", None, sig2d
                return mu2d, meta, "ok", mu2d, sig2d
    
            if what_u == "sigma":
                if sig2d is None:
                    # mean may exist even if sigma doesn't
                    meta = dict(meta)
                    meta["reason"] = "published_sigma_not_available"
                    return None, meta, "missing_uncertainty", mu2d, None
                return sig2d, meta, "ok", mu2d, sig2d
    
            raise ValueError(f"Unknown what='{what}'. Use 'mean' or 'sigma'.")
    
        # -------------------------
        # Method fields (bayes/rk/ens/...)
        # -------------------------
        if not hasattr(self, "_uq__get_method_grids"):
            raise RuntimeError("Missing _uq__get_method_grids() implementation; cannot fetch non-ShakeMap grids.")
    
        # IMPORTANT: your _uq__get_method_grids() expects vkey=..., NOT version=...
        mu2d, sig2d, meta_m, status = self._uq__get_method_grids(
            vkey=vkey,
            method=method_u,
            imt=imt_u,
            compute_if_missing=bool(compute_if_missing),
            method_kwargs=method_kwargs,
            debug=False,
        )
    
        meta = meta_m if isinstance(meta_m, dict) else {}
        meta.setdefault("method", method_u)
        meta.setdefault("imt", imt_u)
        meta.setdefault("version", vkey)
        meta["what"] = what_u
    
        if str(status).lower().strip() != "ok":
            return None, meta, status, mu2d, sig2d
    
        if what_u == "mean":
            if mu2d is None:
                return None, meta, "missing", None, sig2d
            return mu2d, meta, "ok", mu2d, sig2d
    
        if what_u == "sigma":
            if sig2d is None:
                meta = dict(meta)
                meta.setdefault("reason", "method_sigma_missing")
                return None, meta, "missing_uncertainty", mu2d, None
            return sig2d, meta, "ok", mu2d, sig2d
    
        raise ValueError(f"Unknown what='{what}'. Use 'mean' or 'sigma'.")







    def uq_plot_method_map_panel(
        self,
        *,
        version,
        imt: str = "MMI",
        method: str = "bayes",
        method_kwargs: dict = None,
        compute_if_missing: bool = False,
        include_deltas: bool = True,
    
        # overlays
        show_obs: bool = True,
        obs_size: float = 10.0,
        obs_kwargs: dict = None,
        show_rupture: bool = True,
        rupture_kwargs: dict = None,
    
        # base features + zorders
        zorders: dict = None,
        use_utm: bool = True,
        add_ocean: bool = True,
        add_land: bool = True,
        add_borders: bool = True,
        add_coastlines: bool = True,
        add_gridlines: bool = True,
    
        # colorbar controls
        mean_scale_type: str = "usgs",
        mean_pga_units: str = "%g",
        custom_colorbar_mean=None,
        custom_colorbar_sigma=None,
        colorbar_shrink: float = 0.70,
        colorbar_kwargs: dict = None,
        colorbar_label_mean: str = None,
        colorbar_label_sigma: str = None,
        colorbar_label_delta: str = None,
        colorbar_label_delta_sigma: str = None,
    
        # figure
        title_fs: float = 11.0,
        label_fs: float = 10.0,
        tick_fs: float = 9.0,
        figsize=(13.0, 7.0),
        dpi: int = 200,
    
        # output
        output_path: str = None,
        save: bool = True,
        save_formats=("png",),
        show: bool = True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
    
        vkey = self._uq__norm_version_safe(version)
        imt_u = str(imt).upper().strip()
        method_u = str(method).strip()
    
        lon2d, lat2d = self._uq__get_unified_grid_safe()
        extent = self._get_grid_extent(vkey, grid_mode="unified", margin_deg=0.0)
    
        # --- Fetch fields ---
        Zs_mean, _, status_s_m, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method="ShakeMap", what="mean"
        )
        Zs_sig, _, status_s_s, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method="ShakeMap", what="sigma"
        )
        Zm_mean, _, status_m_m, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method=method_u, what="mean",
            compute_if_missing=bool(compute_if_missing), method_kwargs=method_kwargs
        )
        Zm_sig, _, status_m_s, *_ = self._uq__get_field_for_plot(
            version=vkey, imt=imt_u, method=method_u, what="sigma",
            compute_if_missing=bool(compute_if_missing), method_kwargs=method_kwargs
        )
    
        if Zs_mean is None or str(status_s_m).lower() != "ok":
            raise RuntimeError(f"Panel requires ShakeMap mean; status={status_s_m}")
        if Zm_mean is None or str(status_m_m).lower() != "ok":
            raise RuntimeError(f"Panel requires {method_u} mean; status={status_m_m}")
    
        have_sigma = (
            Zs_sig is not None and str(status_s_s).lower() == "ok" and
            Zm_sig is not None and str(status_m_s).lower() == "ok"
        )
        if not have_sigma:
            include_deltas = False
    
        ncols = 3 if include_deltas else 2
        nrows = 2 if have_sigma else 1
    
        ftmp, axtmp, used_cartopy, ccrs, zdef = self._uq__cartopy_axes(
            figsize=(1, 1),
            extent=extent,
            use_utm=bool(use_utm),
            zorders=zorders,
            add_ocean=False, add_land=False,
            add_borders=False, add_coastlines=False, add_gridlines=False,
        )
        proj = getattr(axtmp, "projection", None)
        plt.close(ftmp)
    
        fig = plt.figure(figsize=figsize, dpi=dpi)
        axs = []
        for i in range(nrows * ncols):
            if used_cartopy and proj is not None:
                ax = fig.add_subplot(nrows, ncols, i + 1, projection=proj)
            else:
                ax = fig.add_subplot(nrows, ncols, i + 1)
            axs.append(ax)
    
        tr = ccrs.PlateCarree() if used_cartopy and ccrs else None
    
        def _plot(ax, Z, *, cmap, norm=None, ttl=""):
            if Z is None:
                ax.set_title(ttl)
                return None
            kw = dict(cmap=cmap, shading="auto", zorder=zdef.get("field", 2))
            if tr is not None:
                kw["transform"] = tr
            if norm is not None:
                kw["norm"] = norm
            pm = ax.pcolormesh(lon2d, lat2d, np.asarray(Z, float), **kw)
            ax.set_title(ttl)
            return pm
    
        cmap_m, norm_m, *_ = self._uq__resolve_colorbar_spec(
            imt=imt_u, what="mean",
            mean_scale_type=mean_scale_type, mean_pga_units=mean_pga_units,
            custom_colorbar=custom_colorbar_mean,
        )
        cmap_s, norm_s, *_ = self._uq__resolve_colorbar_spec(
            imt=imt_u, what="sigma",
            mean_scale_type=mean_scale_type, mean_pga_units=mean_pga_units,
            custom_colorbar=custom_colorbar_sigma,
        )
    
        _plot(axs[0], Zs_mean, cmap=cmap_m, norm=norm_m, ttl=f"ShakeMap mean (v{vkey})")
        _plot(axs[1], Zm_mean, cmap=cmap_m, norm=norm_m, ttl=f"{method_u} mean (v{vkey})")
    
        if include_deltas:
            dmean = Zm_mean - Zs_mean
            dn = self._uq__two_slope_norm(*self._uq__autosym_limits(dmean), vcenter=0.0)
            _plot(axs[2], dmean, cmap="seismic", norm=dn, ttl=f"Δmean ({method_u} − ShakeMap)")
    
        if have_sigma:
            base = ncols
            _plot(axs[base + 0], Zs_sig, cmap=cmap_s, norm=norm_s, ttl=f"ShakeMap σ (v{vkey})")
            _plot(axs[base + 1], Zm_sig, cmap=cmap_s, norm=norm_s, ttl=f"{method_u} σ (v{vkey})")
    
            if include_deltas:
                ds = Zm_sig - Zs_sig
                dn2 = self._uq__two_slope_norm(*self._uq__autosym_limits(ds), vcenter=0.0)
                _plot(axs[base + 2], ds, cmap="seismic", norm=dn2, ttl=f"Δσ ({method_u} − ShakeMap)")
    
        fig.tight_layout()
    
        # ✅ NOTE: no "kind" here anymore
        rec = dict(
            version=vkey,
            imt=imt_u,
            method=method_u,
            include_deltas=bool(include_deltas),
            have_sigma=bool(have_sigma),
            status=dict(
                shakemap_mean=status_s_m,
                shakemap_sigma=status_s_s,
                method_mean=status_m_m,
                method_sigma=status_m_s,
            ),
        )
    
        if save and output_path:
            self._uq__save_figure_bundle(
                fig=fig,
                output_path=output_path,
                base_name=f"panel__v{vkey}__{imt_u}__{method_u}".replace(" ", ""),
                save_formats=save_formats,
            )
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig, axs, rec
    
    
    
    
































