# SHAKEuq.py
# Fresh, traceable SHAKEuq rebuild (inspired by SHAKEuq_legacy, but clean architecture)
# - Parses ShakeMap grid.xml + uncertainty.xml
# - Parses stationlist.json (instrumented + DYFI-in-stationlist) via USGSParser
# - Parses CDI file via USGSParser(dyfi_data)
# - Builds per-version store + unified grids + sanity table
# - Keeps ALL station/cdi dataframe columns (no cropping)
# - Supports event_id="simulation" to generate synthetic dataset (expand later)

from __future__ import annotations

import os
import re
import glob
import math
import json
import shutil
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import os
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import numpy as np
import statistics
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import logging

from typing import List, Optional, Tuple, Dict, Any
from itertools import combinations
import pickle

import matplotlib.ticker as mticker
from cartopy.crs import PlateCarree

from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from sklearn.neighbors import BallTree
from scipy.spatial import cKDTree

import geopandas as gpd
from shapely.geometry import Point, Polygon
from sklearn.metrics import mean_squared_error, mean_absolute_error

from nolds import lyap_r, corr_dim, sampen
from sklearn.preprocessing import StandardScaler
import joblib

from scipy.spatial.distance import cdist

from scipy import stats
from scipy.stats import norm, lognorm, gamma, gumbel_r
from sklearn.decomposition import PCA
import seaborn as sns

# local modules import
from modules.SHAKEparser import *
from modules.SHAKEmapper import *
from modules.SHAKEtools import *
from modules.SHAKEgmice import *

# -----------------------------
# Small utilities
# -----------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _parse_iso_utc(ts: Optional[str]) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    try:
        # USGS timestamps are often "YYYY-MM-DDTHH:MM:SS"
        return pd.to_datetime(ts, utc=True)
    except Exception:
        return None


def _xml_tag_endswith(elem: ET.Element, suffix: str) -> bool:
    return elem.tag.endswith(suffix)


def _xml_find_first(root: ET.Element, suffix: str) -> Optional[ET.Element]:
    for e in root.iter():
        if _xml_tag_endswith(e, suffix):
            return e
    return None


def _xml_find_all(root: ET.Element, suffix: str) -> List[ET.Element]:
    out = []
    for e in root.iter():
        if _xml_tag_endswith(e, suffix):
            out.append(e)
    return out


def _coerce_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _guess_time_after_event_hours(event_time_utc: Optional[pd.Timestamp],
                                  process_time_utc: Optional[pd.Timestamp]) -> Optional[float]:
    if event_time_utc is None or process_time_utc is None:
        return None
    try:
        dt = (process_time_utc - event_time_utc).total_seconds() / 3600.0
        return float(dt)
    except Exception:
        return None


# -----------------------------
# SHAKEuq
# -----------------------------

class SHAKEuq:
    """
    Fresh SHAKEuq rebuild.

    State layout (self.uq_state):
      {
        "config": {...},
        "versions": {
            "001": {
                "paths": {...},
                "exists": {...},
                "meta": {...},         # parsed shakemap_grid attrs + <event> attrs + timestamps
                "grid": {...},         # mean fields 2D arrays, units, spec, orientation
                "uncert": {...},       # sigma fields 2D arrays, units, spec (aligned)
                "stations": {...},     # full dataframes: instruments + dyfi_stationlist
                "cdi": {...},          # full dataframe (if used)
                "obs_audit": {...},    # counts and notes
            },
            ...
        },
        "unified": {
            "grid_spec": {...},
            "lon2d": array,
            "lat2d": array,
            "fields": { "MMI": stack[nver, nlat, nlon], ... },
            "sigma":  { "STDMMI": stack[nver, nlat, nlon], ... },
            "version_keys": [...],
        },
        "sanity": pd.DataFrame(...)
      }
    """

    # -------------------------
    # init
    # -------------------------
    def __init__(
        self,
        event_id: str,
        event_time: Optional[Union[str, pd.Timestamp]] = None,
        shakemap_folder: Optional[str] = None,
        version_list: Optional[List[Union[int, str]]] = None,
        base_folder: str = "./export/SHAKEuq",
        stations_folder: Optional[str] = None,
        rupture_folder: Optional[str] = None,
        dyfi_cdi_file: Optional[str] = None,
        # future knobs
        include_cdi_from_version: int = 2,
        prefer_sigma_field_prefix: str = "STD",
        verbose: bool = True,
        simulation_defaults: Optional[Dict[str, Any]] = None,
    ):
        self.event_id = str(event_id)
        self.verbose = bool(verbose)
        self.include_cdi_from_version = int(include_cdi_from_version)
        self.prefer_sigma_field_prefix = str(prefer_sigma_field_prefix)

        # event_time override (user-provided)
        if isinstance(event_time, pd.Timestamp):
            self.event_time_utc = event_time.tz_convert("UTC") if event_time.tzinfo else event_time.tz_localize("UTC")
        elif isinstance(event_time, str):
            ts = _parse_iso_utc(event_time)
            self.event_time_utc = ts
        else:
            self.event_time_utc = None

        # folders / paths
        self.base_folder = os.path.abspath(base_folder)
        self.out_folder = _ensure_dir(os.path.join(self.base_folder, self.event_id, "uq"))

        self.shakemap_folder = os.path.abspath(shakemap_folder) if shakemap_folder else None
        self.stations_folder = os.path.abspath(stations_folder) if stations_folder else None
        self.rupture_folder = os.path.abspath(rupture_folder) if rupture_folder else None
        self.dyfi_cdi_file = os.path.abspath(dyfi_cdi_file) if dyfi_cdi_file else None


        self.shakemap_folder = self._norm_event_folder(shakemap_folder)
        self.stations_folder = self._norm_event_folder(stations_folder)
        self.rupture_folder = self._norm_event_folder(rupture_folder)


        # versions
        if version_list is None:
            self.version_list = []
        else:
            self.version_list = [self._norm_version(v) for v in version_list]

        # primary state holder
        self.uq_state: Dict[str, Any] = {
            "config": {
                "event_id": self.event_id,
                "event_time_utc": str(self.event_time_utc) if self.event_time_utc is not None else None,
                "shakemap_folder": self.shakemap_folder,
                "stations_folder": self.stations_folder,
                "rupture_folder": self.rupture_folder,
                "dyfi_cdi_file": self.dyfi_cdi_file,
                "version_list": list(self.version_list),
                "include_cdi_from_version": self.include_cdi_from_version,
                "prefer_sigma_field_prefix": self.prefer_sigma_field_prefix,
                "out_folder": self.out_folder,
            },
            "versions": {},
            "unified": {},
            "sanity": None,
        }

        # simulation support (expand later)
        self.simulation_defaults = simulation_defaults or {
            "nlat": 80,
            "nlon": 100,
            "lon_min": 0.0,
            "lat_min": 0.0,
            "d_lon": 0.02,
            "d_lat": 0.02,
            "fields": ["MMI", "PGA", "PGV", "PSA03", "PSA10", "PSA30", "SVEL"],
            "sigma_fields": ["STDMMI", "STDPGA", "STDPGV", "STDPSA03", "STDPSA10", "STDPSA30"],
            "seed": 42,
            "n_instruments": 50,
            "n_dyfi_stationlist": 80,
            "n_cdi": 300,
        }

        if self.event_id.lower() == "simulation":
            # If user didn't pass versions, create a default set (001..005)
            if not self.version_list:
                self.version_list = [self._norm_version(v) for v in [1, 2, 3, 4, 5]]
                self.uq_state["config"]["version_list"] = list(self.version_list)

        if self.verbose:
            print(f"[SHAKEuq] init event_id={self.event_id} versions={self.version_list}")
            print(f"[SHAKEuq] out_folder={self.out_folder}")

    # -------------------------
    # public: build dataset
    # -------------------------
    def uq_build_dataset(self) -> Dict[str, Any]:
        """
        Build per-version parsed store + unified grid stacks + sanity table.
        """
        if self.event_id.lower() == "simulation":
            return self._uq_build_simulation_dataset()

        if not self.version_list:
            raise ValueError("version_list is empty. Provide versions explicitly.")

        sanity_rows: List[Dict[str, Any]] = []
        per_version: Dict[str, Any] = {}

        for vkey in self.version_list:
            paths = self._discover_version_paths(vkey)
            exists = {k: (p is not None and os.path.exists(p)) for k, p in paths.items()}

            # read grids / meta
            grid_pack = None
            uncert_pack = None
            meta_pack = {}

            if exists.get("grid_xml", False):
                grid_pack = self._read_shakemap_grid_xml(paths["grid_xml"])
                # event_time fallback from grid.xml event tag
                if self.event_time_utc is None:
                    et = _parse_iso_utc(grid_pack["meta"].get("event_timestamp"))
                    if et is not None:
                        self.event_time_utc = et
                        self.uq_state["config"]["event_time_utc"] = str(self.event_time_utc)
                meta_pack.update(grid_pack["meta"])

            if exists.get("uncertainty_xml", False):
                # if grid_pack exists, pass its orientation so uncertainty aligns
                orientation = grid_pack["grid"]["orientation"] if grid_pack is not None else None
                uncert_pack = self._read_uncertainty_xml(paths["uncertainty_xml"], orientation=orientation)
                # also keep uncertainty meta attrs
                meta_pack.update(uncert_pack.get("meta", {}))

            # stations: keep full dfs
            stations_pack = {"instruments": None, "dyfi_stationlist": None, "debug": {}}
            if exists.get("stationlist_json", False):
                stations_pack = self._read_stationlist_json(paths["stationlist_json"])

            # CDI: global file, include from version >= include_cdi_from_version
            cdi_pack = {"df": None, "debug": {}}
            use_cdi = False
            try:
                use_cdi = int(vkey) >= int(self.include_cdi_from_version)
            except Exception:
                use_cdi = False

            if use_cdi and self.dyfi_cdi_file and os.path.exists(self.dyfi_cdi_file):
                cdi_pack = self._read_cdi_file(self.dyfi_cdi_file)

            # audit counts
            n_inst = int(stations_pack["instruments"].shape[0]) if isinstance(stations_pack["instruments"], pd.DataFrame) else 0
            n_dyfi_sl = int(stations_pack["dyfi_stationlist"].shape[0]) if isinstance(stations_pack["dyfi_stationlist"], pd.DataFrame) else 0
            n_cdi = int(cdi_pack["df"].shape[0]) if isinstance(cdi_pack["df"], pd.DataFrame) else 0

            # TAE
            process_ts = meta_pack.get("process_timestamp")
            process_time_utc = _parse_iso_utc(process_ts)
            tae_h = _guess_time_after_event_hours(self.event_time_utc, process_time_utc)

            # per-version store
            per_version[vkey] = {
                "paths": paths,
                "exists": exists,
                "meta": meta_pack,
                "grid": grid_pack["grid"] if grid_pack else None,
                "uncert": uncert_pack["uncert"] if uncert_pack else None,
                "stations": stations_pack,
                "cdi": cdi_pack,
                "obs_audit": {
                    "n_instruments": n_inst,
                    "n_dyfi_stationlist": n_dyfi_sl,
                    "n_cdi": n_cdi,
                    "use_cdi": use_cdi,
                },
            }

            sanity_rows.append({
                "version": vkey,
                "process_timestamp": process_ts,
                "event_timestamp": meta_pack.get("event_timestamp"),
                "TAE_hours": tae_h,
                "grid_xml": exists.get("grid_xml", False),
                "uncertainty_xml": exists.get("uncertainty_xml", False),
                "stationlist_json": exists.get("stationlist_json", False),
                "rupture": exists.get("rupture_json", False),
                "n_instruments": n_inst,
                "n_dyfi_stationlist": n_dyfi_sl,
                "n_cdi": n_cdi,
                "magnitude": _safe_float(meta_pack.get("magnitude")),
                "depth_km": _safe_float(meta_pack.get("depth")),
                "event_lat": _safe_float(meta_pack.get("lat")),
                "event_lon": _safe_float(meta_pack.get("lon")),
                "event_description": meta_pack.get("event_description"),
                "code_version": meta_pack.get("code_version"),
                "shakemap_originator": meta_pack.get("shakemap_originator"),
                "map_status": meta_pack.get("map_status"),
            })

            if self.verbose:
                print(f"[SHAKEuq] parsed v={vkey} grid={exists.get('grid_xml')} unc={exists.get('uncertainty_xml')} "
                      f"stations={exists.get('stationlist_json')} cdi={use_cdi and (self.dyfi_cdi_file is not None)} "
                      f"(n_inst={n_inst}, n_dyfi_sl={n_dyfi_sl}, n_cdi={n_cdi})")

        self.uq_state["versions"] = per_version
        self.uq_state["sanity"] = pd.DataFrame(sanity_rows)

        # build unified grid stacks
        self._build_unified_grids()

        return self.uq_state

    # -------------------------
    # version normalization
    # -------------------------
    def _norm_version(self, v: Union[int, str]) -> str:
        if isinstance(v, int):
            return f"{v:03d}"
        s = str(v).strip()
        # accept "1", "001", "v1", "version1"
        m = re.search(r"(\d+)", s)
        if not m:
            raise ValueError(f"Cannot parse version from: {v}")
        return f"{int(m.group(1)):03d}"



    # -------------------------
    # XML readers
    # -------------------------
    def _read_shakemap_grid_xml(self, xml_path: str) -> Dict[str, Any]:
        """
        Parse ShakeMap mean grid xml into:
          - meta: shakemap_grid attrs + event attrs
          - grid: {spec, fields, units, orientation, lon2d, lat2d}
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # meta from shakemap_grid root attrs
        meta = dict(root.attrib) if root.attrib else {}
        # meta from <event .../>
        event_elem = _xml_find_first(root, "event")
        if event_elem is not None and event_elem.attrib:
            meta.update(event_elem.attrib)

        spec_elem = _xml_find_first(root, "grid_specification")
        if spec_elem is None:
            raise ValueError(f"No grid_specification in {xml_path}")
        spec = {k: spec_elem.attrib.get(k) for k in spec_elem.attrib.keys()}

        nlon = int(spec_elem.attrib["nlon"])
        nlat = int(spec_elem.attrib["nlat"])

        # field list
        field_elems = _xml_find_all(root, "grid_field")
        if not field_elems:
            raise ValueError(f"No grid_field tags in {xml_path}")

        idx_to_name: Dict[int, str] = {}
        units_by_name: Dict[str, str] = {}
        for fe in field_elems:
            idx = int(fe.attrib["index"])
            name = fe.attrib["name"]
            idx_to_name[idx] = name
            units_by_name[name] = fe.attrib.get("units", "")

        nfields = len(idx_to_name)
        order = [idx_to_name[i] for i in sorted(idx_to_name.keys())]

        data_elem = _xml_find_first(root, "grid_data")
        if data_elem is None or data_elem.text is None:
            raise ValueError(f"No grid_data in {xml_path}")

        vals = np.fromstring(data_elem.text.strip(), sep=" ", dtype=float)
        if vals.size % nfields != 0:
            raise ValueError(f"grid_data size mismatch in {xml_path}: vals={vals.size}, nfields={nfields}")
        npts = nlat * nlon
        if vals.size != npts * nfields:
            # sometimes there is newline/extra spaces; fromstring handles spaces.
            # but if mismatch, raise.
            raise ValueError(f"grid_data unexpected size in {xml_path}: got {vals.size}, expected {npts*nfields}")

        arr = vals.reshape((npts, nfields))
        # convert to 2D field grids with presumed row-major fill
        fields_2d: Dict[str, np.ndarray] = {}
        for j, name in enumerate(order):
            fields_2d[name] = arr[:, j].reshape((nlat, nlon))

        # orientation fix using LON/LAT if present
        orientation = self._infer_grid_orientation(fields_2d)
        fields_2d = self._apply_orientation(fields_2d, orientation)

        lon2d = fields_2d.get("LON")
        lat2d = fields_2d.get("LAT")

        return {
            "meta": meta,
            "grid": {
                "path": xml_path,
                "spec": spec,
                "units": units_by_name,
                "fields": fields_2d,
                "orientation": orientation,
                "lon2d": lon2d,
                "lat2d": lat2d,
            }
        }

    def _read_uncertainty_xml(self, xml_path: str, orientation: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        Parse uncertainty.xml into:
          - meta: shakemap_grid attrs + event attrs
          - uncert: {spec, units, sigma_fields (2D), orientation_applied}
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        meta = dict(root.attrib) if root.attrib else {}
        event_elem = _xml_find_first(root, "event")
        if event_elem is not None and event_elem.attrib:
            meta.update(event_elem.attrib)

        spec_elem = _xml_find_first(root, "grid_specification")
        if spec_elem is None:
            raise ValueError(f"No grid_specification in {xml_path}")
        spec = {k: spec_elem.attrib.get(k) for k in spec_elem.attrib.keys()}

        nlon = int(spec_elem.attrib["nlon"])
        nlat = int(spec_elem.attrib["nlat"])

        field_elems = _xml_find_all(root, "grid_field")
        if not field_elems:
            raise ValueError(f"No grid_field tags in {xml_path}")

        idx_to_name: Dict[int, str] = {}
        units_by_name: Dict[str, str] = {}
        for fe in field_elems:
            idx = int(fe.attrib["index"])
            name = fe.attrib["name"]
            idx_to_name[idx] = name
            units_by_name[name] = fe.attrib.get("units", "")

        nfields = len(idx_to_name)
        order = [idx_to_name[i] for i in sorted(idx_to_name.keys())]

        data_elem = _xml_find_first(root, "grid_data")
        if data_elem is None or data_elem.text is None:
            raise ValueError(f"No grid_data in {xml_path}")

        vals = np.fromstring(data_elem.text.strip(), sep=" ", dtype=float)
        if vals.size % nfields != 0:
            raise ValueError(f"uncertainty grid_data size mismatch in {xml_path}: vals={vals.size}, nfields={nfields}")
        npts = nlat * nlon
        if vals.size != npts * nfields:
            raise ValueError(f"uncertainty grid_data unexpected size in {xml_path}: got {vals.size}, expected {npts*nfields}")

        arr = vals.reshape((npts, nfields))
        sigma_2d: Dict[str, np.ndarray] = {}
        for j, name in enumerate(order):
            sigma_2d[name] = arr[:, j].reshape((nlat, nlon))

        # apply orientation from mean grid if given; else infer from LON/LAT
        if orientation is None:
            orientation = self._infer_grid_orientation(sigma_2d)
        sigma_2d = self._apply_orientation(sigma_2d, orientation)

        return {
            "meta": meta,
            "uncert": {
                "path": xml_path,
                "spec": spec,
                "units": units_by_name,
                "fields": sigma_2d,
                "orientation_applied": orientation,
            }
        }

    def _infer_grid_orientation(self, fields_2d: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """
        Infer transpose/flipud/fliplr so that:
          - LAT increases (or at least varies primarily) down axis=0
          - LON increases across axis=1
        This is conservative; if no LON/LAT fields exist, do nothing.
        """
        orient = {"transpose": False, "flipud": False, "fliplr": False}

        lon = fields_2d.get("LON")
        lat = fields_2d.get("LAT")
        if lon is None or lat is None:
            return orient

        # Determine if transpose helps: lon should vary more along axis=1 than axis=0
        def var_axis(a: np.ndarray) -> Tuple[float, float]:
            # average absolute diffs along each axis
            d0 = np.nanmean(np.abs(np.diff(a, axis=0))) if a.shape[0] > 1 else 0.0
            d1 = np.nanmean(np.abs(np.diff(a, axis=1))) if a.shape[1] > 1 else 0.0
            return d0, d1

        lon_d0, lon_d1 = var_axis(lon)
        lat_d0, lat_d1 = var_axis(lat)

        # If lon varies more vertically than horizontally and lat varies more horizontally than vertically, transpose.
        if (lon_d0 > lon_d1) and (lat_d1 > lat_d0):
            orient["transpose"] = True
            lon = lon.T
            lat = lat.T

        # After transpose decision, decide flips
        # lon should increase left->right
        if lon.shape[1] > 1:
            if np.nanmean(lon[:, -1]) < np.nanmean(lon[:, 0]):
                orient["fliplr"] = True
                lon = np.fliplr(lon)
                lat = np.fliplr(lat)

        # lat should increase bottom->top? Actually grid_data often starts at lat_max, so lat decreases downwards.
        # We'll choose a consistent convention: lat increases upward (axis=0 decreasing) is common.
        # But for computations, either is fine as long as consistent. We'll enforce lat increases down axis=0 (top->bottom).
        if lat.shape[0] > 1:
            if np.nanmean(lat[-1, :]) < np.nanmean(lat[0, :]):
                # lat decreases downward -> flipud to make it increase downward
                orient["flipud"] = True

        return orient

    def _apply_orientation(self, fields_2d: Dict[str, np.ndarray], orientation: Dict[str, bool]) -> Dict[str, np.ndarray]:
        out = {}
        for k, a in fields_2d.items():
            aa = a
            if orientation.get("transpose", False):
                aa = aa.T
            if orientation.get("fliplr", False):
                aa = np.fliplr(aa)
            if orientation.get("flipud", False):
                aa = np.flipud(aa)
            out[k] = aa
        return out

    # -------------------------
    # Station readers (keep full dfs)
    # -------------------------
    def _get_usgs_parser(self):
        """
        Import USGSParser in a robust way (toolkit may have different module paths).
        """
        try:
            from SHAKEparser import USGSParser  # type: ignore
            return USGSParser
        except Exception:
            pass
        try:
            from modules.SHAKEparser import USGSParser  # type: ignore
            return USGSParser
        except Exception:
            pass
        try:
            from .SHAKEparser import USGSParser  # type: ignore
            return USGSParser
        except Exception:
            pass
        raise ImportError("USGSParser could not be imported. Ensure SHAKEparser is on PYTHONPATH.")

    def _read_stationlist_json(self, json_path: str) -> Dict[str, Any]:
        """
        Read stationlist.json:
          - instruments dataframe via value_type='pga' (we keep df, no cropping)
          - dyfi stationlist dataframe via value_type='mmi' (keep df)
        """
        USGSParser = self._get_usgs_parser()

        dbg = {"path": json_path, "notes": []}
        instruments_df = None
        dyfi_df = None

        # instruments: choose one value_type to obtain instrumented dataset
        # (USGSParser returns the same row set with pga column populated)
        try:
            p = USGSParser(parser_type="instrumented_data", json_file=json_path)
            instruments_df = p.get_dataframe(value_type="pga")
            if isinstance(instruments_df, pd.DataFrame):
                # coerce common numeric columns but DO NOT drop/crop
                instruments_df = _coerce_numeric_cols(instruments_df, ["longitude", "latitude", "pga", "distance"])
        except Exception as e:
            dbg["notes"].append(f"instruments parse failed: {repr(e)}")

        # dyfi-from-stationlist: value_type mmi
        try:
            p = USGSParser(parser_type="instrumented_data", json_file=json_path)
            dyfi_df = p.get_dataframe(value_type="mmi")
            if isinstance(dyfi_df, pd.DataFrame):
                dyfi_df = _coerce_numeric_cols(dyfi_df, ["longitude", "latitude", "intensity", "distance", "nresp", "intensity_stddev"])
                # NOTE: we do NOT drop rows here; filtering belongs to later obs adapter logic.
        except Exception as e:
            dbg["notes"].append(f"dyfi stationlist parse failed: {repr(e)}")

        return {"instruments": instruments_df, "dyfi_stationlist": dyfi_df, "debug": dbg}

    def _read_cdi_file(self, cdi_file_path: str) -> Dict[str, Any]:
        """
        Read global CDI geocoded file via USGSParser(parser_type='dyfi_data', file_path=...).
        Keep full df (no cropping).
        """
        USGSParser = self._get_usgs_parser()
        dbg = {"path": cdi_file_path, "notes": []}
        df = None
        try:
            p = USGSParser(parser_type="dyfi_data", file_path=cdi_file_path)
            df = p.get_dataframe()
            if isinstance(df, pd.DataFrame):
                # normalize typical columns (case can vary)
                # We'll keep originals and add normalized aliases if present.
                # Common: Latitude/Longitude, CDI, No. of responses, Standard deviation, Suspect?
                for col in df.columns:
                    if isinstance(col, str):
                        pass
                # numeric coercions (best-effort)
                df = _coerce_numeric_cols(df, ["Latitude", "Longitude", "CDI", "No. of responses", "Hypocentral distance",
                                              "Standard deviation", "Suspect?"])
        except Exception as e:
            dbg["notes"].append(f"cdi parse failed: {repr(e)}")

        return {"df": df, "debug": dbg}

    # -------------------------
    # Unified grid builder
    # -------------------------
    def _build_unified_grids(self):
        """
        Build stacked arrays across versions for all available mean fields and sigma fields.
        Stores into self.uq_state["unified"].
        """
        versions = self.uq_state.get("versions", {})
        if not versions:
            self.uq_state["unified"] = {}
            return

        version_keys = list(versions.keys())

        # pick first version with a grid as reference
        ref_v = None
        for vk in version_keys:
            if versions[vk].get("grid") is not None:
                ref_v = vk
                break
        if ref_v is None:
            self.uq_state["unified"] = {}
            return

        ref_grid = versions[ref_v]["grid"]
        ref_spec = ref_grid["spec"]
        ref_lon2d = ref_grid.get("lon2d")
        ref_lat2d = ref_grid.get("lat2d")

        # collect field names present across any version
        mean_field_names = set()
        sigma_field_names = set()

        for vk in version_keys:
            g = versions[vk].get("grid")
            if g and isinstance(g.get("fields"), dict):
                mean_field_names |= set(g["fields"].keys())
            u = versions[vk].get("uncert")
            if u and isinstance(u.get("fields"), dict):
                sigma_field_names |= set(u["fields"].keys())

        mean_field_names = sorted(mean_field_names)
        sigma_field_names = sorted(sigma_field_names)

        # stack per field
        unified_fields: Dict[str, np.ndarray] = {}
        unified_sigma: Dict[str, np.ndarray] = {}

        # Determine grid shape from ref
        any_field = None
        for k, a in ref_grid["fields"].items():
            if isinstance(a, np.ndarray):
                any_field = a
                break
        if any_field is None:
            self.uq_state["unified"] = {}
            return

        nlat, nlon = any_field.shape
        nver = len(version_keys)

        def get_field(vk: str, name: str, kind: str) -> Optional[np.ndarray]:
            if kind == "mean":
                g = versions[vk].get("grid")
                if g and isinstance(g.get("fields"), dict):
                    return g["fields"].get(name)
            else:
                u = versions[vk].get("uncert")
                if u and isinstance(u.get("fields"), dict):
                    return u["fields"].get(name)
            return None

        # mean stacks
        for fname in mean_field_names:
            stack = np.full((nver, nlat, nlon), np.nan, dtype=float)
            for i, vk in enumerate(version_keys):
                a = get_field(vk, fname, "mean")
                if isinstance(a, np.ndarray) and a.shape == (nlat, nlon):
                    stack[i] = a
            unified_fields[fname] = stack

        # sigma stacks
        for sname in sigma_field_names:
            stack = np.full((nver, nlat, nlon), np.nan, dtype=float)
            for i, vk in enumerate(version_keys):
                a = get_field(vk, sname, "sigma")
                if isinstance(a, np.ndarray) and a.shape == (nlat, nlon):
                    stack[i] = a
            unified_sigma[sname] = stack

        self.uq_state["unified"] = {
            "grid_spec": ref_spec,
            "lon2d": ref_lon2d,
            "lat2d": ref_lat2d,
            "fields": unified_fields,
            "sigma": unified_sigma,
            "version_keys": version_keys,
            "shape": (nver, nlat, nlon),
        }

        if self.verbose:
            print(f"[SHAKEuq] unified grids built: nver={nver}, shape=({nlat},{nlon}), "
                  f"mean_fields={len(unified_fields)}, sigma_fields={len(unified_sigma)}")

    # -------------------------
    # Simulation dataset (placeholder but runnable)
    # -------------------------
    def _uq_build_simulation_dataset(self) -> Dict[str, Any]:
        """
        Build a fully synthetic dataset with the same structure as real parsing.
        Expand later with realistic GMPE/GMICE, etc.
        """
        cfg = self.simulation_defaults
        rng = np.random.default_rng(int(cfg.get("seed", 42)))

        nlat = int(cfg["nlat"])
        nlon = int(cfg["nlon"])
        lon_min = float(cfg["lon_min"])
        lat_min = float(cfg["lat_min"])
        dlon = float(cfg["d_lon"])
        dlat = float(cfg["d_lat"])

        lons = lon_min + np.arange(nlon) * dlon
        lats = lat_min + np.arange(nlat) * dlat
        lon2d, lat2d = np.meshgrid(lons, lats)

        # simple synthetic "event"
        meta_base = {
            "event_id": "simulation",
            "magnitude": 6.5,
            "depth": 10.0,
            "lat": float(lat_min + (nlat * dlat) / 2.0),
            "lon": float(lon_min + (nlon * dlon) / 2.0),
            "event_timestamp": str(pd.Timestamp("2020-01-01T00:00:00Z")),
            "event_description": "Synthetic event (simulation)",
            "code_version": "sim",
            "process_timestamp": None,
        }
        self.event_time_utc = _parse_iso_utc(meta_base["event_timestamp"])
        self.uq_state["config"]["event_time_utc"] = str(self.event_time_utc)

        per_version = {}
        sanity_rows = []

        # synthetic base fields
        def gaussian_bump(x, y, x0, y0, sx, sy):
            return np.exp(-(((x - x0) ** 2) / (2 * sx ** 2) + ((y - y0) ** 2) / (2 * sy ** 2)))

        cx = meta_base["lon"]
        cy = meta_base["lat"]
        bump = gaussian_bump(lon2d, lat2d, cx, cy, sx=dlon * nlon / 6.0, sy=dlat * nlat / 6.0)

        for vkey in self.version_list:
            i_v = int(vkey)
            # process timestamp increments
            proc = (self.event_time_utc + pd.to_timedelta(i_v * 30, unit="min")) if self.event_time_utc is not None else None

            # mean fields evolve slightly with version
            fields = {}
            for name in cfg["fields"]:
                noise = rng.normal(0, 0.05, size=(nlat, nlon))
                if name == "MMI":
                    base = 2.5 + 5.0 * bump + 0.2 * (i_v - 1) / max(1, len(self.version_list) - 1)
                    fields[name] = base + noise
                elif name in ("PGA", "PGV"):
                    base = 0.2 + 2.5 * bump + 0.1 * (i_v - 1) / max(1, len(self.version_list) - 1)
                    fields[name] = base + noise
                else:
                    base = 0.1 + 2.0 * bump + 0.05 * (i_v - 1) / max(1, len(self.version_list) - 1)
                    fields[name] = base + noise

            fields["LON"] = lon2d.copy()
            fields["LAT"] = lat2d.copy()

            # sigma fields
            sigma = {}
            for sname in cfg["sigma_fields"]:
                sigma[sname] = np.clip(0.3 + 0.2 * (1.0 - bump) + rng.normal(0, 0.02, size=(nlat, nlon)), 0.05, None)

            grid_spec = {
                "lon_min": str(lon_min),
                "lat_min": str(lat_min),
                "lon_max": str(lons[-1]),
                "lat_max": str(lats[-1]),
                "nominal_lon_spacing": str(dlon),
                "nominal_lat_spacing": str(dlat),
                "nlon": str(nlon),
                "nlat": str(nlat),
            }

            # stations synthetic dataframes (keep full columns)
            n_inst = int(cfg["n_instruments"])
            n_dyfi = int(cfg["n_dyfi_stationlist"])
            inst_df = pd.DataFrame({
                "id": [f"SIM.STA.{k:04d}" for k in range(n_inst)],
                "longitude": rng.uniform(lons.min(), lons.max(), size=n_inst),
                "latitude": rng.uniform(lats.min(), lats.max(), size=n_inst),
                "station_type": "seismic",
                "pga": rng.uniform(0.1, 5.0, size=n_inst),
                "intensity": rng.uniform(2.0, 8.0, size=n_inst),
                "intensity_stddev": rng.uniform(0.2, 0.8, size=n_inst),
                "predictions": None,
            })

            dyfi_df = pd.DataFrame({
                "id": [f"SIM.DYFI.{k:04d}" for k in range(n_dyfi)],
                "longitude": rng.uniform(lons.min(), lons.max(), size=n_dyfi),
                "latitude": rng.uniform(lats.min(), lats.max(), size=n_dyfi),
                "station_type": "macroseismic",
                "nresp": rng.integers(1, 25, size=n_dyfi),
                "intensity": rng.uniform(1.0, 7.0, size=n_dyfi),
                "intensity_stddev": rng.uniform(0.2, 0.6, size=n_dyfi),
                "distance": rng.uniform(0, 300, size=n_dyfi),
                "predictions": None,
            })

            # CDI (global; include from version >= include threshold)
            use_cdi = int(vkey) >= int(self.include_cdi_from_version)
            cdi_df = None
            if use_cdi:
                n_cdi = int(cfg["n_cdi"])
                cdi_df = pd.DataFrame({
                    "Geocoded box": [f"SIM.BOX.{k:05d}" for k in range(n_cdi)],
                    "Latitude": rng.uniform(lats.min(), lats.max(), size=n_cdi),
                    "Longitude": rng.uniform(lons.min(), lons.max(), size=n_cdi),
                    "CDI": rng.uniform(1.0, 6.5, size=n_cdi),
                    "No. of responses": rng.integers(1, 50, size=n_cdi),
                    "Hypocentral distance": rng.uniform(0, 400, size=n_cdi),
                    "Suspect?": 0,
                    "Standard deviation": rng.uniform(0.2, 0.7, size=n_cdi),
                    "City": None,
                    "State": None,
                })

            per_version[vkey] = {
                "paths": {"grid_xml": None, "uncertainty_xml": None, "stationlist_json": None, "rupture_json": None},
                "exists": {"grid_xml": True, "uncertainty_xml": True, "stationlist_json": True, "rupture_json": False},
                "meta": {**meta_base, "process_timestamp": str(proc) if proc is not None else None},
                "grid": {
                    "path": None,
                    "spec": grid_spec,
                    "units": {k: "" for k in fields.keys()},
                    "fields": fields,
                    "orientation": {"transpose": False, "flipud": False, "fliplr": False},
                    "lon2d": lon2d,
                    "lat2d": lat2d,
                },
                "uncert": {
                    "path": None,
                    "spec": grid_spec,
                    "units": {k: "" for k in sigma.keys()},
                    "fields": sigma,
                    "orientation_applied": {"transpose": False, "flipud": False, "fliplr": False},
                },
                "stations": {"instruments": inst_df, "dyfi_stationlist": dyfi_df, "debug": {"notes": []}},
                "cdi": {"df": cdi_df, "debug": {"notes": []}},
                "obs_audit": {
                    "n_instruments": int(inst_df.shape[0]),
                    "n_dyfi_stationlist": int(dyfi_df.shape[0]),
                    "n_cdi": int(cdi_df.shape[0]) if isinstance(cdi_df, pd.DataFrame) else 0,
                    "use_cdi": use_cdi,
                }
            }

            tae_h = _guess_time_after_event_hours(self.event_time_utc, proc if isinstance(proc, pd.Timestamp) else None)
            sanity_rows.append({
                "version": vkey,
                "process_timestamp": str(proc) if proc is not None else None,
                "event_timestamp": meta_base["event_timestamp"],
                "TAE_hours": tae_h,
                "grid_xml": True,
                "uncertainty_xml": True,
                "stationlist_json": True,
                "rupture": False,
                "n_instruments": int(inst_df.shape[0]),
                "n_dyfi_stationlist": int(dyfi_df.shape[0]),
                "n_cdi": int(cdi_df.shape[0]) if isinstance(cdi_df, pd.DataFrame) else 0,
                "magnitude": meta_base["magnitude"],
                "depth_km": meta_base["depth"],
                "event_lat": meta_base["lat"],
                "event_lon": meta_base["lon"],
                "event_description": meta_base["event_description"],
                "code_version": meta_base["code_version"],
                "shakemap_originator": "sim",
                "map_status": "sim",
            })

        self.uq_state["versions"] = per_version
        self.uq_state["sanity"] = pd.DataFrame(sanity_rows)
        self._build_unified_grids()
        return self.uq_state

    # -------------------------
    # Convenience helpers
    # -------------------------
    @staticmethod
    def sigma_field_to_imt(sigma_field_name: str, prefix: str = "STD") -> str:
        s = str(sigma_field_name)
        if s.upper().startswith(prefix.upper()):
            return s[len(prefix):]
        return s

    @staticmethod
    def imt_to_sigma_field(imt: str, prefix: str = "STD") -> str:
        s = str(imt)
        if s.upper().startswith(prefix.upper()):
            return s
        return f"{prefix}{s}"

    def get_sanity_table(self) -> Optional[pd.DataFrame]:
        return self.uq_state.get("sanity", None)

    def get_version(self, v: Union[int, str]) -> Dict[str, Any]:
        vkey = self._norm_version(v)
        if vkey not in self.uq_state.get("versions", {}):
            raise KeyError(f"Version {vkey} not found in uq_state. Run uq_build_dataset() first.")
        return self.uq_state["versions"][vkey]

    def summary(self) -> None:
        df = self.get_sanity_table()
        if df is None:
            print("[SHAKEuq] No sanity table. Run uq_build_dataset().")
            return
        with pd.option_context("display.max_columns", 200, "display.width", 200):
            print(df)



    def debug_discovery(self, v: Union[int, str]) -> Dict[str, Any]:
        """
        Print and return discovery results for one version.
        """
        vkey = self._norm_version(v)
        paths = self._discover_version_paths(vkey)
        exists = {k: (p is not None and os.path.exists(p)) for k, p in paths.items()}
        print(f"\n[SHAKEuq][DISCOVERY] event_id={self.event_id} version={vkey}")
        for k in ["grid_xml", "uncertainty_xml", "stationlist_json", "rupture_json"]:
            print(f"  {k:16s}: {paths.get(k)}  =>  {exists.get(k)}")
        return {"version": vkey, "paths": paths, "exists": exists}

    def debug_discovery_all(self) -> pd.DataFrame:
        """
        Run discovery for all versions and return a small table.
        """
        rows = []
        for v in self.version_list:
            r = self.debug_discovery(v)
            rows.append({
                "version": r["version"],
                "grid_xml": r["exists"]["grid_xml"],
                "uncertainty_xml": r["exists"]["uncertainty_xml"],
                "stationlist_json": r["exists"]["stationlist_json"],
                "rupture_json": r["exists"]["rupture_json"],
                "grid_path": r["paths"]["grid_xml"],
                "uncert_path": r["paths"]["uncertainty_xml"],
                "station_path": r["paths"]["stationlist_json"],
                "rupture_path": r["paths"]["rupture_json"],
            })
        return pd.DataFrame(rows)





    def _norm_event_folder(self, base: Optional[str]) -> Optional[str]:
        """
        Normalize user-provided folder to the event-scoped folder if needed.

        User may pass:
          - .../usgs-shakemap-versions
          - .../usgs-shakemap-versions/us7000m9g4

        We return the existing event folder if found, otherwise return base as-is.
        """
        if not base:
            return None
        base = os.path.abspath(base)

        # If they already passed the event folder:
        if os.path.basename(base) == self.event_id and os.path.isdir(base):
            return base

        # If they passed parent folder, append event_id if that exists:
        candidate = os.path.join(base, self.event_id)
        if os.path.isdir(candidate):
            return candidate

        # Otherwise return the base (maybe they have different naming)
        return base

    def _build_usgs_filename(self, vkey: str, kind: str, originator: str = "us") -> str:
        """
        Deterministic USGS-style filename builder.

        kind in: {"grid", "uncertainty", "stationlist", "rupture"}
        """
        if kind == "grid":
            suffix = "grid.xml"
        elif kind == "uncertainty":
            suffix = "uncertainty.xml"
        elif kind == "stationlist":
            suffix = "stationlist.json"
        elif kind == "rupture":
            suffix = "rupture.json"
        else:
            raise ValueError(f"Unknown kind: {kind}")

        return f"{self.event_id}_{originator}_{vkey}_{suffix}"

    def _find_first_existing(self, candidates: List[str]) -> Optional[str]:
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None


    def _discover_version_paths(self, vkey: str) -> Dict[str, Optional[str]]:
        """
        Deterministic path resolution for your SHAKEfetch layout:

          shakemap_folder/<event_id>/
            us7000m9g4_us_001_grid.xml
            us7000m9g4_us_001_uncertainty.xml
            ...

          stations_folder/<event_id>/
            us7000m9g4_us_001_stationlist.json
            ...

          rupture_folder/<event_id>/
            us7000m9g4_us_001_rupture.json
            ...
        """
        out: Dict[str, Optional[str]] = {
            "grid_xml": None,
            "uncertainty_xml": None,
            "stationlist_json": None,
            "rupture_json": None,
        }

        # ensure we are pointing at event-level folders if possible
        sm_dir = self._norm_event_folder(self.shakemap_folder)
        st_dir = self._norm_event_folder(self.stations_folder)
        rp_dir = self._norm_event_folder(self.rupture_folder)

        # Most of your files use originator "us"
        originator = "us"

        # --- grid.xml and uncertainty.xml
        if sm_dir and os.path.isdir(sm_dir):
            grid_name = self._build_usgs_filename(vkey, "grid", originator=originator)
            unc_name = self._build_usgs_filename(vkey, "uncertainty", originator=originator)

            grid_candidates = [
                os.path.join(sm_dir, grid_name),
                os.path.join(sm_dir, grid_name.replace(".xml", "")),  # if extension missing
            ]
            unc_candidates = [
                os.path.join(sm_dir, unc_name),
                os.path.join(sm_dir, unc_name.replace(".xml", "")),
            ]

            out["grid_xml"] = self._find_first_existing(grid_candidates)
            out["uncertainty_xml"] = self._find_first_existing(unc_candidates)

            # fallback glob if deterministic not found
            if out["grid_xml"] is None:
                hits = sorted(glob.glob(os.path.join(sm_dir, f"{self.event_id}_{originator}_{vkey}_*grid*.xml")))
                if hits:
                    out["grid_xml"] = hits[0]
            if out["uncertainty_xml"] is None:
                hits = sorted(glob.glob(os.path.join(sm_dir, f"{self.event_id}_{originator}_{vkey}_*uncertainty*.xml")))
                if hits:
                    out["uncertainty_xml"] = hits[0]

        # --- stationlist.json
        if st_dir and os.path.isdir(st_dir):
            st_name = self._build_usgs_filename(vkey, "stationlist", originator=originator)
            st_candidates = [
                os.path.join(st_dir, st_name),
                os.path.join(st_dir, st_name.replace(".json", "")),
            ]
            out["stationlist_json"] = self._find_first_existing(st_candidates)

            if out["stationlist_json"] is None:
                hits = sorted(glob.glob(os.path.join(st_dir, f"{self.event_id}_{originator}_{vkey}_*stationlist*.json")))
                if hits:
                    out["stationlist_json"] = hits[0]

        # --- rupture.json
        if rp_dir and os.path.isdir(rp_dir):
            rp_name = self._build_usgs_filename(vkey, "rupture", originator=originator)
            rp_candidates = [
                os.path.join(rp_dir, rp_name),
                os.path.join(rp_dir, rp_name.replace(".json", "")),
            ]
            out["rupture_json"] = self._find_first_existing(rp_candidates)

            if out["rupture_json"] is None:
                hits = sorted(glob.glob(os.path.join(rp_dir, f"{self.event_id}_{originator}_{vkey}_*rupture*.json")))
                if hits:
                    out["rupture_json"] = hits[0]

        return out
