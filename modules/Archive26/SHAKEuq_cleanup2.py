from __future__ import annotations

import os, re, glob, json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import os
import glob
import re

import os
import math
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
        field_defs: Dict[int, Dict[str, Any]] = {}
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
                try:
                    idx = int(attrs.get("index", "-1"))
                    if idx >= 0:
                        field_defs[idx] = attrs
                except Exception:
                    pass
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

        lon_col, lat_col = 0, 1
        for idx, attrs in field_defs.items():
            name = str(attrs.get("name", "")).upper()
            if name in ("LON", "LONGITUDE"):
                lon_col = idx
            if name in ("LAT", "LATITUDE"):
                lat_col = idx

        flat = arr[:npts, :]
        lon2d = flat[:, lon_col].reshape((nlat, nlon))
        lat2d = flat[:, lat_col].reshape((nlat, nlon))
        orient = self._infer_orientation(lon2d, lat2d)

        lon2d = self._apply_orientation_one(lon2d, orient)
        lat2d = self._apply_orientation_one(lat2d, orient)

        fields: Dict[str, np.ndarray] = {}
        for idx, attrs in field_defs.items():
            name = str(attrs.get("name", "")).strip()
            if not name or idx in (lon_col, lat_col):
                continue
            vals = flat[:, idx].reshape((nlat, nlon))
            vals = self._apply_orientation_one(vals, orient)
            fields[name] = vals

        grid = {
            "spec": {"nlat": nlat, "nlon": nlon, **spec},
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
        field_defs: Dict[int, Dict[str, Any]] = {}
        data_text = None

        for ch in root:
            t = self._strip(ch.tag)
            if t == "grid_specification":
                spec = {self._strip(k): v for k, v in ch.attrib.items()}
            elif t == "grid_field":
                attrs = {self._strip(k): v for k, v in ch.attrib.items()}
                try:
                    idx = int(attrs.get("index", "-1"))
                    if idx >= 0:
                        field_defs[idx] = attrs
                except Exception:
                    pass
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

        lon_col = None
        lat_col = None
        for idx, attrs in field_defs.items():
            name = str(attrs.get("name", "")).upper()
            if name in ("LON", "LONGITUDE"):
                lon_col = idx
            if name in ("LAT", "LATITUDE"):
                lat_col = idx

        fields: Dict[str, np.ndarray] = {}
        for idx, attrs in field_defs.items():
            name = str(attrs.get("name", "")).strip()
            if not name:
                continue
            if lon_col is not None and idx == lon_col:
                continue
            if lat_col is not None and idx == lat_col:
                continue
            vals = flat[:, idx].reshape((nlat, nlon))
            if orientation:
                vals = self._apply_orientation_one(vals, orientation)
            fields[name] = vals

        uncert = {"spec": {"nlat": nlat, "nlon": nlon, **spec}, "fields": fields}
        return {"meta": meta, "uncert": uncert}

    # -------------------------
    # unified grids
    # -------------------------
    def _build_unified_grids(self) -> None:
        versions: Dict[str, Any] = self.uq_state.get("versions", {}) or {}
        if not versions:
            self.uq_state["unified"] = {}
            return

        vkeys = list(versions.keys())
        ref_v = next((vk for vk in vkeys if versions[vk].get("grid") is not None), None)
        if ref_v is None:
            self.uq_state["unified"] = {}
            return

        ref_grid = versions[ref_v]["grid"]
        lon2d = ref_grid.get("lon2d")
        lat2d = ref_grid.get("lat2d")

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

        any_field = next(iter(ref_grid["fields"].values()))
        nlat, nlon = any_field.shape

        for fn in mean_names:
            stack = np.full((len(vkeys), nlat, nlon), np.nan, dtype=float)
            for i, vk in enumerate(vkeys):
                g = versions[vk].get("grid")
                if g and fn in g.get("fields", {}):
                    stack[i] = g["fields"][fn]
            unified_fields[fn] = stack

        for sn in sig_keep:
            stack = np.full((len(vkeys), nlat, nlon), np.nan, dtype=float)
            for i, vk in enumerate(vkeys):
                u = versions[vk].get("uncert")
                if u and sn in u.get("fields", {}):
                    stack[i] = u["fields"][sn]
            unified_sigma[sn] = stack

        self.uq_state["unified"] = {
            "lon2d": lon2d,
            "lat2d": lat2d,
            "fields": unified_fields,
            "sigma": unified_sigma,
            "version_keys": vkeys,
        }

    # -------------------------
    # public: build dataset
    # -------------------------
    def uq_build_dataset(self) -> Dict[str, Any]:
        if not self.version_list:
            raise ValueError("version_list is empty.")

        cdi_path, cdi_resolve_note = self._resolve_cdi_path(self.dyfi_cdi_input)

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

            use_cdi = False
            try:
                use_cdi = int(vkey) >= int(self.include_cdi_from_version)
            except Exception:
                use_cdi = False

            cdi_pack = {"df": None, "debug": {}, "cdi_path": None, "cdi_loaded": False, "cdi_note": ""}
            if use_cdi and cdi_path and os.path.exists(cdi_path):
                tmp = self._read_cdi_file(cdi_path)
                cdi_pack.update(tmp)
                cdi_pack["cdi_path"] = cdi_path
                cdi_pack["cdi_loaded"] = isinstance(tmp.get("df"), pd.DataFrame) and not tmp["df"].empty
                cdi_pack["cdi_note"] = cdi_resolve_note
            else:
                cdi_pack["cdi_path"] = cdi_path
                cdi_pack["cdi_loaded"] = False
                cdi_pack["cdi_note"] = cdi_resolve_note if use_cdi else "cdi_gate=off"

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
                    "rupture_loaded": rupture_loaded,
                    "n_instruments": n_inst,
                    "n_dyfi_stationlist": n_dyfi_sl,
                    "use_cdi": use_cdi,
                    "cdi_path": cdi_pack.get("cdi_path"),
                    "cdi_loaded": bool(cdi_pack.get("cdi_loaded")),
                    "n_cdi": n_cdi,
                    "cdi_note": cdi_pack.get("cdi_note"),
                    "magnitude": _safe_float(meta_pack.get("magnitude")),
                    "depth_km": _safe_float(meta_pack.get("depth")),
                    "event_lat": _safe_float(meta_pack.get("lat")),
                    "event_lon": _safe_float(meta_pack.get("lon")),
                    "event_description": meta_pack.get("event_description") or meta_pack.get("event_description"),
                    "code_version": meta_pack.get("code_version"),
                    "shakemap_originator": meta_pack.get("shakemap_originator"),
                    "map_status": meta_pack.get("map_status"),
                }
            )

            if self.verbose:
                print(
                    f"[SHAKEuq] parsed v={vkey} grid={exists.get('grid_xml')} unc={exists.get('uncertainty_xml')} "
                    f"stations={exists.get('stationlist_json')} rupture_loaded={rupture_loaded} "
                    f"use_cdi={use_cdi} cdi_loaded={bool(cdi_pack.get('cdi_loaded'))} n_cdi={n_cdi}"
                )

        self.uq_state["versions"] = per_version
        self._build_unified_grids()
        self.uq_state["sanity"] = pd.DataFrame(sanity_rows)
        return self.uq_state

    # -------------------------
    # Observation adapter
    # -------------------------
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

        obs_frames: List[pd.DataFrame] = []

        # Instruments (seismic)
        inst = vpack.get("stations", {}).get("instruments")
        if isinstance(inst, pd.DataFrame) and not inst.empty and imt_u in ("PGA", "PGV", "MMI_SEISMIC"):
            lon_c = _first_present(inst, ["longitude", "lon", "Longitude", "LON"])
            lat_c = _first_present(inst, ["latitude", "lat", "Latitude", "LAT"])
            val_c = _first_present(inst, [imt_u.lower(), imt_u, "pga", "pgv"])
            if lon_c and lat_c and val_c:
                df = inst.copy()
                df["lon"] = pd.to_numeric(df[lon_c], errors="coerce")
                df["lat"] = pd.to_numeric(df[lat_c], errors="coerce")
                df["value"] = pd.to_numeric(df[val_c], errors="coerce")
                df = df.dropna(subset=["lon", "lat", "value"])
                df["sigma"] = float(sigma_override) if sigma_override is not None else self.sigma_instr
                df["source_type"] = "seismic"
                df["source_detail"] = "station"
                sid = _first_present(df, ["id", "station_id", "code", "station"])
                df["station_id"] = df[sid].astype(str) if sid else None
                df["version"] = vkey
                df["imt"] = imt_u
                df["tae_hours"] = tae_h
                obs_frames.append(df[["lon", "lat", "value", "sigma", "source_type", "source_detail", "station_id", "version", "imt", "tae_hours"]])

        # DYFI stationlist (intensity)
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
                df = df.dropna(subset=["lon", "lat", "value"])
                df["sigma"] = float(sigma_override) if sigma_override is not None else self.sigma_dyfi_stationlist
                df["source_type"] = "intensity"
                df["source_detail"] = "dyfi_stationlist"
                sid = _first_present(df, ["id", "station_id", "code", "station"])
                df["station_id"] = df[sid].astype(str) if sid else None
                df["version"] = vkey
                df["imt"] = imt_u
                df["tae_hours"] = tae_h
                obs_frames.append(df[["lon", "lat", "value", "sigma", "source_type", "source_detail", "station_id", "version", "imt", "tae_hours"]])

        # CDI file (intensity)
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

                if self.cdi_max_dist_km is not None and dist_c and dist_c in df.columns:
                    df[dist_c] = pd.to_numeric(df[dist_c], errors="coerce")
                    df = df[df[dist_c].isna() | (df[dist_c] <= float(self.cdi_max_dist_km))]

                if self.cdi_min_nresp is not None and nresp_c and nresp_c in df.columns:
                    df[nresp_c] = pd.to_numeric(df[nresp_c], errors="coerce")
                    df = df[df[nresp_c].isna() | (df[nresp_c] >= int(self.cdi_min_nresp))]

                df = df.dropna(subset=["lon", "lat", "value"])
                df["sigma"] = float(sigma_override) if sigma_override is not None else self.sigma_cdi
                df["source_type"] = "intensity"
                df["source_detail"] = "cdi_geo"
                df["station_id"] = None
                df["version"] = vkey
                df["imt"] = imt_u
                df["tae_hours"] = tae_h
                obs_frames.append(df[["lon", "lat", "value", "sigma", "source_type", "source_detail", "station_id", "version", "imt", "tae_hours"]])

        if not obs_frames:
            return pd.DataFrame(columns=["lon", "lat", "value", "sigma", "source_type", "source_detail", "station_id", "version", "imt", "tae_hours"])

        out = pd.concat(obs_frames, ignore_index=True)
        out = out.dropna(subset=["lon", "lat", "value"])
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
        lon_rev = False
        lat_rev = False
        if len(src_lon1) > 1 and np.nanmean(np.diff(src_lon1)) < 0:
            src_lon1 = src_lon1[::-1]
            src_field2d = np.fliplr(src_field2d)
            lon_rev = True
        if len(src_lat1) > 1 and np.nanmean(np.diff(src_lat1)) < 0:
            src_lat1 = src_lat1[::-1]
            src_field2d = np.flipud(src_field2d)
            lat_rev = True
    
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
            # quick report: which versions had shape mismatches?
            mism = []
            for vk in vkeys:
                g = versions[vk].get("grid")
                if g and g.get("lon2d") is not None:
                    if g["lon2d"].shape != (nlat_ref, nlon_ref):
                        mism.append((vk, g["lon2d"].shape))
            if mism:
                print(f"[SHAKEuq] unified: remapped {len(mism)} version(s) onto ref grid {ref_v} shape={(nlat_ref,nlon_ref)}.")
                print("[SHAKEuq] unified mismatches (version, shape):", mism[:10], "..." if len(mism) > 10 else "")
    



    # ============================
    # PATCH: CDI behavior hard-fix
    # Paste EVERYTHING below at the END of your current SHAKEuq.py
    # ============================
    
    def _shakeuq__norm_abs(p):
        if p is None:
            return None
        if not isinstance(p, str):
            return None
        p = p.strip().strip('"').strip("'")
        if not p:
            return None
        return os.path.abspath(os.path.expanduser(p))
    
    
    def _shakeuq__is_file(p):
        try:
            return bool(p) and os.path.isfile(p)
        except Exception:
            return False
    
    
    def _shakeuq__is_dir(p):
        try:
            return bool(p) and os.path.isdir(p)
        except Exception:
            return False
    
    
    def _shakeuq__safe_glob(patterns):
        hits = []
        for pat in patterns:
            try:
                hits += glob.glob(pat, recursive=True)
            except Exception:
                pass
        hits = [h for h in hits if os.path.isfile(h)]
        # dedupe while keeping order
        seen = set()
        out = []
        for h in hits:
            if h not in seen:
                out.append(h)
                seen.add(h)
        return out
    
    
    def _shakeuq__score_cdi_path(p):
        b = os.path.basename(p).lower()
        sc = 0
        if "cdi" in b:
            sc += 10
        if "geo" in b:
            sc += 10
        if "1km" in b:
            sc += 3
        if "_cdi_" in b:
            sc += 2
        if b.endswith(".txt"):
            sc += 1
        return sc
    
    
    def _shakeuq__find_cdi_in_dir(self, folder, event_id=None):
        """
        Robust CDI file finder for a DYFI folder (Windows-safe).
        If event_id is given, prefer matching event_id.
        """
        if not _shakeuq__is_dir(folder):
            return None, "cdi_dir_missing"
    
        folder = _shakeuq__norm_abs(folder)
        eid = (event_id or "").lower().strip()
    
        patterns = [
            os.path.join(folder, "**", "*cdi*geo*1km*.txt"),
            os.path.join(folder, "**", "*cdi*geo*.txt"),
            os.path.join(folder, "**", "*_cdi_*geo*.txt"),
            os.path.join(folder, "**", "*cdi*.txt"),
            os.path.join(folder, "**", "*cdi*.csv"),
            os.path.join(folder, "**", "*dyfi*geo*.txt"),
            os.path.join(folder, "**", "*dyfi*.txt"),
        ]
        hits = _shakeuq__safe_glob(patterns)
        if not hits:
            return None, "cdi_dir_no_match"
    
        # prefer event_id matches if available
        if eid:
            eid_hits = [h for h in hits if eid in os.path.basename(h).lower()]
            if eid_hits:
                hits = eid_hits
    
        hits = sorted(hits, key=lambda p: (_shakeuq__score_cdi_path(p), len(os.path.basename(p))), reverse=True)
        return hits[0], "cdi_dir_match"
    
    
    def _shakeuq__resolve_cdi_input(self, dyfi_cdi_file, event_id=None):
        """
        New CDI behavior:
          - if dyfi_cdi_file is a FILE: use exactly that file (fixed).
          - if dyfi_cdi_file is a DIR: search inside and pick best match.
          - if dyfi_cdi_file is None: do NOT auto-search anywhere (explicit only).
        """
        if dyfi_cdi_file is None:
            return None, "cdi_none_explicit"
    
        p = _shakeuq__norm_abs(dyfi_cdi_file)
        if _shakeuq__is_file(p):
            return p, "cdi_input=file"
        if _shakeuq__is_dir(p):
            f, note = _shakeuq__find_cdi_in_dir(p, event_id=event_id)
            return f, note
        return None, "cdi_input_invalid"
    
    
     # ======================================================================
    # DROP-IN METHODS (NO "PATCH" NAMING)
    # Paste this whole block at the END of SHAKEuq.py
    # - No _PATCH__ function names
    # - No "PATCH" print banners
    # - Still attaches methods onto SHAKEuq (because you're pasting at file end)
    # ======================================================================
    

    
    # ----------------------------
    # CDI: explicit-only resolver
    # ----------------------------
    def resolve_cdi_path(self, cdi_input):
        """
        Explicit-only CDI resolution.
        - If cdi_input is a FILE: use it if it exists
        - If cdi_input is a DIR: find a CDI-like file inside (best effort)
        - If cdi_input is None/empty: return (None, "cdi_none_explicit")
        """
        cdi_path, note = _shakeuq__resolve_cdi_input(cdi_input, event_id=self.event_id)
        return cdi_path, note
    
    
    def uq_build_dataset(self):
        """
        Builds per-version dataset + unified grids + sanity table.
        Forces:
          - CDI resolution ONCE (explicit-only) and reuses fixed path for all versions.
          - CDI gate by include_cdi_from_version.
        """
        if not self.version_list:
            raise ValueError("version_list is empty.")
    
        # Resolve CDI ONCE (explicit-only) and reuse for every version.
        cdi_path, cdi_resolve_note = self.resolve_cdi_path(self.dyfi_cdi_input)
    
        sanity_rows = []
        per_version = {}
    
        for vkey in self.version_list:
            paths = self._discover_version_paths(vkey)
            exists = {k: (p is not None and os.path.exists(p)) for k, p in paths.items()}
    
            grid_pack = None
            uncert_pack = None
            meta_pack: dict = {}
    
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
                "cdi_path": None,
                "cdi_loaded": False,
                "cdi_note": "",
            }
    
            if use_cdi and cdi_path and os.path.exists(cdi_path):
                tmp = self._read_cdi_file(cdi_path)
                cdi_pack.update(tmp)
                cdi_pack["cdi_path"] = cdi_path
                cdi_pack["cdi_loaded"] = isinstance(tmp.get("df"), pd.DataFrame) and not tmp["df"].empty
                cdi_pack["cdi_note"] = cdi_resolve_note
            else:
                cdi_pack["cdi_path"] = cdi_path
                cdi_pack["cdi_loaded"] = False
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
                    "rupture_loaded": rupture_loaded,
                    "n_instruments": n_inst,
                    "n_dyfi_stationlist": n_dyfi_sl,
                    "use_cdi": use_cdi,
                    "cdi_path": cdi_pack.get("cdi_path"),
                    "cdi_loaded": bool(cdi_pack.get("cdi_loaded")),
                    "n_cdi": n_cdi,
                    "cdi_note": cdi_pack.get("cdi_note"),
                    "magnitude": _safe_float(meta_pack.get("magnitude")),
                    "depth_km": _safe_float(meta_pack.get("depth")),
                    "event_lat": _safe_float(meta_pack.get("lat")),
                    "event_lon": _safe_float(meta_pack.get("lon")),
                    "code_version": meta_pack.get("code_version"),
                    "map_status": meta_pack.get("map_status"),
                }
            )
    
            if self.verbose:
                print(
                    f"[SHAKEuq] parsed v={vkey} grid={exists.get('grid_xml')} unc={exists.get('uncertainty_xml')} "
                    f"stations={exists.get('stationlist_json')} rupture_loaded={rupture_loaded} "
                    f"use_cdi={use_cdi} cdi_loaded={bool(cdi_pack.get('cdi_loaded'))} n_cdi={n_cdi}"
                )
    
        self.uq_state["versions"] = per_version
        self._build_unified_grids()
        self.uq_state["sanity"] = pd.DataFrame(sanity_rows)
        return self.uq_state
    
    
    # ----------------------------
    # Raw ShakeMap baseline extractor
    # ----------------------------
    def extract_raw_shakemap(self, imt: str):
        """
        Baseline extractor for raw USGS ShakeMap data (IMT-aware).
    
        Returns dict with:
          - per-version mean/sigma grid availability
          - counts for instruments/dyfi_stationlist/cdi
          - summary DataFrame + log lines
        """
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
    
    
    # ----------------------------
    # Units + VS30 helpers
    # ----------------------------
    def parse_grid_field_units_from_xml(self, xml_path: str):
        """
        Parse <grid_field name="..." units="..."/> from a ShakeMap-style grid xml.
        Returns dict[name -> units]. Safe: returns {} on failure.
        """
        try:
            if not xml_path or not os.path.exists(xml_path):
                return {}
            tree = ET.parse(xml_path)
            root = tree.getroot()
            ns = {"sm": "http://earthquake.usgs.gov/eqcenter/shakemap"}
            units = {}
            for gf in root.findall(".//sm:grid_field", ns):
                name = gf.attrib.get("name")
                u = gf.attrib.get("units")
                if name:
                    units[str(name).upper()] = str(u) if u is not None else None
            return units
        except Exception:
            return {}
    
    
    def ensure_units_cache(self, version: str):
        """
        Ensures vpack has:
          vpack["grid_units"]  : dict[field -> units] from grid.xml
          vpack["uncert_units"]: dict[field -> units] from uncertainty.xml
        """
        vkey = str(version)
        vpack = self.uq_state.get("versions", {}).get(vkey, {})
        if not vpack:
            return
    
        if "grid_units" not in vpack:
            grid_xml = (vpack.get("paths") or {}).get("grid_xml")
            vpack["grid_units"] = self.parse_grid_field_units_from_xml(grid_xml) if grid_xml else {}
    
        if "uncert_units" not in vpack:
            unc_xml = (vpack.get("paths") or {}).get("uncertainty_xml")
            vpack["uncert_units"] = self.parse_grid_field_units_from_xml(unc_xml) if unc_xml else {}
    
    
    def get_vs30_grid(self, version: str):
        """
        Return (vs30_grid_2d, source_field_name, units) for a version.
        Priority:
          1) VS30 field if present
          2) SVEL field treated as VS30 proxy (common in m/s)
        """
        vkey = str(version)
        vpack = self.uq_state.get("versions", {}).get(vkey, {})
        grid = (vpack.get("grid") or {}).get("fields", {}) if vpack else {}
        if not grid:
            return None, None, None
    
        self.ensure_units_cache(vkey)
        units_map = vpack.get("grid_units", {}) or {}
    
        if "VS30" in grid:
            return grid["VS30"], "VS30", units_map.get("VS30")
        if "SVEL" in grid:
            return grid["SVEL"], "SVEL", units_map.get("SVEL")
    
        return None, None, None
    
    
    def convert_units_array(self, arr, from_units: str, to_units: str):
        """
        Convert a numpy array between common ShakeMap units.
        Supported:
          - %g  <-> g
          - g   <-> m/s^2
          - cm/s <-> m/s
          - intensity stays intensity
        """
        if arr is None:
            return None
        if from_units is None or to_units is None:
            return arr
    
        fu = str(from_units).strip().lower()
        tu = str(to_units).strip().lower()
        if fu == tu:
            return arr
    
        out = np.asarray(arr, dtype=float)
    
        # intensity (no conversion)
        if fu in ("intensity", "mmi") or tu in ("intensity", "mmi"):
            return out
    
        # %g <-> g
        if fu in ("%g", "percent g", "pctg", "%(g)") and tu == "g":
            return out / 100.0
        if fu == "g" and tu in ("%g", "percent g", "pctg", "%(g)"):
            return out * 100.0
    
        # g <-> m/s^2
        g0 = 9.80665
        if fu == "g" and tu in ("m/s^2", "m/s2", "ms^-2"):
            return out * g0
        if fu in ("m/s^2", "m/s2", "ms^-2") and tu == "g":
            return out / g0
    
        # cm/s <-> m/s
        if fu in ("cm/s", "cmps") and tu in ("m/s", "mps"):
            return out / 100.0
        if fu in ("m/s", "mps") and tu in ("cm/s", "cmps"):
            return out * 100.0
    
        return out
    
    
    def to_log_space(self, arr_linear, space: str, eps: float = 1e-16):
        """
        Convert a positive linear array to ln-space if requested.
        space in {"linear","ln"}.
        """
        if arr_linear is None:
            return None
        s = str(space).lower().strip()
        if s in ("linear", "lin", "native"):
            return arr_linear
        if s in ("ln", "log", "loge"):
            x = np.maximum(np.asarray(arr_linear, dtype=float), eps)
            return np.log(x)
        return arr_linear
    
    
    def from_log_space(self, arr_log, space: str):
        """
        Convert ln-space array back to linear if requested.
        """
        if arr_log is None:
            return None
        s = str(space).lower().strip()
        if s in ("linear", "lin", "native"):
            return arr_log
        if s in ("ln", "log", "loge"):
            return np.exp(np.asarray(arr_log, dtype=float))
        return arr_log
    
    
    def get_shakemap_layer(self, version: str, imt: str):
        """
        Returns (mean_2d, mean_units, sigma_2d, sigma_units).
        sigma layer uses prefer_sigma_field_prefix + IMT (e.g., "STDMMI").
        """
        vkey = str(version)
        imt_u = str(imt).upper().strip()
        vpack = self.uq_state.get("versions", {}).get(vkey, {})
        if not vpack:
            return None, None, None, None
    
        grid_fields = (vpack.get("grid") or {}).get("fields", {}) or {}
        unc_fields = (vpack.get("uncert") or {}).get("fields", {}) or {}
    
        self.ensure_units_cache(vkey)
        grid_units = vpack.get("grid_units", {}) or {}
        unc_units = vpack.get("uncert_units", {}) or {}
    
        mean = grid_fields.get(imt_u)
        mean_u = grid_units.get(imt_u)
    
        sig_name = f"{self.prefer_sigma_field_prefix}{imt_u}".upper()
        sigma = unc_fields.get(sig_name)
        sigma_u = unc_units.get(sig_name)
    
        return mean, mean_u, sigma, sigma_u
    
    
    def normalize_shakemap_layer(
        self,
        version: str,
        imt: str,
        target_units: str = None,
        target_mean_space: str = "linear",
        target_sigma_space: str = "native",
    ):
        """
        Normalize ShakeMap mean + sigma to requested units/space.
    
        Mean:
          - optionally convert units (e.g., %g -> g)
          - optionally convert to ln-space
    
        Sigma:
          - default "native" (no guessing)
          - if user asks sigma_space="linear" and sigma units contain "ln", returns exp(sigma)
        """
        mean, mean_u, sigma, sigma_u = self.get_shakemap_layer(version, imt)
    
        if mean is None:
            return {
                "mean": None, "mean_units": mean_u, "mean_space": str(target_mean_space).lower().strip(),
                "sigma": None, "sigma_units": sigma_u, "sigma_space": str(target_sigma_space).lower().strip(),
            }
    
        # units conversion for mean
        if target_units:
            mean_conv = self.convert_units_array(mean, mean_u, target_units)
            mean_units_out = target_units
        else:
            mean_conv = mean
            mean_units_out = mean_u
    
        # mean space
        mean_out = self.to_log_space(mean_conv, target_mean_space)
        mean_space_out = str(target_mean_space).lower().strip()
    
        # sigma handling
        sigma_out = sigma
        sigma_units_out = sigma_u
        sigma_space_out = "native"
    
        if sigma is not None:
            ts = str(target_sigma_space).lower().strip()
            if ts in ("ln", "log", "loge"):
                sigma_space_out = "ln"
            elif ts in ("linear", "lin"):
                su = str(sigma_u or "").lower()
                if "ln" in su:
                    sigma_out = np.exp(np.asarray(sigma, dtype=float))
                sigma_space_out = "linear"
    
        return {
            "mean": mean_out,
            "mean_units": mean_units_out,
            "mean_space": mean_space_out,
            "sigma": sigma_out,
            "sigma_units": sigma_units_out,
            "sigma_space": sigma_space_out,
        }
    
    
    def collect_baseline_inputs(
        self,
        imt: str,
        target_units: str = None,
        target_mean_space: str = "linear",
        dyfi_source: str = "auto",
        include_vs30: bool = True,
    ):
        """
        Collect per-version baseline inputs for methods WITHOUT losing any stream:
          - ShakeMap mean/sigma for chosen IMT (normalized if requested)
          - VS30 grid (VS30 or SVEL)
          - Observations for chosen IMT (via build_observations)
          - Audit counts and a summary DataFrame
        """
        imt_u = str(imt).upper().strip()
        log = []
        out = {"imt": imt_u, "versions": {}, "summary": None, "log": log}
    
        if "versions" not in self.uq_state:
            raise RuntimeError("uq_state['versions'] missing. Run uq_build_dataset() first.")
    
        rows = []
        for vkey in list(self.uq_state["versions"].keys()):
            vpack = self.uq_state["versions"][vkey]
    
            layer = self.normalize_shakemap_layer(
                version=vkey,
                imt=imt_u,
                target_units=target_units,
                target_mean_space=target_mean_space,
            )
            mean2d = layer["mean"]
            sig2d = layer["sigma"]
            n_grid = int(mean2d.size) if isinstance(mean2d, np.ndarray) else 0
    
            vs30_2d, vs30_src, vs30_units = (None, None, None)
            if include_vs30:
                vs30_2d, vs30_src, vs30_units = self.get_vs30_grid(vkey)
    
            try:
                if imt_u == "MMI":
                    obs = self.build_observations(version=vkey, imt="MMI", dyfi_source=dyfi_source)
                else:
                    obs = self.build_observations(version=vkey, imt=imt_u)
            except Exception:
                obs = pd.DataFrame()
    
            stations = vpack.get("stations", {}) or {}
            cdi = vpack.get("cdi", {}) or {}
            n_inst = int(stations.get("instruments", pd.DataFrame()).shape[0]) if isinstance(stations.get("instruments"), pd.DataFrame) else 0
            n_dyfi = int(stations.get("dyfi_stationlist", pd.DataFrame()).shape[0]) if isinstance(stations.get("dyfi_stationlist"), pd.DataFrame) else 0
            n_cdi = int(cdi.get("df", pd.DataFrame()).shape[0]) if isinstance(cdi.get("df"), pd.DataFrame) else 0
            n_obs = int(obs.shape[0]) if isinstance(obs, pd.DataFrame) else 0
    
            audit = {
                "n_grid_points": n_grid,
                "n_obs_used": n_obs,
                "n_instruments_raw": n_inst,
                "n_dyfi_stationlist_raw": n_dyfi,
                "n_cdi_raw": n_cdi,
                "mean_units": layer["mean_units"],
                "mean_space": layer["mean_space"],
                "sigma_units": layer["sigma_units"],
                "sigma_space": layer["sigma_space"],
                "vs30_source": vs30_src,
                "vs30_units": vs30_units,
                "has_mean": bool(isinstance(mean2d, np.ndarray)),
                "has_sigma": bool(isinstance(sig2d, np.ndarray)),
                "has_vs30": bool(isinstance(vs30_2d, np.ndarray)),
            }
    
            log.append(
                f"[BASE] v={vkey} imt={imt_u} "
                f"mean={'Y' if audit['has_mean'] else 'N'} "
                f"sigma={'Y' if audit['has_sigma'] else 'N'} "
                f"vs30={'Y' if audit['has_vs30'] else 'N'}({vs30_src}) "
                f"grid_pts={n_grid} obs={n_obs} inst={n_inst} dyfi={n_dyfi} cdi={n_cdi} "
                f"mean_units={audit['mean_units']} sigma_units={audit['sigma_units']}"
            )
    
            out["versions"][vkey] = {
                "mean": mean2d,
                "sigma": sig2d,
                "vs30": vs30_2d,
                "obs": obs,
                "audit": audit,
            }
    
            rows.append(
                {
                    "version": vkey,
                    "imt": imt_u,
                    "has_mean": audit["has_mean"],
                    "has_sigma": audit["has_sigma"],
                    "has_vs30": audit["has_vs30"],
                    "vs30_source": vs30_src,
                    "n_grid_points": n_grid,
                    "n_obs_used": n_obs,
                    "n_instruments_raw": n_inst,
                    "n_dyfi_stationlist_raw": n_dyfi,
                    "n_cdi_raw": n_cdi,
                    "mean_units": audit["mean_units"],
                    "sigma_units": audit["sigma_units"],
                    "mean_space": audit["mean_space"],
                }
            )
    
        out["summary"] = pd.DataFrame(rows)
        return out
    

    # ======================================================================
    # CDI RESOLVER FIX (module-scope helper)
    # Paste this at the VERY END of SHAKEuq.py (OUTSIDE the class)
    #
    # This fixes:
    #   NameError: name '_shakeuq__resolve_cdi_input' is not defined
    # because SHAKEuq.resolve_cdi_path() calls this helper.
    # ======================================================================
    def _shakeuq__resolve_cdi_input(cdi_input, event_id: str = None):
        """
        Explicit-only CDI resolution.
        - FILE  -> use it if exists
        - DIR   -> find a CDI-like file inside (best effort)
        - None  -> (None, "cdi_none_explicit")
        Returns: (cdi_path_or_None, note)
        """
        if cdi_input is None:
            return None, "cdi_none_explicit"
    
        if not isinstance(cdi_input, str):
            return None, "cdi_invalid_type"
    
        p = cdi_input.strip()
        if not p:
            return None, "cdi_empty_explicit"
    
        p_abs = os.path.abspath(p)
    
        # 1) direct file path
        if os.path.isfile(p_abs):
            if os.path.exists(p_abs):
                return p_abs, "cdi_input=file"
            return None, "cdi_input=file_missing"
    
        # 2) directory path -> best-effort file search
        if os.path.isdir(p_abs):
            # Common DYFI CDI exports are *.txt; prefer names containing "cdi"
            patterns = [
                os.path.join(p_abs, "*cdi*geo*km*.txt"),
                os.path.join(p_abs, "*cdi*geo*.txt"),
                os.path.join(p_abs, "*cdi*.txt"),
                os.path.join(p_abs, "*.txt"),
            ]
            candidates = []
            for pat in patterns:
                candidates.extend(glob.glob(pat))
    
            # filter down: must be a file and non-empty
            candidates = [c for c in candidates if os.path.isfile(c)]
            if not candidates:
                return None, "cdi_input=dir_no_match"
    
            # prefer: contains event_id if given, then shortest name, then newest mtime
            def _score(fp):
                name = os.path.basename(fp).lower()
                has_eid = (event_id is not None) and (str(event_id).lower() in name)
                size = 0
                try:
                    size = os.path.getsize(fp)
                except Exception:
                    size = 0
                mtime = 0
                try:
                    mtime = os.path.getmtime(fp)
                except Exception:
                    mtime = 0
                return (1 if has_eid else 0, 1 if size > 0 else 0, -len(name), mtime)
    
            candidates.sort(key=_score, reverse=True)
            best = os.path.abspath(candidates[0])
            if os.path.exists(best):
                return best, "cdi_input=dir_best_match"
            return None, "cdi_input=dir_best_missing"
    
        # 3) path is neither file nor dir
        return None, "cdi_input=path_not_found"




    # ============================================================
    # FIXED METHODS (paste INSIDE class SHAKEuq, replacing existing)
    # - no external helper dependency
    # - dyfi_cdi_file can be a FILE or DIR (or None)
    # - explicit-only: if None -> no auto-search
    # ============================================================
    
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
    
    
    def uq_build_dataset(self) -> Dict[str, Any]:
        """
        Builds per-version dataset + unified grids + sanity table.
        Uses explicit-only CDI resolution:
          - resolves CDI ONCE from self.dyfi_cdi_input
          - applies CDI only when version gate is on
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
                    "rupture_loaded": rupture_loaded,
                    "n_instruments": n_inst,
                    "n_dyfi_stationlist": n_dyfi_sl,
                    "use_cdi": use_cdi,
                    "cdi_path": cdi_pack.get("cdi_path"),
                    "cdi_loaded": bool(cdi_pack.get("cdi_loaded")),
                    "n_cdi": n_cdi,
                    "cdi_note": cdi_pack.get("cdi_note"),
                    "magnitude": _safe_float(meta_pack.get("magnitude")),
                    "depth_km": _safe_float(meta_pack.get("depth")),
                    "event_lat": _safe_float(meta_pack.get("lat")),
                    "event_lon": _safe_float(meta_pack.get("lon")),
                    "code_version": meta_pack.get("code_version"),
                    "map_status": meta_pack.get("map_status"),
                }
            )
    
            if self.verbose:
                print(
                    f"[SHAKEuq] parsed v={vkey} grid={exists.get('grid_xml')} unc={exists.get('uncertainty_xml')} "
                    f"stations={exists.get('stationlist_json')} rupture_loaded={rupture_loaded} "
                    f"use_cdi={use_cdi} cdi_loaded={bool(cdi_pack.get('cdi_loaded'))} n_cdi={n_cdi}"
                )
    
        self.uq_state["versions"] = per_version
        self._build_unified_grids()
        self.uq_state["sanity"] = pd.DataFrame(sanity_rows)
        return self.uq_state

