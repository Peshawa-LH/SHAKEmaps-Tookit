"""
SHAKEdataset: dataset builder for SHAKEuq workflows.

Extracted from SHAKEuq to isolate dataset construction from UQ methods.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class SHAKEdataset:
    """
    Dataset builder for SHAKEuq.

    Parameters mirror SHAKEuq dataset inputs to preserve compatibility.
    """

    def __init__(
        self,
        event_id: str,
        event_time=None,
        shakemap_folder: Optional[str] = None,
        rupture_folder: Optional[str] = None,
        stations_folder: Optional[str] = None,
        dyfi_cdi_file: Optional[str] = None,
        version_list=None,
        include_cdi_from_version: int = 4,
        base_folder: str = "./export/SHAKEuq",
        verbose: bool = True,
        **kwargs,
    ):
        self.event_id = str(event_id)
        self.event_time = self._parse_event_time(event_time)
        self.shakemap_folder = os.path.normpath(shakemap_folder) if shakemap_folder else None
        self.rupture_folder = os.path.normpath(rupture_folder) if rupture_folder else None
        self.stations_folder = os.path.normpath(stations_folder) if stations_folder else None
        self.dyfi_cdi_file = os.path.normpath(dyfi_cdi_file) if dyfi_cdi_file else None
        self.version_list = list(version_list) if version_list is not None else None
        self.cdi_attach_from_version = int(include_cdi_from_version) if include_cdi_from_version is not None else 4
        self.base_folder = base_folder
        self.verbose = bool(verbose)

        self.dyfi_source = str(kwargs.get("dyfi_source", "stationlist")).lower().strip()
        self.dyfi_cdi_max_dist_km = float(kwargs.get("dyfi_cdi_max_dist_km", 400.0))
        self.dyfi_cdi_min_nresp = int(kwargs.get("dyfi_cdi_min_nresp", 1))
        self.dyfi_weight_rule = str(kwargs.get("dyfi_weight_rule", "nresp_threshold"))
        self.dyfi_weight_threshold = int(kwargs.get("dyfi_weight_threshold", 3))
        self.dyfi_weight_low = float(kwargs.get("dyfi_weight_low", 1.0))
        self.dyfi_weight_high = float(kwargs.get("dyfi_weight_high", 2.0))
        self.dyfi_weight_max = float(kwargs.get("dyfi_weight_max", 10.0))

        self._dyfi_cdi_df_cache = None
        self.dataset_state: Optional[Dict[str, Any]] = None

    def _parse_event_time(self, event_time):
        if isinstance(event_time, datetime):
            return event_time
        if event_time:
            s = str(event_time).strip()
            try:
                return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            except Exception:
                try:
                    return datetime.fromisoformat(s.replace("Z", ""))
                except Exception:
                    return None
        return None

    def validate_paths(self, strict: bool = True) -> Dict[str, Any]:
        """
        Validate expected input paths for each version.
        """
        if self.version_list is None:
            return {"status": "MISSING", "reason": "version_list is None"}

        results = {
            "event_id": self.event_id,
            "versions": {},
            "ok": True,
        }
        for v in self.version_list:
            grid_path, unc_path, st_path, rup_path, _ = self._uq_resolve_paths(
                int(v), stations_folder=self.stations_folder, rupture_folder=self.rupture_folder
            )
            vres = {
                "grid_xml": "OK" if grid_path.exists() else "MISSING",
                "uncertainty_xml": "OK" if unc_path is not None and Path(unc_path).exists() else "MISSING",
                "stationlist_json": "OK" if Path(st_path).exists() else "MISSING",
                "rupture_json": "OK" if Path(rup_path).exists() else "MISSING",
            }
            if strict and any(v != "OK" for v in vres.values()):
                results["ok"] = False
            results["versions"][int(v)] = vres
        return results

    def summary(self) -> Dict[str, Any]:
        """Return summary counts from the latest dataset state."""
        if self.dataset_state is None:
            return {"status": "EMPTY"}
        state = self.dataset_state
        versions = state.get("version_list", [])
        obs_by_version = state.get("obs_by_version", {})
        counts = {
            "versions": len(versions),
            "stations": 0,
            "dyfi": 0,
            "cdi": 0,
        }
        for v in versions:
            obs = obs_by_version.get(v, {})
            counts["stations"] += len(obs.get("obs_seismic", []) or [])
            counts["dyfi"] += len(obs.get("obs_intensity_stationlist", []) or [])
            counts["cdi"] += len(obs.get("obs_intensity_cdi", []) or [])
        unified_spec = state.get("unified_spec", {})
        grid_shape = (
            unified_spec.get("nlat", None),
            unified_spec.get("nlon", None),
        )
        return {
            "versions": len(versions),
            "counts": counts,
            "unified_grid_shape": grid_shape,
            "has_unified": bool(state.get("unified_axes")),
        }

    def build(
        self,
        version_list=None,
        base_folder: Optional[str] = None,
        stations_folder: Optional[str] = None,
        rupture_folder: Optional[str] = None,
        imts=("MMI", "PGA", "PGV", "PSA"),
        grid_unify: str = "intersection",
        resolution: str = "finest",
        export: bool = True,
        interp_method: str = "nearest",
        interp_kwargs: dict = None,
        output_units: dict = None,
    ) -> Dict[str, Any]:
        """
        Build dataset state for SHAKEuq.
        """
        import json
        import warnings

        version_list = list(version_list) if version_list is not None else list(self.version_list or [])
        version_list = [int(v) for v in list(version_list)]
        interp_kwargs = {} if interp_kwargs is None else dict(interp_kwargs)

        bf = Path(base_folder or self.base_folder).expanduser()
        if bf.name == "uq" and bf.parent.name == str(self.event_id):
            base = bf
        elif bf.name.lower() == "SHAKEuq":
            base = bf / str(self.event_id) / "uq"
        else:
            parts_lower = [p.lower() for p in bf.parts]
            if "SHAKEuq" in parts_lower:
                base = bf / str(self.event_id) / "uq"
            else:
                base = bf / "SHAKEuq" / str(self.event_id) / "uq"

        base = self._uq_ensure_dir(base)

        per_avail = self.uq_list_available_imts(
            version_list, stations_folder=stations_folder, rupture_folder=rupture_folder
        )
        global_imts = sorted({k for vv in per_avail.values() for k in vv})
        requested = self._uq_expand_requested_imts(imts, global_imts)

        versions_raw = {}
        file_traces = {}
        sanity_rows = []
        obs_by_version = {}
        cdi_df = self._uq_load_dyfi_cdi_df() if getattr(self, "dyfi_cdi_file", None) else None

        for v in version_list:
            grid_path, unc_path, st_path, rup_path, trace = self._uq_resolve_paths(
                v, stations_folder=stations_folder, rupture_folder=rupture_folder
            )
            file_traces[v] = trace

            raw = {
                "version": v,
                "grid_path": str(grid_path),
                "unc_path": str(unc_path) if unc_path is not None else "",
                "station_path": str(st_path),
                "rupture_path": str(rup_path),
                "grid_spec": None,
                "mean_fields": {},
                "mean_units": {},
                "sigma_fields_xml": {},
                "sigma_units_xml": {},
                "sigma_fields": {},
                "sigma_units": {},
                "vs30": None,
                "lats_1d": None,
                "lons_1d": None,
                "stations": {"instrumented": [], "dyfi": []},
                "counts": {"n_instrumented": 0, "n_dyfi": 0},
                "station_parse_debug": {},
                "rupture_loaded": bool(rup_path.exists()),
                "uncertainty_xml_exists": bool(unc_path is not None and Path(unc_path).exists()),
                "stationlist_exists": bool(Path(st_path).exists()),
            }

            if Path(grid_path).exists():
                spec, mean_fields, vs30, mean_units = self._uq_parse_grid_xml(grid_path)
                raw["grid_spec"] = spec
                raw["mean_fields"] = mean_fields
                raw["mean_units"] = mean_units or {}
                raw["vs30"] = vs30
                lats, lons = self._uq_axes_from_spec(spec)
                raw["lats_1d"] = lats
                raw["lons_1d"] = lons
            else:
                warnings.warn(f"[UQ] Missing grid XML for v{v}: {grid_path}")

            if raw["uncertainty_xml_exists"]:
                try:
                    _, sig_fields_xml, sig_units_xml = self._uq_parse_uncertainty_xml(unc_path)
                    raw["sigma_fields_xml"] = sig_fields_xml or {}
                    raw["sigma_units_xml"] = sig_units_xml or {}

                    mapped = {}
                    mapped_units = {}
                    for sf_name, sf_grid in raw["sigma_fields_xml"].items():
                        imt_name = self._uq_sigma_field_to_imt(sf_name)
                        mapped[str(imt_name)] = sf_grid
                        if sf_name in raw["sigma_units_xml"]:
                            mapped_units[str(imt_name)] = raw["sigma_units_xml"][sf_name]
                    raw["sigma_fields"] = mapped
                    raw["sigma_units"] = mapped_units

                except Exception as e:
                    warnings.warn(f"[UQ] Failed parsing uncertainty XML for v{v}: {e}")
                    raw["sigma_fields_xml"] = {}
                    raw["sigma_units_xml"] = {}
                    raw["sigma_fields"] = {}
                    raw["sigma_units"] = {}

            for imt in requested:
                if imt not in raw["sigma_fields"]:
                    m = raw["mean_fields"].get(imt, None)
                    raw["sigma_fields"][imt] = self._uq_default_sigma_for_imt(imt, m)
                    if imt not in raw["sigma_units"]:
                        raw["sigma_units"][imt] = None

            if raw["stationlist_exists"]:
                try:
                    inst, dyfi, dbg = self._uq_parse_stationlist_with_usgsparser(st_path)
                    raw["stations"]["instrumented"] = inst
                    raw["stations"]["dyfi"] = dyfi
                    raw["counts"]["n_instrumented"] = int(len(inst))
                    raw["counts"]["n_dyfi"] = int(len(dyfi))
                    raw["station_parse_debug"] = dbg
                except Exception as e:
                    warnings.warn(f"[UQ] Stationlist parsing failed for v{v}: {e}")

            versions_raw[v] = raw
            obs_by_version[v] = self._uq_build_obs_pool_for_version(int(v), raw, cdi_df=cdi_df)

            sanity_rows.append(
                {
                    "version": v,
                    "grid_shape": None if raw["grid_spec"] is None else (raw["grid_spec"]["nlat"], raw["grid_spec"]["nlon"]),
                    "n_instrumented": raw["counts"]["n_instrumented"],
                    "n_dyfi": raw["counts"]["n_dyfi"],
                    "rupture_loaded": raw["rupture_loaded"],
                    "uncertainty_xml_exists": raw["uncertainty_xml_exists"],
                    "stationlist_exists": raw["stationlist_exists"],
                    "aligned_to_unified": False,
                    "pga_df_rows": raw.get("station_parse_debug", {}).get("pga_df_rows", 0),
                    "pga_rows_after_lonlat_filter": raw.get("station_parse_debug", {}).get("pga_rows_after_lonlat_filter", 0),
                    "pga_rows_after_value_filter": raw.get("station_parse_debug", {}).get("pga_rows_after_value_filter", 0),
                    "pgv_df_rows": raw.get("station_parse_debug", {}).get("pgv_df_rows", 0),
                    "sa_df_rows": raw.get("station_parse_debug", {}).get("sa_df_rows", 0),
                    "mmi_df_rows": raw.get("station_parse_debug", {}).get("mmi_df_rows", 0),
                    "mmi_rows_after_lonlat_filter": raw.get("station_parse_debug", {}).get("mmi_rows_after_lonlat_filter", 0),
                    "mmi_rows_after_intensity_filter": raw.get("station_parse_debug", {}).get("mmi_rows_after_intensity_filter", 0),
                    "station_parse_note": raw.get("station_parse_debug", {}).get("note", ""),
                    "mean_units_keys": sorted(list((raw.get("mean_units") or {}).keys())),
                    "sigma_units_keys": sorted(list((raw.get("sigma_units") or {}).keys())),
                }
            )

        specs = [versions_raw[v]["grid_spec"] for v in version_list if versions_raw[v]["grid_spec"] is not None]
        if not specs:
            raise FileNotFoundError("[UQ] No grid XML found across versions; cannot build dataset.")

        unified_spec = self._uq_build_unified_spec(specs, grid_unify=grid_unify, resolution=resolution)
        ulats_1d, ulons_1d = self._uq_axes_from_spec(unified_spec)
        ULON2, ULAT2 = np.meshgrid(ulons_1d, ulats_1d)

        unified = {}
        for v in version_list:
            raw = versions_raw[v]
            if raw["grid_spec"] is None:
                continue

            slats = np.asarray(raw["lats_1d"], dtype=float)
            slons = np.asarray(raw["lons_1d"], dtype=float)

            uv = {
                "version": v,
                "unified_mean": {},
                "unified_sigma_prior_total": {},
                "unified_vs30": None,
                "aligned_to_unified": True,
                "unified_mean_units": dict(raw.get("mean_units") or {}),
                "unified_sigma_units": dict(raw.get("sigma_units") or {}),
                "interp_method": str(interp_method),
            }

            if raw["vs30"] is not None:
                uv["unified_vs30"] = self._uq_interp_to_unified(
                    slats, slons, raw["vs30"], ULAT2, ULON2, method=interp_method, **interp_kwargs
                )
            else:
                uv["unified_vs30"] = np.full((unified_spec["nlat"], unified_spec["nlon"]), np.nan, dtype=float)

            for imt in requested:
                if imt in raw["mean_fields"]:
                    uv["unified_mean"][imt] = self._uq_interp_to_unified(
                        slats, slons, raw["mean_fields"][imt], ULAT2, ULON2, method=interp_method, **interp_kwargs
                    )
                else:
                    uv["unified_mean"][imt] = np.full((unified_spec["nlat"], unified_spec["nlon"]), np.nan, dtype=float)

                sig = raw["sigma_fields"].get(imt, None)
                if sig is None:
                    uv["unified_sigma_prior_total"][imt] = self._uq_default_sigma_for_imt(imt, uv["unified_mean"][imt])
                elif isinstance(sig, (float, int)):
                    uv["unified_sigma_prior_total"][imt] = np.full((unified_spec["nlat"], unified_spec["nlon"]), float(sig), dtype=float)
                else:
                    sig = np.asarray(sig, dtype=float)
                    if sig.shape == (raw["grid_spec"]["nlat"], raw["grid_spec"]["nlon"]):
                        uv["unified_sigma_prior_total"][imt] = self._uq_interp_to_unified(
                            slats, slons, sig, ULAT2, ULON2, method=interp_method, **interp_kwargs
                        )
                    elif sig.shape == (unified_spec["nlat"], unified_spec["nlon"]):
                        uv["unified_sigma_prior_total"][imt] = sig
                    else:
                        uv["unified_sigma_prior_total"][imt] = np.full(
                            (unified_spec["nlat"], unified_spec["nlon"]), float(np.nanmedian(sig)), dtype=float
                        )

            unified[v] = uv
            for row in sanity_rows:
                if row["version"] == v:
                    row["aligned_to_unified"] = True

        if export:
            with open(base / "uq_unified_grid_spec.json", "w", encoding="utf-8") as f:
                json.dump(unified_spec, f, indent=2)

            np.savez_compressed(
                base / "uq_unified_axes.npz",
                lats_1d=ulats_1d,
                lons_1d=ulons_1d,
                lat2d=ULAT2,
                lon2d=ULON2,
            )

            for v, uv in unified.items():
                vdir = self._uq_ensure_dir(base / f"v{int(v)}")
                np.savez_compressed(vdir / "uq_unified_mean.npz", **{k: uv["unified_mean"][k] for k in uv["unified_mean"]})
                np.savez_compressed(
                    vdir / "uq_unified_sigma_prior_total.npz",
                    **{k: uv["unified_sigma_prior_total"][k] for k in uv["unified_sigma_prior_total"]},
                )
                np.savez_compressed(vdir / "uq_unified_vs30.npz", vs30=uv["unified_vs30"])

                with open(vdir / "uq_unified_units.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "mean_units": uv.get("unified_mean_units", {}),
                            "sigma_units": uv.get("unified_sigma_units", {}),
                            "interp_method": uv.get("interp_method", None),
                        },
                        f,
                        indent=2,
                    )

            with open(base / "uq_file_trace.json", "w", encoding="utf-8") as f:
                json.dump(file_traces, f, indent=2)

            with open(base / "uq_sanity_table.json", "w", encoding="utf-8") as f:
                json.dump(sanity_rows, f, indent=2)

            if output_units is not None:
                with open(base / "uq_output_units_requested.json", "w", encoding="utf-8") as f:
                    json.dump(dict(output_units), f, indent=2)

        self.dataset_state = {
            "event_id": str(self.event_id),
            "version_list": version_list,
            "requested_imts": requested,
            "per_version_available_imts": per_avail,
            "unified_spec": unified_spec,
            "unified_axes": {"lats_1d": ulats_1d, "lons_1d": ulons_1d, "lat2d": ULAT2, "lon2d": ULON2},
            "versions_raw": versions_raw,
            "versions_unified": unified,
            "obs_by_version": obs_by_version,
            "sanity_rows": sanity_rows,
            "file_traces": file_traces,
            "base_folder": str(base),
            "stations_folder_used": str(stations_folder) if stations_folder is not None else None,
            "rupture_folder_used": str(rupture_folder) if rupture_folder is not None else None,
            "interp_method": str(interp_method),
            "interp_kwargs": dict(interp_kwargs),
            "output_units_requested": dict(output_units) if output_units is not None else None,
        }
        return self.dataset_state

    # ---------------------------
    # Utilities
    # ---------------------------
    def _uq_safe_float(self, x):
        try:
            if x is None:
                return None
            if isinstance(x, str):
                xs = x.strip()
                if xs == "" or xs.lower() in {"null", "none", "nan"}:
                    return None
            v = float(x)
            if not np.isfinite(v):
                return None
            return v
        except Exception:
            return None

    def _uq_isfinite(self, x) -> bool:
        v = self._uq_safe_float(x)
        return v is not None and np.isfinite(v)

    def _uq_ensure_dir(self, p):
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _uq_regular_axis(self, start: float, step: float, n: int):
        return start + step * np.arange(n, dtype=float)

    def _uq_bilinear_interp_regular_grid(
        self,
        src_lats_1d,
        src_lons_1d,
        src_field_2d,
        tgt_lats_2d,
        tgt_lons_2d,
        fill_value=float("nan"),
    ):
        src_lats = np.asarray(src_lats_1d, dtype=float)
        src_lons = np.asarray(src_lons_1d, dtype=float)
        Z = np.asarray(src_field_2d, dtype=float)

        if Z.ndim != 2:
            raise ValueError("src_field_2d must be 2D.")
        if src_lats.size < 2 or src_lons.size < 2:
            return np.full_like(tgt_lats_2d, fill_value, dtype=float)

        if src_lats[1] < src_lats[0]:
            src_lats = src_lats[::-1]
            Z = Z[::-1, :]
        if src_lons[1] < src_lons[0]:
            src_lons = src_lons[::-1]
            Z = Z[:, ::-1]

        lat0 = float(src_lats[0])
        lon0 = float(src_lons[0])
        dlat = float(src_lats[1] - src_lats[0])
        dlon = float(src_lons[1] - src_lons[0])
        if dlat == 0 or dlon == 0:
            return np.full_like(tgt_lats_2d, fill_value, dtype=float)

        fi = (tgt_lats_2d - lat0) / dlat
        fj = (tgt_lons_2d - lon0) / dlon
        i0 = np.floor(fi).astype(int)
        j0 = np.floor(fj).astype(int)
        di = fi - i0
        dj = fj - j0

        nlat, nlon = Z.shape
        valid = (i0 >= 0) & (i0 < nlat - 1) & (j0 >= 0) & (j0 < nlon - 1)
        out = np.full_like(tgt_lats_2d, fill_value, dtype=float)
        if not np.any(valid):
            return out

        i0v = i0[valid]
        j0v = j0[valid]
        div = di[valid]
        djv = dj[valid]

        z00 = Z[i0v, j0v]
        z10 = Z[i0v + 1, j0v]
        z01 = Z[i0v, j0v + 1]
        z11 = Z[i0v + 1, j0v + 1]

        w00 = (1.0 - div) * (1.0 - djv)
        w10 = div * (1.0 - djv)
        w01 = (1.0 - div) * djv
        w11 = div * djv

        outv = w00 * z00 + w10 * z10 + w01 * z01 + w11 * z11

        nanmask = ~np.isfinite(outv)
        if np.any(nanmask):
            zstack = np.vstack([z00, z10, z01, z11]).T
            wstack = np.vstack([w00, w10, w01, w11]).T
            finite = np.isfinite(zstack)
            ws = np.where(finite, wstack, 0.0)
            zs = np.where(finite, zstack, 0.0)
            wsum = ws.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                repaired = np.where(wsum > 0, (ws * zs).sum(axis=1) / wsum, fill_value)
            outv[nanmask] = repaired[nanmask]

        out[valid] = outv
        return out

    def _uq_nearest_interp_regular_grid(
        self,
        src_lats_1d,
        src_lons_1d,
        src_field_2d,
        tgt_lats_2d,
        tgt_lons_2d,
        fill_value=float("nan"),
    ):
        src_lats = np.asarray(src_lats_1d, dtype=float)
        src_lons = np.asarray(src_lons_1d, dtype=float)
        Z = np.asarray(src_field_2d, dtype=float)

        if Z.ndim != 2:
            raise ValueError("src_field_2d must be 2D.")
        if src_lats.size < 1 or src_lons.size < 1:
            return np.full_like(tgt_lats_2d, fill_value, dtype=float)

        if src_lats.size >= 2 and src_lats[1] < src_lats[0]:
            src_lats = src_lats[::-1]
            Z = Z[::-1, :]
        if src_lons.size >= 2 and src_lons[1] < src_lons[0]:
            src_lons = src_lons[::-1]
            Z = Z[:, ::-1]

        if src_lats.size < 2 or src_lons.size < 2:
            return np.full_like(tgt_lats_2d, fill_value, dtype=float)

        lat0 = float(src_lats[0])
        lon0 = float(src_lons[0])
        dlat = float(src_lats[1] - src_lats[0])
        dlon = float(src_lons[1] - src_lons[0])
        if dlat == 0.0 or dlon == 0.0:
            return np.full_like(tgt_lats_2d, fill_value, dtype=float)

        fi = (tgt_lats_2d - lat0) / dlat
        fj = (tgt_lons_2d - lon0) / dlon
        ii = np.rint(fi).astype(int)
        jj = np.rint(fj).astype(int)

        nlat, nlon = Z.shape
        valid = (ii >= 0) & (ii < nlat) & (jj >= 0) & (jj < nlon)
        out = np.full_like(tgt_lats_2d, fill_value, dtype=float)
        if not np.any(valid):
            return out

        out[valid] = Z[ii[valid], jj[valid]]
        return out

    def _uq_interp_to_unified(
        self,
        src_lats_1d,
        src_lons_1d,
        src_field_2d,
        tgt_lat2d,
        tgt_lon2d,
        method="nearest",
        fill_value=float("nan"),
        **kwargs,
    ):
        m = str(method).strip().lower()
        if m in {"nearest", "nn"}:
            return self._uq_nearest_interp_regular_grid(
                src_lats_1d, src_lons_1d, src_field_2d, tgt_lat2d, tgt_lon2d, fill_value=fill_value
            )
        if m in {"bilinear", "linear"}:
            return self._uq_bilinear_interp_regular_grid(
                src_lats_1d, src_lons_1d, src_field_2d, tgt_lat2d, tgt_lon2d, fill_value=fill_value
            )
        raise ValueError(f"Unsupported interp method '{method}'. Use 'nearest' or 'bilinear'.")

    def _uq_sm_xml_find_first(self, root, tag_suffix: str):
        for e in root.iter():
            if str(e.tag).endswith(tag_suffix):
                return e
        return None

    def _uq_sm_xml_find_all(self, root, tag_suffix: str):
        out = []
        for e in root.iter():
            if str(e.tag).endswith(tag_suffix):
                out.append(e)
        return out

    def _uq_sm_xml_get_attrib_float(self, elem, key: str, default=None):
        if elem is None or key not in getattr(elem, "attrib", {}):
            return default
        return self._uq_safe_float(elem.attrib.get(key))

    def _uq_axes_from_spec(self, spec):
        nlat = int(spec["nlat"])
        nlon = int(spec["nlon"])
        lat_min = float(spec["lat_min"])
        lon_min = float(spec["lon_min"])
        dy = float(spec["dy"])
        dx = float(spec["dx"])
        lats = self._uq_regular_axis(lat_min, dy, nlat)
        lons = self._uq_regular_axis(lon_min, dx, nlon)
        return lats, lons

    def _uq_sigma_field_to_imt(self, sigma_field: str) -> str:
        s = str(sigma_field).strip().upper()
        if s.startswith("STD") and len(s) > 3:
            return s[3:]
        return s

    def _uq_imt_to_sigma_field(self, imt: str) -> str:
        imt_u = str(imt).strip().upper()
        if imt_u.startswith("STD"):
            return imt_u
        return "STD" + imt_u

    def _uq_parse_grid_xml(self, xml_path):
        import xml.etree.ElementTree as ET

        xml_path = Path(xml_path)
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        spec_elem = self._uq_sm_xml_find_first(root, "grid_specification")
        if spec_elem is None:
            raise ValueError(f"Missing grid_specification in {xml_path}")

        nlat = int(self._uq_sm_xml_get_attrib_float(spec_elem, "nlat"))
        nlon = int(self._uq_sm_xml_get_attrib_float(spec_elem, "nlon"))
        lon_min = float(self._uq_sm_xml_get_attrib_float(spec_elem, "lon_min"))
        lat_min = float(self._uq_sm_xml_get_attrib_float(spec_elem, "lat_min"))

        dx = self._uq_sm_xml_get_attrib_float(
            spec_elem, "nominal_lon_spacing", self._uq_sm_xml_get_attrib_float(spec_elem, "lon_spacing")
        )
        dy = self._uq_sm_xml_get_attrib_float(
            spec_elem, "nominal_lat_spacing", self._uq_sm_xml_get_attrib_float(spec_elem, "lat_spacing")
        )
        dx = float(dx)
        dy = float(dy)

        spec = {"nlat": nlat, "nlon": nlon, "lon_min": lon_min, "lat_min": lat_min, "dx": dx, "dy": dy}

        fields = self._uq_sm_xml_find_all(root, "grid_field")
        idx_to_name = {}
        name_to_units = {}
        for f in fields:
            idx = f.attrib.get("index", None)
            name = f.attrib.get("name", None)
            units = f.attrib.get("units", None)
            if idx is None or name is None:
                continue
            try:
                idxi = int(idx)
                nm = str(name)
                idx_to_name[idxi] = nm
                if units is not None:
                    name_to_units[nm] = str(units)
            except Exception:
                continue

        data_elem = self._uq_sm_xml_find_first(root, "grid_data")
        if data_elem is None or data_elem.text is None:
            raise ValueError(f"Missing grid_data in {xml_path}")

        tokens = data_elem.text.strip().split()
        nfields = len(idx_to_name)
        expected = nlat * nlon * nfields
        if len(tokens) < expected:
            raise ValueError(f"grid_data too short: got {len(tokens)}, expected {expected} in {xml_path}")

        arr = np.array([float(x) for x in tokens[:expected]], dtype=float).reshape((nlat * nlon, nfields))

        mean_fields = {}
        mean_units = {}
        vs30 = None

        grids_by_name = {}
        lat_grid = None
        lon_grid = None

        for idx, name in idx_to_name.items():
            col = idx - 1
            if col < 0 or col >= nfields:
                continue
            g2 = arr[:, col].reshape((nlat, nlon))
            grids_by_name[str(name)] = g2

            nU = str(name).upper().strip()
            if nU == "LAT":
                lat_grid = g2
            elif nU == "LON":
                lon_grid = g2

        transpose = False
        flipud = False
        fliplr = False

        if lat_grid is not None and lon_grid is not None:
            lat_axis_increasing_down = np.nanmean(lat_grid[-1, :] - lat_grid[0, :]) > 0
            lon_axis_increasing_right = np.nanmean(lon_grid[:, -1] - lon_grid[:, 0]) > 0

            if not lat_axis_increasing_down:
                flipud = True
            if not lon_axis_increasing_right:
                fliplr = True

        if transpose:
            for kname in list(grids_by_name.keys()):
                grids_by_name[kname] = grids_by_name[kname].T
            if lat_grid is not None:
                lat_grid = lat_grid.T
            if lon_grid is not None:
                lon_grid = lon_grid.T

        if flipud:
            for kname in list(grids_by_name.keys()):
                grids_by_name[kname] = np.flipud(grids_by_name[kname])
            if lat_grid is not None:
                lat_grid = np.flipud(lat_grid)
            if lon_grid is not None:
                lon_grid = np.flipud(lon_grid)

        if fliplr:
            for kname in list(grids_by_name.keys()):
                grids_by_name[kname] = np.fliplr(grids_by_name[kname])
            if lat_grid is not None:
                lat_grid = np.fliplr(lat_grid)
            if lon_grid is not None:
                lon_grid = np.fliplr(lon_grid)

        try:
            self._uq_last_grid_orientation = {"transpose": bool(transpose), "flipud": bool(flipud), "fliplr": bool(fliplr)}
        except Exception:
            self._uq_last_grid_orientation = {"transpose": False, "flipud": False, "fliplr": False}

        for idx, name in idx_to_name.items():
            n = str(name)
            if n not in grids_by_name:
                continue
            grid2d = grids_by_name[n]

            if str(name).upper() == "VS30":
                vs30 = grid2d
            else:
                mean_fields[str(name)] = grid2d
                if str(name) in name_to_units:
                    mean_units[str(name)] = name_to_units[str(name)]

        return spec, mean_fields, vs30, mean_units

    def _uq_parse_uncertainty_xml(self, xml_path):
        import xml.etree.ElementTree as ET

        xml_path = Path(xml_path)
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        spec_elem = self._uq_sm_xml_find_first(root, "grid_specification")
        if spec_elem is None:
            raise ValueError(f"Missing grid_specification in {xml_path}")

        nlat = int(self._uq_sm_xml_get_attrib_float(spec_elem, "nlat"))
        nlon = int(self._uq_sm_xml_get_attrib_float(spec_elem, "nlon"))
        lon_min = float(self._uq_sm_xml_get_attrib_float(spec_elem, "lon_min"))
        lat_min = float(self._uq_sm_xml_get_attrib_float(spec_elem, "lat_min"))

        dx = self._uq_sm_xml_get_attrib_float(
            spec_elem, "nominal_lon_spacing", self._uq_sm_xml_get_attrib_float(spec_elem, "lon_spacing")
        )
        dy = self._uq_sm_xml_get_attrib_float(
            spec_elem, "nominal_lat_spacing", self._uq_sm_xml_get_attrib_float(spec_elem, "lat_spacing")
        )
        dx = float(dx)
        dy = float(dy)

        spec = {"nlat": nlat, "nlon": nlon, "lon_min": lon_min, "lat_min": lat_min, "dx": dx, "dy": dy}

        fields = self._uq_sm_xml_find_all(root, "grid_field")
        idx_to_name = {}
        name_to_units = {}
        for f in fields:
            idx = f.attrib.get("index", None)
            name = f.attrib.get("name", None)
            units = f.attrib.get("units", None)
            if idx is None or name is None:
                continue
            try:
                idxi = int(idx)
                nm = str(name)
                idx_to_name[idxi] = nm
                if units is not None:
                    name_to_units[nm] = str(units)
            except Exception:
                continue

        data_elem = self._uq_sm_xml_find_first(root, "grid_data")
        if data_elem is None or data_elem.text is None:
            raise ValueError(f"Missing grid_data in {xml_path}")

        tokens = data_elem.text.strip().split()
        nfields = len(idx_to_name)
        expected = nlat * nlon * nfields
        if len(tokens) < expected:
            raise ValueError(f"grid_data too short: got {len(tokens)}, expected {expected} in {xml_path}")

        arr = np.array([float(x) for x in tokens[:expected]], dtype=float).reshape((nlat * nlon, nfields))

        sigma_fields = {}
        sigma_units = {}

        orient = getattr(self, "_uq_last_grid_orientation", None)
        transpose = bool(orient.get("transpose", False)) if isinstance(orient, dict) else False
        flipud = bool(orient.get("flipud", False)) if isinstance(orient, dict) else False
        fliplr = bool(orient.get("fliplr", False)) if isinstance(orient, dict) else False

        for idx, name in idx_to_name.items():
            col = idx - 1
            if col < 0 or col >= nfields:
                continue

            grid2d = arr[:, col].reshape((nlat, nlon))
            if transpose:
                grid2d = grid2d.T
            if flipud:
                grid2d = np.flipud(grid2d)
            if fliplr:
                grid2d = np.fliplr(grid2d)

            if str(name).upper() == "VS30":
                continue
            sigma_fields[str(name)] = grid2d
            if str(name) in name_to_units:
                sigma_units[str(name)] = name_to_units[str(name)]

        return spec, sigma_fields, sigma_units

    def _uq_resolve_paths(self, version: int, stations_folder=None, rupture_folder=None):
        v = int(version)
        if self.event_id is None:
            raise AttributeError("SHAKEdataset must have self.event_id for UQ file resolution.")

        shakemap_folder = Path(self.shakemap_folder)

        stations_folder_eff = Path(stations_folder) if stations_folder is not None else Path(
            self.stations_folder or shakemap_folder
        )
        rupture_folder_eff = Path(rupture_folder) if rupture_folder is not None else Path(
            self.rupture_folder or shakemap_folder
        )

        grid_fname = f"{self.event_id}_us_{str(v).zfill(3)}_grid.xml"
        grid_path = shakemap_folder / str(self.event_id) / str(grid_fname)

        unc_path = None
        gname = grid_path.name
        candidates = []
        if "grid" in gname:
            candidates.append(gname.replace("grid", "uncertainty"))
            candidates.append(gname.replace("grid", "uncertainty_grid"))
        candidates += ["uncertainty.xml", "uncertainty_grid.xml"]
        chosen = None
        for c in candidates:
            p = grid_path.parent / c
            if p.exists():
                chosen = p
                break
        unc_path = chosen if chosen is not None else (grid_path.parent / "uncertainty.xml")

        st_fname = f"{self.event_id}_us_{str(v).zfill(3)}_stationlist.json"
        station_path = stations_folder_eff / str(self.event_id) / str(st_fname)

        rup_fname = f"{self.event_id}_us_{str(v).zfill(3)}_rupture.json"
        rupture_path = rupture_folder_eff / str(self.event_id) / str(rup_fname)

        trace = {
            "grid_xml": str(grid_path),
            "uncertainty_xml": str(unc_path) if unc_path is not None else "",
            "stationlist_json": str(station_path),
            "rupture_json": str(rupture_path),
        }
        return grid_path, unc_path, station_path, rupture_path, trace

    def uq_list_available_imts(self, version_list, stations_folder=None, rupture_folder=None):
        out = {}
        for v in list(version_list):
            grid_path, _, _, _, _ = self._uq_resolve_paths(int(v), stations_folder=stations_folder, rupture_folder=rupture_folder)
            if not grid_path.exists():
                out[int(v)] = []
                continue
            try:
                _, fields, _, _ = self._uq_parse_grid_xml(grid_path)
                out[int(v)] = sorted(list(fields.keys()))
            except Exception:
                out[int(v)] = []
        return out

    def _uq_expand_requested_imts(self, requested_tokens, global_imts):
        requested = []
        for token in requested_tokens:
            t = str(token).upper()
            if t == "PSA":
                requested.extend([k for k in global_imts if str(k).upper().startswith("PSA")])
            else:
                requested.append(str(token))
        seen = set()
        out = []
        for x in requested:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def _uq_default_sigma_for_imt(self, imt: str, shape2d=None):
        imt_u = str(imt).upper()
        if imt_u == "MMI":
            s = 0.7
        elif imt_u in {"PGA", "PGV"} or imt_u.startswith("PSA"):
            s = 0.6
        else:
            s = 0.8
        if shape2d is None:
            return float(s)
        if isinstance(shape2d, np.ndarray):
            return np.full(shape2d.shape, float(s), dtype=float)
        return float(s)

    def _uq_parse_stationlist_with_usgsparser(self, stationlist_json):
        dbg = {
            "pga_df_rows": 0,
            "pga_rows_after_lonlat_filter": 0,
            "pga_rows_after_value_filter": 0,
            "pgv_df_rows": 0,
            "pgv_rows_after_lonlat_filter": 0,
            "pgv_rows_after_value_filter": 0,
            "sa_df_rows": 0,
            "sa_rows_after_lonlat_filter": 0,
            "sa_rows_after_value_filter": 0,
            "mmi_df_rows": 0,
            "mmi_rows_after_lonlat_filter": 0,
            "mmi_rows_after_intensity_filter": 0,
            "pga_columns": [],
            "pgv_columns": [],
            "sa_columns": [],
            "mmi_columns": [],
            "note": "",
        }

        stationlist_json = Path(stationlist_json)
        if not stationlist_json.exists():
            dbg["note"] = "stationlist_json_missing"
            return [], [], dbg

        try:
            from modules.SHAKEparser import USGSParser
        except Exception:
            try:
                from SHAKEmaps_Toolkit.modules.SHAKEparser import USGSParser
            except Exception as e:
                dbg["note"] = f"USGSParser_import_failed: {e}"
                return [], [], dbg

        inst = []
        dyfi = []

        def _numcol(df, candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        def _latlon_cols(df):
            latc = "latitude" if "latitude" in df.columns else ("lat" if "lat" in df.columns else None)
            lonc = "longitude" if "longitude" in df.columns else ("lon" if "lon" in df.columns else None)
            return latc, lonc

        try:
            p = USGSParser(parser_type="instrumented_data", json_file=str(stationlist_json))
            df_pga = p.get_dataframe(value_type="pga")
        except Exception as e:
            df_pga = None
            dbg["note"] += f"|get_dataframe_pga_failed:{e}"

        if df_pga is not None:
            dbg["pga_df_rows"] = int(len(df_pga))
            dbg["pga_columns"] = list(df_pga.columns)
            if len(df_pga) > 0:
                df = df_pga.copy()
                latc, lonc = _latlon_cols(df)
                if latc is None or lonc is None:
                    dbg["note"] += "|pga_missing_lonlat_cols"
                else:
                    df[latc] = pd.to_numeric(df[latc], errors="coerce")
                    df[lonc] = pd.to_numeric(df[lonc], errors="coerce")
                    df = df.dropna(subset=[latc, lonc])
                    dbg["pga_rows_after_lonlat_filter"] = int(len(df))

                    pga_col = _numcol(df, ["pga", "PGA", "value", "amplitude", "amp"])
                    unit_col = _numcol(df, ["pga_unit", "PGA_unit", "unit"])

                    if pga_col is not None:
                        df[pga_col] = pd.to_numeric(df[pga_col], errors="coerce")
                        df_val = df.dropna(subset=[pga_col])
                        dbg["pga_rows_after_value_filter"] = int(len(df_val))
                        for _, r in df_val.iterrows():
                            d = {"lon": float(r[lonc]), "lat": float(r[latc]), "pga": float(r[pga_col]), "w": 1.0}
                            if unit_col is not None:
                                uu = r.get(unit_col, None)
                                if uu is not None and str(uu).strip() != "":
                                    d["pga_unit"] = str(uu).strip()
                            inst.append(d)

        try:
            p = USGSParser(parser_type="instrumented_data", json_file=str(stationlist_json))
            df_pgv = p.get_dataframe(value_type="pgv")
        except Exception:
            df_pgv = None

        if df_pgv is not None:
            dbg["pgv_df_rows"] = int(len(df_pgv))
            dbg["pgv_columns"] = list(df_pgv.columns)
            if len(df_pgv) > 0:
                df = df_pgv.copy()
                latc, lonc = _latlon_cols(df)
                if latc is not None and lonc is not None:
                    df[latc] = pd.to_numeric(df[latc], errors="coerce")
                    df[lonc] = pd.to_numeric(df[lonc], errors="coerce")
                    df = df.dropna(subset=[latc, lonc])
                    dbg["pgv_rows_after_lonlat_filter"] = int(len(df))

                    pgv_col = _numcol(df, ["pgv", "PGV", "value", "amplitude", "amp"])
                    unit_col = _numcol(df, ["pgv_unit", "PGV_unit", "unit"])

                    if pgv_col is not None:
                        df[pgv_col] = pd.to_numeric(df[pgv_col], errors="coerce")
                        df_val = df.dropna(subset=[pgv_col])
                        dbg["pgv_rows_after_value_filter"] = int(len(df_val))
                        for _, r in df_val.iterrows():
                            d = {"lon": float(r[lonc]), "lat": float(r[latc]), "pgv": float(r[pgv_col]), "w": 1.0}
                            if unit_col is not None:
                                uu = r.get(unit_col, None)
                                if uu is not None and str(uu).strip() != "":
                                    d["pgv_unit"] = str(uu).strip()
                            inst.append(d)

        df_sa = None
        for sa_type in ["sa", "psa"]:
            try:
                p = USGSParser(parser_type="instrumented_data", json_file=str(stationlist_json))
                df_sa = p.get_dataframe(value_type=sa_type)
                if df_sa is not None:
                    break
            except Exception:
                df_sa = None

        if df_sa is not None:
            dbg["sa_df_rows"] = int(len(df_sa))
            dbg["sa_columns"] = list(df_sa.columns)
            if len(df_sa) > 0:
                df = df_sa.copy()
                latc, lonc = _latlon_cols(df)
                if latc is not None and lonc is not None:
                    df[latc] = pd.to_numeric(df[latc], errors="coerce")
                    df[lonc] = pd.to_numeric(df[lonc], errors="coerce")
                    df = df.dropna(subset=[latc, lonc])
                    dbg["sa_rows_after_lonlat_filter"] = int(len(df))

                    sa_col = _numcol(df, ["sa", "SA", "psa", "PSA", "value", "amplitude", "amp"])
                    unit_col = _numcol(df, ["sa_unit", "SA_unit", "psa_unit", "unit"])

                    if sa_col is not None:
                        df[sa_col] = pd.to_numeric(df[sa_col], errors="coerce")
                        df_val = df.dropna(subset=[sa_col])
                        dbg["sa_rows_after_value_filter"] = int(len(df_val))
                        for _, r in df_val.iterrows():
                            d = {"lon": float(r[lonc]), "lat": float(r[latc]), "sa": float(r[sa_col]), "w": 1.0}
                            if unit_col is not None:
                                uu = r.get(unit_col, None)
                                if uu is not None and str(uu).strip() != "":
                                    d["sa_unit"] = str(uu).strip()
                            inst.append(d)

        df_mmi = None
        for parser_type in ["instrumented_data", "dyfi_data"]:
            try:
                p = USGSParser(parser_type=parser_type, json_file=str(stationlist_json))
                df_mmi = p.get_dataframe(value_type="mmi")
                if df_mmi is not None:
                    break
            except Exception:
                df_mmi = None

        if df_mmi is not None:
            dbg["mmi_df_rows"] = int(len(df_mmi))
            dbg["mmi_columns"] = list(df_mmi.columns)
            if len(df_mmi) > 0:
                df = df_mmi.copy()
                latc, lonc = _latlon_cols(df)
                if latc is None or lonc is None:
                    dbg["note"] += "|mmi_missing_lonlat_cols"
                else:
                    df[latc] = pd.to_numeric(df[latc], errors="coerce")
                    df[lonc] = pd.to_numeric(df[lonc], errors="coerce")
                    df = df.dropna(subset=[latc, lonc])
                    dbg["mmi_rows_after_lonlat_filter"] = int(len(df))

                    if "intensity" not in df.columns:
                        for alt in ["mmi", "cdi", "value", "MMI", "Intensity"]:
                            if alt in df.columns:
                                df["intensity"] = df[alt]
                                break

                    if "intensity" in df.columns:
                        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
                        df = df.dropna(subset=["intensity"])
                        dbg["mmi_rows_after_intensity_filter"] = int(len(df))

                        nresp_col = None
                        for c in ["nresp", "numResp", "numresp", "nResp", "responses"]:
                            if c in df.columns:
                                nresp_col = c
                                break
                        if nresp_col is not None:
                            df[nresp_col] = pd.to_numeric(df[nresp_col], errors="coerce")

                        for _, r in df.iterrows():
                            nresp = None
                            if nresp_col is not None and np.isfinite(r[nresp_col]):
                                nresp = float(r[nresp_col])
                            w = float(max(1.0, nresp)) if nresp is not None else 1.0
                            dyfi.append(
                                {
                                    "lon": float(r[lonc]),
                                    "lat": float(r[latc]),
                                    "intensity": float(r["intensity"]),
                                    "nresp": nresp,
                                    "w": w,
                                }
                            )

        inst2 = [o for o in inst if np.isfinite(o.get("lon", np.nan)) and np.isfinite(o.get("lat", np.nan))]
        dyfi2 = [
            o
            for o in dyfi
            if np.isfinite(o.get("lon", np.nan))
            and np.isfinite(o.get("lat", np.nan))
            and np.isfinite(o.get("intensity", np.nan))
        ]

        if dbg["pga_df_rows"] == 0 and dbg["mmi_df_rows"] == 0 and dbg["pgv_df_rows"] == 0 and dbg["sa_df_rows"] == 0:
            dbg["note"] += "|all_dataframes_empty_or_none"
        if (dbg["pga_df_rows"] > 0 and dbg["pga_rows_after_lonlat_filter"] == 0) or (
            dbg["mmi_df_rows"] > 0 and dbg["mmi_rows_after_lonlat_filter"] == 0
        ):
            dbg["note"] += "|filtered_all_lonlat"

        return inst2, dyfi2, dbg

    def _uq_build_unified_spec(self, specs, grid_unify="intersection", resolution="finest"):
        import math

        if not specs:
            raise ValueError("No grid specs provided.")

        lon_lo = max([float(s["lon_min"]) for s in specs])
        lat_lo = max([float(s["lat_min"]) for s in specs])
        lon_hi = min([float(s["lon_min"]) + float(s["dx"]) * (int(s["nlon"]) - 1) for s in specs])
        lat_hi = min([float(s["lat_min"]) + float(s["dy"]) * (int(s["nlat"]) - 1) for s in specs])

        if resolution == "finest":
            dxs = [abs(float(s["dx"])) for s in specs]
            dys = [abs(float(s["dy"])) for s in specs]
            u_dx = min(dxs)
            u_dy = min(dys)
        else:
            dxs = [abs(float(s["dx"])) for s in specs]
            dys = [abs(float(s["dy"])) for s in specs]
            u_dx = max(dxs)
            u_dy = max(dys)

        u_lon_lo = lon_lo
        u_lon_hi = lon_hi
        u_lat_lo = lat_lo
        u_lat_hi = lat_hi

        if u_dy > 0:
            lat_min = u_lat_lo
            nlat = int(math.floor((u_lat_hi - u_lat_lo) / u_dy)) + 1
            lat_max = lat_min + u_dy * (nlat - 1)
        else:
            lat_min = u_lat_hi
            nlat = int(math.floor((u_lat_hi - u_lat_lo) / abs(u_dy))) + 1
            lat_max = lat_min + u_dy * (nlat - 1)

        if u_dx > 0:
            lon_min = u_lon_lo
            nlon = int(math.floor((u_lon_hi - u_lon_lo) / u_dx)) + 1
            lon_max = lon_min + u_dx * (nlon - 1)
        else:
            lon_min = u_lon_hi
            nlon = int(math.floor((u_lon_hi - u_lon_lo) / abs(u_dx))) + 1
            lon_max = lon_min + u_dx * (nlon - 1)

        lon_lo_final, lon_hi_final = min(lon_min, lon_max), max(lon_min, lon_max)
        lat_lo_final, lat_hi_final = min(lat_min, lat_max), max(lat_min, lat_max)

        return {
            "lon_min": float(lon_min),
            "lat_min": float(lat_min),
            "dx": float(u_dx),
            "dy": float(u_dy),
            "nlon": int(nlon),
            "nlat": int(nlat),
            "lon_max": float(lon_hi_final),
            "lat_max": float(lat_hi_final),
            "grid_unify": str(grid_unify),
            "resolution": str(resolution),
        }

    def _uq_is_cdi_available_for_version(self, version, stations_folder=None, rupture_folder=None):
        from pathlib import Path

        cdi_path = getattr(self, "dyfi_cdi_file", None)
        if not cdi_path:
            return False
        if not Path(str(cdi_path)).exists():
            return False

        mode = str(getattr(self, "dyfi_source", "stationlist")).lower().strip()
        if mode == "cdi":
            return True
        if mode != "auto":
            return False
        return int(version) >= int(getattr(self, "cdi_attach_from_version", 4))

    def _uq_load_dyfi_cdi_df(self):
        from pathlib import Path

        if getattr(self, "_dyfi_cdi_df_cache", None) is not None:
            return self._dyfi_cdi_df_cache

        f = getattr(self, "dyfi_cdi_file", None)
        if not f:
            self._dyfi_cdi_df_cache = None
            return None

        p = Path(str(f))
        if not p.exists():
            self._dyfi_cdi_df_cache = None
            return None

        try:
            from modules.SHAKEparser import USGSParser
            parser = USGSParser(parser_type="dyfi_data", file_path=str(p))
            df = parser.get_dataframe()
        except Exception:
            self._dyfi_cdi_df_cache = None
            return None

        if df is None or len(df) == 0:
            self._dyfi_cdi_df_cache = None
            return None

        colmap = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ("latitude", "lat"):
                colmap[c] = "lat"
            elif cl in ("longitude", "lon"):
                colmap[c] = "lon"
            elif cl in ("cdi",):
                colmap[c] = "cdi"
            elif "no. of responses" in cl or cl in ("nresp", "responses"):
                colmap[c] = "nresp"
            elif "hypocentral distance" in cl or cl in ("distance", "dist", "dist_km"):
                colmap[c] = "dist_km"
            elif "standard deviation" in cl or cl in ("std", "stddev", "sigma"):
                colmap[c] = "std"
            elif "suspect" in cl:
                colmap[c] = "suspect"
        df = df.rename(columns=colmap)

        need = ["lat", "lon", "cdi"]
        for k in need:
            if k not in df.columns:
                self._dyfi_cdi_df_cache = None
                return None

        for k in ("lat", "lon", "cdi", "nresp", "dist_km", "std", "suspect"):
            if k in df.columns:
                df[k] = pd.to_numeric(df[k], errors="coerce")

        df = df.dropna(subset=["lat", "lon", "cdi"]).copy()
        self._dyfi_cdi_df_cache = df
        return df

    def _uq_obs_bounds(self, obs_list):
        if not obs_list:
            return None
        lats = [o.get("lat") for o in obs_list if o.get("lat") is not None]
        lons = [o.get("lon") for o in obs_list if o.get("lon") is not None]
        lats = [float(x) for x in lats if np.isfinite(x)]
        lons = [float(x) for x in lons if np.isfinite(x)]
        if not lats or not lons:
            return None
        return {
            "lat_min": float(np.min(lats)),
            "lat_max": float(np.max(lats)),
            "lon_min": float(np.min(lons)),
            "lon_max": float(np.max(lons)),
        }

    def _uq_build_obs_pool_for_version(self, version, raw, cdi_df=None):
        stations_raw = list(raw.get("stations", {}).get("instrumented", []) or [])
        dyfi_stationlist_raw = list(raw.get("stations", {}).get("dyfi", []) or [])

        cdi_raw = []
        if cdi_df is not None and int(version) >= int(getattr(self, "cdi_attach_from_version", 4)):
            max_dist = float(getattr(self, "dyfi_cdi_max_dist_km", 400.0))
            min_nresp = int(getattr(self, "dyfi_cdi_min_nresp", 1))
            for _, r in cdi_df.iterrows():
                lat = r.get("lat", np.nan)
                lon = r.get("lon", np.nan)
                cdi = r.get("cdi", np.nan)
                nresp = r.get("nresp", np.nan)
                dist = r.get("dist_km", np.nan)
                std = r.get("std", np.nan)
                suspect = r.get("suspect", np.nan)
                if not (np.isfinite(lat) and np.isfinite(lon) and np.isfinite(cdi)):
                    continue
                if np.isfinite(suspect) and int(suspect) != 0:
                    continue
                if np.isfinite(dist) and float(dist) > max_dist:
                    continue
                if np.isfinite(nresp) and int(nresp) < min_nresp:
                    continue
                cdi_raw.append(
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "cdi": float(cdi),
                        "nresp": float(nresp) if np.isfinite(nresp) else None,
                        "dist_km": float(dist) if np.isfinite(dist) else None,
                        "std": float(std) if np.isfinite(std) else None,
                        "suspect": float(suspect) if np.isfinite(suspect) else None,
                    }
                )

        obs_seismic = []
        for o in stations_raw:
            lat = o.get("lat")
            lon = o.get("lon")
            if not (np.isfinite(lat) and np.isfinite(lon)):
                continue
            for key, imt, unit_key in (
                ("pga", "PGA", "pga_unit"),
                ("pgv", "PGV", "pgv_unit"),
                ("sa", "PSA", "sa_unit"),
            ):
                if key not in o:
                    continue
                try:
                    val = float(o.get(key))
                except Exception:
                    continue
                if not np.isfinite(val):
                    continue
                obs_seismic.append(
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "value": float(val),
                        "imt": imt,
                        "domain": "seismic",
                        "source": "stationlist",
                        "type": "instrumented",
                        "unit": o.get(unit_key, None),
                        "w": float(o.get("w", 1.0)),
                    }
                )

        obs_intensity_stationlist = []
        for o in dyfi_stationlist_raw:
            lat = o.get("lat")
            lon = o.get("lon")
            if not (np.isfinite(lat) and np.isfinite(lon)):
                continue
            try:
                val = float(o.get("intensity"))
            except Exception:
                continue
            if not np.isfinite(val):
                continue
            obs_intensity_stationlist.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "value": float(val),
                    "imt": "MMI",
                    "domain": "intensity",
                    "source": "dyfi_stationlist",
                    "type": "dyfi",
                    "nresp": o.get("nresp", None),
                    "w": float(o.get("w", 1.0)),
                }
            )

        obs_intensity_cdi = []
        for o in cdi_raw:
            obs_intensity_cdi.append(
                {
                    "lat": float(o["lat"]),
                    "lon": float(o["lon"]),
                    "value": float(o["cdi"]),
                    "imt": "MMI",
                    "domain": "intensity",
                    "source": "dyfi_cdi",
                    "type": "dyfi_cdi",
                    "nresp": o.get("nresp", None),
                    "stddev": o.get("std", None),
                    "distance_km": o.get("dist_km", None),
                }
            )

        summary = {
            "counts": {
                "stations_raw": len(stations_raw),
                "dyfi_stationlist_raw": len(dyfi_stationlist_raw),
                "cdi_raw": len(cdi_raw),
                "obs_seismic": len(obs_seismic),
                "obs_intensity_stationlist": len(obs_intensity_stationlist),
                "obs_intensity_cdi": len(obs_intensity_cdi),
            },
            "bounds": {
                "obs_seismic": self._uq_obs_bounds(obs_seismic),
                "obs_intensity_stationlist": self._uq_obs_bounds(obs_intensity_stationlist),
                "obs_intensity_cdi": self._uq_obs_bounds(obs_intensity_cdi),
            },
            "samples": {
                "obs_seismic": obs_seismic[:3],
                "obs_intensity_stationlist": obs_intensity_stationlist[:3],
                "obs_intensity_cdi": obs_intensity_cdi[:3],
            },
        }
        logger.info("[UQ OBS POOL] v%s counts=%s bounds=%s", int(version), summary["counts"], summary["bounds"])
        return {
            "stations_raw": stations_raw,
            "dyfi_stationlist_raw": dyfi_stationlist_raw,
            "cdi_raw": cdi_raw,
            "obs_seismic": obs_seismic,
            "obs_intensity_stationlist": obs_intensity_stationlist,
            "obs_intensity_cdi": obs_intensity_cdi,
            "summary": summary,
        }
