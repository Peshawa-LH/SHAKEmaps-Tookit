"""
SHAKElab: multi-event CDI analysis workflows for SHAKEuq.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.SHAKEuq import SHAKEuq

logger = logging.getLogger(__name__)


class SHAKElab:
    """
    Multi-event CDI analysis runner built on SHAKEuq datasets.
    """

    def __init__(
        self,
        root_folder: Union[str, Path],
        export_base: Union[str, Path] = "./export/SHAKElab",
        default_imt: str = "MMI",
        cdi_filter_defaults: Optional[Dict[str, Any]] = None,
    ):
        self.root_folder = Path(root_folder)
        self.export_base = Path(export_base)
        self.default_imt = str(default_imt)
        self.cdi_filter_defaults = cdi_filter_defaults or {}
        self.export_base.mkdir(parents=True, exist_ok=True)

    def index_events(self) -> pd.DataFrame:
        """
        Scan root folder for events with required data folders.
        """
        root = self.root_folder
        shakemap_root = root / "usgs-shakemap-versions"
        rupture_root = root / "usgs-rupture-versions"
        stations_root = root / "usgs-instruments_data-versions"
        dyfi_root = root / "usgs-dyfi-versions"

        events = []
        if not shakemap_root.exists():
            return pd.DataFrame(events)

        for event_dir in shakemap_root.iterdir():
            if not event_dir.is_dir():
                continue
            event_id = event_dir.name
            rupture_dir = rupture_root / event_id
            stations_dir = stations_root / event_id
            dyfi_dir = dyfi_root / event_id
            if not (rupture_dir.exists() and stations_dir.exists() and dyfi_dir.exists()):
                continue

            dyfi_cdi_file = None
            for ext in ("*.txt", "*.xml"):
                matches = sorted([p for p in dyfi_dir.glob(ext) if "cdi" in p.name.lower()])
                if matches:
                    dyfi_cdi_file = str(matches[0])
                    break

            versions = self._discover_versions(event_dir)

            events.append(
                {
                    "event_id": event_id,
                    "shakemap_folder": str(shakemap_root),
                    "rupture_folder": str(rupture_root),
                    "stations_folder": str(stations_root),
                    "dyfi_cdi_file": dyfi_cdi_file,
                    "version_list": versions,
                }
            )

        return pd.DataFrame(events)

    def build_event(
        self,
        event_row_or_id: Union[pd.Series, str],
        version_list: Union[str, Iterable[int]] = "auto",
        include_cdi_from_version: int = 2,
        **kwargs,
    ) -> SHAKEuq:
        """
        Build SHAKEuq for an event row or event_id.
        """
        if isinstance(event_row_or_id, pd.Series):
            row = event_row_or_id
        else:
            df = self.index_events()
            matches = df[df["event_id"] == str(event_row_or_id)]
            if matches.empty:
                raise ValueError(f"Event not found: {event_row_or_id}")
            row = matches.iloc[0]

        if version_list == "auto":
            version_list = row.get("version_list")
        if not version_list:
            event_dir = Path(row["shakemap_folder"]) / str(row["event_id"])
            version_list = self._discover_versions(event_dir)

        uq = SHAKEuq(
            event_id=str(row["event_id"]),
            event_time=None,
            shakemap_folder=row["shakemap_folder"],
            stations_folder=row["stations_folder"],
            rupture_folder=row["rupture_folder"],
            dyfi_cdi_file=row.get("dyfi_cdi_file"),
            version_list=version_list,
            cdi_attach_from_version=include_cdi_from_version,
            **kwargs,
        )
        uq.uq_build_dataset(
            event_id=str(row["event_id"]),
            version_list=version_list,
            base_folder=str(self.export_base / "SHAKEuq"),
            stations_folder=row["stations_folder"],
            rupture_folder=row["rupture_folder"],
            imts=(self.default_imt,),
            export=True,
        )
        return uq

    def run_cdi_filter_sweep(
        self,
        events: Union[pd.DataFrame, Iterable[Union[str, pd.Series]]],
        filter_configs: List[Dict[str, Any]],
        *,
        imt: str = "MMI",
    ) -> pd.DataFrame:
        """
        Run CDI filter sweeps across events and return tidy results.
        """
        if isinstance(events, pd.DataFrame):
            event_rows = [row for _, row in events.iterrows()]
        else:
            event_rows = list(events)

        normalized_configs = self._normalize_filter_configs(filter_configs)
        results = []

        for event in event_rows:
            uq = self.build_event(event, version_list="auto", include_cdi_from_version=2)
            cdi_df = uq._uq_load_dyfi_cdi_df()
            if cdi_df is None or len(cdi_df) == 0:
                continue

            state = uq.uq_get_dataset_state()
            version = max(state.get("version_list", []) or [0])
            mean_grid, spec = self._get_unified_mean_grid(state, version, imt)
            if mean_grid is None:
                continue

            for config in normalized_configs:
                filtered = self._apply_cdi_filters(cdi_df, config)
                residuals = self._compute_residuals(filtered, mean_grid, spec)
                metrics = self._summarize_residuals(residuals)
                coverage = self._coverage_proxy(filtered, spec)

                results.append(
                    {
                        "event_id": uq.event_id,
                        "version": int(version),
                        "config_id": config["config_id"],
                        "strictness_index": config["strictness_index"],
                        "points_total": int(len(cdi_df)),
                        "points_kept": int(len(filtered)),
                        "retained_fraction": float(len(filtered) / len(cdi_df)) if len(cdi_df) else np.nan,
                        "coverage_fraction": coverage,
                        **metrics,
                    }
                )

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            out_dir = self.export_base / "cdi_sweep"
            out_dir.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(out_dir / "cdi_filter_sweep_results.csv", index=False)
            self._write_manifest(out_dir, normalized_configs)
        return results_df

    def plot_cdi_sweep_summary(self, results_df: pd.DataFrame, out_dir: Union[str, Path], **style):
        """
        Produce summary plots for CDI filter sweeps.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if results_df is None or results_df.empty:
            return []

        plt.style.use(style.get("style", "seaborn-v0_8-whitegrid"))
        files = []

        grouped = results_df.groupby("config_id")
        strictness = grouped["strictness_index"].mean().sort_values()
        retention = grouped["retained_fraction"].median().reindex(strictness.index)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(strictness.values, retention.values, marker="o")
        ax.set_xlabel("Strictness index")
        ax.set_ylabel("Retained fraction")
        ax.set_title("CDI Retention vs Strictness")
        files.extend(self._save_plot(fig, out_dir, "retention_vs_strictness"))

        fig, ax = plt.subplots(figsize=(7, 4))
        data = [
            results_df[results_df["config_id"] == cid]["median_abs_residual"].dropna().values
            for cid in strictness.index
        ]
        ax.boxplot(data, labels=strictness.index, showfliers=False)
        ax.set_ylabel("Median |residual| (MMI)")
        ax.set_title("Residual Improvement vs Baseline")
        ax.tick_params(axis="x", rotation=30)
        files.extend(self._save_plot(fig, out_dir, "residual_improvement_boxplot"))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(results_df["retained_fraction"], results_df["coverage_fraction"], c=results_df["strictness_index"])
        ax.set_xlabel("Retained fraction")
        ax.set_ylabel("Coverage fraction")
        ax.set_title("Retention vs Coverage")
        files.extend(self._save_plot(fig, out_dir, "retention_vs_coverage"))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(results_df["coverage_fraction"], results_df["median_abs_residual"], c=results_df["strictness_index"])
        ax.set_xlabel("Coverage fraction")
        ax.set_ylabel("Median |residual| (MMI)")
        ax.set_title("Coverage vs Residual Tradeoff")
        files.extend(self._save_plot(fig, out_dir, "coverage_vs_residual"))

        self._write_manifest(out_dir, None, extra_files=files)
        return files

    def _discover_versions(self, event_dir: Path) -> List[int]:
        versions = set()
        if not event_dir.exists():
            return []
        for path in event_dir.rglob("grid.xml"):
            for part in path.parts:
                part_lower = part.lower()
                if part_lower.startswith("v") and part_lower[1:].isdigit():
                    versions.add(int(part_lower[1:]))
        if not versions:
            for path in event_dir.glob("*.xml"):
                name = path.name
                if "grid.xml" in name:
                    tokens = [t for t in name.split("_") if t.isdigit()]
                    for t in tokens:
                        versions.add(int(t))
        return sorted(list(versions))

    def _normalize_filter_configs(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for idx, cfg in enumerate(configs):
            merged = dict(self.cdi_filter_defaults)
            merged.update(cfg)
            merged["config_id"] = merged.get("config_id", f"cfg_{idx:02d}")
            merged["strictness_index"] = merged.get("strictness_index", idx)
            normalized.append(merged)
        return normalized

    def _apply_cdi_filters(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        work = df.copy()
        if "max_dist_km" in config and "dist_km" in work.columns:
            work = work[work["dist_km"] <= float(config["max_dist_km"])]
        if "min_nresp" in config and "nresp" in work.columns:
            work = work[work["nresp"] >= float(config["min_nresp"])]
        if "max_std" in config and "std" in work.columns:
            work = work[work["std"] <= float(config["max_std"])]
        if "min_cdi" in config and "cdi" in work.columns:
            work = work[work["cdi"] >= float(config["min_cdi"])]
        if "max_cdi" in config and "cdi" in work.columns:
            work = work[work["cdi"] <= float(config["max_cdi"])]
        if "allow_suspect" in config and "suspect" in work.columns:
            if not bool(config["allow_suspect"]):
                work = work[work["suspect"].fillna(0) == 0]
        return work

    def _get_unified_mean_grid(self, state: Dict[str, Any], version: int, imt: str):
        versions = state.get("versions_unified", {})
        vinfo = versions.get(int(version))
        if not vinfo:
            return None, None
        mean_grid = vinfo.get("unified_mean", {}).get(str(imt))
        spec = state.get("unified_spec", {})
        return mean_grid, spec

    def _compute_residuals(self, df: pd.DataFrame, mean_grid: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
        if df.empty:
            return np.asarray([])
        lat_min = float(spec.get("lat_min"))
        lon_min = float(spec.get("lon_min"))
        dy = float(spec.get("dy"))
        dx = float(spec.get("dx"))
        nlat = int(spec.get("nlat"))
        nlon = int(spec.get("nlon"))

        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()
        i = np.rint((lats - lat_min) / dy).astype(int) if dy != 0 else np.zeros_like(lats, dtype=int)
        j = np.rint((lons - lon_min) / dx).astype(int) if dx != 0 else np.zeros_like(lons, dtype=int)
        i = np.clip(i, 0, nlat - 1)
        j = np.clip(j, 0, nlon - 1)
        modeled = mean_grid[i, j]
        obs = df["cdi"].to_numpy()
        return obs - modeled

    def _summarize_residuals(self, residuals: np.ndarray) -> Dict[str, Any]:
        if residuals.size == 0:
            return {
                "median_abs_residual": np.nan,
                "mean_abs_residual": np.nan,
                "rmse": np.nan,
            }
        abs_res = np.abs(residuals)
        rmse = float(np.sqrt(np.nanmean(residuals ** 2)))
        return {
            "median_abs_residual": float(np.nanmedian(abs_res)),
            "mean_abs_residual": float(np.nanmean(abs_res)),
            "rmse": rmse,
        }

    def _coverage_proxy(self, df: pd.DataFrame, spec: Dict[str, Any]) -> float:
        if df.empty:
            return np.nan
        lat_min = float(spec.get("lat_min"))
        lon_min = float(spec.get("lon_min"))
        dy = float(spec.get("dy"))
        dx = float(spec.get("dx"))
        nlat = int(spec.get("nlat"))
        nlon = int(spec.get("nlon"))

        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()
        i = np.rint((lats - lat_min) / dy).astype(int) if dy != 0 else np.zeros_like(lats, dtype=int)
        j = np.rint((lons - lon_min) / dx).astype(int) if dx != 0 else np.zeros_like(lons, dtype=int)
        i = np.clip(i, 0, nlat - 1)
        j = np.clip(j, 0, nlon - 1)
        unique_cells = len({(ii, jj) for ii, jj in zip(i, j)})
        total_cells = max(1, nlat * nlon)
        return float(unique_cells / total_cells)

    def _save_plot(self, fig, out_dir: Path, stem: str) -> List[str]:
        png = out_dir / f"{stem}.png"
        pdf = out_dir / f"{stem}.pdf"
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        fig.savefig(pdf)
        plt.close(fig)
        return [str(png), str(pdf)]

    def _write_manifest(self, out_dir: Path, configs: Optional[List[Dict[str, Any]]], extra_files: Optional[List[str]] = None):
        manifest = {
            "output_dir": str(out_dir),
            "configs": configs,
            "files": extra_files or [],
        }
        manifest_path = out_dir / "manifest.csv"
        rows = []
        if configs:
            for cfg in configs:
                rows.append({"type": "config", "config_id": cfg.get("config_id"), "payload": json.dumps(cfg)})
        for f in manifest["files"]:
            rows.append({"type": "file", "config_id": "", "payload": str(f)})
        if rows:
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
        else:
            pd.DataFrame([manifest]).to_csv(manifest_path, index=False)
