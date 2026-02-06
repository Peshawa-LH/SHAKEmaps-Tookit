"""
SHAKEfetch (v2026-ready)
========================

Fetch USGS/ComCat event products (e.g., shakemap, dyfi, losspager, origin, moment-tensor)
with a focus on robustness, reproducibility, and time-evolving SHAKEmaps research workflows.

Key design goals (relative to v2025):
- Backward-compatible public API (existing method names preserved).
- Safe behavior when optional products are missing (no IndexError).
- Deterministic output layout + run manifest for reproducibility.
- Explicit product version selection policy ("first/last/all/preferred").
- Source-evolution timeline extraction + version comparison helpers.

Notes on terminology:
- This module fetches data from USGS ComCat. "ShakeMap" refers to the USGS implementation.
- In your thesis text, prefer "SHAKEmaps" for the general family.

Dependencies:
- libcomcat (for event discovery)
- getproduct CLI (recommended; used for downloads)
Optional:
- pandas/matplotlib for DYFI parsing/plotting and timeline tables.

© SHAKEmaps Toolkit (v2025 → v2026)
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import shutil
import logging
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
# Library-friendly default: no logging configuration side effects
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# -------------------------
# Helpers
# -------------------------

@contextmanager
def change_dir(path: str):
    """Temporarily change working directory (safe for notebooks/scripts)."""
    old = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _coerce_datetime_any(x: Any) -> Optional[str]:
    """
    Best-effort conversion of various libcomcat time types to ISO-8601 string.
    Returns None if unknown.
    """
    if x is None:
        return None
    if isinstance(x, str):
        return x
    # epoch milliseconds/seconds
    if isinstance(x, (int, float)):
        # heuristic: ms if large
        try:
            if x > 1e12:
                return datetime.fromtimestamp(x / 1000.0, tz=timezone.utc).isoformat()
            if x > 1e9:
                return datetime.fromtimestamp(x, tz=timezone.utc).isoformat()
        except Exception:
            return None
    # datetime-like
    if isinstance(x, datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=timezone.utc).isoformat()
        return x.astimezone(timezone.utc).isoformat()
    return None


def _ensure_json_serializable(obj: Any) -> Any:
    """Make common libcomcat objects JSON-serializable (best-effort)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_ensure_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return _coerce_datetime_any(obj)
    # try dataclass
    try:
        return asdict(obj)  # type: ignore[arg-type]
    except Exception:
        pass
    # fallback string representation
    return str(obj)


@dataclass
class DownloadResult:
    product_type: str
    version: str
    output_dir: str
    requested_files: List[str]
    downloaded_files: List[str]
    skipped_files: List[str]
    missing_files: List[str]
    errors: List[str]
    stdout: List[str]
    stderr: List[str]


# -------------------------
# Main class
# -------------------------

class SHAKEfetch:
    """
    Backward-compatible SHAKEfetch with v2026 upgrades.

    Parameters
    ----------
    event_id:
        ComCat event id (e.g., "us7000m9g4").
    export_dir:
        Root export directory (default "export").
    base_subdir:
        Subfolder under export_dir (default "SHAKEfetch").
    strict:
        If True, raise on missing products/dependencies; if False, warn and continue.
    overwrite:
        If True, overwrite existing files; if False, skip existing files.
    timeout:
        Default subprocess timeout for getproduct calls (seconds).
    """

    def __init__(
        self,
        event_id: str = "us7000m9g4",
        export_dir: str = "export",
        base_subdir: str = "SHAKEfetch",
        strict: bool = False,
        overwrite: bool = False,
        timeout: int = 120,
    ):
        self.event_id = event_id
        self.export_dir = export_dir
        self.base_subdir = base_subdir
        self.strict = strict
        self.overwrite_default = overwrite
        self.timeout_default = timeout

        self.original_dir = os.getcwd()
        self.base_dir = os.path.join(self.original_dir, self.export_dir, self.base_subdir)

        self.earthquake = None  # libcomcat event object
        self._getproduct_path = shutil.which("getproduct")

        # init: fetch event (robust)
        self.check_shakefetch_inputs()

    # -------------------------
    # Core checks & discovery
    # -------------------------

    def check_shakefetch_inputs(self) -> None:
        """Fetch event, validate optional toolchain. Keeps backward-compatible name."""
        # libcomcat import lazily for better module usability in environments
        try:
            from libcomcat.search import get_event_by_id  # type: ignore
        except Exception as e:
            raise ImportError(
                "libcomcat is required for SHAKEfetch. Install libcomcat in this environment."
            ) from e

        try:
            self.earthquake = get_event_by_id(self.event_id)
        except Exception as e:
            raise ValueError(f"Could not resolve event_id '{self.event_id}' via ComCat.") from e

        os.makedirs(self.base_dir, exist_ok=True)

        if self._getproduct_path is None:
            msg = (
                "getproduct CLI not found on PATH. Downloads will fail unless you install "
                "libcomcat utilities / getproduct."
            )
            if self.strict:
                raise EnvironmentError(msg)
            logger.warning(msg)

        # Backward-compatible: do not raise if shakemap missing (many events won't have it)
        # but provide a warning to preserve old intent.
        if not self._has_product("shakemap"):
            logger.warning("No 'shakemap' product found for event %s.", self.event_id)

    def _get_products_mapping(self) -> Dict[str, List[Any]]:
        """Return products as a dict {product_type: [product_instances,...]} across libcomcat variants."""
        if self.earthquake is None:
            return {}
        products = _safe_getattr(self.earthquake, "products", None)

        # Common case: dict mapping to list of Product
        if isinstance(products, dict):
            return {k: (v if isinstance(v, list) else [v]) for k, v in products.items()}

        # Some libcomcat builds expose a flat list of Product-like objects
        if isinstance(products, list):
            out: Dict[str, List[Any]] = {}
            for item in products:
                ptype = None
                if isinstance(item, dict):
                    ptype = item.get("type") or item.get("product_type") or item.get("name")
                else:
                    ptype = getattr(item, "type", None) or getattr(item, "product_type", None) or getattr(item, "name", None)
                if not ptype:
                    continue
                out.setdefault(str(ptype), []).append(item)
            return out

        # Fallback: try known methods (best-effort)
        out: Dict[str, List[Any]] = {}
        for meth in ("getProductTypes", "get_product_types", "getProducttypes"):
            fn = getattr(self.earthquake, meth, None)
            if callable(fn):
                try:
                    types = fn()
                    for t in types or []:
                        out[str(t)] = []
                    return out
                except Exception:
                    break
        return {}

    def _has_product(self, product_type: str) -> bool:
        mapping = self._get_products_mapping()
        return product_type in mapping and bool(mapping[product_type])

    def check_event_files(self) -> Dict[str, Any]:
        """
        Logs basic info and lists available content names for the shakemap product.
        Returns a structured summary (new; non-breaking).
        """
        if self.earthquake is None:
            raise RuntimeError("Event not loaded.")

        info = {
            "event_id": self.event_id,
            "time": _coerce_datetime_any(_safe_getattr(self.earthquake, "time", None)),
            "magnitude": _safe_getattr(self.earthquake, "magnitude", None),
            "latitude": _safe_getattr(self.earthquake, "latitude", None),
            "longitude": _safe_getattr(self.earthquake, "longitude", None),
            "depth": _safe_getattr(self.earthquake, "depth", None),
            "products": {},
        }

        products = self._get_products_mapping()
        for ptype, plist in products.items():
            info["products"][ptype] = len(plist) if plist is not None else 0

        logger.info("Event: %s | M=%s | time=%s", self.event_id, info["magnitude"], info["time"])
        logger.info("Products available: %s", ", ".join(sorted(info["products"].keys())))

        # shakemap content listing (safe)
        contents = []
        if self._has_product("shakemap"):
            try:
                p0 = products["shakemap"][0]
                contents = sorted(list(getattr(p0, "contents", {}).keys()))
                for c in contents:
                    logger.info("shakemap content: %s", c)
            except Exception as e:
                logger.warning("Could not list shakemap contents: %s", e)

        info["shakemap_contents_first"] = contents
        return info

    # -------------------------
    # Download engine
    # -------------------------

    def _run_download_commands(
        self,
        product_type: str,
        file_list: Sequence[str],
        output_subdir: str,
        version: str = "all",
        overwrite: Optional[bool] = None,
        timeout: Optional[int] = None,
        pause_s: float = 0.0,
    ) -> DownloadResult:
        """
        Backward-compatible helper that downloads product contents using getproduct.

        - Validates version parameter.
        - Creates output directory.
        - Skips existing files if overwrite is False.
        - Captures stdout/stderr for debugging.
        - Never uses shell=True.

        Returns
        -------
        DownloadResult (new, non-breaking).
        """
        allowed = {"last", "all", "first", "preferred"}
        if version not in allowed:
            raise ValueError(f"Invalid version '{version}', choose from {sorted(allowed)}")

        if overwrite is None:
            overwrite = self.overwrite_default
        if timeout is None:
            timeout = self.timeout_default

        output_dir = os.path.join(self.base_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        res = DownloadResult(
            product_type=product_type,
            version=version,
            output_dir=output_dir,
            requested_files=list(file_list),
            downloaded_files=[],
            skipped_files=[],
            missing_files=[],
            errors=[],
            stdout=[],
            stderr=[],
        )

        if self._getproduct_path is None:
            msg = "getproduct CLI not available. Cannot download product contents."
            if self.strict:
                raise EnvironmentError(msg)
            res.errors.append(msg)
            logger.warning(msg)
            return res

        with change_dir(output_dir):
            for content in file_list:
                # Skip existing file if not overwriting (best-effort: check basename)
                target_name = os.path.basename(content)
                if (not overwrite) and os.path.exists(target_name):
                    res.skipped_files.append(content)
                    continue

                cmd = [
                    self._getproduct_path,
                    product_type,
                    content,
                    "-i",
                    self.event_id,
                    f"--get-version={version}",
                ]
                logger.info("[%s] %s", product_type, " ".join(cmd))

                try:
                    cp = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    if cp.stdout:
                        res.stdout.append(cp.stdout.strip())
                    if cp.stderr:
                        res.stderr.append(cp.stderr.strip())
                    # after download, expect basename file to exist
                    if os.path.exists(target_name):
                        res.downloaded_files.append(content)
                    else:
                        # could be written to subpaths; treat as missing but non-fatal
                        res.missing_files.append(content)
                except subprocess.TimeoutExpired as e:
                    msg = f"Timeout downloading {product_type}:{content}"
                    res.errors.append(msg)
                    res.stderr.append(str(e))
                    logger.error(msg)
                    if self.strict:
                        raise
                except subprocess.CalledProcessError as e:
                    msg = f"Failed downloading {product_type}:{content}"
                    res.errors.append(msg)
                    if e.stdout:
                        res.stdout.append(e.stdout.strip())
                    if e.stderr:
                        res.stderr.append(e.stderr.strip())
                    logger.error("%s | %s", msg, (e.stderr or "").strip())
                    # do not raise by default; keep running
                    if self.strict:
                        raise

                if pause_s and pause_s > 0:
                    time.sleep(pause_s)

        return res

    # -------------------------
    # Deterministic output + manifest
    # -------------------------

    def _event_root(self) -> str:
        """Stable output root for this event."""
        return os.path.join(self.base_dir, self.event_id)

    def _write_manifest(
        self,
        tag: str,
        extra: Optional[Dict[str, Any]] = None,
        downloaded_paths: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Write a JSON manifest capturing event metadata and file hashes.
        Returns manifest path.
        """
        if output_dir is None:
            output_dir = self._event_root()
        os.makedirs(output_dir, exist_ok=True)

        eq = self.earthquake
        manifest: Dict[str, Any] = {
            "tag": tag,
            "created_utc": _utc_now_iso(),
            "event_id": self.event_id,
            "event_time": _coerce_datetime_any(_safe_getattr(eq, "time", None)),
            "magnitude": _safe_getattr(eq, "magnitude", None),
            "latitude": _safe_getattr(eq, "latitude", None),
            "longitude": _safe_getattr(eq, "longitude", None),
            "depth": _safe_getattr(eq, "depth", None),
            "products_available": sorted(list((_safe_getattr(eq, "products", {}) or {}).keys())),
            "tooling": {
                "getproduct_path": self._getproduct_path,
            },
            "files": [],
        }

        if downloaded_paths:
            for p in downloaded_paths:
                try:
                    if os.path.exists(p) and os.path.isfile(p):
                        manifest["files"].append(
                            {"path": p, "sha256": _sha256_file(p), "bytes": os.path.getsize(p)}
                        )
                except Exception as e:
                    manifest["files"].append({"path": p, "error": str(e)})

        if extra:
            manifest["extra"] = _ensure_json_serializable(extra)

        outpath = os.path.join(output_dir, f"manifest_{tag}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return outpath

    # -------------------------
    # Product version utilities (Phase B)
    # -------------------------

    def _list_product_versions(self, product_type: str) -> List[Dict[str, Any]]:
        """
        Return product version metadata for product_type (best-effort).
        Each entry includes update time, preferred flag, source/code, and content keys.
        """
        if self.earthquake is None:
            raise RuntimeError("Event not loaded.")
        products = self._get_products_mapping()
        plist = products.get(product_type, []) or []
        out: List[Dict[str, Any]] = []
        for i, p in enumerate(plist):
            try:
                contents = getattr(p, "contents", {}) or {}
                out.append(
                    {
                        "index": i,
                        "product_type": product_type,
                        "update_time": _coerce_datetime_any(_safe_getattr(p, "update_time", None))
                        or _coerce_datetime_any(_safe_getattr(p, "updateTime", None))
                        or _coerce_datetime_any(_safe_getattr(p, "time", None)),
                        "preferred": bool(_safe_getattr(p, "preferred", False)),
                        "source": _safe_getattr(p, "source", None),
                        "code": _safe_getattr(p, "code", None),
                        "status": _safe_getattr(p, "status", None),
                        "contents": sorted(list(contents.keys())),
                        "properties": _ensure_json_serializable(_safe_getattr(p, "properties", {})),
                    }
                )
            except Exception as e:
                out.append({"index": i, "product_type": product_type, "error": str(e)})
        return out

    def get_source_evolution(
        self,
        product_types: Sequence[str] = ("origin", "moment-tensor", "focal-mechanism", "finite-fault"),
        as_dataframe: bool = True,
    ):
        """
        Phase B: Extract a "source evolution timeline" across product versions.

        Returns a pandas DataFrame if pandas is available and as_dataframe=True, else a list of dicts.
        """
        records: List[Dict[str, Any]] = []
        for ptype in product_types:
            if not self._has_product(ptype):
                continue
            records.extend(self._list_product_versions(ptype))

        if as_dataframe:
            try:
                import pandas as pd  # type: ignore
                return pd.DataFrame.from_records(records)
            except Exception:
                return records
        return records

    def compare_product_versions(
        self,
        product_type: str,
        a: Union[int, str] = "preferred",
        b: Union[int, str] = "last",
    ) -> Dict[str, Any]:
        """
        Phase B: Compare two versions (a vs b) of a product by their content keys and properties.

        a/b can be:
        - integer indices
        - "preferred" | "first" | "last"
        """
        versions = self._list_product_versions(product_type)
        if not versions:
            msg = f"No versions found for product '{product_type}'"
            if self.strict:
                raise ValueError(msg)
            return {"product_type": product_type, "error": msg}

        def resolve(sel: Union[int, str]) -> Dict[str, Any]:
            if isinstance(sel, int):
                return versions[sel]
            s = sel.lower()
            if s == "first":
                return versions[0]
            if s == "last":
                return versions[-1]
            if s == "preferred":
                for v in versions:
                    if v.get("preferred"):
                        return v
                return versions[0]
            raise ValueError(f"Unknown selector '{sel}'")

        va = resolve(a)
        vb = resolve(b)

        ca = set(va.get("contents", []) or [])
        cb = set(vb.get("contents", []) or [])

        diff = {
            "product_type": product_type,
            "a": va,
            "b": vb,
            "contents_added_in_b": sorted(list(cb - ca)),
            "contents_removed_in_b": sorted(list(ca - cb)),
        }

        # property diff (shallow)
        pa = va.get("properties", {}) if isinstance(va.get("properties", {}), dict) else {}
        pb = vb.get("properties", {}) if isinstance(vb.get("properties", {}), dict) else {}
        keys = set(pa.keys()) | set(pb.keys())
        changed = {}
        for k in sorted(keys):
            if pa.get(k) != pb.get(k):
                changed[k] = {"a": pa.get(k), "b": pb.get(k)}
        diff["properties_changed"] = changed
        return diff

    # -------------------------
    # Backward-compatible product download methods
    # (kept names from v2025)
    # -------------------------

    def get_shakemaps(self, version: str = "all"):
        """Download shakemap core files (backward-compatible wrapper)."""
        return self.get_shakemap_files(version=version)

    def get_pagers(self, version: str = "all"):
        """Download PAGER (losspager) files (backward-compatible wrapper)."""
        return self.get_losspager_files(version=version)

    def get_shakemap_files(self, version: str = "all"):
        """
        Download common ShakeMap files into a deterministic per-event folder.
        """
        files = [
            "grid.xml",
            "grid.xml.zip",
            "stationlist.json",
            "rupture.json",
            "info.json",
            "metadata.json",
        ]
        out = os.path.join(self.event_id, "shakemap")
        res = self._run_download_commands("shakemap", files, out, version=version)
        self._write_manifest(tag=f"shakemap_{version}", extra=asdict(res), output_dir=os.path.join(self.base_dir, out))
        return res

    def get_losspager_files(self, version: str = "all"):
        """
        Download losspager files (PAGER) into a deterministic per-event folder.
        """
        files = [
            "pager.xml",
            "pager.json",
            "impact.json",
            "onepager.pdf",
            "onepager.png",
        ]
        out = os.path.join(self.event_id, "losspager")
        res = self._run_download_commands("losspager", files, out, version=version)
        self._write_manifest(tag=f"losspager_{version}", extra=asdict(res), output_dir=os.path.join(self.base_dir, out))
        return res

    def get_dyfi_files(self, version: str = "all"):
        """
        Download DYFI files into a deterministic per-event folder.
        """
        files = [
            "dyfi_geo_1km.geojson",
            "dyfi_geo_10km.geojson",
            "dyfi_geo_100km.geojson",
            "dyfi_plot_atten.json",
            "dyfi_plot_numresp.json",
            "dyfi_plot_resp.json",
            "dyfi_plot_ciim.json",
            "dyfi_plot_pager.json",
            "dyfi.zip",
            "dyfi.json",
        ]
        out = os.path.join(self.event_id, "dyfi")
        res = self._run_download_commands("dyfi", files, out, version=version)
        self._write_manifest(tag=f"dyfi_{version}", extra=asdict(res), output_dir=os.path.join(self.base_dir, out))
        return res

    # The following methods preserve v2025 names but route to generalized downloader.
    # They focus on shakemap product content families.

    def get_stations(self, version: str = "all"):
        return self._run_download_commands(
            "shakemap",
            ["stationlist.json"],
            os.path.join(self.event_id, "shakemap", "usgs-instruments_data-versions"),
            version=version,
        )

    def get_ruptures(self, version: str = "all"):
        return self._run_download_commands(
            "shakemap",
            ["rupture.json"],
            os.path.join(self.event_id, "shakemap", "usgs-rupture-versions"),
            version=version,
        )

    def get_attenuation_curves(self, version: str = "all"):
        return self._run_download_commands(
            "shakemap",
            ["attenuation_curves.json"],
            os.path.join(self.event_id, "shakemap", "usgs-attenuation_curves-versions"),
            version=version,
        )

    def get_contours(self, version: str = "all"):
        files = [
            "cont_mi.json",
            "cont_pga.json",
            "cont_pgv.json",
            "cont_psa03.json",
            "cont_psa10.json",
            "cont_psa30.json",
        ]
        return self._run_download_commands(
            "shakemap",
            files,
            os.path.join(self.event_id, "shakemap", "usgs-contours-versions"),
            version=version,
        )

    def get_coverages(self, version: str = "all"):
        files = [
            "coverage_mi.json",
            "coverage_pga.json",
            "coverage_pgv.json",
            "coverage_psa03.json",
            "coverage_psa10.json",
            "coverage_psa30.json",
        ]
        return self._run_download_commands(
            "shakemap",
            files,
            os.path.join(self.event_id, "shakemap", "usgs-coverage-versions"),
            version=version,
        )

    def get_event_info(self, version: str = "all"):
        return self._run_download_commands(
            "shakemap",
            ["info.json", "metadata.json"],
            os.path.join(self.event_id, "shakemap", "usgs-event_info-versions"),
            version=version,
        )

    def get_shapefiles(self, version: str = "all"):
        files = ["shakemap.kmz", "shape.zip"]
        return self._run_download_commands(
            "shakemap",
            files,
            os.path.join(self.event_id, "shakemap", "usgs-shapefiles-versions"),
            version=version,
        )

    def get_figures_all(self, version: str = "all"):
        # This list is intentionally broad; missing files will be logged, not fatal.
        files = [
            "download/intensity.jpg",
            "download/intensity.pdf",
            "download/intensity.png",
            "download/intensity.ps",
            "download/intensity.ps.zip",
            "download/intensity.tif",
            "download/intensity.tif.zip",
            "download/pga.jpg",
            "download/pga.pdf",
            "download/pga.png",
            "download/pga.tif",
            "download/pgv.jpg",
            "download/pgv.pdf",
            "download/pgv.png",
            "download/mi.jpg",
            "download/mi.pdf",
            "download/mi.png",
            "download/mi.tif",
        ]
        return self._run_download_commands(
            "shakemap",
            files,
            os.path.join(self.event_id, "shakemap", "usgs-figures-all-versions"),
            version=version,
        )

    # Source / other ComCat product families
    def get_origin(self, version: str = "all"):
        # common origin contents vary; attempt a small set
        files = ["origin.xml", "origin.json", "quakeml.xml"]
        return self._run_download_commands(
            "origin",
            files,
            os.path.join(self.event_id, "origin"),
            version=version,
        )

    def get_moment_tensor(self, version: str = "all"):
        files = ["moment-tensor.xml", "moment-tensor.json", "quakeml.xml"]
        return self._run_download_commands(
            "moment-tensor",
            files,
            os.path.join(self.event_id, "moment-tensor"),
            version=version,
        )

    def get_focal_mechanism(self, version: str = "all"):
        files = ["focal-mechanism.xml", "focal-mechanism.json", "quakeml.xml"]
        return self._run_download_commands(
            "focal-mechanism",
            files,
            os.path.join(self.event_id, "focal-mechanism"),
            version=version,
        )

    def get_finite_fault(self, version: str = "all"):
        files = ["finite-fault.json", "finite-fault.xml", "quakeml.xml"]
        return self._run_download_commands(
            "finite-fault",
            files,
            os.path.join(self.event_id, "finite-fault"),
            version=version,
        )

    def get_phase_data(self, version: str = "all"):
        files = ["phase-data.xml", "phase-data.json"]
        return self._run_download_commands(
            "phase-data",
            files,
            os.path.join(self.event_id, "phase-data"),
            version=version,
        )

    # -------------------------
    # DYFI parsing + plotting (backward-compatible)
    # -------------------------

    def get_dyfi(self, version: str = "all"):
        """
        Load DYFI JSON (if downloaded) into a pandas DataFrame.
        Will attempt to download dyfi.json first if missing.
        """
        # ensure dyfi.json exists (best-effort)
        dyfi_dir = os.path.join(self.base_dir, self.event_id, "dyfi")
        dyfi_json_path = os.path.join(dyfi_dir, "dyfi.json")
        if not os.path.exists(dyfi_json_path):
            self.get_dyfi_files(version=version)

        if not os.path.exists(dyfi_json_path):
            msg = f"DYFI file not found: {dyfi_json_path}"
            if self.strict:
                raise FileNotFoundError(msg)
            logger.warning(msg)
            return None

        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise ImportError("pandas is required for get_dyfi().") from e

        with open(dyfi_json_path, "r", encoding="utf-8") as f:
            js = json.load(f)

        # Best-effort schema: handle GeoJSON-like structures
        features = js.get("features", [])
        rows = []
        for feat in features:
            props = feat.get("properties", {}) or {}
            geom = feat.get("geometry", {}) or {}
            coords = geom.get("coordinates", None)
            row = dict(props)
            # stable location columns
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                row["longitude"] = coords[0]
                row["latitude"] = coords[1]
            rows.append(row)

        df = pd.DataFrame(rows)

        # Light coercions for stability across versions
        for col in ["cdi", "mmi", "intensity", "nresp", "dist", "lat", "lon"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def plot_dyfi(self, df=None, x: str = "dist", y: str = "cdi"):
        """
        Simple DYFI scatter plot (keeps v2025 name).
        """
        if df is None:
            df = self.get_dyfi()
        if df is None:
            return None

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            raise ImportError("matplotlib is required for plot_dyfi().") from e

        if x not in df.columns or y not in df.columns:
            logger.warning("DYFI dataframe missing columns for plot: %s vs %s", x, y)
            return None

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(df[x], df[y], s=10)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"DYFI {y} vs {x} | {self.event_id}")
        return fig

    # -------------------------
    # Optional: event search (large-scale studies)
    # -------------------------

    def search_events(self, **kwargs):
        """
        Search ComCat for events (optional convenience).
        Requires libcomcat.search.search (if available in your libcomcat version).

        Example kwargs: starttime, endtime, minmagnitude, maxmagnitude, latitude, longitude, maxradiuskm, boundingbox, etc.
        """
        try:
            from libcomcat.search import search  # type: ignore
        except Exception as e:
            raise ImportError(
                "This libcomcat installation does not expose search(). "
                "Update libcomcat or use the ComCat API directly."
            ) from e
        return search(**kwargs)

    # -------------------------
    # Documentation helper
    # -------------------------

    def print_doc(self):
        """Keep v2025 helper name; prints module/class docstring."""
        print(self.__doc__)
