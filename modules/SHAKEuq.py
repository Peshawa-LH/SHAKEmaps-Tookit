"""
Uncertainty Quantification for Time-Dependent SHAKEmap Evolution
Class Name: SHAKEuq

Description:
    The SHAKEuq class provides a dedicated uncertainty-quantification (UQ) framework for analyzing
    time-evolving SHAKEmap products for a single seismic event. It is designed to work alongside
    SHAKEuq and focuses on uncertainty representation, uncertainty propagation, and uncertainty
    updating across multiple ShakeMap versions as observations (instrumented stations and macroseismic
    reports) become available.

    SHAKEuq supports construction of unified-grid UQ datasets across versions, extraction of RAW
    ShakeMap-published uncertainty layers, and generation of alternative uncertainty estimates using
    lightweight, Bayesian-inspired updating and comparison methods. The framework is intended for
    rapid-response analysis and research workflows where interpretability, reproducibility, and
    scientific defensibility are prioritized over complex spatial Bayesian filtering.

Core Functionality:
    • UQ dataset builder (per-version mean fields + uncertainty components on a unified grid)
    • RAW ShakeMap uncertainty extraction (e.g., sigma_total where available)
    • Bayesian-inspired local updating (mean and/or sigma; observation-weighted precision fusion)
    • Hierarchical / scale aggregation utilities (point → area → global; scale-only options)
    • Comparator methods for diagnostics (e.g., kriging- and Monte Carlo-style diagnostics when enabled)
    • Target-based evolution plots and audits (point/area/global uncertainty decay, stabilization checks)
    • Export/import of UQ states, audits, and intermediate products for reproducibility

Prerequisites:
    ShakeMap products must be available locally (e.g., fetched via SHAKEfetch) and follow standard USGS
    directory structures and naming conventions. Required dependencies include NumPy, Pandas,
    Matplotlib, SciPy, and XML parsing libraries. Optional diagnostics may require geospatial and ML
    libraries (Cartopy, GeoPandas, scikit-learn, etc.) depending on enabled methods.

Relationship to SHAKEmaps:
    “SHAKEmaps” refers to the general family of ShakeMap-type procedures.
    “ShakeMap” refers to the USGS implementation.
    “shakemap” refers to a specific calculated product.

Notes on Architecture:
    SHAKEuq is intended to separate UQ-specific logic from SHAKEuq to improve maintainability and
    enable iterative research development without expanding the SHAKEuq class. In the v26.6 cleanup,
    SHAKEuq can be used either as:
      (A) a standalone class initialized with event metadata and folder paths, or
      (B) a companion class attached to a SHAKEuq instance (composition pattern).

Date:
    February, 2026
Version:
    26.6 (draft split from SHAKEuq v26.5)

"""

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
from modules.SHAKEdataset import SHAKEdataset


from hashlib import md5 as _md5

try:
    from modules.SHAKEtime import *
except Exception:
    SHAKEtime = None

            

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class SHAKEuq:
    """
    SHAKEuq: Uncertainty Quantification Engine for Time-Evolving SHAKEmaps (v26.6)

    Overview
    ========
    The SHAKEuq class contains the dedicated Uncertainty Quantification (UQ) workflow for time-evolving
    SHAKEmap products. It builds reproducible UQ datasets across multiple ShakeMap versions on a unified
    spatial grid and provides tools for extracting and comparing uncertainty representations, including
    RAW ShakeMap uncertainty layers and Bayesian-inspired update surrogates.

    Core Capabilities (v26.6)
    =========================
    1) UQ dataset construction
       - Constructs a unified grid across versions (intersection-based by default) and exports per-version
         mean fields and uncertainty layers into an event-scoped dataset root.

    2) Uncertainty representation + extraction
       - Reads ShakeMap-published uncertainty components (when present) and exposes consistent accessors
         for total and epistemic uncertainty fields.

    3) Bayesian-inspired updating utilities (lightweight)
       - Local, distance-weighted precision fusion surrogates to simulate uncertainty reduction with
         incoming observations.
       - Supports multiple observation types (instrumented and macroseismic) with optional per-type noise.

    4) Target-based diagnostics + audits
       - Point, area, and global uncertainty evolution curves comparing published ShakeMap to alternative
         methods.
       - CSV/JSON audit export for reproducibility.

    5) Optional comparators and diagnostics
       - Kriging- and Monte Carlo-style comparators can be enabled for sensitivity analysis, with clear
         labeling as diagnostic comparators rather than Bayesian filters.

    Inputs and Directory Layout
    ---------------------------
    SHAKEuq expects folders organized as:
        <shakemap_folder>/<event_id>/<versioned_shakemap_files>
        <stations_folder>/<event_id>/<versioned_stationlist_files>   (if separate)
        <rupture_folder>/<event_id>/<versioned_rupture_files>        (if separate)

    Canonical UQ dataset root:
        <base_folder>/<event_id>/...

    Notes
    -----
    - This class is split out of SHAKEuq to reduce file size and isolate UQ development.
    - SHAKEuq may keep a reference to a parent SHAKEuq instance in future refactors, but the initial
      split supports standalone initialization to minimize breaking changes.

    © SHAKEmaps version 26.6
    """

    def __init__(
        self,
        event_id: str,
        event_time=None,
        shakemap_folder=None,
        pager_folder=None,
        file_type: int = 2,
        version_list=None,
        base_folder: str = "./export/SHAKEuq",
        stations_folder: str = None,
        rupture_folder: str = None,
        # --- NEW (optional CDI inputs / policy) ---
        dyfi_cdi_file: str = None,
        dyfi_source: str = "stationlist",      # "stationlist" | "cdi" | "auto"
        cdi_attach_from_version: int = 4,      # CDI becomes "available" from this version index (auto mode)
        dyfi_use_after_hours: float = 24.0,    # (legacy) TaE threshold; not used for auto routing
        dyfi_cdi_max_dist_km: float = 400.0,   # filter far CDI points
        dyfi_cdi_min_nresp: int = 1,           # minimal responses to keep
        dyfi_weight_rule: str = "nresp_threshold",  # "none" | "nresp_threshold" | "sqrt_nresp"
        dyfi_weight_threshold: int = 3,        # if nresp >= threshold => higher weight
        dyfi_weight_low: float = 1.0,
        dyfi_weight_high: float = 2.0,
        dyfi_weight_max: float = 10.0,
    ):
        """
        Initialize SHAKEuq with event details and folder paths.

        Parameters
        ----------
        event_id : str
            USGS event identifier (e.g., "us7000pn9s").
        event_time : str or datetime, optional
            Event origin time (format "%Y-%m-%d %H:%M:%S") or datetime.
        shakemap_folder : str, optional
            Root folder containing versioned ShakeMap grid products.
        pager_folder : str, optional
            Root folder containing versioned PAGER products (optional for UQ).
        file_type : int
            Naming convention selector (1 or 2).
        version_list : list, optional
            Optional default version list for UQ routines. Can be overridden per method call.
        base_folder : str
            Default export/import root for UQ datasets (e.g., "./export/SHAKEuq").
        stations_folder : str, optional
            Optional folder containing stationlist files if separate from shakemap_folder.
        rupture_folder : str, optional
            Optional folder containing rupture JSON files if separate from shakemap_folder.
        """
        
        import os
        from datetime import datetime
    
        self.event_id = event_id
    
        # keep your existing event_time parsing behavior
        if isinstance(event_time, datetime):
            self.event_time = event_time
        elif event_time:
            # accept either "%Y-%m-%d %H:%M:%S" or ISO-like strings
            s = str(event_time).strip()
            try:
                self.event_time = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            except Exception:
                # tolerate "2025-03-28T06:20:54" etc.
                try:
                    self.event_time = datetime.fromisoformat(s.replace("Z", ""))
                except Exception:
                    self.event_time = None
        else:
            self.event_time = None
    
        self.shakemap_folder = os.path.normpath(shakemap_folder) if shakemap_folder else None
        self.pager_folder = os.path.normpath(pager_folder) if pager_folder else None
        self.file_type = int(file_type)
    
        self.version_list = list(version_list) if version_list is not None else None
        self.base_folder = os.path.normpath(base_folder) if base_folder else "./export/SHAKEuq"
    
        self.stations_folder = os.path.normpath(stations_folder) if stations_folder else None
        self.rupture_folder = os.path.normpath(rupture_folder) if rupture_folder else None
    
        # caches (keep names stable)
        self._instrument_data_cache = {}
        self._dyfi_data_cache = {}
        self._rupture_geom_cache = {}
    
        # NEW: CDI cache
        self.dyfi_cdi_file = os.path.normpath(dyfi_cdi_file) if dyfi_cdi_file else None
        self._dyfi_cdi_df_cache = None
    
        # NEW: policy knobs
        self.dyfi_source = str(dyfi_source).lower().strip()  # stationlist|cdi|auto
        self.cdi_attach_from_version = int(cdi_attach_from_version) if cdi_attach_from_version is not None else 4
        self.dyfi_use_after_hours = float(dyfi_use_after_hours) if dyfi_use_after_hours is not None else 24.0
        self.dyfi_cdi_max_dist_km = float(dyfi_cdi_max_dist_km) if dyfi_cdi_max_dist_km is not None else 400.0
        self.dyfi_cdi_min_nresp = int(dyfi_cdi_min_nresp) if dyfi_cdi_min_nresp is not None else 1
    
        self.dyfi_weight_rule = str(dyfi_weight_rule).lower().strip()
        self.dyfi_weight_threshold = int(dyfi_weight_threshold)
        self.dyfi_weight_low = float(dyfi_weight_low)
        self.dyfi_weight_high = float(dyfi_weight_high)
        self.dyfi_weight_max = float(dyfi_weight_max)
    
        # UQ state container (built by uq_build_dataset)
        self.uq_state = None
    

# ==========================================================
# Paste the COMPLETE UQ FRAMEWORK block below this line
# (moved from SHAKEuq.py, then clean duplicate functions)
# ==========================================================
# ##################################################
# #
# # COMPLETE UQ FRAMEWORK (DATASET + ANALYSIS + PLOTS + EXPORTS)
# # Export/Import root = export/<event_id>
# # UPDATE v26.4 (Consolidated patches 1+2+3)
# #
# ##################################################

    ##################################################
    #
    # COMPLETE UQ FRAMEWORK (DATASET + ANALYSIS + PLOTS + EXPORTS)
    # Export/Import root = export/<event_id>
    # UPDATE v26.4 (Consolidated patches 1+2+3)
    #
    ##################################################
    
    # ---------------------------
    # Utilities
    # ---------------------------
    def _uq_safe_float(self, x):
        import numpy as np
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
        import numpy as np
        v = self._uq_safe_float(x)
        return v is not None and np.isfinite(v)
    
    
    def _uq_ensure_dir(self, p):
        from pathlib import Path
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    
  
    
    def _uq_regular_axis(self, start: float, step: float, n: int):
        import numpy as np
        return start + step * np.arange(n, dtype=float)
    
    
    # ---------------------------
    # Canonical UQ directories (avoid uq/uq)
    # ---------------------------
    def _uq_uqdir(self):
        """
        Canonical UQ root directory.
    
        Policy:
        - If output_path is NOT provided elsewhere, UQ output goes to:
            ./export/SHAKEuq/<event_id>/
        - This function derives the canonical root from uq_state["base_folder"].
    
        Robustness:
        - If uq_state["base_folder"] looks like a generic export root (e.g., ./export),
          we automatically expand to ./export/SHAKEuq/<event_id>.
        - If base_folder already ends with ".../uq", use it as-is.
        """
        from pathlib import Path
    
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ state not initialized yet. Run uq_build_dataset(...) first.")
    
        base = Path(self.uq_state.get("base_folder", "")).expanduser()
    
        # If someone accidentally set base_folder to a generic export root, fix it.
        # We treat these as "roots" that should contain SHAKEuq/<event_id>.
        if base.name.lower() in ("export", ".") or str(base).rstrip("/\\").lower().endswith("export"):
            base = base / "SHAKEuq" / str(self.event_id)
    
        # If base already points to ".../uq", keep it; else append "uq"
        uq = base if base.name.lower() == "uq" else (base / "uq")
        uq.mkdir(parents=True, exist_ok=True)
        return uq

    def _uq_results_dir(self, version: int = None):
        """
        Standard results directory:
          export/<event_id>/v<version>/uq_results   (per-version)
          export/<event_id>/uq_results             (global)
        """
        uq = self._uq_uqdir()
        p = (uq / "uq_results") if version is None else (uq / f"v{int(version)}" / "uq_results")
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    
    def _uq_plots_dir(self, sub: str = ""):
        """
        Standard plots directory:
          export/<event_id>/uq_plots[/sub]
        """
        uq = self._uq_uqdir()
        p = uq / "uq_plots"
        if sub:
            p = p / str(sub)
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    
    # ---------------------------
    # Interpolation helpers
    # ---------------------------
    def _uq_bilinear_interp_regular_grid(
        self,
        src_lats_1d,
        src_lons_1d,
        src_field_2d,
        tgt_lats_2d,
        tgt_lons_2d,
        fill_value=float("nan"),
    ):
        import numpy as np
    
        src_lats = np.asarray(src_lats_1d, dtype=float)
        src_lons = np.asarray(src_lons_1d, dtype=float)
        Z = np.asarray(src_field_2d, dtype=float)
    
        if Z.ndim != 2:
            raise ValueError("src_field_2d must be 2D.")
        if src_lats.size < 2 or src_lons.size < 2:
            return np.full_like(tgt_lats_2d, fill_value, dtype=float)
    
        # ensure ascending axes
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
    
        # repair NaNs locally using finite corners
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
        import numpy as np
    
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
    
    
    # ---------------------------
    # XML parsing helpers
    # ---------------------------
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
    


    # ORIENTATION FIX update v26.5 - 15.01.2025 
    def _uq_parse_grid_xml(self, xml_path):
        """
        Returns:
          spec, mean_fields, vs30, mean_units
        mean_units maps field name -> units (string)
        """
        import numpy as np
        import xml.etree.ElementTree as ET
        from pathlib import Path
    
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
    
        # -----------------------------
        # ORIENTATION FIX (minimal)
        # -----------------------------
        # First pass: build all grids and identify LAT/LON if present
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
    
        # Default: no transforms
        transpose = False
        flipud = False
        fliplr = False
    
        # If LAT/LON exist, infer correct layout and fix grids
        if lat_grid is not None and lon_grid is not None:
            # Decide transpose:
            # Expect: lat varies mainly with axis=0, lon varies mainly with axis=1.
            # If reversed, we transpose.
            try:
                lat_var0 = np.nanmedian(np.abs(np.diff(lat_grid, axis=0)))
                lat_var1 = np.nanmedian(np.abs(np.diff(lat_grid, axis=1)))
                lon_var0 = np.nanmedian(np.abs(np.diff(lon_grid, axis=0)))
                lon_var1 = np.nanmedian(np.abs(np.diff(lon_grid, axis=1)))
            except Exception:
                lat_var0 = lat_var1 = lon_var0 = lon_var1 = np.nan
    
            # If lat changes more across columns than rows OR lon changes more across rows than cols → transpose
            if np.isfinite(lat_var0) and np.isfinite(lat_var1) and np.isfinite(lon_var0) and np.isfinite(lon_var1):
                if (lat_var1 > lat_var0) or (lon_var0 > lon_var1):
                    transpose = True
    
            if transpose:
                for kname in list(grids_by_name.keys()):
                    grids_by_name[kname] = grids_by_name[kname].T
                lat_grid = lat_grid.T
                lon_grid = lon_grid.T
    
            # Enforce monotonic directions: lat increasing downward, lon increasing rightward
            try:
                lat_top = float(np.nanmean(lat_grid[0, :]))
                lat_bot = float(np.nanmean(lat_grid[-1, :]))
                if np.isfinite(lat_top) and np.isfinite(lat_bot) and (lat_top > lat_bot):
                    flipud = True
            except Exception:
                pass
    
            try:
                lon_left = float(np.nanmean(lon_grid[:, 0]))
                lon_right = float(np.nanmean(lon_grid[:, -1]))
                if np.isfinite(lon_left) and np.isfinite(lon_right) and (lon_left > lon_right):
                    fliplr = True
            except Exception:
                pass
    
            if flipud:
                for kname in list(grids_by_name.keys()):
                    grids_by_name[kname] = np.flipud(grids_by_name[kname])
                lat_grid = np.flipud(lat_grid)
                lon_grid = np.flipud(lon_grid)
    
            if fliplr:
                for kname in list(grids_by_name.keys()):
                    grids_by_name[kname] = np.fliplr(grids_by_name[kname])
                lat_grid = np.fliplr(lat_grid)
                lon_grid = np.fliplr(lon_grid)
    
        # Store the orientation so uncertainty.xml can follow the same transforms
        try:
            self._uq_last_grid_orientation = {"transpose": bool(transpose), "flipud": bool(flipud), "fliplr": bool(fliplr)}
        except Exception:
            self._uq_last_grid_orientation = {"transpose": False, "flipud": False, "fliplr": False}
    
        # Second pass: fill outputs from corrected grids
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
    
        


    # ORIENTATION FIX update v26.5 15.01.2026
    def _uq_parse_uncertainty_xml(self, xml_path):
        """
        Returns:
          spec, sigma_fields, sigma_units
        sigma_fields uses field names as found in XML (STDPGA, STDMMI, STDPSA03,...)
        sigma_units maps field name -> units string
        """
        import numpy as np
        import xml.etree.ElementTree as ET
        from pathlib import Path
    
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
            raise ValueError(f"uncertainty grid_data too short: got {len(tokens)}, expected {expected} in {xml_path}")
    
        arr = np.array([float(x) for x in tokens[:expected]], dtype=float).reshape((nlat * nlon, nfields))
    
        sigma_fields = {}
        sigma_units = {}
    
        # -----------------------------
        # ORIENTATION FIX (follow grid.xml)
        # -----------------------------
        orient = getattr(self, "_uq_last_grid_orientation", None)
        transpose = bool(orient.get("transpose", False)) if isinstance(orient, dict) else False
        flipud = bool(orient.get("flipud", False)) if isinstance(orient, dict) else False
        fliplr = bool(orient.get("fliplr", False)) if isinstance(orient, dict) else False
    
        for idx, name in idx_to_name.items():
            col = idx - 1
            if col < 0 or col >= nfields:
                continue
    
            grid2d = arr[:, col].reshape((nlat, nlon))
    
            # Apply same orientation transforms as grid.xml
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



    # ---------------------------
    # File resolution (match your conventions)
    # ---------------------------
    def _uq_resolve_paths(self, version: int, stations_folder=None, rupture_folder=None):
        from pathlib import Path
    
        v = int(version)
        event_id = getattr(self, "event_id", None)
        if event_id is None:
            raise AttributeError("SHAKEuq must have self.event_id for UQ file resolution.")
    
        shakemap_folder = Path(getattr(self, "shakemap_folder"))
    
        stations_folder_eff = Path(stations_folder) if stations_folder is not None else Path(getattr(self, "stations_folder", shakemap_folder))
        rupture_folder_eff = Path(rupture_folder) if rupture_folder is not None else Path(getattr(self, "rupture_folder", shakemap_folder))
    
        if hasattr(self, "_get_shakemap_filename"):
            grid_fname = self._get_shakemap_filename(v)
        else:
            grid_fname = f"{event_id}_us_{str(v).zfill(3)}_grid.xml"
        grid_path = shakemap_folder / str(event_id) / str(grid_fname)
    
        # uncertainty
        unc_path = None
        if hasattr(self, "_get_uncertainty_filename"):
            try:
                unc_fname = self._get_uncertainty_filename(v)
                if unc_fname:
                    unc_path = shakemap_folder / str(event_id) / str(unc_fname)
            except Exception:
                unc_path = None
    
        if unc_path is None:
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
    
        # stationlist
        if hasattr(self, "_get_stations_filename"):
            st_fname = self._get_stations_filename(v)
        else:
            st_fname = f"{event_id}_us_{str(v).zfill(3)}_stationlist.json"
        station_path = stations_folder_eff / str(event_id) / str(st_fname)
    
        # rupture
        if hasattr(self, "_get_rupture_filename"):
            rup_fname = self._get_rupture_filename(v)
        else:
            rup_fname = f"{event_id}_us_{str(v).zfill(3)}_rupture.json"
        rupture_path = rupture_folder_eff / str(event_id) / str(rup_fname)
    
        trace = {
            "grid_xml": str(grid_path),
            "uncertainty_xml": str(unc_path) if unc_path is not None else "",
            "stationlist_json": str(station_path),
            "rupture_json": str(rupture_path),
        }
        return grid_path, unc_path, station_path, rupture_path, trace
    
    
    # ---------------------------
    # IMT helpers
    # ---------------------------
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
    
    
    def uq_list_available_imts_global(self, version_list, stations_folder=None, rupture_folder=None):
        per = self.uq_list_available_imts(version_list, stations_folder=stations_folder, rupture_folder=rupture_folder)
        s = set()
        for imts in per.values():
            for k in imts:
                s.add(k)
        return sorted(list(s))
    
    
    def _uq_expand_requested_imts(self, requested_tokens, global_imts):
        requested = []
        for token in requested_tokens:
            t = str(token).upper()
            if t == "PSA":
                requested.extend([k for k in global_imts if str(k).upper().startswith("PSA")])
            else:
                requested.append(str(token))
        # unique stable
        seen = set()
        out = []
        for x in requested:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out
    
    
    def _uq_default_sigma_for_imt(self, imt: str, shape2d=None):
        import numpy as np
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
    
    
    def _uq_sigma_aleatory_default(self, imt: str) -> float:
        imt_u = str(imt).upper()
        if imt_u == "MMI":
            return 0.40
        if imt_u in {"PGA", "PGV"} or imt_u.startswith("PSA"):
            return 0.35
        return 0.35
    
    
    # ---------------------------
    # Stations/DYFI parsing (USGSParser), with PGV+SA and unit capture
    # Returns: inst_list, dyfi_list, debug_dict
    # ---------------------------
    def _uq_parse_stationlist_with_usgsparser(self, stationlist_json):
        """
        Uses USGSParser("instrumented_data", json_file=...).get_dataframe(value_type=...)
        and returns:
          inst_list: list of dicts (lat,lon, pga/pgv/sa optional, unit fields optional)
          dyfi_list: list of dicts (lat,lon, intensity, nresp optional, w)
          dbg: audit counters + notes
        """
        import numpy as np
        import pandas as pd
        from pathlib import Path
    
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
    
        # USGSParser availability
        try:
            # most common in your toolkit
            from modules.SHAKEparser import USGSParser
        except Exception:
            try:
                from SHAKEmaps_Toolkit.modules.SHAKEparser import USGSParser
            except Exception as e:
                dbg["note"] = f"USGSParser_import_failed: {e}"
                return [], [], dbg
    
        inst = []
        dyfi = []
    
        # helpers
        def _numcol(df, candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None
    
        def _latlon_cols(df):
            latc = "latitude" if "latitude" in df.columns else ("lat" if "lat" in df.columns else None)
            lonc = "longitude" if "longitude" in df.columns else ("lon" if "lon" in df.columns else None)
            return latc, lonc
    
        # ---------- instrumented PGA ----------
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
    
        # ---------- instrumented PGV (optional) ----------
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
    
        # ---------- instrumented SA (optional generic) ----------
        # NOTE: Some USGSParser implementations use "sa" or "psa" types. We try both.
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
    
        # ---------- DYFI MMI ----------
        # Your older pattern used instrumented_data parser for MMI; here we try both:
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
    
        # final finite cleanup
        inst2 = [o for o in inst if np.isfinite(o.get("lon", np.nan)) and np.isfinite(o.get("lat", np.nan))]
        dyfi2 = [o for o in dyfi if np.isfinite(o.get("lon", np.nan)) and np.isfinite(o.get("lat", np.nan)) and np.isfinite(o.get("intensity", np.nan))]
    
        if dbg["pga_df_rows"] == 0 and dbg["mmi_df_rows"] == 0 and dbg["pgv_df_rows"] == 0 and dbg["sa_df_rows"] == 0:
            dbg["note"] += "|all_dataframes_empty_or_none"
        if (dbg["pga_df_rows"] > 0 and dbg["pga_rows_after_lonlat_filter"] == 0) or (dbg["mmi_df_rows"] > 0 and dbg["mmi_rows_after_lonlat_filter"] == 0):
            dbg["note"] += "|filtered_all_lonlat"
    
        return inst2, dyfi2, dbg
    
    
    # ---------------------------
    # Unified grid spec (intersection + finest)
    # ---------------------------
    def _uq_build_unified_spec(self, specs, grid_unify="intersection", resolution="finest"):
        import math
        grid_unify = str(grid_unify).strip().lower()
        resolution = str(resolution).strip().lower()
        if grid_unify != "intersection":
            raise ValueError("Only grid_unify='intersection' is implemented.")
        if resolution != "finest":
            raise ValueError("Only resolution='finest' is implemented.")
    
        def bounds(minv, step, n):
            maxv = minv + step * (n - 1)
            return min(minv, maxv), max(minv, maxv)
    
        lat_lows, lat_highs, lon_lows, lon_highs = [], [], [], []
        dxs, dys = [], []
    
        for s in specs:
            nlat, nlon = int(s["nlat"]), int(s["nlon"])
            lat_min, lon_min = float(s["lat_min"]), float(s["lon_min"])
            dy, dx = float(s["dy"]), float(s["dx"])
            la0, la1 = bounds(lat_min, dy, nlat)
            lo0, lo1 = bounds(lon_min, dx, nlon)
            lat_lows.append(la0)
            lat_highs.append(la1)
            lon_lows.append(lo0)
            lon_highs.append(lo1)
            dxs.append(abs(dx))
            dys.append(abs(dy))
    
        u_lat_lo = max(lat_lows)
        u_lat_hi = min(lat_highs)
        u_lon_lo = max(lon_lows)
        u_lon_hi = min(lon_highs)
        if not (u_lat_lo < u_lat_hi and u_lon_lo < u_lon_hi):
            raise ValueError("[UQ] Unified intersection is empty.")
    
        u_dx_abs = min(dxs)
        u_dy_abs = min(dys)
    
        ref = specs[0]
        dy_sign = 1.0 if float(ref["dy"]) >= 0 else -1.0
        dx_sign = 1.0 if float(ref["dx"]) >= 0 else -1.0
        u_dx = dx_sign * u_dx_abs
        u_dy = dy_sign * u_dy_abs
    
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
            "grid_unify": "intersection",
            "resolution": "finest",
        }
    
    
    
    def uq_sanity_report(self):
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")
        rows = self.uq_state.get("sanity_rows", [])
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except Exception:
            return rows

    #update file path issue 
    def uq_build_dataset(
        self,
        event_id: str,
        version_list,
        base_folder: str = "./export/SHAKEuq",
        stations_folder: str = None,
        rupture_folder: str = None,
        imts=("MMI", "PGA", "PGV", "PSA"),
        grid_unify: str = "intersection",
        resolution: str = "finest",
        export: bool = True,
        interp_method: str = "nearest",
        interp_kwargs: dict = None,
        output_units: dict = None,
    ):
        """
        Dataset builder delegated to SHAKEdataset while keeping API stable.
        """
        dataset_builder = SHAKEdataset(
            event_id=event_id,
            event_time=self.event_time,
            shakemap_folder=self.shakemap_folder,
            rupture_folder=rupture_folder or self.rupture_folder,
            stations_folder=stations_folder or self.stations_folder,
            dyfi_cdi_file=self.dyfi_cdi_file,
            version_list=version_list,
            include_cdi_from_version=getattr(self, "cdi_attach_from_version", 4),
            base_folder=base_folder,
            verbose=True,
            dyfi_source=getattr(self, "dyfi_source", "stationlist"),
            dyfi_cdi_max_dist_km=getattr(self, "dyfi_cdi_max_dist_km", 400.0),
            dyfi_cdi_min_nresp=getattr(self, "dyfi_cdi_min_nresp", 1),
            dyfi_weight_rule=getattr(self, "dyfi_weight_rule", "nresp_threshold"),
            dyfi_weight_threshold=getattr(self, "dyfi_weight_threshold", 3),
            dyfi_weight_low=getattr(self, "dyfi_weight_low", 1.0),
            dyfi_weight_high=getattr(self, "dyfi_weight_high", 2.0),
            dyfi_weight_max=getattr(self, "dyfi_weight_max", 10.0),
        )
        ds_state = dataset_builder.build(
            version_list=version_list,
            base_folder=base_folder,
            stations_folder=stations_folder,
            rupture_folder=rupture_folder,
            imts=imts,
            grid_unify=grid_unify,
            resolution=resolution,
            export=export,
            interp_method=interp_method,
            interp_kwargs=interp_kwargs,
            output_units=output_units,
        )
        self.uq_state = self._uq__coerce_dataset_state(ds_state)
        return self.uq_state

    def _uq__coerce_dataset_state(self, ds_state):
        """
        Compatibility shim for dataset states built by SHAKEdataset.
        Ensures legacy keys exist for downstream UQ methods.
        """
        if ds_state is None:
            return None
        state = dict(ds_state)
        state.setdefault("event_id", str(getattr(self, "event_id", "")))
        state.setdefault("version_list", [])
        state.setdefault("requested_imts", [])
        state.setdefault("per_version_available_imts", {})
        state.setdefault("unified_spec", {})
        state.setdefault("unified_axes", {})
        state.setdefault("versions_raw", {})
        state.setdefault("versions_unified", {})
        state.setdefault("obs_by_version", {})
        state.setdefault("sanity_rows", [])
        state.setdefault("file_traces", {})
        state.setdefault("interp_method", "nearest")
        state.setdefault("interp_kwargs", {})
        state.setdefault("output_units_requested", None)
        if "stations_folder_used" not in state:
            state["stations_folder_used"] = str(getattr(self, "stations_folder", "")) or None
        if "rupture_folder_used" not in state:
            state["rupture_folder_used"] = str(getattr(self, "rupture_folder", "")) or None
        if "base_folder" not in state or not state.get("base_folder"):
            state["base_folder"] = str(self._uq_uqdir())
        return state

    def uq_get_dataset_state(self):
        return getattr(self, "uq_state", None)

    def uq_get_versions(self):
        if not hasattr(self, "uq_state") or self.uq_state is None:
            return []
        return list(self.uq_state.get("version_list", []) or [])

    def uq_get_obs(self, version: int):
        if not hasattr(self, "uq_state") or self.uq_state is None:
            return {}
        return (self.uq_state.get("obs_by_version", {}) or {}).get(int(version), {})

    
    
    # ---------------------------
    # Units + working space conversions
    # ---------------------------
    def uq_set_imt_units(self, pga_unit: str = None, pgv_unit: str = None, psa_unit: str = None):
        if not hasattr(self, "uq_units") or self.uq_units is None:
            self.uq_units = {}
        if pga_unit is not None:
            self.uq_units["PGA"] = str(pga_unit)
        if psa_unit is not None:
            self.uq_units["PSA"] = str(psa_unit)
        if pgv_unit is not None:
            self.uq_units["PGV"] = str(pgv_unit)
        return self.uq_units
    
    
    def _uq_is_lognormal_imt(self, imt: str) -> bool:
        imt_u = str(imt).upper()
        return (imt_u in {"PGA", "PGV"}) or imt_u.startswith("PSA")
    
    
    def _uq_default_imt_unit(self, imt: str) -> str:
        imt_u = str(imt).upper()
        if imt_u == "MMI":
            return "MMI"
        if imt_u == "PGV":
            return "cm/s"
        if imt_u == "PGA" or imt_u.startswith("PSA"):
            return "%g"
        return ""
    
    
    def _uq_get_imt_unit(self, imt: str) -> str:
        imt_u = str(imt).upper()
        if hasattr(self, "uq_units") and isinstance(self.uq_units, dict):
            if imt_u.startswith("PSA"):
                if "PSA" in self.uq_units:
                    return str(self.uq_units["PSA"])
                if "PGA" in self.uq_units:
                    return str(self.uq_units["PGA"])
            if imt_u in self.uq_units:
                return str(self.uq_units[imt_u])
        return self._uq_default_imt_unit(imt)
    
    
    def _uq_base_unit_for_sigma(self, imt: str) -> str:
        imt_u = str(imt).upper()
        if imt_u == "PGV":
            return "cm/s"
        if imt_u == "PGA" or imt_u.startswith("PSA"):
            return "g"
        return "MMI"
    
    
    def _uq_convert_units(self, x, imt: str, from_unit: str, to_unit: str):
        import numpy as np
        if x is None:
            return None
        arr = np.asarray(x, dtype=float)
    
        imt_u = str(imt).upper()
        fu = str(from_unit)
        tu = str(to_unit)
        if fu == tu or imt_u == "MMI":
            return arr
    
        if imt_u == "PGA" or imt_u.startswith("PSA"):
            if fu in {"%g", "percent_g", "pct_g"}:
                g = arr / 100.0
            elif fu == "g":
                g = arr
            elif fu in {"m/s^2", "m/s2"}:
                g = arr / 9.80665
            else:
                g = arr
            if tu in {"%g", "percent_g", "pct_g"}:
                return g * 100.0
            if tu == "g":
                return g
            if tu in {"m/s^2", "m/s2"}:
                return g * 9.80665
            return g
    
        if imt_u == "PGV":
            if fu in {"cm/s", "cmps"}:
                cms = arr
            elif fu in {"m/s", "mps"}:
                cms = arr * 100.0
            else:
                cms = arr
            if tu in {"cm/s", "cmps"}:
                return cms
            if tu in {"m/s", "mps"}:
                return cms / 100.0
            return cms
    
        return arr
    
    
    def _uq_mu_to_working_space(self, mu_linear, imt: str):
        import numpy as np
        imt_u = str(imt).upper()
        if not self._uq_is_lognormal_imt(imt_u):
            return np.asarray(mu_linear, dtype=float)
    
        from_unit = self._uq_default_imt_unit(imt_u)        # source grid.xml convention
        base_unit = self._uq_base_unit_for_sigma(imt_u)     # sigma base unit
        mu_base = self._uq_convert_units(mu_linear, imt_u, from_unit, base_unit)
    
        mu_base = np.asarray(mu_base, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(np.maximum(1e-30, mu_base))
    
    
    def _uq_obs_to_working_space(self, obs_value_linear, imt: str, obs_unit: str = None):
        import numpy as np
        imt_u = str(imt).upper()
        if obs_value_linear is None:
            return None
        v = float(obs_value_linear)
    
        if not self._uq_is_lognormal_imt(imt_u):
            return v
    
        from_unit = str(obs_unit) if (obs_unit is not None and str(obs_unit).strip() != "") else self._uq_default_imt_unit(imt_u)
        base_unit = self._uq_base_unit_for_sigma(imt_u)
    
        vb = float(self._uq_convert_units(np.array([v]), imt_u, from_unit, base_unit)[0])
        if vb <= 0:
            return None
        return float(np.log(vb))
    
    
    def _uq_threshold_to_working_space(self, threshold, imt: str):
        import numpy as np
        imt_u = str(imt).upper()
        thr = float(threshold)
        if not self._uq_is_lognormal_imt(imt_u):
            return thr
    
        from_unit = self._uq_get_imt_unit(imt_u)            # user-facing thresholds
        base_unit = self._uq_base_unit_for_sigma(imt_u)
    
        tb = float(self._uq_convert_units(np.array([thr]), imt_u, from_unit, base_unit)[0])
        if tb <= 0:
            return float("-inf")
        return float(np.log(tb))
    
    
    def _uq_mu_from_working_space(self, mu_working, imt: str):
        import numpy as np
        imt_u = str(imt).upper()
        muw = np.asarray(mu_working, dtype=float)
        if not self._uq_is_lognormal_imt(imt_u):
            return muw
    
        base_unit = self._uq_base_unit_for_sigma(imt_u)
        to_unit = self._uq_get_imt_unit(imt_u)
        mu_base = np.exp(muw)
        return self._uq_convert_units(mu_base, imt_u, base_unit, to_unit)
    
    
    # ---------------------------
    # Sigma decomposition
    # ---------------------------
    def _uq_decompose_sigma(
        self,
        sigma_prior_total,
        imt: str,
        sigma_aleatory=None,
        sigma_total_from_shakemap: bool = True,
        eps: float = 1e-6,
    ):
        import numpy as np
        sig_total = np.asarray(sigma_prior_total, dtype=float)
        sigma_a = float(self._uq_sigma_aleatory_default(imt) if sigma_aleatory is None else sigma_aleatory)
    
        if sigma_total_from_shakemap:
            sig_ep2 = np.maximum(0.0, sig_total * sig_total - sigma_a * sigma_a)
            sig_ep = np.sqrt(sig_ep2)
        else:
            sig_ep = np.maximum(eps, sig_total)
            sig_total = np.sqrt(np.maximum(0.0, sigma_a * sigma_a + sig_ep * sig_ep))
    
        sig_ep = np.maximum(eps, sig_ep)
        sig_total = np.sqrt(np.maximum(0.0, sigma_a * sigma_a + sig_ep * sig_ep))
        return sig_total, sigma_a, sig_ep
    
    
    # ---------------------------
    # Observation collection (MMI/PGA/PGV/PSA*)
    # obs dict includes optional 'unit'
    # ---------------------------
    def _uq_bayes_impact_audit(self, sigma_ep_prior, sigma_ep_post, obs_used: int):
        import numpy as np
        prior = np.asarray(sigma_ep_prior, dtype=float)
        post = np.asarray(sigma_ep_post, dtype=float)
        d = prior - post
        finite = np.isfinite(d)
        d = d[finite]
    
        if d.size == 0:
            return {
                "n_obs_used": int(obs_used),
                "n_finite_cells": 0,
                "frac_decreased": 0.0,
                "frac_increased": 0.0,
                "mean_drop_all": 0.0,
                "median_drop_all": 0.0,
                "p90_drop_all": 0.0,
                "max_drop_all": 0.0,
                "mean_drop_changed": 0.0,
                "median_drop_changed": 0.0,
                "p90_drop_changed": 0.0,
                "max_drop_changed": 0.0,
                "note": "no_finite_cells_for_audit",
            }
    
        dec = d > 0
        inc = d < 0
    
        out = {
            "n_obs_used": int(obs_used),
            "n_finite_cells": int(d.size),
            "frac_decreased": float(dec.mean()) if d.size else 0.0,
            "frac_increased": float(inc.mean()) if d.size else 0.0,
            "mean_drop_all": float(np.mean(d)),
            "median_drop_all": float(np.median(d)),
            "p90_drop_all": float(np.percentile(d, 90)),
            "max_drop_all": float(np.max(d)),
            "note": "" if obs_used > 0 else "no_observations_used",
        }
    
        if np.any(dec):
            dd = d[dec]
            out.update(
                {
                    "mean_drop_changed": float(np.mean(dd)),
                    "median_drop_changed": float(np.median(dd)),
                    "p90_drop_changed": float(np.percentile(dd, 90)),
                    "max_drop_changed": float(np.max(dd)),
                }
            )
        else:
            out.update({"mean_drop_changed": 0.0, "median_drop_changed": 0.0, "p90_drop_changed": 0.0, "max_drop_changed": 0.0})
    
        return out
    
    
    # ---------------------------
    # Bayes update (epistemic-only)
    # ---------------------------
    def uq_bayes_update(
        self,
        version_list=None,
        imt: str = "MMI",
        update_radius_km: float = 25.0,
        update_kernel: str = "gaussian",  # gaussian|tophat
        sigma_aleatory=None,
        sigma_total_from_shakemap: bool = True,
        measurement_sigma=None,
        min_effective_weight: float = 1e-6,
        export: bool = True,
        make_audit: bool = True,
    ):
        import numpy as np
        import json
        from pathlib import Path
    
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")
    
        base = Path(self.uq_state["base_folder"])
        versions = self.uq_state["version_list"] if version_list is None else [int(v) for v in list(version_list)]
        imt = str(imt)
    
        axes = self.uq_state["unified_axes"]
        LAT2 = np.asarray(axes["lat2d"], dtype=float)
        LON2 = np.asarray(axes["lon2d"], dtype=float)
    
        kernel = str(update_kernel).strip().lower()
        radius = float(update_radius_km)
        if radius < 0:
            radius = 0.0
    
        results = {
            "method": "bayes_update",
            "imt": imt,
            "per_version": {},
            "audit": {},
            "kernel": kernel,
            "radius_km": float(radius),
            "min_effective_weight": float(min_effective_weight),
            "measurement_sigma": None if measurement_sigma is None else float(measurement_sigma),
            "mu_unit_preferred": str(self._uq_get_imt_unit(imt)),
            "sigma_base_unit": str(self._uq_base_unit_for_sigma(imt)),
            "sigma_total_from_shakemap": bool(sigma_total_from_shakemap),
        }
    
        for v in versions:
            uv = self.uq_state["versions_unified"].get(int(v), None)
            if uv is None:
                continue
    
            mu_prior_linear = np.asarray(uv["unified_mean"].get(imt, None), dtype=float)
    
            sig_total_prior_grid = np.asarray(
                uv["unified_sigma_prior_total"].get(imt, self._uq_default_sigma_for_imt(imt, mu_prior_linear)),
                dtype=float,
            )
    
            sig_total_prior, sig_a, sig_ep_prior = self._uq_decompose_sigma(
                sig_total_prior_grid,
                imt=imt,
                sigma_aleatory=sigma_aleatory,
                sigma_total_from_shakemap=sigma_total_from_shakemap,
            )
    
            mu_prior_work = self._uq_mu_to_working_space(mu_prior_linear, imt)
            mu_post_work = np.asarray(mu_prior_work, dtype=float).copy()
            sig_ep_post = np.asarray(sig_ep_prior, dtype=float).copy()
    
            obs = self._uq_collect_observations(int(v), imt)
            obs_used = 0
    
            sig_meas = 0.0 if measurement_sigma is None else float(measurement_sigma)
    
            for o in obs:
                lat_o = float(o["lat"])
                lon_o = float(o["lon"])
                val_o_lin = o.get("value", None)
                if val_o_lin is None:
                    continue
    
                val_o_work = self._uq_obs_to_working_space(val_o_lin, imt, obs_unit=o.get("unit", None))
                if val_o_work is None or (not np.isfinite(val_o_work)):
                    continue
    
                w_o = float(o.get("w", 1.0))
                dist = self._uq_haversine_km(LAT2, LON2, lat_o, lon_o)
    
                if radius == 0.0:
                    idx = np.unravel_index(np.nanargmin(dist), dist.shape)
                    mask = np.zeros(dist.shape, dtype=bool)
                    mask[idx] = True
                    dsel = np.array([0.0], dtype=float)
                else:
                    mask = dist <= radius
                    if not np.any(mask):
                        continue
                    dsel = dist[mask]
    
                if kernel == "gaussian":
                    sigma_k = max(1e-6, radius / 3.0)
                    wdist = np.exp(-0.5 * (dsel / sigma_k) ** 2)
                else:
                    wdist = np.ones_like(dsel, dtype=float)
    
                w_eff = w_o * wdist
                if np.nanmax(w_eff) < min_effective_weight:
                    continue
    
                sig_obs2 = (sig_a * sig_a) + (sig_meas * sig_meas)
                sig_obs2 = max(1e-12, float(sig_obs2))
    
                sig_ep2 = sig_ep_post[mask] ** 2
                prec_prior = 1.0 / np.maximum(1e-12, sig_ep2)
                prec_like = w_eff / sig_obs2
                prec_post = prec_prior + prec_like
    
                mu_sel = mu_post_work[mask]
                mu_post_sel = (mu_sel * prec_prior + val_o_work * prec_like) / np.maximum(1e-12, prec_post)
                sig_ep_post_sel = np.sqrt(1.0 / np.maximum(1e-12, prec_post))
    
                mu_post_work[mask] = mu_post_sel
                sig_ep_post[mask] = sig_ep_post_sel
                obs_used += 1
    
            sig_total_post = np.sqrt(np.maximum(0.0, sig_a * sig_a + sig_ep_post * sig_ep_post))
            audit = self._uq_bayes_impact_audit(sig_ep_prior, sig_ep_post, obs_used=obs_used) if make_audit else {}
    
            mu_post_linear = self._uq_mu_from_working_space(mu_post_work, imt)
            mu_prior_linear_out = self._uq_mu_from_working_space(mu_prior_work, imt)
    
            results["per_version"][int(v)] = {
                "sigma_aleatory": float(sig_a),
                "n_obs_used": int(obs_used),
                "mu_prior_mean": float(np.nanmean(mu_prior_linear_out)),
                "mu_post_mean": float(np.nanmean(mu_post_linear)),
                "sigma_ep_prior_mean": float(np.nanmean(sig_ep_prior)),
                "sigma_ep_post_mean": float(np.nanmean(sig_ep_post)),
                "sigma_total_prior_mean": float(np.nanmean(sig_total_prior)),
                "sigma_total_post_mean": float(np.nanmean(sig_total_post)),
            }
            if make_audit:
                results["audit"][int(v)] = audit
    
            if export:
                vdir = self._uq_ensure_dir(base / f"v{int(v)}" / "uq_results")
                np.savez_compressed(
                    vdir / f"bayesupdate_{imt}_posterior.npz",
                    mu_post=np.asarray(mu_post_linear, dtype=float),
                    sigma_ep_post=np.asarray(sig_ep_post, dtype=float),
                    sigma_total_post=np.asarray(sig_total_post, dtype=float),
                    mu_prior=np.asarray(mu_prior_linear_out, dtype=float),
                    sigma_ep_prior=np.asarray(sig_ep_prior, dtype=float),
                    sigma_total_prior=np.asarray(sig_total_prior, dtype=float),
                    sigma_aleatory=np.array([sig_a], dtype=float),
                    mu_post_working=np.asarray(mu_post_work, dtype=float),
                    mu_prior_working=np.asarray(mu_prior_work, dtype=float),
                    mu_unit=str(self._uq_get_imt_unit(imt)),
                    sigma_base_unit=str(self._uq_base_unit_for_sigma(imt)),
                )
                if make_audit:
                    with open(vdir / f"bayesupdate_{imt}_audit.json", "w", encoding="utf-8") as f:
                        json.dump(audit, f, indent=2)
    
        if export:
            with open(base / f"bayesupdate_{imt}_summary.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    
        self.uq_last_bayes = results
        return results
    
    
    # ---------------------------
    # Posterior loaders for local metrics/plots
    # ---------------------------
    def _uq_load_bayes_posterior_npz(self, version: int, imt: str):
        import numpy as np
        from pathlib import Path
    
        v = int(version)
        imt = str(imt)
    
        uq = self._uq_uqdir()
        base = Path(self.uq_state["base_folder"])
    
        candidates = [
            uq / f"v{v}" / "uq_results" / f"bayesupdate_{imt}_posterior.npz",
            uq / ("v%03d" % v) / "uq_results" / f"bayesupdate_{imt}_posterior.npz",
            uq / f"v{v}" / f"bayesupdate_{imt}_posterior.npz",
            base / f"v{v}" / "uq_results" / f"bayesupdate_{imt}_posterior.npz",
            base / ("v%03d" % v) / "uq_results" / f"bayesupdate_{imt}_posterior.npz",
            base / "uq" / f"v{v}" / "uq_results" / f"bayesupdate_{imt}_posterior.npz",
            base / "uq" / ("v%03d" % v) / "uq_results" / f"bayesupdate_{imt}_posterior.npz",
        ]
    
        for p in candidates:
            try:
                if p.exists():
                    dat = np.load(p, allow_pickle=True)
                    return {k: dat[k] for k in dat.files}
            except Exception:
                continue
        return None
    
    
    def _uq_load_hierarchical_posterior_npz(self, imt: str):
        import numpy as np
        from pathlib import Path
    
        imt = str(imt)
    
        uq = self._uq_uqdir()
        base = Path(self.uq_state["base_folder"])
    
        candidates = [
            uq / "uq_results" / f"hierarchical_{imt}_posterior.npz",
            base / "uq_results" / f"hierarchical_{imt}_posterior.npz",
            base / "uq" / "uq_results" / f"hierarchical_{imt}_posterior.npz",
        ]
        for p in candidates:
            try:
                if p.exists():
                    dat = np.load(p, allow_pickle=True)
                    return {k: dat[k] for k in dat.files}
            except Exception:
                continue
        return None
    
    
    # ---------------------------
    # PoE helpers
    # ---------------------------
    def _uq_norm_cdf(self, x):
        import numpy as np
        from math import sqrt
        x = np.asarray(x, dtype=float)
        try:
            from scipy.special import erf as sp_erf
            return 0.5 * (1.0 + sp_erf(x / sqrt(2.0)))
        except Exception:
            from math import erf
            return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))
    
    
    def _uq_poe_exceed(self, mu, sigma, threshold, imt: str = None):
        import numpy as np
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        thr = float(threshold)
    
        if imt is None:
            imt = getattr(self, "_uq_poe_imt_context", "MMI")
        imt = str(imt)
    
        if self._uq_is_lognormal_imt(imt):
            mu_work = self._uq_mu_to_working_space(mu, imt)
            thr_work = self._uq_threshold_to_working_space(thr, imt)
            if not np.isfinite(thr_work):
                poe = np.ones_like(mu_work, dtype=float)
                poe[~np.isfinite(mu_work)] = np.nan
                return poe
            z = (thr_work - mu_work) / np.maximum(1e-12, sigma)
            return 1.0 - self._uq_norm_cdf(z)
    
        z = (thr - mu) / np.maximum(1e-12, sigma)
        return 1.0 - self._uq_norm_cdf(z)
    
    
    # ---------------------------
    # Local metrics helpers
    # ---------------------------
    def _uq_collect_obs_coords_all(self, version: int, imt: str):
        import numpy as np
        obs = self._uq_collect_observations(int(version), str(imt))
        if not obs:
            return np.array([]), np.array([]), np.array([])
        lats = np.array([o["lat"] for o in obs], dtype=float)
        lons = np.array([o["lon"] for o in obs], dtype=float)
        w = np.array([o.get("w", 1.0) for o in obs], dtype=float)
        good = np.isfinite(lats) & np.isfinite(lons) & np.isfinite(w)
        return lats[good], lons[good], w[good]
    
    
    def _uq_mask_within_km_of_points(self, lat2d, lon2d, pts_lat, pts_lon, radius_km: float):
        import numpy as np
        if pts_lat.size == 0:
            return np.zeros(lat2d.shape, dtype=bool)
    
        radius = float(radius_km)
        if radius <= 0:
            dist0 = self._uq_haversine_km(lat2d, lon2d, float(pts_lat[0]), float(pts_lon[0]))
            idx = np.unravel_index(np.nanargmin(dist0), dist0.shape)
            m = np.zeros(lat2d.shape, dtype=bool)
            m[idx] = True
            return m
    
        mask = np.zeros(lat2d.shape, dtype=bool)
        for la, lo in zip(pts_lat, pts_lon):
            d = self._uq_haversine_km(lat2d, lon2d, float(la), float(lo))
            mask |= (d <= radius)
        return mask
    
    
    def _uq_prior_sigma_fields(self, version: int, imt: str):
        import numpy as np
        v = int(version)
        imt = str(imt)
    
        uv = self.uq_state["versions_unified"].get(v, None)
        if uv is None:
            raise KeyError(f"No unified data for version {v}")
    
        mu = np.asarray(uv["unified_mean"].get(imt, None), dtype=float)
        sig_total_prior_grid = np.asarray(
            uv["unified_sigma_prior_total"].get(imt, self._uq_default_sigma_for_imt(imt, mu)),
            dtype=float,
        )
    
        sig_total_prior, sig_a, sig_ep_prior = self._uq_decompose_sigma(
            sig_total_prior_grid, imt=imt, sigma_aleatory=None, sigma_total_from_shakemap=True
        )
        return sig_ep_prior, sig_total_prior, float(sig_a)
    
    
    
    def uq_compute_local_metrics(
        self,
        imt: str = "MMI",
        method: str = "bayesupdate",
        which_sigma: str = "epistemic",   # epistemic|total
        station_radius_km: float = 40.0,
        poe_threshold: float = 5.0,
        poe_cut: float = 0.5,
        version_list=None,
        export: bool = True,
    ):
        import numpy as np
        import json
    
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")
    
        imt = str(imt)
        method = str(method).strip().lower()
        which = str(which_sigma).strip().lower()
        versions = self.uq_state["version_list"] if version_list is None else [int(v) for v in list(version_list)]
    
        axes = self.uq_state["unified_axes"]
        LAT2 = np.asarray(axes["lat2d"], dtype=float)
        LON2 = np.asarray(axes["lon2d"], dtype=float)
    
        def mean_on(mask, arr):
            mask = np.asarray(mask, dtype=bool)
            a = np.asarray(arr, dtype=float)
            sel = a[mask & np.isfinite(a)]
            return float("nan") if sel.size == 0 else float(np.mean(sel))
    
        rows = []
        self._uq_poe_imt_context = imt
    
        for v in versions:
            sig_ep_prior, sig_total_prior, _ = self._uq_prior_sigma_fields(v, imt)
            sig_prior = sig_total_prior if which == "total" else sig_ep_prior
    
            sig_ep_post, sig_total_post = self._uq_posterior_sigma_fields(v, imt, method=method)
            sig_post = sig_total_post if which == "total" else sig_ep_post
    
            pts_lat, pts_lon, _w = self._uq_collect_obs_coords_all(v, imt)
            m_station = self._uq_mask_within_km_of_points(LAT2, LON2, pts_lat, pts_lon, float(station_radius_km))
    
            mu = np.asarray(self.uq_state["versions_unified"][int(v)]["unified_mean"].get(imt, None), dtype=float)
    
            poe = self._uq_poe_exceed(mu, sig_post, float(poe_threshold), imt=imt)
            m_poe = np.isfinite(poe) & (poe >= float(poe_cut))
    
            d_sigma = sig_prior - sig_post
    
            rows.append({
                "version": int(v),
                "n_obs": int(pts_lat.size),
                "station_radius_km": float(station_radius_km),
                "poe_threshold": float(poe_threshold),
                "poe_cut": float(poe_cut),
    
                "mean_sigma_prior_global": mean_on(np.isfinite(sig_prior), sig_prior),
                "mean_sigma_post_global": mean_on(np.isfinite(sig_post), sig_post),
                "mean_delta_sigma_global": mean_on(np.isfinite(d_sigma), d_sigma),
    
                "mean_sigma_prior_near_stations": mean_on(m_station, sig_prior),
                "mean_sigma_post_near_stations": mean_on(m_station, sig_post),
                "mean_delta_sigma_near_stations": mean_on(m_station, d_sigma),
    
                "mean_sigma_prior_poe": mean_on(m_poe, sig_prior),
                "mean_sigma_post_poe": mean_on(m_poe, sig_post),
                "mean_delta_sigma_poe": mean_on(m_poe, d_sigma),
    
                "frac_cells_near_stations": float(np.mean(m_station)) if m_station.size else 0.0,
                "frac_cells_poe": float(np.mean(m_poe)) if m_poe.size else 0.0,
            })
    
        out = {
            "method": "local_metrics",
            "posterior_method": method,
            "imt": imt,
            "which_sigma": which,
            "mu_unit": str(self._uq_get_imt_unit(imt)),
            "sigma_base_unit": str(self._uq_base_unit_for_sigma(imt)),
            "rows": rows,
        }
    
        if export:
            rdir = self._uq_results_dir(None)
            json_path = rdir / f"local_metrics_{method}_{imt}_{which}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
    
            csv_path = rdir / f"local_metrics_{method}_{imt}_{which}.csv"
            if rows:
                cols = list(rows[0].keys())
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(",".join(cols) + "\n")
                    for r in rows:
                        f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    
        self.uq_last_local_metrics = out
        return out
    
    

    
 
    
    def _uq_sigma_unit_label(self, imt: str):
        imt_u = str(imt).upper()
        if self._uq_is_lognormal_imt(imt_u):
            base = self._uq_base_unit_for_sigma(imt_u)
            return f"ln({base})"
        return "MMI"
    
    
    def uq_plot_shakemap_sigma_and_data_map(
        self,
        version: int,
        imt: str = "MMI",
        which_sigma: str = "total",     # total|epistemic
        station_radius_km: float = 0.0,
        cmap: str = "viridis",
        vmin=None,
        vmax=None,
        show_obs: bool = True,
        obs_size: float = 10.0,
        figsize=(10, 8),
        show_title: bool = True,
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        dpi: int = 300,
        show: bool = True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        try:
            import cartopy.crs as ccrs
            has_cartopy = True
        except Exception:
            has_cartopy = False
            ccrs = None
    
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")
    
        v = int(version)
        imt = str(imt)
        which = str(which_sigma).strip().lower()
    
        sig_ep_prior, sig_total_prior, _sig_a = self._uq_prior_sigma_fields(v, imt)
        sig = sig_total_prior if which == "total" else sig_ep_prior
    
        pts_lat, pts_lon, _w = self._uq_collect_obs_coords_all(v, imt)
    
        axes = self.uq_state["unified_axes"]
        LAT2 = np.asarray(axes["lat2d"], dtype=float)
        LON2 = np.asarray(axes["lon2d"], dtype=float)
    
        fig, ax = self._uq_cartopy_axes(figsize=figsize)
        self._uq_set_map_extent_from_axes(ax)
    
        pm = ax.pcolormesh(
            LON2, LAT2, np.asarray(sig, dtype=float),
            transform=(ccrs.PlateCarree() if has_cartopy else None),
            cmap=str(cmap),
            vmin=vmin, vmax=vmax,
            shading="auto",
            zorder=5,
        )
    
        if show_obs and pts_lat.size > 0:
            ax.scatter(
                pts_lon, pts_lat,
                s=float(obs_size),
                transform=(ccrs.PlateCarree() if has_cartopy else None),
                zorder=30,
                label="Observations",
            )
    
        if station_radius_km and station_radius_km > 0 and pts_lat.size > 0:
            m = self._uq_mask_within_km_of_points(LAT2, LON2, pts_lat, pts_lon, float(station_radius_km))
            try:
                ax.contour(
                    LON2, LAT2, m.astype(float),
                    levels=[0.5],
                    transform=(ccrs.PlateCarree() if has_cartopy else None),
                    zorder=25,
                )
            except Exception:
                pass
    
        cbar = fig.colorbar(pm, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(f"ShakeMap σ ({which}); units={self._uq_sigma_unit_label(imt)}")
    
        raw = self.uq_state["versions_raw"].get(v, {})
        n_inst = int(raw.get("counts", {}).get("n_instrumented", 0))
        n_dyfi = int(raw.get("counts", {}).get("n_dyfi", 0))
        ax.text(
            0.01, 0.01,
            f"v{v}\ninst={n_inst}\ndyfi={n_dyfi}",
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.8),
            zorder=40,
        )
    
        if show_title:
            ax.set_title(f"ShakeMap σ + data included — {imt} ({which}) — v{v}", fontsize=14, pad=10)
    
        if show_obs and pts_lat.size > 0:
            ax.legend(loc="upper right", fontsize=10, frameon=True)
    
        fig.tight_layout()
    
        if save:
            self._uq_save_figure(
                fig,
                fname_stem=f"qa_shakemap_sigma_and_data_{imt}_{which}_v{v}",
                subdir="uq_plots/qa_maps",
                output_path=output_path,
                save_formats=save_formats,
                dpi=dpi,
            )
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig, ax
    
    
    def uq_plot_shakemap_sigma_change_map(
        self,
        imt: str = "MMI",
        which_sigma: str = "total",
        v_from: int = None,
        v_to: int = None,
        mask_to_data_radius_km: float = 0.0,
        cmap: str = "viridis",
        vmin=None,
        vmax=None,
        show_obs: bool = True,
        obs_size: float = 10.0,
        figsize=(10, 8),
        show_title: bool = True,
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        dpi: int = 300,
        show: bool = True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        try:
            import cartopy.crs as ccrs
            has_cartopy = True
        except Exception:
            has_cartopy = False
            ccrs = None
    
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")
    
        imt = str(imt)
        which = str(which_sigma).strip().lower()
    
        versions = [int(v) for v in self.uq_state["version_list"]]
        if v_from is None:
            v_from = versions[0]
        if v_to is None:
            v_to = versions[-1]
        v_from = int(v_from)
        v_to = int(v_to)
    
        ep1, tot1, _ = self._uq_prior_sigma_fields(v_from, imt)
        ep2, tot2, _ = self._uq_prior_sigma_fields(v_to, imt)
    
        s1 = tot1 if which == "total" else ep1
        s2 = tot2 if which == "total" else ep2
        d = np.asarray(s1 - s2, dtype=float)
    
        axes = self.uq_state["unified_axes"]
        LAT2 = np.asarray(axes["lat2d"], dtype=float)
        LON2 = np.asarray(axes["lon2d"], dtype=float)
    
        pts_lat, pts_lon, _w = self._uq_collect_obs_coords_all(v_to, imt)
        if mask_to_data_radius_km and mask_to_data_radius_km > 0 and pts_lat.size > 0:
            m = self._uq_mask_within_km_of_points(LAT2, LON2, pts_lat, pts_lon, float(mask_to_data_radius_km))
            d = np.where(m, d, np.nan)
    
        fig, ax = self._uq_cartopy_axes(figsize=figsize)
        self._uq_set_map_extent_from_axes(ax)
    
        pm = ax.pcolormesh(
            LON2, LAT2, d,
            transform=(ccrs.PlateCarree() if has_cartopy else None),
            cmap=str(cmap),
            vmin=vmin, vmax=vmax,
            shading="auto",
            zorder=5,
        )
    
        if show_obs and pts_lat.size > 0:
            ax.scatter(
                pts_lon, pts_lat,
                s=float(obs_size),
                transform=(ccrs.PlateCarree() if has_cartopy else None),
                zorder=30,
                label=f"Obs (v{v_to})",
            )
    
        cbar = fig.colorbar(pm, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(f"Δσ = σ(v{v_from}) − σ(v{v_to}) ({which}); units={self._uq_sigma_unit_label(imt)}")
    
        if show_title:
            extra = f", masked to R={mask_to_data_radius_km:g} km" if mask_to_data_radius_km and mask_to_data_radius_km > 0 else ""
            ax.set_title(f"ShakeMap local uncertainty change — {imt} ({which}) — v{v_from}→v{v_to}{extra}", fontsize=14, pad=10)
    
        if show_obs and pts_lat.size > 0:
            ax.legend(loc="upper right", fontsize=10, frameon=True)
    
        fig.tight_layout()
    
        if save:
            self._uq_save_figure(
                fig,
                fname_stem=f"qa_shakemap_sigma_change_{imt}_{which}_v{v_from}_to_v{v_to}_maskR{mask_to_data_radius_km:g}",
                subdir="uq_plots/qa_maps",
                output_path=output_path,
                save_formats=save_formats,
                dpi=dpi,
            )
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig, ax
    
    
    
    
    # ============================================================
    #
    # UQ PLOTTING UPGRADE PATCH (PUBLISHING-READY + CARTOPY MAPS)
    # UPDATED PATCH 4 — PATH SAFE + BUG FIXES
    #
    # Fixes:
    #  - Uses robust posterior loaders (bayes/hier) instead of brittle Path(...)
    #  - Fixes recursion bug in uq_export_sigma_reduction_maps()
    #  - Makes _uq_cartopy_axes() safe when fig+ax are passed (multi-panel plots)
    #
    # ============================================================
    
    def _uq_plot_style_defaults(self, font_sizes=None, grid_kwargs=None, legend_kwargs=None):
        """
        Centralized style defaults to match other SHAKEuq plotting patterns.
        """
        if font_sizes is None:
            font_sizes = {"title": 14, "labels": 12, "ticks": 10, "legend": 10}
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.4}
        if legend_kwargs is None:
            legend_kwargs = {"loc": "best", "fontsize": font_sizes.get("legend", 10), "frameon": True}
        return font_sizes, grid_kwargs, legend_kwargs
    

    
    def _uq_save_figure(
        self,
        fig,
        fname_stem: str,
        subdir: str = "",
        output_path: str = None,
        save_formats=("png", "pdf"),
        dpi: int = 300,
    ):
        """
        Save helper with strict UQ policy.
    
        Policy:
          - If output_path is provided:
              <output_path>/SHAKEuq/<event_id>/<subdir>/
          - Else:
              ./export/SHAKEuq/<event_id>/<subdir>/
        """
        from pathlib import Path
    
        # Determine base UQ directory
        if output_path:
            uq_root = Path(output_path).expanduser() / "SHAKEuq" / str(self.event_id) / "uq"
            uq_root.mkdir(parents=True, exist_ok=True)
        else:
            uq_root = self._uq_uqdir()
    
        out_dir = uq_root / str(subdir) if subdir else uq_root
        out_dir.mkdir(parents=True, exist_ok=True)
    
        for ext in list(save_formats):
            ext = str(ext).lstrip(".")
            fig.savefig(out_dir / f"{fname_stem}.{ext}", bbox_inches="tight", dpi=int(dpi))
    


    def _uq_cartopy_axes(
        self,
        fig=None,
        ax=None,
        figsize=(12, 6),
        add_ocean=True,
        add_borders=True,
        add_coastlines=True,
        add_gridlines=True,
        draw_labels=True,
    ):
        """
        Create OR style a Cartopy GeoAxes.
        - If fig+ax are None -> create new.
        - If fig+ax are provided -> only style, do not recreate.
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    
        if fig is None and ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
        # style existing axis
        if add_coastlines:
            try:
                ax.coastlines(zorder=10)
            except Exception:
                pass
        if add_borders:
            try:
                ax.add_feature(cfeature.BORDERS, zorder=10)
            except Exception:
                pass
        if add_ocean:
            try:
                ax.add_feature(cfeature.OCEAN, facecolor="skyblue", zorder=1)
            except Exception:
                pass
    
        if add_gridlines:
            try:
                gl = ax.gridlines(
                    crs=ccrs.PlateCarree(),
                    draw_labels=bool(draw_labels),
                    linewidth=1,
                    color="gray",
                    alpha=0.5,
                    linestyle="--",
                    zorder=20,
                )
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {"size": 10}
                gl.ylabel_style = {"size": 10}
            except Exception:
                pass
    
        return fig, ax


    
    def _uq_set_map_extent_from_axes(self, ax):
        """
        Set map extent to unified grid bounds.
        """
        import numpy as np
        import cartopy.crs as ccrs
    
        axes = self.uq_state["unified_axes"]
        lat2d = np.asarray(axes["lat2d"], dtype=float)
        lon2d = np.asarray(axes["lon2d"], dtype=float)
    
        lon_min = float(np.nanmin(lon2d))
        lon_max = float(np.nanmax(lon2d))
        lat_min = float(np.nanmin(lat2d))
        lat_max = float(np.nanmax(lat2d))
    
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    
    # ------------------------------------------------------------
    # 1) GLOBAL UNCERTAINTY DECAY (multi-curves; loader-safe)
    # ------------------------------------------------------------
    def uq_plot_uncertainty_decay(
        self,
        imt: str = "MMI",
        which: str = "total",           # "total"|"epistemic"
        method: str = "prior",          # "prior"|"bayesupdate"|"hierarchical"
        version_list=None,
        x_axis: str = "version",
        tae_hours=None,
        tae_seconds=None,
        xrotation: float = 45,
        show_title: bool = True,
        figsize=(12, 5),
        font_sizes: dict = None,
        grid: bool = True,
        grid_kwargs: dict = None,
        legend: bool = True,
        legend_kwargs: dict = None,
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        dpi: int = 300,
        show: bool = True,
        overlay_other_sigma: bool = True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
    
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet.")
    
        font_sizes, grid_kwargs, legend_kwargs = self._uq_plot_style_defaults(
            font_sizes=font_sizes, grid_kwargs=grid_kwargs, legend_kwargs=legend_kwargs
        )
    
        imt = str(imt)
        which = str(which).strip().lower()
        method = str(method).strip().lower()
    
        versions = self.uq_state["version_list"] if version_list is None else [int(v) for v in list(version_list)]
        x, xlabel = self._uq_make_x(versions, x_axis=x_axis, tae_hours=tae_hours, tae_seconds=tae_seconds)
    
        y_main, y_other = [], []
    
        for v in versions:
            uv = self.uq_state["versions_unified"].get(int(v), None)
            if uv is None:
                y_main.append(np.nan); y_other.append(np.nan)
                continue
    
            # --- prior sigma fields ---
            mu = np.asarray(uv["unified_mean"].get(imt, None), dtype=float)
            sig_total_prior_grid = np.asarray(
                uv["unified_sigma_prior_total"].get(imt, self._uq_default_sigma_for_imt(imt, mu)),
                dtype=float,
            )
            sig_total_prior, sig_a, sig_ep_prior = self._uq_decompose_sigma(
                sig_total_prior_grid, imt=imt, sigma_aleatory=None, sigma_total_from_shakemap=True
            )
    
            sig_total = sig_total_prior
            sig_ep = sig_ep_prior
    
            # --- posterior selection (ROBUST loaders) ---
            if method == "bayesupdate":
                dat = self._uq_load_bayes_posterior_npz(int(v), imt) if hasattr(self, "_uq_load_bayes_posterior_npz") else None
                if dat is not None:
                    sig_total = np.asarray(dat.get("sigma_total_post", sig_total), dtype=float)
                    sig_ep = np.asarray(dat.get("sigma_ep_post", sig_ep), dtype=float)
    
            elif method == "hierarchical":
                dat = self._uq_load_hierarchical_posterior_npz(imt) if hasattr(self, "_uq_load_hierarchical_posterior_npz") else None
                if dat is not None:
                    sig_total = np.asarray(dat.get("sigma_total_post", sig_total), dtype=float)
                    sig_ep = np.asarray(dat.get("sigma_ep_post", sig_ep), dtype=float)
    
            main_field = sig_total if which == "total" else sig_ep
            other_field = sig_ep if which == "total" else sig_total
    
            y_main.append(float(np.nanmean(main_field)))
            y_other.append(float(np.nanmean(other_field)))
    
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y_main, marker="o", label=f"Mean σ ({which})")
    
        if overlay_other_sigma:
            other_name = "epistemic" if which == "total" else "total"
            ax.plot(x, y_other, marker="s", linestyle="--", label=f"Mean σ ({other_name})")
    
        ax.set_xlabel(xlabel, fontsize=font_sizes["labels"])
        ax.set_ylabel("Mean σ", fontsize=font_sizes["labels"])
        ax.tick_params(axis="both", labelsize=font_sizes["ticks"])
        ax.tick_params(axis="x", labelrotation=xrotation)
    
        if grid:
            ax.grid(True, **grid_kwargs)
        if legend:
            ax.legend(**legend_kwargs)
    
        if show_title:
            ax.set_title(f"{method} uncertainty decay — {imt}", fontsize=font_sizes["title"])
    
        fig.tight_layout()
    
        if save:
            self._uq_save_figure(
                fig,
                fname_stem=f"{method}_uncertainty_decay_{imt}_{which}_overlay{int(bool(overlay_other_sigma))}",
                subdir="uq_plots",
                output_path=output_path,
                save_formats=save_formats,
                dpi=dpi,
            )
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig, ax
    
    
    # ------------------------------------------------------------
    # 2) AREA WHERE PoE >= cut (loader-safe)
    # ------------------------------------------------------------
    def uq_plot_metric_area_ge_threshold(
        self,
        imt: str = "MMI",
        threshold: float = 5.0,
        method: str = "prior",
        version_list=None,
        which_sigma: str = "total",
        x_axis: str = "version",
        tae_hours=None,
        tae_seconds=None,
        xrotation: float = 45,
        poe_cut: float = 0.5,
        show_title: bool = True,
        figsize=(12, 5),
        font_sizes: dict = None,
        grid: bool = True,
        grid_kwargs: dict = None,
        legend: bool = True,
        legend_kwargs: dict = None,
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        dpi: int = 300,
        show: bool = True,
        overlay_other_sigma: bool = True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
    
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet.")
    
        font_sizes, grid_kwargs, legend_kwargs = self._uq_plot_style_defaults(
            font_sizes=font_sizes, grid_kwargs=grid_kwargs, legend_kwargs=legend_kwargs
        )
    
        imt = str(imt)
        method = str(method).strip().lower()
        which_sigma = str(which_sigma).strip().lower()
    
        versions = self.uq_state["version_list"] if version_list is None else [int(v) for v in list(version_list)]
        x, xlabel = self._uq_make_x(versions, x_axis=x_axis, tae_hours=tae_hours, tae_seconds=tae_seconds)
    
        cell_area = float(self._uq_cell_area_km2())
    
        areas_main, areas_other = [], []
    
        # Set PoE context
        self._uq_poe_imt_context = imt
    
        for v in versions:
            uv = self.uq_state["versions_unified"].get(int(v), None)
            if uv is None:
                areas_main.append(np.nan); areas_other.append(np.nan)
                continue
    
            mu = np.asarray(uv["unified_mean"].get(imt, None), dtype=float)
    
            # prior sigma
            sig_total_prior_grid = np.asarray(
                uv["unified_sigma_prior_total"].get(imt, self._uq_default_sigma_for_imt(imt, mu)),
                dtype=float,
            )
            sig_total_prior, sig_a, sig_ep_prior = self._uq_decompose_sigma(
                sig_total_prior_grid, imt=imt, sigma_aleatory=None, sigma_total_from_shakemap=True
            )
    
            mu_use = mu
            sig_total = sig_total_prior
            sig_ep = sig_ep_prior
    
            if method == "bayesupdate":
                dat = self._uq_load_bayes_posterior_npz(int(v), imt) if hasattr(self, "_uq_load_bayes_posterior_npz") else None
                if dat is not None:
                    mu_use = np.asarray(dat.get("mu_post", mu_use), dtype=float)
                    sig_total = np.asarray(dat.get("sigma_total_post", sig_total), dtype=float)
                    sig_ep = np.asarray(dat.get("sigma_ep_post", sig_ep), dtype=float)
    
            elif method == "hierarchical":
                dat = self._uq_load_hierarchical_posterior_npz(imt) if hasattr(self, "_uq_load_hierarchical_posterior_npz") else None
                if dat is not None:
                    mu_use = np.asarray(dat.get("mu_post", mu_use), dtype=float)
                    sig_total = np.asarray(dat.get("sigma_total_post", sig_total), dtype=float)
                    sig_ep = np.asarray(dat.get("sigma_ep_post", sig_ep), dtype=float)
    
            main_sig = sig_total if which_sigma == "total" else sig_ep
            other_sig = sig_ep if which_sigma == "total" else sig_total
    
            poe_main = self._uq_poe_exceed(mu_use, main_sig, float(threshold))
            poe_other = self._uq_poe_exceed(mu_use, other_sig, float(threshold))
    
            areas_main.append(float(np.nansum(poe_main >= float(poe_cut)) * cell_area))
            areas_other.append(float(np.nansum(poe_other >= float(poe_cut)) * cell_area))
    
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, areas_main, marker="o", label=f"Area (PoE≥{poe_cut}) using σ({which_sigma})")
    
        if overlay_other_sigma:
            other_name = "epistemic" if which_sigma == "total" else "total"
            ax.plot(x, areas_other, marker="s", linestyle="--", label=f"Area (PoE≥{poe_cut}) using σ({other_name})")
    
        ax.set_xlabel(xlabel, fontsize=font_sizes["labels"])
        ax.set_ylabel("Area (km²)", fontsize=font_sizes["labels"])
        ax.tick_params(axis="both", labelsize=font_sizes["ticks"])
        ax.tick_params(axis="x", labelrotation=xrotation)
    
        if grid:
            ax.grid(True, **grid_kwargs)
        if legend:
            ax.legend(**legend_kwargs)
    
        if show_title:
            ax.set_title(f"{method} area where PoE≥{poe_cut} — {imt} thr={threshold:g}", fontsize=font_sizes["title"])
    
        fig.tight_layout()
    
        if save:
            self._uq_save_figure(
                fig,
                fname_stem=f"{method}_area_poe_ge_{poe_cut:g}_{imt}_thr{threshold:g}_{which_sigma}_overlay{int(bool(overlay_other_sigma))}",
                subdir="uq_plots",
                output_path=output_path,
                save_formats=save_formats,
                dpi=dpi,
            )
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig, ax
    
    
    # ------------------------------------------------------------
    # 3) MEAN PoE (loader-safe)
    # ------------------------------------------------------------
    def uq_plot_metric_mean_poe(
        self,
        imt: str = "MMI",
        threshold: float = 5.0,
        method: str = "prior",
        version_list=None,
        which_sigma: str = "total",
        x_axis: str = "version",
        tae_hours=None,
        tae_seconds=None,
        xrotation: float = 45,
        show_title: bool = True,
        figsize=(12, 5),
        font_sizes: dict = None,
        grid: bool = True,
        grid_kwargs: dict = None,
        legend: bool = True,
        legend_kwargs: dict = None,
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        dpi: int = 300,
        show: bool = True,
        overlay_other_sigma: bool = True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
    
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet.")
    
        font_sizes, grid_kwargs, legend_kwargs = self._uq_plot_style_defaults(
            font_sizes=font_sizes, grid_kwargs=grid_kwargs, legend_kwargs=legend_kwargs
        )
    
        imt = str(imt)
        method = str(method).strip().lower()
        which_sigma = str(which_sigma).strip().lower()
    
        versions = self.uq_state["version_list"] if version_list is None else [int(v) for v in list(version_list)]
        x, xlabel = self._uq_make_x(versions, x_axis=x_axis, tae_hours=tae_hours, tae_seconds=tae_seconds)
    
        mpoe_main, mpoe_other = [], []
    
        self._uq_poe_imt_context = imt
    
        for v in versions:
            uv = self.uq_state["versions_unified"].get(int(v), None)
            if uv is None:
                mpoe_main.append(np.nan); mpoe_other.append(np.nan)
                continue
    
            mu = np.asarray(uv["unified_mean"].get(imt, None), dtype=float)
            sig_total_prior_grid = np.asarray(
                uv["unified_sigma_prior_total"].get(imt, self._uq_default_sigma_for_imt(imt, mu)),
                dtype=float,
            )
            sig_total_prior, sig_a, sig_ep_prior = self._uq_decompose_sigma(
                sig_total_prior_grid, imt=imt, sigma_aleatory=None, sigma_total_from_shakemap=True
            )
    
            mu_use = mu
            sig_total = sig_total_prior
            sig_ep = sig_ep_prior
    
            if method == "bayesupdate":
                dat = self._uq_load_bayes_posterior_npz(int(v), imt) if hasattr(self, "_uq_load_bayes_posterior_npz") else None
                if dat is not None:
                    mu_use = np.asarray(dat.get("mu_post", mu_use), dtype=float)
                    sig_total = np.asarray(dat.get("sigma_total_post", sig_total), dtype=float)
                    sig_ep = np.asarray(dat.get("sigma_ep_post", sig_ep), dtype=float)
    
            elif method == "hierarchical":
                dat = self._uq_load_hierarchical_posterior_npz(imt) if hasattr(self, "_uq_load_hierarchical_posterior_npz") else None
                if dat is not None:
                    mu_use = np.asarray(dat.get("mu_post", mu_use), dtype=float)
                    sig_total = np.asarray(dat.get("sigma_total_post", sig_total), dtype=float)
                    sig_ep = np.asarray(dat.get("sigma_ep_post", sig_ep), dtype=float)
    
            main_sig = sig_total if which_sigma == "total" else sig_ep
            other_sig = sig_ep if which_sigma == "total" else sig_total
    
            poe_main = self._uq_poe_exceed(mu_use, main_sig, float(threshold))
            poe_other = self._uq_poe_exceed(mu_use, other_sig, float(threshold))
    
            mpoe_main.append(float(np.nanmean(poe_main)))
            mpoe_other.append(float(np.nanmean(poe_other)))
    
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, mpoe_main, marker="o", label=f"Mean PoE using σ({which_sigma})")
    
        if overlay_other_sigma:
            other_name = "epistemic" if which_sigma == "total" else "total"
            ax.plot(x, mpoe_other, marker="s", linestyle="--", label=f"Mean PoE using σ({other_name})")
    
        ax.set_xlabel(xlabel, fontsize=font_sizes["labels"])
        ax.set_ylabel("Mean PoE", fontsize=font_sizes["labels"])
        ax.tick_params(axis="both", labelsize=font_sizes["ticks"])
        ax.tick_params(axis="x", labelrotation=xrotation)
    
        if grid:
            ax.grid(True, **grid_kwargs)
        if legend:
            ax.legend(**legend_kwargs)
    
        if show_title:
            ax.set_title(f"{method} mean PoE — {imt} thr={threshold:g}", fontsize=font_sizes["title"])
    
        fig.tight_layout()
    
        if save:
            self._uq_save_figure(
                fig,
                fname_stem=f"{method}_mean_poe_{imt}_thr{threshold:g}_{which_sigma}_overlay{int(bool(overlay_other_sigma))}",
                subdir="uq_plots",
                output_path=output_path,
                save_formats=save_formats,
                dpi=dpi,
            )
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig, ax
    
    
    # ------------------------------------------------------------
    # 4) LOCAL UNCERTAINTY DECAY (already path-safe via uq_compute_local_metrics)
    # ------------------------------------------------------------
    def uq_plot_local_uncertainty_decay(
        self,
        imt: str = "MMI",
        method: str = "bayesupdate",
        which_sigma: str = "epistemic",
        station_radius_km: float = 40.0,
        poe_threshold: float = 5.0,
        poe_cut: float = 0.5,
        version_list=None,
        x_axis: str = "version",
        tae_hours=None,
        tae_seconds=None,
        xrotation: float = 45,
        show_title: bool = True,
        figsize=(12, 5),
        font_sizes: dict = None,
        grid: bool = True,
        grid_kwargs: dict = None,
        legend: bool = True,
        legend_kwargs: dict = None,
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        dpi: int = 300,
        show: bool = True,
        include_prior: bool = False,
    ):
        import matplotlib.pyplot as plt
    
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet.")
    
        font_sizes, grid_kwargs, legend_kwargs = self._uq_plot_style_defaults(
            font_sizes=font_sizes, grid_kwargs=grid_kwargs, legend_kwargs=legend_kwargs
        )
    
        metrics = self.uq_compute_local_metrics(
            imt=imt,
            method=method,
            which_sigma=which_sigma,
            station_radius_km=station_radius_km,
            poe_threshold=poe_threshold,
            poe_cut=poe_cut,
            version_list=version_list,
            export=True,
        )
    
        rows = metrics["rows"]
        versions = [r["version"] for r in rows]
        x, xlabel = self._uq_make_x(versions, x_axis=x_axis, tae_hours=tae_hours, tae_seconds=tae_seconds)
    
        y_post_global = [r["mean_sigma_post_global"] for r in rows]
        y_post_station = [r["mean_sigma_post_near_stations"] for r in rows]
        y_post_poe = [r["mean_sigma_post_poe"] for r in rows]
    
        fig, ax = plt.subplots(figsize=figsize)
    
        ax.plot(x, y_post_global, marker="o", label="Posterior: global mean σ")
        ax.plot(x, y_post_station, marker="s", linestyle="--", label=f"Posterior: within {station_radius_km:g} km of obs")
        ax.plot(x, y_post_poe, marker="^", linestyle="-.", label=f"Posterior: PoE≥{poe_cut:g} (thr={poe_threshold:g})")
    
        if include_prior:
            y_prior_global = [r["mean_sigma_prior_global"] for r in rows]
            ax.plot(x, y_prior_global, linestyle=":", label="Prior: global mean σ")
    
        ax.set_xlabel(xlabel, fontsize=font_sizes["labels"])
        ax.set_ylabel(f"Mean σ ({which_sigma})", fontsize=font_sizes["labels"])
        ax.tick_params(axis="both", labelsize=font_sizes["ticks"])
        ax.tick_params(axis="x", labelrotation=xrotation)
    
        if grid:
            ax.grid(True, **grid_kwargs)
        if legend:
            ax.legend(**legend_kwargs)
    
        if show_title:
            ax.set_title(f"{method} local uncertainty decay — {imt}", fontsize=font_sizes["title"])
    
        fig.tight_layout()
    
        if save:
            self._uq_save_figure(
                fig,
                fname_stem=f"{method}_local_uncertainty_decay_{which_sigma}_{imt}_R{station_radius_km:g}_thr{poe_threshold:g}_cut{poe_cut:g}_prior{int(bool(include_prior))}",
                subdir="uq_plots/local_decay",
                output_path=output_path,
                save_formats=save_formats,
                dpi=dpi,
            )
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig, ax
    
    
    # ------------------------------------------------------------
    # 5) SIGMA REDUCTION MAP (already uses your core helpers)
    # (No changes needed here unless you want symmetric color scaling)
    # ------------------------------------------------------------
    
    
    # ------------------------------------------------------------
    # 6) BATCH EXPORT SIGMA REDUCTION MAPS (FIXED: no recursion)
    # ------------------------------------------------------------
    def uq_export_sigma_reduction_maps(
        self,
        imt: str = "MMI",
        which_sigma: str = "epistemic",
        method: str = "bayesupdate",
        version_list=None,
        cmap: str = "viridis",
        vmin=None,
        vmax=None,
        output_path: str = None,
        save_formats=("png", "pdf"),
        dpi: int = 300,
        figsize=(10, 8),
        show_obs: bool = True,
        obs_size: float = 12.0,
        plot_colorbar: bool = True,
    ):
        """
        Export sigma reduction maps for all versions.
        FIXED: calls uq_plot_sigma_reduction_map(), not itself.
        """
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet.")
    
        imt = str(imt)
        which = str(which_sigma).strip().lower()
        method = str(method).strip().lower()
    
        versions = self.uq_state["version_list"] if version_list is None else [int(v) for v in list(version_list)]
    
        for v in versions:
            self.uq_plot_sigma_reduction_map(
                version=int(v),
                imt=imt,
                which_sigma=which,
                method=method,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                figsize=figsize,
                show_obs=show_obs,
                obs_size=obs_size,
                output_path=output_path,
                save=True,
                save_formats=save_formats,
                dpi=dpi,
                show=False,
                plot_colorbar=plot_colorbar,
            )
    
        return True
    
    
    # ------------------------------------------------------------
    # VERSION-CHANGE + DECOMPOSITION (only small fix: cartopy styling call)
    # (Your compute + plot functions are OK; decomposition now uses safe styling.)
    # ------------------------------------------------------------
    def uq_plot_net_uncertainty_decomposition_first_to_final(
        self,
        imt: str = "MMI",
        which_sigma: str = "epistemic",
        posterior_method: str = "bayesupdate",
        first_version: int = None,
        final_version: int = None,
        cmap: str = "viridis",
        vmin=None,
        vmax=None,
        figsize=(14, 9),
        output_path: str = None,
        save: bool = True,
        save_formats=("png", "pdf"),
        dpi: int = 300,
        show: bool = True,
    ):
        """
        One figure with 3 panels (Cartopy) summarizing the decomposition:
          Panel 1: Δσ_post
          Panel 2: Δσ_prior
          Panel 3: Δreduction
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    
        dat = self.uq_compute_uncertainty_change_first_to_final(
            imt=imt,
            which_sigma=which_sigma,
            posterior_method=posterior_method,
            first_version=first_version,
            final_version=final_version,
            export=True,
            output_path=output_path,
        )
    
        Z1 = np.asarray(dat["delta_sigma_post"], dtype=float)
        Z2 = np.asarray(dat["delta_sigma_prior"], dtype=float)
        Z3 = np.asarray(dat["delta_reduction"], dtype=float)
    
        axes = self.uq_state["unified_axes"]
        LAT2 = np.asarray(axes["lat2d"], dtype=float)
        LON2 = np.asarray(axes["lon2d"], dtype=float)
    
        fig = plt.figure(figsize=figsize)
    
        ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
        ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
    
        # style axes safely
        for ax in (ax1, ax2, ax3):
            self._uq_cartopy_axes(fig=fig, ax=ax)  # ignore figsize when styling
            self._uq_set_map_extent_from_axes(ax)
    
        pm1 = ax1.pcolormesh(LON2, LAT2, Z1, transform=ccrs.PlateCarree(), cmap=str(cmap), vmin=vmin, vmax=vmax, shading="auto", zorder=5)
        pm2 = ax2.pcolormesh(LON2, LAT2, Z2, transform=ccrs.PlateCarree(), cmap=str(cmap), vmin=vmin, vmax=vmax, shading="auto", zorder=5)
        pm3 = ax3.pcolormesh(LON2, LAT2, Z3, transform=ccrs.PlateCarree(), cmap=str(cmap), vmin=vmin, vmax=vmax, shading="auto", zorder=5)
    
        ax1.set_title("Net posterior change\nΔσ_post = σ_post(v1) − σ_post(vN)", fontsize=12)
        ax2.set_title("Model/procedure change\nΔσ_prior = σ_prior(v1) − σ_prior(vN)", fontsize=12)
        ax3.set_title("Data/update-effect change\nΔreduction = red(vN) − red(v1)", fontsize=12)
    
        cbar = fig.colorbar(pm1, ax=[ax1, ax2, ax3], shrink=0.75, pad=0.02)
        cbar.set_label("Positive = decrease in σ (or increase in reduction)")
    
        fig.suptitle(
            f"{posterior_method} uncertainty decomposition — {imt} ({which_sigma}) — v{dat['first_version']}→v{dat['final_version']}",
            fontsize=14,
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    
        if save:
            self._uq_save_figure(
                fig,
                fname_stem=f"{posterior_method}_uncertainty_decomposition_{which_sigma}_{imt}_v{dat['first_version']}_to_v{dat['final_version']}",
                subdir="uq_plots/version_change",
                output_path=output_path,
                save_formats=save_formats,
                dpi=dpi,
            )
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig
    
    
    


    
    # ======================================================================
    # PATCH 4 (FULL): Target-based time-evolving prediction + uncertainty comparison
    #   - Compare ShakeMap-published (prediction + RAW sigma_total from uncertainty.xml)
    #     vs alternative methods predicted given observations available at each version.
    #   - Targets: point(s) / area(s) / global
    #   - Area/global: add ShakeMap-only min/max envelopes (shaded) for published curves
    #   - NEW: Raw XML vs Unified-grid curve comparison (point/area; with envelopes)
    #   - Robust wrappers for differing internal signatures
    #   - Robust audit export (CSV + JSON)
    #
    # Drop this whole block INSIDE the SHAKEuq class (replace the whole section).
    # Relies on imports already present in SHAKEuq.py: numpy as np, pandas as pd,
    # matplotlib.pyplot as plt, Path, os, xml.etree.ElementTree as ET, etc.
    # ======================================================================
    
    # ----------------------------------------------------------------------
    # 0) Signature-robust wrappers (mu + sigma + obs conversions)
    # ----------------------------------------------------------------------
    def _uq_mu_to_ws(self, imt, mu_linear):
        """Robust wrapper for _uq_mu_to_working_space (supports both arg orders)."""
        if not hasattr(self, "_uq_mu_to_working_space"):
            return np.asarray(mu_linear, dtype=float)
        fn = self._uq_mu_to_working_space
        try:
            return fn(mu_linear, imt)
        except Exception:
            return fn(imt, mu_linear)
    
    
    def _uq_mu_from_ws(self, imt, mu_working):
        """Robust wrapper for _uq_mu_from_working_space (supports both arg orders)."""
        if not hasattr(self, "_uq_mu_from_working_space"):
            return np.asarray(mu_working, dtype=float)
        fn = self._uq_mu_from_working_space
        try:
            return fn(mu_working, imt)
        except Exception:
            return fn(imt, mu_working)
    
    
    def _uq_obs_to_ws(self, imt, y_linear):
        """Robust wrapper for _uq_obs_to_working_space (supports both arg orders)."""
        if not hasattr(self, "_uq_obs_to_working_space"):
            return float(y_linear)
        fn = self._uq_obs_to_working_space
        try:
            return float(fn(y_linear, imt))
        except Exception:
            return float(fn(imt, y_linear))
    
    
    def _uq_decompose_sigma_safe(
        self,
        imt,
        sigma_prior_total,
        sigma_aleatory=None,
        sigma_total_from_shakemap=True,
    ):
        """
        Robust wrapper for _uq_decompose_sigma supporting both signatures:
          - _uq_decompose_sigma(sigma_prior_total, imt, ...)  [current SHAKEuq]
          - _uq_decompose_sigma(imt, sigma_prior_total, ...)  [older drafts]
    
        IMPORTANT:
        - ShakeMap plotted sigma_total is RAW from uncertainty.xml, NOT decomposed.
        - This decomposition is used ONLY for update-method bookkeeping (sigma_a + sigma_ep).
        Returns: (sigma_total, sigma_aleatory, sigma_epistemic)
        """
        def _as_grid(x, like):
            like = np.asarray(like, dtype=float)
            x = np.asarray(x, dtype=float)
            if x.ndim == 0:
                return np.full_like(like, float(x))
            if x.size == 1 and like.size > 1:
                return np.full_like(like, float(x.ravel()[0]))
            if x.shape != like.shape and x.size == like.size:
                return x.reshape(like.shape)
            return x
    
        if not hasattr(self, "_uq_decompose_sigma"):
            sig_tot = np.asarray(sigma_prior_total, dtype=float)
            if sigma_aleatory is None:
                sig_a = np.zeros_like(sig_tot)
            else:
                sig_a = np.full_like(sig_tot, float(sigma_aleatory))
            sig_ep = np.sqrt(np.maximum(0.0, sig_tot**2 - sig_a**2))
            return sig_tot, sig_a, sig_ep
    
        fn = self._uq_decompose_sigma
        try:
            sig_tot, sig_a, sig_ep = fn(
                sigma_prior_total, imt,
                sigma_aleatory=sigma_aleatory,
                sigma_total_from_shakemap=bool(sigma_total_from_shakemap),
            )
        except Exception:
            sig_tot, sig_a, sig_ep = fn(
                imt, sigma_prior_total,
                sigma_aleatory=sigma_aleatory,
                sigma_total_from_shakemap=bool(sigma_total_from_shakemap),
            )
    
        sig_tot = np.asarray(sig_tot, dtype=float)
        sig_a = _as_grid(sig_a, sig_tot)
        sig_ep = _as_grid(sig_ep, sig_tot)
    
        sig_a = np.clip(sig_a, 0.0, np.inf)
        sig_ep = np.clip(sig_ep, 0.0, np.inf)
        return sig_tot, sig_a, sig_ep

    
    
    # ----------------------------------------------------------------------
    # 1) Geometry helpers
    # ----------------------------------------------------------------------
    def _uq_haversine_km(self, lat1, lon1, lat2, lon2):
        """Vectorized haversine distance in km (lat/lon degrees)."""
        R = 6371.0
        lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
        lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return R * c
    
    
    # ----------------------------------------------------------------------
    # 2) Target parsing
    # ----------------------------------------------------------------------
    def _uq_parse_targets(self, points=None, areas=None, allow_empty=False):
        """
        Normalize targets into list[dict]:
          - points: (lat,lon) or list of tuples, or list of dicts {"id","lat","lon"}
          - areas: dict or list of dicts:
              circle: {"id","kind":"circle","lat","lon","radius_km"}
              bbox:   {"id","kind":"bbox","minlat","minlon","maxlat","maxlon"}
        """
        targets = []
    
        if points is not None:
            if isinstance(points, tuple) and len(points) == 2:
                points = [points]
            if isinstance(points, dict):
                points = [points]
            if isinstance(points, list):
                for i, p in enumerate(points):
                    if isinstance(p, dict):
                        tid = p.get("id", f"pt_{i+1}")
                        targets.append({"id": tid, "type": "point", "lat": float(p["lat"]), "lon": float(p["lon"])})
                    else:
                        lat, lon = p
                        targets.append({"id": f"pt_{i+1}", "type": "point", "lat": float(lat), "lon": float(lon)})
    
        if areas is not None:
            if isinstance(areas, dict):
                areas = [areas]
            for i, a in enumerate(areas):
                if not isinstance(a, dict):
                    raise ValueError("areas must be dict or list[dict].")
                kind = str(a.get("kind", "")).lower().strip()
                tid = a.get("id", f"area_{i+1}")
                if kind == "circle":
                    targets.append({
                        "id": tid, "type": "area", "kind": "circle",
                        "lat": float(a["lat"]), "lon": float(a["lon"]),
                        "radius_km": float(a["radius_km"]),
                    })
                elif kind == "bbox":
                    targets.append({
                        "id": tid, "type": "area", "kind": "bbox",
                        "minlat": float(a["minlat"]), "minlon": float(a["minlon"]),
                        "maxlat": float(a["maxlat"]), "maxlon": float(a["maxlon"]),
                    })
                else:
                    raise ValueError('Area kind must be "circle" or "bbox".')
    
        if not targets and (not allow_empty):
            raise ValueError("No targets provided. Provide points and/or areas.")
    
        seen = set()
        for t in targets:
            if t["id"] in seen:
                raise ValueError(f'Duplicate target id="{t["id"]}".')
            seen.add(t["id"])
        return targets
    
    
    # ----------------------------------------------------------------------
    # 3) Unified grid access
    # ----------------------------------------------------------------------
    def _uq_get_unified_for_versions(self, version_list, imt="MMI", grid_res=None, interp_method="nearest", interp_kwargs=None):
        """
        Ensure unified axes exist. Prefer uq_state built by uq_build_dataset.
        Returns (df_or_None, lat2d, lon2d)
        """
        if interp_kwargs is None:
            interp_kwargs = {}
    
        versions = [int(v) for v in (version_list or [])]
        if not versions:
            raise ValueError("version_list is empty.")
    
        # Prefer uq_state (fast, no file reads)
        if hasattr(self, "uq_state") and self.uq_state is not None:
            st = self.uq_state
            lat2d = None
            lon2d = None
            if isinstance(st, dict):
                lat2d = st.get("lat2d", None) or st.get("unified_axes", {}).get("lat2d", None)
                lon2d = st.get("lon2d", None) or st.get("unified_axes", {}).get("lon2d", None)
            if lat2d is not None and lon2d is not None and "versions_unified" in st:
                return None, np.asarray(lat2d), np.asarray(lon2d)
    
        # Fallback: build via get_unified_grid (may trigger file reading internally)
        df = self.get_unified_grid(
            version_list=versions,
            metric=imt,
            grid_res=grid_res,
            interp_method=interp_method,
            interp_kwargs=interp_kwargs,
            use_cache=True,
        )
    
        lat2d = getattr(self, "_unified_lat2d", None)
        lon2d = getattr(self, "_unified_lon2d", None)
        if lat2d is None or lon2d is None:
            if isinstance(df, pd.DataFrame) and ("lat" in df.columns) and ("lon" in df.columns):
                lats = np.sort(df["lat"].unique())
                lons = np.sort(df["lon"].unique())
                lon2d, lat2d = np.meshgrid(lons, lats)
            else:
                raise RuntimeError("Could not infer unified lat/lon grids.")
        return df, np.asarray(lat2d), np.asarray(lon2d)
    
    
    def _uq_target_mask(self, target, lat2d, lon2d):
        """
        Return (mask2d, meta) for selecting grid cells for point/area targets.
        - point: nearest cell (one True)
        - area circle/bbox: all cells inside
        """
        if target.get("type") == "point":
            d = self._uq_haversine_km(target["lat"], target["lon"], lat2d, lon2d)
            ij = np.unravel_index(np.nanargmin(d), d.shape)
            mask = np.zeros(lat2d.shape, dtype=bool)
            mask[ij] = True
            meta = {"kind": "nearest_cell", "ij": (int(ij[0]), int(ij[1])), "n_cells": 1, "min_dist_km": float(d[ij])}
            return mask, meta
    
        kind = target.get("kind", "").lower()
        if kind == "circle":
            d = self._uq_haversine_km(target["lat"], target["lon"], lat2d, lon2d)
            mask = d <= float(target["radius_km"])
            meta = {"kind": "circle", "n_cells": int(mask.sum()), "radius_km": float(target["radius_km"])}
            return mask, meta
    
        if kind == "bbox":
            mask = (
                (lat2d >= float(target["minlat"])) & (lat2d <= float(target["maxlat"])) &
                (lon2d >= float(target["minlon"])) & (lon2d <= float(target["maxlon"]))
            )
            meta = {"kind": "bbox", "n_cells": int(mask.sum())}
            return mask, meta
    
        raise ValueError(f"Unknown target kind: {target}")
    
    
    def _uq_agg(self, values, agg="mean"):
        """Aggregate with nan safety."""
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan")
        a = str(agg).lower().strip()
        if a == "mean":
            return float(np.nanmean(v))
        if a == "median":
            return float(np.nanmedian(v))
        if a in ("p05", "p5"):
            return float(np.nanpercentile(v, 5.0))
        if a in ("p95", "p95.0"):
            return float(np.nanpercentile(v, 95.0))
        if a == "min":
            return float(np.nanmin(v))
        if a == "max":
            return float(np.nanmax(v))
        raise ValueError(f"Unknown agg={agg}")
    
    
    def _uq_minmax(self, values):
        """Return (nanmin, nanmax) with safety."""
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan"), float("nan")
        return float(np.nanmin(v)), float(np.nanmax(v))
    
    
    def _uq_get_mu_sigma_unified(self, version, imt, lat2d, lon2d, grid_res=None, interp_method="nearest", interp_kwargs=None):
        """
        Retrieve (mu_linear, sigma_total_raw) on unified grid for given version/imt.
        - sigma_total_raw = the uncertainty grid values as stored (from uncertainty.xml)
        Prefer uq_state built by uq_build_dataset.
        """
        if interp_kwargs is None:
            interp_kwargs = {}
    
        imtU = str(imt).upper()
        v = int(version)
    
        if hasattr(self, "uq_state") and isinstance(self.uq_state, dict) and "versions_unified" in self.uq_state:
            vu = self.uq_state["versions_unified"].get(v, None)
            if vu is not None:
                mu = None
                sig = None
                if isinstance(vu.get("unified_mean", {}), dict):
                    mu = vu["unified_mean"].get(imtU, None)
                if isinstance(vu.get("unified_sigma_prior_total", {}), dict):
                    # stored as sigma_total for this imt
                    sig = vu["unified_sigma_prior_total"].get(imtU, None)
                if mu is not None and sig is not None:
                    return np.asarray(mu, dtype=float), np.asarray(sig, dtype=float)
    
        # fallback: call get_unified_grid for just this version
        df = self.get_unified_grid(
            version_list=[v],
            metric=imtU,
            grid_res=grid_res,
            interp_method=interp_method,
            interp_kwargs=interp_kwargs,
            use_cache=False,
        )
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("get_unified_grid did not return a DataFrame; build uq_state via uq_build_dataset for reliability.")
    
        if ("lat" not in df.columns) or ("lon" not in df.columns):
            raise RuntimeError("Unified DataFrame missing lat/lon columns.")
    
        # Try to infer columns
        mu_col = None
        sig_col = None
        imtL = imtU.lower()
    
        for c in df.columns:
            cl = c.lower()
            if (f"v{v}" in cl) and ("sigma" in cl or "std" in cl or "unc" in cl):
                sig_col = c
            if (f"v{v}" in cl) and (("mean" in cl) or (cl.endswith(imtL)) or (imtL in cl and "sigma" not in cl and "std" not in cl)):
                mu_col = c
    
        if mu_col is None or sig_col is None:
            raise RuntimeError(
                "Could not infer unified mean/sigma columns from unified DataFrame. "
                "Build uq_state first via uq_build_dataset(...) for reliability."
            )
    
        lats = np.sort(df["lat"].unique())
        lons = np.sort(df["lon"].unique())
        lon2d2, lat2d2 = np.meshgrid(lons, lats)
        if lat2d2.shape != lat2d.shape:
            lat2d = lat2d2
            lon2d = lon2d2
    
        mu = df.pivot(index="lat", columns="lon", values=mu_col).values
        sig = df.pivot(index="lat", columns="lon", values=sig_col).values
        return np.asarray(mu, dtype=float), np.asarray(sig, dtype=float)
    

    
    
    def _uq_hierarchical_posterior_at_mask(
        self,
        mu0_ws,
        sigma_ep0,
        sigma_a,
        lat2d,
        lon2d,
        target_mask,
        obs_list,
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
    ):
        """
        Lightweight hierarchical-style update:
          1) estimate global bias from obs residuals (nearest prior at obs)
          2) shift prior mean by bias
          3) run local Bayes update
        """
        mu0_ws = np.asarray(mu0_ws, dtype=float)
        if not obs_list:
            return np.full_like(mu0_ws, np.nan, dtype=float), np.full_like(mu0_ws, np.nan, dtype=float)
    
        num = 0.0
        den = 0.0
        for ob in obs_list:
            d = self._uq_haversine_km(ob["lat"], ob["lon"], lat2d, lon2d)
            ij = np.unravel_index(np.nanargmin(d), d.shape)
            mu_at = float(mu0_ws[ij])
            r = float(ob["y_ws"]) - mu_at
            w = float(ob.get("w", 1.0))
            num += w * r
            den += w
        delta = num / den if den > 0 else 0.0
    
        mu0_shift = mu0_ws + float(delta)
        return self._uq_bayes_local_posterior_at_mask(
            mu0_shift, sigma_ep0, sigma_a,
            lat2d, lon2d, target_mask, obs_list,
            update_radius_km=update_radius_km,
            kernel=kernel,
            kernel_scale_km=kernel_scale_km,
            measurement_sigma=measurement_sigma,
        )
    
    
    def _uq_ok_residual_posterior_at_mask(
        self,
        mu0_ws,
        sigma_ep0,
        sigma_a,
        lat2d,
        lon2d,
        target_mask,
        obs_list,
        variogram="exponential",
        range_km=60.0,
        nugget=1e-6,
        sill=None,
        measurement_sigma=0.30,
        sigma_ep_cap_to_prior=True,
    ):
        """
        Ordinary kriging of residuals in working space:
          r_i = y_i - mu0(x_i)
          mu_post = mu0 + r_hat
          sigma_ep_post ~ sqrt(kriging_var) (capped to prior ep by default)
        """
        mu0_ws = np.asarray(mu0_ws, dtype=float)
        sigma_ep0 = np.asarray(sigma_ep0, dtype=float)
    
        mu_post = np.full(mu0_ws.shape, np.nan, dtype=float)
        sig_ep_post = np.full(mu0_ws.shape, np.nan, dtype=float)
    
        m = target_mask.astype(bool)
        if (not m.any()) or (not obs_list):
            return mu_post, sig_ep_post
    
        lats = np.array([o["lat"] for o in obs_list], dtype=float)
        lons = np.array([o["lon"] for o in obs_list], dtype=float)
        y = np.array([o["y_ws"] for o in obs_list], dtype=float)
        w_obs = np.array([float(o.get("w", 1.0)) for o in obs_list], dtype=float)
    
        r = np.zeros_like(y)
        for i in range(len(y)):
            d = self._uq_haversine_km(lats[i], lons[i], lat2d, lon2d)
            ij = np.unravel_index(np.nanargmin(d), d.shape)
            r[i] = y[i] - float(mu0_ws[ij])
    
        if sill is None:
            ww = np.clip(w_obs, 0.0, np.inf)
            if ww.sum() > 0:
                rm = (ww * r).sum() / ww.sum()
                sill = float((ww * (r - rm) ** 2).sum() / max(ww.sum(), 1.0))
            else:
                sill = float(np.nanvar(r))
            sill = max(sill, 1e-6)
        else:
            sill = float(sill)
    
        def cov(h):
            h = np.asarray(h, dtype=float)
            a = max(float(range_km), 1e-6)
            if str(variogram).lower().strip() in ("exp", "exponential"):
                return sill * np.exp(-h / a)
            return sill * np.exp(-h / a)
    
        n = len(r)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            D[i, :] = self._uq_haversine_km(lats[i], lons[i], lats, lons)
    
        meas_var = float(measurement_sigma) ** 2
        C = cov(D)
        C = C + np.eye(n) * (float(nugget) + meas_var)
    
        A = np.zeros((n + 1, n + 1), dtype=float)
        A[:n, :n] = C
        A[:n, n] = 1.0
        A[n, :n] = 1.0
    
        idx = np.argwhere(m)
        for (ii, jj) in idx:
            lat_t = float(lat2d[ii, jj])
            lon_t = float(lon2d[ii, jj])
            d_to = self._uq_haversine_km(lat_t, lon_t, lats, lons)
            c = cov(d_to)
    
            b = np.zeros(n + 1, dtype=float)
            b[:n] = c
            b[n] = 1.0
    
            try:
                sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                mu_post[ii, jj] = mu0_ws[ii, jj]
                sig_ep_post[ii, jj] = sigma_ep0[ii, jj]
                continue
    
            w = sol[:n]
            lam = sol[n]
    
            rhat = float(np.dot(w, r))
            var_k = float(sill - np.dot(w, c) + lam)
            var_k = max(var_k, 0.0)
    
            mu_post[ii, jj] = mu0_ws[ii, jj] + rhat
            sig = float(np.sqrt(var_k))
            if sigma_ep_cap_to_prior:
                sig = min(sig, float(sigma_ep0[ii, jj]))
            sig_ep_post[ii, jj] = sig
    
        return mu_post, sig_ep_post
    
    
    # ----------------------------------------------------------------------
    # 6) Audit utilities
    # ----------------------------------------------------------------------
    def _uq_write_audit(self, audit_rows, output_path=None, prefix="UQ-TargetAudit"):
        """
        Write audit CSV + JSON following strict UQ path policy.
    
        Policy:
          - If output_path provided:
              <output_path>/SHAKEuq/<event_id>/uq_audit/
          - Else:
              ./export/SHAKEuq/<event_id>/uq_audit/
        """
        from pathlib import Path
        import pandas as pd
    
        try:
            if output_path:
                uq_root = Path(output_path).expanduser() / "SHAKEuq" / str(self.event_id) / "uq"
            else:
                uq_root = self._uq_uqdir()
    
            outp = uq_root / "uq_audit"
            outp.mkdir(parents=True, exist_ok=True)
    
            df = pd.DataFrame(audit_rows)
            csv_path = outp / f"{prefix}.csv"
            json_path = outp / f"{prefix}.json"
            df.to_csv(csv_path, index=False)
            df.to_json(json_path, orient="records", indent=2)
            return str(csv_path), str(json_path)
        except Exception:
            return None, None
    
        
    def _uq_save_figure_safe(
        self,
        fig,
        fname_stem,
        subdir=None,
        output_path=None,
        save_formats=("png", "pdf"),
        dpi=300
    ):
        """
        Safe figure save that always follows UQ path policy.
    
        Policy:
          - If output_path provided:
              <output_path>/SHAKEuq/<event_id>/<subdir>/
          - Else:
              ./export/SHAKEuq/<event_id>/<subdir>/
        """
        # Always prefer the canonical saver
        if hasattr(self, "_uq_save_figure"):
            try:
                return self._uq_save_figure(
                    fig,
                    fname_stem=fname_stem,
                    subdir=subdir or "",
                    output_path=output_path,
                    save_formats=save_formats,
                    dpi=dpi,
                )
            except Exception:
                # fall through to strict-policy fallback below
                pass
    
        # Strict-policy fallback (no direct saving to bare output_path)
        from pathlib import Path
    
        if output_path:
            uq_root = Path(output_path).expanduser() / "SHAKEuq" / str(self.event_id) / "uq"
        else:
            uq_root = self._uq_uqdir()
    
        out_dir = uq_root / str(subdir) if subdir else uq_root
        out_dir.mkdir(parents=True, exist_ok=True)
    
        for ext in save_formats:
            ext = str(ext).lstrip(".")
            fig.savefig(out_dir / f"{fname_stem}.{ext}", dpi=dpi, bbox_inches="tight")
    

    # ----------------------------------------------------------------------
    # 7) RAW ShakeMap XML extraction (baseline truth for validation)
    # ----------------------------------------------------------------------
    def _uq_find_shakemap_xml_legacy(self, version, which="grid", base_folder=None, shakemap_folder=None):
        v = int(version)
        which = str(which).lower().strip()
        if which not in ("grid", "uncertainty"):
            raise ValueError('which must be "grid" or "uncertainty".')
    
        patterns = ["grid.xml"] if which == "grid" else ["uncertainty.xml"]
        # also accept alternate suffixes
        patterns += ["*grid.xml", "*_grid.xml"] if which == "grid" else ["*uncertainty.xml", "*_uncertainty.xml"]
    
        search_roots = []
        if base_folder:
            search_roots.append(Path(base_folder))
        if shakemap_folder:
            search_roots.append(Path(shakemap_folder))
    
        if hasattr(self, "uq_state") and isinstance(self.uq_state, dict):
            bf = self.uq_state.get("base_folder", None)
            if bf:
                search_roots.insert(0, Path(bf) / f"v{v}")
                search_roots.insert(0, Path(bf) / f"v{str(v).zfill(2)}")
                search_roots.insert(0, Path(bf))
    
        # Search best match
        for root in search_roots:
            try:
                if not root.exists():
                    continue
                hits = []
                for pat in patterns:
                    hits.extend(list(root.rglob(pat)))
                if not hits:
                    continue
                scored = []
                for p in hits:
                    name = p.name.lower()
                    score = 0
                    if f"v{v}" in name or f"v{str(v).zfill(2)}" in name:
                        score += 3
                    if which in name:
                        score += 1
                    scored.append((score, len(str(p)), p))
                scored.sort(key=lambda x: (-x[0], x[1]))
                return scored[0][2]
            except Exception:
                continue
    
        return None
    
    
    
    def _uq_raw_mask_on_xml_grid(self, G, target):
        """
        Build mask on RAW XML grid (rectilinear) for point/area targets.
        Returns (mask2d, meta).
        """
        lat_vec = np.asarray(G["lat_vec"], dtype=float)
        lon_vec = np.asarray(G["lon_vec"], dtype=float)
        lon2d, lat2d = np.meshgrid(lon_vec, lat_vec)
    
        if target.get("type") == "point":
            d = self._uq_haversine_km(target["lat"], target["lon"], lat2d, lon2d)
            ij = np.unravel_index(np.nanargmin(d), d.shape)
            mask = np.zeros(lat2d.shape, dtype=bool)
            mask[ij] = True
            meta = {"kind": "nearest_cell", "ij": (int(ij[0]), int(ij[1])), "n_cells": 1, "min_dist_km": float(d[ij])}
            return mask, meta
    
        kind = target.get("kind", "").lower()
        if kind == "circle":
            d = self._uq_haversine_km(target["lat"], target["lon"], lat2d, lon2d)
            mask = d <= float(target["radius_km"])
            meta = {"kind": "circle", "n_cells": int(mask.sum()), "radius_km": float(target["radius_km"])}
            return mask, meta
    
        if kind == "bbox":
            mask = (
                (lat2d >= float(target["minlat"])) & (lat2d <= float(target["maxlat"])) &
                (lon2d >= float(target["minlon"])) & (lon2d <= float(target["maxlon"]))
            )
            meta = {"kind": "bbox", "n_cells": int(mask.sum())}
            return mask, meta
    
        raise ValueError(f"Unknown target kind: {target}")
    
    # orientation fix update v26.5 15.01.2026
    def _uq_raw_get_field_grid(self, D, field_name):
        """Return 2D grid for a field name from parsed XML dict D."""
        import numpy as np
    
        nameU = str(field_name).upper()
        if nameU not in D["fields"]:
            return None
        c = D["fields"][nameU]
        if c < 0 or c >= D["data"].shape[1]:
            return None
        # reshape to (nlat,nlon) row-major
        grid = D["data"][:, c].reshape((D["nlat"], D["nlon"]))
    
        # Optional: apply stored orientation if present
        orient = D.get("orientation", None) if isinstance(D, dict) else None
        if isinstance(orient, dict):
            if bool(orient.get("transpose", False)):
                grid = grid.T
            if bool(orient.get("flipud", False)):
                grid = np.flipud(grid)
            if bool(orient.get("fliplr", False)):
                grid = np.fliplr(grid)
    
        return grid


    
    
    # ----------------------------------------------------------------------
    # 9) Plotting: target curves (prediction and uncertainty) with ShakeMap envelopes
    # ----------------------------------------------------------------------
    def uq_plot_targets_comparison(
        self,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        # what to plot
        what="sigma",  # "mean" | "sigma" | "delta_mean" | "delta_sigma"
        methods=("ShakeMap", "bayes", "hierarchical", "kriging", "montecarlo"),
        agg="mean",
        global_stat=None,
        prior_version=None,
        # UQ controls
        sigma_total_from_shakemap=True,
        sigma_aleatory=None,
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
        ok_range_km=60.0,
        ok_variogram="exponential",
        ok_nugget=1e-6,
        ok_sill=None,
        ok_cap_sigma_to_prior=True,
        mc_nsim=2000,
        mc_include_aleatory=True,
        # unified controls
        grid_res=None,
        interp_method="nearest",
        interp_kwargs=None,
        # plotting kwargs
        figsize=(9.5, 5.2),
        dpi=300,
        ylog=False,
        ymin=None,
        ymax=None,
        xrotation=45,
        show_title=True,
        title=None,
        show_grid=True,
        legend=True,
        legend_kwargs=None,
        tight=True,
        output_path=None,
        save=False,
        save_formats=("png", "pdf"),
        show=True,
        # labels
        xlabel="ShakeMap version",
        ylabel=None,
        # envelopes
        show_shakemap_envelope=True,
        envelope_alpha=0.18,
        # audit
        audit=True,
        audit_output_path=None,
        audit_prefix=None,
    ):
        """
        Comparison plotter (Patch 4):
    
        - what="mean": plot mean_published for ShakeMap, mean_predicted for methods
        - what="sigma": plot sigma_total_published_raw for ShakeMap, sigma_total_predicted for methods
        - what="delta_mean": plot delta_mean_vs_published for methods (ShakeMap baseline at 0)
        - what="delta_sigma": plot delta_sigma_vs_published for methods (ShakeMap baseline at 0)
    
        Envelopes:
          For area/global targets, if show_shakemap_envelope=True:
            - Shade min/max of published over mask for ShakeMap curve only.
        """
        if legend_kwargs is None:
            legend_kwargs = {}
    
        methods = [("ShakeMap" if str(m).lower().strip() in ("published", "shakemap") else m) for m in methods]
        methods = tuple(methods)
    
        df = self.uq_extract_target_series(
            version_list=version_list,
            imt=imt,
            points=points,
            areas=areas,
            agg=agg,
            global_stat=global_stat,
            shakemap_total_sigma_mode="raw",
            sigma_total_from_shakemap=sigma_total_from_shakemap,
            sigma_aleatory=sigma_aleatory,
            prior_version=prior_version,
            update_radius_km=update_radius_km,
            kernel=kernel,
            kernel_scale_km=kernel_scale_km,
            measurement_sigma=measurement_sigma,
            ok_range_km=ok_range_km,
            ok_variogram=ok_variogram,
            ok_nugget=ok_nugget,
            ok_sill=ok_sill,
            ok_cap_sigma_to_prior=ok_cap_sigma_to_prior,
            mc_nsim=mc_nsim,
            mc_include_aleatory=mc_include_aleatory,
            grid_res=grid_res,
            interp_method=interp_method,
            interp_kwargs=interp_kwargs,
            audit=audit,
            audit_output_path=audit_output_path or output_path,
            audit_prefix=audit_prefix,
        )
    
        w = str(what).lower().strip()
        if w not in ("mean", "sigma", "delta_mean", "delta_sigma"):
            raise ValueError('what must be one of: "mean","sigma","delta_mean","delta_sigma".')
    
        targets = sorted(df["target_id"].unique().tolist())
    
        for tid in targets:
            sub = df[(df["target_id"] == tid) & (df["method"].isin(methods))].copy()
            sub = sub.sort_values(["version", "method"])
    
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
            # Shade ShakeMap envelopes (area/global)
            if show_shakemap_envelope:
                sm = df[(df["target_id"] == tid) & (df["method"] == "ShakeMap")].copy().sort_values("version")
                # envelope meaningful only if min/max differ
                if not sm.empty:
                    x = sm["version"].values
                    if w == "mean":
                        lo = sm["mean_published_min"].astype(float).values
                        hi = sm["mean_published_max"].astype(float).values
                    elif w == "sigma":
                        lo = sm["sigma_published_min"].astype(float).values
                        hi = sm["sigma_published_max"].astype(float).values
                    else:
                        lo = None
                        hi = None
    
                    if lo is not None and hi is not None:
                        # only shade if not all equal and finite
                        if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)) and (np.nanmax(hi - lo) > 0):
                            ax.fill_between(x, lo, hi, alpha=float(envelope_alpha), label="ShakeMap min/max")
    
            for m in methods:
                s = sub[sub["method"] == m].sort_values("version")
                if s.empty:
                    continue
    
                if w == "mean":
                    y = s["mean_published"].values if m == "ShakeMap" else s["mean_predicted"].values
                elif w == "sigma":
                    y = s["sigma_total_published_raw"].values if m == "ShakeMap" else s["sigma_total_predicted"].values
                elif w == "delta_mean":
                    y = np.zeros_like(s["version"].values, dtype=float) if m == "ShakeMap" else s["delta_mean_vs_published"].values
                else:  # delta_sigma
                    y = np.zeros_like(s["version"].values, dtype=float) if m == "ShakeMap" else s["delta_sigma_vs_published"].values
    
                ax.plot(s["version"].values, y, marker="o", linewidth=1.8, label=m)
    
            if ylog:
                ax.set_yscale("log")
            if ymin is not None or ymax is not None:
                ax.set_ylim(ymin, ymax)
    
            ax.set_xlabel(str(xlabel))
            if ylabel is not None:
                ax.set_ylabel(str(ylabel))
            else:
                if w == "mean":
                    ax.set_ylabel(f"{str(imt).upper()} (prediction)")
                elif w == "sigma":
                    ax.set_ylabel(f"sigma_total ({str(imt).upper()})")
                elif w == "delta_mean":
                    ax.set_ylabel(f"Δ mean vs ShakeMap ({str(imt).upper()})")
                else:
                    ax.set_ylabel(f"Δ sigma vs ShakeMap ({str(imt).upper()})")
    
            if show_grid:
                ax.grid(True, which="both", alpha=0.3)
    
            if show_title:
                if title is not None:
                    ax.set_title(str(title).replace("{target}", tid).replace("{imt}", str(imt).upper()))
                else:
                    ax.set_title(f"Patch4 comparison @ {tid} — {str(imt).upper()} — {w}")
    
            ax.tick_params(axis="x", rotation=float(xrotation))
    
            if legend:
                ax.legend(**legend_kwargs)
    
            if tight:
                fig.tight_layout()
    
            if save and output_path is not None:
                self._uq_save_figure_safe(
                    fig,
                    fname_stem=f"UQ-Patch4-Compare-{str(imt).upper()}-{w}-{tid}",
                    subdir="uq_plots/patch4_comparison",
                    output_path=output_path,
                    save_formats=save_formats,
                    dpi=dpi,
                )
    
            if show:
                plt.show()
            else:
                plt.close(fig)
    
        return df
    


    # ======================================================================
    # PATCH 4.1 (ADD-ON): Prediction+Uncertainty Comparative Framework
    #   - Adds explicit "published vs predicted" framing for BOTH mean and sigma
    #   - Adds USGS (ShakeMap) area-band shading (min/max across target mask)
    #   - Adds raw-XML vs unified-grid curve comparison (point OR area)
    #   - Hardens raw XML discovery + parsing + sampling
    #
    # Paste at the VERY END of the SHAKEuq class (after Patch 4).
    # Later definitions override earlier ones inside the class.
    # ======================================================================
    
    # ----------------------------------------------------------------------
    # 4.1-A) RAW ShakeMap XML discovery + parsing (hardened)
    # ----------------------------------------------------------------------
    def _uq_find_shakemap_xml(self, version, which="grid", base_folder=None, shakemap_folder=None):
        """
        Patch 4.1-A++:
          - Event-safe discovery: prefers (and when possible constrains to) self.event_id folder.
          - Content-checked: skips candidates that do not contain <grid_spec>.
          - Still supports explicit args, uq_state roots, and class-known folders.
    
        Returns
        -------
        Path or None
        """
        from pathlib import Path
    
        v = int(version)
        which = str(which).lower().strip()
        if which not in ("grid", "uncertainty"):
            raise ValueError('which must be "grid" or "uncertainty".')
    
        patterns = (
            ["grid.xml", "*grid.xml", "*_grid.xml"]
            if which == "grid"
            else ["uncertainty.xml", "*uncertainty.xml", "*_uncertainty.xml"]
        )
    
        # Try to get the current event id (best-effort)
        event_id = None
        for attr in ("event_id", "eventid", "event", "event_name"):
            if hasattr(self, attr):
                val = getattr(self, attr)
                if isinstance(val, str) and val.strip():
                    event_id = val.strip()
                    break
    
        roots = []
        def _add_root(r):
            try:
                if r is None:
                    return
                p = Path(r)
                if p not in roots:
                    roots.append(p)
            except Exception:
                return
    
        # 1) Explicit args first
        _add_root(base_folder)
        _add_root(shakemap_folder)
    
        # 2) uq_state base_folder
        if hasattr(self, "uq_state") and isinstance(self.uq_state, dict):
            bf = self.uq_state.get("base_folder", None)
            if bf:
                _add_root(Path(bf) / f"v{v}")
                _add_root(Path(bf) / f"v{str(v).zfill(2)}")
                _add_root(Path(bf) / f"v{str(v).zfill(3)}")
                _add_root(bf)
    
        # 3) Common class attributes
        for attr in ("base_folder", "shakemap_folder", "event_folder", "export_folder", "output_folder"):
            if hasattr(self, attr):
                _add_root(getattr(self, attr, None))
    
        # If we know event_id, add common event-specific roots early
        # (This is what prevents cross-event mixing)
        if event_id:
            event_roots = []
            for r in list(roots):
                try:
                    rp = Path(r)
                    if not rp.exists():
                        continue
                    # If the current root already includes the event_id, keep it as-is
                    if event_id.lower() in str(rp).lower():
                        event_roots.append(rp)
                        continue
                    # Otherwise, if there's a direct child folder matching the event_id, prefer that
                    cand = rp / event_id
                    if cand.exists():
                        event_roots.append(cand)
                except Exception:
                    continue
            # Put event_roots first (but keep the originals as fallbacks)
            roots = event_roots + [r for r in roots if r not in event_roots]
    
        # Small helper: content validation (avoid wrong XML / HTML)
        def _looks_like_shakemap_gridxml(p: Path) -> bool:
            try:
                # read a small chunk only
                txt = p.read_text(encoding="utf-8", errors="ignore")
                # ShakeMap grid + uncertainty XMLs both include <grid_spec ...>
                return "<grid_spec" in txt
            except Exception:
                return False
    
        # Scoring: prefer current event_id, then exact filename, then vXX/vXXX, then shorter path
        def _score(path: Path):
            s = str(path).lower()
            name = path.name.lower()
            score = 0.0
    
            if event_id and event_id.lower() in s:
                score += 100.0  # BIG bias: must not cross events
    
            if which == "grid" and name == "grid.xml":
                score += 10.0
            if which == "uncertainty" and name == "uncertainty.xml":
                score += 10.0
    
            # version hints (can appear as v002 or _002_)
            if f"v{v}" in s:
                score += 3.0
            if f"v{str(v).zfill(2)}" in s:
                score += 2.0
            if f"v{str(v).zfill(3)}" in s:
                score += 2.0
            if f"_{str(v).zfill(3)}_" in s:
                score += 2.0
    
            score -= 0.001 * len(s)
            return score
    
        # Search
        candidates = []
        for root in roots:
            try:
                root = Path(root)
                if not root.exists():
                    continue
    
                hits = []
                for pat in patterns:
                    # direct file check for exact names
                    if pat in ("grid.xml", "uncertainty.xml"):
                        p0 = root / pat
                        if p0.exists():
                            hits.append(p0)
                    hits.extend(list(root.rglob(pat)))
    
                if not hits:
                    continue
    
                for h in hits:
                    if not h.is_file():
                        continue
                    # If we know event_id, reject cross-event hits early
                    if event_id and event_id.lower() not in str(h).lower():
                        continue
                    # Content check: skip non-grid-like XML
                    if not _looks_like_shakemap_gridxml(h):
                        continue
                    candidates.append(h.resolve())
            except Exception:
                continue
    
        if not candidates:
            return None
    
        # Choose best candidate
        candidates = list({c for c in candidates})
        candidates.sort(key=_score, reverse=True)
        return candidates[0]
    

    # orientation update v26.5 15.01.2025
    def _uq_parse_gridxml_to_arrays(self, grid_xml_path):
        """
        ShakeMap XML parser (grid.xml or uncertainty.xml), with:
          - grid_spec OR grid_specification (namespace tolerant)
          - grid_field mapping (NAME -> 0-based index within field block)
          - grid_data (nlat*nlon rows)
        PLUS: in-memory caching to avoid re-parsing the same big XML repeatedly.
    
        Returns dict:
          {
            "nlat","nlon","lat_min","lat_max","lon_min","lon_max",
            "lat_vec","lon_vec","fields","data"
          }
        """
        from pathlib import Path
        import numpy as np
        import xml.etree.ElementTree as ET
    
        p = Path(grid_xml_path)
    
        # -----------------------------
        # Cache (keyed by resolved path + mtime)
        # -----------------------------
        if not hasattr(self, "_uq_xml_cache") or not isinstance(getattr(self, "_uq_xml_cache"), dict):
            self._uq_xml_cache = {}
    
        try:
            rp = p.resolve()
        except Exception:
            rp = p
    
        try:
            mtime = rp.stat().st_mtime
        except Exception:
            mtime = None
    
        cache_key = (str(rp), mtime)
        if cache_key in self._uq_xml_cache:
            return self._uq_xml_cache[cache_key]
    
        # -----------------------------
        # Parse XML
        # -----------------------------
        tree = ET.parse(str(rp))
        root = tree.getroot()
    
        # --- grid spec: tolerate tag name + namespaces ---
        spec = root.find(".//{*}grid_spec")
        if spec is None:
            spec = root.find(".//{*}grid_specification")
        if spec is None:
            for el in root.iter():
                tag = str(el.tag).lower()
                if tag.endswith("grid_spec") or tag.endswith("grid_specification"):
                    spec = el
                    break
        if spec is None:
            raise RuntimeError(f"No grid_spec/grid_specification found in {rp}")
    
        nlon = int(float(spec.attrib.get("nlon")))
        nlat = int(float(spec.attrib.get("nlat")))
        lon_min = float(spec.attrib.get("lon_min"))
        lon_max = float(spec.attrib.get("lon_max"))
        lat_min = float(spec.attrib.get("lat_min"))
        lat_max = float(spec.attrib.get("lat_max"))
    
        # Prefer spacing if provided
        dlon = (
            spec.attrib.get("lon_spacing")
            or spec.attrib.get("nominal_lon_spacing")
            or spec.attrib.get("dlon")
        )
        dlat = (
            spec.attrib.get("lat_spacing")
            or spec.attrib.get("nominal_lat_spacing")
            or spec.attrib.get("dlat")
        )
    
        try:
            dlon = float(dlon) if dlon is not None else None
        except Exception:
            dlon = None
        try:
            dlat = float(dlat) if dlat is not None else None
        except Exception:
            dlat = None
    
        lon_vec = lon_min + np.arange(nlon, dtype=float) * dlon if (dlon is not None and nlon > 1) else np.linspace(lon_min, lon_max, nlon, dtype=float)
        lat_vec = lat_min + np.arange(nlat, dtype=float) * dlat if (dlat is not None and nlat > 1) else np.linspace(lat_min, lat_max, nlat, dtype=float)
    
        # fields: NAME -> 0-based field-block index (idx-1)
        fields = {}
        for gf in root.findall(".//{*}grid_field"):
            try:
                idx = int(gf.attrib.get("index"))
                name = str(gf.attrib.get("name")).strip().upper()
                fields[name] = idx - 1
            except Exception:
                continue
    
        gd = root.find(".//{*}grid_data")
        if gd is None or gd.text is None:
            raise RuntimeError(f"No grid_data found in {rp}")
    
        lines = [ln.strip() for ln in gd.text.strip().splitlines() if ln.strip()]
    
        # Parse numbers (this is heavy, hence caching)
        arr = np.array([[float(x) for x in ln.split()] for ln in lines], dtype=float)
    
        expect = nlat * nlon
        if arr.shape[0] != expect:
            if arr.shape[0] > expect:
                arr = arr[-expect:, :]
            else:
                raise RuntimeError(f"grid_data rows mismatch in {rp.name}: got {arr.shape[0]}, expected {expect}")
    
        # ------------------------------------------------------------------
        # FIX: enforce consistent row order (lat-major, lon-minor) when possible
        #      If LON/LAT columns exist, reorder rows by (LAT, LON) and rebuild
        #      lat_vec/lon_vec from those columns (sorted unique).
        # ------------------------------------------------------------------
        row_order = None
        try:
            lon_keys = {"LON", "LONGITUDE", "LON_DEG", "LONDD"}
            lat_keys = {"LAT", "LATITUDE", "LAT_DEG", "LATDD"}
    
            fkeys = set((fields or {}).keys())
            has_lon = len(fkeys.intersection(lon_keys)) > 0
            has_lat = len(fkeys.intersection(lat_keys)) > 0
    
            if has_lon and has_lat:
                lon_name = next(iter(fkeys.intersection(lon_keys)))
                lat_name = next(iter(fkeys.intersection(lat_keys)))
    
                lon_col = int(fields[lon_name])
                lat_col = int(fields[lat_name])
    
                if 0 <= lon_col < arr.shape[1] and 0 <= lat_col < arr.shape[1]:
                    lons = np.asarray(arr[:, lon_col], dtype=float)
                    lats = np.asarray(arr[:, lat_col], dtype=float)
    
                    # Sort rows to lat-major / lon-minor (stable, deterministic)
                    row_order = np.lexsort((lons, lats))
                    arr = arr[row_order, :]
    
                    # Rebuild axes from the actual lon/lat columns
                    lon_vec2 = np.unique(np.round(lons, 12))
                    lat_vec2 = np.unique(np.round(lats, 12))
                    lon_vec2.sort()
                    lat_vec2.sort()
    
                    # Only override if sizes match (guard against weird metadata)
                    if lon_vec2.size == nlon and lat_vec2.size == nlat:
                        lon_vec = lon_vec2
                        lat_vec = lat_vec2
                    else:
                        # Still enforce increasing axes from spec
                        lon_vec = np.sort(np.asarray(lon_vec, dtype=float))
                        lat_vec = np.sort(np.asarray(lat_vec, dtype=float))
                else:
                    lon_vec = np.sort(np.asarray(lon_vec, dtype=float))
                    lat_vec = np.sort(np.asarray(lat_vec, dtype=float))
            else:
                lon_vec = np.sort(np.asarray(lon_vec, dtype=float))
                lat_vec = np.sort(np.asarray(lat_vec, dtype=float))
        except Exception:
            # Fail-soft: keep original but ensure monotonic increasing
            lon_vec = np.sort(np.asarray(lon_vec, dtype=float))
            lat_vec = np.sort(np.asarray(lat_vec, dtype=float))
            row_order = None
    
        out = {
            "nlat": nlat, "nlon": nlon,
            "lat_min": lat_min, "lat_max": lat_max,
            "lon_min": lon_min, "lon_max": lon_max,
            "lat_vec": np.asarray(lat_vec, dtype=float),
            "lon_vec": np.asarray(lon_vec, dtype=float),
            "fields": fields,
            "data": arr,
            "row_order": row_order,
        }
    
        # Store in cache
        self._uq_xml_cache[cache_key] = out
    
        # Optional: prevent cache blow-up (keep last ~30 files)
        if len(self._uq_xml_cache) > 30:
            # drop oldest inserted (simple strategy)
            try:
                first_key = next(iter(self._uq_xml_cache.keys()))
                self._uq_xml_cache.pop(first_key, None)
            except Exception:
                pass
    
        return out



    # orientation update v26.5 15.01.2026
    def _uq_raw_sample_from_xml(self, grid_xml_path, uncertainty_xml_path, lat, lon, imt="MMI", sample="nearest"):
        """
        Sample RAW mean and RAW sigma from ShakeMap grid.xml + uncertainty.xml at (lat, lon).
        """
        import numpy as np
    
        imtU = str(imt).upper().strip()
        stdU = f"STD{imtU}"
        sample = str(sample).lower().strip()
    
        # Parse (fail-soft)
        try:
            G = self._uq_parse_gridxml_to_arrays(grid_xml_path)
            U = self._uq_parse_gridxml_to_arrays(uncertainty_xml_path)
        except Exception as e:
            return {"raw_mean": np.nan, "raw_sigma": np.nan, "nearest_km": np.nan, "error": str(e)}
    
        # ------------------------------------------------------------------
        # FIX: if grid rows were reordered using lon/lat, apply SAME ordering
        #      to uncertainty rows so row indexing matches.
        # ------------------------------------------------------------------
        try:
            ro = G.get("row_order", None)
            if ro is not None:
                ro = np.asarray(ro, dtype=int)
                if ("row_order" not in U) or (U.get("row_order", None) is None):
                    if "data" in U and U["data"] is not None and U["data"].shape[0] == ro.size:
                        U = dict(U)
                        U["data"] = np.asarray(U["data"], dtype=float)[ro, :]
                        U["row_order"] = ro
        except Exception:
            pass
    
        def _has_lonlat(fields_dict):
            f = set((fields_dict or {}).keys())
            lon_keys = {"LON", "LONGITUDE", "LON_DEG", "LONDD"}
            lat_keys = {"LAT", "LATITUDE", "LAT_DEG", "LATDD"}
            return (len(f.intersection(lon_keys)) > 0) and (len(f.intersection(lat_keys)) > 0)
    
        def _pick_field_col(D, name_candidates):
            fields = D.get("fields", {}) or {}
            arr = D.get("data", None)
            if arr is None:
                return None
    
            found = None
            for nm in name_candidates:
                if nm in fields:
                    found = int(fields[nm])
                    break
            if found is None:
                return None
    
            ncols = int(np.asarray(arr).shape[1])
    
            if _has_lonlat(fields):
                col = found
                if 0 <= col < ncols:
                    return col
                return None
    
            col = found + 2
            if 0 <= col < ncols:
                return col
    
            if 0 <= found < ncols:
                return found
    
            return None
    
        g_candidates = [imtU]
        u_candidates = [stdU, f"{imtU}_STD", f"STD_{imtU}"]
    
        gcol = _pick_field_col(G, g_candidates)
        ucol = _pick_field_col(U, u_candidates)
    
        if gcol is None:
            return {"raw_mean": np.nan, "raw_sigma": np.nan, "nearest_km": np.nan,
                    "error": f"IMT '{imtU}' not found or col invalid. Fields={list((G.get('fields') or {}).keys())}"}
        if ucol is None:
            return {"raw_mean": np.nan, "raw_sigma": np.nan, "nearest_km": np.nan,
                    "error": f"STD '{stdU}' not found or col invalid. Fields={list((U.get('fields') or {}).keys())}"}
    
        latv = np.asarray(G.get("lat_vec", []), dtype=float)
        lonv = np.asarray(G.get("lon_vec", []), dtype=float)
        if latv.size == 0 or lonv.size == 0:
            return {"raw_mean": np.nan, "raw_sigma": np.nan, "nearest_km": np.nan,
                    "error": "Missing lat_vec/lon_vec from RAW grid spec."}
    
        iy = int(np.argmin(np.abs(latv - float(lat))))
        ix = int(np.argmin(np.abs(lonv - float(lon))))
    
        dlat = (latv[iy] - float(lat)) * 111.0
        dlon = (lonv[ix] - float(lon)) * 111.0 * np.cos(np.deg2rad(float(lat)))
        nearest_km = float(np.sqrt(dlat * dlat + dlon * dlon))
    
        arrG = np.asarray(G["data"], dtype=float)
        arrU = np.asarray(U["data"], dtype=float)
        nlon = int(G["nlon"])
    
        def _row(i_y, i_x):
            return int(i_y) * nlon + int(i_x)
    
        if sample == "bilinear":
            iy0 = max(0, min(len(latv) - 2, iy))
            ix0 = max(0, min(len(lonv) - 2, ix))
    
            y1, y2 = float(latv[iy0]), float(latv[iy0 + 1])
            x1, x2 = float(lonv[ix0]), float(lonv[ix0 + 1])
    
            wy = 0.0 if y2 == y1 else (float(lat) - y1) / (y2 - y1)
            wx = 0.0 if x2 == x1 else (float(lon) - x1) / (x2 - x1)
            wy = float(np.clip(wy, 0.0, 1.0))
            wx = float(np.clip(wx, 0.0, 1.0))
    
            def _val(arr, col):
                q11 = arr[_row(iy0, ix0), col]
                q21 = arr[_row(iy0, ix0 + 1), col]
                q12 = arr[_row(iy0 + 1, ix0), col]
                q22 = arr[_row(iy0 + 1, ix0 + 1), col]
                return (q11 * (1 - wx) * (1 - wy) +
                        q21 * (wx) * (1 - wy) +
                        q12 * (1 - wx) * (wy) +
                        q22 * (wx) * (wy))
    
            raw_mean = float(_val(arrG, gcol))
            raw_sigma = float(_val(arrU, ucol))
        else:
            r = _row(iy, ix)
            raw_mean = float(arrG[r, gcol])
            raw_sigma = float(arrU[r, ucol])
    
        return {"raw_mean": raw_mean, "raw_sigma": raw_sigma, "nearest_km": nearest_km}
    
        



    
    # ----------------------------------------------------------------------
    # 4.1-B) Target-area band extraction for published (ShakeMap) curves
    # ----------------------------------------------------------------------
    def _uq_band_minmax_unified(self, version, imt, lat2d, lon2d, mask, grid_res=None, interp_method="nearest", interp_kwargs=None):
        """
        Patch 4.1: For a given version + target mask, return min/max for:
          - mean (mu)
          - sigma_raw (published sigma_total from uncertainty.xml)
        on the unified grid.
        """
        if interp_kwargs is None:
            interp_kwargs = {}
        imtU = str(imt).upper()
        mu, sig = self._uq_get_mu_sigma_unified(
            int(version), imtU, lat2d, lon2d,
            grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
        )
        mu_m = np.asarray(mu, dtype=float)[mask]
        sg_m = np.asarray(sig, dtype=float)[mask]
    
        def _minmax(x):
            x = x[np.isfinite(x)]
            if x.size == 0:
                return (np.nan, np.nan)
            return (float(np.nanmin(x)), float(np.nanmax(x)))
    
        mu_min, mu_max = _minmax(mu_m)
        sg_min, sg_max = _minmax(sg_m)
        return {"mu_min": mu_min, "mu_max": mu_max, "sig_min": sg_min, "sig_max": sg_max}
    



            
    
    # ----------------------------------------------------------------------
    # 4.1-D) NEW: Explicit prediction+uncertainty comparison plot (publish-ready)
    # ----------------------------------------------------------------------
    def uq_plot_published_vs_predicted(
        self,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        methods=("bayes", "hierarchical", "kriging", "montecarlo"),
        agg="mean",
        global_stat=None,
        prior_version=None,
        # UQ controls
        sigma_total_from_shakemap=True,
        sigma_aleatory=None,
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
        ok_range_km=60.0,
        ok_variogram="exponential",
        ok_nugget=1e-6,
        ok_sill=None,
        ok_cap_sigma_to_prior=True,
        mc_nsim=2000,
        mc_include_aleatory=True,
        # unified controls
        grid_res=None,
        interp_method="nearest",
        interp_kwargs=None,
        # plotting
        figsize=(11, 6),
        dpi=300,
        ylog_sigma=False,
        ymin_mu=None,
        ymax_mu=None,
        ymin_sigma=None,
        ymax_sigma=None,
        xrotation=45,
        show_grid=True,
        title=None,
        legend=True,
        legend_kwargs=None,
        tight=True,
        output_path=None,
        save=False,
        save_formats=("png", "pdf"),
        show=True,
        # Patch 4.1: published area-band shading
        published_band=True,
        published_band_alpha=0.18,
        # labels
        xlabel="ShakeMap version",
        ylabel_mu=None,
        ylabel_sigma=None,
        # audit
        audit=True,
        audit_output_path=None,
        audit_prefix=None,
    ):
        """
        Patch 4.1: Explicit framing plot:
          Panel A: Published ShakeMap mean vs Predicted mean (methods) over versions
          Panel B: Published ShakeMap RAW sigma_total vs Predicted sigma_total over versions
    
        If target is AREA/GLOBAL, optional min/max band is drawn for Published curves only.
        """
        if legend_kwargs is None:
            legend_kwargs = {}
    
        df = self.uq_extract_target_series(
            version_list=version_list,
            imt=imt,
            points=points,
            areas=areas,
            agg=agg,
            global_stat=global_stat,
            prior_version=prior_version,
            shakemap_total_sigma_mode="raw",
            sigma_total_from_shakemap=sigma_total_from_shakemap,
            sigma_aleatory=sigma_aleatory,
            update_radius_km=update_radius_km,
            kernel=kernel,
            kernel_scale_km=kernel_scale_km,
            measurement_sigma=measurement_sigma,
            ok_range_km=ok_range_km,
            ok_variogram=ok_variogram,
            ok_nugget=ok_nugget,
            ok_sill=ok_sill,
            ok_cap_sigma_to_prior=ok_cap_sigma_to_prior,
            mc_nsim=mc_nsim,
            mc_include_aleatory=mc_include_aleatory,
            grid_res=grid_res,
            interp_method=interp_method,
            interp_kwargs=interp_kwargs,
            audit=audit,
            audit_output_path=audit_output_path or output_path,
            audit_prefix=audit_prefix,
        )
    
        # unify axes for bands
        versions = [int(v) for v in (version_list or [])]
        _, lat2d, lon2d = self._uq_get_unified_for_versions(
            versions, imt=str(imt).upper(),
            grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
        )
    
        # rebuild targets for masks
        if global_stat is not None:
            gs = str(global_stat).lower().strip()
            gid = f"GLOBAL_{gs.replace(' ', '_')}".replace("__", "_")
            targets = [{"id": gid, "type": "global"}]
        else:
            targets = self._uq_parse_targets(points=points, areas=areas)
    
        masks = {}
        for t in targets:
            tid = t.get("id", "GLOBAL")
            if t.get("type") == "global":
                masks[tid] = ("global", np.isfinite(lat2d))
            else:
                m, _ = self._uq_target_mask(t, lat2d, lon2d)
                masks[tid] = ("area" if t.get("type") == "area" else "target", m)
    
        targets = sorted(df["target_id"].unique().tolist())
        pred_methods = [m for m in methods]
        for tid in targets:
            sub_pub = df[(df["target_id"] == tid) & (df["method"] == "ShakeMap")].sort_values("version")
            sub_pred = df[(df["target_id"] == tid) & (df["method"].isin(pred_methods))].copy()
    
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
    
            # Published band (area/global only)
            ttype, mask = masks.get(tid, ("target", None))
            is_area_like = (ttype in ("area", "global")) and (mask is not None)
    
            if published_band and is_area_like and (len(sub_pub) > 0):
                xs = []
                mu_lo = []; mu_hi = []
                sg_lo = []; sg_hi = []
                for v in sub_pub["version"].values:
                    band = self._uq_band_minmax_unified(
                        version=int(v), imt=str(imt).upper(),
                        lat2d=lat2d, lon2d=lon2d, mask=mask,
                        grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
                    )
                    xs.append(int(v))
                    mu_lo.append(band["mu_min"]); mu_hi.append(band["mu_max"])
                    sg_lo.append(band["sig_min"]); sg_hi.append(band["sig_max"])
                ax1.fill_between(xs, mu_lo, mu_hi, alpha=float(published_band_alpha), label="Published min/max (area)")
                ax2.fill_between(xs, sg_lo, sg_hi, alpha=float(published_band_alpha), label="Published sigma min/max (area)")
    
            # Panel A: mean
            ax1.plot(sub_pub["version"].values, sub_pub["mean"].values, marker="o", linewidth=2.0, label="Published (ShakeMap)")
            for m in pred_methods:
                s = sub_pred[sub_pred["method"] == m].sort_values("version")
                if len(s) == 0:
                    continue
                ax1.plot(s["version"].values, s["mean"].values, marker="o", linewidth=1.6, label=f"Predicted ({m})")
    
            if ymin_mu is not None or ymax_mu is not None:
                ax1.set_ylim(ymin_mu, ymax_mu)
            ax1.set_ylabel(ylabel_mu if ylabel_mu is not None else f"{str(imt).upper()} mean")
            if show_grid:
                ax1.grid(True, alpha=0.3)
    
            # Panel B: sigma_total (RAW for published)
            ax2.plot(sub_pub["version"].values, sub_pub["sigma_total"].values, marker="o", linewidth=2.0, label="Published σ_total (RAW)")
            for m in pred_methods:
                s = sub_pred[sub_pred["method"] == m].sort_values("version")
                if len(s) == 0:
                    continue
                ax2.plot(s["version"].values, s["sigma_total"].values, marker="o", linewidth=1.6, label=f"Predicted σ_total ({m})")
    
            if ylog_sigma:
                ax2.set_yscale("log")
            if ymin_sigma is not None or ymax_sigma is not None:
                ax2.set_ylim(ymin_sigma, ymax_sigma)
            ax2.set_ylabel(ylabel_sigma if ylabel_sigma is not None else f"{str(imt).upper()} σ_total")
            ax2.set_xlabel(str(xlabel))
            ax2.tick_params(axis="x", rotation=float(xrotation))
            if show_grid:
                ax2.grid(True, which="both", alpha=0.3)
    
            if title is None:
                fig.suptitle(f"Published vs Predicted — {str(imt).upper()} — {tid}")
            else:
                fig.suptitle(str(title).replace("{target}", tid).replace("{imt}", str(imt).upper()))
    
            if legend:
                # single combined legend
                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                all_h = handles1 + [h for h, lab in zip(handles2, labels2) if lab not in labels1]
                all_l = labels1 + [lab for lab in labels2 if lab not in labels1]
                fig.legend(all_h, all_l, loc="upper right", **legend_kwargs)
    
            if tight:
                fig.tight_layout(rect=[0, 0, 1, 0.95])
    
            if save and output_path is not None:
                self._uq_save_figure_safe(
                    fig,
                    fname_stem=f"UQ-PublishedVsPredicted-{str(imt).upper()}-{tid}",
                    subdir="uq_plots/published_vs_predicted",
                    output_path=output_path,
                    save_formats=save_formats,
                    dpi=dpi,
                )
    
            if show:
                plt.show()
            else:
                plt.close(fig)
    
        return df
    
    
    
    # ======================================================================
    # PATCH 4.1-G (STANDALONE / DIAGNOSTICS): RAW XML vs UNIFIED GRID Comparison
    #   - Test-oriented tools to compare published ShakeMap grid.xml values
    #     against values re-sampled onto a user-defined unified grid.
    #   - Builds its OWN unified grid (does not depend on uq_state).
    #   - Supports POINT and AREA targets, with controllable:
    #       * grid_res
    #       * interp_method (nearest / bilinear)
    #   - Intended for validation and sanity checks only.
    #     Production UQ pipeline MUST NOT depend on these helpers.
    #
    # Functions in this section:
    #   - uq_compare_raw_vs_unified_curves
    #   - _pick_col
    #   - _nearest_idx
    #   - _interp_nearest
    #   - _interp_bilinear
    # ======================================================================

    
    # ----------------------------------------------------------------------
    # 4.1-F) Mask subsampling helpers (for RAW area sampling speed/robustness)
    # ----------------------------------------------------------------------
    def _uq_mask_indices(self, mask, max_cells=None, stride=None, random_state=42, method="auto"):
        """
        Return (iy, ix) indices for True cells in mask with optional downsampling.
    
        Parameters
        ----------
        mask : 2D bool
        max_cells : int|None
            Maximum number of cells returned. If None, return all.
        stride : int|None
            If provided and >1, take every stride-th cell in the flattened index order.
        random_state : int
            For reproducible random subsampling.
        method : {"auto","stride","random"}
            auto: uses stride if given, else random if max_cells is given and mask too large.
    
        Returns
        -------
        ij : ndarray shape (k,2) with rows [iy,ix]
        meta : dict describing subsampling
        """
        import numpy as np
    
        m = np.asarray(mask, dtype=bool)
        ij = np.argwhere(m)
        n = int(ij.shape[0])
    
        meta = {"n_cells": n, "n_returned": n, "method": "all", "stride": None, "max_cells": None}
    
        if n == 0:
            meta["n_returned"] = 0
            return ij, meta
    
        if max_cells is None and (stride is None or int(stride) <= 1):
            return ij, meta
    
        stride = int(stride) if stride is not None else None
        max_cells = int(max_cells) if max_cells is not None else None
    
        how = str(method).lower().strip()
    
        # Decide strategy
        if how == "auto":
            if stride is not None and stride > 1:
                how = "stride"
            elif max_cells is not None and n > max_cells:
                how = "random"
            else:
                how = "all"
    
        if how == "stride":
            if stride is None or stride <= 1:
                return ij, meta
            ij2 = ij[::stride, :]
            meta.update({"n_returned": int(ij2.shape[0]), "method": "stride", "stride": int(stride), "max_cells": max_cells})
            # If still too many, cap with random
            if max_cells is not None and ij2.shape[0] > max_cells:
                rs = np.random.RandomState(int(random_state))
                pick = rs.choice(ij2.shape[0], size=int(max_cells), replace=False)
                ij2 = ij2[pick, :]
                meta.update({"n_returned": int(ij2.shape[0]), "method": "stride+random"})
            return ij2, meta
    
        if how == "random":
            if max_cells is None or n <= max_cells:
                return ij, meta
            rs = np.random.RandomState(int(random_state))
            pick = rs.choice(n, size=int(max_cells), replace=False)
            ij2 = ij[pick, :]
            meta.update({"n_returned": int(ij2.shape[0]), "method": "random", "stride": stride, "max_cells": int(max_cells)})
            return ij2, meta
    
        return ij, meta


    # orientation update 26.5 15.01.2026
    def _uq_area_raw_values_from_xml(
        self,
        grid_p,
        unc_p,
        imtU,
        lat2d,
        lon2d,
        mask,
        max_cells=25000,
        stride=None,
        random_state=42,
        subsample_method="auto",
    ):
        """
        For an AREA-like target, return RAW values sampled from published XML
        at the coordinates of the UNIFIED grid cells inside `mask`.
        """
        import numpy as np
    
        meta = {
            "method": None,
            "n_cells": int(np.sum(mask)) if mask is not None else 0,
            "n_returned": 0,
            "stride": None,
            "max_cells": max_cells,
            "status": "init",
            "missing": None,
        }
    
        if mask is None or int(np.sum(mask)) == 0:
            meta["status"] = "empty_mask"
            return np.asarray([], dtype=float), np.asarray([], dtype=float), meta
    
        try:
            G = self._uq_parse_gridxml_to_arrays(grid_p)
            U = self._uq_parse_gridxml_to_arrays(unc_p)
        except Exception as e:
            meta["status"] = "parse_failed"
            meta["missing"] = str(e)
            return np.asarray([], dtype=float), np.asarray([], dtype=float), meta
    
        # ------------------------------------------------------------------
        # FIX: if grid rows were reordered using lon/lat, apply SAME ordering
        #      to uncertainty rows so row indexing matches.
        # ------------------------------------------------------------------
        try:
            ro = G.get("row_order", None)
            if ro is not None:
                ro = np.asarray(ro, dtype=int)
                if ("row_order" not in U) or (U.get("row_order", None) is None):
                    if "data" in U and U["data"] is not None and U["data"].shape[0] == ro.size:
                        U = dict(U)
                        U["data"] = np.asarray(U["data"], dtype=float)[ro, :]
                        U["row_order"] = ro
        except Exception:
            pass
    
        def _has_lonlat(fields_dict):
            f = set((fields_dict or {}).keys())
            lon_keys = {"LON", "LONGITUDE", "LON_DEG", "LONDD"}
            lat_keys = {"LAT", "LATITUDE", "LAT_DEG", "LATDD"}
            return (len(f.intersection(lon_keys)) > 0) and (len(f.intersection(lat_keys)) > 0)
    
        def _pick_field_col(D, name_candidates):
            fields = D.get("fields", {}) or {}
            arr = D.get("data", None)
            if arr is None:
                return None
            found = None
            for nm in name_candidates:
                if nm in fields:
                    found = int(fields[nm])
                    break
            if found is None:
                return None
            ncols = int(np.asarray(arr).shape[1])
    
            if _has_lonlat(fields):
                col = found
                if 0 <= col < ncols:
                    return col
                return None
    
            col = found + 2
            if 0 <= col < ncols:
                return col
            if 0 <= found < ncols:
                return found
            return None
    
        imtU = str(imtU).upper().strip()
        stdU = f"STD{imtU}"
        gcol = _pick_field_col(G, [imtU])
        ucol = _pick_field_col(U, [stdU, f"{imtU}_STD", f"STD_{imtU}"])
    
        if gcol is None:
            meta["status"] = "missing_field"
            meta["missing"] = f"grid missing {imtU}"
            return np.asarray([], dtype=float), np.asarray([], dtype=float), meta
        if ucol is None:
            meta["status"] = "missing_field"
            meta["missing"] = f"uncertainty missing {stdU}"
            return np.asarray([], dtype=float), np.asarray([], dtype=float), meta
    
        latv = np.asarray(G.get("lat_vec", []), dtype=float)
        lonv = np.asarray(G.get("lon_vec", []), dtype=float)
        if latv.size == 0 or lonv.size == 0:
            meta["status"] = "missing_latlon_vec"
            meta["missing"] = "lat_vec/lon_vec missing"
            return np.asarray([], dtype=float), np.asarray([], dtype=float), meta
    
        iy, ix = np.where(mask)
        n = iy.size
    
        method = str(subsample_method or "auto").lower().strip()
        rng = np.random.default_rng(int(random_state) if random_state is not None else 42)
    
        if max_cells is not None and n > int(max_cells):
            if method == "stride":
                st = int(stride) if stride else int(np.ceil(np.sqrt(n / float(max_cells))))
                st = max(1, st)
                sel = np.arange(0, n, st, dtype=int)
                meta["method"] = "stride"
                meta["stride"] = st
            elif method == "random":
                k = int(max_cells)
                sel = rng.choice(n, size=k, replace=False)
                meta["method"] = "random"
                meta["stride"] = None
            else:
                if stride is not None:
                    st = max(1, int(stride))
                    sel = np.arange(0, n, st, dtype=int)
                    meta["method"] = "stride"
                    meta["stride"] = st
                else:
                    k = int(max_cells)
                    sel = rng.choice(n, size=k, replace=False)
                    meta["method"] = "random"
                    meta["stride"] = None
        else:
            sel = np.arange(n, dtype=int)
            meta["method"] = "none"
            meta["stride"] = None
    
        iy = iy[sel]
        ix = ix[sel]
        meta["n_returned"] = int(iy.size)
    
        lat_q = np.asarray(lat2d[iy, ix], dtype=float)
        lon_q = np.asarray(lon2d[iy, ix], dtype=float)
    
        def _nearest_index(vec, vals):
            vec = np.asarray(vec, dtype=float)
            vals = np.asarray(vals, dtype=float)
            inc = (vec[-1] >= vec[0])
            if not inc:
                vec2 = vec[::-1]
                idx = np.searchsorted(vec2, vals, side="left")
                idx = np.clip(idx, 0, vec2.size - 1)
                left = np.clip(idx - 1, 0, vec2.size - 1)
                choose_left = (np.abs(vec2[left] - vals) <= np.abs(vec2[idx] - vals))
                idx = np.where(choose_left, left, idx)
                return (vec.size - 1 - idx).astype(int)
            else:
                idx = np.searchsorted(vec, vals, side="left")
                idx = np.clip(idx, 0, vec.size - 1)
                left = np.clip(idx - 1, 0, vec.size - 1)
                choose_left = (np.abs(vec[left] - vals) <= np.abs(vec[idx] - vals))
                idx = np.where(choose_left, left, idx)
                return idx.astype(int)
    
        iy_raw = _nearest_index(latv, lat_q)
        ix_raw = _nearest_index(lonv, lon_q)
    
        nlon = int(G["nlon"])
        ridx = iy_raw * nlon + ix_raw
    
        arrG = np.asarray(G["data"], dtype=float)
        arrU = np.asarray(U["data"], dtype=float)
    
        raw_mu_vals = arrG[ridx, int(gcol)].astype(float)
        raw_sg_vals = arrU[ridx, int(ucol)].astype(float)
    
        meta["status"] = "ok"
        return raw_mu_vals, raw_sg_vals, meta
    
        
    
    # ----------------------------------------------------------------------
    # 4.1-H) NEW: Published-vs-Predicted residual tables (mean + sigma)
    # ----------------------------------------------------------------------
    def uq_published_vs_predicted_residuals(
        self,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        methods=("bayes", "hierarchical", "kriging", "montecarlo"),
        agg="mean",
        global_stat=None,
        prior_version=None,
        # UQ controls (passed through)
        sigma_total_from_shakemap=True,
        sigma_aleatory=None,
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
        ok_range_km=60.0,
        ok_variogram="exponential",
        ok_nugget=1e-6,
        ok_sill=None,
        ok_cap_sigma_to_prior=True,
        mc_nsim=2000,
        mc_include_aleatory=True,
        # unified controls
        grid_res=None,
        interp_method="nearest",
        interp_kwargs=None,
        # export
        output_path=None,
        export_table=True,
        export_prefix="UQ-PublishedVsPredicted-Residuals",
        # audit passthrough
        audit=True,
        audit_output_path=None,
        audit_prefix=None,
    ):
        """
        Build a tidy residual table comparing predicted methods to published ShakeMap for BOTH:
          - mean residual: (pred_mean - pub_mean)
          - sigma residual: (pred_sigma_total - pub_sigma_total_raw)
    
        Returns
        -------
        df_res : DataFrame with per-target/version/method residuals + simple skill metrics
        """
        if interp_kwargs is None:
            interp_kwargs = {}
    
        df = self.uq_extract_target_series(
            version_list=version_list,
            imt=imt,
            points=points,
            areas=areas,
            agg=agg,
            global_stat=global_stat,
            prior_version=prior_version,
            shakemap_total_sigma_mode="raw",
            sigma_total_from_shakemap=sigma_total_from_shakemap,
            sigma_aleatory=sigma_aleatory,
            update_radius_km=update_radius_km,
            kernel=kernel,
            kernel_scale_km=kernel_scale_km,
            measurement_sigma=measurement_sigma,
            ok_range_km=ok_range_km,
            ok_variogram=ok_variogram,
            ok_nugget=ok_nugget,
            ok_sill=ok_sill,
            ok_cap_sigma_to_prior=ok_cap_sigma_to_prior,
            mc_nsim=mc_nsim,
            mc_include_aleatory=mc_include_aleatory,
            grid_res=grid_res,
            interp_method=interp_method,
            interp_kwargs=interp_kwargs,
            audit=audit,
            audit_output_path=audit_output_path or output_path,
            audit_prefix=audit_prefix,
        )
    
        df = df.copy()
        df["method"] = df["method"].astype(str)
    
        pub = df[df["method"] == "ShakeMap"].copy()
        if pub.empty:
            raise RuntimeError("No published (ShakeMap) rows found in uq_extract_target_series output.")
    
        pub = pub[["target_id", "version", "mean", "sigma_total"]].rename(
            columns={"mean": "pub_mean", "sigma_total": "pub_sigma_total_raw"}
        )
    
        pred = df[df["method"].isin([str(m) for m in methods])].copy()
        if pred.empty:
            raise RuntimeError("No predicted rows found for requested methods.")
    
        # join
        j = pred.merge(pub, on=["target_id", "version"], how="left")
    
        # residuals
        j["res_mean"] = j["mean"].astype(float) - j["pub_mean"].astype(float)
        j["res_sigma_total"] = j["sigma_total"].astype(float) - j["pub_sigma_total_raw"].astype(float)
    
        j["abs_res_mean"] = np.abs(j["res_mean"].astype(float))
        j["abs_res_sigma_total"] = np.abs(j["res_sigma_total"].astype(float))
    
        # simple skill summaries (per target+method)
        out_rows = []
        for tid in j["target_id"].unique():
            for m in j["method"].unique():
                s = j[(j["target_id"] == tid) & (j["method"] == m)]
                if s.empty:
                    continue
                rmse_mu = float(np.sqrt(np.nanmean(np.square(s["res_mean"].astype(float))))) if np.any(np.isfinite(s["res_mean"])) else np.nan
                rmse_sg = float(np.sqrt(np.nanmean(np.square(s["res_sigma_total"].astype(float))))) if np.any(np.isfinite(s["res_sigma_total"])) else np.nan
                mae_mu = float(np.nanmean(s["abs_res_mean"].astype(float))) if np.any(np.isfinite(s["abs_res_mean"])) else np.nan
                mae_sg = float(np.nanmean(s["abs_res_sigma_total"].astype(float))) if np.any(np.isfinite(s["abs_res_sigma_total"])) else np.nan
                out_rows.append({
                    "target_id": tid,
                    "method": m,
                    "rmse_mean": rmse_mu,
                    "mae_mean": mae_mu,
                    "rmse_sigma_total": rmse_sg,
                    "mae_sigma_total": mae_sg,
                    "n_versions": int(s["version"].nunique()),
                })
    
        df_skill = pd.DataFrame(out_rows)
        df_res = j
    
        if export_table and output_path is not None:
            outp = Path(output_path)
            outp.mkdir(parents=True, exist_ok=True)
            imtU = str(imt).upper()
            df_res.to_csv(outp / f"{export_prefix}-{imtU}-residuals.csv", index=False)
            df_skill.to_csv(outp / f"{export_prefix}-{imtU}-skill.csv", index=False)
    
        return df_res, df_skill
    
    
    # ----------------------------------------------------------------------
    # 4.1-I) NEW: Residual plots (mean + sigma) per target (publish-ready)
    # ----------------------------------------------------------------------
    def uq_plot_published_vs_predicted_residuals(
        self,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        methods=("bayes", "hierarchical", "kriging", "montecarlo"),
        agg="mean",
        global_stat=None,
        prior_version=None,
        # passthrough controls
        sigma_total_from_shakemap=True,
        sigma_aleatory=None,
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
        ok_range_km=60.0,
        ok_variogram="exponential",
        ok_nugget=1e-6,
        ok_sill=None,
        ok_cap_sigma_to_prior=True,
        mc_nsim=2000,
        mc_include_aleatory=True,
        grid_res=None,
        interp_method="nearest",
        interp_kwargs=None,
        # plotting
        figsize=(11, 6),
        dpi=300,
        xrotation=45,
        ylog_sigma=False,
        ylim_mean=None,
        ylim_sigma=None,
        show_grid=True,
        title=None,
        legend=True,
        legend_kwargs=None,
        tight=True,
        output_path=None,
        save=False,
        save_formats=("png", "pdf"),
        show=True,
        # export residual tables
        export_tables=True,
        export_prefix="UQ-PublishedVsPredicted-Residuals",
        # audit
        audit=True,
        audit_output_path=None,
        audit_prefix=None,
    ):
        """
        Two-panel residual plot per target:
          - Panel A: (pred_mean - pub_mean)
          - Panel B: (pred_sigma_total - pub_sigma_total_raw)
    
        Uses uq_published_vs_predicted_residuals() internally.
        """
        if legend_kwargs is None:
            legend_kwargs = {}
        if interp_kwargs is None:
            interp_kwargs = {}
    
        df_res, df_skill = self.uq_published_vs_predicted_residuals(
            version_list=version_list,
            imt=imt,
            points=points,
            areas=areas,
            methods=methods,
            agg=agg,
            global_stat=global_stat,
            prior_version=prior_version,
            sigma_total_from_shakemap=sigma_total_from_shakemap,
            sigma_aleatory=sigma_aleatory,
            update_radius_km=update_radius_km,
            kernel=kernel,
            kernel_scale_km=kernel_scale_km,
            measurement_sigma=measurement_sigma,
            ok_range_km=ok_range_km,
            ok_variogram=ok_variogram,
            ok_nugget=ok_nugget,
            ok_sill=ok_sill,
            ok_cap_sigma_to_prior=ok_cap_sigma_to_prior,
            mc_nsim=mc_nsim,
            mc_include_aleatory=mc_include_aleatory,
            grid_res=grid_res,
            interp_method=interp_method,
            interp_kwargs=interp_kwargs,
            output_path=output_path,
            export_table=export_tables,
            export_prefix=export_prefix,
            audit=audit,
            audit_output_path=audit_output_path or output_path,
            audit_prefix=audit_prefix,
        )
    
        imtU = str(imt).upper()
        targets = sorted(df_res["target_id"].unique().tolist())
        meths = [str(m) for m in methods]
    
        for tid in targets:
            s = df_res[df_res["target_id"] == tid].copy().sort_values(["version", "method"])
    
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
    
            for m in meths:
                sm = s[s["method"] == m].sort_values("version")
                if sm.empty:
                    continue
                ax1.plot(sm["version"].values, sm["res_mean"].values, marker="o", linewidth=1.6, label=m)
                ax2.plot(sm["version"].values, sm["res_sigma_total"].values, marker="o", linewidth=1.6, label=m)
    
            ax1.axhline(0.0, linewidth=1.2)
            ax2.axhline(0.0, linewidth=1.2)
    
            ax1.set_ylabel(f"Δ mean (pred - pub) [{imtU}]")
            ax2.set_ylabel(f"Δ σ_total (pred - pub) [{imtU}]")
            ax2.set_xlabel("ShakeMap version")
            ax2.tick_params(axis="x", rotation=float(xrotation))
    
            if ylog_sigma:
                # NOTE: residuals can be negative; log-scale is generally not meaningful here.
                # If user insists, they should plot abs residuals instead.
                pass
    
            if ylim_mean is not None:
                ax1.set_ylim(*ylim_mean)
            if ylim_sigma is not None:
                ax2.set_ylim(*ylim_sigma)
    
            if show_grid:
                ax1.grid(True, alpha=0.3)
                ax2.grid(True, alpha=0.3)
    
            # annotate with skill summary (small text in axes)
            try:
                sk = df_skill[df_skill["target_id"] == tid].copy()
                if not sk.empty:
                    # make a compact note
                    lines = []
                    for _, r in sk.sort_values("method").iterrows():
                        lines.append(
                            f"{r['method']}: RMSEμ={r['rmse_mean']:.3g}, RMSEσ={r['rmse_sigma_total']:.3g}"
                        )
                    ax1.text(0.01, 0.02, "\n".join(lines), transform=ax1.transAxes, va="bottom", ha="left", fontsize=8)
            except Exception:
                pass
    
            if title is None:
                fig.suptitle(f"Published vs Predicted residuals — {imtU} — {tid}")
            else:
                fig.suptitle(str(title).replace("{target}", tid).replace("{imt}", imtU))
    
            if legend:
                fig.legend(loc="upper right", **legend_kwargs)
    
            if tight:
                fig.tight_layout(rect=[0, 0, 1, 0.95])
    
            if save and output_path is not None:
                self._uq_save_figure_safe(
                    fig,
                    fname_stem=f"UQ-PublishedVsPredicted-Residuals-{imtU}-{tid}",
                    subdir="uq_plots/published_vs_predicted_residuals",
                    output_path=output_path,
                    save_formats=save_formats,
                    dpi=dpi,
                )
    
            if show:
                plt.show()
            else:
                plt.close(fig)
    
        return df_res, df_skill

    
    
 
    # ----------------------------------------------------------------------
    # 4.1-G) Override: RAW vs UNIFIED curves with subsampling + exports + diagnostics
    #   IMPORTANT: This version builds its OWN unified grid (no uq_state, no get_unified_grid)
    # ----------------------------------------------------------------------
    def uq_compare_raw_vs_unified_curves(
        self,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        agg="mean",
        # raw discovery
        base_folder=None,
        shakemap_folder=None,
        # sampling
        raw_sample="nearest",         # for POINT targets: "nearest" | "bilinear"
        # unified settings
        grid_res=None,
        interp_method="nearest",
        interp_kwargs=None,
        # plotting
        figsize=(11, 5),
        dpi=250,
        xrotation=45,
        show_grid=True,
        title=None,
        legend=True,
        legend_kwargs=None,
        tight=True,
        output_path=None,
        save=False,
        save_formats=("png", "pdf"),
        show=True,
        # area band shading
        band=True,
        band_alpha=0.18,
        # subsampling RAW area mapping
        raw_area_max_cells=25000,     # None disables cap
        raw_area_stride=None,         # e.g., 2, 3, 5
        raw_area_random_state=42,
        raw_area_subsample_method="auto",  # "auto"|"stride"|"random"
        # export table
        export_table=True,
        export_prefix="UQ-RawVsUnifiedCurves",
    ):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path
    
        if legend_kwargs is None:
            legend_kwargs = {}
        if interp_kwargs is None:
            interp_kwargs = {}
    
        versions = [int(v) for v in (version_list or [])]
        if not versions:
            raise ValueError("version_list cannot be empty.")
    
        imtU = str(imt).upper().strip()
        stdU = f"STD{imtU}"
    
        # ------------------------------------------------------------
        # A) Discover + parse RAW XML once per version
        # ------------------------------------------------------------
        raw_cache = {}   # v -> dict with parsed arrays + field cols
        spacings = []
    
        def _pick_col(D, name_candidates):
            fields = D.get("fields", {}) or {}
            arr = D.get("data", None)
            if arr is None:
                return None
            for nm in name_candidates:
                if nm in fields:
                    col = int(fields[nm])
                    ncols = int(np.asarray(arr).shape[1])
                    if 0 <= col < ncols:
                        return col
                    # fallback: sometimes legacy mapping expected +2
                    if 0 <= (col + 2) < ncols:
                        return col + 2
            return None
    
        for v in versions:
            grid_p = self._uq_find_shakemap_xml(int(v), which="grid", base_folder=base_folder, shakemap_folder=shakemap_folder)
            unc_p  = self._uq_find_shakemap_xml(int(v), which="uncertainty", base_folder=base_folder, shakemap_folder=shakemap_folder)
    
            if grid_p is None or unc_p is None:
                raw_cache[int(v)] = {"grid_p": grid_p, "unc_p": unc_p, "ok": False, "reason": "missing_paths"}
                continue
    
            try:
                G = self._uq_parse_gridxml_to_arrays(grid_p)
                U = self._uq_parse_gridxml_to_arrays(unc_p)
    
                gcol = _pick_col(G, [imtU])
                ucol = _pick_col(U, [stdU, f"{imtU}_STD", f"STD_{imtU}"])
    
                if gcol is None or ucol is None:
                    raw_cache[int(v)] = {"grid_p": grid_p, "unc_p": unc_p, "ok": False, "reason": "missing_fields"}
                    continue
    
                latv = np.asarray(G.get("lat_vec", []), dtype=float)
                lonv = np.asarray(G.get("lon_vec", []), dtype=float)
                if latv.size < 2 or lonv.size < 2:
                    raw_cache[int(v)] = {"grid_p": grid_p, "unc_p": unc_p, "ok": False, "reason": "bad_latlon"}
                    continue
    
                # spacing estimate (abs median diff)
                dlat = float(np.nanmedian(np.abs(np.diff(latv))))
                dlon = float(np.nanmedian(np.abs(np.diff(lonv))))
                if np.isfinite(dlat) and dlat > 0:
                    spacings.append(dlat)
                if np.isfinite(dlon) and dlon > 0:
                    spacings.append(dlon)
    
                raw_cache[int(v)] = {
                    "grid_p": grid_p, "unc_p": unc_p, "ok": True,
                    "G": G, "U": U,
                    "gcol": int(gcol), "ucol": int(ucol),
                    "latv": latv, "lonv": lonv,
                    "nlat": int(G["nlat"]), "nlon": int(G["nlon"]),
                }
            except Exception as e:
                raw_cache[int(v)] = {"grid_p": grid_p, "unc_p": unc_p, "ok": False, "reason": f"parse_error:{e}"}
    
        # ------------------------------------------------------------
        # B) Build UNIFIED grid (intersection bbox) from RAW grids
        # ------------------------------------------------------------
        ok_versions = [v for v in versions if raw_cache.get(int(v), {}).get("ok", False)]
        if not ok_versions:
            raise RuntimeError("No versions had readable RAW grid+uncertainty. Cannot build unified grid.")
    
        # intersection bbox
        lon_min = max(float(raw_cache[v]["lonv"].min()) for v in ok_versions)
        lon_max = min(float(raw_cache[v]["lonv"].max()) for v in ok_versions)
        lat_min = max(float(raw_cache[v]["latv"].min()) for v in ok_versions)
        lat_max = min(float(raw_cache[v]["latv"].max()) for v in ok_versions)
    
        if not (lon_max > lon_min and lat_max > lat_min):
            raise RuntimeError("Intersection bbox is empty. RAW versions do not overlap in lon/lat.")
    
        # pick grid_res
        if grid_res is None:
            grid_res_use = float(min(spacings)) if spacings else 0.033333
        else:
            grid_res_use = float(grid_res)
    
        # build axes
        lon_vals = np.arange(lon_min, lon_max + 0.5 * grid_res_use, grid_res_use, dtype=float)
        lat_vals = np.arange(lat_min, lat_max + 0.5 * grid_res_use, grid_res_use, dtype=float)
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    
        # ------------------------------------------------------------
        # C) Interpolate RAW mean/sigma onto unified grid for each version
        #     (rectilinear grid => we can do nearest or bilinear without scipy)
        # ------------------------------------------------------------
        def _nearest_idx(vec, vals):
            vec = np.asarray(vec, dtype=float)
            vals = np.asarray(vals, dtype=float)
            inc = (vec[-1] >= vec[0])
            if not inc:
                vec2 = vec[::-1]
                idx = np.searchsorted(vec2, vals, side="left")
                idx = np.clip(idx, 0, vec2.size - 1)
                left = np.clip(idx - 1, 0, vec2.size - 1)
                choose_left = (np.abs(vec2[left] - vals) <= np.abs(vec2[idx] - vals))
                idx = np.where(choose_left, left, idx)
                return (vec.size - 1 - idx).astype(int)
            idx = np.searchsorted(vec, vals, side="left")
            idx = np.clip(idx, 0, vec.size - 1)
            left = np.clip(idx - 1, 0, vec.size - 1)
            choose_left = (np.abs(vec[left] - vals) <= np.abs(vec[idx] - vals))
            idx = np.where(choose_left, left, idx)
            return idx.astype(int)
    
        def _interp_nearest(cache_v, qlat2d, qlon2d, col_idx, use_unc=False):
            latv = cache_v["latv"]; lonv = cache_v["lonv"]
            nlon = cache_v["nlon"]
            arr = np.asarray(cache_v["U"]["data"] if use_unc else cache_v["G"]["data"], dtype=float)
    
            iy = _nearest_idx(latv, qlat2d.ravel())
            ix = _nearest_idx(lonv, qlon2d.ravel())
            ridx = iy * nlon + ix
            out = arr[ridx, int(col_idx)].astype(float)
            return out.reshape(qlat2d.shape)
    
        def _interp_bilinear(cache_v, qlat2d, qlon2d, col_idx, use_unc=False):
            # bilinear on rectilinear grid (assumes latv/lonv monotonic but can be decreasing)
            latv = np.asarray(cache_v["latv"], dtype=float)
            lonv = np.asarray(cache_v["lonv"], dtype=float)
            nlat = int(cache_v["nlat"]); nlon = int(cache_v["nlon"])
            arr = np.asarray(cache_v["U"]["data"] if use_unc else cache_v["G"]["data"], dtype=float)[:, int(col_idx)]
    
            # reshape flat (row-major lat-major like your ridx formula)
            Z = arr.reshape(nlat, nlon)
    
            # handle decreasing vectors by flipping
            lat_inc = (latv[-1] >= latv[0])
            lon_inc = (lonv[-1] >= lonv[0])
            if not lat_inc:
                latv2 = latv[::-1]
                Z = Z[::-1, :]
            else:
                latv2 = latv
            if not lon_inc:
                lonv2 = lonv[::-1]
                Z = Z[:, ::-1]
            else:
                lonv2 = lonv
    
            qlat = qlat2d.ravel()
            qlon = qlon2d.ravel()
    
            # clamp query into bounds
            qlat = np.clip(qlat, latv2.min(), latv2.max())
            qlon = np.clip(qlon, lonv2.min(), lonv2.max())
    
            # find cell indices
            iy1 = np.searchsorted(latv2, qlat, side="right") - 1
            ix1 = np.searchsorted(lonv2, qlon, side="right") - 1
            iy1 = np.clip(iy1, 0, nlat - 2)
            ix1 = np.clip(ix1, 0, nlon - 2)
            iy2 = iy1 + 1
            ix2 = ix1 + 1
    
            y1 = latv2[iy1]; y2 = latv2[iy2]
            x1 = lonv2[ix1]; x2 = lonv2[ix2]
    
            # weights
            wy = np.where((y2 - y1) != 0, (qlat - y1) / (y2 - y1), 0.0)
            wx = np.where((x2 - x1) != 0, (qlon - x1) / (x2 - x1), 0.0)
    
            z11 = Z[iy1, ix1]
            z12 = Z[iy1, ix2]
            z21 = Z[iy2, ix1]
            z22 = Z[iy2, ix2]
    
            z = (1 - wy) * ((1 - wx) * z11 + wx * z12) + wy * ((1 - wx) * z21 + wx * z22)
            return z.reshape(qlat2d.shape)
    
        im = str(interp_method).lower().strip()
        if im not in ("nearest", "bilinear", "linear", "cubic"):
            im = "nearest"
        use_bilinear = (im in ("bilinear", "linear", "cubic"))
    
        uni_cache = {}  # v -> (mu2d, sg2d)
        for v in versions:
            cv = raw_cache.get(int(v), {})
            if not cv.get("ok", False):
                uni_cache[int(v)] = (np.full(lat2d.shape, np.nan), np.full(lat2d.shape, np.nan))
                continue
            if use_bilinear:
                mu2d = _interp_bilinear(cv, lat2d, lon2d, cv["gcol"], use_unc=False)
                sg2d = _interp_bilinear(cv, lat2d, lon2d, cv["ucol"], use_unc=True)
            else:
                mu2d = _interp_nearest(cv, lat2d, lon2d, cv["gcol"], use_unc=False)
                sg2d = _interp_nearest(cv, lat2d, lon2d, cv["ucol"], use_unc=True)
            uni_cache[int(v)] = (np.asarray(mu2d, dtype=float), np.asarray(sg2d, dtype=float))
    
        # ------------------------------------------------------------
        # D) Main loop: targets, curves, RAW sampling, plots
        # ------------------------------------------------------------
        targets = self._uq_parse_targets(points=points, areas=areas)
        rows = []
    
        for t in targets:
            tid = t.get("id", "target")
            mask, meta = self._uq_target_mask(t, lat2d, lon2d)
            is_area_like = (t.get("type") == "area")
    
            if t.get("type") == "point":
                t_lat = float(t["lat"])
                t_lon = float(t["lon"])
    
            for v in versions:
                mu_u, sg_u = uni_cache[int(v)]
                uni_mu_agg = self._uq_agg(mu_u[mask], agg=agg)
                uni_sg_agg = self._uq_agg(sg_u[mask], agg=agg)
    
                if band and is_area_like:
                    uni_mu_min = float(np.nanmin(mu_u[mask])) if np.any(np.isfinite(mu_u[mask])) else np.nan
                    uni_mu_max = float(np.nanmax(mu_u[mask])) if np.any(np.isfinite(mu_u[mask])) else np.nan
                    uni_sg_min = float(np.nanmin(sg_u[mask])) if np.any(np.isfinite(sg_u[mask])) else np.nan
                    uni_sg_max = float(np.nanmax(sg_u[mask])) if np.any(np.isfinite(sg_u[mask])) else np.nan
                else:
                    uni_mu_min = uni_mu_max = np.nan
                    uni_sg_min = uni_sg_max = np.nan
    
                # RAW paths
                grid_p = raw_cache.get(int(v), {}).get("grid_p", None)
                unc_p  = raw_cache.get(int(v), {}).get("unc_p", None)
    
                raw_mu_agg = raw_sg_agg = np.nan
                raw_mu_min = raw_mu_max = np.nan
                raw_sg_min = raw_sg_max = np.nan
                subs_meta = {}
    
                if (grid_p is not None) and (unc_p is not None):
                    if t.get("type") == "point":
                        raw = self._uq_raw_sample_from_xml(grid_p, unc_p, t_lat, t_lon, imt=imtU, sample=raw_sample)
                        if raw is not None:
                            raw_mu_agg = float(raw.get("raw_mean", np.nan))
                            raw_sg_agg = float(raw.get("raw_sigma", np.nan))
                            raw_mu_min = raw_mu_max = raw_mu_agg
                            raw_sg_min = raw_sg_max = raw_sg_agg
                            subs_meta = {
                                "raw_sample": str(raw_sample),
                                "raw_nearest_km": float(raw.get("nearest_km", np.nan)),
                                "status": raw.get("error", None),
                            }
                    else:
                        raw_mu_vals, raw_sg_vals, subs_meta = self._uq_area_raw_values_from_xml(
                            grid_p, unc_p, imtU, lat2d, lon2d, mask,
                            max_cells=raw_area_max_cells,
                            stride=raw_area_stride,
                            random_state=raw_area_random_state,
                            subsample_method=raw_area_subsample_method,
                        )
                        raw_mu_agg = self._uq_agg(raw_mu_vals, agg=agg)
                        raw_sg_agg = self._uq_agg(raw_sg_vals, agg=agg)
    
                        if band:
                            vv = raw_mu_vals[np.isfinite(raw_mu_vals)]
                            ww = raw_sg_vals[np.isfinite(raw_sg_vals)]
                            raw_mu_min = float(np.nanmin(vv)) if vv.size else np.nan
                            raw_mu_max = float(np.nanmax(vv)) if vv.size else np.nan
                            raw_sg_min = float(np.nanmin(ww)) if ww.size else np.nan
                            raw_sg_max = float(np.nanmax(ww)) if ww.size else np.nan
    
                rows.append({
                    "target_id": tid,
                    "target_type": t.get("type", "target"),
                    "version": int(v),
                    "imt": imtU,
                    "agg": str(agg),
                    "unified_mean": float(uni_mu_agg),
                    "unified_sigma": float(uni_sg_agg),
                    "unified_mean_min": float(uni_mu_min),
                    "unified_mean_max": float(uni_mu_max),
                    "unified_sigma_min": float(uni_sg_min),
                    "unified_sigma_max": float(uni_sg_max),
                    "raw_mean": float(raw_mu_agg),
                    "raw_sigma": float(raw_sg_agg),
                    "raw_mean_min": float(raw_mu_min),
                    "raw_mean_max": float(raw_mu_max),
                    "raw_sigma_min": float(raw_sg_min),
                    "raw_sigma_max": float(raw_sg_max),
                    "raw_grid_path": str(grid_p) if grid_p else None,
                    "raw_unc_path": str(unc_p) if unc_p else None,
                    "mask_kind": meta.get("kind", ""),
                    "n_cells": int(meta.get("n_cells", int(mask.sum()))),
                    "raw_area_subsample_method": subs_meta.get("method", None),
                    "raw_area_n_cells": subs_meta.get("n_cells", None),
                    "raw_area_n_returned": subs_meta.get("n_returned", None),
                    "raw_area_stride": subs_meta.get("stride", None),
                    "raw_area_max_cells": subs_meta.get("max_cells", None),
                    "raw_status": subs_meta.get("status", None),
                    "raw_missing_field": subs_meta.get("missing", None),
                    "raw_nearest_km": subs_meta.get("raw_nearest_km", None),
                    # extra debug (does not break anything)
                    "unified_grid_res_used": float(grid_res_use),
                    "unified_interp_method_used": str(im),
                })
    
            # ---- Plot per target ----
            df_t = pd.DataFrame([r for r in rows if r["target_id"] == tid]).sort_values("version")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
            x = df_t["version"].values
    
            if band and is_area_like:
                ax1.fill_between(x, df_t["raw_mean_min"].values, df_t["raw_mean_max"].values, alpha=float(band_alpha), label="RAW min/max (area)")
                ax1.fill_between(x, df_t["unified_mean_min"].values, df_t["unified_mean_max"].values, alpha=float(band_alpha), label="UNIFIED min/max (area)")
            ax1.plot(x, df_t["raw_mean"].values, marker="o", linewidth=2.0, label="RAW mean")
            ax1.plot(x, df_t["unified_mean"].values, marker="o", linewidth=2.0, label="UNIFIED mean")
            ax1.set_ylabel(f"{imtU} mean")
            if show_grid:
                ax1.grid(True, alpha=0.3)
    
            if band and is_area_like:
                ax2.fill_between(x, df_t["raw_sigma_min"].values, df_t["raw_sigma_max"].values, alpha=float(band_alpha), label="RAW σ min/max (area)")
                ax2.fill_between(x, df_t["unified_sigma_min"].values, df_t["unified_sigma_max"].values, alpha=float(band_alpha), label="UNIFIED σ min/max (area)")
            ax2.plot(x, df_t["raw_sigma"].values, marker="o", linewidth=2.0, label="RAW σ_total (STD*)")
            ax2.plot(x, df_t["unified_sigma"].values, marker="o", linewidth=2.0, label="UNIFIED σ_total (STD*)")
            ax2.set_ylabel(f"{imtU} σ_total")
            ax2.set_xlabel("ShakeMap version")
            ax2.tick_params(axis="x", rotation=float(xrotation))
            if show_grid:
                ax2.grid(True, which="both", alpha=0.3)
    
            if title is None:
                fig.suptitle(f"RAW vs UNIFIED curves — {imtU} — {tid}")
            else:
                fig.suptitle(str(title).replace("{target}", tid).replace("{imt}", imtU))
    
            if legend:
                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                all_h = handles1 + [h for h, lab in zip(handles2, labels2) if lab not in labels1]
                all_l = labels1 + [lab for lab in labels2 if lab not in labels1]
                fig.legend(all_h, all_l, loc="upper right", **(legend_kwargs or {}))
    
            if tight:
                fig.tight_layout(rect=[0, 0, 1, 0.95])
    
            if save and output_path is not None:
                self._uq_save_figure_safe(
                    fig,
                    fname_stem=f"{export_prefix}-{imtU}-{tid}",
                    subdir="uq_plots/raw_vs_unified_curves",
                    output_path=output_path,
                    save_formats=save_formats,
                    dpi=dpi,
                )
    
            if show:
                plt.show()
            else:
                plt.close(fig)
    
        df_curves = pd.DataFrame(rows)
    
        # -----------------------------
        # Export CSV to export/SHAKEuq/<event_id>/
        # -----------------------------
        if export_table and output_path is not None:
            out_root = Path(output_path)
    
            eid = getattr(self, "event_id", None) or getattr(self, "eventid", None) or getattr(self, "event", None)
            if isinstance(eid, dict):
                eid = eid.get("id", None)
            if eid is None and hasattr(self, "uq_state") and isinstance(self.uq_state, dict):
                eid = self.uq_state.get("event_id", None) or self.uq_state.get("eventid", None)
            if eid is None:
                eid = "unknown_event"
    
            outp = out_root / "SHAKEuq" / str(eid) / "uq"
            outp.mkdir(parents=True, exist_ok=True)
            df_curves.to_csv(outp / f"{export_prefix}-{imtU}.csv", index=False)
    
        return df_curves


    # ======================================================================
    # PATCH 4.2 (REPLACEMENT): Posterior Save/Load Infrastructure + Robust UQ Map Driver
    #
    # This patch REPLACES your previous Patch 4.2.
    # Paste this at the END of class SHAKEuq (later defs override earlier ones).
    #
    # What this patch fixes vs your previous 4.2:
    # -----------------------------------------
    # 1) Bayes loader signature mismatch:
    #    - Supports legacy _uq_load_bayes_posterior_npz() whether it expects (imt) OR (version, imt).
    #
    # 2) Cartopy extent / wrong map projection handling:
    #    - Uses transform=ccrs.PlateCarree() for lon/lat plotting on Cartopy GeoAxes.
    #    - For panel plots, creates Cartopy subplots when cartopy is available and sets extent.
    #
    # 3) "Compute -> Save -> Load" inside the mapper:
    #    - For bayesupdate/hierarchical/kriging the mapper can compute results, save standardized NPZ,
    #      then load them back via a single generic loader.
    #
    # Supported methods:
    # ------------------
    #   - "shakemap"   : prior + change (published sigma on unified grid)
    #   - "bayesupdate": post + reduction + change (computed/exported on demand)
    #   - "hierarchical": post + reduction + change (computed on demand via grid wrapper in this patch)
    #   - "kriging"    : post + reduction + change (computed on demand via grid wrapper in this patch)
    #
    # Supported kinds:
    # ----------------
    #   kind = "prior" | "post" | "reduction" | "change"
    #
    # Panel plotting:
    # --------------
    #   panel_all=True produces a multi-panel figure.
    #   - If kind in ("prior","post","reduction"): one panel per version
    #   - If kind=="change": panels follow panel_mode:
    #         "cumulative"  : first -> each
    #         "sequential"  : (i-1) -> i
    #     plus optional last-vs-first extra panel.
    #
    # ======================================================================
    
    # ----------------------------------------------------------------------
    # 0) Small normalizers
    # ----------------------------------------------------------------------
    def _uq_method_norm(self, method: str) -> str:
        m = str(method).strip().lower()
        alias = {
            "sm": "shakemap",
            "shake": "shakemap",
            "raw": "shakemap",
            "published": "shakemap",
            "bayes": "bayesupdate",
            "bayes_update": "bayesupdate",
            "bayesupdate": "bayesupdate",
            "hier": "hierarchical",
            "hierarchical": "hierarchical",
            "ok": "kriging",
            "ordinarykriging": "kriging",
            "kriging": "kriging",
        }
        return alias.get(m, m)
    
    def _uq_imt_norm(self, imt: str) -> str:
        return str(imt).upper().strip()
    
    def _uq_which_sigma_norm(self, which_sigma: str) -> str:
        ws = str(which_sigma).lower().strip()
        if ws not in ("epistemic", "total"):
            raise ValueError("which_sigma must be 'epistemic' or 'total'.")
        return ws
    
    def _uq_kind_norm(self, kind: str) -> str:
        k = str(kind).lower().strip()
        if k not in ("prior", "post", "reduction", "change"):
            raise ValueError("kind must be: 'prior' | 'post' | 'reduction' | 'change'.")
        return k
    
    # ----------------------------------------------------------------------
    # 1) Posterior persistence: standardized NPZ schema (method-agnostic)
    # ----------------------------------------------------------------------
    def _uq_param_hash(self, method: str, imt: str, version: int, params: dict) -> str:
        import json, hashlib
        m = self._uq_method_norm(method)
        imtU = self._uq_imt_norm(imt)
        payload = {"method": m, "imt": imtU, "version": int(version), "params": params or {}}
        s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(s).hexdigest()[:10]
    
    def _uq_posterior_outdir(self, output_path: str = None) -> str:
        from pathlib import Path
        if output_path is None:
            if hasattr(self, "uq_state") and isinstance(self.uq_state, dict):
                bf = self.uq_state.get("base_folder", None)
                if bf:
                    output_path = str(bf)
        if output_path is None:
            output_path = "./export"
        p = Path(output_path) / "SHAKEuq" / str(self.event_id) / "uq" / "uq_posteriors"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    
    def _uq_save_posterior_npz(
        self,
        *,
        method: str,
        imt: str,
        version: int,
        lat2d,
        lon2d,
        sigma_ep_prior=None,
        sigma_total_prior=None,
        sigma_ep_post=None,
        sigma_total_post=None,
        mu_prior_ws=None,
        mu_post_ws=None,
        meta: dict = None,
        output_path: str = None,
        filename: str = None,
    ):
        import json
        import numpy as np
        from pathlib import Path
    
        m = self._uq_method_norm(method)
        imtU = self._uq_imt_norm(imt)
        v = int(version)
    
        outdir = Path(self._uq_posterior_outdir(output_path))
        if filename is None:
            h = None
            if meta and isinstance(meta, dict) and "param_hash" in meta:
                h = meta["param_hash"]
            filename = f"{m}_{imtU}_v{v:03d}_{h}.npz" if h else f"{m}_{imtU}_v{v:03d}.npz"
    
        arr = {
            "lat2d": np.asarray(lat2d, dtype=float),
            "lon2d": np.asarray(lon2d, dtype=float),
        }
        if sigma_ep_prior is not None:
            arr["sigma_ep_prior"] = np.asarray(sigma_ep_prior, dtype=float)
        if sigma_total_prior is not None:
            arr["sigma_total_prior"] = np.asarray(sigma_total_prior, dtype=float)
        if sigma_ep_post is not None:
            arr["sigma_ep_post"] = np.asarray(sigma_ep_post, dtype=float)
        if sigma_total_post is not None:
            arr["sigma_total_post"] = np.asarray(sigma_total_post, dtype=float)
        if mu_prior_ws is not None:
            arr["mu_prior_ws"] = np.asarray(mu_prior_ws, dtype=float)
        if mu_post_ws is not None:
            arr["mu_post_ws"] = np.asarray(mu_post_ws, dtype=float)
    
        meta0 = dict(meta or {})
        meta0.update({"method": m, "imt": imtU, "version": v, "event_id": str(self.event_id)})
        arr["meta_json"] = np.array(json.dumps(meta0, sort_keys=True, default=str), dtype=object)
    
        outpath = outdir / filename
        np.savez_compressed(outpath, **arr)
        return str(outpath)
    
    def _uq_find_latest_posterior_file(self, *, method: str, imt: str, version: int, output_path: str = None):
        import glob, os
        from pathlib import Path
    
        m = self._uq_method_norm(method)
        imtU = self._uq_imt_norm(imt)
        v = int(version)
    
        d = Path(self._uq_posterior_outdir(output_path))
        pat = str(d / f"{m}_{imtU}_v{v:03d}*.npz")
        files = glob.glob(pat)
        if not files:
            return None
        files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        return files[0]
    
    # ======================================================================
    # Patch 4.2-PFIX2: Robust posterior NPZ loader (search multiple roots)
    # ======================================================================
    def _uq_load_posterior_npz(self, *, method: str, imt: str, version: int, output_path: str = None):
        import json
        import numpy as np
        from pathlib import Path
    
        m = self._uq_method_norm(method)
        imtU = self._uq_imt_norm(imt)
        v = int(version)
    
        # Candidate roots to search (in order)
        roots = []
        if output_path:
            roots.append(str(output_path))
    
        # uq_state base_folder (if exists)
        try:
            if hasattr(self, "uq_state") and isinstance(self.uq_state, dict):
                bf = self.uq_state.get("base_folder", None)
                if bf:
                    roots.append(str(bf))
        except Exception:
            pass
    
        # default fallback
        roots.append("./export")
    
        # Search each root for newest matching file
        best_file = None
        best_mtime = None
        for rt in roots:
            try:
                d = Path(self._uq_posterior_outdir(rt))
                pat = str(d / f"{m}_{imtU}_v{v:03d}*.npz")
                import glob, os
                files = glob.glob(pat)
                if not files:
                    continue
                files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
                f0 = files[0]
                mt = os.path.getmtime(f0)
                if (best_mtime is None) or (mt > best_mtime):
                    best_mtime = mt
                    best_file = f0
            except Exception:
                continue
    
        if best_file is None:
            return None
    
        dat = dict(np.load(best_file, allow_pickle=True))
        meta = {}
        if "meta_json" in dat:
            try:
                meta = json.loads(str(dat["meta_json"].item()))
            except Exception:
                meta = {}
        dat["_meta"] = meta
        dat["_file"] = best_file
        return dat







    
    # ----------------------------------------------------------------------
    # 2) Legacy bayes loader signature compatibility + standard export bridge
    # ----------------------------------------------------------------------
    def _uq_call_legacy_bayes_loader(self, version: int, imt: str):
        """
        Call _uq_load_bayes_posterior_npz with whichever signature exists:
          - (imt)
          - (version, imt)
        Returns whatever the legacy loader returns, or None.
        """
        import inspect
    
        if not hasattr(self, "_uq_load_bayes_posterior_npz"):
            return None
    
        fn = self._uq_load_bayes_posterior_npz
        try:
            sig = inspect.signature(fn)
            nreq = 0
            for p in sig.parameters.values():
                if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                    nreq += 1
        except Exception:
            # if signature introspection fails, just try both
            nreq = None
    
        # Try the most likely patterns safely
        try:
            if nreq == 2:
                return fn(imt)
        except TypeError:
            pass
        try:
            return fn(int(version), imt)
        except TypeError:
            try:
                return fn(imt)
            except Exception:
                return None
        except Exception:
            return None
    
    def _uq_extract_sigma_from_legacy_dict(self, legacy: dict):
        """
        Extract sigma arrays from a legacy bayes/hier dict using flexible key names.
        Returns (sigma_ep_post, sigma_total_post, mu_post_ws) or (None, None, None).
        """
        if legacy is None or not isinstance(legacy, dict):
            return None, None, None
    
        # common possibilities
        ep = legacy.get("sigma_ep_post", None)
        if ep is None:
            ep = legacy.get("sigma_epistemic_post", None)
    
        tot = legacy.get("sigma_total_post", None)
        if tot is None:
            tot = legacy.get("sigma_post", None)
    
        mu = legacy.get("mu_post_ws", None)
        if mu is None:
            mu = legacy.get("mu_post", None)
        if mu is None:
            mu = legacy.get("mu_post_mean", None)
    
        return ep, tot, mu
    
    def _uq_export_standard_posterior_from_legacy(
        self,
        *,
        method: str,
        imt: str,
        version: int,
        legacy: dict,
        output_path: str,
        meta: dict,
    ):
        """
        Bridge: take whatever legacy loader produced and re-save in standardized NPZ format.
        """
        import numpy as np
    
        axes = self.uq_state["unified_axes"]
        lat2d = np.asarray(axes["lat2d"], dtype=float)
        lon2d = np.asarray(axes["lon2d"], dtype=float)
    
        sig_ep0, sig_tot0, _sig_a = self._uq_prior_sigma_fields(int(version), self._uq_imt_norm(imt))
        ep_post, tot_post, mu_post_ws = self._uq_extract_sigma_from_legacy_dict(legacy)
    
        if ep_post is None or tot_post is None:
            return None
    
        return self._uq_save_posterior_npz(
            method=method,
            imt=imt,
            version=int(version),
            lat2d=lat2d,
            lon2d=lon2d,
            sigma_ep_prior=sig_ep0,
            sigma_total_prior=sig_tot0,
            sigma_ep_post=ep_post,
            sigma_total_post=tot_post,
            mu_prior_ws=None,
            mu_post_ws=mu_post_ws,
            meta=meta,
            output_path=output_path,
        )
    
    # ----------------------------------------------------------------------
    # 3) Full-grid hierarchical + kriging builders (compute -> standardized NPZ)
    # ----------------------------------------------------------------------
    def uq_hierarchical_update_grid(
        self,
        *,
        version_list,
        imt: str = "MMI",
        update_radius_km: float = 30.0,
        kernel: str = "gaussian",
        kernel_scale_km: float = 20.0,
        measurement_sigma: float = 0.30,
        output_path: str = None,
        export: bool = True,
    ):
        import numpy as np
    
        imtU = self._uq_imt_norm(imt)
        vlist = sorted([int(v) for v in (version_list or [])])
        if not vlist:
            raise ValueError("version_list is required.")
    
        axes = self.uq_state["unified_axes"]
        lat2d = np.asarray(axes["lat2d"], dtype=float)
        lon2d = np.asarray(axes["lon2d"], dtype=float)
        target_mask = np.isfinite(lat2d) & np.isfinite(lon2d)
    
        out = {}
        for v in vlist:
            sig_ep0, sig_tot0, sig_a = self._uq_prior_sigma_fields(int(v), imtU)
    
            # Best-effort prior mean field on unified grid
            mu0_ws = np.full_like(sig_ep0, np.nan, dtype=float)
            try:
                vkey = str(int(v)).zfill(3)
                dfv = self.uq_state["versions_unified"][vkey]
                col = f"{imtU.lower()}"
                if col in dfv.columns:
                    lats = np.sort(dfv["lat"].unique())
                    lons = np.sort(dfv["lon"].unique())
                    mu0_ws = dfv[col].values.reshape(len(lats), len(lons))
            except Exception:
                pass
    
            obs_list, _counts = self._uq_collect_obs_for_version(
                int(v),
                imtU,
                measurement_sigma=float(measurement_sigma),
                include_weights=True,
                prefer_domain=True,
                allow_fallback=True,
            )
    
            mu_post_ws, sig_ep_post = self._uq_hierarchical_posterior_at_mask(
                mu0_ws=mu0_ws,
                sigma_ep0=sig_ep0,
                sigma_a=sig_a,
                lat2d=lat2d,
                lon2d=lon2d,
                target_mask=target_mask,
                obs_list=obs_list,
                update_radius_km=float(update_radius_km),
                kernel=str(kernel),
                kernel_scale_km=float(kernel_scale_km),
                measurement_sigma=float(measurement_sigma),
            )
    
            sig_tot_post = np.sqrt(np.asarray(sig_ep_post, float) ** 2 + np.asarray(sig_a, float) ** 2)
    
            meta = {
                "update_radius_km": float(update_radius_km),
                "kernel": str(kernel),
                "kernel_scale_km": float(kernel_scale_km),
                "measurement_sigma": float(measurement_sigma),
            }
            meta["param_hash"] = self._uq_param_hash("hierarchical", imtU, int(v), meta)
    
            if export:
                self._uq_save_posterior_npz(
                    method="hierarchical",
                    imt=imtU,
                    version=int(v),
                    lat2d=lat2d,
                    lon2d=lon2d,
                    sigma_ep_prior=sig_ep0,
                    sigma_total_prior=sig_tot0,
                    sigma_ep_post=sig_ep_post,
                    sigma_total_post=sig_tot_post,
                    mu_prior_ws=mu0_ws,
                    mu_post_ws=mu_post_ws,
                    meta=meta,
                    output_path=output_path,
                )
    
            out[int(v)] = {"sigma_ep_post": sig_ep_post, "sigma_total_post": sig_tot_post, "meta": meta}
    
        return out
    
    def uq_kriging_update_grid(
        self,
        *,
        version_list,
        imt: str = "MMI",
        variogram: str = "exponential",
        range_km: float = 60.0,
        nugget: float = 1e-6,
        sill=None,
        measurement_sigma: float = 0.30,
        sigma_ep_cap_to_prior: bool = True,
        output_path: str = None,
        export: bool = True,
    ):
        import numpy as np
    
        imtU = self._uq_imt_norm(imt)
        vlist = sorted([int(v) for v in (version_list or [])])
        if not vlist:
            raise ValueError("version_list is required.")
    
        axes = self.uq_state["unified_axes"]
        lat2d = np.asarray(axes["lat2d"], dtype=float)
        lon2d = np.asarray(axes["lon2d"], dtype=float)
        target_mask = np.isfinite(lat2d) & np.isfinite(lon2d)
    
        out = {}
        for v in vlist:
            sig_ep0, sig_tot0, sig_a = self._uq_prior_sigma_fields(int(v), imtU)
    
            mu0_ws = np.full_like(sig_ep0, np.nan, dtype=float)
            try:
                vkey = str(int(v)).zfill(3)
                dfv = self.uq_state["versions_unified"][vkey]
                col = f"{imtU.lower()}"
                if col in dfv.columns:
                    lats = np.sort(dfv["lat"].unique())
                    lons = np.sort(dfv["lon"].unique())
                    mu0_ws = dfv[col].values.reshape(len(lats), len(lons))
            except Exception:
                pass
    
            obs_list, _counts = self._uq_collect_obs_for_version(
                int(v),
                imtU,
                measurement_sigma=float(measurement_sigma),
                include_weights=True,
                prefer_domain=True,
                allow_fallback=True,
            )
    
            mu_post_ws, sig_ep_post = self._uq_ok_residual_posterior_at_mask(
                mu0_ws=mu0_ws,
                sigma_ep0=sig_ep0,
                sigma_a=sig_a,
                lat2d=lat2d,
                lon2d=lon2d,
                target_mask=target_mask,
                obs_list=obs_list,
                variogram=str(variogram),
                range_km=float(range_km),
                nugget=float(nugget),
                sill=sill,
                measurement_sigma=float(measurement_sigma),
                sigma_ep_cap_to_prior=bool(sigma_ep_cap_to_prior),
            )
    
            sig_tot_post = np.sqrt(np.asarray(sig_ep_post, float) ** 2 + np.asarray(sig_a, float) ** 2)
    
            meta = {
                "variogram": str(variogram),
                "range_km": float(range_km),
                "nugget": float(nugget),
                "sill": None if sill is None else float(sill),
                "measurement_sigma": float(measurement_sigma),
                "sigma_ep_cap_to_prior": bool(sigma_ep_cap_to_prior),
            }
            meta["param_hash"] = self._uq_param_hash("kriging", imtU, int(v), meta)
    
            if export:
                self._uq_save_posterior_npz(
                    method="kriging",
                    imt=imtU,
                    version=int(v),
                    lat2d=lat2d,
                    lon2d=lon2d,
                    sigma_ep_prior=sig_ep0,
                    sigma_total_prior=sig_tot0,
                    sigma_ep_post=sig_ep_post,
                    sigma_total_post=sig_tot_post,
                    mu_prior_ws=mu0_ws,
                    mu_post_ws=mu_post_ws,
                    meta=meta,
                    output_path=output_path,
                )
    
            out[int(v)] = {"sigma_ep_post": sig_ep_post, "sigma_total_post": sig_tot_post, "meta": meta}
    
        return out
    
    # ----------------------------------------------------------------------
    # 4) UPDATED: Posterior sigma field accessor (robust + standardized first)
    # ----------------------------------------------------------------------
    def _uq_posterior_sigma_fields(self, version, imt, method):
        """
        Return (sigma_ep_post_2d, sigma_total_post_2d, meta_dict).
    
        Load order:
          1) standardized uq_posteriors NPZ (any method)
          2) legacy bayes loader (signature-flex) for bayesupdate, then bridge-save optional
        """
        import numpy as np
    
        v = int(version)
        imtU = self._uq_imt_norm(imt)
        m = self._uq_method_norm(method)
    
        # 1) standardized store
        dat = self._uq_load_posterior_npz(method=m, imt=imtU, version=v, output_path=None)
        if dat is not None and ("sigma_ep_post" in dat) and ("sigma_total_post" in dat):
            return (
                np.asarray(dat["sigma_ep_post"], float),
                np.asarray(dat["sigma_total_post"], float),
                dict(dat.get("_meta", {})),
            )
    
        # 2) legacy bayes loader support (only if method is bayesupdate)
        if m == "bayesupdate":
            legacy = self._uq_call_legacy_bayes_loader(v, imtU)
            if legacy is not None and isinstance(legacy, dict):
                ep, tot, _mu = self._uq_extract_sigma_from_legacy_dict(legacy)
                if ep is not None and tot is not None:
                    meta = {}
                    try:
                        meta = dict(legacy.get("_meta", {}))
                    except Exception:
                        meta = {}
                    return np.asarray(ep, float), np.asarray(tot, float), meta
    
        # 3) hierarchical/kriging should already be in standardized store (computed by this patch)
        return None, None, {}
    
    # ----------------------------------------------------------------------
    # 5) Cartopy-safe plotting helpers (transform + extent)
    # ----------------------------------------------------------------------
    def _uq_get_cartopy_transform(self):
        try:
            import cartopy.crs as ccrs
            return ccrs.PlateCarree()
        except Exception:
            return None
    
    def _uq_is_cartopy_geoaxes(self, ax) -> bool:
        # Light check; cartopy GeoAxes typically has attribute 'projection'
        return hasattr(ax, "projection")
    
    def _uq_set_ax_extent_lonlat(self, ax, lon2d, lat2d, pad: float = 0.0):
        """
        For cartopy axes: set extent in PlateCarree.
        """
        tr = self._uq_get_cartopy_transform()
        if tr is None or not self._uq_is_cartopy_geoaxes(ax):
            return
        import numpy as np
        lo1 = float(np.nanmin(lon2d))
        lo2 = float(np.nanmax(lon2d))
        la1 = float(np.nanmin(lat2d))
        la2 = float(np.nanmax(lat2d))
        if pad and pad > 0:
            lo_pad = (lo2 - lo1) * pad
            la_pad = (la2 - la1) * pad
            lo1 -= lo_pad; lo2 += lo_pad
            la1 -= la_pad; la2 += la_pad
        ax.set_extent([lo1, lo2, la1, la2], crs=tr)
    


    # ----------------------------------------------------------------------
    # 6) NEW: Unified UQ sigma map driver (compute -> save -> load -> plot)
    # ----------------------------------------------------------------------
    def uq_plot_uq_sigma_map(
        self,
        *,
        version_list,
        imt: str = "MMI",
        method: str = "shakemap",            # "shakemap" | "bayesupdate" | "hierarchical" | "kriging"
        which_sigma: str = "epistemic",      # "epistemic" | "total"
        kind: str = "prior",                 # "prior" | "post" | "reduction" | "change"
        # version controls
        version: int = None,                 # used for prior/post/reduction
        first_version: int = None,           # used for change; defaults to min(version_list)
        last_version: int = None,            # used for change; defaults to max(version_list)
        # dataset build controls
        build_dataset_if_missing: bool = True,
        base_folder: str = "./export/SHAKEuq",
        stations_folder: str = None,
        rupture_folder: str = None,
        grid_unify: str = "intersection",
        resolution: str = "finest",
        interp_method: str = "nearest",
        interp_kwargs: dict = None,
        export_dataset: bool = True,
        # compute controls (on-demand)
        compute_if_missing: bool = True,
        export_posteriors: bool = True,
        posterior_output_path: str = None,
        # bayes knobs (passed through to uq_bayes_update)
        update_radius_km: float = 25.0,
        update_kernel: str = "gaussian",
        sigma_aleatory=None,
        sigma_total_from_shakemap: bool = True,
        measurement_sigma=None,
        make_audit: bool = True,
        # hierarchical knobs
        hier_update_radius_km: float = 30.0,
        hier_kernel: str = "gaussian",
        hier_kernel_scale_km: float = 20.0,
        hier_measurement_sigma: float = 0.30,
        # kriging knobs
        ok_variogram: str = "exponential",
        ok_range_km: float = 60.0,
        ok_nugget: float = 1e-6,
        ok_sill=None,
        ok_measurement_sigma: float = 0.30,
        ok_cap_sigma_to_prior: bool = True,
        # plotting knobs
        cmap: str = "viridis_r",
        vmin=None,
        vmax=None,
        plot_colorbar: bool = True,
        cbar_label: str = None,
        cbar_orientation: str = "vertical",
        show_obs: bool = True,
        obs_size: float = 10.0,
        mask_to_data_radius_km: float = 0.0,
        figsize=(10, 8),
        dpi: int = 300,
        title: str = None,
        show_title: bool = True,
        # save/show
        output_path: str = "./export",
        save: bool = True,
        save_formats=("png", "pdf"),
        show: bool = True,
        # panel mode
        panel_all: bool = False,
        panel_mode: str = "cumulative",      # "cumulative" | "sequential"
        panel_ncol: int = 3,
        panel_figsize=None,
        include_last_vs_first_panel: bool = True,
    
        # ------------------------------------------------------------------
        # NEW (v26.5 patch): panel saving + spacing + colorbar placement
        # ------------------------------------------------------------------
        panel_save_mode: str = "panel",       # "panel" | "split"  (split saves each panel as its own figure)
        panel_split_prefix: str = None,        # optional override prefix for split figure filenames
    
        # subplot spacing controls (panel figure)
        panel_wspace: float = None,           # subplot horizontal spacing
        panel_hspace: float = None,           # subplot vertical spacing
        panel_left: float = None,             # subplots_adjust(left=...)
        panel_right: float = None,            # subplots_adjust(right=...)
        panel_bottom: float = None,           # subplots_adjust(bottom=...)
        panel_top: float = None,              # subplots_adjust(top=...)
        panel_use_tight_layout: bool = True,  # apply tight_layout (can fight with cartopy; disable if needed)
    
        # colorbar controls
        cbar_outside: bool = True,            # if True in panel mode, place colorbar in separate axes
        cbar_rect: tuple = None,              # (x0,y0,w,h) in figure fraction for outside colorbar
        cbar_pad: float = 0.02,               # used when not using outside cax
        cbar_shrink: float = 0.85,
        cbar_fraction: float = 0.05,
        cbar_aspect: int = 30,
    
        # ------------------------------------------------------------------
        # NEW (v26.5 patch): title/label font sizes
        # ------------------------------------------------------------------
        title_fontsize: float = None,               # fig suptitle
        subplot_title_fontsize: float = None,       # each panel title
        cbar_label_fontsize: float = None,          # colorbar label size
        cbar_tick_fontsize: float = None,           # colorbar tick labels size
        axis_tick_fontsize: float = None,           # non-cartopy axes tick labels
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
    
        imtU = self._uq_imt_norm(imt)
        m = self._uq_method_norm(method)
        ws = self._uq_which_sigma_norm(which_sigma)
        k = self._uq_kind_norm(kind)
    
        vlist = sorted([int(v) for v in (version_list or [])])
        if not vlist:
            raise ValueError("version_list must be provided (non-empty).")
    
        v_first = int(first_version) if first_version is not None else int(vlist[0])
        v_last  = int(last_version)  if last_version  is not None else int(vlist[-1])
        if version is None:
            version = v_last
        version = int(version)
    
        # Ensure uq_state exists (unified grid + axes)
        if (not hasattr(self, "uq_state") or self.uq_state is None) and build_dataset_if_missing:
            self.uq_build_dataset(
                event_id=self.event_id,
                version_list=vlist,
                base_folder=base_folder,
                stations_folder=stations_folder,
                rupture_folder=rupture_folder,
                imts=(imtU,),
                grid_unify=grid_unify,
                resolution=resolution,
                export=export_dataset,
                interp_method=interp_method,
                interp_kwargs=interp_kwargs,
            )
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("uq_state missing. Run uq_build_dataset or set build_dataset_if_missing=True.")
    
        axes_uq = self.uq_state["unified_axes"]
        lat2d = np.asarray(axes_uq["lat2d"], float)
        lon2d = np.asarray(axes_uq["lon2d"], float)
    
        tr = self._uq_get_cartopy_transform()
    
        # One “posterior root” for compute/save/load
        posterior_root = posterior_output_path or output_path
    
        # --------------------------
        # small helpers
        # --------------------------
        def _pick_sigma(ep, tot):
            return np.asarray(tot if ws == "total" else ep, float)
    
        def _prior_sigma(v):
            ep0, tot0, _sa = self._uq_prior_sigma_fields(int(v), imtU)
            return _pick_sigma(ep0, tot0)
    
        def _obs_overlay(v_for_obs):
            if not show_obs:
                return np.array([]), np.array([])
            try:
                pts_lat, pts_lon, _w = self._uq_collect_obs_coords_all(int(v_for_obs), imtU)
                return np.asarray(pts_lat, float), np.asarray(pts_lon, float)
            except Exception:
                return np.array([]), np.array([])
    
        def _apply_mask(field, v_for_obs):
            if mask_to_data_radius_km and float(mask_to_data_radius_km) > 0:
                try:
                    pts_lat, pts_lon = _obs_overlay(v_for_obs)
                    if pts_lat.size > 0:
                        msk = self._uq_mask_within_km_of_points(
                            lat2d, lon2d, pts_lat, pts_lon, float(mask_to_data_radius_km)
                        )
                        return np.where(msk, field, np.nan)
                except Exception:
                    pass
            return field
    
        def _ensure_posteriors(method_name: str, v_need: int):
            """
            Compute -> Save standardized NPZ -> then future loads use generic loader.
            """
            method_name = self._uq_method_norm(method_name)
    
            dat = self._uq_load_posterior_npz(method=method_name, imt=imtU, version=int(v_need), output_path=posterior_root)
            if dat is not None and ("sigma_ep_post" in dat) and ("sigma_total_post" in dat):
                return
    
            if not compute_if_missing:
                raise FileNotFoundError(f"Missing posterior for method={method_name} v{int(v_need):03d} imt={imtU}")
    
            if method_name == "bayesupdate":
                self.uq_bayes_update(
                    version_list=vlist,
                    imt=imtU,
                    update_radius_km=float(update_radius_km),
                    update_kernel=str(update_kernel),
                    sigma_aleatory=sigma_aleatory,
                    sigma_total_from_shakemap=bool(sigma_total_from_shakemap),
                    measurement_sigma=measurement_sigma,
                    export=bool(export_posteriors),
                    make_audit=bool(make_audit),
                )
    
                dat2 = self._uq_load_posterior_npz(method="bayesupdate", imt=imtU, version=int(v_need), output_path=posterior_root)
                if dat2 is not None and ("sigma_ep_post" in dat2) and ("sigma_total_post" in dat2):
                    return
    
                legacy = self._uq_call_legacy_bayes_loader(int(v_need), imtU)
                if legacy is None:
                    raise FileNotFoundError("Bayes update ran, but legacy posterior could not be loaded for bridging.")
    
                meta = {
                    "update_radius_km": float(update_radius_km),
                    "update_kernel": str(update_kernel),
                    "measurement_sigma": measurement_sigma,
                    "sigma_total_from_shakemap": bool(sigma_total_from_shakemap),
                    "sigma_aleatory": sigma_aleatory,
                }
                meta["param_hash"] = self._uq_param_hash("bayesupdate", imtU, int(v_need), meta)
    
                self._uq_export_standard_posterior_from_legacy(
                    method="bayesupdate",
                    imt=imtU,
                    version=int(v_need),
                    legacy=legacy,
                    output_path=posterior_root,
                    meta=meta,
                )
                return
    
            if method_name == "hierarchical":
                self.uq_hierarchical_update_grid(
                    version_list=vlist,
                    imt=imtU,
                    update_radius_km=float(hier_update_radius_km),
                    kernel=str(hier_kernel),
                    kernel_scale_km=float(hier_kernel_scale_km),
                    measurement_sigma=float(hier_measurement_sigma),
                    output_path=posterior_root,
                    export=bool(export_posteriors),
                )
                return
    
            if method_name == "kriging":
                self.uq_kriging_update_grid(
                    version_list=vlist,
                    imt=imtU,
                    variogram=str(ok_variogram),
                    range_km=float(ok_range_km),
                    nugget=float(ok_nugget),
                    sill=ok_sill,
                    measurement_sigma=float(ok_measurement_sigma),
                    sigma_ep_cap_to_prior=bool(ok_cap_sigma_to_prior),
                    output_path=posterior_root,
                    export=bool(export_posteriors),
                )
                return
    
            raise ValueError("method must be one of: 'shakemap', 'bayesupdate', 'hierarchical', 'kriging'.")
    
        def _post_sigma(v):
            _ensure_posteriors(m, int(v))
            ep, tot, _meta = self._uq_posterior_sigma_fields(int(v), imtU, m)
            if ep is None or tot is None:
                dat = self._uq_load_posterior_npz(method=m, imt=imtU, version=int(v), output_path=posterior_root)
                if dat is None:
                    raise FileNotFoundError(f"Posterior still missing after compute for {m} v{int(v):03d} imt={imtU}")
                ep = dat.get("sigma_ep_post", None)
                tot = dat.get("sigma_total_post", None)
                if ep is None or tot is None:
                    raise FileNotFoundError(f"Posterior file found but missing sigma keys for {m} v{int(v):03d}")
            return _pick_sigma(ep, tot)
    
        def _field_for(kind_name, vA, vB=None):
            """
            Returns (field2d, label, obs_version_for_overlay).
            Sign convention for change: positive means decreased uncertainty (A - B).
            """
            kind_name = self._uq_kind_norm(kind_name)
    
            if kind_name == "prior":
                fld = _prior_sigma(vA)
                lbl = f"{imtU} σ_{ws} (PRIOR) — ShakeMap"
                return _apply_mask(fld, vA), lbl, vA
    
            if kind_name == "post":
                fld = _post_sigma(vA)
                lbl = f"{imtU} σ_{ws} (POST) — {m}"
                return _apply_mask(fld, vA), lbl, vA
    
            if kind_name == "reduction":
                fld = _prior_sigma(vA) - _post_sigma(vA)
                lbl = f"{imtU} Δσ_{ws} = PRIOR − POST — {m}"
                return _apply_mask(fld, vA), lbl, vA
    
            if vB is None:
                raise ValueError("change requires vB.")
    
            if m == "shakemap":
                fld = _prior_sigma(vA) - _prior_sigma(vB)
                lbl = f"{imtU} Δσ_{ws} (PRIOR change) = v{int(vA):03d} − v{int(vB):03d}"
                return _apply_mask(fld, vB), lbl, vB
            else:
                fld = _post_sigma(vA) - _post_sigma(vB)
                lbl = f"{imtU} Δσ_{ws} (POST change) = v{int(vA):03d} − v{int(vB):03d} — {m}"
                return _apply_mask(fld, vB), lbl, vB
    
        # cartopy basemap helper (robust)
        def _draw_cartopy_basemap(ax):
            try:
                if hasattr(self, "_uq_draw_cartopy_basemap"):
                    self._uq_draw_cartopy_basemap(ax)
                    return
            except Exception:
                pass
            try:
                import cartopy.feature as cfeature
                try:
                    ax.coastlines(resolution="10m", linewidth=0.9, zorder=20)
                except Exception:
                    ax.coastlines(linewidth=0.9, zorder=20)
                try:
                    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.7, edgecolor="black", zorder=21)
                except Exception:
                    ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="black", zorder=21)
                try:
                    ax.add_feature(cfeature.LAKES, alpha=0.25, zorder=10)
                except Exception:
                    pass
            except Exception:
                return
    
        # Guard: shakemap does not have post/reduction
        if m == "shakemap" and k in ("post", "reduction"):
            raise ValueError("method='shakemap' supports kind='prior' or kind='change' only.")
    
        # ==================================================================
        # SINGLE MAP MODE
        # ==================================================================
        if not bool(panel_all):
            if k in ("prior", "post", "reduction"):
                fld, lbl, vobs = _field_for(k, version)
            else:
                fld, lbl, vobs = _field_for("change", v_first, v_last)
    
            fig, ax = self._uq_cartopy_axes(figsize=figsize)
            self._uq_set_ax_extent_lonlat(ax, lon2d, lat2d, pad=0.02)
    
            if tr is not None and self._uq_is_cartopy_geoaxes(ax):
                _draw_cartopy_basemap(ax)
                pm = ax.pcolormesh(
                    lon2d, lat2d, fld, transform=tr, cmap=cmap, vmin=vmin, vmax=vmax,
                    shading="auto", zorder=5
                )
            else:
                pm = ax.pcolormesh(
                    lon2d, lat2d, fld, cmap=cmap, vmin=vmin, vmax=vmax,
                    shading="auto", zorder=5
                )
    
            pts_lat, pts_lon = _obs_overlay(vobs)
            if show_obs and pts_lat.size > 0:
                if tr is not None and self._uq_is_cartopy_geoaxes(ax):
                    ax.scatter(pts_lon, pts_lat, transform=tr, s=float(obs_size), zorder=30)
                else:
                    ax.scatter(pts_lon, pts_lat, s=float(obs_size), zorder=30)
    
            if title is None:
                if k == "change":
                    title = f"{self.event_id} | {lbl}"
                else:
                    title = f"{self.event_id} | v{int(version):03d} | {lbl}"
            if show_title:
                ax.set_title(title, fontsize=title_fontsize)
    
            if plot_colorbar:
                cb = fig.colorbar(pm, ax=ax, shrink=float(cbar_shrink), orientation=str(cbar_orientation),
                                 pad=float(cbar_pad), fraction=float(cbar_fraction), aspect=int(cbar_aspect))
                cb.set_label(cbar_label or lbl, fontsize=cbar_label_fontsize)
                if cbar_tick_fontsize is not None:
                    cb.ax.tick_params(labelsize=float(cbar_tick_fontsize))
    
            # axis tick fontsize (non-cartopy only)
            if axis_tick_fontsize is not None and not (tr is not None and self._uq_is_cartopy_geoaxes(ax)):
                try:
                    ax.tick_params(labelsize=float(axis_tick_fontsize))
                except Exception:
                    pass
    
            if save and output_path:
                out_dir = Path(output_path) / "SHAKEuq" / str(self.event_id) / "uq" / "uq_maps"
                out_dir.mkdir(parents=True, exist_ok=True)
                if k == "change":
                    stem = f"UQMap-{imtU}-{m}-{k}-{ws}-v{v_first:03d}-v{v_last:03d}"
                else:
                    stem = f"UQMap-{imtU}-{m}-{k}-{ws}-v{version:03d}"
                for fmt in save_formats:
                    fig.savefig(out_dir / f"{stem}.{fmt}", dpi=int(dpi), bbox_inches="tight")
    
            if show:
                plt.show()
            else:
                plt.close(fig)
            return fig, ax
    
        # ==================================================================
        # PANEL MODE
        # ==================================================================
        panel_mode = str(panel_mode).lower().strip()
        if panel_mode not in ("cumulative", "sequential"):
            panel_mode = "cumulative"
    
        panel_save_mode = str(panel_save_mode).lower().strip()
        if panel_save_mode not in ("panel", "split"):
            panel_save_mode = "panel"
    
        pairs = []
        if k in ("prior", "post", "reduction"):
            for v in vlist:
                pairs.append((int(v), None, None))
        else:
            # semantics: keep your existing behavior
            if panel_mode == "sequential":
                for i in range(1, len(vlist)):
                    pairs.append((int(vlist[i - 1]), int(vlist[i]), None))
            else:
                for v in vlist[1:]:
                    pairs.append((int(v_first), int(v), None))
            if include_last_vs_first_panel:
                pairs.append((int(v_first), int(v_last), "LASTvsFIRST"))
    
        n = len(pairs)
    
        # ---------------------------------------------------------
        # SPLIT SAVE MODE: each pair rendered as a standalone figure
        # ---------------------------------------------------------
        if panel_save_mode == "split":
            out_dir = Path(output_path) / "SHAKEuq" / str(self.event_id) / "uq" / "uq_maps"
            if save and output_path:
                out_dir.mkdir(parents=True, exist_ok=True)
    
            figs = []
            axes_out = []
    
            base_prefix = panel_split_prefix
            if base_prefix is None:
                base_prefix = f"UQMapStep-{imtU}-{m}-{k}-{ws}-{panel_mode}"
    
            for (a, b, tag) in pairs:
                if k in ("prior", "post", "reduction"):
                    fld, lbl, vobs = _field_for(k, a)
                    ttl = f"v{int(a):03d}"
                    stem = f"{base_prefix}-v{int(a):03d}"
                else:
                    fld, lbl, vobs = _field_for("change", a, b)
                    ttl = f"v{int(a):03d}−v{int(b):03d}"
                    if tag == "LASTvsFIRST":
                        ttl = "FIRST−LAST"
                    stem = f"{base_prefix}-v{int(a):03d}-v{int(b):03d}"
    
                fig, ax = self._uq_cartopy_axes(figsize=figsize)
                self._uq_set_ax_extent_lonlat(ax, lon2d, lat2d, pad=0.02)
    
                if tr is not None and self._uq_is_cartopy_geoaxes(ax):
                    _draw_cartopy_basemap(ax)
                    pm = ax.pcolormesh(lon2d, lat2d, fld, transform=tr, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", zorder=5)
                else:
                    pm = ax.pcolormesh(lon2d, lat2d, fld, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", zorder=5)
    
                pts_lat, pts_lon = _obs_overlay(vobs)
                if show_obs and pts_lat.size > 0:
                    if tr is not None and self._uq_is_cartopy_geoaxes(ax):
                        ax.scatter(pts_lon, pts_lat, transform=tr, s=float(obs_size), zorder=30)
                    else:
                        ax.scatter(pts_lon, pts_lat, s=float(obs_size), zorder=30)
    
                ax.set_title(ttl, fontsize=subplot_title_fontsize)
    
                if plot_colorbar:
                    cb = fig.colorbar(pm, ax=ax, shrink=float(cbar_shrink), orientation=str(cbar_orientation),
                                      pad=float(cbar_pad), fraction=float(cbar_fraction), aspect=int(cbar_aspect))
                    cb.set_label(cbar_label or lbl, fontsize=cbar_label_fontsize)
                    if cbar_tick_fontsize is not None:
                        cb.ax.tick_params(labelsize=float(cbar_tick_fontsize))
    
                if save and output_path:
                    for fmt in save_formats:
                        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=int(dpi), bbox_inches="tight")
    
                if show:
                    plt.show()
                else:
                    plt.close(fig)
    
                figs.append(fig)
                axes_out.append(ax)
    
            return figs, axes_out
    
        # ---------------------------------------------------------
        # PANEL FIGURE MODE
        # ---------------------------------------------------------
        ncol = max(1, int(panel_ncol))
        nrow = int(np.ceil(n / ncol))
        if panel_figsize is None:
            panel_figsize = (ncol * 5.4, nrow * 4.8)
    
        if tr is not None:
            try:
                import cartopy.crs as ccrs
                proj = ccrs.PlateCarree()
                fig, axes = plt.subplots(
                    nrow, ncol, figsize=panel_figsize, dpi=int(dpi),
                    subplot_kw={"projection": proj},
                    constrained_layout=False
                )
                cartopy_panels = True
            except Exception:
                fig, axes = plt.subplots(nrow, ncol, figsize=panel_figsize, dpi=int(dpi))
                cartopy_panels = False
        else:
            fig, axes = plt.subplots(nrow, ncol, figsize=panel_figsize, dpi=int(dpi))
            cartopy_panels = False
    
        axes = np.atleast_1d(axes).ravel()
    
        # Robust shared vmin/vmax if not supplied
        if vmin is None or vmax is None:
            vals = []
            for (a, b, _tag) in pairs:
                if k in ("prior", "post", "reduction"):
                    fld, _lbl, _vobs = _field_for(k, a)
                else:
                    fld, _lbl, _vobs = _field_for("change", a, b)
                vals.append(fld)
            try:
                allv = np.concatenate([np.ravel(x[np.isfinite(x)]) for x in vals if x is not None and np.isfinite(x).any()])
            except Exception:
                allv = np.array([])
            if allv.size > 0:
                lo = np.nanpercentile(allv, 2.0)
                hi = np.nanpercentile(allv, 98.0)
                if vmin is None:
                    vmin = float(lo)
                if vmax is None:
                    vmax = float(hi)
    
        mappable = None
        last_lbl = None
    
        for i, (a, b, tag) in enumerate(pairs):
            ax = axes[i]
    
            if cartopy_panels:
                self._uq_set_ax_extent_lonlat(ax, lon2d, lat2d, pad=0.02)
                _draw_cartopy_basemap(ax)
    
            if k in ("prior", "post", "reduction"):
                fld, lbl, vobs = _field_for(k, a)
                ttl = f"v{int(a):03d}"
            else:
                fld, lbl, vobs = _field_for("change", a, b)
                ttl = f"v{int(a):03d}−v{int(b):03d}"
                if tag == "LASTvsFIRST":
                    ttl = "FIRST−LAST"
    
            last_lbl = lbl
    
            if cartopy_panels:
                pm = ax.pcolormesh(lon2d, lat2d, fld, transform=tr, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", zorder=5)
            else:
                pm = ax.pcolormesh(lon2d, lat2d, fld, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", zorder=5)
    
            mappable = pm
    
            pts_lat, pts_lon = _obs_overlay(vobs)
            if show_obs and pts_lat.size > 0:
                if cartopy_panels:
                    ax.scatter(pts_lon, pts_lat, transform=tr, s=float(obs_size) * 0.6, zorder=30)
                else:
                    ax.scatter(pts_lon, pts_lat, s=float(obs_size) * 0.6, zorder=30)
    
            ax.set_title(ttl, fontsize=subplot_title_fontsize)
    
            if not cartopy_panels:
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")
                if axis_tick_fontsize is not None:
                    try:
                        ax.tick_params(labelsize=float(axis_tick_fontsize))
                    except Exception:
                        pass
    
        # Hide unused axes
        for j in range(n, axes.size):
            axes[j].axis("off")
    
        if title is None:
            title = f"{self.event_id} | {imtU} | method={m} | kind={k} | σ={ws}"
        if show_title:
            fig.suptitle(title, fontsize=title_fontsize)
    
        # -------------------------
        # spacing controls
        # -------------------------
        adjust_kwargs = {}
        if panel_left is not None:
            adjust_kwargs["left"] = float(panel_left)
        if panel_right is not None:
            adjust_kwargs["right"] = float(panel_right)
        if panel_bottom is not None:
            adjust_kwargs["bottom"] = float(panel_bottom)
        if panel_top is not None:
            adjust_kwargs["top"] = float(panel_top)
        if panel_wspace is not None:
            adjust_kwargs["wspace"] = float(panel_wspace)
        if panel_hspace is not None:
            adjust_kwargs["hspace"] = float(panel_hspace)
    
        if adjust_kwargs:
            try:
                fig.subplots_adjust(**adjust_kwargs)
            except Exception:
                pass
        elif panel_use_tight_layout:
            try:
                fig.tight_layout(rect=[0, 0, 1, 0.96] if show_title else None)
            except Exception:
                pass
    
        # -------------------------
        # colorbar placement
        # -------------------------
        if plot_colorbar and (mappable is not None):
            if cbar_outside:
                if cbar_rect is None:
                    # Reserve space for cbar if user didn't already
                    try:
                        if panel_right is None:
                            fig.subplots_adjust(right=0.88)
                    except Exception:
                        pass
                    cbar_rect_use = (0.90, 0.15, 0.025, 0.70)
                else:
                    cbar_rect_use = tuple(float(x) for x in cbar_rect)
    
                try:
                    cax = fig.add_axes(cbar_rect_use)
                    cb = fig.colorbar(mappable, cax=cax, orientation=str(cbar_orientation))
                except Exception:
                    cb = fig.colorbar(
                        mappable, ax=axes[:n], shrink=float(cbar_shrink),
                        fraction=float(cbar_fraction), pad=float(cbar_pad),
                        aspect=int(cbar_aspect), orientation=str(cbar_orientation)
                    )
            else:
                cb = fig.colorbar(
                    mappable, ax=axes[:n], shrink=float(cbar_shrink),
                    fraction=float(cbar_fraction), pad=float(cbar_pad),
                    aspect=int(cbar_aspect), orientation=str(cbar_orientation)
                )
    
            cb.set_label(cbar_label or (last_lbl or ""), fontsize=cbar_label_fontsize)
            if cbar_tick_fontsize is not None:
                try:
                    cb.ax.tick_params(labelsize=float(cbar_tick_fontsize))
                except Exception:
                    pass
    
        if save and output_path:
            out_dir = Path(output_path) / "SHAKEuq" / str(self.event_id) / "uq" / "uq_maps"
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"UQMapPanels-{imtU}-{m}-{k}-{ws}-{panel_mode}-n{n}"
            for fmt in save_formats:
                fig.savefig(out_dir / f"{stem}.{fmt}", dpi=int(dpi), bbox_inches="tight")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        return fig, axes[:n]






    # ======================================================================
    # PATCH 26.6 (Step 5): Cross-IMT pseudo-observations (weak, GMICE default)
    #   - Adds optional pseudo-observations by converting IMT across MMI<->PGA
    #   - Default: GMICE conversion (SHAKEgmice.GMICE)
    #   - Optional: dataset extraction (predictions / mmi_from_pgm) if present
    #   - Weak influence enforced by large sigma_pseudo
    # ======================================================================
    
    def _uq_g_to_cms2(self, g_val):
        """Convert g (gravity) to cm/s^2."""
        try:
            return float(g_val) * 980.665
        except Exception:
            return None
    
    def _uq_percent_g_to_cms2(self, pct_g):
        """Convert %g to cm/s^2."""
        try:
            return float(pct_g) * 0.01 * 980.665
        except Exception:
            return None
    
    def _uq_cms2_to_percent_g(self, cms2):
        """Convert cm/s^2 to %g."""
        try:
            return float(cms2) / 980.665 * 100.0
        except Exception:
            return None
    
    def _uq_try_extract_cross_imt_from_dataset(self, row, target_imt):
        """
        Optional dataset-mode extraction of cross-IMT values.
        This is a best-effort helper (not guaranteed for all feeds).
        Expected examples:
          - Instrumented PGA df: column 'mmi_from_pgm' is list of dicts with entries like {'name':'pga','value':MMI}
          - MMI-side feeds sometimes have 'predictions' entries with PGA/PGV etc.
        Returns scalar in the *target IMT domain units*:
          - target_imt == 'MMI' -> returns MMI value
          - target_imt == 'PGA' -> returns PGA in %g if available
        """
        try:
            tgt = str(target_imt).upper().strip()
    
            if tgt == "MMI":
                # prefer mmi_from_pgm list-of-dicts
                mfp = row.get("mmi_from_pgm", None)
                if isinstance(mfp, (list, tuple)):
                    # try pga first, else any available
                    for name_try in ("pga", "pgv", "sa(1.0)", "sa(0.3)", "sa(3.0)"):
                        for d in mfp:
                            if isinstance(d, dict) and str(d.get("name", "")).lower() == name_try:
                                v = d.get("value", None)
                                if v is not None:
                                    return float(v)
                    for d in mfp:
                        if isinstance(d, dict) and d.get("value", None) is not None:
                            return float(d["value"])
    
                # fallback: sometimes "intensity" already present
                if row.get("intensity", None) is not None:
                    return float(row["intensity"])
    
                return None
    
            if tgt == "PGA":
                # try predictions list-of-dicts
                preds = row.get("predictions", None)
                if isinstance(preds, (list, tuple)):
                    for d in preds:
                        if isinstance(d, dict) and str(d.get("name", "")).lower() == "pga":
                            v = d.get("value", None)
                            if v is not None:
                                return float(v)
    
                # fallback: direct 'pga' column if present (often %g)
                if row.get("pga", None) is not None:
                    return float(row["pga"])
    
                return None
    
            return None
        except Exception:
            return None
    
    def _uq_convert_cross_imt_gmice(self, value, from_imt, to_imt, gmice_model="Globalgmice"):
        """
        Convert across IMT via SHAKEgmice.GMICE.
    
        Conventions (matching your parsing examples):
          - PGA observations often stored as %g in your dataframes
          - GMICE expects PGA in cm/s^2 (per SHAKEgmice header)
          - Output:
              to_imt == MMI -> returns MMI
              to_imt == PGA -> returns PGA in %g
        """
        try:
            fromU = str(from_imt).upper().strip()
            toU   = str(to_imt).upper().strip()
    
            if fromU == toU:
                return float(value)
    
            # Import here to avoid hard dependency if user never enables Step 5
            from modules.SHAKEgmice import GMICE
    
            if fromU == "PGA" and toU == "MMI":
                # input is %g -> cm/s^2
                cms2 = self._uq_percent_g_to_cms2(value)
                if cms2 is None:
                    return None
                conv = GMICE(model=str(gmice_model), input_value=[cms2], input_type="PGA", output_type="MMI").convert()
                # GMICE returns array-like
                return float(conv[0]) if hasattr(conv, "__len__") else float(conv)
    
            if fromU == "MMI" and toU == "PGA":
                conv = GMICE(model=str(gmice_model), input_value=[float(value)], input_type="MMI", output_type="PGA").convert()
                out_cms2 = float(conv[0]) if hasattr(conv, "__len__") else float(conv)
                # convert cm/s^2 -> %g
                pctg = self._uq_cms2_to_percent_g(out_cms2)
                return float(pctg) if pctg is not None else None
    
            # (Optional extensions: PGV, SA, etc.) – keep Step 5 minimal for now.
            return None
        except Exception:
            return None
    
    def _uq_append_pseudo_obs(
        self,
        obs_rows,
        target_imt,
        pseudo_mode="gmice",
        gmice_model="Globalgmice",
        sigma_pseudo_mmi=1.0,
        sigma_pseudo_pga_ln=1.0,
        measurement_sigma_base=0.30,
        prefer_dataset_first=False,
    ):
        """
        Build pseudo-observations by converting each real observation's IMT into target_imt,
        then adding it with reduced weight ~ (sigma_base / sigma_pseudo)^2.
    
        IMPORTANT:
          - This assumes obs_rows are dict-like and include:
              'imt' or 'obs_imt' (source IMT name), 'value' (domain value),
              'lat','lon', and optionally 'weight' (base weight).
          - This does NOT change your core Bayes math; it only downweights pseudo obs.
    
        Returns (obs_rows_extended, n_added)
        """
        tgt = str(target_imt).upper().strip()
        pseudo_added = 0
    
        # choose pseudo sigma in *working space*:
        #  - MMI: linear, so sigma_pseudo_mmi is already OK
        #  - PGA: your working space for PGA uses ln(PGA), so sigma_pseudo_pga_ln is in ln-units
        sigma_pseudo_ws = float(sigma_pseudo_mmi) if tgt == "MMI" else float(sigma_pseudo_pga_ln)
    
        # base sigma used to translate pseudo sigma into a relative weight
        # (this keeps pseudo influence weak while staying compatible with your existing interfaces)
        sigma_base_ws = float(measurement_sigma_base)
    
        # weight multiplier: w' = w * (sigma_base / sigma_pseudo)^2
        # if sigma_pseudo is large, multiplier is small.
        try:
            wmult = (sigma_base_ws / sigma_pseudo_ws) ** 2
        except Exception:
            wmult = 0.0
    
        out = list(obs_rows)
    
        for o in obs_rows:
            if not isinstance(o, dict):
                continue
    
            # try to identify source IMT + value
            src_imt = o.get("imt", o.get("obs_imt", None))
            val     = o.get("value", o.get("obs_value", None))
            if src_imt is None or val is None:
                continue
    
            srcU = str(src_imt).upper().strip()
            if srcU == tgt:
                continue  # already in target IMT
    
            # Conversion: dataset-mode optionally, else GMICE
            converted = None
    
            # If this observation carried its raw row payload, dataset-mode can use it.
            raw_row = o.get("_raw_row", None)
    
            mode = str(pseudo_mode).lower().strip()
            if prefer_dataset_first and raw_row is not None and mode in ("dataset", "auto"):
                converted = self._uq_try_extract_cross_imt_from_dataset(raw_row, tgt)
    
            if converted is None and mode in ("gmice", "auto", "dataset"):
                # If dataset was requested but failed, we still allow GMICE as fallback.
                converted = self._uq_convert_cross_imt_gmice(val, srcU, tgt, gmice_model=gmice_model)
    
            if converted is None:
                continue
    
            # Build pseudo obs row: keep location, set IMT to target, mark pseudo, reduce weight
            w0 = o.get("weight", 1.0)
            try:
                w0 = float(w0)
            except Exception:
                w0 = 1.0
    
            pseudo = dict(o)
            pseudo["imt"] = tgt
            pseudo["value"] = float(converted)
            pseudo["is_pseudo"] = True
            pseudo["pseudo_from_imt"] = srcU
            pseudo["weight"] = float(w0) * float(wmult)
    
            # type labeling (optional)
            pseudo["obs_type"] = str(pseudo.get("obs_type", "unknown")) + "+pseudo"
    
            out.append(pseudo)
            pseudo_added += 1
    
        return out, pseudo_added






    # ======================================================================
    # PATCH v26.6 — True Two-Likelihood Bayesian Update
    # Overrides local Bayes fusion to consume per-observation sigma
    # ======================================================================

  

    # ======================================================================
    # PATCH v26.7 — Target-decay pipeline (ShakeMap raw + Bayes + 2-lik + DYFI-weighted)
    # Copy-paste this at the END of class SHAKEuq (inside the class indentation).
    #
    # What you get:
    #   1) "ShakeMap" curve: published mean + published total sigma (raw from uncertainty grid)
    #   2) "bayes"          : v0 prior + single-likelihood update (domain-preferred obs)
    #   3) "bayes_2lik"     : v0 prior + mixed obs with per-observation meas_var
    #   4) "dyfi_weighted"  : like bayes_2lik, but applies DYFI weighting (nresp-based) if present
    #
    # Debug:
    #   - set self.debug_uq = True  (prints meas_var usage summary per update call)
    # ======================================================================
    
    def _uq_infer_obs_domain(self, imt):
        imtU = str(imt).upper().strip()
        return "intensity" if imtU == "MMI" else "seismic"
    
    
    
    
    def _uq_bayes_local_posterior_at_mask(
        self,
        mu0_ws,
        sigma_ep0,
        sigma_a,
        lat2d,
        lon2d,
        target_mask,
        obs_list,
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
    ):
        """
        Local conjugate Gaussian update in working space.
    
        Uses:
          - ob["meas_var"] if present
          - else ob["sigma_obs"]**2
          - else measurement_sigma**2
    
        Returns:
          mu_post_ws, sigma_ep_post arrays on unified grid (nan outside mask).
        """
        import numpy as np
    
        mu0_ws = np.asarray(mu0_ws, dtype=float)
        sigma_ep0 = np.asarray(sigma_ep0, dtype=float)
        sigma_a = np.asarray(sigma_a, dtype=float)
    
        # broadcast guards
        if sigma_ep0.ndim == 0:
            sigma_ep0 = np.full_like(mu0_ws, float(sigma_ep0), dtype=float)
        elif sigma_ep0.shape != mu0_ws.shape:
            sigma_ep0 = np.broadcast_to(sigma_ep0, mu0_ws.shape).astype(float)
    
        if sigma_a.ndim == 0:
            sigma_a = np.full_like(mu0_ws, float(sigma_a), dtype=float)
        elif sigma_a.shape != mu0_ws.shape:
            sigma_a = np.broadcast_to(sigma_a, mu0_ws.shape).astype(float)
    
        mu_post = np.full(mu0_ws.shape, np.nan, dtype=float)
        sigma_ep_post = np.full(mu0_ws.shape, np.nan, dtype=float)
    
        m = np.asarray(target_mask, dtype=bool)
        if not m.any():
            return mu_post, sigma_ep_post
    
        mu_post[m] = mu0_ws[m]
        sigma_ep_post[m] = sigma_ep0[m]
    
        if not obs_list:
            return mu_post, sigma_ep_post
    
        ker = str(kernel).lower().strip()
        default_meas_var = float(measurement_sigma) ** 2
    
        used_meas_vars = []
    
        for ob in obs_list:
            try:
                lat_o = float(ob["lat"]); lon_o = float(ob["lon"])
                y = float(ob["y_ws"])
            except Exception:
                continue
    
            # measurement variance (per-observation)
            meas_var = None
            if "meas_var" in ob:
                try:
                    mv = float(ob["meas_var"])
                    if np.isfinite(mv) and mv > 0:
                        meas_var = mv
                except Exception:
                    pass
            if meas_var is None and "sigma_obs" in ob:
                try:
                    so = float(ob["sigma_obs"])
                    if np.isfinite(so) and so > 0:
                        meas_var = so ** 2
                except Exception:
                    pass
            if meas_var is None:
                meas_var = default_meas_var
    
            used_meas_vars.append(meas_var)
    
            w0 = 1.0
            try:
                w0 = float(ob.get("w", 1.0))
            except Exception:
                w0 = 1.0
    
            # distances
            d = self._uq_haversine_km(lat_o, lon_o, lat2d, lon2d)
    
            # local mask
            if update_radius_km is None or float(update_radius_km) <= 0:
                d2 = d.copy()
                d2[~m] = np.nan
                if np.all(~np.isfinite(d2)):
                    continue
                ij = np.unravel_index(np.nanargmin(d2), d2.shape)
                local = np.zeros_like(m, dtype=bool)
                local[ij] = True
            else:
                local = (d <= float(update_radius_km)) & m
    
            if not local.any():
                continue
    
            # spatial weights
            if ker in ("tophat", "uniform"):
                w_space = np.ones_like(d, dtype=float)
            else:
                s = float(kernel_scale_km) if kernel_scale_km is not None else float(update_radius_km) / 2.0
                s = max(s, 1e-6)
                w_space = np.exp(-(d**2) / (2.0 * s**2))
    
            w_eff = w0 * w_space
            w_eff[~local] = 0.0
    
            # likelihood variance per cell: sigma_ep_post^2 + sigma_a^2 + meas_var
            like_var = np.clip((sigma_ep_post ** 2) + (sigma_a ** 2) + float(meas_var), 1e-16, np.inf)
    
            # precision update
            prec_prior = np.zeros_like(mu0_ws, dtype=float)
            prec_prior[local] = 1.0 / np.clip(sigma_ep_post[local] ** 2, 1e-16, np.inf)
    
            prec_like = np.zeros_like(mu0_ws, dtype=float)
            prec_like[local] = np.clip(w_eff[local] / like_var[local], 0.0, np.inf)
    
            prec_post = prec_prior + prec_like
            upd = local & np.isfinite(prec_post) & (prec_post > 0)
    
            mu_post[upd] = (mu_post[upd] * prec_prior[upd] + y * prec_like[upd]) / prec_post[upd]
            sigma_ep_post[upd] = np.sqrt(1.0 / np.clip(prec_post[upd], 1e-16, np.inf))
    
        if getattr(self, "debug_uq", False):
            import numpy as np
            if used_meas_vars:
                u = np.unique(np.round(np.asarray(used_meas_vars, float), 6))
                print(f"[UQ] bayes update meas_var used (unique={u.size}): min={u.min():.4f}, max={u.max():.4f}")
            else:
                print("[UQ] bayes update: obs_list empty / unusable")
    
        return mu_post, sigma_ep_post


    def _uq_bayes_local_posterior_2lik_at_mask(
        self,
        mu0_ws,
        sigma_ep0,
        sigma_a,
        lat2d,
        lon2d,
        target_mask,
        obs_list,
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
    ):
        """
        Two-likelihood update wrapper.
        Uses per-observation meas_var when provided (same kernel as bayes).
        """
        return self._uq_bayes_local_posterior_at_mask(
            mu0_ws,
            sigma_ep0,
            sigma_a,
            lat2d,
            lon2d,
            target_mask,
            obs_list,
            update_radius_km=update_radius_km,
            kernel=kernel,
            kernel_scale_km=kernel_scale_km,
            measurement_sigma=measurement_sigma,
        )


    def _uq_dyfi_weighted_posterior_at_mask(
        self,
        mu0_ws,
        sigma_ep0,
        sigma_a,
        lat2d,
        lon2d,
        target_mask,
        obs_list,
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
    ):
        """
        DYFI-weighted update wrapper (expects obs_list already weighted).
        """
        return self._uq_bayes_local_posterior_2lik_at_mask(
            mu0_ws,
            sigma_ep0,
            sigma_a,
            lat2d,
            lon2d,
            target_mask,
            obs_list,
            update_radius_km=update_radius_km,
            kernel=kernel,
            kernel_scale_km=kernel_scale_km,
            measurement_sigma=measurement_sigma,
        )
    
    
    def _uq_apply_dyfi_weights(self, obs_list, mode="sqrt_nresp", w_max=10.0):
        """
        DYFI-weight helper.
          - If ob['domain']=="intensity" and ob has 'nresp', boosts weight.
          - Keeps non-DYFI obs unchanged.
        """
        import numpy as np
        out = []
        for ob in (obs_list or []):
            o = dict(ob)
            if str(o.get("domain", "")).lower() == "intensity" and ("nresp" in o):
                try:
                    n = float(o["nresp"])
                    if np.isfinite(n) and n > 0:
                        if mode == "linear_nresp":
                            f = n
                        elif mode == "log1p_nresp":
                            f = np.log1p(n)
                        else:  # default sqrt
                            f = np.sqrt(n)
                        o["w"] = float(min(float(o.get("w", 1.0)) * f, float(w_max)))
                except Exception:
                    pass
            out.append(o)
        return out
    
    

    # ==========================================================
    # PATCH v26.8 — CDI-aware DYFI stream + per-version timestamp helper
    # Paste at END of SHAKEuq class (last definition wins).
    # ==========================================================
    
    def _uq_collect_stationlist_obs_for_version(
        self,
        version,
        imtU,
        measurement_sigma=0.30,
        include_weights=True,
        prefer_domain=True,
        allow_fallback=True,
        measurement_sigma_instr=None,
        measurement_sigma_dyfi=None,
        attach_per_obs_sigma=False,
    ):
        """
        Adapter around the existing stationlist collector you already have:
            self._uq_collect_observations(version, imt)
    
        Returns:
          obs_list: list[dict] with keys lat, lon, value, w, domain, (optional) sigma_obs, meas_var
          counts: dict(total,seismic,intensity,unknown)
        """
        import numpy as np
    
        imtU = str(imtU).upper().strip()
        domain_pref = self._uq_infer_obs_domain(imtU)
    
        # --- source: your existing collector ---
        if not hasattr(self, "_uq_collect_observations"):
            return [], {"total": 0, "seismic": 0, "intensity": 0, "unknown": 0}
    
        raw = self._uq_collect_observations(int(version), imtU)
        if raw is None:
            raw = []
    
        obs_all = []
        counts_all = {"total": 0, "seismic": 0, "intensity": 0, "unknown": 0}
    
        def _classify_domain(o):
            # robust classification from your current obs dicts
            t = str(o.get("type", "")).lower().strip()
            # DYFI / macroseismic -> intensity
            if ("dyfi" in t) or ("intensity" in t) or ("macro" in t):
                return "intensity"
            # instrumented / seismic -> seismic
            if ("instrument" in t) or ("seismic" in t) or ("station" in t):
                return "seismic"
            # fallback: IMT-based preference
            return domain_pref if domain_pref in ("seismic", "intensity") else "unknown"
    
        def _pick_sigma(domain):
            # Step-4: per-domain likelihood sigma
            if domain == "seismic" and measurement_sigma_instr is not None:
                return float(measurement_sigma_instr)
            if domain == "intensity" and measurement_sigma_dyfi is not None:
                return float(measurement_sigma_dyfi)
            return float(measurement_sigma)
    
        for o in raw:
            try:
                lat = float(o["lat"])
                lon = float(o["lon"])
            except Exception:
                continue
    
            val = o.get("value", None)
            if val is None:
                continue
            try:
                val = float(val)
            except Exception:
                continue
            if not np.isfinite(val):
                continue
    
            domain = _classify_domain(o)
            w = float(o.get("w", 1.0)) if include_weights else 1.0
    
            rec = {
                "lat": lat,
                "lon": lon,
                "value": val,
                "w": w,
                "domain": domain,
                "type": o.get("type", None),
                "unit": o.get("unit", None),
            }
    
            if attach_per_obs_sigma:
                sig = _pick_sigma(domain)
                rec["sigma_obs"] = float(sig)
                rec["meas_var"] = float(sig) ** 2
    
            obs_all.append(rec)
            counts_all["total"] += 1
            if domain in counts_all:
                counts_all[domain] += 1
            else:
                counts_all["unknown"] += 1
    
        # --- prefer_domain behavior ---
        if prefer_domain:
            obs_pref = [o for o in obs_all if o.get("domain") == domain_pref]
            if len(obs_pref) > 0:
                # recompute counts on filtered set
                c = {"total": 0, "seismic": 0, "intensity": 0, "unknown": 0}
                for o in obs_pref:
                    c["total"] += 1
                    d = o.get("domain", "unknown")
                    if d in c:
                        c[d] += 1
                    else:
                        c["unknown"] += 1
                return obs_pref, c
    
            # if no preferred-domain obs, optionally fall back
            if not allow_fallback:
                return [], {"total": 0, "seismic": 0, "intensity": 0, "unknown": 0}
    
        return obs_all, counts_all
    
    




    def _uq_get_obs_for_method(
        self,
        version,
        imtU,
        method,
        measurement_sigma=0.30,
        measurement_sigma_instr=None,
        measurement_sigma_dyfi=None,
    ):
        """
        Centralized obs selection for UQ methods.
        Ensures CDI/stationlist switching is honored via self.dyfi_source.
        """
        imtU = str(imtU).upper().strip()
        m = str(method).lower().strip()
    
        # bayes: keep legacy prefer_domain=True behavior
        if m == "bayes":
            obs, c = self._uq_collect_obs_for_version(
                int(version), imtU,
                measurement_sigma=measurement_sigma,
                include_weights=True,
                prefer_domain=True,
                allow_fallback=True,
                measurement_sigma_instr=measurement_sigma_instr,
                measurement_sigma_dyfi=measurement_sigma_dyfi,
                attach_per_obs_sigma=False,
            )
            return obs, c
    
        # bayes_2lik + dyfi_weighted: mixed stream + per-obs sigma + DYFI source policy
        if m in ("bayes_2lik", "dyfi_weighted"):
            obs, c = self._uq_collect_obs_for_version(
                int(version), imtU,
                measurement_sigma=measurement_sigma,
                include_weights=True,
                prefer_domain=False,
                allow_fallback=True,
                measurement_sigma_instr=measurement_sigma_instr,
                measurement_sigma_dyfi=measurement_sigma_dyfi,
                attach_per_obs_sigma=True,
                dyfi_source=getattr(self, "dyfi_source", "stationlist"),
            )
            return obs, c
    
        # default: no obs
        return [], {"total": 0, "seismic": 0, "intensity": 0, "unknown": 0}


    



    # ==========================================================
    # PATCH v26.8+ — dyfi_source override + CDI switching helpers
    # Paste at END of SHAKEuq class (last definition wins).
    # ==========================================================
    
    def smoke_uq_v26p9_cdi(
        self,
        *,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        agg="mean",
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
        measurement_sigma_instr=None,
        measurement_sigma_dyfi=None,
        max_versions_print=6,
    ):
        """
        Compact smoke test for CDI switching + per-obs sigma (v26.9).
        Prints:
          - version → dyfi_source(auto) → counts by source
          - divergence stats between stationlist vs cdi for bayes_2lik and dyfi_weighted
          - high-radius check (update_radius_km=200)
        """
        import numpy as np
        from pathlib import Path

        imtU = str(imt).upper().strip()
        versions = sorted([int(v) for v in version_list])

        orig_attach = getattr(self, "cdi_attach_from_version", 4)

        def _meas_var_stats(obs_list):
            m = [o.get("meas_var") for o in (obs_list or []) if o.get("meas_var") is not None]
            m = [float(x) for x in m if np.isfinite(x)]
            if not m:
                return None
            return {"min": float(np.min(m)), "median": float(np.median(m)), "max": float(np.max(m)), "n": len(m)}
    
        def _collect_obs_summary(v, src):
            obs, c = self._uq_get_obs_for_update(
                int(v),
                imtU,
                method="bayes_2lik",
                dyfi_source=src,
                measurement_sigma=measurement_sigma,
                measurement_sigma_instr=measurement_sigma_instr,
                measurement_sigma_dyfi=measurement_sigma_dyfi,
                attach_per_obs_sigma=True,
            )
            n_cdi = sum(1 for o in (obs or []) if str(o.get("source", "")).lower() == "dyfi_cdi")
            stats = _meas_var_stats(obs)
            return obs, c, n_cdi, stats

        def _auto_src(v):
            return self._uq_select_dyfi_source_for_version(int(v), dyfi_source="auto")

        try:
            print("\n[UQ SMOKE] CDI switching diagnostics (v26.9)")
            for src in ("stationlist", "auto", "cdi"):
                print(f"\n[UQ SMOKE] dyfi_source={src}")
                for v in versions[: int(max_versions_print)]:
                    src_eff = _auto_src(v) if src == "auto" else src
                    obs, c, n_cdi, stats = _collect_obs_summary(v, src)
                    pool = (self.uq_state or {}).get("obs_by_version", {}).get(int(v), {})
                    n_station = len(pool.get("obs_intensity_stationlist", []))
                    n_cdi_pool = len(pool.get("obs_intensity_cdi", []))
                    print(
                        f"  v{int(v):03d} | src_eff={src_eff} | "
                        f"n_obs={int(c.get('total', 0))} | n_cdi_obs={int(n_cdi)} | "
                        f"pool_stationlist={n_station} pool_cdi={n_cdi_pool}"
                    )
                    if stats:
                        print(
                            f"    meas_var: n={stats['n']} "
                            f"min={stats['min']:.4f} med={stats['median']:.4f} max={stats['max']:.4f}"
                        )
    
            if getattr(self, "dyfi_cdi_file", None) is None:
                print("[UQ SMOKE] CDI file not configured (dyfi_cdi_file=None).")
            else:
                cdi_path = Path(str(self.dyfi_cdi_file))
                if not cdi_path.exists():
                    print(f"[UQ SMOKE] CDI file missing: {cdi_path}")
                else:
                    df_cdi = self._uq_load_dyfi_cdi_df() if hasattr(self, "_uq_load_dyfi_cdi_df") else None
                    if df_cdi is None or len(df_cdi) == 0:
                        print("[UQ SMOKE] CDI dataframe empty after parsing.")
    
            df_station = self.uq_extract_target_series(
                version_list=versions,
                imt=imtU,
                points=points,
                areas=areas,
                agg=agg,
                update_radius_km=update_radius_km,
                kernel=kernel,
                kernel_scale_km=kernel_scale_km,
                measurement_sigma=measurement_sigma,
                measurement_sigma_instr=measurement_sigma_instr,
                measurement_sigma_dyfi=measurement_sigma_dyfi,
                methods_to_compute=("ShakeMap", "bayes_2lik", "dyfi_weighted"),
                dyfi_source="stationlist",
                audit=False,
            )
            df_cdi = self.uq_extract_target_series(
                version_list=versions,
                imt=imtU,
                points=points,
                areas=areas,
                agg=agg,
                update_radius_km=update_radius_km,
                kernel=kernel,
                kernel_scale_km=kernel_scale_km,
                measurement_sigma=measurement_sigma,
                measurement_sigma_instr=measurement_sigma_instr,
                measurement_sigma_dyfi=measurement_sigma_dyfi,
                methods_to_compute=("ShakeMap", "bayes_2lik", "dyfi_weighted"),
                dyfi_source="cdi",
                audit=False,
            )
    
            def _diff_stats(method, col):
                a = df_station[df_station["method"] == method][["version", "target_id", col]].rename(columns={col: f"{col}_a"})
                b = df_cdi[df_cdi["method"] == method][["version", "target_id", col]].rename(columns={col: f"{col}_b"})
                m = a.merge(b, on=["version", "target_id"], how="inner")
                if m.empty:
                    return None
                d = (m[f"{col}_b"] - m[f"{col}_a"]).astype(float)
                return {"n": int(len(d)), "min": float(np.min(d)), "median": float(np.median(d)), "max": float(np.max(d))}
    
            for method in ("bayes_2lik", "dyfi_weighted"):
                for col in ("sigma_total_predicted", "mean_predicted"):
                    stats = _diff_stats(method, col)
                    if stats:
                        print(
                            f"[UQ SMOKE] diff {method} {col}: "
                            f"n={stats['n']} min={stats['min']:.4f} med={stats['median']:.4f} max={stats['max']:.4f}"
                        )
                    else:
                        print(f"[UQ SMOKE] diff {method} {col}: no overlap rows.")

            self.cdi_attach_from_version = 2
            df_auto = self.uq_extract_target_series(
                version_list=versions,
                imt=imtU,
                points=points,
                areas=areas,
                agg=agg,
                update_radius_km=update_radius_km,
                kernel=kernel,
                kernel_scale_km=kernel_scale_km,
                measurement_sigma=measurement_sigma,
                measurement_sigma_instr=measurement_sigma_instr,
                measurement_sigma_dyfi=measurement_sigma_dyfi,
                methods_to_compute=("ShakeMap", "bayes_2lik", "dyfi_weighted"),
                dyfi_source="auto",
                audit=False,
            )
            print("[UQ SMOKE] AUTO check: cdi_attach_from_version=2 (forced early switch).")

            df_high = self.uq_extract_target_series(
                version_list=versions,
                imt=imtU,
                points=points,
                areas=areas,
                agg=agg,
                update_radius_km=200.0,
                kernel=kernel,
                kernel_scale_km=kernel_scale_km,
                measurement_sigma=measurement_sigma,
                measurement_sigma_instr=measurement_sigma_instr,
                measurement_sigma_dyfi=measurement_sigma_dyfi,
                methods_to_compute=("ShakeMap", "bayes_2lik", "dyfi_weighted"),
                dyfi_source="cdi",
                audit=False,
            )
            print(f"[UQ SMOKE] High-radius check: rows={len(df_high)} auto_rows={len(df_auto)}")

        finally:
            self.cdi_attach_from_version = orig_attach

        return True


    def smoke_uq_v26p8_cdi(self, *args, **kwargs):
        """Backward-compatible wrapper for v26.9 smoke diagnostics."""
        return self.smoke_uq_v26p9_cdi(*args, **kwargs)
    
    
    def _uq_parse_iso_datetime(self, s):
        from datetime import datetime
        if s is None:
            return None
        ss = str(s).strip()
        if not ss:
            return None
        try:
            return datetime.fromisoformat(ss.replace("Z", ""))
        except Exception:
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(ss.replace("Z", ""), fmt)
                except Exception:
                    pass
        return None
    
    
    def _uq_read_grid_timestamps(self, grid_path):
        """
        Read per-version timestamps from grid.xml header.
        Returns dict: {"process_time": dt|None, "event_time": dt|None, "TaE_hours": float|None}
        """
        from pathlib import Path
        import xml.etree.ElementTree as ET
    
        p = Path(grid_path)
        if not p.exists():
            return {"process_time": None, "event_time": None, "TaE_hours": None}
    
        try:
            root = ET.parse(str(p)).getroot()
        except Exception:
            return {"process_time": None, "event_time": None, "TaE_hours": None}
    
        proc_s = root.attrib.get("process_timestamp", None)
    
        evt_node = None
        for ch in list(root):
            if str(ch.tag).lower().endswith("event"):
                evt_node = ch
                break
        evt_s = evt_node.attrib.get("event_timestamp", None) if evt_node is not None else None
    
        proc_dt = self._uq_parse_iso_datetime(proc_s)
        evt_dt = self._uq_parse_iso_datetime(evt_s)
    
        # fallback to object-level event_time (if set)
        if evt_dt is None and getattr(self, "event_time", None) is not None:
            evt_dt = self.event_time
    
        tae_h = None
        if proc_dt is not None and evt_dt is not None:
            try:
                tae_h = (proc_dt - evt_dt).total_seconds() / 3600.0
            except Exception:
                tae_h = None
    
        return {"process_time": proc_dt, "event_time": evt_dt, "TaE_hours": tae_h}
    
    
    def _uq_get_version_timestamp(self, version, stations_folder=None, rupture_folder=None):
        """Return ShakeMap process timestamp (datetime) for a version."""
        try:
            grid_path, _, _, _, _ = self._uq_resolve_paths(
                int(version),
                stations_folder=stations_folder,
                rupture_folder=rupture_folder,
            )
        except Exception:
            return None
        meta = self._uq_read_grid_timestamps(grid_path)
        return meta.get("process_time", None)
    
    
    def _uq_hours_since_origin(self, version, stations_folder=None, rupture_folder=None):
        """Compute hours since origin for a version (TaE_hours)."""
        try:
            grid_path, _, _, _, _ = self._uq_resolve_paths(
                int(version),
                stations_folder=stations_folder,
                rupture_folder=rupture_folder,
            )
        except Exception:
            return None
    
        meta = self._uq_read_grid_timestamps(grid_path)
        tae_h = meta.get("TaE_hours", None)
        if tae_h is not None:
            try:
                return float(tae_h)
            except Exception:
                return None
    
        proc_dt = meta.get("process_time", None)
        evt_dt = meta.get("event_time", None)
        if proc_dt is None or evt_dt is None:
            return None
        try:
            return float((proc_dt - evt_dt).total_seconds() / 3600.0)
        except Exception:
            return None
    
    
    def _uq_is_cdi_available_for_version(self, version, stations_folder=None, rupture_folder=None):
        """
        CDI gate:
          - requires dyfi_cdi_file exists
          - if self.dyfi_source == "cdi": True (force)
          - if self.dyfi_source == "auto": True only if version >= cdi_attach_from_version
          - otherwise False
        """
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
        """
        Load CDI dataframe ONCE and cache it.
        Expects USGSParser(parser_type="dyfi_data", file_path=...).
        Produces columns: lat, lon, cdi, nresp, dist_km, std, suspect
        """
        from pathlib import Path
        import pandas as pd
    
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
    
        # normalize likely column names to our internal ones
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
    
        # coerce numerics
        for k in ("lat", "lon", "cdi", "nresp", "dist_km", "std", "suspect"):
            if k in df.columns:
                df[k] = pd.to_numeric(df[k], errors="coerce")
    
        df = df.dropna(subset=["lat", "lon", "cdi"]).copy()
        self._dyfi_cdi_df_cache = df
        return df
    
    
    def _uq_dyfi_weight_from_row(self, nresp):
        """
        First-step DYFI weighting rule (simple + stable):
          - rule="nresp_threshold" (default): weight=high if nresp>=thr else low
          - rule="sqrt_nresp": weight=sqrt(nresp) capped by dyfi_weight_max
          - rule="none": weight=1
        """
        import numpy as np
    
        rule = str(getattr(self, "dyfi_weight_rule", "nresp_threshold")).lower().strip()
        thr = int(getattr(self, "dyfi_weight_threshold", 3))
        wlo = float(getattr(self, "dyfi_weight_low", 1.0))
        whi = float(getattr(self, "dyfi_weight_high", 2.0))
        wmax = float(getattr(self, "dyfi_weight_max", 10.0))
    
        if nresp is None or (not np.isfinite(nresp)):
            return float(wlo)
    
        n = int(max(0, float(nresp)))
    
        if rule == "none":
            return 1.0
        if rule == "sqrt_nresp":
            return float(min(wmax, np.sqrt(max(1, n))))
        return float(whi if n >= thr else wlo)
    
    
    def _uq_collect_dyfi_cdi_obs(
        self,
        imtU,
        measurement_sigma=0.30,
        measurement_sigma_dyfi=None,
        attach_per_obs_sigma=True,
        include_weights=True,
    ):
        """
        Build obs list from CDI file (MMI only).
        """
        import numpy as np
    
        df = self._uq_load_dyfi_cdi_df()
        if df is None or len(df) == 0:
            return [], {"total": 0, "intensity": 0, "seismic": 0, "unknown": 0}
    
        if str(imtU).upper().strip() not in ("MMI", "INTENSITY"):
            return [], {"total": 0, "intensity": 0, "seismic": 0, "unknown": 0}
    
        max_dist = float(getattr(self, "dyfi_cdi_max_dist_km", 400.0))
        min_nresp = int(getattr(self, "dyfi_cdi_min_nresp", 1))
    
        obs = []
        for _, r in df.iterrows():
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
    
            # sigma: prefer CDI std; then dyfi scalar; then base
            if np.isfinite(std) and float(std) > 0:
                sigma_obs = float(std)
            elif measurement_sigma_dyfi is not None:
                sigma_obs = float(measurement_sigma_dyfi)
            else:
                sigma_obs = float(measurement_sigma)
    
            o = {
                "lat": float(lat),
                "lon": float(lon),
                "value": float(cdi),
                "domain": "intensity",
                "source": "dyfi_cdi",
                "type": "dyfi_cdi",
            }
            if include_weights:
                o["w"] = float(self._uq_dyfi_weight_from_row(nresp))
            if attach_per_obs_sigma:
                o["sigma_obs"] = float(sigma_obs)
                o["meas_var"] = float(max(1e-12, sigma_obs ** 2))
    
            # keep helpful metadata
            if np.isfinite(nresp):
                o["nresp"] = float(nresp)
            if np.isfinite(dist):
                o["distance_km"] = float(dist)
            if np.isfinite(std):
                o["stddev"] = float(std)
            if np.isfinite(suspect):
                o["suspect"] = float(suspect)
    
            obs.append(o)
    
        return obs, {"total": len(obs), "intensity": len(obs), "seismic": 0, "unknown": 0}


    def _uq_obs_bounds(self, obs_list):
        import numpy as np
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
        import numpy as np
        logger = logging.getLogger(__name__)

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
        logger.info(
            "[UQ OBS POOL] v%s counts=%s bounds=%s",
            int(version),
            summary["counts"],
            summary["bounds"],
        )
        return {
            "stations_raw": stations_raw,
            "dyfi_stationlist_raw": dyfi_stationlist_raw,
            "cdi_raw": cdi_raw,
            "obs_seismic": obs_seismic,
            "obs_intensity_stationlist": obs_intensity_stationlist,
            "obs_intensity_cdi": obs_intensity_cdi,
            "summary": summary,
        }
    
    
    def _uq_debug_obs_signature(self, obs_list, label="obs"):
        """Small debug print: domains + meas_var presence."""
        import numpy as np
        if obs_list is None:
            obs_list = []
        n = len(obs_list)
        dom = {"seismic": 0, "intensity": 0, "unknown": 0}
        mv = []
        for o in obs_list:
            d = str(o.get("domain", "unknown")).lower().strip()
            dom[d] = dom.get(d, 0) + 1
            if o.get("meas_var", None) is not None:
                try:
                    mv.append(float(o["meas_var"]))
                except Exception:
                    pass
        mv = np.asarray(mv, float) if mv else np.asarray([], float)
        if mv.size:
            mv = mv[np.isfinite(mv)]
        msg = f"[UQ DEBUG] {label}: n={n} dom={dom}"
        if mv.size:
            msg += f" meas_var(n={mv.size} min={float(np.min(mv)):.4g} med={float(np.median(mv)):.4g} max={float(np.max(mv)):.4g})"
        else:
            msg += " meas_var(n=0)"
        print(msg)


    def _uq_select_dyfi_source_for_version(self, version, dyfi_source=None):
        from pathlib import Path
        mode = dyfi_source if dyfi_source is not None else getattr(self, "dyfi_source", "stationlist")
        mode = str(mode).lower().strip()
        if mode == "cdi":
            if getattr(self, "dyfi_cdi_file", None) and Path(str(self.dyfi_cdi_file)).exists():
                return "cdi"
            return "stationlist"
        if mode != "auto":
            return "stationlist"
        if getattr(self, "dyfi_cdi_file", None) is None:
            return "stationlist"
        if not Path(str(self.dyfi_cdi_file)).exists():
            return "stationlist"
        if int(version) >= int(getattr(self, "cdi_attach_from_version", 4)):
            return "cdi"
        return "stationlist"

    def _select_intensity_obs(self, version, dyfi_source, obs_pool, allow_fallback=True):
        """
        Select intensity observations for a version based on dyfi_source routing.
        Returns (obs_intensity, chosen_source).
        """
        mode = dyfi_source if dyfi_source is not None else getattr(self, "dyfi_source", "stationlist")
        mode = str(mode).lower().strip()
        stationlist_obs = (obs_pool or {}).get("obs_intensity_stationlist", []) or []
        cdi_obs = (obs_pool or {}).get("obs_intensity_cdi", []) or []

        if mode == "stationlist":
            return stationlist_obs, "stationlist"

        if mode == "cdi":
            if cdi_obs:
                return cdi_obs, "cdi"
            if allow_fallback and stationlist_obs:
                return stationlist_obs, "stationlist"
            return [], "cdi"

        # auto: stationlist until CDI becomes available for this version
        cdi_ready = int(version) >= int(getattr(self, "cdi_attach_from_version", 4))
        if cdi_ready and cdi_obs:
            return cdi_obs, "cdi"
        if stationlist_obs:
            return stationlist_obs, "stationlist"
        if allow_fallback and cdi_obs:
            return cdi_obs, "cdi"
        return [], "stationlist"


    def _uq_obs_audit_summary(self, obs_list):
        import numpy as np
        src_counts = {}
        dom_counts = {"seismic": 0, "intensity": 0, "unknown": 0}
        mv_by_source = {}
        for o in obs_list or []:
            src = str(o.get("source", "unknown"))
            dom = str(o.get("domain", "unknown")).lower().strip()
            dom_counts[dom] = dom_counts.get(dom, 0) + 1
            src_counts[src] = src_counts.get(src, 0) + 1
            mv = o.get("meas_var", None)
            if mv is not None:
                mv_by_source.setdefault(src, []).append(float(mv))
        mv_stats = {}
        for src, vals in mv_by_source.items():
            arr = np.asarray(vals, float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                mv_stats[src] = {
                    "n": int(arr.size),
                    "min": float(np.min(arr)),
                    "median": float(np.median(arr)),
                    "max": float(np.max(arr)),
                }
        return {"counts_by_source": src_counts, "counts_by_domain": dom_counts, "meas_var_stats": mv_stats}


    def _uq_get_obs_for_update(
        self,
        version,
        target_imt,
        method,
        dyfi_source=None,
        measurement_sigma=0.30,
        measurement_sigma_instr=None,
        measurement_sigma_dyfi=None,
        attach_per_obs_sigma=True,
        gmice_model="Globalgmice",
        conversion_sigma_mmi=0.6,
        conversion_sigma_pga_ln=0.6,
        allow_inverse_gmice=False,
        dyfi_weight_mode="sqrt_nresp",
        dyfi_w_max=10.0,
    ):
        import numpy as np
        logger = logging.getLogger(__name__)

        v = int(version)
        imtU = str(target_imt).upper().strip()
        method_u = str(method).lower().strip()

        obs_pool = (self.uq_state or {}).get("obs_by_version", {}).get(v, None)
        if obs_pool is None:
            return [], {"total": 0, "seismic": 0, "intensity": 0, "unknown": 0}

        intensity_obs, dyfi_src_eff = self._select_intensity_obs(
            v,
            dyfi_source=dyfi_source,
            obs_pool=obs_pool,
            allow_fallback=True,
        )
        seismic_obs = obs_pool.get("obs_seismic", [])

        use_mixed = method_u in ("bayes_2lik", "dyfi_weighted")
        selected = []

        def _pick_sigma(domain, obs):
            if obs.get("sigma_obs", None) is not None:
                try:
                    so = float(obs.get("sigma_obs"))
                    if np.isfinite(so) and so > 0:
                        return so
                except Exception:
                    pass
            if domain == "intensity" and obs.get("stddev", None) is not None:
                try:
                    so = float(obs.get("stddev"))
                    if np.isfinite(so) and so > 0:
                        return so
                except Exception:
                    pass
            if domain == "seismic" and measurement_sigma_instr is not None:
                return float(measurement_sigma_instr)
            if domain == "intensity" and measurement_sigma_dyfi is not None:
                return float(measurement_sigma_dyfi)
            return float(measurement_sigma)

        def _append_obs(lat, lon, value, domain, source, w, unit=None, meta=None, sigma_obs=None, conv_sigma=None):
            if unit is not None and self._uq_is_lognormal_imt(imtU):
                try:
                    value = float(self._uq_convert_units([value], imtU, unit, self._uq_default_imt_unit(imtU))[0])
                except Exception:
                    value = float(value)
            y_ws = self._uq_obs_to_ws(imtU, value)
            if y_ws is None or not np.isfinite(y_ws):
                return
            if attach_per_obs_sigma:
                if sigma_obs is None:
                    sigma_obs = _pick_sigma(domain, meta or {})
                meas_var = float(max(1e-12, sigma_obs ** 2))
                if conv_sigma is not None:
                    meas_var += float(conv_sigma) ** 2
            else:
                meas_var = None
            rec = {
                "lat": float(lat),
                "lon": float(lon),
                "value": float(value),
                "y_ws": float(y_ws),
                "domain": domain,
                "source": source,
                "w": float(w),
            }
            if meas_var is not None:
                rec["meas_var"] = float(meas_var)
                rec["sigma_obs"] = float(np.sqrt(meas_var))
            if meta:
                rec.update({k: v for k, v in meta.items() if k not in rec})
            selected.append(rec)

        if imtU == "MMI":
            for o in intensity_obs:
                _append_obs(
                    o["lat"],
                    o["lon"],
                    o["value"],
                    "intensity",
                    o.get("source", dyfi_src_eff),
                    o.get("w", 1.0),
                    meta={"nresp": o.get("nresp", None)},
                    sigma_obs=_pick_sigma("intensity", o),
                )
            if use_mixed:
                for o in seismic_obs:
                    if str(o.get("imt", "")).upper().strip() != "PGA":
                        continue
                    val = o.get("value")
                    if val is None:
                        continue
                    unit = o.get("unit", "%g")
                    val_pct = self._uq_convert_units([val], "PGA", unit, "%g")[0]
                    mmi = self._uq_convert_cross_imt_gmice(val_pct, "PGA", "MMI", gmice_model=gmice_model)
                    if mmi is None:
                        continue
                    _append_obs(
                        o["lat"],
                        o["lon"],
                        float(mmi),
                        "seismic",
                        o.get("source", "stationlist"),
                        o.get("w", 1.0),
                        meta={"converted_from": "PGA"},
                        sigma_obs=_pick_sigma("seismic", o),
                        conv_sigma=conversion_sigma_mmi,
                    )
        else:
            for o in seismic_obs:
                if str(o.get("imt", "")).upper().strip() != imtU:
                    continue
                _append_obs(
                    o["lat"],
                    o["lon"],
                    o["value"],
                    "seismic",
                    o.get("source", "stationlist"),
                    o.get("w", 1.0),
                    unit=o.get("unit", None),
                    sigma_obs=_pick_sigma("seismic", o),
                )
            if use_mixed and allow_inverse_gmice and imtU == "PGA":
                for o in intensity_obs:
                    mmi = o.get("value", None)
                    if mmi is None:
                        continue
                    pga = self._uq_convert_cross_imt_gmice(mmi, "MMI", "PGA", gmice_model=gmice_model)
                    if pga is None:
                        continue
                    _append_obs(
                        o["lat"],
                        o["lon"],
                        float(pga),
                        "intensity",
                        o.get("source", dyfi_src_eff),
                        o.get("w", 1.0),
                        meta={"converted_from": "MMI"},
                        sigma_obs=_pick_sigma("intensity", o),
                        conv_sigma=conversion_sigma_pga_ln,
                    )

        if method_u == "dyfi_weighted" and selected:
            weighted = []
            for o in selected:
                oo = dict(o)
                if str(oo.get("domain", "")).lower().strip() == "intensity":
                    nresp = oo.get("nresp", None)
                    try:
                        n = float(nresp) if nresp is not None else np.nan
                    except Exception:
                        n = np.nan
                    if dyfi_weight_mode == "sqrt_nresp" and np.isfinite(n):
                        oo["w"] = float(min(dyfi_w_max, np.sqrt(max(1.0, n))))
                    elif dyfi_weight_mode != "none":
                        oo["w"] = float(self._uq_dyfi_weight_from_row(n))
                weighted.append(oo)
            selected = weighted

        audit = self._uq_obs_audit_summary(selected)
        logger.info(
            "[UQ OBS ADAPTER] v=%s imt=%s method=%s dyfi_source=%s counts=%s meas_var=%s",
            v,
            imtU,
            method_u,
            dyfi_src_eff,
            audit.get("counts_by_domain"),
            audit.get("meas_var_stats"),
        )

        counts = {
            "total": len(selected),
            "seismic": audit["counts_by_domain"].get("seismic", 0),
            "intensity": audit["counts_by_domain"].get("intensity", 0),
            "unknown": audit["counts_by_domain"].get("unknown", 0),
        }
        return selected, counts
    
    
    def _uq_collect_observations(self, version: int, imt: str):
        """
        IMPORTANT PATCH:
        - For MMI: return BOTH
            (a) stationlist DYFI intensity points
            (b) stationlist instrumented intensity (seismic) points, if present
          This fixes: MMI had only DYFI → bayes_2lik looked identical to bayes.
        - For PGA/PGV: return instrumented points (as before).
        """
        import numpy as np
    
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet.")
    
        v = int(version)
        imt_u = str(imt).upper().strip()
        obs = []

        obs_pool = self.uq_state.get("obs_by_version", {}).get(v, None)
        if obs_pool is not None:
            if imt_u == "MMI":
                for o in obs_pool.get("obs_intensity_stationlist", []):
                    obs.append(
                        {
                            "lat": o["lat"],
                            "lon": o["lon"],
                            "value": o["value"],
                            "w": o.get("w", 1.0),
                            "type": o.get("type", "DYFI"),
                            "unit": "MMI",
                            "source": o.get("source", "stationlist"),
                            "nresp": o.get("nresp", None),
                        }
                    )
                return obs
            if imt_u in ("PGA", "PGV", "PSA"):
                for o in obs_pool.get("obs_seismic", []):
                    if str(o.get("imt", "")).upper().strip() != imt_u:
                        continue
                    obs.append(
                        {
                            "lat": o["lat"],
                            "lon": o["lon"],
                            "value": o["value"],
                            "w": o.get("w", 1.0),
                            "type": o.get("type", "instrumented"),
                            "unit": o.get("unit", None),
                            "source": o.get("source", "stationlist"),
                        }
                    )
                return obs
            return obs

        raw = self.uq_state["versions_raw"].get(v, None)
        if raw is None:
            return []

        stations = raw.get("stations", {})
        dyfi = stations.get("dyfi", []) or []
        inst = stations.get("instrumented", []) or []
    
        if imt_u == "MMI":
            # (a) DYFI from stationlist
            for o in dyfi:
                val = o.get("intensity", None)
                try:
                    val = float(val)
                except Exception:
                    continue
                if not np.isfinite(val):
                    continue
                rec = {
                    "lat": float(o["lat"]),
                    "lon": float(o["lon"]),
                    "value": float(val),
                    "w": float(o.get("w", 1.0)),
                    "type": "DYFI",
                    "unit": "MMI",
                    "source": "stationlist",
                }
                # keep per-observation sigma if it exists
                if o.get("intensity_stddev", None) is not None:
                    rec["intensity_stddev"] = o.get("intensity_stddev", None)
                if o.get("nresp", None) is not None:
                    rec["nresp"] = o.get("nresp", None)
                obs.append(rec)
    
            # (b) instrumented intensity (seismic) from stationlist (if present)
            for o in inst:
                val = o.get("intensity", None)
                if val is None:
                    continue
                try:
                    val = float(val)
                except Exception:
                    continue
                if not np.isfinite(val):
                    continue
                rec = {
                    "lat": float(o["lat"]),
                    "lon": float(o["lon"]),
                    "value": float(val),
                    "w": float(o.get("w", 1.0)),
                    "type": "seismic_intensity",
                    "unit": "MMI",
                    "source": "stationlist",
                }
                if o.get("intensity_stddev", None) is not None:
                    rec["intensity_stddev"] = o.get("intensity_stddev", None)
                obs.append(rec)
    
            return obs
    
        if imt_u == "PGA":
            for o in inst:
                val = o.get("pga", None)
                try:
                    val = float(val)
                except Exception:
                    continue
                if not np.isfinite(val):
                    continue
                rec = {
                    "lat": float(o["lat"]),
                    "lon": float(o["lon"]),
                    "value": float(val),
                    "w": float(o.get("w", 1.0)),
                    "type": "instrumented",
                    "unit": o.get("pga_unit", "%g"),
                    "source": "stationlist",
                }
                obs.append(rec)
            return obs
    
        if imt_u == "PGV":
            for o in inst:
                val = o.get("pgv", None)
                try:
                    val = float(val)
                except Exception:
                    continue
                if not np.isfinite(val):
                    continue
                rec = {
                    "lat": float(o["lat"]),
                    "lon": float(o["lon"]),
                    "value": float(val),
                    "w": float(o.get("w", 1.0)),
                    "type": "instrumented",
                    "unit": o.get("pgv_unit", "cm/s"),
                    "source": "stationlist",
                }
                obs.append(rec)
            return obs
    
        return []
    
    
    def _uq_collect_obs_for_version(
        self,
        version,
        imtU,
        measurement_sigma=0.30,
        include_weights=True,
        prefer_domain=True,
        allow_fallback=True,
        measurement_sigma_instr=None,
        measurement_sigma_dyfi=None,
        attach_per_obs_sigma=False,
        dyfi_source=None,  # None -> use self.dyfi_source ; "stationlist"|"cdi"|"auto"
    ):
        """
        Unified obs collector (legacy wrapper).
        Uses the obs adapter to return working-space-ready obs for the requested IMT.
        """
        method = "bayes" if prefer_domain else "bayes_2lik"
        obs, counts = self._uq_get_obs_for_update(
            int(version),
            str(imtU).upper().strip(),
            method=method,
            dyfi_source=dyfi_source,
            measurement_sigma=measurement_sigma,
            measurement_sigma_instr=measurement_sigma_instr,
            measurement_sigma_dyfi=measurement_sigma_dyfi,
            attach_per_obs_sigma=attach_per_obs_sigma,
            dyfi_weight_mode="none",
        )
        if prefer_domain and not obs and not allow_fallback:
            return [], {"total": 0, "seismic": 0, "intensity": 0, "unknown": 0}
        return obs, counts
    
    
    def uq_extract_target_series(
        self,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        agg="mean",
        global_stat=None,
        sigma_total_from_shakemap=True,
        sigma_aleatory=None,
        prior_version=None,
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
        measurement_sigma_instr=None,
        measurement_sigma_dyfi=None,
        ok_range_km=60.0,
        ok_variogram="exponential",
        ok_nugget=1e-6,
        ok_sill=None,
        ok_cap_sigma_to_prior=True,
        # dyfi weighting method
        dyfi_weight_mode="sqrt_nresp",
        dyfi_w_max=10.0,
        # unified grid controls
        grid_res=None,
        interp_method="nearest",
        interp_kwargs=None,
        # Option B safety
        methods_to_compute=None,
        # audit
        audit=True,
        audit_output_path=None,
        audit_prefix=None,
        # NEW: route DYFI source downstream
        dyfi_source=None,
    ):
        """
        Target series extraction:
          - ShakeMap: mean & sigma_total from file-derived unified grid
          - bayes: v0 prior + prefer_domain=True obs
          - bayes_2lik: v0 prior + prefer_domain=False obs + meas_var if present
          - dyfi_weighted: bayes_2lik + DYFI weights (nresp-based) on intensity points
          - kriging: ordinary kriging on residuals in working space
          - dyfi_source: stationlist | cdi | auto
        """
        import numpy as np
        import pandas as pd
        logger = logging.getLogger(__name__)
    
        def _norm_method(m):
            s = str(m).strip()
            return "ShakeMap" if s.lower() == "published" else s
    
        if methods_to_compute is None:
            compute = {"ShakeMap", "bayes", "bayes_2lik", "dyfi_weighted"}
        else:
            compute = {_norm_method(m) for m in methods_to_compute}
            compute.add("ShakeMap")
    
        imtU = str(imt).upper().strip()
        versions = sorted([int(v) for v in version_list])
    
        # targets
        if global_stat is not None:
            gs = str(global_stat).lower().strip()
            gid = f"GLOBAL_{gs.replace(' ', '_')}".replace("__", "_")
            targets = [{"id": gid, "type": "global"}]
        else:
            targets = self._uq_parse_targets(points=points, areas=areas)
    
        # unified grid
        _, lat2d, lon2d = self._uq_get_unified_for_versions(
            versions, imt=imtU, grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
        )
    
        v0 = int(prior_version) if prior_version is not None else int(versions[0])
    
        # prior fields
        mu0_lin, sig0_raw = self._uq_get_mu_sigma_unified(
            v0, imtU, lat2d, lon2d, grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
        )
        mu0_ws = self._uq_mu_to_ws(imtU, mu0_lin)
    
        _sig_tot0, s_a0, s_ep0 = self._uq_decompose_sigma_safe(
            imtU, sig0_raw, sigma_aleatory=sigma_aleatory, sigma_total_from_shakemap=sigma_total_from_shakemap
        )
    
        rows = []
        agg_effective = str(agg).lower().strip()
    
        for t in targets:
            if t.get("type") == "global":
                mask = np.isfinite(lat2d)
                ttype = "global"
            else:
                mask, _meta = self._uq_target_mask(t, lat2d, lon2d)
                ttype = t.get("type", "target")
    
            n_cells = int(np.sum(mask)) if mask is not None else 0
            if (mask is None) or (n_cells <= 0):
                continue
    
            # v0 scalars
            mu0_lin_t = self._uq_agg(mu0_lin[mask], agg=agg_effective)
            sig0_raw_t = self._uq_agg(sig0_raw[mask], agg=agg_effective)
            s_a0_t = self._uq_agg(s_a0[mask], agg=agg_effective)
            s_ep0_t = self._uq_agg(s_ep0[mask], agg=agg_effective)
    
            for v in versions:
                mu_v_lin, sig_v_raw = self._uq_get_mu_sigma_unified(
                    int(v), imtU, lat2d, lon2d, grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
                )
                mean_pub = self._uq_agg(mu_v_lin[mask], agg=agg_effective)
                sig_pub = self._uq_agg(sig_v_raw[mask], agg=agg_effective)
    
                # published
                rows.append({
                    "target_id": t.get("id", "GLOBAL"),
                    "target_type": ttype,
                    "version": int(v),
                    "imt": imtU,
                    "method": "ShakeMap",
                    "mean_predicted": float(mean_pub),
                    "sigma_total_predicted": float(sig_pub),
                    "sigma_epistemic_predicted": np.nan,
                    "sigma_aleatoric_used": np.nan,
                    "n_obs_total": 0,
                    "n_obs_seismic": 0,
                    "n_obs_intensity": 0,
                    "n_obs_unknown": 0,
                    "n_cells": int(n_cells),
                })
    
                # obs streams via adapter (dyfi_source aware)
                obs_pref, c_pref = self._uq_get_obs_for_update(
                    int(v),
                    imtU,
                    method="bayes",
                    dyfi_source=dyfi_source,
                    measurement_sigma=measurement_sigma,
                    measurement_sigma_instr=measurement_sigma_instr,
                    measurement_sigma_dyfi=measurement_sigma_dyfi,
                    attach_per_obs_sigma=True,
                    dyfi_weight_mode="none",
                )
                obs_mix, c_mix = self._uq_get_obs_for_update(
                    int(v),
                    imtU,
                    method="bayes_2lik",
                    dyfi_source=dyfi_source,
                    measurement_sigma=measurement_sigma,
                    measurement_sigma_instr=measurement_sigma_instr,
                    measurement_sigma_dyfi=measurement_sigma_dyfi,
                    attach_per_obs_sigma=True,
                    dyfi_weight_mode="none",
                )
    
                if getattr(self, "debug_uq", False) or getattr(self, "uq_debug", False):
                    self._uq_debug_obs_signature(obs_pref, label=f"bayes v{int(v):03d} dyfi_source={dyfi_source or getattr(self,'dyfi_source','stationlist')}")
                    self._uq_debug_obs_signature(obs_mix,  label=f"bayes_2lik v{int(v):03d} dyfi_source={dyfi_source or getattr(self,'dyfi_source','stationlist')}")
    
                # bayes (domain-preferred)
                if "bayes" in compute:
                    if c_pref.get("total", 0) == 0:
                        mean_b, sig_b, s_ep_b_t = mu0_lin_t, sig0_raw_t, s_ep0_t
                    else:
                        mu_b_ws, s_ep_b = self._uq_bayes_local_posterior_at_mask(
                            mu0_ws, s_ep0, s_a0,
                            lat2d, lon2d, mask, [o.copy() for o in obs_pref],
                            update_radius_km=update_radius_km,
                            kernel=kernel,
                            kernel_scale_km=kernel_scale_km,
                            measurement_sigma=measurement_sigma,  # fallback only
                        )
                        mu_b_lin = self._uq_mu_from_ws(imtU, mu_b_ws)
                        mean_b = self._uq_agg(mu_b_lin[mask], agg=agg_effective)
                        s_ep_b_t = self._uq_agg(s_ep_b[mask], agg=agg_effective)
                        sig_b = float(np.sqrt(max(0.0, float(s_a0_t) ** 2 + float(s_ep_b_t) ** 2)))
    
                    rows.append({
                        "target_id": t.get("id", "GLOBAL"),
                        "target_type": ttype,
                        "version": int(v),
                        "imt": imtU,
                        "method": "bayes",
                        "mean_predicted": float(mean_b),
                        "sigma_total_predicted": float(sig_b),
                        "sigma_epistemic_predicted": float(s_ep_b_t),
                        "sigma_aleatoric_used": float(s_a0_t),
                        "n_obs_total": int(c_pref.get("total", 0)),
                        "n_obs_seismic": int(c_pref.get("seismic", 0)),
                        "n_obs_intensity": int(c_pref.get("intensity", 0)),
                        "n_obs_unknown": int(c_pref.get("unknown", 0)),
                        "n_cells": int(n_cells),
                    })
    
                # bayes_2lik (mixed + meas_var)
                if "bayes_2lik" in compute:
                    if c_mix.get("total", 0) == 0:
                        mean2, sig2, s_ep2_t = mu0_lin_t, sig0_raw_t, s_ep0_t
                    else:
                        mu2_ws, s_ep2 = self._uq_bayes_local_posterior_2lik_at_mask(
                            mu0_ws, s_ep0, s_a0,
                            lat2d, lon2d, mask, [o.copy() for o in obs_mix],
                            update_radius_km=update_radius_km,
                            kernel=kernel,
                            kernel_scale_km=kernel_scale_km,
                            measurement_sigma=measurement_sigma,  # fallback only
                        )
                        mu2_lin = self._uq_mu_from_ws(imtU, mu2_ws)
                        mean2 = self._uq_agg(mu2_lin[mask], agg=agg_effective)
                        s_ep2_t = self._uq_agg(s_ep2[mask], agg=agg_effective)
                        sig2 = float(np.sqrt(max(0.0, float(s_a0_t) ** 2 + float(s_ep2_t) ** 2)))
    
                    rows.append({
                        "target_id": t.get("id", "GLOBAL"),
                        "target_type": ttype,
                        "version": int(v),
                        "imt": imtU,
                        "method": "bayes_2lik",
                        "mean_predicted": float(mean2),
                        "sigma_total_predicted": float(sig2),
                        "sigma_epistemic_predicted": float(s_ep2_t),
                        "sigma_aleatoric_used": float(s_a0_t),
                        "n_obs_total": int(c_mix.get("total", 0)),
                        "n_obs_seismic": int(c_mix.get("seismic", 0)),
                        "n_obs_intensity": int(c_mix.get("intensity", 0)),
                        "n_obs_unknown": int(c_mix.get("unknown", 0)),
                        "n_cells": int(n_cells),
                    })
    
                # dyfi_weighted (mixed + meas_var + DYFI weights)
                if "dyfi_weighted" in compute:
                    obs_w, c_w = self._uq_get_obs_for_update(
                        int(v),
                        imtU,
                        method="dyfi_weighted",
                        dyfi_source=dyfi_source,
                        measurement_sigma=measurement_sigma,
                        measurement_sigma_instr=measurement_sigma_instr,
                        measurement_sigma_dyfi=measurement_sigma_dyfi,
                        attach_per_obs_sigma=True,
                        dyfi_weight_mode=dyfi_weight_mode,
                        dyfi_w_max=dyfi_w_max,
                    )
                    if c_w.get("total", 0) == 0:
                        meanw, sigw, s_epw_t = mu0_lin_t, sig0_raw_t, s_ep0_t
                    else:
                        muw_ws, s_epw = self._uq_dyfi_weighted_posterior_at_mask(
                            mu0_ws, s_ep0, s_a0,
                            lat2d, lon2d, mask, [o.copy() for o in obs_w],
                            update_radius_km=update_radius_km,
                            kernel=kernel,
                            kernel_scale_km=kernel_scale_km,
                            measurement_sigma=measurement_sigma,  # fallback only
                        )
                        muw_lin = self._uq_mu_from_ws(imtU, muw_ws)
                        meanw = self._uq_agg(muw_lin[mask], agg=agg_effective)
                        s_epw_t = self._uq_agg(s_epw[mask], agg=agg_effective)
                        sigw = float(np.sqrt(max(0.0, float(s_a0_t) ** 2 + float(s_epw_t) ** 2)))
    
                    rows.append({
                        "target_id": t.get("id", "GLOBAL"),
                        "target_type": ttype,
                        "version": int(v),
                        "imt": imtU,
                        "method": "dyfi_weighted",
                        "mean_predicted": float(meanw),
                        "sigma_total_predicted": float(sigw),
                        "sigma_epistemic_predicted": float(s_epw_t),
                        "sigma_aleatoric_used": float(s_a0_t),
                        "n_obs_total": int(c_w.get("total", 0)),
                        "n_obs_seismic": int(c_w.get("seismic", 0)),
                        "n_obs_intensity": int(c_w.get("intensity", 0)),
                        "n_obs_unknown": int(c_w.get("unknown", 0)),
                        "n_cells": int(n_cells),
                    })

                # kriging (residual kriging in working space)
                if "kriging" in compute:
                    audit_krig = self._uq_obs_audit_summary(obs_mix)
                    logger.info(
                        "[UQ KRIGING] v=%s imt=%s n_obs=%s domains=%s variogram=%s range_km=%s nugget=%s sill=%s",
                        int(v),
                        imtU,
                        int(c_mix.get("total", 0)),
                        audit_krig.get("counts_by_domain"),
                        ok_variogram,
                        float(ok_range_km),
                        float(ok_nugget),
                        "auto" if ok_sill is None else float(ok_sill),
                    )
                    if c_mix.get("total", 0) == 0:
                        meank, sigk, s_epk_t = mu0_lin_t, sig0_raw_t, s_ep0_t
                    else:
                        muk_ws, s_epk = self._uq_ok_residual_posterior_at_mask(
                            mu0_ws, s_ep0, s_a0,
                            lat2d, lon2d, mask, [o.copy() for o in obs_mix],
                            variogram=ok_variogram,
                            range_km=ok_range_km,
                            nugget=ok_nugget,
                            sill=ok_sill,
                            measurement_sigma=measurement_sigma,
                            sigma_ep_cap_to_prior=ok_cap_sigma_to_prior,
                        )
                        muk_lin = self._uq_mu_from_ws(imtU, muk_ws)
                        meank = self._uq_agg(muk_lin[mask], agg=agg_effective)
                        s_epk_t = self._uq_agg(s_epk[mask], agg=agg_effective)
                        sigk = float(np.sqrt(max(0.0, float(s_a0_t) ** 2 + float(s_epk_t) ** 2)))

                    rows.append({
                        "target_id": t.get("id", "GLOBAL"),
                        "target_type": ttype,
                        "version": int(v),
                        "imt": imtU,
                        "method": "kriging",
                        "mean_predicted": float(meank),
                        "sigma_total_predicted": float(sigk),
                        "sigma_epistemic_predicted": float(s_epk_t),
                        "sigma_aleatoric_used": float(s_a0_t),
                        "n_obs_total": int(c_mix.get("total", 0)),
                        "n_obs_seismic": int(c_mix.get("seismic", 0)),
                        "n_obs_intensity": int(c_mix.get("intensity", 0)),
                        "n_obs_unknown": int(c_mix.get("unknown", 0)),
                        "n_cells": int(n_cells),
                    })
    
        return pd.DataFrame(rows)
    
    
    def uq_plot_targets_decay(
        self,
        *,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        what="sigma_total_predicted",
        methods=("ShakeMap", "bayes", "bayes_2lik"),
        agg="mean",
        global_stat=None,
        prior_version=None,
        sigma_total_from_shakemap=True,
        sigma_aleatory=None,
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
        measurement_sigma_instr=None,
        measurement_sigma_dyfi=None,
        dyfi_weight_mode="sqrt_nresp",
        dyfi_w_max=10.0,
        grid_res=None,
        interp_method="nearest",
        interp_kwargs=None,
        figsize=(10, 5),
        dpi=150,
        xrotation=0,
        title=None,
        output_path=None,
        save=True,
        save_formats=("png",),
        show=False,
        plot_combined=True,
        combined_figsize=(11, 6),
        combined_legend_ncol=2,
        audit=True,
        audit_output_path=None,
        audit_prefix=None,
        dyfi_source=None,  # NEW
    ):
        """
        Plot target decay curves for points/areas.
        """
        import matplotlib.pyplot as plt
    
        imtU = str(imt).upper().strip()
        versions = sorted([int(v) for v in version_list])
    
        df = self.uq_extract_target_series(
            version_list=versions,
            imt=imtU,
            points=points,
            areas=areas,
            agg=agg,
            global_stat=global_stat,
            sigma_total_from_shakemap=sigma_total_from_shakemap,
            sigma_aleatory=sigma_aleatory,
            prior_version=prior_version,
            update_radius_km=update_radius_km,
            kernel=kernel,
            kernel_scale_km=kernel_scale_km,
            measurement_sigma=measurement_sigma,
            measurement_sigma_instr=measurement_sigma_instr,
            measurement_sigma_dyfi=measurement_sigma_dyfi,
            dyfi_weight_mode=dyfi_weight_mode,
            dyfi_w_max=dyfi_w_max,
            grid_res=grid_res,
            interp_method=interp_method,
            interp_kwargs=interp_kwargs,
            methods_to_compute=methods,
            audit=audit,
            audit_output_path=audit_output_path,
            audit_prefix=audit_prefix,
            dyfi_source=dyfi_source,
        )
    
        targets = sorted(df["target_id"].unique().tolist())
        for tid in targets:
            sub = df[(df["target_id"] == tid) & (df["method"].isin(methods))].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["version", "method"])
    
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            for m in methods:
                s = sub[sub["method"] == m].sort_values("version")
                if s.empty:
                    continue
                ax.plot(s["version"].values, s[what].values, marker="o", linewidth=1.5, label=str(m))
    
            ax.set_xlabel("ShakeMap version")
            ax.set_ylabel(str(what))
            ax.tick_params(axis="x", rotation=float(xrotation))
            ax.grid(True, alpha=0.3)
    
            ttl = title if title is not None else f"{imtU} target decay @ {tid} ({what})"
            ax.set_title(ttl)
            ax.legend()
    
            fig.tight_layout()
            if save and (output_path is not None):
                self._uq_save_figure_safe(
                    fig,
                    fname_stem=f"UQ-TargetDecay-{tid}-{imtU}-{what}",
                    subdir="uq_plots/targets_decay",
                    output_path=output_path,
                    save_formats=save_formats,
                    dpi=dpi,
                )
            if show:
                plt.show()
            else:
                plt.close(fig)
    
        return df
