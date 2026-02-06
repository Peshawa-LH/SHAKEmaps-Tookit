"""
Analyze Time-Dependent SHAKEmap Evolution
Class Name: SHAKEtime

Description:
    The SHAKEtime class provides tools to analyze the temporal evolution of SHAKEmap products for a
    single seismic event. It processes multiple SHAKEmap versions (grid, uncertainty, station, rupture,
    and optional PAGER products) to quantify how shaking estimates and uncertainties change as new
    information becomes available.

    The class supports unified-grid comparisons across versions, computation of global and local update
    metrics, temporal evolution plots, and uncertainty quantification (UQ) workflows, including
    Bayesian-style uncertainty updating. SHAKEtime is intended for rapid-response analysis, SHAKEmap
    reliability studies, and research on time-evolving SHAKEmaps.

Core Functionality:
    • Versioned ingestion of SHAKEmap products
    • Unified spatial grid construction for cross-version comparison
    • Delta mapping and temporal evolution analysis
    • Optional PAGER analysis and auxiliary diagnostics
    • Uncertainty Quantification (UQ) and Bayesian updating tools

Prerequisites:
    SHAKEmap products must be available locally (e.g., fetched via SHAKEfetch) and follow standard USGS
    directory structures and naming conventions. Required dependencies include NumPy, Pandas,
    Matplotlib, SciPy, and XML parsing libraries.

Relationship to SHAKEmaps:
    “SHAKEmaps” refers to the general family of ShakeMap-type procedures.
    “ShakeMap” refers to the USGS implementation.
    “shakemap” refers to a specific calculated product.

Date:
    January, 2026
Version:
    26.5

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


from typing import List 
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
from modules.SHAKEtools import *

from modules.SHAKEgmice import *


try:
    from modules.SHAKEuq import *
except Exception:
    SHAKEuq = None




# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')



class SHAKEtime:
    """
    SHAKEtime: A Class for Time-Dependent SHAKEmap Evolution and Rapid-Response Analysis (v26.1.4)

    Overview
    ========
    The SHAKEtime class analyzes the *temporal evolution* of earthquake shaking products by processing
    multiple versions of ShakeMap-derived files (grid + uncertainty + stationlist + rupture) and, when
    available, PAGER products. The class is designed to reconstruct how SHAKEmap products change through
    time as additional information becomes available (stations, macroseismic observations, refined rupture,
    etc.), and to quantify/visualize those changes using unified-grid comparisons, global and local metrics,
    and uncertainty quantification (UQ) workflows.

    Core Capabilities (v26.1.4)
    ==========================
    1) Versioned ingestion and bookkeeping
       - Reads multiple ShakeMap grid XML versions and associated metadata (event parameters, grid specs).
       - Optional PAGER XML ingestion for alert probabilities and exposure time-series.
       - Built-in version zero-padding helpers to keep filenames consistent across products.

    2) Unified-grid workflow for cross-version comparison
       - Constructs a common spatial grid across versions (intersection-based unification) so that
         direct, cell-by-cell comparisons are valid.
       - Computes per-version statistics on the unified grid (mean, median, std, quantiles, skewness),
         plus bootstrap confidence intervals where requested.
       - Computes pairwise delta fields between versions and summarizes update magnitudes with global
         error metrics (e.g., MAE/RMSE/mean-diff) and correlation-based measures.

    3) Temporal evolution diagnostics and mapping
       - Hazard Temporal Display (HTD) style plots for time-series evolution of hazard/uncertainty.
       - Rate-of-change and “delta maps” to localize where (and how strongly) shaking updates.
       - Uncertainty maps (standard deviation / sigma layers when present).
       - Rupture overlay support for map visualization.

    4) Auxiliary covariates, data influence, and regression-style diagnostics
       - Builds auxiliary tables that track how inputs change between versions (counts of stations,
         macroseismic points, rupture updates, etc. when extractable).
       - Provides “data influence” plots that relate Δ(data quantity) to Δ(map update magnitude)
         derived from unified-grid delta statistics.
       - Supports exporting and re-importing auxiliary results in common tabular formats.

    5) Chaos-informed evolution metrics (research extensions)
       - Computes chaos metrics (e.g., Largest Lyapunov Exponent, Sample Entropy, Correlation Dimension)
         from version-to-version evolution signals, including rolling-window analysis and plotting.

    6) Uncertainty Quantification (UQ) framework (v26.1.4)
       - Dataset builder that exports per-version mean fields and uncertainty components into a structured
         on-disk dataset for reproducible UQ analysis.
       - Bayesian-style uncertainty updating utilities (e.g., Bayes update / hierarchical update patterns),
         supporting posterior mean and posterior epistemic sigma updates using observations.
       - Canonical UQ output routing is standardized to avoid nested folders (no "uq/uq") and keep all UQ
         outputs under a single event-scoped root.

    File Naming Conventions
    -----------------------
    The class supports two naming conventions via `file_type`:

    file_type = 1:
        - ShakeMap grid:   "{event_id}_grid_{version}.xml"
        - PAGER:           "{event_id}_pager_{version}.xml"
        - Rupture:         "{event_id}_rupture_{version}.json"

    file_type = 2 (default):
        - ShakeMap grid:   "{event_id}_us_{version}_grid.xml"
        - PAGER:           "{event_id}_us_{version}_pager.xml"
        - Rupture:         "{event_id}_us_{version}_rupture.json"

    (Versions are internally normalized using zero-padding, typically to 3 digits.)

    Inputs and Directory Layout
    ---------------------------
    SHAKEtime expects folders organized as:
        <shakemap_folder>/<event_id>/<versioned_shakemap_files>
        <pager_folder>/<event_id>/<versioned_pager_files>   (optional)

    Several analysis and export methods write outputs under:
        <output_path>/SHAKEtime/<event_id>/<module_subfolder>/

    UQ outputs (v26.1.4 canonical root):
        <base_folder>/<event_id>/uq/...

    Caching and Performance Notes
    -----------------------------
    The class caches parsed station and macroseismic datasets per version (when extracted) to reduce
    repeated I/O in iterative notebooks. Unified-grid computations can also be cached depending on the
    method options.

    Notes and Conventions
    ---------------------
    - "ShakeMap" (capitalized) refers to the USGS implementation; "shakemap" may refer to a specific
      computed product in downstream analyses.
    - Many plotting functions support consistent styling knobs (figure size, font sizes, grid style,
      save formats, DPI) and can optionally save figures/tables automatically.
    - When uncertainty layers are available, analyses distinguish epistemic vs total uncertainty where
      possible, but behavior depends on the data present in each version.

    
    © SHAKEmaps version 26.5
    """
    
    def __init__(self,
             event_id: str,
             event_time=None,
             shakemap_folder=None,
             pager_folder=None,
             file_type: int = 2):

        """
        Initialize with event details and folder paths.
        file_type 1 uses "event_id_grid_{version}.xml" while type 2 uses "event_id_us_{version}_grid.xml".
        """ 
        self.event_id = event_id
        self.event_time = (datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S")
                           if event_time else None)
        self.shakemap_folder = (os.path.normpath(shakemap_folder)
                                if shakemap_folder else None)
        self.pager_folder = (os.path.normpath(pager_folder)
                             if pager_folder else None)
        self.file_type = file_type
        self.summary_dict_list = []
        self.summary_df = pd.DataFrame()


        self._instrument_data_cache = {}     # version → station‐list DataFrame
        self._dyfi_data_cache       = {}     # version → dyfi DataFrame
        self._rupture_geom_cache    = {}     # version → rupture‐geometry dict
    
    ###################################################
    #
    #
    #
    # --- Helper methods for file names ---
    #
    #
    #
    #################################################

    def _pad_version(self, version: str) -> str:
        """Ensure numeric versions are zero‑padded to 3 digits."""
        vs = str(version)
        return vs.zfill(3) if vs.isdigit() else vs

    def _get_shakemap_filename(self, version: str) -> str:
        """Return ShakeMap filename with padded version."""
        v = self._pad_version(version)
        if self.file_type == 1:
            return f"{self.event_id}_grid_{v}.xml"
        else:
            return f"{self.event_id}_us_{v}_grid.xml"

    def _get_pager_filename(self, version: str) -> str:
        """Return Pager filename with padded version."""
        v = self._pad_version(version)
        if self.file_type == 1:
            return f"{self.event_id}_pager_{v}.xml"
        else:
            return f"{self.event_id}_us_{v}_pager.xml"

    def _get_rupture_filename(self, version: str) -> str:
        """Return Rupture filename with padded version."""
        v = self._pad_version(version)
        if self.file_type == 1:
            return f"{self.event_id}_rupture_{v}.json"
        else:
            return f"{self.event_id}_us_{v}_rupture.json"

    # --- End Helper Methods ---++

    
    

    ##################################################
    #
    #
    #
    # # --- SHAKEsummary methods  ---++
    #
    #
    #
    #################################################

    def update_summary_state(self):
        """Update internal summary DataFrame."""
        self.summary_df = pd.DataFrame(self.summary_dict_list)

    def get_shake_summary(self,
                          version_list: list,
                          shakemap_folder: str = None) -> list:
        """
        Parse ShakeMap XML for versions; update and return summary list.
        You can pass shakemap_folder here, or it will use the one from __init__.
        """
        # Decide which folder to use
        folder = os.path.normpath(shakemap_folder) if shakemap_folder else self.shakemap_folder
        if not folder:
            raise ValueError("Must provide shakemap_folder either in __init__ or here.")
    
        summary_dict_list = []
        ns = {'sm': 'http://earthquake.usgs.gov/eqcenter/shakemap'}
    
        # 1) Loop through versions and build summary entries
        for version in version_list:
            filename = self._get_shakemap_filename(version)
            xml_path = os.path.join(folder, self.event_id, filename)
            logging.info(f"Checking ShakeMap file: {xml_path}")
            if not os.path.exists(xml_path):
                logging.warning(f"File not found for version {version}: {xml_path}")
                continue
    
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                shakemap_info = root.attrib
    
                event_elem = root.find('sm:event', ns)
                event_info = event_elem.attrib if event_elem is not None else {}
    
                grid_spec_elem = root.find('sm:grid_specification', ns)
                grid_spec = grid_spec_elem.attrib if grid_spec_elem is not None else {}
    
                summary = {
                    'version': version,
                    'shakemap_version': shakemap_info.get('shakemap_version'),
                    'process_timestamp': shakemap_info.get('process_timestamp'),
                    'map_status': shakemap_info.get('map_status'),
                    'event_id': event_info.get('event_id'),
                    'magnitude': event_info.get('magnitude'),
                    'depth': event_info.get('depth'),
                    'lon': event_info.get('lon'),
                    'lat': event_info.get('lat'),
                    'event_timestamp': event_info.get('event_timestamp'),
                    'intensity_observations': event_info.get('intensity_observations'),
                    'seismic_stations': event_info.get('seismic_stations'),
                    'point_source': event_info.get('point_source'),
                    'grid_space': grid_spec.get('nominal_lon_spacing'),
                    'nlon': grid_spec.get('nlon'),
                    'nlat': grid_spec.get('nlat')
                }
                summary_dict_list.append(summary)
    
            except Exception as e:
                logging.error(f"Error processing version {version}: {e}")
    
        # 2) If no event_time was provided, grab it from the first entry's event_timestamp
        if self.event_time is None and summary_dict_list:
            first_ts = summary_dict_list[0].get('event_timestamp')
            if first_ts:
                try:
                    self.event_time = datetime.strptime(first_ts, "%Y-%m-%dT%H:%M:%S")
                    logging.info(f"Set event_time from ShakeMap: {self.event_time}")
                except Exception as e:
                    logging.error(f"Failed to parse event_timestamp '{first_ts}': {e}")
    
        # 3) Compute Time-after-Event (TaE) metrics if we have an event_time
        if self.event_time:
            for entry in summary_dict_list:
                proc_ts = entry.get("process_timestamp")
                if proc_ts:
                    try:
                        parsed_proc = datetime.strptime(proc_ts, "%Y-%m-%dT%H:%M:%S")
                        delta = parsed_proc - self.event_time
                        entry["TaE_d"] = delta.total_seconds() / 86400
                        entry["TaE_h"] = delta.total_seconds() / 3600
                        entry["TaE_m"] = delta.total_seconds() / 60
                    except Exception as e:
                        logging.error(
                            f"Error parsing process_timestamp for version "
                            f"{entry.get('version')}: {e}"
                        )
    
        # 4) Store and return
        self.summary_dict_list = summary_dict_list
        self.update_summary_state()
        return self.summary_dict_list




    def add_shakemap_pgm(self) -> list:
        """Update summary with main ShakeMap metrics: MMI, PGA, PGV."""
        if not self.summary_dict_list:
            logging.warning("Summary not created. Run get_shake_summary() first.")
            return []
        ns = {'sm': 'http://earthquake.usgs.gov/eqcenter/shakemap'}
        metrics = ["MMI", "PGA", "PGV","PSA10"]
        for entry in self.summary_dict_list:
            version = entry.get("version")
            filename = self._get_shakemap_filename(version)
            xml_path = os.path.normpath(os.path.join(self.shakemap_folder, self.event_id, filename))
            logging.info(f"Processing ground motion metrics for version {version}")
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                grid_fields = root.findall('sm:grid_field', ns)
                field_mapping = {gf.attrib.get("name").upper(): (i, gf.attrib.get("units")) 
                                 for i, gf in enumerate(grid_fields)}
                grid_data_elem = root.find('sm:grid_data', ns)
                if grid_data_elem is None:
                    logging.warning(f"No grid_data for version {version}")
                    continue
                data_text = grid_data_elem.text.strip()
                flat_data = [float(x) for x in data_text.split()]
                nfields = len(grid_fields)
                for metric in metrics:
                    if metric.upper() in field_mapping:
                        col_index, unit = field_mapping[metric.upper()]
                        values = [flat_data[i] for i in range(col_index, len(flat_data), nfields)]
                        if values:
                            entry[f"{metric.lower()}_min"] = min(values)
                            entry[f"{metric.lower()}_mean"] = sum(values) / len(values)
                            entry[f"{metric.lower()}_max"] = max(values)
                            entry[f"{metric.lower()}_std"] = statistics.stdev(values) if len(values) > 1 else 0
                            entry[f"{metric.lower()}_unit"] = unit
                        else:
                            logging.warning(f"No values for {metric} in version {version}")
                    else:
                        logging.warning(f"Metric {metric} not found in version {version}")
            except Exception as e:
                logging.error(f"Error processing grid data for version {version}: {e}")
        self.update_summary_state()
        return self.summary_dict_list

    #passed 
    def add_shakemap_stdpgm(self) -> list:
        """Update summary with uncertainty metrics: STDMMI, STDPGA, STDPGV, STDPSA10."""
        if not self.summary_dict_list:
            logging.warning("Summary not created. Run get_shake_summary() first.")
            return []
        ns = {'sm': 'http://earthquake.usgs.gov/eqcenter/shakemap'}
        metrics = ["STDMMI", "STDPGA", "STDPGV", "STDPSA10"]
        for entry in self.summary_dict_list:
            version = entry.get("version")
            filename = self._get_shakemap_filename(version).replace("grid", "uncertainty")
            xml_path = os.path.normpath(os.path.join(self.shakemap_folder, self.event_id, filename))
            logging.info(f"Processing uncertainty metrics for version {version}")
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                grid_fields = root.findall('sm:grid_field', ns)
                field_mapping = {gf.attrib.get("name").upper(): (i, gf.attrib.get("units"))
                                 for i, gf in enumerate(grid_fields)}
                grid_data_elem = root.find('sm:grid_data', ns)
                if grid_data_elem is None:
                    logging.warning(f"No grid_data in uncertainty file for version {version}")
                    continue
                data_text = grid_data_elem.text.strip()
                flat_data = [float(x) for x in data_text.split()]
                nfields = len(grid_fields)
                for metric in metrics:
                    if metric.upper() in field_mapping:
                        col_index, unit = field_mapping[metric.upper()]
                        values = [flat_data[i] for i in range(col_index, len(flat_data), nfields)]
                        if values:
                            entry[f"{metric.lower()}_min"] = min(values)
                            entry[f"{metric.lower()}_mean"] = sum(values) / len(values)
                            entry[f"{metric.lower()}_max"] = max(values)
                            entry[f"{metric.lower()}_std"] = statistics.stdev(values) if len(values) > 1 else 0
                            entry[f"{metric.lower()}_unit"] = unit
                        else:
                            logging.warning(f"No values for {metric} in uncertainty file, version {version}")
                    else:
                        logging.warning(f"Metric {metric} not found in uncertainty file for version {version}")
            except Exception as e:
                logging.error(f"Error processing uncertainty data for version {version}: {e}")
        self.update_summary_state()
        return self.summary_dict_list


    def add_pager_exposure(self) -> list:
        """Extract Pager exposure data and update summary."""
        for entry in self.summary_dict_list:
            version = entry.get("version")
            filename = self._get_pager_filename(version)
            xml_path = os.path.normpath(os.path.join(self.pager_folder, self.event_id, filename))
            logging.info(f"Processing Pager exposure for version {version} at {xml_path}")
            if not os.path.exists(xml_path):
                logging.warning(f"Pager file not found for version {version}: {xml_path}")
                continue
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                exposures = []
                for exp_elem in root.findall('exposure'):
                    exposures.append({
                        'dmin': exp_elem.get('dmin'),
                        'dmax': exp_elem.get('dmax'),
                        'exposure': exp_elem.get('exposure'),
                        'rangeInsideMap': exp_elem.get('rangeInsideMap')
                    })
                if len(exposures) >= 10:
                    entry['exposure_IV_V'] = exposures[4].get('exposure')
                    entry['exposure_V_VI'] = exposures[5].get('exposure')
                    entry['exposure_VI_VII'] = exposures[6].get('exposure')
                    entry['exposure_VII_VIII'] = exposures[7].get('exposure')
                    entry['exposure_VIII_IX'] = exposures[8].get('exposure')
                    entry['exposure_IX_X'] = exposures[9].get('exposure')
                else:
                    logging.warning(f"Exposure data too short for version {version}.")
            except Exception as e:
                logging.error(f"Error processing Pager exposure for version {version}: {e}")
        self.update_summary_state()
        return self.summary_dict_list

    def add_cities_impact(self, selected_cities: list) -> list:
        """Extract city impact data and update summary."""
        for entry in self.summary_dict_list:
            version = entry.get("version")
            filename = self._get_pager_filename(version)
            xml_path = os.path.normpath(os.path.join(self.pager_folder, self.event_id, filename))
            logging.info(f"Processing city impact for version {version} at {xml_path}")
            if not os.path.exists(xml_path):
                logging.warning(f"Pager file not found for version {version}: {xml_path}")
                continue
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                cities_impact = [ {attr: city.get(attr) for attr in city.keys()} for city in root.findall('city') ]
                city_lookup = {c['name'].strip().lower(): c for c in cities_impact if 'name' in c}
                for city_name in selected_cities:
                    key = city_name.strip().lower()
                    if key in city_lookup:
                        city_info = city_lookup[key]
                        try:
                            mmi_val = float(city_info.get('mmi', 'nan'))
                        except Exception:
                            mmi_val = np.nan
                        try:
                            pop_val = int(city_info.get('population', '0'))
                        except Exception:
                            pop_val = 0
                        entry[f'{city_name}_mmi'] = mmi_val
                        entry[f'{city_name}_pop'] = pop_val
                    else:
                        logging.warning(f"City '{city_name}' not found in version {version}")
            except Exception as e:
                logging.error(f"Failed to parse Pager XML for version {version}: {e}")
        self.update_summary_state()
        return self.summary_dict_list


    def add_alerts(self, version_list: list, alert_type: str = "fatality") -> list:
        """
        Extract Pager alert bins (economic or fatality) into the summary.
        
        After running, each summary entry has a key
          'alert_{alert_type}_bins' → [ {min, max, probability, color}, … ]
        """
        for entry in self.summary_dict_list:
            version = entry.get("version")
            fn = self._get_pager_filename(version)
            xml_path = Path(self.pager_folder) / self.event_id / fn
            logging.info(f"Loading Pager alerts '{alert_type}' for v{version} from {xml_path}")
            entry[f"alert_{alert_type}_bins"] = []
            if not xml_path.exists():
                logging.warning(f"Pager XML not found for v{version}")
                continue
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                alert_elem = root.find(f"alert[@type='{alert_type}']")
                if alert_elem is None:
                    logging.warning(f"No <alert type='{alert_type}'> in v{version}")
                    continue
                bins = []
                for b in alert_elem.findall("bin"):
                    bins.append({
                        "min": float(b.get("min")),
                        "max": float(b.get("max")),
                        "probability": float(b.get("probability")),
                        "color": b.get("color")
                    })
                entry[f"alert_{alert_type}_bins"] = bins
            except Exception as e:
                logging.error(f"Failed parsing alerts for v{version}: {e}")
        self.update_summary_state()
        return self.summary_dict_list


    def get_dataframe(self, summary_dict_list: list = None) -> pd.DataFrame:
        """Return summary as a pandas DataFrame."""
        if summary_dict_list is None:
            summary_dict_list = self.summary_dict_list
        self.summary_df = pd.DataFrame(summary_dict_list)
        return self.summary_df
    
    def get_summary_dict(self) -> dict:
        """Return summary as a dictionary."""
        self.update_summary_state()
        return self.summary_df.to_dict(orient="list")


    
    def export_summary(
        self,
        output_dir: str,
        file_type: str = 'csv',
        txt_sep: str = '\t'
    ):
        """
        Export self.summary_df to disk in any of the supported formats,
        naming it SHAKEtime-Summary-{event_id}.{file_type}.

        Parameters
        ----------
        output_dir : str
            Directory to write the file.
        file_type : str, default 'csv'
            One of: 'csv', 'txt', 'xlsx', 'json', 'feather',
            'parquet', 'pickle'
        txt_sep : str, default '\\t'
            Field delimiter when file_type == 'txt'.

        Raises
        ------
        ValueError
            If summary_df is not set or file_type is unsupported.
        """
        if not hasattr(self, 'summary_df') or self.summary_df is None:
            raise ValueError("summary_df is not available.")

        os.makedirs(output_dir, exist_ok=True)
        ft = file_type.lower()
        fname = f"SHAKEtime-Summary-{self.event_id}.{ft}"
        fpath = os.path.join(output_dir, fname)

        if ft == 'csv':
            self.summary_df.to_csv(fpath, index=False)
        elif ft == 'txt':
            self.summary_df.to_csv(fpath, index=False, sep=txt_sep)
        elif ft == 'xlsx':
            self.summary_df.to_excel(fpath, index=False)
        elif ft == 'json':
            self.summary_df.to_json(fpath, orient='records', lines=True)
        elif ft == 'feather':
            self.summary_df.to_feather(fpath)
        elif ft == 'parquet':
            self.summary_df.to_parquet(fpath, index=False)
        elif ft == 'pickle':
            self.summary_df.to_pickle(fpath)
        else:
            raise ValueError(f"Unsupported export file type: '{file_type}'")

        logging.info(f"Summary exported to {fpath!r} in format '{ft}'")


    def import_summary(self, file_path: str) -> pd.DataFrame:
        """
        Import a pre-computed summary from disk (any supported format),
        zero-pad its 'version' column to three digits, and cache it.

        Parameters
        ----------
        file_path : str
            Path to a summary file in one of:
            csv, txt, xlsx, json, feather, parquet, pickle.

        Returns
        -------
        pd.DataFrame
            The imported summary, with version zero-padded.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')

        if ext == 'csv':
            df = pd.read_csv(file_path)  # :contentReference[oaicite:0]{index=0}
        elif ext == 'txt':
            df = pd.read_csv(file_path, sep='\t')  # :contentReference[oaicite:1]{index=1}
        elif ext in ('xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt'):
            df = pd.read_excel(file_path)  # :contentReference[oaicite:2]{index=2}
        elif ext == 'json':
            df = pd.read_json(file_path, orient='records', lines=True)  # :contentReference[oaicite:3]{index=3}
        elif ext == 'feather':
            df = pd.read_feather(file_path)  # :contentReference[oaicite:4]{index=4}
        elif ext == 'parquet':
            df = pd.read_parquet(file_path)  # :contentReference[oaicite:5]{index=5}
        elif ext in ('pkl', 'pickle'):
            df = pd.read_pickle(file_path)  # :contentReference[oaicite:6]{index=6}
        else:
            raise ValueError(f"Unsupported import file type: '{ext}'")

        # Zero-pad the 'version' column to 3 digits
        if 'version' in df.columns:
            df['version'] = df['version'].astype(int).astype(str).str.zfill(3)

        # Cache for downstream use
        self.summary_df = df
        self.summary_dict_list = df.to_dict(orient='records')
        logging.info(f"Imported summary from {file_path!r} ({len(df)} rows)")
        return df




    def add_dyfi_lonlat(
        self,
        dyfi_df: pd.DataFrame,
        box_col: str = "Geocoded box"
    ) -> pd.DataFrame:
        """
        Given a DYFI DataFrame with a UTM‐encoded box string in column `box_col`
        (e.g. 'UTM:(43P 0785 1455 1000)'), parse out zone, easting, northing,
        convert to latitude/longitude, and return a new DataFrame with
        'latitude' and 'longitude' appended.

        Parameters
        ----------
        dyfi_df : pd.DataFrame
            Input DYFI table containing `box_col` with UTM cell specs.
        box_col : str, default "Geocoded box"
            Name of the column holding UTM box strings.

        Returns
        -------
        pd.DataFrame
            A copy of `dyfi_df` with two new columns:
            - 'latitude': decimal degrees
            - 'longitude': decimal degrees
        """
        def parse_utm(box_str: str):
            m = re.match(
                r'UTM:\(\s*(\d+)([C-HJ-NP-X])\s+(\d+)\s+(\d+)\s+(\d+)\s*\)',
                box_str
            )
            if not m:
                raise ValueError(f"Invalid UTM format: {box_str!r}")
            zone_number = int(m.group(1))
            zone_letter = m.group(2)
            e_idx = int(m.group(3))
            n_idx = int(m.group(4))
            resolution = int(m.group(5))
            # center of the grid cell:
            easting  = e_idx * resolution + resolution / 2.0
            northing = n_idx * resolution + resolution / 2.0
            return zone_number, zone_letter, easting, northing

        # 1) parse each box into numeric UTM components
        parsed = dyfi_df[box_col].apply(lambda s: pd.Series(
            parse_utm(s),
            index=["zone_number","zone_letter","easting","northing"]
        ))
        df = pd.concat([dyfi_df.reset_index(drop=True), parsed], axis=1)

        # 2) convert UTM → lat/lon
        def to_latlon(row):
            lat, lon = utm.to_latlon(
                row["easting"], row["northing"],
                int(row["zone_number"]), row["zone_letter"]
            )
            return pd.Series({"latitude": lat, "longitude": lon})

        latlon = df.apply(to_latlon, axis=1)
        df = pd.concat([df, latlon], axis=1)

        # 3) drop the intermediate UTM columns if you like
        df = df.drop(columns=["zone_number","zone_letter","easting","northing"])

        return df


    
    # --- END of SHAKEsummary methods  ---++


    
    ##################################################
    #
    #
    #
    #
    # --- Unified Grid Methods ---
    #
    #
    #
    ###################################################

    def get_unified_grid(
        self,
        version_list: List[str],
        metric: str = "mmi",
        grid_res: float = None,
        use_cache: bool = False,
        interp_method: str = "nearest",
        interp_kwargs: dict = None
    ) -> pd.DataFrame:
        """
        Build—or fetch—a unified lon–lat grid for the chosen intensity metric.

        If use_cache=True and we've already computed a unified grid, return it immediately.

        Parameters
        ----------
        version_list : list of str
            ShakeMap version identifiers; required if no cached grid exists.
        metric : str, default "mmi"
            Intensity measure to interpolate ('mmi', 'pga', etc.).
        grid_res : float, optional
            Desired grid spacing in degrees; if None, uses the smallest native spacing.
        use_cache : bool, default False
            If True and a grid has already been computed, return the cached grid.
        interp_method : str, default "linear"
            Interpolation method for scipy.interpolate.griddata: one of
            'linear', 'nearest', or 'cubic'.
        interp_kwargs : dict, optional
            Additional keyword args to forward to scipy.interpolate.griddata.

        Returns
        -------
        pd.DataFrame
            Unified grid with columns ['lon','lat', f"{metric}_v{version}", …].
        """
        import logging
        from scipy.interpolate import griddata

        # (1) Return cache if asked
        if use_cache and hasattr(self, "_unified_grid") and self._unified_grid is not None:
            logging.info("Using cached unified grid")
            return self._unified_grid

        if version_list is None:
            raise ValueError("No cached grid and no version_list provided; cannot compute unified grid.")

        logging.info(f"Computing unified grid for metric='{metric}', versions={version_list}, interp_method='{interp_method}'")

        ns = {'sm': 'http://earthquake.usgs.gov/eqcenter/shakemap'}
        shakemap_dfs = {}

        # (2) auto-detect resolution if not provided
        if grid_res is None:
            spacings = []
            for v in version_list:
                try:
                    fn = self._get_shakemap_filename(v)
                    path = os.path.join(self.shakemap_folder, self.event_id, fn)
                    root = ET.parse(path).getroot()
                    spec = root.find('sm:grid_specification', ns)
                    spacings.append(float(spec.attrib['nominal_lon_spacing']))
                except Exception:
                    continue
            grid_res = min(spacings) if spacings else 0.033333
            logging.info(f"  determined grid_res = {grid_res}")

        # (3) load each version’s raw grid
        for v in version_list:
            try:
                fn = self._get_shakemap_filename(v)
                path = os.path.join(self.shakemap_folder, self.event_id, fn)
                root = ET.parse(path).getroot()
                fields = root.findall('sm:grid_field', ns)
                names = [f.attrib['name'].lower() for f in fields]
                flat = [float(x) for x in root.find('sm:grid_data', ns).text.split()]
                arr = np.array(flat).reshape(-1, len(names))
                shakemap_dfs[v] = pd.DataFrame(arr, columns=names)
                logging.info(f"  loaded grid v{v} ({len(shakemap_dfs[v])} pts)")
            except Exception as e:
                logging.warning(f"  failed to load grid for v{v}: {e}")

        if not shakemap_dfs:
            logging.warning("No ShakeMap grids loaded → returning empty DataFrame")
            self._unified_grid = pd.DataFrame()
            return self._unified_grid

        # (4) compute overlap bbox
        lon_min = max(df.lon.min() for df in shakemap_dfs.values())
        lon_max = min(df.lon.max() for df in shakemap_dfs.values())
        lat_min = max(df.lat.min() for df in shakemap_dfs.values())
        lat_max = min(df.lat.max() for df in shakemap_dfs.values())

        lon_vals = np.arange(lon_min, lon_max + grid_res, grid_res)
        lat_vals = np.arange(lat_min, lat_max + grid_res, grid_res)
        lon_g, lat_g = np.meshgrid(lon_vals, lat_vals)
        unified = pd.DataFrame({'lon': lon_g.ravel(), 'lat': lat_g.ravel()})
        xi = unified[['lon', 'lat']].values
        logging.info(f"  built target grid with {len(unified)} points")

        # default empty interp_kwargs
        interp_kwargs = interp_kwargs or {}

        # (5) interpolate each version onto the unified grid
        for v, df in shakemap_dfs.items():
            col = f"{metric}_v{v}"
            if metric not in df.columns:
                logging.error(f"Metric '{metric}' not in v{v}; filling with NaN")
                unified[col] = np.nan
                continue

            pts = df[['lon', 'lat']].values
            vals = df[metric].values
            try:
                unified[col] = griddata(pts, vals, xi,
                                        method=interp_method,
                                        **interp_kwargs)
                logging.info(f"  interpolated '{metric}' for v{v} ({interp_method})")
            except Exception as e:
                logging.error(f"  interpolation failed for v{v}: {e}")
                unified[col] = np.nan

        # (6) cache & return
        self._unified_grid = unified
        logging.info("Unified grid computation complete and cached")
        return unified


    def add_rate_of_change(
        self,
        unified_grid: pd.DataFrame = None,
        version_list: list = None,
        metric: str = "mmi",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Compute delta (v2−v1) and relative rate columns for each consecutive
        version pair, plus one extra pair: final version minus first version.

        If unified_grid is None, fetch/build it via get_unified_grid(..., use_cache=use_cache).

        Returns the augmented DataFrame.
        """
        if unified_grid is None:
            unified_grid = self.get_unified_grid(version_list, metric, use_cache=use_cache)

        logging.info(f"Computing delta/rate for metric='{metric}'")

        # 1) consecutive pairs
        for i in range(len(version_list) - 1):
            v1, v2 = version_list[i], version_list[i + 1]
            c1, c2 = f"{metric}_v{v1}", f"{metric}_v{v2}"
            dcol, rcol = f"delta_{v2}_{v1}_{metric}", f"rate_{v2}_{v1}_{metric}"

            if c1 not in unified_grid.columns or c2 not in unified_grid.columns:
                logging.warning(f"  missing '{c1}' or '{c2}'; filling '{dcol}','{rcol}' with NaN")
                unified_grid[dcol] = np.nan
                unified_grid[rcol] = np.nan
            else:
                unified_grid[dcol] = unified_grid[c2] - unified_grid[c1]
                unified_grid[rcol] = unified_grid[dcol] / (unified_grid[c1] + 1e-6)
                logging.debug(f"  computed {dcol}, {rcol}")

        # 2) final vs first
        if len(version_list) >= 2:
            v_first, v_last = version_list[0], version_list[-1]
            c_first, c_last = f"{metric}_v{v_first}", f"{metric}_v{v_last}"
            dfcol, frcol = f"delta_{v_last}_{v_first}_{metric}", f"rate_{v_last}_{v_first}_{metric}"

            if c_first not in unified_grid.columns or c_last not in unified_grid.columns:
                logging.warning(f"  missing '{c_first}' or '{c_last}'; filling '{dfcol}','{frcol}' with NaN")
                unified_grid[dfcol] = np.nan
                unified_grid[frcol] = np.nan
            else:
                unified_grid[dfcol] = unified_grid[c_last] - unified_grid[c_first]
                unified_grid[frcol] = unified_grid[dfcol] / (unified_grid[c_first] + 1e-6)
                logging.info(f"  computed final-first delta/rate: {dfcol}, {frcol}")

        return unified_grid


    

    def get_rate_grid(
        self,
        version_list: list,
        metric: str = "mmi",
        grid_res: float = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Return a unified grid augmented with delta and rate columns.
    
        Parameters
        ----------
        version_list : list of str
            ShakeMap version identifiers.
        metric : str, default "mmi"
            Intensity measure to interpolate.
        grid_res : float, optional
            Desired grid spacing.
        use_cache : bool, default True
            If True and a rate grid is already cached, return it immediately
            without recomputing either unified or rate grids.
    
        Returns
        -------
        pd.DataFrame
            The rate‐augmented grid, cached in self._rate_grid.
        """
        # 1) If allowed, return existing rate grid
        if use_cache and hasattr(self, "_rate_grid") and self._rate_grid is not None:
            logging.info("Using cached rate grid")
            return self._rate_grid
    
        # 2) Build or fetch the unified grid (respects its own use_cache flag)
        ug = self.get_unified_grid(
            version_list,
            metric,
            grid_res,
            use_cache=use_cache
        )
    
        # 3) Compute deltas and rates
        ug = self.add_rate_of_change(
            unified_grid=ug,
            version_list=version_list,
            metric=metric,
            use_cache=use_cache
        )
    
        # 4) Cache and return
        self._rate_grid = ug
        logging.info("Rate grid computed and cached in self._rate_grid")
        return ug


    def add_rate_to_summary(
        self,
        unified_grid: pd.DataFrame = None,
        version_list: list = None,
        metric: str = "mmi",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate per-cell delta & rate statistics into the summary DataFrame,
        including both each consecutive version pair and the overall change
        from the first to the last version.

        Uses cached rate grid by default; pass use_cache=False to recompute.
        """
        import numpy as np

        # 1) Get or build the rate grid
        if unified_grid is None:
            unified_grid = self.get_rate_grid(version_list, metric, use_cache=use_cache)

        if version_list is None:
            raise ValueError("version_list must be provided to aggregate rate stats.")

        # 2) Ensure we have a base summary
        summary_df = self.get_dataframe()
        if summary_df.empty:
            logging.info("Summary empty → regenerating base summary")
            self.get_shake_summary(version_list)
            summary_df = self.get_dataframe()

        # 3) Ensure aggregate columns exist (consecutive + final-first “full”)
        agg_cols = []
        # consecutive‐pair stats:
        agg_cols += [f"{metric}_delta_{s}" for s in ("min", "mean", "max", "std")]
        agg_cols += [f"{metric}_rate_{s}"  for s in ("min", "mean", "max", "std")]
        # final-first stats:
        agg_cols += [f"{metric}_delta_full_{s}" for s in ("min", "mean", "max", "std")]
        agg_cols += [f"{metric}_rate_full_{s}"  for s in ("min", "mean", "max", "std")]

        for col in agg_cols:
            if col not in summary_df.columns:
                summary_df[col] = np.nan

        # 4) Fill in stats for each version after the first (consecutive pairs)
        for i in range(1, len(version_list)):
            v1, v2 = version_list[i-1], version_list[i]
            dcol = f"delta_{v2}_{v1}_{metric}"
            rcol = f"rate_{v2}_{v1}_{metric}"
            mask = summary_df['version'] == v2

            if dcol in unified_grid.columns and rcol in unified_grid.columns:
                summary_df.loc[mask, f"{metric}_delta_min"]   = round(unified_grid[dcol].min(),  2)
                summary_df.loc[mask, f"{metric}_delta_mean"]  = round(unified_grid[dcol].mean(), 2)
                summary_df.loc[mask, f"{metric}_delta_max"]   = round(unified_grid[dcol].max(),  2)
                summary_df.loc[mask, f"{metric}_delta_std"]   = round(unified_grid[dcol].std(),  2)
                summary_df.loc[mask, f"{metric}_rate_min"]    = round(unified_grid[rcol].min(),  2)
                summary_df.loc[mask, f"{metric}_rate_mean"]   = round(unified_grid[rcol].mean(), 2)
                summary_df.loc[mask, f"{metric}_rate_max"]    = round(unified_grid[rcol].max(),  2)
                summary_df.loc[mask, f"{metric}_rate_std"]    = round(unified_grid[rcol].std(),  2)
                logging.debug(f"  aggregated stats for version {v2}")
            else:
                logging.warning(f"  missing grid cols {dcol} or {rcol} for v{v2}")

        # 5) Aggregate final-first for only the last version row
        if len(version_list) >= 2:
            v_first, v_last = version_list[0], version_list[-1]
            dfcol = f"delta_{v_last}_{v_first}_{metric}"
            frcol = f"rate_{v_last}_{v_first}_{metric}"
            mask_last = summary_df['version'] == v_last

            if dfcol in unified_grid.columns and frcol in unified_grid.columns:
                # delta_full_*
                summary_df.loc[mask_last, f"{metric}_delta_full_min"]   = round(unified_grid[dfcol].min(),  2)
                summary_df.loc[mask_last, f"{metric}_delta_full_mean"]  = round(unified_grid[dfcol].mean(), 2)
                summary_df.loc[mask_last, f"{metric}_delta_full_max"]   = round(unified_grid[dfcol].max(),  2)
                summary_df.loc[mask_last, f"{metric}_delta_full_std"]   = round(unified_grid[dfcol].std(),  2)
                # rate_full_*
                summary_df.loc[mask_last, f"{metric}_rate_full_min"]    = round(unified_grid[frcol].min(),  2)
                summary_df.loc[mask_last, f"{metric}_rate_full_mean"]   = round(unified_grid[frcol].mean(), 2)
                summary_df.loc[mask_last, f"{metric}_rate_full_max"]    = round(unified_grid[frcol].max(),  2)
                summary_df.loc[mask_last, f"{metric}_rate_full_std"]    = round(unified_grid[frcol].std(),  2)
                logging.info(f"  aggregated final-first stats for version {v_last}")
            else:
                logging.warning(f"  missing final-first grid cols {dfcol} or {frcol} for v{v_last}")

        # 6) Persist and return
        self.summary_df = summary_df
        self.summary_dict_list = summary_df.to_dict(orient="records")
        logging.info("Summary updated with delta/rate statistics (including final-first)")
        return summary_df


    def get_rate_grid_dataframe(
        self,
        version_list: list,
        metric: str = "mmi",
        grid_res: float = None
    ) -> pd.DataFrame:
        """
        Return the rate-of-change grid as a DataFrame, building it if needed.
        """
        if not hasattr(self, "_rate_grid") or self._rate_grid is None:
            logging.info("Rate grid not cached → computing now")
            self._rate_grid = self.get_rate_grid(version_list, metric, grid_res)
        return self._rate_grid


    def clear_grid_cache(self):
        """
        Drop any stored unified or rate grids so they’ll be recomputed next time.
        """
        self._unified_grid = None
        self._rate_grid = None
        logging.info("Cleared cached unified and rate grids")

    def clear_summary_cache(self):
        """
        Drop the ShakeMap summary cache so it must be rebuilt on next access.
        """
        self.summary_dict_list = []
        self.summary_df = pd.DataFrame()
        logging.info("Cleared ShakeMap summary cache")



    def import_unified_grid(self, file_path: str) -> pd.DataFrame:
        """
        Load a pre-computed unified grid from disk in any supported format
        (csv, txt, xlsx, json, feather, parquet, pickle), caching it in
        self._unified_grid for downstream calls.

        Parameters
        ----------
        file_path : str
            Path to a file containing the unified grid.

        Returns
        -------
        pd.DataFrame
            The imported unified grid.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        # 1) Detect extension
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')

        # 2) Dispatch to the appropriate pandas reader
        if ext == 'csv':
            df = pd.read_csv(file_path)  # :contentReference[oaicite:0]{index=0}
        elif ext == 'txt':
            # assume tab-delimited by default
            df = pd.read_csv(file_path, sep='\t')  # :contentReference[oaicite:1]{index=1}
        elif ext in ('xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt'):
            df = pd.read_excel(file_path)  # :contentReference[oaicite:2]{index=2}
        elif ext == 'json':
            # one JSON record per line, matching our export
            df = pd.read_json(file_path, orient='records', lines=True)  # :contentReference[oaicite:3]{index=3}
        elif ext == 'feather':
            df = pd.read_feather(file_path)  # :contentReference[oaicite:4]{index=4}
        elif ext == 'parquet':
            df = pd.read_parquet(file_path)  # :contentReference[oaicite:5]{index=5}
        elif ext in ('pkl', 'pickle'):
            df = pd.read_pickle(file_path)  # :contentReference[oaicite:6]{index=6}
        else:
            raise ValueError(f"Unsupported import file type: '{ext}'")

        # 3) Cache & log
        self._unified_grid = df
        logging.info(f"Imported unified grid from {file_path!r}: {len(df)} points cached")
        return df


    def export_unified_grid(
        self,
        output_dir: str,
        file_type: str = 'csv',
        txt_sep: str = ';',
    ):
        """
        Export the cached unified grid to a file, naming it
        SHAKEtime_unified_grid_{event_id}_{imt}.{file_type}.

        Parameters
        ----------
        output_dir : str
            Directory to write the file.
        file_type : str, default 'csv'
            One of: 'csv', 'txt', 'xlsx', 'json', 'feather', 'parquet', 'pickle'
        txt_sep : str, default '\\t'
            Field delimiter to use when file_type == 'txt'.

        Raises
        ------
        RuntimeError
            If no unified grid is cached.
        ValueError
            If file_type is not supported.
        """
        if not hasattr(self, "_unified_grid") or self._unified_grid is None:
            raise RuntimeError("No unified grid to export; compute or import one first.")

        # Detect IMT as before...
        imt = None
        for metric in ('pga', 'mmi', 'pgv', 'psa10'):
            if any(col.startswith(f"{metric}_v") for col in self._unified_grid.columns):
                imt = metric
                break
        if imt is None:
            logging.warning("Could not detect IMT; defaulting to 'unknown'")
            imt = 'unknown'

        os.makedirs(output_dir, exist_ok=True)
        ft = file_type.lower()
        fname = f"SHAKEtime_unified_grid_{self.event_id}_{imt}.{ft}"
        fpath = os.path.join(output_dir, fname)

        if ft == 'csv':
            self._unified_grid.to_csv(fpath, index=False)
        elif ft == 'txt':
            # use the user‐supplied separator here
            self._unified_grid.to_csv(fpath, index=False, sep=txt_sep)
        elif ft == 'xlsx':
            self._unified_grid.to_excel(fpath, index=False)
        elif ft == 'json':
            self._unified_grid.to_json(fpath, orient='records', lines=True)
        elif ft == 'feather':
            self._unified_grid.to_feather(fpath)
        elif ft == 'parquet':
            self._unified_grid.to_parquet(fpath, index=False)
        elif ft == 'pickle':
            self._unified_grid.to_pickle(fpath)
        else:
            raise ValueError(f"Unsupported export file type: {file_type}")

        logging.info(f"Unified grid exported to {fpath!r} in format '{ft}'")



    def clip_grid(self, min_lon, max_lon, min_lat, max_lat):
        """Subset the cached unified grid to a smaller region."""
        if not hasattr(self, "_unified_grid") or self._unified_grid is None:
            raise RuntimeError("No unified grid in cache to clip.")
        ug = self._unified_grid
        mask = (ug.lon>=min_lon)&(ug.lon<=max_lon)&(ug.lat>=min_lat)&(ug.lat<=max_lat)
        clipped = ug[mask].reset_index(drop=True)
        self._unified_grid = clipped
        logging.info(f"Clipped unified grid to {len(clipped)} points within [{min_lon},{max_lon},{min_lat},{max_lat}]")
        return clipped
    
    def grid_stats(self, column: str) -> dict:
        """Return min, max, mean, std for a given unified‐grid column."""
        if not hasattr(self, "_unified_grid") or column not in self._unified_grid:
            raise KeyError(f"{column!r} not found in unified grid cache.")
        s = self._unified_grid[column].dropna()
        return {"min": s.min(), "mean": s.mean(), "max": s.max(), "std": s.std()}
    


    ##################################################
    #
    #
    #
    #
    # --- SHAKEmaps plotting methods  ---++
    #
    #
    #
    ##################################################
    
    def available_metrics(self) -> List[str]:
        """
        Returns all metric prefixes for which you have *_mean columns in your summary.
        e.g. ['mmi','pga','psa10','mmi_delta','pga_rate',…]
        """
        cols = [c for c in self.summary_df.columns if c.endswith("_mean")]
        return sorted({c.rsplit("_",1)[0] for c in cols})


    
    def plot_thd(
        self,
        metric_type: str = "mmi",
        show_title: bool = True,
        x_ticks: str = "version",
        output_path: str = None,
        save_formats: list = ["png", "pdf"],
        dpi: int = 300,
    
        # -------- NEW: styling kwargs --------
        figsize: tuple = (24, 12),
        marker: str = "o",
        marker_size: int = 20,
        line_width: float = 2.5,
        alpha_fill: float = 0.2,
        errorbar_capsize: int = 10,
        errorbar_color: str = "k",
    
        font_sizes: dict = None,
        xlabel: str = None,
        ylabel: str = None,
    
        legend: bool = True,
        legend_loc: str = "best",
        legend_fontsize: int = None,
    
        grid: bool = True,
        grid_kwargs: dict = None,
    
        x_rotation: int = None,
        x_ha: str = None,
    
        tight_layout: bool = True,
        show: bool = False,
        close: bool = False
    ):
        """
        Temporal Hazard Display (THD) for any metric in the summary.
        """
    
        import numpy as np
        import matplotlib.pyplot as plt
        import logging
        from pathlib import Path
    
        # --------------------
        # Defaults
        # --------------------
        if font_sizes is None:
            font_sizes = {"labels": 14, "ticks": 12, "legend": 12, "title": 18}
    
        lbl_fs = font_sizes.get("labels", 14)
        tck_fs = font_sizes.get("ticks", 12)
        lgd_fs = legend_fontsize or font_sizes.get("legend", 12)
        ttl_fs = font_sizes.get("title", 18)
    
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.5}
    
        # --------------------
        # Data preparation (UNCHANGED)
        # --------------------
        df = self.get_dataframe()
        df_plot = df.copy()
    
        if "version" not in df_plot.columns:
            raise ValueError("Column 'version' not found in summary")
    
        df_plot["version_int"] = df_plot["version"].astype(int)
        df_plot = df_plot.sort_values("version_int").reset_index(drop=True)
    
        x_pos = np.arange(len(df_plot))
    
        if x_ticks == "version":
            tick_labels = df_plot["version"].astype(str)
            xlabel_eff = "Version"
        else:
            if x_ticks not in df_plot.columns:
                raise ValueError(f"x_ticks='{x_ticks}' not in summary columns")
            vals = df_plot[x_ticks]
            if np.issubdtype(vals.dtype, np.number):
                tick_labels = [f"{v:.1f}" for v in vals.astype(float)]
            else:
                tick_labels = vals.astype(str)
            xlabel_map = {
                "TaE_h": "Time After Event (hours)",
                "TaE_d": "Time After Event (days)",
                "shakemap_version": "ShakeMap Version"
            }
            xlabel_eff = xlabel_map.get(x_ticks, x_ticks)
    
        # --------------------
        # Metric parsing
        # --------------------
        if "_" in metric_type:
            base, kind = metric_type.split("_", 1)
        elif metric_type.startswith("std"):
            base, kind = metric_type[3:], "std"
        else:
            base, kind = metric_type, "mean"
    
        cols = {
            "min":  f"{metric_type}_min",
            "mean": f"{metric_type}_mean",
            "max":  f"{metric_type}_max",
            "std":  f"{metric_type}_std"
        }
    
        missing = [c for c in cols.values() if c not in df_plot.columns]
        if missing:
            raise ValueError(f"Missing columns for '{metric_type}': {missing}")
    
        mean_s = df_plot[cols["mean"]]
        min_s  = df_plot[cols["min"]]
        max_s  = df_plot[cols["max"]]
        std_s  = df_plot[cols["std"]]
    
        # --------------------
        # Colors (UNCHANGED)
        # --------------------
        base_colors  = {"mmi":"#4C72B0","pga":"#DD8452","pgv":"#55A868","psa10":"#8172B2"}
        delta_colors = {"mmi":"#2E3C72","pga":"#B23F24","pgv":"#3C8E50","psa10":"#5F4F88"}
        std_colors   = {"mmi":"#A1BEDC","pga":"#F3BCA3","pgv":"#A8D5B0","psa10":"#B9AEE3"}
    
        if kind == "mean":
            line_color = base_colors.get(base, "C0")
        elif kind in ("delta", "rate"):
            line_color = delta_colors.get(base, "C0")
        else:
            line_color = std_colors.get(base, "C0")
    
        # --------------------
        # Plot
        # --------------------
        fig, ax = plt.subplots(figsize=figsize)
    
        ax.plot(
            x_pos, mean_s,
            label="Mean",
            color=line_color,
            marker=marker,
            markersize=marker_size,
            linewidth=line_width
        )
    
        ax.fill_between(
            x_pos, min_s, max_s,
            color=line_color,
            alpha=alpha_fill,
            label="Min–Max"
        )
    
        ax.errorbar(
            x_pos, mean_s,
            yerr=std_s,
            fmt="none",
            ecolor=errorbar_color,
            capsize=errorbar_capsize,
            label="Std Dev"
        )
    
        ax.set_xlabel(xlabel or xlabel_eff, fontsize=lbl_fs)
        ax.set_ylabel(
            ylabel or (base.upper() if kind == "mean" else f"{base} ({kind})"),
            fontsize=lbl_fs
        )
    
        if grid:
            ax.grid(True, **grid_kwargs)
    
        if legend:
            ax.legend(fontsize=lgd_fs, loc=legend_loc)
    
        # --------------------
        # Ticks
        # --------------------
        rot = x_rotation if x_rotation is not None else (45 if x_ticks != "version" else 0)
        ha  = x_ha if x_ha is not None else ("right" if x_ticks != "version" else "center")
    
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tick_labels, rotation=rot, ha=ha, fontsize=tck_fs)
        ax.tick_params(axis="y", labelsize=tck_fs)
    
        if show_title:
            ax.set_title(
                f"{base.upper()} over {xlabel_eff} ({metric_type})",
                fontsize=ttl_fs
            )
    
        if tight_layout:
            plt.tight_layout()
    
        # --------------------
        # Save
        # --------------------
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "temporal_hazard"
            out_dir.mkdir(parents=True, exist_ok=True)
            for fmt in save_formats:
                p = out_dir / f"{self.event_id}_THD_{metric_type}.{fmt}"
                fig.savefig(p, bbox_inches="tight", dpi=dpi)
                logging.info(f"Saved THD plot to {p}")
    
        if show:
            plt.show()
        if close:
            plt.close(fig)
    
        return fig, ax
    

   
    def plot_pager_data(
        self,
        version_list: list,
        output_path: str = None,
        figsize: tuple = (8, 6),
        save_formats: list = ["png", "pdf"],
        dpi: int = 300,
        font_sizes: dict = None,
        xlim: tuple = None,
        ylim: tuple = None,
        legend_kwargs: dict = None,
        grid_kwargs: dict = None,
        annotate: bool = False,
        annotate_fmt: str = "{:.2f}"
    ) -> list:
        """
        Plots Pager probability distributions for each version in version_list.

        Parameters
        ----------
        version_list : list
            ShakeMap version identifiers.
        output_path : str, optional
            Base directory under which to save the plots.
        figsize : tuple, default (8,6)
            Figure size to apply to each plot.
        save_formats : list of str, default ["png","pdf"]
            File formats for saving figures.
        dpi : int, default 300
            Resolution in dots per inch for saved figures.
        font_sizes : dict, optional
            Font sizes: {"title":..., "labels":..., "ticks":...}.
        xlim : tuple, optional
            X-axis limits as (min, max).
        ylim : tuple, optional
            Y-axis limits as (min, max).
        legend_kwargs : dict, optional
            Keyword args for ax.legend(...).
        grid_kwargs : dict, optional
            Keyword args for ax.grid(...).
        annotate : bool, default False
            If True, annotate each bar with its value.
        annotate_fmt : str, default "{:.2f}"
            Format string for annotations.
        
        Returns
        -------
        List[tuple]
            A list of (fig, ax, prob_type, version) for each generated plot.
        """
        figures = []

        # default font sizes
        if font_sizes is None:
            font_sizes = {}
        title_fs = font_sizes.get("title", 14)
        label_fs = font_sizes.get("labels", 12)
        tick_fs  = font_sizes.get("ticks", 10)

        # default legend/grid kwargs
        if legend_kwargs is None:
            legend_kwargs = {}
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.5}

        for version in version_list:
            fn = self._get_pager_filename(version)
            xml_path = os.path.normpath(os.path.join(self.pager_folder, self.event_id, fn))
            logging.info(f"Processing Pager data for version {version} from {xml_path}")
            if not os.path.exists(xml_path):
                logging.warning(f"Pager XML not found for version {version}: {xml_path}")
                continue

            try:
                pager_parser = USGSParser(parser_type='pager_xml', mode='parse', xml_file=xml_path)

                for prob_type in ['economic', 'fatality']:
                    fig, ax = pager_parser.plot_probability_distribution(prob_type)

                    # apply figure size
                    fig.set_size_inches(*figsize)

                    # font sizes
                    ax.title.set_fontsize(title_fs)
                    ax.xaxis.label.set_size(label_fs)
                    ax.yaxis.label.set_size(label_fs)
                    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                        lbl.set_fontsize(tick_fs)

                    # set limits if provided
                    if xlim:
                        ax.set_xlim(*xlim)
                    if ylim:
                        ax.set_ylim(*ylim)

                    # grid and legend
                    ax.grid(True, **grid_kwargs)
                    ax.legend(**legend_kwargs)

                    # annotate bars
                    if annotate:
                        for bar in ax.patches:
                            height = bar.get_height()
                            ax.annotate(
                                annotate_fmt.format(height),
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=tick_fs
                            )

                    figures.append((fig, ax, prob_type, version))

                    # save if requested
                    if output_path:
                        out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "/pager_data/PagerPlots"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        for fmt in save_formats:
                            save_path = out_dir / f"{self.event_id}_Pager{prob_type.capitalize()}_{version}.{fmt}"
                            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
                            logging.info(f"Saved Pager plot to {save_path}")

            except Exception as e:
                logging.error(f"Error processing Pager XML for version {version}: {e}")

        return figures


    
    def _get_stations_filename(self, version: str) -> str:
        """
        Return station‐list filename with padded version.
        """
        v = self._pad_version(version)
        if self.file_type == 1:
            return f"{self.event_id}_stationlist_{v}.json"
        else:
            return f"{self.event_id}_us_{v}_stationlist.json"



    def plot_shakemaps(
        self,
        version_list: list,
        metric: str = "mmi",
        rupture_folder: str = None,
        stations_folder: str = None,
        add_cities: bool = False,
        cities_population: int = 1000000,
        output_path: str = None,
        plot_colorbar: bool = True,
        show_title: bool = True,
        save_formats: list = ["png", "pdf"],
        dpi: int = 300,
        mode: str = "shakemap",
        use_cache: bool = True,
        grid_res: float = None,
        interp_method: str = "nearest",
        interp_kwargs: dict = None,
    ) -> list:
        """
        Plots ShakeMap maps for the specified versions, optionally overlaying
        rupture traces, seismic stations (PGA), DYFI reports (MMI), and cities.
    
        Two rendering modes:
          - mode="shakemap"     : parse each ShakeMap XML and render as usual (default)
          - mode="unified_grid" : render each version from the cached/computed unified grid
                                  (calls get_unified_grid(..., use_cache=use_cache) with fallback)
    
        Notes
        -----
        - In mode="unified_grid", ShakeMap *values* come from the unified grid, but
          metadata is read from the version-specific ShakeMap XML so SHAKEmapper can
          use parser.metadata as expected.
        - Stations/DYFI are plotted for EACH version independently (no subtraction),
          but numeric fields are cleaned to avoid NULL/null/empty-string issues.
        """
        from pathlib import Path
        import logging
        import numpy as np
        import pandas as pd
    
        # -----------------------------
        # helper: robust numeric cleaning
        # -----------------------------
        _NULL_TOKENS = {
            "", " ", "nan", "NaN", "NAN",
            "null", "NULL", "Null",
            "none", "None", "NONE",
            "na", "NA", "N/A", "n/a",
        }
    
        def _clean_null_tokens(df: pd.DataFrame, cols: list) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].apply(
                        lambda x: np.nan
                        if (isinstance(x, str) and x.strip() in _NULL_TOKENS)
                        else x
                    )
            return df
    
        def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            df = _clean_null_tokens(df, cols)
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
    
        figures = []
    
        # ---------------------------------------------------------
        # validate mode
        # ---------------------------------------------------------
        if mode not in ("shakemap", "unified_grid"):
            raise ValueError("mode must be either 'shakemap' or 'unified_grid'")
    
        # ---------------------------------------------------------
        # unified grid setup (only if requested)
        # ---------------------------------------------------------
        ug = None
        lon_unique = lat_unique = None
        nlon = nlat = None
        dlon = dlat = None
    
        if mode == "unified_grid":
            # try cache first (if requested)
            try:
                ug = self.get_unified_grid(
                    version_list=version_list,
                    metric=metric,
                    grid_res=grid_res,
                    use_cache=use_cache,
                    interp_method=interp_method,
                    interp_kwargs=interp_kwargs,
                )
            except Exception as e:
                logging.warning(f"[plot_shakemaps] get_unified_grid(use_cache={use_cache}) failed: {e}")
                ug = None
    
            # fallback: compute if empty/missing
            if ug is None or getattr(ug, "empty", True):
                try:
                    ug = self.get_unified_grid(
                        version_list=version_list,
                        metric=metric,
                        grid_res=grid_res,
                        use_cache=False,
                        interp_method=interp_method,
                        interp_kwargs=interp_kwargs,
                    )
                except Exception as e:
                    logging.error(f"[plot_shakemaps] get_unified_grid(fallback compute) failed: {e}")
                    ug = None
    
            if ug is None or getattr(ug, "empty", True):
                logging.error("[plot_shakemaps] unified grid is empty; cannot plot in mode='unified_grid'")
                return figures
    
            # derive grid geometry (for metadata grid_specification)
            lon_unique = np.sort(pd.unique(ug["lon"]))
            lat_unique = np.sort(pd.unique(ug["lat"]))
            nlon = len(lon_unique)
            nlat = len(lat_unique)
            dlon = float(np.min(np.diff(lon_unique))) if nlon > 1 else 0.0
            dlat = float(np.min(np.diff(lat_unique))) if nlat > 1 else 0.0
    
            # Helper to pull metadata from XML for this version (least-risk)
            def _get_xml_metadata_for_version(v: str) -> dict:
                sm_fn = self._get_shakemap_filename(v)
                sm_path = Path(self.shakemap_folder) / self.event_id / sm_fn
                try:
                    meta_parser = USGSParser(
                        parser_type="shakemap_xml",
                        xml_file=str(sm_path),
                        imt=metric,
                        value_type="mean",
                    )
                    # USGSParser likely has .metadata OR .get_metadata(); handle both
                    if hasattr(meta_parser, "metadata"):
                        return meta_parser.metadata
                    if hasattr(meta_parser, "get_metadata"):
                        return meta_parser.get_metadata()
                except Exception as e:
                    logging.warning(f"[plot_shakemaps] metadata parse failed for v{v}: {e}")
                return {}
    
            # Build unified "parser" that matches SHAKEmapper expectations
            def _make_unified_parser(v: str):
                col = f"{metric}_v{v}"
                if col not in ug.columns:
                    raise KeyError(f"Unified grid column '{col}' not found in unified grid.")
    
                # Sort by LAT desc then LON asc so values reshape correctly for SHAKEmapper
                df = ug[["lon", "lat", col]].copy()
                df = df.sort_values(["lat", "lon"], ascending=[False, True]).reset_index(drop=True)
    
                xml_meta = _get_xml_metadata_for_version(v) or {}
    
                # Ensure grid_specification exists (some code paths rely on it)
                # We keep XML metadata if present, but we can safely supplement it.
                grid_spec = xml_meta.get("grid_specification", {}) if isinstance(xml_meta, dict) else {}
                if isinstance(grid_spec, dict):
                    grid_spec.setdefault("nlon", int(nlon))
                    grid_spec.setdefault("nlat", int(nlat))
                    grid_spec.setdefault("nominal_lon_spacing", float(dlon) if dlon else 0.0)
                    grid_spec.setdefault("nominal_lat_spacing", float(dlat) if dlat else 0.0)
                    if isinstance(xml_meta, dict):
                        xml_meta["grid_specification"] = grid_spec
    
                class _UnifiedGridParser:
                    def __init__(self, df_in: pd.DataFrame, metric_name: str, meta: dict, colname: str):
                        self._df = df_in
                        self._metric = metric_name
                        self._colname = colname
                        # SHAKEmapper expects parser.metadata (attribute)
                        self.metadata = meta
    
                    def get_metadata(self):
                        return self.metadata
    
                    def get_dataframe(self):
                        return pd.DataFrame({
                            "LON": self._df["lon"].values,
                            "LAT": self._df["lat"].values,
                            self._metric: self._df[self._colname].values,
                        })
    
                return _UnifiedGridParser(df, metric, xml_meta, col)
    
        # ---------------------------------------------------------
        # main loop
        # ---------------------------------------------------------
        for version in version_list:
            parser = None
    
            if mode == "shakemap":
                sm_fn = self._get_shakemap_filename(version)
                sm_path = Path(self.shakemap_folder) / self.event_id / sm_fn
                logging.info(f"Plotting ShakeMap v{version} ({metric}) from {sm_path}")
                try:
                    parser = USGSParser(
                        parser_type="shakemap_xml",
                        xml_file=str(sm_path),
                        imt=metric,
                        value_type="mean"
                    )
                except Exception as e:
                    logging.error(f"  ✖ parse failed for v{version}: {e}")
                    continue
    
            else:
                # unified grid
                logging.info(f"Plotting ShakeMap v{version} ({metric}) from unified grid")
                try:
                    parser = _make_unified_parser(version)
                except Exception as e:
                    logging.error(f"  ✖ unified-grid parser failed for v{version}: {e}")
                    continue
    
            try:
                # 1) base map + ShakeMap
                mapper = SHAKEmapper()
                mapper.create_basemap(label_size=22)
                mapper.add_usgs_shakemap(parser, plot_colorbar=plot_colorbar)
    
                # 2) rupture overlay
                if rupture_folder:
                    ru_fn = self._get_rupture_filename(version)
                    ru_path = Path(rupture_folder) / self.event_id / ru_fn
                    logging.info(f"  adding rupture from {ru_path}")
                    try:
                        rup = USGSParser(parser_type="rupture_json", rupture_json=str(ru_path))
                        xs, ys = rup.get_rupture_xy()
                        mapper.add_rupture(xs, ys)
                    except Exception as e:
                        logging.warning(f"  ⚠ rupture failed for v{version}: {e}")
    
                # 3) stations + DYFI overlay (plot ALL for that version, just clean bad values)
                if stations_folder:
                    st_fn = self._get_stations_filename(version)
                    st_path = Path(stations_folder) / self.event_id / st_fn
                    logging.info(f"  adding stations/DYFI from {st_path}")
                    try:
                        inst_parser = USGSParser(parser_type="instrumented_data", json_file=str(st_path))
    
                        # seismic instruments (PGA)
                        inst_df = inst_parser.get_dataframe(value_type="pga")
                        if inst_df is not None and not inst_df.empty:
                            inst_df = _coerce_numeric(inst_df, ["longitude", "latitude"])
                            inst_df = inst_df.dropna(subset=["longitude", "latitude"])
                            if not inst_df.empty:
                                mapper.add_stations(
                                    inst_df["longitude"].values,
                                    inst_df["latitude"].values
                                )
    
                        # DYFI reports (MMI)
                        dyfi_df = inst_parser.get_dataframe(value_type="mmi")
                        if dyfi_df is not None and not dyfi_df.empty:
                            dyfi_df = _coerce_numeric(dyfi_df, ["longitude", "latitude", "intensity", "nresp"])
                            dyfi_df = dyfi_df.dropna(subset=["longitude", "latitude", "intensity"])
                            if not dyfi_df.empty:
                                mapper.add_dyfi(
                                    dyfi_df["longitude"].values,
                                    dyfi_df["latitude"].values,
                                    dyfi_df["intensity"].values,
                                    plot_colorbar=plot_colorbar
                                )
                    except Exception as e:
                        logging.warning(f"  ⚠ stations/DYFI failed for v{version}: {e}")
    
                # 4) epicenter (optional)
                try:
                    # support both parser.metadata and parser.get_metadata()
                    meta = parser.metadata if hasattr(parser, "metadata") else parser.get_metadata()
                    ev_meta = meta.get("event", {}) if isinstance(meta, dict) else {}
                    if "lon" in ev_meta and "lat" in ev_meta:
                        mapper.add_epicenter(float(ev_meta["lon"]), float(ev_meta["lat"]))
                except Exception as e:
                    logging.warning(f"  ⚠ epicenter failed for v{version}: {e}")
    
                # 5) cities overlay
                if add_cities:
                    try:
                        mapper.add_cities(population=cities_population)
                    except Exception as e:
                        logging.warning(f"  ⚠ cities overlay failed for v{version}: {e}")
    
                # 6) finalize & title
                fig, ax = mapper.get_figure()
                if show_title:
                    ax.set_title(f"{self.event_id} – ShakeMap v{version} ({metric.upper()})", pad=12)
                else:
                    ax.set_title("")
    
                figures.append((fig, ax, version))
    
                # 7) save if requested
                if output_path:
                    out_dir = (
                        Path(output_path)
                        / "SHAKEtime" / self.event_id
                        / "shakemaps" / metric / mode
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for fmt in save_formats:
                        save_path = out_dir / f"{self.event_id}_shakemap_v{version}_{metric}_{mode}.{fmt}"
                        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
                        logging.info(f"Saved ShakeMap plot to {save_path}")
    
            except Exception as e:
                logging.error(f"  ✖ plotting failed for v{version}: {e}")
    
        return figures

    




    #v26.01.4
    def plot_std_maps(
        self,
        version_list: list,
        metric: str = "mmi",
        rupture_folder: str = None,
        stations_folder: str = None,
        add_cities: bool = False,
        cities_population: int = 1000000,
        output_path: str = None,
        plot_colorbar: bool = True,
        show_title: bool = True,
        save_formats: list = ["png", "pdf"],
        dpi: int = 300,
        # --- new plotting controls ---
        cmap: str = "viridis",
        vmin: float = None,
        vmax: float = None,
        norm_type: str = "linear",   # "linear" | "log"
        cbar_label: str = None,
        cbar_orientation: str = "vertical",
        cbar_pad: float = 0.05,
        marker_size: float = 15.0,
        marker_edgecolor: str = "k",
        marker_linewidth: float = 0.1,
        show_obs: bool = True,
        obs_size: float = 18.0,
    ) -> list:
        """
        Plot ShakeMap uncertainty (standard deviation) maps for each version,
        with the same per-version stations/DYFI overlay behavior as plot_shakemaps().
    
        Notes
        -----
        - Reads the version-specific *uncertainty.xml* file by replacing "grid" with "uncertainty"
          in the base ShakeMap filename.
        - STD layers are non-negative; defaults and normalization reflect that.
        - `stations_folder` overlays instrumental stations and DYFI points for EACH version.
        """
        from pathlib import Path
        import logging
        import numpy as np
        import pandas as pd
        import os
    
        import cartopy.crs as ccrs
        from matplotlib.colors import Normalize, LogNorm
    
        # -----------------------------
        # helper: robust numeric cleaning (same spirit as plot_shakemaps)
        # -----------------------------
        _NULL_TOKENS = {
            "", " ", "nan", "NaN", "NAN",
            "null", "NULL", "Null",
            "none", "None", "NONE",
            "na", "NA", "N/A", "n/a",
        }
    
        def _clean_null_tokens(df: pd.DataFrame, cols: list) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].apply(
                        lambda x: np.nan
                        if (isinstance(x, str) and x.strip() in _NULL_TOKENS)
                        else x
                    )
            return df
    
        def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            df = _clean_null_tokens(df, cols)
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
    
        # -----------------------------
        # map metric -> std column in uncertainty.xml
        # -----------------------------
        m = (metric or "").strip().lower()
        std_col_map = {
            "mmi": "STDMMI",
            "pga": "STDPGA",
            "pgv": "STDPGV",
            # common PSA naming variants (if your parser supports them)
            "psa03": "STDPSA03",
            "psa0.3": "STDPSA03",
            "psa10": "STDPSA10",
            "psa1.0": "STDPSA10",
            "psa30": "STDPSA30",
            "psa3.0": "STDPSA30",
        }
        std_col = std_col_map.get(m)
        if std_col is None:
            raise ValueError(
                f"Unsupported metric={metric!r} for std maps. "
                f"Supported: {sorted(std_col_map.keys())}"
            )
    
        # -----------------------------
        # validate norm type
        # -----------------------------
        if norm_type not in ("linear", "log"):
            raise ValueError("norm_type must be 'linear' or 'log'")
    
        figures = []
    
        for version in version_list:
            # 1) Load uncertainty XML for this version
            base_fn = self._get_shakemap_filename(version)
            un_fn = base_fn.replace("grid", "uncertainty")
            xml_path = os.path.join(self.shakemap_folder, self.event_id, un_fn)
    
            logging.info(f"[plot_std_maps] v{version}: loading {xml_path!r}")
    
            try:
                parser = USGSParser(
                    parser_type="shakemap_xml",
                    xml_file=xml_path,
                    imt=m,               # parser may use this internally; ok to pass
                    value_type="std"
                )
            except Exception as e:
                logging.error(f"  ✖ parse failed for v{version} std XML: {e}")
                continue
    
            # 2) Base map
            mapper = SHAKEmapper()
            try:
                meta = parser.metadata if hasattr(parser, "metadata") else parser.get_metadata()
                gs = meta.get("grid_specification", {}) if isinstance(meta, dict) else {}
                extent = [
                    float(gs["lon_min"]), float(gs["lon_max"]),
                    float(gs["lat_min"]), float(gs["lat_max"]),
                ]
                mapper.set_extent(extent)
            except Exception as e:
                logging.warning(f"  ⚠ extent not set from metadata for v{version}: {e}")
    
            fig, ax = mapper.create_basemap(label_size=22)
    
            # 3) Extract grid dataframe + validate std column
            df = parser.get_dataframe()
            if df is None or df.empty:
                logging.error(f"  ✖ empty dataframe for v{version}")
                continue
    
            # enforce numeric
            df = _coerce_numeric(df, ["LON", "LAT", std_col])
            df = df.dropna(subset=["LON", "LAT", std_col])
            if df.empty:
                logging.error(f"  ✖ no valid (lon,lat,{std_col}) after cleaning for v{version}")
                continue
    
            # 4) Determine vmin/vmax defaults (std is non-negative)
            data = df[std_col].values
            # avoid log issues
            data_pos = data[data > 0] if data is not None else np.array([])
    
            if vmin is None:
                # sensible default: 0 for linear; small positive for log
                vmin_eff = 0.0 if norm_type == "linear" else (float(np.nanmin(data_pos)) if data_pos.size else 1e-6)
            else:
                vmin_eff = float(vmin)
    
            if vmax is None:
                vmax_eff = float(np.nanpercentile(data, 99))  # robust upper bound
            else:
                vmax_eff = float(vmax)
    
            # protect log norm
            if norm_type == "log":
                if vmin_eff <= 0:
                    vmin_eff = float(np.nanmin(data_pos)) if data_pos.size else 1e-6
                if vmax_eff <= vmin_eff:
                    vmax_eff = vmin_eff * 10.0
    
                norm = LogNorm(vmin=vmin_eff, vmax=vmax_eff)
            else:
                norm = Normalize(vmin=vmin_eff, vmax=vmax_eff)
    
            # 5) Scatter plot of std grid
            scatter = ax.scatter(
                df["LON"].values,
                df["LAT"].values,
                c=df[std_col].values,
                cmap=cmap,
                norm=norm,
                s=float(marker_size),
                edgecolor=marker_edgecolor,
                linewidth=float(marker_linewidth),
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
    
            # 6) Optional rupture overlay
            if rupture_folder:
                rup_fn = self._get_rupture_filename(version)
                rup_path = os.path.join(rupture_folder, self.event_id, rup_fn)
                logging.info(f"  adding rupture from {rup_path}")
                try:
                    rup_p = USGSParser(parser_type="rupture_json", rupture_json=str(rup_path))
                    xs, ys = rup_p.get_rupture_xy()
                    mapper.add_rupture(xs, ys)
                except Exception as e:
                    logging.warning(f"  ⚠ rupture failed for v{version}: {e}")
    
            # 7) Optional stations + DYFI overlay (per version)
            if stations_folder and show_obs:
                st_fn = self._get_stations_filename(version)
                st_path = Path(stations_folder) / self.event_id / st_fn
                logging.info(f"  adding stations/DYFI from {st_path}")
                try:
                    inst_parser = USGSParser(parser_type="instrumented_data", json_file=str(st_path))
    
                    # instrumental stations (PGA points)
                    inst_df = inst_parser.get_dataframe(value_type="pga")
                    if inst_df is not None and not inst_df.empty:
                        inst_df = _coerce_numeric(inst_df, ["longitude", "latitude"])
                        inst_df = inst_df.dropna(subset=["longitude", "latitude"])
                        if not inst_df.empty:
                            ax.scatter(
                                inst_df["longitude"].values,
                                inst_df["latitude"].values,
                                s=float(obs_size),
                                marker="^",
                                facecolor="none",
                                edgecolor="black",
                                linewidth=0.8,
                                transform=ccrs.PlateCarree(),
                                zorder=5,
                            )
    
                    # DYFI points (MMI)
                    dyfi_df = inst_parser.get_dataframe(value_type="mmi")
                    if dyfi_df is not None and not dyfi_df.empty:
                        dyfi_df = _coerce_numeric(dyfi_df, ["longitude", "latitude", "intensity", "nresp"])
                        dyfi_df = dyfi_df.dropna(subset=["longitude", "latitude"])
                        if not dyfi_df.empty:
                            ax.scatter(
                                dyfi_df["longitude"].values,
                                dyfi_df["latitude"].values,
                                s=float(obs_size),
                                marker="o",
                                facecolor="none",
                                edgecolor="tab:red",
                                linewidth=0.8,
                                transform=ccrs.PlateCarree(),
                                zorder=5,
                            )
                except Exception as e:
                    logging.warning(f"  ⚠ stations/DYFI overlay failed for v{version}: {e}")
    
            # 8) Epicenter
            try:
                meta = parser.metadata if hasattr(parser, "metadata") else parser.get_metadata()
                ev = meta.get("event", {}) if isinstance(meta, dict) else {}
                if "lon" in ev and "lat" in ev:
                    mapper.add_epicenter(float(ev["lon"]), float(ev["lat"]))
            except Exception as e:
                logging.warning(f"  ⚠ epicenter failed for v{version}: {e}")
    
            # 9) Cities overlay
            if add_cities:
                try:
                    mapper.add_cities(population=cities_population)
                except Exception as e:
                    logging.warning(f"  ⚠ cities overlay failed for v{version}: {e}")
    
            # 10) Title
            if show_title:
                ax.set_title(f"{self.event_id} – StdDev v{version} ({metric.upper()})", pad=12)
            else:
                ax.set_title("")
    
            # 11) Colorbar
            if plot_colorbar:
                if cbar_label is None:
                    # default label based on std column
                    cbar_label_eff = f"{std_col} (std dev)"
                else:
                    cbar_label_eff = str(cbar_label)
    
                cbar = fig.colorbar(
                    scatter,
                    ax=ax,
                    orientation=cbar_orientation,
                    pad=float(cbar_pad),
                )
                cbar.ax.tick_params(labelsize=16)
                cbar.set_label(cbar_label_eff, fontsize=18)
    
            figures.append((fig, ax, version))
    
            # 12) Save
            if output_path:
                out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "std_maps" / metric.lower()
                out_dir.mkdir(parents=True, exist_ok=True)
                for fmt in save_formats:
                    save_path = out_dir / f"{self.event_id}_std_map_v{version}_{metric.lower()}.{fmt}"
                    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                    logging.info(f"Saved std map to {save_path}")
    
        return figures
    

    
    


        
    def plot_rate_maps(
        self,
        version_list: list,
        metric: str = "mmi",
        specific_columns: list = None,
        extent: list = None,
        output_path: str = None,
        plot_colorbar: bool = True,
        show_title: bool = True,
        save_formats: list = ["png", "pdf"],
        dpi: int = 300
    ) -> list:
        """
        Plots and optionally saves rate-of-change maps using Cartopy.

        If specific_columns is provided, only those columns in the unified grid are plotted.
        Otherwise, for each consecutive version pair, this method plots two maps:
          one for the delta and one for the rate.

        Parameters
        ----------
        version_list : list
            Versions to process.
        metric : str, default "mmi"
            'mmi', 'pga', or 'pgv'.
        specific_columns : list, optional
            If provided, only these columns are plotted.
        extent : list, optional
            [min_lon, max_lon, min_lat, max_lat] extent to apply.
        output_path : str, optional
            Base folder to save outputs.
        plot_colorbar : bool, default True
            Whether to draw colorbar on each map.
        show_title : bool, default True
            Whether to show map title.
        save_formats : list of str, default ["png","pdf"]
            File extensions to save ("png", "pdf", "svg", etc.).
        dpi : int, default 300
            Resolution in dots per inch for saved figures.
        Returns
        -------
        List[tuple]
            List of (fig, ax, column) tuples for each generated plot.
        """
        metric_labels = {
            'mmi': 'Modified Mercalli Intensity',
            'pga': 'Peak Ground Acceleration',
            'pgv': 'Peak Ground Velocity'
        }
        human = metric_labels.get(metric.lower(), metric.upper())

        # build rate grid
        ug = self.get_rate_grid(version_list, metric=metric, use_cache=False)

        figures = []

        def _plot_map(column: str):
            fig, ax = plt.subplots(figsize=(24,16), subplot_kw={'projection': ccrs.PlateCarree()})
            parts = column.split('_')
            kind = 'Change' if parts[0] == 'delta' else 'Rate of Change'
            v2, v1 = parts[1], parts[2]

            if show_title:
                ax.set_title(f"{kind} of {human} [{v1}→{v2}]", fontsize=18)

            norm = Normalize(vmin=-2, vmax=2)
            sc = ax.scatter(
                ug['lon'], ug['lat'], c=ug[column],
                cmap='seismic', norm=norm,
                s=15, edgecolor='k', linewidth=0.1,
                transform=ccrs.PlateCarree(), zorder=8
            )

            if extent:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
            else:
                ax.set_extent([
                    ug['lon'].min(), ug['lon'].max(),
                    ug['lat'].min(), ug['lat'].max()
                ], crs=ccrs.PlateCarree())

            ax.coastlines(zorder=10)
            ax.add_feature(cfeature.BORDERS, zorder=10)
            ax.add_feature(cfeature.OCEAN, facecolor='skyblue', zorder=9)

            gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7, zorder=99)
            gl.top_labels = False; gl.right_labels = False
            gl.xlabel_style = {"size": 18}; gl.ylabel_style = {"size": 18}

            if plot_colorbar:
                cb = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02)
                label = f"Δ {human}" if parts[0] == 'delta' else f"Rate of Change ({human})"
                cb.set_label(label, fontsize=25)

            plt.tight_layout()
            return fig, ax

        # decide which columns to plot
        cols = specific_columns or []
        if not cols:
            for i in range(len(version_list) - 1):
                a, b = version_list[i], version_list[i+1]
                cols.extend([f"delta_{b}_{a}_{metric}", f"rate_{b}_{a}_{metric}"])

        # generate and optionally save
        for col in cols:
            fig, ax = _plot_map(col)
            figures.append((fig, ax, col))
            if output_path:
                out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "rate_of_change_plots" / metric
                out_dir.mkdir(parents=True, exist_ok=True)
                for fmt in save_formats:
                    save_path = out_dir / f"{col}.{fmt}"
                    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
                    logging.info(f"Saved rate map to {save_path}")

        return figures




    def plot_ratemap_details(
        self,
        version_list: list,
        metric: str = "mmi",
        specific_columns: list = None,
        rupture_folder: str = None,
        stations_folder: str = None,
        add_cities: bool = False,
        cities_population: int = 1_000_000,
        output_path: str = None,
        plot_colorbar: bool = True,
        show_title: bool = True,
        save_formats: list = ["png", "pdf"],
        dpi: int = 300,
        use_cache: bool = False,
        which: str = "both",
    ) -> list:
        """
        Detailed rate‐of‐change maps with SHAKEmapper, always overlaying
        rupture + only the *new* stations/DYFI for each delta/rate, but for
        the final‐vs‐first column, plot all stations/DYFI (no subtraction).
        Can choose to plot only 'delta', only 'rate', or 'both'.
        """
        from pathlib import Path
        from matplotlib.colors import Normalize
        import logging
        import pandas as pd
    
        # 1) build the rate grid
        ug = self.get_rate_grid(version_list, metric=metric, use_cache=use_cache)
    
        # 2) pick which columns to plot
        cols = list(specific_columns) if specific_columns else []
        if not cols:
            # consecutive‐pairs
            for i in range(len(version_list) - 1):
                v1, v2 = version_list[i], version_list[i + 1]
                cols += [f"delta_{v2}_{v1}_{metric}", f"rate_{v2}_{v1}_{metric}"]
            # append final vs first
            first, last = version_list[0], version_list[-1]
            cols += [f"delta_{last}_{first}_{metric}", f"rate_{last}_{first}_{metric}"]
    
        # 3) filter by 'which'
        if which not in ("delta", "rate", "both"):
            raise ValueError(f"Invalid which='{which}'; must be 'delta','rate', or 'both'")
        if which == "delta":
            cols = [c for c in cols if c.startswith("delta_")]
        elif which == "rate":
            cols = [c for c in cols if c.startswith("rate_")]
    
        # registry of already‐plotted station & DYFI IDs/codes
        plotted_station_ids = set()
        plotted_station_codes = set()
        plotted_dyfi_ids = set()
        plotted_dyfi_codes = set()
    
        first, last = version_list[0], version_list[-1]
        final_delta_col = f"delta_{last}_{first}_{metric}"
        final_rate_col = f"rate_{last}_{first}_{metric}"
    
        figs = []
    
        for col in cols:
            # detect if this is the final‐vs‐first column
            is_final_first = col in (final_delta_col, final_rate_col)
    
            # extract v2 from "delta_v2_v1_metric"
            parts = col.split("_")
            v2 = parts[1]
    
            # compute map extent from entire grid
            extent = [
                float(ug.lon.min()),
                float(ug.lon.max()),
                float(ug.lat.min()),
                float(ug.lat.max()),
            ]
    
            # 4) create mapper at that extent
            mapper = SHAKEmapper(extent=extent)
            fig, ax = mapper.create_basemap(figsize=figsize,label_size=22)
    
            # 5) plot the scatter
            norm = Normalize(vmin=-2, vmax=2)
            sc = ax.scatter(
                ug.lon,
                ug.lat,
                c=ug[col],
                cmap="seismic",
                norm=norm,
                s=15,
                edgecolor="none",
                transform=mapper.ax.projection,
                zorder=8,
            )
    
            # 6) title + colorbar
            if show_title:
                kind = "Change" if col.startswith("delta") else "Rate of Change"
                ax.set_title(f"{kind} ({metric.upper()}) {col}", fontsize=16)
            if plot_colorbar:
                cb = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
                cb.set_label(col, fontsize=12)
    
            # 7) overlay rupture for version v2
            if rupture_folder:
                rf = self._get_rupture_filename(v2)
                rp = Path(rupture_folder) / self.event_id / rf
                logging.info(f"[delta {col}] rupture JSON: {rp} (exists={rp.exists()})")
                if rp.exists():
                    try:
                        rup = USGSParser(parser_type="rupture_json", rupture_json=str(rp))
                        xs, ys = rup.get_rupture_xy()
                        if xs:
                            mapper.add_rupture(xs, ys)
                            logging.info("  ✓ rupture plotted")
                    except Exception as e:
                        logging.warning(f"  ⚠ rupture parse failed: {e}")
    
            # 8) overlay *new* stations + DYFI for version v2
            if stations_folder:
                sf = self._get_stations_filename(v2)
                sp = Path(stations_folder) / self.event_id / sf
                logging.info(f"[delta {col}] stations JSON: {sp} (exists={sp.exists()})")
                if sp.exists():
                    try:
                        ip = USGSParser(parser_type="instrumented_data", json_file=str(sp))
    
                        # ---------------------------
                        # PGA → station markers
                        # ---------------------------
                        df_sta = ip.get_dataframe(value_type="pga")
    
                        # Robust numeric cleaning (handles "null"/"NULL"/""/strings)
                        for c in ["longitude", "latitude"]:
                            if c in df_sta.columns:
                                df_sta[c] = pd.to_numeric(df_sta[c], errors="coerce")
    
                        df_sta = df_sta.dropna(subset=["longitude", "latitude"])
    
                        if not df_sta.empty:
                            if is_final_first:
                                new_sta = df_sta
                            else:
                                mask_sta = (
                                    ~df_sta["id"].astype(str).isin(plotted_station_ids)
                                    & ~df_sta["station_code"].astype(str).isin(plotted_station_codes)
                                )
                                new_sta = df_sta[mask_sta]
                            if not new_sta.empty:
                                mapper.add_stations(
                                    new_sta["longitude"].values,
                                    new_sta["latitude"].values,
                                )
                                if not is_final_first:
                                    plotted_station_ids.update(new_sta["id"].astype(str))
                                    plotted_station_codes.update(new_sta["station_code"].astype(str))
                                logging.info(f"  ✓ plotted {len(new_sta)} station points")
    
                        # ---------------------------
                        # MMI → DYFI
                        # ---------------------------
                        df_dy = ip.get_dataframe(value_type="mmi")
    
                        # Robust numeric cleaning (handles "null"/"NULL"/""/strings)
                        for c in ["longitude", "latitude", "intensity"]:
                            if c in df_dy.columns:
                                df_dy[c] = pd.to_numeric(df_dy[c], errors="coerce")
    
                        # Optional: coerce nresp if present (won't affect filtering)
                        if "nresp" in df_dy.columns:
                            df_dy["nresp"] = pd.to_numeric(df_dy["nresp"], errors="coerce")
    
                        df_dy = df_dy.dropna(subset=["longitude", "latitude", "intensity"])
    
                        if not df_dy.empty:
                            if is_final_first:
                                new_dy = df_dy
                            else:
                                mask_dy = (
                                    ~df_dy["id"].astype(str).isin(plotted_dyfi_ids)
                                    & ~df_dy["station_code"].astype(str).isin(plotted_dyfi_codes)
                                )
                                new_dy = df_dy[mask_dy]
                            if not new_dy.empty:
                                mapper.add_dyfi(
                                    new_dy["longitude"].values,
                                    new_dy["latitude"].values,
                                    new_dy["intensity"].values,
                                    nresp=new_dy.get("nresp"),
                                )
                                if not is_final_first:
                                    plotted_dyfi_ids.update(new_dy["id"].astype(str))
                                    plotted_dyfi_codes.update(new_dy["station_code"].astype(str))
                                logging.info(f"  ✓ plotted {len(new_dy)} DYFI points")
    
                    except Exception as e:
                        logging.warning(f"  ⚠ stations/DYFI parse failed: {e}")
    
            # 9) optional cities
            if add_cities:
                try:
                    mapper.add_cities(population=cities_population)
                    logging.info("  ✓ cities plotted")
                except Exception as e:
                    logging.warning(f"  ⚠ cities overlay failed: {e}")
    
            # 10) save each
            if output_path:
                od = (
                    Path(output_path)
                    / "SHAKEtime"
                    / self.event_id
                    / "rate_map_details"
                    / metric
                )
                od.mkdir(parents=True, exist_ok=True)
                for ext in save_formats:
                    fp = od / f"{col}.{ext}"
                    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
                    logging.info(f"  ✓ saved {fp}")
    
            figs.append((fig, ax, col))
    
        return figs
    

    # upadted 26
    def plot_ratemap_details(
        self,
        version_list: list,
        metric: str = "mmi",
        specific_columns: list = None,
        rupture_folder: str = None,
        stations_folder: str = None,
        add_cities: bool = False,
        cities_population: int = 1_000_000,
        output_path: str = None,
        plot_colorbar: bool = True,
        show_title: bool = True,
        save_formats: list = ("png", "pdf"),
        dpi: int = 300,
        use_cache: bool = False,
        which: str = "both",
        # ---- figure controls ----
        figsize: tuple = (6, 6),
        label_size: int = 22,          # basemap gridline label size (SHAKEmapper.create_basemap)
        tick_size: int = 12,           # axes tick font size
        title_size: int = 16,          # title font size
        # ---- scatter controls ----
        scatter_size: float = 15,      # size of the delta/rate grid points
        scatter_edgecolor: str = "none",
        scatter_alpha: float = 1.0,
        # ---- colormap / norm controls ----
        cmap: str = "seismic",
        vmin: float = -2.0,
        vmax: float = 2.0,
        # ---- colorbar controls ----
        cbar_pad: float = 0.02,
        cbar_shrink: float = 0.9,
        cbar_label: str = None,        # override cbar label (defaults to `col`)
        cbar_labelsize: int = 12,
        cbar_ticksize: int = 11,
        # ---- overlay marker sizes ----
        station_marker_size: float = 150,  # SHAKEmapper.add_stations default is 150
        dyfi_marker_size: float = 150,     # SHAKEmapper.add_dyfi default is 150
        # ---- cities controls ----
        cities_marker_size: float = 7,     # SHAKEmapper.add_cities default markersize=7
        cities_label_fontsize: int = 15,   # SHAKEmapper.add_cities default label_fontsize=15
        # ---- behavior ----
        show: bool = False,
        close_figs: bool = False,
    ) -> list:
        """
        Detailed rate-of-change maps using SHAKEmapper.
        For each delta/rate column:
          - plot delta/rate grid
          - overlay rupture (version v2)
          - overlay only NEW stations/DYFI (except final-vs-first: plot ALL)
          - optional cities overlay
    
        Returns
        -------
        list of (fig, ax, col)
        """
        from pathlib import Path
        import logging
        import pandas as pd
        import cartopy.crs as ccrs
        from matplotlib.colors import Normalize
    
        # 1) build the rate grid
        ug = self.get_rate_grid(version_list, metric=metric, use_cache=use_cache)
    
        # 2) choose columns to plot
        cols = list(specific_columns) if specific_columns else []
        if not cols:
            for i in range(len(version_list) - 1):
                v1, v2 = version_list[i], version_list[i + 1]
                cols += [f"delta_{v2}_{v1}_{metric}", f"rate_{v2}_{v1}_{metric}"]
            first, last = version_list[0], version_list[-1]
            cols += [f"delta_{last}_{first}_{metric}", f"rate_{last}_{first}_{metric}"]
    
        # 3) filter by 'which'
        if which not in ("delta", "rate", "both"):
            raise ValueError(f"Invalid which='{which}'; must be 'delta','rate', or 'both'")
        if which == "delta":
            cols = [c for c in cols if c.startswith("delta_")]
        elif which == "rate":
            cols = [c for c in cols if c.startswith("rate_")]
    
        # track already plotted stations/dyfi across steps
        plotted_station_ids = set()
        plotted_station_codes = set()
        plotted_dyfi_ids = set()
        plotted_dyfi_codes = set()
    
        first, last = version_list[0], version_list[-1]
        final_delta_col = f"delta_{last}_{first}_{metric}"
        final_rate_col  = f"rate_{last}_{first}_{metric}"
    
        figs = []
    
        # precompute extent once from the unified grid
        extent = [
            float(ug.lon.min()),
            float(ug.lon.max()),
            float(ug.lat.min()),
            float(ug.lat.max()),
        ]
    
        for col in cols:
            is_final_first = col in (final_delta_col, final_rate_col)
    
            parts = col.split("_")
            if len(parts) < 4:
                logging.warning(f"Skipping unexpected column name format: {col}")
                continue
            v2 = parts[1]
    
            # 4) create basemap
            mapper = SHAKEmapper(extent=extent)
            fig, ax = mapper.create_basemap(figsize=figsize, label_size=label_size)
    
            # ticks (gridline labels are controlled by label_size; this is for tick labels if they appear)
            try:
                ax.tick_params(axis="both", labelsize=tick_size)
            except Exception:
                pass
    
            # 5) plot delta/rate grid as scatter
            norm = Normalize(vmin=vmin, vmax=vmax)
            sc = ax.scatter(
                ug.lon,
                ug.lat,
                c=ug[col],
                cmap=cmap,
                norm=norm,
                s=scatter_size,
                edgecolor=scatter_edgecolor,
                alpha=scatter_alpha,
                transform=ccrs.PlateCarree(),
                zorder=8,
                rasterized=True,
            )
    
            # 6) title + colorbar
            if show_title:
                kind = "Change" if col.startswith("delta") else "Rate of Change"
                ax.set_title(f"{kind} ({metric.upper()}) {col}", fontsize=title_size)
    
            if plot_colorbar:
                cb = fig.colorbar(
                    sc,
                    ax=ax,
                    orientation="vertical",
                    pad=cbar_pad,
                    shrink=cbar_shrink,
                )
                label_to_use = col if cbar_label is None else cbar_label
                cb.set_label(label_to_use, fontsize=cbar_labelsize)
                cb.ax.tick_params(labelsize=cbar_ticksize)
    
            # 7) overlay rupture for version v2
            if rupture_folder:
                rf = self._get_rupture_filename(v2)
                rp = Path(rupture_folder) / self.event_id / rf
                logging.info(f"[rate_map {col}] rupture JSON: {rp} (exists={rp.exists()})")
                if rp.exists():
                    try:
                        rup = USGSParser(parser_type="rupture_json", rupture_json=str(rp))
                        xs, ys = rup.get_rupture_xy()
                        if xs:
                            mapper.add_rupture(xs, ys)  # uses mapper defaults
                            logging.info("  ✓ rupture plotted")
                    except Exception as e:
                        logging.warning(f"  ⚠ rupture parse failed: {e}")
    
            # 8) overlay stations + DYFI for version v2
            if stations_folder:
                sf = self._get_stations_filename(v2)
                sp = Path(stations_folder) / self.event_id / sf
                logging.info(f"[rate_map {col}] stations JSON: {sp} (exists={sp.exists()})")
    
                if sp.exists():
                    try:
                        ip = USGSParser(parser_type="instrumented_data", json_file=str(sp))
    
                        # ---------- Stations (PGA dataframe) ----------
                        df_sta = ip.get_dataframe(value_type="pga").copy()
    
                        for c in ("longitude", "latitude"):
                            if c in df_sta.columns:
                                df_sta[c] = pd.to_numeric(df_sta[c], errors="coerce")
                        df_sta = df_sta.dropna(subset=["longitude", "latitude"])
    
                        if not df_sta.empty:
                            if is_final_first:
                                new_sta = df_sta
                            else:
                                mask_sta = (
                                    ~df_sta["id"].astype(str).isin(plotted_station_ids)
                                    & ~df_sta["station_code"].astype(str).isin(plotted_station_codes)
                                )
                                new_sta = df_sta[mask_sta]
    
                            if not new_sta.empty:
                                mapper.add_stations(
                                    new_sta["longitude"].values,
                                    new_sta["latitude"].values,
                                    s=station_marker_size,
                                )
                                if not is_final_first:
                                    plotted_station_ids.update(new_sta["id"].astype(str))
                                    plotted_station_codes.update(new_sta["station_code"].astype(str))
                                logging.info(f"  ✓ plotted {len(new_sta)} station points")
    
                        # ---------- DYFI (MMI dataframe) ----------
                        df_dy = ip.get_dataframe(value_type="mmi").copy()
    
                        for c in ("longitude", "latitude", "intensity"):
                            if c in df_dy.columns:
                                df_dy[c] = pd.to_numeric(df_dy[c], errors="coerce")
                        if "nresp" in df_dy.columns:
                            df_dy["nresp"] = pd.to_numeric(df_dy["nresp"], errors="coerce")
    
                        df_dy = df_dy.dropna(subset=["longitude", "latitude", "intensity"])
    
                        if not df_dy.empty:
                            if is_final_first:
                                new_dy = df_dy
                            else:
                                mask_dy = (
                                    ~df_dy["id"].astype(str).isin(plotted_dyfi_ids)
                                    & ~df_dy["station_code"].astype(str).isin(plotted_dyfi_codes)
                                )
                                new_dy = df_dy[mask_dy]
    
                            if not new_dy.empty:
                                # Important: keep DYFI colorbar off (your map already has a delta/rate colorbar)
                                mapper.add_dyfi(
                                    new_dy["longitude"].values,
                                    new_dy["latitude"].values,
                                    new_dy["intensity"].values,
                                    nresp=new_dy.get("nresp"),
                                    plot_colorbar=False,
                                    s=dyfi_marker_size,
                                )
                                if not is_final_first:
                                    plotted_dyfi_ids.update(new_dy["id"].astype(str))
                                    plotted_dyfi_codes.update(new_dy["station_code"].astype(str))
                                logging.info(f"  ✓ plotted {len(new_dy)} DYFI points")
    
                    except Exception as e:
                        logging.warning(f"  ⚠ stations/DYFI parse failed: {e}")
    
            # 9) cities overlay
            if add_cities:
                try:
                    mapper.add_cities(
                        population=cities_population,
                        markersize=cities_marker_size,
                        label_fontsize=cities_label_fontsize,
                    )
                    logging.info("  ✓ cities plotted")
                except Exception as e:
                    logging.warning(f"  ⚠ cities overlay failed: {e}")
    
            # 10) save
            if output_path:
                od = Path(output_path) / "SHAKEtime" / self.event_id / "rate_map_details" / metric
                od.mkdir(parents=True, exist_ok=True)
                for ext in save_formats:
                    fp = od / f"{col}.{ext}"
                    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
                    logging.info(f"  ✓ saved {fp}")
    
            if show:
                import matplotlib.pyplot as plt
                plt.show()
    
            if close_figs:
                import matplotlib.pyplot as plt
                plt.close(fig)
    
            figs.append((fig, ax, col))
    
        return figs
    



    def _plot_thd_axes(
        self,
        ax,
        metric_type: str,
        up_to_version: str,
        full_n: int,
        tick_positions: List[int],
        tick_labels: List[str],
        delta: bool = False,
        show_title: bool = True,
        units: str = "%g"
    ):
        """
        Plot a locked‐x‐axis THD (or ΔTHD) time series up to a given version.
    
        ...
        """
        # 1) prepare sorted summary
        df = (
            self.summary_df
                .assign(v_int=lambda d: d["version"].astype(int))
                .sort_values("v_int")
                .reset_index(drop=True)
        )
    
        base = metric_type.lower()
        # choose columns, labels, colors
        if delta:
            cols = {
                "mean": f"{base}_delta_mean",
                "min":  f"{base}_delta_min",
                "max":  f"{base}_delta_max",
                "std":  f"{base}_delta_std",
            }
            ylabel_base = f"Δ{base.upper()}"
            title_text  = f"Temporal Hazard (Δ{base.upper()})"
            line_colors = {"mmi":"#2E3C72","pga":"#B23F24","pgv":"#3C8E50","psa10":"#5F4F88"}
        else:
            cols = {
                "mean": f"{base}_mean",
                "min":  f"{base}_min",
                "max":  f"{base}_max",
                "std":  f"{base}_std",
            }
            ylabel_base = base.upper()
            title_text  = f"Temporal Hazard ({base.upper()})"
            line_colors = {"mmi":"#4C72B0","pga":"#DD8452","pgv":"#55A868","psa10":"#8172B2"}
    
        color     = line_colors.get(base, "C0")
        std_color = {"mmi":"#A1BEDC","pga":"#F3BCA3","pgv":"#A8D5B0","psa10":"#B9AEE3"}.get(base, color)
    
        # 2) select up-to-version
        cutoff = int(up_to_version)
        df_up  = df[df.v_int <= cutoff].copy()
        x_up   = df_up.index.values
    
        # 3) extract series
        mean_s = df_up[cols["mean"]].copy()
        min_s  = df_up[cols["min"]].copy()
        max_s  = df_up[cols["max"]].copy()
        std_s  = df_up[cols["std"]] if cols["std"] in df_up else None
    
        # 4) conversion (only if units != "%g" and base in pga/psa10)
        if units != "%g" and base in ("pga","psa10"):
            try:
                conv = AccelerationUnitConverter()
            except NameError:
                conv = None
            if conv:
                mean_s[:] = conv.convert_unit(mean_s, "%g", units)
                min_s[:]  = conv.convert_unit(min_s,  "%g", units)
                max_s[:]  = conv.convert_unit(max_s,  "%g", units)
                if std_s is not None:
                    std_s[:] = conv.convert_unit(std_s, "%g", units)
    
        # 5) build ylabel: show for anything except any MMI variant
        if "mmi" not in base:
            ylabel = f"{ylabel_base} ({units})"
        else:
            ylabel = ylabel_base
    
        # 6) plot
        ax.plot(x_up, mean_s, "-o", ms=6, color=color, zorder=3)
        ax.fill_between(x_up, min_s, max_s, color=color, alpha=0.25, zorder=2)
        if std_s is not None:
            ax.errorbar(
                x_up, mean_s,
                yerr=std_s,
                fmt="none",
                ecolor=std_color,
                capsize=3,
                zorder=4
            )
    
        # 7) formatting
        ax.set_xlim(-0.5, full_n - 0.5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
    
        if show_title:
            ax.set_title(title_text, fontsize=12)

            



    def _plot_single_delta(
        self,
        ax,
        v1: str,
        v2: str,
        metric: str = "mmi",
        rupture_xy: tuple[list[float], list[float]] = (None, None),
        new_sta_lons: list[float] | None = None,
        new_sta_lats: list[float] | None = None,
        new_dyfi_lons: list[float] | None = None,
        new_dyfi_lats: list[float] | None = None,
        new_dyfi_int: list[float] | None = None,
        add_cities: bool = False,
        cities_population: int = 1_000_000,
        plot_colorbar: bool = True,
    ):
        """
        Draw ΔShakeMap (v2 − v1) on `ax`, locked to the full grid extent,
        then overlay:
          • rupture_xy = (xs, ys) if provided
          • new_sta_{lons,lats} if provided
          • new_dyfi_{lons,lats,int} if provided
          • optional city markers
        """
        col = f"delta_{v2}_{v1}_{metric}"
        if col not in self._rate_grid.columns:
            logging.warning(f"No '{col}' in rate grid → skipping delta plot")
            ax.axis("off")
            return

        logging.info(f"Plotting ΔShake {v1}→{v2} ({metric})")

        # 1) lock to grid extent
        lon_min, lon_max = self._rate_grid.lon.min(), self._rate_grid.lon.max()
        lat_min, lat_max = self._rate_grid.lat.min(), self._rate_grid.lat.max()
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], PlateCarree())

        # 2) map background (coastlines/oceans/borders already drawn upstream)

        # 3) scatter the delta grid
        sc = ax.scatter(
            self._rate_grid.lon,
            self._rate_grid.lat,
            c=self._rate_grid[col],
            cmap="seismic",
            norm=Normalize(vmin=-2, vmax=2),
            s=12,
            transform=PlateCarree(),
            zorder=5
        )
        if plot_colorbar:
            cb = ax.figure.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
            cb.set_label(col, fontsize=9)

        ax.set_title(f"ΔShake {v1}→{v2}", fontsize=12)

        # 4) rupture overlay
        xs, ys = rupture_xy
        if xs:
            logging.info(f"Overlaying rupture (n={len(xs)})")
            try:
                mapper = SHAKEmapper(extent=[lon_min, lon_max, lat_min, lat_max])
                mapper.create_basemap()
                mapper.ax = ax
                mapper.add_rupture(xs, ys)
            except Exception as e:
                logging.warning(f"  Failed to overlay rupture: {e!r}")

        # 5) stations overlay
        if new_sta_lons is not None and len(new_sta_lons):
            logging.info(f"Overlaying {len(new_sta_lons)} new stations")
            try:
                mapper = SHAKEmapper(extent=[lon_min, lon_max, lat_min, lat_max])
                mapper.ax = ax
                mapper.add_stations(new_sta_lons, new_sta_lats)
            except Exception as e:
                logging.warning(f"  Failed to overlay stations: {e!r}")

        # 6) DYFI overlay
        if new_dyfi_lons is not None and len(new_dyfi_lons):
            logging.info(f"Overlaying {len(new_dyfi_lons)} new DYFI points")
            try:
                mapper = SHAKEmapper(extent=[lon_min, lon_max, lat_min, lat_max])
                mapper.ax = ax
                mapper.add_dyfi(new_dyfi_lons, new_dyfi_lats, new_dyfi_int)
            except Exception as e:
                logging.warning(f"  Failed to overlay DYFI: {e!r}")

        # 7) cities if requested
        if add_cities:
            logging.info("Overlaying cities")
            try:
                mapper = SHAKEmapper(extent=[lon_min, lon_max, lat_min, lat_max])
                mapper.ax = ax
                mapper.add_cities(population=cities_population)
            except Exception as e:
                logging.warning(f"  Failed to overlay cities: {e!r}")


    

    def create_overview_panels(
        self,
        version_list: List[str],
        rupture_folder: str = None,
        stations_folder: str = None,
        add_cities: bool = False,
        cities_population: int = 1_000_000,
        output_path: str = None,
        plot_colorbar: bool = True,
        show_title: bool = True,
        save_formats: List[str] = ["png","pdf"],
        figsize: tuple[float,float] = (18,12),
        dpi: int = 300,
        use_cache: bool = True
    ) -> List[plt.Figure]:
        """
        For each ShakeMap version, build a 4×3 panel (ShakeMap, ΔShakeMap, THDs, ΔTHDs, stds,
        plus two aux time‐series).  Adds a "time‐after‐event" annotation on each map,
        and only plots ΔShakeMap when available (with rupture + new stations/DYFI overlays).
        On the final version, shows two panels: one for Δ(final–prev) in the usual slot,
        and then a separate extra panel for Δ(final–first).
        """
        from cartopy.crs import PlateCarree
        import cartopy.feature as cfeature
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import logging

        # 1) ensure all summaries & grids exist
        if not use_cache or not hasattr(self, 'summary_df') or self.summary_df.empty \
           or set(version_list) - set(self.summary_df.version.astype(str)):
            self.get_shake_summary(version_list)
        if "mmi_mean" not in self.summary_df.columns or not use_cache:
            self.add_shakemap_pgm()
        if "stdmmi_mean" not in self.summary_df.columns or not use_cache:
            self.add_shakemap_stdpgm()
        for m in ("mmi","pga"):
            if f"{m}_delta_mean" not in self.summary_df.columns or not use_cache:
                self.add_rate_to_summary(version_list=version_list, metric=m, use_cache=use_cache)
        # prepare rate grid for ΔShakeMap
        self.get_rate_grid(version_list, metric="mmi", use_cache=use_cache)

        # 2) prepare THD ticks/labels
        df_sum = (
            self.summary_df
                .assign(v_int=lambda d: d.version.astype(int))
                .sort_values("v_int")
                .reset_index(drop=True)
        )
        total    = len(version_list)
        tick_pos = list(range(total))
        tick_labs = (
            [f"{v:.1f}" for v in df_sum["TaE_h"]]
            if "TaE_h" in df_sum.columns else
            df_sum["version"].astype(str).tolist()
        )

        # 3) compute auxiliary influences once
        aux_res = self.analyze_auxiliary_influences(
            version_list,
            station_folder=stations_folder,
            dyfi_folder=stations_folder,
            rupture_folder=rupture_folder,
            thresholds=[6.0,7.0,8.0],
            uncertainty_percentile=90.0,
            bootstrap_iters=200,
            radius_km=30,
            metric='mmi',
            cache_folder=None,
            use_cache=use_cache,
            file_type="csv"
        )
        df_aux  = aux_res.get("aux", pd.DataFrame()).reset_index()
        df_diag = aux_res.get("diag", pd.DataFrame()).reset_index()

        panels = []
        # Loop through each version: plot ShakeMap + Δ(prev→curr)
        for idx, ver in enumerate(version_list):
            df_aux_cur  = df_aux.iloc[:idx+1]
            df_diag_cur = df_diag.iloc[:idx+1]

            fig = plt.figure(figsize=figsize)
            gs  = fig.add_gridspec(4,3,
                                   width_ratios=[2,1,1],
                                   height_ratios=[1,1,1,1],
                                   wspace=0.3, hspace=0.4)

            # ShakeMap panel (rows 0–1, col 0)
            ax0 = fig.add_subplot(gs[0:2,0], projection=PlateCarree())
            ax0.coastlines(zorder=10)
            ax0.add_feature(cfeature.BORDERS, zorder=10)
            ax0.add_feature(cfeature.OCEAN, facecolor="skyblue", zorder=9)
            ax0.grid(linestyle="--", alpha=0.3, zorder=20)

            mapper = SHAKEmapper()
            mapper.create_basemap(); mapper.fig = fig; mapper.ax = ax0
            
            sm_fn = self._get_shakemap_filename(ver)
            sm_p  = Path(self.shakemap_folder)/self.event_id/sm_fn
            parser = USGSParser("shakemap_xml", xml_file=str(sm_p), imt="mmi", value_type="mean")
            try:
                md = parser.get_metadata()["grid_specification"]
                ext = [float(md["lon_min"]), float(md["lon_max"]),
                       float(md["lat_min"]), float(md["lat_max"])]
                mapper.set_extent(ext); ax0.set_extent(ext, PlateCarree())
            except:
                pass
            mapper.add_usgs_shakemap(parser, plot_colorbar=plot_colorbar)

            ax0.set_title(f"SHAKEmap v{ver}", fontsize=14, pad=12)

            gl = ax0.gridlines(crs=ccrs.PlateCarree(),
                               draw_labels=True, linewidth=2,
                               color='gray', alpha=0.7,
                               linestyle='--', zorder=99)
            gl.top_labels   = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 12}
            gl.ylabel_style = {"size": 12}

            # Time-after-event annotation
            if "TaE_h" in df_sum.columns:
                tae = df_sum.loc[df_sum["version"].astype(str)==ver, "TaE_h"].iloc[0]
                lab = f"{tae:.1f}\u2009hr" if tae<=24 else f"{(tae/24):.1f}\u2009d"
                ax0.text(
                    0.02, 0.98, lab,
                    transform=ax0.transAxes, ha="left", va="top",
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                    zorder=20
                )

            # Overlays on ShakeMap
            if rupture_folder:
                rj = Path(rupture_folder)/self.event_id/self._get_rupture_filename(ver)
                if rj.exists():
                    try:
                        xs, ys = USGSParser("rupture_json", rupture_json=str(rj)).get_rupture_xy()
                        if xs: mapper.add_rupture(xs, ys)
                    except:
                        pass
            if stations_folder:
                sj = Path(stations_folder)/self.event_id/self._get_stations_filename(ver)
                
                if sj.exists():
                    try:
                        ip = USGSParser("instrumented_data", json_file=str(sj))
                        # PGA stations, only if columns exist
                        dfp = ip.get_dataframe(value_type="pga")
                        
                        if {'longitude','latitude'}.issubset(dfp.columns):
                            dfp = dfp.dropna(subset=["longitude","latitude"])
                            
                            if not dfp.empty:
                                mapper.add_stations(dfp.longitude.values, dfp.latitude.values)
                        # DYFI, only if columns exist

                        dfd = ip.get_dataframe(value_type="mmi")
                        for col in ("longitude","latitude","intensity","distance"):
                            if col in dfd.columns:
                                dfd[col] = pd.to_numeric(dfd[col], errors="coerce")
                        dfd = dfd.dropna(subset=["longitude","latitude","intensity"])
                        if "distance" in dfd.columns:
                            dfd = dfd[dfd["distance"] <= 400]
                        if not dfd.empty:
                            mapper.add_dyfi(
                                dfd.longitude.values,
                                dfd.latitude.values,
                                dfd.intensity.values
                            )
           
                    except Exception:
                        logging.warning(f"No station/DYFI overlay for version {ver}, skipping.")
            try:
                ev = parser.get_metadata()["event"]
                mapper.add_epicenter(float(ev["lon"]), float(ev["lat"]))
            except:
                pass
            if add_cities:
                try: mapper.add_cities(population=cities_population)
                except: pass

            # Delta ShakeMap (rows 2–3, col 0) for idx>0
            if idx > 0:
                prev = version_list[idx-1]
                ax1 = fig.add_subplot(gs[2:4,0], projection=PlateCarree())
                ax1.coastlines(zorder=10)
                ax1.add_feature(cfeature.BORDERS, zorder=10)
                ax1.add_feature(cfeature.OCEAN, facecolor="skyblue", zorder=9)
                ax1.grid(linestyle="--", alpha=0.3, zorder=20)

                gl1 = ax1.gridlines(crs=ccrs.PlateCarree(),
                               draw_labels=True, linewidth=2,
                               color='gray', alpha=0.7,
                               linestyle='--', zorder=99)
                gl1.top_labels   = False
                gl1.right_labels = False
                gl1.xlabel_style = {"size": 12}
                gl1.ylabel_style = {"size": 12}

                # init registries on first delta
                if idx == 1:
                    self._plotted_station_ids = set()
                    self._plotted_station_codes = set()
                    self._plotted_dyfi_ids = set()
                    self._plotted_dyfi_codes = set()

                # pull rupture coords
                rupture_xy = (None, None)
                if rupture_folder:
                    rj = Path(rupture_folder)/self.event_id/self._get_rupture_filename(ver)
                    if rj.exists():
                        try:
                            rupture_xy = USGSParser(
                                parser_type="rupture_json", rupture_json=str(rj)
                            ).get_rupture_xy()
                        except:
                            rupture_xy = (None, None)

                # pull only-new stations & DYFI, with guards
                new_sta_lons = new_sta_lats = None
                new_dyfi_lons = new_dyfi_lats = new_dyfi_int = None
                if stations_folder:
                    sf = Path(stations_folder)/self.event_id/self._get_stations_filename(ver)
                    if sf.exists():
                        ip = USGSParser(parser_type="instrumented_data", json_file=str(sf))
                        # PGA stations
                        df_sta = ip.get_dataframe(value_type="pga")
                        if {'longitude','latitude'}.issubset(df_sta.columns):
                            df_sta = df_sta.dropna(subset=["longitude","latitude"])
                            if not df_sta.empty:
                                mask = (
                                    ~df_sta["id"].astype(str).isin(self._plotted_station_ids) &
                                    ~df_sta["station_code"].astype(str).isin(self._plotted_station_codes)
                                )
                                new_sta = df_sta[mask]
                                if not new_sta.empty:
                                    new_sta_lons = new_sta["longitude"].values
                                    new_sta_lats = new_sta["latitude"].values
                                    self._plotted_station_ids |= set(new_sta["id"].astype(str))
                                    self._plotted_station_codes |= set(new_sta["station_code"].astype(str))
                        # DYFI
                        df_dyfi = ip.get_dataframe(value_type="mmi")

                        if {'longitude','latitude','intensity'}.issubset(df_dyfi.columns):
                            for col in ("longitude","latitude","intensity","distance"):
                                if col in df_dyfi.columns:
                                    df_dyfi[col] = pd.to_numeric(df_dyfi[col], errors="coerce")

                                if "distance" in df_dyfi.columns:
                                    df_dyfi = df_dyfi[df_dyfi["distance"] <= 400]
                            df_dyfi = df_dyfi.dropna(subset=["longitude","latitude","intensity"])
                            if not df_dyfi.empty:
                                mask = (
                                    ~df_dyfi["id"].astype(str).isin(self._plotted_dyfi_ids) &
                                    ~df_dyfi["station_code"].astype(str).isin(self._plotted_dyfi_codes)
                                )
                                new_dy = df_dyfi[mask]
                                if not new_dy.empty:
                                    new_dyfi_lons = new_dy["longitude"].values
                                    new_dyfi_lats = new_dy["latitude"].values
                                    new_dyfi_int  = new_dy["intensity"].values
                                    self._plotted_dyfi_ids |= set(new_dy["id"].astype(str))
                                    self._plotted_dyfi_codes |= set(new_dy["station_code"].astype(str))

                # helper call
                self._plot_single_delta(
                    ax=ax1, v1=prev, v2=ver, metric="mmi",
                    rupture_xy=rupture_xy,
                    new_sta_lons=new_sta_lons, new_sta_lats=new_sta_lats,
                    new_dyfi_lons=new_dyfi_lons, new_dyfi_lats=new_dyfi_lats, new_dyfi_int=new_dyfi_int,
                    add_cities=add_cities, cities_population=cities_population,
                    plot_colorbar=plot_colorbar
                )

            # AUX slot always blank
            ax_blank = fig.add_subplot(gs[3,0])
            ax_blank.axis("off")

            # THD & ΔTHD & std panels (cols 1–2)
            ax2 = fig.add_subplot(gs[0,1])
            self._plot_thd_axes(ax2, "mmi", ver, total, tick_pos, tick_labs, delta=False)
            ax3 = fig.add_subplot(gs[0,2])
            self._plot_thd_axes(ax3, "mmi", ver, total, tick_pos, tick_labs, delta=True)
            ax4 = fig.add_subplot(gs[1,1])
            self._plot_thd_axes(ax4, "pga", ver, total, tick_pos, tick_labs, delta=False)
            ax5 = fig.add_subplot(gs[1,2])
            self._plot_thd_axes(ax5, "pga", ver, total, tick_pos, tick_labs, delta=True)
            ax6 = fig.add_subplot(gs[2,1])
            self._plot_thd_axes(ax6, "stdmmi", ver, total, tick_pos, tick_labs, delta=False)
            ax7 = fig.add_subplot(gs[2,2])
            self._plot_thd_axes(ax7, "stdpga", ver, total, tick_pos, tick_labs, delta=False)

            # AUX time series (row 3, cols 1–2)
            x_here = np.arange(idx+1)
            ax8 = fig.add_subplot(gs[3,1], sharex=ax2)
            for col in ("station_count","dyfi_count","trace_length_km"):
                if col in df_aux_cur:
                    ax8.plot(x_here, df_aux_cur[col].values, "o-", label=col)
            import matplotlib.ticker as mticker
            ax8.set_yscale("log")
            ax8.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
            sf = mticker.ScalarFormatter(); sf.set_scientific(False); sf.set_useOffset(False)
            ax8.yaxis.set_major_formatter(sf)
            ax8.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax8.set_ylabel("Count / Length (log)")
            ax8.set_xlabel("Time after Event (TaE)")
            ax8.grid(which="both", linestyle="--", alpha=0.5)
            ax8.legend(loc="best")
            ax8.set_xticks(tick_pos)
            ax8.set_xticklabels(tick_labs, rotation=45)
            ax8.set_xlim(-0.5, total-0.5)


            ax9 = fig.add_subplot(gs[3,2], sharex=ax8)
            style_map = {
                "unc_area_pct90":  {"marker":"s","linestyle":"--"},
                "area_exceed_7.0": {"marker":"o","linestyle":"-"},
                "area_exceed_6.0": {"marker":"^","linestyle":"-"},
                "area_exceed_8.0": {"marker":"D","linestyle":"-"},
            }
            for col, st in style_map.items():
                if col in df_diag_cur:
                    ax9.plot(x_here, df_diag_cur[col].values,
                             marker=st["marker"], linestyle=st["linestyle"], label=col)
            ax9.set_ylabel("Area (km²)")
            ax9.grid(linestyle="--", alpha=0.5)
            ax9.legend(loc="best")
            ax9.set_xticks(tick_pos); ax9.set_xticklabels(tick_labs,rotation=45)
            ax9.set_xlim(-0.5, total-0.5); ax9.set_xlabel("Time after Event (TaE)")

            if show_title:
                fig.suptitle(f"{self.event_id} Overview → v{ver}", fontsize=18)

            # save
            if output_path:
                od = Path(output_path)/"SHAKEtime"/self.event_id/"overview_panels"
                od.mkdir(parents=True, exist_ok=True)
                for ext in save_formats:
                    fig.savefig(od/f"{self.event_id}_overview_v{ver}.{ext}",
                                bbox_inches="tight", dpi=dpi)

            panels.append(fig)

        # ─── extra Δ(final–first) ───
        # (constructs a final figure exactly like above, but compares first→last)
        # ──────────────────────────────────────────────────────────────────────────
        # extra panel for Δ(final–first)
        v_first, v_last = version_list[0], version_list[-1]
        
        fig2 = plt.figure(figsize=figsize)
        gs2 = fig2.add_gridspec(
            4, 3,
            width_ratios =[2,1,1],
            height_ratios=[1,1,1,1],
            wspace=0.3, hspace=0.4
        )
        
        # ── ShakeMap v_last (rows 0–1, col 0) ──
        ax0 = fig2.add_subplot(gs2[0:2, 0], projection=PlateCarree())
        ax0.coastlines(zorder=10)
        ax0.add_feature(cfeature.BORDERS, zorder=10)
        ax0.add_feature(cfeature.OCEAN,   facecolor="skyblue", zorder=9)
        ax0.grid(linestyle="--", alpha=0.3, zorder=20)
        
        mapper = SHAKEmapper(); mapper.fig = fig2; mapper.ax = ax0
        sm_fn   = self._get_shakemap_filename(v_last)
        sm_p    = Path(self.shakemap_folder) / self.event_id / sm_fn
        parser  = USGSParser("shakemap_xml", xml_file=str(sm_p), imt="mmi", value_type="mean")
        # lock to grid extent if available
        try:
            md = parser.get_metadata()["grid_specification"]
            ext = [
                float(md["lon_min"]), float(md["lon_max"]),
                float(md["lat_min"]), float(md["lat_max"])
            ]
            mapper.set_extent(ext)
            ax0.set_extent(ext, PlateCarree())
        except:
            pass
        
        mapper.add_usgs_shakemap(parser, plot_colorbar=plot_colorbar)
        ax0.set_title(f"SHAKEmap v{ver}", fontsize=14, pad=12)

        gl = ax0.gridlines(crs=ccrs.PlateCarree(),
                               draw_labels=True, linewidth=2,
                               color='gray', alpha=0.7,
                               linestyle='--', zorder=99)
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 12}
        gl.ylabel_style = {"size": 12}

        
        # — rupture overlay —
        if rupture_folder:
            rj = Path(rupture_folder) / self.event_id / self._get_rupture_filename(v_last)
            if rj.exists():
                try:
                    xs, ys = USGSParser("rupture_json", rupture_json=str(rj)).get_rupture_xy()
                    if xs:
                        mapper.add_rupture(xs, ys)
                except:
                    pass
        
        # — all stations overlay —
        if stations_folder:
            sj = Path(stations_folder) / self.event_id / self._get_stations_filename(v_last)
            if sj.exists():
                try:
                    ip   = USGSParser("instrumented_data", json_file=str(sj))
                    dfp  = ip.get_dataframe(value_type="pga").dropna(subset=["longitude","latitude"])
                    if not dfp.empty:
                        mapper.add_stations(dfp.longitude.values, dfp.latitude.values)




                        ###########

                    dfd = ip.get_dataframe(value_type="mmi")
                    for col in ("longitude","latitude","intensity","distance"):
                        if col in dfd.columns:
                            dfd[col] = pd.to_numeric(dfd[col], errors="coerce")
                    dfd = dfd.dropna(subset=["longitude","latitude","intensity"])
                    if "distance" in dfd.columns:
                        dfd = dfd[dfd["distance"] <= 400]
                    if not dfd.empty:
                        mapper.add_dyfi(
                            dfd.longitude.values,
                            dfd.latitude.values,
                            dfd.intensity.values
                        )
                except:
                    pass
        
        # — epicenter —
        try:
            ev = parser.get_metadata()["event"]
            mapper.add_epicenter(float(ev["lon"]), float(ev["lat"]))
        except:
            pass
        
        # — cities (optional) —
        if add_cities:
            try:
                mapper.add_cities(population=cities_population)
            except:
                pass
        
        
        # ── ΔShakeMap(final–first) (rows 2–3, col 0) ──
        ax1 = fig2.add_subplot(gs2[2:4, 0], projection=PlateCarree())
        ax1.coastlines(zorder=10)
        ax1.add_feature(cfeature.BORDERS, zorder=10)
        ax1.add_feature(cfeature.OCEAN,   facecolor="skyblue", zorder=9)
        ax1.grid(linestyle="--", alpha=0.3, zorder=20)
        gl = ax1.gridlines(crs=ccrs.PlateCarree(),
                               draw_labels=True, linewidth=2,
                               color='gray', alpha=0.7,
                               linestyle='--', zorder=99)
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 12}
        gl.ylabel_style = {"size": 12}
        
        # compute rupture_xy for v_last
        rupture_xy = (None, None)
        if rupture_folder:
            rj = Path(rupture_folder) / self.event_id / self._get_rupture_filename(v_last)
            if rj.exists():
                try:
                    rupture_xy = USGSParser("rupture_json", rupture_json=str(rj)).get_rupture_xy()
                except:
                    rupture_xy = (None, None)
        
        # grab all stations & DYFI from final version
        new_sta_lons = new_sta_lats = None
        new_dyfi_lons = new_dyfi_lats = new_dyfi_int = None
        if stations_folder:
            sf = Path(stations_folder) / self.event_id / self._get_stations_filename(v_last)
            if sf.exists():
                ip     = USGSParser("instrumented_data", json_file=str(sf))
                df_sta = ip.get_dataframe(value_type="pga").dropna(subset=["longitude","latitude"])
                if not df_sta.empty:
                    new_sta_lons = df_sta["longitude"].values
                    new_sta_lats = df_sta["latitude"].values

                df_dyfi = ip.get_dataframe(value_type="mmi")
                for col in ("longitude","latitude","intensity","distance"):
                    if col in df_dyfi.columns:
                        df_dyfi[col] = pd.to_numeric(df_dyfi[col], errors="coerce")
                df_dyfi = df_dyfi.dropna(subset=["longitude","latitude","intensity"])
                if "distance" in df_dyfi.columns:
                    df_dyfi = df_dyfi[df_dyfi["distance"] <= 400]
                    
                #df_dyfi = ip.get_dataframe(value_type="mmi").dropna(subset=["longitude","latitude","intensity"])
                if not df_dyfi.empty:
                    new_dyfi_lons = df_dyfi["longitude"].values
                    new_dyfi_lats = df_dyfi["latitude"].values
                    new_dyfi_int  = df_dyfi["intensity"].values
        
        # plot the delta now with full data
        self._plot_single_delta(
            ax=ax1,
            v1=v_first, v2=v_last, metric="mmi",
            rupture_xy=rupture_xy,
            new_sta_lons=new_sta_lons, new_sta_lats=new_sta_lats,
            new_dyfi_lons=new_dyfi_lons, new_dyfi_lats=new_dyfi_lats, new_dyfi_int=new_dyfi_int,
            add_cities=add_cities, cities_population=cities_population,
            plot_colorbar=plot_colorbar
        )
        
        # ── copy THD, std, and aux-time-series panels for v_last ──
        ax2 = fig2.add_subplot(gs2[0,1])
        self._plot_thd_axes(ax2,  "mmi",   v_last, total, tick_pos, tick_labs, delta=False)
        ax3 = fig2.add_subplot(gs2[0,2])
        self._plot_thd_axes(ax3,  "mmi",   v_last, total, tick_pos, tick_labs, delta=True)
        ax4 = fig2.add_subplot(gs2[1,1])
        self._plot_thd_axes(ax4,  "pga",   v_last, total, tick_pos, tick_labs, delta=False)
        ax5 = fig2.add_subplot(gs2[1,2])
        self._plot_thd_axes(ax5,  "pga",   v_last, total, tick_pos, tick_labs, delta=True)
        ax6 = fig2.add_subplot(gs2[2,1])
        self._plot_thd_axes(ax6,  "stdmmi",v_last, total, tick_pos, tick_labs, delta=False)
        ax7 = fig2.add_subplot(gs2[2,2])
        self._plot_thd_axes(ax7,  "stdpga",v_last, total, tick_pos, tick_labs, delta=False)
        
        x2 = np.arange(len(version_list))
        ax8 = fig2.add_subplot(gs2[3,1], sharex=ax2)
        for col in ("station_count","dyfi_count","trace_length_km"):
            if col in df_aux:
                ax8.plot(x2, df_aux[col].values[:len(version_list)], "o-", label=col)
        ax8.set_yscale("log")
        ax8.legend(loc="best")
        ax8.set_xticks(tick_pos); ax8.set_xticklabels(tick_labs, rotation=45)
        import matplotlib.ticker as mticker
        ax8.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
        sf = mticker.ScalarFormatter(); sf.set_scientific(False); sf.set_useOffset(False)
        ax8.yaxis.set_major_formatter(sf)
        ax8.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax8.set_ylabel("Count / Length (log)")
        ax8.grid(which="both", linestyle="--", alpha=0.5)
        ax8.set_xticks(tick_pos)
        ax8.set_xticklabels(tick_labs, rotation=45)
        ax8.set_xlim(-0.5, total-0.5)

        
        ax9 = fig2.add_subplot(gs2[3,2], sharex=ax2)
        for col, st in style_map.items():
            if col in df_diag:
                ax9.plot(
                    x2,
                    df_diag[col].values[:len(version_list)],
                    marker=st["marker"],
                    linestyle=st["linestyle"],
                    label=col
                )
        ax9.legend(loc="best")
        ax9.set_ylabel("Area (km²)")
        ax9.grid(linestyle="--", alpha=0.5)
        ax9.set_xticks(tick_pos); ax9.set_xticklabels(tick_labs,rotation=45)
        ax9.set_xlim(-0.5, total-0.5); ax9.set_xlabel("Version")

        
        if show_title:
            fig2.suptitle(f"{self.event_id} Overview → v{v_last} (final, Δfirst)", fontsize=18)
        
        if output_path:
            od2 = Path(output_path)/"SHAKEtime"/self.event_id/"overview_panels" 
            od2.mkdir(parents=True, exist_ok=True)
            for ext in save_formats:
                fig2.savefig(
                    od2/f"{self.event_id}_overview_v{v_last}_extra.{ext}",
                    bbox_inches="tight", dpi=dpi
                )
        
        panels.append(fig2)

        return panels





        
    def create_final_overview_panel(
        self,
        version_list: List[str],
        rupture_folder: str = None,
        stations_folder: str = None,
        add_cities: bool = False,
        cities_population: int = 1_000_000,
        output_path: str = None,
        plot_colorbar: bool = True,
        show_title: bool = True,
        save_formats: List[str] = ["png","pdf"],
        figsize: tuple[float,float] = (18,12),
        dpi: int = 300,
        use_cache: bool = True,
        thd_panels: List[str] = None
    ) -> plt.Figure:
        """
        Create only the final overview panel (Δ(final–first)) for ShakeMap versions.

        Matches the "extra" panel in create_overview_panels but as a standalone.
        thd_panels: list of any of
          ["mmi", "mmi_delta", "pga", "pga_delta",
           "stdmmi", "stdpga", "aux", "diag"]
        to draw.  Default None → all eight.
        """
        from cartopy.crs import PlateCarree
        import cartopy.feature as cfeature
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import logging

        # ─── 1) Ensure summary & rate‐grid ───
        if (not use_cache
           or not hasattr(self,'summary_df')
           or self.summary_df.empty
           or set(version_list)-set(self.summary_df.version.astype(str))):
            self.get_shake_summary(version_list)
        if "mmi_mean" not in self.summary_df.columns or not use_cache:
            self.add_shakemap_pgm()
        if "stdmmi_mean" not in self.summary_df.columns or not use_cache:
            self.add_shakemap_stdpgm()
        for m in ("mmi","pga"):
            col = f"{m}_delta_mean"
            if col not in self.summary_df.columns or not use_cache:
                self.add_rate_to_summary(version_list=version_list, metric=m, use_cache=use_cache)
        self.get_rate_grid(version_list, metric="mmi", use_cache=use_cache)

        # ─── 2) Prepare ticks/labels ───
        df_sum = (
            self.summary_df
                .assign(v_int=lambda d: d.version.astype(int))
                .sort_values("v_int")
                .reset_index(drop=True)
        )
        total    = len(version_list)
        tick_pos = list(range(total))
        tick_labs = (
            [f"{v:.1f}" for v in df_sum["TaE_h"]]
            if "TaE_h" in df_sum.columns else
            df_sum["version"].astype(str).tolist()
        )

        # ─── 3) Aux & diag ───
        aux_res = self.analyze_auxiliary_influences(
            version_list,
            station_folder=stations_folder,
            dyfi_folder=stations_folder,
            rupture_folder=rupture_folder,
            thresholds=[6.0,7.0,8.0],
            uncertainty_percentile=90.0,
            bootstrap_iters=200,
            radius_km=30,
            metric='mmi',
            cache_folder=None,
            use_cache=use_cache,
            file_type="csv"
        )
        df_aux  = aux_res.get("aux", pd.DataFrame()).reset_index()
        df_diag = aux_res.get("diag", pd.DataFrame()).reset_index()

        # ─── 4) Panel selection ───
        ALL = ["mmi","mmi_delta","pga","pga_delta","stdmmi","stdpga","aux","diag"]
        if thd_panels is None:
            thd_panels = ALL.copy()
        else:
            bad = set(thd_panels) - set(ALL)
            if bad:
                raise ValueError(f"Unknown thd_panels: {bad}")

        def blank(ax):
            ax.axis('off')

        # ─── 5) First and last ───
        v_first, v_last = version_list[0], version_list[-1]

        # ─── 6) Build figure ───
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(4,3,
                               width_ratios =[2,1,1],
                               height_ratios=[1,1,1,1],
                               wspace=0.3, hspace=0.4)

        # — ShakeMap final —
        ax0 = fig.add_subplot(gs[0:2,0], projection=PlateCarree())
        ax0.coastlines(zorder=10); ax0.add_feature(cfeature.BORDERS, zorder=10)
        ax0.add_feature(cfeature.OCEAN, facecolor="skyblue", zorder=9)
        ax0.grid(linestyle="--", alpha=0.3, zorder=20)
        mapper = SHAKEmapper(); mapper.fig, mapper.ax = fig, ax0
        sm_fn = self._get_shakemap_filename(v_last)
        sm_p  = Path(self.shakemap_folder)/self.event_id/sm_fn
        parser=USGSParser("shakemap_xml", xml_file=str(sm_p), imt="mmi", value_type="mean")
        try:
            md  = parser.get_metadata()["grid_specification"]
            ext = [float(md[k]) for k in ("lon_min","lon_max","lat_min","lat_max")]
            mapper.set_extent(ext); ax0.set_extent(ext, PlateCarree())
        except: pass
        mapper.add_usgs_shakemap(parser, plot_colorbar=("on" if plot_colorbar else "off"))
        if rupture_folder:
            rj = Path(rupture_folder)/self.event_id/self._get_rupture_filename(v_last)
            if rj.exists():
                try:
                    xs, ys = USGSParser("rupture_json",rupture_json=str(rj)).get_rupture_xy()
                    mapper.add_rupture(xs, ys)
                except: pass
        if stations_folder:
            sj = Path(stations_folder)/self.event_id/self._get_stations_filename(v_last)
            if sj.exists():
                try:
                    ip  = USGSParser("instrumented_data", json_file=str(sj))
                    dfp = ip.get_dataframe(value_type="pga").dropna(subset=["longitude","latitude"])
                    if not dfp.empty:
                        mapper.add_stations(dfp.longitude.values, dfp.latitude.values)
                    dfd = ip.get_dataframe(value_type="mmi")
                    # numeric + drop + distance filter
                    for c in ("longitude","latitude","intensity","distance"):
                        if c in dfd.columns:
                            dfd[c] = pd.to_numeric(dfd[c], errors="coerce")
                    dfd = dfd.dropna(subset=["longitude","latitude","intensity"])
                    if "distance" in dfd.columns:
                        dfd = dfd[dfd.distance <= 400]
                    if not dfd.empty:
                        mapper.add_dyfi(dfd.longitude.values, dfd.latitude.values, dfd.intensity.values)
                except: pass
        if add_cities:
            try: mapper.add_cities(population=cities_population)
            except: pass

        # — Δ(final–first) —
        ax1 = fig.add_subplot(gs[2:4,0], projection=PlateCarree())
        ax1.coastlines(zorder=10); ax1.add_feature(cfeature.BORDERS, zorder=10)
        ax1.add_feature(cfeature.OCEAN, facecolor="skyblue", zorder=9)
        ax1.grid(linestyle="--", alpha=0.3, zorder=20)
        # rupture coords
        rupture_xy=(None,None)
        if rupture_folder:
            rj = Path(rupture_folder)/self.event_id/self._get_rupture_filename(v_last)
            if rj.exists():
                try: rupture_xy = USGSParser("rupture_json",rupture_json=str(rj)).get_rupture_xy()
                except: rupture_xy=(None,None)
        # all stations & DYFI
        new_sta_lons=new_sta_lats=None
        new_dyfi_lons=new_dyfi_lats=new_dyfi_int=None
        if stations_folder:
            sf = Path(stations_folder)/self.event_id/self._get_stations_filename(v_last)
            if sf.exists():
                ip     = USGSParser("instrumented_data", json_file=str(sf))
                df_sta = ip.get_dataframe(value_type="pga").dropna(subset=["longitude","latitude"])
                if not df_sta.empty:
                    new_sta_lons,new_sta_lats = df_sta.longitude.values, df_sta.latitude.values
                df_dy  = ip.get_dataframe(value_type="mmi")
                for c in ("longitude","latitude","intensity","distance"):
                    if c in df_dy.columns:
                        df_dy[c] = pd.to_numeric(df_dy[c], errors="coerce")
                df_dy = df_dy.dropna(subset=["longitude","latitude","intensity"])
                if "distance" in df_dy.columns:
                    df_dy = df_dy[df_dy.distance <= 400]
                if not df_dy.empty:
                    new_dyfi_lons, new_dyfi_lats, new_dyfi_int = (
                        df_dy.longitude.values,
                        df_dy.latitude.values,
                        df_dy.intensity.values
                    )
        self._plot_single_delta(
            ax=ax1, v1=v_first, v2=v_last, metric="mmi",
            rupture_xy=rupture_xy,
            new_sta_lons=new_sta_lons, new_sta_lats=new_sta_lats,
            new_dyfi_lons=new_dyfi_lons, new_dyfi_lats=new_dyfi_lats, new_dyfi_int=new_dyfi_int,
            add_cities=add_cities, cities_population=cities_population,
            plot_colorbar=plot_colorbar
        )

        # ─── THD / ΔTHD / STD / AUX / DIAG ───
        # positions:
        ax2 = fig.add_subplot(gs[0,1]);  ax3 = fig.add_subplot(gs[0,2])
        ax4 = fig.add_subplot(gs[1,1]);  ax5 = fig.add_subplot(gs[1,2])
        ax6 = fig.add_subplot(gs[2,1]);  ax7 = fig.add_subplot(gs[2,2])
        ax8 = fig.add_subplot(gs[3,1], sharex=ax2)
        ax9 = fig.add_subplot(gs[3,2], sharex=ax8)

        # helper for THD
        def plot_thd_slot(name, ax, metric, delta):
            if name in thd_panels:
                self._plot_thd_axes(ax, metric, v_last, total, tick_pos, tick_labs, delta=delta)
            else:
                blank(ax)

        plot_thd_slot("mmi",        ax2, "mmi",    False)
        plot_thd_slot("mmi_delta",  ax3, "mmi",    True)
        plot_thd_slot("pga",        ax4, "pga",    False)
        plot_thd_slot("pga_delta",  ax5, "pga",    True)
        plot_thd_slot("stdmmi",     ax6, "stdmmi", False)
        plot_thd_slot("stdpga",     ax7, "stdpga", False)

        # aux
        if "aux" in thd_panels:
            x2 = np.arange(len(version_list))
            for col in ("station_count","dyfi_count","trace_length_km"):
                if col in df_aux:
                    ax8.plot(x2, df_aux[col].values[:len(version_list)], "o-", label=col)
            import matplotlib.ticker as mticker
            ax8.set_yscale("log")
            ax8.yaxis.set_major_locator(mticker.LogLocator(base=10,numticks=6))
            sf = mticker.ScalarFormatter(); sf.set_scientific(False); sf.set_useOffset(False)
            ax8.yaxis.set_major_formatter(sf); ax8.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax8.set_ylabel("Count / Length (log)"); ax8.set_xlabel("Time After Event")
            ax8.grid(linestyle="--", alpha=0.5); ax8.legend(loc="best")
            ax8.set_xticks(tick_pos); ax8.set_xticklabels(tick_labs, rotation=45)
            ax8.set_xlim(-0.5, total-0.5)
        else:
            blank(ax8)

        # diag
        if "diag" in thd_panels:
            x2 = np.arange(len(version_list))
            style_map = {
                "unc_area_pct90":  {"marker":"s","linestyle":"--"},
                "area_exceed_7.0": {"marker":"o","linestyle":"-"},
                "area_exceed_6.0": {"marker":"^","linestyle":"-"},
                "area_exceed_8.0": {"marker":"D","linestyle":"-"},
            }
            for col, st in style_map.items():
                if col in df_diag:
                    ax9.plot(x2, df_diag[col].values[:len(version_list)],
                             marker=st["marker"], linestyle=st["linestyle"], label=col)
            ax9.set_ylabel("Area (km²)"); ax9.grid(linestyle="--", alpha=0.5)
            ax9.legend(loc="best")
            ax9.set_xticks(tick_pos); ax9.set_xticklabels(tick_labs, rotation=45)
            ax9.set_xlim(-0.5, total-0.5); ax9.set_xlabel("Time After Event")
        else:
            blank(ax9)

        # ─── Title & Save ───
        if show_title:
            fig.suptitle(f"{self.event_id} Overview → v{v_last} (final, Δfirst)", fontsize=18)

        if output_path:
            od = Path(output_path)/"SHAKEtime"/self.event_id/"overview_panels"
            od.mkdir(parents=True, exist_ok=True)
            for ext in save_formats:
                fig.savefig(od/f"{self.event_id}_overview_v{v_last}_extra.{ext}",
                            bbox_inches="tight", dpi=dpi)

        return fig



    def create_evolution_panel(
        self,
        version_list: List[str],
        metric: str = "mmi",
        thresholds: List[float] = [3.0],
        uncertainty_percentile: float = 90.0,
        bootstrap_iters: int = 1000,
        figsize: tuple = (16, 12),
        font_sizes: dict = None,
        grid_kwargs: dict = None,
        output_path: str = None,
        save_formats: List[str] = ["png", "pdf"],
        dpi: int = 300,
        show_title: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Create a 2x2 panel summarizing evolution diagnostics:
          1) Bootstrap CIs for mean, std, min, and max
          2) Global difference stats: MAE, mean_diff, RMSE
          3) Spatial correlation over time (hours after event)
          4) Variogram parameters: sill and nugget
    
        X-axis uses version indices for equal spacing, labeled with TaE_h.
        """
        import numpy as np
        import pandas as pd
        from pathlib import Path
    
        # Ensure evolution data is available
        evo = self.quantify_evolution(
            version_list,
            metric=metric,
            thresholds=thresholds,
            uncertainty_percentile=uncertainty_percentile,
            bootstrap_iters=bootstrap_iters
        )
    
        # Retrieve Time After Event (hours) labels for ticks
        summary = self.summary_df.set_index('version')
        tae = summary.loc[version_list, 'TaE_h'].astype(float)
        x_labels = [f"{h:.1f}" for h in tae]
        # X-axis positions (equally spaced by version)
        x = np.arange(len(version_list))
    
        # Bootstrap CIs
        mean_lo = evo['bootstrap_mean_lo']
        mean_hi = evo['bootstrap_mean_hi']
        std_lo  = evo['bootstrap_std_lo']
        std_hi  = evo['bootstrap_std_hi']
        min_lo  = evo.get('bootstrap_min_lo', pd.Series(index=evo.index, dtype=float))
        min_hi  = evo.get('bootstrap_min_hi', pd.Series(index=evo.index, dtype=float))
        max_lo  = evo.get('bootstrap_max_lo', pd.Series(index=evo.index, dtype=float))
        max_hi  = evo.get('bootstrap_max_hi', pd.Series(index=evo.index, dtype=float))
    
        # Global diff stats for each consecutive pair
        diff_stats = self.compute_global_diff_stats(
            version_list=version_list,
            metric=metric,
            pairwise=True
        )
        mae = [np.nan] * len(version_list)
        mean_diff = [np.nan] * len(version_list)
        rmse = [np.nan] * len(version_list)
        for i in range(1, len(version_list)):
            key = f"{version_list[i-1]}_{version_list[i]}"
            ds = diff_stats.get(key, {})
            mae[i]       = ds.get('mae', np.nan)
            mean_diff[i] = ds.get('mean_diff', np.nan)
            rmse[i]      = ds.get('rmse', np.nan)
    
        # Spatial correlation (ordered by version)
        corr_series = self.spatial_correlation(version_list, metric)
        corr_vals = corr_series.reindex(version_list).values
    
        # Variogram parameters
        variogram_params = {
            v: self.variogram_analysis(v, metric=metric, plot=False, **kwargs)
            for v in version_list
        }
        nugget = [variogram_params[v].get('nugget', np.nan) for v in version_list]
        sill   = [variogram_params[v].get('sill',   np.nan) for v in version_list]
    
        # Plot configuration
        if font_sizes is None:
            font_sizes = {'title': 14, 'labels': 12, 'ticks': 10}
        if grid_kwargs is None:
            grid_kwargs = {'linestyle': '--', 'alpha': 0.5}
    
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax1, ax2, ax3, ax4 = axes.flatten()
    
        # Common x-axis label
        x_label = "Time After Event (hours)"
    
        # Subplot 1: Bootstrap CIs
        ax1.fill_between(x, mean_lo, mean_hi, color='C0', alpha=0.2)
        ax1.fill_between(x, std_lo, std_hi, color='C1', alpha=0.2)
        ax1.fill_between(x, min_lo, min_hi, color='C2', alpha=0.2)
        ax1.fill_between(x, max_lo, max_hi, color='C3', alpha=0.2)
        ax1.plot(x, mean_lo, '--', label='Mean CI', color='C0')
        ax1.plot(x, mean_hi, '--', color='C0')
        ax1.plot(x, std_lo, '-.', label='Std CI', color='C1')
        ax1.plot(x, std_hi, '-.', color='C1')
        ax1.plot(x, min_lo, ':', label='Min CI', color='C2')
        ax1.plot(x, min_hi, ':', color='C2')
        ax1.plot(x, max_lo, '-', label='Max CI', color='C3')
        ax1.plot(x, max_hi, '-', color='C3')
        if show_title:
            ax1.set_title('Bootstrap CIs: mean, std, min, max', fontsize=font_sizes['title'])
        ax1.set_xlabel(x_label, fontsize=font_sizes['labels'])
        ax1.set_ylabel(metric.upper(), fontsize=font_sizes['labels'])
        ax1.legend(fontsize=font_sizes['ticks'])
        ax1.grid(**grid_kwargs)
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, rotation=45)
        ax1.tick_params(axis='both', labelsize=font_sizes['ticks'])
    
        # Subplot 2: Global difference metrics
        ax2.plot(x, mae, 'o-', label='MAE')
        ax2.plot(x, mean_diff, 's-', label='Mean Δ')
        ax2.plot(x, rmse, '^-', label='RMSE')
        if show_title:
            ax2.set_title('Global Difference Metrics', fontsize=font_sizes['title'])
        ax2.set_xlabel(x_label, fontsize=font_sizes['labels'])
        ax2.set_ylabel(metric.upper(), fontsize=font_sizes['labels'])
        ax2.legend(fontsize=font_sizes['ticks'])
        ax2.grid(**grid_kwargs)
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels, rotation=45)
        ax2.tick_params(axis='both', labelsize=font_sizes['ticks'])
    
        # Subplot 3: Spatial correlation
        ax3.plot(x, corr_vals, 'o-', color='C4')
        if show_title:
            ax3.set_title('Spatial Correlation', fontsize=font_sizes['title'])
        ax3.set_xlabel(x_label, fontsize=font_sizes['labels'])
        ax3.set_ylabel('Correlation', fontsize=font_sizes['labels'])
        ax3.grid(**grid_kwargs)
        ax3.set_xticks(x)
        ax3.set_xticklabels(x_labels, rotation=45)
        ax3.tick_params(axis='both', labelsize=font_sizes['ticks'])
    
        # Subplot 4: Variogram parameters
        ax4.plot(x, nugget, 'o-', label='Nugget')
        ax4.plot(x, sill, 's-', label='Sill')
        if show_title:
            ax4.set_title('Variogram Parameters', fontsize=font_sizes['title'])
        ax4.set_xlabel(x_label, fontsize=font_sizes['labels'])
        ax4.set_ylabel('Value', fontsize=font_sizes['labels'])
        ax4.legend(fontsize=font_sizes['ticks'])
        ax4.grid(**grid_kwargs)
        ax4.set_xticks(x)
        ax4.set_xticklabels(x_labels, rotation=45)
        ax4.tick_params(axis='both', labelsize=font_sizes['ticks'])
    
        plt.tight_layout()
        if show_title:
            fig.suptitle(f"Evolution Panel — {self.event_id}", fontsize=font_sizes['title'] + 2)
            plt.subplots_adjust(top=0.92)
    
        # Save if requested
        if output_path:
            out_dir = Path(output_path) / 'SHAKEtime' / self.event_id / 'evolution_panel'
            out_dir.mkdir(parents=True, exist_ok=True)
    
            # 1) full panel
            for fmt in save_formats:
                full_path = out_dir / f"{self.event_id}_evolution_panel.{fmt}"
                fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
    
            # 2) individual subplots
            # draw once so renderer is available
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for idx, ax in enumerate((ax1, ax2, ax3, ax4), start=1):
                # get tight bbox (includes title, labels, ticks, legend)
                bb = ax.get_tightbbox(renderer)
                # pad by 10% in both directions
                bb_padded = bb.padded(0.1)
                # convert to inches
                bb_inches = bb_padded.transformed(fig.dpi_scale_trans.inverted())
                for fmt in save_formats:
                    subpath = out_dir / f"{self.event_id}_evolution_subplot{idx}.{fmt}"
                    fig.savefig(subpath,
                                dpi=dpi,
                                bbox_inches=bb_inches,
                                pad_inches=0.05)
        
            return fig
    



    def plot_pop_exposure(
        self,
        version_list: list = None,
        output_path: str = None,
        x_ticks: str = 'TaE_h',
        save_formats: list = ['png', 'pdf'],
        dpi: int = 100,
        show_title: bool = True,
        figsize: tuple = (24, 12),
    
        # --- NEW: styling kwargs (aligned with plot_alerts) ---
        font_sizes: dict = None,
        xlabel: str = None,
        ylabel: str = None,
        xrotation: int = 45,
        legend_fontsize: int = None,
        legend_loc: str = 'best',
        title_fontsize: int = None,
        grid: bool = False,
        grid_kwargs: dict = None,
    
        # --- figure hygiene ---
        show: bool = False,
        close: bool = False
    
    ) -> (plt.Figure, plt.Axes):
        """
        Plots a stacked bar chart of population exposure over time for different MMI bins.
        """
    
        import logging
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
    
        # --------------------
        # Defaults
        # --------------------
        if font_sizes is None:
            font_sizes = {"labels": 14, "ticks": 12}
        lbl_fs = font_sizes.get("labels", 14)
        tck_fs = font_sizes.get("ticks", 12)
        lgd_fs = legend_fontsize or lbl_fs
        ttl_fs = title_fontsize or (lbl_fs + 4)
    
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.4}
    
        # --------------------
        # Load / validate data
        # --------------------
        df_exposure = self.get_dataframe()
        required_cols = [
            'exposure_IV_V', 'exposure_V_VI', 'exposure_VI_VII',
            'exposure_VII_VIII', 'exposure_VIII_IX', 'exposure_IX_X'
        ]
    
        if (
            df_exposure.empty
            or x_ticks not in df_exposure.columns
            or not all(col in df_exposure.columns for col in required_cols)
        ):
            logging.info("Exposure columns or selected x_ticks missing, regenerating...")
            if version_list is not None:
                self.get_shake_summary(version_list)
                self.add_pager_exposure()
                df_exposure = self.get_dataframe()
            else:
                logging.error("No version list provided and summary is empty.")
                return None, None
    
        # --------------------
        # Styling constants
        # --------------------
        usgs_colors = [
            (122/255, 255/255, 147/255, 1.0),
            (255/255, 255/255,   0/255, 1.0),
            (255/255, 200/255,   0/255, 1.0),
            (255/255, 145/255,   0/255, 1.0),
            (255/255,   0/255,   0/255, 1.0),
            (200/255,   0/255,   0/255, 1.0),
        ]
    
        mmi_labels = [
            'Exposure IV–V', 'Exposure V–VI', 'Exposure VI–VII',
            'Exposure VII–VIII', 'Exposure VIII–IX', 'Exposure IX–X'
        ]
    
        # --------------------
        # Prepare data
        # --------------------
        df_sorted = df_exposure.sort_values(x_ticks).reset_index(drop=True)
        x = np.arange(len(df_sorted))
        heights = [df_sorted[col].astype(float).values for col in required_cols]
    
        # --------------------
        # Plot
        # --------------------
        fig, ax = plt.subplots(figsize=figsize)
        bottoms = np.zeros(len(df_sorted))
    
        for h, color, label in zip(heights, usgs_colors, mmi_labels):
            ax.bar(x, h, bottom=bottoms, color=color, label=label)
            bottoms += h
    
        # --------------------
        # Axes formatting
        # --------------------
        ax.set_xticks(x)
        ax.set_xticklabels(
            df_sorted[x_ticks].round(1),
            rotation=xrotation,
            ha='right',
            fontsize=tck_fs
        )
    
        xlabel_lookup = {
            'TaE_h': "Time After Event (hours)",
            'TaE_d': "Time After Event (days)",
            'shakemap_version': "ShakeMap Version"
        }
    
        ax.set_xlabel(
            xlabel or xlabel_lookup.get(x_ticks, x_ticks),
            fontsize=lbl_fs
        )
        ax.set_ylabel(
            ylabel or "Exposed Population",
            fontsize=lbl_fs
        )
    
        if show_title:
            ax.set_title(
                "Population Exposure by MMI Bins Over Time",
                fontsize=ttl_fs
            )
    
        ax.legend(
            title="MMI Bins",
            fontsize=lgd_fs,
            title_fontsize=lgd_fs,
            loc=legend_loc
        )
    
        if grid:
            ax.grid(True, **grid_kwargs)
    
        ax.tick_params(axis='y', labelsize=tck_fs)

        #ax.set_yscale("log")
    
        plt.tight_layout()
    
        # --------------------
        # Save
        # --------------------
        if output_path is not None:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "pager_data"
            out_dir.mkdir(parents=True, exist_ok=True)
            for fmt in save_formats:
                save_path = out_dir / f"{self.event_id}_pop_exposure.{fmt}"
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                logging.info(f"Saved exposure plot to {save_path}")
    
        # --------------------
        # Show / close
        # --------------------
        if show:
            plt.show()
        if close:
            plt.close(fig)
    
        return fig, ax

    



    def plot_alerts(
        self,
        version_list: list,
        alert_type: str = "fatality",
        x_ticks: str = "version",
        figsize: tuple = (24, 12),
        font_sizes: dict = None,
        xlabel: str = None,
        ylabel: str = None,
        xrotation: int = 45,
        legend_fontsize: int = None,
        legend_loc='best',
        title_fontsize: int = None,
        grid_kwargs: dict = None,
        output_path: str = None,
        save_formats: list = ["png", "pdf"],
        
        dpi: int = 300,
        show_title: bool = True
    ) -> (plt.Figure, plt.Axes):
        """
        Plot a single stacked bar chart of alert‐type probabilities for all versions,
        with customizable figure size, font sizes, axis labels, legend & title sizes,
        and grid style.  Automatically normalizes each bar so its probabilities sum to 1.
        """
        import os
        import numpy as np
        import matplotlib.colors as mcolors
        from collections import defaultdict
        from pathlib import Path
        import xml.etree.ElementTree as ET
        import logging
        import pandas as pd

        # --- defaults ---
        if font_sizes is None:
            font_sizes = {"labels":12, "ticks":10}
        lbl_fs = font_sizes.get("labels", 12)
        tck_fs = font_sizes.get("ticks", 10)
        lgd_fs = legend_fontsize or lbl_fs
        ttl_fs = title_fontsize or (lbl_fs + 2)
        if grid_kwargs is None:
            grid_kwargs = {"linestyle":"--","alpha":0.5}

        # --- x-axis setup ---
        versions_z = [v.zfill(3) for v in version_list]
        if x_ticks == "version":
            x_labels = versions_z
            xlabel_eff = xlabel or "Version"
        else:
            summary = self.get_dataframe().set_index('version')
            summary.index = summary.index.str.zfill(3)
            if x_ticks not in summary.columns:
                raise ValueError(f"x_ticks='{x_ticks}' not in summary columns.")
            x_vals = summary.loc[versions_z, x_ticks].astype(float)
            order = x_vals.argsort()
            versions_z = [versions_z[i] for i in order]
            x_labels = x_vals.iloc[order].round(1).astype(str).tolist()
            xlabel_map = {"TaE_h":"Time After Event (hours)",
                          "TaE_d":"Time After Event (days)"}
            xlabel_eff = xlabel or xlabel_map.get(x_ticks, x_ticks)
        ylabel_eff = ylabel or "Probability"

        # --- data gathering ---
        unit_map = {"fatality":"Fatalities","economic":"USD (Millions)"}
        legend_title = f"Alert Bin {unit_map.get(alert_type, '')}"
        all_probs, bin_labels, base_colors = [], None, None

        for version in version_list:
            fn = self._get_pager_filename(version)
            xml_path = os.path.join(self.pager_folder, self.event_id, fn)
            if not os.path.exists(xml_path):
                if bin_labels:
                    all_probs.append([0.0] * len(bin_labels))
                continue

            root = ET.parse(xml_path).getroot()
            alert = root.find(f"alert[@type='{alert_type}']")
            if alert is None:
                if bin_labels:
                    all_probs.append([0.0] * len(bin_labels))
                continue

            bins = []
            for b in alert.findall("bin"):
                mn, bx = b.get("min"), b.get("max")
                prob    = float(b.get("probability","0"))
                clr     = b.get("color","#888")
                bins.append((mn, bx, prob, clr))

            if bin_labels is None:
                bin_labels  = [f"{mn}–{bx}" for mn, bx, *_ in bins]
                base_colors = [clr for *_, clr in bins]

            all_probs.append([prob for *_, prob, _ in bins])

        # --- normalize each row to sum exactly to 1 ---
        for i, probs in enumerate(all_probs):
            total = sum(probs)
            if len(probs) and abs(total - 1.0) > 1e-6:
                # adjust the largest-probability bin by the difference
                idx_max = int(np.argmax(probs))
                all_probs[i][idx_max] += (1.0 - total)

        # --- shades ---
        shades = [None] * len(base_colors)
        groups = defaultdict(list)
        for idx, c in enumerate(base_colors):
            groups[c].append(idx)
        for base_c, idxs in groups.items():
            rgb   = mcolors.to_rgb(base_c)
            count = len(idxs)
            for j, idx in enumerate(sorted(idxs)):
                alpha = 0.5 * (1 - j/(count-1)) if count>1 else 0.0
                shades[idx] = tuple(rgb[k] + (1-rgb[k])*alpha for k in range(3))

        # --- DataFrame ---
        df = pd.DataFrame(all_probs, index=versions_z, columns=bin_labels)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(len(df))
        for col, color in zip(df.columns, shades):
            ax.bar(df.index, df[col], bottom=bottom,
                   label=col, color=color, edgecolor="k")
            bottom += df[col].values

        # --- Labels & Title & Legend & Grid ---
        ax.set_xlabel(xlabel_eff, fontsize=lbl_fs)
        ax.set_ylabel(ylabel_eff, fontsize=lbl_fs)
        if show_title:
            ax.set_title(f"{alert_type.title()} Alert Probabilities by {xlabel_eff}",
                         fontsize=ttl_fs)
        ax.legend(title=legend_title,
                  fontsize=lgd_fs,
                  title_fontsize=lgd_fs,
                  loc=legend_loc)
        ax.grid(True, **grid_kwargs)
        plt.xticks(rotation=xrotation, ha="right", fontsize=tck_fs)
        plt.yticks(fontsize=tck_fs)
        if x_ticks != "version":
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=tck_fs)
        plt.tight_layout()

        # --- Save ---
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "pager_data"
            out_dir.mkdir(parents=True, exist_ok=True)
            for fmt in save_formats:
                p = out_dir / f"{self.event_id}_{alert_type}_alerts.{fmt}"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                logging.info(f"Saved alerts plot to {p}")

        return fig, ax



 
    # --- END of plotting methods  ---++







    ##################################################
    #
    #
    #
    # ---Quantify Evolution methods for SHAKEtime ---
    #
    #
    #
    ##################################################

    
    def compute_threshold_exceedance_area(
        self,
        threshold: float,
        version_list: List[str],
        metric: str = "mmi"
    ) -> pd.DataFrame:
        """
        For each version in version_list, compute the total approximate area (km^2)
        where the metric exceeds threshold. Uses cached unified grid when available.
        """
        records = []
        deg2km = 111.0
        for v in version_list:
            ug = self.get_unified_grid([v], metric=metric, use_cache=True)
            col = f"{metric}_v{v}"
            if col not in ug.columns:
                continue
            mask = ug[col] > threshold
            if len(ug) < 2:
                area = 0.0
            else:
                dx = abs(np.diff(sorted(ug['lon'].unique()))).min() * deg2km
                dy = abs(np.diff(sorted(ug['lat'].unique()))).min() * deg2km
                lats = ug.loc[mask, 'lat'].values
                cell_areas = dx * dy * np.cos(np.radians(lats))
                area = float(np.nansum(cell_areas))
            records.append({'version': v, 'area_km2': area})
        return pd.DataFrame(records)
    
    
    def compute_global_diff_stats(
        self,
        v1: str = None,
        v2: str = None,
        version_list: List[str] = None,
        metric: str = "mmi",
        pairwise: bool = True
    ) -> dict:
        """
        Compute global difference metrics (MAE, RMSE, mean_diff).
    
        If version_list is provided, computes for multiple pairs:
          - pairwise=True: consecutive pairs
          - pairwise=False: all unique combinations
    
        Otherwise, v1 and v2 must be given to compute a single comparison.
    
        Returns
        -------
        For single v1/v2: a dict {'mae', 'rmse', 'mean_diff'}.
        For multiple: a dict mapping "v1_v2" → {'mae', 'rmse', 'mean_diff'}
        """
        # multiple pairs
        if version_list is not None:
            results = {}
            if pairwise:
                pairs = [(version_list[i], version_list[i+1]) for i in range(len(version_list)-1)]
            else:
                pairs = list(combinations(version_list, 2))
            for a, b in pairs:
                results[f"{a}_{b}"] = self.compute_global_diff_stats(
                    v1=a, v2=b, metric=metric, pairwise=False
                )
            return results
    
        # single pair
        if v1 is None or v2 is None:
            raise ValueError("Must provide either version_list or both v1 and v2")
    
        ug = self.get_unified_grid([v1, v2], metric=metric, use_cache=True)
        c1 = f"{metric}_v{v1}"
        c2 = f"{metric}_v{v2}"
        if c1 not in ug.columns or c2 not in ug.columns:
            return {'mae': np.nan, 'rmse': np.nan, 'mean_diff': np.nan}
        df = ug[[c1, c2]].dropna()
        if df.empty:
            return {'mae': np.nan, 'rmse': np.nan, 'mean_diff': np.nan}
        diff = df[c2] - df[c1]
        return {
            'mae': float(np.mean(np.abs(diff))),
            'rmse': float(np.sqrt(np.mean(diff**2))),
            'mean_diff': float(np.mean(diff))
        }


    def spatial_correlation(
        self,
        version_list: List[str],
        metric: str = "mmi",
        method: str = "pearson"
    ) -> pd.Series:
        """
        Compute Pearson or Spearman correlation for each consecutive pair.
        """
        import scipy.stats as stats
        corrs, versions = [], []
        for i in range(1, len(version_list)):
            v1, v2 = version_list[i-1], version_list[i]
            ug = self.get_unified_grid([v1, v2], metric=metric, use_cache=True)
            c1, c2 = f"{metric}_v{v1}", f"{metric}_v{v2}"
            if c1 not in ug.columns or c2 not in ug.columns:
                corr = np.nan
            else:
                df = ug[[c1, c2]].dropna()
                if df.empty:
                    corr = np.nan
                else:
                    corr = (stats.spearmanr(df[c1], df[c2])[0]
                            if method == 'spearman'
                            else stats.pearsonr(df[c1], df[c2])[0])
            versions.append(v2)
            corrs.append(corr)
        return pd.Series(corrs, index=versions, name='correlation')
    
    
    def variogram_analysis(
        self,
        version: str,
        metric: str = "mmi",
        n_lags: int = 10,
        max_dist: float = None,
        max_samples: int = 5000,
        random_state: int = 0,
        plot: bool = False
    ) -> dict:
        """
        Fit a semivariogram manually using SciPy on a subsample of the grid.
        """
        from scipy.spatial.distance import pdist
    
        ug = self.get_unified_grid([version], metric=metric, use_cache=True)
        coords = ug[['lon', 'lat']].values
        values = ug[f"{metric}_v{version}"].values
    
        # subsample
        n_pts = coords.shape[0]
        if n_pts > max_samples:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(n_pts, size=max_samples, replace=False)
            coords = coords[idx]
            values = values[idx]
    
        # distances & semivariance
        dists = pdist(coords)
        diffs = pdist(values.reshape(-1, 1))
        semivar = 0.5 * (diffs ** 2)
    
        # binning
        max_d = max_dist if max_dist is not None else dists.max()
        bin_edges = np.linspace(0, max_d, n_lags + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        gammas = []
        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (dists >= low) & (dists < high)
            gammas.append(np.nanmean(semivar[mask]) if np.any(mask) else np.nan)
        gammas = np.array(gammas)
    
        # parameters
        nugget = float(gammas[0]) if gammas.size else np.nan
        sill = float(np.nanmean(gammas)) if np.any(~np.isnan(gammas)) else np.nan
        try:
            idx95 = np.where(gammas >= 0.95 * sill)[0][0]
            range_dist = float(bin_centers[idx95])
        except Exception:
            range_dist = float(max_d)
    
        params = {'nugget': nugget, 'sill': sill, 'range': range_dist}
    
        if plot:
            fig, ax = plt.subplots()
            ax.plot(bin_centers, gammas, marker='o', linestyle='-')
            ax.axhline(sill, color='red', linestyle='--', label='Sill')
            ax.axhline(nugget, color='green', linestyle='--', label='Nugget')
            ax.axvline(range_dist, color='blue', linestyle='--', label='Range')
            ax.set_xlabel('Distance')
            ax.set_ylabel('Semivariance')
            ax.set_title(f'Experimental Semivariogram: {metric.upper()} v{version}')
            ax.legend()
            plt.tight_layout()
            return params, fig, ax
    
        return params
    
    
    def area_of_uncertainty_change(
        self,
        version_list: List[str],
        metric: str = "mmi",
        percentile: float = 90.0
    ) -> pd.DataFrame:
        """
        Area where std-dev exceeds its percentile threshold.
        """
        records = []
        deg2km = 111.0
        std_map = {"mmi":"STDMMI","pga":"STDPGA","pgv":"STDPGV"}[metric]
        for v in version_list:
            fn = self._get_shakemap_filename(v).replace('grid','uncertainty')
            xml = os.path.join(self.shakemap_folder, self.event_id, fn)
            parser = USGSParser(parser_type='shakemap_xml', xml_file=xml, imt=metric, value_type='std')
            df = parser.get_dataframe()
            if std_map not in df.columns:
                area = np.nan
            else:
                thr = np.nanpercentile(df[std_map], percentile)
                mask = df[std_map] > thr
                dx = abs(np.diff(sorted(df['LON'].unique()))).min() * deg2km
                dy = abs(np.diff(sorted(df['LAT'].unique()))).min() * deg2km
                lats = df.loc[mask,'LAT'].values
                area = float(np.nansum(dx * dy * np.cos(np.radians(lats))))
            records.append({'version': v, f'unc_area_pct{int(percentile)}': area})
        return pd.DataFrame(records)
    
    
    def export_to_geotiff(
        self,
        version: str,
        metric: str = "mmi",
        output_path: str = None
    ):
        """
        Export a single version's grid to GeoTIFF for GIS use. Uses cached grid when available.
        Requires rasterio.
        """
        import rasterio
        from rasterio.transform import from_origin
        ug = self.get_unified_grid([version], metric=metric, use_cache=True)
        col = f"{metric}_v{version}"
        arr = ug[col].values.reshape(int(np.sqrt(len(ug))), -1)
        lon_min, lon_max = ug['lon'].min(), ug['lon'].max()
        lat_min, lat_max = ug['lat'].min(), ug['lat'].max()
        nrows, ncols = arr.shape
        dx = (lon_max - lon_min) / (ncols - 1)
        dy = (lat_max - lat_min) / (nrows - 1)
        transform = from_origin(lon_min, lat_max, dx, dy)
        out = output_path or os.getcwd()
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"{self.event_id}_{metric}_v{version}.tif")
        with rasterio.open(
            path, 'w', driver='GTiff', height=nrows, width=ncols,
            count=1, dtype=arr.dtype, crs="EPSG:4326", transform=transform
        ) as dst:
            dst.write(arr, 1)
        logging.info(f"GeoTIFF exported to {path}")
        return path
    
    
    def bootstrap_uncertainty(
        self,
        version_list: List[str],
        metric: str = "mmi",
        n_iter: int = 1000,
        alpha: float = 0.05
    ) -> dict:
        """
        Bootstrap-resample grid points for each version to compute
        confidence intervals on: mean, std, min and max of the metric.
        Uses cached unified grid when available.
        """
        results = {}
        for v in version_list:
            ug = self.get_unified_grid([v], metric=metric, use_cache=True)
            data = ug[f"{metric}_v{v}"].dropna().values
    
            boot_means, boot_stds = [], []
            boot_mins, boot_maxs = [], []
    
            for _ in range(n_iter):
                samp = np.random.choice(data, size=len(data), replace=True)
                boot_means.append(samp.mean())
                boot_stds .append(samp.std(ddof=1))
                boot_mins .append(samp.min())
                boot_maxs .append(samp.max())
    
            # two‐sided CIs for each statistic
            m_lo,  m_hi  = np.percentile(boot_means, [100*alpha/2, 100*(1-alpha/2)])
            s_lo,  s_hi  = np.percentile(boot_stds,  [100*alpha/2, 100*(1-alpha/2)])
            min_lo, min_hi = np.percentile(boot_mins, [100*alpha/2, 100*(1-alpha/2)])
            max_lo, max_hi = np.percentile(boot_maxs, [100*alpha/2, 100*(1-alpha/2)])
    
            results[v] = {
                'bootstrap_mean_lo': m_lo,    'bootstrap_mean_hi': m_hi,
                'bootstrap_std_lo':  s_lo,    'bootstrap_std_hi':  s_hi,
                'bootstrap_min_lo':  min_lo,  'bootstrap_min_hi':  min_hi,
                'bootstrap_max_lo':  max_lo,  'bootstrap_max_hi':  max_hi
            }
    
        return results
    
    
    def quantify_evolution(
        self,
        version_list: List[str],
        metric: str = "mmi",
        thresholds: List[float] = [7.0],
        uncertainty_percentile: float = 90.0,
        bootstrap_iters: int = 100,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Run all diagnostics across ShakeMap versions, producing a single summary table.
    
        Caches the result in self._evolution_df if use_cache=True.
        """
        if use_cache and hasattr(self, '_evolution_df') and self._evolution_df is not None:
            return self._evolution_df
        records = []
        # threshold exceedance
        for thr in thresholds:
            df_area = self.compute_threshold_exceedance_area(thr, version_list, metric)
            for row in df_area.itertuples(index=False):
                records.append({'version': row.version, f'area_exceed_{thr}': row.area_km2})
        # diffs & correlation
        corr_series = self.spatial_correlation(version_list, metric)
        diff_stats = self.compute_global_diff_stats(version_list=version_list, metric=metric, pairwise=True)
        for i in range(1, len(version_list)):
            v1, v2 = version_list[i-1], version_list[i]
            rec = {'version': v2}
            rec.update(diff_stats.get(f"{v1}_{v2}", {}))
            rec['spatial_corr'] = corr_series.get(v2, np.nan)
            records.append(rec)
        # variogram
        for v in version_list:
            params = self.variogram_analysis(v, metric)
            rec = {'version': v, **params}
            records.append(rec)
        # uncertainty-area change
        df_unc = self.area_of_uncertainty_change(version_list, metric, uncertainty_percentile)
        for row in df_unc.itertuples(index=False):
            records.append({'version': row.version, f'unc_area_pct{int(uncertainty_percentile)}': row[1]})
        # bootstrap CIs
        boot = self.bootstrap_uncertainty(version_list, metric, bootstrap_iters)
        for v, ci in boot.items():
            records.append({
                'version': v,
                'bootstrap_mean_lo': ci['bootstrap_mean_lo'],
                'bootstrap_mean_hi': ci['bootstrap_mean_hi'],
                'bootstrap_std_lo': ci['bootstrap_std_lo'],
                'bootstrap_std_hi': ci['bootstrap_std_hi'],
                'bootstrap_min_lo': ci['bootstrap_min_lo'],
                'bootstrap_min_hi': ci['bootstrap_min_hi'],
                'bootstrap_max_lo': ci['bootstrap_max_lo'],
                'bootstrap_max_hi': ci['bootstrap_max_hi']


            })
    
        df = pd.DataFrame(records).groupby('version').first()
        self._evolution_df = df
        return df
    
    
    def export_evolution(
        self,
        output_dir: str,
        file_type: str = 'csv'
    ):
        """
        Export the cached evolution DataFrame to a file in the specified format.
        """
        if not hasattr(self, '_evolution_df') or self._evolution_df is None:
            raise RuntimeError('No evolution data to export; run quantify_evolution first.')
        os.makedirs(output_dir, exist_ok=True)
        fname = f'evolution_export.{file_type.lower()}'
        fpath = os.path.join(output_dir, fname)
        ft = file_type.lower()
        df = self._evolution_df
        if ft == 'csv':
            df.to_csv(fpath, index=True)
        elif ft == 'txt':
            df.to_csv(fpath, index=True, sep='\t')
        elif ft == 'xlsx':
            df.to_excel(fpath, index=True)
        elif ft == 'json':
            df.to_json(fpath, orient='records', lines=True)
        elif ft == 'feather':
            df.to_feather(fpath)
        elif ft == 'parquet':
            df.to_parquet(fpath)
        elif ft == 'pickle':
            df.to_pickle(fpath)
        else:
            raise ValueError(f"Unsupported export file type: {file_type}")
        logging.info(f"Evolution data exported to {fpath!r} in format '{ft}'")
    
    
    def plot_evolution(
        self,
        version_list: List[str],
        metric: str = "mmi",
        thresholds: List[float] = [3.0],
        uncertainty_percentile: float = 90.0,
        bootstrap_iters: int = 100,
        plot_types: List[str] = None,
        figsize: tuple = (12, 8),
        font_sizes: dict = None,
        grid_kwargs: dict = None,
        output_path: str = None,
        save_formats: List[str] = ["png", "pdf"],
        dpi: int = 300
    ):
        """
        Plot selected evolution diagnostics over ShakeMap versions.
    
        Raises ValueError if an invalid plot_type is requested.
        """
        evo = self.quantify_evolution(
            version_list,
            metric=metric,
            thresholds=thresholds,
            uncertainty_percentile=uncertainty_percentile,
            bootstrap_iters=bootstrap_iters
        )
        if font_sizes is None:
            font_sizes = {"title": 14, "labels": 12, "ticks": 10}
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.5}
        all_cols = list(evo.columns)
        cols_to_plot = plot_types or all_cols
        for col in cols_to_plot:
            if col not in evo.columns:
                raise ValueError(f"Cannot plot '{col}': not a computed diagnostic.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(evo.index, evo[col], marker='o')
            ax.set_title(f"Evolution of {col}", fontsize=font_sizes["title"])
            ax.set_xlabel("Version", fontsize=font_sizes["labels"])
            ax.set_ylabel(col, fontsize=font_sizes["labels"])
            ax.grid(True, **grid_kwargs)
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=font_sizes["ticks"])
            plt.tight_layout()
            if output_path:
                out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "EvolutionPlots"
                out_dir.mkdir(parents=True, exist_ok=True)
                for fmt in save_formats:
                    path = out_dir / f"{self.event_id}_evolution_{col}.{fmt}"
                    fig.savefig(path, dpi=dpi, bbox_inches='tight')
                    logging.info(f"Saved evolution plot to {path}")
        return evo


    ##################################################
    #
    #
    #
    #
    # --- Analyze Auxiliary Influences methods for SHAKEtime ---
    #
    #
    #
    ##################################################

        
    def compute_station_covariates(
        self,
        version: str,
        station_folder: str,
        metric: str = "pga",
        radius_km: float = 50.0
    ) -> dict:
        """
        Compute station-network covariates for one ShakeMap version.
    
        Parameters
        ----------
        version : str
            ShakeMap version identifier (e.g. "020").
        station_folder : str
            Base directory containing per-version stationlist JSON.
        metric : {'pga','pgv'}, default 'pga'
            Which observed ground motion to use.
        radius_km : float, default 50.0
            Radius around each grid cell (in km) to count stations.
    
        Returns
        -------
        dict
            {
                'station_count': int,
                'mean_pgm': float (NaN if no stations),
                'std_pgm': float (NaN if no stations),
                'station_density': pd.Series,  # stations per km² at each grid cell
            }
        """
        from sklearn.neighbors import BallTree
    
        # ensure unified grid is ready
        ug = self.get_unified_grid([version], metric="mmi", use_cache=True)
        density_default = pd.Series(0.0, index=ug.index, name='station_density')
        defaults = {
            'station_count': 0,
            'mean_pgm': np.nan,
            'std_pgm': np.nan,
            'station_density': density_default
        }
    
        # locate file
        fp = Path(station_folder) / self.event_id / f"{self.event_id}_us_{version}_stationlist.json"
        if not fp.exists():
            logging.warning(f"[Station] no file for v{version}: {fp}")
            return defaults
    
        # parse
        try:
            parser = USGSParser(parser_type="instrumented_data", json_file=str(fp))
            df_sta = parser.get_dataframe(value_type=metric)[['latitude','longitude',metric]].dropna()
        except Exception as e:
            logging.error(f"[Station] failed to parse v{version}: {e}")
            return defaults
    
        if df_sta.empty:
            logging.info(f"[Station] parsed but empty for v{version}")
            return defaults
    
        # compute scalars
        station_count = len(df_sta)
        mean_pgm = float(df_sta[metric].mean())
        std_pgm  = float(df_sta[metric].std())
    
        # compute density via haversine BallTree
        pts_rad = np.deg2rad(ug[['lat','lon']].values)
        sta_rad = np.deg2rad(df_sta[['latitude','longitude']].values)
        tree = BallTree(sta_rad, metric="haversine")
        r = radius_km / 6371.0  # km → radians
        counts = tree.query_radius(pts_rad, r=r, count_only=True)
        cell_area = np.pi * radius_km**2
        density = pd.Series(counts / cell_area, index=ug.index, name='station_density')
    
        return {
            'station_count': station_count,
            'mean_pgm': mean_pgm,
            'std_pgm': std_pgm,
            'station_density': density
        }
    
    
    def compute_dyfi_covariates(
        self,
        version: str,
        dyfi_folder: str,
        metric: str = "mmi"
    ) -> dict:
        """
        For one version, compute:
          • dyfi_count
          • dyfi_footprint_km2   (convex-hull area of reports)
          • dyfi_mean_resid, dyfi_std_resid
        Residual = (reported DYFI MMI – ShakeMap MMI_at_station)
    
        Returns a dict with zeros/NaNs if no usable DYFI data.
        """
        from shapely.geometry import MultiPoint
        from scipy.interpolate import griddata
        import logging
    
        defaults = {
            'dyfi_count': 0,
            'dyfi_footprint_km2': 0.0,
            'dyfi_mean_resid': np.nan,
            'dyfi_std_resid': np.nan
        }
    
        # 1) load & cache DYFI list
        fp = Path(dyfi_folder) / self.event_id / f"{self.event_id}_us_{version}_stationlist.json"
        if not fp.exists():
            logging.warning(f"[DYFI] no file for v{version}: {fp}")
            return defaults
    
        try:
            parser = USGSParser(parser_type="instrumented_data", json_file=str(fp))
            df_raw = parser.get_dataframe(value_type=metric)
        except Exception as e:
            logging.error(f"[DYFI] failed to parse v{version}: {e}")
            return defaults
    
        if df_raw.empty:
            logging.info(f"[DYFI] parsed but empty for v{version}")
            return defaults
    
        # map columns case-insensitively
        col_map = {c.lower(): c for c in df_raw.columns}
        lat_col = col_map.get('latitude') or col_map.get('lat')
        lon_col = col_map.get('longitude') or col_map.get('lon')
        int_col = col_map.get('intensity') or col_map.get(metric.lower()) or col_map.get('mmi')
        if not (lat_col and lon_col and int_col):
            logging.warning(f"[DYFI] missing lat/lon/int cols in v{version}")
            return defaults
    
        df = df_raw[[lat_col, lon_col, int_col]].copy()
        df.columns = ['latitude', 'longitude', 'intensity']
    
        # *** Force intensity to numeric, drop invalid ***
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude', 'intensity'])
        if df.empty:
            logging.info(f"[DYFI] after coercion & dropna empty for v{version}")
            return defaults
    
        # 2) footprint area via convex hull
        pts = MultiPoint(list(zip(df.longitude, df.latitude)))
        hull = pts.convex_hull
        deg2km = 111.0
        lat0 = df.latitude.mean()
        area_km2 = hull.area * (deg2km**2) * np.cos(np.deg2rad(lat0))
    
        # 3) interpolate ShakeMap metric at those points
        ug = self.get_unified_grid([version], metric=metric, use_cache=True)
        coords = ug[['lon','lat']].values
        vals   = ug[f"{metric}_v{version}"].values
        sm_vals = griddata(coords,
                           vals,
                           df[['longitude','latitude']].values,
                           method='linear')
    
        resid = df['intensity'].values - sm_vals
        mean_resid = float(np.nanmean(resid))
        std_resid  = float(np.nanstd(resid))
    
        return {
            'dyfi_count': len(df),
            'dyfi_footprint_km2': area_km2,
            'dyfi_mean_resid': mean_resid,
            'dyfi_std_resid': std_resid
        }
    
        
    def compute_rupture_geometry(
        self,
        version: str,
        rupture_folder: str
    ) -> dict:
        """
        Compute rupture‐trace geometry covariates for one version.
    
        Parameters
        ----------
        version : str
            ShakeMap version identifier.
        rupture_folder : str
            Base directory containing per-version rupture JSON.
    
        Returns
        -------
        dict
            {
                'trace_length_km': float,
                'segment_count': int
            }
        """
        from shapely.geometry import LineString
    
        defaults = {'trace_length_km': 0.0, 'segment_count': 0}
        fp = Path(rupture_folder) / self.event_id / f"{self.event_id}_us_{version}_rupture.json"
        if not fp.exists():
            logging.warning(f"[Rupture] no file for v{version}: {fp}")
            return defaults
    
        try:
            rup = USGSParser(parser_type='rupture_json', mode='parse', rupture_json=str(fp))
            xs, ys = rup.get_rupture_xy()
            if not xs or not ys:
                logging.info(f"[Rupture] parsed but no coords for v{version}")
                return defaults
            line = LineString(list(zip(xs, ys)))
            deg2km = 111.0
            length_km = float(line.length * deg2km)
            seg_count = max(len(xs)-1, 0)
            return {'trace_length_km': length_km, 'segment_count': seg_count}
        except Exception as e:
            logging.error(f"[Rupture] failed to parse v{version}: {e}")
            return defaults


    
    def analyze_auxiliary_influences(
        self,
        version_list,
        station_folder=None,
        dyfi_folder=None,
        rupture_folder=None,
        metric="mmi",
        radius_km=50.0,
        extra=None,
        *,
        cache_folder=None,
        use_cache=True,
        file_type="csv",
        txt_sep="\t",
        **extra_kwargs
    ):
        """
        Build or load your per-version covariates (station, DYFI, rupture),
        optionally run extra analyses, then merge with the map-change diagnostics.
    
        If use_cache=True, we first try to import from cache_folder (if given)
        or ./export/SHAKEtime/{event_id}.  If that works and no `extra` analyses
        are requested, we return immediately.  If `extra` is non-empty, we append
        just those analyses on top of the cached aux table, rebuild merged/corr/
        regs, re-export into the same cache, and return.
        """
        from pathlib import Path
        import logging, numpy as np, pandas as pd, statsmodels.api as sm
    
        # stash for the Moran routines
        self._last_station_folder = station_folder
        self._last_dyfi_folder    = dyfi_folder
        self._last_radius_km      = radius_km
        self._last_metric         = metric
    
        def _is_complete(res):
            return all(res.get(k) is not None for k in ("aux","diag","merged","correlation"))
    
        # 1) try to load cache
        if use_cache:
            for load_dir in ([cache_folder] if cache_folder else []) + \
                            [Path("export")/"SHAKEtime"/self.event_id]:
                if not load_dir:
                    continue
                logging.info(f"[aux] trying cache at {load_dir!r}")
                try:
                    res = self.import_auxiliary_results(str(load_dir),
                                                        file_type=file_type,
                                                        txt_sep=txt_sep)
                    if _is_complete(res):
                        logging.info(f"[aux] loaded all tables from {load_dir!r}")
                        # if no extras, just return it
                        if not extra:
                            return res
    
                        # else append just the extras
                        logging.info("[aux] appending extra analyses to cached aux")
                        aux_df  = res["aux"].copy()
                        diag_df = res["diag"]
                        new_aux = self.add_auxiliary(
                            aux_df,
                            version_list,
                            analyses=extra,
                            metric=metric,
                            **extra_kwargs
                        )
                        # rebuild merged
                        merged = ( new_aux.join(diag_df, how="inner")
                                         .apply(pd.to_numeric, errors="coerce")
                                         .replace([np.inf, -np.inf], np.nan) )
                        # re-inject any time columns
                        for t in ("TaE_h","TaE_d"):
                            if t in res["merged"].columns:
                                merged[t] = res["merged"][t]
                        # recompute corr & regs
                        corr = merged.corr(method="pearson")
                        regs = {}
                        aux_cols = list(new_aux.columns) + [t for t in ("TaE_h","TaE_d") if t in merged]
                        for diag_col in diag_df.columns:
                            tmp = merged[aux_cols + [diag_col]].dropna()
                            if len(tmp) >= 2:
                                X = sm.add_constant(tmp[aux_cols])
                                regs[diag_col] = sm.OLS(tmp[diag_col], X).fit()
                            else:
                                regs[diag_col] = None
    
                        out = {
                            "aux":         new_aux,
                            "diag":        diag_df,
                            "merged":      merged,
                            "correlation": corr,
                            "regressions": regs
                        }
                        logging.info(f"[aux] re-exporting updated cache to {load_dir!r}")
                        try:
                            self.export_auxiliary_results(
                                out,
                                output_dir=str(load_dir),
                                file_type=file_type,
                                txt_sep=txt_sep
                            )
                        except Exception as e:
                            logging.warning(f"[aux] failed re-export: {e!r}")
                        return out
    
                except Exception as e:
                    logging.warning(f"[aux] cache import from {load_dir!r} failed: {e!r}")
    
            logging.info("[aux] cache import unsuccessful → computing from scratch")
    
        # 2) compute base covariates
        aux_recs = []
        for v in version_list:
            rec = {"version": v}
            if station_folder:
                st = self.compute_station_covariates(v,
                                                     station_folder,
                                                     metric="pga",
                                                     radius_km=radius_km)
                rec.update({
                    "station_count":        st["station_count"],
                    "mean_pgm":             st["mean_pgm"],
                    "std_pgm":              st["std_pgm"],
                    "station_density_mean": float(st["station_density"].mean())
                })
            if dyfi_folder:
                dy = self.compute_dyfi_covariates(v,
                                                  dyfi_folder,
                                                  metric=metric)
                rec.update({
                    "dyfi_count":         dy["dyfi_count"],
                    "dyfi_footprint_km2": dy["dyfi_footprint_km2"],
                    "dyfi_mean_resid":    dy["dyfi_mean_resid"],
                    "dyfi_std_resid":     dy["dyfi_std_resid"]
                })
            if rupture_folder:
                rp = self.compute_rupture_geometry(v, rupture_folder)
                rec.update(rp)
            aux_recs.append(rec)
    
        df_aux = pd.DataFrame(aux_recs).set_index("version")
        self._last_aux      = df_aux.copy()
        self._last_versions = list(version_list)
    
        # 3) extras
        if extra:
            df_aux = self.add_auxiliary(df_aux,
                                        version_list,
                                        analyses=extra,
                                        metric=metric,
                                        **extra_kwargs)
            self._last_aux = df_aux.copy()
    
        # 4) map-change diagnostics
        df_diag = self.quantify_evolution(
            version_list,
            metric=metric,
            thresholds=extra_kwargs.get("thresholds", [3.0]),
            uncertainty_percentile=extra_kwargs.get("uncertainty_percentile", 90.0),
            bootstrap_iters=extra_kwargs.get("bootstrap_iters", 100)
        )
    
        # 5) merge
        df_merged = ( df_aux.join(df_diag, how="inner")
                             .apply(pd.to_numeric, errors="coerce")
                             .replace([np.inf, -np.inf], np.nan) )
    
        # 6) time‐since‐event
        if hasattr(self, "summary_df") and not self.summary_df.empty:
            summary = self.summary_df.set_index("version")
            summary.index = summary.index.astype(str)
            for t in ("TaE_h","TaE_d"):
                if t in summary.columns:
                    df_merged[t] = summary[t]
    
        # 7) corr & regs
        corr = df_merged.corr(method="pearson")
        regs = {}
        aux_cols = list(df_aux.columns) + [t for t in ("TaE_h","TaE_d") if t in df_merged]
        for diag_col in df_diag.columns:
            tmp = df_merged[aux_cols + [diag_col]].dropna()
            if len(tmp) >= 2:
                X = sm.add_constant(tmp[aux_cols])
                regs[diag_col] = sm.OLS(tmp[diag_col], X).fit()
            else:
                regs[diag_col] = None
    
        result = {
            "aux":         df_aux,
            "diag":        df_diag,
            "merged":      df_merged,
            "correlation": corr,
            "regressions": regs
        }
    
        # 8) export to cache
        if use_cache:
            out_folder = Path(cache_folder) if cache_folder else Path("export")/"SHAKEtime"/self.event_id
            logging.info(f"[aux] exporting results to cache at {out_folder!r}")
            try:
                self.export_auxiliary_results(
                    result,
                    output_dir=str(out_folder),
                    file_type=file_type,
                    txt_sep=txt_sep
                )
            except Exception as e:
                logging.warning(f"[aux] failed export: {e!r}")
    
        return result
    
        



    def plot_auxiliary_results(
        self,
        res: dict,
        kind: str = "all",
        figsize: tuple = (12, 8),
        output_path: str = None,
        save_formats: list[str] = ["png","pdf"],
        dpi: int = 300,
        **kwargs
    ):
        """
        Plot auxiliary‐vs‐diagnostic results from analyze_auxiliary_influences().

        Polished with grid lines, rotated ticks, and tight layouts.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import logging
        from pathlib import Path

        df   = res['merged']
        corr = res['correlation']
        regs = res['regressions']
        versions = df.index.astype(str).tolist()

        # use a light style
        #plt.style.use("seaborn-whitegrid")

        def _make_plot(k: str):
            if k == 'time_series':
                fig, (ax1, ax2) = plt.subplots(2,1,figsize=figsize, sharex=True)
                # top: station/dyfi/rupture counts
                ax1.set_yscale('log')
                for col in ['station_count','dyfi_count','trace_length_km']:
                    if col in df:
                        ax1.plot(versions, df[col], 'o-', markersize=6, label=col, alpha=0.8)
                ax1.set_ylabel("Count/Length (log)")
                ax1.legend()
                ax1.grid(True, linestyle="--", alpha=0.5)

                # bottom: area diagnostics
                for col in ['unc_area_pct90','area_exceed_7.0','area_exceed_6.0','area_exceed_8.0']:
                    if col in df:
                        ax2.plot(versions, df[col], 's--', markersize=6, label=col, alpha=0.8)
                ax2.set_ylabel("Area (km²)")
                ax2.set_xlabel("Version")
                ax2.legend()
                ax2.grid(True, linestyle="--", alpha=0.5)
                plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                return fig, (ax1, ax2)

            if k == 'scatter':
                fig, (ax1, ax2) = plt.subplots(1,2,figsize=figsize)

                # Unc vs trace_length_km
                if 'trace_length_km' in df and 'unc_area_pct90' in df:
                    x,y = df['trace_length_km'], df['unc_area_pct90']
                    ax1.scatter(x,y, s=50, alpha=0.7)
                    ax1.set(xlabel="trace_length_km", ylabel="unc_area_pct90",
                            title="Unc90 vs trace_length_km")
                    if regs.get('unc_area_pct90'):
                        m = regs['unc_area_pct90']
                        xs = np.linspace(x.min(), x.max(), 100)
                        ys = m.params['const'] + m.params['trace_length_km']*xs
                        ax1.plot(xs, ys, 'r--', lw=2)
                    ax1.grid(True, linestyle="--", alpha=0.5)

                # Area exceedance vs station_count
                if 'station_count' in df and 'area_exceed_7.0' in df:
                    x,y = df['station_count'], df['area_exceed_7.0']
                    ax2.scatter(x,y, s=50, alpha=0.7)
                    ax2.set(xlabel="station_count", ylabel="area_exceed_7.0",
                            title="Area > MMI7 vs station_count")
                    if regs.get('area_exceed_7.0'):
                        m = regs['area_exceed_7.0']
                        xs = np.linspace(x.min(), x.max(), 100)
                        ys = m.params['const'] + m.params['station_count']*xs
                        ax2.plot(xs, ys, 'r--', lw=2)
                    ax2.grid(True, linestyle="--", alpha=0.5)

                plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
                plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                return fig, (ax1, ax2)

            if k == 'heatmap':
                fig, ax = plt.subplots(figsize=figsize)
                sns.heatmap(corr, cmap='coolwarm', center=0, square=True,
                            cbar_kws={"shrink":.8}, ax=ax)
                ax.set_title("Correlation Matrix", pad=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                return fig, ax

            if k == 'annotated_heatmap':
                fig, ax = plt.subplots(figsize=figsize)
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                            cmap='vlag', center=0, square=True,
                            cbar_kws={"shrink":.8}, ax=ax)
                ax.set_title("Masked & Annotated Correlation Matrix", pad=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                return fig, ax

            if k == 'clustered_heatmap':
                corr_clean = (
                    corr.dropna(axis=0, how='any')
                        .dropna(axis=1, how='any')
                )
                if corr_clean.shape[0] < 2 or corr_clean.shape[1] < 2:
                    logging.warning("[clustered_heatmap] not enough complete variables to cluster")
                    return None
                cg = sns.clustermap(
                    corr_clean, cmap="vlag", center=0,
                    method="average", metric="correlation",
                    annot=True, fmt=".2f", figsize=figsize,
                    cbar_kws={"shrink": .8}
                )
                cg.ax_heatmap.set_title("Clustered Correlation Matrix", pad=12)
                cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
                cg.fig.tight_layout()
                return cg.fig, cg.ax_heatmap

            if k == 'bootstrap_ci':
                fig, ax = plt.subplots(figsize=figsize)
                lo = df['bootstrap_mean_lo']; hi = df['bootstrap_mean_hi']
                mid = 0.5*(lo+hi)
                ax.errorbar(versions, mid, yerr=[mid-lo, hi-mid],
                            fmt='o', capsize=5, alpha=0.8)
                ax.set(xlabel="Version", ylabel="Mean ± CI",
                       title="Bootstrap CI on Mean")
                ax.grid(True, linestyle="--", alpha=0.5)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                return fig, ax

            if k == 'dyfi_resid':
                fig, ax = plt.subplots(figsize=figsize)
                means = df['dyfi_mean_resid']; stds = df['dyfi_std_resid']
                ax.errorbar(versions, means, yerr=stds, fmt='s', capsize=5, alpha=0.8)
                ax.set(xlabel="Version", ylabel="DYFI Residual (MMI)",
                       title="DYFI Residuals ± σ")
                ax.grid(True, linestyle="--", alpha=0.5)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                return fig, ax

            if k == 'semivariogram':
                figs = []
                for v in versions:
                    try:
                        params, fig_v, ax_v = self.variogram_analysis(
                            v, plot=True,
                            **{kk:kwargs[kk] for kk in ('n_lags','max_dist') if kk in kwargs}
                        )
                        ax_v.set_title(f"Semivariogram v{v}")
                        ax_v.grid(True, linestyle="--", alpha=0.5)
                        fig_v.tight_layout()
                        figs.append((v, fig_v, ax_v))
                    except Exception as e:
                        logging.warning(f"variogram_analysis failed for v{v}: {e}")
                return figs

            raise ValueError(f"Unknown kind '{k}'")

        all_kinds = [
            'time_series','scatter','heatmap','annotated_heatmap',
            'clustered_heatmap','bootstrap_ci','dyfi_resid','semivariogram'
        ]
        to_run = all_kinds if kind=='all' else [kind]

        results = {}
        for k in to_run:
            out = _make_plot(k)
            results[k] = out

            # ── save if requested ──
            if output_path and k != 'semivariogram' and out is not None:
                fig = out[0] if isinstance(out, tuple) else out
                save_dir = Path(output_path) / "SHAKEtime" / self.event_id / "auxiliary_influences"
                save_dir.mkdir(parents=True, exist_ok=True)
                for fmt in save_formats:
                    p = save_dir / f"{self.event_id}_{k}.{fmt}"
                    fig.savefig(p, dpi=dpi, bbox_inches='tight')
                    logging.info(f"Saved {k} plot to {p!r}")

        return results if kind=='all' else results[kind]




    def export_auxiliary_results(
        self,
        res: dict,
        output_dir: str,
        file_type: str = 'csv',
        txt_sep: str = '\t'
    ):
        """
        Export the auxiliary‐analysis results (aux, diag, merged, correlation)
        produced by analyze_auxiliary_influences().

        Always exports the four tables in the requested format AND
        always exports the full regressions dict as a pickle.
        """
        import pickle

        # build base output dir: ./export/SHAKEtime/<event_id>/
        out_base = Path(output_dir) / "SHAKEtime" / self.event_id
        out_base.mkdir(parents=True, exist_ok=True)

        ft = file_type.lower()
        dfs = {
            'aux':         res.get('aux'),
            'diag':        res.get('diag'),
            'merged':      res.get('merged'),
            'correlation': res.get('correlation')
        }

        # 1) export each DataFrame in the chosen format
        for key, df in dfs.items():
            if df is None:
                logging.warning(f"No DataFrame found for '{key}', skipping export.")
                continue

            fname = f"SHAKEtime_aux_{key}_{self.event_id}.{ft}"
            fpath = out_base / fname

            if ft == 'csv':
                df.to_csv(fpath, index=True)
            elif ft == 'txt':
                df.to_csv(fpath, index=True, sep=txt_sep)
            elif ft == 'xlsx':
                df.to_excel(fpath, index=True)
            elif ft == 'json':
                df.to_json(fpath, orient='records', lines=True)
            elif ft == 'feather':
                df.to_feather(fpath)
            elif ft == 'parquet':
                df.to_parquet(fpath, index=True)
            elif ft == 'pickle':
                df.to_pickle(fpath)
            else:
                raise ValueError(f"Unsupported export file type: {file_type!r}")

            logging.info(f"Exported '{key}' → {fpath}")

        # 2) always export regressions dict as pickle
        regs = res.get('regressions')
        if regs is not None:
            reg_fname = f"SHAKEtime_aux_regressions_{self.event_id}.pkl"
            reg_path  = out_base / reg_fname
            with open(reg_path, 'wb') as f:
                pickle.dump(regs, f)
            logging.info(f"Exported 'regressions' → {reg_path}")
        else:
            logging.warning("No regressions to export.")


    def import_auxiliary_results(
        self,
        input_dir: str,
        file_type: str = 'csv',
        txt_sep: str = '\t'
    ) -> dict:
        """
        Read back the four main tables AND always attempt to load
        the regressions pickle, regardless of file_type.
        """
        import pickle

        ft = file_type.lower()
        keys = ['aux', 'diag', 'merged', 'correlation']
        results = {}

        # 1) load the four tables
        for key in keys:
            fname = f"SHAKEtime_aux_{key}_{self.event_id}.{ft}"
            fpath = Path(input_dir) / fname
            if not fpath.exists():
                logging.warning(f"File not found: {fpath!r}, skipping '{key}'")
                results[key] = None
                continue

            try:
                if ft == 'csv':
                    df = pd.read_csv(fpath, index_col=0)
                elif ft == 'txt':
                    df = pd.read_csv(fpath, index_col=0, sep=txt_sep)
                elif ft == 'xlsx':
                    df = pd.read_excel(fpath, index_col=0)
                elif ft == 'json':
                    df = pd.read_json(fpath, orient='records', lines=True)
                    if 'version' in df.columns:
                        df.set_index('version', inplace=True)
                elif ft == 'feather':
                    df = pd.read_feather(fpath).set_index('version', drop=False)
                elif ft == 'parquet':
                    df = pd.read_parquet(fpath).set_index('version', drop=False)
                elif ft == 'pickle':
                    df = pd.read_pickle(fpath)
                else:
                    raise ValueError(f"Unsupported import file type: {file_type!r}")
                results[key] = df
                logging.info(f"Imported '{key}' from {fpath!r}")
            except Exception as e:
                logging.error(f"Failed to import '{key}' from {fpath!r}: {e}")
                results[key] = None

        # 2) always attempt to load regressions pickle
        reg_fname = f"SHAKEtime_aux_regressions_{self.event_id}.pkl"
        reg_path  = Path(input_dir) / reg_fname
        if reg_path.exists():
            try:
                with open(reg_path, 'rb') as f:
                    regs = pickle.load(f)
                logging.info(f"Imported 'regressions' from {reg_path!r}")
            except Exception as e:
                logging.error(f"Failed to load regressions from {reg_path!r}: {e}")
                regs = None
        else:
            logging.warning(f"No regressions pickle found at {reg_path!r}")
            regs = None

        results['regressions'] = regs
        return results


        ##################################################
        #
        #
        #
        # --- Spatial statistics methods Not fully implemented ---
        # --- Pipiline section for adding other functions ---
        # --- Pipiline unstable placeholder for future development ---
        #
        #
        #
        #
        ##################################################


    def add_auxiliary(self,
                      df_aux=None,
                      version_list=None,
                      analyses=None,
                      metric="mmi",
                      **kwargs):
        """
        Take an existing aux‐DataFrame (indexed by version) and tack on
        whichever analyses you request.  If you omit df_aux & version_list,
        it reuses the last one created by analyze_auxiliary_influences().
    
        analyses : list of strings; choose any of:
          ['morans_i', 'getis_ord_gi', 'pca',
           'bayesian_field', 'jaccard', 'functional_depth',
           'global_moran', 'local_moran']
    
        kwargs passed through:
          • k                for morans_i / getis_ord_gi / global/local Moran
          • n_components     for pca
          • thresholds       for jaccard
          • moran_fields     list of fields for global/local Moran
          • export_path, save_formats, dpi for local_moran
        """
        import pandas as pd
    
        # reuse last if needed
        if df_aux is None or version_list is None:
            if not hasattr(self, "_last_aux"):
                raise ValueError("No previous auxiliary table; run analyze_auxiliary_influences() first.")
            df_aux = self._last_aux.copy()
            version_list = self._last_versions
    
        # existing extra methods...
        if "morans_i" in analyses:
            df_aux["morans_i"] = [
                self.morans_i(v, metric=metric, k=kwargs.get("k", 8))
                for v in version_list
            ]
    
        if "getis_ord_gi" in analyses:
            df_aux["getis_gi"] = [
                self.getis_ord_gi(v, metric=metric, k=kwargs.get("k", 8))
                for v in version_list
            ]
    
        if "pca" in analyses:
            pca_res = self.pca_analysis(
                version_list,
                metric=metric,
                n_components=kwargs.get("n_components", 3)
            )
            for i, ev in enumerate(pca_res["explained_variance"], start=1):
                df_aux[f"pca_ev_{i}"] = ev
    
        if "bayesian_field" in analyses:
            bf = self.bayesian_true_field(version_list, metric=metric)
            df_aux["bayes_mu"]    = bf["posterior_mu"]
            df_aux["bayes_sigma"] = bf["posterior_sd"]
    
        if "jaccard" in analyses:
            thr = kwargs.get("thresholds", [4.0])
            jdf = self.jaccard_similarity(version_list, metric=metric, thresholds=thr)
            mean_j = {}
            for v in version_list:
                mask = (jdf.v1 == v) | (jdf.v2 == v)
                mean_j[v] = jdf.loc[mask, "jaccard"].mean()
            df_aux[f"jaccard_mean_{thr[0]}"] = pd.Series(mean_j)
    
        if "functional_depth" in analyses:
            depths = self.functional_data_analysis(version_list, metric=metric)
            df_aux["functional_depth"] = depths
    
        # new global Moran
        if "global_moran" in analyses:
            gm = self.global_moran(
                version_list,
                fields=kwargs.get("moran_fields", ["mmi", "station_density", "dyfi_count"]),
                k=kwargs.get("k", 8)
            )
            df_aux = df_aux.join(gm)
    
        # new local Moran (only generates & saves heatmaps)
        if "local_moran" in analyses:
            for v in version_list:
                self.local_moran(
                    v,
                    fields=kwargs.get("moran_fields", ["mmi", "station_density", "dyfi_count"]),
                    k=kwargs.get("k", 8),
                    export_path=kwargs.get("export_path", "./export"),
                    save_formats=kwargs.get("save_formats", ("png",)),
                    dpi=kwargs.get("dpi", 150)
                )
    
        # refresh stored copy
        self._last_aux = df_aux.copy()
        return df_aux
    

    

    def global_moran(
        self,
        version_list,
        fields=None,
        k: int = 8
    ):
        """
        Compute *global* Moran’s I for one or more fields on the ShakeMap grid.
        Re-computes station‐density and DYFI‐count inside this routine,
        so it has no outside dependencies.

        Parameters
        ----------
        version_list : list[str]
            ShakeMap versions to iterate over.
        fields : list[str], optional
            Which fields to evaluate.  Supported:
              – 'mmi'              ShakeMap MMI
              – 'station_density'  stations per km² (within self._last_radius_km)
              – 'dyfi_count'       count of DYFI reports per cell
        k : int, default 8
            Number of neighbors for the KNN weight matrix.

        Returns
        -------
        pd.DataFrame
            Index = version_list, columns = ["moran_<field>"].
        """
        # defaults
        if fields is None:
            fields = ["mmi", "station_density", "dyfi_count"]

        out = {f"moran_{f}": [] for f in fields}

        for v in version_list:
            # 1) load the unified grid once
            ug = self.get_unified_grid([v], metric="mmi", use_cache=True)
            lonlat = list(zip(ug["lon"], ug["lat"]))

            for f in fields:
                try:
                    # —————————————————————————————
                    #  a) MMI on grid
                    if f == "mmi":
                        vals = ug[f"mmi_v{v}"].values

                    # b) station_density on grid
                    elif f == "station_density":
                        # locate station‐list JSON (must have set _last_station_folder earlier)
                        fp = Path(self._last_station_folder) / self.event_id / f"{self.event_id}_us_{v}_stationlist.json"
                        parser = USGSParser("instrumented_data", json_file=str(fp))
                        df_sta = parser.get_dataframe(value_type="pga")\
                                       .dropna(subset=["latitude","longitude"])
                        # if no stations, fill with zeros
                        if df_sta.empty:
                            vals = np.zeros(len(ug))
                        else:
                            pts_rad = np.deg2rad(ug[["lat","lon"]].values)
                            sta_rad = np.deg2rad(df_sta[["latitude","longitude"]].values)
                            tree = BallTree(sta_rad, metric="haversine")
                            r = self._last_radius_km / 6371.0
                            counts = tree.query_radius(pts_rad, r=r, count_only=True)
                            area = np.pi * (self._last_radius_km**2)
                            vals = counts / area

                    # c) dyfi_count on grid
                    elif f == "dyfi_count":
                        fp = Path(self._last_dyfi_folder) / self.event_id / f"{self.event_id}_us_{v}_stationlist.json"
                        parser = USGSParser("instrumented_data", json_file=str(fp))
                        df_dy = parser.get_dataframe(value_type="mmi")\
                                      .dropna(subset=["latitude","longitude"])
                        if df_dy.empty:
                            vals = np.zeros(len(ug), dtype=int)
                        else:
                            # assign each report to nearest cell
                            tree = cKDTree(ug[["lon","lat"]].values)
                            idx = tree.query(df_dy[["longitude","latitude"]].values, k=1)[1]
                            counts = np.bincount(idx, minlength=len(ug))
                            vals = counts

                    else:
                        raise ValueError(f"Field '{f}' not supported for global Moran")

                    # 2) build weights & compute Moran
                    w = KNN(lonlat, k=k)
                    m = Moran(vals, w)
                    out[f"moran_{f}"].append(m.I)

                except Exception as e:
                    logging.error("global_moran failed for '%s' v%s: %s", f, v, e)
                    out[f"moran_{f}"].append(np.nan)

        import pandas as pd
        return pd.DataFrame(out, index=version_list)


    def local_moran(
        self,
        version_list,
        fields=None,
        k: int = 8,
        export_path: str = "./export",
        save_formats=("png",),
        dpi: int = 150
    ):
        """
        Compute *local* Moran’s I over the final version’s grid, for each requested field,
        and write out heatmaps under:
          {export_path}/SHAKEtime/{event_id}/local_moran/{field}_v{version}.{fmt}
        """
        # choose last version
        v = version_list[-1] if isinstance(version_list, (list,tuple)) else version_list

        # defaults
        if fields is None:
            fields = ["mmi", "station_density", "dyfi_count"]

        # load unified grid
        ug = self.get_unified_grid([v], metric="mmi", use_cache=True)
        lonlat = list(zip(ug["lon"], ug["lat"]))

        # prepare export dir
        out_dir = Path(export_path) / "SHAKEtime" / self.event_id / "local_moran"
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in fields:
            try:
                # same data‐prep as global_moran...
                if f == "mmi":
                    vals = ug[f"mmi_v{v}"].values

                elif f == "station_density":
                    fp = Path(self._last_station_folder) / self.event_id / f"{self.event_id}_us_{v}_stationlist.json"
                    parser = USGSParser("instrumented_data", json_file=str(fp))
                    df_sta = parser.get_dataframe(value_type="pga")\
                                   .dropna(subset=["latitude","longitude"])
                    if df_sta.empty:
                        vals = np.zeros(len(ug))
                    else:
                        pts_rad = np.deg2rad(ug[["lat","lon"]].values)
                        sta_rad = np.deg2rad(df_sta[["latitude","longitude"]].values)
                        tree = BallTree(sta_rad, metric="haversine")
                        r = self._last_radius_km / 6371.0
                        counts = tree.query_radius(pts_rad, r=r, count_only=True)
                        area = np.pi * (self._last_radius_km**2)
                        vals = counts / area

                elif f == "dyfi_count":
                    fp = Path(self._last_dyfi_folder) / self.event_id / f"{self.event_id}_us_{v}_stationlist.json"
                    parser = USGSParser("instrumented_data", json_file=str(fp))
                    df_dy = parser.get_dataframe(value_type="mmi")\
                                  .dropna(subset=["latitude","longitude"])
                    if df_dy.empty:
                        vals = np.zeros(len(ug), dtype=int)
                    else:
                        tree = cKDTree(ug[["lon","lat"]].values)
                        idx = tree.query(df_dy[["longitude","latitude"]].values, k=1)[1]
                        counts = np.bincount(idx, minlength=len(ug))
                        vals = counts

                else:
                    logging.warning("Field '%s' not supported for local Moran → skipping", f)
                    continue

                # compute local Moran
                w   = KNN(lonlat, k=k)
                lm  = Moran_Local(vals, w)
                Is  = lm.Is   # an array of length=len(ug)

                # plot
                fig, ax = plt.subplots()
                sc = ax.scatter(ug["lon"], ug["lat"], c=Is, cmap="RdBu", s=20)
                ax.set_title(f"Local Moran I — {f} v{v}")
                plt.colorbar(sc, ax=ax, label="Local I")

                # save
                for fmt in save_formats:
                    p = out_dir / f"{self.event_id}_local_moran_{f}_v{v}.{fmt}"
                    fig.savefig(p, dpi=dpi, bbox_inches="tight")
                    logging.info("Saved local Moran map to %s", p)
                plt.close(fig)

            except Exception as e:
                logging.error("local_moran failed for '%s' v%s: %s", f, v, e)

                
            
        
    def morans_i(
        self,
        version: str,
        metric: str = "mmi",
        k: int = 8
    ) -> float:
        """
        Compute Moran's I statistic for spatial autocorrelation of the metric grid.
        Requires libpysal.
        """
        try:
            from libpysal.weights import KNN
            from esda.moran import Moran
        except ImportError:
            raise ImportError("libpysal and esda are required for Moran's I")
        ug = self.get_unified_grid([version], metric=metric, use_cache=True)
        coords = list(zip(ug['lon'], ug['lat']))
        w = KNN(coords, k=k)
        vals = ug[f'{metric}_v{version}'].values
        m = Moran(vals, w)
        return m.I


    def getis_ord_gi(
        self,
        version: str,
        metric: str = "mmi",
        k: int = 8
    ) -> float:
        """
        Compute Getis-Ord Gi* statistic for hotspots of the metric grid.
        """
        try:
            from libpysal.weights import KNN
            from esda.getisord import G_Local
        except ImportError:
            raise ImportError("libpysal and esda are required for Getis-Ord Gi*")
        ug = self.get_unified_grid([version], metric=metric, use_cache=True)
        coords = list(zip(ug['lon'], ug['lat']))
        w = KNN(coords, k=k)
        vals = ug[f'{metric}_v{version}'].values
        g = G_Local(vals, w)
        return float(np.nanmean(g.Gs))

    def lisa_clusters(
        self,
        version: str,
        metric: str = "mmi",
        k: int = 8
    ):
        """
        Compute Local Indicators of Spatial Association (LISA).
        Returns a GeoDataFrame with cluster categories.
        """
        try:
            import geopandas as gpd
            from libpysal.weights import KNN
            from esda.moran import Moran_Local
        except ImportError:
            raise ImportError("geopandas, libpysal, and esda are required for LISA")
        ug = self.get_unified_grid([version], metric=metric, use_cache=False)
        gdf = gpd.GeoDataFrame(
            ug,
            geometry=gpd.points_from_xy(ug.lon, ug.lat),
            crs="EPSG:4326"
        )
        coords = list(zip(ug['lon'], ug['lat']))
        w = KNN(coords, k=k)
        m_local = Moran_Local(ug[f'{metric}_v{version}'].values, w)
        gdf['lisa_cluster'] = m_local.q
        return gdf

    def pca_analysis(
        self,
        version_list: List[str],
        metric: str = "mmi",
        n_components: int = 3
    ):
        """
        Stack difference grids across version pairs and run PCA.
        Returns PCA components and explained variance.
        """
        from sklearn.decomposition import PCA
        # build rate grid for consecutive diffs
        ug = self.get_rate_grid(version_list, metric=metric, use_cache=False)
        # select delta columns
        cols = [c for c in ug.columns if c.startswith('delta_') and c.endswith(f'_{metric}')]
        data = ug[cols].fillna(0).values
        pca = PCA(n_components=n_components)
        comps = pca.fit_transform(data)
        return {'components': comps, 'explained_variance': pca.explained_variance_ratio_}

    def forecast_summary(
        self,
        version_list: List[str],
        metric: str = "mmi"
    ):
        """
        Fit an ARIMA model on the summary stat (mean) time series and forecast stabilization.
        Requires statsmodels.
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError("statsmodels is required for forecast_summary")
        df = self.quantify_evolution(version_list, metric=metric)
        ts = df[f'{metric}_mean']
        model = ARIMA(ts, order=(1,1,1)).fit()
        forecast = model.get_forecast(steps=3)
        return forecast.summary_frame()

    def bayesian_true_field(
        self,
        version_list: List[str],
        metric: str = "mmi"
    ):
        """
        Fit a simple Bayesian hierarchical model treating versions as draws.
        Returns posterior mean and sd arrays.
        """
        import pymc3 as pm
        ug = self.get_unified_grid(version_list, metric=metric, use_cache=False)
        # stack values: shape (n_versions, n_points)
        vals = np.vstack([ug[f'{metric}_v{v}'].values for v in version_list])
        with pm.Model() as model:
            mu = pm.Normal('mu', mu=vals.mean(), sd=10)
            sigma = pm.HalfNormal('sigma', sd=5)
            obs = pm.Normal('obs', mu=mu, sd=sigma, observed=vals)
            trace = pm.sample(1000, tune=500, chains=2, cores=1)
        post_mu = trace['mu'].mean()
        post_sd = trace['sigma'].mean()
        return {'posterior_mu': post_mu, 'posterior_sd': post_sd}

    def jaccard_similarity(
        self,
        version_list: List[str],
        metric: str = "mmi",
        thresholds: List[float] = [4.0]
    ) -> pd.DataFrame:
        """
        Compute Jaccard similarity of exceedance sets between each pair of versions.
        Returns DataFrame with pairs and similarity scores.
        """
        records = []
        for i in range(len(version_list)):
            for j in range(i+1, len(version_list)):
                v1, v2 = version_list[i], version_list[j]
                ug = self.get_unified_grid([v1, v2], metric=metric, use_cache=False)
                for thr in thresholds:
                    mask1 = ug[f'{metric}_v{v1}'] > thr
                    mask2 = ug[f'{metric}_v{v2}'] > thr
                    inter = np.logical_and(mask1, mask2).sum()
                    union = np.logical_or(mask1, mask2).sum()
                    sim = inter/union if union>0 else np.nan
                    records.append({'v1':v1,'v2':v2,'threshold':thr,'jaccard':sim})
        return pd.DataFrame(records)

    def functional_data_analysis(
        self,
        version_list: List[str],
        metric: str = "mmi"
    ):
        """
        Perform functional data analysis on the spatial fields.
        Uses the fda package if available to compute depth metrics.
        """
        try:
            from fda import FDataGrid, depth
        except ImportError:
            raise ImportError("fda package is required for functional_data_analysis")
        ug = self.get_unified_grid(version_list, metric=metric, use_cache=False)
        # shape: (n_versions, n_points)
        data = np.vstack([ug[f'{metric}_v{v}'].values for v in version_list])
        coords = ug[['lon','lat']].values
        fd = FDataGrid(data_matrix=data, grid_points=coords)
        d = depth.band_depth(fd)
        return d



    ##################################################
    #
    #
    #
    #
    # --- Analyze SHAKEmaps Using Chaos Theory for SHAKEtime ---
    #
    #
    #
    #
    ##################################################

    def compute_rosenstein_lle(self, ts, emb_dim=6, tau=1, mean_range=5):
        """
        Estimate Largest Lyapunov Exponent using Rosenstein's method.
    
        Parameters
        ----------
        ts : np.ndarray
            1D time series.
        emb_dim : int
            Embedding dimension.
        tau : int
            Time delay (default 1).
        mean_range : int
            Number of steps to average over in divergence.
    
        Returns
        -------
        float
            Estimated largest Lyapunov exponent.
        """
        ts = np.asarray(ts)
        N = len(ts)
        M = N - (emb_dim - 1) * tau
        if M <= 0:
            raise ValueError("Time series too short for embedding.")
    
        X = np.array([ts[i:i + emb_dim * tau:tau] for i in range(M)])
        dists = cdist(X, X)
    
        # Remove temporal neighbors
        np.fill_diagonal(dists, np.inf)
        for i in range(M):
            dists[i, max(0, i - tau):i + tau + 1] = np.inf
    
        nn_idx = np.argmin(dists, axis=1)
    
        div = []
        for j in range(1, mean_range):
            div_j = []
            for i in range(M - j):
                xi = X[i + j]
                xj = X[nn_idx[i] + j] if (nn_idx[i] + j < M) else None
                if xj is not None:
                    dist = np.linalg.norm(xi - xj)
                    if dist > 0:
                        div_j.append(np.log(dist))
            if div_j:
                div.append(np.mean(div_j))
    
        if len(div) < 2:
            return np.nan
    
        t = np.arange(1, len(div) + 1)
        slope, _ = np.polyfit(t, div, 1)
        return slope
    

        
    def compute_chaos_metrics(
        self,
        metric: str = "mmi",
        version_list: list = None,
        mode: str = "both",
        output_path: str = None,
        save_formats: list = ["csv", "json", "pkl"],
        emb_dim: int = 6,
        use_pca: bool = True,
        plot: bool = False,
        max_points: int = 5000
    ):
        """
        Compute global and/or spatial chaos theory metrics on a unified ShakeMap intensity grid.
    
        Parameters
        ----------
        metric : str
            The ground motion intensity measure to use (e.g., "mmi", "pga").
        version_list : list of str
            List of ShakeMap versions to analyze.
        mode : str
            One of "global", "spatial", or "both". Determines what metrics are computed.
        output_path : str or Path, optional
            If provided, exports results to disk in a structured folder.
        save_formats : list of str
            Which formats to save: any combination of ["csv", "json", "pkl"].
        emb_dim : int
            Embedding dimension for chaos metrics (Lyapunov, D2).
        use_pca : bool
            If True, extract ts_global using PCA(1). Else, use mean across grid.
        plot : bool
            If True, plots the ts_global sequence.
        max_points : int
            Maximum number of spatial grid points to compute chaos for (reduces cost).
    
        Returns
        -------
        results : dict
            Dictionary containing global chaos metrics, spatial chaos map, and metadata.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import warnings
        import joblib
        import json
        from pathlib import Path
        from sklearn.exceptions import UndefinedMetricWarning
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from nolds import lyap_r, corr_dim, sampen
    
        if version_list is None or not isinstance(version_list, list):
            raise ValueError("You must provide a valid list of ShakeMap versions.")
    
        logging.info(f"[Chaos] Computing chaos metrics for metric={metric}, versions={version_list}, mode={mode}")
    
        unified_grid = (
            self._unified_grid
            if hasattr(self, "_unified_grid") and self._unified_grid is not None
            else self.get_unified_grid(version_list, metric=metric, use_cache=True)
        )
    
        metric_cols = [f"{metric}_v{v}" for v in version_list]
        clean_grid = unified_grid[metric_cols].dropna()
        if clean_grid.empty:
            raise RuntimeError("Unified grid has no valid data for selected versions/metric.")
    
        results = {}
    
        # --- GLOBAL CHAOS METRICS ---
        if mode in ("global", "both"):
            try:
                if use_pca:
                    ts_global = PCA(n_components=1).fit_transform(clean_grid.T).flatten()
                    logging.info("[Chaos] Used PCA(1) for global signal extraction.")
                else:
                    ts_global = clean_grid.mean(axis=0).values
                    logging.info("[Chaos] Used mean() over grid for global signal extraction.")
    
                ts_global = StandardScaler().fit_transform(ts_global.reshape(-1, 1)).flatten()
                ts_global = np.ascontiguousarray(ts_global, dtype=np.float64)
    
                if plot:
                    plt.plot(ts_global, marker='o')
                    plt.title("Global Time Series for Chaos Metrics")
                    plt.xlabel("Version index")
                    plt.grid()
                    plt.show()
    
                if len(ts_global) < emb_dim * 2:
                    emb_dim = max(2, len(ts_global) // 2)
                    logging.info(f"[Chaos] Reduced embedding dimension to {emb_dim} due to short series.")
    
                try:
                    lle = lyap_r(ts_global, emb_dim=emb_dim)
                    logging.info("[Chaos] LLE computed using lyap_r()")
                except Exception as e1:
                    logging.warning(f"[Chaos] LLE computation with lyap_r() failed: {e1}")
                    try:
                        lle = self.compute_rosenstein_lle(ts_global, emb_dim=emb_dim)
                        logging.info("[Chaos] LLE fallback computed using Rosenstein method")
                    except Exception as e2:
                        logging.warning(f"[Chaos] Rosenstein LLE computation failed: {e2}")
                        lle = np.nan
    
                try:
                    d2 = corr_dim(ts_global, emb_dim=emb_dim)
                except Exception as e:
                    logging.warning(f"[Chaos] CorrDim computation failed: {e}")
                    d2 = np.nan
    
                try:
                    entropy = sampen(ts_global)
                except Exception as e:
                    logging.warning(f"[Chaos] Sample Entropy computation failed: {e}")
                    entropy = np.nan
    
            except Exception as e:
                logging.warning(f"[Chaos] Global chaos preprocessing failed: {e}")
                lle, d2, entropy = np.nan, np.nan, np.nan
                ts_global = np.array([])
    
            results["global"] = {
                "lyapunov_exponent": lle,
                "correlation_dimension": d2,
                "sample_entropy": entropy,
                "length": len(ts_global)
            }
            results["ts_global"] = ts_global
    
        # --- SPATIAL CHAOS METRICS ---
        if mode in ("spatial", "both"):
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            warnings.filterwarnings("ignore", message="RANSAC did not reach consensus*")
    
            lle_vals, d2_vals, se_vals = [], [], []
    
            if len(clean_grid) > max_points:
                clean_grid = clean_grid.sample(n=max_points, random_state=42)
    
            for idx, row in clean_grid.iterrows():
                ts = row.values.astype(np.float64)
                if np.isnan(ts).any() or np.std(ts) < 1e-6:
                    lle, d2, se = np.nan, np.nan, np.nan
                else:
                    ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-8)
                    ts = np.ascontiguousarray(ts, dtype=np.float64)
    
                    try:
                        lle = lyap_r(ts, emb_dim=emb_dim)
                    except Exception:
                        try:
                            lle = self.compute_rosenstein_lle(ts, emb_dim=emb_dim)
                        except Exception:
                            lle = np.nan
    
                    try:
                        d2 = corr_dim(ts, emb_dim=emb_dim)
                    except Exception:
                        d2 = np.nan
    
                    try:
                        se = sampen(ts)
                    except Exception:
                        se = np.nan
    
                lle_vals.append(lle)
                d2_vals.append(d2)
                se_vals.append(se)
    
            chaos_df = unified_grid[['lon', 'lat']].iloc[clean_grid.index].copy()
            chaos_df["lyapunov"] = lle_vals
            chaos_df["corr_dim"] = d2_vals
            chaos_df["sampen"] = se_vals
            results["spatial"] = chaos_df
    
        # --- VERSION METADATA ---
        if hasattr(self, 'summary_df') and not self.summary_df.empty:
            try:
                version_times = (
                    self.summary_df.set_index("version")
                    .loc[version_list]
                    .get("TaE_h", pd.Series(index=version_list, dtype=float))
                    .to_dict()
                )
                results["time_after_event"] = version_times
            except Exception as e:
                logging.warning(f"[Chaos] Failed to attach version timing metadata: {e}")
    
        # --- EXPORT CHAOS METRICS ---
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "choas_theory_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            base_path = out_dir / f"{self.event_id}_chaos_metrics"
    
            if "csv" in save_formats:
                if "global" in results and "ts_global" in results:
                    df_csv = pd.DataFrame({
                        "version": list(results.get("time_after_event", {}).keys()),
                        "time_after_event_h": list(results.get("time_after_event", {}).values()),
                        "ts_global": results["ts_global"]
                    })
                    df_csv["lyapunov_exponent"] = results["global"].get("lyapunov_exponent")
                    df_csv["correlation_dimension"] = results["global"].get("correlation_dimension")
                    df_csv["sample_entropy"] = results["global"].get("sample_entropy")
                    df_csv.to_csv(base_path.with_suffix(".csv"), index=False)
                if "spatial" in results and isinstance(results["spatial"], pd.DataFrame):
                    results["spatial"].to_csv(out_dir / f"{self.event_id}_chaos_spatial.csv", index=False)
    
            if "json" in save_formats:
                try:
                    with open(base_path.with_suffix(".json"), "w") as f:
                        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
                except Exception as e:
                    logging.warning(f"[Chaos] Failed to export JSON: {e}")
    
            if "pkl" in save_formats:
                try:
                    joblib.dump(results, base_path.with_suffix(".pkl"))
                except Exception as e:
                    logging.warning(f"[Chaos] Failed to export Pickle: {e}")
    
        return results

    
        
    def compute_rolling_chaos_metrics(
        self,
        version_list,
        metric="mmi",
        window_size=7,
        step=1,
        emb_dim=6,
        use_pca=True,
        output_path=None,
    ):
        """
        Compute rolling chaos metrics (LLE, SampEn, CorrDim) over sliding windows of ShakeMap versions.
    
        Parameters
        ----------
        self : SHAKEtime
            Instance of SHAKEtime class.
        version_list : list of str
            Full list of ShakeMap versions in temporal order.
        metric : str
            Intensity measure to analyze (e.g., "mmi", "pga").
        window_size : int
            Number of versions per rolling window.
        step : int
            How many versions to slide the window each step.
        emb_dim : int
            Embedding dimension for LLE and CorrDim.
        use_pca : bool
            If True, use PCA to extract ts_global, else use mean.
        output_path : str or Path or None
            If provided, saves the result DataFrame to this directory under "SHAKEtime/{event_id}/choas_theory_results/".
        save_name : str
            Filename to use for saving the CSV file (default "rolling_chaos_metrics.csv").
    
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            ['start_version', 'end_version', 'mid_time', 'lyapunov', 'sampen', 'corr_dim']
        """
        import pandas as pd
        import numpy as np
        from pathlib import Path
    
        results = []
    
        for i in range(0, len(version_list) - window_size + 1, step):
            win_versions = version_list[i:i + window_size]
            chaos = self.compute_chaos_metrics(
                metric=metric,
                version_list=win_versions,
                mode="global",
                use_pca=use_pca,
                plot=False,
                emb_dim=emb_dim
            )
    
            chaos_global = chaos["global"]
            times = chaos["time_after_event"]
            mid_time = np.mean([times[v] for v in win_versions if v in times])
    
            results.append({
                "start_version": win_versions[0],
                "end_version": win_versions[-1],
                "mid_time": mid_time,
                "lyapunov": chaos_global.get("lyapunov_exponent", np.nan),
                "sampen": chaos_global.get("sample_entropy", np.nan),
                "corr_dim": chaos_global.get("correlation_dimension", np.nan),
                "window_length": window_size
            })
    
        df = pd.DataFrame(results)
        

    
        if output_path:
            save_name=f"{self.event_id}_rolling_chaos_metrics.csv"
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "choas_theory_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / save_name, index=False)
    
        return df

    def plot_rolling_chaos(
        self,
        version_list,
        metric= "mmi",
        window_size=7,
        step=1,
        emb_dim=6,
        use_pca=True,
        output_path=None,
        save_formats=["png", "pdf"],
        dpi=300,
        show=True,
        highlight_chaos=True,
        compute_only=False, figsize=(12, 5), show_title = True
    ):
        """
        Compute and plot rolling chaos metrics over time with optional chaos highlighting and export.
    
        Parameters
        ----------
        self : SHAKEtime
            Instance of SHAKEtime class.
        version_list : list of str
            ShakeMap versions in time order.
        metric : str
            Intensity measure.
        window_size : int
            Rolling window size.
        step : int
            Step size between windows.
        emb_dim : int
            Embedding dimension for chaos metrics.
        use_pca : bool
            Whether to use PCA for ts_global extraction.
        output_path : str or Path or None
            Where to save the plots and CSV.
        save_formats : list of str
            Image file formats to save.
        dpi : int
            Plot resolution.
        show : bool
            Whether to display the plot.
        highlight_chaos : bool
            If True, highlights LLE > 0 zones.
        compute_only : bool
            If True, skip plotting and only return DataFrame.
    
        Returns
        -------
        pd.DataFrame
            Rolling chaos metrics DataFrame.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from pathlib import Path
    
        df = self.compute_rolling_chaos_metrics(
            version_list=version_list,
            metric=metric,
            window_size=window_size,
            step=step,
            emb_dim=emb_dim,
            use_pca=use_pca,
            output_path=output_path
        )
    
        if compute_only:
            return df
    
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df["mid_time"], df["lyapunov"], label="LLE", color="red", marker='o')
        ax.plot(df["mid_time"], df["sampen"], label="Sample Entropy", color="blue", marker='s')
        ax.plot(df["mid_time"], df["corr_dim"], label="Corr Dimension", color="green", marker='^')
    
        if highlight_chaos:
            chaotic = df["lyapunov"] > 0
            ax.fill_between(df["mid_time"], ax.get_ylim()[0], ax.get_ylim()[1],
                            where=chaotic, color="red", alpha=0.1, label="LLE > 0")
    
        ax.set_xlabel("Time After Event (hrs)")
        ax.set_ylabel("Metric Value")
        if show_title:
            ax.set_title(f"Rolling Chaos Metrics (window={window_size}, step={step})")

        ax.tick_params(axis='x', labelrotation=45)

        ax.legend()
        ax.grid(True)
    
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "choas_theory_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            for ext in save_formats:
                fig.savefig(out_dir / f"{self.event_id}_rolling_chaos_metrics.{ext}", bbox_inches="tight", dpi=dpi)
    
        if show:
            plt.show()
    
        return df
    
             


    def plot_global_chaos(
        self,
        version_list,
        metric="mmi",
        use_pca=True,
        output_path=None,
        save_formats=["png", "pdf"],
        dpi=300,
        show=True, show_title = True,
        export_csv=True,
        x_axis_mode="time",figsize=(12, 5)  # 'time' for TaE_h, 'version' for version labels
    ):
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from pathlib import Path
    
        results = self.compute_chaos_metrics(
            metric=metric,
            version_list=version_list,
            mode="global",
            use_pca=use_pca,
            plot=False
        )
    
        chaos = results["global"]
        ts_global = results["ts_global"]
        tae_dict = results["time_after_event"]
    
        x_vals = list(range(len(version_list)))
        tick_labels = [f"{tae_dict[v]:.1f}" if x_axis_mode == "time" else v for v in version_list]
    
        # Figure 1: ts_global
        fig1, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(x_vals, ts_global, marker='o', color='black', label="ts_global (PCA/Mean)")
        ax1.set_ylabel("Standardized Intensity")
        if show_title:
            ax1.set_title("ShakeMap Evolution (Global Signal)")
        ax1.set_xlabel("ShakeMap Version" if x_axis_mode == "version" else "Time After Event (hrs)")
        ax1.set_xticks(x_vals)
        ax1.set_xticklabels(tick_labels, rotation=45)
        ax1.grid(True)
    
        # Figure 2: Chaos metrics
        fig2, ax2 = plt.subplots(figsize=figsize)
        ax2.hlines(chaos["lyapunov_exponent"], xmin=0, xmax=len(version_list)-1, colors='r', label="LLE")
        ax2.hlines(chaos["correlation_dimension"], xmin=0, xmax=len(version_list)-1, colors='g', label="Corr Dim")
        ax2.hlines(chaos["sample_entropy"], xmin=0, xmax=len(version_list)-1, colors='b', label="SampEn")
        ax2.legend(loc="upper right")
        ax2.set_ylabel("Chaos Metric Value")
        ax2.set_xlabel("ShakeMap Version" if x_axis_mode == "version" else "Time After Event (hrs)")
        ax2.set_xticks(x_vals)
        ax2.set_xticklabels(tick_labels, rotation=45)
        if show_title:
            ax2.set_title("Chaos Metrics (Global)")
        ax2.grid(True)
    
        if output_path:
            od = Path(output_path) / "SHAKEtime" / self.event_id / "choas_theory_results"
            od.mkdir(parents=True, exist_ok=True)
    
            for ext in save_formats:
                fig1.savefig(od / f"{self.event_id}_global_signal.{ext}", bbox_inches="tight", dpi=dpi)
                fig2.savefig(od / f"{self.event_id}_global_chaos_metrics.{ext}", bbox_inches="tight", dpi=dpi)
    
            if export_csv:
                times = [tae_dict[v] for v in version_list if v in tae_dict]
                df_out = pd.DataFrame({
                    "version": version_list,
                    "time_after_event_h": times,
                    "ts_global": ts_global
                })
                df_out["lyapunov_exponent"] = chaos["lyapunov_exponent"]
                df_out["correlation_dimension"] = chaos["correlation_dimension"]
                df_out["sample_entropy"] = chaos["sample_entropy"]
    
                df_out.to_csv(od / f"{self.event_id}_global_chaos_metrics.csv", index=False)
    
        if show:
            plt.show()
    
        return fig1, fig2

    def plot_spatial_chaos(
        self,
        version_list,
        output_path=None,
        save_formats=["png", "pdf"],
        event_id=None,
        cmap="viridis",
        dpi=300,
        show_title = True,
        show=True,
        emb_dim=6,
        max_points=5000, figsize=(10,8),label_size= 20,s=30, colorbar_font = 15
    ):
        """
        Compute and plot all spatial chaos maps (lyapunov, corr_dim, sampen) using Cartopy base map.
    
        Parameters
        ----------
        version_list : list of str
            ShakeMap versions to analyze.
        output_path : str or Path, optional
            Where to save the outputs. If None, only displays.
        save_formats : list of str
            List of file extensions to save plots in.
        event_id : str, optional
            Earthquake event identifier. Defaults to self.event_id.
        cmap : str
            Colormap for plotting.
        dpi : int
            Plot resolution.
        show : bool
            Whether to display figures interactively.
        emb_dim : int
            Embedding dimension for chaos metrics.
        max_points : int
            Max number of points to evaluate spatially.
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        import cartopy.crs as ccrs
        from cartopy.feature import BORDERS, OCEAN
    
        event_id = event_id or self.event_id
    
        results = self.compute_chaos_metrics(
            version_list=version_list,
            mode="spatial",
            output_path=output_path,
            save_formats=["csv"],
            emb_dim=emb_dim,
            max_points=max_points
        )
    
        if "spatial" not in results:
            raise ValueError("No spatial chaos results found.")
    
        df = results["spatial"]
        metrics = ["lyapunov", "corr_dim", "sampen"]
        figs = []
    
        for metric in metrics:
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.add_feature(BORDERS)
            ax.add_feature(OCEAN, facecolor='lightblue', alpha=0.3)
            #ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

            gl = ax.gridlines(crs=ccrs.PlateCarree(),
                               draw_labels=True, linewidth=2,
                               color='gray', alpha=0.7,
                               linestyle='--', zorder=999)
            gl.top_labels   = False
            gl.right_labels = False
            gl.xlabel_style = {"size": label_size}
            gl.ylabel_style = {"size": label_size}
    
            sc = ax.scatter(df["lon"], df["lat"], c=df[metric], cmap=cmap, s=s, edgecolor="k", linewidth=0.3, transform=ccrs.PlateCarree())
            cbar = plt.colorbar(sc, ax=ax, label=metric)

            # set the main label’s font size
            cbar.set_label(metric, fontsize=colorbar_font)
            
            # optionally also set the tick‐label font size
            cbar.ax.tick_params(labelsize=colorbar_font)


            if show_title:
                ax.set_title(f"{event_id} Spatial Chaos Map - {metric.title()}")
    
            if output_path:
                out_dir = Path(output_path) / "SHAKEtime" / event_id / "choas_theory_results"
                out_dir.mkdir(parents=True, exist_ok=True)
                for fmt in save_formats:
                    fig.savefig(out_dir / f"{event_id}_chaos_spatial_{metric}_map.{fmt}", dpi=dpi, bbox_inches="tight")
    
            if show:
                plt.show()
    
            figs.append(fig)
    
        return figs

    


    def plot_spatial_chaos_maps(
        self,
        version_list,
        output_path=None,
        save_formats=["png", "pdf"],
        event_id=None,
        cmap="viridis",
        dpi=300,
        show=True,
        emb_dim=6,
        max_points=5000, figsize=(8, 6), show_title= True
    ):
        """
        Compute and plot all spatial chaos maps (lyapunov, corr_dim, sampen) from version list.
    
        Parameters
        ----------
        version_list : list of str
            ShakeMap versions to analyze.
        output_path : str or Path, optional
            Where to save the outputs. If None, only displays.
        save_formats : list of str
            List of file extensions to save plots in.
        event_id : str, optional
            Earthquake event identifier. Defaults to self.event_id.
        cmap : str
            Colormap for plotting.
        dpi : int
            Plot resolution.
        show : bool
            Whether to display figures interactively.
        emb_dim : int
            Embedding dimension for chaos metrics.
        max_points : int
            Max number of points to evaluate spatially.
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
    
        event_id = event_id or self.event_id
    
        results = self.compute_chaos_metrics(
            version_list=version_list,
            mode="spatial",
            output_path=output_path,
            save_formats=["csv"],
            emb_dim=emb_dim,
            max_points=max_points
        )
    
        if "spatial" not in results:
            raise ValueError("No spatial chaos results found.")
    
        df = results["spatial"]
        metrics = ["lyapunov", "corr_dim", "sampen"]
        figs = []
    
        for metric in metrics:
            fig, ax = plt.subplots(figsize=figsize)
            sc = ax.scatter(df["lon"], df["lat"], c=df[metric], cmap=cmap, s=30, edgecolor="k", linewidth=0.2)
            plt.colorbar(sc, ax=ax, label=metric)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            if show_title:
                ax.set_title(f"{event_id} Spatial Chaos Map - {metric.title()}")
            ax.grid(True)
    
            if output_path:
                out_dir = Path(output_path) / "SHAKEtime" / event_id / "choas_theory_results"
                out_dir.mkdir(parents=True, exist_ok=True)
                for fmt in save_formats:
                    fig.savefig(out_dir / f"{event_id}_chaos_spatial_{metric}.{fmt}", dpi=dpi, bbox_inches="tight")
    
            if show:
                plt.show()
    
            figs.append(fig)
    
        return figs


    ##################################################
    #
    #
    #       AUX regression and City Progress 
    #
    #
    #
    ##################################################


    
    def plot_auxiliary_regression_summary(
        self,
        version_list,
        intensity_threshold=None,
        p_uncertainty=None,
        output_path: str = None,
        show: bool = False,
        save_formats=("png","pdf"),
        dpi: int = 300,
        **aux_kwargs
    ):
        """
        Plot a single grouped‐bar figure of regression slopes (+95% CI) for
        all diagnostics vs. all auxiliary covariates.
    
        Returns the combined DataFrame of slopes & CIs.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
    
        # 1) load precomputed regression models
        res = self.analyze_auxiliary_influences(
            version_list,
            metric="mmi",
            use_cache=True,
            **aux_kwargs
        )
        regressions = res["regressions"]  # dict: diag -> OLSResults
    
        if not regressions:
            raise RuntimeError("No regression models found to plot.")
    
        # 2) extract slopes & CIs into a single DataFrame
        all_diags = sorted(regressions.keys())
        combined = {}
        for d in all_diags:
            model = regressions[d]
            slopes = model.params.drop("const")
            ci = model.conf_int().loc[slopes.index]
            df = slopes.to_frame(name="slope")
            df["ci_lower"] = ci[0]
            df["ci_upper"] = ci[1]
            combined[d] = df
    
        # pivot into wide form: index=covariate, columns=(diag_slope, diag_ci_lower, diag_ci_upper)
        covariates = list(combined[all_diags[0]].index)
        n_cov = len(covariates)
        n_diag = len(all_diags)
    
        # 3) build figure
        fig, ax = plt.subplots(figsize=(1.5 * n_cov, 6))
    
        x = np.arange(n_cov)
        width = 0.8 / n_diag
    
        for i, diag in enumerate(all_diags):
            df = combined[diag]
            slopes = df["slope"].values
            err = [slopes - df["ci_lower"].values,
                   df["ci_upper"].values - slopes]
            ax.bar(
                x + i * width,
                slopes,
                width=width,
                yerr=err,
                capsize=4,
                label=diag
            )
    
        ax.set_xticks(x + width * (n_diag - 1) / 2)
        ax.set_xticklabels(covariates, rotation=30, ha="right")
        ax.set_ylabel("Regression slope")
        ax.set_title("Auxiliary Covariate Influence on Diagnostics")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend(title="Diagnostic", bbox_to_anchor=(1.02,1), loc="upper left")
    
        # 4) save if requested
        if output_path:
            save_dir = Path(output_path) / "SHAKEtime" / self.event_id / "auxiliary_influences"
            save_dir.mkdir(parents=True, exist_ok=True)
            for fmt in save_formats:
                p = save_dir / f"{self.event_id}_aux_regression_summary.{fmt}"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                logging.info(f"Saved auxiliary regression summary to {p!r}")
    
        if show:
            plt.show()
        else:
            plt.close(fig)
    
        # return wide‐format DataFrame for further inspection
        # e.g., columns = MultiIndex[(diag, metric), ...]
        wide = pd.concat(
            {diag: combined[diag] for diag in all_diags},
            axis=1
        )
        return wide



    def plot_city_and_global_progression(
        self,
        version_list,        # list of version strings, e.g. ["001","002",…]
        cities,              # list of city names matching "<city>_mmi" columns in summary_df
        threshold=None,      # float intensity at which to draw a horizontal line (optional)
        x_axis="version",    # either "version" or "TaE" for tick‐labels
        xrotation=45,
        x_limits=None,       # tuple (xmin, xmax) to fix x-axis limits (optional)
        y_limits=None,       # tuple (ymin, ymax) to fix y-axis limits (optional)
        figsize=(12, 8),     # tuple (width, height) in inches
        output_path=None,    # base directory under which to save
        save_formats=("png", "pdf"),
        dpi=300,
        linewidth=2,
        legendloc="best", plot_title=True
    ):
        """
        Plot per‐city intensity progression alongside global min/median/mean/max
        across ShakeMap versions, using discrete x=0,1,2… but showing either
        the version strings or TaE_h values as tick labels. Optionally highlights
        a threshold and fixes x/y limits. Saves into EvolutionPlots if output_path
        is given. Returns (fig, ax).
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        import numpy as np
        import logging

        # 1) Ensure we have summary_df in memory
        if not hasattr(self, 'summary_df') or self.summary_df is None or self.summary_df.empty:
            self.get_shake_summary(version_list)
        df_sum = self.summary_df

        # 2) Fetch unified grid from cache
        ug = self.get_unified_grid(version_list, metric="mmi", use_cache=True)

        # 3) Extract each city's MMI over versions
        city_series = {}
        for city in cities:
            col = f"{city}_mmi"
            if col not in df_sum.columns:
                raise ValueError(f"City '{city}' not found in summary_df columns")
            vals = []
            for v in version_list:
                row = df_sum.loc[df_sum["version"].astype(str) == str(v)]
                vals.append(row[col].iloc[0] if not row.empty else np.nan)
            city_series[city] = np.array(vals)

        # 4) Compute global stats from unified grid
        mins, meds, means, maxs = [], [], [], []
        for v in version_list:
            col_v = f"mmi_v{v}"
            if col_v in ug.columns:
                data = ug[col_v].dropna().values
                if data.size:
                    mins.append(data.min())
                    meds.append(np.median(data))
                    means.append(data.mean())
                    maxs.append(data.max())
                    continue
            mins.append(np.nan)
            meds.append(np.nan)
            means.append(np.nan)
            maxs.append(np.nan)

        # 5) Prepare discrete x positions
        x = np.arange(len(version_list))

        # 6) Determine tick labels
        if x_axis == "TaE":
            if "TaE_h" not in df_sum.columns:
                raise ValueError("No 'TaE_h' column in summary_df")
            # map versions → TaE_h
            tae = (
                df_sum
                .set_index(df_sum["version"].astype(str))
                .loc[version_list, "TaE_h"]
                .astype(float)
                .tolist()
            )
            x_labels = [f"{h:.1f}" for h in tae]
            xlabel = "Time After Event (hrs)"
        else:
            x_labels = version_list
            xlabel = "ShakeMap Version"

        # 7) Plot everything
        fig, ax = plt.subplots(figsize=figsize)

        # 7a) city lines
        for city, vals in city_series.items():
            ax.plot(x, vals,
                    marker='o', linewidth=linewidth+1,
                    label=city)

        # 7b) global metrics
        ax.plot(x, mins,  linestyle='--', linewidth=linewidth, color='#D3D3D3', label='Global min')     # LightGray
        ax.plot(x, meds,  linestyle='-.', linewidth=linewidth, color='#A9A9A9', label='Global median')  # DarkGray
        ax.plot(x, means, linestyle=(0, (3, 5, 1, 5)), linewidth=linewidth, color='#808080', label='Global mean')  # Gray
        ax.plot(x, maxs,  linestyle=':', linewidth=linewidth, color='#505050', label='Global max')      # DimGray

        # 7c) optional threshold
        if threshold is not None:
            ax.axhline(threshold,
                       linewidth=linewidth-1, linestyle='-', color='k',
                       label=f'threshold = {threshold}')

        # 8) Styling & ticks
        ax.set_xlabel(xlabel)
        ax.set_ylabel("MMI Intensity")

        if plot_title:
            ax.set_title(f"{self.event_id} — City & Global MMI Progression")
        ax.legend(loc=legendloc)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=xrotation)

        if x_limits is not None:
            ax.set_xlim(x_limits)
        if y_limits is not None:
            ax.set_ylim(y_limits)

        plt.tight_layout()

        # 9) Save if requested
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "EvolutionPlots"
            out_dir.mkdir(parents=True, exist_ok=True)
            for fmt in save_formats:
                fname = f"{self.event_id}_city_global_progression.{fmt}"
                fpath = out_dir / fname
                fig.savefig(fpath, dpi=dpi, bbox_inches='tight')
                logging.info(f"Saved city/global progression plot to {fpath!r}")

        return fig, ax


    ##############################################################
    #
    #
    #
    #              Unified Grid Analysis Methods 
    #
    #
    #
    #
    #######################################################

    def analyze_unified_grid_mean_median(
        self,
        version_list: list,
        metric: str = "mmi",
        use_cache: bool = True,
        x_ticks: str = "version",   # "version", "TaE_h", "TaE_d" (uses summary if available)
        xrotation: int = 45,
        n_boot: int = 500,
        ci: float = 0.95,
        random_state: int = 42,
        output_path: str = None,
        save_formats: list = ["png", "pdf"],
        dpi: int = 300,
        make_plots: bool = True,
        close_figs: bool = True,
        # ---- styling kwargs (similar spirit to plot_alerts) ----
        figsize: tuple = (12, 6),
        diff_figsize: tuple = (12, 4),
        font_sizes: dict = None,         # {"labels": 12, "ticks": 10, "title": 14, "legend": 10}
        marker_size: int = 6,
        line_width: float = 2.0,
        grid_kwargs: dict = None,        # {"linestyle":"--","alpha":0.4}
        legend_loc: str = "best",
        show_title: bool = True,
    ):
        """
        Compare mean vs median of a ShakeMap metric (e.g., MMI) across versions using the unified grid.
    
        What you get:
          - summary table per version (mean, median, std, skew, q05/q95, mean-median, bootstrap CI)
          - optional plots:
              (1) mean & median vs x
              (2) (mean-median) vs x with CI
          - optional CSV + figure saving under:
              output_path/SHAKEtime/<event_id>/unified_grid_analysis/
    
        Notes:
          - This is a *spatial* comparison across grid cells within each version.
          - If bootstrap CI for (mean-median) excludes 0, you can say mean and median differ meaningfully.
        """
        import numpy as np
        import pandas as pd
        import logging
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from pathlib import Path
    
        # ---------------- defaults ----------------
        if font_sizes is None:
            font_sizes = {"labels": 12, "ticks": 10, "title": 14, "legend": 10}
        lbl_fs = font_sizes.get("labels", 12)
        tck_fs = font_sizes.get("ticks", 10)
        ttl_fs = font_sizes.get("title", 14)
        lgd_fs = font_sizes.get("legend", 10)
    
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.4}
    
        # ---------------- load unified grid ----------------
        unified = self.get_unified_grid(version_list=version_list, metric=metric, use_cache=use_cache)
    
        if unified is None or unified.empty:
            logging.error("Unified grid is empty; cannot analyze mean vs median.")
            return None
    
        # metric columns (mmi_v001, etc.)
        metric_cols = [c for c in unified.columns if c.startswith(f"{metric}_v")]
        if not metric_cols:
            raise ValueError(f"No columns found like '{metric}_v###' in unified grid for metric='{metric}'.")
    
        # enforce order by the provided version_list (not by column sort)
        versions = [v.zfill(3) for v in version_list]
        cols_ordered = [f"{metric}_v{v}" for v in versions if f"{metric}_v{v}" in unified.columns]
        if not cols_ordered:
            raise ValueError("None of the requested version metric columns exist in unified grid.")
    
        rng = np.random.default_rng(random_state)
        alpha = (1.0 - ci) / 2.0
        lo_q = 100.0 * alpha
        hi_q = 100.0 * (1.0 - alpha)
    
        rows = []
        for v, col in zip(versions, cols_ordered):
            s = pd.to_numeric(unified[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            n = int(s.shape[0])
            if n == 0:
                rows.append({
                    "version": v, "n_cells": 0,
                    "mean": np.nan, "median": np.nan, "std": np.nan, "skew": np.nan,
                    "q05": np.nan, "q95": np.nan,
                    "mean_minus_median": np.nan,
                    "diff_ci_low": np.nan, "diff_ci_high": np.nan,
                    "diff_ci_excludes_zero": False
                })
                continue
    
            mean_v = float(s.mean())
            med_v  = float(s.median())
            std_v  = float(s.std())
            skew_v = float(s.skew())
            q05_v  = float(s.quantile(0.05))
            q95_v  = float(s.quantile(0.95))
            diff_v = mean_v - med_v
    
            # bootstrap CI for (mean - median)
            # (resample cells; measures sensitivity to outliers / skew)
            idx = np.arange(n)
            diffs = np.empty(n_boot, dtype=float)
            arr = s.to_numpy()
            for i in range(n_boot):
                samp = arr[rng.choice(idx, size=n, replace=True)]
                diffs[i] = float(np.mean(samp) - np.median(samp))
    
            ci_low = float(np.percentile(diffs, lo_q))
            ci_high = float(np.percentile(diffs, hi_q))
            excludes0 = (ci_low > 0.0) or (ci_high < 0.0)
    
            rows.append({
                "version": v,
                "n_cells": n,
                "mean": mean_v,
                "median": med_v,
                "std": std_v,
                "skew": skew_v,
                "q05": q05_v,
                "q95": q95_v,
                "mean_minus_median": diff_v,
                "diff_ci_low": ci_low,
                "diff_ci_high": ci_high,
                "diff_ci_excludes_zero": bool(excludes0)
            })
    
        df_out = pd.DataFrame(rows).set_index("version")
    
        # ---------------- x-axis labels ----------------
        x_pos = np.arange(len(df_out))
        x_labels = df_out.index.tolist()
        xlabel = "Version"
    
        if x_ticks != "version":
            # try to pull from summary if present
            try:
                summary = self.get_dataframe().copy()
                if "version" in summary.columns:
                    summary["version"] = summary["version"].astype(str).str.zfill(3)
                    summary = summary.set_index("version")
    
                    if x_ticks in summary.columns:
                        vals = pd.to_numeric(summary.loc[df_out.index, x_ticks], errors="coerce")
                        x_labels = [f"{v:.1f}" if pd.notnull(v) else "" for v in vals]
                        xlabel_map = {"TaE_h": "Time After Event (hours)",
                                      "TaE_d": "Time After Event (days)"}
                        xlabel = xlabel_map.get(x_ticks, x_ticks)
            except Exception as e:
                logging.warning(f"Could not use x_ticks='{x_ticks}' from summary; falling back to version. ({e})")
    
        # ---------------- plots ----------------
        figs = {}
        if make_plots:
            # Plot 1: mean & median
            fig1, ax1 = plt.subplots(figsize=figsize)
            ax1.plot(x_pos, df_out["mean"].values, marker="o", markersize=marker_size,
                     linewidth=line_width, label="Mean")
            ax1.plot(x_pos, df_out["median"].values, marker="s", markersize=marker_size,
                     linewidth=line_width, label="Median")
    
            ax1.set_xlabel(xlabel, fontsize=lbl_fs)
            ax1.set_ylabel(metric.upper(), fontsize=lbl_fs)
            ax1.grid(True, **grid_kwargs)
            ax1.tick_params(axis="both", labelsize=tck_fs)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(x_labels, rotation=xrotation,
                                ha="right" if x_ticks != "version" else "center")
            ax1.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax1.legend(loc=legend_loc, fontsize=lgd_fs)
    
            if show_title:
                ax1.set_title(f"{metric.upper()} spatial mean vs median (unified grid)", fontsize=ttl_fs)
    
            fig1.tight_layout()
            figs["mean_median"] = (fig1, ax1)
    
            # Plot 2: mean-median with CI
            fig2, ax2 = plt.subplots(figsize=diff_figsize)
            diff = df_out["mean_minus_median"].values
            lo = df_out["diff_ci_low"].values
            hi = df_out["diff_ci_high"].values
    
            ax2.axhline(0.0, linewidth=1.5)
            ax2.plot(x_pos, diff, marker="o", markersize=marker_size, linewidth=line_width,
                     label="Mean − Median")
            ax2.fill_between(x_pos, lo, hi, alpha=0.25, label=f"{int(ci*100)}% bootstrap CI")
    
            ax2.set_xlabel(xlabel, fontsize=lbl_fs)
            ax2.set_ylabel("Mean − Median", fontsize=lbl_fs)
            ax2.grid(True, **grid_kwargs)
            ax2.tick_params(axis="both", labelsize=tck_fs)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(x_labels, rotation=xrotation,
                                ha="right" if x_ticks != "version" else "center")
            ax2.legend(loc=legend_loc, fontsize=lgd_fs)
    
            if show_title:
                ax2.set_title(f"{metric.upper()} mean−median difference (unified grid)", fontsize=ttl_fs)
    
            fig2.tight_layout()
            figs["diff_ci"] = (fig2, ax2)
    
        # ---------------- save outputs ----------------
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "unified_grid_analysis"
            out_dir.mkdir(parents=True, exist_ok=True)
    
            # save table
            csv_path = out_dir / f"{self.event_id}_unified_{metric}_mean_median_summary.csv"
            df_out.to_csv(csv_path)
            logging.info(f"Saved mean/median summary CSV to {csv_path}")
    
            # save figs
            for key, (fig, ax) in figs.items():
                for fmt in save_formats:
                    p = out_dir / f"{self.event_id}_unified_{metric}_{key}.{fmt}"
                    fig.savefig(p, dpi=dpi, bbox_inches="tight")
                    logging.info(f"Saved figure to {p}")
    
        # ---------------- cleanup ----------------
        if close_figs and make_plots:
            for fig, ax in figs.values():
                plt.close(fig)
    
        return df_out, figs
    

    def analyze_spatial_change_concentration(
        self,
        version_list: list = None,
        metric: str = "mmi",
        use_cache: bool = True,
        delta_columns: list = None,
        tol: float = 1e-4,
        top_fracs: tuple = (0.01, 0.05, 0.10),
        x_ticks: str = "version",  # "version", "TaE_h", "TaE_d"
        xrotation: int = 45,
        output_path: str = None,
        save_csv: bool = True,
        csv_name: str = None,
        make_plots: bool = True,
        plot_key_fracs: tuple = (0.05, 0.10),  # which top_fracs to plot
        save_formats: list = ("png", "pdf"),
        dpi: int = 300,
        show_title: bool = True,
        # ---- styling kwargs (similar to other SHAKEtime plotters) ----
        figsize: tuple = (14, 5),
        font_sizes: dict = None,  # {"labels":12,"ticks":10,"title":14,"legend":10}
        grid: bool = True,
        grid_kwargs: dict = None,
        legend_loc: str = "best",
        show: bool = False,
        close_figs: bool = True,
    ):
        """
        Spatial change concentration index on unified-grid delta fields.
    
        For each delta column (e.g., delta_012_011_mmi), compute how concentrated the
        total absolute change is within the top X% of grid cells.
    
        Outputs per delta:
          - N cells (finite)
          - changed_frac (|delta| > tol)
          - abs_sum = sum(|delta|)
          - top_{p}_share for p in top_fracs: share of abs_sum contributed by top p fraction of cells
          - concentration_slope (optional simple indicator): top_5% share / 0.05
    
        Returns
        -------
        df_conc : pd.DataFrame indexed by delta column
        figs    : dict of matplotlib figures (if make_plots)
        """
        import numpy as np
        import pandas as pd
        import logging
        import matplotlib.pyplot as plt
        from pathlib import Path
    
        # ---- defaults ----
        if font_sizes is None:
            font_sizes = {"labels": 12, "ticks": 10, "title": 14, "legend": 10}
        lbl_fs = font_sizes.get("labels", 12)
        tck_fs = font_sizes.get("ticks", 10)
        ttl_fs = font_sizes.get("title", 14)
        lgd_fs = font_sizes.get("legend", 10)
    
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.4}
    
        # ---- unified grid ----
        if version_list is None and not (hasattr(self, "_unified_grid") and self._unified_grid is not None):
            raise ValueError("Provide version_list or ensure unified grid is already cached.")
        ug = self.get_unified_grid(version_list=version_list, metric=metric, use_cache=use_cache)
    
        if ug is None or ug.empty:
            logging.error("Unified grid is empty; cannot analyze concentration.")
            return None, {}
    
        # ---- pick delta columns ----
        if delta_columns is None:
            # Prefer metric-specific delta columns if present
            metric_suffix = f"_{metric}"
            delta_columns = [c for c in ug.columns if c.startswith("delta_") and c.endswith(metric_suffix)]
            if not delta_columns:
                # fallback to all deltas
                delta_columns = [c for c in ug.columns if c.startswith("delta_")]
    
        if not delta_columns:
            raise ValueError("No delta columns found in unified grid.")
    
        # ---- helper: parse versions from delta name delta_012_011_mmi ----
        def _parse_delta_name(colname: str):
            parts = colname.split("_")
            # expected: delta, new, old, metric...
            v_new = parts[1] if len(parts) > 2 else None
            v_old = parts[2] if len(parts) > 2 else None
            return v_new, v_old
    
        # ---- compute concentration metrics ----
        top_fracs = tuple(sorted(set(float(f) for f in top_fracs)))
        rows = []
    
        for col in delta_columns:
            s = pd.to_numeric(ug[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            n = int(s.shape[0])
    
            if n == 0:
                v_new, v_old = _parse_delta_name(col)
                row = {"delta_col": col, "v_new": v_new, "v_old": v_old, "n_cells": 0,
                       "changed_frac": np.nan, "abs_sum": np.nan}
                for f in top_fracs:
                    row[f"top_{int(100*f)}pct_share"] = np.nan
                rows.append(row)
                continue
    
            vals = s.to_numpy()
            absvals = np.abs(vals)
    
            # changed fraction with tolerance
            changed_frac = float(np.mean(absvals > tol))
            abs_sum = float(absvals.sum())
    
            # if all-zero (within tol) -> shares are undefined; set to 0
            if abs_sum <= 0.0:
                shares = {f: 0.0 for f in top_fracs}
            else:
                # Sort descending abs change
                sorted_abs = np.sort(absvals)[::-1]
                cumsum = np.cumsum(sorted_abs)
    
                shares = {}
                for f in top_fracs:
                    k = max(1, int(np.ceil(f * n)))
                    share = float(cumsum[k - 1] / abs_sum)
                    shares[f] = share
    
            v_new, v_old = _parse_delta_name(col)
            row = {
                "delta_col": col,
                "v_new": v_new,
                "v_old": v_old,
                "n_cells": n,
                "changed_frac": changed_frac,
                "abs_sum": abs_sum,
            }
            for f, share in shares.items():
                row[f"top_{int(100*f)}pct_share"] = share
    
            # simple interpretability indicator (optional)
            if 0.05 in shares:
                row["concentration_slope_top5"] = shares[0.05] / 0.05 if 0.05 > 0 else np.nan
            rows.append(row)
    
        df_conc = pd.DataFrame(rows).set_index("delta_col")
    
        # ---- order for plotting (try by old/new versions) ----
        # if v_old/v_new parseable, order by v_new then v_old; else leave
        try:
            df_conc["_vnew_int"] = pd.to_numeric(df_conc["v_new"], errors="coerce")
            df_conc = df_conc.sort_values(["_vnew_int"]).drop(columns=["_vnew_int"])
        except Exception:
            pass
    
        # ---- x labels via summary ----
        x_labels = df_conc.index.tolist()
        xlabel = "Delta column"
        if x_ticks in ("version", "TaE_h", "TaE_d"):
            if x_ticks == "version":
                # show "old→new" labels if possible
                lab = []
                for col in df_conc.index:
                    v_new, v_old = _parse_delta_name(col)
                    lab.append(f"{v_old}→{v_new}" if (v_old and v_new) else col)
                x_labels = lab
                xlabel = "Version step"
            else:
                # Use the summary to label by time after event at v_new (if available)
                try:
                    summary = self.get_dataframe().copy()
                    if "version" in summary.columns and x_ticks in summary.columns:
                        summary["version"] = summary["version"].astype(str).str.zfill(3)
                        summary = summary.set_index("version")
                        lab = []
                        for col in df_conc.index:
                            v_new, _ = _parse_delta_name(col)
                            if v_new in summary.index:
                                val = pd.to_numeric(summary.loc[v_new, x_ticks], errors="coerce")
                                lab.append("" if pd.isna(val) else f"{float(val):.1f}")
                            else:
                                lab.append("")
                        x_labels = lab
                        xlabel_map = {"TaE_h": "Time After Event (hours)",
                                      "TaE_d": "Time After Event (days)"}
                        xlabel = xlabel_map.get(x_ticks, x_ticks)
                except Exception as e:
                    logging.warning(f"[concentration] could not derive x tick labels from summary: {e}")
    
        # ---- plotting ----
        figs = {}
        if make_plots:
            plot_fracs = tuple(f for f in plot_key_fracs if f in top_fracs)
            if not plot_fracs:
                plot_fracs = (top_fracs[0],)
    
            fig, ax = plt.subplots(figsize=figsize)
    
            x = np.arange(len(df_conc))
            # plot one or more concentration shares
            for f in plot_fracs:
                col = f"top_{int(100*f)}pct_share"
                ax.plot(x, df_conc[col].values, marker="o", label=f"Top {int(100*f)}% share")
    
            # overlay changed fraction
            ax.plot(x, df_conc["changed_frac"].values, marker="s", linestyle="--", label=f"Changed frac (|Δ|>{tol:g})")
    
            ax.set_xlabel(xlabel, fontsize=lbl_fs)
            ax.set_ylabel("Share / Fraction", fontsize=lbl_fs)
            ax.tick_params(axis="both", labelsize=tck_fs)
    
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=xrotation, ha="right")
    
            if grid:
                ax.grid(True, **grid_kwargs)
    
            ax.legend(loc=legend_loc, fontsize=lgd_fs)
    
            if show_title:
                ax.set_title("Spatial change concentration & changed fraction (unified grid deltas)", fontsize=ttl_fs)
    
            fig.tight_layout()
            figs["concentration"] = (fig, ax)
    
        # ---- saving ----
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "unified_grid_analysis"
            out_dir.mkdir(parents=True, exist_ok=True)
    
            if save_csv:
                if csv_name is None:
                    csv_name = f"{self.event_id}_delta_change_concentration_{metric}.csv"
                csv_path = out_dir / csv_name
                df_conc.to_csv(csv_path)
                logging.info(f"Saved concentration table to {csv_path}")
    
            for key, (fig, ax) in figs.items():
                for fmt in save_formats:
                    p = out_dir / f"{self.event_id}_{key}_change_concentration_{metric}.{fmt}"
                    fig.savefig(p, dpi=dpi, bbox_inches="tight")
                    logging.info(f"Saved figure to {p}")
    
        if show:
            import matplotlib.pyplot as plt
            plt.show()
    
        if close_figs:
            import matplotlib.pyplot as plt
            for fig, _ in figs.values():
                plt.close(fig)
    
        return df_conc, figs
    

    def analyze_directional_change(
        self,
        version_list: list = None,
        metric: str = "mmi",
        use_cache: bool = True,
        delta_columns: list = None,
        tol: float = 1e-4,
        x_ticks: str = "version",  # "version", "TaE_h", "TaE_d"
        xrotation: int =45,
        output_path: str = None,
        save_csv: bool = True,
        csv_name: str = None,
        make_plots: bool = True,
        save_formats: list = ("png", "pdf"),
        dpi: int = 300,
        show_title: bool = True,
        # ---- styling kwargs ----
        figsize: tuple = (14, 5),
        font_sizes: dict = None,
        grid: bool = True,
        grid_kwargs: dict = None,
        legend_loc: str = "best",
        show: bool = False,
        close_figs: bool = True,
    ):
        """
        Directional stability / sign-consistency assessment on unified-grid deltas.
    
        Per delta column, computes:
          - pos_frac, neg_frac, zero_frac (based on tol)
          - signed_sum, abs_sum, directionality_ratio = signed_sum / abs_sum
          - mean_delta, median_delta
    
        This answers: do updates mostly increase shaking, decrease it, or refine symmetrically?
        """
        import numpy as np
        import pandas as pd
        import logging
        import matplotlib.pyplot as plt
        from pathlib import Path
    
        # ---- defaults ----
        if font_sizes is None:
            font_sizes = {"labels": 12, "ticks": 10, "title": 14, "legend": 10}
        lbl_fs = font_sizes.get("labels", 12)
        tck_fs = font_sizes.get("ticks", 10)
        ttl_fs = font_sizes.get("title", 14)
        lgd_fs = font_sizes.get("legend", 10)
    
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.4}
    
        # ---- unified grid ----
        if version_list is None and not (hasattr(self, "_unified_grid") and self._unified_grid is not None):
            raise ValueError("Provide version_list or ensure unified grid is already cached.")
        ug = self.get_unified_grid(version_list=version_list, metric=metric, use_cache=use_cache)
    
        if ug is None or ug.empty:
            logging.error("Unified grid is empty; cannot analyze directionality.")
            return None, {}
    
        # ---- pick delta columns ----
        if delta_columns is None:
            metric_suffix = f"_{metric}"
            delta_columns = [c for c in ug.columns if c.startswith("delta_") and c.endswith(metric_suffix)]
            if not delta_columns:
                delta_columns = [c for c in ug.columns if c.startswith("delta_")]
    
        if not delta_columns:
            raise ValueError("No delta columns found in unified grid.")
    
        def _parse_delta_name(colname: str):
            parts = colname.split("_")
            v_new = parts[1] if len(parts) > 2 else None
            v_old = parts[2] if len(parts) > 2 else None
            return v_new, v_old
    
        rows = []
        for col in delta_columns:
            s = pd.to_numeric(ug[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            n = int(s.shape[0])
    
            v_new, v_old = _parse_delta_name(col)
    
            if n == 0:
                rows.append({
                    "delta_col": col, "v_new": v_new, "v_old": v_old, "n_cells": 0,
                    "pos_frac": np.nan, "neg_frac": np.nan, "zero_frac": np.nan,
                    "signed_sum": np.nan, "abs_sum": np.nan,
                    "directionality_ratio": np.nan,
                    "mean_delta": np.nan, "median_delta": np.nan
                })
                continue
    
            vals = s.to_numpy()
            pos = vals > tol
            neg = vals < -tol
            zer = (~pos) & (~neg)
    
            pos_frac = float(np.mean(pos))
            neg_frac = float(np.mean(neg))
            zero_frac = float(np.mean(zer))
    
            signed_sum = float(vals.sum())
            abs_sum = float(np.abs(vals).sum())
            ratio = float(signed_sum / abs_sum) if abs_sum > 0 else 0.0
    
            rows.append({
                "delta_col": col,
                "v_new": v_new,
                "v_old": v_old,
                "n_cells": n,
                "pos_frac": pos_frac,
                "neg_frac": neg_frac,
                "zero_frac": zero_frac,
                "signed_sum": signed_sum,
                "abs_sum": abs_sum,
                "directionality_ratio": ratio,
                "mean_delta": float(np.mean(vals)),
                "median_delta": float(np.median(vals)),
            })
    
        df_dir = pd.DataFrame(rows).set_index("delta_col")
    
        # order (attempt by new version)
        try:
            df_dir["_vnew_int"] = pd.to_numeric(df_dir["v_new"], errors="coerce")
            df_dir = df_dir.sort_values(["_vnew_int"]).drop(columns=["_vnew_int"])
        except Exception:
            pass
    
        # x labels
        x_labels = df_dir.index.tolist()
        xlabel = "Delta column"
        if x_ticks == "version":
            lab = []
            for col in df_dir.index:
                v_new, v_old = _parse_delta_name(col)
                lab.append(f"{v_old}→{v_new}" if (v_old and v_new) else col)
            x_labels = lab
            xlabel = "Version step"
        elif x_ticks in ("TaE_h", "TaE_d"):
            try:
                summary = self.get_dataframe().copy()
                if "version" in summary.columns and x_ticks in summary.columns:
                    summary["version"] = summary["version"].astype(str).str.zfill(3)
                    summary = summary.set_index("version")
                    lab = []
                    for col in df_dir.index:
                        v_new, _ = _parse_delta_name(col)
                        if v_new in summary.index:
                            val = pd.to_numeric(summary.loc[v_new, x_ticks], errors="coerce")
                            lab.append("" if pd.isna(val) else f"{float(val):.1f}")
                        else:
                            lab.append("")
                    x_labels = lab
                    xlabel_map = {"TaE_h": "Time After Event (hours)",
                                  "TaE_d": "Time After Event (days)"}
                    xlabel = xlabel_map.get(x_ticks, x_ticks)
            except Exception as e:
                import logging
                logging.warning(f"[directional] could not derive x tick labels from summary: {e}")
    
        figs = {}
        if make_plots:
            x = np.arange(len(df_dir))
    
            # Plot 1: directionality ratio
            fig1, ax1 = plt.subplots(figsize=figsize)
            ax1.axhline(0.0, linewidth=1.5)
            ax1.plot(x, df_dir["directionality_ratio"].values, marker="o", label="signed_sum / abs_sum")
            ax1.set_xlabel(xlabel, fontsize=lbl_fs)
            ax1.set_ylabel("Directionality ratio [-1..+1]", fontsize=lbl_fs)
            ax1.tick_params(axis="both", labelsize=tck_fs)
            ax1.set_xticks(x)
            ax1.set_xticklabels(x_labels, rotation=xrotation, ha="right")
            if grid:
                ax1.grid(True, **grid_kwargs)
            ax1.legend(loc=legend_loc, fontsize=lgd_fs)
            if show_title:
                ax1.set_title(f"Directional stability of Δ{metric.upper()} (tol={tol:g})", fontsize=ttl_fs)
            fig1.tight_layout()
            figs["directionality_ratio"] = (fig1, ax1)
    
            # Plot 2: sign fractions (stacked)
            fig2, ax2 = plt.subplots(figsize=figsize)
            pos = df_dir["pos_frac"].values
            neg = df_dir["neg_frac"].values
            zer = df_dir["zero_frac"].values
    
            ax2.bar(x, neg, label="Δ < -tol")
            ax2.bar(x, zer, bottom=neg, label="|Δ| ≤ tol")
            ax2.bar(x, pos, bottom=neg + zer, label="Δ > +tol")
    
            ax2.set_xlabel(xlabel, fontsize=lbl_fs)
            ax2.set_ylabel("Fraction of cells", fontsize=lbl_fs)
            ax2.tick_params(axis="both", labelsize=tck_fs)
            ax2.set_xticks(x)
            ax2.set_xticklabels(x_labels, rotation=xrotation, ha="right")
            if grid:
                ax2.grid(True, axis="y", **grid_kwargs)
            ax2.legend(loc=legend_loc, fontsize=lgd_fs)
            if show_title:
                ax2.set_title(f"Sign consistency of Δ{metric.upper()} (tol={tol:g})", fontsize=ttl_fs)
            fig2.tight_layout()
            figs["sign_fractions"] = (fig2, ax2)
    
        # saving
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "unified_grid_analysis"
            out_dir.mkdir(parents=True, exist_ok=True)
    
            if save_csv:
                if csv_name is None:
                    csv_name = f"{self.event_id}_delta_directional_change_{metric}.csv"
                csv_path = out_dir / csv_name
                df_dir.to_csv(csv_path)
                logging.info(f"Saved directional change table to {csv_path}")
    
            for key, (fig, ax) in figs.items():
                for fmt in save_formats:
                    p = out_dir / f"{self.event_id}_{key}_directional_change_{metric}.{fmt}"
                    fig.savefig(p, dpi=dpi, bbox_inches="tight")
                    logging.info(f"Saved figure to {p}")
    
        if show:
            import matplotlib.pyplot as plt
            plt.show()
    
        if close_figs:
            import matplotlib.pyplot as plt
            for fig, _ in figs.values():
                plt.close(fig)
    
        return df_dir, figs
    
    
    def analyze_quantile_convergence(
        self,
        version_list: list,
        metric: str = "mmi",
        use_cache: bool = True,
        quantiles: tuple = (0.05, 0.50, 0.95),
        x_ticks: str = "version",  # "version", "TaE_h", "TaE_d"
        xrotation: int = 45,
        output_path: str = None,
        save_csv: bool = True,
        csv_name: str = None,
        make_plots: bool = True,
        save_formats: list = ("png", "pdf"),
        dpi: int = 300,
        show_title: bool = True,
        # ---- styling kwargs ----
        figsize: tuple = (14, 5),
        font_sizes: dict = None,
        grid: bool = True,
        grid_kwargs: dict = None,
        legend_loc: str = "best",
        show: bool = False,
        close_figs: bool = True,
    ):
        """
        Quantile-based convergence (robust convergence) for unified-grid metric fields.
    
        For each version's metric grid (e.g., mmi_v001), computes quantiles (q05, q50, q95 ...)
        and their distance to the final version's quantiles.
    
        Returns
        -------
        df_q   : pd.DataFrame indexed by version (strings zfilled)
        figs   : dict of figures (if make_plots)
        """
        import numpy as np
        import pandas as pd
        import logging
        import matplotlib.pyplot as plt
        from pathlib import Path
    
        if font_sizes is None:
            font_sizes = {"labels": 12, "ticks": 10, "title": 14, "legend": 10}
        lbl_fs = font_sizes.get("labels", 12)
        tck_fs = font_sizes.get("ticks", 10)
        ttl_fs = font_sizes.get("title", 14)
        lgd_fs = font_sizes.get("legend", 10)
    
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.4}
    
        # unified grid
        ug = self.get_unified_grid(version_list=version_list, metric=metric, use_cache=use_cache)
        if ug is None or ug.empty:
            logging.error("Unified grid is empty; cannot analyze quantile convergence.")
            return None, {}
    
        versions = [str(v).zfill(3) for v in version_list]
        metric_cols = [f"{metric}_v{v}" for v in versions if f"{metric}_v{v}" in ug.columns]
        if not metric_cols:
            raise ValueError(f"No metric columns like '{metric}_v###' found for provided versions.")
    
        qs = tuple(float(q) for q in quantiles)
        for q in qs:
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"Quantile {q} out of [0,1].")
    
        # compute quantiles per version
        rows = []
        for v in versions:
            col = f"{metric}_v{v}"
            if col not in ug.columns:
                continue
            s = pd.to_numeric(ug[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                row = {"version": v, "n_cells": 0}
                for q in qs:
                    row[f"q{int(q*100):02d}"] = np.nan
                rows.append(row)
                continue
    
            row = {"version": v, "n_cells": int(s.shape[0])}
            for q in qs:
                row[f"q{int(q*100):02d}"] = float(s.quantile(q))
            rows.append(row)
    
        df_q = pd.DataFrame(rows).set_index("version").sort_index()
    
        # distances to final version quantiles
        final_v = df_q.index[-1]
        for q in qs:
            c = f"q{int(q*100):02d}"
            df_q[f"{c}_minus_final"] = df_q[c] - df_q.loc[final_v, c]
            df_q[f"abs_{c}_minus_final"] = (df_q[c] - df_q.loc[final_v, c]).abs()
    
        # x-axis labels
        x_pos = np.arange(len(df_q))
        x_labels = df_q.index.tolist()
        xlabel = "Version"
    
        if x_ticks != "version":
            try:
                summary = self.get_dataframe().copy()
                if "version" in summary.columns and x_ticks in summary.columns:
                    summary["version"] = summary["version"].astype(str).str.zfill(3)
                    summary = summary.set_index("version")
                    vals = pd.to_numeric(summary.loc[df_q.index, x_ticks], errors="coerce")
                    x_labels = ["" if pd.isna(v) else f"{float(v):.1f}" for v in vals]
                    xlabel_map = {"TaE_h": "Time After Event (hours)",
                                  "TaE_d": "Time After Event (days)"}
                    xlabel = xlabel_map.get(x_ticks, x_ticks)
            except Exception as e:
                logging.warning(f"[quantiles] could not derive x tick labels from summary: {e}")
    
        figs = {}
        if make_plots:
            # Plot 1: quantile trajectories
            fig1, ax1 = plt.subplots(figsize=figsize)
            for q in qs:
                c = f"q{int(q*100):02d}"
                ax1.plot(x_pos, df_q[c].values, marker="o", label=c)
    
            ax1.set_xlabel(xlabel, fontsize=lbl_fs)
            ax1.set_ylabel(metric.upper(), fontsize=lbl_fs)
            ax1.tick_params(axis="both", labelsize=tck_fs)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(x_labels, rotation=xrotation if x_ticks != "version" else 0,
                                ha="right" if x_ticks != "version" else "center")
            if grid:
                ax1.grid(True, **grid_kwargs)
            ax1.legend(loc=legend_loc, fontsize=lgd_fs)
            if show_title:
                ax1.set_title(f"{metric.upper()} quantile evolution (unified grid)", fontsize=ttl_fs)
            fig1.tight_layout()
            figs["quantile_trajectories"] = (fig1, ax1)
    
            # Plot 2: absolute distance to final
            fig2, ax2 = plt.subplots(figsize=figsize)
            for q in qs:
                c = f"abs_q{int(q*100):02d}_minus_final"
                ax2.plot(x_pos, df_q[c].values, marker="o", label=c.replace("abs_", "|") + "|")
    
            ax2.set_xlabel(xlabel, fontsize=lbl_fs)
            ax2.set_ylabel(f"|Quantile − final| ({metric.upper()})", fontsize=lbl_fs)
            ax2.tick_params(axis="both", labelsize=tck_fs)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(x_labels, rotation=xrotation if x_ticks != "version" else 0,
                                ha="right" if x_ticks != "version" else "center")
            if grid:
                ax2.grid(True, **grid_kwargs)
            ax2.legend(loc=legend_loc, fontsize=lgd_fs)
            if show_title:
                ax2.set_title(f"{metric.upper()} quantile convergence to final version {final_v}", fontsize=ttl_fs)
            fig2.tight_layout()
            figs["quantile_convergence_to_final"] = (fig2, ax2)
    
        # saving
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "unified_grid_analysis"
            out_dir.mkdir(parents=True, exist_ok=True)
    
            if save_csv:
                if csv_name is None:
                    csv_name = f"{self.event_id}_quantile_convergence_{metric}.csv"
                csv_path = out_dir / csv_name
                df_q.to_csv(csv_path)
                logging.info(f"Saved quantile convergence table to {csv_path}")
    
            for key, (fig, ax) in figs.items():
                for fmt in save_formats:
                    p = out_dir / f"{self.event_id}_{key}_quantile_convergence_{metric}.{fmt}"
                    fig.savefig(p, dpi=dpi, bbox_inches="tight")
                    logging.info(f"Saved figure to {p}")
    
        if show:
            import matplotlib.pyplot as plt
            plt.show()
    
        if close_figs:
            import matplotlib.pyplot as plt
            for fig, _ in figs.values():
                plt.close(fig)
    
        return df_q, figs
    



    #########################################################
    #
    #
    #
    # Data Influcene V26.1
    #
    #
    #
    #
    ###################################################3
    def plot_data_availability_timeseries(
        self,
        version_list: list,
        metric: str = "mmi",
        use_cache: bool = True,
        x_ticks: str = "version",        # "version", "TaE_h", "TaE_d"
        # Optional CDI/DCI dataframe
        cdi_df: "pd.DataFrame" = None,
        cdi_version_col: str = "version",  # if present, we group by this
        cdi_value_col: str = None,         # optional filter, e.g. "cdi" (not required)
        cdi_min_value: float = None,       # optional threshold on cdi_value_col
        # tolerance / robustness
        zfill_versions: bool = True,
        output_path: str = None,
        save_formats: list = ("png", "pdf"),
        dpi: int = 300,
        show_title: bool = True,
        title: str = None,
        # ---- NEW: axis label + scaling + ticks ----
        xlabel: str = None,
        ylabel: str = None,
        xscale: str = "linear",            # "linear" or "log"
        yscale: str = "log",               # default stays log like before
        xticks: list = None,
        xticklabels: list = None,
        yticks: list = None,
        yticklabels: list = None,
        # ---- styling ----
        figsize: tuple = (14, 6),
        font_sizes: dict = None,          # {"labels":12,"ticks":10,"title":14,"legend":10}
        legend_loc: str = "best",
        grid: bool = True,
        grid_kwargs: dict = None,
        line_width: float = 2.5,
        marker_size: float = 8,
        xrotation: int = 45,
        # ---- behavior ----
        show: bool = False,
        close_figs: bool = True,
        return_df: bool = True,
    ):
        """
        Figure 1: data-availability time series.
        Plots station_count, dyfi_count (used in ShakeMap), trace_length_km
        and optional dyfi_cdi_count from user-provided CDI/DCI dataframe.
    
        New options:
          - xlabel/ylabel override
          - xscale/yscale ("linear" or "log")
          - custom ticks/labels for x and y (xticks/xticklabels/yticks/yticklabels)
    
        Returns
        -------
        (fig, ax) and optionally df_used.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import logging
        from pathlib import Path
    
        # defaults
        if font_sizes is None:
            font_sizes = {"labels": 12, "ticks": 10, "title": 14, "legend": 10}
        lbl_fs = font_sizes.get("labels", 12)
        tck_fs = font_sizes.get("ticks", 10)
        ttl_fs = font_sizes.get("title", 14)
        lgd_fs = font_sizes.get("legend", 10)
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.4}
    
        # compute aux/diag using existing engine
        res = self.analyze_auxiliary_influences(
            version_list=version_list,
            metric=metric,
            use_cache=use_cache
        )
        if not res or "merged" not in res or res["merged"] is None or res["merged"].empty:
            logging.error("analyze_auxiliary_influences returned empty merged dataframe.")
            return (None, None, None) if return_df else (None, None)
    
        df = res["merged"].copy()
    
        # normalize versions
        if "version" in df.columns:
            df["version"] = df["version"].astype(str)
            if zfill_versions:
                df["version"] = df["version"].str.zfill(3)
            df = df.set_index("version", drop=False)
        else:
            # if already indexed by version
            df.index = df.index.astype(str)
            if zfill_versions:
                df.index = df.index.str.zfill(3)
            df["version"] = df.index
    
        # order by provided version_list
        vlist = [str(v) for v in version_list]
        if zfill_versions:
            vlist = [v.zfill(3) for v in vlist]
        df = df.loc[[v for v in vlist if v in df.index]].copy()
    
        # build x labels (default behavior, unless user passes xticks/xticklabels)
        x = np.arange(len(df))
        x_labels = df.index.tolist()
        xlab_default = "Version"
        if x_ticks in ("TaE_h", "TaE_d") and x_ticks in df.columns:
            vals = pd.to_numeric(df[x_ticks], errors="coerce")
            x_labels = [("" if pd.isna(v) else f"{float(v):.1f}") for v in vals]
            xlab_default = "Time After Event (hours)" if x_ticks == "TaE_h" else "Time After Event (days)"
    
        # optional CDI counts
        if cdi_df is not None:
            try:
                cdi_work = cdi_df.copy()
                if cdi_version_col in cdi_work.columns:
                    cdi_work[cdi_version_col] = cdi_work[cdi_version_col].astype(str)
                    if zfill_versions:
                        cdi_work[cdi_version_col] = cdi_work[cdi_version_col].str.zfill(3)
    
                    if (cdi_value_col is not None) and (cdi_value_col in cdi_work.columns) and (cdi_min_value is not None):
                        cdi_work = cdi_work[pd.to_numeric(cdi_work[cdi_value_col], errors="coerce") >= float(cdi_min_value)]
    
                    cdi_counts = cdi_work.groupby(cdi_version_col).size()
                    df["dyfi_cdi_count"] = [int(cdi_counts.get(v, 0)) for v in df.index]
                else:
                    logging.warning(f"cdi_df provided but column '{cdi_version_col}' not found. Skipping CDI count.")
            except Exception as e:
                logging.warning(f"Failed to compute dyfi_cdi_count from cdi_df: {e}")
    
        # series to plot
        series_map = [
            ("station_count", "Stations"),
            ("dyfi_count", "DYFI used"),
            ("trace_length_km", "Trace length (km)"),
        ]
        if "dyfi_cdi_count" in df.columns:
            series_map.append(("dyfi_cdi_count", "DYFI total (CDI/DCI)"))
    
        # plot
        fig, ax = plt.subplots(figsize=figsize)
    
        # NEW: axis scaling
        if xscale not in ("linear", "log"):
            raise ValueError(f"xscale must be 'linear' or 'log', got: {xscale}")
        if yscale not in ("linear", "log"):
            raise ValueError(f"yscale must be 'linear' or 'log', got: {yscale}")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
    
        for col, label in series_map:
            if col in df.columns:
                y = pd.to_numeric(df[col], errors="coerce").to_numpy()
                ax.plot(
                    x, y,
                    marker="o",
                    markersize=marker_size,
                    linewidth=line_width,
                    label=label
                )
    
        # NEW: label overrides
        ax.set_xlabel(xlabel if xlabel is not None else xlab_default, fontsize=lbl_fs)
        ax.set_ylabel(ylabel if ylabel is not None else ("Count / Length" + (" (log scale)" if yscale == "log" else "")),
                      fontsize=lbl_fs)
    
        ax.tick_params(axis="both", labelsize=tck_fs)
    
        # X ticks: allow full override
        if xticks is not None:
            ax.set_xticks(xticks)
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels, rotation=xrotation, ha="right")
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels if xticklabels is None else xticklabels,
                               rotation=xrotation, ha="right")
    
        # Y ticks: allow full override
        if yticks is not None:
            ax.set_yticks(yticks)
            if yticklabels is not None:
                ax.set_yticklabels(yticklabels)
    
        # keep your nice log tick formatting only if y is log AND user didn't override yticks
        if yscale == "log" and yticks is None:
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    
        if grid:
            ax.grid(True, which="both" if (xscale == "log" or yscale == "log") else "major", **grid_kwargs)
        ax.legend(loc=legend_loc, fontsize=lgd_fs)
    
        if show_title:
            ax.set_title(title or "Data availability per SHAKEmap version", fontsize=ttl_fs)
    
        fig.tight_layout()
    
        # save
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "data_influence"
            out_dir.mkdir(parents=True, exist_ok=True)
            for fmt in save_formats:
                p = out_dir / f"{self.event_id}_data_availability_timeseries.{fmt}"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                logging.info(f"Saved data availability plot to {p}")
    
            if return_df:
                csvp = out_dir / f"{self.event_id}_data_availability_timeseries.csv"
                df.to_csv(csvp)
                logging.info(f"Saved data availability table to {csvp}")
    
        if show:
            plt.show()
        if close_figs:
            plt.close(fig)
    
        return (fig, ax, df) if return_df else (fig, ax)
    
    


    def plot_hazard_footprint_timeseries(
        self,
        version_list: list,
        metric: str = "mmi",
        use_cache: bool = True,
        x_ticks: str = "version",     # "version", "TaE_h", "TaE_d"
        zfill_versions: bool = True,
        output_path: str = None,
        save_formats: list = ("png", "pdf"),
        dpi: int = 300,
        show_title: bool = True,
        title: str = None,
        # ---- NEW: axis label + scaling + ticks ----
        xlabel: str = None,
        ylabel: str = None,
        xscale: str = "linear",       # "linear" or "log"
        yscale: str = "linear",       # hazard footprint usually linear, but now optional
        xticks: list = None,
        xticklabels: list = None,
        yticks: list = None,
        yticklabels: list = None,
        # ---- styling ----
        figsize: tuple = (14, 6),
        font_sizes: dict = None,
        legend_loc: str = "best",
        grid: bool = True,
        grid_kwargs: dict = None,
        line_width: float = 2.5,
        marker_size: float = 8,
        xrotation: int = 45,
        # ---- behavior ----
        show: bool = False,
        close_figs: bool = True,
        return_df: bool = True,
    ):
        """
        Figure 2: global hazard footprint time series.
        Plots uncertainty area (90% percentile) and exceedance areas for MMI >= 6, 7, 8.
    
        New options:
          - xlabel/ylabel override
          - xscale/yscale ("linear" or "log")
          - custom ticks/labels for x and y (xticks/xticklabels/yticks/yticklabels)
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import logging
        from pathlib import Path
    
        if font_sizes is None:
            font_sizes = {"labels": 12, "ticks": 10, "title": 14, "legend": 10}
        lbl_fs = font_sizes.get("labels", 12)
        tck_fs = font_sizes.get("ticks", 10)
        ttl_fs = font_sizes.get("title", 14)
        lgd_fs = font_sizes.get("legend", 10)
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.4}
    
        res = self.analyze_auxiliary_influences(
            version_list=version_list,
            metric=metric,
            use_cache=use_cache
        )
        if not res or "merged" not in res or res["merged"] is None or res["merged"].empty:
            logging.error("analyze_auxiliary_influences returned empty merged dataframe.")
            return (None, None, None) if return_df else (None, None)
    
        df = res["merged"].copy()
    
        # normalize versions
        if "version" in df.columns:
            df["version"] = df["version"].astype(str)
            if zfill_versions:
                df["version"] = df["version"].str.zfill(3)
            df = df.set_index("version", drop=False)
        else:
            df.index = df.index.astype(str)
            if zfill_versions:
                df.index = df.index.str.zfill(3)
            df["version"] = df.index
    
        vlist = [str(v) for v in version_list]
        if zfill_versions:
            vlist = [v.zfill(3) for v in vlist]
        df = df.loc[[v for v in vlist if v in df.index]].copy()
    
        # x labels (default behavior, unless user overrides)
        x = np.arange(len(df))
        x_labels = df.index.tolist()
        xlab_default = "Version"
        if x_ticks in ("TaE_h", "TaE_d") and x_ticks in df.columns:
            vals = pd.to_numeric(df[x_ticks], errors="coerce")
            x_labels = [("" if pd.isna(v) else f"{float(v):.1f}") for v in vals]
            xlab_default = "Time After Event (hours)" if x_ticks == "TaE_h" else "Time After Event (days)"
    
        cols = [
            ("unc_area_pct90", "Uncertainty area (90% pct)"),
            ("area_exceed_6.0", "Area MMI ≥ 6"),
            ("area_exceed_7.0", "Area MMI ≥ 7"),
            ("area_exceed_8.0", "Area MMI ≥ 8"),
        ]
    
        fig, ax = plt.subplots(figsize=figsize)
    
        # NEW: axis scaling
        if xscale not in ("linear", "log"):
            raise ValueError(f"xscale must be 'linear' or 'log', got: {xscale}")
        if yscale not in ("linear", "log"):
            raise ValueError(f"yscale must be 'linear' or 'log', got: {yscale}")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
    
        for col, label in cols:
            if col in df.columns:
                y = pd.to_numeric(df[col], errors="coerce").to_numpy()
                ax.plot(
                    x, y,
                    marker="o",
                    markersize=marker_size,
                    linewidth=line_width,
                    label=label
                )
    
        # NEW: label overrides
        ax.set_xlabel(xlabel if xlabel is not None else xlab_default, fontsize=lbl_fs)
        ax.set_ylabel(ylabel if ylabel is not None else ("Area (km²)" + (" (log scale)" if yscale == "log" else "")),
                      fontsize=lbl_fs)
    
        ax.tick_params(axis="both", labelsize=tck_fs)
    
        # X ticks override
        if xticks is not None:
            ax.set_xticks(xticks)
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels, rotation=xrotation, ha="right")
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels if xticklabels is None else xticklabels,
                               rotation=xrotation, ha="right")
    
        # Y ticks override
        if yticks is not None:
            ax.set_yticks(yticks)
            if yticklabels is not None:
                ax.set_yticklabels(yticklabels)
    
        # Optional nice log formatting if y is log and user did not override yticks
        if yscale == "log" and yticks is None:
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    
        if grid:
            ax.grid(True, which="both" if (xscale == "log" or yscale == "log") else "major", **grid_kwargs)
        ax.legend(loc=legend_loc, fontsize=lgd_fs)
    
        if show_title:
            ax.set_title(title or "Hazard footprint & uncertainty evolution", fontsize=ttl_fs)
    
        fig.tight_layout()
    
        # save
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "data_influence"
            out_dir.mkdir(parents=True, exist_ok=True)
    
            for fmt in save_formats:
                p = out_dir / f"{self.event_id}_hazard_footprint_timeseries.{fmt}"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                logging.info(f"Saved hazard footprint plot to {p}")
    
            if return_df:
                csvp = out_dir / f"{self.event_id}_hazard_footprint_timeseries.csv"
                df.to_csv(csvp)
                logging.info(f"Saved hazard footprint table to {csvp}")
    
        if show:
            plt.show()
        if close_figs:
            plt.close(fig)
    
        return (fig, ax, df) if return_df else (fig, ax)

    
    def plot_update_magnitude_vs_data_increment(
        self,
        version_list: list,
        metric: str = "mmi",
        use_cache: bool = True,
        tol: float = 1e-4,
        # Optional CDI counts to include as predictor
        cdi_df: "pd.DataFrame" = None,
        cdi_version_col: str = "version",
        cdi_value_col: str = None,
        cdi_min_value: float = None,
        zfill_versions: bool = True,
        # which response measure from deltas
        response: str = "mean_abs_delta",  # "mean_abs_delta" or "sum_abs_delta"
        output_path: str = None,
        save_formats: list = ("png", "pdf"),
        dpi: int = 300,
        show_title: bool = True,
        title: str = None,
        # ---- styling ----
        figsize: tuple = (16, 5),
        font_sizes: dict = None,
        grid: bool = True,
        grid_kwargs: dict = None,
        marker_size: float = 60,   # scatter size
        alpha: float = 0.85,
        # ---- behavior ----
        show: bool = False,
        close_figs: bool = True,
        return_df: bool = True,
    ):
        """
        Scatter-based influence plot:
          x = Δ(data quantity) per version step
          y = global update magnitude per step (from unified-grid delta fields)
    
        Returns
        -------
        df_steps : per-step table with predictors and response
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import logging
        from pathlib import Path
    
        if font_sizes is None:
            font_sizes = {"labels": 12, "ticks": 10, "title": 14, "legend": 10}
        lbl_fs = font_sizes.get("labels", 12)
        tck_fs = font_sizes.get("ticks", 10)
        ttl_fs = font_sizes.get("title", 14)
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.4}
    
        # 1) get merged aux/diag for counts
        res = self.analyze_auxiliary_influences(version_list=version_list, metric=metric, use_cache=use_cache)
        if not res or "merged" not in res or res["merged"] is None or res["merged"].empty:
            logging.error("analyze_auxiliary_influences returned empty merged dataframe.")
            return (None, None) if not return_df else (None, None, None)
    
        dfm = res["merged"].copy()
        if "version" in dfm.columns:
            dfm["version"] = dfm["version"].astype(str)
            if zfill_versions:
                dfm["version"] = dfm["version"].str.zfill(3)
            dfm = dfm.set_index("version", drop=False)
        else:
            dfm.index = dfm.index.astype(str)
            if zfill_versions:
                dfm.index = dfm.index.str.zfill(3)
            dfm["version"] = dfm.index
    
        vlist = [str(v) for v in version_list]
        if zfill_versions:
            vlist = [v.zfill(3) for v in vlist]
        dfm = dfm.loc[[v for v in vlist if v in dfm.index]].copy()
    
        # add optional CDI counts as a predictor series
        if cdi_df is not None:
            try:
                cdi_work = cdi_df.copy()
                if cdi_version_col in cdi_work.columns:
                    cdi_work[cdi_version_col] = cdi_work[cdi_version_col].astype(str)
                    if zfill_versions:
                        cdi_work[cdi_version_col] = cdi_work[cdi_version_col].str.zfill(3)
    
                    if (cdi_value_col is not None) and (cdi_value_col in cdi_work.columns) and (cdi_min_value is not None):
                        cdi_work = cdi_work[pd.to_numeric(cdi_work[cdi_value_col], errors="coerce") >= float(cdi_min_value)]
    
                    counts = cdi_work.groupby(cdi_version_col).size()
                    dfm["dyfi_cdi_count"] = [int(counts.get(v, 0)) for v in dfm.index]
            except Exception as e:
                logging.warning(f"Failed to compute dyfi_cdi_count from cdi_df: {e}")
    
        # 2) compute response per step from unified-grid deltas
        ug = self.get_unified_grid(version_list=version_list, metric=metric, use_cache=use_cache)
        if ug is None or ug.empty:
            logging.error("Unified grid is empty; cannot compute delta response.")
            return (None, None) if not return_df else (None, None, None)
    
        # build step table
        steps = []
        for i in range(1, len(vlist)):
            v_old = vlist[i-1]
            v_new = vlist[i]
            dcol = f"delta_{v_new}_{v_old}_{metric}"
            if dcol not in ug.columns:
                # try generic naming (some builds omit metric suffix)
                dcol2 = f"delta_{v_new}_{v_old}_{metric}"
                dcol = dcol2
            if dcol not in ug.columns:
                # fallback: find any delta matching new/old and metric
                candidates = [c for c in ug.columns if c.startswith(f"delta_{v_new}_{v_old}") and (c.endswith(f"_{metric}") or c.endswith(metric))]
                dcol = candidates[0] if candidates else None
    
            if dcol is None or dcol not in ug.columns:
                continue
    
            s = pd.to_numeric(ug[dcol], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                continue
            vals = s.to_numpy()
            absvals = np.abs(vals)
            absvals = absvals[absvals > tol]  # ignore numerical noise
            if absvals.size == 0:
                mean_abs = 0.0
                sum_abs = 0.0
            else:
                mean_abs = float(absvals.mean())
                sum_abs = float(absvals.sum())
    
            steps.append({"v_old": v_old, "v_new": v_new, "delta_col": dcol,
                          "mean_abs_delta": mean_abs, "sum_abs_delta": sum_abs})
    
        df_steps = pd.DataFrame(steps)
        if df_steps.empty:
            logging.error("No delta steps found; cannot plot update magnitude vs increments.")
            return (None, None) if not return_df else (None, None, None)
    
        # 3) compute predictor increments per step
        predictors = ["station_count", "dyfi_count", "trace_length_km"]
        if "dyfi_cdi_count" in dfm.columns:
            predictors.append("dyfi_cdi_count")
    
        for p in predictors:
            if p not in dfm.columns:
                continue
            inc = []
            for _, r in df_steps.iterrows():
                v_old, v_new = r["v_old"], r["v_new"]
                a = pd.to_numeric(dfm.loc[v_old, p], errors="coerce")
                b = pd.to_numeric(dfm.loc[v_new, p], errors="coerce")
                inc.append(float(b - a) if (pd.notna(a) and pd.notna(b)) else np.nan)
            df_steps[f"delta_{p}"] = inc
    
        # 4) plot: one row of scatter panels (kept simple)
        fig, axes = plt.subplots(1, len([p for p in predictors if f"delta_{p}" in df_steps.columns]), figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
    
        y = df_steps[response].to_numpy()
        used_axes = 0
        for ax, p in zip(axes, [p for p in predictors if f"delta_{p}" in df_steps.columns]):
            x = df_steps[f"delta_{p}"].to_numpy()
            ax.scatter(x, y, s=marker_size, alpha=alpha)
            ax.set_xlabel(f"Δ{p}", fontsize=lbl_fs)
            ax.tick_params(axis="both", labelsize=tck_fs)
            if grid:
                ax.grid(True, **grid_kwargs)
            used_axes += 1
    
        axes[0].set_ylabel(response.replace("_", " "), fontsize=lbl_fs)
        if show_title:
            fig.suptitle(title or f"Update magnitude vs data increments ({metric.upper()})", fontsize=ttl_fs)
        fig.tight_layout()
    
        # save
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "data_influence"
            out_dir.mkdir(parents=True, exist_ok=True)
            for fmt in save_formats:
                p = out_dir / f"{self.event_id}_update_vs_data_increment_{metric}_{response}.{fmt}"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                logging.info(f"Saved update-vs-increment plot to {p}")
            if return_df:
                csvp = out_dir / f"{self.event_id}_update_vs_data_increment_{metric}_{response}.csv"
                df_steps.to_csv(csvp, index=False)
                logging.info(f"Saved update-vs-increment table to {csvp}")
    
        if show:
            plt.show()
        if close_figs:
            plt.close(fig)
    
        return (fig, axes, df_steps) if return_df else (fig, axes)


    def plot_standardized_data_influence(
        self,
        version_list: list,
        metric: str = "mmi",
        use_cache: bool = True,
        tol: float = 1e-4,
        response_set: tuple = ("mean_abs_delta", "sum_abs_delta"),
        # Optional CDI counts
        cdi_df: "pd.DataFrame" = None,
        cdi_version_col: str = "version",
        cdi_value_col: str = None,
        cdi_min_value: float = None,
        zfill_versions: bool = True,
        output_path: str = None,
        save_formats: list = ("png", "pdf"),
        dpi: int = 300,
        show_title: bool = True,
        title: str = None,
        # ---- styling ----
        figsize: tuple = (10, 6),
        font_sizes: dict = None,
        grid: bool = True,
        grid_kwargs: dict = None,
        # ---- behavior ----
        show: bool = False,
        close_figs: bool = True,
        return_df: bool = True,
    ):
        """
        Fit standardized OLS models:
          response ~ Δstation + Δdyfi + Δtrace (+ optional Δcdi)
        and plot absolute standardized coefficients (|beta|).
    
        Internally reuses plot_update_magnitude_vs_data_increment() to build the step table
        (predictor increments + update magnitude response).
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import logging
        from pathlib import Path
    
        try:
            import statsmodels.api as sm
        except Exception:
            sm = None
    
        # ---- defaults ----
        if font_sizes is None:
            font_sizes = {"labels": 12, "ticks": 10, "title": 14, "legend": 10}
        lbl_fs = font_sizes.get("labels", 12)
        tck_fs = font_sizes.get("ticks", 10)
        ttl_fs = font_sizes.get("title", 14)
    
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.4}
    
        # ---- build step table using existing function (NO make_plots kwarg!) ----
        # We accept that this helper function creates a figure internally;
        # passing close_figs=True ensures it won’t leak memory.
        try:
            _, _, df_steps = self.plot_update_magnitude_vs_data_increment(
                version_list=version_list,
                metric=metric,
                use_cache=use_cache,
                tol=tol,
                cdi_df=cdi_df,
                cdi_version_col=cdi_version_col,
                cdi_value_col=cdi_value_col,
                cdi_min_value=cdi_min_value,
                zfill_versions=zfill_versions,
                response="mean_abs_delta",   # just to build df_steps consistently
                output_path=None,
                return_df=True,
                close_figs=True
            )
        except TypeError as e:
            # If your plot_update_magnitude_vs_data_increment signature differs,
            # this gives a clearer message.
            logging.error(f"Failed to call plot_update_magnitude_vs_data_increment(): {e}")
            return (None, None, None) if return_df else (None, None)
    
        if df_steps is None or df_steps.empty:
            logging.error("df_steps unavailable/empty; cannot fit standardized influence.")
            return (None, None, None) if return_df else (None, None)
    
        # predictors (increments)
        pred_cols = [c for c in df_steps.columns if c.startswith("delta_")]
        # remove non-predictor delta column if present
        pred_cols = [c for c in pred_cols if c not in ("delta_col",)]
        if not pred_cols:
            logging.error("No predictor delta_* columns found in df_steps.")
            return (None, None, None) if return_df else (None, None)
    
        # standardize helper
        def zscore(series):
            s = pd.to_numeric(series, errors="coerce")
            mu = s.mean()
            sd = s.std()
            if pd.isna(sd) or sd == 0:
                return s * np.nan
            return (s - mu) / sd
    
        betas = {}
    
        for resp in response_set:
            if resp not in df_steps.columns:
                continue
    
            df_fit = df_steps[pred_cols + [resp]].copy()
            df_fit = df_fit.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(df_fit) < 3:
                logging.warning(f"Not enough rows to fit standardized influence for response='{resp}'.")
                continue
    
            Xz = df_fit[pred_cols].apply(zscore)
            yz = zscore(df_fit[resp])
    
            # drop any columns that became all-NaN after zscore
            good_cols = [c for c in Xz.columns if Xz[c].notna().any()]
            Xz = Xz[good_cols]
            if Xz.shape[1] == 0 or yz.notna().sum() < 3:
                logging.warning(f"Predictors collapsed after standardization for response='{resp}'.")
                continue
    
            # Fit
            if sm is not None:
                X = sm.add_constant(Xz)
                model = sm.OLS(yz, X).fit()
                b = model.params.drop("const", errors="ignore").abs()
            else:
                # fallback: absolute correlation with standardized response
                b = Xz.apply(lambda col: abs(col.corr(yz)))
    
            betas[resp] = b
    
        if not betas:
            logging.error("No standardized influence results computed (insufficient data or predictors).")
            return (None, None, None) if return_df else (None, None)
    
        df_betas = pd.DataFrame(betas).T.fillna(0.0)
    
        # ---- plot ----
        fig, ax = plt.subplots(figsize=figsize)
        df_betas.plot(kind="bar", ax=ax)
    
        ax.set_xlabel("Response metric", fontsize=lbl_fs)
        ax.set_ylabel("|Standardized coefficient|", fontsize=lbl_fs)
        ax.tick_params(axis="both", labelsize=tck_fs)
    
        if grid:
            ax.grid(True, axis="y", **grid_kwargs)
    
        if show_title:
            ax.set_title(title or f"Standardized data influence ranking ({metric.upper()})", fontsize=ttl_fs)
    
        fig.tight_layout()
    
        # ---- save ----
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "data_influence"
            out_dir.mkdir(parents=True, exist_ok=True)
    
            for fmt in save_formats:
                p = out_dir / f"{self.event_id}_standardized_data_influence_{metric}.{fmt}"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                logging.info(f"Saved standardized influence plot to {p}")
    
            if return_df:
                csvp = out_dir / f"{self.event_id}_standardized_data_influence_{metric}.csv"
                df_betas.to_csv(csvp)
                logging.info(f"Saved standardized influence table to {csvp}")
    
        if show:
            plt.show()
    
        if close_figs:
            plt.close(fig)
    
        return (fig, ax, df_betas) if return_df else (fig, ax)




    def plot_data_effect_lag(
        self,
        version_list: list,
        metric: str = "mmi",
        use_cache: bool = True,
        tol: float = 1e-4,
        response: str = "mean_abs_delta",
        max_lag: int = 4,
        # Optional CDI counts
        cdi_df: "pd.DataFrame" = None,
        cdi_version_col: str = "version",
        cdi_value_col: str = None,
        cdi_min_value: float = None,
        zfill_versions: bool = True,
        output_path: str = None,
        save_formats: list = ("png", "pdf"),
        dpi: int = 300,
        show_title: bool = True,
        title: str = None,
        # ---- styling ----
        figsize: tuple = (12, 6),
        font_sizes: dict = None,
        legend_loc: str = "best",
        grid: bool = True,
        grid_kwargs: dict = None,
        line_width: float = 2.5,
        marker_size: float = 8,
        # ---- behavior ----
        show: bool = False,
        close_figs: bool = True,
        return_df: bool = True,
    ):
        """
        Lagged correlation: response(step i) vs predictor increment(step i-lag).
        Outputs a table of correlations for each lag and predictor and a plot.
    
        This is not causal proof, but it helps detect delayed influence (e.g., DYFI arriving later).
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import logging
        from pathlib import Path
    
        if font_sizes is None:
            font_sizes = {"labels": 12, "ticks": 10, "title": 14, "legend": 10}
        lbl_fs = font_sizes.get("labels", 12)
        tck_fs = font_sizes.get("ticks", 10)
        ttl_fs = font_sizes.get("title", 14)
        lgd_fs = font_sizes.get("legend", 10)
        if grid_kwargs is None:
            grid_kwargs = {"linestyle": "--", "alpha": 0.4}
    
        # build step table
        _, _, df_steps = self.plot_update_magnitude_vs_data_increment(
            version_list=version_list,
            metric=metric,
            use_cache=use_cache,
            tol=tol,
            cdi_df=cdi_df,
            cdi_version_col=cdi_version_col,
            cdi_value_col=cdi_value_col,
            cdi_min_value=cdi_min_value,
            zfill_versions=zfill_versions,
            response=response,
            output_path=None,
            return_df=True,
            close_figs=True,
        )
    
        if df_steps is None or df_steps.empty or response not in df_steps.columns:
            logging.error("df_steps unavailable; cannot compute lag effects.")
            return (None, None) if not return_df else (None, None, None)
    
        pred_cols = [c for c in df_steps.columns if c.startswith("delta_") and c not in ("delta_col",)]
        pred_cols = [c for c in pred_cols if c in df_steps.columns]
        if not pred_cols:
            logging.error("No predictor delta_* columns found.")
            return (None, None) if not return_df else (None, None, None)
    
        # compute correlations by lag
        rows = []
        y = pd.to_numeric(df_steps[response], errors="coerce")
    
        for lag in range(0, max_lag + 1):
            for p in pred_cols:
                x = pd.to_numeric(df_steps[p], errors="coerce")
                # align response at i with predictor at i-lag
                x_shift = x.shift(lag)
                tmp = pd.concat([y, x_shift], axis=1).dropna()
                if len(tmp) < 3:
                    corr = np.nan
                else:
                    corr = float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1]))
                rows.append({"lag": lag, "predictor": p, "corr": corr})
    
        df_lag = pd.DataFrame(rows)
    
        # plot
        fig, ax = plt.subplots(figsize=figsize)
        for p in pred_cols:
            sub = df_lag[df_lag["predictor"] == p].sort_values("lag")
            ax.plot(
                sub["lag"].values,
                sub["corr"].values,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
                label=p.replace("delta_", "Δ")
            )
    
        ax.axhline(0.0, linewidth=1.5)
        ax.set_xlabel("Lag (steps)", fontsize=lbl_fs)
        ax.set_ylabel(f"Corr( {response} , Δpredictor@lag )", fontsize=lbl_fs)
        ax.tick_params(axis="both", labelsize=tck_fs)
        if grid:
            ax.grid(True, **grid_kwargs)
        ax.legend(loc=legend_loc, fontsize=lgd_fs)
    
        if show_title:
            ax.set_title(title or f"Lagged influence of data increments on updates ({metric.upper()})", fontsize=ttl_fs)
    
        fig.tight_layout()
    
        # save
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "data_influence"
            out_dir.mkdir(parents=True, exist_ok=True)
            for fmt in save_formats:
                p = out_dir / f"{self.event_id}_lag_effect_{metric}_{response}. {fmt}".replace(" ", "")
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                logging.info(f"Saved lag effect plot to {p}")
            if return_df:
                csvp = out_dir / f"{self.event_id}_lag_effect_{metric}_{response}.csv"
                df_lag.to_csv(csvp, index=False)
                logging.info(f"Saved lag effect table to {csvp}")
    
        if show:
            plt.show()
        if close_figs:
            plt.close(fig)
    
        return (fig, ax, df_lag) if return_df else (fig, ax)


    def _uq_sync_paths(self):
        """Keep SHAKEuq paths consistent with SHAKEtime paths."""
        if getattr(self, "uq", None) is None:
            return
        self.uq.event_id = self.event_id
        self.uq.event_time = self.event_time
        self.uq.shakemap_folder = self.shakemap_folder
        self.uq.pager_folder = self.pager_folder
        self.uq.file_type = self.file_type
        # optional paths that may be set later in notebooks
        self.uq.stations_folder = getattr(self, "stations_folder", None)
        self.uq.rupture_folder = getattr(self, "rupture_folder", None)

        



    def uq_build_dataset(self, *args, **kwargs):
        if self.uq is None:
            raise RuntimeError("SHAKEuq is not attached. Check modules.SHAKEuq import.")
        self._uq_sync_paths()
        return self.uq.uq_build_dataset(*args, **kwargs)
    
    def uq_plot_targets_decay(self, *args, **kwargs):
        if self.uq is None:
            raise RuntimeError("SHAKEuq is not attached. Check modules.SHAKEuq import.")
        self._uq_sync_paths()
        return self.uq.uq_plot_targets_decay(*args, **kwargs)
    
    def uq_list_available_imts(self, *args, **kwargs):
        if self.uq is None:
            raise RuntimeError("SHAKEuq is not attached. Check modules.SHAKEuq import.")
        self._uq_sync_paths()
        return self.uq.uq_list_available_imts(*args, **kwargs)

        
        

