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
    26.1.4

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

    
    © SHAKEmaps version 26.1.4
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

    

    



    def plot_std_maps(
        self,
        version_list: list,
        metric: str = "mmi",
        rupture_folder: str = None,
        output_path: str = None,
        plot_colorbar: bool = True,
        show_title: bool = True,
        save_formats: list = ["png", "pdf"],
        dpi: int = 300
    ) -> list:
        """
        Plot uncertainty (standard deviation) maps for each ShakeMap version.

        Parameters
        ----------
        version_list : list of str
            ShakeMap version identifiers.
        metric : str, default "mmi"
            Intensity measure to plot ('mmi', 'pga', or 'pgv').
        rupture_folder : str, optional
            Folder path for rupture files; if provided, overlays rupture traces.
        output_path : str, optional
            Base directory in which to save maps.
        plot_colorbar : bool, default True
            If True, draw a colorbar with ticks and label.
        show_title : bool, default True
            If True, set a map title; otherwise omit it.
        save_formats : list of str, default ["png","pdf"]
            File extensions to save ("png", "pdf", "svg", etc.).
        dpi : int, default 300
            Resolution in dots per inch for saved figures.

        Returns
        -------
        List[(fig, ax, version)]
        """
        figures = []
        for version in version_list:
            # 1) Load uncertainty XML
            base_fn = self._get_shakemap_filename(version)
            un_fn = base_fn.replace("grid", "uncertainty")
            xml_path = os.path.join(self.shakemap_folder, self.event_id, un_fn)
            logging.info(f"Std‐dev map v{version}: loading {xml_path!r}")
            try:
                parser = USGSParser(
                    parser_type="shakemap_xml",
                    xml_file=xml_path,
                    imt=metric,
                    value_type="std"
                )
            except Exception as e:
                logging.error(f"  ✖ failed to parse std XML for v{version}: {e}")
                continue

            # 2) Set up base map
            mapper = SHAKEmapper()
            try:
                md = parser.get_metadata()["grid_specification"]
                extent = [
                    float(md["lon_min"]), float(md["lon_max"]),
                    float(md["lat_min"]), float(md["lat_max"])
                ]
                mapper.set_extent(extent)
            except Exception:
                logging.warning("  ⚠ extent not set from metadata")

            fig, ax = mapper.create_basemap(label_size=22)

            # 3) Add gridlines
            #gl = ax.gridlines(
            #    crs=ccrs.PlateCarree(),
            #    draw_labels=True,
            #    linewidth=1,
            #    color="gray",
            #    alpha=0.5,
            #    linestyle="--",
            #    zorder=2
            #)
            #gl.top_labels = False
            #gl.right_labels = False
            #gl.xlabel_style = {"size": 18}
            #gl.ylabel_style = {"size": 18}

            # 4) Scatter the std‐dev points
            df = parser.get_dataframe()
            lon, lat = df["LON"], df["LAT"]
            std_col = {
                "mmi": "STDMMI",
                "pga": "STDPGA",
                "pgv": "STDPGV"
            }.get(metric.lower())
            if std_col not in df:
                logging.error(f"  ✖ no column {std_col!r} for metric {metric}")
                continue

            scatter = ax.scatter(
                lon, lat,
                c=df[std_col],
                cmap="seismic",
                norm=Normalize(vmin=-4, vmax=4),
                s=15,
                edgecolor="k",
                linewidth=0.1,
                transform=ccrs.PlateCarree(),
                zorder=3
            )

            # 5) Overlay rupture & epicenter
            if rupture_folder:
                rup_fn = self._get_rupture_filename(version)
                rup_path = os.path.join(rupture_folder, self.event_id, rup_fn)
                try:
                    rup_p = USGSParser(parser_type="rupture_json", rupture_json=rup_path)
                    x, y = rup_p.get_rupture_xy()
                    mapper.add_rupture(x, y)
                except Exception as e:
                    logging.warning(f"  ⚠ rupture failed for v{version}: {e}")

            try:
                epic = parser.get_metadata()["event"]
                mapper.add_epicenter(float(epic["lon"]), float(epic["lat"]))
            except Exception:
                logging.warning("  ⚠ epicenter failed")

            # 6) Title
            if show_title:
                ax.set_title(f"Standard Deviation of {metric.upper()}", fontsize=24)

            # 7) Colorbar
            if plot_colorbar:
                cbar = fig.colorbar(scatter, ax=ax, orientation="vertical", pad=0.05)
                cbar.ax.tick_params(labelsize=16)
                cbar.set_label(f"{metric.upper()} σ (std dev)", fontsize=18)

            figures.append((fig, ax, version))

            # 8) Save if requested
            if output_path:
                out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "std_maps" / metric
                out_dir.mkdir(parents=True, exist_ok=True)
                for fmt in save_formats:
                    save_path = out_dir / f"{self.event_id}_std_map_v{version}_{metric}.{fmt}"
                    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                    logging.info(f"Saved std‐dev map to {save_path}")

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
            fig, ax = mapper.create_basemap(label_size=22)
    
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


    #v25.3
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
        ax1.set_title('Bootstrap CIs: mean, std, min, max', fontsize=font_sizes['title'])
        ax1.set_ylabel(metric.upper(), fontsize=font_sizes['labels'])
        ax1.legend()
        ax1.grid(**grid_kwargs)
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, rotation=45, fontsize=font_sizes['ticks'])

        # Subplot 2: Global difference metrics
        ax2.plot(x, mae, 'o-', label='MAE')
        ax2.plot(x, mean_diff, 's-', label='Mean Δ')
        ax2.plot(x, rmse, '^-', label='RMSE')
        ax2.set_title('Global Difference Metrics', fontsize=font_sizes['title'])
        ax2.set_ylabel(metric.upper(), fontsize=font_sizes['labels'])
        ax2.legend()
        ax2.grid(**grid_kwargs)
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels, rotation=45, fontsize=font_sizes['ticks'])

        # Subplot 3: Spatial correlation
        ax3.plot(x, corr_vals, 'o-', color='C4')
        ax3.set_title('Spatial Correlation', fontsize=font_sizes['title'])
        ax3.set_ylabel('Correlation', fontsize=font_sizes['labels'])
        ax3.grid(**grid_kwargs)
        ax3.set_xticks(x)
        ax3.set_xticklabels(x_labels, rotation=45, fontsize=font_sizes['ticks'])

        # Subplot 4: Variogram parameters
        ax4.plot(x, nugget, 'o-', label='Nugget')
        ax4.plot(x, sill, 's-', label='Sill')
        ax4.set_title('Variogram Parameters', fontsize=font_sizes['title'])
        ax4.set_ylabel('Value', fontsize=font_sizes['labels'])
        ax4.legend()
        ax4.grid(**grid_kwargs)
        ax4.set_xticks(x)
        ax4.set_xticklabels(x_labels, rotation=45, fontsize=font_sizes['ticks'])

        plt.tight_layout()

        if show_title:
            fig.suptitle(f"Evolution Panel — {self.event_id}", fontsize=font_sizes['title']+2)
            plt.subplots_adjust(top=0.92)

        # Save if requested
        if output_path:
            out_dir = Path(output_path) / 'SHAKEtime' / self.event_id / 'evolution_panel'
            out_dir.mkdir(parents=True, exist_ok=True)
            for fmt in save_formats:
                path = out_dir / f"{self.event_id}_evolution_panel.{fmt}"
                fig.savefig(path, dpi=dpi, bbox_inches='tight')
                # export each subplot separately   
                # 2) save each subplot individually
                fig.canvas.draw()  # ensure the layout is computed
                for idx, ax in enumerate((ax1, ax2, ax3, ax4), start=1):
                    # get the bounding box of the axes in display coords
                    bbox = ax.get_window_extent()
                    # convert to inches (figure coords)
                    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
                    subpath = out_dir / f"{self.event_id}_evolution_panel_subplot{idx}.{fmt}"
                    fig.savefig(subpath, dpi=dpi, bbox_inches=bbox_inches)

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
        Figure 1: data-availability time series (log y).
        Plots station_count, dyfi_count (used in ShakeMap), trace_length_km
        and optional dyfi_cdi_count from user-provided CDI/DCI dataframe.
    
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
    
        # build x labels
        x = np.arange(len(df))
        x_labels = df.index.tolist()
        xlabel = "Version"
        if x_ticks in ("TaE_h", "TaE_d") and x_ticks in df.columns:
            vals = pd.to_numeric(df[x_ticks], errors="coerce")
            x_labels = [("" if pd.isna(v) else f"{float(v):.1f}") for v in vals]
            xlabel = "Time After Event (hours)" if x_ticks == "TaE_h" else "Time After Event (days)"
    
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
        ax.set_yscale("log")
    
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
    
        ax.set_xlabel(xlabel, fontsize=lbl_fs)
        ax.set_ylabel("Count / Length (log scale)", fontsize=lbl_fs)
        ax.tick_params(axis="both", labelsize=tck_fs)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=xrotation, ha="right")
    
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    
        if grid:
            ax.grid(True, which="both", **grid_kwargs)
        ax.legend(loc=legend_loc, fontsize=lgd_fs)
    
        if show_title:
            ax.set_title(title or "Data availability per ShakeMap version", fontsize=ttl_fs)
    
        fig.tight_layout()
    
        # save
        if output_path:
            out_dir = Path(output_path) / "SHAKEtime" / self.event_id / "data_influence"
            out_dir.mkdir(parents=True, exist_ok=True)
            for fmt in save_formats:
                p = out_dir / f"{self.event_id}_data_availability_timeseries.{fmt}"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                logging.info(f"Saved data availability plot to {p}")
    
            # also save dataframe if requested
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
    
        # x labels
        x = np.arange(len(df))
        x_labels = df.index.tolist()
        xlabel = "Version"
        if x_ticks in ("TaE_h", "TaE_d") and x_ticks in df.columns:
            vals = pd.to_numeric(df[x_ticks], errors="coerce")
            x_labels = [("" if pd.isna(v) else f"{float(v):.1f}") for v in vals]
            xlabel = "Time After Event (hours)" if x_ticks == "TaE_h" else "Time After Event (days)"
    
        cols = [
            ("unc_area_pct90", "Uncertainty area (90% pct)"),
            ("area_exceed_6.0", "Area MMI ≥ 6"),
            ("area_exceed_7.0", "Area MMI ≥ 7"),
            ("area_exceed_8.0", "Area MMI ≥ 8"),
        ]
    
        fig, ax = plt.subplots(figsize=figsize)
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
    
        ax.set_xlabel(xlabel, fontsize=lbl_fs)
        ax.set_ylabel("Area (km²)", fontsize=lbl_fs)
        ax.tick_params(axis="both", labelsize=tck_fs)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=xrotation, ha="right")
    
        if grid:
            ax.grid(True, **grid_kwargs)
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



############################
    #######################################
    ###############################
    ###############################3
    ####################################
    ######################################




    ##################################################
    #
    # COMPLETE UQ FRAMEWORK (DATASET + ANALYSIS + PLOTS + EXPORTS)
    # Export/Import root = export/<event_id>/uq
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
    
    
    def _uq_haversine_km(self, lat1, lon1, lat2, lon2):
        import numpy as np
        R = 6371.0
        lat1r = np.radians(lat1)
        lon1r = np.radians(lon1)
        lat2r = np.radians(lat2)
        lon2r = np.radians(lon2)
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
        return R * c
    
    
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
            ./export/SHAKEtime/<event_id>/uq/
        - This function derives the canonical root from uq_state["base_folder"].
    
        Robustness:
        - If uq_state["base_folder"] looks like a generic export root (e.g., ./export),
          we automatically expand to ./export/SHAKEtime/<event_id>/uq.
        - If base_folder already ends with ".../uq", use it as-is.
        """
        from pathlib import Path
    
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ state not initialized yet. Run uq_build_dataset(...) first.")
    
        base = Path(self.uq_state.get("base_folder", "")).expanduser()
    
        # If someone accidentally set base_folder to a generic export root, fix it.
        # We treat these as "roots" that should contain SHAKEtime/<event_id>/uq.
        if base.name.lower() in ("export", ".") or str(base).rstrip("/\\").lower().endswith("export"):
            base = base / "SHAKEtime" / str(self.event_id)
    
        # If base already points to ".../uq", keep it; else append "uq"
        uq = base if base.name.lower() == "uq" else (base / "uq")
        uq.mkdir(parents=True, exist_ok=True)
        return uq

    def _uq_results_dir(self, version: int = None):
        """
        Standard results directory:
          export/<event_id>/uq/v<version>/uq_results   (per-version)
          export/<event_id>/uq/uq_results             (global)
        """
        uq = self._uq_uqdir()
        p = (uq / "uq_results") if version is None else (uq / f"v{int(version)}" / "uq_results")
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    
    def _uq_plots_dir(self, sub: str = ""):
        """
        Standard plots directory:
          export/<event_id>/uq/uq_plots[/sub]
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
    
        for idx, name in idx_to_name.items():
            col = idx - 1
            if col < 0 or col >= nfields:
                continue
            grid2d = arr[:, col].reshape((nlat, nlon))
            if str(name).upper() == "VS30":
                vs30 = grid2d
            else:
                mean_fields[str(name)] = grid2d
                if str(name) in name_to_units:
                    mean_units[str(name)] = name_to_units[str(name)]
    
        return spec, mean_fields, vs30, mean_units
    
    
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
    
        for idx, name in idx_to_name.items():
            col = idx - 1
            if col < 0 or col >= nfields:
                continue
            grid2d = arr[:, col].reshape((nlat, nlon))
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
            raise AttributeError("SHAKEtime must have self.event_id for UQ file resolution.")
    
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
    
    
    # ---------------------------
    # DATASET BUILDER
    # ---------------------------
    def uq_build_dataset(
        self,
        event_id: str,
        version_list,
        base_folder: str = "./export/SHAKEtime",
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
        import numpy as np
        import json
        from pathlib import Path
        import warnings
    
        version_list = [int(v) for v in list(version_list)]
        interp_kwargs = {} if interp_kwargs is None else dict(interp_kwargs)
    
        # Canonical base = export/<event_id>/uq
        base = self._uq_ensure_dir(Path(base_folder) / str(event_id) / "uq")
    
        per_avail = self.uq_list_available_imts(version_list, stations_folder=stations_folder, rupture_folder=rupture_folder)
        global_imts = sorted({k for vv in per_avail.values() for k in vv})
        requested = self._uq_expand_requested_imts(imts, global_imts)
    
        versions_raw = {}
        file_traces = {}
        sanity_rows = []
    
        for v in version_list:
            grid_path, unc_path, st_path, rup_path, trace = self._uq_resolve_paths(v, stations_folder=stations_folder, rupture_folder=rupture_folder)
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
                "sigma_fields_xml": {},   # XML names (STDPGA...)
                "sigma_units_xml": {},
                "sigma_fields": {},       # mapped IMT names (PGA...)
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
    
            # grid.xml
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
    
            # uncertainty.xml
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
    
            # Ensure requested sigma exists (defaults if missing)
            for imt in requested:
                if imt not in raw["sigma_fields"]:
                    m = raw["mean_fields"].get(imt, None)
                    raw["sigma_fields"][imt] = self._uq_default_sigma_for_imt(imt, m)
                    if imt not in raw["sigma_units"]:
                        raw["sigma_units"][imt] = None
    
            # stationlist.json
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
    
            # vs30
            if raw["vs30"] is not None:
                uv["unified_vs30"] = self._uq_interp_to_unified(slats, slons, raw["vs30"], ULAT2, ULON2, method=interp_method, **interp_kwargs)
            else:
                uv["unified_vs30"] = np.full((unified_spec["nlat"], unified_spec["nlon"]), np.nan, dtype=float)
    
            # fields
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
    
            np.savez_compressed(base / "uq_unified_axes.npz", lats_1d=ulats_1d, lons_1d=ulons_1d, lat2d=ULAT2, lon2d=ULON2)
    
            for v, uv in unified.items():
                vdir = self._uq_ensure_dir(base / f"v{int(v)}")
                np.savez_compressed(vdir / "uq_unified_mean.npz", **{k: uv["unified_mean"][k] for k in uv["unified_mean"]})
                np.savez_compressed(vdir / "uq_unified_sigma_prior_total.npz", **{k: uv["unified_sigma_prior_total"][k] for k in uv["unified_sigma_prior_total"]})
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
    
        self.uq_state = {
            "event_id": str(event_id),
            "version_list": version_list,
            "requested_imts": requested,
            "per_version_available_imts": per_avail,
            "unified_spec": unified_spec,
            "unified_axes": {"lats_1d": ulats_1d, "lons_1d": ulons_1d, "lat2d": ULAT2, "lon2d": ULON2},
            "versions_raw": versions_raw,
            "versions_unified": unified,
            "sanity_rows": sanity_rows,
            "file_traces": file_traces,
            "base_folder": str(base),  # export/<event_id>/uq
            "stations_folder_used": str(stations_folder) if stations_folder is not None else None,
            "rupture_folder_used": str(rupture_folder) if rupture_folder is not None else None,
            "interp_method": str(interp_method),
            "interp_kwargs": dict(interp_kwargs),
            "output_units_requested": dict(output_units) if output_units is not None else None,
        }
        return self.uq_state
    
    
    def uq_sanity_report(self):
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet. Run uq_build_dataset(...) first.")
        rows = self.uq_state.get("sanity_rows", [])
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except Exception:
            return rows
    
    
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
    def _uq_collect_observations(self, version: int, imt: str):
        import numpy as np
        if not hasattr(self, "uq_state") or self.uq_state is None:
            raise RuntimeError("UQ dataset not built yet.")
    
        v = int(version)
        raw = self.uq_state["versions_raw"].get(v, None)
        if raw is None:
            return []
    
        imt_u = str(imt).upper()
        obs = []
    
        if imt_u == "MMI":
            for o in raw["stations"]["dyfi"]:
                val = o.get("intensity", None)
                if val is None or (not np.isfinite(val)):
                    continue
                obs.append({"lat": float(o["lat"]), "lon": float(o["lon"]), "value": float(val), "w": float(o.get("w", 1.0)), "type": "DYFI", "unit": "MMI"})
            return obs
    
        inst = raw["stations"]["instrumented"]
    
        if imt_u == "PGA":
            for o in inst:
                val = o.get("pga", None)
                if val is None or (not np.isfinite(val)):
                    continue
                obs.append({"lat": float(o["lat"]), "lon": float(o["lon"]), "value": float(val), "w": float(o.get("w", 1.0)), "type": "instrumented", "unit": o.get("pga_unit", "%g")})
            return obs
    
        if imt_u == "PGV":
            for o in inst:
                val = o.get("pgv", None)
                if val is None or (not np.isfinite(val)):
                    continue
                obs.append({"lat": float(o["lat"]), "lon": float(o["lon"]), "value": float(val), "w": float(o.get("w", 1.0)), "type": "instrumented", "unit": o.get("pgv_unit", "cm/s")})
            return obs
    
        if imt_u.startswith("PSA"):
            # generic fallback: use "sa" if parser provided it (unit default %g)
            for o in inst:
                val = o.get("sa", None)
                if val is None or (not np.isfinite(val)):
                    continue
                obs.append({"lat": float(o["lat"]), "lon": float(o["lon"]), "value": float(val), "w": float(o.get("w", 1.0)), "type": "instrumented", "unit": o.get("sa_unit", "%g")})
            return obs
    
        return obs
    
    
    # ---------------------------
    # Bayes impact audit
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
    
    
    def _uq_posterior_sigma_fields(self, version: int, imt: str, method: str):
        import numpy as np
        v = int(version)
        imt = str(imt)
        method = str(method).strip().lower()
    
        if method == "bayesupdate":
            dat = self._uq_load_bayes_posterior_npz(v, imt)
            if dat is None:
                raise FileNotFoundError(f"Missing Bayes posterior for v{v}, imt={imt}. Run uq_bayes_update(export=True).")
            return np.asarray(dat["sigma_ep_post"], dtype=float), np.asarray(dat["sigma_total_post"], dtype=float)
    
        if method == "hierarchical":
            dat = self._uq_load_hierarchical_posterior_npz(imt)
            if dat is None:
                raise FileNotFoundError(f"Missing hierarchical posterior for imt={imt}. Run uq_hierarchical(export=True).")
            return np.asarray(dat["sigma_ep_post"], dtype=float), np.asarray(dat["sigma_total_post"], dtype=float)
    
        raise ValueError("method must be 'bayesupdate' or 'hierarchical'")
    
    
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
    
    
    # ---------------------------
    # Plot helpers (self-contained)
    # ---------------------------
    def _uq_cartopy_axes(self, figsize=(10, 8)):
        import matplotlib.pyplot as plt
        try:
            import cartopy.crs as ccrs
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(projection=ccrs.PlateCarree())
            return fig, ax
        except Exception:
            fig, ax = plt.subplots(figsize=figsize)
            return fig, ax
    
    
    def _uq_set_map_extent_from_axes(self, ax, pad_deg: float = 0.1):
        # Uses unified_spec bounds if available
        try:
            spec = self.uq_state["unified_spec"]
            lon_min = float(spec["lon_min"])
            lon_max = float(spec.get("lon_max", lon_min + float(spec["dx"]) * (int(spec["nlon"]) - 1)))
            lat_min = float(spec["lat_min"])
            lat_max = float(spec.get("lat_max", lat_min + float(spec["dy"]) * (int(spec["nlat"]) - 1)))
            lo0, lo1 = min(lon_min, lon_max) - pad_deg, max(lon_min, lon_max) + pad_deg
            la0, la1 = min(lat_min, lat_max) - pad_deg, max(lat_min, lat_max) + pad_deg
            if hasattr(ax, "set_extent"):
                ax.set_extent([lo0, lo1, la0, la1])
            else:
                ax.set_xlim(lo0, lo1)
                ax.set_ylim(la0, la1)
        except Exception:
            return
    
    
    def _uq_save_figure(self, fig, fname_stem: str, subdir: str = "", output_path: str = None, save_formats=("png", "pdf"), dpi: int = 300):
        from pathlib import Path
        outbase = Path(output_path) if output_path is not None else self._uq_uqdir()
        outdir = self._uq_ensure_dir(outbase / str(subdir)) if subdir else self._uq_ensure_dir(outbase)
        for ext in save_formats:
            ext = str(ext).lstrip(".")
            fig.savefig(outdir / f"{fname_stem}.{ext}", dpi=int(dpi), bbox_inches="tight")
    
    
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
        Centralized style defaults to match other SHAKEtime plotting patterns.
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
              <output_path>/SHAKEtime/<event_id>/uq/<subdir>/
          - Else:
              ./export/SHAKEtime/<event_id>/uq/<subdir>/
        """
        from pathlib import Path
    
        # Determine base UQ directory
        if output_path:
            uq_root = Path(output_path).expanduser() / "SHAKEtime" / str(self.event_id) / "uq"
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
    # Drop this whole block INSIDE the SHAKEtime class (replace the whole section).
    # Relies on imports already present in SHAKEtime.py: numpy as np, pandas as pd,
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
          - _uq_decompose_sigma(sigma_prior_total, imt, ...)  [current SHAKEtime]
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
    
    
    # ----------------------------------------------------------------------
    # 4) Observations collection (IMT-aware, robust; logs counts by type)
    # ----------------------------------------------------------------------
    def _uq_infer_obs_domain(self, imt):
        """
        Return preferred observation domain for an IMT:
          - "intensity" for MMI
          - "seismic" for PGA/PGV/SA*
        """
        imtU = str(imt).upper()
        if imtU == "MMI":
            return "intensity"
        return "seismic"
    
    
    def _uq_collect_obs_for_version(
        self,
        version,
        imt,
        measurement_sigma=0.30,
        include_weights=True,
        prefer_domain=True,
        allow_fallback=True,
    ):
        """
        Collect observations available for a given version/imt.
    
        Priority:
          1) If SHAKEtime has _uq_collect_observations(version, imt), use it.
             - We will FILTER by domain when possible (seismic vs intensity)
          2) Else: return [].
    
        Expected record fields (flexible):
          lat, lon, value, (optional) w, type/domain/source
    
        Returns:
          obs_list: list of dicts {"lat","lon","y_ws","w","type"}
          obs_counts: dict with counts by inferred type
        """
        imtU = str(imt).upper()
        domain_pref = self._uq_infer_obs_domain(imtU)
        obs = []
        counts = {"total": 0, "seismic": 0, "intensity": 0, "unknown": 0}
    
        raw = None
        if hasattr(self, "_uq_collect_observations"):
            try:
                raw = self._uq_collect_observations(int(version), imtU)
            except Exception:
                raw = None
    
        if not raw:
            return [], counts
    
        # normalize and (optionally) domain-filter
        tmp = []
        for r in raw:
            if r is None or (not isinstance(r, dict)):
                continue
            if ("lat" not in r) or ("lon" not in r) or ("value" not in r):
                continue
            try:
                y = float(r.get("value"))
            except Exception:
                continue
    
            typ = str(r.get("type", r.get("domain", r.get("source", "unknown")))).lower()
            if "dyfi" in typ or "mmi" in typ or "intensity" in typ:
                dom = "intensity"
            elif "station" in typ or "seismic" in typ or "pga" in typ or "pgv" in typ:
                dom = "seismic"
            else:
                dom = "unknown"
    
            tmp.append((r, dom))
    
        # domain preference
        if prefer_domain:
            preferred = [r for (r, dom) in tmp if dom == domain_pref]
            if preferred:
                use = preferred
            else:
                use = [r for (r, dom) in tmp] if allow_fallback else []
        else:
            use = [r for (r, dom) in tmp]
    
        for r in use:
            try:
                y = float(r.get("value"))
            except Exception:
                continue
            y_ws = self._uq_obs_to_ws(imtU, y)
            w = float(r.get("w", 1.0)) if include_weights else 1.0
    
            typ = str(r.get("type", r.get("domain", r.get("source", "obs"))))
            typL = typ.lower()
            if "dyfi" in typL or "intensity" in typL or "mmi" in typL:
                counts["intensity"] += 1
            elif "station" in typL or "seismic" in typL or "pga" in typL or "pgv" in typL:
                counts["seismic"] += 1
            else:
                counts["unknown"] += 1
    
            obs.append({
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "y_ws": float(y_ws),
                "w": float(w),
                "type": typ,
            })
    
        counts["total"] = len(obs)
        return obs, counts
    
    
    # ----------------------------------------------------------------------
    # 5) Bayesian / Hierarchical / OK residual update engines (working space)
    # ----------------------------------------------------------------------
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
        Returns arrays (mu_post_ws, sigma_ep_post) valid on mask; else nan.
        """
        mu0_ws = np.asarray(mu0_ws, dtype=float)
        sigma_ep0 = np.asarray(sigma_ep0, dtype=float)
        sigma_a = np.asarray(sigma_a, dtype=float)
    
        mu_post = np.full(mu0_ws.shape, np.nan, dtype=float)
        sig_ep_post = np.full(mu0_ws.shape, np.nan, dtype=float)
    
        m = target_mask.astype(bool)
        if not m.any():
            return mu_post, sig_ep_post
    
        prior_var = np.clip(sigma_ep0**2, 1e-16, np.inf)
        prior_prec = 1.0 / prior_var
    
        mu_post[m] = mu0_ws[m]
        sig_ep_post[m] = sigma_ep0[m]
        prec_post = np.zeros_like(mu0_ws, dtype=float)
        prec_post[m] = prior_prec[m]
    
        if not obs_list:
            return mu_post, sig_ep_post
    
        meas_var = float(measurement_sigma) ** 2
        like_var = np.clip(sigma_a**2 + meas_var, 1e-16, np.inf)
    
        ker = str(kernel).lower().strip()
        for ob in obs_list:
            lat_o, lon_o = ob["lat"], ob["lon"]
            y = float(ob["y_ws"])
            w0 = float(ob.get("w", 1.0))
    
            d = self._uq_haversine_km(lat_o, lon_o, lat2d, lon2d)
    
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
    
            if ker in ("tophat", "uniform"):
                w_space = np.ones_like(d, dtype=float)
            else:
                s = float(kernel_scale_km) if kernel_scale_km is not None else float(update_radius_km) / 2.0
                s = max(s, 1e-6)
                w_space = np.exp(-(d**2) / (2.0 * s**2))
    
            w_eff = w0 * w_space
            w_eff[~local] = 0.0
    
            prec_like = np.zeros_like(mu0_ws, dtype=float)
            prec_like[local] = np.clip(w_eff[local] / like_var[local], 0.0, np.inf)
    
            prec_new = prec_post + prec_like
            upd = local & np.isfinite(prec_new) & (prec_new > 0)
    
            mu_post[upd] = (mu_post[upd] * prec_post[upd] + y * prec_like[upd]) / prec_new[upd]
            prec_post[upd] = prec_new[upd]
            sig_ep_post[upd] = np.sqrt(1.0 / np.clip(prec_post[upd], 1e-16, np.inf))
    
        return mu_post, sig_ep_post
    
    
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
              <output_path>/SHAKEtime/<event_id>/uq/uq_audit/
          - Else:
              ./export/SHAKEtime/<event_id>/uq/uq_audit/
        """
        from pathlib import Path
        import pandas as pd
    
        try:
            if output_path:
                uq_root = Path(output_path).expanduser() / "SHAKEtime" / str(self.event_id) / "uq"
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
              <output_path>/SHAKEtime/<event_id>/uq/<subdir>/
          - Else:
              ./export/SHAKEtime/<event_id>/uq/<subdir>/
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
            uq_root = Path(output_path).expanduser() / "SHAKEtime" / str(self.event_id) / "uq"
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
    
    
    def _uq_parse_gridxml_to_arrays_legacy(self, grid_xml_path):
        """
        Parse ShakeMap grid.xml OR uncertainty.xml with:
          - grid_spec (nlat, nlon, lat_min/max, lon_min/max, and optionally spacing)
          - grid_field mapping (index -> name)
          - grid_data text rows (nlat*nlon lines)
        Returns dict:
          {
            "nlat","nlon","lat_min","lat_max","lon_min","lon_max",
            "lat_vec","lon_vec","fields": {NAME: col_index0},
            "data": ndarray shape (nlat*nlon, nfields)
          }
    
        IMPORTANT FIX:
          Prefer grid_spec spacing (lat_spacing/lon_spacing or nominal) when present,
          instead of linspace, to match ShakeMap indexing precisely.
        """
        p = Path(grid_xml_path)
        tree = ET.parse(p)
        root = tree.getroot()
    
        spec = root.find(".//grid_spec")
        if spec is None:
            raise RuntimeError(f"No grid_spec found in {p.name}")
    
        nlon = int(float(spec.attrib.get("nlon")))
        nlat = int(float(spec.attrib.get("nlat")))
        lon_min = float(spec.attrib.get("lon_min"))
        lon_max = float(spec.attrib.get("lon_max"))
        lat_min = float(spec.attrib.get("lat_min"))
        lat_max = float(spec.attrib.get("lat_max"))
    
        # Try spacing keys used in ShakeMap
        lon_spacing = spec.attrib.get("lon_spacing", spec.attrib.get("nominal_lon_spacing", None))
        lat_spacing = spec.attrib.get("lat_spacing", spec.attrib.get("nominal_lat_spacing", None))
    
        lon_vec = None
        lat_vec = None
        try:
            if lon_spacing is not None:
                dlon = float(lon_spacing)
                lon_vec = lon_min + dlon * np.arange(nlon, dtype=float)
            if lat_spacing is not None:
                dlat = float(lat_spacing)
                lat_vec = lat_min + dlat * np.arange(nlat, dtype=float)
        except Exception:
            lon_vec = None
            lat_vec = None
    
        # Fall back to linspace if needed
        if lon_vec is None or len(lon_vec) != nlon:
            lon_vec = np.linspace(lon_min, lon_max, nlon)
        if lat_vec is None or len(lat_vec) != nlat:
            lat_vec = np.linspace(lat_min, lat_max, nlat)
    
        # fields (store 0-based column indices)
        fields = {}
        for gf in root.findall(".//grid_field"):
            try:
                idx = int(gf.attrib.get("index"))
                name = str(gf.attrib.get("name")).strip().upper()
                fields[name] = idx - 1  # ShakeMap is typically 1-based
            except Exception:
                continue
    
        gd = root.find(".//grid_data")
        if gd is None or gd.text is None:
            raise RuntimeError(f"No grid_data found in {p.name}")
    
        lines = [ln.strip() for ln in gd.text.strip().splitlines() if ln.strip()]
        arr = np.array([[float(x) for x in ln.split()] for ln in lines], dtype=float)
    
        expected = nlat * nlon
        if arr.shape[0] != expected:
            # trim (keep last expected rows) if larger; else error
            if arr.shape[0] > expected:
                arr = arr[-expected:, :]
            else:
                raise RuntimeError(f"grid_data row count mismatch in {p.name}: got {arr.shape[0]} expected {expected}")
    
        return {
            "nlat": nlat, "nlon": nlon,
            "lat_min": lat_min, "lat_max": lat_max,
            "lon_min": lon_min, "lon_max": lon_max,
            "lat_vec": lat_vec, "lon_vec": lon_vec,
            "fields": fields,
            "data": arr,
        }
    
    
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
    
    
    def _uq_raw_get_field_grid(self, D, field_name):
        """Return 2D grid for a field name from parsed XML dict D."""
        nameU = str(field_name).upper()
        if nameU not in D["fields"]:
            return None
        c = D["fields"][nameU]
        if c < 0 or c >= D["data"].shape[1]:
            return None
        # reshape to (nlat,nlon) row-major
        grid = D["data"][:, c].reshape((D["nlat"], D["nlon"]))
        return grid
    
    
    # ----------------------------------------------------------------------
    # 8) Core extraction: per-target series (PUBLISHED vs PREDICTED + RAW sigma_total)
    # ----------------------------------------------------------------------
    def uq_extract_target_series(
        self,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        agg="mean",
        global_stat=None,  # None|"mean"|"median"|"delta mean"|"delta median" (applies to *reported* y)
        # IMPORTANT: ShakeMap sigma_total is RAW from uncertainty.xml (no decomposition)
        shakemap_total_sigma_mode="raw",
        # decomposition controls (used ONLY for update methods bookkeeping)
        sigma_total_from_shakemap=True,
        sigma_aleatory=None,
        # updates
        prior_version=None,
        update_radius_km=30.0,
        kernel="gaussian",
        kernel_scale_km=20.0,
        measurement_sigma=0.30,
        # kriging
        ok_range_km=60.0,
        ok_variogram="exponential",
        ok_nugget=1e-6,
        ok_sill=None,
        ok_cap_sigma_to_prior=True,
        # monte carlo
        mc_nsim=2000,
        mc_include_aleatory=True,
        # unified grid controls
        grid_res=None,
        interp_method="nearest",
        interp_kwargs=None,
        # audit
        audit=True,
        audit_output_path=None,
        audit_prefix=None,
    ):
        """
        Build tidy DataFrame of per-target, per-version:
          - ShakeMap published prediction (mean_published) and RAW sigma_total_published (from uncertainty.xml)
          - Method predicted prediction (mean_predicted) and updated sigma_total_predicted (from v0 prior + obs)
          - Deltas vs ShakeMap published for explicit comparison.
    
        Output rows are method-specific (method = "ShakeMap" or update method).
        BUT for all rows we ALSO include the published reference columns so you can compare directly.
    
        Core columns:
          target_id, target_type, version, imt, method,
          mean_published, sigma_total_published_raw,
          mean_predicted, sigma_total_predicted,
          delta_mean_vs_published, delta_sigma_vs_published,
          sigma_epistemic_predicted, sigma_aleatoric_used,
          n_obs_total, n_obs_seismic, n_obs_intensity, n_obs_unknown,
          n_cells,
          (ShakeMap-only envelopes for area/global): mean_published_min/max, sigma_published_min/max
    
        Notes:
          - ShakeMap sigma_total is ALWAYS the RAW uncertainty grid value (STD*) aggregated over mask.
          - Update methods use prior decomposition (sigma_a0 + sigma_ep_post) to reconstruct sigma_total_predicted.
        """
        if interp_kwargs is None:
            interp_kwargs = {}
    
        versions = [int(v) for v in (version_list or [])]
        if not versions:
            raise ValueError("version_list is required and cannot be empty.")
    
        imtU = str(imt).upper()
        if str(shakemap_total_sigma_mode).lower().strip() != "raw":
            # enforce invariant for now (future extension hook)
            shakemap_total_sigma_mode = "raw"
    
        # GLOBAL target mode
        global_stat_eff = None
        if global_stat is not None:
            gs = str(global_stat).lower().strip()
            if gs in ("mean", "median"):
                global_stat_eff = gs
            elif gs in ("delta mean", "deltamean", "delta_mean"):
                global_stat_eff = "delta mean"
            elif gs in ("delta median", "deltamedian", "delta_median"):
                global_stat_eff = "delta median"
            else:
                raise ValueError('global_stat must be None or one of: "mean","median","delta mean","delta median".')
    
        if global_stat_eff is not None:
            gid = f"GLOBAL_{str(global_stat_eff).replace(' ', '_')}".replace("__", "_")
            targets = [{"id": gid, "type": "global"}]
        else:
            targets = self._uq_parse_targets(points=points, areas=areas)
    
        # unified axes
        _, lat2d, lon2d = self._uq_get_unified_for_versions(
            versions, imt=imtU, grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
        )
    
        # choose prior version
        v0 = int(prior_version) if prior_version is not None else int(versions[0])
    
        # prior fields on unified grid (from v0)
        mu0_lin, sig0_raw = self._uq_get_mu_sigma_unified(
            v0, imtU, lat2d, lon2d, grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
        )
        mu0_ws = self._uq_mu_to_ws(imtU, mu0_lin)
    
        # decompose prior sigma (ONLY for update methods)
        _, s_a0, s_ep0 = self._uq_decompose_sigma_safe(
            imtU,
            sig0_raw,
            sigma_aleatory=sigma_aleatory,
            sigma_total_from_shakemap=bool(sigma_total_from_shakemap),
        )
    
        rows = []
        audit_rows = []
    
        # helper: global mask
        def _global_mask(lat2d_):
            return np.isfinite(lat2d_)
    
        for t in targets:
            # determine mask + agg
            if t.get("type") == "global":
                mask = _global_mask(lat2d)
                n_cells = int(np.sum(mask))
                agg_effective = "mean" if ("mean" in global_stat_eff) else "median"
                meta = {"kind": "global", "n_cells": n_cells}
            else:
                mask, meta = self._uq_target_mask(t, lat2d, lon2d)
                n_cells = int(meta.get("n_cells", int(mask.sum())))
                agg_effective = agg
    
            if n_cells <= 0:
                for v in versions:
                    # publish row
                    rows.append({
                        "target_id": t.get("id", "unknown"),
                        "target_type": t.get("type", "unknown"),
                        "version": int(v),
                        "imt": imtU,
                        "method": "ShakeMap",
                        "mean_published": np.nan,
                        "sigma_total_published_raw": np.nan,
                        "mean_published_min": np.nan,
                        "mean_published_max": np.nan,
                        "sigma_published_min": np.nan,
                        "sigma_published_max": np.nan,
                        "mean_predicted": np.nan,
                        "sigma_total_predicted": np.nan,
                        "delta_mean_vs_published": np.nan,
                        "delta_sigma_vs_published": np.nan,
                        "sigma_epistemic_predicted": np.nan,
                        "sigma_aleatoric_used": np.nan,
                        "n_obs_total": 0,
                        "n_obs_seismic": 0,
                        "n_obs_intensity": 0,
                        "n_obs_unknown": 0,
                        "n_cells": 0,
                    })
                continue
    
            # prior target scalars (used when no obs)
            s_a0_t = self._uq_agg(s_a0[mask], agg=agg_effective)
            s_ep0_t = self._uq_agg(s_ep0[mask], agg=agg_effective)
            sig0_raw_t = self._uq_agg(sig0_raw[mask], agg=agg_effective)
            mu0_lin_t = self._uq_agg(mu0_lin[mask], agg=agg_effective)
    
            base_mu = float(mu0_lin_t)
            base_sig = float(sig0_raw_t)
    
            for v in versions:
                # ---- PUBLISHED (ShakeMap) reference for this version ----
                mu_v_lin, sig_v_raw = self._uq_get_mu_sigma_unified(
                    int(v), imtU, lat2d, lon2d,
                    grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
                )
                mean_pub = self._uq_agg(mu_v_lin[mask], agg=agg_effective)
                sig_pub_raw = self._uq_agg(sig_v_raw[mask], agg=agg_effective)
    
                # envelopes (only meaningful for area/global; for point they collapse)
                mean_pub_min, mean_pub_max = self._uq_minmax(mu_v_lin[mask])
                sig_pub_min, sig_pub_max = self._uq_minmax(sig_v_raw[mask])
    
                # publish row (explicit)
                rows.append({
                    "target_id": t.get("id", "GLOBAL"),
                    "target_type": t.get("type", "global" if t.get("type") == "global" else "target"),
                    "version": int(v),
                    "imt": imtU,
                    "method": "ShakeMap",
                    "mean_published": float(mean_pub),
                    "sigma_total_published_raw": float(sig_pub_raw),
                    "mean_published_min": float(mean_pub_min),
                    "mean_published_max": float(mean_pub_max),
                    "sigma_published_min": float(sig_pub_min),
                    "sigma_published_max": float(sig_pub_max),
                    "mean_predicted": float(mean_pub),               # for ShakeMap row, predicted == published
                    "sigma_total_predicted": float(sig_pub_raw),     # for ShakeMap row, predicted == published
                    "delta_mean_vs_published": 0.0,
                    "delta_sigma_vs_published": 0.0,
                    "sigma_epistemic_predicted": np.nan,
                    "sigma_aleatoric_used": np.nan,
                    "n_obs_total": 0,
                    "n_obs_seismic": 0,
                    "n_obs_intensity": 0,
                    "n_obs_unknown": 0,
                    "n_cells": int(n_cells),
                })
    
                # ---- OBSERVATIONS available at version v for imt ----
                obs, counts = self._uq_collect_obs_for_version(
                    int(v), imtU,
                    measurement_sigma=measurement_sigma,
                    include_weights=True,
                    prefer_domain=True,
                    allow_fallback=True,
                )
                n_obs = len(obs)
    
                # ---- If no obs: update methods revert to prior scalars (v0) ----
                if n_obs == 0:
                    for method in ("bayes", "hierarchical", "kriging", "montecarlo"):
                        mean_pred = float(mu0_lin_t)
                        sig_pred = float(sig0_raw_t)
                        rows.append({
                            "target_id": t.get("id", "GLOBAL"),
                            "target_type": t.get("type", "global" if t.get("type") == "global" else "target"),
                            "version": int(v),
                            "imt": imtU,
                            "method": method,
                            # published reference
                            "mean_published": float(mean_pub),
                            "sigma_total_published_raw": float(sig_pub_raw),
                            "mean_published_min": float(mean_pub_min),
                            "mean_published_max": float(mean_pub_max),
                            "sigma_published_min": float(sig_pub_min),
                            "sigma_published_max": float(sig_pub_max),
                            # predicted by method
                            "mean_predicted": mean_pred,
                            "sigma_total_predicted": sig_pred,
                            "delta_mean_vs_published": mean_pred - float(mean_pub) if np.isfinite(mean_pub) else np.nan,
                            "delta_sigma_vs_published": sig_pred - float(sig_pub_raw) if np.isfinite(sig_pub_raw) else np.nan,
                            "sigma_epistemic_predicted": float(s_ep0_t),
                            "sigma_aleatoric_used": float(s_a0_t),
                            "n_obs_total": 0,
                            "n_obs_seismic": int(counts.get("seismic", 0)),
                            "n_obs_intensity": int(counts.get("intensity", 0)),
                            "n_obs_unknown": int(counts.get("unknown", 0)),
                            "n_cells": int(n_cells),
                        })
                    if audit:
                        audit_rows.append({
                            "target_id": t.get("id", "GLOBAL"),
                            "target_type": t.get("type", "global" if t.get("type") == "global" else "target"),
                            "version": int(v),
                            "imt": imtU,
                            "agg": str(agg_effective),
                            "mask_kind": meta.get("kind", "global" if t.get("type") == "global" else "mask"),
                            "mask_meta": str(meta)[:800],
                            "n_cells": int(n_cells),
                            "n_obs_total": 0,
                            "n_obs_seismic": int(counts.get("seismic", 0)),
                            "n_obs_intensity": int(counts.get("intensity", 0)),
                            "n_obs_unknown": int(counts.get("unknown", 0)),
                            "note": "No obs => methods revert to v0 prior scalars at target.",
                        })
                    continue
    
                # ---- BAYES local update (working space) ----
                mu_b_ws, s_ep_b = self._uq_bayes_local_posterior_at_mask(
                    mu0_ws, s_ep0, s_a0,
                    lat2d, lon2d, mask, obs,
                    update_radius_km=update_radius_km,
                    kernel=kernel,
                    kernel_scale_km=kernel_scale_km,
                    measurement_sigma=measurement_sigma,
                )
                mu_b_lin = self._uq_mu_from_ws(imtU, mu_b_ws)
                mean_b = self._uq_agg(mu_b_lin[mask], agg=agg_effective)
                s_ep_b_t = self._uq_agg(s_ep_b[mask], agg=agg_effective)
                sig_b = float(np.sqrt(max(0.0, s_a0_t**2 + s_ep_b_t**2)))
    
                rows.append({
                    "target_id": t.get("id", "GLOBAL"),
                    "target_type": t.get("type", "global" if t.get("type") == "global" else "target"),
                    "version": int(v),
                    "imt": imtU,
                    "method": "bayes",
                    "mean_published": float(mean_pub),
                    "sigma_total_published_raw": float(sig_pub_raw),
                    "mean_published_min": float(mean_pub_min),
                    "mean_published_max": float(mean_pub_max),
                    "sigma_published_min": float(sig_pub_min),
                    "sigma_published_max": float(sig_pub_max),
                    "mean_predicted": float(mean_b),
                    "sigma_total_predicted": float(sig_b),
                    "delta_mean_vs_published": float(mean_b) - float(mean_pub) if np.isfinite(mean_pub) else np.nan,
                    "delta_sigma_vs_published": float(sig_b) - float(sig_pub_raw) if np.isfinite(sig_pub_raw) else np.nan,
                    "sigma_epistemic_predicted": float(s_ep_b_t),
                    "sigma_aleatoric_used": float(s_a0_t),
                    "n_obs_total": int(n_obs),
                    "n_obs_seismic": int(counts.get("seismic", 0)),
                    "n_obs_intensity": int(counts.get("intensity", 0)),
                    "n_obs_unknown": int(counts.get("unknown", 0)),
                    "n_cells": int(n_cells),
                })
    
                # ---- Hierarchical-like update ----
                mu_h_ws, s_ep_h = self._uq_hierarchical_posterior_at_mask(
                    mu0_ws, s_ep0, s_a0,
                    lat2d, lon2d, mask, obs,
                    update_radius_km=update_radius_km,
                    kernel=kernel,
                    kernel_scale_km=kernel_scale_km,
                    measurement_sigma=measurement_sigma,
                )
                mu_h_lin = self._uq_mu_from_ws(imtU, mu_h_ws)
                mean_h = self._uq_agg(mu_h_lin[mask], agg=agg_effective)
                s_ep_h_t = self._uq_agg(s_ep_h[mask], agg=agg_effective)
                sig_h = float(np.sqrt(max(0.0, s_a0_t**2 + s_ep_h_t**2)))
    
                rows.append({
                    "target_id": t.get("id", "GLOBAL"),
                    "target_type": t.get("type", "global" if t.get("type") == "global" else "target"),
                    "version": int(v),
                    "imt": imtU,
                    "method": "hierarchical",
                    "mean_published": float(mean_pub),
                    "sigma_total_published_raw": float(sig_pub_raw),
                    "mean_published_min": float(mean_pub_min),
                    "mean_published_max": float(mean_pub_max),
                    "sigma_published_min": float(sig_pub_min),
                    "sigma_published_max": float(sig_pub_max),
                    "mean_predicted": float(mean_h),
                    "sigma_total_predicted": float(sig_h),
                    "delta_mean_vs_published": float(mean_h) - float(mean_pub) if np.isfinite(mean_pub) else np.nan,
                    "delta_sigma_vs_published": float(sig_h) - float(sig_pub_raw) if np.isfinite(sig_pub_raw) else np.nan,
                    "sigma_epistemic_predicted": float(s_ep_h_t),
                    "sigma_aleatoric_used": float(s_a0_t),
                    "n_obs_total": int(n_obs),
                    "n_obs_seismic": int(counts.get("seismic", 0)),
                    "n_obs_intensity": int(counts.get("intensity", 0)),
                    "n_obs_unknown": int(counts.get("unknown", 0)),
                    "n_cells": int(n_cells),
                })
    
                # ---- OK residual kriging ----
                mu_k_ws, s_ep_k = self._uq_ok_residual_posterior_at_mask(
                    mu0_ws, s_ep0, s_a0,
                    lat2d, lon2d, mask, obs,
                    variogram=ok_variogram,
                    range_km=ok_range_km,
                    nugget=ok_nugget,
                    sill=ok_sill,
                    measurement_sigma=measurement_sigma,
                    sigma_ep_cap_to_prior=ok_cap_sigma_to_prior,
                )
                mu_k_lin = self._uq_mu_from_ws(imtU, mu_k_ws)
                mean_k = self._uq_agg(mu_k_lin[mask], agg=agg_effective)
                s_ep_k_t = self._uq_agg(s_ep_k[mask], agg=agg_effective)
                sig_k = float(np.sqrt(max(0.0, s_a0_t**2 + s_ep_k_t**2)))
    
                rows.append({
                    "target_id": t.get("id", "GLOBAL"),
                    "target_type": t.get("type", "global" if t.get("type") == "global" else "target"),
                    "version": int(v),
                    "imt": imtU,
                    "method": "kriging",
                    "mean_published": float(mean_pub),
                    "sigma_total_published_raw": float(sig_pub_raw),
                    "mean_published_min": float(mean_pub_min),
                    "mean_published_max": float(mean_pub_max),
                    "sigma_published_min": float(sig_pub_min),
                    "sigma_published_max": float(sig_pub_max),
                    "mean_predicted": float(mean_k),
                    "sigma_total_predicted": float(sig_k),
                    "delta_mean_vs_published": float(mean_k) - float(mean_pub) if np.isfinite(mean_pub) else np.nan,
                    "delta_sigma_vs_published": float(sig_k) - float(sig_pub_raw) if np.isfinite(sig_pub_raw) else np.nan,
                    "sigma_epistemic_predicted": float(s_ep_k_t),
                    "sigma_aleatoric_used": float(s_a0_t),
                    "n_obs_total": int(n_obs),
                    "n_obs_seismic": int(counts.get("seismic", 0)),
                    "n_obs_intensity": int(counts.get("intensity", 0)),
                    "n_obs_unknown": int(counts.get("unknown", 0)),
                    "n_cells": int(n_cells),
                })
    
                # ---- Monte Carlo (target-level) around Bayes posterior by default ----
                mu_ref = float(mean_b)
                sig_ref = float(sig_b) if mc_include_aleatory else float(s_ep_b_t)
    
                if np.isfinite(mu_ref) and np.isfinite(sig_ref) and (sig_ref > 0) and mc_nsim and (int(mc_nsim) > 10):
                    draws = np.random.normal(loc=mu_ref, scale=sig_ref, size=int(mc_nsim))
                    mean_mc = float(np.nanmean(draws))
                    sig_mc = float(np.nanstd(draws))
                else:
                    mean_mc = mu_ref
                    sig_mc = sig_ref
    
                if mc_include_aleatory:
                    sig_tot_mc = sig_mc
                    sig_ep_mc = float(np.sqrt(max(0.0, sig_tot_mc**2 - s_a0_t**2)))
                else:
                    sig_ep_mc = sig_mc
                    sig_tot_mc = float(np.sqrt(max(0.0, s_a0_t**2 + sig_ep_mc**2)))
    
                rows.append({
                    "target_id": t.get("id", "GLOBAL"),
                    "target_type": t.get("type", "global" if t.get("type") == "global" else "target"),
                    "version": int(v),
                    "imt": imtU,
                    "method": "montecarlo",
                    "mean_published": float(mean_pub),
                    "sigma_total_published_raw": float(sig_pub_raw),
                    "mean_published_min": float(mean_pub_min),
                    "mean_published_max": float(mean_pub_max),
                    "sigma_published_min": float(sig_pub_min),
                    "sigma_published_max": float(sig_pub_max),
                    "mean_predicted": float(mean_mc),
                    "sigma_total_predicted": float(sig_tot_mc),
                    "delta_mean_vs_published": float(mean_mc) - float(mean_pub) if np.isfinite(mean_pub) else np.nan,
                    "delta_sigma_vs_published": float(sig_tot_mc) - float(sig_pub_raw) if np.isfinite(sig_pub_raw) else np.nan,
                    "sigma_epistemic_predicted": float(sig_ep_mc),
                    "sigma_aleatoric_used": float(s_a0_t),
                    "n_obs_total": int(n_obs),
                    "n_obs_seismic": int(counts.get("seismic", 0)),
                    "n_obs_intensity": int(counts.get("intensity", 0)),
                    "n_obs_unknown": int(counts.get("unknown", 0)),
                    "n_cells": int(n_cells),
                })
    
                # Audit (per target/version)
                if audit:
                    audit_rows.append({
                        "target_id": t.get("id", "GLOBAL"),
                        "target_type": t.get("type", "global" if t.get("type") == "global" else "target"),
                        "version": int(v),
                        "imt": imtU,
                        "agg": str(agg_effective),
                        "n_cells": int(n_cells),
                        "n_obs_total": int(n_obs),
                        "n_obs_seismic": int(counts.get("seismic", 0)),
                        "n_obs_intensity": int(counts.get("intensity", 0)),
                        "n_obs_unknown": int(counts.get("unknown", 0)),
                        "mask_kind": meta.get("kind", "global" if t.get("type") == "global" else "mask"),
                        "mask_meta": str(meta)[:800],
                        "note": "Published sigma_total is RAW uncertainty.xml. Methods predict mean+sigma via v0 prior + obs at version.",
                    })
    
            # Optional: delta global modes could be applied downstream (plotter supports delta columns).
            # We keep extraction explicit and do delta transforms in plotter for clarity.
    
        df = pd.DataFrame(rows)
    
        # audit write
        if audit:
            pref = audit_prefix or f"UQ-TargetAudit-Patch4-{imtU}"
            self._uq_write_audit(audit_rows, output_path=audit_output_path, prefix=pref)
    
        return df
    
    
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
    
    """
    # ----------------------------------------------------------------------
    # 10) Validation: RAW XML vs UNIFIED grid curves (point/area; with envelopes)
    # ----------------------------------------------------------------------
    def uq_compare_raw_vs_unified_curves(
        self,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        # raw xml search paths
        base_folder=None,
        shakemap_folder=None,
        # extraction controls
        agg="mean",
        # unified settings
        grid_res=None,
        interp_method="nearest",
        interp_kwargs=None,
        # plotting
        make_plots=True,
        figsize=(9.5, 5.0),
        dpi=220,
        xrotation=45,
        show_grid=True,
        show_title=True,
        title=None,
        legend=True,
        legend_kwargs=None,
        envelope_alpha=0.18,
        # save
        output_path=None,
        save=False,
        save_formats=("png",),
        # audit
        export_tables=True,
        export_prefix="UQ-RawVsUnified-Curves",
    ):
        if legend_kwargs is None:
            legend_kwargs = {}
        if interp_kwargs is None:
            interp_kwargs = {}
    
        versions = [int(v) for v in (version_list or [])]
        if not versions:
            raise ValueError("version_list cannot be empty.")
        imtU = str(imt).upper()
    
        targets = self._uq_parse_targets(points=points, areas=areas)
        if not targets:
            raise ValueError("Provide points and/or areas.")
    
        # unified axes
        _, lat2d, lon2d = self._uq_get_unified_for_versions(
            versions, imt=imtU, grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
        )
    
        rows = []
    
        for v in versions:
            # unified grids for this version
            mu_u, sig_u = self._uq_get_mu_sigma_unified(
                v, imtU, lat2d, lon2d, grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
            )
    
            # raw xml locate
            grid_p = self._uq_find_shakemap_xml(v, which="grid", base_folder=base_folder, shakemap_folder=shakemap_folder)
            unc_p = self._uq_find_shakemap_xml(v, which="uncertainty", base_folder=base_folder, shakemap_folder=shakemap_folder)
    
            G = None
            U = None
            if grid_p is not None and unc_p is not None:
                try:
                    G = self._uq_parse_gridxml_to_arrays(grid_p)
                    U = self._uq_parse_gridxml_to_arrays(unc_p)
                except Exception:
                    G = None
                    U = None
    
            for t in targets:
                # unified mask
                umask, umeta = self._uq_target_mask(t if t.get("type") != "global" else {"type": "global"}, lat2d, lon2d)
                u_mean = self._uq_agg(np.asarray(mu_u)[umask], agg=agg)
                u_sig = self._uq_agg(np.asarray(sig_u)[umask], agg=agg)
                u_mean_min, u_mean_max = self._uq_minmax(np.asarray(mu_u)[umask])
                u_sig_min, u_sig_max = self._uq_minmax(np.asarray(sig_u)[umask])
    
                rows.append({
                    "target_id": t["id"],
                    "target_type": t["type"],
                    "version": int(v),
                    "imt": imtU,
                    "source": "unified",
                    "mean": float(u_mean),
                    "sigma": float(u_sig),
                    "mean_min": float(u_mean_min),
                    "mean_max": float(u_mean_max),
                    "sigma_min": float(u_sig_min),
                    "sigma_max": float(u_sig_max),
                    "note": "",
                })
    
                # raw extraction
                if (G is None) or (U is None):
                    rows.append({
                        "target_id": t["id"],
                        "target_type": t["type"],
                        "version": int(v),
                        "imt": imtU,
                        "source": "raw",
                        "mean": np.nan,
                        "sigma": np.nan,
                        "mean_min": np.nan,
                        "mean_max": np.nan,
                        "sigma_min": np.nan,
                        "sigma_max": np.nan,
                        "note": "Missing raw XML or parse failure (pass base_folder/shakemap_folder).",
                    })
                    continue
    
                rmask, rmeta = self._uq_raw_mask_on_xml_grid(G, t)
                g_mean_grid = self._uq_raw_get_field_grid(G, imtU)
                u_sig_grid = self._uq_raw_get_field_grid(U, f"STD{imtU}")
    
                if g_mean_grid is None or u_sig_grid is None:
                    rows.append({
                        "target_id": t["id"],
                        "target_type": t["type"],
                        "version": int(v),
                        "imt": imtU,
                        "source": "raw",
                        "mean": np.nan,
                        "sigma": np.nan,
                        "mean_min": np.nan,
                        "mean_max": np.nan,
                        "sigma_min": np.nan,
                        "sigma_max": np.nan,
                        "note": f"Missing fields in raw XML: {imtU} and/or STD{imtU}.",
                    })
                    continue
    
                r_mean = self._uq_agg(g_mean_grid[rmask], agg=agg)
                r_sig = self._uq_agg(u_sig_grid[rmask], agg=agg)
                r_mean_min, r_mean_max = self._uq_minmax(g_mean_grid[rmask])
                r_sig_min, r_sig_max = self._uq_minmax(u_sig_grid[rmask])
    
                rows.append({
                    "target_id": t["id"],
                    "target_type": t["type"],
                    "version": int(v),
                    "imt": imtU,
                    "source": "raw",
                    "mean": float(r_mean),
                    "sigma": float(r_sig),
                    "mean_min": float(r_mean_min),
                    "mean_max": float(r_mean_max),
                    "sigma_min": float(r_sig_min),
                    "sigma_max": float(r_sig_max),
                    "note": "",
                })
    
        df = pd.DataFrame(rows)
    
        # export
        if export_tables and output_path is not None:
            outp = Path(output_path)
            outp.mkdir(parents=True, exist_ok=True)
            df.to_csv(outp / f"{export_prefix}.csv", index=False)
    
        # plots
        if make_plots:
            for tid in sorted(df["target_id"].dropna().unique()):
                sub = df[df["target_id"] == tid].copy().sort_values(["version", "source"])
                if sub.empty:
                    continue
    
                for param in ("mean", "sigma"):
                    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
                    # raw
                    sraw = sub[sub["source"] == "raw"].sort_values("version")
                    if not sraw.empty:
                        ax.plot(sraw["version"].values, sraw[param].values, marker="o", linewidth=1.8, label="raw")
                        # envelope for areas/global
                        lo = sraw[f"{param}_min"].astype(float).values
                        hi = sraw[f"{param}_max"].astype(float).values
                        if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)) and (np.nanmax(hi - lo) > 0):
                            ax.fill_between(sraw["version"].values, lo, hi, alpha=float(envelope_alpha), label="raw min/max")
    
                    # unified
                    suni = sub[sub["source"] == "unified"].sort_values("version")
                    if not suni.empty:
                        ax.plot(suni["version"].values, suni[param].values, marker="o", linewidth=1.8, label="unified")
                        lo = suni[f"{param}_min"].astype(float).values
                        hi = suni[f"{param}_max"].astype(float).values
                        if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)) and (np.nanmax(hi - lo) > 0):
                            ax.fill_between(suni["version"].values, lo, hi, alpha=float(envelope_alpha), label="unified min/max")
    
                    ax.set_xlabel("ShakeMap version")
                    ax.set_ylabel(f"{param} ({imtU})" if param == "sigma" else f"{imtU}")
                    if show_grid:
                        ax.grid(True, alpha=0.3)
                    ax.tick_params(axis="x", rotation=float(xrotation))
                    if show_title:
                        if title is not None:
                            ax.set_title(str(title).replace("{target}", tid).replace("{imt}", imtU).replace("{param}", param))
                        else:
                            ax.set_title(f"RAW vs UNIFIED @ {tid} — {imtU} — {param}")
                    if legend:
                        ax.legend(**(legend_kwargs or {}))
                    fig.tight_layout()
    
                    if save and output_path is not None:
                        self._uq_save_figure_safe(
                            fig,
                            fname_stem=f"{export_prefix}-{imtU}-{param}-{tid}",
                            subdir="uq_plots/raw_vs_unified_curves",
                            output_path=output_path,
                            save_formats=save_formats,
                            dpi=dpi,
                        )
                    plt.show()
    
        return df
    """

    # ======================================================================
    # PATCH 4.1 (ADD-ON): Prediction+Uncertainty Comparative Framework
    #   - Adds explicit "published vs predicted" framing for BOTH mean and sigma
    #   - Adds USGS (ShakeMap) area-band shading (min/max across target mask)
    #   - Adds raw-XML vs unified-grid curve comparison (point OR area)
    #   - Hardens raw XML discovery + parsing + sampling
    #
    # Paste at the VERY END of the SHAKEtime class (after Patch 4).
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
    




    def _uq_parse_gridxml_to_arrays(self, grid_xml_path):
        """
        Patch 4.1++: Safer ShakeMap XML parser (schema + namespace tolerant)
    
        Supports both ShakeMap XML variants:
          - <grid_spec ...>
          - <grid_specification ...>   (common USGS files)
        and tolerates XML namespaces.
    
        Keeps EXACTLY the same return schema as your current Patch 4.1 version:
          {
            "nlat","nlon","lat_min","lat_max","lon_min","lon_max",
            "lat_vec","lon_vec","fields","data"
          }
    
        Notes
        -----
        - lat_vec/lon_vec are constructed from spacing if available, else linspace.
        - data is the raw numeric table with nlat*nlon rows, not reshaped.
        """
        from pathlib import Path
        import numpy as np
        import xml.etree.ElementTree as ET
    
        p = Path(grid_xml_path)
        tree = ET.parse(p)
        root = tree.getroot()
    
        # --- grid spec: tolerate tag name + namespaces ---
        spec = root.find(".//{*}grid_spec")
        if spec is None:
            spec = root.find(".//{*}grid_specification")
        if spec is None:
            # brute-force fallback for weird namespace/tag variants
            for el in root.iter():
                tag = str(el.tag).lower()
                if tag.endswith("grid_spec") or tag.endswith("grid_specification"):
                    spec = el
                    break
        if spec is None:
            raise RuntimeError(f"No grid_spec/grid_specification found in {p}")
    
        nlon = int(float(spec.attrib.get("nlon")))
        nlat = int(float(spec.attrib.get("nlat")))
        lon_min = float(spec.attrib.get("lon_min"))
        lon_max = float(spec.attrib.get("lon_max"))
        lat_min = float(spec.attrib.get("lat_min"))
        lat_max = float(spec.attrib.get("lat_max"))
    
        # Prefer explicit spacing if provided (support multiple attribute names)
        dlon = (
            spec.attrib.get("nominal_lon_spacing", None)
            or spec.attrib.get("dlon", None)
            or spec.attrib.get("lon_spacing", None)
        )
        dlat = (
            spec.attrib.get("nominal_lat_spacing", None)
            or spec.attrib.get("dlat", None)
            or spec.attrib.get("lat_spacing", None)
        )
    
        try:
            dlon = float(dlon) if dlon is not None else None
        except Exception:
            dlon = None
        try:
            dlat = float(dlat) if dlat is not None else None
        except Exception:
            dlat = None
    
        if (dlon is not None) and (nlon > 1):
            lon_vec = lon_min + np.arange(nlon, dtype=float) * dlon
        else:
            lon_vec = np.linspace(lon_min, lon_max, nlon, dtype=float)
    
        if (dlat is not None) and (nlat > 1):
            lat_vec = lat_min + np.arange(nlat, dtype=float) * dlat
        else:
            lat_vec = np.linspace(lat_min, lat_max, nlat, dtype=float)
    
        # fields: tolerate namespaces
        fields = {}
        for gf in root.findall(".//{*}grid_field"):
            try:
                idx = int(gf.attrib.get("index"))
                name = str(gf.attrib.get("name")).strip().upper()
                fields[name] = idx - 1  # 1-based in XML
            except Exception:
                continue
    
        gd = root.find(".//{*}grid_data")
        if gd is None or gd.text is None:
            raise RuntimeError(f"No grid_data found in {p}")
    
        lines = [ln.strip() for ln in gd.text.strip().splitlines() if ln.strip()]
        arr = np.array([[float(x) for x in ln.split()] for ln in lines], dtype=float)
    
        expect = nlat * nlon
        if arr.shape[0] != expect:
            if arr.shape[0] > expect:
                arr = arr[-expect:, :]
            else:
                raise RuntimeError(f"grid_data rows mismatch in {p.name}: got {arr.shape[0]}, expected {expect}")
    
        return {
            "nlat": nlat, "nlon": nlon,
            "lat_min": lat_min, "lat_max": lat_max,
            "lon_min": lon_min, "lon_max": lon_max,
            "lat_vec": np.asarray(lat_vec, dtype=float),
            "lon_vec": np.asarray(lon_vec, dtype=float),
            "fields": fields,
            "data": arr,
        }
    

    
    # ----------------------------------------------------------------------
    # Patch: RAW sampler (fail-soft)
    # ----------------------------------------------------------------------
    def _uq_raw_sample_from_xml(self, grid_xml_path, uncertainty_xml_path, lat, lon, imt="MMI", sample="nearest"):
        """
        Sample RAW mean and RAW sigma from ShakeMap-published grid/uncertainty XMLs at (lat, lon).
    
        Returns dict with:
          - raw_mean, raw_sigma, nearest_km
          - error (optional)
        """
        import numpy as np
    
        imtU = str(imt).upper().strip()
        stdU = f"STD{imtU}"
    
        try:
            G = self._uq_parse_gridxml_to_arrays(grid_xml_path)
            U = self._uq_parse_gridxml_to_arrays(uncertainty_xml_path)
        except Exception as e:
            return {"raw_mean": np.nan, "raw_sigma": np.nan, "nearest_km": np.nan, "error": str(e)}
    
        # Field lookup
        gidx = G["field_index"].get(imtU, None)
        uidx = U["field_index"].get(stdU, None)
    
        if gidx is None:
            return {"raw_mean": np.nan, "raw_sigma": np.nan, "nearest_km": np.nan,
                    "error": f"IMT field '{imtU}' not found in grid fields: {G['fields']}"}
        if uidx is None:
            return {"raw_mean": np.nan, "raw_sigma": np.nan, "nearest_km": np.nan,
                    "error": f"STD field '{stdU}' not found in uncertainty fields: {U['fields']}"}
    
        # Need lat/lon vectors for nearest/bilinear
        latv = G["lat"]
        lonv = G["lon"]
        if latv is None or lonv is None:
            return {"raw_mean": np.nan, "raw_sigma": np.nan, "nearest_km": np.nan,
                    "error": "Missing lat/lon vectors from grid spec; cannot sample reliably."}
    
        latv = np.asarray(latv, dtype=float)
        lonv = np.asarray(lonv, dtype=float)
    
        # nearest indices
        iy = int(np.argmin(np.abs(latv - float(lat))))
        ix = int(np.argmin(np.abs(lonv - float(lon))))
    
        # Very light “nearest_km” estimate (planar approx) for diagnostics only
        # (kept simple; you already use it only as a diagnostic)
        dlat = (latv[iy] - float(lat)) * 111.0
        dlon = (lonv[ix] - float(lon)) * 111.0 * np.cos(np.deg2rad(float(lat)))
        nearest_km = float(np.sqrt(dlat * dlat + dlon * dlon))
    
        if str(sample).lower().strip() == "bilinear":
            # bilinear interpolation in index space
            # clamp to valid interior
            iy0 = max(0, min(len(latv) - 2, iy))
            ix0 = max(0, min(len(lonv) - 2, ix))
            y1, y2 = latv[iy0], latv[iy0 + 1]
            x1, x2 = lonv[ix0], lonv[ix0 + 1]
    
            # Handle decreasing vectors
            if y2 == y1 or x2 == x1:
                w_y = 0.0
                w_x = 0.0
            else:
                w_y = (float(lat) - y1) / (y2 - y1)
                w_x = (float(lon) - x1) / (x2 - x1)
    
            w_y = float(np.clip(w_y, 0.0, 1.0))
            w_x = float(np.clip(w_x, 0.0, 1.0))
    
            def _bilin(A):
                q11 = A[iy0, ix0]
                q21 = A[iy0, ix0 + 1]
                q12 = A[iy0 + 1, ix0]
                q22 = A[iy0 + 1, ix0 + 1]
                return (q11 * (1 - w_x) * (1 - w_y) +
                        q21 * (w_x) * (1 - w_y) +
                        q12 * (1 - w_x) * (w_y) +
                        q22 * (w_x) * (w_y))
    
            raw_mean = float(_bilin(G["data"][:, :, gidx]))
            raw_sigma = float(_bilin(U["data"][:, :, uidx]))
        else:
            # nearest
            raw_mean = float(G["data"][iy, ix, gidx])
            raw_sigma = float(U["data"][iy, ix, uidx])
    
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
    # 4.1-C) Override uq_plot_targets_decay to add USGS area-band shading
    # ----------------------------------------------------------------------
    def uq_plot_targets_decay(
        self,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        what="sigma_total",
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
        figsize=(9, 4.8),
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
        # NEW: label sizing
        label_size=None,
        tick_size=None,
        title_size=None,
        legend_size=None,
        xlabel="ShakeMap version",
        ylabel=None,
        linewidth=1.5,
        marker_style='o',
        markersize=10,
        markevery=1,

        # combined plot controls
        plot_combined=True,
        combined_figsize=(11, 6),
        combined_legend_ncol=2,
        # NEW Patch 4.1: area-band shading for published curves
        published_band=True,
        published_band_alpha=0.18,
        # audit
        audit=True,
        audit_output_path=None,
        audit_prefix=None,
    ):
        """
        Patch 4.1:
          - Same as Patch 4 uq_plot_targets_decay, PLUS:
            If target is an AREA (circle/bbox) or GLOBAL and method includes "ShakeMap",
            draw a min/max band across the target mask for ShakeMap ONLY.
    
          Band rules:
            - what == "mean": band uses mu_min/mu_max
            - what in sigma cols: band uses sig_min/sig_max (published RAW sigma)
            - band is NOT drawn for predicted methods (bayes/kriging/etc).
        """
        if legend_kwargs is None:
            legend_kwargs = {}
    
        methods = [("ShakeMap" if str(m).lower().strip() == "published" else m) for m in methods]
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
    
        # what aliases + delta support
        w = str(what).strip()
        wl = w.lower().replace("_", " ").strip()
        if wl in ("raw", "sigma raw", "sigma total raw", "sigma_total_raw"):
            w = "sigma_total_raw"
    
        if wl in ("delta mean", "delta median"):
            base_v = int(prior_version) if prior_version is not None else int([int(v) for v in version_list][0])
            df2 = df.copy()
            df2["delta_sigma_total"] = np.nan
            for tid in df2["target_id"].unique():
                for m in df2["method"].unique():
                    sub0 = df2[(df2["target_id"] == tid) & (df2["method"] == m) & (df2["version"] == base_v)]
                    if sub0.empty:
                        continue
                    base = float(sub0["sigma_total"].values[0])
                    idx = (df2["target_id"] == tid) & (df2["method"] == m)
                    df2.loc[idx, "delta_sigma_total"] = df2.loc[idx, "sigma_total"].astype(float) - base
            df = df2
            w = "delta_sigma_total"
    
        if w not in df.columns:
            raise ValueError(f'what="{what}" not available. Choose from: {sorted(df.columns.tolist())}')
    
        # For band shading we need target masks again -> rebuild unified axes once
        versions = [int(v) for v in (version_list or [])]
        _, lat2d, lon2d = self._uq_get_unified_for_versions(
            versions, imt=str(imt).upper(),
            grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
        )
    
        # reconstruct targets consistently with extractor
        targets_meta = {}
        if global_stat is not None:
            gs = str(global_stat).lower().strip()
            gid = f"GLOBAL_{gs.replace(' ', '_')}".replace("__", "_")
            targets = [{"id": gid, "type": "global"}]
        else:
            targets = self._uq_parse_targets(points=points, areas=areas)
    
        for t in targets:
            tid = t.get("id", "GLOBAL")
            if t.get("type") == "global":
                mask = np.isfinite(lat2d)
                ttype = "global"
            else:
                mask, _meta = self._uq_target_mask(t, lat2d, lon2d)
                ttype = t.get("type", "target")
            targets_meta[tid] = {"type": ttype, "mask": mask}
    
        targets = sorted(df["target_id"].unique().tolist())
    
        # per-target plots
        for tid in targets:
            sub = df[(df["target_id"] == tid) & (df["method"].isin(methods))].copy()
            sub = sub.sort_values(["version", "method"])
    
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
            # Patch 4.1: published band (area/global only)
            tinfo = targets_meta.get(tid, {})
            is_area_like = (tinfo.get("type") in ("area", "global")) and (tinfo.get("mask") is not None)
    
            if published_band and is_area_like and ("ShakeMap" in methods) and (w in ("mean", "sigma_total", "sigma_total_raw", "sigma_epistemic", "sigma_aleatoric")):
                mask = tinfo["mask"]
                xs = []
                lo = []
                hi = []
                for v in sorted(sub["version"].unique().tolist()):
                    band = self._uq_band_minmax_unified(
                        version=int(v), imt=str(imt).upper(),
                        lat2d=lat2d, lon2d=lon2d, mask=mask,
                        grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
                    )
                    xs.append(int(v))
                    if w == "mean":
                        lo.append(band["mu_min"]); hi.append(band["mu_max"])
                    else:
                        # uncertainty bands always from RAW published sigma field
                        lo.append(band["sig_min"]); hi.append(band["sig_max"])
                if len(xs) > 0:
                    ax.fill_between(xs, lo, hi, alpha=float(published_band_alpha), label="ShakeMap area min/max")
    
            # plot series
            for m in methods:
                s = sub[sub["method"] == m].sort_values("version")
                #ax.plot(s["version"].values, s[w].values, marker=marker_style ,linewidth=linewidth, label=m)
                ax.plot(
                    s["version"].values,
                    s[w].values,
                    marker=marker_style,
                    markersize=markersize,
                    markevery=markevery,
                    linewidth=linewidth,
                    label=m
                )

    
            if ylog:
                ax.set_yscale("log")
            if ymin is not None or ymax is not None:
                ax.set_ylim(ymin, ymax)
    
            ax.set_xlabel(str(xlabel))
            if ylabel is None:
                if w == "mean":
                    ax.set_ylabel(f"{str(imt).upper()}")
                else:
                    ax.set_ylabel(f"{w} ({str(imt).upper()})")
            else:
                ax.set_ylabel(str(ylabel))
    
            if show_grid:
                ax.grid(True, which="both", alpha=0.3)
    
            if show_title:
                if title is not None:
                    ax.set_title(str(title).replace("{target}", tid).replace("{imt}", str(imt).upper()), fontsize=title_size)
                else:
                    ax.set_title(f"{str(imt).upper()} target decay @ {tid} ({w})", fontsize=title_size)
    
            ax.tick_params(axis="x", rotation=float(xrotation))
            if tick_size is not None:
                ax.tick_params(axis="both", labelsize=float(tick_size))
            if label_size is not None:
                ax.xaxis.label.set_size(float(label_size))
                ax.yaxis.label.set_size(float(label_size))
    
            if legend:
                lk = dict(legend_kwargs)
                if legend_size is not None:
                    lk.setdefault("fontsize", float(legend_size))
                ax.legend(**lk)
    
            if tight:
                fig.tight_layout()
    
            if save and output_path is not None:
                self._uq_save_figure_safe(
                    fig,
                    fname_stem=f"UQ-TargetDecay-{str(imt).upper()}-{w}-{tid}",
                    subdir="uq_plots/targets_decay",
                    output_path=output_path,
                    save_formats=save_formats,
                    dpi=dpi,
                )
    
            if show:
                plt.show()
            else:
                plt.close(fig)
    
        # combined plot unchanged (kept simple; no bands to avoid clutter)
        if plot_combined and len(targets) > 1:
            fig, ax = plt.subplots(figsize=combined_figsize, dpi=dpi)
    
            for tid in targets:
                sub = df[(df["target_id"] == tid) & (df["method"].isin(methods))].copy()
                for m in methods:
                    s = sub[sub["method"] == m].sort_values("version")
                    ax.plot(s["version"].values, s[w].values, marker="o", linewidth=1.4, label=f"{tid} | {m}")
    
            if ylog:
                ax.set_yscale("log")
            if ymin is not None or ymax is not None:
                ax.set_ylim(ymin, ymax)
    
            ax.set_xlabel(str(xlabel))
            ax.set_ylabel(f"{w} ({str(imt).upper()})" if w != "mean" else f"{str(imt).upper()}")
    
            if show_grid:
                ax.grid(True, which="both", alpha=0.3)
    
            if show_title:
                ax.set_title(title if title is not None else f"Combined target decay ({str(imt).upper()} / {w})", fontsize=title_size)
    
            ax.tick_params(axis="x", rotation=float(xrotation))
            if tick_size is not None:
                ax.tick_params(axis="both", labelsize=float(tick_size))
            if label_size is not None:
                ax.xaxis.label.set_size(float(label_size))
                ax.yaxis.label.set_size(float(label_size))
    
            if legend:
                lk = dict(legend_kwargs)
                if legend_size is not None:
                    lk.setdefault("fontsize", float(legend_size))
                ax.legend(ncol=int(combined_legend_ncol), **lk)
    
            fig.tight_layout()
    
            if save and output_path is not None:
                self._uq_save_figure_safe(
                    fig,
                    fname_stem=f"UQ-TargetDecay-COMBINED-{str(imt).upper()}-{w}",
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


    def uq_plot_targets_decay(
        self,
        version_list,
        imt="MMI",
        points=None,
        areas=None,
        what="sigma_total",
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
        figsize=(9, 4.8),
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
        # NEW: label sizing
        label_size=None,
        tick_size=None,
        title_size=None,
        legend_size=None,
        xlabel="ShakeMap version",
        ylabel=None,
        linewidth=1.5,
        markerstyle="o",
        markersize=10,
        markevery=1,
        
        # combined plot controls
        plot_combined=True,
        combined_figsize=(11, 6),
        combined_legend_ncol=2,
        # NEW Patch 4.1: area-band shading for published curves
        published_band=True,
        published_band_alpha=0.18,
        # audit
        audit=True,
        audit_output_path=None,
        audit_prefix=None,
    ):
        """
        Patch 4.1 (fixed):
          - Draw ShakeMap published min/max band for AREA/GLOBAL targets whenever "ShakeMap" is in methods.
          - Band type auto-selected:
              * mean-like `what` -> mu_min/mu_max
              * otherwise -> sig_min/sig_max (RAW published sigma field)
        """
    
        import numpy as np
        import matplotlib.pyplot as plt
    
        # normalize methods (allow "published" alias)
        methods = [("ShakeMap" if str(m).lower().strip() == "published" else m) for m in methods]
        methods = tuple(methods)
    
        # -----------------------------
        # 1) Extract series
        # -----------------------------
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
    
        # -----------------------------
        # 2) "what" aliases + delta support
        # -----------------------------
        w = str(what).strip()
    
        # common aliases you used in notebooks
        alias = {
            "sigma_total": "sigma_total_predicted",
            "sigma_total_raw": "sigma_total_published_raw",
            "sigma_epistemic": "sigma_epistemic_predicted",
            "sigma_aleatoric": "sigma_aleatoric_used",
            "mean": "mean_predicted",
        }
        if w in alias:
            w = alias[w]
    
        # delta support: delta_<col> means relative to first shown version (per target/method)
        if w.lower().startswith("delta_"):
            base_col = w[len("delta_") :].strip()
            if base_col not in df.columns:
                raise ValueError(f'delta requested but base column "{base_col}" not found. Choose from: {sorted(df.columns.tolist())}')
            df2 = df.copy()
            df2[w] = np.nan
            for tid in df2["target_id"].unique().tolist():
                for m in df2["method"].unique().tolist():
                    sub0 = df2[(df2["target_id"] == tid) & (df2["method"] == m)].sort_values("version")
                    if len(sub0) == 0:
                        continue
                    base = float(sub0[base_col].values[0])
                    idx = (df2["target_id"] == tid) & (df2["method"] == m)
                    df2.loc[idx, w] = df2.loc[idx, base_col].astype(float) - base
            df = df2
    
        if w not in df.columns:
            raise ValueError(f'what="{what}" not available. Choose from: {sorted(df.columns.tolist())}')
    
        # -----------------------------
        # 3) Rebuild unified axes once (for band shading masks)
        # -----------------------------
        versions = [int(v) for v in (version_list or [])]
        _, lat2d, lon2d = self._uq_get_unified_for_versions(
            versions, imt=str(imt).upper(),
            grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
        )
    
        # reconstruct targets consistently with extractor
        targets_meta = {}
        if global_stat is not None:
            gs = str(global_stat).lower().strip()
            gid = f"GLOBAL_{gs.replace(' ', '_')}".replace("__", "_")
            targets = [{"id": gid, "type": "global"}]
        else:
            targets = self._uq_parse_targets(points=points, areas=areas)
    
        for t in targets:
            tid = t.get("id", "GLOBAL")
            if t.get("type") == "global":
                mask = np.isfinite(lat2d)
                ttype = "global"
            else:
                mask, _meta = self._uq_target_mask(t, lat2d, lon2d)
                ttype = t.get("type", "target")
            targets_meta[tid] = {"type": ttype, "mask": mask}
    
        targets = sorted(df["target_id"].unique().tolist())
    
        # helper: detect if the plotted y is mean-like
        w_is_mean_like = str(w).lower().startswith("mean")
    
        # -----------------------------
        # 4) Per-target plots
        # -----------------------------
        for tid in targets:
            sub = df[(df["target_id"] == tid) & (df["method"].isin(methods))].copy()
            sub = sub.sort_values(["version", "method"])
    
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
            # Patch 4.1 FIXED: published band for AREA/GLOBAL whenever ShakeMap is present
            tinfo = targets_meta.get(tid, {})
            is_area_like = (tinfo.get("type") in ("area", "global")) and (tinfo.get("mask") is not None)
    
            if published_band and is_area_like and ("ShakeMap" in methods):
                mask = tinfo["mask"]
                xs, lo, hi = [], [], []
                for v in sorted(sub["version"].unique().tolist()):
                    band = self._uq_band_minmax_unified(
                        version=int(v), imt=str(imt).upper(),
                        lat2d=lat2d, lon2d=lon2d, mask=mask,
                        grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
                    )
                    xs.append(int(v))
                    if w_is_mean_like:
                        lo.append(band["mu_min"]); hi.append(band["mu_max"])
                    else:
                        # uncertainty bands always from RAW published sigma field
                        lo.append(band["sig_min"]); hi.append(band["sig_max"])
                if len(xs) > 0:
                    ax.fill_between(xs, lo, hi, alpha=float(published_band_alpha), label="ShakeMap min/max (area/global)")
    
            # plot series
            for m in methods:
                s = sub[sub["method"] == m].sort_values("version")
                #ax.plot(s["version"].values, s[w].values, marker=marker, linewidth=linewidth, label=m)
                ax.plot(
                        s["version"].values,
                        s[w].values,
                        marker=markerstyle,
                        markersize=markersize,
                        markevery=markevery,
                        linewidth=linewidth,
                        label=m
                    )

    
            if ylog:
                ax.set_yscale("log")
            if ymin is not None or ymax is not None:
                ax.set_ylim(ymin, ymax)
    
            ax.set_xlabel(str(xlabel))
            if ylabel is None:
                if w_is_mean_like:
                    ax.set_ylabel(f"{str(imt).upper()} mean")
                else:
                    ax.set_ylabel(f"{w}")
            else:
                ax.set_ylabel(str(ylabel))
    
            if show_grid:
                ax.grid(True, which="both", alpha=0.3)
    
            if show_title:
                if title is not None:
                    ax.set_title(str(title).replace("{target}", tid).replace("{imt}", str(imt).upper()), fontsize=title_size)
                else:
                    ax.set_title(f"{str(imt).upper()} target decay @ {tid} ({w})", fontsize=title_size)
    
            ax.tick_params(axis="x", rotation=float(xrotation))
            if tick_size is not None:
                ax.tick_params(axis="both", labelsize=float(tick_size))
            if label_size is not None:
                ax.xaxis.label.set_size(float(label_size))
                ax.yaxis.label.set_size(float(label_size))
    
            if legend:
                lk = dict(legend_kwargs or {})
                if legend_size is not None:
                    lk.setdefault("fontsize", float(legend_size))
                ax.legend(**lk)
    
            if tight:
                fig.tight_layout()
    
            if save and output_path is not None:
                self._uq_save_figure_safe(
                    fig,
                    fname_stem=f"UQ-TargetDecay-{str(imt).upper()}-{tid}-{w}",
                    subdir="uq_plots/targets_decay",
                    output_path=output_path,
                    save_formats=save_formats,
                    dpi=dpi,
                )
    
            if show:
                plt.show()
            else:
                plt.close(fig)
    
        # -----------------------------
        # 5) Combined plot (all targets)
        # -----------------------------
        if plot_combined:
            fig, ax = plt.subplots(figsize=combined_figsize, dpi=dpi)
    
            for tid in targets:
                sub = df[(df["target_id"] == tid) & (df["method"].isin(methods))].copy()
                sub = sub.sort_values(["version", "method"])
                for m in methods:
                    s = sub[sub["method"] == m].sort_values("version")
                    if len(s) == 0:
                        continue
                    #ax.plot(s["version"].values, s[w].values, marker=marker, linewidth=linewidth, label=f"{tid} — {m}")
                    ax.plot(
                        s["version"].values,
                        s[w].values,
                        marker=markerstyle,
                        markersize=markersize,
                        markevery=markevery,
                        linewidth=linewidth,
                        label=f"{tid} — {m}"
                    )


    
            if ylog:
                ax.set_yscale("log")
            if ymin is not None or ymax is not None:
                ax.set_ylim(ymin, ymax)
    
            ax.set_xlabel(str(xlabel))
            if ylabel is None:
                ax.set_ylabel(f"{w}")
            else:
                ax.set_ylabel(str(ylabel))
    
            if show_grid:
                ax.grid(True, which="both", alpha=0.3)
    
            if show_title:
                ax.set_title(f"{str(imt).upper()} combined targets decay ({w})", fontsize=title_size)
    
            ax.tick_params(axis="x", rotation=float(xrotation))
            if tick_size is not None:
                ax.tick_params(axis="both", labelsize=float(tick_size))
            if label_size is not None:
                ax.xaxis.label.set_size(float(label_size))
                ax.yaxis.label.set_size(float(label_size))
    
            if legend:
                lk = dict(legend_kwargs or {})
                if legend_size is not None:
                    lk.setdefault("fontsize", float(legend_size))
                ax.legend(ncol=int(combined_legend_ncol), **lk)
    
            fig.tight_layout()
    
            if save and output_path is not None:
                self._uq_save_figure_safe(
                    fig,
                    fname_stem=f"UQ-TargetDecay-COMBINED-{str(imt).upper()}-{w}",
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
    
    
    # ----------------------------------------------------------------------
    # 4.1-E) NEW: Raw XML vs Unified curve comparison (point OR area)
    # ----------------------------------------------------------------------
    """
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
        raw_sample="nearest",         # "nearest" | "bilinear"
        unified_sample="nearest",     # currently nearest via mask/agg; kept for API symmetry
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
    ):
        if legend_kwargs is None:
            legend_kwargs = {}
        if interp_kwargs is None:
            interp_kwargs = {}
    
        versions = [int(v) for v in (version_list or [])]
        if not versions:
            raise ValueError("version_list cannot be empty.")
    
        imtU = str(imt).upper()
    
        # targets
        targets = self._uq_parse_targets(points=points, areas=areas)
    
        # unified axes
        _, lat2d, lon2d = self._uq_get_unified_for_versions(
            versions, imt=imtU, grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
        )
    
        rows = []
    
        for t in targets:
            tid = t.get("id", "target")
            mask, meta = self._uq_target_mask(t, lat2d, lon2d)
            is_area = (t.get("type") == "area")
    
            # Precompute unified min/max bands easily
            uni_mu_min = []; uni_mu_max = []
            uni_sg_min = []; uni_sg_max = []
    
            raw_mu_min = []; raw_mu_max = []
            raw_sg_min = []; raw_sg_max = []
    
            xs = []
    
            # If point target, representative lat/lon = the point
            if t["type"] == "point":
                t_lat = float(t["lat"]); t_lon = float(t["lon"])
    
            for v in versions:
                xs.append(int(v))
    
                # --- UNIFIED ---
                mu_u, sg_u = self._uq_get_mu_sigma_unified(
                    int(v), imtU, lat2d, lon2d,
                    grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
                )
                mu_u = np.asarray(mu_u, dtype=float)
                sg_u = np.asarray(sg_u, dtype=float)
    
                uni_mu_agg = self._uq_agg(mu_u[mask], agg=agg)
                uni_sg_agg = self._uq_agg(sg_u[mask], agg=agg)
    
                if band and is_area:
                    b = self._uq_band_minmax_unified(
                        version=int(v), imt=imtU,
                        lat2d=lat2d, lon2d=lon2d, mask=mask,
                        grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
                    )
                    uni_mu_min.append(b["mu_min"]); uni_mu_max.append(b["mu_max"])
                    uni_sg_min.append(b["sig_min"]); uni_sg_max.append(b["sig_max"])
                else:
                    uni_mu_min.append(np.nan); uni_mu_max.append(np.nan)
                    uni_sg_min.append(np.nan); uni_sg_max.append(np.nan)
    
                # --- RAW ---
                grid_p = self._uq_find_shakemap_xml(int(v), which="grid", base_folder=base_folder, shakemap_folder=shakemap_folder)
                unc_p  = self._uq_find_shakemap_xml(int(v), which="uncertainty", base_folder=base_folder, shakemap_folder=shakemap_folder)
    
                if (grid_p is None) or (unc_p is None):
                    raw_mu_agg = np.nan
                    raw_sg_agg = np.nan
                    rmin_mu = rmax_mu = np.nan
                    rmin_sg = rmax_sg = np.nan
                else:
                    if t["type"] == "point":
                        raw = self._uq_raw_sample_from_xml(grid_p, unc_p, t_lat, t_lon, imt=imtU, sample=raw_sample)
                        raw_mu_agg = float(raw["raw_mean"]) if raw is not None else np.nan
                        raw_sg_agg = float(raw["raw_sigma"]) if raw is not None else np.nan
                        rmin_mu = rmax_mu = raw_mu_agg
                        rmin_sg = rmax_sg = raw_sg_agg
                    else:
                        # area/global: approximate raw sampling by mapping each unified masked cell
                        # to raw nearest node (fast enough for moderate masks; for huge masks consider thinning)
                        G = self._uq_parse_gridxml_to_arrays(grid_p)
                        U = self._uq_parse_gridxml_to_arrays(unc_p)
    
                        latv = G["lat_vec"]; lonv = G["lon_vec"]
                        nlon = G["nlon"]
    
                        def _get(D, i, j, fieldU):
                            if fieldU not in D["fields"]:
                                return np.nan
                            c = D["fields"][fieldU]
                            idx = i * nlon + j
                            if c < 0 or c >= D["data"].shape[1]:
                                return np.nan
                            return float(D["data"][idx, c])
    
                        # iterate masked indices
                        ij = np.argwhere(mask)
                        raw_mu_vals = []
                        raw_sg_vals = []
                        for (ii, jj) in ij:
                            latc = float(lat2d[ii, jj])
                            lonc = float(lon2d[ii, jj])
                            ir = int(np.nanargmin(np.abs(latv - latc)))
                            jr = int(np.nanargmin(np.abs(lonv - lonc)))
                            raw_mu_vals.append(_get(G, ir, jr, imtU))
                            raw_sg_vals.append(_get(U, ir, jr, f"STD{imtU}"))
    
                        raw_mu_agg = self._uq_agg(np.asarray(raw_mu_vals, dtype=float), agg=agg)
                        raw_sg_agg = self._uq_agg(np.asarray(raw_sg_vals, dtype=float), agg=agg)
    
                        vv = np.asarray(raw_mu_vals, dtype=float); vv = vv[np.isfinite(vv)]
                        ww = np.asarray(raw_sg_vals, dtype=float); ww = ww[np.isfinite(ww)]
                        rmin_mu = float(np.nanmin(vv)) if vv.size else np.nan
                        rmax_mu = float(np.nanmax(vv)) if vv.size else np.nan
                        rmin_sg = float(np.nanmin(ww)) if ww.size else np.nan
                        rmax_sg = float(np.nanmax(ww)) if ww.size else np.nan
    
                if band and is_area:
                    raw_mu_min.append(rmin_mu); raw_mu_max.append(rmax_mu)
                    raw_sg_min.append(rmin_sg); raw_sg_max.append(rmax_sg)
                else:
                    raw_mu_min.append(np.nan); raw_mu_max.append(np.nan)
                    raw_sg_min.append(np.nan); raw_sg_max.append(np.nan)
    
                rows.append({
                    "target_id": tid,
                    "target_type": t.get("type", "target"),
                    "version": int(v),
                    "imt": imtU,
                    "agg": str(agg),
                    "unified_mean": float(uni_mu_agg),
                    "unified_sigma": float(uni_sg_agg),
                    "unified_mean_min": float(uni_mu_min[-1]),
                    "unified_mean_max": float(uni_mu_max[-1]),
                    "unified_sigma_min": float(uni_sg_min[-1]),
                    "unified_sigma_max": float(uni_sg_max[-1]),
                    "raw_mean": float(raw_mu_agg),
                    "raw_sigma": float(raw_sg_agg),
                    "raw_mean_min": float(raw_mu_min[-1]),
                    "raw_mean_max": float(raw_mu_max[-1]),
                    "raw_sigma_min": float(raw_sg_min[-1]),
                    "raw_sigma_max": float(raw_sg_max[-1]),
                    "raw_grid_path": str(grid_p) if grid_p else None,
                    "raw_unc_path": str(unc_p) if unc_p else None,
                    "mask_kind": meta.get("kind", ""),
                    "n_cells": int(meta.get("n_cells", int(mask.sum()))),
                })
    
            # ---- Plot per target ----
            df_t = pd.DataFrame([r for r in rows if r["target_id"] == tid]).sort_values("version")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
    
            x = df_t["version"].values
    
            # Mean panel
            if band and is_area:
                ax1.fill_between(x, df_t["raw_mean_min"].values, df_t["raw_mean_max"].values, alpha=float(band_alpha), label="RAW min/max (area)")
                ax1.fill_between(x, df_t["unified_mean_min"].values, df_t["unified_mean_max"].values, alpha=float(band_alpha), label="UNIFIED min/max (area)")
            ax1.plot(x, df_t["raw_mean"].values, marker="o", linewidth=2.0, label="RAW mean")
            ax1.plot(x, df_t["unified_mean"].values, marker="o", linewidth=2.0, label="UNIFIED mean")
            ax1.set_ylabel(f"{imtU} mean")
            if show_grid:
                ax1.grid(True, alpha=0.3)
    
            # Sigma panel
            if band and is_area:
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
                    fname_stem=f"UQ-RawVsUnifiedCurves-{imtU}-{tid}",
                    subdir="uq_plots/raw_vs_unified_curves",
                    output_path=output_path,
                    save_formats=save_formats,
                    dpi=dpi,
                )
    
            if show:
                plt.show()
            else:
                plt.close(fig)
    
        return pd.DataFrame(rows)
    """

    # ======================================================================
    # PATCH 4.1-B (ADD-ON): Fixes + Residual/Skill Diagnostics + Safer Area RAW sampling
    #   - Adds mask subsampling utilities to keep RAW area sampling tractable
    #   - Overrides uq_compare_raw_vs_unified_curves() to support subsampling + CSV export
    #   - Adds published-vs-predicted residual tables + optional plots
    #
    # Paste AFTER your Patch 4.1 block (end of class). Later defs override earlier ones.
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
    
    
    def _uq_area_raw_values_from_xml(
        self,
        grid_p,
        unc_p,
        imtU,
        lat2d,
        lon2d,
        mask,
        *,
        max_cells=None,
        stride=None,
        random_state=42,
        subsample_method="auto",
    ):
        """
        Vectorized-ish RAW sampling for an area mask by mapping each (lat,lon) cell to nearest RAW node.
    
        Returns
        -------
        raw_mu_vals, raw_sg_vals : 1D arrays (length = n_returned)
        meta : dict with subsampling + availability info
        """
        meta = {"raw_grid_path": str(grid_p) if grid_p else None, "raw_unc_path": str(unc_p) if unc_p else None}
    
        if grid_p is None or unc_p is None:
            meta.update({"status": "missing_raw_xml"})
            return np.asarray([], dtype=float), np.asarray([], dtype=float), meta
    
        # parse once
        G = self._uq_parse_gridxml_to_arrays(grid_p)
        U = self._uq_parse_gridxml_to_arrays(unc_p)
    
        latv = np.asarray(G["lat_vec"], dtype=float)
        lonv = np.asarray(G["lon_vec"], dtype=float)
        nlon = int(G["nlon"])
    
        f_mu = str(imtU).upper().strip()
        f_sg = f"STD{f_mu}"
    
        # field lookup
        if f_mu not in G["fields"]:
            meta.update({"status": "missing_field", "missing": f_mu})
            return np.asarray([], dtype=float), np.asarray([], dtype=float), meta
        if f_sg not in U["fields"]:
            meta.update({"status": "missing_field", "missing": f_sg})
            return np.asarray([], dtype=float), np.asarray([], dtype=float), meta
    
        c_mu = int(G["fields"][f_mu])
        c_sg = int(U["fields"][f_sg])
    
        ij, smeta = self._uq_mask_indices(mask, max_cells=max_cells, stride=stride, random_state=random_state, method=subsample_method)
        meta.update({"status": "ok", **smeta})
    
        if ij.shape[0] == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), meta
    
        # map each unified cell to nearest raw node indices
        latc = np.asarray(lat2d[ij[:, 0], ij[:, 1]], dtype=float)
        lonc = np.asarray(lon2d[ij[:, 0], ij[:, 1]], dtype=float)
    
        # nearest indices (rectilinear)
        ir = np.nanargmin(np.abs(latv.reshape(-1, 1) - latc.reshape(1, -1)), axis=0)
        jr = np.nanargmin(np.abs(lonv.reshape(-1, 1) - lonc.reshape(1, -1)), axis=0)
    
        idx = ir * nlon + jr
    
        mu = G["data"][idx, c_mu].astype(float)
        sg = U["data"][idx, c_sg].astype(float)
    
        return mu, sg, meta
    
    
    # ----------------------------------------------------------------------
    # 4.1-G) Override: RAW vs UNIFIED curves with subsampling + exports + diagnostics
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
        # NEW (Patch 4.1-B): subsampling RAW area mapping
        raw_area_max_cells=25000,     # None disables cap
        raw_area_stride=None,         # e.g., 2, 3, 5
        raw_area_random_state=42,
        raw_area_subsample_method="auto",  # "auto"|"stride"|"random"
        # NEW: export table
        export_table=True,
        export_prefix="UQ-RawVsUnifiedCurves",
    ):
        """
        Patch 4.1-B (override):
          - Same output/plot as Patch 4.1-E, but with RAW area subsampling controls and CSV export.
          - Adds diagnostics columns for RAW subsampling decisions.
    
        Returns
        -------
        df_curves : DataFrame
        """
        if legend_kwargs is None:
            legend_kwargs = {}
        if interp_kwargs is None:
            interp_kwargs = {}
    
        versions = [int(v) for v in (version_list or [])]
        if not versions:
            raise ValueError("version_list cannot be empty.")
    
        imtU = str(imt).upper()
    
        targets = self._uq_parse_targets(points=points, areas=areas)
    
        # unified axes
        _, lat2d, lon2d = self._uq_get_unified_for_versions(
            versions, imt=imtU, grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
        )
    
        rows = []
    
        for t in targets:
            tid = t.get("id", "target")
            mask, meta = self._uq_target_mask(t, lat2d, lon2d)
            is_area_like = (t.get("type") == "area")
    
            # point lat/lon
            if t.get("type") == "point":
                t_lat = float(t["lat"])
                t_lon = float(t["lon"])
    
            for v in versions:
                # ---- UNIFIED ----
                mu_u, sg_u = self._uq_get_mu_sigma_unified(
                    int(v), imtU, lat2d, lon2d,
                    grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
                )
                mu_u = np.asarray(mu_u, dtype=float)
                sg_u = np.asarray(sg_u, dtype=float)
    
                uni_mu_agg = self._uq_agg(mu_u[mask], agg=agg)
                uni_sg_agg = self._uq_agg(sg_u[mask], agg=agg)
    
                if band and is_area_like:
                    b = self._uq_band_minmax_unified(
                        version=int(v), imt=imtU,
                        lat2d=lat2d, lon2d=lon2d, mask=mask,
                        grid_res=grid_res, interp_method=interp_method, interp_kwargs=interp_kwargs
                    )
                    uni_mu_min, uni_mu_max = b["mu_min"], b["mu_max"]
                    uni_sg_min, uni_sg_max = b["sig_min"], b["sig_max"]
                else:
                    uni_mu_min = uni_mu_max = np.nan
                    uni_sg_min = uni_sg_max = np.nan
    
                # ---- RAW ----
                grid_p = self._uq_find_shakemap_xml(int(v), which="grid", base_folder=base_folder, shakemap_folder=shakemap_folder)
                unc_p  = self._uq_find_shakemap_xml(int(v), which="uncertainty", base_folder=base_folder, shakemap_folder=shakemap_folder)
    
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
                            subs_meta = {"raw_sample": str(raw_sample), "raw_nearest_km": float(raw.get("nearest_km", np.nan))}
                    else:
                        # area/global: subsampled mapping
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
                    # subsampling diagnostics
                    "raw_area_subsample_method": subs_meta.get("method", None),
                    "raw_area_n_cells": subs_meta.get("n_cells", None),
                    "raw_area_n_returned": subs_meta.get("n_returned", None),
                    "raw_area_stride": subs_meta.get("stride", None),
                    "raw_area_max_cells": subs_meta.get("max_cells", None),
                    "raw_status": subs_meta.get("status", None),
                    "raw_missing_field": subs_meta.get("missing", None),
                    "raw_nearest_km": subs_meta.get("raw_nearest_km", None),
                })
    
            # ---- Plot per target (same layout as 4.1-E) ----
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
    
        if export_table and output_path is not None:
            outp = Path(output_path)
            outp.mkdir(parents=True, exist_ok=True)
            df_curves.to_csv(outp / f"{export_prefix}-{imtU}.csv", index=False)
    
        return df_curves
    
    
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


    # ============================ END PATCH 4 ============================
