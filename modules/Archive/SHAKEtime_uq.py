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

from modules.SHAKEparser import *
from modules.SHAKEmapper import *
from modules.SHAKEtools import *
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



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Assume that AccelerationUnitConverter, USGSParser, and SHAKEmapper are defined and imported elsewhere.
# For example:
# from my_unit_converter import AccelerationUnitConverter
# from my_parser_module import USGSParser
# from my_mapper_module import SHAKEmapper

class SHAKEtime:
    """
    SHAKEtime: A Class for Time Dependent Earthquake Impact Assessment
    
    Overview
    ========
    The SHAKEtime class processes multiple versions of ShakeMap files and Pager data
    to construct a timeline of ShakeMap product development for a given earthquake event.
    It supports reading ShakeMap XML (and corresponding uncertainty files) and Pager XML files,
    computing summary statistics (e.g. mean, std) for key intensity measures, and plotting
    various maps, including:
        - Hazard Temporal Display (HTD) plots.
        - Rate-of-change maps.
        - Standard deviation (uncertainty) maps.
        - Population exposure maps.
        - ShakeMap maps with rupture data overlaid.
    
    File Naming Conventions
    -----------------------
    The class accepts a parameter `file_type` to determine the file naming conventions:
    
    For file_type = 1:
        - Pager:         "{event_id}_pager_{version}.xml"
        - ShakeMap:      "{event_id}_grid_{version}.xml"
        - Rupture:       "{event_id}_rupture_{version}.json"
    
    For file_type = 2 (default):
        - Pager:         "{event_id}_us_{version}_pager.xml"
        - ShakeMap:      "{event_id}_us_{version}_grid.xml"
        - Rupture:       "{event_id}_us_{version}_rupture.json"
    
    Colorbar Options
    ----------------
    For map plotting functions (plot_rate_maps, plot_shakemaps, and plot_std_maps),
    there is an option (plot_colorbar) to turn the colorbar on or off.
    
    Usage Example
    =============
        event_id = 'us7000pn9s'
        event_time = '2025-03-28 06:20:52'
        shakemap_folder = "/export/SHAKEfetch/usgs-shakemap-versions"
        pager_folder = "/export/SHAKEfetch/usgs-pager-versions"
        rupture_folder = "/export/SHAKEfetch/usgs-rupture-versions"
        version_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012']
        
        # Create an instance (default file_type is 2):
        shake = SHAKEtime(event_id, event_time, shakemap_folder, pager_folder, file_type=2)
        
        # Build summary and update metrics:
        shake.get_shake_summary(version_list)
        shake.add_shakemap_pgm()
        shake.add_shakemap_stdpgm()
        shake.add_pager_exposure()
        shake.add_cities_impact(['Mandalay', 'Nay Pyi Taw', 'Chiang Mai', 'Bangkok', 'Taungoo'])
        shake.add_rate_to_summary(version_list=version_list, metric="mmi")
        
        # Plot maps (HTD, rate maps, standard deviation maps, exposure, ShakeMap)
        fig_htd, ax_htd = shake.plot_HTD(metric_type="mmi")
        plt.show()
        
        rate_maps = shake.plot_rate_maps(version_list, plot_colorbar='on', output_path="./export")
        for fig, ax, version in rate_maps:
            plt.show()
        
        std_maps = shake.plot_std_maps(version_list, metric="mmi", plot_colorbar="on", rupture_folder=rupture_folder, output_path="./export")
        for fig, ax, version in std_maps:
            plt.show()
        
        exposure_fig, exposure_ax = shake.plot_pop_exposure(version_list=version_list, output_path="./export")
        plt.show()
        
        shakemap_figures = shake.plot_shakemaps(version_list, rupture_folder, plot_colorbar='on', output_path="./export")
        for fig, ax, version in shakemap_figures:
            plt.show()



        ──────────────────────────────────────────────────────────────────────────
    Analysis routines usage summary
    ──────────────────────────────────────────────────────────────────────────

    1) analyze_auxiliary_influences(...)
       • Always runs the three “base” covariate generators when folders are provided:
         – compute_station_covariates(): local station density, mean/std of PGA observations.
         – compute_dyfi_covariates(): count, footprint area, residuals vs. ShakeMap.
         – compute_rupture_geometry(): trace length & segment count.
       • Always merges in all diagnostics from quantify_evolution:
         – compute_threshold_exceedance_area(): area above a chosen MMI threshold.
         – spatial_correlation(): Pearson/Spearman between successive maps.
         – compute_global_diff_stats(): MAE, RMSE, mean-bias between grids.
         – variogram_analysis(): semivariance vs. lag → nugget, sill, range.
         – area_of_uncertainty_change(): area where σ exceeds a percentile.
         – bootstrap_uncertainty(): CI on mean/std of each map.
       • If you pass extra=[…], it calls add_auxiliary(), which dispatches to exactly
         those of the six “extra” methods you requested:
           • morans_i()                – global Moran’s I (spatial autocorrelation)
           • getis_ord_gi()            – Getis–Ord Gi* (hotspot score)
           • pca_analysis()            – PCA on stacked delta-grids → pca_ev_*
           • bayesian_true_field()     – Bayesian hierarchical field (μ, σ)
           • jaccard_similarity()      – Jaccard exceedance-set similarity
           • functional_data_analysis()– functional-data band-depth

    2) quantify_evolution(...)
       • Computes only its six core diagnostics, in this order:
         – compute_threshold_exceedance_area()
         – spatial_correlation()
         – compute_global_diff_stats()
         – variogram_analysis()
         – area_of_uncertainty_change()
         – bootstrap_uncertainty()

    3) Unused analysis methods
       • lisa_clusters()
       • forecast_summary()

    ──────────────────────────────────────────────────────────────────────────
    Implemented method definitions
    ──────────────────────────────────────────────────────────────────────────

    Base covariates:
      • compute_station_covariates(): local station density, mean/std of PGA observations.
      • compute_dyfi_covariates(): count, footprint area, residuals vs. ShakeMap.
      • compute_rupture_geometry(): rupture trace length & segment count.

    Evolution diagnostics:
      • compute_threshold_exceedance_area(): area above a chosen MMI threshold.
      • spatial_correlation(): Pearson or Spearman correlation between successive maps.
      • compute_global_diff_stats(): MAE, RMSE, mean-bias between two ShakeMap versions.
      • variogram_analysis(): experimental semivariogram → nugget, sill, range.
      • area_of_uncertainty_change(): area where ShakeMap σ exceeds a given percentile.
      • bootstrap_uncertainty(): bootstrap CIs on map mean and std.

    Optional “extra” analyses (via add_auxiliary()):
      • morans_i(): Moran’s I for global spatial autocorrelation.
      • getis_ord_gi(): Getis–Ord Gi* hotspot statistic.
      • pca_analysis(): PCA on the stack of delta-grids → principal components.
      • bayesian_true_field(): Bayesian hierarchical posterior mean & σ of the field.
      • jaccard_similarity(): Jaccard index of exceedance‐set overlap.
      • functional_data_analysis(): functional‐data band‐depth of spatial fields.

    ──────────────────────────────────────────────────────────────────────────
    Future extensions (ideas for additional methods)
    ──────────────────────────────────────────────────────────────────────────

    In quantify_evolution:
      • Trend & change-point detection
        – Mann–Kendall trend test on global metrics.
        – CUSUM or Bayesian change-point detection to flag abrupt shifts.
      • Information-theoretic distances
        – Kullback–Leibler or Hellinger distance between grid‐value histograms.
      • Directional variogram
        – Compute variograms along fault-parallel vs. fault-normal axes to reveal anisotropy.

    In analyze_auxiliary_influences / add_auxiliary:
      • Local Indicators of Spatial Association (LISA)
        – Expose lisa_clusters() to identify spatial clusters of change.
      • Network centrality of station network
        – Build a station-graph (e.g. Delaunay triangulation) and compute betweenness or closeness centrality.
      • Minkowski functionals / Morphological metrics
        – Compute area, perimeter, Euler characteristic of >MMI contours to capture shape complexity.
      • Entropy & complexity measures
        – Shannon or Rényi entropy on the continuous MMI field to quantify “randomness” of updates.
    
    © SHAKEmaps version 25.3.4
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

    
    # --- Helper methods for file names ---

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

    
    # --- SHAKEsummary methods  ---++

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


    

    # --- Unified Grid Methods ---


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
    



    # --- plotting methods  ---++

    
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



##########################################

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

    




######################################

    



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
    



    ###################3



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



#################################################################

    

    


        
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



        ##############################

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
    

        #########################



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





    ##############3

 
    # --- END of plotting methods  ---++








   # ---Quantify Evolution methods for SHAKEtime ---

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



   # --- Analyze Auxiliary Influences methods for SHAKEtime ---

        
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


    
        # --- Spatial statistics methods Not fully implemented ---
        # --- Pipiline section for adding other functions ---
        # --- Pipiline unstable placeholder for future development ---


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

   # --- Analyze Using Chaos Theory for SHAKEtime ---


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
##########################################################

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


    # =========================
    #
    #
    #
    # UQ core framework + UQ unified dataset builder 
    #
    #
    #
    # =========================

    
    def uq_build_dataset(
        self,
        version_list=None,
        imts=None,
        shakemap_folder=None,
        output_path="./export",
        load_vs30=True,
        vs30_path=None,
        strict=False,
        cache=True,
        overwrite_cache=False,
        verbose=True,
    ):
        """
        Build an uncertainty-aware unified dataset for UQ analyses across versions and IMTs.
    
        This method is intentionally self-contained (independent of sensitivity routines).
        It loads and organizes, per version:
          - grid mean:        *_grid.xml
          - uncertainty:      *_uncertainty.xml
          - stationlist:      *_stationlist.json (includes data actually used; DYFI can be extracted here too)
          - rupture:          *_rupture.json
          - vs30 grid:        optional; imported/attached to dataset (full alignment handled in Patch 2)
    
        Parameters
        ----------
        version_list : list[int|str] or None
            ShakeMap shakemap version identifiers to include. If None, attempts to discover versions.
        imts : list[str] or None
            IMTs to include (e.g., ["MMI","PGA","PGV","PSA03","PSA10"]). If None, include all available IMTs.
        shakemap_folder : str or None
            Base folder containing event/version products. If None, attempts to use self.shakemap_folder.
        output_path : str
            Export root. Creates: output_path/SHAKEtime/<event_id>/uq/{data,posterior,maps,metrics,figures}
        load_vs30 : bool
            If True, tries to load Vs30 from vs30_path (simple formats) and attach to dataset
            (full grid alignment & resampling in Patch 2).
        vs30_path : str or None
            Optional file path for Vs30 grid/data. Supported here (simple):
              - .npz with keys: lon, lat, vs30  OR x,y,vs30
              - .npy with structured array with fields lon,lat,vs30
              - .csv with columns lon,lat,vs30 (header required)
            If None, dataset stores a placeholder for later use.
        strict : bool
            If True, missing required files raise; otherwise warnings are logged and version is partially loaded.
        cache : bool
            If True, stores built dataset at self.uq_data and optionally writes a compact cache file.
        overwrite_cache : bool
            If True, overwrite on-disk cache if present.
        verbose : bool
            If True, log progress.
    
        Returns
        -------
        dict
            Unified dataset: per version/per IMT arrays of lon/lat/mean/sigma (+ vs30 if available),
            and auxiliary station/DYFI/rupture metadata.
        """
        import os
    
        event_id = self._uq_get_event_id()
        sm_folder = shakemap_folder or getattr(self, "shakemap_folder", None) or getattr(self, "shakemap_dir", None)
        if sm_folder is None:
            raise ValueError("uq_build_dataset: shakemap_folder is None and self.shakemap_folder not found.")
    
        # Prepare output directories
        uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
        self._uq_log(f"UQ export dirs ready: {uq_dirs}", level="debug", verbose=verbose)
    
        # Discover versions if not provided
        if version_list is None:
            version_list = self._uq_discover_versions(sm_folder)
            self._uq_log(f"Discovered versions: {version_list}", level="info", verbose=verbose)
        else:
            version_list = list(version_list)
    
        # Build dataset scaffold
        dataset = {
            "event_id": event_id,
            "shakemap_folder": sm_folder,
            "imts_requested": None if imts is None else list(imts),
            "versions": {},
            "vs30": {
                "attached": False,
                "source": vs30_path,
                "data": None,         # (lon,lat,vs30) arrays or structured object
                "note": "Vs30 is loaded/attached here if possible; grid alignment/resampling handled in Patch 2.",
            },
            "meta": {
                "created_by": "SHAKEtime.uq_build_dataset",
                "uq_export_dirs": uq_dirs,
            },
        }
    
        # Attempt to load Vs30 (simple supported formats in Patch 1)
        if load_vs30:
            try:
                vs30_obj = self._uq_try_load_vs30(vs30_path)
                if vs30_obj is not None:
                    dataset["vs30"]["data"] = vs30_obj
                    dataset["vs30"]["attached"] = True
                    self._uq_log("Vs30 loaded (raw). Grid alignment will be handled later.", level="info", verbose=verbose)
                else:
                    self._uq_log("Vs30 not loaded (no vs30_path or unsupported/missing). Placeholder kept.", level="warning", verbose=verbose)
            except Exception as e:
                msg = f"Vs30 load failed: {e}"
                if strict:
                    raise
                self._uq_log(msg, level="warning", verbose=verbose)
    
        # Load per-version products
        for ver in version_list:
            ver_key = str(ver)
            try:
                vpaths = self._uq_find_version_products(sm_folder, ver)
                self._uq_log(f"[v{ver_key}] product paths: {vpaths}", level="debug", verbose=verbose)
    
                # Mean grid
                grid = None
                if vpaths.get("grid_xml"):
                    grid = self._uq_parse_grid_xml(vpaths["grid_xml"])
                else:
                    raise FileNotFoundError(f"Missing grid xml for version {ver_key}")
    
                # Uncertainty grid
                unc = None
                if vpaths.get("uncertainty_xml"):
                    unc = self._uq_parse_uncertainty_xml(vpaths["uncertainty_xml"])
                else:
                    # Not strictly fatal if strict=False; we can still store means
                    if strict:
                        raise FileNotFoundError(f"Missing uncertainty xml for version {ver_key}")
                    self._uq_log(f"[v{ver_key}] uncertainty xml missing; sigma unavailable for this version.", level="warning", verbose=verbose)
    
                # Stationlist (stations + DYFI)
                stations = None
                dyfi = None
                if vpaths.get("stationlist_json"):
                    stations, dyfi = self._uq_parse_stationlist_json(vpaths["stationlist_json"])
                else:
                    if strict:
                        raise FileNotFoundError(f"Missing stationlist json for version {ver_key}")
                    self._uq_log(f"[v{ver_key}] stationlist json missing; station-based UQ modes may be limited.", level="warning", verbose=verbose)
    
                # Rupture (optional for patch 1; stored raw)
                rupture = None
                if vpaths.get("rupture_json"):
                    rupture = self._uq_read_json(vpaths["rupture_json"])
                else:
                    self._uq_log(f"[v{ver_key}] rupture json missing (ok).", level="debug", verbose=verbose)
    
                # Decide IMTs to keep
                imts_available = sorted(grid["fields"].keys())
                if imts is None:
                    imts_keep = imts_available
                else:
                    imts_keep = [x for x in imts if x in imts_available]
                    missing = [x for x in imts if x not in imts_available]
                    if missing:
                        self._uq_log(f"[v{ver_key}] requested IMTs missing in grid: {missing}", level="warning", verbose=verbose)
    
                # Assemble per-IMT arrays (lon/lat shared)
                lon = grid["lon"]
                lat = grid["lat"]
    
                ver_block = {
                    "version": ver,
                    "paths": vpaths,
                    "grid_meta": grid.get("meta", {}),
                    "unc_meta": None if unc is None else unc.get("meta", {}),
                    "stations": stations,
                    "dyfi": dyfi,
                    "rupture": rupture,
                    "imts": {},
                    "notes": [],
                }
    
                for imt in imts_keep:
                    mean_arr = grid["fields"][imt]
                    sig_arr = None
                    if unc is not None:
                        # uncertainty xml may provide same IMT names; if not, try some light normalization
                        if imt in unc["fields"]:
                            sig_arr = unc["fields"][imt]
                        else:
                            # Try a few safe normalizations for PSA variants, case, etc.
                            alt = self._uq_normalize_imt(imt)
                            sig_arr = unc["fields"].get(alt, None)
    
                    ver_block["imts"][imt] = {
                        "imt": imt,
                        "lon": lon,
                        "lat": lat,
                        "mean": mean_arr,
                        "sigma": sig_arr,
                        "vs30": None,  # attached/aligned in Patch 2
                    }
    
                dataset["versions"][ver_key] = ver_block
                self._uq_log(f"[v{ver_key}] loaded IMTs: {list(ver_block['imts'].keys())}", level="info", verbose=verbose)
    
            except Exception as e:
                msg = f"[v{ver_key}] load failed: {e}"
                if strict:
                    raise
                self._uq_log(msg, level="error", verbose=verbose)
                dataset["versions"][ver_key] = {
                    "version": ver,
                    "error": str(e),
                    "paths": {},
                    "imts": {},
                }
    
        # Cache in object
        if cache:
            self.uq_data = dataset
    
            # Optional on-disk compact cache (npz with pickled dict using numpy)
            try:
                cache_path = os.path.join(uq_dirs["data"], f"uq_dataset_cache_{event_id}.npz")
                if (not os.path.exists(cache_path)) or overwrite_cache:
                    self._uq_write_npz_dict(cache_path, dataset)
                    self._uq_log(f"UQ dataset cache written: {cache_path}", level="info", verbose=verbose)
                else:
                    self._uq_log(f"UQ dataset cache exists (skip): {cache_path}", level="debug", verbose=verbose)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"Failed to write UQ cache: {e}", level="warning", verbose=verbose)
    
        return dataset


    # -------------------------
    # Private helpers (PATCH 1)
    # -------------------------
    
    def _uq_get_event_id(self):
        """Return a robust event_id string from common SHAKEtime attribute names."""
        for k in ("event_id", "eventid", "eventID", "id", "event"):
            if hasattr(self, k):
                v = getattr(self, k)
                if v is None:
                    continue
                # event may be object; try to extract id
                if isinstance(v, str):
                    return v
                if isinstance(v, dict) and "id" in v:
                    return str(v["id"])
                if hasattr(v, "id"):
                    return str(getattr(v, "id"))
                try:
                    return str(v)
                except Exception:
                    pass
        return "unknown_event"
    
    
    def _uq_prepare_export_dirs(self, output_path, event_id):
        """Create standardized UQ export directories."""
        import os
        base = os.path.join(output_path, "SHAKEtime", str(event_id), "uq")
        dirs = {
            "base": base,
            "data": os.path.join(base, "data"),
            "posterior": os.path.join(base, "posterior"),
            "maps": os.path.join(base, "maps"),
            "metrics": os.path.join(base, "metrics"),
            "figures": os.path.join(base, "figures"),
        }
        for _, p in dirs.items():
            os.makedirs(p, exist_ok=True)
        # expose for later patches
        self._uq_export_dirs = dirs
        return dirs
    
    
    def _uq_log(self, msg, level="info", verbose=True):
        """
        Lightweight logger that respects common patterns:
          - self.logger.<level>(msg)
          - self.log(msg, level=...)
          - fallback: print
        """
        if not verbose and level in ("debug",):
            return
        logger = getattr(self, "logger", None)
        if logger is not None:
            fn = getattr(logger, level, None)
            if callable(fn):
                fn(msg)
                return
            # fallback to info
            fn = getattr(logger, "info", None)
            if callable(fn):
                fn(msg)
                return
        # alternative project-style logger
        log_fn = getattr(self, "log", None)
        if callable(log_fn):
            try:
                log_fn(msg, level=level)
                return
            except Exception:
                pass
        # final fallback
        print(f"[UQ:{level}] {msg}")
    
    
    def _uq_discover_versions(self, shakemap_folder):
        """
        Discover shakemap versions under shakemap_folder with a conservative strategy:
        - Look for subfolders containing 'shakemap' version patterns (v##) or numeric folders.
        - If not found, return ['preferred'] as a single version key to allow downstream resolution.
        """
        import os
        versions = []
        try:
            for name in os.listdir(shakemap_folder):
                p = os.path.join(shakemap_folder, name)
                if not os.path.isdir(p):
                    continue
                # common patterns: "shakemap_v12", "v12", "12"
                lname = name.lower()
                if "v" in lname:
                    # extract digits after 'v'
                    idx = lname.find("v")
                    dig = "".join([c for c in lname[idx + 1 :] if c.isdigit()])
                    if dig:
                        versions.append(int(dig))
                        continue
                if name.isdigit():
                    versions.append(int(name))
            versions = sorted(list({v for v in versions}))
        except Exception:
            versions = []
    
        if not versions:
            # Fallback: allow user to pass version_list later; keep a placeholder
            return ["preferred"]
        return versions
    
    
    def _uq_find_version_products(self, shakemap_folder, version):
        """
        Locate product files for a given version. Tries several likely layouts without assuming your exact tree.
    
        Returns dict with keys:
          - grid_xml
          - uncertainty_xml
          - stationlist_json
          - rupture_json
          - base_dir
        """
        import os
        v = str(version)
    
        # Candidate base dirs
        candidates = []
        # direct version folder
        candidates.append(os.path.join(shakemap_folder, v))
        # v-prefixed folder
        candidates.append(os.path.join(shakemap_folder, f"v{v}"))
        # shakemap_v## folder
        candidates.append(os.path.join(shakemap_folder, f"shakemap_v{v}"))
        # preferred / current
        candidates.append(os.path.join(shakemap_folder, "preferred"))
        candidates.append(os.path.join(shakemap_folder, "current"))
    
        base_dir = None
        for c in candidates:
            if os.path.isdir(c):
                base_dir = c
                break
    
        # If we can't find a base_dir, still attempt recursive search from root
        search_root = base_dir or shakemap_folder
    
        def find_first(pattern_suffix):
            # search locally first (base_dir), then one level deep, then recursively as last resort
            if base_dir:
                for fn in os.listdir(base_dir):
                    if fn.endswith(pattern_suffix):
                        return os.path.join(base_dir, fn)
                # one level deep
                for sub in os.listdir(base_dir):
                    sp = os.path.join(base_dir, sub)
                    if os.path.isdir(sp):
                        for fn in os.listdir(sp):
                            if fn.endswith(pattern_suffix):
                                return os.path.join(sp, fn)
            # recursive fallback
            for root, _, files in os.walk(search_root):
                for fn in files:
                    if fn.endswith(pattern_suffix):
                        return os.path.join(root, fn)
            return None
    
        grid_xml = find_first("_grid.xml")
        uncertainty_xml = find_first("_uncertainty.xml")
        stationlist_json = find_first("_stationlist.json")
        rupture_json = find_first("_rupture.json")
    
        return {
            "base_dir": base_dir,
            "grid_xml": grid_xml,
            "uncertainty_xml": uncertainty_xml,
            "stationlist_json": stationlist_json,
            "rupture_json": rupture_json,
        }
    
    
    def _uq_parse_grid_xml(self, grid_xml_path):
        """
        Parse ShakeMap *_grid.xml (mean grid).
        Returns dict: lon, lat, fields{name->array}, meta.
        """
        import numpy as np
        import xml.etree.ElementTree as ET
    
        tree = ET.parse(grid_xml_path)
        root = tree.getroot()
    
        # Identify grid_field definitions
        fields = {}
        field_order = []  # list of (index:int, name:str)
        meta = {}
    
        # Root attributes often include grid specs
        try:
            meta.update(dict(root.attrib))
        except Exception:
            pass
    
        for gf in root.iter():
            if gf.tag.endswith("grid_field"):
                name = gf.attrib.get("name", "").strip()
                idx = gf.attrib.get("index", None)
                if name and idx is not None:
                    try:
                        field_order.append((int(idx), name))
                    except Exception:
                        field_order.append((None, name))
    
        # Sort by index if possible
        field_order_sorted = sorted(field_order, key=lambda x: (1e9 if x[0] is None else x[0]))
    
        # Read grid_data text block
        grid_data_node = None
        for node in root.iter():
            if node.tag.endswith("grid_data"):
                grid_data_node = node
                break
        if grid_data_node is None or (grid_data_node.text is None):
            raise ValueError(f"grid_data not found or empty in: {grid_xml_path}")
    
        # Load numeric matrix
        text = grid_data_node.text.strip().splitlines()
        rows = []
        for ln in text:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            rows.append([float(x) for x in parts])
    
        mat = np.asarray(rows, dtype=float)
        if mat.ndim != 2 or mat.shape[1] < 2:
            raise ValueError(f"Unexpected grid_data matrix shape {mat.shape} in: {grid_xml_path}")
    
        lon = mat[:, 0]
        lat = mat[:, 1]
    
        # Remaining columns map to field_order indices (usually starts at 1 for lon, 2 for lat, then IMTs)
        # Many ShakeMap grids index lon=1, lat=2, IMTs begin at 3
        # We therefore assign any field with index >=3 to mat[:, index-1]
        for idx, name in field_order_sorted:
            if idx is None:
                continue
            col = idx - 1
            if col < 0 or col >= mat.shape[1]:
                continue
            # Skip lon/lat if present in definition
            if name.lower() in ("lon", "longitude", "x", "lat", "latitude", "y"):
                continue
            fields[name] = mat[:, col]
    
        # If we failed to map any fields but there are extra columns, create fallback names
        if not fields and mat.shape[1] > 2:
            for j in range(2, mat.shape[1]):
                fields[f"FIELD{j+1}"] = mat[:, j]
    
        return {"lon": lon, "lat": lat, "fields": fields, "meta": meta}
    
    
    def _uq_parse_uncertainty_xml(self, unc_xml_path):
        """
        Parse ShakeMap *_uncertainty.xml (sigma grid).
        Returns dict: lon, lat, fields{name->array}, meta.
        """
        import numpy as np
        import xml.etree.ElementTree as ET
    
        tree = ET.parse(unc_xml_path)
        root = tree.getroot()
    
        fields = {}
        field_order = []
        meta = {}
        try:
            meta.update(dict(root.attrib))
        except Exception:
            pass
    
        for gf in root.iter():
            if gf.tag.endswith("grid_field"):
                name = gf.attrib.get("name", "").strip()
                idx = gf.attrib.get("index", None)
                if name and idx is not None:
                    try:
                        field_order.append((int(idx), name))
                    except Exception:
                        field_order.append((None, name))
    
        field_order_sorted = sorted(field_order, key=lambda x: (1e9 if x[0] is None else x[0]))
    
        grid_data_node = None
        for node in root.iter():
            if node.tag.endswith("grid_data"):
                grid_data_node = node
                break
        if grid_data_node is None or (grid_data_node.text is None):
            raise ValueError(f"grid_data not found or empty in: {unc_xml_path}")
    
        text = grid_data_node.text.strip().splitlines()
        rows = []
        for ln in text:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            rows.append([float(x) for x in parts])
    
        mat = np.asarray(rows, dtype=float)
        if mat.ndim != 2 or mat.shape[1] < 2:
            raise ValueError(f"Unexpected uncertainty grid_data matrix shape {mat.shape} in: {unc_xml_path}")
    
        lon = mat[:, 0]
        lat = mat[:, 1]
    
        for idx, name in field_order_sorted:
            if idx is None:
                continue
            col = idx - 1
            if col < 0 or col >= mat.shape[1]:
                continue
            if name.lower() in ("lon", "longitude", "x", "lat", "latitude", "y"):
                continue
            fields[name] = mat[:, col]
    
        if not fields and mat.shape[1] > 2:
            for j in range(2, mat.shape[1]):
                fields[f"SIGMA{j+1}"] = mat[:, j]
    
        return {"lon": lon, "lat": lat, "fields": fields, "meta": meta}
    
    
    def _uq_parse_stationlist_json(self, stationlist_path):
        """
        Parse *_stationlist.json with flexible extraction:
          - Returns (stations, dyfi) lists of dicts with at minimum: lon, lat, value (IMT-dependent if present), source/type.
          - Keeps raw entries in each dict under 'raw' for traceability.
    
        Notes
        -----
        ShakeMap stationlist schemas vary. We implement best-effort extraction:
          - If GeoJSON-like: features[].geometry.coordinates + properties
          - If dict with 'stations': iterate stations
          - DYFI often appears as "DYFI" network or a distinct feature type; we separate if recognized.
        """
        data = self._uq_read_json(stationlist_path)
    
        def _coord_from(obj):
            # returns (lon,lat) or (None,None)
            if isinstance(obj, dict):
                # GeoJSON geometry
                geom = obj.get("geometry", None)
                if isinstance(geom, dict) and "coordinates" in geom:
                    c = geom["coordinates"]
                    if isinstance(c, (list, tuple)) and len(c) >= 2:
                        return float(c[0]), float(c[1])
                # direct fields
                for a, b in (("lon", "lat"), ("longitude", "latitude"), ("x", "y")):
                    if a in obj and b in obj:
                        return float(obj[a]), float(obj[b])
            return None, None
    
        stations = []
        dyfi = []
    
        # GeoJSON-like
        if isinstance(data, dict) and "features" in data and isinstance(data["features"], list):
            for feat in data["features"]:
                props = feat.get("properties", {}) if isinstance(feat, dict) else {}
                lon, lat = _coord_from(feat)
                if lon is None or lat is None:
                    continue
    
                net = str(props.get("network", props.get("net", "")) or "").upper()
                stype = str(props.get("station_type", props.get("type", "")) or "").upper()
                # Attempt to get an intensity-like value
                val = props.get("intensity", None)
                if val is None:
                    val = props.get("mmi", props.get("MMI", None))
    
                rec = {
                    "lon": lon,
                    "lat": lat,
                    "value": None if val is None else float(val),
                    "network": net,
                    "station_type": stype,
                    "properties": props,
                    "raw": feat,
                }
                # Heuristic DYFI split
                if "DYFI" in net or "DYFI" in stype or ("DYFI" in str(props.get("source", "")).upper()):
                    dyfi.append(rec)
                else:
                    stations.append(rec)
    
            return stations, dyfi
    
        # Alternative schema: 'stations' list
        if isinstance(data, dict) and "stations" in data and isinstance(data["stations"], list):
            for st in data["stations"]:
                if not isinstance(st, dict):
                    continue
                lon, lat = _coord_from(st)
                if lon is None or lat is None:
                    continue
                net = str(st.get("network", st.get("net", "")) or "").upper()
                stype = str(st.get("station_type", st.get("type", "")) or "").upper()
                val = st.get("intensity", st.get("mmi", st.get("MMI", None)))
                rec = {
                    "lon": lon,
                    "lat": lat,
                    "value": None if val is None else float(val),
                    "network": net,
                    "station_type": stype,
                    "properties": st,
                    "raw": st,
                }
                if "DYFI" in net or "DYFI" in stype or ("DYFI" in str(st.get("source", "")).upper()):
                    dyfi.append(rec)
                else:
                    stations.append(rec)
    
            return stations, dyfi
    
        # If schema unknown, return empty lists but keep raw for debugging
        return stations, dyfi
    
    
    def _uq_read_json(self, path):
        """Read JSON file safely."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    
    def _uq_normalize_imt(self, imt):
        """
        Normalize IMT naming to improve cross-file matching.
        Keeps PSA variants (e.g., PSA03/PSA3/SA(0.3)) as best-effort.
        """
        s = str(imt).strip().upper()
        s = s.replace(" ", "")
        s = s.replace("SA(", "PSA").replace(")", "")
        s = s.replace("SA", "PSA") if s.startswith("SA") and not s.startswith("PSA") else s
        # common: PSA0.3 -> PSA03
        if s.startswith("PSA") and "." in s:
            try:
                per = float(s.replace("PSA", ""))
                # represent as 2 digits of seconds*10 if < 1, else seconds*10 too (e.g., 1.0 -> 10)
                val = int(round(per * 10))
                s = f"PSA{val:02d}"
            except Exception:
                pass
        return s
    
    
    def _uq_try_load_vs30(self, vs30_path):
        """
        Try to load Vs30 from simple formats for later alignment.
        Returns object dict with arrays, or None.
        """
        import os
        import numpy as np
    
        if vs30_path is None:
            return None
        if not os.path.exists(vs30_path):
            return None
    
        ext = os.path.splitext(vs30_path)[1].lower()
        if ext == ".npz":
            z = np.load(vs30_path, allow_pickle=True)
            keys = set(z.files)
            # common keys: lon, lat, vs30 OR x,y,vs30
            if {"lon", "lat", "vs30"}.issubset(keys):
                return {"lon": z["lon"], "lat": z["lat"], "vs30": z["vs30"], "source": vs30_path}
            if {"x", "y", "vs30"}.issubset(keys):
                return {"lon": z["x"], "lat": z["y"], "vs30": z["vs30"], "source": vs30_path}
            # if unknown, store raw dict-like
            return {"raw_npz": {k: z[k] for k in z.files}, "source": vs30_path}
    
        if ext == ".npy":
            arr = np.load(vs30_path, allow_pickle=True)
            # structured array with fields
            if hasattr(arr, "dtype") and arr.dtype.names:
                names = set([n.lower() for n in arr.dtype.names])
                if {"lon", "lat", "vs30"}.issubset(names):
                    return {
                        "lon": arr["lon"],
                        "lat": arr["lat"],
                        "vs30": arr["vs30"],
                        "source": vs30_path,
                    }
            # otherwise assume Nx3
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return {"lon": arr[:, 0], "lat": arr[:, 1], "vs30": arr[:, 2], "source": vs30_path}
            return {"raw_npy": arr, "source": vs30_path}
    
        if ext == ".csv":
            # expects header with lon,lat,vs30
            import csv
            lon, lat, vs = [], [], []
            with open(vs30_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row is None:
                        continue
                    lon.append(float(row["lon"]))
                    lat.append(float(row["lat"]))
                    vs.append(float(row["vs30"]))
            if lon:
                return {"lon": np.asarray(lon), "lat": np.asarray(lat), "vs30": np.asarray(vs), "source": vs30_path}
            return None
    
        # Unsupported here (GeoTIFF/NetCDF/etc. handled later if needed)
        return None
    
    
    def _uq_write_npz_dict(self, npz_path, obj):
        """Write a python dict into npz as a single pickled object (lightweight cache)."""
        import numpy as np
        np.savez_compressed(npz_path, uq_dataset=np.array([obj], dtype=object))



    # =========================
    # PATCH 2 — Vs30 loader + grid alignment utilities (ADD INSIDE SHAKEtime CLASS)
    # =========================
    
    def uq_attach_vs30(
        self,
        dataset=None,
        vs30_path=None,
        method="nearest",
        max_distance_km=None,
        output_path=None,
        strict=False,
        verbose=True,
    ):
        """
        Attach (and align) Vs30 values to every version/IMT grid point in an existing UQ dataset.
    
        This patch adds the practical grid alignment step that Patch 1 deferred:
          - Loads Vs30 if not already loaded
          - Maps Vs30 onto each shakemap grid point and writes into dataset["versions"][v]["imts"][imt]["vs30"]
          - Optionally aligns sigma grid to mean grid if lon/lat differ between grid and uncertainty files.
    
        Parameters
        ----------
        dataset : dict or None
            UQ dataset from uq_build_dataset. If None, uses self.uq_data.
        vs30_path : str or None
            Optional path to Vs30 data. If None, uses dataset["vs30"]["source"].
        method : str
            Mapping method: "nearest" (default) or "bilinear" (if Vs30 is gridded and dependencies allow).
        max_distance_km : float or None
            If set, any gridpoint farther than this from nearest Vs30 sample is set to NaN.
        output_path : str or None
            If provided, exports a simple summary CSV of Vs30 attachment stats into uq/data.
        strict : bool
            If True, raise on failures; otherwise warn and continue.
        verbose : bool
            Verbose logging.
    
        Returns
        -------
        dict
            Updated dataset (also stored back to self.uq_data if dataset was None).
        """
        import os
        import numpy as np
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_attach_vs30: dataset is None and self.uq_data not found. Run uq_build_dataset first.")
    
        # Ensure export dirs exist
        event_id = ds.get("event_id", self._uq_get_event_id())
        out_root = output_path if output_path is not None else ds.get("meta", {}).get("uq_export_dirs", {}).get("base", None)
        if out_root is None:
            # fall back to self._uq_export_dirs if available
            uq_dirs = getattr(self, "_uq_export_dirs", None)
            if uq_dirs is None:
                uq_dirs = self._uq_prepare_export_dirs(output_path="./export", event_id=event_id)
            else:
                uq_dirs = uq_dirs
        else:
            uq_dirs = self._uq_prepare_export_dirs(output_path=os.path.dirname(os.path.dirname(out_root)), event_id=event_id)
    
        # Load Vs30 if missing
        if not ds.get("vs30", {}).get("attached", False) or ds.get("vs30", {}).get("data", None) is None:
            src = vs30_path or ds.get("vs30", {}).get("source", None)
            try:
                vs30_obj = self._uq_try_load_vs30(src)
                if vs30_obj is None:
                    msg = "uq_attach_vs30: Vs30 not available (no path or unsupported)."
                    if strict:
                        raise FileNotFoundError(msg)
                    self._uq_log(msg, level="warning", verbose=verbose)
                    return ds
                ds["vs30"]["data"] = vs30_obj
                ds["vs30"]["source"] = src
                ds["vs30"]["attached"] = True
                self._uq_log(f"Vs30 loaded for attachment from: {src}", level="info", verbose=verbose)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"uq_attach_vs30: failed to load Vs30: {e}", level="warning", verbose=verbose)
                return ds
    
        vs30_obj = ds["vs30"]["data"]
    
        # Pre-build interpolator/mapping function
        try:
            mapper = self._uq_build_vs30_mapper(vs30_obj, method=method, verbose=verbose)
        except Exception as e:
            if strict:
                raise
            self._uq_log(f"uq_attach_vs30: could not build Vs30 mapper: {e}", level="warning", verbose=verbose)
            return ds
    
        # Attach Vs30 to each version/IMT
        total_pts = 0
        nan_pts = 0
        for vkey, vblock in ds.get("versions", {}).items():
            if not isinstance(vblock, dict) or "imts" not in vblock:
                continue
    
            # Optional: align sigma arrays to mean lon/lat if needed (safe + lightweight)
            try:
                self._uq_align_sigma_to_mean_inplace(vblock, strict=False, verbose=verbose)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"[v{vkey}] sigma alignment skipped/failed: {e}", level="debug", verbose=verbose)
    
            for imt, iblock in vblock.get("imts", {}).items():
                lon = np.asarray(iblock.get("lon", []), dtype=float)
                lat = np.asarray(iblock.get("lat", []), dtype=float)
                if lon.size == 0 or lat.size == 0:
                    continue
    
                vs = mapper(lon, lat)
                if max_distance_km is not None:
                    # enforce max distance constraint using nearest-distance if available
                    dkm = self._uq_nearest_distance_km(vs30_obj, lon, lat, verbose=verbose)
                    vs = np.where(dkm <= float(max_distance_km), vs, np.nan)
    
                iblock["vs30"] = vs
    
                total_pts += vs.size
                nan_pts += int(np.isnan(vs).sum())
    
        self._uq_log(
            f"Vs30 attached. Total grid points: {total_pts:,}; NaNs: {nan_pts:,} ({(nan_pts/total_pts*100 if total_pts else 0):.2f}%)",
            level="info",
            verbose=verbose,
        )
    
        # Optional stats export
        try:
            if output_path is not None:
                csv_path = os.path.join(uq_dirs["data"], f"uq_vs30_attach_stats_{event_id}.csv")
                self._uq_write_simple_kv_csv(
                    csv_path,
                    {
                        "event_id": event_id,
                        "total_points": total_pts,
                        "nan_points": nan_pts,
                        "nan_percent": (nan_pts / total_pts * 100.0) if total_pts else 0.0,
                        "method": method,
                        "max_distance_km": "" if max_distance_km is None else float(max_distance_km),
                        "vs30_source": ds.get("vs30", {}).get("source", ""),
                    },
                )
                self._uq_log(f"Vs30 attachment stats exported: {csv_path}", level="debug", verbose=verbose)
        except Exception as e:
            if strict:
                raise
            self._uq_log(f"Vs30 stats export failed (non-fatal): {e}", level="debug", verbose=verbose)
    
        # Store back if using self.uq_data
        if dataset is None:
            self.uq_data = ds
        return ds
    
    
    def _uq_build_vs30_mapper(self, vs30_obj, method="nearest", verbose=True):
        """
        Build a callable f(lon,lat)->vs30 array from a loaded Vs30 object.
    
        Supports:
          A) scattered samples: vs30_obj has 1D arrays lon,lat,vs30 same length
          B) gridded: vs30_obj has lon2d/lat2d/vs30 2D or x1d/y1d with vs30 2D (best-effort)
    
        method:
          - "nearest": always available (uses scipy KDTree if present, else numpy fallback)
          - "bilinear": for gridded data if scipy RegularGridInterpolator available, else falls back to nearest
        """
        import numpy as np
    
        # ---- Scattered points ----
        if all(k in vs30_obj for k in ("lon", "lat", "vs30")):
            lon = np.asarray(vs30_obj["lon"], dtype=float)
            lat = np.asarray(vs30_obj["lat"], dtype=float)
            vs = np.asarray(vs30_obj["vs30"], dtype=float)
    
            if lon.ndim == 1 and lat.ndim == 1 and vs.ndim == 1 and (lon.size == lat.size == vs.size):
                # Build KDTree if available; else brute nearest
                tree = self._uq_try_build_kdtree(lon, lat)
                if tree is not None:
                    def mapper(xlon, xlat):
                        xlon = np.asarray(xlon, dtype=float)
                        xlat = np.asarray(xlat, dtype=float)
                        d, idx = tree.query(np.c_[xlon, xlat], k=1)
                        return vs[idx]
                    return mapper
                else:
                    self._uq_log("Vs30 mapper using numpy brute nearest (scattered).", level="debug", verbose=verbose)
                    def mapper(xlon, xlat):
                        xlon = np.asarray(xlon, dtype=float)
                        xlat = np.asarray(xlat, dtype=float)
                        return self._uq_brute_nearest_values(lon, lat, vs, xlon, xlat)
                    return mapper
    
            # allow lon/lat as 2D grids
            if lon.ndim == 2 and lat.ndim == 2 and vs.ndim == 2 and lon.shape == lat.shape == vs.shape:
                return self._uq_build_gridded_mapper(lon, lat, vs, method=method, verbose=verbose)
    
        # ---- Alternative gridded keys (best effort) ----
        # If someone saved as x/y grids
        for xk, yk, vk in (("x", "y", "vs30"), ("lon2d", "lat2d", "vs30")):
            if all(k in vs30_obj for k in (xk, yk, vk)):
                X = np.asarray(vs30_obj[xk], dtype=float)
                Y = np.asarray(vs30_obj[yk], dtype=float)
                V = np.asarray(vs30_obj[vk], dtype=float)
                if X.ndim == 2 and Y.ndim == 2 and V.ndim == 2 and X.shape == Y.shape == V.shape:
                    return self._uq_build_gridded_mapper(X, Y, V, method=method, verbose=verbose)
    
        raise ValueError("Unsupported Vs30 object structure for mapper building.")
    
    
    def _uq_build_gridded_mapper(self, lon2d, lat2d, vs30_2d, method="nearest", verbose=True):
        """
        Build mapper for gridded lon/lat/vs30 arrays.
    
        - If method="bilinear" and scipy RegularGridInterpolator is available AND grid is rectilinear,
          uses bilinear interpolation on a rectilinear grid.
        - Otherwise uses nearest-neighbor on the flattened grid (KDTree if available).
        """
        import numpy as np
    
        lon2d = np.asarray(lon2d, dtype=float)
        lat2d = np.asarray(lat2d, dtype=float)
        v2d = np.asarray(vs30_2d, dtype=float)
    
        # Try rectilinear inference: lon varies primarily along axis 1, lat along axis 0
        rectilinear = False
        try:
            lon1 = lon2d[0, :]
            lat1 = lat2d[:, 0]
            # check consistency
            if np.allclose(lon2d, np.tile(lon1, (lat1.size, 1)), rtol=0, atol=1e-10) and \
               np.allclose(lat2d, np.tile(lat1[:, None], (1, lon1.size)), rtol=0, atol=1e-10):
                rectilinear = True
        except Exception:
            rectilinear = False
    
        if method.lower() == "bilinear" and rectilinear:
            interp = self._uq_try_build_regulargrid_interpolator(lon2d[0, :], lat2d[:, 0], v2d)
            if interp is not None:
                self._uq_log("Vs30 mapper using bilinear interpolation on rectilinear grid.", level="debug", verbose=verbose)
                def mapper(xlon, xlat):
                    xlon = np.asarray(xlon, dtype=float)
                    xlat = np.asarray(xlat, dtype=float)
                    pts = np.c_[xlat, xlon]  # (lat,lon) ordering for interpolator
                    out = interp(pts)
                    return np.asarray(out, dtype=float)
                return mapper
    
        # Nearest fallback on flattened grid
        self._uq_log("Vs30 mapper using nearest on flattened grid.", level="debug", verbose=verbose)
        lonf = lon2d.ravel()
        latf = lat2d.ravel()
        vf = v2d.ravel()
    
        tree = self._uq_try_build_kdtree(lonf, latf)
        if tree is not None:
            def mapper(xlon, xlat):
                xlon = np.asarray(xlon, dtype=float)
                xlat = np.asarray(xlat, dtype=float)
                _, idx = tree.query(np.c_[xlon, xlat], k=1)
                return vf[idx]
            return mapper
    
        def mapper(xlon, xlat):
            xlon = np.asarray(xlon, dtype=float)
            xlat = np.asarray(xlat, dtype=float)
            return self._uq_brute_nearest_values(lonf, latf, vf, xlon, xlat)
        return mapper
    
    
    def _uq_try_build_kdtree(self, lon, lat):
        """Try to build a scipy.spatial.cKDTree; return None if scipy unavailable."""
        try:
            import numpy as np
            from scipy.spatial import cKDTree
            pts = np.c_[np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)]
            return cKDTree(pts)
        except Exception:
            return None
    
    
    def _uq_try_build_regulargrid_interpolator(self, lon1d, lat1d, v2d):
        """Try to build scipy RegularGridInterpolator on (lat, lon) grid; return None if unavailable."""
        try:
            import numpy as np
            from scipy.interpolate import RegularGridInterpolator
    
            lon1d = np.asarray(lon1d, dtype=float)
            lat1d = np.asarray(lat1d, dtype=float)
            v2d = np.asarray(v2d, dtype=float)
    
            # Ensure increasing order for interpolator
            lon_inc = np.all(np.diff(lon1d) > 0)
            lat_inc = np.all(np.diff(lat1d) > 0)
    
            if not lon_inc:
                lon1d = lon1d[::-1]
                v2d = v2d[:, ::-1]
            if not lat_inc:
                lat1d = lat1d[::-1]
                v2d = v2d[::-1, :]
    
            # interpolator expects (lat,lon) axes
            return RegularGridInterpolator((lat1d, lon1d), v2d, bounds_error=False, fill_value=np.nan)
        except Exception:
            return None
    
    
    def _uq_brute_nearest_values(self, lon_src, lat_src, val_src, lon_tgt, lat_tgt, chunk=5000):
        """
        Numpy-only nearest-neighbor mapping (brute force, chunked).
        Works for scattered sources; O(N*M) so chunked to avoid memory blowups.
        """
        import numpy as np
    
        lon_src = np.asarray(lon_src, dtype=float).ravel()
        lat_src = np.asarray(lat_src, dtype=float).ravel()
        val_src = np.asarray(val_src, dtype=float).ravel()
    
        lon_tgt = np.asarray(lon_tgt, dtype=float).ravel()
        lat_tgt = np.asarray(lat_tgt, dtype=float).ravel()
    
        out = np.empty(lon_tgt.size, dtype=float)
        for i0 in range(0, lon_tgt.size, int(chunk)):
            i1 = min(lon_tgt.size, i0 + int(chunk))
            dx = lon_src[None, :] - lon_tgt[i0:i1, None]
            dy = lat_src[None, :] - lat_tgt[i0:i1, None]
            d2 = dx * dx + dy * dy
            idx = np.argmin(d2, axis=1)
            out[i0:i1] = val_src[idx]
        return out.reshape(np.asarray(lon_tgt).shape)
    
    
    def _uq_nearest_distance_km(self, vs30_obj, lon_tgt, lat_tgt, verbose=True):
        """
        Compute approximate nearest distance (km) from each target point to Vs30 samples.
        If scipy KDTree available, uses it; else brute (chunked).
        """
        import numpy as np
    
        # Extract source points
        if not all(k in vs30_obj for k in ("lon", "lat")):
            # if gridded, flatten if possible
            if "lon2d" in vs30_obj and "lat2d" in vs30_obj:
                lon_src = np.asarray(vs30_obj["lon2d"], dtype=float).ravel()
                lat_src = np.asarray(vs30_obj["lat2d"], dtype=float).ravel()
            else:
                # cannot compute
                return np.full(np.asarray(lon_tgt).size, np.nan, dtype=float)
    
        else:
            lon_src = np.asarray(vs30_obj["lon"], dtype=float).ravel()
            lat_src = np.asarray(vs30_obj["lat"], dtype=float).ravel()
    
        lon_tgt = np.asarray(lon_tgt, dtype=float).ravel()
        lat_tgt = np.asarray(lat_tgt, dtype=float).ravel()
    
        # KDTree
        tree = self._uq_try_build_kdtree(lon_src, lat_src)
        if tree is not None:
            ddeg, _ = tree.query(np.c_[lon_tgt, lat_tgt], k=1)
        else:
            # brute distances in degrees
            self._uq_log("Nearest distance using numpy brute mode.", level="debug", verbose=verbose)
            ddeg = np.empty(lon_tgt.size, dtype=float)
            chunk = 3000
            for i0 in range(0, lon_tgt.size, chunk):
                i1 = min(lon_tgt.size, i0 + chunk)
                dx = lon_src[None, :] - lon_tgt[i0:i1, None]
                dy = lat_src[None, :] - lat_tgt[i0:i1, None]
                d2 = dx * dx + dy * dy
                ddeg[i0:i1] = np.sqrt(np.min(d2, axis=1))
    
        # Convert degrees to km approximately (scale by latitude)
        # 1 deg lat ~ 111.32 km; 1 deg lon ~ 111.32*cos(lat)
        latm = np.deg2rad(lat_tgt)
        km_per_deg_lon = 111.32 * np.cos(latm)
        km_per_deg_lat = 111.32
    
        # We only have scalar ddeg, not separated; approximate using mean scaling
        km_per_deg = np.sqrt((km_per_deg_lon * km_per_deg_lon + km_per_deg_lat * km_per_deg_lat) / 2.0)
        dkm = ddeg * km_per_deg
        return dkm
    
    
    def _uq_align_sigma_to_mean_inplace(self, vblock, strict=False, verbose=True):
        """
        Ensure that sigma arrays (if present) correspond to the same lon/lat ordering as mean arrays.
    
        In Patch 1 we assumed lon/lat match across *_grid.xml and *_uncertainty.xml.
        In practice they sometimes do, but this method provides a safe correction:
          - If sigma is None: do nothing
          - If sigma length matches mean length: do nothing
          - If lengths differ: attempt to map sigma from its own lon/lat if available in vblock["unc_meta"]
            (often not stored), else skip with warning.
    
        Note: Patch 1 parsers currently store lon/lat for uncertainty internally, but only mean lon/lat
        are persisted in per-IMT blocks. So here we can only fix obvious mismatches (shape mismatches)
        by skipping, unless you later decide to store unc lon/lat in vblock (Patch 3+ can also do it).
        """
        import numpy as np
    
        # This is intentionally conservative: only handle trivial cases without extra unc lon/lat storage.
        for imt, iblock in vblock.get("imts", {}).items():
            mean = iblock.get("mean", None)
            sig = iblock.get("sigma", None)
            if mean is None or sig is None:
                continue
            mean = np.asarray(mean)
            sig = np.asarray(sig)
            if mean.shape == sig.shape:
                continue
            # Non-trivial mismatch: cannot safely align without unc lon/lat. Keep sigma None.
            msg = f"[sigma-align] IMT {imt}: mean shape {mean.shape} != sigma shape {sig.shape}. Setting sigma=None."
            if strict:
                raise ValueError(msg)
            self._uq_log(msg, level="warning", verbose=verbose)
            iblock["sigma"] = None
    
    
    def _uq_write_simple_kv_csv(self, path, kv_dict):
        """Write key-value pairs to a 2-column CSV file."""
        import csv
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "value"])
            for k, v in kv_dict.items():
                w.writerow([k, v])
    

    # =========================
    # PATCH 3 — mode="bayes_update" (fast empirical Bayes update using stations/DYFI)
    # ADD INSIDE SHAKEtime CLASS
    # =========================
    
    def uq_run(
        self,
        mode="bayes_update",
        dataset=None,
        version_list=None,
        imts=None,
        output_path="./export",
        strict=False,
        verbose=True,
        **kwargs,
    ):
        """
        Run a UQ analysis mode on a built UQ dataset.
    
        Supported modes (this patch implements bayes_update):
          - "bayes_update" : fast empirical Bayes / Bayesian update using stations (+ DYFI if present)
          - "monte_carlo"  : implemented in Patch 4
          - "hierarchical" : implemented in Patch 5
    
        Parameters
        ----------
        mode : str
            UQ mode.
        dataset : dict or None
            If None, uses self.uq_data.
        version_list : list or None
            Versions to process (subset). If None, all in dataset.
        imts : list[str] or None
            IMTs to process. If None, all available per version.
        output_path : str
            Export root; standard uq dirs used.
        strict : bool
            Raise on errors if True.
        verbose : bool
            Verbose logging.
        **kwargs
            Passed through to mode function.
    
        Returns
        -------
        dict
            Updated dataset with posterior results added.
        """
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_run: dataset is None and self.uq_data not found. Run uq_build_dataset first.")
    
        mode = str(mode).strip().lower()
        if mode == "bayes_update":
            return self.uq_bayes_update(
                dataset=ds,
                version_list=version_list,
                imts=imts,
                output_path=output_path,
                strict=strict,
                verbose=verbose,
                **kwargs,
            )
        elif mode == "monte_carlo":
            raise NotImplementedError("uq_run: mode='monte_carlo' will be implemented in Patch 4.")
        elif mode == "hierarchical":
            raise NotImplementedError("uq_run: mode='hierarchical' will be implemented in Patch 5.")
        else:
            raise ValueError(f"uq_run: unsupported mode '{mode}'.")
    
    
    def uq_bayes_update(
        self,
        dataset=None,
        version_list=None,
        imts=None,
        use_dyfi=True,
        prefer_instrumental=True,
        obs_sigma_overrides=None,
        kernel_km=60.0,
        min_stations=3,
        min_weight_sum=1e-8,
        inflation_floor=0.0,
        posterior_floor=1e-6,
        add_posterior_to_block=True,
        export_summary=True,
        output_path="./export",
        strict=False,
        verbose=True,
    ):
        """
        Fast Bayesian / empirical Bayes update of grid mean+sigma using station observations.
    
        Practical approach (runs without heavy dependencies):
          1) For each version + IMT:
              - Collect observations (instrumental + optionally DYFI) from stationlist.json (already loaded by Patch 1).
              - Interpolate prior grid mean/sigma to observation locations (nearest-neighbor).
              - Compute residuals r = y - m0_at_obs.
          2) For each grid point:
              - Compute distance-weighted local residual mean (bias) and local residual variance using a Gaussian kernel.
              - Aggregate local observation uncertainty se^2 (from obs_sigma_overrides or defaults) + local residual variance.
              - Apply conjugate Normal-Normal update:
                    k = s0^2 / (s0^2 + se^2)
                    m_post = m0 + k * r_local_mean
                    s_post^2 = (s0^2 * se^2) / (s0^2 + se^2)
            If there is insufficient station support near a grid point, keep prior (m_post=m0, s_post=s0).
    
        Notes
        -----
        - For PGA/PGV/PSA* stations, station schemas vary; we best-effort extract values from station properties.
        - For MMI, we use station['value'] and/or properties['mmi'|'intensity'] (and DYFI if enabled).
        - Units:
            * This implementation assumes station and grid IMT values are already in consistent units.
              (Commonly: MMI in intensity units; PGA/PGV/PSA in natural log or linear depends on your grids.
               If your grids are log units, ensure station values are also log units; Patch 3 does not transform units.)
        - Posterior is stored as mean_post/sigma_post per IMT block; original mean/sigma are preserved.
    
        Parameters
        ----------
        dataset : dict or None
            If None, uses self.uq_data.
        version_list : list or None
            Which versions to process; defaults to all present.
        imts : list[str] or None
            Which IMTs to process; defaults to all per version.
        use_dyfi : bool
            If True, include DYFI observations extracted from stationlist (when present).
        prefer_instrumental : bool
            If True, instrumental stations take precedence when duplicates occur at same location (simple heuristic).
        obs_sigma_overrides : dict or None
            Optional mapping {IMT: obs_sigma}. Used as per-observation standard deviation (measurement / reporting).
            Example: {"MMI": 0.3, "PGA": 0.6}
        kernel_km : float
            Gaussian kernel length scale (km) for local update.
        min_stations : int
            Minimum number of contributing observations (nonzero weights) required to update a grid point.
        min_weight_sum : float
            Minimum sum of weights required to update (guards numerical issues).
        inflation_floor : float
            Optional extra variance added to prior sigma^2 to avoid overconfidence before update.
        posterior_floor : float
            Minimum posterior sigma to avoid exact zeros.
        add_posterior_to_block : bool
            If True, writes into each IMT block: mean_post, sigma_post, update_meta.
        export_summary : bool
            If True, writes a per-version summary CSV into uq/posterior.
        output_path : str
            Export root.
        strict : bool
            Raise on errors if True.
        verbose : bool
            Verbose logging.
    
        Returns
        -------
        dict
            Updated dataset with posterior results under each version/IMT.
        """
        import os
        import numpy as np
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_bayes_update: dataset is None and self.uq_data not found. Run uq_build_dataset first.")
    
        event_id = ds.get("event_id", self._uq_get_event_id())
        uq_dirs = ds.get("meta", {}).get("uq_export_dirs", None)
        if uq_dirs is None:
            uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        # Defaults for observation noise (tunable)
        obs_sig_default = {
            "MMI": 0.35,   # intensity reporting scatter (typical range ~0.3-0.6)
            "PGA": 0.60,   # often log units ~0.5-0.7; if linear units, override!
            "PGV": 0.60,
        }
        # PSA variants default to PGA-like unless overridden
        obs_sigma_overrides = obs_sigma_overrides or {}
        def _obs_sigma_for(imt):
            imt_u = str(imt).upper()
            if imt_u in obs_sigma_overrides:
                return float(obs_sigma_overrides[imt_u])
            if imt_u.startswith("PSA"):
                return float(obs_sigma_overrides.get("PSA", obs_sig_default.get("PGA", 0.60)))
            return float(obs_sig_default.get(imt_u, 0.60))
    
        # Decide versions
        vkeys = list(ds.get("versions", {}).keys())
        if version_list is not None:
            vset = set([str(v) for v in version_list])
            vkeys = [k for k in vkeys if k in vset]
    
        summaries = []
    
        for vkey in vkeys:
            vblock = ds["versions"].get(vkey, {})
            if not isinstance(vblock, dict) or not vblock.get("imts"):
                continue
    
            # IMTs to process
            imt_keys = list(vblock["imts"].keys())
            if imts is not None:
                req = set([str(x) for x in imts])
                imt_keys = [k for k in imt_keys if k in req]
    
            # Collect obs pools
            stations = vblock.get("stations", []) or []
            dyfi = vblock.get("dyfi", []) or []
            if (not stations) and (not (use_dyfi and dyfi)):
                self._uq_log(f"[v{vkey}] No stations/DYFI available; bayes_update skipped.", level="warning", verbose=verbose)
                continue
    
            for imt in imt_keys:
                iblock = vblock["imts"][imt]
                lon_g = np.asarray(iblock.get("lon", []), dtype=float)
                lat_g = np.asarray(iblock.get("lat", []), dtype=float)
                m0 = np.asarray(iblock.get("mean", []), dtype=float)
                s0 = iblock.get("sigma", None)
    
                if lon_g.size == 0 or lat_g.size == 0 or m0.size == 0:
                    continue
    
                if s0 is None:
                    # If no sigma available, we still can create an "effective" sigma to allow updates
                    # but we warn; practical default based on obs sigma.
                    s0 = np.full_like(m0, _obs_sigma_for(imt), dtype=float)
                    self._uq_log(f"[v{vkey}][{imt}] sigma missing -> using default prior sigma={_obs_sigma_for(imt):.3f}", level="warning", verbose=verbose)
                else:
                    s0 = np.asarray(s0, dtype=float)
    
                # Optional inflation floor to avoid premature overconfidence
                if inflation_floor and float(inflation_floor) > 0:
                    s0 = np.sqrt(np.maximum(s0 * s0 + float(inflation_floor), posterior_floor**2))
    
                # Build nearest-neighbor interpolator from grid to obs
                grid_tree = self._uq_try_build_kdtree(lon_g, lat_g)  # returns KDTree or None
                def grid_nn(xlon, xlat, arr):
                    if grid_tree is not None:
                        _, idx = grid_tree.query(np.c_[xlon, xlat], k=1)
                        return arr[idx]
                    return self._uq_brute_nearest_values(lon_g, lat_g, arr, xlon, xlat)
    
                # Gather observations for this IMT
                obs = self._uq_collect_observations_for_imt(
                    imt=imt,
                    stations=stations,
                    dyfi=dyfi if use_dyfi else [],
                    prefer_instrumental=prefer_instrumental,
                    verbose=verbose,
                )
                if len(obs["y"]) < 1:
                    self._uq_log(f"[v{vkey}][{imt}] No usable observations found; skip.", level="warning", verbose=verbose)
                    continue
    
                xlon = obs["lon"]
                xlat = obs["lat"]
                y = obs["y"]
                n_obs = y.size
    
                # Prior at obs
                m0_obs = grid_nn(xlon, xlat, m0)
                s0_obs = grid_nn(xlon, xlat, s0)
    
                # Residuals
                r = y - m0_obs
    
                # Localized update to each grid point
                # Compute weights W_{g,i} = exp(-0.5*(d/kernel)^2)
                # We'll do chunked to control memory if grids are huge.
                kernel_km = float(kernel_km)
                if kernel_km <= 0:
                    kernel_km = 60.0
    
                # Precompute observation sigma and variance
                obs_sig = np.full(n_obs, _obs_sigma_for(imt), dtype=float)
                # Effective observation variance at each location (adds prior-at-obs uncertainty to be conservative)
                # se^2 = obs_sig^2 + max(0, residual_var_local) [added later] + (s0_obs^2 optional)
                # Here we keep s0_obs in the update through k; we do NOT add it to se by default.
    
                # Chunk update
                gN = lon_g.size
                m_post = np.array(m0, copy=True)
                s_post = np.array(s0, copy=True)
    
                chunk = 20000  # tune for speed/memory
                updated_count = 0
    
                for i0 in range(0, gN, chunk):
                    i1 = min(gN, i0 + chunk)
                    glon = lon_g[i0:i1]
                    glat = lat_g[i0:i1]
                    # distances (km): shape (chunk, n_obs)
                    dkm = self._uq_pairwise_distance_km(glon, glat, xlon, xlat)
                    w = np.exp(-0.5 * (dkm / kernel_km) ** 2)
    
                    wsum = np.sum(w, axis=1)
                    # Determine contributing count (weights above tiny epsilon)
                    contrib = np.sum(w > 1e-12, axis=1)
    
                    # Where insufficient data, keep prior
                    ok = (wsum >= min_weight_sum) & (contrib >= int(min_stations))
                    if not np.any(ok):
                        continue
    
                    # Local bias (weighted residual mean)
                    # r_local = sum(w*r)/sum(w)
                    rloc = np.zeros(i1 - i0, dtype=float)
                    rloc[ok] = (w[ok, :] @ r) / wsum[ok]
    
                    # Local residual variance
                    # var_local = sum(w*(r-rloc)^2)/sum(w)
                    # compute efficiently: E[r^2]-E[r]^2
                    Er2 = np.zeros(i1 - i0, dtype=float)
                    Er = np.zeros(i1 - i0, dtype=float)
                    Er[ok] = (w[ok, :] @ r) / wsum[ok]
                    Er2[ok] = (w[ok, :] @ (r * r)) / wsum[ok]
                    var_local = np.maximum(Er2 - Er * Er, 0.0)
    
                    # Effective observation variance at grid point: obs noise + local residual variance
                    se2 = (float(_obs_sigma_for(imt)) ** 2) + var_local
    
                    # Prior variance at grid point
                    s02 = np.maximum(s0[i0:i1] * s0[i0:i1], posterior_floor**2)
    
                    # Bayesian gain
                    k = np.zeros(i1 - i0, dtype=float)
                    k[ok] = s02[ok] / (s02[ok] + se2[ok])
    
                    # Posterior mean/sigma
                    m_post[i0:i1][ok] = m0[i0:i1][ok] + k[ok] * rloc[ok]
                    sp2 = np.zeros(i1 - i0, dtype=float)
                    sp2[ok] = (s02[ok] * se2[ok]) / (s02[ok] + se2[ok])
                    s_post[i0:i1][ok] = np.sqrt(np.maximum(sp2[ok], posterior_floor**2))
    
                    updated_count += int(np.sum(ok))
    
                # Store results
                if add_posterior_to_block:
                    iblock["mean_post"] = m_post
                    iblock["sigma_post"] = s_post
                    iblock["update_meta"] = {
                        "mode": "bayes_update",
                        "kernel_km": kernel_km,
                        "min_stations": int(min_stations),
                        "n_obs_used": int(n_obs),
                        "obs_sigma": float(_obs_sigma_for(imt)),
                        "updated_gridpoints": int(updated_count),
                        "gridpoints_total": int(gN),
                        "use_dyfi": bool(use_dyfi),
                    }
    
                # Version/IMT summary
                summaries.append({
                    "event_id": event_id,
                    "version": vkey,
                    "imt": imt,
                    "n_obs": int(n_obs),
                    "gridpoints": int(gN),
                    "updated_gridpoints": int(updated_count),
                    "kernel_km": kernel_km,
                    "obs_sigma": float(_obs_sigma_for(imt)),
                    "prior_sigma_median": float(np.nanmedian(s0)),
                    "post_sigma_median": float(np.nanmedian(s_post)),
                    "prior_mean_median": float(np.nanmedian(m0)),
                    "post_mean_median": float(np.nanmedian(m_post)),
                })
    
                self._uq_log(
                    f"[v{vkey}][{imt}] bayes_update done: obs={n_obs}, updated={updated_count}/{gN}, "
                    f"median sigma {np.nanmedian(s0):.3f}->{np.nanmedian(s_post):.3f}",
                    level="info",
                    verbose=verbose,
                )
    
        # Export summary CSV
        if export_summary and summaries:
            try:
                csv_path = os.path.join(uq_dirs["posterior"], f"uq_bayes_update_summary_{event_id}.csv")
                self._uq_write_dictlist_csv(csv_path, summaries)
                self._uq_log(f"Bayes update summary exported: {csv_path}", level="info", verbose=verbose)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"Bayes update summary export failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        # Store back if using self.uq_data
        if dataset is None:
            self.uq_data = ds
        return ds
    
    
    # -------------------------
    # Private helpers (PATCH 3)
    # -------------------------
    
    def _uq_collect_observations_for_imt(self, imt, stations, dyfi, prefer_instrumental=True, verbose=True):
        """
        Collect usable observations for a given IMT from stationlist-derived station dicts.
    
        Returns dict with numpy arrays: lon, lat, y, source
        """
        import numpy as np
    
        imt_u = str(imt).upper()
        obs_lon = []
        obs_lat = []
        obs_y = []
        obs_src = []
    
        # Combine stations and optional dyfi
        # Simple de-duplication by rounding lon/lat
        seen = set()
    
        def add_rec(rec, source_label):
            lon = rec.get("lon", None)
            lat = rec.get("lat", None)
            if lon is None or lat is None:
                return
            try:
                lonf = float(lon); latf = float(lat)
            except Exception:
                return
    
            y = self._uq_extract_station_imt_value(rec, imt_u)
            if y is None:
                return
            try:
                y = float(y)
            except Exception:
                return
    
            key = (round(lonf, 4), round(latf, 4), imt_u)
            if key in seen:
                # If prefer instrumental, keep first added as instrumental
                return
            seen.add(key)
    
            obs_lon.append(lonf)
            obs_lat.append(latf)
            obs_y.append(y)
            obs_src.append(source_label)
    
        # Instrumental stations first if preferred
        if prefer_instrumental:
            for s in stations or []:
                add_rec(s, "station")
            for d in dyfi or []:
                add_rec(d, "dyfi")
        else:
            # DYFI first
            for d in dyfi or []:
                add_rec(d, "dyfi")
            for s in stations or []:
                add_rec(s, "station")
    
        if len(obs_y) == 0:
            return {"lon": np.array([]), "lat": np.array([]), "y": np.array([]), "source": np.array([])}
    
        return {
            "lon": np.asarray(obs_lon, dtype=float),
            "lat": np.asarray(obs_lat, dtype=float),
            "y": np.asarray(obs_y, dtype=float),
            "source": np.asarray(obs_src, dtype=str),
        }
    
    
    def _uq_extract_station_imt_value(self, station_rec, imt_u):
        """
        Best-effort extraction of an IMT observation value from a station record produced by Patch 1 parser.
    
        Patch 1 records include:
          - station_rec["value"] (often MMI/intensity)
          - station_rec["properties"] (raw properties dict)
          - station_rec["network"], station_rec["station_type"]
    
        Extraction heuristics:
          - For MMI: use 'value' first, else properties['mmi'|'MMI'|'intensity']
          - For other IMTs: search properties for keys matching IMT (case-insensitive), or nested dicts under 'channels',
            'amplitudes', 'values', etc.
        """
        props = station_rec.get("properties", {}) or {}
    
        # MMI / intensity
        if imt_u == "MMI":
            v = station_rec.get("value", None)
            if v is not None:
                return v
            for k in ("mmi", "MMI", "intensity", "Intensity"):
                if k in props:
                    return props.get(k, None)
            return None
    
        # Try direct property key matches
        # common keys: PGA, PGV, PSA03, PSA10, SA(0.3), etc.
        if imt_u in props:
            return props.get(imt_u)
        if imt_u.lower() in props:
            return props.get(imt_u.lower())
    
        # PSA variants sometimes stored differently
        alt = self._uq_normalize_imt(imt_u)
        if alt in props:
            return props.get(alt)
    
        # Look into nested structures
        # Examples (vary widely):
        #   props['amplitudes'] = {'PGA': ..., 'PGV': ...}
        #   props['values'] = {'PGA': ..., ...}
        #   props['channels'] list of dicts with 'name'/'imt'/'value'
        for container_key in ("amplitudes", "values", "imts", "metrics"):
            if container_key in props and isinstance(props[container_key], dict):
                d = props[container_key]
                if imt_u in d:
                    return d.get(imt_u)
                if alt in d:
                    return d.get(alt)
    
        if "channels" in props and isinstance(props["channels"], list):
            for ch in props["channels"]:
                if not isinstance(ch, dict):
                    continue
                # try keys
                name = str(ch.get("imt", ch.get("name", ch.get("type", ""))) or "").upper()
                if name == imt_u or name == alt:
                    if "value" in ch:
                        return ch.get("value")
                    if "val" in ch:
                        return ch.get("val")
                    # sometimes peak stored
                    for k in ("p", "peak", "amplitude"):
                        if k in ch:
                            return ch.get(k)
    
        return None
    
    
    def _uq_pairwise_distance_km(self, lon_a, lat_a, lon_b, lat_b):
        """
        Compute pairwise distances in km between points A and points B.
    
        Parameters
        ----------
        lon_a, lat_a : 1D arrays length NA
        lon_b, lat_b : 1D arrays length NB
    
        Returns
        -------
        ndarray
            Distances shape (NA, NB) in km.
    
        Notes
        -----
        Uses a fast equirectangular approximation (good for regional distances).
        """
        import numpy as np
    
        lon_a = np.asarray(lon_a, dtype=float).ravel()
        lat_a = np.asarray(lat_a, dtype=float).ravel()
        lon_b = np.asarray(lon_b, dtype=float).ravel()
        lat_b = np.asarray(lat_b, dtype=float).ravel()
    
        # radians
        la = np.deg2rad(lat_a)[:, None]
        lb = np.deg2rad(lat_b)[None, :]
        dlat = lb - la
        dlon = np.deg2rad(lon_b)[None, :] - np.deg2rad(lon_a)[:, None]
        latm = 0.5 * (la + lb)
    
        # equirectangular
        x = dlon * np.cos(latm)
        y = dlat
        R = 6371.0
        return R * np.sqrt(x * x + y * y)
    
    
    def _uq_write_dictlist_csv(self, path, rows):
        """Write a list of dicts to CSV with union of keys as columns."""
        import csv
        if not rows:
            return
        # union keys
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    
    
    # =========================
    # PATCH 4 — mode="monte_carlo" (sampling + propagation, PoE + credible intervals for maps + metrics hooks)
    # ADD INSIDE SHAKEtime CLASS
    # =========================
    
    def uq_monte_carlo(
        self,
        dataset=None,
        version_list=None,
        imts=None,
        n_samples=500,
        use_posterior=True,
        correlation="independent",
        corr_length_km=40.0,
        random_seed=42,
        chunksize=250000,
        store_samples=False,
        compute_poe=True,
        poe_thresholds=None,
        compute_ci=True,
        ci_quantiles=(0.05, 0.5, 0.95),
        compute_metrics=False,
        metrics_thresholds=None,
        output_path="./export",
        export_summary=True,
        strict=False,
        verbose=True,
    ):
        """
        Monte Carlo uncertainty propagation for UQ dataset.
    
        For each version and IMT:
          - Select prior or posterior mean/sigma (posterior if available and use_posterior=True)
          - Sample realizations at each grid point:
                X ~ Normal(mean, sigma)  (independent by default)
            Optional lightweight spatial correlation:
                - "exp": exponential kernel with length corr_length_km via low-rank random features
                         (practical, avoids huge covariance matrices)
                - "gauss": gaussian kernel via random features
            If scipy is unavailable or correlation not requested, defaults to independent.
    
          - Products:
             * PoE maps for thresholds (if compute_poe=True)
             * Credible interval maps (if compute_ci=True)
             * (Optional) metrics distributions (implemented as hook; full metrics in Patch 7)
    
        Outputs are written into:
          output_path/SHAKEtime/<event_id>/uq/{posterior,maps,metrics}
    
        Parameters
        ----------
        dataset : dict or None
            If None, uses self.uq_data.
        version_list : list or None
            Versions to process.
        imts : list[str] or None
            IMTs to process.
        n_samples : int
            Number of MC samples per IMT/version.
        use_posterior : bool
            If True and mean_post/sigma_post exist, use them; else fallback to mean/sigma.
        correlation : str
            "independent" (default), "exp", or "gauss".
        corr_length_km : float
            Correlation length for correlated sampling (km).
        random_seed : int
            Random seed for reproducibility.
        chunksize : int
            Chunk size for sampling computations to control memory.
        store_samples : bool
            If True, stores samples (compressed) to disk; otherwise stores only summary products.
            WARNING: storing full samples can be large.
        compute_poe : bool
            Compute probability-of-exceedance maps.
        poe_thresholds : dict or None
            Dict {IMT: [thresholds...]} or global list applied to all IMTs.
            If None, uses defaults for common IMTs (MMI, PGA, PGV, PSA*).
        compute_ci : bool
            Compute credible interval maps from samples.
        ci_quantiles : tuple
            Quantiles to compute (e.g., (0.05,0.5,0.95)).
        compute_metrics : bool
            If True, compute simple global metrics distributions (area>=k) using metrics_thresholds.
            (Full uncertainty bands + exposures handled in Patch 7.)
        metrics_thresholds : dict or None
            Dict {IMT: [thresholds...]} or global list for area>=k metric.
        output_path : str
            Export root.
        export_summary : bool
            Export a CSV summary per run.
        strict : bool
            Raise on errors if True.
        verbose : bool
            Verbose logging.
    
        Returns
        -------
        dict
            Updated dataset with MC products (and optionally sample references).
        """
        import os
        import numpy as np
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_monte_carlo: dataset is None and self.uq_data not found. Run uq_build_dataset first.")
    
        event_id = ds.get("event_id", self._uq_get_event_id())
        uq_dirs = ds.get("meta", {}).get("uq_export_dirs", None)
        if uq_dirs is None:
            uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        # defaults for thresholds
        poe_thresholds = poe_thresholds if poe_thresholds is not None else self._uq_default_thresholds()
        metrics_thresholds = metrics_thresholds if metrics_thresholds is not None else self._uq_default_thresholds()
    
        rng = np.random.default_rng(int(random_seed))
    
        # versions
        vkeys = list(ds.get("versions", {}).keys())
        if version_list is not None:
            vset = set([str(v) for v in version_list])
            vkeys = [k for k in vkeys if k in vset]
    
        summaries = []
        for vkey in vkeys:
            vblock = ds["versions"].get(vkey, {})
            if not isinstance(vblock, dict) or not vblock.get("imts"):
                continue
    
            # imts
            imt_keys = list(vblock["imts"].keys())
            if imts is not None:
                req = set([str(x) for x in imts])
                imt_keys = [k for k in imt_keys if k in req]
    
            for imt in imt_keys:
                try:
                    iblock = vblock["imts"][imt]
                    lon = np.asarray(iblock.get("lon", []), dtype=float)
                    lat = np.asarray(iblock.get("lat", []), dtype=float)
    
                    if use_posterior and ("mean_post" in iblock) and ("sigma_post" in iblock) and (iblock["mean_post"] is not None):
                        mean = np.asarray(iblock["mean_post"], dtype=float)
                        sig = np.asarray(iblock["sigma_post"], dtype=float)
                        basis = "posterior"
                    else:
                        mean = np.asarray(iblock.get("mean", []), dtype=float)
                        sig = iblock.get("sigma", None)
                        if sig is None:
                            # fallback
                            sig = np.full_like(mean, float(self._uq_default_obs_sigma(imt)), dtype=float)
                        else:
                            sig = np.asarray(sig, dtype=float)
                        basis = "prior"
    
                    N = mean.size
                    if N == 0:
                        continue
    
                    # Determine thresholds
                    th_poe = self._uq_thresholds_for_imt(poe_thresholds, imt)
                    th_met = self._uq_thresholds_for_imt(metrics_thresholds, imt)
    
                    # Sampling
                    # independent: X = mean + sig * Z
                    # correlated:  X = mean + sig * (sqrt(rho)*Zcorr + sqrt(1-rho)*Zind)
                    # We use random features for Zcorr with approximate kernel correlation.
                    if str(correlation).lower() == "independent":
                        # draw in chunks to control memory
                        poe_maps = {}  # threshold -> prob array
                        ci_maps = {}   # q -> array
                        met_dist = {}  # threshold -> metric array samples (optional)
    
                        # initialize accumulators
                        if compute_poe and th_poe:
                            for t in th_poe:
                                poe_maps[float(t)] = np.zeros(N, dtype=float)
    
                        # For CI, store running samples only if needed; we compute quantiles using chunked reservoir if small
                        # Here: we compute quantiles by storing samples if N*n_samples small, else do per-point chunk sort using memmap to disk.
                        # Practical approach: if store_samples or N*n_samples <= 2e7, store in float32.
                        store_for_ci = compute_ci and (store_samples or (N * int(n_samples) <= 2e7))
                        samples_mem = None
                        samples_path = None
    
                        if store_samples or store_for_ci:
                            # store samples in a memmap file to avoid RAM blowup
                            samples_path = os.path.join(uq_dirs["posterior"], f"uq_mc_samples_{event_id}_v{vkey}_{imt}.dat")
                            samples_mem = np.memmap(samples_path, mode="w+", dtype="float32", shape=(int(n_samples), N))
    
                        # metrics distributions (area>=k) use cell areas; if not available, we compute proxy using uniform cell area
                        cell_area = None
                        if compute_metrics and th_met:
                            cell_area = self._uq_estimate_cell_area_km2(lon, lat)
                            for t in th_met:
                                met_dist[float(t)] = np.zeros(int(n_samples), dtype=float)
    
                        # sample loop
                        for s0 in range(0, int(n_samples), 1):
                            z = rng.standard_normal(N, dtype=float)
                            x = mean + sig * z
    
                            if samples_mem is not None:
                                samples_mem[s0, :] = x.astype("float32")
    
                            if compute_poe and th_poe:
                                for t in th_poe:
                                    poe_maps[float(t)] += (x >= float(t)).astype(float)
    
                            if compute_metrics and th_met:
                                # area>=k = sum(cell_area where x>=k)
                                mask = None
                                for t in th_met:
                                    mask = (x >= float(t))
                                    met_dist[float(t)][s0] = float(np.nansum(cell_area[mask])) if cell_area is not None else float(np.nansum(mask))
    
                        # finalize poe
                        if compute_poe and th_poe:
                            for t in th_poe:
                                poe_maps[float(t)] /= float(n_samples)
    
                        # credible intervals
                        if compute_ci and ci_quantiles:
                            if samples_mem is None:
                                # should not happen due to store_for_ci logic
                                self._uq_log(f"[v{vkey}][{imt}] CI requested but samples not stored; skipping CI.", level="warning", verbose=verbose)
                            else:
                                # compute quantiles per grid point; do in chunks to avoid RAM
                                qs = [float(q) for q in ci_quantiles]
                                ci_maps = {q: np.full(N, np.nan, dtype=float) for q in qs}
                                # chunk over points
                                pchunk = 50000
                                for i0 in range(0, N, pchunk):
                                    i1 = min(N, i0 + pchunk)
                                    block = np.asarray(samples_mem[:, i0:i1], dtype=float)  # (S, P)
                                    # sort along samples axis
                                    block.sort(axis=0)
                                    for q in qs:
                                        idx = int(round((int(n_samples) - 1) * q))
                                        idx = max(0, min(int(n_samples) - 1, idx))
                                        ci_maps[q][i0:i1] = block[idx, :]
    
                        # Export products
                        # store under iblock
                        iblock.setdefault("mc", {})
                        iblock["mc"]["basis"] = basis
                        iblock["mc"]["n_samples"] = int(n_samples)
                        iblock["mc"]["correlation"] = str(correlation)
                        iblock["mc"]["corr_length_km"] = float(corr_length_km)
    
                        if compute_poe and poe_maps:
                            iblock["mc"]["poe"] = poe_maps
                            self._uq_export_map_dict(
                                uq_dirs["maps"],
                                event_id=event_id,
                                version=vkey,
                                imt=imt,
                                kind="poe",
                                lon=lon,
                                lat=lat,
                                maps=poe_maps,
                                verbose=verbose,
                            )
    
                        if compute_ci and ci_maps:
                            iblock["mc"]["ci"] = ci_maps
                            self._uq_export_map_dict(
                                uq_dirs["maps"],
                                event_id=event_id,
                                version=vkey,
                                imt=imt,
                                kind="ci",
                                lon=lon,
                                lat=lat,
                                maps=ci_maps,
                                verbose=verbose,
                            )
    
                        if compute_metrics and met_dist:
                            iblock["mc"]["metrics"] = {"area_ge": met_dist}
                            self._uq_export_metric_dist(
                                uq_dirs["metrics"],
                                event_id=event_id,
                                version=vkey,
                                imt=imt,
                                metric_name="area_ge",
                                dist_dict=met_dist,
                                verbose=verbose,
                            )
    
                        if store_samples:
                            iblock["mc"]["samples_path"] = samples_path
    
                        # If not storing samples, clean up memmap file
                        if (samples_mem is not None) and (not store_samples):
                            try:
                                # release file handle then remove
                                del samples_mem
                                if samples_path and os.path.exists(samples_path):
                                    os.remove(samples_path)
                            except Exception:
                                pass
    
                    else:
                        # correlated sampling (random features approx)
                        corr = str(correlation).lower()
                        if corr not in ("exp", "gauss"):
                            raise ValueError(f"Unsupported correlation='{correlation}'. Use 'independent','exp','gauss'.")
    
                        # Build random features once per (v,imt)
                        # We approximate a stationary kernel by mapping coordinates->features and drawing correlated field.
                        # Zcorr = Phi @ w, where w~N(0,I), Phi shape (N, D)
                        D = int(min(512, max(64, int(np.sqrt(N) // 1))))  # heuristic
                        Phi = self._uq_random_features(lon, lat, D=D, kernel=corr, length_km=float(corr_length_km), rng=rng)
                        # Normalize features to unit variance approx
                        Phi = Phi / np.sqrt(np.maximum(np.sum(Phi * Phi, axis=1, keepdims=True), 1e-12))
    
                        poe_maps = {}
                        ci_maps = {}
                        met_dist = {}
    
                        if compute_poe and th_poe:
                            for t in th_poe:
                                poe_maps[float(t)] = np.zeros(N, dtype=float)
    
                        store_for_ci = compute_ci and (store_samples or (N * int(n_samples) <= 1.5e7))
                        samples_mem = None
                        samples_path = None
                        if store_samples or store_for_ci:
                            samples_path = os.path.join(uq_dirs["posterior"], f"uq_mc_samples_{event_id}_v{vkey}_{imt}.dat")
                            samples_mem = np.memmap(samples_path, mode="w+", dtype="float32", shape=(int(n_samples), N))
    
                        cell_area = None
                        if compute_metrics and th_met:
                            cell_area = self._uq_estimate_cell_area_km2(lon, lat)
                            for t in th_met:
                                met_dist[float(t)] = np.zeros(int(n_samples), dtype=float)
    
                        for s0 in range(int(n_samples)):
                            # correlated component
                            w = rng.standard_normal(Phi.shape[1], dtype=float)
                            zcorr = Phi @ w
                            # independent component
                            zind = rng.standard_normal(N, dtype=float)
                            # mix (simple): z = alpha*zcorr + sqrt(1-alpha^2)*zind
                            alpha = 0.7  # fixed mixing coefficient (practical); can be exposed later
                            z = alpha * zcorr + np.sqrt(max(1e-8, 1.0 - alpha * alpha)) * zind
                            x = mean + sig * z
    
                            if samples_mem is not None:
                                samples_mem[s0, :] = x.astype("float32")
    
                            if compute_poe and th_poe:
                                for t in th_poe:
                                    poe_maps[float(t)] += (x >= float(t)).astype(float)
    
                            if compute_metrics and th_met:
                                for t in th_met:
                                    mask = (x >= float(t))
                                    met_dist[float(t)][s0] = float(np.nansum(cell_area[mask])) if cell_area is not None else float(np.nansum(mask))
    
                        if compute_poe and th_poe:
                            for t in th_poe:
                                poe_maps[float(t)] /= float(n_samples)
    
                        if compute_ci and ci_quantiles and (samples_mem is not None):
                            qs = [float(q) for q in ci_quantiles]
                            ci_maps = {q: np.full(N, np.nan, dtype=float) for q in qs}
                            pchunk = 50000
                            for i0 in range(0, N, pchunk):
                                i1 = min(N, i0 + pchunk)
                                block = np.asarray(samples_mem[:, i0:i1], dtype=float)
                                block.sort(axis=0)
                                for q in qs:
                                    idx = int(round((int(n_samples) - 1) * q))
                                    idx = max(0, min(int(n_samples) - 1, idx))
                                    ci_maps[q][i0:i1] = block[idx, :]
    
                        iblock.setdefault("mc", {})
                        iblock["mc"]["basis"] = basis
                        iblock["mc"]["n_samples"] = int(n_samples)
                        iblock["mc"]["correlation"] = corr
                        iblock["mc"]["corr_length_km"] = float(corr_length_km)
    
                        if compute_poe and poe_maps:
                            iblock["mc"]["poe"] = poe_maps
                            self._uq_export_map_dict(
                                uq_dirs["maps"], event_id, vkey, imt, "poe", lon, lat, poe_maps, verbose=verbose
                            )
    
                        if compute_ci and ci_maps:
                            iblock["mc"]["ci"] = ci_maps
                            self._uq_export_map_dict(
                                uq_dirs["maps"], event_id, vkey, imt, "ci", lon, lat, ci_maps, verbose=verbose
                            )
    
                        if compute_metrics and met_dist:
                            iblock["mc"]["metrics"] = {"area_ge": met_dist}
                            self._uq_export_metric_dist(
                                uq_dirs["metrics"], event_id, vkey, imt, "area_ge", met_dist, verbose=verbose
                            )
    
                        if store_samples:
                            iblock["mc"]["samples_path"] = samples_path
    
                        if (samples_mem is not None) and (not store_samples):
                            try:
                                del samples_mem
                                if samples_path and os.path.exists(samples_path):
                                    os.remove(samples_path)
                            except Exception:
                                pass
    
                    summaries.append({
                        "event_id": event_id,
                        "version": vkey,
                        "imt": imt,
                        "basis": basis,
                        "n_samples": int(n_samples),
                        "correlation": str(correlation),
                        "corr_length_km": float(corr_length_km) if str(correlation).lower() != "independent" else "",
                        "poe_thresholds": ";".join([str(t) for t in th_poe]) if (compute_poe and th_poe) else "",
                        "ci_quantiles": ";".join([str(q) for q in ci_quantiles]) if (compute_ci and ci_quantiles) else "",
                        "metrics_thresholds": ";".join([str(t) for t in th_met]) if (compute_metrics and th_met) else "",
                    })
    
                    self._uq_log(f"[v{vkey}][{imt}] monte_carlo complete ({basis}, S={n_samples}, corr={correlation}).", level="info", verbose=verbose)
    
                except Exception as e:
                    msg = f"[v{vkey}][{imt}] monte_carlo failed: {e}"
                    if strict:
                        raise
                    self._uq_log(msg, level="error", verbose=verbose)
    
        if export_summary and summaries:
            try:
                csv_path = os.path.join(uq_dirs["posterior"], f"uq_monte_carlo_summary_{event_id}.csv")
                self._uq_write_dictlist_csv(csv_path, summaries)
                self._uq_log(f"Monte Carlo summary exported: {csv_path}", level="info", verbose=verbose)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"MC summary export failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        if dataset is None:
            self.uq_data = ds
        return ds


    # -------------------------
    # Private helpers (PATCH 4)
    # -------------------------
    
    def _uq_default_thresholds(self):
        """
        Default threshold dictionary for PoE/metrics if user does not supply.
        These are practical defaults; override for your study.
        """
        return {
            "MMI": [4.0, 5.0, 6.0, 7.0],
            "PGA": [0.05, 0.10, 0.20, 0.30],
            "PGV": [5.0, 10.0, 20.0, 30.0],
            "PSA": [0.10, 0.20, 0.30],
        }
    
    
    def _uq_thresholds_for_imt(self, thresholds_obj, imt):
        """Resolve thresholds from dict or list."""
        if thresholds_obj is None:
            return []
        imt_u = str(imt).upper()
        if isinstance(thresholds_obj, (list, tuple)):
            return list(thresholds_obj)
        if isinstance(thresholds_obj, dict):
            if imt_u in thresholds_obj:
                return list(thresholds_obj[imt_u])
            if imt_u.startswith("PSA") and "PSA" in thresholds_obj:
                return list(thresholds_obj["PSA"])
            # fallback to any generic key
            if "ALL" in thresholds_obj:
                return list(thresholds_obj["ALL"])
        return []
    
    
    def _uq_default_obs_sigma(self, imt):
        """Reuse Patch 3-like defaults when sigma missing in Monte Carlo."""
        imt_u = str(imt).upper()
        if imt_u == "MMI":
            return 0.35
        if imt_u.startswith("PSA"):
            return 0.60
        if imt_u in ("PGA", "PGV"):
            return 0.60
        return 0.60
    
    
    def _uq_estimate_cell_area_km2(self, lon, lat):
        """
        Estimate per-point cell area (km^2) from scattered grid points by inferring spacing.
        This is approximate but stable enough for uncertainty bands and trends.
    
        If grid is rectilinear, spacing inferred from nearest-neighbor distances.
        Returns array length N with constant area estimate.
        """
        import numpy as np
    
        lon = np.asarray(lon, dtype=float).ravel()
        lat = np.asarray(lat, dtype=float).ravel()
        N = lon.size
        if N < 4:
            return np.ones(N, dtype=float)
    
        # infer typical dx,dy in degrees
        # Use random subset for speed
        rng = np.random.default_rng(123)
        idx = rng.choice(N, size=min(N, 5000), replace=False)
        lon_s = lon[idx]
        lat_s = lat[idx]
    
        tree = self._uq_try_build_kdtree(lon_s, lat_s)
        if tree is not None and len(lon_s) > 10:
            # query 2 nearest to estimate spacing (first is self)
            d, _ = tree.query(np.c_[lon_s, lat_s], k=2)
            ddeg = np.median(d[:, 1])
        else:
            # crude fallback
            ddeg = np.median(np.sqrt((lon_s - np.mean(lon_s))**2 + (lat_s - np.mean(lat_s))**2)) / 20.0
            ddeg = max(ddeg, 1e-4)
    
        # Convert to km: 1 deg ~111 km (rough), area ~ (d*111)^2
        km = 111.32 * ddeg
        area = km * km
        return np.full(N, float(area), dtype=float)
    
    
    def _uq_export_map_dict(self, out_dir, event_id, version, imt, kind, lon, lat, maps, verbose=True):
        """
        Export a dict of map arrays to CSV(s) for portability.
        Each CSV: lon,lat,value
        """
        import os
        import numpy as np
        import csv
    
        lon = np.asarray(lon, dtype=float).ravel()
        lat = np.asarray(lat, dtype=float).ravel()
    
        for key, arr in (maps or {}).items():
            a = np.asarray(arr, dtype=float).ravel()
            fn = f"uq_{kind}_{event_id}_v{version}_{imt}_{str(key).replace('.','p')}.csv"
            path = os.path.join(out_dir, fn)
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["lon", "lat", "value"])
                for i in range(lon.size):
                    w.writerow([lon[i], lat[i], a[i]])
        self._uq_log(f"Exported {kind} maps for v{version} {imt} -> {out_dir}", level="debug", verbose=verbose)
    
    
    def _uq_export_metric_dist(self, out_dir, event_id, version, imt, metric_name, dist_dict, verbose=True):
        """
        Export metric distributions (per threshold) to CSV.
        Columns: sample_index, threshold, value
        """
        import os
        import csv
        import numpy as np
    
        rows = []
        for thr, vals in (dist_dict or {}).items():
            vals = np.asarray(vals, dtype=float).ravel()
            for i, v in enumerate(vals):
                rows.append({"sample": i, "threshold": float(thr), "value": float(v)})
    
        fn = f"uq_metric_{metric_name}_{event_id}_v{version}_{imt}.csv"
        path = os.path.join(out_dir, fn)
        self._uq_write_dictlist_csv(path, rows)
        self._uq_log(f"Exported metric dist {metric_name} for v{version} {imt} -> {path}", level="debug", verbose=verbose)
    
    
    def _uq_random_features(self, lon, lat, D=128, kernel="exp", length_km=40.0, rng=None):
        """
        Random Fourier features for stationary kernels (approximate correlated fields).
    
        Inputs:
          lon,lat: 1D arrays
          D: number of features
          kernel: "exp" or "gauss"
          length_km: correlation length
        Returns:
          Phi: (N, D) feature matrix
    
        Notes:
          - We first project lon/lat to km using equirectangular projection around median latitude.
          - For gaussian kernel: spectral density is normal; for exp kernel: Cauchy-like.
            We implement a practical approximation:
              * gauss: w ~ N(0, (1/ell)^2 I)
              * exp:   w ~ t-dist like; approximate via normal / scale mixture (good enough for trends)
        """
        import numpy as np
    
        lon = np.asarray(lon, dtype=float).ravel()
        lat = np.asarray(lat, dtype=float).ravel()
        N = lon.size
        if rng is None:
            rng = np.random.default_rng(0)
    
        # project to km
        lat0 = np.deg2rad(np.nanmedian(lat))
        x = (lon - np.nanmedian(lon)) * 111.32 * np.cos(lat0)
        y = (lat - np.nanmedian(lat)) * 111.32
        XY = np.c_[x, y]  # km
    
        ell = max(1e-6, float(length_km))
        kernel = str(kernel).lower()
    
        if kernel == "gauss":
            W = rng.normal(0.0, 1.0 / ell, size=(D, 2))
        else:
            # "exp" approximate: scale-mixture of normals
            # draw scales from inverse-gamma-ish via 1/sqrt(u) with u~Uniform
            u = rng.uniform(0.05, 1.0, size=(D, 1))
            scale = (1.0 / ell) / np.sqrt(u)
            W = rng.normal(0.0, 1.0, size=(D, 2)) * scale
    
        b = rng.uniform(0.0, 2.0 * np.pi, size=(D,))
        # Phi = sqrt(2/D) * cos(XW^T + b)
        Z = XY @ W.T + b[None, :]
        Phi = np.sqrt(2.0 / float(D)) * np.cos(Z)
        return Phi




    # =========================
    # PATCH 5 — mode="hierarchical" (practical hierarchical EB across versions; uncertainty decay vs time)
    # ADD INSIDE SHAKEtime CLASS
    # =========================
    
    def uq_hierarchical(
        self,
        dataset=None,
        version_list=None,
        imts=None,
        use_posterior_if_present=True,
        pool_strength=1.0,
        smooth_strength=0.5,
        min_versions=3,
        min_points=200,
        robust=True,
        max_iter=10,
        tol=1e-4,
        export_summary=True,
        output_path="./export",
        strict=False,
        verbose=True,
    ):
        """
        Practical hierarchical Bayesian (empirical Bayes) model across versions.
    
        Goal
        ----
        Stabilize version-to-version trajectories of (mean, sigma) and produce a practical
        "uncertainty decay vs time" signal WITHOUT heavy dependencies or MCMC.
    
        Model (practical EB)
        --------------------
        For each IMT and each grid point g, we have per-version estimates:
            m_v(g), s_v(g)
        from either prior (mean/sigma) or posterior (mean_post/sigma_post).
    
        We treat m_v(g) as noisy observations of a latent smooth trajectory μ_v(g) with:
            m_v(g) ~ Normal( μ_v(g), s_v(g)^2 )
    
        We impose (i) pooling across versions via a global mean and (ii) smoothing across time (versions)
        using a simple quadratic penalty (like a random-walk prior). This is equivalent to MAP estimation
        of μ_v(g) with:
            sum_v (m_v-μ_v)^2 / s_v^2   +   λ * sum_v (Δμ_v)^2
        plus optional global shrinkage toward the across-version mean at each g.
    
        We also estimate a pooled “extra variance” component τ^2 (heterogeneity not captured by s_v)
        via iterative empirical Bayes:
            s_eff_v^2 = s_v^2 + τ^2
        and re-solve.
    
        Outputs
        -------
        Adds under dataset["versions"][v]["imts"][imt]:
          - mean_hier : hierarchical-smoothed mean (same length as grid)
          - sigma_hier: effective sigma (sqrt(s_v^2 + τ^2) optionally smoothed)
          - hier_meta : dict with settings and estimated τ
    
        Also adds dataset["hierarchical"] summary:
          - per IMT: tau2 estimate, and global median sigma vs version (decay curve)
    
        Parameters
        ----------
        use_posterior_if_present : bool
            If True, use mean_post/sigma_post when available; else fall back to mean/sigma.
        pool_strength : float
            Strength of shrinkage toward across-version mean at each grid point (0 disables).
        smooth_strength : float
            Strength of smoothing across versions (0 disables).
        robust : bool
            If True, use robust aggregation (median/MAD) for tau estimation and sigma summaries.
        max_iter : int
            EB iterations for tau2 refinement.
        tol : float
            Convergence tolerance for tau2.
        """
        import os
        import numpy as np
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_hierarchical: dataset is None and self.uq_data not found. Run uq_build_dataset first.")
    
        event_id = ds.get("event_id", self._uq_get_event_id())
        uq_dirs = ds.get("meta", {}).get("uq_export_dirs", None)
        if uq_dirs is None:
            uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        # versions ordering (numeric where possible, otherwise stable string)
        vkeys_all = list(ds.get("versions", {}).keys())
        vkeys = self._uq_sort_version_keys(vkeys_all)
    
        if version_list is not None:
            vset = set([str(v) for v in version_list])
            vkeys = [k for k in vkeys if k in vset]
    
        if len(vkeys) < int(min_versions):
            msg = f"uq_hierarchical: need at least {min_versions} versions, got {len(vkeys)}."
            if strict:
                raise ValueError(msg)
            self._uq_log(msg, level="warning", verbose=verbose)
            return ds
    
        # Determine IMTs intersection/union
        # We'll process each IMT if it exists for at least min_versions versions.
        imt_counts = {}
        for vk in vkeys:
            vblock = ds["versions"].get(vk, {})
            if not isinstance(vblock, dict) or "imts" not in vblock:
                continue
            for imt in vblock["imts"].keys():
                imt_counts[imt] = imt_counts.get(imt, 0) + 1
    
        imt_list = [k for k, c in imt_counts.items() if c >= int(min_versions)]
        if imts is not None:
            req = set([str(x) for x in imts])
            imt_list = [k for k in imt_list if k in req]
    
        if not imt_list:
            msg = "uq_hierarchical: no IMTs meet min_versions criterion."
            if strict:
                raise ValueError(msg)
            self._uq_log(msg, level="warning", verbose=verbose)
            return ds
    
        hier_summary = ds.setdefault("hierarchical", {"per_imt": {}, "created_by": "SHAKEtime.uq_hierarchical"})
    
        for imt in imt_list:
            try:
                # Build matrices for this IMT: M (V,N) and S (V,N)
                M_list = []
                S_list = []
                lon = lat = None
    
                used_vkeys = []
                for vk in vkeys:
                    vblock = ds["versions"].get(vk, {})
                    if not vblock or "imts" not in vblock or imt not in vblock["imts"]:
                        continue
                    ib = vblock["imts"][imt]
                    if lon is None:
                        lon = np.asarray(ib.get("lon", []), dtype=float)
                        lat = np.asarray(ib.get("lat", []), dtype=float)
                    mean, sig, basis = self._uq_pick_mean_sigma(ib, prefer_posterior=use_posterior_if_present)
                    if mean is None or sig is None:
                        continue
                    mean = np.asarray(mean, dtype=float)
                    sig = np.asarray(sig, dtype=float)
                    if mean.size < int(min_points):
                        continue
                    M_list.append(mean)
                    S_list.append(sig)
                    used_vkeys.append(vk)
    
                V = len(used_vkeys)
                if V < int(min_versions):
                    self._uq_log(f"[{imt}] insufficient versions with data after filtering: {V}", level="warning", verbose=verbose)
                    continue
    
                M = np.vstack(M_list)  # (V,N)
                S = np.vstack(S_list)  # (V,N)
                N = M.shape[1]
    
                # Initial tau2 estimate (per IMT): based on across-version residual variance
                tau2 = self._uq_initial_tau2(M, S, robust=robust)
    
                # Iterative EB refinement
                prev_tau2 = tau2
                for it in range(int(max_iter)):
                    # effective variances
                    Se2 = np.maximum(S * S + tau2, 1e-12)
    
                    # Solve for smoothed μ_v(g): per grid point (independent) with smoothing across versions.
                    # We solve a banded linear system for each g:
                    #   (W + λ*L + γ*I) μ = W m + γ * m_bar
                    # where W is diag(1/Se2[:,g]), L is 1D Laplacian, γ from pool_strength.
                    mu = self._uq_smooth_across_versions(
                        M,
                        Se2,
                        smooth_strength=float(smooth_strength),
                        pool_strength=float(pool_strength),
                    )  # (V,N)
    
                    # Update tau2 from residuals (method-of-moments):
                    # residual variance beyond Se2 should inform tau2; we re-estimate as max(0, median(var(res) - median(S^2)))
                    tau2_new = self._uq_update_tau2(M, mu, S, robust=robust)
                    if abs(tau2_new - prev_tau2) / (abs(prev_tau2) + 1e-12) < float(tol):
                        tau2 = tau2_new
                        break
                    prev_tau2 = tau2_new
                    tau2 = tau2_new
    
                # Final effective sigma (not smoothed): sqrt(S^2 + tau2)
                sigma_eff = np.sqrt(np.maximum(S * S + tau2, 1e-12))
    
                # Write back results per version
                for i, vk in enumerate(used_vkeys):
                    ib = ds["versions"][vk]["imts"][imt]
                    ib["mean_hier"] = mu[i, :].copy()
                    ib["sigma_hier"] = sigma_eff[i, :].copy()
                    ib["hier_meta"] = {
                        "mode": "hierarchical",
                        "basis": "posterior" if use_posterior_if_present else "prior",
                        "tau2": float(tau2),
                        "pool_strength": float(pool_strength),
                        "smooth_strength": float(smooth_strength),
                        "robust": bool(robust),
                        "versions_used": used_vkeys,
                    }
    
                # Build uncertainty decay curve (median sigma over grid) by version
                med_sig = []
                for i, vk in enumerate(used_vkeys):
                    svec = sigma_eff[i, :]
                    if robust:
                        med_sig.append(float(np.nanmedian(svec)))
                    else:
                        med_sig.append(float(np.nanmean(svec)))
                decay = [{"event_id": event_id, "imt": imt, "version": vk, "sigma_median": med_sig[i]} for i, vk in enumerate(used_vkeys)]
    
                hier_summary["per_imt"][imt] = {
                    "tau2": float(tau2),
                    "versions_used": used_vkeys,
                    "sigma_decay": decay,
                }
    
                self._uq_log(f"[{imt}] hierarchical EB done: versions={V}, tau2={tau2:.4g}", level="info", verbose=verbose)
    
                if export_summary:
                    try:
                        csv_path = os.path.join(uq_dirs["posterior"], f"uq_hierarchical_decay_{event_id}_{imt}.csv")
                        self._uq_write_dictlist_csv(csv_path, decay)
                    except Exception as e:
                        if strict:
                            raise
                        self._uq_log(f"[{imt}] decay export failed (non-fatal): {e}", level="warning", verbose=verbose)
    
            except Exception as e:
                msg = f"[{imt}] hierarchical failed: {e}"
                if strict:
                    raise
                self._uq_log(msg, level="error", verbose=verbose)
    
        if dataset is None:
            self.uq_data = ds
        return ds


    # -------------------------
    # Private helpers (PATCH 5)
    # -------------------------
    
    def _uq_sort_version_keys(self, vkeys):
        """Sort version keys numerically when possible, else lexicographically."""
        def _to_num(k):
            try:
                # handle "v12" or "12"
                s = str(k).lower().strip()
                if s.startswith("v"):
                    s = s[1:]
                return float(s)
            except Exception:
                return None
        nums = [(k, _to_num(k)) for k in vkeys]
        if all(n is not None for _, n in nums):
            return [k for k, _ in sorted(nums, key=lambda x: x[1])]
        # mixed: keep numeric first then string
        num_part = [x for x in nums if x[1] is not None]
        str_part = [x for x in nums if x[1] is None]
        out = [k for k, _ in sorted(num_part, key=lambda x: x[1])] + [k for k, _ in sorted(str_part, key=lambda x: str(x[0]))]
        return out
    
    
    def _uq_pick_mean_sigma(self, imt_block, prefer_posterior=True):
        """Pick mean/sigma arrays from block: posterior > hierarchical > prior (depending on preference)."""
        # If user runs hierarchical after bayes_update, posterior exists; we use it if requested.
        if prefer_posterior:
            mp = imt_block.get("mean_post", None)
            sp = imt_block.get("sigma_post", None)
            if mp is not None and sp is not None:
                return mp, sp, "posterior"
        # If hierarchical already exists, allow reuse
        mh = imt_block.get("mean_hier", None)
        sh = imt_block.get("sigma_hier", None)
        if mh is not None and sh is not None:
            return mh, sh, "hierarchical"
        m0 = imt_block.get("mean", None)
        s0 = imt_block.get("sigma", None)
        if m0 is None or s0 is None:
            return None, None, "missing"
        return m0, s0, "prior"
    
    
    def _uq_initial_tau2(self, M, S, robust=True):
        """
        Initial tau^2 estimate for heterogeneity beyond reported sigma.
    
        We estimate across-version variance of M at each grid point, then subtract typical S^2.
        """
        import numpy as np
        # variance across versions per grid point
        if robust:
            # robust var via MAD scaled: var ≈ (1.4826*MAD)^2
            med = np.nanmedian(M, axis=0)
            mad = np.nanmedian(np.abs(M - med[None, :]), axis=0)
            var_m = (1.4826 * mad) ** 2
            s2_typ = np.nanmedian(S * S, axis=0)
            tau2 = np.nanmedian(np.maximum(var_m - s2_typ, 0.0))
        else:
            var_m = np.nanvar(M, axis=0)
            s2_typ = np.nanmean(S * S, axis=0)
            tau2 = np.nanmean(np.maximum(var_m - s2_typ, 0.0))
        if not np.isfinite(tau2):
            tau2 = 0.0
        return float(max(0.0, tau2))
    
    
    def _uq_update_tau2(self, M, mu, S, robust=True):
        """
        Update tau^2 using residuals r = M - mu.
    
        tau2 ≈ typical( var(r) - typical(S^2) ), clipped at 0.
        """
        import numpy as np
        r = M - mu
        if robust:
            med = np.nanmedian(r, axis=0)
            mad = np.nanmedian(np.abs(r - med[None, :]), axis=0)
            var_r = (1.4826 * mad) ** 2
            s2_typ = np.nanmedian(S * S, axis=0)
            tau2 = np.nanmedian(np.maximum(var_r - s2_typ, 0.0))
        else:
            var_r = np.nanvar(r, axis=0)
            s2_typ = np.nanmean(S * S, axis=0)
            tau2 = np.nanmean(np.maximum(var_r - s2_typ, 0.0))
        if not np.isfinite(tau2):
            tau2 = 0.0
        return float(max(0.0, tau2))
    
    
    def _uq_smooth_across_versions(self, M, Se2, smooth_strength=0.5, pool_strength=1.0):
        """
        Solve for μ (V,N) minimizing:
            sum_v (M_v-μ_v)^2/Se2_v  +  λ * sum_v (μ_v-μ_{v-1})^2  +  γ * sum_v (μ_v - mbar)^2
    
        Implemented per-grid-point as a tridiagonal system (Thomas algorithm) for speed.
    
        Parameters
        ----------
        M : array (V,N)
        Se2 : array (V,N) effective variance
        smooth_strength : float λ
        pool_strength : float γ
    
        Returns
        -------
        mu : array (V,N)
        """
        import numpy as np
        V, N = M.shape
        lam = float(max(0.0, smooth_strength))
        gam = float(max(0.0, pool_strength))
    
        # weights
        W = 1.0 / np.maximum(Se2, 1e-12)  # (V,N)
        mbar = np.nanmedian(M, axis=0)  # (N,) robust pool target (median)
    
        mu = np.zeros_like(M, dtype=float)
    
        # Tridiagonal coefficients per grid point g:
        # a_v = -lam (subdiag), b_v = W_v + 2lam + gam (diag), c_v = -lam (superdiag)
        # endpoints have 1*lam
        # rhs d_v = W_v*M_v + gam*mbar
        # We solve V-length tri-system for each g. Vectorize over N with loops over V (small).
        a = np.full((V, N), -lam, dtype=float)
        b = W + (2.0 * lam + gam)
        c = np.full((V, N), -lam, dtype=float)
    
        # endpoints adjust (random-walk prior)
        b[0, :] = W[0, :] + (lam + gam)
        b[-1, :] = W[-1, :] + (lam + gam)
        a[0, :] = 0.0
        c[-1, :] = 0.0
    
        d = W * M + gam * mbar[None, :]
    
        # Thomas algorithm (vectorized across N)
        # forward sweep
        cp = np.zeros_like(c)
        dp = np.zeros_like(d)
        cp[0, :] = c[0, :] / np.maximum(b[0, :], 1e-12)
        dp[0, :] = d[0, :] / np.maximum(b[0, :], 1e-12)
    
        for i in range(1, V):
            denom = np.maximum(b[i, :] - a[i, :] * cp[i - 1, :], 1e-12)
            cp[i, :] = c[i, :] / denom
            dp[i, :] = (d[i, :] - a[i, :] * dp[i - 1, :]) / denom
    
        # back substitution
        mu[-1, :] = dp[-1, :]
        for i in range(V - 2, -1, -1):
            mu[i, :] = dp[i, :] - cp[i, :] * mu[i + 1, :]
    
        return mu



    # =========================
    # PATCH 6 — Probability of Exceedance + Credible Intervals + Uncertainty decay utilities
    # ADD INSIDE SHAKEtime CLASS
    # =========================
    
    def uq_probability_of_exceedance(
        self,
        dataset=None,
        version_list=None,
        imt="MMI",
        thresholds=None,
        basis="auto",
        distribution="normal",
        output_path="./export",
        export_maps=True,
        export_tables=True,
        strict=False,
        verbose=True,
    ):
        """
        Compute probability-of-exceedance (PoE) products for a given IMT across versions.
    
        PoE is computed analytically assuming Normal(mean, sigma) at each grid point:
            PoE = P(X >= k) = 1 - Phi((k - mean)/sigma)
    
        If Monte Carlo products exist (Patch 4), you can set basis="mc" to use MC PoE maps if present.
        Otherwise it uses analytical Normal.
    
        Parameters
        ----------
        dataset : dict or None
            If None, uses self.uq_data.
        version_list : list or None
            Subset of versions. Default all.
        imt : str
            IMT to process.
        thresholds : list[float] or None
            Thresholds k for exceedance. If None, uses defaults.
        basis : str
            "auto" (default): choose in order: hierarchical -> posterior -> prior (if sigma exists); else fallback
            "prior", "posterior", "hierarchical", "mc"
        distribution : str
            Currently "normal" only (analytical).
        output_path : str
            Export root.
        export_maps : bool
            Export lon/lat/value CSV maps into uq/maps.
        export_tables : bool
            Export a per-version summary table into uq/metrics.
        strict : bool
            Raise on errors.
        verbose : bool
            Verbose logging.
    
        Returns
        -------
        dict
            dict with structure:
              poe["versions"][vkey]["thresholds"][k] = poe_array
              poe["summary"] = list of dict rows (optional)
        """
        import os
        import numpy as np
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_probability_of_exceedance: dataset is None and self.uq_data not found. Run uq_build_dataset first.")
    
        event_id = ds.get("event_id", self._uq_get_event_id())
        uq_dirs = ds.get("meta", {}).get("uq_export_dirs", None)
        if uq_dirs is None:
            uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        imt = str(imt)
        if thresholds is None:
            thresholds = self._uq_thresholds_for_imt(self._uq_default_thresholds(), imt)
        thresholds = [float(t) for t in thresholds]
    
        vkeys = self._uq_sort_version_keys(list(ds.get("versions", {}).keys()))
        if version_list is not None:
            vset = set([str(v) for v in version_list])
            vkeys = [k for k in vkeys if k in vset]
    
        poe_out = {"event_id": event_id, "imt": imt, "thresholds": thresholds, "versions": {}}
        summary_rows = []
    
        for vkey in vkeys:
            vblock = ds["versions"].get(vkey, {})
            if not vblock or "imts" not in vblock or imt not in vblock["imts"]:
                continue
            ib = vblock["imts"][imt]
            lon = np.asarray(ib.get("lon", []), dtype=float)
            lat = np.asarray(ib.get("lat", []), dtype=float)
            if lon.size == 0:
                continue
    
            poe_maps = {}
    
            # MC basis
            if str(basis).lower() == "mc" or (str(basis).lower() == "auto" and "mc" in ib and "poe" in ib["mc"]):
                mc_poe = ib.get("mc", {}).get("poe", {})
                for t in thresholds:
                    if float(t) in mc_poe:
                        poe_maps[float(t)] = np.asarray(mc_poe[float(t)], dtype=float)
                    else:
                        # allow string keys too
                        poe_maps[float(t)] = np.asarray(mc_poe.get(str(t), np.full(lon.size, np.nan)), dtype=float)
            else:
                mean, sig = self._uq_get_mean_sigma_by_basis(ib, basis=basis)
                mean = np.asarray(mean, dtype=float)
                sig = np.asarray(sig, dtype=float)
                sig = np.maximum(sig, 1e-12)
                # analytical normal cdf
                poe_maps = {t: self._uq_poe_normal(mean, sig, t) for t in thresholds}
    
            poe_out["versions"][vkey] = {"thresholds": poe_maps}
    
            # Summary: global mean PoE across grid (simple)
            for t in thresholds:
                pv = poe_maps.get(t, None)
                if pv is None:
                    continue
                summary_rows.append({
                    "event_id": event_id,
                    "version": vkey,
                    "imt": imt,
                    "threshold": float(t),
                    "poe_mean": float(np.nanmean(pv)),
                    "poe_median": float(np.nanmedian(pv)),
                    "poe_p95": float(np.nanpercentile(pv, 95)),
                })
    
            if export_maps and poe_maps:
                try:
                    self._uq_export_map_dict(
                        uq_dirs["maps"],
                        event_id=event_id,
                        version=vkey,
                        imt=imt,
                        kind="poe",
                        lon=lon,
                        lat=lat,
                        maps=poe_maps,
                        verbose=verbose,
                    )
                except Exception as e:
                    if strict:
                        raise
                    self._uq_log(f"[v{vkey}][{imt}] PoE map export failed: {e}", level="warning", verbose=verbose)
    
        if export_tables and summary_rows:
            try:
                csv_path = os.path.join(uq_dirs["metrics"], f"uq_poe_summary_{event_id}_{imt}.csv")
                self._uq_write_dictlist_csv(csv_path, summary_rows)
                self._uq_log(f"PoE summary exported: {csv_path}", level="info", verbose=verbose)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"PoE summary export failed: {e}", level="warning", verbose=verbose)
    
        poe_out["summary"] = summary_rows
        return poe_out
    
    
    def uq_credible_intervals(
        self,
        dataset=None,
        version_list=None,
        imt="MMI",
        quantiles=(0.05, 0.5, 0.95),
        basis="auto",
        method="auto",
        output_path="./export",
        export_maps=True,
        export_tables=True,
        strict=False,
        verbose=True,
    ):
        """
        Compute credible interval (CI) maps for a given IMT across versions.
    
        Sources:
          - If Monte Carlo samples exist and method allows, uses MC quantiles already stored as iblock["mc"]["ci"]
            OR recomputes quantiles from stored sample file if present.
          - Otherwise, uses analytical Normal quantiles:
                q_p = mean + sigma * Phi^{-1}(p)
    
        Parameters
        ----------
        quantiles : tuple
            e.g. (0.05, 0.5, 0.95)
        basis : str
            "auto" (default): hierarchical -> posterior -> prior; or "prior"/"posterior"/"hierarchical"/"mc"
        method : str
            "auto" (default): prefer MC CIs if available, else analytical
            "mc" : require MC
            "analytical" : force analytical
        export_maps : bool
            Export CI maps to uq/maps.
        export_tables : bool
            Export summary table (grid-median CI) to uq/metrics.
    
        Returns
        -------
        dict
            ci["versions"][vkey]["quantiles"][q] = ci_array
        """
        import os
        import numpy as np
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_credible_intervals: dataset is None and self.uq_data not found. Run uq_build_dataset first.")
    
        event_id = ds.get("event_id", self._uq_get_event_id())
        uq_dirs = ds.get("meta", {}).get("uq_export_dirs", None)
        if uq_dirs is None:
            uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        imt = str(imt)
        qs = [float(q) for q in quantiles]
    
        vkeys = self._uq_sort_version_keys(list(ds.get("versions", {}).keys()))
        if version_list is not None:
            vset = set([str(v) for v in version_list])
            vkeys = [k for k in vkeys if k in vset]
    
        out = {"event_id": event_id, "imt": imt, "quantiles": qs, "versions": {}}
        summary_rows = []
    
        for vkey in vkeys:
            vblock = ds["versions"].get(vkey, {})
            if not vblock or "imts" not in vblock or imt not in vblock["imts"]:
                continue
            ib = vblock["imts"][imt]
            lon = np.asarray(ib.get("lon", []), dtype=float)
            lat = np.asarray(ib.get("lat", []), dtype=float)
            if lon.size == 0:
                continue
    
            ci_maps = {}
    
            # Decide method
            method_eff = str(method).lower()
            if method_eff == "auto":
                method_eff = "mc" if ("mc" in ib and "ci" in (ib.get("mc") or {})) else "analytical"
    
            if method_eff == "mc":
                mc_ci = ib.get("mc", {}).get("ci", {})
                for q in qs:
                    if q in mc_ci:
                        ci_maps[q] = np.asarray(mc_ci[q], dtype=float)
                    else:
                        ci_maps[q] = np.asarray(mc_ci.get(str(q), np.full(lon.size, np.nan)), dtype=float)
    
                # If missing but sample path exists, attempt compute
                if any(np.all(~np.isfinite(ci_maps[q])) for q in qs) and ("samples_path" in (ib.get("mc") or {})):
                    try:
                        ci_maps = self._uq_ci_from_samples_file(ib["mc"]["samples_path"], qs, lon.size)
                    except Exception as e:
                        if strict:
                            raise
                        self._uq_log(f"[v{vkey}][{imt}] CI from samples failed: {e}", level="warning", verbose=verbose)
                        ci_maps = {}
    
            if method_eff == "analytical":
                mean, sig = self._uq_get_mean_sigma_by_basis(ib, basis=basis)
                mean = np.asarray(mean, dtype=float)
                sig = np.maximum(np.asarray(sig, dtype=float), 1e-12)
                z = {q: self._uq_norm_ppf(q) for q in qs}
                ci_maps = {q: mean + sig * z[q] for q in qs}
    
            out["versions"][vkey] = {"quantiles": ci_maps}
    
            # Summary
            if ci_maps:
                row = {"event_id": event_id, "version": vkey, "imt": imt}
                for q in qs:
                    arr = ci_maps.get(q, None)
                    if arr is None:
                        continue
                    row[f"q{int(round(q*100)):02d}_median"] = float(np.nanmedian(arr))
                summary_rows.append(row)
    
            if export_maps and ci_maps:
                try:
                    self._uq_export_map_dict(
                        uq_dirs["maps"],
                        event_id=event_id,
                        version=vkey,
                        imt=imt,
                        kind="ci",
                        lon=lon,
                        lat=lat,
                        maps=ci_maps,
                        verbose=verbose,
                    )
                except Exception as e:
                    if strict:
                        raise
                    self._uq_log(f"[v{vkey}][{imt}] CI map export failed: {e}", level="warning", verbose=verbose)
    
        if export_tables and summary_rows:
            try:
                csv_path = os.path.join(uq_dirs["metrics"], f"uq_ci_summary_{event_id}_{imt}.csv")
                self._uq_write_dictlist_csv(csv_path, summary_rows)
                self._uq_log(f"CI summary exported: {csv_path}", level="info", verbose=verbose)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"CI summary export failed: {e}", level="warning", verbose=verbose)
    
        return out
    
    
    def uq_uncertainty_decay(
        self,
        dataset=None,
        imt="MMI",
        basis="auto",
        summary_stat="median",
        output_path="./export",
        export=True,
        strict=False,
        verbose=True,
    ):
        """
        Compute uncertainty decay vs version for a given IMT.
    
        This produces a compact curve:
            sigma_summary(version) = median/mean of sigma grid for that version.
    
        If hierarchical results exist and basis="auto", this will prefer sigma_hier.
        Otherwise it chooses posterior sigma_post then prior sigma.
    
        Returns list of dicts with version and sigma_summary.
        """
        import os
        import numpy as np
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_uncertainty_decay: dataset is None and self.uq_data not found. Run uq_build_dataset first.")
    
        event_id = ds.get("event_id", self._uq_get_event_id())
        uq_dirs = ds.get("meta", {}).get("uq_export_dirs", None)
        if uq_dirs is None:
            uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        imt = str(imt)
        vkeys = self._uq_sort_version_keys(list(ds.get("versions", {}).keys()))
    
        rows = []
        for vkey in vkeys:
            vblock = ds["versions"].get(vkey, {})
            if not vblock or "imts" not in vblock or imt not in vblock["imts"]:
                continue
            ib = vblock["imts"][imt]
            _, sig = self._uq_get_mean_sigma_by_basis(ib, basis=basis)
            sig = np.asarray(sig, dtype=float)
            if str(summary_stat).lower() == "mean":
                sval = float(np.nanmean(sig))
            else:
                sval = float(np.nanmedian(sig))
            rows.append({"event_id": event_id, "imt": imt, "version": vkey, "sigma_summary": sval})
    
        if export and rows:
            try:
                csv_path = os.path.join(uq_dirs["metrics"], f"uq_uncertainty_decay_{event_id}_{imt}.csv")
                self._uq_write_dictlist_csv(csv_path, rows)
                self._uq_log(f"Uncertainty decay exported: {csv_path}", level="info", verbose=verbose)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"Uncertainty decay export failed: {e}", level="warning", verbose=verbose)
    
        return rows
    
    
    # -------------------------
    # Private helpers (PATCH 6)
    # -------------------------
    
    def _uq_get_mean_sigma_by_basis(self, iblock, basis="auto"):
        """
        Resolve (mean, sigma) arrays from an IMT block according to basis.
    
        basis:
          - "auto": hierarchical -> posterior -> prior
          - "hierarchical": mean_hier/sigma_hier
          - "posterior": mean_post/sigma_post
          - "prior": mean/sigma
        """
        b = str(basis).lower()
        if b == "auto":
            if iblock.get("mean_hier") is not None and iblock.get("sigma_hier") is not None:
                return iblock["mean_hier"], iblock["sigma_hier"]
            if iblock.get("mean_post") is not None and iblock.get("sigma_post") is not None:
                return iblock["mean_post"], iblock["sigma_post"]
            return iblock.get("mean"), iblock.get("sigma") if iblock.get("sigma") is not None else (iblock.get("mean"), None)
    
        if b == "hierarchical":
            return iblock.get("mean_hier"), iblock.get("sigma_hier")
        if b == "posterior":
            return iblock.get("mean_post"), iblock.get("sigma_post")
        if b == "prior":
            return iblock.get("mean"), iblock.get("sigma")
        raise ValueError(f"_uq_get_mean_sigma_by_basis: unsupported basis '{basis}'.")
    
    
    def _uq_poe_normal(self, mean, sigma, threshold):
        """Analytical PoE for Normal distribution."""
        import numpy as np
        z = (float(threshold) - mean) / np.maximum(sigma, 1e-12)
        # use erf approximation (no scipy needed)
        return 1.0 - 0.5 * (1.0 + self._uq_erf(z / np.sqrt(2.0)))
    
    
    def _uq_erf(self, x):
        """
        Vectorized erf approximation (Abramowitz & Stegun 7.1.26).
        Works without scipy.
        """
        import numpy as np
        x = np.asarray(x, dtype=float)
        sign = np.sign(x)
        ax = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
        return sign * y
    
    
    def _uq_norm_ppf(self, p):
        """
        Approximate inverse CDF (ppf) for standard normal without scipy.
        Uses Peter John Acklam approximation.
        """
        import numpy as np
        p = float(p)
        if p <= 0.0:
            return -np.inf
        if p >= 1.0:
            return np.inf
    
        # Coefficients in rational approximations
        a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
              1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
              6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
        d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
              3.754408661907416e+00]
    
        plow = 0.02425
        phigh = 1 - plow
    
        if p < plow:
            q = np.sqrt(-2 * np.log(p))
            return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        if phigh < p:
            q = np.sqrt(-2 * np.log(1 - p))
            return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                     ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    
    
    def _uq_ci_from_samples_file(self, samples_path, quantiles, n_points):
        """
        Compute CI maps from a stored float32 memmap sample file produced in Patch 4.
    
        samples file layout: (n_samples, n_points) float32
        """
        import numpy as np
        # infer n_samples from file size
        itemsize = np.dtype("float32").itemsize
        n_items = int(np.floor((__import__("os").path.getsize(samples_path)) / itemsize))
        n_samples = int(n_items // int(n_points))
        mm = np.memmap(samples_path, mode="r", dtype="float32", shape=(n_samples, int(n_points)))
        qs = [float(q) for q in quantiles]
        out = {q: np.full(int(n_points), np.nan, dtype=float) for q in qs}
        pchunk = 50000
        for i0 in range(0, int(n_points), pchunk):
            i1 = min(int(n_points), i0 + pchunk)
            block = np.asarray(mm[:, i0:i1], dtype=float)
            block.sort(axis=0)
            for q in qs:
                idx = int(round((n_samples - 1) * q))
                idx = max(0, min(n_samples - 1, idx))
                out[q][i0:i1] = block[idx, :]
        return out
    
    

    # =========================
    # PATCH 7 — Global metrics + uncertainty bands (area≥k, optional exposure if available)
    # ADD INSIDE SHAKEtime CLASS
    # =========================
    
    def uq_compute_metrics(
        self,
        dataset=None,
        version_list=None,
        imts=None,
        thresholds=None,
        basis="auto",
        prefer_mc=True,
        n_mc_if_missing=0,
        mc_kwargs=None,
        metrics=("area_ge",),
        include_exposure_if_available=True,
        exposure_imt="MMI",
        exposure_threshold=4.0,
        output_path="./export",
        export=True,
        strict=False,
        verbose=True,
    ):
        """
        Compute global metrics with uncertainty bands across versions.
    
        Implemented metrics (Patch 7):
          - "area_ge": area where IMT >= k (for each threshold k)
    
        Optional (use-if-present, does NOT create dependencies):
          - exposure metrics (population / exposure) if this SHAKEtime instance already has
            data layers or callable exposure functions available.
            Because your codebase may store exposure in different ways, this patch:
              * detects common attributes and callables
              * otherwise skips exposure with a clear log message
    
        Uncertainty bands
        -----------------
        Preferred source: Monte Carlo distributions (Patch 4) if available and prefer_mc=True.
          - If MC metrics distributions exist in iblock["mc"]["metrics"]["area_ge"][k], we use them.
          - Else if MC samples exist, we can compute from samples file if present.
          - Else if n_mc_if_missing > 0, we will run uq_monte_carlo() internally (self-contained) to generate metrics.
        Fallback: Analytical Normal approximation (no sampling):
          - Uses PoE at each gridpoint and approximates distribution of area>=k by:
                E[A] = sum(area_i * p_i)
                Var[A] ≈ sum(area_i^2 * p_i*(1-p_i))   (independent Bernoulli assumption)
            Then reports mean ± z*sd (approx), plus median≈mean.
          - This is fast and often good enough for trend curves, but MC is better.
    
        Outputs
        -------
        Adds dataset["metrics"]["area_ge"][imt][threshold] = list of per-version rows with bands.
        Also stores per-version results inside each IMT block under:
          iblock["metrics"]["area_ge"][threshold] = {"mean":..., "p05":..., "p50":..., "p95":..., "method":...}
    
        Export
        ------
        Writes CSVs under: output_path/SHAKEtime/<event_id>/uq/metrics
    
        Parameters
        ----------
        thresholds : dict or list or None
            If dict: {IMT:[k1,k2,...]} (PSA* can use key "PSA")
            If list: applied to all IMTs
            If None: uses defaults.
        basis : str
            "auto" (default) chooses hierarchical->posterior->prior for analytical method.
            For MC it uses whatever MC was generated from.
        prefer_mc : bool
            Prefer MC-based uncertainty bands if available.
        n_mc_if_missing : int
            If >0 and MC metrics missing, run uq_monte_carlo with this many samples (metrics-only).
        mc_kwargs : dict or None
            Extra kwargs forwarded to uq_monte_carlo when invoked internally.
        include_exposure_if_available : bool
            Attempt exposure computations (use-if-present).
        exposure_imt, exposure_threshold : used for exposure hooks (if available).
        """
        import os
        import numpy as np
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_compute_metrics: dataset is None and self.uq_data not found. Run uq_build_dataset first.")
    
        event_id = ds.get("event_id", self._uq_get_event_id())
        uq_dirs = ds.get("meta", {}).get("uq_export_dirs", None)
        if uq_dirs is None:
            uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        thresholds = thresholds if thresholds is not None else self._uq_default_thresholds()
        mc_kwargs = mc_kwargs or {}
    
        # versions
        vkeys = self._uq_sort_version_keys(list(ds.get("versions", {}).keys()))
        if version_list is not None:
            vset = set([str(v) for v in version_list])
            vkeys = [k for k in vkeys if k in vset]
    
        # decide IMTs list based on dataset
        imt_set = set()
        for vk in vkeys:
            vb = ds["versions"].get(vk, {})
            if vb and "imts" in vb:
                imt_set.update(vb["imts"].keys())
        imt_list = sorted(list(imt_set))
        if imts is not None:
            req = set([str(x) for x in imts])
            imt_list = [k for k in imt_list if k in req]
    
        ds.setdefault("metrics", {})
        ds["metrics"].setdefault("area_ge", {})
        ds["metrics"].setdefault("exposure_ge", {})
    
        # If we need MC but missing, optionally run MC once per IMT/version set (metrics-only)
        # We will run MC only if n_mc_if_missing>0 and prefer_mc=True.
        if prefer_mc and int(n_mc_if_missing) > 0:
            # determine if MC metrics exist for any IMT; if not, run MC in metrics-only mode
            need_mc = False
            for vk in vkeys:
                vb = ds["versions"].get(vk, {})
                if not vb or "imts" not in vb:
                    continue
                for imt in imt_list:
                    if imt not in vb["imts"]:
                        continue
                    ib = vb["imts"][imt]
                    if "mc" not in ib or "metrics" not in (ib.get("mc") or {}) or "area_ge" not in ib["mc"]["metrics"]:
                        need_mc = True
                        break
                if need_mc:
                    break
    
            if need_mc:
                self._uq_log(
                    f"MC metrics missing; running uq_monte_carlo(metrics-only) with n_samples={int(n_mc_if_missing)}",
                    level="info",
                    verbose=verbose,
                )
                try:
                    self.uq_monte_carlo(
                        dataset=ds,
                        version_list=vkeys,
                        imts=imt_list,
                        n_samples=int(n_mc_if_missing),
                        compute_poe=False,
                        compute_ci=False,
                        compute_metrics=True,
                        metrics_thresholds=thresholds,
                        store_samples=bool(mc_kwargs.get("store_samples", False)),
                        correlation=mc_kwargs.get("correlation", "independent"),
                        corr_length_km=mc_kwargs.get("corr_length_km", 40.0),
                        random_seed=mc_kwargs.get("random_seed", 42),
                        output_path=output_path,
                        export_summary=True,
                        strict=strict,
                        verbose=verbose,
                    )
                except Exception as e:
                    if strict:
                        raise
                    self._uq_log(f"Internal MC generation for metrics failed: {e}", level="warning", verbose=verbose)
    
        # helper for quantiles
        def qstats(vals):
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return {"mean": np.nan, "p05": np.nan, "p50": np.nan, "p95": np.nan}
            return {
                "mean": float(np.mean(vals)),
                "p05": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
            }
    
        rows_export = []
    
        # -------- area_ge metric --------
        for imt in imt_list:
            ths = self._uq_thresholds_for_imt(thresholds, imt)
            if not ths:
                continue
            ds["metrics"]["area_ge"].setdefault(imt, {})
    
            for thr in ths:
                thr = float(thr)
                ds["metrics"]["area_ge"][imt].setdefault(thr, [])
    
                for vk in vkeys:
                    vb = ds["versions"].get(vk, {})
                    if not vb or "imts" not in vb or imt not in vb["imts"]:
                        continue
                    ib = vb["imts"][imt]
                    lon = np.asarray(ib.get("lon", []), dtype=float)
                    lat = np.asarray(ib.get("lat", []), dtype=float)
                    if lon.size == 0:
                        continue
    
                    cell_area = self._uq_estimate_cell_area_km2(lon, lat)
    
                    method_used = None
                    stats = None
    
                    # 1) MC-based if available
                    if prefer_mc and ("mc" in ib) and ("metrics" in (ib.get("mc") or {})) and ("area_ge" in ib["mc"]["metrics"]):
                        dist = ib["mc"]["metrics"]["area_ge"].get(thr, None)
                        if dist is None:
                            # allow string keys
                            dist = ib["mc"]["metrics"]["area_ge"].get(str(thr), None)
                        if dist is not None:
                            stats = qstats(dist)
                            method_used = "mc_metric_dist"
    
                    # 2) If MC samples file exists but no metric dist, compute quickly
                    if stats is None and prefer_mc and ("mc" in ib) and ("samples_path" in (ib.get("mc") or {})):
                        try:
                            dist = self._uq_area_ge_from_samples_file(
                                ib["mc"]["samples_path"], thr, lon.size, cell_area
                            )
                            stats = qstats(dist)
                            method_used = "mc_samples"
                            # cache for reuse
                            ib.setdefault("mc", {}).setdefault("metrics", {}).setdefault("area_ge", {})[thr] = dist
                        except Exception as e:
                            if strict:
                                raise
                            self._uq_log(f"[v{vk}][{imt}] area_ge from samples failed: {e}", level="debug", verbose=verbose)
    
                    # 3) Analytical fallback
                    if stats is None:
                        mean, sig = self._uq_get_mean_sigma_by_basis(ib, basis=basis)
                        mean = np.asarray(mean, dtype=float)
                        sig = np.maximum(np.asarray(sig if sig is not None else np.full_like(mean, self._uq_default_obs_sigma(imt)), dtype=float), 1e-12)
                        p = self._uq_poe_normal(mean, sig, thr)
                        EA = float(np.nansum(cell_area * p))
                        # Var under independence
                        VarA = float(np.nansum((cell_area * cell_area) * p * (1.0 - p)))
                        sdA = np.sqrt(max(VarA, 0.0))
                        # approximate 90% interval using normal
                        stats = {
                            "mean": EA,
                            "p50": EA,
                            "p05": EA - 1.6448536269514722 * sdA,
                            "p95": EA + 1.6448536269514722 * sdA,
                        }
                        method_used = "analytical_indep"
    
                    # Store per-version result
                    ib.setdefault("metrics", {}).setdefault("area_ge", {})[thr] = {
                        **stats,
                        "method": method_used,
                        "threshold": thr,
                        "units": "km2" if method_used else "",
                    }
    
                    row = {
                        "event_id": event_id,
                        "version": vk,
                        "imt": imt,
                        "metric": "area_ge",
                        "threshold": thr,
                        "method": method_used,
                        "mean": stats["mean"],
                        "p05": stats["p05"],
                        "p50": stats["p50"],
                        "p95": stats["p95"],
                    }
                    ds["metrics"]["area_ge"][imt][thr].append(row)
                    rows_export.append(row)
    
        # -------- exposure metric (optional, best-effort) --------
        if include_exposure_if_available:
            try:
                exp_rows = self._uq_try_compute_exposure_bands(
                    ds,
                    vkeys=vkeys,
                    basis=basis,
                    prefer_mc=prefer_mc,
                    imt=str(exposure_imt),
                    threshold=float(exposure_threshold),
                    verbose=verbose,
                )
                if exp_rows:
                    ds["metrics"]["exposure_ge"].setdefault(str(exposure_imt), {}).setdefault(float(exposure_threshold), [])
                    ds["metrics"]["exposure_ge"][str(exposure_imt)][float(exposure_threshold)].extend(exp_rows)
                    rows_export.extend(exp_rows)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"Exposure metrics skipped/failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        # Export combined metrics
        if export and rows_export:
            try:
                csv_path = os.path.join(uq_dirs["metrics"], f"uq_metrics_summary_{event_id}.csv")
                self._uq_write_dictlist_csv(csv_path, rows_export)
                self._uq_log(f"UQ metrics summary exported: {csv_path}", level="info", verbose=verbose)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"Metrics export failed: {e}", level="warning", verbose=verbose)
    
            # Also export one CSV per (imt,threshold) for area_ge for convenience
            try:
                for imt, d in ds.get("metrics", {}).get("area_ge", {}).items():
                    for thr, rows in d.items():
                        if not rows:
                            continue
                        fn = f"uq_metric_area_ge_{event_id}_{imt}_{str(thr).replace('.','p')}.csv"
                        self._uq_write_dictlist_csv(os.path.join(uq_dirs["metrics"], fn), rows)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"Per-metric exports failed (non-fatal): {e}", level="debug", verbose=verbose)
    
        if dataset is None:
            self.uq_data = ds
        return ds
    
    
    # -------------------------
    # Private helpers (PATCH 7)
    # -------------------------
    
    def _uq_area_ge_from_samples_file(self, samples_path, threshold, n_points, cell_area):
        """
        Compute area>=threshold distribution from a float32 memmap samples file.
    
        samples layout: (n_samples, n_points)
        returns array length n_samples with area in km^2.
        """
        import numpy as np
        import os
    
        threshold = float(threshold)
        n_points = int(n_points)
        cell_area = np.asarray(cell_area, dtype=float).ravel()
    
        itemsize = np.dtype("float32").itemsize
        n_items = int(np.floor(os.path.getsize(samples_path) / itemsize))
        n_samples = int(n_items // n_points)
    
        mm = np.memmap(samples_path, mode="r", dtype="float32", shape=(n_samples, n_points))
        out = np.zeros(n_samples, dtype=float)
    
        # compute in chunks of samples to reduce memory
        s_chunk = 50
        for s0 in range(0, n_samples, s_chunk):
            s1 = min(n_samples, s0 + s_chunk)
            block = np.asarray(mm[s0:s1, :], dtype=float)
            mask = block >= threshold
            out[s0:s1] = np.nansum(mask * cell_area[None, :], axis=1)
        return out
    
    
    def _uq_try_compute_exposure_bands(
        self,
        dataset,
        vkeys,
        basis="auto",
        prefer_mc=True,
        imt="MMI",
        threshold=4.0,
        verbose=True,
    ):
        """
        Best-effort exposure computation without imposing dependencies.
    
        This tries several common patterns that may exist in your SHAKEtime ecosystem:
          1) self.population_grid or self.pop_grid arrays aligned to shakemap grid
          2) self.exposure_grid / self.exposure arrays
          3) callable: self.get_population_exposure(lon,lat,imt_values) or similar
    
        Metric computed:
          exposure_ge: sum(pop_i * I(IMT_i >= threshold))  (or if pop already in people per cell)
    
        Uncertainty bands:
          - if MC samples exist, compute distribution
          - else analytical using PoE: E[Exp] = sum(pop_i * p_i); Var ≈ sum(pop_i^2 * p_i(1-p_i))
    
        Returns list of metric rows (dicts) consistent with uq_compute_metrics export format.
        """
        import numpy as np
    
        ds = dataset
        event_id = ds.get("event_id", self._uq_get_event_id())
        imt = str(imt)
        threshold = float(threshold)
    
        # Detect population/exposure grid
        pop = None
    
        for attr in ("population_grid", "pop_grid", "population", "exposure_grid", "exposure"):
            if hasattr(self, attr):
                pop = getattr(self, attr)
                break
    
        # If pop is callable, treat differently
        pop_callable = None
        if callable(pop):
            pop_callable = pop
            pop = None
    
        if pop is None and pop_callable is None:
            # Try a method
            for fn in ("get_population_grid", "get_exposure_grid", "population", "exposure"):
                if hasattr(self, fn) and callable(getattr(self, fn)):
                    pop_callable = getattr(self, fn)
                    break
    
        if pop is None and pop_callable is None:
            self._uq_log("Exposure not available (no population/exposure grid or callable found).", level="debug", verbose=verbose)
            return []
    
        rows = []
    
        for vk in vkeys:
            vb = ds["versions"].get(vk, {})
            if not vb or "imts" not in vb or imt not in vb["imts"]:
                continue
            ib = vb["imts"][imt]
            lon = np.asarray(ib.get("lon", []), dtype=float)
            lat = np.asarray(ib.get("lat", []), dtype=float)
            if lon.size == 0:
                continue
    
            # Acquire population aligned to grid points:
            # - if pop is array length N, use directly
            # - if pop is 2D or otherwise, attempt nearest mapping (requires lon/lat for pop, not assumed here)
            pop_vec = None
            if pop is not None:
                try:
                    pop_arr = np.asarray(pop, dtype=float)
                    if pop_arr.ndim == 1 and pop_arr.size == lon.size:
                        pop_vec = pop_arr
                except Exception:
                    pop_vec = None
    
            if pop_vec is None and pop_callable is not None:
                # Try calling with (lon,lat) signature
                try:
                    pv = pop_callable(lon, lat)
                    pv = np.asarray(pv, dtype=float).ravel()
                    if pv.size == lon.size:
                        pop_vec = pv
                except Exception:
                    pop_vec = None
    
            if pop_vec is None:
                self._uq_log(f"[v{vk}] Exposure skipped (pop not aligned/mappable).", level="debug", verbose=verbose)
                continue
    
            # MC if possible
            method_used = None
            stats = None
    
            if prefer_mc and ("mc" in ib) and ("samples_path" in (ib.get("mc") or {})):
                try:
                    dist = self._uq_exposure_ge_from_samples_file(
                        ib["mc"]["samples_path"], threshold, lon.size, pop_vec
                    )
                    stats = {
                        "mean": float(np.mean(dist)),
                        "p05": float(np.percentile(dist, 5)),
                        "p50": float(np.percentile(dist, 50)),
                        "p95": float(np.percentile(dist, 95)),
                    }
                    method_used = "mc_samples"
                except Exception:
                    stats = None
    
            if stats is None:
                mean, sig = self._uq_get_mean_sigma_by_basis(ib, basis=basis)
                mean = np.asarray(mean, dtype=float)
                sig = np.maximum(np.asarray(sig if sig is not None else np.full_like(mean, self._uq_default_obs_sigma(imt)), dtype=float), 1e-12)
                p = self._uq_poe_normal(mean, sig, threshold)
                E = float(np.nansum(pop_vec * p))
                Var = float(np.nansum((pop_vec * pop_vec) * p * (1.0 - p)))
                sd = np.sqrt(max(Var, 0.0))
                stats = {
                    "mean": E,
                    "p50": E,
                    "p05": E - 1.6448536269514722 * sd,
                    "p95": E + 1.6448536269514722 * sd,
                }
                method_used = "analytical_indep"
    
            # store in iblock
            ib.setdefault("metrics", {}).setdefault("exposure_ge", {})[threshold] = {
                **stats,
                "method": method_used,
                "threshold": threshold,
                "units": "people"  # assumed; user can reinterpret
            }
    
            rows.append({
                "event_id": event_id,
                "version": vk,
                "imt": imt,
                "metric": "exposure_ge",
                "threshold": threshold,
                "method": method_used,
                "mean": stats["mean"],
                "p05": stats["p05"],
                "p50": stats["p50"],
                "p95": stats["p95"],
            })
    
        return rows
    
    
    def _uq_exposure_ge_from_samples_file(self, samples_path, threshold, n_points, pop_vec):
        """
        Compute exposure>=threshold distribution from a float32 memmap samples file.
        exposure = sum(pop_i * I(x_i >= threshold))
        """
        import numpy as np
        import os
    
        threshold = float(threshold)
        n_points = int(n_points)
        pop_vec = np.asarray(pop_vec, dtype=float).ravel()
    
        itemsize = np.dtype("float32").itemsize
        n_items = int(np.floor(os.path.getsize(samples_path) / itemsize))
        n_samples = int(n_items // n_points)
    
        mm = np.memmap(samples_path, mode="r", dtype="float32", shape=(n_samples, n_points))
        out = np.zeros(n_samples, dtype=float)
    
        s_chunk = 50
        for s0 in range(0, n_samples, s_chunk):
            s1 = min(n_samples, s0 + s_chunk)
            block = np.asarray(mm[s0:s1, :], dtype=float)
            mask = block >= threshold
            out[s0:s1] = np.nansum(mask * pop_vec[None, :], axis=1)
        return out
    
        



    # =========================
    # PATCH 8 — Plotting + export helpers in SHAKEtime style (metrics/decay/PoE quicklook)
    # ADD INSIDE SHAKEtime CLASS
    # =========================
    
    def uq_plot_uncertainty_decay(
        self,
        dataset=None,
        imt="MMI",
        basis="auto",
        summary_stat="median",
        figsize=(10, 4),
        font_sizes=None,
        xrotation=0,
        line_width=2.0,
        marker_size=5.0,
        legend_loc="best",
        grid_kwargs=None,
        show_title=True,
        dpi=200,
        save_formats=("png",),
        output_path="./export",
        close_figs=True,
        verbose=True,
    ):
        """
        Plot uncertainty decay vs version for a given IMT.
    
        Uses uq_uncertainty_decay() (Patch 6) to compute the curve.
        Saves to: output_path/SHAKEtime/<event_id>/uq/figures
    
        Returns (fig, ax).
        """
        import os
        import matplotlib.pyplot as plt
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_plot_uncertainty_decay: dataset is None and self.uq_data not found.")
    
        event_id = ds.get("event_id", self._uq_get_event_id())
        uq_dirs = ds.get("meta", {}).get("uq_export_dirs", None)
        if uq_dirs is None:
            uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        rows = self.uq_uncertainty_decay(
            dataset=ds, imt=imt, basis=basis, summary_stat=summary_stat,
            output_path=output_path, export=False, verbose=verbose
        )
    
        versions = [r["version"] for r in rows]
        values = [r["sigma_summary"] for r in rows]
    
        font_sizes = font_sizes or {"title": 12, "label": 11, "tick": 10, "legend": 10}
        grid_kwargs = grid_kwargs or {"alpha": 0.3}
    
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(range(len(versions)), values, linewidth=line_width, marker="o", markersize=marker_size, label=f"{imt} σ")
    
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels(versions, rotation=xrotation, fontsize=font_sizes["tick"])
        ax.tick_params(axis="y", labelsize=font_sizes["tick"])
        ax.set_ylabel("Sigma summary", fontsize=font_sizes["label"])
        ax.set_xlabel("Version", fontsize=font_sizes["label"])
        ax.grid(True, **grid_kwargs)
    
        if show_title:
            ax.set_title(f"Uncertainty decay — {event_id} — {imt} ({basis})", fontsize=font_sizes["title"])
    
        ax.legend(loc=legend_loc, fontsize=font_sizes["legend"])
    
        # Save
        base = os.path.join(uq_dirs["figures"], f"uq_decay_{event_id}_{imt}_{basis}")
        for ext in save_formats or ():
            fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=dpi)
        self._uq_log(f"Saved decay figure(s): {base}.*", level="info", verbose=verbose)
    
        if close_figs:
            plt.close(fig)
        return fig, ax
    
    
    def uq_plot_metric_timeseries(
        self,
        dataset=None,
        metric="area_ge",
        imt="MMI",
        thresholds=None,
        figsize=(10, 4),
        font_sizes=None,
        xrotation=0,
        line_width=2.0,
        marker_size=4.0,
        legend_loc="best",
        grid_kwargs=None,
        show_title=True,
        dpi=200,
        save_formats=("png",),
        output_path="./export",
        close_figs=True,
        verbose=True,
    ):
        """
        Plot metric timeseries with uncertainty bands (p05/p50/p95) across versions.
    
        Requires uq_compute_metrics (Patch 7) to have been run, OR it will attempt to read from dataset.
    
        Saves to: .../uq/figures
    
        Returns (fig, ax).
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_plot_metric_timeseries: dataset is None and self.uq_data not found.")
    
        event_id = ds.get("event_id", self._uq_get_event_id())
        uq_dirs = ds.get("meta", {}).get("uq_export_dirs", None)
        if uq_dirs is None:
            uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        metric = str(metric)
        imt = str(imt)
        font_sizes = font_sizes or {"title": 12, "label": 11, "tick": 10, "legend": 10}
        grid_kwargs = grid_kwargs or {"alpha": 0.3}
    
        # Resolve thresholds
        if thresholds is None:
            thresholds = self._uq_thresholds_for_imt(self._uq_default_thresholds(), imt)
        thresholds = [float(t) for t in thresholds]
    
        # Pull data
        mroot = ds.get("metrics", {}).get(metric, {})
        if metric == "area_ge":
            series_dict = mroot.get(imt, {})
        elif metric == "exposure_ge":
            series_dict = ds.get("metrics", {}).get("exposure_ge", {}).get(imt, {})
        else:
            series_dict = {}
    
        if not series_dict:
            self._uq_log(
                f"Metric '{metric}' not found for IMT '{imt}'. Run uq_compute_metrics first.",
                level="warning",
                verbose=verbose,
            )
            # still produce an empty plot
            series_dict = {}
    
        vkeys = self._uq_sort_version_keys(list(ds.get("versions", {}).keys()))
    
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
        for thr in thresholds:
            rows = series_dict.get(thr, series_dict.get(str(thr), []))
            if not rows:
                continue
    
            # build lookup by version
            byv = {r["version"]: r for r in rows if "version" in r}
            x = []
            p05 = []
            p50 = []
            p95 = []
            for i, vk in enumerate(vkeys):
                if vk not in byv:
                    continue
                r = byv[vk]
                x.append(i)
                p05.append(r.get("p05", np.nan))
                p50.append(r.get("p50", np.nan))
                p95.append(r.get("p95", np.nan))
    
            if not x:
                continue
    
            ax.plot(x, p50, linewidth=line_width, marker="o", markersize=marker_size, label=f"{metric} ≥ {thr}")
            ax.fill_between(x, p05, p95, alpha=0.2)
    
        ax.set_xticks(range(len(vkeys)))
        ax.set_xticklabels(vkeys, rotation=xrotation, fontsize=font_sizes["tick"])
        ax.tick_params(axis="y", labelsize=font_sizes["tick"])
    
        ylabel = "Area (km²)" if metric == "area_ge" else "Exposure (people)"
        ax.set_ylabel(ylabel, fontsize=font_sizes["label"])
        ax.set_xlabel("Version", fontsize=font_sizes["label"])
        ax.grid(True, **grid_kwargs)
    
        if show_title:
            ax.set_title(f"Metric timeseries — {event_id} — {metric} — {imt}", fontsize=font_sizes["title"])
    
        ax.legend(loc=legend_loc, fontsize=font_sizes["legend"])
    
        base = os.path.join(uq_dirs["figures"], f"uq_metric_{metric}_{event_id}_{imt}")
        for ext in save_formats or ():
            fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=dpi)
        self._uq_log(f"Saved metric figure(s): {base}.*", level="info", verbose=verbose)
    
        if close_figs:
            plt.close(fig)
        return fig, ax
    
    
    def uq_plot_poe_quicklook(
        self,
        dataset=None,
        version=None,
        imt="MMI",
        threshold=5.0,
        basis="auto",
        figsize=(6, 5),
        font_sizes=None,
        show_title=True,
        dpi=200,
        save_formats=("png",),
        output_path="./export",
        close_figs=True,
        verbose=True,
    ):
        """
        Quicklook scatter plot of PoE over map points (lon/lat colored by PoE).
        This is intentionally lightweight (no cartopy dependency).
    
        If MC PoE exists and basis='mc' or basis='auto' with mc available, uses that.
        Otherwise uses analytical PoE.
    
        Saves to: .../uq/figures
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_plot_poe_quicklook: dataset is None and self.uq_data not found.")
    
        event_id = ds.get("event_id", self._uq_get_event_id())
        uq_dirs = ds.get("meta", {}).get("uq_export_dirs", None)
        if uq_dirs is None:
            uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        imt = str(imt)
        thr = float(threshold)
    
        # pick version
        vkeys = self._uq_sort_version_keys(list(ds.get("versions", {}).keys()))
        if version is None:
            version = vkeys[-1] if vkeys else None
        vkey = str(version)
    
        vb = ds.get("versions", {}).get(vkey, {})
        if not vb or "imts" not in vb or imt not in vb["imts"]:
            raise ValueError(f"uq_plot_poe_quicklook: version {vkey} / IMT {imt} not found in dataset.")
    
        ib = vb["imts"][imt]
        lon = np.asarray(ib.get("lon", []), dtype=float)
        lat = np.asarray(ib.get("lat", []), dtype=float)
        if lon.size == 0:
            raise ValueError("uq_plot_poe_quicklook: grid empty.")
    
        # PoE array
        poe = None
        if str(basis).lower() == "mc" or (str(basis).lower() == "auto" and "mc" in ib and "poe" in ib["mc"]):
            poe = ib.get("mc", {}).get("poe", {}).get(thr, None)
            if poe is None:
                poe = ib.get("mc", {}).get("poe", {}).get(str(thr), None)
    
        if poe is None:
            mean, sig = self._uq_get_mean_sigma_by_basis(ib, basis=basis)
            mean = np.asarray(mean, dtype=float)
            sig = np.maximum(np.asarray(sig if sig is not None else np.full_like(mean, self._uq_default_obs_sigma(imt)), dtype=float), 1e-12)
            poe = self._uq_poe_normal(mean, sig, thr)
    
        poe = np.asarray(poe, dtype=float)
    
        font_sizes = font_sizes or {"title": 12, "label": 11, "tick": 10}
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sc = ax.scatter(lon, lat, c=poe, s=6, linewidths=0.0)
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label("PoE", fontsize=font_sizes["label"])
        ax.set_xlabel("Longitude", fontsize=font_sizes["label"])
        ax.set_ylabel("Latitude", fontsize=font_sizes["label"])
        ax.tick_params(labelsize=font_sizes["tick"])
    
        if show_title:
            ax.set_title(f"PoE quicklook — {event_id} — v{vkey} — {imt} ≥ {thr}", fontsize=font_sizes["title"])
    
        base = os.path.join(uq_dirs["figures"], f"uq_poe_{event_id}_v{vkey}_{imt}_{str(thr).replace('.','p')}")
        for ext in save_formats or ():
            fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=dpi)
        self._uq_log(f"Saved PoE quicklook figure(s): {base}.*", level="info", verbose=verbose)
    
        if close_figs:
            plt.close(fig)
        return fig, ax
    
    
    def uq_export_uq_dataset(
        self,
        dataset=None,
        output_path="./export",
        formats=("npz", "json"),
        slim=True,
        strict=False,
        verbose=True,
    ):
        """
        Export the UQ dataset (mean/sigma/vs30 + posterior/hier/metrics summaries) to disk.
    
        formats:
          - "npz": compressed pickled dict
          - "json": slim JSON (only metadata + per-version summaries; does NOT dump full arrays unless slim=False)
    
        slim=True strongly recommended for json to avoid huge files.
    
        Exports into: .../uq/data
        """
        import os
        import json
        import numpy as np
    
        ds = dataset if dataset is not None else getattr(self, "uq_data", None)
        if ds is None:
            raise ValueError("uq_export_uq_dataset: dataset is None and self.uq_data not found.")
    
        event_id = ds.get("event_id", self._uq_get_event_id())
        uq_dirs = ds.get("meta", {}).get("uq_export_dirs", None)
        if uq_dirs is None:
            uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        exported = []
    
        if "npz" in formats:
            try:
                path = os.path.join(uq_dirs["data"], f"uq_dataset_{event_id}.npz")
                self._uq_write_npz_dict(path, ds)
                exported.append(path)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"NPZ export failed: {e}", level="warning", verbose=verbose)
    
        if "json" in formats:
            try:
                path = os.path.join(uq_dirs["data"], f"uq_dataset_{event_id}.json")
                if slim:
                    js = self._uq_slim_dataset_for_json(ds)
                else:
                    # Warning: could be huge; convert arrays to lists
                    js = self._uq_full_dataset_to_jsonable(ds)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(js, f, indent=2)
                exported.append(path)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"JSON export failed: {e}", level="warning", verbose=verbose)
    
        self._uq_log(f"UQ dataset exported: {exported}", level="info", verbose=verbose)
        return exported
    
    
    # -------------------------
    # Private helpers (PATCH 8)
    # -------------------------
    
    def _uq_slim_dataset_for_json(self, ds):
        """Create a slim JSON-serializable summary (no full arrays)."""
        out = {
            "event_id": ds.get("event_id"),
            "shakemap_folder": ds.get("shakemap_folder"),
            "imts_requested": ds.get("imts_requested"),
            "vs30": {
                "attached": ds.get("vs30", {}).get("attached"),
                "source": ds.get("vs30", {}).get("source"),
                "note": ds.get("vs30", {}).get("note"),
            },
            "meta": ds.get("meta", {}),
            "versions": {},
            "metrics": ds.get("metrics", {}),
            "hierarchical": ds.get("hierarchical", {}),
        }
    
        for vkey, vb in (ds.get("versions", {}) or {}).items():
            if not isinstance(vb, dict):
                continue
            vout = {
                "version": vb.get("version"),
                "paths": vb.get("paths", {}),
                "grid_meta": vb.get("grid_meta", {}),
                "unc_meta": vb.get("unc_meta", {}),
                "n_stations": len(vb.get("stations", []) or []),
                "n_dyfi": len(vb.get("dyfi", []) or []),
                "imts": {},
            }
            for imt, ib in (vb.get("imts", {}) or {}).items():
                # summary stats only
                def sstats(arr):
                    import numpy as np
                    if arr is None:
                        return None
                    a = np.asarray(arr, dtype=float)
                    return {
                        "n": int(a.size),
                        "median": float(np.nanmedian(a)) if a.size else None,
                        "mean": float(np.nanmean(a)) if a.size else None,
                    }
                vout["imts"][imt] = {
                    "mean": sstats(ib.get("mean")),
                    "sigma": sstats(ib.get("sigma")),
                    "mean_post": sstats(ib.get("mean_post")),
                    "sigma_post": sstats(ib.get("sigma_post")),
                    "mean_hier": sstats(ib.get("mean_hier")),
                    "sigma_hier": sstats(ib.get("sigma_hier")),
                    "has_vs30": ib.get("vs30") is not None,
                    "update_meta": ib.get("update_meta", None),
                    "hier_meta": ib.get("hier_meta", None),
                    "mc_meta": {
                        "basis": (ib.get("mc") or {}).get("basis"),
                        "n_samples": (ib.get("mc") or {}).get("n_samples"),
                        "correlation": (ib.get("mc") or {}).get("correlation"),
                        "corr_length_km": (ib.get("mc") or {}).get("corr_length_km"),
                        "has_samples_path": "samples_path" in (ib.get("mc") or {}),
                    } if "mc" in ib else None,
                }
            out["versions"][vkey] = vout
        return out
    
    
    def _uq_full_dataset_to_jsonable(self, ds):
        """Convert full dataset to JSONable form by converting numpy arrays to lists (can be HUGE)."""
        import numpy as np
    
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            if isinstance(obj, tuple):
                return [convert(x) for x in obj]
            return obj
    
        return convert(ds)
    


    # =========================
    # PATCH 9 — Convenience orchestration: one-call UQ pipeline runner
    # ADD INSIDE SHAKEtime CLASS
    # =========================
    
    def uq_pipeline(
        self,
        # --- data build ---
        version_list=None,
        imts=None,
        shakemap_folder=None,
        output_path="./export",
        load_vs30=True,
        vs30_path=None,
        attach_vs30_kwargs=None,
        # --- UQ mode ---
        mode="bayes_update",
        mode_kwargs=None,
        # --- products ---
        compute_poe=True,
        poe_imts=None,
        poe_thresholds=None,
        poe_basis="auto",
        compute_ci=True,
        ci_imts=None,
        ci_quantiles=(0.05, 0.5, 0.95),
        ci_basis="auto",
        ci_method="auto",
        # --- metrics ---
        compute_metrics=True,
        metrics_imts=None,
        metrics_thresholds=None,
        metrics_basis="auto",
        prefer_mc_metrics=True,
        n_mc_if_missing=0,
        mc_kwargs=None,
        include_exposure_if_available=True,
        exposure_imt="MMI",
        exposure_threshold=4.0,
        # --- plotting ---
        make_plots=True,
        plot_decay=True,
        plot_metrics=True,
        plot_poe_quicklook=False,
        poe_quicklook_version=None,
        poe_quicklook_imt="MMI",
        poe_quicklook_threshold=5.0,
        # --- export ---
        export_dataset=True,
        export_formats=("npz", "json"),
        export_dataset_slim_json=True,
        # --- behavior ---
        strict=False,
        verbose=True,
    ):
        """
        One-call orchestration wrapper for the full UQ pipeline.
    
        This is a convenience function for notebooks/scripts to run:
          1) uq_build_dataset(...)
          2) uq_attach_vs30(...)  (optional)
          3) run UQ mode:
               - bayes_update (Patch 3)
               - monte_carlo  (Patch 4)  [if selected]
               - hierarchical (Patch 5)
          4) compute products:
               - PoE (Patch 6)
               - Credible intervals (Patch 6)
          5) compute metrics + bands:
               - area>=k (+ optional exposure if available) (Patch 7)
          6) plots (Patch 8)
          7) export dataset (Patch 8)
    
        It is intentionally robust: failures in optional steps are logged and skipped unless strict=True.
    
        Returns
        -------
        dict
            The final updated UQ dataset (also stored in self.uq_data).
        """
        ds = None
        attach_vs30_kwargs = attach_vs30_kwargs or {}
        mode_kwargs = mode_kwargs or {}
        mc_kwargs = mc_kwargs or {}
    
        # 1) Build dataset
        try:
            ds = self.uq_build_dataset(
                version_list=version_list,
                imts=imts,
                shakemap_folder=shakemap_folder,
                output_path=output_path,
                load_vs30=load_vs30,
                vs30_path=vs30_path,
                strict=strict,
                cache=True,
                verbose=verbose,
            )
        except Exception as e:
            if strict:
                raise
            self._uq_log(f"uq_pipeline: uq_build_dataset failed: {e}", level="error", verbose=verbose)
            return getattr(self, "uq_data", None)
    
        # 2) Attach Vs30 (alignment step)
        if load_vs30:
            try:
                self.uq_attach_vs30(
                    dataset=ds,
                    vs30_path=vs30_path,
                    output_path=output_path,
                    strict=strict,
                    verbose=verbose,
                    **attach_vs30_kwargs,
                )
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"uq_pipeline: uq_attach_vs30 failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        # 3) Run mode
        try:
            m = str(mode).lower().strip()
            if m == "bayes_update":
                self.uq_bayes_update(dataset=ds, version_list=version_list, imts=imts, output_path=output_path, strict=strict, verbose=verbose, **mode_kwargs)
            elif m == "monte_carlo":
                # if user selects monte_carlo mode, run MC as primary (products/bands can also use MC)
                self.uq_monte_carlo(dataset=ds, version_list=version_list, imts=imts, output_path=output_path, strict=strict, verbose=verbose, **mode_kwargs)
            elif m == "hierarchical":
                # hierarchical assumes you may already have posterior, but works either way
                self.uq_hierarchical(dataset=ds, version_list=version_list, imts=imts, output_path=output_path, strict=strict, verbose=verbose, **mode_kwargs)
            else:
                raise ValueError(f"uq_pipeline: unsupported mode '{mode}'.")
        except Exception as e:
            if strict:
                raise
            self._uq_log(f"uq_pipeline: mode '{mode}' failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        # 4) Products: PoE
        if compute_poe:
            try:
                poe_imt_list = poe_imts or (imts if imts is not None else None)
                # If poe_imt_list is None, compute only for imt="MMI" as a safe default
                if poe_imt_list is None:
                    poe_imt_list = ["MMI"]
                for imt_i in poe_imt_list:
                    self.uq_probability_of_exceedance(
                        dataset=ds,
                        version_list=version_list,
                        imt=str(imt_i),
                        thresholds=(None if poe_thresholds is None else self._uq_thresholds_for_imt(poe_thresholds, str(imt_i)) if isinstance(poe_thresholds, dict) else poe_thresholds),
                        basis=poe_basis,
                        output_path=output_path,
                        export_maps=True,
                        export_tables=True,
                        strict=strict,
                        verbose=verbose,
                    )
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"uq_pipeline: PoE computation failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        # 5) Products: Credible intervals
        if compute_ci:
            try:
                ci_imt_list = ci_imts or (imts if imts is not None else None)
                if ci_imt_list is None:
                    ci_imt_list = ["MMI"]
                for imt_i in ci_imt_list:
                    self.uq_credible_intervals(
                        dataset=ds,
                        version_list=version_list,
                        imt=str(imt_i),
                        quantiles=ci_quantiles,
                        basis=ci_basis,
                        method=ci_method,
                        output_path=output_path,
                        export_maps=True,
                        export_tables=True,
                        strict=strict,
                        verbose=verbose,
                    )
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"uq_pipeline: CI computation failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        # 6) Metrics + bands
        if compute_metrics:
            try:
                mt_imts = metrics_imts or (imts if imts is not None else None)
                self.uq_compute_metrics(
                    dataset=ds,
                    version_list=version_list,
                    imts=mt_imts,
                    thresholds=(metrics_thresholds if metrics_thresholds is not None else None),
                    basis=metrics_basis,
                    prefer_mc=prefer_mc_metrics,
                    n_mc_if_missing=int(n_mc_if_missing),
                    mc_kwargs=mc_kwargs,
                    include_exposure_if_available=include_exposure_if_available,
                    exposure_imt=exposure_imt,
                    exposure_threshold=float(exposure_threshold),
                    output_path=output_path,
                    export=True,
                    strict=strict,
                    verbose=verbose,
                )
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"uq_pipeline: metrics computation failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        # 7) Plots
        if make_plots:
            try:
                if plot_decay:
                    # plot decay for each IMT requested (or MMI default)
                    decay_imts = (imts if imts is not None else ["MMI"])
                    for imt_i in decay_imts:
                        try:
                            self.uq_plot_uncertainty_decay(
                                dataset=ds,
                                imt=str(imt_i),
                                basis="auto",
                                output_path=output_path,
                                save_formats=("png",),
                                close_figs=True,
                                verbose=verbose,
                            )
                        except Exception as ee:
                            if strict:
                                raise
                            self._uq_log(f"uq_pipeline: decay plot failed for {imt_i}: {ee}", level="debug", verbose=verbose)
    
                if plot_metrics:
                    # plot area_ge timeseries for each IMT (or MMI)
                    pm_imts = (metrics_imts if metrics_imts is not None else (imts if imts is not None else ["MMI"]))
                    for imt_i in pm_imts:
                        try:
                            self.uq_plot_metric_timeseries(
                                dataset=ds,
                                metric="area_ge",
                                imt=str(imt_i),
                                thresholds=(None if metrics_thresholds is None else self._uq_thresholds_for_imt(metrics_thresholds, str(imt_i)) if isinstance(metrics_thresholds, dict) else metrics_thresholds),
                                output_path=output_path,
                                save_formats=("png",),
                                close_figs=True,
                                verbose=verbose,
                            )
                        except Exception as ee:
                            if strict:
                                raise
                            self._uq_log(f"uq_pipeline: metric plot failed for {imt_i}: {ee}", level="debug", verbose=verbose)
    
                if plot_poe_quicklook:
                    self.uq_plot_poe_quicklook(
                        dataset=ds,
                        version=poe_quicklook_version,
                        imt=str(poe_quicklook_imt),
                        threshold=float(poe_quicklook_threshold),
                        basis=poe_basis,
                        output_path=output_path,
                        save_formats=("png",),
                        close_figs=True,
                        verbose=verbose,
                    )
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"uq_pipeline: plotting failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        # 8) Export dataset
        if export_dataset:
            try:
                # export npz and slim json by default
                self.uq_export_uq_dataset(
                    dataset=ds,
                    output_path=output_path,
                    formats=export_formats,
                    slim=bool(export_dataset_slim_json),
                    strict=strict,
                    verbose=verbose,
                )
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"uq_pipeline: dataset export failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        # Store and return
        self.uq_data = ds
        return ds
    




    # =========================
    # PATCH 10 (HOTFIX) — Fix mean/sigma basis resolver + robust product discovery by event_id + version
    # ADD AT THE END OF SHAKEtime CLASS
    # =========================
    
    def _uq_get_mean_sigma_by_basis(self, iblock, basis="auto"):
        """
        HOTFIX override.
    
        Resolve (mean, sigma) arrays from an IMT block according to basis.
    
        basis:
          - "auto": hierarchical -> posterior -> prior
          - "hierarchical": mean_hier/sigma_hier
          - "posterior": mean_post/sigma_post
          - "prior": mean/sigma
    
        Always returns a 2-tuple (mean, sigma). If sigma is missing, returns (mean, None).
        """
        b = str(basis).lower()
    
        if b == "auto":
            if iblock.get("mean_hier") is not None and iblock.get("sigma_hier") is not None:
                return iblock.get("mean_hier"), iblock.get("sigma_hier")
            if iblock.get("mean_post") is not None and iblock.get("sigma_post") is not None:
                return iblock.get("mean_post"), iblock.get("sigma_post")
            return iblock.get("mean"), iblock.get("sigma")
    
        if b == "hierarchical":
            return iblock.get("mean_hier"), iblock.get("sigma_hier")
        if b == "posterior":
            return iblock.get("mean_post"), iblock.get("sigma_post")
        if b == "prior":
            return iblock.get("mean"), iblock.get("sigma")
    
        raise ValueError(f"_uq_get_mean_sigma_by_basis: unsupported basis '{basis}'.")
    
    
    def _uq_find_product_files(self, event_id, version, shakemap_folder, verbose=True):
        """
        HOTFIX override.
    
        Robustly locate grid/uncertainty/stationlist/rupture files for a given event_id + version.
    
        This fixes the observed failure mode where the builder accidentally loads a different event's
        *_grid.xml because it searches too broadly.
    
        Expected naming (USGS-style):
            <event_id>_<region>_<ver:03d>_grid.xml
            <event_id>_<region>_<ver:03d>_uncertainty.xml
            <event_id>_<region>_<ver:03d>_stationlist.json
            <event_id>_<region>_<ver:03d>_rupture.json
    
        We match with glob patterns:
            f"{event_id}_*_{ver:03d}_grid.xml", etc.
    
        Returns dict:
          {"base_dir":..., "grid_xml":..., "uncertainty_xml":..., "stationlist_json":..., "rupture_json":...}
        """
        import os
        import glob
    
        # normalize
        event_id = str(event_id).strip()
        v = int(version)
        vtag = f"{v:03d}"
    
        # Candidate search roots (try event subdir first)
        roots = []
        if shakemap_folder is not None:
            roots.append(os.path.abspath(str(shakemap_folder)))
            roots.append(os.path.abspath(os.path.join(str(shakemap_folder), event_id)))
    
        # Also try common repo layout if present
        roots.append(os.path.abspath(os.path.join(".", "event_data", "SHAKEfetch", "usgs-shakemap-versions", event_id)))
        roots.append(os.path.abspath(os.path.join(".", "event_data", "SHAKEfetch", "usgs-shakemap-versions")))
    
        # de-dup
        roots = [r for i, r in enumerate(roots) if r and r not in roots[:i]]
    
        # Build patterns
        pat_grid = f"{event_id}_*_{vtag}_grid.xml"
        pat_unc  = f"{event_id}_*_{vtag}_uncertainty.xml"
        pat_sta  = f"{event_id}_*_{vtag}_stationlist.json"
        pat_rup  = f"{event_id}_*_{vtag}_rupture.json"
    
        def first_match(root, pattern):
            # recursive match
            hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
            if not hits:
                return None
            # prefer shortest path (closest to root)
            hits = sorted(hits, key=lambda p: (len(p), p))
            return hits[0]
    
        grid_xml = unc_xml = sta_json = rup_json = None
        base_dir = None
    
        # Search in order of roots
        for root in roots:
            if not os.path.exists(root):
                continue
            g = first_match(root, pat_grid)
            u = first_match(root, pat_unc)
            s = first_match(root, pat_sta)
            r = first_match(root, pat_rup)
    
            # If we got at least grid+uncertainty, accept this root
            if g is not None and u is not None:
                grid_xml = g
                unc_xml = u
                sta_json = s
                rup_json = r
                base_dir = os.path.dirname(g)
                break
    
        out = {
            "base_dir": base_dir,
            "grid_xml": grid_xml,
            "uncertainty_xml": unc_xml,
            "stationlist_json": sta_json,
            "rupture_json": rup_json,
        }
    
        if verbose:
            self._uq_log(f"[HOTFIX] _uq_find_product_files({event_id}, v{vtag}) -> {out}", level="debug", verbose=True)
    
        return out
    
    
    
    


    # =========================
    # PATCH 11 — Builder v2 that uses _uq_find_product_files() + Pipeline v2 (full orchestration)
    # ADD AT END OF SHAKEtime CLASS
    # =========================
    
    def uq_build_dataset_v2(
        self,
        version_list=None,
        imts=None,
        shakemap_folder=None,
        output_path="./export",
        load_vs30=True,
        vs30_path=None,
        strict=False,
        verbose=True,
    ):
        """
        v2 dataset builder: uses robust _uq_find_product_files(event_id, version, shakemap_folder)
        so it cannot mix up events (fixes the us7000pn9s vs us6000jllz issue).
    
        Returns dataset in the same structure used by later uq_* methods.
        """
        import os
    
        event_id = self._uq_get_event_id()
        uq_dirs = self._uq_prepare_export_dirs(output_path=output_path, event_id=event_id)
    
        ds = {
            "event_id": event_id,
            "shakemap_folder": shakemap_folder,
            "imts_requested": imts,
            "meta": {"uq_export_dirs": uq_dirs},
            "vs30": {"attached": False, "source": None, "note": "placeholder"},
            "versions": {},
        }
    
        if version_list is None:
            version_list = getattr(self, "version_list", None)
        if version_list is None:
            raise ValueError("uq_build_dataset_v2: version_list is None.")
    
        for v in version_list:
            try:
                v_int = int(str(v).lower().replace("v", ""))
            except Exception:
                v_int = int(v)
    
            vkey = f"{v_int:03d}"
    
            paths = self._uq_find_product_files(event_id, v_int, shakemap_folder, verbose=verbose)
    
            ds["versions"][vkey] = {
                "version": vkey,
                "paths": paths,
                "stations": [],
                "dyfi": [],
                "rupture": None,
                "imts": {},
            }
    
            # grid + uncertainty must exist
            if not paths.get("grid_xml") or not paths.get("uncertainty_xml"):
                msg = f"[v{vkey}] missing grid/uncertainty files for event {event_id}."
                if strict:
                    raise FileNotFoundError(msg)
                self._uq_log(msg, level="warning", verbose=verbose)
                continue
    
            # parse grid/uncertainty
            grid = self._uq_parse_grid_xml(paths["grid_xml"], imts=imts)
            unc = self._uq_parse_uncertainty_xml(paths["uncertainty_xml"], imts=imts)
    
            for imt_name, gblock in grid.items():
                ds["versions"][vkey]["imts"][imt_name] = {
                    "imt": imt_name,
                    "lon": gblock["lon"],
                    "lat": gblock["lat"],
                    "mean": gblock["mean"],
                    "sigma": unc.get(imt_name, {}).get("sigma", None),
                    "vs30": None,
                }
    
            # stationlist + rupture if present
            if paths.get("stationlist_json"):
                try:
                    stations, dyfi = self._uq_parse_stationlist_json(paths["stationlist_json"])
                    ds["versions"][vkey]["stations"] = stations
                    ds["versions"][vkey]["dyfi"] = dyfi
                except Exception as e:
                    self._uq_log(f"[v{vkey}] stationlist parse failed: {e}", level="warning", verbose=verbose)
            else:
                self._uq_log(f"[v{vkey}] stationlist json missing; station-based UQ modes may be limited.", level="warning", verbose=verbose)
    
            if paths.get("rupture_json"):
                try:
                    ds["versions"][vkey]["rupture"] = self._uq_parse_rupture_json(paths["rupture_json"])
                except Exception as e:
                    self._uq_log(f"[v{vkey}] rupture parse failed: {e}", level="warning", verbose=verbose)
    
            self._uq_log(f"[v{vkey}] loaded IMTs: {list(ds['versions'][vkey]['imts'].keys())}", level="info", verbose=verbose)
    
        # attach vs30 (optional)
        if load_vs30:
            try:
                self.uq_attach_vs30(dataset=ds, vs30_path=vs30_path, output_path=output_path, strict=False, verbose=verbose)
            except Exception as e:
                self._uq_log(f"uq_build_dataset_v2: vs30 attach failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        self.uq_data = ds
        return ds
    
    
    def uq_pipeline_v2(
        self,
        version_list=None,
        imts=None,
        shakemap_folder=None,
        output_path="./export",
        load_vs30=True,
        vs30_path=None,
        attach_vs30_kwargs=None,
        mode="bayes_update",
        mode_kwargs=None,
        compute_poe=True,
        poe_imts=None,
        poe_thresholds=None,
        poe_basis="auto",
        compute_ci=True,
        ci_imts=None,
        ci_quantiles=(0.05, 0.5, 0.95),
        ci_basis="auto",
        ci_method="auto",
        compute_metrics=True,
        metrics_imts=None,
        metrics_thresholds=None,
        metrics_basis="auto",
        prefer_mc_metrics=True,
        n_mc_if_missing=0,
        mc_kwargs=None,
        include_exposure_if_available=True,
        exposure_imt="MMI",
        exposure_threshold=4.0,
        make_plots=True,
        plot_decay=True,
        plot_metrics=True,
        plot_poe_quicklook=False,
        poe_quicklook_version=None,
        poe_quicklook_imt="MMI",
        poe_quicklook_threshold=5.0,
        export_dataset=True,
        export_formats=("npz", "json"),
        export_dataset_slim_json=True,
        strict=False,
        verbose=True,
    ):
        """
        v2 full orchestration: identical intent to uq_pipeline(), but uses uq_build_dataset_v2()
        to guarantee correct per-event file selection.
        """
        attach_vs30_kwargs = attach_vs30_kwargs or {}
        mode_kwargs = mode_kwargs or {}
        mc_kwargs = mc_kwargs or {}
    
        ds = self.uq_build_dataset_v2(
            version_list=version_list,
            imts=imts,
            shakemap_folder=shakemap_folder,
            output_path=output_path,
            load_vs30=load_vs30,
            vs30_path=vs30_path,
            strict=strict,
            verbose=verbose,
        )
    
        # Apply attach_vs30_kwargs if user wants to override defaults (optional)
        if load_vs30 and attach_vs30_kwargs:
            try:
                self.uq_attach_vs30(dataset=ds, vs30_path=vs30_path, output_path=output_path, strict=False, verbose=verbose, **attach_vs30_kwargs)
            except Exception as e:
                if strict:
                    raise
                self._uq_log(f"uq_pipeline_v2: uq_attach_vs30 failed (non-fatal): {e}", level="warning", verbose=verbose)
    
        # Mode
        m = str(mode).lower().strip()
        if m == "bayes_update":
            self.uq_bayes_update(dataset=ds, version_list=version_list, imts=imts, output_path=output_path, strict=strict, verbose=verbose, **mode_kwargs)
        elif m == "monte_carlo":
            self.uq_monte_carlo(dataset=ds, version_list=version_list, imts=imts, output_path=output_path, strict=strict, verbose=verbose, **mode_kwargs)
        elif m == "hierarchical":
            self.uq_hierarchical(dataset=ds, version_list=version_list, imts=imts, output_path=output_path, strict=strict, verbose=verbose, **mode_kwargs)
        else:
            self._uq_log(f"uq_pipeline_v2: unsupported mode '{mode}'", level="warning", verbose=verbose)
    
        # PoE
        if compute_poe and poe_thresholds is not None:
            imt_list = poe_imts or imts or ["MMI"]
            for imt_i in imt_list:
                self.uq_probability_of_exceedance(
                    dataset=ds,
                    version_list=version_list,
                    imt=str(imt_i),
                    thresholds=(self._uq_thresholds_for_imt(poe_thresholds, str(imt_i)) if isinstance(poe_thresholds, dict) else poe_thresholds),
                    basis=poe_basis,
                    output_path=output_path,
                    export_maps=True,
                    export_tables=True,
                    strict=strict,
                    verbose=verbose,
                )
    
        # CI
        if compute_ci:
            imt_list = ci_imts or imts or ["MMI"]
            for imt_i in imt_list:
                self.uq_credible_intervals(
                    dataset=ds,
                    version_list=version_list,
                    imt=str(imt_i),
                    quantiles=ci_quantiles,
                    basis=ci_basis,
                    method=ci_method,
                    output_path=output_path,
                    export_maps=True,
                    export_tables=True,
                    strict=strict,
                    verbose=verbose,
                )
    
        # Metrics
        if compute_metrics:
            self.uq_compute_metrics(
                dataset=ds,
                version_list=version_list,
                imts=(metrics_imts or imts),
                thresholds=metrics_thresholds,
                basis=metrics_basis,
                prefer_mc=prefer_mc_metrics,
                n_mc_if_missing=int(n_mc_if_missing),
                mc_kwargs=mc_kwargs,
                include_exposure_if_available=include_exposure_if_available,
                exposure_imt=exposure_imt,
                exposure_threshold=float(exposure_threshold),
                output_path=output_path,
                export=True,
                strict=strict,
                verbose=verbose,
            )
    
        # Plots
        if make_plots:
            if plot_decay:
                for imt_i in (imts or ["MMI"]):
                    try:
                        self.uq_plot_uncertainty_decay(dataset=ds, imt=str(imt_i), basis="auto", output_path=output_path, save_formats=("png",), close_figs=True, verbose=verbose)
                    except Exception as e:
                        if strict:
                            raise
                        self._uq_log(f"uq_pipeline_v2: decay plot failed for {imt_i}: {e}", level="debug", verbose=verbose)
    
            if plot_metrics:
                for imt_i in (metrics_imts or imts or ["MMI"]):
                    try:
                        self.uq_plot_metric_timeseries(
                            dataset=ds,
                            metric="area_ge",
                            imt=str(imt_i),
                            thresholds=(self._uq_thresholds_for_imt(metrics_thresholds, str(imt_i)) if isinstance(metrics_thresholds, dict) else metrics_thresholds),
                            output_path=output_path,
                            save_formats=("png",),
                            close_figs=True,
                            verbose=verbose,
                        )
                    except Exception as e:
                        if strict:
                            raise
                        self._uq_log(f"uq_pipeline_v2: metrics plot failed for {imt_i}: {e}", level="debug", verbose=verbose)
    
            if plot_poe_quicklook:
                self.uq_plot_poe_quicklook(
                    dataset=ds,
                    version=poe_quicklook_version,
                    imt=str(poe_quicklook_imt),
                    threshold=float(poe_quicklook_threshold),
                    basis=poe_basis,
                    output_path=output_path,
                    save_formats=("png",),
                    close_figs=True,
                    verbose=verbose,
                )
    
        # Export dataset
        if export_dataset:
            self.uq_export_uq_dataset(
                dataset=ds,
                output_path=output_path,
                formats=export_formats,
                slim=bool(export_dataset_slim_json),
                strict=strict,
                verbose=verbose,
            )
    
        self.uq_data = ds
        return ds




















    
    
    ##############################################################
    ##############################################################
    ##########################################################
    #
    #
    #                     Just In Case Backups 
    #
    #
    #
    #########################################################
    #back up version   
    def plot_ratemap_details_v25(
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

        # 1) build the rate grid
        ug = self.get_rate_grid(version_list, metric=metric, use_cache=use_cache)

        # 2) pick which columns to plot
        cols = list(specific_columns) if specific_columns else []
        if not cols:
            # consecutive‐pairs
            for i in range(len(version_list) - 1):
                v1, v2 = version_list[i], version_list[i+1]
                cols += [f"delta_{v2}_{v1}_{metric}", f"rate_{v2}_{v1}_{metric}"]
            # append final vs first
            first, last = version_list[0], version_list[-1]
            cols += [
                f"delta_{last}_{first}_{metric}",
                f"rate_{last}_{first}_{metric}"
            ]

        # 3) filter by 'which'
        if which not in ("delta", "rate", "both"):
            raise ValueError(f"Invalid which='{which}'; must be 'delta','rate', or 'both'")
        if which == "delta":
            cols = [c for c in cols if c.startswith("delta_")]
        elif which == "rate":
            cols = [c for c in cols if c.startswith("rate_")]

        # registry of already‐plotted station & DYFI IDs/codes
        plotted_station_ids   = set()
        plotted_station_codes = set()
        plotted_dyfi_ids      = set()
        plotted_dyfi_codes    = set()

        first, last = version_list[0], version_list[-1]
        final_delta_col = f"delta_{last}_{first}_{metric}"
        final_rate_col  = f"rate_{last}_{first}_{metric}"

        figs = []

        for col in cols:
            # detect if this is the final‐vs‐first column
            is_final_first = col in (final_delta_col, final_rate_col)

            # extract v2 from "delta_v2_v1_metric"
            parts = col.split("_")
            v2 = parts[1]

            # compute map extent from entire grid
            extent = [
                float(ug.lon.min()), float(ug.lon.max()),
                float(ug.lat.min()), float(ug.lat.max()),
            ]

            # 4) create mapper at that extent
            mapper = SHAKEmapper(extent=extent)
            fig, ax = mapper.create_basemap(label_size=22)

            # 5) plot the scatter
            norm = Normalize(vmin=-2, vmax=2)
            sc = ax.scatter(
                ug.lon, ug.lat, c=ug[col],
                cmap="seismic", norm=norm,
                s=15, edgecolor="none",
                transform=mapper.ax.projection, zorder=8
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

                        # PGA → station markers
                        df_sta = ip.get_dataframe(value_type="pga")\
                                   .dropna(subset=["longitude", "latitude"])
                        if not df_sta.empty:
                            if is_final_first:
                                new_sta = df_sta
                            else:
                                mask_sta = (
                                    ~df_sta["id"].astype(str).isin(plotted_station_ids) &
                                    ~df_sta["station_code"].astype(str).isin(plotted_station_codes)
                                )
                                new_sta = df_sta[mask_sta]
                            if not new_sta.empty:
                                mapper.add_stations(new_sta["longitude"].values,
                                                    new_sta["latitude"].values)
                                if not is_final_first:
                                    plotted_station_ids.update(new_sta["id"].astype(str))
                                    plotted_station_codes.update(new_sta["station_code"].astype(str))
                                logging.info(f"  ✓ plotted {len(new_sta)} station points")

                        # MMI → DYFI
                        df_dy = ip.get_dataframe(value_type="mmi")\
                                  .dropna(subset=["longitude", "latitude", "intensity"])
                        if not df_dy.empty:
                            if is_final_first:
                                new_dy = df_dy
                            else:
                                mask_dy = (
                                    ~df_dy["id"].astype(str).isin(plotted_dyfi_ids) &
                                    ~df_dy["station_code"].astype(str).isin(plotted_dyfi_codes)
                                )
                                new_dy = df_dy[mask_dy]
                            if not new_dy.empty:
                                mapper.add_dyfi(new_dy["longitude"].values,
                                                new_dy["latitude"].values,
                                                new_dy["intensity"].values,
                                                nresp=new_dy.get("nresp"))
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
                od = Path(output_path) / "SHAKEtime" / self.event_id / "rate_map_details" / metric
                od.mkdir(parents=True, exist_ok=True)
                for ext in save_formats:
                    fp = od / f"{col}.{ext}"
                    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
                    logging.info(f"  ✓ saved {fp}")

            figs.append((fig, ax, col))

        return figs




    def plot_shakemaps_v25(
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
        dpi: int = 300
    ) -> list:
        """
        Plots ShakeMap maps for the specified versions, optionally overlaying
        rupture traces, seismic stations (PGA), DYFI reports (MMI), and cities.
    
        Parameters
        ----------
        version_list : list of str
            ShakeMap version identifiers to plot.
        metric : str, default "mmi"
            Intensity measure to render.
        rupture_folder : str, optional
            Path to directory of rupture JSONs.
        stations_folder : str, optional
            Path to directory of station‐list JSONs.
        add_cities : bool, default False
            If True, overlay cities above `cities_population`.
        cities_population : int, default 1_000_000
            Minimum population for cities to plot.
        output_path : str, optional
            Base directory for saving maps.
        plot_colorbar : bool, default True
            Whether to draw the colorbar.
        show_title : bool, default True
            Whether to set a title on each map.
        save_formats : list of str, default ["png","pdf"]
            File extensions to save ("png","pdf","svg",…).
        dpi : int, default 300
            Resolution for saved figures.
    
        Returns
        -------
        List[tuple]
            A list of (fig, ax, version) for each generated map.
        """
        figures = []
    
        for version in version_list:
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
    
            try:
                # 1) base map + ShakeMap
                mapper = SHAKEmapper()
                mapper.create_basemap(label_size=22)
                mapper.add_usgs_shakemap(
                    parser,
                    plot_colorbar=plot_colorbar
                )
    
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
    
                # 3) stations + DYFI overlay
                if stations_folder:
                    st_fn = self._get_stations_filename(version)
                    st_path = Path(stations_folder) / self.event_id / st_fn
                    logging.info(f"  adding stations/DYFI from {st_path}")
                    try:
                        inst_parser = USGSParser(
                            parser_type="instrumented_data",
                            json_file=str(st_path)
                        )
                        # seismic instruments (PGA)
                        inst_df = inst_parser.get_dataframe(value_type="pga")\
                                             .dropna(subset=["longitude", "latitude"])
                        mapper.add_stations(
                            inst_df["longitude"].values,
                            inst_df["latitude"].values
                        )
                        # DYFI reports (MMI)
                        dyfi_df = inst_parser.get_dataframe(value_type="mmi")\
                                             .dropna(subset=["longitude", "latitude", "intensity"])
                        mapper.add_dyfi(
                            dyfi_df["longitude"].values,
                            dyfi_df["latitude"].values,
                            dyfi_df["intensity"].values,plot_colorbar=plot_colorbar
                        )
                    except Exception as e:
                        logging.warning(f"  ⚠ stations/DYFI failed for v{version}: {e}")
    
                # 4) epicenter
                try:
                    ev_meta = parser.get_metadata()["event"]
                    mapper.add_epicenter(
                        float(ev_meta["lon"]),
                        float(ev_meta["lat"])
                    )
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
                    out_dir = (Path(output_path)
                               / "SHAKEtime" / self.event_id
                               / "shakemaps" / metric)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for fmt in save_formats:
                        save_path = out_dir / f"{self.event_id}_shakemap_v{version}_{metric}.{fmt}"
                        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
                        logging.info(f"Saved ShakeMap plot to {save_path}")
    
            except Exception as e:
                logging.error(f"  ✖ plotting failed for v{version}: {e}")
    
        return figures



    def create_overview_panels_(
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
            ax2 = fig.add_subplot(gs[0:2,1:3])
            self._plot_thd_axes(ax2, "mmi", ver, total, tick_pos, tick_labs, delta=False)
            ax3 = fig.add_subplot(gs[0:2,1:3])
            self._plot_thd_axes(ax3, "mmi", ver, total, tick_pos, tick_labs, delta=True)

            # AUX time series (row 3, cols 1–2)
            x_here = np.arange(idx+1)
            ax8 = fig.add_subplot(gs[2:4, 2:4], sharex=ax2)
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


            ax9 = fig.add_subplot(gs[2:4, 2:4], sharex=ax8)
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



#######################################################################


   # --- Statistical Window for data ---
    # === SHAKEtime Statistical Analysis & Plotting Extensions ===

