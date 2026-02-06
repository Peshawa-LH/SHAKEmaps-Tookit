"""
Fetch USGS ShakeMap data 
Class Name: SHAKEfetch

Description:
    The SHAKEfetch class provides functionality to retrieve SHAKEMaps and related datasets from USGS servers, including PAGERs.
    This class is particularly useful for accessing data for specific seismic events, such as the (e.g., 2024 Taiwan earthquake with Mw 7.4) with The default mode provides the fetching of ShakeMaps and related datasets for this event from 
    USGS servers.

Prerequisites:
    To use this class effectively, ensure it is executed within an environment where the libcomcat library is installed. 
    Follow the libcomcat installation guide: [libcomcat Installation Guide](https://code.usgs.gov/ghsc/esi/libcomcat-python)

About libcomcat:
    libcomcat is a Python library that interfaces with the ANSS ComCat search API, providing a Pythonic context for accessing 
    earthquake-related data. It includes various classes and functions for interacting with the ComCat API, as well as command-line 
    programs for data retrieval and manipulation.


Date:
    December, 2025
Version:
    26.1
    
"""




import os
import subprocess
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libcomcat.search import get_event_by_id
from contextlib import contextmanager

# Set up basic logging (configured once at the module level)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@contextmanager
def change_dir(new_dir):
    """Context manager for temporarily changing the working directory."""
    original_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(original_dir)

class SHAKEfetch:
    """
    SHAKEfetch: A comprehensive, event-centric interface for inspecting and
    retrieving earthquake products from the USGS ComCat system.
    
    SHAKEfetch provides tools to:
      • inspect which products, versions, and files exist for a given event
      • selectively download products and files with full version control
      • support research workflows involving time-evolving SHAKEmaps,
        DYFI updates, and source evolution products
    
    The class is intentionally **single-event scoped**: one SHAKEfetch instance
    corresponds to one USGS event ID. Multi-event workflows are handled by
    instantiating the class multiple times (e.g. in a loop).
    
    ────────────────────────────────────────────────────────────────────────────
    PRODUCT COVERAGE
    ────────────────────────────────────────────────────────────────────────────
    Depending on availability in ComCat, SHAKEfetch can inspect and/or download:
    
    • ShakeMap products
      (grid.xml, raster.zip, uncertainty.xml, stationlist.json, contours,
       coverage files, figures, rupture.json, shapefiles, etc.)
    
    • DYFI (Did-You-Feel-It?) products
      (geospatial files, CIIM plots, response counts, GeoJSON, KMZ, etc.)
    
    • LossPager products
      (PDFs, XML, JSON summaries, exposure grids)
    
    • Source and catalog products
      (origin, moment-tensor, finite-fault, phase-data)
    
    • Textual impact products
      (impact-text, general-text, tectonic summaries — if present)
    
    Availability varies by event; SHAKEfetch never assumes a product exists
    without checking ComCat.
    
    ────────────────────────────────────────────────────────────────────────────
    VERSION HANDLING
    ────────────────────────────────────────────────────────────────────────────
    Most inspection and download methods accept a `version` selector.
    
    Allowed values:
      - "preferred" : USGS-preferred authoritative version
      - "last"      : Most recent published version
      - "first"     : First published version
      - "all"       : All available versions
      - int         : Explicit version number (e.g. 12)
    
    Note:
      • ComCat inspection (via libcomcat) and file downloads (via getproduct)
        may expose versions differently.
      • This difference is expected and documented in the toolkit.
    
    ────────────────────────────────────────────────────────────────────────────
    CORE ATTRIBUTES
    ────────────────────────────────────────────────────────────────────────────
    event_id : str
        USGS ComCat event ID associated with this instance.
    original_dir : str
        Working directory where the class was instantiated.
    earthquake : object
        libcomcat event object containing metadata and product declarations.
    dyfi_data : pandas.DataFrame
        Processed DYFI response data (populated by get_dyfi).
    
    ────────────────────────────────────────────────────────────────────────────
    KEY METHODS (v26.1)
    ────────────────────────────────────────────────────────────────────────────
    check_event_files(...)
        Inspect and log available products, versions, and content trees
        declared by ComCat (read-only, no downloads).
    
    modular_fetcher(plan, ...)
        Modular, plan-driven downloader allowing per-product control over:
          • on/off state
          • version selection (including explicit version numbers)
          • file subsets or full product trees
          • dry-run previews for reproducibility
    
    Legacy convenience download methods (e.g. get_shakemaps, get_stations,
    get_dyfi_files, etc.) are retained for backward compatibility and
    specialized workflows.
    
    ────────────────────────────────────────────────────────────────────────────
    EXAMPLE
    ────────────────────────────────────────────────────────────────────────────
    >>> sf = SHAKEfetch("us7000pn9s")
    >>> sf.check_event_files(products=["shakemap","dyfi"])
    
    >>> plan = {
    ...   "shakemap": {"on": True, "version": "all", "files": "download/grid.xml"},
    ...   "dyfi": {"on": True, "version": "last", "files": "__ALL__"},
    ... }
    >>> sf.modular_fetcher(plan, dry_run=True)   # preview
    >>> sf.modular_fetcher(plan, dry_run=False)  # download
    
    ────────────────────────────────────────────────────────────────────────────
    NOTES
    ────────────────────────────────────────────────────────────────────────────
    • SHAKEfetch relies on the `libcomcat` Python package for catalog inspection.
    • File downloads use the USGS `getproduct` command-line utility.
    • Downloaded files are organized under export/SHAKEfetch/.
    • Designed to support uncertainty analysis and time-evolving SHAKEmaps
      in rapid-response and research contexts.
    
    © SHAKEmaps Toolkit — SHAKEfetch v26.1
    """

    def __init__(self, event_id='us7000m9g4', export_dir='export'):
        """
        Initializes the SHAKEfetch class with a specified or default event ID.

        Parameters:
        -----------
        event_id : str
            The event ID of the earthquake to be fetched. Defaults to 'us7000m9g4'.
        """


        self.event_id    = event_id
        self.original_dir = os.getcwd()
        self.export_dir   = export_dir
        # Build <cwd>/<export_dir>/SHAKEfetch once, reuse in all downloads
        self.base_dir     = os.path.join(self.original_dir, self.export_dir, 'SHAKEfetch')
        self.earthquake   = None
        self.check_shakefetch_inputs()
        # self.dyfi_data = None  # Uncomment if you plan to use get_dyfi later.


    def _run_download_commands(self, product_type, file_list, output_subdir, version):
        """
        Helper method to download files.

        This helper:
          - Validates the version parameter.
          - Creates the output directory if needed.
          - Changes to the output directory.
          - Iterates over the file_list to build and run the getproduct commands.
          - Logs successes and failures.
        
        Parameters:
        -----------
        product_type : str
            The type of product to download (e.g., 'shakemap', 'dyfi', 'losspager').
        file_list : list of str
            List of file paths (relative) to download.
        output_subdir : str
            The subdirectory under 'export/SHAKEfetch' where files will be saved.
        version : str
            The version flag to use. Allowed values: "last", "all", "first", "preferred".
        
        Returns:
        --------
        int
            The number of files successfully downloaded.
        """
        
        allowed = {"last", "all", "first", "preferred"}
        if version not in allowed:
            raise ValueError(f"Invalid version '{version}', choose from {allowed}")

        # ensure root exists
        output_dir = os.path.join(self.base_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        success = 0
        with change_dir(output_dir):
            for content in file_list:
                cmd = f"getproduct {product_type} {content} -i {self.event_id} --get-version={version}"
                logging.info(f"[{product_type}] {cmd}")
                try:
                    subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True)
                    success += 1
                except subprocess.CalledProcessError as e:
                    logging.error(f"[{product_type}] failed: {e.stderr.strip()}")
        return success


    


    def get_stations(self, version="all"):
        """
        Downloads station data files (stationlist.json) for the specified event from USGS.

        Files are saved under: export/SHAKEfetch/usgs-instruments_data-versions.

        Parameters:
        -----------
        version : str, optional
            Version flag for the download. Allowed values: "last", "all", "first", "preferred".
            Default is "all".
        """
        if not self.earthquake:
            logging.error(f"[get_stations] Earthquake information not available for event ID: {self.event_id}.")
            return
        logging.info(f"[get_stations] Starting station data download for event ID: {self.event_id}.")
        file_list = ["download/stationlist.json"]
        count = self._run_download_commands("shakemap", file_list, "usgs-instruments_data-versions", version)
        logging.info(f"[get_stations] Station list retrieval completed. Files saved: {count}/{len(file_list)}.")

    def get_ruptures(self, version='all'):
        """
        Downloads rupture data files (rupture.json) for the specified event from USGS.

        Files are saved under: export/SHAKEfetch/usgs-rupture-versions.

        Parameters:
        -----------
        version : str, optional
            Version flag for the download. Allowed values: "last", "all", "first", "preferred".
            Default is "all".
        """
        if not self.earthquake:
            logging.error(f"[get_ruptures] Earthquake information not available for event ID: {self.event_id}.")
            return
        logging.info(f"[get_ruptures] Starting rupture download for event ID: {self.event_id}.")
        file_list = ["download/rupture.json"]
        count = self._run_download_commands("shakemap", file_list, "usgs-rupture-versions", version)
        logging.info(f"[get_ruptures] Rupture retrieval completed. Files saved: {count}/{len(file_list)}.")

    def get_attenuation_curves(self, version='all'):
        """
        Downloads attenuation curves (attenuation_curves.json) for the specified event from USGS.

        Files are saved under: export/SHAKEfetch/usgs-attenuation_curves-versions.

        Parameters:
        -----------
        version : str, optional
            Version flag for the download. Allowed values: "last", "all", "first", "preferred".
            Default is "all".
        """
        if not self.earthquake:
            logging.error(f"[get_attenuation_curves] Earthquake information not available for event ID: {self.event_id}.")
            return
        logging.info(f"[get_attenuation_curves] Starting attenuation curves download for event ID: {self.event_id}.")
        file_list = ["download/attenuation_curves.json"]
        count = self._run_download_commands("shakemap", file_list, "usgs-attenuation_curves-versions", version)
        logging.info(f"[get_attenuation_curves] Attenuation curves retrieval completed. Files saved: {count}/{len(file_list)}.")

    def get_contours(self, version="all"):
        """
        Downloads contour files (e.g., cont_mi.json, cont_pga.json, etc.) for the specified event from USGS.

        Files are saved under: export/SHAKEfetch/usgs-contours-versions.

        Parameters:
        -----------
        version : str, optional
            Version flag for the download. Allowed values: "last", "all", "first", "preferred".
            Default is "all".
        """
        if not self.earthquake:
            logging.error(f"[get_contours] Earthquake information not available for event ID: {self.event_id}.")
            return
        logging.info(f"[get_contours] Starting contour download for event ID: {self.event_id}.")
        file_list = [
            "download/cont_mi.json",
            "download/cont_mmi.json",
            "download/cont_pga.json",
            "download/cont_pgv.json",
            "download/cont_psa0p3.json",
            "download/cont_psa0p6.json",
            "download/cont_psa1p0.json",
            "download/cont_psa3p0.json"
        ]
        count = self._run_download_commands("shakemap", file_list, "usgs-contours-versions", version)
        logging.info(f"[get_contours] Contour retrieval completed. Files saved: {count}/{len(file_list)}.")

    def get_coverages(self, version="all"):
        """
        Downloads coverage files (covjson) for various ground-motion parameters for the specified event from USGS.

        Files are saved under: export/SHAKEfetch/usgs-coverage-versions.

        Parameters:
        -----------
        version : str, optional
            Version flag for the download. Allowed values: "last", "all", "first", "preferred".
            Default is "all".
        """
        if not self.earthquake:
            logging.error(f"[get_coverage] Earthquake information not available for event ID: {self.event_id}.")
            return
        logging.info(f"[get_coverage] Starting coverage files download for event ID: {self.event_id}.")
        file_list = [
            "download/coverage_mmi_high_res.covjson",
            "download/coverage_mmi_low_res.covjson",
            "download/coverage_mmi_medium_res.covjson",
            "download/coverage_pga_high_res.covjson",
            "download/coverage_pga_low_res.covjson",
            "download/coverage_pga_medium_res.covjson",
            "download/coverage_pgv_high_res.covjson",
            "download/coverage_pgv_low_res.covjson",
            "download/coverage_pgv_medium_res.covjson",
            "download/coverage_psa0p3_high_res.covjson",
            "download/coverage_psa0p3_low_res.covjson",
            "download/coverage_psa0p3_medium_res.covjson",
            "download/coverage_psa0p6_high_res.covjson",
            "download/coverage_psa0p6_low_res.covjson",
            "download/coverage_psa0p6_medium_res.covjson",
            "download/coverage_psa1p0_high_res.covjson",
            "download/coverage_psa1p0_low_res.covjson",
            "download/coverage_psa1p0_medium_res.covjson",
            "download/coverage_psa3p0_high_res.covjson",
            "download/coverage_psa3p0_low_res.covjson",
            "download/coverage_psa3p0_medium_res.covjson"
        ]
        count = self._run_download_commands("shakemap", file_list, "usgs-coverage-versions", version)
        logging.info(f"[get_coverage] Coverage retrieval completed. Files saved: {count}/{len(file_list)}.")

    def get_event_info(self, version="all"):
        """
        Downloads the event information file (info.json) for the specified event from USGS.

        File is saved under: export/SHAKEfetch/usgs-event_info-versions.

        Parameters:
        -----------
        version : str, optional
            Version flag for the download (e.g., "all", "first", "preferred", "last"). Default is "all".
        """
        if not self.earthquake:
            logging.error(f"[get_event_info] Earthquake information not available for event ID: {self.event_id}.")
            return
        logging.info(f"[get_event_info] Starting event information download for event ID: {self.event_id}.")
        file_list = ["download/info.json"]
        count = self._run_download_commands("shakemap", file_list, "usgs-event_info-versions", version)
        logging.info(f"[get_event_info] Event info retrieval completed. Files saved: {count}/{len(file_list)}.")

    def get_shapefiles(self, version="all"):
        """
        Downloads shapefile products for the specified event from USGS.

        Files downloaded:
          - download/shakemap.kmz
          - download/shape.zip

        Saved under: export/SHAKEfetch/usgs-shapefiles-versions.

        Parameters:
        -----------
        version : str, optional
            Version flag for the download. Allowed values: "last", "all", "first", "preferred".
            Default is "all".
        """
        if not self.earthquake:
            logging.error(f"[get_shapefiles] Earthquake information not available for event ID: {self.event_id}.")
            return
        logging.info(f"[get_shapefiles] Starting shapefile download for event ID: {self.event_id}.")
        file_list = [
            "download/shakemap.kmz",
            "download/shape.zip"
        ]
        count = self._run_download_commands("shakemap", file_list, "usgs-shapefiles-versions", version)
        logging.info(f"[get_shapefiles] Shapefile retrieval completed. Files saved: {count}/{len(file_list)}.")

    def get_figures_all(self, version="all"):
        """
        Downloads all ShakeMap figure files for the specified event from USGS.

        Files downloaded include:
          - download/intensity.jpg, intensity.pdf, intensity_overlay.png, intensity_overlay.pngw,
            mmi_legend.png, mmi_regr.png, pga.jpg, pga.pdf, pga_regr.png, pgv.jpg, pgv.pdf,
            pgv_regr.png, pin-thumbnail.png, psa0p3.jpg, psa0p3.pdf, psa0p3_regr.png, psa0p6.jpg,
            psa0p6.pdf, psa0p6_regr.png, psa1p0.jpg, psa1p0.pdf, psa1p0_regr.png, psa3p0.jpg,
            psa3p0.pdf, psa3p0_regr.png.

        Saved under: export/SHAKEfetch/usgs-figures-versions.

        Parameters:
        -----------
        version : str, optional
            Version flag for the download. Allowed values: "last", "all", "first", "preferred".
            Default is "all".
        """
        if not self.earthquake:
            logging.error(f"[get_figures_all] Earthquake information not available for event ID: {self.event_id}.")
            return
        logging.info(f"[get_figures_all] Starting figure files download for event ID: {self.event_id}.")
        file_list = [
            "download/intensity.jpg",
            "download/intensity.pdf",
            "download/intensity_overlay.png",
            "download/intensity_overlay.pngw",
            "download/mmi_legend.png",
            "download/mmi_regr.png",
            "download/pga.jpg",
            "download/pga.pdf",
            "download/pga_regr.png",
            "download/pgv.jpg",
            "download/pgv.pdf",
            "download/pgv_regr.png",
            "download/pin-thumbnail.png",
            "download/psa0p3.jpg",
            "download/psa0p3.pdf",
            "download/psa0p3_regr.png",
            "download/psa0p6.jpg",
            "download/psa0p6.pdf",
            "download/psa0p6_regr.png",
            "download/psa1p0.jpg",
            "download/psa1p0.pdf",
            "download/psa1p0_regr.png",
            "download/psa3p0.jpg",
            "download/psa3p0.pdf",
            "download/psa3p0_regr.png"
        ]
        count = self._run_download_commands("shakemap", file_list, "usgs-figures-versions", version)
        logging.info(f"[get_figures_all] Figure files retrieval completed. Files saved: {count}/{len(file_list)}.")


    def get_shakemaps(self, version="all"):
        """
        Downloads ShakeMap files for the specified event from USGS.
    
        The following files are downloaded:
          - download/grid.xml
          - download/raster.zip
          - download/uncertainty.xml
    
        The files are saved in the directory:
          export/SHAKEfetch/usgs-shakemap-versions
    
        Parameters:
        -----------
        version : str, optional
            Version flag for the download. Allowed values: "last", "all", "first", "preferred".
            Default is "all".
        """
        if not self.earthquake:
            logging.error(f"[get_shakemaps] Earthquake information not available for event ID: {self.event_id}.")
            return
        logging.info(f"[get_shakemaps] Starting ShakeMap download for event ID: {self.event_id}.")
        file_list = [
             "download/grid.xml",
            "download/raster.zip",
            "download/uncertainty.xml"
        ]
        count = self._run_download_commands("shakemap", file_list, "usgs-shakemap-versions", version)
        logging.info(f"[get_shakemaps] Shakemaps retrieval completed. Files saved: {count}/{len(file_list)}.")

    def get_pagers(self, version="all"):
        """
        Downloads Loss Pager products for the specified event from the USGS server.
    
        Files downloaded include:
          - onepager.pdf
          - pager.xml
    
        Saved under: export/SHAKEfetch/usgs-pager-versions.
    
        Parameters:
        -----------
        version : str, optional
            Version flag for the download. Allowed values: "last", "all", "first", "preferred".
            Default is "all".
        """
        if not self.earthquake:
            logging.error(f"[get_pagers] Earthquake information not available for event ID: {self.event_id}.")
            return
        
        try:
            product = self.earthquake.getProducts('losspager')[0]
        except Exception as e:
            logging.error(f"[get_pagers] Error retrieving Loss Pager product for event ID: {self.event_id}: {e}")
            return
        
        logging.info(f"[get_pagers] Checking available pager product properties for event ID: {self.event_id}:")
        for prop in product.properties:
            logging.info(f"[get_pagers] {prop}: {product[prop]}")
        
        logging.info(f"[get_pagers] Starting pager files download for event ID: {self.event_id}.")
        # Define the files to download
        file_list = [
            "onepager.pdf",
            "pager.xml"
        ]
        # Use the helper function to perform the download in the correct directory.
        count = self._run_download_commands("losspager", file_list, "usgs-pager-versions", version)
        logging.info(f"[get_pagers] Pager product retrieval completed. Files saved: {count}/{len(file_list)}.")
    

    def check_shakefetch_inputs(self):
        """
        Fetches and displays information about an earthquake event based on its event ID.

        If no event is found with the specified ID, it defaults to event ID 'us7000m9g4'.

        Returns:
        --------
        None
        """
        logging.info('Searching for event ...')
        try:
            self.earthquake = get_event_by_id(self.event_id)
            if not self.earthquake:
                logging.info(f"No earthquake found with event ID: {self.event_id}. Proceeding with default event ID 'us7000m9g4'.")
                self.event_id = 'us7000m9g4'
                self.earthquake = get_event_by_id(self.event_id)
        except Exception as e:
            logging.error(f"An error occurred while fetching the event: {e}")
            return
        time.sleep(1)
        logging.info('Retrieved Event Information:')
        event_info = {
            'Event Type': self.earthquake.getProducts('shakemap')[0]['event-type'],
            'Event Title': self.earthquake.getProducts('origin')[0]['title'],
            'Event Magnitude': self.earthquake.getProducts('origin')[0]['magnitude'],
            'Event Depth': self.earthquake.getProducts('origin')[0]['depth'],
            'Event Latitude': self.earthquake.getProducts('origin')[0]['latitude'],
            'Event Longitude': self.earthquake.getProducts('origin')[0]['longitude'],
            'Event Event time': self.earthquake.getProducts('origin')[0]['eventtime'],
            'Includes "shakemap" as a product': self.earthquake.hasProduct('shakemap'),
            '[Pager] | Alert level': self.earthquake.getProducts('losspager')[0]['alertlevel'],
            'Max MMI (Pager)': self.earthquake.getProducts('losspager')[0]['maxmmi'],
            'Includes "station" as a property': self.earthquake.hasProperty('station'),
            '[DYFI?] | Nrsp': self.earthquake.getProducts('dyfi')[0]['numResp'],
            'Max MMI (DYFI)': self.earthquake.getProducts('dyfi')[0]['maxmmi']
        }
        for key, value in event_info.items():
            logging.info(f'{key}: {value}')

    def get_origin(self):
        if not self.earthquake:
            logging.error("Earthquake information not available. Please check the event ID.")
            return None
        product = self.earthquake.getProducts('origin')[0]
        logging.info('Retrieving origin data')
        origin_data = {prop: product[prop] for prop in product.properties}
        origin_df = pd.DataFrame([origin_data])
        logging.info(f"Origin data retrieval for event ID: {self.event_id} completed.")
        return origin_df

    def get_moment_tensor(self):
        if not self.earthquake:
            logging.error("Earthquake information not available. Please check the event ID.")
            return None
        product = self.earthquake.getProducts('moment-tensor')[0]
        logging.info('Retrieving moment tensor data')
        moment_tensor_data = {prop: product[prop] for prop in product.properties}
        moment_tensor_df = pd.DataFrame([moment_tensor_data])
        logging.info(f"Moment tensor data retrieval for event ID: {self.event_id} completed.")
        return moment_tensor_df

    def get_focal_mechanism(self):
        if not self.earthquake:
            logging.error("Earthquake information not available. Please check the event ID.")
            return None
        product = self.earthquake.getProducts('focal-mechanism')[0]
        logging.info('Retrieving focal mechanism data')
        focal_mechanism_data = {prop: product[prop] for prop in product.properties}
        focal_mechanism_df = pd.DataFrame([focal_mechanism_data])
        logging.info(f"Focal mechanism data retrieval for event ID: {self.event_id} completed.")
        return focal_mechanism_df

    def get_finite_fault(self):
        if not self.earthquake:
            logging.error("Earthquake information not available. Please check the event ID.")
            return None
        product = self.earthquake.getProducts('finite-fault')[0]
        logging.info('Retrieving finite fault data')
        finite_fault_data = {prop: product[prop] for prop in product.properties}
        finite_fault_df = pd.DataFrame([finite_fault_data])
        logging.info(f"Finite fault data retrieval for event ID: {self.event_id} completed.")
        return finite_fault_df

    def get_phase_data(self):
        if not self.earthquake:
            logging.error("Earthquake information not available. Please check the event ID.")
            return None
        product = self.earthquake.getProducts('phase-data')[0]
        logging.info('Retrieving phase data')
        phase_data = {prop: product[prop] for prop in product.properties}
        phase_data_df = pd.DataFrame([phase_data])
        logging.info(f"Phase data retrieval for event ID: {self.event_id} completed.")
        return phase_data_df

    def get_dyfi(self):
        """
        Retrieves DYFI data for the specified event from USGS and returns a DataFrame
        with the following columns (in order):
          Version, eventsourcecode, depth, latitude, longitude, maxmmi, num-responses, Update Time, Elapsed (sec)

        Returns:
        --------
        pd.DataFrame or None
            DataFrame containing DYFI response information with correct data types.
        """
        if not self.earthquake:
            logging.error(f"[get_dyfi] Earthquake information not available for event ID: {self.event_id}.")
            return None
        try:
            detail = get_event_by_id(self.event_id, includesuperseded=True)
        except Exception as e:
            logging.error(f"[get_dyfi] Error fetching event detail for {self.event_id}: {e}")
            return None
        try:
            dyfi_responses = detail.getProducts('dyfi', source='us', version='all')
        except Exception as e:
            logging.error(f"[get_dyfi] Error fetching DYFI products for {self.event_id}: {e}")
            return None
        if not dyfi_responses:
            logging.warning(f"[get_dyfi] No DYFI products found for event ID: {self.event_id}.")
            return None
        rows = []
        origin = detail.time
        for idx, response in enumerate(dyfi_responses):
            row = {}
            if isinstance(response.properties, list):
                for key in response.properties:
                    try:
                        row[key] = response[key]
                    except Exception as e:
                        logging.debug(f"[get_dyfi] Could not retrieve value for key '{key}' in response {idx}: {e}")
                        row[key] = np.nan
            elif isinstance(response.properties, dict):
                row.update(response.properties)
            else:
                row['properties'] = str(response.properties)
            row['Version'] = response.version
            row['Update Time'] = response.update_time
            try:
                row['Elapsed (sec)'] = (response.update_time - origin).total_seconds()
            except Exception as e:
                logging.debug(f"[get_dyfi] Error calculating elapsed time for response {idx}: {e}")
                row['Elapsed (sec)'] = np.nan
            rows.append(row)
        response_frame = pd.DataFrame(rows)
        desired_columns = [
            "Version",
            "eventsourcecode",
            "depth",
            "latitude",
            "longitude",
            "maxmmi",
            "num-responses",
            "Update Time",
            "Elapsed (sec)"
        ]
        for col in desired_columns:
            if col not in response_frame.columns:
                response_frame[col] = np.nan
        response_frame = response_frame[desired_columns]
        try:
            response_frame['Version'] = response_frame['Version'].astype(int)
        except Exception as e:
            logging.warning(f"Could not convert 'Version' to int: {e}")
        response_frame['eventsourcecode'] = response_frame['eventsourcecode'].astype(str)
        try:
            response_frame['depth'] = response_frame['depth'].astype(float)
        except Exception as e:
            logging.warning(f"Could not convert 'depth' to float: {e}")
        try:
            response_frame['latitude'] = response_frame['latitude'].astype(float)
        except Exception as e:
            logging.warning(f"Could not convert 'latitude' to float: {e}")
        try:
            response_frame['longitude'] = response_frame['longitude'].astype(float)
        except Exception as e:
            logging.warning(f"Could not convert 'longitude' to float: {e}")
        try:
            response_frame['maxmmi'] = response_frame['maxmmi'].astype(float)
        except Exception as e:
            logging.warning(f"Could not convert 'maxmmi' to float: {e}")
        try:
            response_frame['num-responses'] = response_frame['num-responses'].astype(int)
        except Exception as e:
            logging.warning(f"Could not convert 'num-responses' to int: {e}")
        try:
            response_frame['Update Time'] = pd.to_datetime(response_frame['Update Time'], errors='coerce')
        except Exception as e:
            logging.warning(f"Could not convert 'Update Time' to datetime: {e}")
        try:
            response_frame['Elapsed (sec)'] = response_frame['Elapsed (sec)'].astype(float)
        except Exception as e:
            logging.warning(f"Could not convert 'Elapsed (sec)' to float: {e}")
        self.dyfi_data = response_frame
        return response_frame

    def plot_dyfi(self):
        if self.dyfi_data is None:
            logging.error("DYFI data not available. Please retrieve DYFI data first using SHAKEfetch.get_dyfi()")
            return

        response_frame = self.dyfi_data.copy()

        # Convert columns to numeric explicitly
        for col in ['Elapsed (sec)', 'num-responses']:
            response_frame[col] = pd.to_numeric(response_frame[col], errors='coerce')

        # Drop rows with NaN values in critical columns
        response_frame.dropna(subset=['Elapsed (sec)', 'num-responses'], inplace=True)

        if response_frame.empty:
            logging.error("No valid DYFI data to plot after filtering for numeric values.")
            return

        # Plotting
        try:
            hours = response_frame['Elapsed (sec)'] / 3600
            responses = response_frame['num-responses']

            r95 = responses.max() * 0.95
            i95 = np.argmax(responses.values >= r95)
            t95 = hours.iloc[i95]

            fig = plt.figure(figsize=(12, 6))
            plt.plot(hours, responses, 'b.')
            plt.axvline(x=t95, color='red', linestyle='--', label='95% Response Time')

            xmin, xmax, ymin, ymax = plt.axis()
            plt.axis([0, 24, ymin, ymax])
            plt.xlabel('Elapsed time (hours)', fontsize=14)
            plt.ylabel('Number of DYFI responses', fontsize=14)

            plt.grid(which='major', color='dimgray', linestyle=':', linewidth='0.5')
            plt.grid(which='minor', color='dimgray', linestyle=':', linewidth='0.5')

            tstr = f'Responses vs Time ({self.earthquake.id})'
            plt.title(tstr, fontsize=16)
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot DYFI data: {e}")

    def get_dyfi_files(self, version="last"):
        """
        Downloads DYFI product files for the specified event from USGS.
        
        The targeted files (if available) are:
          - dyfi_dat.xml
          - [event_id]_ciim.jpg
          - dyfi_geo_10km.geojson
          - dyfi_geo_1km.geojson
          - cdi_geo.txt
          - contents.xml
          - [event_id]_ciim_geo.jpg
          - [event_id]_plot_atten.jpg
          - cdi_geo_1km.txt
          - dyfi_plot_atten.json
          - cdi_geo.xml
          - cdi_zip.xml
          - cdi_zip.txt
          - event_data.xml
          - dyfi_plot_numresp.json
          - [event_id]_plot_numresp.jpg
          - plot_atten.txt
          - [event_id]_plot_numresp.txt
        
        Files are saved under: export/SHAKEfetch/usgs-dyfi-versions.

        Parameters:
        -----------
        version : str, optional
            Version flag for the download (allowed values: "last", "all", "first", "preferred").
            Default is "last".
        """
        if not self.earthquake:
            logging.error(f"[get_dyfi_files] Earthquake information not available for event ID: {self.event_id}.")
            return
        try:
            product = self.earthquake.getProducts('dyfi')[0]
        except Exception as e:
            logging.error(f"[get_dyfi_files] Error retrieving DYFI product for event ID: {self.event_id}: {e}")
            return
        logging.info(f"[get_dyfi_files] Checking available DYFI product contents for event ID: {self.event_id}.")
        for content in product.contents:
            logging.info(f"[get_dyfi_files] Available DYFI content: {content}")
        base_dir = os.path.join(self.original_dir, 'export', 'SHAKEfetch')
        output_dir = os.path.join(base_dir, "usgs-dyfi-versions")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"[get_dyfi_files] Created directory: {output_dir}")
        else:
            logging.info(f"[get_dyfi_files] Using existing directory: {output_dir}")
        previous_dir = os.getcwd()
        os.chdir(output_dir)
        file_list = [
            "dyfi_dat.xml",
            f"{self.event_id}_ciim.jpg",
            "dyfi_geo_10km.geojson",
            "dyfi_geo_1km.geojson",
            "cdi_geo.txt",
            "contents.xml",
            f"{self.event_id}_ciim_geo.jpg",
            f"{self.event_id}_plot_atten.jpg",
            "cdi_geo_1km.txt",
            "dyfi_plot_atten.json",
            "cdi_geo.xml",
            "cdi_zip.xml",
            "cdi_zip.txt",
            "event_data.xml",
            "dyfi_plot_numresp.json",
            f"{self.event_id}_plot_numresp.jpg",
            "plot_atten.txt",
            f"{self.event_id}_plot_numresp.txt"
        ]
        logging.info(f"[get_dyfi_files] Files to download: {file_list}")
        success_count = 0
        for file in file_list:
            cmd = f"getproduct dyfi {file} -i {self.event_id} --get-version={version}"
            logging.info(f"[get_dyfi_files] Executing command: {cmd}")
            try:
                result = subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True)
                logging.debug(f"[get_dyfi_files] Command output for {file}: {result.stdout}")
                logging.info(f"[get_dyfi_files] Successfully downloaded {file} for event ID: {self.event_id}.")
                success_count += 1
            except subprocess.CalledProcessError as e:
                logging.error(f"[get_dyfi_files] Error executing command: {cmd}")
                logging.error(f"[get_dyfi_files] Error details: {e}. Stderr: {e.stderr}. Skipping {file}.")
        os.chdir(previous_dir)
        logging.info(f"[get_dyfi_files] DYFI files retrieval for event ID: {self.event_id} completed. Files saved: {success_count}/{len(file_list)}.")

    def print_doc(self):
        """Prints the docstring of the SHAKEfetch class."""
        print(self.__doc__)







    # ---------------------------------------------------------------------
    # v25+ EXTENSIONS (additive, safe)
    # Product / version / timeline utilities for SHAKEtime
    # ---------------------------------------------------------------------

    def _get_detail(self, includesuperseded=True):
        try:
            return get_event_by_id(self.event_id, includesuperseded=includesuperseded)
        except Exception as e:
            logging.error(f"[SHAKEfetch] get_event_by_id failed for {self.event_id}: {e}")
            return None

    def _products_map(self, ev):
        products = getattr(ev, "products", None)
        if isinstance(products, dict):
            return products

        mapping = {}
        if isinstance(products, list):
            for p in products:
                ptype = getattr(p, "type", None) or getattr(p, "product_type", None) or getattr(p, "product", None)
                if not ptype:
                    continue
                mapping.setdefault(str(ptype), []).append(p)
        return mapping

    def list_product_types(self, includesuperseded=True):
        ev = self._get_detail(includesuperseded)
        if ev is None:
            return []
        types = sorted(self._products_map(ev).keys())
        logging.info(f"[list_product_types] {self.event_id}: {types}")
        return types

    def list_product_versions(self, product_type, source=None, includesuperseded=True):
        ev = self._get_detail(includesuperseded)
        if ev is None:
            return []
        try:
            prods = ev.getProducts(product_type, source=source, version="all") if source else ev.getProducts(product_type, version="all")
        except Exception as e:
            logging.error(f"[list_product_versions] {product_type}: {e}")
            return []

        versions = []
        for p in prods or []:
            try:
                versions.append(int(p.version))
            except Exception:
                pass
        versions = sorted(set(versions))
        logging.info(f"[list_product_versions] {self.event_id} {product_type}: {versions}")
        return versions

    def list_product_contents(self, product_type, version="preferred", source=None, includesuperseded=True):
        ev = self._get_detail(includesuperseded)
        if ev is None:
            return []

        try:
            if isinstance(version, int):
                prods = ev.getProducts(product_type, source=source, version="all") if source else ev.getProducts(product_type, version="all")
                prod = next((p for p in prods if p.version == version), None)
            else:
                prods = ev.getProducts(product_type, source=source, version=version) if source else ev.getProducts(product_type, version=version)
                prod = prods[0] if prods else None

            if prod is None:
                return []

            contents = prod.contents
            return sorted(contents.keys()) if isinstance(contents, dict) else sorted(contents)

        except Exception as e:
            logging.error(f"[list_product_contents] {product_type}: {e}")
            return []

    def download_product(self, product_type, contents, output_subdir=None, version="preferred"):
        if output_subdir is None:
            output_subdir = f"usgs-{product_type}-versions"

        if isinstance(contents, str):
            contents = [contents]

        logging.info(f"[download_product] {self.event_id} {product_type} ({version})")
        return self._run_download_commands(product_type, contents, output_subdir, version)

    def get_product_timeline(self, product_type, source=None, includesuperseded=True, properties=None):
        ev = self._get_detail(includesuperseded)
        if ev is None:
            return None

        try:
            prods = ev.getProducts(product_type, source=source, version="all") if source else ev.getProducts(product_type, version="all")
        except Exception as e:
            logging.error(f"[get_product_timeline] {product_type}: {e}")
            return None

        rows = []
        for p in prods or []:
            row = {
                "event_id": self.event_id,
                "product_type": product_type,
                "version": p.version,
                "update_time": getattr(p, "update_time", None),
                "source": getattr(p, "source", None),
                "code": getattr(p, "code", None),
                "status": getattr(p, "status", None),
            }
            if properties:
                for k in properties:
                    try:
                        row[k] = p[k]
                    except Exception:
                        row[k] = None
            rows.append(row)

        if not rows:
            return None

        import pandas as pd
        df = pd.DataFrame(rows)
        df["update_time"] = pd.to_datetime(df["update_time"], errors="coerce")
        return df.sort_values(["update_time", "version"], na_position="last")

    def get_source_evolution(self, includesuperseded=True):
        product_props = {
            "origin": ["magnitude","latitude","longitude","depth","eventtime","reviewstatus","location"],
            "moment-tensor": ["scalar-moment","derived-depth","derived-latitude","derived-longitude"],
            "focal-mechanism": [
                "nodal-plane-1-strike","nodal-plane-1-dip","nodal-plane-1-rake",
                "nodal-plane-2-strike","nodal-plane-2-dip","nodal-plane-2-rake"
            ],
            "finite-fault": ["magnitude","depth","latitude","longitude"],
            "phase-data": ["num-phases","num-stations","azimuthal-gap","minimum-distance"],
        }

        out = {}
        for ptype, props in product_props.items():
            out[ptype] = self.get_product_timeline(
                ptype,
                includesuperseded=includesuperseded,
                properties=props
            )
        return out


######################




    def check_shakemap_files(self):
        """
        Retrieves and logs basic information about the event and lists available ShakeMap product content names.
        
        This method uses the event ID to fetch data from ComCat. If the event is not found,
        it raises an exception.
        """
        event_id = self.event_id
        try:
            event = get_event_by_id(event_id)
        except Exception as e:
            logging.error(f"Error retrieving event {event_id}: {e}")
            raise RuntimeError(f"Failed to retrieve event {event_id}.")
    
        if not event:
            logging.error(f"No event found with event ID: {event_id}")
            raise ValueError(f"No event found with event ID: {event_id}")
    
        shakemap_products = event.getProducts('shakemap')
        if not shakemap_products:
            logging.error("No ShakeMap product available for this event.")
            raise ValueError("No ShakeMap product available for this event.")
    
        product = shakemap_products[0]
    
        logging.info("Available content names in the ShakeMap product:")
        for content_name in product.contents:
            logging.info(f" - {content_name}")







    def check_event_files(self, products=None, version="preferred",
                          includesuperseded=True, only_prefix=None):
        """
        Logs detailed information about available ComCat product contents
        for the event, in a verbose, file-by-file style.
    
        Parameters
        ----------
        products : list[str] or None
            Product types to check (e.g. ['shakemap','dyfi','losspager']).
            If None, checks all known/common products.
        version : str or int
            'preferred' | 'last' | 'first' | 'all' or an explicit version number.
        includesuperseded : bool
            Include superseded products (recommended).
        only_prefix : str or None
            If set (e.g. 'download/'), only log contents under this prefix.
            If None, logs the entire product tree.
        """
    
        event_id = self.event_id
    
        # ------------------------------------------------------------------
        # Fetch event (detail-aware)
        # ------------------------------------------------------------------
        try:
            event = get_event_by_id(event_id, includesuperseded=includesuperseded)
            try:
                event = event.getDetail()
            except Exception:
                pass
        except Exception as e:
            logging.error(f"[check_event_files] Error retrieving event {event_id}: {e}")
            raise RuntimeError(f"Failed to retrieve event {event_id}.")
    
        if not event:
            logging.error(f"[check_event_files] No event found with event ID: {event_id}")
            raise ValueError(f"No event found with event ID: {event_id}")
    
        logging.info("===================================================")
        logging.info(f"[check_event_files] Event ID: {event_id}")
        logging.info("===================================================")
    
        # ------------------------------------------------------------------
        # Determine which products to check
        # ------------------------------------------------------------------
        if products is None:
            products = [
                "shakemap","dyfi","losspager",
                "origin","moment-tensor","focal-mechanism",
                "finite-fault","phase-data",
                "impact-text","general-text","tectonic-summary",
                "significance","nearby-cities",
                "scitech-text","scitech-link",
            ]
    
        # ------------------------------------------------------------------
        # Loop over products
        # ------------------------------------------------------------------
        for ptype in products:
            try:
                has_prod = event.hasProduct(ptype)
            except Exception:
                has_prod = False
    
            if not has_prod:
                logging.info(f"[{ptype}] Product NOT available.")
                continue
    
            logging.info("---------------------------------------------------")
            logging.info(f"[{ptype}] Product available.")
    
            # --------------------------------------------------------------
            # Retrieve product versions
            # --------------------------------------------------------------
            try:
                if version == "all":
                    prod_list = event.getProducts(ptype, version="all")
                elif isinstance(version, int):
                    prod_list = event.getProducts(ptype, version="all")
                else:
                    prod_list = event.getProducts(ptype, version=version)
            except Exception as e:
                logging.error(f"[{ptype}] Failed to retrieve products (version={version}): {e}")
                continue
    
            if not prod_list:
                logging.warning(f"[{ptype}] No product instances found.")
                continue
    
            # If explicit version requested, filter
            if isinstance(version, int):
                prod_list = [p for p in prod_list if getattr(p, "version", None) == version]
                if not prod_list:
                    logging.warning(f"[{ptype}] Version {version} not found.")
                    continue
    
            # --------------------------------------------------------------
            # Loop over product instances (versions)
            # --------------------------------------------------------------
            for prod in prod_list:
                v = getattr(prod, "version", "unknown")
                ut = getattr(prod, "update_time", None)
    
                logging.info(f"[{ptype}] Version: {v} | Update time: {ut}")
    
                contents = getattr(prod, "contents", {})
                if isinstance(contents, dict):
                    content_names = sorted(contents.keys())
                else:
                    try:
                        content_names = sorted(list(contents))
                    except Exception:
                        content_names = []
    
                if not content_names:
                    logging.info(f"[{ptype}]   (no contents listed)")
                    continue
    
                logging.info(f"[{ptype}]   Available contents:")
                for name in content_names:
                    if only_prefix and not str(name).startswith(only_prefix):
                        continue
                    logging.info(f"[{ptype}]     - {name}")
    
        logging.info("===================================================")
        logging.info("[check_event_files] Completed.")




    def modular_fetcher(self, plan, default_version="preferred", includesuperseded=True,
                       only_prefix=None, strict=False, dry_run=False):
        """
        Modular downloader controlled by a user-provided dictionary ("plan").
    
        Version modularity:
          - version: "preferred" | "last" | "first" | "all" | int
    
        plan example:
        -------------
        plan = {
          "shakemap": {
            "on": True,
            "version": "all",
            "output_subdir": "usgs-shakemap-versions",
            "files": ["download/grid.xml","download/uncertainty.xml","download/stationlist.json"]
            # OR files="__ALL_DOWNLOAD__"  -> download everything under download/
            # OR files="__ALL__"           -> download entire tree (root + folders)
          },
          "dyfi": {
            "on": True,
            "version": "last",
            "output_subdir": "usgs-dyfi-versions",
            "files": "__ALL__"
          },
          "losspager": {"on": False},
        }
        """
    
        # -------- helpers --------
        def _get_detail():
            try:
                ev = get_event_by_id(self.event_id, includesuperseded=includesuperseded)
                try:
                    return ev.getDetail()
                except Exception:
                    return ev
            except Exception as e:
                msg = f"[modular_fetcher] Failed to fetch event detail for {self.event_id}: {e}"
                logging.error(msg)
                if strict:
                    raise RuntimeError(msg)
                return None
    
        def _list_contents(evd, product_type, version_sel):
            """Return union of content names for the selected version(s)."""
            try:
                if version_sel == "all":
                    prod_list = evd.getProducts(product_type, version="all")
                elif isinstance(version_sel, int):
                    prod_list = evd.getProducts(product_type, version="all")
                else:
                    prod_list = evd.getProducts(product_type, version=version_sel)
            except Exception:
                return []
    
            if not prod_list:
                return []
    
            # If explicit version number requested, pick that one.
            if isinstance(version_sel, int):
                chosen = None
                for pr in prod_list:
                    if getattr(pr, "version", None) == version_sel:
                        chosen = pr
                        break
                if chosen is None:
                    return []
                prod_list = [chosen]
    
            names = set()
            for pr in prod_list:
                contents = getattr(pr, "contents", {})
                if isinstance(contents, dict):
                    keys = contents.keys()
                else:
                    try:
                        keys = list(contents)
                    except Exception:
                        keys = []
                for k in keys:
                    names.add(str(k))
            return sorted(names)
    
        def _normalize_version(v):
            if isinstance(v, str):
                v = v.strip().lower()
            return v
    
        # -------- main --------
        evd = _get_detail()
        if evd is None:
            return {}
    
        results = {}
        logging.info("===================================================")
        logging.info(f"[modular_fetcher] Event: {self.event_id}")
        logging.info("===================================================")
    
        for product_type, cfg in (plan or {}).items():
            cfg = cfg or {}
            on = bool(cfg.get("on", False))
    
            if not on:
                results[product_type] = {
                    "attempted": False,
                    "downloaded_count": 0,
                    "requested_count": 0,
                    "skipped_reason": "off",
                }
                logging.info(f"[modular_fetcher] [{product_type}] OFF (skipping).")
                continue
    
            # Check product existence
            try:
                exists = bool(evd.hasProduct(product_type))
            except Exception:
                exists = False
    
            if not exists:
                msg = f"[modular_fetcher] [{product_type}] Product not available for this event."
                if strict:
                    raise ValueError(msg)
                logging.warning(msg)
                results[product_type] = {
                    "attempted": False,
                    "downloaded_count": 0,
                    "requested_count": 0,
                    "skipped_reason": "product_not_available",
                }
                continue
    
            version_sel = _normalize_version(cfg.get("version", default_version))
            output_subdir = cfg.get("output_subdir", f"usgs-{product_type}-versions")
            files_spec = cfg.get("files", [])
    
            # Expand magic tokens for file selection
            if files_spec in ("__ALL__", "__ALL_DOWNLOAD__"):
                contents_all = _list_contents(evd, product_type, version_sel)
    
                # default behavior: "__ALL_DOWNLOAD__" means only download/*
                if files_spec == "__ALL_DOWNLOAD__":
                    prefix = only_prefix if only_prefix is not None else "download/"
                    selected = [c for c in contents_all if c.startswith(prefix)]
                else:
                    # "__ALL__" means entire tree; optionally filter if only_prefix is set
                    if only_prefix:
                        selected = [c for c in contents_all if c.startswith(only_prefix)]
                    else:
                        selected = contents_all
    
                file_list = selected
    
            else:
                # explicit list of files
                if isinstance(files_spec, str):
                    file_list = [files_spec]
                else:
                    file_list = list(files_spec)
    
                # optional filtering (if user provided both explicit files and only_prefix)
                if only_prefix:
                    file_list = [f for f in file_list if str(f).startswith(only_prefix)]
    
            if not file_list:
                msg = f"[modular_fetcher] [{product_type}] No files selected (nothing to download)."
                if strict:
                    raise ValueError(msg)
                logging.warning(msg)
                results[product_type] = {
                    "attempted": False,
                    "downloaded_count": 0,
                    "requested_count": 0,
                    "skipped_reason": "no_files_selected",
                }
                continue
    
            # ---- log what will happen ----
            logging.info("---------------------------------------------------")
            logging.info(f"[modular_fetcher] [{product_type}] ON")
            logging.info(f"[modular_fetcher] [{product_type}] version={version_sel}, output_subdir={output_subdir}")
            logging.info(f"[modular_fetcher] [{product_type}] files={len(file_list)}")
            for f in file_list[:30]:
                logging.info(f"[modular_fetcher] [{product_type}]   - {f}")
            if len(file_list) > 30:
                logging.info(f"[modular_fetcher] [{product_type}]   ... +{len(file_list)-30} more")
    
            if dry_run:
                results[product_type] = {
                    "attempted": True,
                    "downloaded_count": 0,
                    "requested_count": len(file_list),
                    "skipped_reason": "dry_run",
                }
                logging.info(f"[modular_fetcher] [{product_type}] DRY RUN: no downloads executed.")
                continue
    
            # ---- download via your existing downloader ----
            try:
                count = self._run_download_commands(product_type, file_list, output_subdir, version_sel)
            except Exception as e:
                msg = f"[modular_fetcher] [{product_type}] download failed: {e}"
                if strict:
                    raise RuntimeError(msg)
                logging.error(msg)
                count = 0
    
            results[product_type] = {
                "attempted": True,
                "downloaded_count": int(count) if count is not None else 0,
                "requested_count": len(file_list),
                "skipped_reason": None,
            }
            logging.info(f"[modular_fetcher] [{product_type}] Done. Files saved: {count}/{len(file_list)}")
    
        logging.info("===================================================")
        logging.info("[modular_fetcher] Completed.")
        return results






















