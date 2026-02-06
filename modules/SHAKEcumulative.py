import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import logging

class SHAKEcumulative:
    """
    SHAKEcumulative:
    A class for combining multiple ShakeMap XMLs into interpolated and aggregated ground motion fields.
    
    Features:
    - Metadata extraction & classification of events (mainshock/aftershock/foreshock)
    - ShakeMap grid ingestion and interpolation
    - Standard deviation (uncertainty) map ingestion
    - ShakeMap combination (mean, max, weighted, Bayesian)
    - Extent, exceedance, and plotting support


    Method	Description	Advantage
    Bayesian Model Averaging:	Probabilistic combination using model likelihoods	Better handling of epistemic uncertainty
    Ensemble Learning (ML): 	     Regression over multiple event ShakeMaps	Learns nonlinear patterns
    Robust Aggregation:	Uses statistics less sensitive to outliers	Reliable when some maps are wrong
    Space-Time Kernel:	Weighs by proximity and recency	Intuitive and smooth decay
    Empirical Fusion:	Learns from observed ShakeMap residuals	Real-world correction
    Monte Carlo Ensemble:	Generates ground motion samples	Full uncertainty propagation




    Add soil consideration
    
    """
    def __init__(self):
        # Metadata
        self.metadata_df = None
        self._registry = None
        self._main_event_time = None

        # ShakeMap data
        self._ingest_metric = None
        self._event_raw = {}
        self._event_base = None
        self._event_stack = {}
        self._event_std_stack = {}
        self._event_order = []
        self._event_extents = {}
        self._event_spacing = None
        self._event_svel = {}  # <- SVEL cache here
        self._failed_std_ingestions = []

        # Output
        self.combined_grid = None




    def get_combined_data(self):
        """
        Return the current cached combined grid if available.
        """
        if self.combined_grid is not None:
            logging.info("Returning cached combined ShakeMap grid.")
            return self.combined_grid
        logging.warning("No combined ShakeMap grid available.")
        return pd.DataFrame()

    def import_combined_shakemap(self, csv_file: str) -> pd.DataFrame:
        """
        Load a CSV of previously combined ShakeMap data into the class.

        Parameters
        ----------
        csv_file : str
            Path to the combined grid CSV file (must include lon, lat columns)

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame, also cached in self.combined_grid
        """
        try:
            df = pd.read_csv(csv_file)
            if not {'lon', 'lat'}.issubset(df.columns):
                raise ValueError("CSV must contain 'lon' and 'lat' columns.")
            self.combined_grid = df
            logging.info(f"Loaded ShakeMap CSV with {len(df)} rows.")
            return df
        except Exception as e:
            logging.error(f"Failed to load CSV: {e}")
            return pd.DataFrame()


    def parse_shakemap_metadata(self, event: dict) -> dict:
        """
        Parse basic metadata from a ShakeMap grid XML file.
    
        Parameters
        ----------
        event : dict
            Dictionary with keys like 'file_path', 'event_id', 'event_type'.
    
        Returns
        -------
        dict
            Parsed metadata including timestamp, location, magnitude, etc.
    
        Raises
        ------
        FileNotFoundError or ET.ParseError if XML cannot be read or parsed.
        """
        
        xml_file = event['file_path']
        root = ET.parse(xml_file).getroot()
        ns = {'sm': 'http://earthquake.usgs.gov/eqcenter/shakemap'}
        ev = root.find('sm:event', ns)
        if ev is None:
            raise ValueError("Missing <event> tag")
        return {
            'event_id': event['event_id'],
            'event_type': event['event_type'],
            'event_timestamp': ev.attrib.get('event_timestamp'),
            'magnitude': float(ev.attrib.get('magnitude', 'nan')),
            'depth': float(ev.attrib.get('depth', 'nan')),
            'lat': float(ev.attrib.get('lat', 'nan')),
            'lon': float(ev.attrib.get('lon', 'nan')),
        }

    def build_metadata_table(self, registry: list[dict]) -> pd.DataFrame:
        """
        Build a metadata DataFrame from ShakeMap registry.
    
        Each event is parsed from its XML ShakeMap file to extract time, location,
        magnitude, and other metadata. Events are classified as 'main', 'aftershock',
        or 'foreshock' based on their timestamp relative to the main event.
    
        Parameters
        ----------
        registry : list of dict
            List of event metadata dictionaries including 'file_path' and 'event_type'.
    
        Returns
        -------
        pd.DataFrame
            Metadata table including shock classification.
        """
        records = []
        for rec in registry:
            try:
                m = self.parse_shakemap_metadata(rec)
                records.append(m)
            except Exception as e:
                logging.warning(f"Failed to parse metadata for {rec.get('event_id')}: {e}")
        df = pd.DataFrame(records)
        main = df[df['event_type'] == 'main_event']
        self._main_event_time = pd.to_datetime(main.iloc[0]['event_timestamp']) if not main.empty else None
        def classify(r):
            if r['event_type'] == 'main_event':
                return 'main'
            ts = pd.to_datetime(r['event_timestamp'])
            return 'aftershock' if ts >= self._main_event_time else 'foreshock'
        df['shock_type'] = df.apply(classify, axis=1)
        self.metadata_df = df
        return df
    
        
    def _parse_shake_xml(self, file_path, metric, ns):
        '''
        Internal: Parse grid XML file into DataFrame, store all fields (e.g., PGA, MMI, SVEL).
    
        Parameters
        ----------
        file_path : str
            Path to the ShakeMap grid XML file.
        metric : str
            The intensity measure used for the primary analysis (e.g., 'mmi').
        ns : dict
            Namespace dictionary for XML parsing.
    
        Returns
        -------
        pd.DataFrame
            DataFrame containing all grid fields (e.g., MMI, PGA, PGV, SVEL).
        '''
        root = ET.parse(file_path).getroot()
        fields = root.findall('sm:grid_field', ns)
        names = [f.attrib['name'].lower() for f in fields]
        data_text = root.find('sm:grid_data', ns).text
        data_values = [float(x) for x in data_text.split()]
        df = pd.DataFrame(np.array(data_values).reshape(-1, len(names)), columns=names)
    
        # Store spacing and extent
        spec = root.find('sm:grid_specification', ns)
        if spec is not None:
            self._event_spacing = float(spec.attrib.get('nominal_lon_spacing', 0.01))
            self._event_extents[file_path] = [
                float(spec.attrib['lon_min']), float(spec.attrib['lon_max']),
                float(spec.attrib['lat_min']), float(spec.attrib['lat_max'])
            ]
    
        return df

    def _parse_std_xml(self, std_path, metric, base_df, ns):
        """
        Internal: Parse std grid file and interpolate to the base ShakeMap grid.

        Parameters
        ----------
        std_path : str
        metric : str
        base_df : pd.DataFrame
        ns : dict

        Returns
        -------
        np.ndarray or None
        """
        try:
            root = ET.parse(std_path).getroot()
            fields = root.findall('sm:grid_field', ns)
            names = [f.attrib['name'].lower() for f in fields]
            data = [float(x) for x in root.find('sm:grid_data', ns).text.split()]
            df = pd.DataFrame(np.array(data).reshape(-1, len(names)), columns=names)
            std_col = f"std{metric.lower()}"
            if std_col not in df.columns:
                return None
            pts = df[['lon', 'lat']].values
            vals = df[std_col].values
            xi = base_df[['lon', 'lat']].values
            return griddata(pts, vals, xi, method='linear')
        except Exception as e:
            logging.warning(f"Could not read or interpolate std file {std_path}: {e}")
            return None
            

    def ingest_event_shakemaps(self, registry, metric='mmi', interp_method='linear'):
        """
        Ingest ShakeMap XMLs and interpolate data to the main event grid.
    
        Parameters
        ----------
        registry : list of dict
            List of event ShakeMap metadata dictionaries.
        metric : str, default 'mmi'
            Intensity type to ingest (e.g., 'mmi', 'pga').
        interp_method : str, default 'linear'
            Interpolation method for aftershock ShakeMaps.
        """
        self._registry = registry
        self._ingest_metric = metric
        ns = {'sm': 'http://earthquake.usgs.gov/eqcenter/shakemap'}
        self._event_vs30_stack = {}
        self._failed_std_ingestions = []
    
        # Load main event
        mains = [r for r in registry if r['event_type'] == 'main_event']
        if len(mains) != 1:
            raise ValueError("Exactly one main event must be specified.")
        main = mains[0]
    
        try:
            logging.info(f"🔹 Ingesting main event: {main['event_id']}")
            main_df = self._parse_shake_xml(main['file_path'], metric, ns)
            self._event_raw[main['event_id']] = main_df
            self._event_base = main_df[['lon', 'lat']].reset_index(drop=True)
            self._event_stack[main['event_id']] = main_df[metric].values
            self._event_order = [main['event_id']]
    
            # Store Vs30 for main event
            if 'svel' in main_df.columns:
                self._event_vs30_stack[main['event_id']] = main_df['svel'].values
    
            if 'std_file_path' in main:
                std_vals = self._parse_std_xml(main['std_file_path'], metric, self._event_base, ns)
                self._event_std_stack[main['event_id']] = std_vals
    
        except Exception as e:
            logging.error(f"❌ Critical: Failed to ingest main event {main['event_id']}: {e}")
            return  # Cannot proceed without main event
    
        # Process aftershocks
        for rec in registry:
            if rec['event_type'] != 'aftershock':
                continue
            eid = rec['event_id']
            try:
                logging.info(f"🔸 Ingesting aftershock: {eid}")
                df = self._parse_shake_xml(rec['file_path'], metric, ns)
                self._event_raw[eid] = df
                self._event_order.append(eid)
    
                xi = self._event_base[['lon', 'lat']].values
                pts = df[['lon', 'lat']].values
                vals = griddata(pts, df[metric].values, xi, method=interp_method)
                self._event_stack[eid] = vals
    
                # Cache Vs30
                if 'svel' in df.columns:
                    vs_interp = griddata(pts, df['svel'].values, xi, method=interp_method)
                    self._event_vs30_stack[eid] = vs_interp
    
                # Try std map
                if 'std_file_path' in rec:
                    try:
                        std_vals = self._parse_std_xml(rec['std_file_path'], metric, self._event_base, ns)
                        self._event_std_stack[eid] = std_vals
                    except Exception as e:
                        logging.warning(f"⚠️ Failed to parse std for aftershock {eid}: {e}")
                        self._event_std_stack[eid] = None
                        self._failed_std_ingestions.append(eid)
    
            except Exception as e:
                logging.error(f"❌ Failed to ingest aftershock {eid}: {e}")
                continue
    
        logging.info(f"✅ ShakeMap ingestion complete. Events: {len(self._event_order)}")
        if self._failed_std_ingestions:
            logging.warning(f"⚠️ STD ingestion failed for events: {self._failed_std_ingestions}")
    
        

    def combine_event_shakemaps(
        self,
        registry=None,
        metric=None,
        extent='main',
        grid_res=None,
        interp_method='nearest',
        use_cache=True,
        write_std_to_df=False,
        soil_consideration=False,
        soil_method=None,
        gmice_model='WordenEtAl12'
    ):
        """
        Combine all ingested ShakeMap event grids into a single spatial DataFrame.
    
        Parameters
        ----------
        registry : list, optional
            ShakeMap registry to ingest (if not already loaded).
        metric : str, optional
            The intensity metric to process (default is 'mmi').
        extent : str or list, default 'main'
            Specifies target grid: 'main', 'union', or a bounding box [lon_min, lon_max, lat_min, lat_max].
        grid_res : float, optional
            Grid resolution in degrees for interpolation.
        interp_method : str, default 'nearest' 
            Interpolation method for scipy's griddata.
        use_cache : bool, default True
            If True, use previously cached grid if available.
        write_std_to_df : bool, default False
            Whether to write standard deviation columns into output DataFrame.
        soil_consideration : bool, default False
            Whether to apply soil-to-rock and rock-to-site conversion logic.
        soil_method : str, optional
            One of {'usgs', 'ec8', 'ems-98', 'mmi_hybrid'}
        gmice_model : str, optional
            GMICE model to use if needed for intensity conversion (used in 'mmi_hybrid')
        """
    
        metric = metric or self._ingest_metric or 'mmi'
        if not self._event_order:
            registry = registry or self._registry
            self.build_metadata_table(registry)
            self.ingest_event_shakemaps(registry, metric, interp_method)
    
        if use_cache and self.combined_grid is not None:
            logging.info("Returning cached combined grid.")
            return self.combined_grid
    
        acc_converter = AccelerationUnitConverter()
        soil_converter = SiteAmplificationConverter()
    
        # Create or get the base grid
        if extent == 'main':
            grid_df = self._event_base.copy()
            if 'svel' in self._event_raw[self._event_order[0]].columns:
                grid_df['vs30'] = self._event_raw[self._event_order[0]]['svel'].values
            else:
                grid_df['vs30'] = np.full(len(grid_df), np.nan)
        else:
            exts = np.array(list(self._event_extents.values()))
            lon_min, lon_max = exts[:, 0].min(), exts[:, 1].max()
            lat_min, lat_max = exts[:, 2].min(), exts[:, 3].max()
            dr = grid_res or self._event_spacing
            LONS = np.arange(lon_min, lon_max + dr, dr)
            LATS = np.arange(lat_min, lat_max + dr, dr)
            lon_g, lat_g = np.meshgrid(LONS, LATS)
            grid_df = pd.DataFrame({'lon': lon_g.ravel(), 'lat': lat_g.ravel()})
            grid_df['vs30'] = self.interpolate_vs30(grid_df[['lon', 'lat']])
    
        for eid in self._event_order:
            raw_df = self._event_raw[eid]
            pts = self._event_base[['lon', 'lat']].values
            xi = grid_df[['lon', 'lat']].values
            shock_type = self.metadata_df.loc[self.metadata_df['event_id'] == eid, 'shock_type'].values[0]
    
            raw_vals = self._event_stack[eid]
    
            # Default interpolation
            vals = griddata(pts, raw_vals, xi, method=interp_method)
    
            if soil_consideration:
                if 'svel' not in raw_df.columns:
                    logging.warning(f"⚠️ No SVEL data for event {eid}; skipping soil correction.")
                else:
                    source_vs30 = griddata(pts, raw_df['svel'].values, xi, method=interp_method)
                    target_vs30 = grid_df['vs30'].values
    
                    if soil_method == 'mmi_hybrid':
                        if 'pga' not in raw_df.columns:
                            logging.warning(f"⚠️ No PGA data in ShakeMap for event {eid}. Cannot apply mmi_hybrid.")
                            continue
    
                        raw_pga_vals = griddata(pts, raw_df['pga'].values, xi, method=interp_method)
                        pga_cmps2 = acc_converter.convert_unit(raw_pga_vals, '%g', 'cm/s2')
    
                        # Step 1: deamplify to rock
                        pga_rock = np.array([
                            soil_converter.convert(p, v_src, 760, measure='PGA', method='USGS')
                            for p, v_src in zip(pga_cmps2, source_vs30)
                        ])
    
                        # Step 2: reamplify to target
                        pga_target = np.array([
                            soil_converter.convert(p, 760, v_tgt, measure='PGA', method='USGS')
                            for p, v_tgt in zip(pga_rock, target_vs30)
                        ])
    
                        # Step 3: convert PGA to MMI using GMICE
                        gmice_mmi = GMICE(model=gmice_model, input_value=pga_target, input_type='PGA', output_type='MMI')
                        vals = gmice_mmi.result
    
            grid_df[f"{metric}_{eid}"] = vals
    
            if eid in self._event_std_stack and self._event_std_stack[eid] is not None and write_std_to_df:
                std_vals = self._event_std_stack[eid]
                grid_df[f"std_{metric}_{eid}"] = (
                    std_vals if extent == 'main' else griddata(pts, std_vals, xi, method=interp_method)
                )
    
        self.combined_grid = grid_df
        logging.info("✅ Combined ShakeMap grid has been built.")
        return grid_df
    



    def get_vs30_grid(self, lon, lat, vs30_filepath: str = './SHAKEdata/global_vs30.grd') -> np.ndarray:
        """
        Interpolate global VS30 grid onto a set of lon/lat points.
    
        Parameters
        ----------
        lon : np.ndarray
            Longitudes (1D or flattened array).
        lat : np.ndarray
            Latitudes (1D or flattened array).
        vs30_filepath : str, default './SHAKEdata/global_vs30.grd'
            Path to the global Vs30 .grd file (or GeoTIFF).
    
        Returns
        -------
        np.ndarray
            Interpolated Vs30 values at the input coordinates.
        """
        try:
            with rasterio.open(vs30_filepath) as src:
                vs30_data = src.read(1)
                bounds = src.bounds
                transform = src.transform
    
                # Build lat/lon arrays from raster grid
                nx, ny = src.width, src.height
                lon_vals = np.linspace(bounds.left, bounds.right, nx)
                lat_vals = np.linspace(bounds.top, bounds.bottom, ny)
    
                # Flip lat values to match image origin
                lat_vals = lat_vals[::-1]
    
                # Interpolator over regular grid
                interpolator = RegularGridInterpolator(
                    (lat_vals, lon_vals),
                    vs30_data,
                    bounds_error=False,
                    fill_value=np.nan
                )
    
                # Prepare points for interpolation
                target_pts = np.column_stack([lat, lon])
                vs30_interp = interpolator(target_pts)
    
            logging.info(f"Interpolated VS30 for {len(lon)} points using: {vs30_filepath}")
            return vs30_interp
    
        except Exception as e:
            logging.error(f"Failed to interpolate VS30 from file '{vs30_filepath}': {e}")
            return np.full_like(lon, np.nan)
    

        

    def get_custom_grid(self, extent: list, spacing: float) -> pd.DataFrame:
        """
        Create a regular lon-lat grid for evaluation over the specified extent.

        Parameters
        ----------
        extent : list
            Bounding box [lon_min, lon_max, lat_min, lat_max]
        spacing : float
            Grid spacing in degrees (e.g., 0.01)

        Returns
        -------
        pd.DataFrame
            Grid with 'lon', 'lat', and interpolated 'vs30' columns
        """
        lon_min, lon_max, lat_min, lat_max = extent
        LONS = np.arange(lon_min, lon_max + spacing, spacing)
        LATS = np.arange(lat_min, lat_max + spacing, spacing)
        lon_g, lat_g = np.meshgrid(LONS, LATS)
        grid_df = pd.DataFrame({'lon': lon_g.ravel(), 'lat': lat_g.ravel()})

        try:
            grid_df['vs30'] = self.get_vs30_grid(grid_df)
            logging.info("✅ Added interpolated Vs30 column to generated grid.")
        except Exception as e:
            logging.warning(f"⚠️ Could not interpolate Vs30 onto grid: {e}")
            grid_df['vs30'] = np.nan

        return grid_df









    

    def available_plots(self) -> list:
        """
        Return a list of plot-ready columns from the combined ShakeMap grid.

        Returns
        -------
        list
            Column names that can be plotted (excluding 'lon' and 'lat').
        """
        if self.combined_grid is not None:
            return [col for col in self.combined_grid.columns if col not in ('lon', 'lat')]
        logging.warning("No combined grid available.")
        return []

    def plot_shakemap(self, shakemap: str, imt_type: str = "mmi", zorder: int = 8, plot_colorbar: bool = True):
        """
        Plot a specific ShakeMap column using the SHAKEmapper visualization engine.

        Parameters
        ----------
        shakemap : str
            Column name in the combined grid to plot (e.g., 'mmi_us7000pn9s').
        imt_type : str, default "mmi"
            Intensity Measure Type (used for labels and color scale).
        zorder : int, default 8
            Layer order on map plot.
        plot_colorbar : bool, default True
            Whether to show a colorbar on the plot.

        Returns
        -------
        Object
            Output from the SHAKEmapper plotting utility.
        """
        if self.combined_grid is None:
            logging.error("No combined grid available. Run combine_event_shakemaps first.")
            return

        if shakemap not in self.combined_grid.columns:
            logging.error(f"'{shakemap}' not found in combined grid. Available columns: {self.available_plots()}")
            return


        mapper = SHAKEmapper()
        mapper.create_basemap()
        return mapper.add_shakemap(
            self.combined_grid['lon'],
            self.combined_grid['lat'],
            self.combined_grid[shakemap],
            imt_type=imt_type,
            zorder=zorder,
            plot_colorbar="on" if plot_colorbar else "off"
        )

    def get_data_extent(self, column_name: str) -> dict:
        """
        Compute the geographic extent of valid (non-NaN) data for a specific column.

        Parameters
        ----------
        column_name : str
            Name of the column in combined grid to evaluate.

        Returns
        -------
        dict
            Dictionary with bounding box coordinates of valid data:
            {
                'lon_min': float,
                'lon_max': float,
                'lat_min': float,
                'lat_max': float
            }

        Raises
        ------
        ValueError
            If the column is missing or has no valid data.
        """
        if self.combined_grid is None:
            raise ValueError("Combined ShakeMap grid not initialized.")
        if column_name not in self.combined_grid.columns:
            raise ValueError(f"Column '{column_name}' not found in combined grid.")

        vals = self.combined_grid[column_name]
        mask = vals.notna()
        if not mask.any():
            raise ValueError(f"Column '{column_name}' contains no valid (non-NaN) values.")

        lon_valid = self.combined_grid.loc[mask, 'lon']
        lat_valid = self.combined_grid.loc[mask, 'lat']

        return {
            'lon_min': lon_valid.min(),
            'lon_max': lon_valid.max(),
            'lat_min': lat_valid.min(),
            'lat_max': lat_valid.max()
        }




    def get_mean_shakemap(self) -> pd.DataFrame:
        """
        Compute and add the cumulative mean of all event intensity values per grid point.
        
        Returns
        -------
        pd.DataFrame
            Updated combined grid with 'cumulative_mean' column.
        """
        if self.combined_grid is None:
            raise ValueError("Run combine_event_shakemaps() first.")

        metric_cols = [col for col in self.combined_grid.columns if col.startswith(f"{self._ingest_metric}_")]
        if not metric_cols:
            raise ValueError("No event intensity columns found.")

        self.combined_grid['cumulative_mean'] = self.combined_grid[metric_cols].mean(axis=1, skipna=True)
        logging.info("Added 'cumulative_mean' column to combined grid.")
        return self.combined_grid

    def get_max_shakemap(self) -> pd.DataFrame:
        """
        Compute and add the cumulative maximum of all event intensity values per grid point.

        Returns
        -------
        pd.DataFrame
            Updated combined grid with 'cumulative_max' column.
        """
        if self.combined_grid is None:
            raise ValueError("Run combine_event_shakemaps() first.")

        metric_cols = [col for col in self.combined_grid.columns if col.startswith(f"{self._ingest_metric}_")]
        if not metric_cols:
            raise ValueError("No event intensity columns found.")

        self.combined_grid['cumulative_max'] = self.combined_grid[metric_cols].max(axis=1, skipna=True)
        logging.info("Added 'cumulative_max' column to combined grid.")
        return self.combined_grid

    def get_rms_shakemap(self, column_prefix="mmi"):
        """
        Compute and add the root-mean-square (RMS) of ShakeMap values across events.

        Parameters
        ----------
        column_prefix : str, default "mmi"
            Column prefix to identify event-specific intensity columns (e.g., 'mmi_us7000pn9s').

        Returns
        -------
        None
            Adds 'cumulative_rms' column to combined grid.
        """
        if self.combined_grid is None:
            raise ValueError("Run combine_event_shakemaps() first.")

        cols = [col for col in self.combined_grid.columns if col.startswith(f"{column_prefix}_")]
        squared = self.combined_grid[cols] ** 2
        self.combined_grid["cumulative_rms"] = np.sqrt(squared.mean(axis=1, skipna=True))
        logging.info("Added 'cumulative_rms' column to combined grid.")

    def get_quantile_shakemap(self, quantile=0.75, column_prefix="mmi"):
        """
        Compute and add a quantile-based estimate of intensity at each grid point.

        Parameters
        ----------
        quantile : float, default 0.75
            Quantile to compute (e.g., 0.75 = 75th percentile).
        column_prefix : str, default "mmi"
            Column prefix to identify event-specific intensity columns.

        Returns
        -------
        None
            Adds a column like 'cumulative_q75' to the combined grid.
        """
        if self.combined_grid is None:
            raise ValueError("Run combine_event_shakemaps() first.")

        cols = [col for col in self.combined_grid.columns if col.startswith(f"{column_prefix}_")]
        name = f"cumulative_q{int(quantile * 100)}"
        self.combined_grid[name] = self.combined_grid[cols].quantile(q=quantile, axis=1, numeric_only=True)
        logging.info(f"Added '{name}' column to combined grid.")

    def get_exceedance_count(self, threshold: float, column_prefix: str = "mmi"):
        """
        Count how many events exceed a given threshold per grid point.

        Parameters
        ----------
        threshold : float
            Intensity threshold.
        column_prefix : str, default "mmi"
            Prefix for ShakeMap columns (e.g., "mmi").

        Returns
        -------
        None
            Adds column 'exceedance_count_{threshold}' to combined grid.
        """
        if self.combined_grid is None:
            raise ValueError("Run combine_event_shakemaps() first.")

        cols = [col for col in self.combined_grid.columns if col.startswith(f"{column_prefix}_")]
        values = self.combined_grid[cols].to_numpy()
        exceedances = (values >= threshold).astype(float)
        exceedances[np.isnan(values)] = 0  # Ignore NaNs
        count = exceedances.sum(axis=1)

        colname = f"exceedance_count_{threshold:.1f}"
        self.combined_grid[colname] = count
        logging.info(f"Added '{colname}' to combined grid.")

    def get_probability_of_exceedance(self, threshold: float, column_prefix="mmi"):
        """
        Compute and add the probability of exceedance across events.

        Parameters
        ----------
        threshold : float
            Threshold value to check.
        column_prefix : str, default "mmi"
            Prefix for ShakeMap intensity columns.

        Returns
        -------
        None
            Adds column like 'poe_{threshold}' to combined grid.
        """
        if self.combined_grid is None:
            raise ValueError("Run combine_event_shakemaps() first.")

        cols = [col for col in self.combined_grid.columns if col.startswith(f"{column_prefix}_")]
        values = self.combined_grid[cols].to_numpy()

        exceed = (values > threshold).astype(float)
        exceed[np.isnan(values)] = np.nan  # NaNs should not count

        count_exceed = np.nansum(exceed, axis=1)
        count_valid = np.sum(~np.isnan(values), axis=1)

        poe = np.divide(count_exceed, count_valid, out=np.full_like(count_exceed, np.nan), where=count_valid != 0)
        colname = f"poe_{threshold:.1f}"
        self.combined_grid[colname] = poe
        logging.info(f"Added '{colname}' (probability of exceedance) to combined grid.")








    def haversine(lon1, lat1, lon2, lat2):
        """
        Compute the great-circle distance between two points on Earth in kilometers.
        """
        R = 6371  # Earth radius in km
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))
    
    def get_weighted_mean_shakemap(
        self,
        weights: dict = None,
        column_prefix: str = "mmi",
        use_magnitude: bool = False,
        use_time_decay: bool = False,
        use_distance: bool = False,
        decay_half_life_hours: float = 48.0
    ):
        """
        Compute a weighted mean ShakeMap and append the result to the combined grid.
    
        Parameters
        ----------
        weights : dict, optional
            Custom manual weights keyed by event_id.
        column_prefix : str, default "mmi"
            ShakeMap intensity type (e.g., 'mmi', 'pga').
        use_magnitude : bool
            Whether to include magnitude-based weights.
        use_time_decay : bool
            Whether to apply time-based exponential decay.
        use_distance : bool
            Whether to apply distance decay from the main event epicenter.
        decay_half_life_hours : float
            Time decay half-life in hours.
        """
        if self.combined_grid is None:
            raise ValueError("Run combine_event_shakemaps first.")
        if self.metadata_df is None:
            raise ValueError("Metadata table is required.")
    
        cols = [c for c in self.combined_grid.columns if c.startswith(f"{column_prefix}_")]
        event_ids = [c.split("_")[-1] for c in cols]
        weights_arr = np.ones(len(event_ids))
    
        weight_sources = []
    
        # Manual Weights
        if weights:
            for i, eid in enumerate(event_ids):
                weights_arr[i] *= weights.get(eid, 1.0)
            weight_sources.append("weighted")
    
        # Magnitude Weights
        if use_magnitude:
            mag_lookup = self.metadata_df.set_index("event_id")["magnitude"].to_dict()
            for i, eid in enumerate(event_ids):
                mag = mag_lookup.get(eid)
                if mag is not None and not np.isnan(mag):
                    weights_arr[i] *= mag
            weight_sources.append("mweighted")
    
        # Time Decay
        if use_time_decay and self._main_event_time is not None:
            times = self.metadata_df.set_index("event_id")["event_timestamp"]
            times = pd.to_datetime(times, errors="coerce")
            delta_hrs = (times - self._main_event_time).dt.total_seconds().div(3600)
            decay_weights = np.exp2(-delta_hrs / decay_half_life_hours)
            for i, eid in enumerate(event_ids):
                decay_w = decay_weights.get(eid)
                if decay_w is not None and not np.isnan(decay_w):
                    weights_arr[i] *= decay_w
            weight_sources.append("tweighted")
    
        # Distance Decay
        if use_distance:
            meta = self.metadata_df.set_index("event_id")
            main_lat = meta.loc[meta['shock_type'] == 'main', 'lat'].values[0]
            main_lon = meta.loc[meta['shock_type'] == 'main', 'lon'].values[0]
            for i, eid in enumerate(event_ids):
                try:
                    lat2, lon2 = meta.loc[eid, ['lat', 'lon']]
                    dist = haversine(main_lon, main_lat, lon2, lat2)
                    decay = 1 / (dist + 1e-3)  # Avoid division by zero
                    weights_arr[i] *= decay
                except Exception:
                    continue
            weight_sources.append("dweighted")
    
        if not weight_sources:
            raise ValueError("No weighting method selected.")
    
        # Final column name
        colname = f"cumulative_{'_'.join(weight_sources)}_mean"
    
        # Compute weighted mean, NaN-safe
        values = self.combined_grid[cols].to_numpy()
        valid_mask = ~np.isnan(values)
        weighted_values = np.where(valid_mask, values * weights_arr, 0.0)
        sum_weights = np.where(valid_mask, weights_arr, 0.0).sum(axis=1)
    
        self.combined_grid[colname] = np.divide(
            weighted_values.sum(axis=1),
            sum_weights,
            out=np.full_like(sum_weights, np.nan),
            where=sum_weights != 0
        )
        logging.info(f"Added '{colname}' column to combined grid.")



        

    def get_data_extent(self, column_name: str) -> dict:
        """
        Get the geographic bounding box where a column (e.g., intensity) has valid data.

        Parameters
        ----------
        column_name : str
            Name of the column in the combined grid to inspect.

        Returns
        -------
        dict
            Dictionary with keys: lon_min, lon_max, lat_min, lat_max.
        """
        if self.combined_grid is None:
            raise ValueError("Combined grid not initialized.")
        if column_name not in self.combined_grid:
            raise ValueError(f"Column '{column_name}' not found.")

        mask = self.combined_grid[column_name].notna()
        if not mask.any():
            raise ValueError(f"Column '{column_name}' contains no valid data.")

        lon_valid = self.combined_grid.loc[mask, 'lon']
        lat_valid = self.combined_grid.loc[mask, 'lat']

        extent = {
            'lon_min': lon_valid.min(),
            'lon_max': lon_valid.max(),
            'lat_min': lat_valid.min(),
            'lat_max': lat_valid.max()
        }
        logging.info(f"Computed extent for '{column_name}': {extent}")
        return extent


        

    def get_bayesian_shakemap(
        self,
        column_prefix: str = "mmi",
        mode: str = "auto",
        default_sigma: float = 0.6,
        use_bma: bool = False,
        bma_weight_mode: str = "uniform",  # options: "uniform", "magnitude", "distance"
        std_event_id: str = None
    ):
        """
        Compute a Bayesian hierarchical estimate with optional Bayesian Model Averaging (BMA).
    
        Parameters
        ----------
        column_prefix : str
            Column prefix (e.g., 'mmi', 'pga').
        mode : str
            One of {'main_only', 'all_events', 'auto', 'model_default', 'version_diff', 'gmm_residual'}.
        default_sigma : float
            Fallback standard deviation if mode='model_default'.
        use_bma : bool
            If True, apply Bayesian Model Averaging (BMA) weighting.
        bma_weight_mode : str
            BMA weighting mode: 'uniform', 'magnitude', or 'distance'.
        std_event_id : str
            If provided, use this event's std for all events (forced std control).




        Extended Bayesian ShakeMap Fusion Options (Planned and Proposed)
        ------------------------------------------------------------------
        
        The current Bayesian shakemap fusion supports:
          - Inverse-variance weighting
          - Bayesian Model Averaging (BMA)
          - Support for main-only, all-events, or model-default sigma usage
          - Optional event-specific standard deviation selection
        
        Future Enhancements & Advanced Options (to implement):
        
        1. Full Posterior Sampling (Bayesian MCMC)
           - Sample full posterior distributions per grid point.
           - Enables confidence intervals and threshold exceedance probability maps.
        
        2. Spatial Correlation Modeling
           - Account for spatial autocorrelation using:
             - Gaussian Processes (GP)
             - Kriging / geostatistical models
             - Conditional Autoregressive (CAR) models
        
        3. Metadata-Based Prior Weighting
           - Use priors derived from:
             - Station count
             - ShakeMap status (e.g. 'automatic' vs 'reviewed')
             - Magnitude or DYFI confidence
           - Incorporate into BMA as model prior probabilities.
        
        4. Epistemic vs Aleatory Uncertainty Separation
           - Distinguish between:
             - Aleatory variability (natural randomness)
             - Epistemic variability (modeling limitations)
           - Compute total sigma as:
             σ²_total = σ²_aleatory + σ²_epistemic
        
        5. Version-Weighted Averaging
           - Use differences between multiple ShakeMap versions:
             - Estimate intra-event std (as model disagreement)
             - Prefer newer or manually reviewed versions
        
        6. Robust Bayesian Fusion
           - Replace Gaussian likelihood with Student’s t-distribution:
             - More resilient to outliers or biased ShakeMaps
             - Useful in low-confidence or sparse data regions
        
        7. Ensemble Filtering Methods
           - Use techniques like:
             - Ensemble Kalman Filter (EnKF)
             - Particle filters
           - Dynamically assimilate new ShakeMaps over time.
        
        8. Grid-Adaptive Fusion
           - Adjust uncertainty or fusion strategy per grid cell:
             - High station count → more confident
             - Sparse areas → increase std or fallback to priors
        
        9. DYFI Integration
           - Incorporate “Did You Feel It?” observations:
             - As pseudo-ShakeMaps with higher uncertainty
             - As priors or constraints on the posterior
        
        10. Multi-IMT Joint Modeling
            - Combine MMI, PGA, PGV simultaneously using:
              - Multivariate Bayesian inference
              - Correlation-aware fusion
              
        """
        if self.combined_grid is None:
            raise ValueError("Run combine_event_shakemaps first.")
        if self.metadata_df is None:
            raise ValueError("Run build_metadata_table first.")
    
        supported_modes = {"auto", "main_only", "all_events", "model_default", "version_diff", "gmm_residual"}
        if mode not in supported_modes:
            raise ValueError(f"Unsupported mode '{mode}'.")
    
        event_cols = [c for c in self.combined_grid.columns if c.startswith(f"{column_prefix}_")]
        event_ids = [c.split("_")[-1] for c in event_cols]
    
        data_stack, var_stack, usable_ids = [], [], []
    
        for eid, col in zip(event_ids, event_cols):
            vals = self.combined_grid[col].values
    
            if std_event_id:
                stds = self._event_std_stack.get(std_event_id)
                if stds is None or np.all(np.isnan(stds)):
                    logging.warning(f"No usable std from '{std_event_id}'")
                    continue
            elif mode in {"main_only", "all_events", "auto"}:
                stds = self._event_std_stack.get(eid)
                if stds is None or np.all(np.isnan(stds)):
                    if mode == "main_only" and self.metadata_df.loc[self.metadata_df['event_id'] == eid, 'shock_type'].values[0] != "main":
                        continue
                    if mode == "all_events":
                        raise ValueError(f"No std data for {eid}")
                    if mode == "auto":
                        continue
            elif mode == "model_default":
                stds = np.full_like(vals, default_sigma)
            elif mode in {"version_diff", "gmm_residual"}:
                logging.warning(f"Bayesian mode '{mode}' is not yet implemented.")
                return
    
            data_stack.append(vals)
            var_stack.append(stds**2)
            usable_ids.append(eid)
    
        if not usable_ids:
            raise ValueError("No usable data found for Bayesian combination.")
    
        X = np.stack(data_stack)
        V = np.stack(var_stack)
    
        # ----- Bayesian Model Averaging weights -----
        if use_bma:
            bma_weights = self._compute_bma_weights(usable_ids, mode=bma_weight_mode)
            bma_weights = np.array(bma_weights)
            bma_weights = bma_weights / bma_weights.sum()  # normalize
            inv_var = 1 / V
            weighted_sum = np.sum(X * inv_var * bma_weights[:, None], axis=0)
            total_inv_var = np.sum(inv_var * bma_weights[:, None], axis=0)
        else:
            inv_var = 1 / V
            total_inv_var = np.sum(inv_var, axis=0)
            weighted_sum = np.sum(X * inv_var, axis=0)
    
        estimate = np.divide(
            weighted_sum,
            total_inv_var,
            out=np.full_like(weighted_sum, np.nan),
            where=total_inv_var > 0
        )
    
        label_map = {
            "main_only": "mbayesian",
            "all_events": "abayesian",
            "model_default": "modelbayesian",
            "auto": "abayesian"
        }
        suffix = label_map.get(mode, "bayesian")
        if use_bma:
            suffix = f"bma_{suffix}"
        if std_event_id:
            suffix += f"_std_{std_event_id}"
    
        colname = f"cumulative_{suffix}"
        self.combined_grid[colname] = estimate
        logging.info(f"Bayesian shakemap ({mode}, BMA={use_bma}) stored in '{colname}' using {len(usable_ids)} events.")
    
    




    def _compute_bma_weights(self, event_ids, mode="uniform"):
        """
        Estimate BMA weights based on metadata.
    
        Parameters
        ----------
        event_ids : list of str
            List of event IDs used in Bayesian stacking.
        mode : str
            One of {"uniform", "magnitude", "distance"}.
    
        Returns
        -------
        list of float
            Weights (non-normalized) for each event ID.
        """
        if self.metadata_df is None:
            raise ValueError("Metadata table is required for BMA weight estimation.")
    
        meta = self.metadata_df.set_index("event_id")
        weights = []
    
        for eid in event_ids:
            if mode == "uniform":
                weights.append(1.0)
            elif mode == "magnitude":
                mag = meta.loc[eid, "magnitude"]
                weights.append(mag if not np.isnan(mag) else 1.0)
            elif mode == "distance":
                main = meta[meta["shock_type"] == "main"]
                if main.empty:
                    weights.append(1.0)
                else:
                    main_lat, main_lon = main.iloc[0][["lat", "lon"]]
                    lat, lon = meta.loc[eid, ["lat", "lon"]]
                    dist = self.haversine(main_lon, main_lat, lon, lat)
                    weights.append(1 / (dist + 1e-3))
            else:
                raise ValueError(f"Unsupported BMA weight mode: '{mode}'")
    
        return weights



    def reset(self):
        """
        Clear all internal state and cached grids.
        """
        self.metadata_df = None
        self._registry = None
        self._main_event_time = None
        self._ingest_metric = None
        self._event_raw.clear()
        self._event_base = None
        self._event_stack.clear()
        self._event_std_stack.clear()
        self._event_order.clear()
        self._event_extents.clear()
        self._event_spacing = None
        self._event_svel.clear()
        self.combined_grid = None
        self._failed_std_ingestions = []
        logging.info("🔁 SHAKEcumulative state has been reset.")



