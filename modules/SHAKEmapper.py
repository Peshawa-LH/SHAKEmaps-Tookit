"""
Fetch and Visualize ShakeMap Data
Class Name: SHAKEmapper

Description:
    The SHAKEmapper class provides a comprehensive set of tools to create, visualize, and analyze seismic ShakeMap 
    datasets. It supports reading grid files (e.g., VS30 data) from USGS and other sources, overlaying additional 
    geospatial information (such as rupture traces, seismic stations, earthquake epicenters, and DYFI responses), 
    and extracting and resampling grid data for further analysis. The class leverages libraries like Cartopy, 
    Matplotlib, and Rasterio to streamline the visualization and manipulation of ShakeMap data.

Prerequisites:
    To use this class effectively, ensure that the following Python libraries are installed:
      - Matplotlib
      - Cartopy
      - Rasterio
      - NumPy
      - Pandas
      - SciPy
    Additional geospatial libraries may be required for advanced functionalities.

Date:
    March, 2025
Version:
    26.1
"""


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import xml.etree.ElementTree as ET
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from rasterio.windows import from_bounds
from rasterio.transform import array_bounds
from scipy.ndimage import zoom
import os


class SHAKEmapper:
    """
    SHAKEmapper: A Comprehensive Class for Shakemap Visualization and Data Extraction

    Overview
    --------
    The SHAKEmapper class provides a flexible and modular framework for creating, visualizing,
    and analyzing seismic shakemaps using Python. It leverages Matplotlib and Cartopy for geospatial
    plotting and Rasterio for reading raster (grid) files. This class supports:
      - Creation of a basemap with customizable geographic extent, coastlines, borders, ocean features,
        gridlines, and an outline frame.
      - Overlaying VS30 data from raster files with a discrete categorical color mapping.
      - Extraction of VS30 data from a specified geographic extent, with options for resampling
        (subsampling or upsampling) to a desired grid spacing.
      - Computation of basic grid statistics (number of rows/columns, grid spacing, min/max/mean/std of VS30).
      - Conversion of the VS30 grid data into a Pandas DataFrame for further analysis.
      - Overlay of additional seismic and geospatial information such as rupture traces, seismic station
        locations, earthquake epicenters, and DYFI (Did You Feel It?) responses.
      - Retrieval and plotting of cities from external CSV files that fall within the current map extent.
      - Dynamic updating of the map extent and legend.

    Attributes
    ----------
    extent : list
        Geographic extent of the map in the format [lon_min, lon_max, lat_min, lat_max]. If not
        provided during initialization, a default global extent of [-180, 180, -90, 90] is used.
    fig : matplotlib.figure.Figure
        The Matplotlib Figure object created by the create_basemap() method.
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The Cartopy Axes object used for plotting the basemap and overlays.
    legend_zorder : int
        Z-order value used to ensure the legend appears on top of map elements.

    Key Methods
    -----------
    __init__(extent=None)
        Initialize the SHAKEmapper instance with an optional geographic extent.
    set_extent(extent)
        Set or update the internal geographic extent.
    create_basemap()
        Create a basemap with a PlateCarree projection, add coastlines, borders, ocean features,
        gridlines (with 2° intervals), and a thin black frame.
    add_vs30_layer(vs30_filepath, alpha, zorder)
        Overlay a VS30 layer from a raster file using a discrete colormap. The default color mapping
        classifies VS30 values into five categories (e.g., <90 m/s, 90–180 m/s, 180–360 m/s, 360–800 m/s, >800 m/s).
    extract_vs30_data(vs30_filepath, extent)
        Extract a subset of VS30 data from a raster file for a specified geographic extent and return
        the data array along with 1D coordinate arrays (longitude, latitude) and the new extent.
    extract_grid_data(vs30_filepath, extent, target_spacing)
        Extract and optionally resample the VS30 grid data. If a target spacing (in kilometers) is specified,
        the data is either subsampled or upsampled (using nearest-neighbor interpolation) accordingly.
    get_grid_stats(vs30_filepath, extent, target_spacing)
        Compute basic grid statistics such as number of rows, columns, grid spacing, and VS30 statistics.
    grid_to_dataframe(vs30_filepath, extent, target_spacing)
        Convert the extracted (and optionally resampled) VS30 grid data into a Pandas DataFrame with columns
        for longitude, latitude, and VS30.
    add_rupture(x_coords, y_coords, line_color, line_width, rupture_label, zorder)
        Overlay a rupture trace (given as longitude and latitude arrays) on the current basemap.
    contour_scale(imt)
        Generate a discrete colormap, normalization, and descriptive label for a given intensity measure type.
    _get_column(df, col_name)
        Helper method that retrieves a DataFrame column in a case-insensitive manner.
    add_stations(lon, lat, fig, ax, zorder)
        Plot seismic station markers on the map if they fall within the current extent.
    add_epicenter(lon, lat, fig, ax, zorder)
        Mark the earthquake epicenter on the map.
    get_cities(cities_csv)
        Retrieve city data from a CSV file that falls within the current map extent.
    add_cities(fig, ax, population, cities_csv, zorder, label)
        Plot and label cities on the map that exceed a specified population threshold.
    get_extent()
        Return the current geographic extent.
    update_extent(extent)
        Update the internal extent and redraw the basemap accordingly.
    add_usgs_shakemap(shakemap_data, zorder)
        Overlay USGS shakemap data (from a parser object) on the current basemap and update the extent.
    add_dyfi(lon, lat, values, nresp, fig, ax, zorder, label)
        Overlay DYFI data points with conditional styling based on the number of responses.
    update_legend(loc, alpha)
        Update and reposition the map legend with a specified location and transparency.

    Usage Example
    -------------
    Below is an example of how you might use the SHAKEmapper class:

        from shakemapper import SHAKEmapper

        # Initialize with a custom extent.
        mapper = SHAKEmapper(extent=[90.4, 103.4, 13.0, 26.8])
        fig, ax = mapper.create_basemap()

        # Overlay VS30 data.
        mapper.add_vs30_layer(vs30_filepath='./SHAKEdata/global_vs30.grd')

        # Extract and analyze grid data.
        grid_stats = mapper.get_grid_stats(vs30_filepath='./SHAKEdata/global_vs30.grd')
        print("Grid Stats:", grid_stats)

        # Convert grid data to a DataFrame.
        df_vs30 = mapper.grid_to_dataframe(vs30_filepath='./SHAKEdata/global_vs30.grd')
        print(df_vs30.head())

        # Overlay additional data (e.g., rupture trace, seismic stations, epicenter, DYFI, cities).
        # mapper.add_rupture(x_coords, y_coords)
        # mapper.add_stations(station_lons, station_lats)
        # mapper.add_epicenter(epi_lon, epi_lat)
        # mapper.add_dyfi(dyfi_lons, dyfi_lats, dyfi_values, nresp=dyfi_nresp)
        # mapper.add_cities(population=100000)

        # Update the legend if necessary.
        mapper.update_legend()

        plt.show()


    References
    ----------
    - Matplotlib Documentation: https://matplotlib.org/stable/contents.html
    - Cartopy Documentation: https://scitools.org.uk/cartopy/docs/latest/
    - Rasterio Documentation: https://rasterio.readthedocs.io/en/latest/
    - SciPy ndimage.zoom: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html


    © SHAKEmaps Version 26.1
    """
    def __init__(self, extent=None):
        """
        Initialize a SHAKEmapper instance with an optional geographic extent.

        If no extent is provided, the default global extent [-180, 180, -90, 90] is used.

        Parameters
        ----------
        extent : list, optional
            A list of four floats [lon_min, lon_max, lat_min, lat_max]. If None, the default extent is used.
        
        References
        ----------
        See the official Python documentation for classes:
        https://docs.python.org/3/tutorial/classes.html
        """
        if extent is None:
            self.extent = [-180, 180, -90, 90]
        else:
            self.set_extent(extent)
        self.fig = None
        self.ax = None
        self._plate = ccrs.PlateCarree()
        self._shakemap_colorbar = None
        self._dyfi_colorbar     = None



        
        self.legend_zorder = 100


    def set_extent(self, extent):
        """
        Set the geographical extent for the shakemap.

        Parameters
        ----------
        extent : list
            A list of four values [lon_min, lon_max, lat_min, lat_max].

        Raises
        ------
        ValueError
            If the provided extent does not contain exactly four elements.
        """
        if len(extent) != 4:
            raise ValueError("Extent must be a list of four values: [lon_min, lon_max, lat_min, lat_max].")
        self.extent = extent

    def create_basemap(self, figsize=(24, 12), label_size = 20 ):
        """
        Create a basemap using Cartopy and Matplotlib with default visual settings.

        The method performs the following:
          - Initializes a figure and axes with PlateCarree projection.
          - Sets the map extent.
          - Adds coastlines, country borders, and an ocean feature.
          - Configures gridlines with labels.
          - Adds a thin black frame around the map.

        Returns
        -------
        tuple
            A tuple (fig, ax) where fig is the Matplotlib Figure and ax is the Cartopy Axes.

        References
        ----------
        - Cartopy Documentation: https://scitools.org.uk/cartopy/docs/latest/
        - Matplotlib Documentation: https://matplotlib.org/stable/contents.html
        """
        self.fig = plt.figure(figsize=figsize)
        self.ax  = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    
        # Rasterize coastlines and borders as bitmaps
        self.ax.coastlines(zorder=10, rasterized=True)
        self.ax.add_feature(cfeature.BORDERS,  zorder=10,
                            linestyle='-', rasterized=True)
    
        # Rasterize the ocean fill
        self.ax.add_feature(cfeature.OCEAN, zorder=9,
                            facecolor='skyblue', rasterized=True)
    
        gl = self.ax.gridlines(crs=ccrs.PlateCarree(),
                               draw_labels=True, linewidth=2,
                               color='gray', alpha=0.7,
                               linestyle='--', zorder=999)
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {"size": label_size}
        gl.ylabel_style = {"size": label_size}
    
        # Thin black frame around the map
        frame = mpatches.Rectangle((0, 0), 1, 1,
                                   transform=self.ax.transAxes,
                                   fill=False, edgecolor='black',
                                   linewidth=1, zorder=100)
        self.ax.add_patch(frame)
    
        self.ax.set_extent(self.extent, crs=ccrs.PlateCarree())
        return self.fig, self.ax

        


    def add_vs30_layer(self,
                       vs30_filepath='./SHAKEdata/global_vs30.grd',
                       alpha=0.6,
                       zorder=8):
        """
        Overlay a VS30 layer onto the basemap using a discrete categorical colormap,
        but only reads the data within the current map extent for speed.

        The method reads the VS30 grid file only within `self.extent` and applies the
        following color mapping based on VS30 values:
          - 0 to 90 m/s: red
          - 90 to 180 m/s: orange
          - 180 to 360 m/s: yellow
          - 360 to 800 m/s: lightgreen
          - 800 to 1500 m/s: green

        Parameters
        ----------
        vs30_filepath : str
            Path to the VS30 grid file.
        alpha : float, optional
            Transparency level for the VS30 layer (default is 0.6).
        zorder : int, optional
            Drawing order for the VS30 layer (default is 8).

        Raises
        ------
        RuntimeError
            If the basemap has not been created (i.e. create_basemap() has not been called).
        """
        if self.ax is None:
            raise RuntimeError("Basemap has not been created yet. Call create_basemap() first.")

        # 1) Extract only the subset within the current extent
        data = self.extract_vs30_data(vs30_filepath, extent=self.extent)
        vs30_subset = data['vs30']
        extent = data['extent']     # [left, right, bottom, top]

        # 2) Set up the discrete colormap exactly as before
        boundaries = [0, 90, 180, 360, 800, 1500]
        colors     = ["red", "orange", "yellow", "lightgreen", "green"]
        cmap       = ListedColormap(colors)
        norm       = BoundaryNorm(boundaries, len(colors))

        # 3) Plot only the subset
        im = self.ax.imshow(
            vs30_subset,
            origin='upper',
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            zorder=zorder
        )

        # 4) Draw the colorbar with custom ticks
        cbar = self.fig.colorbar(
            im,
            ax=self.ax,
            orientation='vertical',
            pad=0.05,
            ticks=boundaries
        )
        tick_labels = ["", "<90 $m/s$", "180 $m/s$", "360 $m/s$", ">800 $m/s$", ""]
        cbar.set_ticklabels(tick_labels)
        cbar.set_label('$V_{S30}$ ($m/s$)')

    
        
        
        
        
    def extract_vs30_data(self,
                          vs30_filepath='./SHAKEdata/global_vs30.grd',
                          extent=None,
                          lons=None,
                          lats=None):
        """
        Extract VS30 data from a raster file for a given extent, and/or sample at specific points.

        This function can do two things in one call:
          1. If no point arrays are provided, it behaves as before: extracts a 2D subset for
             the specified geographic extent and returns lon/lat coordinate arrays plus the grid.
          2. If arrays `lons` and `lats` are provided (same length), it additionally
             samples VS30 at each of those point locations and returns their values.

        Parameters
        ----------
        vs30_filepath : str
            Path to the VS30 grid file.
        extent : list of float, optional
            A list [lon_min, lon_max, lat_min, lat_max]. If None, defaults to self.extent.
        lons : array-like, optional
            Longitudes of individual points to sample. If provided, `lats` must also be provided.
        lats : array-like, optional
            Latitudes of individual points to sample. Must match length of `lons`.

        Returns
        -------
        dict
            Always contains:
              - 'lon': 1D NumPy array of grid longitudes for the extent.
              - 'lat': 1D NumPy array of grid latitudes for the extent.
              - 'vs30': 2D NumPy array of VS30 grid data for the extent.
              - 'extent': [left, right, bottom, top] bounds of the extracted grid.

            Additionally, if points were provided:
              - 'point_values': 1D NumPy array of VS30 values at each (lons, lats) location.

        Raises
        ------
        ValueError
            If only one of `lons`/`lats` is provided, or if their lengths differ.

        References
        ----------
        Rasterio sampling docs: https://rasterio.readthedocs.io/en/latest/topics/sample.html
        """
        if extent is None:
            extent = self.extent

        with rasterio.open(vs30_filepath) as src:
            # 1) Extract the grid subset
            window = from_bounds(extent[0], extent[2], extent[1], extent[3], src.transform)
            vs30_subset = src.read(1, window=window)
            window_transform = src.window_transform(window)
            new_bounds = array_bounds(vs30_subset.shape[0],
                                      vs30_subset.shape[1],
                                      window_transform)
            new_extent = [new_bounds[0], new_bounds[2], new_bounds[1], new_bounds[3]]

            # Build grid coordinate arrays
            nrows, ncols = vs30_subset.shape
            a = window_transform.a
            c = window_transform.c
            e = window_transform.e
            f = window_transform.f
            lon_coords = c + np.arange(ncols) * a
            lat_coords = f + np.arange(nrows) * e

            result = {
                'lon': lon_coords,
                'lat': lat_coords,
                'vs30': vs30_subset,
                'extent': new_extent
            }

            # 2) If point arrays are provided, sample those
            if (lons is not None) or (lats is not None):
                if lons is None or lats is None:
                    raise ValueError("Both lons and lats must be provided together.")
                lons = np.asarray(lons)
                lats = np.asarray(lats)
                if lons.shape != lats.shape:
                    raise ValueError("lons and lats must have the same shape.")

                coords = list(zip(lons, lats))
                samples = list(src.sample(coords))
                point_values = np.array([s[0] for s in samples], dtype=vs30_subset.dtype)
                result['point_values'] = point_values

        return result


    
    def extract_grid_data(self, vs30_filepath='./SHAKEdata/global_vs30.grd', extent=None, target_spacing=None):
        """
        Extract and optionally resample VS30 data from a raster file.

        The function first extracts the native grid for the specified extent. It then:
          - Subsamples the grid if the target spacing is larger than the native spacing.
          - Upsamples the grid via nearest-neighbor interpolation if the target spacing is smaller.
          - Returns the native grid if target_spacing is None.

        Parameters
        ----------
        vs30_filepath : str
            Path to the VS30 grid file.
        extent : list, optional
            Geographic extent as [lon_min, lon_max, lat_min, lat_max]. Defaults to self.extent.
        target_spacing : float, optional
            Desired grid spacing in kilometers. If None, no resampling is performed.

        Returns
        -------
        dict
            Dictionary with keys:
              - 'vs30': 2D NumPy array of the (resampled) VS30 data.
              - 'lon': 1D NumPy array of longitudes.
              - 'lat': 1D NumPy array of latitudes.
              - 'extent': Geographic bounds [left, right, bottom, top] of the data.

        References
        ----------
        - Rasterio documentation: https://rasterio.readthedocs.io/en/latest/
        - SciPy ndimage.zoom documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
        """
        # Use the provided extent or fallback to self.extent.
        if extent is None:
            extent = self.extent

        # Open the file and extract the native grid for the given extent.
        with rasterio.open(vs30_filepath) as src:
            window = from_bounds(extent[0], extent[2], extent[1], extent[3], src.transform)
            vs30_subset = src.read(1, window=window)
            window_transform = src.window_transform(window)
            new_bounds = array_bounds(vs30_subset.shape[0], vs30_subset.shape[1], window_transform)
            # new_bounds is (left, bottom, right, top)
            new_extent = [new_bounds[0], new_bounds[2], new_bounds[1], new_bounds[3]]

        nrows, ncols = vs30_subset.shape

        # Compute native grid spacing in degrees.
        native_dx_deg = (new_extent[1] - new_extent[0]) / ncols
        native_dy_deg = (new_extent[3] - new_extent[2]) / nrows

        # Estimate native spacing in kilometers.
        # 1° latitude is ~111 km. For longitude, the conversion depends on latitude.
        mean_lat = (new_extent[2] + new_extent[3]) / 2.0
        native_dx_km = native_dx_deg * 111 * np.cos(np.deg2rad(mean_lat))
        native_dy_km = native_dy_deg * 111
        # We'll use the average spacing as an approximation.
        native_spacing_km = (native_dx_km + native_dy_km) / 2.0

        if target_spacing is None:
            # No resampling requested; use the native grid.
            resampled_vs30 = vs30_subset
            new_nrows_res, new_ncols_res = nrows, ncols
        else:
            if target_spacing > native_spacing_km:
                # Subsample the data: choose every 'step' row/column.
                step = int(round(target_spacing / native_spacing_km))
                step = max(1, step)
                resampled_vs30 = vs30_subset[::step, ::step]
            elif target_spacing < native_spacing_km:
                # Upsample the data using nearest-neighbor interpolation.
                zoom_factor_lat = native_dy_km / target_spacing
                zoom_factor_lon = native_dx_km / target_spacing
                resampled_vs30 = zoom(vs30_subset, (zoom_factor_lat, zoom_factor_lon), order=0)
            else:
                resampled_vs30 = vs30_subset
            new_nrows_res, new_ncols_res = resampled_vs30.shape

        # Create new coordinate arrays based on the new shape and original extent.
        left, right, bottom, top = new_extent
        new_lon = np.linspace(left, right, new_ncols_res)
        # We assume that the original data are arranged from top (maximum latitude) to bottom.
        new_lat = np.linspace(top, bottom, new_nrows_res)

        return {'vs30': resampled_vs30, 'lon': new_lon, 'lat': new_lat, 'extent': new_extent}


    
    def get_grid_stats(self, vs30_filepath='./SHAKEdata/global_vs30.grd', extent=None,target_spacing=None):
        """
        Extract VS30 grid data and compute basic statistical metrics.

        The function computes:
          - The number of rows and columns.
          - Total number of grid points.
          - Approximate grid spacing (dx and dy).
          - Minimum, maximum, mean, and standard deviation of VS30 values.

        Parameters
        ----------
        vs30_filepath : str
            Path to the VS30 grid file.
        extent : list, optional
            Geographic extent as [lon_min, lon_max, lat_min, lat_max]. Defaults to self.extent.
        target_spacing : float, optional
            Desired grid spacing in kilometers. If None, the native grid is used.

        Returns
        -------
        dict
            Dictionary containing grid statistics:
              - 'nrows', 'ncols', 'total_points'
              - 'dx', 'dy'
              - 'min', 'max', 'mean', 'std'
              - 'extent' of the extracted data.
        """

        result = self.extract_grid_data(vs30_filepath, extent,target_spacing)
        
        vs30_array = result['vs30']
        new_extent = result['extent']

        # Get grid dimensions.
        nrows, ncols = vs30_array.shape
        total_points = nrows * ncols

        # Calculate grid spacing.
        # new_extent is [left, right, bottom, top]
        dx = (new_extent[1] - new_extent[0]) / ncols
        dy = (new_extent[3] - new_extent[2]) / nrows

        # Compute basic statistics, ignoring NaN values.
        stats = {
            'nrows': nrows,
            'ncols': ncols,
            'total_points': total_points,
            'dx': dx,
            'dy': dy,
            'min': float(np.nanmin(vs30_array)),
            'max': float(np.nanmax(vs30_array)),
            'mean': float(np.nanmean(vs30_array)),
            'std': float(np.nanstd(vs30_array)),
            'extent': new_extent
        }
        return stats
    
    
    def grid_to_dataframe(self, vs30_filepath='./SHAKEdata/global_vs30.grd', extent=None,target_spacing=None):
        """
        Convert VS30 grid data to a Pandas DataFrame with coordinate columns.

        The function extracts (and optionally resamples) VS30 data and returns a DataFrame
        containing longitude, latitude, and VS30 value for each grid cell.

        Parameters
        ----------
        vs30_filepath : str
            Path to the VS30 grid file.
        extent : list, optional
            Geographic extent as [lon_min, lon_max, lat_min, lat_max]. Defaults to self.extent.
        target_spacing : float, optional
            Desired grid spacing in kilometers. If None, the native grid is used.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
              - 'lon': Longitude values.
              - 'lat': Latitude values.
              - 'vs30': VS30 values.
        """
        # Extract the data as a dictionary using the previously defined function.
        data = self.extract_grid_data(vs30_filepath, extent, target_spacing)
        vs30_array = data['vs30']
        lon_coords = data['lon']
        lat_coords = data['lat']

        # Create a 2D meshgrid for the coordinates.
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

        # Flatten the arrays so each row corresponds to one grid cell.
        df = pd.DataFrame({
            'lon': lon_grid.ravel(),
            'lat': lat_grid.ravel(),
            'vs30': vs30_array.ravel()
        })
        return df




    def add_rupture(self, x_coords, y_coords,
                    line_color='k-', line_width=2,
                    rupture_label='Rupture Extent', zorder=12):
        """
        Overlay rupture trace data on the current basemap.

        The function plots a rupture trace on the map using the provided x (longitude)
        and y (latitude) coordinates.

        Parameters
        ----------
        x_coords : array-like
            Array of x-coordinates (longitudes) for the rupture trace.
        y_coords : array-like
            Array of y-coordinates (latitudes) for the rupture trace.
        line_color : str, optional
            Matplotlib line style and color (default 'k-').
        line_width : int, optional
            Line width for the rupture trace (default 2).
        rupture_label : str, optional
            Label for the rupture trace (default 'Rupture Extent').
        zorder : int, optional
            Drawing order for the rupture trace (default 6).

        Returns
        -------
        None
            The rupture trace is added to the basemap.
        """
        
        if self.ax is None:
            raise RuntimeError("Call create_basemap() first.")
    
        # 1) ensure numpy arrays
        x = np.asarray(x_coords)
        y = np.asarray(y_coords)
    
        # 2) plot once
        self.ax.plot(x, y,
                     line_color, linewidth=line_width,
                     label=rupture_label,
                     zorder=zorder,
                     transform=self._plate)
    
        # 3) add legend only if missing
        if self.ax.get_legend() is None:
            self.ax.legend(loc='upper right')
    
        # no canvas.draw() here





    def contour_scale(self, imt="MMI"):
        """
        Generate a colormap and normalization parameters for seismic intensity mapping.

        Based on the specified intensity measure type (IMT), this method returns:
          - A discrete ListedColormap.
          - A list of boundaries and ticks.
          - A BoundaryNorm normalization object.
          - A descriptive label for the intensity scale.

        Parameters
        ----------
        imt : str, optional
            Intensity measure type (e.g., 'MMI', 'PGA', 'PGV', 'PSA03', 'PSA10'). Default is 'MMI'.

        Returns
        -------
        tuple
            A tuple (cmap, bounds, ticks, norm, used_scale) where:
              - cmap is the ListedColormap.
              - bounds is the list of boundary values.
              - ticks is the list of tick values.
              - norm is a BoundaryNorm instance.
              - used_scale is a descriptive label for the scale.
        
        Raises
        ------
        ValueError
            If an invalid intensity measure type is provided.
        """

        imt = imt.upper()
        usgs_colors = [
            (255/255, 255/255, 255/255, 1.0),
            (191/255, 204/255, 255/255, 1.0),
            (160/255, 230/255, 255/255, 1.0),
            (128/255, 255/255, 255/255, 1.0),
            (122/255, 255/255, 147/255, 1.0),
            (255/255, 255/255, 0/255, 1.0),
            (255/255, 200/255, 0/255, 1.0),
            (255/255, 145/255, 0/255, 1.0),
            (255/255, 0/255, 0/255, 1.0),
            (200/255, 0/255, 0/255, 1.0),
            (128/255, 0/255, 0/255, 1.0)
        ]
        cmap = mpl.colors.ListedColormap(usgs_colors)
        usgs_table = {
            "MMI": [0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10],
            "PGA": [0, 0.05, 0.3, 2.8, 6.2, 11.5, 21.5, 40.1, 74.7, 139],
            "PGV": [0, 0.02, 0.1, 1.4, 4.7, 9.6, 20, 41, 86, 178],
            "PSA03": [0, 0.02, 0.1, 1, 4.6, 10, 23, 50, 110, 244],
            "PSA10": [0, 0.02, 0.1, 1, 4.6, 10, 23, 50, 110, 244]
        }
        default_units = {
            "MMI": "MMI",
            "PGA": "%g",
            "PGV": "cm/s",
            "PSA03": "%g",
            "PSA10": "%g"
        }
        labels = {
            "%g": "%g",
            "cm/s": "cm/s",
            "MMI": "MMI"
        }
        scale_labels = {
            "MMI": "Modified Mercalli Intensity Scale",
            "PGA": "Peak Ground Acceleration",
            "PGV": "Peak Ground Velocity",
            "PSA03": "Spectral Acceleration $Sa_{0.3s}$",
            "PSA10": "Spectral Acceleration $Sa_{1s}$"
        }
        if imt not in usgs_table:
            raise ValueError(f"Invalid intensity measure type '{imt}'")
        bounds = usgs_table[imt]
        ticks = bounds
        used_scale = f'{scale_labels[imt]} ({labels[default_units[imt]]})'
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        return cmap, bounds, ticks, norm, used_scale
    
    def _get_column(self, df, col_name):
        """
        Helper method to return a column from df using different case variants.
        """
        for key in [col_name, col_name.lower(), col_name.upper()]:
            if key in df.columns:
                return df[key].values
        raise KeyError(f"Column '{col_name}' not found in DataFrame.")
        
        


    

    def add_stations(self, lon, lat, zorder=14, s= 150):
        """
        Add seismic station markers to the current shakemap plot.

        The function plots station markers if their coordinates lie within the map extent.

        Parameters
        ----------
        lon : array-like
            Array of station longitudes.
        lat : array-like
            Array of station latitudes.
        fig : matplotlib.figure.Figure, optional
            The figure object; defaults to the current figure if not provided.
        ax : matplotlib.axes.Axes, optional
            The axes object; defaults to the current axes if not provided.
        zorder : int, optional
            Drawing order for the station markers (default is 7).

        Raises
        ------
        ValueError
            If either longitude or latitude data is not provided.
        """
        
        if self.ax is None:
            raise RuntimeError("Call create_basemap() first.")
    
        # 1) numpy arrays
        lon = np.asarray(lon)
        lat = np.asarray(lat)
    
        # 2) extent from stored values (avoid get_xlim()/get_ylim())
        xmin, xmax, ymin, ymax = self.extent
        mask = (lon>=xmin)&(lon<=xmax)&(lat>=ymin)&(lat<=ymax)
        lon2, lat2 = lon[mask], lat[mask]
    
        if lon2.size:
            self.ax.scatter(
                lon2, lat2,
                marker='^', facecolors='r', edgecolors='k',
                s=s, label='Seismic Station',
                zorder=zorder,
                transform=self._plate,
                rasterized=True
            )
            if self.ax.get_legend() is None:
                self.ax.legend(loc='upper right')
        # no draw() here

        

    def add_epicenter(self, lon=None, lat=None, fig=None, ax=None,zorder=30,markersize=20):
        """
        Add an earthquake epicenter marker to the current shakemap plot.

        Parameters
        ----------
        lon : float
            Longitude of the earthquake epicenter.
        lat : float
            Latitude of the earthquake epicenter.
        fig : matplotlib.figure.Figure, optional
            The figure object; defaults to the current figure if not provided.
        ax : matplotlib.axes.Axes, optional
            The axes object; defaults to the current axes if not provided.
        zorder : int, optional
            Drawing order for the epicenter marker (default is 7).

        Raises
        ------
        ValueError
            If either the longitude or latitude is not provided.
        """
        if lon is None or lat is None:
            raise ValueError("Longitude and latitude must be provided.")

            
        fig = fig if fig is not None else self.fig
        ax = ax if ax is not None else self.ax


        ax.plot(lon, lat, 'k*', mfc='y', markersize=markersize, label='Earthquake Epicenter', zorder=zorder, transform=self._plate)
        legend = ax.get_legend()
        if legend is None or 'Earthquake Epicenter' not in [text.get_text() for text in legend.get_texts()]:
            ax.legend(loc='upper right')
        #fig.canvas.draw()


    def get_cities(self, cities_csv='./SHAKEdata/worldcities.csv'):
        """
        Retrieve cities that lie within the current map extent from a CSV file.
        """
        # 1) grab current extent
        extent = self.get_extent()

        # 2) load full CSV
        try:
            cities_df = pd.read_csv(cities_csv)
        except Exception as e:
            raise

        # 3) filter to extent
        lon_min, lon_max, lat_min, lat_max = extent
        mask = (
            (cities_df['Longitude'] >= lon_min) &
            (cities_df['Longitude'] <= lon_max) &
            (cities_df['Latitude']  >= lat_min) &
            (cities_df['Latitude']  <= lat_max)
        )
        within = cities_df[mask]
        return within


    def add_cities(self,
                   fig=None,
                   ax=None,
                   population=100000,
                   cities_csv='./SHAKEdata/worldcities.csv',
                   zorder=16,
                   label='Cities', label_fontsize=15,markersize=7):
        """
        Plot cities on the shakemap that have a population exceeding a specified threshold.
        Instrumented with prints to debug figure‐breaking issues.
        """
        fig = fig if fig is not None else self.fig
        ax  = ax  if ax is not None else self.ax

        # 1) get & filter cities by geographic extent
        cities_df = self.get_cities(cities_csv=cities_csv)

        # 2) filter by population
        large = cities_df[cities_df['population'] > population]
        if large.empty:
            return

        # 3) plot each city
        first = True
        for idx, city in large.iterrows():
            lon = city['Longitude']
            lat = city['Latitude']
            name = city['city_name']
            lbl  = f"{label} pop>{population}" if first else None
            first = False
            ax.plot(
                lon, lat,
                'wo', mfc='k', markersize=markersize,
                label=lbl,
                zorder=zorder,
                transform=ccrs.PlateCarree()
            )
            txt = ax.text(
                lon + 0.05, lat + 0.05, name,
                verticalalignment='center',
                transform=ccrs.Geodetic(),
                fontsize=label_fontsize, style='italic',
                zorder=zorder
            )
            txt.set_path_effects([PathEffects.withStroke(
                linewidth=1, foreground='white'
            )])

        # 4) legend
        legend = ax.get_legend()
        if legend is None or f"{label} pop>{population}" not in [t.get_text() for t in legend.get_texts()]:
            ax.legend(loc='upper right', fontsize='x-large')

        # 5) force redraw
        fig.canvas.draw()



    def get_extent(self):
        """
        Get the current geographical extent of the map.

        Returns
        -------
        list
            A list of four values [lon_min, lon_max, lat_min, lat_max].
        """
        return self.extent
    
    def update_extent(self, extent):
        """
        Update the map extent and redraw the basemap.

        This method sets a new extent and, if a basemap already exists,
        updates its extent accordingly.

        Parameters
        ----------
        extent : list
            A list of four values [lon_min, lon_max, lat_min, lat_max].
        """
        # Validate and set the new extent.
        self.set_extent(extent)

        # If the basemap (axes) exists, update its extent.
        if self.ax is not None:
            self.ax.set_extent(self.extent, crs=ccrs.PlateCarree())
            #self.fig.canvas.draw()


    def add_dyfi(self,
                 lon, lat, values,
                 nresp=None,
                 zorder=13,
                 label='DYFI? Data',
                 plot_colorbar=True, s = 150):
        """
        Overlay DYFI data points but only draw a colorbar if
        - the user requests one, and
        - no shakemap colorbar already exists.

        Parameters
        ----------
        lon : array-like
            Longitudes of DYFI points.
        lat : array-like
            Latitudes of DYFI points.
        values : array-like
            Intensity values for each point.
        nresp : array-like, optional
            Number of responses for each point; used to color edges.
        zorder : int, optional
            Drawing order for the scatter.
        label : str, optional
            Legend label for the DYFI layer.
        plot_colorbar : bool, optional
            If True, attempt to draw a colorbar (but only if no shakemap cb exists).
        """
        if self.ax is None:
            raise RuntimeError("Call create_basemap() first.")

        # 1) convert to arrays and mask invalid
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        vals = np.asarray(values, dtype=float)

        if nresp is not None:
            nresp = np.asarray(nresp, dtype=float)
            valid = (~np.isnan(lon) & ~np.isnan(lat) &
                     ~np.isnan(vals) & ~np.isnan(nresp))
        else:
            valid = (~np.isnan(lon) & ~np.isnan(lat) & ~np.isnan(vals))

        lon = lon[valid]
        lat = lat[valid]
        vals = vals[valid]
        if nresp is not None:
            nresp = nresp[valid]

        if lon.size == 0:
            return  # nothing to plot

        # 2) determine edge colors
        if nresp is not None:
            edges = np.full(lon.shape, 'black', dtype=object)
            edges[nresp < 3] = 'red'
        else:
            edges = 'black'

        # 3) get colormap/norm
        cmap, bounds, ticks, norm, used_scale = self.contour_scale(imt='mmi')

        # 4) scatter (rasterize for speed)
        sc = self.ax.scatter(
            lon, lat,
            c=vals,
            cmap=cmap,
            norm=norm,
            edgecolors=edges,
            s=s,
            alpha=0.9,
            zorder=zorder,
            transform=ccrs.PlateCarree(),
            linewidths=1,
            rasterized=True,
            label=label
        )

        # 5) only add a DYFI colorbar if requested and no shakemap cb exists
        if plot_colorbar and not hasattr(self, '_shakemap_colorbar'):
            dyfi_cb = self.fig.colorbar(
                sc, ax=self.ax,
                orientation='vertical',
                pad=0.05,
                ticks=ticks
            )
            dyfi_cb.set_label(used_scale)
            self._dyfi_colorbar = dyfi_cb

        # 6) legend once
        if self.ax.get_legend() is None:
            self.ax.legend(loc='upper right')

        return sc

        
    def update_legend(self, loc='upper right', alpha=0.9):
        """
        Update the legend so that it appears on top of all layers with a specified location and transparency.

        Parameters
        ----------
        loc : str, optional
            Legend location (default is 'upper right').
        alpha : float, optional
            The transparency level for the legend background (0.0 is fully transparent, 1.0 is fully opaque).
        """
        legend = self.ax.get_legend()
        if legend is None:
            legend = self.ax.legend(loc=loc)
        legend.get_frame().set_alpha(alpha)
        legend.set_zorder(self.legend_zorder)
        #self.fig.canvas.draw()
        
    def print_doc(self):
        """Prints the SHAKEmapper class docstring."""
        print(self.__doc__)


    def get_figure(self):
        """
        Return the current Matplotlib figure and axes.
        
        If they do not exist, a new basemap is created.
        
        Returns
        -------
        tuple
            A tuple (fig, ax) where 'fig' is the Matplotlib Figure and 'ax' is the Cartopy Axes.
        """
        if self.fig is None or self.ax is None:
            self.create_basemap()
        self.fig.canvas.draw_idle()  # Update the drawing if needed.
        return self.fig, self.ax



    
    def add_usgs_shakemap(self, shakemap_data, zorder=8, plot_colorbar=True):
        """
        Overlay USGS shakemap data as a proper grid (pcolormesh),
        using case-insensitive column lookups and equal aspect to avoid distortion.
        """
        if self.ax is None:
            raise RuntimeError("Call create_basemap() first.")
    
        # 1) fetch DataFrame & metadata
        df   = shakemap_data.get_dataframe()
        spec = shakemap_data.metadata['grid_specification']
        imt  = shakemap_data.imt
    
        # 2) extract centers + intensity
        lon_centers = np.unique(self._get_column(df, "LON"))
        lat_centers = np.unique(self._get_column(df, "LAT"))
        intensity   = self._get_column(df, imt)
    
        # 3) dims & consistency
        nlon = int(spec['nlon']); nlat = int(spec['nlat'])
        if intensity.size != nlon * nlat:
            raise ValueError(f"Expected {nlon}×{nlat}={nlon*nlat} points, got {intensity.size}")
    
        # 4) build edges
        dlon = float(spec['nominal_lon_spacing'])
        dlat = float(spec['nominal_lat_spacing'])
        lat_centers = np.sort(lat_centers)[::-1]
    
        lon_edges = np.concatenate([lon_centers - dlon/2,
                                    [lon_centers[-1] + dlon/2]])
        lat_edges = np.concatenate([lat_centers - dlat/2,
                                    [lat_centers[-1] + dlat/2]])
    
        # 5) reshape into 2D grid
        grid = intensity.reshape((nlat, nlon))
    
        # 6) colormap/norm
        cmap, bounds, ticks, norm, used_scale = self.contour_scale(imt=imt)
    
        # 7) draw
        mesh = self.ax.pcolormesh(
            lon_edges, lat_edges, grid,
            cmap=cmap, norm=norm, shading='nearest',
            transform=ccrs.PlateCarree(), zorder=zorder
        )
        if plot_colorbar:
            cb = self.fig.colorbar(mesh, ax=self.ax,
                                   orientation="vertical", pad=0.05,
                                   ticks=ticks)
            cb.set_label(used_scale)
            self._shakemap_colorbar = cb
    
        # 8) sort extents properly, then apply
        lon_min, lon_max = float(lon_edges.min()), float(lon_edges.max())
        lat_min, lat_max = float(lat_edges.min()), float(lat_edges.max())
        self.set_extent([lon_min, lon_max, lat_min, lat_max])
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                           crs=ccrs.PlateCarree())
        self.ax.set_aspect('equal', adjustable='box')




    
        return mesh

    # 2026 update 
    def add_usgs_shakemap(
        self,
        shakemap_data,
        zorder=8,
        plot_colorbar=True,
        # NEW: colorbar controls
        cbar_shrink=0.85,
        cbar_labelsize=14,
        cbar_ticksize=12,
        cbar_pad=0.05,
        cbar_orientation="vertical", cbar_label = None
    ):
        """
        Overlay USGS shakemap data as a proper grid (pcolormesh),
        using case-insensitive column lookups and equal aspect to avoid distortion.
    
        Colorbar controls:
          - cbar_shrink: scales colorbar length relative to axes (prevents overshoot)
          - cbar_labelsize: fontsize for colorbar label
          - cbar_ticksize: fontsize for colorbar ticks
        """
        if self.ax is None:
            raise RuntimeError("Call create_basemap() first.")
    
        # 1) fetch DataFrame & metadata
        df   = shakemap_data.get_dataframe()
        spec = shakemap_data.metadata['grid_specification']
        imt  = shakemap_data.imt
    
        # 2) extract centers + intensity
        lon_centers = np.unique(self._get_column(df, "LON"))
        lat_centers = np.unique(self._get_column(df, "LAT"))
        intensity   = self._get_column(df, imt)
    
        # 3) dims & consistency
        nlon = int(spec['nlon']); nlat = int(spec['nlat'])
        if intensity.size != nlon * nlat:
            raise ValueError(f"Expected {nlon}×{nlat}={nlon*nlat} points, got {intensity.size}")
    
        # 4) build edges
        dlon = float(spec['nominal_lon_spacing'])
        dlat = float(spec['nominal_lat_spacing'])
        lat_centers = np.sort(lat_centers)[::-1]
    
        lon_edges = np.concatenate([lon_centers - dlon/2,
                                    [lon_centers[-1] + dlon/2]])
        lat_edges = np.concatenate([lat_centers - dlat/2,
                                    [lat_centers[-1] + dlat/2]])
    
        # 5) reshape into 2D grid
        grid = intensity.reshape((nlat, nlon))
    
        # 6) colormap/norm
        cmap, bounds, ticks, norm, used_scale = self.contour_scale(imt=imt)


        colorbar_label = used_scale if cbar_label is None else cbar_label

            
            
    
        # 7) draw
        mesh = self.ax.pcolormesh(
            lon_edges, lat_edges, grid,
            cmap=cmap, norm=norm, shading='nearest',
            transform=ccrs.PlateCarree(), zorder=zorder
        )
    
        if plot_colorbar:
            # If you re-plot multiple times, remove old cbar first (prevents stacking)
            if hasattr(self, "_shakemap_colorbar") and self._shakemap_colorbar is not None:
                try:
                    self._shakemap_colorbar.remove()
                except Exception:
                    pass
                self._shakemap_colorbar = None
    
            cb = self.fig.colorbar(
                mesh,
                ax=self.ax,
                orientation=cbar_orientation,
                pad=cbar_pad,
                ticks=ticks,
                shrink=cbar_shrink,   # ✅ key fix: prevents exceeding map height
            )
    
            cb.set_label(colorbar_label, fontsize=cbar_labelsize)
            cb.ax.tick_params(labelsize=cbar_ticksize)
            self._shakemap_colorbar = cb
    
        # 8) sort extents properly, then apply
        lon_min, lon_max = float(lon_edges.min()), float(lon_edges.max())
        lat_min, lat_max = float(lat_edges.min()), float(lat_edges.max())
        self.set_extent([lon_min, lon_max, lat_min, lat_max])
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                           crs=ccrs.PlateCarree())
        self.ax.set_aspect('equal', adjustable='box')
    
        return mesh
    
        
        


    



    
    
    def add_shakemap(self,
                     lon, lat, intensity,
                     imt_type="mmi",
                     zorder=8,
                     plot_colorbar=True,
                     nominal_lon_spacing=None,
                     nominal_lat_spacing=None):
        """
        Overlay generic ShakeMap data onto the basemap with half-cell correction
        and proper extents ordering.
        """
        if self.ax is None:
            raise RuntimeError("Call create_basemap() first.")
    
        lon = np.asarray(lon); lat = np.asarray(lat)
    
        # compute center arrays
        lon_centers = np.unique(lon)
        lat_centers = np.sort(np.unique(lat))[::-1]
    
        # infer spacing
        if nominal_lon_spacing is None and lon_centers.size > 1:
            nominal_lon_spacing = float(np.median(np.diff(lon_centers)))
        if nominal_lat_spacing is None and lat_centers.size > 1:
            # notice we reverse back to increasing for diff
            nominal_lat_spacing = float(np.median(np.diff(lat_centers[::-1])))
    
        # half-cell shift
        if nominal_lon_spacing is not None:
            lon += nominal_lon_spacing / 2.0
        if nominal_lat_spacing is not None:
            lat -= nominal_lat_spacing / 2.0
    
        # recompute centers & edges
        lon_centers = np.unique(lon)
        lat_centers = np.sort(np.unique(lat))[::-1]
    
        if nominal_lon_spacing is not None and nominal_lat_spacing is not None:
            dlon, dlat = nominal_lon_spacing, nominal_lat_spacing
            lon_edges = np.concatenate([lon_centers - dlon/2,
                                        [lon_centers[-1] + dlon/2]])
            lat_edges = np.concatenate([lat_centers - dlat/2,
                                        [lat_centers[-1] + dlat/2]])
            nlon = lon_edges.size - 1
            nlat = lat_edges.size - 1
            grid = np.asarray(intensity).reshape((nlat, nlon))
    
            cmap, bounds, ticks, norm, used_scale = self.contour_scale(imt=imt_type)
            mesh = self.ax.pcolormesh(
                lon_edges, lat_edges, grid,
                cmap=cmap, norm=norm, shading='nearest',
                transform=ccrs.PlateCarree(), zorder=zorder
            )
            if plot_colorbar:
                cb = self.fig.colorbar(mesh, ax=self.ax,
                                       orientation="vertical", pad=0.05,
                                       ticks=ticks)
                cb.set_label(used_scale)
    
            lon_min, lon_max = float(lon_edges.min()), float(lon_edges.max())
            lat_min, lat_max = float(lat_edges.min()), float(lat_edges.max())
        else:
            # scatter fallback
            cmap, bounds, ticks, norm, used_scale = self.contour_scale(imt=imt_type)
            mesh = self.ax.scatter(
                lon, lat, c=intensity,
                cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(), zorder=zorder
            )
            if plot_colorbar:
                cb = self.fig.colorbar(mesh, ax=self.ax,
                                       orientation="vertical", pad=0.05,
                                       ticks=ticks)
                cb.set_label(used_scale)
    
            lon_min, lon_max = float(lon_centers.min()), float(lon_centers.max())
            # remember lat_centers is descending: so max is first
            lat_min, lat_max = float(lat_centers.min()), float(lat_centers.max())
    
        # update & apply extents
        self.set_extent([lon_min, lon_max, lat_min, lat_max])
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                           crs=ccrs.PlateCarree())
        self.ax.set_aspect('equal', adjustable='box')        
            
        return mesh
    
