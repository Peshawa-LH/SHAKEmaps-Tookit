"""
SHAKEparser: Unified Toolkit for Parsing USGS ShakeMap and DYFI Earthquake Data

Class Name: USGSParser (Main entry interface)

Description:
    SHAKEparser is a comprehensive earthquake data parsing toolkit developed for 
    researchers and analysts working with USGS ShakeMap data formats. The toolkit 
    supports parsing, analyzing, and visualizing seismic data from diverse sources, 
    including DYFI (Did You Feel It?), ShakeMap XML/ZIP, rupture and event data, 
    and station instruments.

    Major enhancements include robust support for different ShakeMap data structures, 
    flexible handling of amplitude and intensity types, auto-detection of DYFI file types, 
    advanced geospatial mapping, and statistical analysis features. This makes SHAKEparser 
    a highly modular and user-friendly interface for earthquake data exploration.

Module Contents:
----------------
1. Utility Functions:
    - haversine_distance: Computes great-circle distances using the Haversine formula.

2. Classes and Core Capabilities:

    - AccelerationUnitConverter:
        Converts between acceleration units (e.g., g, m/s², %g, cm/s²).

    - ParseInstrumentsData:
        Parses station data from USGS JSON files. Supports two distinct file formats 
        for amplitude-based and intensity-based extractions. Includes statistical plotting 
        and summary report generation.

    - ParseDYFIDataXML:
        Handles parsing and creation of DYFI XML files. Supports XML-to-DataFrame conversion,
        summary/statistics generation, and CSV-to-XML transformation with UTM code calculation.

    - ParseDYFIData:
        Automatically detects and parses a range of DYFI file types including:
            - CDI GEO (text/XML)
            - CDI ZIP (text/XML)
            - Plot files for number of responses and attenuation
        Supports section-based parsing, advanced plotting, and flexible data extraction 
        for all supported DYFI formats.

    - ParseEventDataXML, ParseRuptureDataJson, ParsePagerDataXML, ParseModelConfig:
        Interfaces for event, rupture, PAGER, and model configuration files.

    - USGSParser:
        Dispatcher class for selecting appropriate parser objects based on input file type.

Key Features:
-------------
- Auto file-type detection for DYFI datasets
- Statistical visualization with histograms, KDE plots, and geospatial scatter plots
- Support for observational vs. predicted data filtering
- Flexible API for extracting station-specific metadata
- Conversion between multiple ground motion units
- File export capabilities (plots, XML files)

Dependencies:
-------------
- pandas, numpy, matplotlib, seaborn
- xml.etree.ElementTree, pyproj, cartopy
- scipy.stats (for KDE, normal distributions)

Usage Examples:
---------------
1. **Parsing Instrument Data:**
    ```python
    parser = ParseInstrumentsData(json_file='station.json', file_type=2)
    df = parser.get_dataframe(value_type='pga')
    parser.show_statistics('pga')
    ```

2. **Parsing DYFI XML:**
    ```python
    parser = ParseDYFIDataXML(mode='parse', xml_file='dyfi_dat.xml')
    print(parser.get_summary())
    ```

3. **Creating DYFI XML from CSV:**
    ```python
    creator = ParseDYFIDataXML(mode='create', csv_file='stations.csv', event_id='event123')
    creator.write_dyfi_xml()
    ```

4. **Auto-detect DYFI Format and Parse:**
    ```python
    parser = ParseDYFIData("us7000_example_cdi_geo.txt")
    df = parser.get_dataframe()
    parser.plot_numresp()
    ```

© SHAKmaps version 25.3.2
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import seaborn as sns 
from scipy.stats import norm
import os  # This might be used for file path operations if needed.

import json
import xml.etree.ElementTree as ET
from pyproj import CRS, Transformer
import re
import math
import cartopy.crs as ccrs
import zipfile
import cartopy.feature as cfeature

from io import StringIO


import logging

#Configure logging for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# You can configure a StreamHandler as needed.
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

logger.disabled = True  # This will disable logging for this logger.




def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculates the Haversine distance between two sets of GPS coordinates.
    This function can handle single values or arrays of values.

    Parameters:
    - lon1, lat1 (float, list, numpy.ndarray, pandas.Series): Longitude and latitude of the first point(s) in degrees.
    - lon2, lat2 (float, list, numpy.ndarray, pandas.Series): Longitude and latitude of the second point(s) in degrees.

    Returns:
    - float or numpy.ndarray: The distance(s) between the two coordinates in kilometers.
    """
    
    import numpy as np
    import pandas as pd

    # Earth radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    # Helper function to ensure data is in numpy array format
    def to_numpy_array(x):
        if isinstance(x, (list, pd.Series)):
            return np.array(x)
        elif isinstance(x, (int, float)):
            return np.array([x])
        elif isinstance(x, np.ndarray):
            return x
        else:
            raise TypeError("Input must be a list, numpy array, pandas Series, int, or float.")

    # Convert inputs to numpy arrays
    lon1 = to_numpy_array(lon1)
    lat1 = to_numpy_array(lat1)
    lon2 = to_numpy_array(lon2)
    lat2 = to_numpy_array(lat2)

    # Ensure lon2 and lat2 match the shape of lon1 and lat1
    if lon2.size == 1:
        lon2 = np.full_like(lon1, lon2.item())
    if lat2.size == 1:
        lat2 = np.full_like(lat1, lat2.item())

    # Debug: Print input types and values
    #print("lon1:", lon1, "type:", type(lon1))
    #print("lat1:", lat1, "type:", type(lat1))
    #print("lon2:", lon2, "type:", type(lon2))
    #print("lat2:", lat2, "type:", type(lat2))

    # Convert all coordinates from degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Differences in coordinates
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    # Haversine formula
    a = np.sin(d_lat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    # If the input was a single value, return a single value rather than an array
    return distance if distance.size > 1 else distance.item()


import numpy as np

class AccelerationUnitConverter:
    """
    A class dedicated to the conversion of acceleration-related units, specifically designed to handle units related to gravity such as g, %g, m/s², and cm/s². This class provides methods to convert between these units using vectorized operations, suitable for handling scalar values, lists, or numpy arrays.

    Attributes:
        G (float): Acceleration due to gravity (approximately 9.81 m/s²).
    """
    
    G = 9.81  # m/s^2, acceleration due to gravity
    
    def __init__(self):
        # Map input units to their corresponding conversion functions
        self.input_conversion = {
            '%g': self.percent_g_to_m_s2,
            'g': self.g_to_m_s2,
            'm/s2': self.m_s2_to_m_s2,
            'cm/s2': self.cm_s2_to_m_s2
        }
        
        # Map output units to their corresponding conversion functions
        self.output_conversion = {
            '%g': self.m_s2_to_percent_g,
            'g': self.m_s2_to_g,
            'm/s2': self.m_s2_to_m_s2,
            'cm/s2': self.m_s2_to_cm_s2
        }

    def percent_g_to_m_s2(self, value):
        return value / 100 * self.G

    def g_to_m_s2(self, value):
        return value * self.G

    def m_s2_to_m_s2(self, value):
        return value

    def cm_s2_to_m_s2(self, value):
        return value / 100

    def m_s2_to_percent_g(self, value):
        return value / self.G * 100

    def m_s2_to_g(self, value):
        return value / self.G

    def m_s2_to_cm_s2(self, value):
        return value * 100

    def convert_unit(self, value, input_unit, output_unit):
        """"
        Converts acceleration values from one unit to another.
        Handles single values, lists, or numpy arrays as input.

        Parameters:
            value (int, float, list, numpy.ndarray): The acceleration values to convert.
            input_unit (str): The current unit of the values (e.g., '%g', 'g', 'm/s2', 'cm/s2').
            output_unit (str): The target unit for the values (e.g., '%g', 'g', 'm/s2', 'cm/s2').

        Returns:
            numpy.ndarray or float: The converted acceleration values in the target unit.

        Raises:
            ValueError: If an unsupported input or output unit is provided.

        Example Usage:
            converter = AccelerationUnitConverter()
            # Single value conversion
            print(converter.convert_unit(1, '%g', 'm/s2'))
            # List of values conversion
            print(converter.convert_unit([1, 2, 3, 4, 5], '%g', 'm/s2'))
            # Numpy array conversion
            input_array = np.array([1, 2, 3, 4, 5])
            print(converter.convert_unit(input_array, '%g', 'm/s2'))
        """

        
        if input_unit not in self.input_conversion or output_unit not in self.output_conversion:
            raise ValueError(f'Unknown unit(s) provided: {input_unit}, {output_unit}')

        # Convert the input to a numpy array if it isn't already one
        if isinstance(value, list):
            value = np.array(value)
        elif isinstance(value, (int, float)):
            value = np.array([value])

        # Perform vectorized conversion
        intermediate_value = self.input_conversion[input_unit](value)
        result_value = self.output_conversion[output_unit](intermediate_value)
        
        # If the input was a single value, return a single value rather than an array
        return result_value if len(result_value) > 1 else result_value.item()





class ParseInstrumentsData:
    """
    ParseInstrumentsData is a class for parsing USGS earthquake instruments data from JSON files.
    It supports two distinct file structures, each corresponding to a different data format.

    File Structures:
    ----------------
    The parser handles two main types of JSON files:

    1. **File Type 1 (Original Instrumented Data):**
       - Typical filenames include "instrumented_dat.json", "instruments", or "ins".
       - This format represents the original USGS instrumented data.
       - For amplitude-based measurements (e.g., 'pga', 'pgv', and spectral acceleration values such as 'sa*'
         that are not meant for intensity), the parser extracts amplitude information from the "channels"
         list embedded within each feature. It computes derived fields such as "HNE", "HNN", and "HNZ" (from
         the respective channels) and selects the maximum amplitude (with its unit) for the requested measure.
       - For non‑amplitude values (e.g., "intensity"), the parser directly extracts the value from the feature’s
         properties without channel‑specific processing.

    2. **File Type 2 (Alternate Station List Data):**
       - Typical filenames contain a string like "us7000pn9s_us_".
       - This format usually comes as a station list including additional fields such as station type,
         various distance measures (e.g., rrup, repi, rhypo, rjb), and prediction values.
       - When intensity measures are requested (e.g., by specifying value_type 'mmi' or 'intensity'),
         the parser extracts intensity information along with metadata (station code, name, location, etc.),
         distance details, and prediction values.
       - For amplitude types (e.g., 'pga', 'pgv' or other 'sa...' values excluding intensity), it extracts a
         comprehensive set of metadata—including predictions and distances—and filters out entries with an
         instrumentType of "OBSERVED" so that only modeled (non‐observed) data are retained.

    Constructor Parameters:
    -----------------------
      - **mode (str):**  
          Currently, only `'parse'` mode is supported. Any other mode will trigger a ValueError.
      - **json_file (str):**  
          Path to the JSON file to be parsed. If not provided, a default file (for File Type 1) is used.
      - **file_type (int):**  
          Specifies the JSON file structure to assume:
            - **1:** Use the original instrumented data structure.
            - **2:** Use the alternate station list structure.

    Key Methods:
    ------------
      - **load_json()**  
          Opens and decodes the JSON data from the provided file path.
      
      - **get_dataframe(value_type='pga')**  
          Processes the loaded JSON data to construct a list of station records (dictionaries) using the 
          `construct_data_list` method, and returns the data as a pandas DataFrame. The columns and fields 
          in the DataFrame depend on the selected file type and the provided value_type (e.g., 'pga', 'intensity', 'mmi').
      
      - **get_dict(value_type='pga')**  
          Similar to get_dataframe, returns the station records as a list of dictionaries.
      
      - **construct_data_list(value_type)**  
          Core method that builds a list of dictionaries by iterating over the "features" in the JSON data.
          Extraction logic branches depending on:
            - The **file_type** (1 for original instrumented data; 2 for the alternate station list).
            - The **value_type**:
                - For amplitude measures ('pga', 'pgv', or spectral acceleration values starting with 'sa'
                  but not intensity), channel-based extraction is performed and derived fields such as HNE, HNN,
                  and HNZ are computed.
                - For intensity measures (or when value_type is not one of the expected amplitude types),
                  values are directly extracted from the feature’s properties.
      
      - **get_channel_amplitude(channel, amplitude_type)**  
          Helper method that, given a channel dictionary, looks for an amplitude entry whose 'name' contains the
          specified amplitude_type, and returns a tuple containing its value and its unit. If no matching entry is 
          found, returns (0, None).
      
      - **get_max_channel_amplitude(channels, amplitude_type)**  
          Iterates over multiple channels to determine the maximum amplitude for the given amplitude type, along
          with its unit and the associated channel name. Returns a tuple (max_value, max_unit, max_channel_id).
      
      - **get_unique_value_types()**  
          Scans through all features and channels to identify all unique amplitude type names present in the data.
      
      - **get_summary()**  
          Constructs and returns a summary dictionary including:
            - Total number of data points.
            - Count of unique stations (identified by station codes).
            - Geographic extent: minimum and maximum latitudes and longitudes.
            - Basic numeric summaries (such as mean and standard deviation) for amplitude values (e.g., 'pga') if available.
      
      - **show_statistics(value_type='pga')**  
          Displays a four-panel matplotlib figure featuring:
            - A histogram with KDE overlay of the requested amplitude data.
            - A histogram with an overlaid normal distribution fit.
            - A geographic scatter plot of station locations, colored by the amplitude or intensity values.
            - A histogram of VS30 values.
      
      - **save_statistics(value_type='pga', filename='statistics.png')**  
          Creates a figure similar to show_statistics but saves the plot to a specified file instead of displaying it.
      
      - **print_doc()**  
          Prints this complete class documentation to standard output.

    Example Usage:
    --------------
      *Parsing an Original USGS Instrumented Data File for Amplitude and Intensity:*
      
          >>> parser1 = ParseInstrumentsData(json_file='path/to/instrumented_dat.json', file_type=1)
          >>> df_amplitude = parser1.get_dataframe(value_type='pga')
          >>> df_intensity = parser1.get_dataframe(value_type='intensity')
          >>> print(df_amplitude.head())
          >>> print(df_intensity.head())
      
      *Parsing an Alternate Station List File for Intensity and Amplitude:*
      
          >>> parser2 = ParseInstrumentsData(json_file='path/to/us7000pn9s_us_020_stationlist.json', file_type=2)
          >>> df_intensity = parser2.get_dataframe(value_type='mmi')
          >>> df_amplitude = parser2.get_dataframe(value_type='pga')
          >>> print(df_intensity.head())
          >>> print(df_amplitude.head())
      
      *Displaying Data Summary and Statistical Plots:*
      
          >>> summary = parser1.get_summary()
          >>> print(summary)
          >>> parser1.show_statistics(value_type='pga')
          >>> parser1.save_statistics(value_type='pga', filename='output/statistics.png')
      
      *Viewing Class Documentation:*
      
          >>> parser1.print_doc()
      
    Additional Notes:
    -----------------
      - The class uses common scientific Python libraries such as pandas, numpy, matplotlib, and seaborn.
      - It employs logging for errors and informational messages.
      - The behavior of data extraction is controlled by the file_type and the specified value_type.
      - This parser was developed as part of the SHAKEMAP version 25.3.1 suite.

    
    © SHAKEmap version 25.3.2
    """

    def __init__(self, mode='parse', json_file=None, file_type=2):
        if mode != 'parse':
            raise ValueError("Currently, only 'parse' mode is supported.")
        if json_file is None:
            # Default to an example file for file_type 1.
            json_file = "./example_data/us7000m9g4/current/instrumented_dat.json"
            logging.info("No file provided, using default file at \"{}\"".format(json_file))
        self.json_file = json_file
        self.file_type = file_type
        self.data = self.load_json()

    def load_json(self):
        try:
            with open(self.json_file, "r") as f:
                return json.load(f)
        except FileNotFoundError as e:
            logging.error("File not found: " + self.json_file)
            raise FileNotFoundError("No event file or example file found at " + self.json_file) from e
        except json.JSONDecodeError as e:
            logging.error("Error parsing JSON data.")
            raise ValueError("Error parsing JSON data.") from e

    def get_dataframe(self, value_type='pga'):
        data_list = self.construct_data_list(value_type)
        return pd.DataFrame(data_list)

    def get_dict(self, value_type='pga'):
        return self.construct_data_list(value_type)

    def construct_data_list(self, value_type):
        """
        Constructs a list of dictionaries from the loaded JSON data. 
        How the data is parsed depends both on the file_type and on the requested value_type.
        """
        data_list = []
        # ------ File Type 1 (original instrumented data file) ------
        if self.file_type == 1 or "instrumented_dat" in self.json_file:
            # If the requested value type is one of the amplitudes,
            # use channel-based extraction.
            if value_type in ['pga', 'pgv'] or (value_type.startswith('sa') and value_type != 'intensity'):
                for feature in self.data.get('features', []):
                    channels = feature['properties'].get('channels', [])
                    channel_data = [self.get_channel_amplitude(channel, value_type) for channel in channels]
                    if channel_data:
                        amplitudes, units = zip(*channel_data)
                    else:
                        amplitudes, units = [], []
                    item = {
                        'id': feature.get('id'),
                        'station_code': feature['properties'].get('code'),
                        'station_name': feature['properties'].get('name'),
                        'longitude': feature['geometry']['coordinates'][0],
                        'latitude': feature['geometry']['coordinates'][1],
                        'location': feature['properties'].get('location'),
                        'instrumentType': feature['properties'].get('instrumentType'),
                        'source': feature['properties'].get('source'),
                        'network': feature['properties'].get('network'),
                        'vs30': feature['properties'].get('vs30'),
                        'channel_number': len(channels)
                    }
                    # Add amplitude columns based on channels if available.
                    item['HNE'] = amplitudes[0] if len(amplitudes) > 0 else 0
                    item['HNN'] = amplitudes[1] if len(amplitudes) > 1 else 0
                    item['HNZ'] = amplitudes[2] if len(amplitudes) > 2 else 0
                    item[value_type] = max(amplitudes, default=0)
                    item[f'{value_type}_unit'] = units[0] if units else None
                    data_list.append(item)
            else:
                # For other values (for example, "intensity") extract directly from the feature's properties.
                for feature in self.data.get('features', []):
                    props = feature.get('properties', {})
                    item = {
                        'id': feature.get('id'),
                        'station_code': props.get('code'),
                        'station_name': props.get('name'),
                        'longitude': feature['geometry']['coordinates'][0],
                        'latitude': feature['geometry']['coordinates'][1],
                        'location': props.get('location'),
                        'instrumentType': props.get('instrumentType'),
                        'source': props.get('source'),
                        'network': props.get('network'),
                        'vs30': props.get('vs30'),
                        value_type: props.get(value_type)
                    }
                    # Also include intensity flag and stddev if available.
                    if value_type == 'intensity':
                        item['intensity_flag'] = props.get('intensity_flag')
                        item['intensity_stddev'] = props.get('intensity_stddev')
                    data_list.append(item)
            return data_list

        # ------ File Type 2 (alternate station list file) ------
        elif self.file_type == 2 or "stationlist" in self.json_file:
            for feature in self.data.get('features', []):
                props = feature.get('properties', {})
                geometry = feature.get('geometry', [])
                # Branch for intensity data – we expect the user to provide "mmi" as the value_type in this branch.
                if value_type.lower() in ['mmi', 'intensity']:
                    # Here we extract many fields from the alternate file.
                    item = {
                        'id': feature.get('id'),
                        'station_code': props.get('code'),
                        'instrumentType': props.get('instrumentType'),
                        'commType': props.get('commType'),
                        'station_name': props.get('name'),
                        'longitude': geometry.get('coordinates')[0],
                        'latitude': geometry.get('coordinates')[1],
                        'location': props.get('location'),
                        'source': props.get('source'),
                        'network': props.get('network'),
                        'station_type': props.get('station_type'),
                        'nresp': props.get('nresp'),
                        'vs30': props.get('vs30'),
                        'intensity': props.get('intensity'),
                        'intensity_flag': props.get('intensity_flag'),
                        'intensity_stddev': props.get('intensity_stddev'),
                        'elev': props.get('elev'),
                        'distance': props.get('distance')
                    }
                    distances = props.get('distances', {})
                    item['rrup'] = distances.get('rrup')
                    item['repi'] = distances.get('repi')
                    item['rhypo'] = distances.get('rhypo')
                    item['rjb'] = distances.get('rjb')
                    # Add predictions for amplitude if available.
                    item['predictions'] = props.get('predictions')
                    item['mmi_from_pgm'] = props.get('mmi_from_pgm')
                    # Optionally, one may include channels data if available for completeness.
                    channels = props.get('channels', [])
                    item['channel_number'] = len(channels)
                    if props.get('instrumentType') == "OBSERVED":
                            data_list.append(item)
                # Branch for amplitude values ("pga", "pgv", "sa" other than intensity).
                elif value_type in ['pga', 'pgv'] or (value_type.startswith('sa') and value_type.lower() != 'mmi'):
                    # In this branch we extract a broader set of metadata.
                    channels = props.get('channels', [])
                    # For channels, attempt to extract amplitude information if available.
                    if channels:
                        channel_data = [self.get_channel_amplitude(channel, value_type) for channel in channels]
                        if channel_data:
                            amplitudes, units = zip(*channel_data)
                        else:
                            amplitudes, units = [], []
                    else:
                        amplitudes, units = [], []
                    item = {
                        'id': feature.get('id'),
                        'station_code': props.get('code'),
                        'instrumentType': props.get('instrumentType'),
                        'commType': props.get('commType'),
                        'station_name': props.get('name'),
                        'longitude': geometry.get('coordinates')[0],
                        'latitude': geometry.get('coordinates')[1],
                        'location': props.get('location'),
                        'source': props.get('source'),
                        'network': props.get('network'),
                        'station_type': props.get('station_type'),
                        'vs30': props.get('vs30'),
                        'elev': props.get('elev'),
                        'distance': props.get('distance')
                    }
                    distances = props.get('distances', {})
                    item['rrup'] = distances.get('rrup')
                    item['repi'] = distances.get('repi')
                    item['rhypo'] = distances.get('rhypo')
                    item['rjb'] = distances.get('rjb')
                    # Also include intensity fields if present.
                    item['intensity'] = props.get('intensity')
                    item['intensity_flag'] = props.get('intensity_flag')
                    item['intensity_stddev'] = props.get('intensity_stddev')
                    item['pga_selected'] = props.get('pga')
                    item['pgv_selected'] = props.get('pgv')
                    # Add predictions for amplitude if available.
                    item['predictions'] = props.get('predictions')
                    item['mmi_from_pgm'] = props.get('mmi_from_pgm')
                    # Process the channels information.
                    item['channel_number'] = len(channels)
                    item['HNE'] = amplitudes[0] if len(amplitudes) > 0 else 0
                    item['HNN'] = amplitudes[1] if len(amplitudes) > 1 else 0
                    item['HNZ'] = amplitudes[2] if len(amplitudes) > 2 else 0
                    item[value_type] = max(amplitudes, default=0)
                    item[f'{value_type}_unit'] = units[0] if units else None
                    # For file_type 2, filter out entries where instrumentType equals "OBSERVED"
                    if props.get('instrumentType') != "OBSERVED":
                        data_list.append(item)
            return data_list

        else:
            logging.error("File type not recognized.")
            return []

    def get_channel_amplitude(self, channel, amplitude_type):
        for amplitude in channel.get('amplitudes', []):
            if amplitude_type in amplitude.get('name', ''):
                return (amplitude.get('value', 0), amplitude.get('units'))
        return (0, None)

    def get_max_channel_amplitude(self, channels, amplitude_type):
        max_value = float("-inf")
        max_unit = None
        max_id = ""
        for channel in channels:
            for amplitude in channel.get('amplitudes', []):
                if amplitude_type in amplitude.get('name', ''):
                    val = amplitude.get('value', 0)
                    if val > max_value:
                        max_value = val
                        max_unit = amplitude.get('units')
                        max_id = channel.get('name', '')
        if max_value == float("-inf"):
            return 0, None, ""
        return max_value, max_unit, max_id

    def get_unique_value_types(self):
        value_types = set()
        for feature in self.data.get('features', []):
            channels = feature['properties'].get('channels', [])
            for channel in channels:
                for amplitude in channel.get('amplitudes', []):
                    if 'name' in amplitude:
                        value_types.add(amplitude['name'])
        return list(value_types)

    def get_summary(self):
        features = self.data.get('features')
        if not features:
            return {"message": "No data available."}
        summary = {"Total Data Points": len(features)}
        # Depending on file type, unique stations may be determined using 'code'
        summary["Unique Stations"] = len({f['properties'].get('code') for f in features})
        # Geographic extent
        summary["Geographic Extent"] = {
            "Min Latitude": min(f['geometry']['coordinates'][1] for f in features),
            "Max Latitude": max(f['geometry']['coordinates'][1] for f in features),
            "Min Longitude": min(f['geometry']['coordinates'][0] for f in features),
            "Max Longitude": max(f['geometry']['coordinates'][0] for f in features)
        }
        # Optionally, add summaries for a given value_type if it is numeric.
        # Here we build a summary based on the dataframe produced.
        try:
            df = self.get_dataframe('pga')
            if not df.empty:
                mean_val = df['pga'].mean()
                std_val = df['pga'].std()
                summary['PGA Average'] = mean_val
                summary['PGA Std Dev'] = std_val
            # Similar statistics for intensity or mmi could be added.
        except Exception as e:
            logging.warning("Could not compute amplitude summary: " + str(e))
        return summary

    def show_statistics(self, value_type='pga'):
        df = self.get_dataframe(value_type)
        if df.empty:
            logging.warning("The dataframe is empty. Check your data source or filters.")
            return
        unit = df.get(f'{value_type}_unit').iloc[0] if f'{value_type}_unit' in df.columns else ""
        data = df[value_type]
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        # Plot 1: Histogram with KDE overlay
        axs[0, 0].hist(data, bins=30, density=True, alpha=0.6, color='b')
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        axs[0, 0].plot(x_range, kde(x_range), color='r', lw=2)
        mean_val = data.mean()
        std_val = data.std()
        axs[0, 0].axvline(mean_val, color='k', linestyle='--', label="Mean: {:.2f}".format(mean_val))
        axs[0, 0].axvline(mean_val + std_val, color='g', linestyle='--', label="+1 Std: {:.2f}".format(std_val))
        axs[0, 0].axvline(mean_val - std_val, color='g', linestyle='--')
        axs[0, 0].set_title("Probability Distribution of {} [{}]".format(value_type, unit))
        axs[0, 0].legend()
        # Plot 2: Histogram with Normal Fit
        axs[0, 1].hist(data, bins=20, density=True, alpha=0.6, color='c')
        xmin, xmax = axs[0, 1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean_val, std_val)
        axs[0, 1].plot(x, p, 'k', linewidth=2)
        axs[0, 1].set_title("Histogram of {} with Normal Fit [{}]".format(value_type, unit))
        # Plot 3: Geographic Scatter Plot
        scatter = axs[1, 0].scatter(df['longitude'], df['latitude'], c=data, cmap='seismic')
        cbar = plt.colorbar(scatter, ax=axs[1, 0])
        cbar.set_label("{} [{}]".format(value_type, unit))
        axs[1, 0].set_xlabel("Longitude")
        axs[1, 0].set_ylabel("Latitude")
        axs[1, 0].set_title("Location Plot Colored by {}".format(value_type))
        # Plot 4: Histogram of VS30 values
        axs[1, 1].hist(df['vs30'], bins=20, color='m', alpha=0.7)
        axs[1, 1].set_title("Histogram of VS30 values")
        axs[1, 1].set_xlabel("VS30")
        axs[1, 1].set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def save_statistics(self, value_type='pga', filename='statistics.png'):
        df = self.get_dataframe(value_type)
        if df.empty:
            logging.warning("Dataframe is empty. Cannot save statistics.")
            return
        unit = df.get(f'{value_type}_unit').iloc[0] if f'{value_type}_unit' in df.columns else ""
        data = df[value_type]
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].hist(data, bins=30, density=True, alpha=0.6, color='b')
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        axs[0, 0].plot(x_range, kde(x_range), color='r', lw=2)
        mean_val = data.mean()
        std_val = data.std()
        axs[0, 0].axvline(mean_val, color='k', linestyle='--', label="Mean: {:.2f}".format(mean_val))
        axs[0, 0].axvline(mean_val + std_val, color='g', linestyle='--', label="+1 Std: {:.2f}".format(std_val))
        axs[0, 0].axvline(mean_val - std_val, color='g', linestyle='--')
        axs[0, 0].set_title("Probability Distribution of {} [{}]".format(value_type, unit))
        axs[0, 0].legend()
        axs[0, 1].hist(data, bins=20, density=True, alpha=0.6, color='c')
        xmin, xmax = axs[0, 1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean_val, std_val)
        axs[0, 1].plot(x, p, 'k', linewidth=2)
        axs[0, 1].set_title("Histogram of {} with Normal Fit [{}]".format(value_type, unit))
        scatter = axs[1, 0].scatter(df['longitude'], df['latitude'], c=data, cmap='seismic')
        cbar = plt.colorbar(scatter, ax=axs[1, 0])
        cbar.set_label("{} [{}]".format(value_type, unit))
        axs[1, 0].set_xlabel("Longitude")
        axs[1, 0].set_ylabel("Latitude")
        axs[1, 0].set_title("Location Plot Colored by {}".format(value_type))
        axs[1, 1].hist(df['vs30'], bins=20, color='m', alpha=0.7)
        axs[1, 1].set_title("Histogram of VS30 values")
        axs[1, 1].set_xlabel("VS30")
        axs[1, 1].set_ylabel("Frequency")
        plt.tight_layout()
        # Ensure the output directory exists.
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        plt.savefig(filename)
        logging.info("Statistics plot saved as {}".format(filename))
        plt.close(fig)

    def print_doc(self):
        print(self.__doc__)



class ParseDYFIDataXML:
    """
     A class for parsing and creating DYFI (Did You Feel It?) XML files conforming to the USGS ShakeMap system format.
    This class operates in two modes:
    
      - 'parse': Reads an existing DYFI XML file and populates the station data from it.
      - 'create': Prepares to generate a new DYFI XML file using station data provided in a CSV file.
      
    In 'parse' mode, the class extracts station data from the XML file and stores it in an internal list (as dictionaries),
    which can later be accessed as a pandas DataFrame or a list of dictionaries. In 'create' mode, the class reads station data 
    from a CSV file, computes UTM coordinate codes for each station based on its latitude and longitude, and stores the 
    prepared station attributes. The XML file is only written when the explicit method write_dyfi_xml() is called.

    Attributes:
    -----------
    mode : str
        Operational mode of the class; must be either 'parse' or 'create'. Any other value raises a ValueError.
    xml_file : str
        In 'parse' mode, the file path to the existing DYFI XML file. In 'create' mode, the file path where the new
        XML file will be written.
    event_id : str
        A unique identifier for the earthquake event; defaults to 'unknown' if not provided. In 'create' mode, this 
        is used to form the default XML file path.
    csv_file : str or None
        In 'create' mode, the file path to the CSV file containing the station data. This parameter is required 
        when mode is 'create'.
    stations : list
        A list of dictionaries where each dictionary contains the data for one station. In 'parse' mode, it is populated 
        by parsing the XML file; in 'create' mode, it is populated by reading the CSV file.
    
    Methods:
    --------
    parse_xml(self):
        Parses the specified DYFI XML file and populates the internal stations list with station data.
        Each station’s data is stored as a dictionary of XML attribute values.
    
    create_xml_from_csv(self):
        Reads station data from a CSV file, computes UTM codes for each station using latitude and longitude,
        and populates the stations list with these attributes. Note that this method prepares the data but does not
        write the XML file; the method write_dyfi_xml() must be explicitly called afterward when in 'create' mode.
    
    write_dyfi_xml(self):
        Writes the prepared station data from the stations list to an XML file at the path specified in xml_file.
        This method should be called only in 'create' mode.
    
    latlon_to_utm_code(lat, lon):
        (Static Method) Converts a given latitude and longitude into a UTM coordinate code string.
        This function uses the EPSG:4326 (WGS84) coordinate system as input and converts it to the
        corresponding UTM zone, returning a formatted string.
    
    prettify(element, indent='  '):
        (Static Method) Recursively formats an XML element tree by adding indentation and newlines.
        This results in an output XML file that is human-readable.
    
    get_dataframe(self):
        Returns a pandas DataFrame constructed from the parsed (or created) station data.
        Raises a ValueError if no data is available.
    
    get_dict(self):
        Returns the station data as a list of dictionaries.
    
    get_summary(self):
        Computes and returns a detailed summary of the dataset, including:
            - The total number of station observations.
            - Basic geographical extents (minimum and maximum latitudes and longitudes).
            - Counts and distributions of responses (e.g., number of responses, distances)
              as available in the dataset.
    
    get_statistics(self):
        Generates and returns simple statistics (mean, standard deviation, maximum, and minimum)
        for the intensity data, after converting the 'intensity' column to numeric values.
        If the intensity data is non-numeric or missing, a warning is printed.
    
    show_statistics(self):
        Displays a matplotlib figure with a four-panel plot that includes:
            - A histogram with a kernel density estimate (KDE) overlay of the intensity observations.
            - A histogram with an overlaid normal distribution fit.
            - A geographic scatter plot of station locations, colored by the intensity values.
            - A histogram of VS30 values.
    
    print_doc(self):
        Prints the complete class documentation (this docstring) to standard output,
        which provides a quick reference for usage and available methods.

    Usage Examples:
    ---------------
    **Parsing an Existing DYFI XML File:**
    
        >>> parser = ParseDYFIDataXML(mode='parse', xml_file='path/to/dyfi_dat.xml')
        >>> df = parser.get_dataframe()
        >>> print(df.head())
    
    **Creating a New DYFI XML File from CSV Data:**
    
        >>> creator = ParseDYFIDataXML(mode='create', csv_file='path/to/stations.csv', event_id='us7000m9g4')
        >>> # The CSV file is read and station data is prepared.
        >>> creator.write_dyfi_xml()  # Call this method to write the XML file.
        >>> summary = creator.get_summary()
        >>> print(summary)
    
    Additional Notes:
    -----------------
    - In 'parse' mode, if no XML file is provided, a default example file is used.
    - In 'create' mode, if no CSV file is provided, a ValueError is raised.
    - The class supports conversion of geographic coordinates to UTM codes,
      which are used to uniquely identify stations in the output XML.
    - The output XML file is not beautified automatically unless prettify() is called within write_dyfi_xml().
    - Use the print_doc() method to view this documentation from within an interactive Python session.

    © SHAKEMAP version 25.3.2
    """

    def __init__(self, mode='parse', xml_file=None, event_id=None, csv_file=None):
        """
        Initialize the object either by parsing an XML file or preparing to create one from CSV.
        """
        self.mode = mode
        self.event_id = event_id or "unknown"
        self.xml_file = xml_file
        self.csv_file = csv_file
        self.stations = []

        if mode == 'create':
            if not csv_file:
                if not event_id:
                    self.event_id = "us7000m9g4"  # Default event ID
                    self.csv_file = "./example_data/us7000m9g4/current/us7000m9g4_dyfi_dat.csv"
                    print("No CSV file or event ID provided, using default values.")
                else:
                    raise ValueError("CSV file path must be provided in 'create' mode.")
            self.xml_file = f"./export/usgs-scenarios/{self.event_id}/current/dyfi_dat.xml"
            self.create_xml_from_csv()
        elif mode == 'parse':
            if not xml_file:
                self.xml_file = "./example_data/us7000m9g4/current/dyfi_dat.xml"
                print("No XML file provided, using default example data path for parsing.")
            self.stations = self.parse_xml()
        else:
            raise ValueError("Mode must be either 'parse' or 'create' try mode='parse'")
    
    def parse_xml(self):
        """
        Parses the specified XML file, extracting data for each station and storing it in the `stations` list.
        Each station's data is represented as a dictionary.
        """

        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        stations = []

        for station_element in root.find('stationlist').findall('station'):
            station_data = {attr: station_element.get(attr) for attr in station_element.keys()}
            stations.append(station_data)

        return stations

    def get_dataframe(self):
        """
        Converts the parsed XML or created XML data into a pandas DataFrame.
        """
        if not self.stations:
            raise ValueError("No data available to convert to DataFrame. Ensure the XML is parsed or created correctly.")
        return pd.DataFrame(self.stations)

    def get_dict(self):
        """
        Converts the station data to a list of dictionaries.
        """
        return self.stations

    def get_statistics(self):
        """Generates simple statistics of the dataset, handling non-numeric data."""
        df = self.get_dataframe()
        if df.empty:
            print("The dataframe is empty. Check your data source or filters.")
            return {}

        # Convert 'intensity' to numeric, handling non-numeric gracefully
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')

        # Check if all values are NaN after conversion
        if df['intensity'].isna().all():
            print("Intensity data is not numeric and couldn't be converted. Check your data source.")
            return {}

        # Compute statistics only on valid numeric data
        statistics = {
            "Mean Intensity": df['intensity'].mean(),
            "Standard Deviation of Intensity": df['intensity'].std(),
            "Max Intensity": df['intensity'].max(),
            "Min Intensity": df['intensity'].min(),
            "Count of Valid Intensity Entries": df['intensity'].notna().sum()  # Count non-NaN entries
        }
        return statistics
    
    def get_summary(self):
        """Provides a detailed summary of the dataset including counts of data points and specifics about other attributes."""
        df = self.get_dataframe()
        if df.empty:
            print("The dataframe is empty. Check your data source or filters.")
            return {}

        # Ensure necessary columns are numeric for accurate comparisons and calculations
        df['nresp'] = pd.to_numeric(df['nresp'], errors='coerce')
        df['dist'] = pd.to_numeric(df['dist'], errors='coerce')

        # Basic geographic and response statistics
        summary = {
            "Total Ovservations": len(df),
            "Stations with More Than 3 Responses": (df['nresp'] > 3).sum(),
            "Min Latitude": df['lat'].min(),
            "Max Latitude": df['lat'].max(),
            "Min Longitude": df['lon'].min(),
            "Max Longitude": df['lon'].max(),
            "Average Distance": df['dist'].mean(),
            "Max Distance": df['dist'].max(),
            "Min Distance": df['dist'].min()
        }

        # Extract unique values and counts for categorical data
        summary['Unique Sources'] = df['source'].dropna().unique().tolist()
        summary['Unique NetIDs'] = df['netid'].dropna().unique().tolist()

        # Adding more categorical data insights
        source_counts = df['source'].value_counts().to_dict()
        netid_counts = df['netid'].value_counts().to_dict()

        summary['Source Distribution'] = source_counts
        summary['NetID Distribution'] = netid_counts

        return summary

    def show_statistics(self):
        """Visualizes the probability distribution of the DYFI dataset using a histogram."""
        df = self.get_dataframe()
        if not df.empty:
            # Ensure the column is numeric, coercing any errors
            df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')

            # Check if the column has only NaNs after conversion
            if df['intensity'].isna().all():
                print("Intensity data is not numeric and couldn't be converted. Check your data source.")
                return

            value_type = 'intensity'  # Assuming 'intensity' is a column in your DataFrame
            unit = 'MMI'  # Assuming the units are in some form of magnitude scale

            # Create figure and axis
            plt.figure(figsize=(10, 6))

            # Probability Distribution with Histogram
            mean = df[value_type].mean()
            std = df[value_type].std()
            sns.histplot(df[value_type].dropna(), bins=30, kde=True, color='blue')
            plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
            plt.axvline(mean + std, color='g', linestyle='--', label=f'Std Dev: ±{std:.2f}')
            plt.axvline(mean - std, color='g', linestyle='--')
            plt.title(f'Probability Distribution of {value_type} DYFI? observations [{unit}]')
            plt.legend()
            plt.xlabel(f'{value_type} [{unit}]')
            plt.ylabel('Frequency')

            plt.tight_layout()
            plt.show()
        else:
            print("The dataframe is empty. Check your data source or filters.")
            
    def create_xml_from_csv(self):
        """
        Reads data from a specified CSV file, calculates UTM codes, and stores each station's data in the `stations` list.
        This method prepares data but does not write the XML file; `write_dyfi_xml()` should be called to save the data.
        """
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"The CSV file {self.csv_file} does not exist.")
        
        df = pd.read_csv(self.csv_file)
        print(f"CSV loaded with {len(df)} rows.")

        for _, row in df.iterrows():
            station_attrs = {
                'code': self.latlon_to_utm_code(row['lat'], row['lon']),
                'lat': str(row['lat']),
                'lon': str(row['lon']),
                'netid': row['netid'],
                'dist': str(row['dist']),
                'intensity': str(row['intensity']),
                'intensity_stddev': str(row['intensity_stddev']),
                'nresp': str(row['nresp']),
                'source': row['source']
            }
            self.stations.append(station_attrs)
        
    def write_dyfi_xml(self):
        """
        Writes the prepared station data from `stations` list to an XML file specified by `xml_file`.
        This method should be explicitly called after data preparation in 'create' mode.
        """
        if self.mode == 'create':
            root = ET.Element('SHAKEdata', attrib={'code_version': "3.5", 'map_version': "3"})
            stationlist = ET.SubElement(root, 'stationlist', attrib={'created': "1710943788", 'reference': "USGS Did You Feel It? System"})

            for station in self.stations:
                ET.SubElement(stationlist, 'station', station)

            self.prettify(root)
            os.makedirs(os.path.dirname(self.xml_file), exist_ok=True)
            tree = ET.ElementTree(root)
            tree.write(self.xml_file, encoding='utf-8', xml_declaration=False)
            print(f'XML file created successfully at {self.xml_file}')
        else: print("Cant write XML file in 'parse' mode.")


    @staticmethod
    def prettify(element, indent='  '):
        """ Recursively add indentations and newlines to the XML element tree """
        queue = [(0, element)]  # (level, element)
        while queue:
            level, element = queue.pop(0)
            children = list(element)
            if children:
                element.text = '\n' + (indent * (level+1))  # Text for children start
            if element.tail is None:
                element.tail = '\n' + (indent * level)  # Tail text for this element
            queue.extend([(level + 1, child) for child in children])

    @staticmethod
    def latlon_to_utm_code(lat, lon):
        try:
            # Define the projections
            proj_latlon = CRS("EPSG:4326")  # WGS 84
            utm_zone = int((lon + 180) / 6) + 1
            proj_utm = CRS(f"EPSG:326{utm_zone}") if lat >= 0 else CRS(f"EPSG:327{utm_zone}")

            # Create a transformer
            transformer = Transformer.from_crs(proj_latlon, proj_utm, always_xy=True)

            # Transform the coordinates
            east, north = transformer.transform(lon, lat)

            # Calculate UTM band
            utm_band = 'CDEFGHJKLMNPQRSTUVWX'[(int(lat) + 80) // 8]
            if lat < -80 or lat >= 84:
                return "Latitude out of range for UTM"

            # Round coordinates to the nearest kilometer and format
            east = int(east / 1000)
            north = int(north / 1000)
            utm_code = f"DYFI.UTM:({utm_zone}{utm_band} {east:04d} {north:04d} 1000)"
            return utm_code
        except Exception as e:
            print(f"Error converting to UTM: {e}")
            return "Conversion Error"
        
    def print_doc(self):
        """Prints the class docstring."""
        print(self.__doc__)




class ParseDYFIData:
    """
    ParseDYFIData is a versatile parser class designed to load and process various types of DYFI data files.
    DYFI (Did You Feel It?) data comes in several different formats, including plain text and XML, representing CDI
    (Community Determined Intensity) GEO or ZIP data as well as specialized plot data such as the number of responses
    over time and attenuation curves. This class automatically detects the file type based on the file extension and
    keywords in the file name (unless explicitly specified) and then calls the corresponding parser.

    **Supported File Types:**
      - **CDI GEO Files (Plain Text and XML):**
          - File type 1 ("cdi_geo_txt"): Standard CDI GEO data in plain text.
          - File type 2 ("cdi_geo_xml"): CDI GEO data in XML.
          - File type 3 ("cdi_geo_1km"): 1km resolution CDI GEO data in plain text.
      - **CDI ZIP Files (Plain Text and XML):**
          - File type 4 ("cdi_zip"): CDI ZIP data in plain text.
          - File type 5 ("cdi_zip_xml"): CDI ZIP data in XML.
      - **Plot Data:**
          - File type 6 ("plot_numresp"): Plot data for the number of responses over time.
          - File type 7 ("plot_atten"): Attenuation curves data with multiple sections. In these files, each section
            is preceded by a header line (beginning with "::") and terminated by an "END" marker.

    **Features and Available Functions:**

      - **Auto-detection and Parsing:**
          - `detect_file_type()`: Automatically detect the file type based on the file name and extension.
          - `parse_file()`: Calls the appropriate parsing method based on the detected (or given) file type.
          - Dedicated parsing methods:
              - `parse_cdi_geo()`: Parse plain text CDI GEO files.
              - `parse_cdi_geo_1km()`: Parse 1km resolution CDI GEO text files.
              - `parse_cdi_zip()`: Parse plain text CDI ZIP files.
              - `parse_cdi_geo_xml()`: Parse CDI GEO files in XML format.
              - `parse_cdi_zip_xml()`: Parse CDI ZIP files in XML format.
              - `parse_plot_numresp()`: Parse plain text files containing plot data for number of responses over time.
              - `parse_plot_atten()`: Parse attenuation curve files that include multiple sections separated by header lines and "END" markers.
    
      - **Data Retrieval:**
          - `get_dataframe()`: Returns the parsed data as a single pandas DataFrame (for non-multi-section files).
          - `get_dict()`: Returns the parsed data as a list of dictionaries.
          - For attenuation plot files:
              - `get_atten_data()`: Returns the parsed attenuation sections as a dictionary mapping section headers to DataFrames.
              - `get_atten_dict()`: Returns the parsed attenuation data as a dictionary mapping section headers to lists of dictionaries.
              - `get_atten_dataframe()`: Combines all parsed attenuation sections into a single DataFrame with an additional column ("section")
                that indicates the original section header.
    
      - **Plotting Capabilities:**
          - `plot_numresp(x_col="absolute_time", y_col="y", ...)`: Generates a plot for number of responses over time by converting the 
            "absolute_time" column to datetime and plotting against the "y" column.
          - `plot_atten()`: Creates scatter plots for each attenuation section using the first two numeric columns.
    
      - **Documentation Display:**
          - `print_doc()`: Prints the class’s detailed docstring to help users understand the functionality and usage.

    **Example Usage:**

      >>> from your_module import ParseDYFIData
      >>> # Initialize a parser for a CDI GEO text file; file type is auto-detected.
      >>> parser = ParseDYFIData("data/us7000pn9s_us_1_cdi_geo.txt")
      >>> # Retrieve the data as a DataFrame and view the first few records.
      >>> df = parser.get_dataframe()
      >>> print(df.head())
      >>> 
      >>> # For plot_numresp data files:
      >>> parser_numresp = ParseDYFIData("data/us7000pn9s_us_1_us7000pn9s_plot_numresp.txt")
      >>> parser_numresp.parse_plot_numresp()
      >>> parser_numresp.plot_numresp()
      >>> 
      >>> # For attenuation curves:
      >>> parser_atten = ParseDYFIData("data/us7000pn9s_us_1_us7000pn9s_plot_atten.txt")
      >>> parser_atten.parse_plot_atten()
      >>> parser_atten.plot_atten()
      >>> # Obtain a combined DataFrame of all attenuation sections:
      >>> combined_atten_df = parser_atten.get_atten_dataframe()
      >>> print(combined_atten_df.head())
      >>> 
      >>> # To display this class's detailed documentation:
      >>> parser.print_doc()

    **Notes:**
      - Files are read using the 'latin-1' encoding by default. Adjust the encoding as needed.
      - When parsing plot data, it is assumed that the columns in plot_numresp files are:
          x, y, seconds_past_origin, and absolute_time (with absolute_time convertible by pandas.to_datetime).
      - In attenuation plot files, sections are identified by lines that start with "::" and terminated by "END".


      © SHAKEmaps version 25.3.2
    """
    

    
    def __init__(self, file_path, file_type=None):
        """
        Initialize the parser with the file path and an optional file_type.
        If file_type is None, auto-detection is performed.
        
        Manual file type selections:
          - 1 or "cdi_geo_txt": Plain text parser for standard CDI GEO files.
          - 2 or "cdi_geo_xml": XML parser for CDI GEO files.
          - 3 or "cdi_geo_1km": Plain text parser for 1km resolution CDI GEO files.
          - 4 or "cdi_zip": Plain text parser for CDI ZIP files.
          - 5 or "cdi_zip_xml": XML parser for CDI ZIP files.
          - 6 or "plot_numresp": Plain text parser for plot data (number of responses vs. time).
          - 7 or "plot_atten": Plain text parser for attenuation curves with multiple sections.
        """
        self.file_path = file_path
        if file_type is None:
            self.file_type = self.detect_file_type()
        else:
            self.file_type = file_type
        self.df = None         # DataFrame to hold parsed data (for single-type files).
        self.data = None       # List of dictionaries.
        # For attenuation plot files with multiple curves.
        self.atten_sections = {}
        self.atten_data = {}
        self.parse_file()      # Parse the file upon initialization

    def detect_file_type(self):
        """
        Auto-detect the file type based on the file name.
        
        For .xml files:
          - If the name contains "cdi_zip" → type 5.
          - Else if the name contains "cdi_geo" → type 2.
        For .txt files:
          - If the name contains "cdi_geo_1km" → type 3.
          - Else if the name contains "cdi_zip" → type 4.
          - Else if the name contains "plot_numresp" → type 6.
          - Else if the name contains "plot_atten" → type 7.
          - Else if the name contains "cdi_geo" → type 1.
          Otherwise returns "unknown".
        """
        file_lower = self.file_path.lower()
        if file_lower.endswith(".xml"):
            if "cdi_zip" in file_lower:
                detected = 5
            elif "cdi_geo" in file_lower:
                detected = 2
            else:
                detected = "unknown"
        elif file_lower.endswith(".txt"):
            if "cdi_geo_1km" in file_lower:
                detected = 3
            elif "cdi_zip" in file_lower:
                detected = 4
            elif "plot_numresp" in file_lower:
                detected = 6
            elif "plot_atten" in file_lower:
                detected = 7
            elif "cdi_geo" in file_lower:
                detected = 1
            else:
                detected = "unknown"
        else:
            detected = "unknown"
        print(f"Auto-detected file type: {detected}")
        return detected

    def parse_file(self):
        """
        Call the appropriate parsing method based on file_type.
        
        Mapping:
          - 1 or "cdi_geo_txt"  → parse_cdi_geo()
          - 2 or "cdi_geo_xml"  → parse_cdi_geo_xml()
          - 3 or "cdi_geo_1km"  → parse_cdi_geo_1km()
          - 4 or "cdi_zip"      → parse_cdi_zip()
          - 5 or "cdi_zip_xml"  → parse_cdi_zip_xml()
          - 6 or "plot_numresp" → parse_plot_numresp()
          - 7 or "plot_atten"   → parse_plot_atten()
        """
        if self.file_type in [1, "cdi_geo_txt"]:
            self.parse_cdi_geo()
        elif self.file_type in [2, "cdi_geo_xml"]:
            self.parse_cdi_geo_xml()
        elif self.file_type in [3, "cdi_geo_1km"]:
            self.parse_cdi_geo_1km()
        elif self.file_type in [4, "cdi_zip"]:
            self.parse_cdi_zip()
        elif self.file_type in [5, "cdi_zip_xml"]:
            self.parse_cdi_zip_xml()
        elif self.file_type in [6, "plot_numresp"]:
            self.parse_plot_numresp()
        elif self.file_type in [7, "plot_atten"]:
            self.parse_plot_atten()
        else:
            raise NotImplementedError(f"Parsing for file type '{self.file_type}' is not implemented.")

    def parse_cdi_geo(self):
        """Parse a plain text file (standard CDI GEO)."""
        try:
            with open(self.file_path, "r", encoding="latin-1") as f:
                lines = f.readlines()
            header_line = next((line.strip() for line in lines if line.strip().startswith("# Columns:")), None)
            if header_line:
                columns_line = header_line[len("# Columns:"):].strip()
                columns = [col.strip() for col in columns_line.split(",")]
            else:
                columns = None
            data_lines = [line for line in lines if not line.strip().startswith("#") and line.strip()]
            data_str = "".join(data_lines)
            if columns:
                self.df = pd.read_csv(StringIO(data_str), header=None, names=columns)
            else:
                self.df = pd.read_csv(StringIO(data_str), header=None)
            self.data = self.df.to_dict(orient="records")
        except Exception as e:
            print("Error parsing CDI GEO text file:", e)

    def parse_cdi_geo_1km(self):
        """Parse a plain text file for 1km resolution CDI GEO data."""
        try:
            with open(self.file_path, "r", encoding="latin-1") as f:
                lines = f.readlines()
            header_line = next((line.strip() for line in lines if line.strip().startswith("# Columns:")), None)
            if header_line:
                columns_line = header_line[len("# Columns:"):].strip()
                columns = [col.strip() for col in columns_line.split(",")]
            else:
                columns = None
            data_lines = [line for line in lines if not line.strip().startswith("#") and line.strip()]
            data_str = "".join(data_lines)
            if columns:
                self.df = pd.read_csv(StringIO(data_str), header=None, names=columns)
            else:
                self.df = pd.read_csv(StringIO(data_str), header=None)
            self.data = self.df.to_dict(orient="records")
        except Exception as e:
            print("Error parsing CDI GEO 1km text file:", e)

    def parse_cdi_zip(self):
        """Parse a plain text file of type CDI ZIP."""
        try:
            with open(self.file_path, "r", encoding="latin-1") as f:
                lines = f.readlines()
            header_line = next((line.strip() for line in lines if line.strip().startswith("# Columns:")), None)
            if header_line:
                columns_line = header_line[len("# Columns:"):].strip()
                columns = [col.strip() for col in columns_line.split(",")]
            else:
                columns = None
            data_lines = [line for line in lines if not line.strip().startswith("#") and line.strip()]
            data_str = "".join(data_lines)
            if columns:
                self.df = pd.read_csv(StringIO(data_str), header=None, names=columns)
            else:
                self.df = pd.read_csv(StringIO(data_str), header=None)
            self.data = self.df.to_dict(orient="records")
        except Exception as e:
            print("Error parsing CDI ZIP text file:", e)

    def parse_cdi_geo_xml(self):
        """Parse an XML file of type CDI GEO."""
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            cdi_elem = root.find("cdi")
            if cdi_elem is None:
                raise ValueError("The XML file does not contain a <cdi> element.")
            parent_attrs = cdi_elem.attrib
            records = []
            for loc in cdi_elem.findall("location"):
                record = dict(parent_attrs)
                record.update(loc.attrib)
                for child in loc:
                    record[child.tag] = child.text
                records.append(record)
            self.data = records
            self.df = pd.DataFrame(records)
        except Exception as e:
            print("Error parsing CDI GEO XML file:", e)

    def parse_cdi_zip_xml(self):
        """Parse an XML file of type CDI ZIP."""
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            cdi_elem = root.find("cdi")
            if cdi_elem is None:
                raise ValueError("The XML file does not contain a <cdi> element.")
            parent_attrs = cdi_elem.attrib
            records = []
            for loc in cdi_elem.findall("location"):
                record = dict(parent_attrs)
                record.update(loc.attrib)
                for child in loc:
                    record[child.tag] = child.text
                records.append(record)
            self.data = records
            self.df = pd.DataFrame(records)
        except Exception as e:
            print("Error parsing CDI ZIP XML file:", e)

    def parse_plot_numresp(self):
        """
        Parse a plain text file for plot data.
        
        The file is space-delimited with four columns (in order):
          x, y, seconds_past_origin, absolute_time
        Header lines (starting with "#") are skipped.
        """
        try:
            default_columns = ["x", "y", "seconds_past_origin", "absolute_time"]
            with open(self.file_path, "r") as f:
                lines = f.readlines()
            data_lines = [line for line in lines if not line.strip().startswith("#") and line.strip()]
            data_str = "".join(data_lines)
            self.df = pd.read_csv(StringIO(data_str), header=None, delim_whitespace=True, names=default_columns)
            self.data = self.df.to_dict(orient="records")
        except Exception as e:
            print("Error parsing Plot NumResp text file:", e)

    def parse_plot_atten(self):
        """
        Parse a plain text file containing multiple attenuation curves.
        
        The file is structured in sections. A section starts with a header line that begins with "::"
        (e.g., "::scatterplot1 All reported data dist=point") and continues with numeric data lines.
        A section's data ends when a line equal (case-insensitive) to "END" is encountered.
        
        For each section, the numeric data is assumed to be space-delimited. The number of columns is
        determined by the first data line; default column names are then assigned (col0, col1, etc.).
        
        The parsed sections are stored in:
            - self.atten_sections: dictionary mapping section header (without the "::" prefix) to DataFrame
            - self.atten_data: dictionary mapping section header to a list of dictionaries (records)
        """
        try:
            with open(self.file_path, "r", encoding="latin-1") as f:
                lines = f.readlines()
                
            sections = {}   # to hold DataFrame for each section
            current_header = None
            data_block = []
            
            for line in lines:
                stripped = line.strip()
                # If we encounter a section header (line starting with "::")
                if stripped.startswith("::"):
                    # If we were processing a section, save its data block first.
                    if current_header is not None and data_block:
                        try:
                            # Determine the number of columns from the first non-empty line.
                            num_cols = len(data_block[0].split())
                            col_names = [f"col{i}" for i in range(num_cols)]
                            df = pd.read_csv(StringIO("\n".join(data_block)),
                                             header=None, delim_whitespace=True, names=col_names)
                            sections[current_header] = df
                        except Exception as e:
                            print(f"Error parsing section '{current_header}':", e)
                        # Reset the data block for the new section.
                        data_block = []
                    # Set new current section header (remove the leading "::")
                    current_header = stripped[2:].strip()
                
                # If we encounter a line that is just "END" (case-insensitive),
                # finish the current section.
                elif stripped.upper() == "END":
                    if current_header is not None and data_block:
                        try:
                            num_cols = len(data_block[0].split())
                            col_names = [f"col{i}" for i in range(num_cols)]
                            df = pd.read_csv(StringIO("\n".join(data_block)),
                                             header=None, delim_whitespace=True, names=col_names)
                            sections[current_header] = df
                        except Exception as e:
                            print(f"Error parsing section '{current_header}':", e)
                    # Reset for the next section.
                    current_header = None
                    data_block = []
                else:
                    # If we are within a section (i.e. current_header is set) and this is non-empty
                    if current_header is not None and stripped:
                        data_block.append(stripped)
                        
            # If the file does not end with an "END", save any remaining data.
            if current_header is not None and data_block:
                try:
                    num_cols = len(data_block[0].split())
                    col_names = [f"col{i}" for i in range(num_cols)]
                    df = pd.read_csv(StringIO("\n".join(data_block)),
                                     header=None, delim_whitespace=True, names=col_names)
                    sections[current_header] = df
                except Exception as e:
                    print(f"Error parsing section '{current_header}':", e)
                    
            self.atten_sections = sections
            self.atten_data = {k: df.to_dict(orient="records") for k, df in sections.items()}
            if sections:
                print("Attenuation sections loaded successfully:")
                for sec, df in sections.items():
                    print(f"Section: {sec}, rows: {len(df)}")
            else:
                print("No sections were found in the attenuation plot file.")
        except Exception as e:
            print("Error parsing Plot Attenuation text file:", e)
        
    def plot_numresp(self, x_col="absolute_time", y_col="y", title="Number of Responses Over Time", xlabel="Absolute Time", ylabel="Number of Responses"):
        """
        Display a plot of the data (for files with a single DataFrame) using matplotlib.
        For plot_numresp files: x-axis is "absolute_time" (converted to datetime) and y-axis is "y".
        """
        if self.df is None:
            print("Data not loaded.")
            return
        if x_col not in self.df.columns or y_col not in self.df.columns:
            print(f"Columns {x_col} and/or {y_col} not found in the DataFrame.")
            return
        try:
            self.df[x_col] = pd.to_datetime(self.df[x_col])
        except Exception as e:
            print("Error converting x-axis column to datetime:", e)
        plt.figure(figsize=(10, 6))
        plt.plot(self.df[x_col], self.df[y_col], marker='o', linestyle='-')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_atten(self):
        """
        Create a plot for each attenuation section.
        
        For each section stored in self.atten_sections, assume that the first two columns are x and y
        values for a scatter plot. The function creates one figure per section.
        """
        if not hasattr(self, "atten_sections") or not self.atten_sections:
            print("No attenuation data loaded to plot.")
            return
    
        for header, df in self.atten_sections.items():
            # Check that the DataFrame has at least two columns
            if df.shape[1] < 2:
                print(f"Section '{header}' does not have enough columns to plot.")
                continue
            plt.figure()
            plt.scatter(df['col0'], df['col1'], c='blue', label=header)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Attenuation Plot: {header}")
            plt.legend()
            plt.grid(True)
            plt.show()

    def get_dataframe(self):
        """Return the parsed single DataFrame (for non-multi-section files)."""
        return self.df

    def get_dict(self):
        """Return the parsed data as a list of dictionaries (for non-multi-section files)."""
        return self.data

    def get_atten_data(self):
        """Return the parsed attenuation sections as a dictionary of DataFrames."""
        return self.atten_sections

    def get_atten_dict(self):
        """Return the parsed attenuation sections as a dictionary 
        mapping section names to lists of dictionaries."""
        return self.atten_data


    def get_atten_dataframe(self):
        """
        Combine all parsed attenuation sections into a single DataFrame.
    
        Each section is tagged with a new column 'section' that contains the section header.
        The individual DataFrames (which were parsed with default column names such as 'col0', 'col1', etc.)
        are concatenated (row-wise) into one DataFrame which is then returned.
    
        Returns:
            pandas.DataFrame: Combined DataFrame with an extra column 'section'.
                              If no attenuation sections have been loaded, None is returned.
        """
        if not hasattr(self, "atten_sections") or not self.atten_sections:
            print("No attenuation sections loaded. Please run parse_plot_atten() first.")
            return None
    
        dfs = []
        for header, df in self.atten_sections.items():
            df_copy = df.copy()
            df_copy['section'] = header
            dfs.append(df_copy)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df


    def print_doc(self):
        """
        Print the class docstring.

        This method prints the documentation string of the ParseDYFIData class
        to the standard output.
        """
        print(self.__class__.__doc__)



class ParseEventDataXML:
    """
    ParseEventDataXML is a class for parsing and creating XML files based on earthquake event data.
    
    This class operates in two distinct modes:
    
    1. **Parse Mode ('parse'):**
       In this mode the class reads an existing earthquake event XML file from a given path (specified by
       `event_xml`), parses its contents, and sets the corresponding object attributes. The extracted data can be 
       converted to a pandas DataFrame or a dictionary for further processing.
       
    2. **Create Mode ('create'):**
       In this mode the class accepts a dictionary of earthquake event attributes (via `event_attributes`) to 
       initialize the object properties. While the attributes are set immediately, the XML file is only created 
       when the `write_event_xml()` method is explicitly called. This mode allows users to generate new XML files 
       with the specified earthquake event data.
       
    **Required Attributes and Modes:**
    
      - **Attributes:**
          - `mode` (str): Operation mode, either `'parse'` or `'create'`.
          - `event_xml` (str): For `'parse'` mode, the path to the XML file to be read. For `'create'` mode, the 
            output path where the XML file will be generated.
          - `event_attributes` (dict): A dictionary of event attributes that is used when operating in
            `'create'` mode. Expected keys include:
              - `'id'`         : Unique event identifier (e.g., "us7000m9g4").
              - `'netid'`      : Network identifier (e.g., "us").
              - `'network'`    : Full network name (e.g., "USGS National Earthquake Information Center, PDE").
              - `'lat'`        : Latitude of the event (string convertible to float).
              - `'lon'`        : Longitude of the event (string convertible to float).
              - `'depth'`      : Depth of the event in km (string convertible to float).
              - `'mag'`        : Magnitude of the earthquake (string convertible to float).
              - `'time'`       : Time of the event in ISO8601 format (e.g., "2024-04-03T01:58:11Z").
              - `'locstring'`  : Descriptive location (e.g., "18 km SSW of Hualien City, Taiwan").
              - `'event_type'` : Type of event (e.g., "ACTUAL").
              - `'reviewed'`   : Review status (e.g., "unknown").
          
      - **Modes:**
          - In **Parse Mode**, the XML file specified by `event_xml` must exist. The class will then parse this 
            file and populate its attributes by reading the XML.
          - In **Create Mode**, a complete `event_attributes` dictionary must be provided. If any required 
            attributes are missing or improperly formatted, a `ValueError` is raised. The XML file is not written 
            automatically; you must call `write_event_xml()` explicitly.
    
    **Available Methods:**
    
      - **Initialization and Mode Handling:**
          - `__init__(self, mode='parse', event_xml=None, event_attributes=None)`: 
            Initializes the instance. If `mode` is `'parse'`, it will attempt to parse the XML file at `event_xml`
            (or use a default path if not provided). If `mode` is `'create'`, it validates and sets the provided 
            `event_attributes`.
    
      - **Parsing and Attribute Setting:**
          - `parse_xml(self)`: Parses the XML file at `event_xml`, verifies that the root element is "earthquake", 
            and calls `set_attributes()` to set the object’s attributes.
          - `set_attributes(self, root)`: Extracts attribute values from the XML root element, performing safe float 
            conversions for numeric fields (lat, lon, depth, mag), and assigns them to object attributes.
          - `safe_float_conversion(self, value)`: Attempts to convert `value` to a float; returns `None` if the 
            conversion fails.
    
      - **XML Creation and Attribute Initialization:**
          - `write_event_xml(self)`: In `'create'` mode, writes a new XML file using the values stored in 
            `event_attributes`. The output path is determined by calling `initialize_path()`, and the XML is written 
            to that location.
          - `set_attributes_directly(self)`: Directly assigns object attributes from the `event_attributes` 
            dictionary (used in `'create'` mode).
          - `initialize_attributes(self)`: Initializes the main event attributes to `None`.
          - `initialize_path(self)`: Constructs a file path for the new XML file using a base directory and the 
            event's ID.
    
      - **Data Conversion and Summary:**
          - `get_dataframe(self)`: Converts the earthquake event data into a pandas DataFrame, with each attribute as 
            a column. Raises an error if any attribute is not set.
          - `get_dict(self)`: Returns the earthquake event data as a dictionary.
          - `get_summary(self)`: Prints a summary of all the earthquake event attributes, formatted for easy 
            reading.
    
      - **Documentation Display:**
          - `print_doc(self)`: Prints the class’s detailed docstring to help users understand its functionality, 
            available methods, and usage examples.
    
    **Example Usage:**
    
    *Parsing an Existing XML File:*
    
        >>> parser = ParseEventDataXML(mode='parse', event_xml='path/to/event.xml')
        >>> parser.get_summary()
        Earthquake Event Summary:
        ID: us7000m9g4
        Network ID: us
        Network: USGS National Earthquake Information Center, PDE
        Latitude: 23.819
        Longitude: 121.5616
        Depth (km): 34.8
        Magnitude: 7.4
        Time: 2024-04-03T01:58:11Z
        Location String: 18 km SSW of Hualien City, Taiwan
        Event Type: ACTUAL
        Reviewed: unknown
        >>> df = parser.get_dataframe()
        >>> print(df.head())
    
    *Creating a New XML File:*
    
        >>> attributes = {
        ...     'id': 'us7000m9g4',
        ...     'netid': 'us',
        ...     'network': 'USGS National Earthquake Information Center, PDE',
        ...     'lat': '23.8190',
        ...     'lon': '121.5616',
        ...     'depth': '34.8',
        ...     'mag': '7.4',
        ...     'time': '2024-04-03T01:58:11Z',
        ...     'locstring': '18 km SSW of Hualien City, Taiwan',
        ...     'event_type': 'ACTUAL',
        ...     'reviewed': 'unknown'
        ... }
        >>> creator = ParseEventDataXML(mode='create', event_attributes=attributes)
        >>> creator.write_event_xml()  # This creates the XML file at the designated path.
        XML file created successfully at: ./export/usgs-scenarios/us7000m9g4/current/event.xml
        >>> creator.get_summary()
    
    *Viewing Class Documentation:*
    
        >>> parser.print_doc()
    
    **Notes:**
      - The XML files are expected to have a root element named "earthquake".
      - In 'create' mode, the XML file is not automatically written – you must call `write_event_xml()` explicitly.
      - Files are read/written using the 'latin-1' encoding by default; change this if necessary.

      © SHAKEmaps version 25.3.1
    """

            
    def __init__(self, mode='parse', event_xml=None, event_attributes=None):
        self.mode = mode
        self.event_xml = event_xml
        self.event_attributes = event_attributes or {}
        self.initialize_attributes()

        
        if self.mode == 'parse':
            if not self.event_xml:
                self.event_xml = "./example_data/us7000m9g4/current/event.xml"
                print("No file provided, proceeding with default file path:", self.event_xml)
            if not os.path.exists(self.event_xml):
                raise FileNotFoundError(f"Event files not found at {self.event_xml}. Please check the path.")
            self.parse_xml()
            print('Event XML parsed successfully.')

        elif self.mode == 'create':
            required_attributes = [
                'id', 'netid', 'network', 'lat', 'lon', 'depth', 'mag', 'time', 
                'locstring', 'event_type', 'reviewed'
            ]

            if not self.event_attributes:
                example_format = {
                    'id': 'us7000m9g4',
                    'netid': 'us',
                    'network': 'USGS National Earthquake Information Center, PDE',
                    'lat': '23.8190',
                    'lon': '121.5616',
                    'depth': '34.8',
                    'mag': '7.4',
                    'time': '2024-04-03T01:58:11Z',
                    'locstring': '18 km SSW of Hualien City, Taiwan',
                    'event_type': 'ACTUAL',
                    'reviewed': 'unknown'
                }
                raise ValueError(
                    "Event attributes must be provided to create an XML file. "
                    "Example format: " + str(example_format)
                )

            missing_attributes = [
                attr for attr in required_attributes if attr not in self.event_attributes
            ]

            if missing_attributes:
                example_format = {
                    'id': 'us7000m9g4',
                    'netid': 'us',
                    'network': 'USGS National Earthquake Information Center, PDE',
                    'lat': '23.8190',
                    'lon': '121.5616',
                    'depth': '34.8',
                    'mag': '7.4',
                    'time': '2024-04-03T01:58:11Z',
                    'locstring': '18 km SSW of Hualien City, Taiwan',
                    'event_type': 'ACTUAL',
                    'reviewed': 'unknown'
                }
                raise ValueError(
                    f"Missing or incorrectly formatted attributes: {missing_attributes}. "
                    "Correct example format: " + str(example_format)
                )

            self.event_id = self.event_attributes.get('id')
            self.set_attributes_directly()
            print('Attributes initialized successfully, XML file not yet created.')
        else:
            raise ValueError("Mode must be either 'parse' or 'create'.")


    def parse_xml(self):
        tree = ET.parse(self.event_xml)
        root = tree.getroot()
        if root.tag != "earthquake":
            raise ValueError("XML does not contain an earthquake event")
        self.set_attributes(root)
        
    def write_event_xml(self):
        if self.mode=='create':
            self.initialize_path()
            if not self.event_attributes:
                raise ValueError("No attributes set for XML creation.")
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.event_xml), exist_ok=True)
            earthquake = ET.Element("earthquake")
            for key, value in self.event_attributes.items():
                earthquake.set(key, str(value))
            tree = ET.ElementTree(earthquake)
            tree.write(self.event_xml)
            print(f'XML file created successfully at: {self.event_xml}')
        else: print("Can not write a file in 'parse' mode" )
        
    def set_attributes_directly(self):
        """Directly sets attributes from the provided attributes dictionary."""
        for attr, value in self.event_attributes.items():
            setattr(self, attr, value)


        
    def initialize_attributes(self):
        self.id = None
        self.netid = None
        self.network = None
        self.lat = None
        self.lon = None
        self.depth = None
        self.mag = None
        self.time = None
        self.locstring = None
        self.event_type = None
        self.reviewed = None
        
    def set_attributes(self, root):
        """Sets attributes from the XML root element, converting specific fields to float with error handling."""
        self.id = root.get('id')
        self.netid = root.get('netid')
        self.network = root.get('network')
        self.lat = self.safe_float_conversion(root.get('lat'))
        self.lon = self.safe_float_conversion(root.get('lon'))
        self.depth = self.safe_float_conversion(root.get('depth'))
        self.mag = self.safe_float_conversion(root.get('mag'))
        self.time = root.get('time')
        self.locstring = root.get('locstring')
        self.event_type = root.get('event_type')
        self.reviewed = root.get('reviewed')
        
    def safe_float_conversion(self, value):
        """Converts a string to a float or returns None if conversion fails."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
        
    def get_dataframe(self):
        """Converts the parsed earthquake event data into a pandas DataFrame."""
        event_data = self.get_dict()
        if None in event_data.values():
            raise ValueError("Not all attributes are set. Ensure the XML is parsed correctly.")
        return pd.Series(event_data).to_frame().T

    def get_dict(self):
        """Returns a dictionary of the earthquake event data from parsed XML."""
        return {
            'id': self.id,
            'netid': self.netid,
            'network': self.network,
            'lat': self.lat,
            'lon': self.lon,
            'depth': self.depth,
            'mag': self.mag,
            'time': self.time,
            'locstring': self.locstring,
            'event_type': self.event_type,
            'reviewed': self.reviewed
        }

    def initialize_path(self):
        """Initializes the file path for creating XML based on event attributes."""
        base_path = "./export/usgs-scenarios"
        self.event_xml = os.path.join(base_path, self.event_id, "current", "event.xml")
    def get_summary(self):
        """
        Prints a summary of all the earthquake event attributes.
        This method collects all relevant attributes and displays them,
        whether they were set by parsing an XML file or provided during creation.
        """
        attributes = {
            'ID': self.id,
            'Network ID': self.netid,
            'Network': self.network,
            'Latitude': self.lat,
            'Longitude': self.lon,
            'Depth (km)': self.depth,
            'Magnitude': self.mag,
            'Time': self.time,
            'Location String': self.locstring,
            'Event Type': self.event_type,
            'Reviewed': self.reviewed
        }
        print("Earthquake Event Summary:")
        for key, value in attributes.items():
            print(f"{key}: {value if value is not None else 'Not Available'}")
            
    def print_doc(self):
        """Prints the class docstring."""
        print(self.__doc__)




class ParseRuptureDataJson:
    """
    A class designed to manage and manipulate JSON files containing earthquake rupture dimensions.
    This class provides functionalities to parse existing JSON files to extract and visualize earthquake rupture data. 
    If no specific JSON file path is provided, it defaults to an example dataset.

    Usage Examples:
    ----------------
    Parsing an existing JSON file:
        parser = ParseRuptureDataJson(mode='parse', rupture_json='path/to/rupture.json')
        metadata = parser.get_metadata()
        print(metadata)  # Display metadata information
        parser.show_rupture()  # Visualize the rupture on a map

    Initializing without a specific path (uses default):
        parser = ParseRuptureDataJson()
        parser.print_doc()  # Print the class documentation

    Parameters:
        mode (str): Specifies the operation mode. Currently, only 'parse' is supported.
        rupture_json (str, optional): Path to the JSON file for parsing. If not provided, defaults to a sample file.

    Attributes:
        mode (str): Operational mode of the class, determines the type of operations that can be performed.
        rupture_json (str): File path to the JSON file containing rupture data.
        data (dict): Parsed JSON data stored as a dictionary.

    Methods:
    -------
        get_metadata(self):
            Extracts and returns metadata from the JSON file.
            Returns:
                dict: A dictionary containing the earthquake event metadata such as event ID, network, product code, event time, location, magnitude, and depth.

        get_rupture_xy(self):
            Extracts X (longitude) and Y (latitude) coordinates from the 'features' section of the JSON data, assuming the geometry type is 'MultiPolygon'.
            Returns:
                tuple: Two lists containing the X and Y coordinates of the rupture.

        get_rupture_coordinates(self):
            Retrieves all coordinates of the rupture dimensions for MultiPolygon geometries.
            Returns:
                list: A list of coordinates for all polygons that make up the rupture.

        show_rupture(self):
            Visualizes the rupture data on a map using matplotlib. Displays the rupture dimensions plot with longitude on the X-axis and latitude on the Y-axis.
            Utilizes:
                get_rupture_xy(self): To fetch the coordinates needed for plotting.
        
        get_dataframe(self):
            Converts the parsed JSON data into a pandas DataFrame for easier manipulation and analysis.

        get_dict(self):
            Converts the parsed JSON data into a dictionary formatted in a manner suitable for further processing or analysis.

        print_doc(self):
            Prints the class's docstring, providing a reference to the class usage and documentation directly from the environment.
    
    Internal Methods:
    ----------------- 
        load_json(self):
            Attempts to load and parse the JSON file specified by `rupture_json`. 
            Raises:
                FileNotFoundError: If the file cannot be found at the specified path.
                json.JSONDecodeError: If the file is not valid JSON.

    Raises:
        ValueError: If the 'mode' is set to anything other than 'parse', since other modes are not supported in the current implementation.

    
    © SHAKEmaps version 25.3.2
    """
    def __init__(self, mode='parse',rupture_json=None):
        
        self.mode = mode
        if mode != 'parse':
            raise ValueError("Currently, only 'parse' mode is supported.")

        self.rupture_json = rupture_json

        if not self.rupture_json:
            default_path = "./example_data/us7000m9g4/current/rupture.json"
            print("No JSON file given, proceeding with default example.")
            if os.path.exists(default_path):
                self.rupture_json = default_path
            else:
                raise FileNotFoundError("Default JSON file not found. Please provide a valid JSON file.")

        self.data = self.load_json()  # Load data from the final path



    def load_json(self):
        """Load and parse JSON data from a file."""
        try:
            with open(self.rupture_json, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            raise ValueError("Failed to decode JSON. The file might be corrupted or improperly formatted.")
        except FileNotFoundError:
            raise FileNotFoundError(f"No JSON file found at {self.rupture_json}. Please provide the correct path.")

    def get_metadata(self):
        """Extract general metadata information."""
        metadata = self.data.get('metadata', {})
        return {
            'event_id': metadata.get('id'),
            'network': metadata.get('network'),
            'product_code': metadata.get('productcode'),
            'event_time': metadata.get('time'),
            'location': metadata.get('locstring'),
            'magnitude': metadata.get('mag'),
            'depth': metadata.get('depth')
        }

    def get_rupture_xy(self):
        """Extract and return separate lists of longitude (x) and latitude (y) coordinates."""
        x_coords, y_coords = [], []
        for feature in self.data.get('features', []):
            geometry = feature.get('geometry', {})
            if geometry.get('type') == 'MultiPolygon':
                for multipolygon in geometry.get('coordinates', []):
                    for polygon in multipolygon:
                        for (lon, lat, _) in polygon:
                            x_coords.append(lon)
                            y_coords.append(lat)
        return x_coords, y_coords
    
    def get_rupture_coordinates(self):
        """Extract all coordinates of the rupture dimensions."""
        features = self.data.get('features', [])
        all_coordinates = []
        for feature in features:
            geometry = feature.get('geometry', {})
            if geometry.get('type') == 'MultiPolygon':
                all_coordinates.extend(geometry.get('coordinates', []))
        return all_coordinates


    def show_rupture(self):
        """Visualize the rupture data on a map."""
        x_coords, y_coords = self.get_rupture_xy()
        
        fig, ax = plt.subplots()
        ax.plot(x_coords, y_coords, '-', label='Rupture Extent')
        ax.set_title('Rupture Dimensions Plot')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        plt.show()
        
    def get_dataframe(self):
        """Convert the entire JSON object into a pandas DataFrame."""
        features = self.data.get('features', [])
        return pd.json_normalize(features)

    def get_dict(self):
        """Convert the entire JSON object into a dictionary."""
        return self.data
        
    def print_doc(self):
        """Prints the class docstring."""
        print(self.__doc__)




class ParsePagerDataXML:
    """
    ParsePagerDataXML: A Comprehensive XML Parser for PAGER Data from Global Earthquakes
    ====================================================================================

    Overview
    --------
    The ParsePagerDataXML class is designed to parse and extract critical earthquake impact data from 
    PAGER (Prompt Assessment of Global Earthquakes for Response) XML files. These files contain detailed 
    information on seismic events including event details, alert levels, city-specific impacts, exposure 
    metrics, and structural comments. With this parser, users can easily transform the XML data into 
    structured formats like pandas DataFrames or dictionaries for subsequent analysis, visualization, 
    or integration into emergency response workflows.

    Installation & Requirements
    -----------------------------
    - Python 3.6 or above.
    - Required modules: 
        * xml.etree.ElementTree (for XML parsing),
        * numpy (for numerical array handling),
        * pandas (for data structuring),
        * matplotlib (for plotting probability distributions),
        * os (for file operations).
    
    Usage
    -----
    To use this parser, instantiate the class by passing the path to a valid PAGER XML file. If no file 
    is provided, a default example is used. Once initialized, you can extract various components of the 
    earthquake data, including:
    
        - Event details,
        - Alerts with probability distributions,
        - City-specific impact data,
        - Structural and additional comments,
        - Exposure data.
    
    Example:
        parser = ParsePagerDataXML(xml_file='path/to/pager.xml')
        
        # Convert parsed data into structured DataFrames.
        df_data = parser.get_dataframe()
        print(df_data)
        
        # Get all parsed data as a dictionary.
        data_dict = parser.get_dict()
        
        # Plot the probability distribution for economic alerts.
        fig, ax = parser.plot_probability_distribution(alert_type='economic')

    Attributes
    ----------
    xml_file : str
        Path to the XML file containing the PAGER data. If omitted, a default example file is used.
    tree : xml.etree.ElementTree.ElementTree
        The XML tree structure parsed from the provided XML file.
    
    User Methods
    ------------
    get_dataframe(self)
        Converts the extracted XML segments into separate pandas DataFrames for event details, alerts,
        cities impact, and structural comments.
    get_dict(self)
        Returns all parsed XML data in a dictionary format, with keys corresponding to each data segment.
    plot_probability_distribution(self, alert_type)
        Plots a histogram-like probability distribution for the specified alert type (e.g., "economic" or 
        "fatalities"). This method formats numeric bin ranges (using abbreviations like 'K' for thousands) 
        for better readability.
    get_pager_details(self)
        Extracts and returns a dictionary of detailed event information from the XML.
    get_alerts(self)
        Parses and returns alert-level information and associated bin data from the XML as a structured list.
    get_cities_impact(self)
        Extracts city-specific impact data from the XML, summarizing how different cities are affected.
    get_structural_comments(self)
        Collects and returns supplementary comments on structural impacts, alert messages, and secondary 
        effects as provided in the XML.
    get_exposure_data(self)
        Retrieves exposure data, such as population exposure by MMI range, from the XML.
    
    Internal Methods
    ----------------
    load_xml(self)
        Loads and parses the XML file specified by 'xml_file', initializing and returning the XML tree.
    format_bin_label(self, min_value, max_value)
        Formats numerical bin range values into abbreviated, human-friendly strings (e.g., converting thousands 
        to "K").

    Error Handling & Logging
    --------------------------
    - The class validates the provided XML file path, using a default example if none is given.
    - It checks for malformed XML and missing elements, raising descriptive exceptions for troubleshooting.
    - Logging (if configured externally) tracks each stage of XML loading and data extraction, aiding in debugging.

    Note
    ----
    Currently, ParsePagerDataXML supports only 'parse' mode. Future updates may extend functionality to support
    additional operational modes.

    © SHAKEmaps version 25.3.2

    """
    
    def __init__(self,mode='parse', xml_file=None):
        
        self.mode = mode
        if mode != 'parse':
            raise ValueError("Currently, only 'parse' mode is supported.")

        
        if not xml_file:
            default_path = './example_data/usgs-pager-versions-us7000m9g4/us7000m9g4/us7000m9g4_us_07_pager.xml'
            print("No XML file given, proceeding with default example.")
            if os.path.exists(default_path):
                self.xml_file = default_path
            else:
                raise FileNotFoundError("Default Pager file not found. Please provide a valid pager file.")
        else:
            self.xml_file = xml_file
        self.tree = self.load_xml()
        
        
    def load_xml(self):
        """
        Load and parse the XML file.
        """
        try:
            tree = ET.parse(self.xml_file)
            return tree
        except ET.ParseError:
            raise ValueError("Failed to parse XML file.")
        except FileNotFoundError:
            raise FileNotFoundError("The XML file was not found.")

    def get_pager_details(self):
        """
        Extracts the main event details from the XML data.
        """
        root = self.tree.getroot()
        event_info = root.find('event')
        event_details = {attr: event_info.get(attr) for attr in event_info.keys()}
        return event_details

    def get_alerts(self):
        """
        Extracts alert information from the XML data.
        """
        root = self.tree.getroot()
        alerts = []
        for alert in root.findall('alert'):
            alert_details = {
                'type': alert.get('type'),
                'level': alert.get('level'),
                'summary': alert.get('summary')
            }
            bins = []
            for bin in alert.findall('bin'):
                bins.append({attr: bin.get(attr) for attr in bin.keys()})
            alert_details['bins'] = bins
            alerts.append(alert_details)
        return alerts

    def get_cities_impact(self):
        """
        Extracts city-specific impact data from the XML data.
        """
        root = self.tree.getroot()
        cities = []
        for city in root.findall('city'):
            city_details = {attr: city.get(attr) for attr in city.keys()}
            cities.append(city_details)
        return cities

    def get_structural_comments(self):
        """
        Extracts comments regarding structural impacts from the XML data.
        """
        root = self.tree.getroot()
        struct_comment = root.find('structcomment').text
        alert_comment = root.find('alertcomment').text
        impact_comment = root.find('impact_comment').text
        secondary_effects = root.find('secondary_effects').text
        return {
            "Structural Comment": struct_comment,
            "Alert Comment": alert_comment,
            "Impact Comment": impact_comment,
            "Secondary Effects": secondary_effects
        }
    
    
    def get_exposure_data(self):
        """
        Extracts exposure data (population by MMI range) from the XML data.
        Returns a list of dictionaries like: [{'dmin': '3.5', 'dmax': '4.5', 'exposure': '69713253'}, ...]
        """
        root = self.tree.getroot()
        exposures = []

        for exp_elem in root.findall('exposure'):
            exp_details = {
                'dmin': exp_elem.get('dmin'),
                'dmax': exp_elem.get('dmax'),
                'exposure': exp_elem.get('exposure'),
                'rangeInsideMap': exp_elem.get('rangeInsideMap')
            }
            exposures.append(exp_details)

        return exposures




    def get_dataframe(self):
        """
        Converts extracted XML data to pandas DataFrames.
        """
        event_df = pd.DataFrame([self.get_pager_details()])
        alerts = self.get_alerts()
        alerts_df = pd.DataFrame([alert for alert in alerts for bin in alert['bins']])
        cities_df = pd.DataFrame(self.get_cities_impact())
        comments = self.get_structural_comments()
        comments_df = pd.DataFrame([comments])

        return {
            "Event Details": event_df,
            "Alerts": alerts_df,
            "Cities Impact": cities_df,
            "Comments": comments_df
        }
    
    def get_dict(self):
        """
        Returns a dictionary containing all the extracted data.
        """
        return {
            "Event Details": self.get_pager_details(),
            "Alerts": self.get_alerts(),
            "Cities Impact": self.get_cities_impact(),
            "Comments": self.get_structural_comments()
        }

    
    def plot_probability_distribution(self, alert_type):
        """
        Plots a histogram-like probability distribution where the height of each bin corresponds to the probability.
        The x-axis label changes based on whether the plot is for 'economic' losses or 'fatalities'.
        Also formats the bin labels to use 'K' for thousands.
        """
        alerts = self.get_alerts()
        # Filter the alerts to only the specified type
        alert_data = next((alert for alert in alerts if alert['type'] == alert_type), None)
        if alert_data is None:
            raise ValueError(f"No alert data found for type '{alert_type}'")

        # Prepare data for plotting
        bin_labels = [self.format_bin_label(bin['min'], bin['max']) for bin in alert_data['bins']]
        probabilities = [float(bin['probability']) for bin in alert_data['bins']]
        colors = [bin['color'] for bin in alert_data['bins']]

        # Define the positions of the bars
        x_positions = range(len(probabilities))

        # Create the bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(x_positions, probabilities, color=colors, edgecolor='black')

        # Add labels on top of each bin
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width() / 2, prob, f"{prob:.2f}",
                     ha='center', va='bottom', fontsize=12)  # Adjust font size here

        # Set the x-axis and y-axis labels
        plt.xlabel("USD (Millions)" if alert_type == 'economic' else "Fatalities", fontsize=13)
        plt.ylabel('Probability', fontsize=13)

        # Set the title of the plot
        plt.title(f'Estimated {alert_type.capitalize()} Probability Distribution', fontsize=14)

        # Set the tick labels font size
        plt.xticks(x_positions, bin_labels, fontsize=12)
        plt.yticks(fontsize=12)

        # Layout and grid settings
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()

        # Return the figure and axes objects instead of showing the plot
        return plt.gcf(), plt.gca()

    @staticmethod
    def format_bin_label(min_value, max_value):
        """
        Formats the bin labels to abbreviate thousands with 'K'.
        """
        def abbreviate(value):
            value = int(value)
            if value >= 1000:
                return f"{value//1000}K"
            return str(value)

        return f"{abbreviate(min_value)} - {abbreviate(max_value)}"
    
    
    def print_doc(self):
        """Prints the class docstring."""
        print(self.__doc__)


class ParseModelConfig:  
    """
    Initializes and manages the operations for parsing and creating configuration files related to seismic event simulations. 
    This class can operate in two modes: 'parse' for reading existing configuration files and 'create' for generating new ones 
    based on provided configuration data.

    Parameters:
    -----------
        mode (str): Specifies the operation mode of the instance; 'parse' reads from an existing file, 'create' generates a new file.
        config_file (str, optional): Path to the configuration file for parsing, or the base directory for creating new files.
        config_data (dict, optional): Data used for creating new configuration files when in 'create' mode, structured according to 
                                      required configuration schema.

    Attributes:
    ----------
        gmpe_sets (dict): Contains predefined GMPE sets and their properties like GMPEs used, their weights, and distance cutoffs.
        gmpe_modules (dict): Maps GMPE short names to their detailed implementation details necessary for simulations.
        config_data (dict): Stores configuration data after parsing or before creation.
        event_id (str): The identifier for the event, used for creating file paths in 'create' mode.

    Methods:
    --------
        print_used_gmpes(self):
            Outputs a list of GMPEs used along with their respective weights, providing a quick overview of the modeling parameters 
            active in the current configuration.

        get_used_gmpes(self):
            Returns a dictionary detailing the GMPEs used and their properties, useful for analysis and debugging of simulation configurations.

        print_doc(self):
            Prints the class's docstring, providing in-context documentation accessible programmatically, useful for understanding 
            class usage without source code access.

        write_config_file(self):
            Constructs a new configuration file using data from config_data. Excludes 'event_id' from being written directly into 
            the file to maintain structural consistency with existing configuration management practices.

    Internal Methods: 
    ----------------
        initialize_parse_path(self):
            Prepares the file path for parsing. Uses a default path if none is provided. Raises an error if the file does not exist.

        parse_config_file(self):
            Reads and parses the configuration file specified by config_file. Extracts and stores configuration sections and 
            their contents into the config_data attribute.


        initialize_create_path(self):
            Prepares the directory and file path settings for creating a new configuration file. It uses event_id to structure 
            the directory path.

        _convert_value(self, value):
            Converts string values from the configuration file into appropriate Python data types (e.g., integers, floats, None).

        _format_value(self, value):
            Formats Python data types back into string representations suitable for writing to the configuration file.


    Usage Examples:
    ---------------
        Parsing an existing configuration file:
            parser = ParseModelConfig(mode='parse', config_file='path/to/config.conf')
            parser.print_used_gmpes()

        Creating a new configuration file:
           config_data = {'event_id': 'us7000m9g4',
                                'gmpe_sets': {
                                    'gmpe_us7000m9g4_custom': {
                                        'gmpes': ['active_crustal_taiwan_deep', 'subduction_interface_nshmp2014'],
                                        'weights': [0.6691643912838412, 0.33083560871615886],
                                        'weights_large_dist': None,
                                        'dist_cutoff': float('nan'),
                                        'site_gmpes': None,
                                        'weights_site_gmpes': None
                                    }
                                },
                                'modeling': {
                                    'gmpe': 'gmpe_us7000m9g4_custom',
                                    'mechanism': 'RS',
                                    'ipe': 'VirtualIPE',
                                    'gmice': 'WGRW12',
                                    'ccf': 'LB13'
                                }
                            }
            creator = ParseModelConfig(mode='create', config_data=config_data)
            creator.create_config_file()  # Saves the new configuration file at the specified location

    Note:
    -----
        - In 'create' mode, ensure that 'event_id' is included in the config_data as it is essential for file path construction.
        - The class does not directly handle the persistence of the 'event_id' in the configuration file, ensuring flexibility in file formatting.
    
    # USGS ShakeMap Documentations 
    ##########################################################################
        # gmpe_sets: Specify the GMPE sets available to ShakeMap's model module.
        # Each sub-section below allows the specification of a set of one or more
        # GMPEs to combine together to be treated as a single GMPE by the model
        # module. The output of the combined module will be the weighted average 
        # combination of the outputs of the individual modules. The parameters
        # for each GMPE set are:
        #
        # - gmpes: A list of one or more GMPE modules. The modules must be the
        #          GMPE's short name as defined in modules.conf. If a specified
        #          GMPE does not produce the required IMT, then that GMPE is
        #          dropped and the weights (see below) of the remaining GMPEs
        #          is rebalanced.
        # - weights: A list of weights to apply to each of the 'gmpes' specified
        #          (in the same order they are specified in the 'gmpes' list).
        #          The weights must sum to 1.0.
        # - weights_large_distance: A list of weights to apply to the 'gmees' for
        #          points at a large distance from the source.
        #          See 'dist_cutoff', below. The weights must sum to 1.0. If
        #          the list is set to 'None', then 'weights' will be used for
        #          all distances.
        # - dist_cutoff: A distance (in kilometers) at which the weights of the
        #          GMPEs switches from the 'weights" list to the 
        #          'weights_large_distance' list. If set to 'nan', then 
        #          'weights_large_distance' will not be used.
        # - weights_site_gmpes: Provides the weighting of the GMPEs' site 
        #          amplification terms applied to the output. This parameter 
        #          allows for the inclusion of GMPEs in the gmpes list that
        #          do not provide site amplifications or provide inadequate 
        #          site amplification terms. The elements of the list must 
        #          sum to 1.0. If the list is 'None' then the normal 'weights'
        #          are used.
        # ##########################################################################

    © SHAKEmaps version 25.3.2
    """


    def __init__(self, mode='parse', config_file=None, config_data=None):

        self.gmpe_sets = {
    "active_crustal_nshmp2014": {
        "gmpes": ["ASK14", "BSSA14", "CB14", "CY14"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "active_crustal_deep": {
        "gmpes": ["ASK14", "CB14", "CY14"],
        "weights": [0.3333, 0.3333, 0.3334],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "active_crustal_california": {
        "gmpes": ["ASK14", "BSSA14ca", "CB14", "CY14"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "active_crustal_taiwan": {
        "gmpes": ["ASK14tw", "BSSA14", "CB14", "CY14"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "active_crustal_taiwan_deep": {
        "gmpes": ["ASK14tw", "CB14", "CY14"],
        "weights": [0.3333, 0.3333, 0.3334],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "active_crustal_japan": {
        "gmpes": ["Zea16c", "ASK14jp", "BSSA14jp", "CB14jp", "CY14"],
        "weights": [0.5, 0.125, 0.125, 0.125, 0.125],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "active_crustal_japan_deep": {
        "gmpes": ["Zea16c", "ASK14jp", "CB14jp", "CY14"],
        "weights": [0.5, 0.1666, 0.1667, 0.1667],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "active_crustal_china": {
        "gmpes": ["ASK14chn", "BSSA14hq", "CB14hq", "CY14"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "active_crustal_china_deep": {
        "gmpes": ["ASK14chn", "CB14hq", "CY14"],
        "weights": [0.3333, 0.3333, 0.3334],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "active_crustal_new_zealand": {
        "gmpes": ["Bradley2013", "ASK14", "BSSA14", "CB14", "CY14"],
        "weights": [0.5, 0.125, 0.125, 0.125, 0.125],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "active_crustal_share": {
        "gmpes": ["Akea14", "Cau14", "CB14lq", "BSSA14lq", "ASK14", "CY14", "Zea16c"],
        "weights": [0.35, 0.35, 0.05, 0.05, 0.05, 0.05, 0.1],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "stable_continental_nshmp2014_rlme": {
        "gmpes": ["Fea96", "Tea97", "Sea02", "C03", "TP05", "AB06p", "Pea11", "Atk08p", "Sea01"],
        "weights": [0.06, 0.11, 0.06, 0.11, 0.11, 0.22, 0.15, 0.08, 0.1],
        "weights_large_dist": [0.16, 0.0, 0.0, 0.17, 0.17, 0.3, 0.2, 0.0, 0.0],
        "dist_cutoff": 500,
        "site_gmpes": ["AB06p"],
        "weights_site_gmpes": [1.0],
    },
    "stable_continental_deep": {
        "gmpes": ["Fea96", "Tea97", "Sea02", "C03", "TP05", "AB06p", "Pea11", "Atk08p", "Sea01"],
        "weights": [0.06, 0.11, 0.06, 0.11, 0.11, 0.22, 0.15, 0.08, 0.1],
        "weights_large_dist": [0.16, 0.0, 0.0, 0.17, 0.17, 0.3, 0.2, 0.0, 0.0],
        "dist_cutoff": 500,
        "site_gmpes": ["AB06p"],
        "weights_site_gmpes": [1.0],
    },
    "stable_continental_induced": {
        "gmpes": ["Atk15"],
        "weights": [1.0],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": ["AB06p"],
        "weights_site_gmpes": [1.0],
    },
    "stable_continental_share": {
        "gmpes": ["Akea14", "Cau14", "CB14lq", "BSSA14lq", "ASK14", "CY14", "Toro02_share", "C03_share"],
        "weights": [0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": ["Akea14", "Cau14", "CY14"],
        "weights_site_gmpes": [0.33, 0.33, 0.34],
    },
    "subduction_interface_nshmp2014": {
        "gmpes": ["AB03i", "Zea16i", "AM09", "Aea15i"],
        "weights": [0.1, 0.3, 0.3, 0.3],
        "site_gmpes": ["Aea15i", "AB03i"],
        "weights_site_gmpes": [0.5, 0.5],
    },
    "subduction_interface_share": {
        "gmpes": ["AB03i", "LinLee08i", "Aea15i", "Zea16i"],
        "weights": [0.2, 0.2, 0.2, 0.4],
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "subduction_interface_chile": {
        "gmpes": ["Mont17i"],
        "weights": [1.0],
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "subduction_slab_nshmp2014": {
        "gmpes": ["AB03s", "AB03sc", "Zea16s", "Aea15s"],
        "weights": [0.1667, 0.1667, 0.3333, 0.3333],
        "site_gmpes": ["Aea15s", "AB03s"],
        "weights_site_gmpes": [0.5, 0.5],
    },
    "subduction_slab_share": {
        "gmpes": ["AB03s", "LinLee08s", "Aea15s", "Zea16s"],
        "weights": [0.2, 0.2, 0.2, 0.4],
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "subduction_slab_chile": {
        "gmpes": ["Mont17s"],
        "weights": [1.0],
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "subduction_crustal": {
        "gmpes": ["ASK14", "BSSA14", "CB14", "CY14"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "subduction_vrancea_share": {
        "gmpes": ["LinLee08s", "Youngs97s"],
        "weights": [0.6, 0.4],
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "volcanic": {
        "gmpes": ["Atk10"],
        "weights": [1.0],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
    },
    "volcanic_new_zealand": {
        "gmpes": ["Bradley2013vol"],
        "weights": [1.0],
        "weights_large_dist": None,
        "dist_cutoff": float("nan"),
        "site_gmpes": None,
        "weights_site_gmpes": None,
            },
            # Add other GMPE sets here as needed
        }
        self.gmpe_modules = {
            "Atk08p": ["Atkinson2008prime", "openquake.hazardlib.gsim.boore_atkinson_2011"],
            "Atk10": ["Atkinson2010Hawaii", "openquake.hazardlib.gsim.boore_atkinson_2008"],
            "Atk15": ["Atkinson2015", "openquake.hazardlib.gsim.atkinson_2015"],
            "AB03i": ["AtkinsonBoore2003SInter", "openquake.hazardlib.gsim.atkinson_boore_2003"],
            "AB03s": ["AtkinsonBoore2003SSlab", "openquake.hazardlib.gsim.atkinson_boore_2003"],
            "AB03sc": ["AtkinsonBoore2003SSlabCascadia", "openquake.hazardlib.gsim.atkinson_boore_2003"],
            "AB06p": ["AtkinsonBoore2006Modified2011", "openquake.hazardlib.gsim.atkinson_boore_2006"],
            "Aea15i": ["AbrahamsonEtAl2015SInter", "openquake.hazardlib.gsim.abrahamson_2015"],
            "Aea15s": ["AbrahamsonEtAl2015SSlab", "openquake.hazardlib.gsim.abrahamson_2015"],
            "AkBo10": ["AkkarBommer2010", "openquake.hazardlib.gsim.akkar_bommer_2010"],
            "Akea14": ["AkkarEtAlRjb2014", "openquake.hazardlib.gsim.akkar_2014"],
            "AM09": ["AtkinsonMacias2009", "openquake.hazardlib.gsim.atkinson_macias_2009"],
            "ASK14": ["AbrahamsonEtAl2014", "openquake.hazardlib.gsim.abrahamson_2014"],
            "ASK14tw": ["AbrahamsonEtAl2014RegTWN", "openquake.hazardlib.gsim.abrahamson_2014"],
            "ASK14jp": ["AbrahamsonEtAl2014RegJPN", "openquake.hazardlib.gsim.abrahamson_2014"],
            "ASK14chn": ["AbrahamsonEtAl2014RegCHN", "openquake.hazardlib.gsim.abrahamson_2014"],
            "BA08": ["BooreAtkinson2008", "openquake.hazardlib.gsim.boore_atkinson_2008"],
            "Bea14": ["BindiEtAl2014Rjb", "openquake.hazardlib.gsim.bindi_2014"],
            "Bea11": ["BindiEtAl2011", "openquake.hazardlib.gsim.bindi_2011"],
            "BJF97": ["BooreEtAl1997GeometricMean", "openquake.hazardlib.gsim.boore_1997"],
            "Bradley2013": ["Bradley2013", "openquake.hazardlib.gsim.bradley_2013"],
            "Bradley2013vol": ["Bradley2013Volc", "openquake.hazardlib.gsim.bradley_2013"],
            "BSSA14": ["BooreEtAl2014", "openquake.hazardlib.gsim.boore_2014"],
            "BSSA14ca": ["BooreEtAl2014CaliforniaBasin", "openquake.hazardlib.gsim.boore_2014"],
            "BSSA14jp": ["BooreEtAl2014JapanBasin", "openquake.hazardlib.gsim.boore_2014"],
            "BSSA14hq": ["BooreEtAl2014HighQ", "openquake.hazardlib.gsim.boore_2014"],
            "BSSA14lq": ["BooreEtAl2014LowQ", "openquake.hazardlib.gsim.boore_2014"],
            "C03": ["Campbell2003MwNSHMP2008", "openquake.hazardlib.gsim.campbell_2003"],
            "C03_share": ["Campbell2003SHARE", "openquake.hazardlib.gsim.campbell_2003"],
            "CB14": ["CampbellBozorgnia2014", "openquake.hazardlib.gsim.campbell_bozorgnia_2014"],
            "CB14jp": ["CampbellBozorgnia2014JapanSite", "openquake.hazardlib.gsim.campbell_bozorgnia_2014"],
            "CB14hq": ["CampbellBozorgnia2014HighQ", "openquake.hazardlib.gsim.campbell_bozorgnia_2014"],
            "CB14lq": ["CampbellBozorgnia2014LowQ", "openquake.hazardlib.gsim.campbell_bozorgnia_2014"],
            "Cau14": ["CauzziEtAl2014", "openquake.hazardlib.gsim.cauzzi_2014"],
            "Cau14nosof": ["CauzziEtAl2014NoSOF", "openquake.hazardlib.gsim.cauzzi_2014"],
            "CY14": ["ChiouYoungs2014", "openquake.hazardlib.gsim.chiou_youngs_2014"],
            "Gea05": ["GarciaEtAl2005SSlab", "openquake.hazardlib.gsim.garcia_2005"],
            "Fea96": ["FrankelEtAl1996MwNSHMP2008", "openquake.hazardlib.gsim.frankel_1996"],
            "Kea06s": ["Kanno2006Shallow", "openquake.hazardlib.gsim.kanno_2006"],
            "Kea06d": ["Kanno2006Deep", "openquake.hazardlib.gsim.kanno_2006"],
            "LinLee08i": ["LinLee2008SInter", "openquake.hazardlib.gsim.lin_lee_2008"],
            "LinLee08s": ["LinLee2008SSlab", "openquake.hazardlib.gsim.lin_lee_2008"],
            "Mont17i": ["MontalvaEtAl2017SInter", "openquake.hazardlib.gsim.montalva_2017"],
            "Mont17s": ["MontalvaEtAl2017SSlab", "openquake.hazardlib.gsim.montalva_2017"],
            "Pea11": ["PezeshkEtAl2011", "openquake.hazardlib.gsim.pezeshk_2011"],
            "Sea01": ["SomervilleEtAl2001NSHMP2008", "openquake.hazardlib.gsim.somerville_2001"],
            "Sea02": ["SilvaEtAl2002MwNSHMP2008", "openquake.hazardlib.gsim.silva_2002"],
            "Tea97": ["ToroEtAl1997MwNSHMP2008", "openquake.hazardlib.gsim.toro_1997"],
            "Toro02_share": ["ToroEtAl2002SHARE", "openquake.hazardlib.gsim.toro_2002"],
            "TP05": ["TavakoliPezeshk2005MwNSHMP2008", "openquake.hazardlib.gsim.tavakoli_pezeshk_2005"],
            "TL16s": ["TusaLanger2016RepiBA08SE", "openquake.hazardlib.gsim.tusa_langer_2016"],
            "TL16d": ["TusaLanger2016RepiBA08DE", "openquake.hazardlib.gsim.tusa_langer_2016"],
            "TL16rhypo": ["TusaLanger2016Rhypo", "openquake.hazardlib.gsim.tusa_langer_2016"],
            "Youngs97i": ["YoungsEtAl1997SInter", "openquake.hazardlib.gsim.youngs_1997"],
            "Youngs97s": ["YoungsEtAl1997SSlab", "openquake.hazardlib.gsim.youngs_1997"],
            "Zea06c": ["ZhaoEtAl2006Asc", "openquake.hazardlib.gsim.zhao_2006"],
            "Zea06i": ["ZhaoEtAl2006SInter", "openquake.hazardlib.gsim.zhao_2006"],
            "Zea06s": ["ZhaoEtAl2006SSlab", "openquake.hazardlib.gsim.zhao_2006"],
            "Zea16c": ["ZhaoEtAl2016Asc", "openquake.hazardlib.gsim.zhao_2016"],
            "Zea16i": ["ZhaoEtAl2016SInter", "openquake.hazardlib.gsim.zhao_2016"],
            "Zea16s": ["ZhaoEtAl2016SSlab", "openquake.hazardlib.gsim.zhao_2016"],
        }

        self.mode = mode
        self.config_file = config_file
        self.config_data = config_data or {}
        self.event_id = None  # Initialized as None, set after loading config


        if mode == "parse":
            self.initialize_parse_path()
            self.parse_config_file()
        elif mode == "create":
            if 'event_id' not in config_data:
                raise ValueError("event_id must be included in config_data for 'create' mode.")
            self.event_id = config_data['event_id']
            self.initialize_create_path()
            self.write_config_file()
        else:
            raise ValueError("Mode must be either 'parse' or 'create'.")

            
    def initialize_parse_path(self):
        """Initialize the file path for parsing. Uses default if none provided."""
        if not self.config_file:
            self.config_file = "./example_data/us7000m9g4/current/model_select.conf"
            print(f"Using default config file at {self.config_file}")
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"No configuration file found at {self.config_file}")


    def parse_config_file(self):
        """Parse the configuration file into a dictionary."""
        with open(self.config_file, "r") as f:
            content = f.read()

        config = {}
        current_section = None
        current_subsection = None

        # Regular expressions for sections, subsections, and key-value pairs
        section_pattern = re.compile(r"\[(\w+)\]")
        subsection_pattern = re.compile(r"\[\[(\w+)\]\]")
        key_value_pattern = re.compile(r"(\w+)\s*=\s*(.*)")

        for line in content.splitlines():
            line = line.strip()
            section_match = section_pattern.match(line)
            subsection_match = subsection_pattern.match(line)
            key_value_match = key_value_pattern.match(line)

            if section_match:
                current_section = section_match.group(1)
                config[current_section] = {}
                current_subsection = None
            elif subsection_match:
                current_subsection = subsection_match.group(1)
                config[current_section][current_subsection] = {}
            elif key_value_match:
                key, value = key_value_match.groups()
                value = self._convert_value(value.strip())
                if current_subsection:
                    config[current_section][current_subsection][key] = value
                else:
                    config[current_section][key] = value

        self.config_data = config

    def write_config_file(self):
        """Create the configuration file from the provided data, excluding 'event_id'."""
        final_path = os.path.join(self.config_file, "model_select.conf")

        with open(final_path, "w") as f:
            for section, section_data in self.config_data.items():
                if section == 'event_id':
                    continue  # Skip writing the 'event_id' section
                if isinstance(section_data, dict):
                    f.write(f"[{section}]\n")
                    for key, value in section_data.items():
                        if isinstance(value, dict):
                            f.write(f"    [[{key}]]\n")
                            for subkey, subvalue in value.items():
                                f.write(f"        {subkey} = {self._format_value(subvalue)}\n")
                        else:
                            f.write(f"    {key} = {self._format_value(value)}\n")
                else:
                    f.write(f"[{section}] {section_data}\n")  # Handle non-dictionary data

        print(f"Config file created at {final_path}")

    def initialize_create_path(self):
        """Initialize the path based on the provided config_file."""
        if not self.event_id:
            raise ValueError("Event ID must be provided for creating.")

        if not self.config_data:
            raise ValueError("Config data must be provided for creating.")

        if self.config_file:
            # Use provided config_file with the event_id
            self.config_file = os.path.join(self.config_file, self.event_id, "current")
        else:
            # Default path used when no specific file_path is provided
            self.config_file = f"./export/usgs-scenarios/{self.event_id}/current"

        os.makedirs(self.config_file, exist_ok=True)

    def _convert_value(self, value):
        """Convert string value to appropriate type."""
        if value.lower() == "none":
            return None
        elif value.lower() == "nan":
            return float("nan")
        elif "," in value:
            value_list = [v.strip() for v in value.split(",")]
            return [float(v) if v.replace(".", "", 1).isdigit() else v for v in value_list]
        else:
            try:
                return float(value)
            except ValueError:
                return value

    def _format_value(self, value):
        """Format value for output to file."""
        if value is None:
            return "None"
        elif isinstance(value, float) and math.isnan(value):
            return "nan"
        elif isinstance(value, list):
            return ", ".join(map(str, value))
        else:
            return str(value)

    def print_used_gmpes(self):
        """Print the GMPEs involved for the set that was used for calculations."""
        for set_name, gmpe_info in self.config_data.get("gmpe_sets", {}).items():
            gmpes = gmpe_info.get("gmpes", [])
            weights = gmpe_info.get("weights", [])

            for gmpe_set_name, set_weight in zip(gmpes, weights):
                gmpe_list = self.gmpe_sets.get(gmpe_set_name, {}).get("gmpes", [])
                gmpe_weights = self.gmpe_sets.get(gmpe_set_name, {}).get("weights", [])

                print(f"'{gmpe_set_name}'  [{set_weight}]")

                for gmpe, gmpe_weight in zip(gmpe_list, gmpe_weights):
                    class_name, module_name = self.gmpe_modules.get(gmpe, ["Unknown", "Unknown"])
                    print(f"  {gmpe} [{gmpe_weight}]: {class_name} from {module_name}")

            print()
            
    def get_used_gmpes(self):
        """Return a dictionary of the GMPEs involved for the set that was used for calculations."""
        gmpe_info_dict = {}

        for set_name, gmpe_info in self.config_data.get("gmpe_sets", {}).items():
            gmpes = gmpe_info.get("gmpes", [])
            weights = gmpe_info.get("weights", [])

            for gmpe_set_name, set_weight in zip(gmpes, weights):
                gmpe_list = self.gmpe_sets.get(gmpe_set_name, {}).get("gmpes", [])
                gmpe_weights = self.gmpe_sets.get(gmpe_set_name, {}).get("weights", [])

                gmpe_info_dict[gmpe_set_name] = {
                    "set_weight": set_weight,
                    "gmpes": [
                        {
                            "gmpe_name": gmpe,
                            "gmpe_weight": gmpe_weight,
                            "class_name": self.gmpe_modules.get(gmpe, ["Unknown", "Unknown"])[0],
                            "module_name": self.gmpe_modules.get(gmpe, ["Unknown", "Unknown"])[1],
                        }
                        for gmpe, gmpe_weight in zip(gmpe_list, gmpe_weights)
                    ],
                }

        return gmpe_info_dict
    
    def print_doc(self):
        """Prints the class docstring."""
        print(self.__doc__)
    
    
    
    
    
class ShakemapParserFLT:
    """
    Parses and visualizes earthquake shakemap data from FLT (raster format) and HDR (header information) files, or directly from ZIP archives containing these files. Supports different types of shakemap data including MMI, PGA, PGV, and spectral accelerations.

    Attributes:
    ----------
    zip_file (str): Path to the .zip file containing shakemap data. If not provided, default example data is used.
    map_type (str): Type of the shakemap ('mmi', 'pga', 'pgv', 'sa0.3', 'sa1.0'), specifying the ground motion parameter.
    data_type (str): Specifies the data aggregation type ('mean' or 'std'), indicating whether to use mean values or standard deviations.
    flt_file (str): Path to the .flt file if using direct file input.
    hdr_file (str): Path to the .hdr file if using direct file input.

    Methods:
    --------
    get_dataframe(self):
        Converts the raster data into a pandas DataFrame, including longitude, latitude, and the data values for easy manipulation and analysis.

    get_dict(self):
        Converts the loaded data and associated metadata into a dictionary format, facilitating easier serialization and access to data across different software environments.

    plot_shakemap(self):
        Visualizes the shakemap data on a geographical map using Cartopy, enhancing data interpretation and presentation.

    get_grid(self):
        Retrieves the geographical grid data used in the shakemap, useful for spatial analysis and further custom visualizations.

    get_stats(self):
        Computes and returns basic statistical measures (min, max, mean, standard deviation) of the shakemap data, aiding in quick data assessment.

    get_extent(self):
        Determines the geographical extent of the shakemap data based on the metadata, essential for setting up map boundaries in visualizations.

    add_stations(self, fig, ax, lon, lat):
        Adds seismic station markers to an existing shakemap plot, enhancing the plot with additional geospatial data points.

    add_rupture(self, fig, ax, x_coords, y_coords):
        Overlays a seismic rupture line based on given coordinates onto an existing shakemap plot, illustrating fault lines or rupture directions.

    add_epicenter(self, fig, ax, xcoord, ycoord):
        Marks the epicenter of the earthquake on the shakemap, highlighting the origin of the seismic event.

    add_dyfi(self, fig, ax, lon, lat, values):
        Overlays DYFI (Did You Feel It?) data points on the shakemap, providing visual insights into public reports of felt intensity.

    add_cities(self, fig, ax, population=100000, cities_csv='./SHAKEdata/worldcities.csv'):
        Plots cities within the extent of the shakemap on the map, highlighting cities with a population of more than the specified threshold.

    cleanup(self):
        Removes temporary files and directories created during the processing, maintaining a clean working environment.

    Internal Methods:
    -----------------
    construct_filenames(self):
        Constructs the filenames for the FLT and HDR files based on the shakemap type and data type, ensuring correct file handling.

    extract_files(self):
        Extracts specific FLT and HDR files from a ZIP archive, facilitating data access and processing.

    parse_hdr_file(self):
        Parses the HDR file to extract metadata about the shakemap, critical for understanding and processing the FLT data.

    read_flt_file(self):
        Reads the FLT file and converts it into a numpy array, enabling the raster data to be manipulated and visualized.
    
    contour_scale(self):
        Generates a color scale for seismic intensity maps based on the given map type and data type.



    Usage Examples:
    ---------------
    # Initializing with a ZIP file
    parser = ShakemapParser(zip_file='path/to/data.zip', map_type='mmi', data_type='mean')

    # Initializing with default example data
    parser = ShakemapParser(map_type='mmi', data_type='mean')

    # Plotting shakemap
    fig, ax = parser.plot_shakemap()

    # Adding additional data overlays
    parser.add_stations(fig, ax, stations_lon, stations_lat)
    parser.add_rupture(fig, ax, rupture_x, rupture_y)
    parser.add_epicenter(fig, ax, epicenter_x, epicenter_y)
    parser.add_dyfi(fig, ax, dyfi_lon, dyfi_lat, dyfi_values)
    parser.add_cities(fig, ax, population=100000, cities_csv='path/to/cities.csv')

    # Generating and retrieving data as DataFrame
    df = parser.get_dataframe()

    # Cleaning up extracted files
    parser.cleanup()

    © SHAKEmaps version 25.3.2
    """


    def __init__(self, zip_file=None, map_type="int", data_type="mean", flt_file=None, hdr_file=None):
        self.map_type = map_type.lower()
        if self.map_type == "int":  # Normalize "int" to "mmi"
            self.map_type = "mmi"
        

        self.data_type = data_type.lower()
        self.data_label = {  # Define data label map at initialization for consistency
            "mmi": "mmi",
            "pga": "pga",
            "pgv": "pgv",
            "sa03": "psa0p3",
            "sa1": "psa1p0"
        }.get(self.map_type, "mmi")
        
                # Default example data
        if zip_file is None and flt_file is None and hdr_file is None:
            zip_file = './example_data/usgs-shakemap-versions-us7000m9g4/us7000m9g4'

        
        # Handle ZIP file extraction or direct file paths
        if zip_file and zip_file.endswith('.zip'):
            self.zip_path = zip_file
            self.flt_file, self.hdr_file = self.construct_filenames()
            self.extract_files()
        elif flt_file and hdr_file:
            self.flt_path = flt_file
            self.hdr_path = hdr_file
        else:
            raise ValueError("Invalid input: specify either a ZIP file or both FLT and HDR filenames.")
        
        self.metadata = self.parse_hdr_file()
        self.data, self.nrows, self.ncols = self.read_flt_file()
        self.grid = self.get_grid()  # Optionally pre-calculate grid if used often

    def construct_filenames(self):
        
        base_name = self.data_label

        file_suffix = "_mean" if self.data_type == "mean" else "_std"
        flt_file = f"{base_name}{file_suffix}.flt"
        hdr_file = f"{base_name}{file_suffix}.hdr"
        return flt_file, hdr_file

    def extract_files(self):
        """Extracts the specific FLT and HDR files from a ZIP archive."""
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extract(self.flt_file, './temp/temp_extract')
            zip_ref.extract(self.hdr_file, './temp/temp_extract')
        self.flt_path = os.path.join('./temp/temp_extract', self.flt_file)
        self.hdr_path = os.path.join('./temp/temp_extract', self.hdr_file)

    def parse_hdr_file(self):
        metadata = {}
        with open(self.hdr_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    try:
                        metadata[key.upper()] = float(value)
                    except ValueError:
                        metadata[key.upper()] = value
        return metadata

    def read_flt_file(self):
        nrows = int(self.metadata['NROWS'])
        ncols = int(self.metadata['NCOLS'])
        data = np.fromfile(self.flt_path, dtype=np.float32).reshape((nrows, ncols))
        nodata = self.metadata.get('NODATA', None)
        if nodata is not None:
            data[data == nodata] = np.nan
        return data, nrows, ncols
        
    def cleanup(self):
        """Remove temporary extracted files and directories."""
        if hasattr(self, 'zip_path'):
            import shutil
            # Using shutil.rmtree to remove the directory and all its contents
            shutil.rmtree('./temp/temp_extract', ignore_errors=True)
    def get_dataframe(self):
        """Generate a DataFrame from the raster data with coordinates."""
        ulx = self.metadata['ULXMAP']
        uly = self.metadata['ULYMAP']
        xdim = self.metadata['XDIM']
        ydim = self.metadata['YDIM']
        x_coords = ulx + xdim * np.arange(self.ncols)
        y_coords = uly - ydim * np.arange(self.nrows)
        col_coord, row_coord = np.meshgrid(x_coords, y_coords)
        data_label = self.data_label
        df = pd.DataFrame({
            'lon': col_coord.ravel(),
            'lat': row_coord.ravel(),
            data_label: self.data.ravel()  # Using dynamic header based on map_type
        })
        return df
    
    def get_dict(self):
        """
        Converts the loaded shake map data and metadata into a dictionary.

        Returns:
            dict: A dictionary containing both the metadata and the raster data.
        """
        # Prepare the raster data as a dictionary
        data_dict = {
            'metadata': self.metadata,
            'data': self.data.tolist()  # Convert the numpy array to a list for JSON compatibility
        }

        # If the grid data has been computed, add it to the dictionary
        if hasattr(self, 'grid'):
            data_dict['grid'] = {
                'lon': self.grid['lon'].tolist(),
                'lat': self.grid['lat'].tolist(),
                self.data_label: self.grid[self.data_label].tolist()
            }

        return data_dict

    
    def get_grid(self):
        """Retrieve the grid without raveling for plotting or other operations."""
        ulx = self.metadata['ULXMAP']
        uly = self.metadata['ULYMAP']
        xdim = self.metadata['XDIM']
        ydim = self.metadata['YDIM']
        x_coords = ulx + xdim * np.arange(self.ncols)
        y_coords = uly + ydim * np.arange(self.nrows)
        
        data_label = self.data_label
            
        return {'lon': x_coords, 'lat': y_coords, data_label: self.data}
    

    
    def get_stats(self):
        """Calculate basic statistics of the raster data."""
        df = self.get_dataframe()
        return df.describe()
    def get_extent(self):
        """ Visualize the raster data on a geographic map. """
        grid = self.get_grid()
        #data_label = self.data_label

        """ Visualize the raster data on a geographic map using the specified extent calculation. """
        ulx = self.metadata['ULXMAP']  # Upper left x-coordinate
        uly = self.metadata['ULYMAP']  # Upper left y-coordinate
        xdim = self.metadata['XDIM']   # Cell width
        ydim = self.metadata['YDIM']   # Cell height (typically negative)
        ncols = self.ncols
        nrows = self.nrows
        extent = [ulx, ulx + ncols * xdim, uly - nrows * ydim, uly]
        
        return extent
    
    def contour_scale(self):
        """Generates a color scale for seismic intensity maps based on the given map type."""
        pgm_type = self.data_label.upper()

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
            "PSA0P3": [0, 0.02, 0.1, 1, 4.6, 10, 23, 50, 110, 244],
            "PSA1P0": [0, 0.02, 0.1, 1, 4.6, 10, 23, 50, 110, 244]
        }

        default_units = {
            "MMI": "MMI",
            "PGA": "%g",
            "PGV": "cm/s",
            "PSA0P3": "%g",
            "PSA1P0": "%g"
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
            "PSA0P3": "Spectral Acceleration $Sa_{0.3s}$",
            "PSA1P0": "Spectral Acceleration $Sa_{1s}$"
        }

        if pgm_type not in usgs_table:
            raise ValueError(f"Invalid PGM type '{pgm_type}'")

        bounds = usgs_table[pgm_type]
        ticks = bounds
        used_scale = f'{scale_labels[pgm_type]} ({labels[default_units[pgm_type]]})'
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        return cmap, bounds, ticks, norm, used_scale

        
    
    def plot_shakemap(self):
        """ Visualize the raster data on a geographic map. """
        grid = self.get_grid()
        data_label = self.data_label
        
        # Determine the colormap and normalization based on data type
        if self.data_type == 'mean':
            cmap, bounds, ticks, norm, used_scale = self.contour_scale()
        elif self.data_type == 'std':
            cmap = 'seismic'
            norm = mpl.colors.Normalize(vmin=-np.nanmax(np.abs(self.data)), vmax=np.nanmax(np.abs(self.data)))
            bounds = None
            ticks = None
            used_scale = 'Standard Deviation'


        """ Visualize the raster data on a geographic map using the specified extent calculation. """
        ulx = self.metadata['ULXMAP']  # Upper left x-coordinate
        uly = self.metadata['ULYMAP']  # Upper left y-coordinate
        xdim = self.metadata['XDIM']   # Cell width
        ydim = self.metadata['YDIM']   # Cell height (typically negative)
        ncols = self.ncols
        nrows = self.nrows

        extent = [ulx, ulx + ncols * xdim, uly - nrows * ydim, uly]
        
        fig, ax = plt.subplots(figsize=(24, 16), subplot_kw={'projection': ccrs.PlateCarree()})
        #cmap, bounds, ticks, norm, used_scale= contour_scale("mmi", "usgs")
        img = ax.imshow(self.data, extent=extent, origin='upper', cmap=cmap, norm=norm, 
                        transform=ccrs.PlateCarree(),zorder=1)
        
        # add map features 
        ax.coastlines(zorder=5)
        ax.add_feature(cfeature.BORDERS,zorder=5, linestyle='-')
        ax.add_feature(cfeature.OCEAN,zorder=2,facecolor='skyblue')

        
        # Adding gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.4, linestyle='--',zorder=6)

        # Add colorbar based on data type
        if self.data_type == 'mean':
            plt.colorbar(img, ax=ax, orientation='vertical', label=used_scale, ticks=ticks)
        else:
            plt.colorbar(img, ax=ax, orientation='vertical', label=used_scale)
        #plt.show()in the 
        
        #ax.legend(loc='upper right', fontsize='x-large')
        
        return fig, ax
    
    def add_stations(self, fig, ax, lon, lat):
        """
        Overlay station data on an existing shakemap figure, only within the map's extent.
        Parameters:
            fig (matplotlib.figure.Figure): The figure object where the shakemap is plotted.
            ax (matplotlib.axes._subplots.AxesSubplot): The axes object of the figure for plotting.
            lon (array-like): Longitudes of the seismic stations.
            lat (array-like): Latitudes of the seismic stations.
        """
        if not isinstance(fig, plt.Figure) or not isinstance(ax, plt.Axes):
            raise ValueError("Invalid figure or axes provided.")

        # Extracting the current extent of the shakemap
        extent = ax.get_xlim() + ax.get_ylim()  # (xmin, xmax, ymin, ymax)

        # Filtering stations to include only those within the extent
        within_extent = (lon >= extent[0]) & (lon <= extent[1]) & (lat >= extent[2]) & (lat <= extent[3])
        lon_within = lon[within_extent]
        lat_within = lat[within_extent]

        # Plotting the station locations as red triangle markers if within extent
        if lon_within.size > 0:  # Check if there are any points to plot
            ax.plot(lon_within, lat_within, 'k^', mfc='r', markersize=10, label='Seismic Station', zorder=5)

            # Check if the legend already exists and update if necessary
            legend = ax.get_legend()
            if legend is None or 'Seismic Station' not in [text.get_text() for text in legend.get_texts()]:
                ax.legend()

            # Redrawing the figure to update the display
            fig.canvas.draw()
            
            ax.legend(loc='upper right', fontsize='x-large')
            #plt.show()
        else:
            print("No seismic stations within the current shakemap extent.")

    def add_rupture(self, fig, ax, x_coords, y_coords):
        """
        Overlay rupture data on an existing shakemap figure.

        Parameters:
            fig (matplotlib.figure.Figure): The figure object where the shakemap is plotted.
            ax (matplotlib.axes._subplots.AxesSubplot): The axes object of the figure for plotting.
            x_coords (array-like): X coordinates (longitude) of the rupture.
            y_coords (array-like): Y coordinates (latitude) of the rupture.
        """
        if not isinstance(fig, plt.Figure) or not isinstance(ax, plt.Axes):
            raise ValueError("Invalid figure or axes provided.")

        # Plotting the rupture as a line on the existing map
        ax.plot(x_coords, y_coords, 'r-', linewidth=2, label='Rupture Extent', zorder=4)

        # Optionally, you can add a legend if not already present
        legend = ax.get_legend()
        if legend is None or 'Rupture Extent' not in [text.get_text() for text in legend.get_texts()]:
            ax.legend()

        # Update the plot to reflect the added rupture
        ax.legend(loc='upper right', fontsize='x-large')
        fig.canvas.draw()
        
        
    def add_epicenter(self, fig, ax, xcoord, ycoord):
        """
        Add the earthquake epicenter marker to an existing shakemap figure.

        Parameters:
            fig (matplotlib.figure.Figure): The figure object where the shakemap is plotted.
            ax (matplotlib.axes._subplots.AxesSubplot): The axes object of the figure for plotting.
            lon (float): Longitude of the earthquake epicenter.
            lat (float): Latitude of the earthquake epicenter.
        """
        if not isinstance(fig, plt.Figure) or not isinstance(ax, plt.Axes):
            raise ValueError("Invalid figure or axes provided.")

        # Plotting the epicenter as a yellow star on the existing map
        ax.plot(xcoord, ycoord, 'k*', mfc='y', markersize=20, label='Earthquake Epicenter', zorder=6)

        # Check for and add/update legend
        legend = ax.get_legend()
        if legend is None or 'Earthquake Epicenter' not in [text.get_text() for text in legend.get_texts()]:
            ax.legend(loc='upper right', fontsize='x-large')

        # Refresh the plot to show the epicenter
        fig.canvas.draw()
        
    def add_dyfi(self, fig, ax, lon, lat, values):
        """
        Add DYFI (Did You Feel It?) data points to an existing shakemap plot.

        Parameters:
            fig (matplotlib.figure.Figure): The figure object where the shakemap is plotted.
            ax (matplotlib.axes._subplots.AxesSubplot): The axes object of the figure for plotting.
            lon (array-like): Longitudes of DYFI data points.
            lat (array-like): Latitudes of DYFI data points.
            values (array-like): Intensity values of DYFI responses.
            cmap (matplotlib.colors.Colormap): The colormap instance to use for color mapping of the values.
            norm (matplotlib.colors.Normalize): The normalization instance to scale the data values into the colormap.
        """
        if not isinstance(fig, plt.Figure) or not isinstance(ax, plt.Axes):
            raise ValueError("Invalid figure or axes provided.")

        # Ensure values are numeric and not strings
        values = pd.to_numeric(values, errors='coerce')
        # Scatter plot for DYFI data points
        scatter = ax.scatter(lon, lat, c=values, cmap=cmap, norm=norm, edgecolor='none', s=35, alpha=0.75, zorder=4)

        # Refresh the plot to show the DYFI data
        fig.canvas.draw()
        
    def get_cities(self, cities_csv='./SHAKEdata/worldcities.csv'):
        """
        Get cities within the extent of the shakemap.

        Parameters:
            cities_csv (str): Path to the CSV file containing city data.

        Returns:
            pd.DataFrame: DataFrame containing cities within the extent.
        """
        extent = self.get_extent()
        cities_df = pd.read_csv(cities_csv)
        cities_within_extent = cities_df[
            (cities_df['Longitude'] >= extent[0]) & 
            (cities_df['Longitude'] <= extent[1]) & 
            (cities_df['Latitude'] >= extent[2]) & 
            (cities_df['Latitude'] <= extent[3])
        ]
        return cities_within_extent

    def add_cities(self, fig, ax,population=100000, cities_csv='./SHAKEdata/worldcities.csv'):
        """
        Plot cities within the extent of the shakemap on the map, 
        with cities having a population of more than 100,000 highlighted.

        Parameters:
            fig (matplotlib.figure.Figure): The figure object where the shakemap is plotted.
            ax (matplotlib.axes._subplots.AxesSubplot): The axes object of the figure for plotting.
            cities_csv (str): Path to the CSV file containing city data.
        """
        import matplotlib.patheffects as PathEffects
        cities_df = self.get_cities(cities_csv)
        large_cities = cities_df[cities_df['population'] > population]

        # Plot cities with a population of more than 100,000
        for _, city in large_cities.iterrows():
            lon = city['Longitude']
            lat = city['Latitude']
            city_name = city['city_name']
            ax.plot(lon, lat, 'wo', mfc='k', markersize=7, transform=ccrs.Geodetic(), zorder=5)  # Plot city as a red dot
            text = ax.text(lon + 0.001, lat, city_name, verticalalignment='center', transform=ccrs.Geodetic(),
                           fontsize=15, color='black', style='italic', zorder=5)
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

        # Check for and add/update legend
        legend = ax.get_legend()
        if legend is None or 'Cities > 100,000' not in [text.get_text() for text in legend.get_texts()]:
            ax.legend(loc='upper right', fontsize='x-large')

        # Refresh the plot to show the cities
        fig.canvas.draw()
        
        
        
    def print_doc(self):
        """Prints the class docstring."""
        print(self.__doc__)
        
        
        

class ShakemapXMLParser:
    """
    ShakemapXMLParser: A Comprehensive Parser for USGS Shakemap XML Files
    ===============================================================
    
    Overview
    --------
    The ShakemapXMLParser is a Python class designed to read and parse USGS Shakemap XML files.
    It extracts metadata (such as event information and grid specifications) and grid data (the actual
    shakemap values) from the XML file. This class supports two types of files:
      - Mean values file (e.g., "us7000pn9s_us_001_grid.xml")
      - Uncertainty file (e.g., "us7000pn9s_us_002_uncertainty.xml")
    
    It supports different intensity measure types (IMT) such as:
      - "int" or "mmi": for intensity values,
      - "pga": for Peak Ground Acceleration,
      - "pgv": for Peak Ground Velocity,
      - "sa": for Spectral Acceleration.
      
    Depending on the imt and value_type provided, a subset of columns is extracted.
    Additionally, for each extracted data column, an adjacent column containing the corresponding unit
    (as defined in the XML file) is inserted.
    
    Installation & Requirements
    -----------------------------
    - Modules: xml.etree.ElementTree (standard library), numpy, pandas, and logging.
    
    Usage
    -----
    Instantiate the class by providing:
        - xml_file: Path to the XML shakemap file.
        - imt: The intensity measure type (default is 'mmi').
        - value_type: Either 'mean' (default) or 'std' for uncertainty.
    
    Example:
        xml_file_path = "us7000pn9s_us_001_grid.xml"
        parser = ShakemapXMLParser(xml_file_path, imt="mmi", value_type="mean")
        
        # Get DataFrame with data and unit columns interleaved.
        df = parser.get_dataframe()
        print(df.head())
        
        # Retrieve metadata, grid extent, basic statistics or the entire parsed dictionary.
        metadata = parser.get_metadata()
        extent = parser.get_extent()
        stats = parser.get_stats()
        data_dict = parser.get_dict()
        
    Error Handling & Logging
    --------------------------
    - This class uses extensive logging to record the progress of XML parsing and DataFrame creation.
    - Errors such as missing XML elements, conversion errors, or mismatches in grid data dimensions
      are logged with an appropriate log level (e.g., ERROR) and then raised as exceptions.
    - This makes it easier to troubleshoot issues when integrating this parser into larger applications.
    
    Detailed Methods
    ----------------
    - __init__: Initializes the parser, normalizes inputs, and parses the XML file.
    - _parse_xml: Internal method that reads the XML file, extracting metadata, grid fields, and data.
    - _get_selected_columns: Determines the list of columns to extract based on the given imt and value_type.
    - get_dataframe: Returns a DataFrame containing the selected data columns with original names intact,
                     plus an added adjacent unit column (e.g., LON_units).
    - get_dict: Returns a dictionary containing metadata, data, and a mapping of units.
    - get_units: Returns a dictionary mapping each selected column to its corresponding unit.
    - get_extent: Calculates the geographical extent (lon_min, lon_max, lat_min, lat_max) and returns it as a list.
    - get_stats: Provides basic statistics (using pandas describe()) on the selected data.
    - get_grid: Returns the grid data in dictionary form (useful for plotting).
    - get_metadata: Returns the extracted metadata.
    - print_doc: Prints the class-level documentation (this README-style docstring).
    
    Note: Make sure the XML file is accessible and well-formed. Detailed logging output can be
          enabled by configuring the logging level.

    © SHAKEmaps version 25.3.2
    """

    def __init__(self, xml_file, imt='mmi', value_type='mean'):
        logger.info("Initializing ShakemapXMLParser with file: %s", xml_file)

        if not os.path.exists(xml_file):
            logger.error("File '%s' not found.", xml_file)
            raise FileNotFoundError(f"File '{xml_file}' not found.")

        self.xml_file = xml_file
        # Normalize the intensity measure to lowercase.
        self.imt = imt.lower()
        self.value_type = value_type.lower()  # Should be either 'mean' or 'std'
        self.metadata = {}       # To store event and grid specification metadata.
        self.field_names = []    # Ordered list of field names.
        self.units = {}          # Dictionary mapping field names to their unit strings.
        self.data = []           # Will hold the grid data as a NumPy array.

        # Parse the XML file
        self._parse_xml()

    def _parse_xml(self):
        """Internal method to read and parse the XML file."""
        logger.debug("Parsing XML file: %s", self.xml_file)
        try:
            tree = ET.parse(self.xml_file)
            root = tree.getroot()
        except ET.ParseError as pe:
            logger.error("Failed to parse XML file: %s", pe)
            raise ValueError("Failed to parse XML file: " + str(pe))
        except Exception as e:
            logger.error("Unexpected error reading XML file: %s", e)
            raise

        # Define namespace used in the USGS shakemap XML.
        ns = {'ns': 'http://earthquake.usgs.gov/eqcenter/shakemap'}

        # Extract event metadata.
        event_el = root.find('ns:event', ns)
        if event_el is not None:
            self.metadata['event'] = event_el.attrib
            logger.debug("Extracted event metadata.")
        else:
            logger.warning("No event metadata found.")
            self.metadata['event'] = {}

        # Extract grid specification metadata.
        grid_spec_el = root.find('ns:grid_specification', ns)
        if grid_spec_el is not None:
            self.metadata['grid_specification'] = grid_spec_el.attrib
            logger.debug("Extracted grid_specification metadata.")
        else:
            logger.warning("No grid_specification metadata found.")
            self.metadata['grid_specification'] = {}

        # Retrieve grid_field elements; enforce order via the 'index' attribute.
        grid_fields = root.findall('ns:grid_field', ns)
        if not grid_fields:
            logger.error("No grid_field elements found in XML.")
            raise ValueError("No grid_field elements found in XML.")
        grid_fields.sort(key=lambda x: int(x.get('index')))
        for gf in grid_fields:
            name = gf.get('name')
            unit = gf.get('units')
            if name is None or unit is None:
                logger.error("A grid_field element is missing 'name' or 'units'.")
                raise ValueError("A grid_field element is missing 'name' or 'units'.")
            self.field_names.append(name)
            self.units[name] = unit
        logger.debug("Extracted grid_field definitions: %s", self.field_names)

        # Extract grid_data element and convert its text into a NumPy array.
        grid_data_el = root.find('ns:grid_data', ns)
        if grid_data_el is None or not grid_data_el.text:
            logger.error("No grid_data element found or grid_data is empty.")
            raise ValueError("No grid_data element found in XML file.")
        tokens = grid_data_el.text.strip().split()
        logger.debug("Found %d tokens in grid_data.", len(tokens))
        try:
            data_float = [float(x) for x in tokens]
        except Exception as e:
            logger.error("Error converting grid_data tokens to float: %s", e)
            raise ValueError("Error converting grid_data tokens to float: " + str(e))

        num_fields = len(self.field_names)
        try:
            nlon = int(self.metadata['grid_specification'].get('nlon', 0))
            nlat = int(self.metadata['grid_specification'].get('nlat', 0))
            num_points = nlon * nlat if nlon and nlat else len(data_float) // num_fields
            logger.debug("Using grid_specification: nlon=%d, nlat=%d, calculated points=%d", nlon, nlat, num_points)
        except Exception as e:
            logger.warning("Issue with grid_specification metadata: %s. Falling back to computed points.", e)
            num_points = len(data_float) // num_fields

        if len(data_float) != num_points * num_fields:
            msg = "Mismatch between grid_data length and expected grid dimensions."
            logger.error(msg)
            raise ValueError(msg)

        self.data = np.array(data_float).reshape((num_points, num_fields))
        logger.info("Successfully parsed XML and created data array with shape %s", self.data.shape)

    def _get_selected_columns(self):
        """
        Determines which columns to select from the XML data based on the 'imt' and 'value_type' properties.
        Returns:
            list: A list of selected column names.
        """
        if self.value_type == 'mean':
            if self.imt in ['int', 'mmi']:
                selected = ['LON', 'LAT', 'SVEL', 'MMI']
            elif self.imt == 'pga':
                selected = ['LON', 'LAT', 'SVEL', 'PGA']
            elif self.imt == 'pgv':
                selected = ['LON', 'LAT', 'SVEL', 'PGV']
            elif self.imt == 'sa':
                selected = ['LON', 'LAT', 'SVEL', 'PSA03', 'PSA06', 'PSA10', 'PSA30']
            else:
                msg = f"Unknown imt value for mean: '{self.imt}'"
                logger.error(msg)
                raise ValueError(msg)
        elif self.value_type == 'std':
            if self.imt in ['int', 'mmi']:
                selected = ['LON', 'LAT', 'STDMMI']
            elif self.imt in ['pga']:
                selected = ['LON', 'LAT', 'STDPGA']
            elif self.imt in ['pgv']:
                selected = ['LON', 'LAT', 'STDPGV']
            elif self.imt in ['sa']:
                selected = ['LON', 'LAT', 'STDPSA03', 'STDPSA06', 'STDPSA10', 'STDPSA30']
            else:
                msg = f"Unknown imt value for std: '{self.imt}'"
                logger.error(msg)
                raise ValueError(msg)
        else:
            msg = f"Unknown value_type: '{self.value_type}'"
            logger.error(msg)
            raise ValueError(msg)
        logger.debug("Selected columns based on imt and value_type: %s", selected)
        return selected

    def get_dataframe(self):
        """
        Generate a pandas DataFrame from the XML data filtered by imt and value_type.
        The DataFrame is built with the original column names intact. For each selected column, an
        additional column (named '<column>_units') is inserted immediately after, containing a constant
        unit value (repeated for each row).

        Returns:
            pandas.DataFrame: DataFrame with interleaved data and unit columns.
        """
        logger.info("Generating DataFrame with selected columns and unit columns interleaved.")
        df_full = pd.DataFrame(self.data, columns=self.field_names)
        selected = self._get_selected_columns()
        missing = [col for col in selected if col not in df_full.columns]
        if missing:
            msg = "The following required fields are missing in the XML: " + ", ".join(missing)
            logger.error(msg)
            raise ValueError(msg)
        
        df_selected = df_full[selected].copy()
        # Interleave the unit columns after their corresponding data columns
        columns_order = []
        for col in selected:
            columns_order.append(col)
            unit_col = col + "_units"
            df_selected[unit_col] = self.units.get(col)
            columns_order.append(unit_col)
        df_result = df_selected[columns_order]
        logger.debug("DataFrame generated with columns: %s", df_result.columns.tolist())
        return df_result

    def get_dict(self):
        """
        Converts the loaded shakemap data and metadata into a dictionary.
        
        The returned dictionary contains:
            - 'metadata': the event and grid specification metadata.
            - 'data': a dictionary of lists (from the DataFrame with interleaved unit columns).
            - 'units': a mapping from each selected column to its unit.

        Returns:
            dict: A dictionary containing metadata, data, and unit mappings.
        """
        try:
            df = self.get_dataframe()
        except Exception as e:
            logger.error("Failed to get DataFrame for dictionary conversion: %s", e)
            raise
        data_dict = {
            'metadata': self.metadata,
            'data': df.to_dict(orient='list'),
            'units': self.get_units()
        }
        logger.info("Created dictionary from parsed data.")
        return data_dict

    def get_units(self):
        """
        Retrieves the units of the selected columns based on imt and value_type.

        Returns:
            dict: A dictionary mapping each selected column to its unit.
        """
        selected = self._get_selected_columns()
        units_dict = {col: self.units.get(col, None) for col in selected}
        logger.debug("Units for selected columns: %s", units_dict)
        return units_dict

    def get_extent(self):
        """
        Determine the geographical extent of the shakemap data based on the data (i.e., min/max longitude and latitude)
        and return it as a list in the order: [lon_min, lon_max, lat_min, lat_max].

        Returns:
            list: List containing the geographical extent.
        """
        logger.info("Calculating geographical extent from data.")
        try:
            df = self.get_dataframe()
            lon_min = df['LON'].min()
            lon_max = df['LON'].max()
            lat_min = df['LAT'].min()
            lat_max = df['LAT'].max()
        except Exception as e:
            logger.error("Failed to calculate geographical extent: %s", e)
            raise
        extent = [lon_min, lon_max, lat_min, lat_max]
        logger.debug("Calculated extent: %s", extent)
        return extent

    def get_stats(self):
        """
        Calculate basic statistics (e.g., count, mean, std, min, max) of the extracted shakemap data using pandas describe().

        Returns:
            pandas.DataFrame: A DataFrame with statistical summary.
        """
        logger.info("Calculating basic statistics from the data.")
        try:
            df = self.get_dataframe()
        except Exception as e:
            logger.error("Failed to calculate statistics: %s", e)
            raise
        stats_df = df.describe()
        logger.debug("Basic statistics calculated.")
        return stats_df

    def get_grid(self):
        """
        Retrieve the grid data in its raw form as a dictionary,
        suitable for plotting or other operations.

        Returns:
            dict: A dictionary containing arrays for longitude, latitude, and each other data field.
        """
        logger.info("Retrieving grid data as a dictionary.")
        try:
            df = self.get_dataframe()
        except Exception as e:
            logger.error("Failed to generate grid data: %s", e)
            raise
        # Identify columns by their exact names, knowing "LON" and "LAT" remain unaltered.
        lon_col = "LON"
        lat_col = "LAT"
        grid = {
            'lon': df[lon_col].values,
            'lat': df[lat_col].values
        }
        for col in df.columns:
            if col not in [lon_col, "LON_units", lat_col, "LAT_units"]:
                grid[col] = df[col].values
        logger.debug("Grid data retrieved with keys: %s", list(grid.keys()))
        return grid

    def get_metadata(self):
        """
        Retrieves the metadata of the shakemap, including event and grid specifications.

        Returns:
            dict: The metadata dictionary.
        """
        logger.info("Retrieving metadata.")
        return self.metadata

    def print_doc(self):
        """Prints the class documentation."""
        print(self.__doc__)





class ShakemapXMLViewer:
    """
    ShakemapXMLViewer: A Comprehensive Viewer and Visualizer for Earthquake Shakemap XML Data
    ================================================================================
    
    Overview
    --------
    The ShakemapXMLViewer class is designed to parse, process, and visualize USGS earthquake shakemap 
    XML files. It extracts key seismic data—such as event metadata, grid specifications, and grid data—from 
    the XML file. In addition, this class offers functionalities to convert the extracted data into a 
    pandas DataFrame or a dictionary, compute basic statistics, and generate high-quality visualizations 
    on geographical maps. It supports overlaying additional geospatial layers like seismic station markers, 
    rupture outlines, earthquake epicenters, DYFI (Did You Feel It?) data points, and city locations.
    
    Installation & Requirements
    -----------------------------
    - Required libraries: xml.etree.ElementTree (standard library), numpy, pandas, matplotlib, cartopy, 
      and matplotlib.patheffects.
    - Data dependencies: Shakemap XML files and optionally a CSV file for world cities (default: 
      './SHAKEdata/worldcities.csv').
    
    Usage
    -----
    To use the viewer, instantiate the class with the path to your XML shakemap file and the desired 
    intensity measure type (IMT). The IMT is normalized to uppercase for consistency.
    
    Example:
        viewer = ShakemapXMLViewer(xml_file="path/to/shakemap.xml", imt="MMI")
        
        # Convert the parsed data to a DataFrame.
        df = viewer.get_dataframe()
        
        # Get a dictionary representation of the parsed data and metadata.
        data_dict = viewer.get_dict()
        
        # Retrieve basic statistical measures of the shakemap data.
        stats = viewer.get_stats()
        
        # Visualize the shakemap.
        fig, ax = viewer.plot_shakemap()
        
        # Overlay additional data, such as seismic stations, rupture lines, epicenters, and cities.
        viewer.add_stations(fig, ax, station_lon_array, station_lat_array)
        viewer.add_epicenter(fig, ax, epicenter_lon, epicenter_lat)
        cities_df = viewer.get_cities()
        viewer.add_cities(fig, ax, population=100000)
    
    Attributes
    ----------
    file_path : str
        Path to the XML file containing the shakemap data.
    imt : str
        Intensity measure type (e.g., 'MMI', 'PGA', 'PGV', 'PSA03', 'PSA10', 'SVEL', etc.). The value is 
        normalized to uppercase.
    extent : list
        Geographic extent of the shakemap data defined as [lon_min, lon_max, lat_min, lat_max].
    
    User Methods
    ------------
    get_dataframe(self)
        Converts the XML data into a pandas DataFrame with the original column names intact and an additional 
        unit column added next to each data column.
    get_dict(self)
        Converts the loaded shakemap data and metadata into a dictionary for easier serialization and access.
    plot_shakemap(self)
        Visualizes the shakemap data on a geographic map using Cartopy and matplotlib, including coastlines, 
        borders, ocean features, and a color bar representing the seismic intensity.
    get_grid(self)
        Retrieves the geographical grid data without raveling, suitable for spatial analysis or custom plotting.
    get_stats(self)
        Computes basic statistical measures (e.g., count, mean, standard deviation, min, max) of the shakemap data.
    get_extent(self)
        Determines and returns the geographical extent of the shakemap data based on the data values, in the 
        form [lon_min, lon_max, lat_min, lat_max].
    set_extent(self, extent)
        Allows manual setting of the geographical extent. Expects a list of four values.
    contour_scale(self)
        Generates a USGS-based color scale for seismic intensity maps, returning the colormap, bounds, ticks, 
        normalization, and descriptive label.
    add_stations(self, fig, ax, lon, lat)
        Adds seismic station markers to an existing shakemap plot.
    add_rupture(self, fig, ax, x_coords, y_coords)
        Overlays a seismic rupture line onto an existing shakemap plot.
    add_epicenter(self, fig, ax, xcoord, ycoord)
        Marks the epicenter of the earthquake on the shakemap.
    add_dyfi(self, fig, ax, lon, lat, values)
        Overlays DYFI (Did You Feel It?) data points on the shakemap, providing visual insight into public 
        intensity reports.
    get_cities(self, cities_csv='./SHAKEdata/worldcities.csv')
        Retrieves a DataFrame of cities within the shakemap extent based on data from a CSV file.
    add_cities(self, fig, ax, population=100000, cities_csv='./SHAKEdata/worldcities.csv')
        Plots cities with populations greater than the specified threshold on the map.
    get_metadata(self)
        Retrieves the metadata extracted from the XML, including event details and grid specifications.
    print_doc(self)
        Prints this complete documentation string for the class.
    
    Internal Methods
    ----------------
    parse_metadata(self)
        Parses the XML file to extract event metadata and grid specification details.
    parse_data(self)
        Parses the XML file to extract grid field definitions (names and units) and grid data, converting the 
        latter into a numpy array.
    
    Error Handling & Logging
    --------------------------
    This class implements thorough error handling: it checks for malformed XML, missing required elements, 
    and data format inconsistencies, and it raises descriptive exceptions to assist troubleshooting. In 
    addition, it uses logging (configurable externally) to track the processing stages and any issues that occur.
    
    Additional Information
    ----------------------
    ShakemapXMLViewer is ideal for researchers and developers focusing on seismic analysis and geospatial 
    data visualization. By integrating data extraction, statistical analysis, and advanced mapping capabilities 
    into a single tool, it streamlines the workflow for processing earthquake shakemap data.


    © SHAKEmaps 24.5.2
    """

    def __init__(self, xml_file=None, imt="MMI"):
        self.xml_file = xml_file or './example_data/usgs-shakemap-versions-us7000m9g4/us7000m9g4/us7000m9g4_us_011_grid.xml'
        self.imt = imt.upper()  # Normalize IMT to upper case
        self.tree = ET.parse(self.xml_file)
        self.root = self.tree.getroot()
        self.metadata = self.parse_metadata()
        self.data, self.fields = self.parse_data()
        self.extent = self.get_extent()  # Initialize extent


    def parse_metadata(self):
        metadata = {}
        for child in self.root:
            if child.tag.endswith("event"):
                metadata.update(child.attrib)
            elif child.tag.endswith("grid_specification"):
                metadata.update(child.attrib)
        return metadata

    def parse_data(self):
        fields = []
        data = []
        units = {}
        for child in self.root:
            if child.tag.endswith("grid_field"):
                field_name = child.attrib["name"]
                fields.append(field_name)
                units[field_name] = child.attrib["units"]
            elif child.tag.endswith("grid_data"):
                rows = child.text.strip().split("\n")
                for row in rows:
                    data.append(list(map(float, row.split())))
        return np.array(data), (fields, units)

    def get_dataframe(self):
        """Generate a DataFrame from the XML data."""
        df = pd.DataFrame(self.data, columns=self.fields[0])
        columns_with_units = []
        for field in self.fields[0]:
            columns_with_units.append(field)
            if field in self.fields[1]:
                columns_with_units.append(field + "_units")
                df[field + "_units"] = self.fields[1][field]
        return df[columns_with_units]


    def get_dict(self):
        """
        Converts the loaded shakemap data and metadata into a dictionary.

        Returns:
            dict: A dictionary containing both the metadata and the grid data.
        """
        data_dict = {
            'metadata': self.metadata,
            'data': self.data.tolist()
        }
        return data_dict

    def get_grid(self):
        """Retrieve the grid without raveling for plotting or other operations."""
        df = self.get_dataframe()
        grid = {
            'lon': df['LON'].values,
            'lat': df['LAT'].values
        }
        for field in self.fields[0][2:]:
            grid[field] = df[field].values
        return grid

    def get_stats(self):
        """Calculate basic statistics of the raster data."""
        df = self.get_dataframe()
        return df.describe()

    def get_extent(self):
        """Determine the geographical extent of the shakemap data."""
        extent_method = 'data'
        if extent_method=='metadata':
            lon_min = float(self.metadata['lon_min'])
            lon_max = float(self.metadata['lon_max'])
            lat_min = float(self.metadata['lat_min'])
            lat_max = float(self.metadata['lat_max'])
        elif extent_method=='data':
            """Determine the geographical extent of the shakemap data based on the data itself."""
            df = self.get_dataframe()
            lon_min = df['LON'].min()
            lon_max = df['LON'].max()
            lat_min = df['LAT'].min()
            lat_max = df['LAT'].max()

        return [lon_min, lon_max, lat_min, lat_max]

    def contour_scale(self):
        """Generates a color scale for seismic intensity maps based on the given map type."""
        pgm_type = self.imt.upper()

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

        if pgm_type not in usgs_table:
            raise ValueError(f"Invalid PGM type '{pgm_type}'")

        bounds = usgs_table[pgm_type]
        ticks = bounds
        used_scale = f'{scale_labels[pgm_type]} ({labels[default_units[pgm_type]]})'
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        return cmap, bounds, ticks, norm, used_scale

    def plot_shakemap(self):
        """ Visualize the shakemap data on a geographic map. """
        df = self.get_dataframe()
        lon = df['LON'].values
        lat = df['LAT'].values
        data_label = self.imt.upper()

        cmap, bounds, ticks, norm, used_scale = self.contour_scale()

        fig, ax = plt.subplots(figsize=(24, 16), subplot_kw={'projection': ccrs.PlateCarree()})
        scatter = ax.scatter(lon, lat, c=df[data_label], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=1)
    
        ax.set_extent(self.extent, crs=ccrs.PlateCarree())
        ax.coastlines(zorder=5)
        ax.add_feature(cfeature.BORDERS, zorder=5, linestyle='-')
        ax.add_feature(cfeature.OCEAN, zorder=2, facecolor='skyblue')

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.4, linestyle='--', zorder=6)

        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', label=used_scale, ticks=ticks)
        #plt.show()

        return fig, ax


    def add_stations(self, fig, ax, lon, lat):
        """Add seismic station markers to an existing shakemap plot."""
        if not isinstance(fig, plt.Figure) or not isinstance(ax, plt.Axes):
            raise ValueError("Invalid figure or axes provided.")
    
        extent = ax.get_xlim() + ax.get_ylim()
        within_extent = (lon >= extent[0]) & (lon <= extent[1]) & (lat >= extent[2]) & (lat <= extent[3])
        lon_within = lon[within_extent]
        lat_within = lat[within_extent]
    
        if lon_within.size > 0:
            ax.plot(lon_within, lat_within, 'k^', mfc='r', markersize=10, label='Seismic Station', zorder=5)
            legend = ax.get_legend()
            if legend is None or 'Seismic Station' not in [text.get_text() for text in legend.get_texts()]:
                ax.legend(loc='upper right')  # Specify a fixed location
            fig.canvas.draw()
        else:
            print("No seismic stations within the current shakemap extent.")

    def add_rupture(self, fig, ax, x_coords, y_coords):
        """Overlay rupture data on an existing shakemap figure."""
        if not isinstance(fig, plt.Figure) or not isinstance(ax, plt.Axes):
            raise ValueError("Invalid figure or axes provided.")
    
        ax.plot(x_coords, y_coords, 'r-', linewidth=2, label='Rupture Extent', zorder=4)
        legend = ax.get_legend()
        if legend is None or 'Rupture Extent' not in [text.get_text() for text in legend.get_texts()]:
            ax.legend(loc='upper right')  # Specify a fixed location
        fig.canvas.draw()
    
    def add_epicenter(self, fig, ax, xcoord, ycoord):
        """Add the earthquake epicenter marker to an existing shakemap figure."""
        if not isinstance(fig, plt.Figure) or not isinstance(ax, plt.Axes):
            raise ValueError("Invalid figure or axes provided.")
    
        ax.plot(xcoord, ycoord, 'k*', mfc='y', markersize=20, label='Earthquake Epicenter', zorder=6)
        legend = ax.get_legend()
        if legend is None or 'Earthquake Epicenter' not in [text.get_text() for text in legend.get_texts()]:
            ax.legend(loc='upper right')  # Specify a fixed location
        fig.canvas.draw()

    def add_dyfi(self, fig, ax, lon, lat, values):
        """Add DYFI (Did You Feel It?) data points to an existing shakemap plot."""
        if not isinstance(fig, plt.Figure) or not isinstance(ax, plt.Axes):
            raise ValueError("Invalid figure or axes provided.")

        values = pd.to_numeric(values, errors='coerce')
        scatter = ax.scatter(lon, lat, c=values, cmap='viridis', edgecolor='none', s=35, alpha=0.75, zorder=4)
        fig.canvas.draw()

    def get_cities(self, cities_csv='./SHAKEdata/worldcities.csv'):
        """
        Get cities within the extent of the shakemap.

        Parameters:
            cities_csv (str): Path to the CSV file containing city data.

        Returns:
            pd.DataFrame: DataFrame containing cities within the extent.
        """
        extent = self.get_extent()
        cities_df = pd.read_csv(cities_csv)
        cities_within_extent = cities_df[
            (cities_df['Longitude'] >= extent[0]) & 
            (cities_df['Longitude'] <= extent[1]) & 
            (cities_df['Latitude'] >= extent[2]) & 
            (cities_df['Latitude'] <= extent[3])
        ]
        return cities_within_extent

    def add_cities(self, fig, ax, population=100000, cities_csv='./SHAKEdata/worldcities.csv'):
        """Plot cities within the extent of the shakemap on the map, highlighting cities with a population of more than the specified threshold."""
        cities_df = self.get_cities(cities_csv)
        large_cities = cities_df[cities_df['population'] > population]

        for _, city in large_cities.iterrows():
            lon = city['Longitude']
            lat = city['Latitude']
            city_name = city['city_name']
            ax.plot(lon, lat, 'wo', mfc='k', markersize=7, transform=ccrs.Geodetic(), zorder=5)
            text = ax.text(lon + 0.001, lat, city_name, verticalalignment='center', transform=ccrs.Geodetic(),
                           fontsize=15, color='black', style='italic', zorder=5)
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

        legend = ax.get_legend()
        if legend is None or 'Cities > 100,000' not in [text.get_text() for text in legend.get_texts()]:
            ax.legend(loc='upper right', fontsize='x-large')
        fig.canvas.draw()

    def get_metadata(self):
        """Retrieves the metadata of the shakemap, including event and grid specifications."""
        return self.metadata

    def print_doc(self):
        """Prints the class docstring."""
        print(self.__doc__)


    def set_extent(self, extent):
        """
        Sets the geographical extent of the shakemap.
    
        Parameters:
        -----------
        extent (list): A list containing [lon_min, lon_max, lat_min, lat_max].
        """
        if len(extent) != 4:
            raise ValueError("Extent must be a list of four values: [lon_min, lon_max, lat_min, lat_max].")
        self.extent = extent


class ParseEventInfo:
    """
    ParseEventInfo Class

    This class is designed to load, parse, and analyze a JSON file containing detailed earthquake event data.
    The JSON file typically includes multiple sections such as:
        - GMPE Selection: Detailed ground motion prediction equation (GMPE) information for various periods.
        - Event Information: Metadata for the earthquake event (e.g., event ID, magnitude, origin time, etc.).
        - STREC: Seismic record parameters related to the event.
        - Output: Ground motion outputs, map information, and uncertainty details.
        - Processing: Details of modules and flags used during data processing.
        - Site Response: Information regarding site response characteristics.

    Key Features:
        - Loads and parses the input JSON file into a Python dictionary.
        - Provides access to the full dataset as a dictionary (via get_dict) and a flattened pandas DataFrame (via get_dataframe).
        - Extracts and summarizes the GMPE section with both dictionary (get_gmpe_summary) and DataFrame (get_gmpe_dataframe) outputs.
        - Supplies methods to extract other specific sections (event information, STREC, output, processing, and site response)
          as either dictionaries or structured DataFrames, facilitating tailored analysis.

    Usage Example:
        file_path = "./example_data/SHAKEfetch/usgs-event_info-versions/us7000pn9s/us7000pn9s_us_020_info.json"
        try:
            event_parser = ParseEventInfo(file_path)

            # Retrieve full event data as a dictionary and DataFrame
            full_data = event_parser.get_dict()
            full_df = event_parser.get_dataframe()

            # Access GMPE details as both a summary dictionary and a DataFrame
            gmpe_summary = event_parser.get_gmpe_summary()
            gmpe_df = event_parser.get_gmpe_dataframe()

            # Access event information details
            event_info = event_parser.get_event_info()
            event_info_df = event_parser.get_event_info_dataframe()

            # Retrieve STREC (seismic record) information
            strec = event_parser.get_strec()
            strec_df = event_parser.get_strec_dataframe()

            # Retrieve output section details
            output = event_parser.get_output()
            output_df = event_parser.get_output_dataframe()

            # Retrieve processing section details
            processing = event_parser.get_processing()
            processing_df = event_parser.get_processing_dataframe()

            # Retrieve site response details
            site_response = event_parser.get_site_response()
            site_response_df = event_parser.get_site_response_dataframe()

        except Exception as e:
            logging.error(f"An error occurred: {e}")

    Parameters:
        file_path (str): The path to the JSON file containing the event information.

    Exceptions:
        Raises an Exception if the JSON file cannot be read or parsed.

    Dependencies:
        - json: Standard Python library for JSON parsing.
        - pandas: Python library for handling structured data via DataFrames.
        - logging: Standard Python library for logging execution details.



    © SHAKEmaps version 25.3.2
    """

    def __init__(self, file_path):
        """
        Initializes the ParseEventInfo instance by loading and parsing the JSON file.

        Parameters:
            file_path (str): The path to the JSON file containing the event information.

        Raises:
            Exception: If there is an error reading or parsing the file.
        """
        self.file_path = file_path
        logging.info(f"Initializing ParseEventInfo with file: {self.file_path}")
        self.data = self._load_file()
        logging.info("JSON file loaded and parsed successfully.")

    def _load_file(self):
        """
        Loads and parses the JSON file from the specified file path.

        Returns:
            dict: A dictionary representing the parsed JSON data.

        Raises:
            Exception: If there is an error opening or decoding the JSON file.
        """
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            logging.info("File read successfully.")
            return data
        except Exception as e:
            logging.error(f"Error reading file {self.file_path}: {e}")
            raise Exception(f"Error reading file {self.file_path}: {e}")

    def get_dict(self):
        """
        Retrieves the full event information data as a dictionary.

        Returns:
            dict: The entire parsed JSON data.
        """
        logging.info("Returning full data as dictionary.")
        return self.data

    def get_dataframe(self):
        """
        Converts the full JSON data into a flattened pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing the flattened JSON structure.
            
        Raises:
            Exception: If there is an error during DataFrame creation.
        """
        logging.info("Converting full data to DataFrame...")
        try:
            df = pd.json_normalize(self.data)
            logging.info("DataFrame conversion successful.")
            return df
        except Exception as e:
            logging.error(f"Error converting JSON to DataFrame: {e}")
            raise Exception(f"Error converting JSON to DataFrame: {e}")

    # -------------------- GMPE Section --------------------
    
    def get_gmpe_summary(self):
        """
        Parses the GMPE (Ground Motion Prediction Equations) section from the 'multigmpe' key and provides a summary.

        The summary contains per-period details including:
            - Overall GMPE group name and associated weights.
            - Details of each individual GMPE selection (name, list of GMPES, and weights).

        Returns:
            dict: A summary dictionary keyed by period (e.g., 'SA(0.3)', 'PGA') with detailed GMPE information.
        """
        logging.info("Parsing GMPE summary information...")
        gmpe_data = self.data.get("multigmpe", {})
        summary = {}
        for period, details in gmpe_data.items():
            summary[period] = {
                "Overall Name": details.get("name"),
                "Overall Weights": details.get("weights"),
                "GMPES": []
            }
            for gmpe_item in details.get("gmpes", []):
                summary[period]["GMPES"].append({
                    "Name": gmpe_item.get("name"),
                    "GMPES List": gmpe_item.get("gmpes"),
                    "Weights": gmpe_item.get("weights")
                })
        logging.info("GMPE summary parsed successfully.")
        return summary

    def get_gmpe_dataframe(self):
        """
        Creates a structured pandas DataFrame summarizing the GMPE selection information.

        Each row in the DataFrame represents a GMPE selection for a specific period. The DataFrame includes:
            - Period (e.g., 'SA(0.3)', 'PGA').
            - Overall GMPE name and weights.
            - Individual GMPE selection details (name, comma-separated list of GMPES, and weights).

        Returns:
            pandas.DataFrame: A DataFrame containing the GMPE summary details.
        """
        logging.info("Converting GMPE summary to DataFrame...")
        gmpe_summary = self.get_gmpe_summary()
        records = []
        for period, details in gmpe_summary.items():
            overall_name = details.get("Overall Name")
            overall_weights = details.get("Overall Weights")
            for gmpe in details.get("GMPES", []):
                records.append({
                    "Period": period,
                    "Overall GMPE Name": overall_name,
                    "Overall Weights": overall_weights,
                    "Selected GMPE Name": gmpe.get("Name"),
                    "GMPES List": ", ".join(gmpe.get("GMPES List", [])),
                    "Selected Weights": gmpe.get("Weights")
                })
        df = pd.DataFrame(records)
        logging.info("GMPE DataFrame created successfully.")
        return df

    # -------------------- Event Information --------------------
    
    def get_event_info(self):
        """
        Retrieves the event information from the 'input' -> 'event_information' section.

        Returns:
            dict: A dictionary containing event details such as depth, event_id, latitude, longitude,
                  magnitude, origin time, and other related information.
        """
        logging.info("Retrieving event information...")
        event_info = self.data.get("input", {}).get("event_information", {})
        return event_info

    def get_event_info_dataframe(self):
        """
        Converts the event information into a pandas DataFrame.

        The resulting DataFrame typically contains a single row with the event attributes.

        Returns:
            pandas.DataFrame: A DataFrame representation of the event information.
        """
        logging.info("Converting event information to DataFrame...")
        event_info = self.get_event_info()
        df = pd.DataFrame([event_info])
        return df

    # -------------------- STREC Section --------------------
    
    def get_strec(self):
        """
        Retrieves the seismic record (strec) section which contains additional event-related parameters.

        Returns:
            dict: The strec section of the event data.
        """
        logging.info("Retrieving STREC information...")
        return self.data.get("strec", {})

    def get_strec_dataframe(self):
        """
        Converts the STREC section into a pandas DataFrame.

        Returns:
            pandas.DataFrame: A one-row DataFrame representing the strec parameters.
        """
        logging.info("Converting STREC information to DataFrame...")
        strec = self.get_strec()
        df = pd.DataFrame([strec])
        return df

    # -------------------- Output Section --------------------
    
    def get_output(self):
        """
        Retrieves the output section containing ground motion details, map information, and uncertainty measures.

        Returns:
            dict: The output section of the event data.
        """
        logging.info("Retrieving output section information...")
        return self.data.get("output", {})

    def get_output_dataframe(self):
        """
        Converts the output section into a flattened pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing the flattened output details.
        """
        logging.info("Converting output section to DataFrame...")
        output = self.get_output()
        try:
            df = pd.json_normalize(output)
            logging.info("Output DataFrame created successfully.")
            return df
        except Exception as e:
            logging.error(f"Error converting output section to DataFrame: {e}")
            raise Exception(f"Error converting output section to DataFrame: {e}")

    # -------------------- Processing Section --------------------
    
    def get_processing(self):
        """
        Retrieves the processing section detailing the modules and flags used for event data generation.

        Returns:
            dict: The processing section information.
        """
        logging.info("Retrieving processing information...")
        return self.data.get("processing", {})

    def get_processing_dataframe(self):
        """
        Converts the processing section into a pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame representation of the processing details.
        """
        logging.info("Converting processing section to DataFrame...")
        processing = self.get_processing()
        try:
            df = pd.json_normalize(processing)
            logging.info("Processing DataFrame created successfully.")
            return df
        except Exception as e:
            logging.error(f"Error converting processing section to DataFrame: {e}")
            raise Exception(f"Error converting processing section to DataFrame: {e}")

    # -------------------- Site Response Section --------------------
    
    def get_site_response(self):
        """
        Retrieves the site response section which includes details like vs30default and site corrections.

        Returns:
            dict: The site response details as a dictionary.
        """
        logging.info("Retrieving site response information...")
        return self.data.get("site_response", {})

    def get_site_response_dataframe(self):
        """
        Converts the site response information into a pandas DataFrame.

        Returns:
            pandas.DataFrame: A one-row DataFrame representing site response details.
        """
        logging.info("Converting site response section to DataFrame...")
        site_response = self.get_site_response()
        df = pd.DataFrame([site_response])
        return df


    def print_doc(self):
        """Prints the class docstring."""
        print(self.__doc__)


class USGSParser:
    """
    USGSParser: Unified Facade for Parsing a Variety of USGS Data Formats
    =====================================================================

    Overview
    --------
    The USGSParser class is designed as a high-level interface that consolidates multiple specialized
    parser classes for different types of USGS data into one unified module. This facade allows users
    to access various data sources—such as instrumented data, DYFI (Did You Feel It?) data, event details,
    rupture information, PAGER (Prompt Assessment of Global Earthquakes for Response) reports, shakemap
    data, model configurations, and event information—through a common and simplified interface.

    Features
    --------
    - **Unified Interface:** Instead of interfacing with multiple parser classes individually, users can
      instantiate a single USGSParser object by specifying the parser type (e.g., 'shakemap_xml', 'pager_xml').
      All subsequent method calls are delegated automatically to the appropriate underlying parser.
    - **Flexible Data Handling:** Supports a variety of USGS data types and formats, including XML, JSON,
      and raster data. Each underlying parser is responsible for extracting and processing its specific format.
    - **Seamless Method Delegation:** The __getattr__ method ensures that any attribute or method call not
      explicitly defined in USGSParser will be transparently passed to the selected parser, providing a smooth
      user experience.
    - **Error Handling:** The class validates the parser type on initialization and, if an unsupported type is 
      provided, raises a ValueError that includes a list of available parser types.
    
    Installation & Requirements
    -----------------------------
    - Requires Python 3.6 or higher.
    - Dependent libraries include: numpy, pandas, xml.etree.ElementTree, and matplotlib (if plotting is used).
    - Ensure that all underlying parser classes (e.g., ParseInstrumentsData, ParseDYFIDataXML, ParseEventDataXML,
      etc.) are available in your module or package.

    Usage
    -----
    Instantiate the USGSParser with a specific parser type by providing the required keyword arguments for
    that parser. Once initialized, you can call methods available on the underlying parser directly through
    the USGSParser instance.

    Example:
        # Parsing a USGS Shakemap XML file.
        parser = USGSParser('shakemap_xml', xml_file='path/to/shakemap.xml')
        df = parser.get_dataframe()          # Retrieves a DataFrame of shakemap data.
        metadata = parser.get_metadata()       # Retrieves event metadata.

        # Parsing a PAGER XML file.
        pager_parser = USGSParser('pager_xml', xml_file='path/to/pager.xml')
        pager_data = pager_parser.get_dict()   # Extracts and returns PAGER data as a dictionary.

    Parameters
    ----------
    parser_type : str
        Identifier for the type of parser to initialize. Available options include:
            - 'instrumented_data'
            - 'dyfi_data_xml'
            - 'dyfi_data'
            - 'event_xml'
            - 'rupture_json'
            - 'pager_xml'
            - 'model_config'
            - 'shakemap_raster'
            - 'shakemap_xml_view'
            - 'shakemap_xml'
            - 'event_info'
    **kwargs
        Additional keyword arguments required for initializing the specific underlying parser.

    Error Handling
    --------------
    - If an unsupported parser_type is provided, the constructor raises a ValueError with a message listing
      all valid parser types.
    - Any exceptions raised by the underlying parser during initialization or parsing are propagated to the user,
      ensuring that errors in data extraction are not silently ignored.

    Advanced Integration
    --------------------
    The USGSParser uses Python's standard delegation mechanism via its __getattr__ method. This means that
    any attribute or method call that is not found on the USGSParser itself is automatically forwarded to the
    underlying parser instance. This design enables:
      - A seamless extension of functionality as new parsers are added.
      - Consistent usage patterns regardless of the specific data type being parsed.
    
    Additional Information
    ----------------------
    USGSParser is part of a larger module aimed at centralizing the processing of USGS data. By serving as a
    single point of contact for a diverse range of data formats, it simplifies integration into applications such
    as seismic analysis tools, emergency response systems, and geospatial data visualization platforms.

    
    © SHAKEmaps version 25.3.2
    """

    
    def __init__(self, parser_type, **kwargs):
        parser_classes = {
            'instrumented_data': ParseInstrumentsData,
            'dyfi_data_xml': ParseDYFIDataXML,
            'dyfi_data': ParseDYFIData,
            'event_xml': ParseEventDataXML,
            'rupture_json': ParseRuptureDataJson,
            'pager_xml': ParsePagerDataXML,
            'model_config': ParseModelConfig,
            'shakemap_raster': ShakemapParserFLT,
            'shakemap_xml_view': ShakemapXMLViewer,
            'shakemap_xml': ShakemapXMLParser,
            'event_info': ParseEventInfo
        }
        if parser_type not in parser_classes:
            available = ", ".join(parser_classes.keys())
            raise ValueError(f"Unsupported parser type: {parser_type}. Available parser types are: {available}.")
        self.parser = parser_classes[parser_type](**kwargs)

    def __getattr__(self, name):
        """
        Delegate method calls to the underlying parser instance.

        Any attribute or method not found on the USGSParser is automatically passed through
        to the internal parser instance. This allows seamless access to all methods of the
        selected parser without needing to explicitly wrap them.
        """
        return getattr(self.parser, name)

    def print_doc(self):
        """
        Prints the full documentation for the USGSParser class, including usage examples and details
        of the available underlying parser types.

        This method also attempts to print the documentation of the currently selected underlying parser,
        providing a comprehensive reference for users who need more information about specific data parsers.
        """
        print(self.__doc__)
        print("\nUnderlying Parser Documentation:")
        try:
            print(self.parser.__doc__)
        except Exception:
            print("No additional documentation available for the underlying parser.")
