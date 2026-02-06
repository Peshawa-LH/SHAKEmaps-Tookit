import matplotlib as mpl
import matplotlib.pyplot as plt


import re
import utm
import pandas as pd

def lonlat_to_dyfi(
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

    # 3) drop the intermediate UTM columns
    df = df.drop(columns=["zone_number","zone_letter","easting","northing"])

    return df



def contour_scale(pgm_type, scale_type="usgs", units=None):
    """
    Generate a color scale for seismic intensity maps based on the given scale type.

    Parameters:
    -----------
    pgm_type : str
        The type of ground motion parameter (PGM), either "PGA", "PGV", "SA_1", or "MMI" (for USGS),
        or "PGA", "PGV", or "EMS" (for EMS). The function accepts both uppercase and lowercase inputs.
    scale_type : str, optional
        The type of scale to use, either "usgs" or "ems". Default is "usgs".
    units : str, optional
        The units for the ground motion parameter. Default is "%g" for USGS and "cm/s^2" for EMS when "PGA" is selected.
        For "PGV", the default is "cm/s" for both scales. For "MMI" or "EMS", the default unit is "MMI" or "EMS", respectively.

    Returns:
    --------
    cmap : ListedColormap
        The colormap for the given scale type.
    bounds : list of float
        The boundaries of the intensity values for the color scale.
    ticks : list of float
        The tick values for the color bar.
    norm : BoundaryNorm
        The normalization for the colormap.
    used_scale : str
        The label for the color bar.
    
    Raises:
    -------
    ValueError
        If an invalid combination of PGM type and units is provided.
    
    Examples:
    ---------
    >>> contour_scale("PGA", "usgs", "%g")
    >>> contour_scale("PGV", "ems", "cm/s")

    """
    pgm_type = pgm_type.upper()
    scale_type = scale_type.lower()
    
    if scale_type not in ["usgs", "ems"]:
        raise ValueError("Invalid scale type. Choose 'usgs' or 'ems'.")

    # Define the color maps for each scale
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
    ems_colors = [
        (1, 1, 1, 0),
        (237/255, 239/255, 243/255, 1.0),              # whitesmoke
        (172/255, 180/255, 206/255, 1.0),              # mediumslateblue
        (161/255, 215/255, 227/255, 1.0),              # lightskyblue
        (143/255, 200/255, 145/255, 1.0),              # seagreen
        (249/255, 236/255, 51/255, 1.0),               # yellow
        (238/255, 181/255, 9/255, 1.0),                # gold
        (233/255, 135/255, 45/255, 1.0),               # orange
        (223/255, 83/255, 42/255, 1.0),                # darkorange
        (217/255, 38/255, 42/255, 1.0),                # red
        (136/255, 0/255, 0/255, 1.0),                  # darkred
        (68/255, 0/255, 1/255, 1.0)                    # verydarkred
    ]
    colors = usgs_colors if scale_type == "usgs" else ems_colors
    cmap = mpl.colors.ListedColormap(colors)

    # Define the intensity bounds for each scale
    usgs_table = {
        "pga_values_%g": [0, 0.05, 0.3, 2.8, 6.2, 11.5, 21.5, 40.1, 74.7, 139],
        "pga_values_g": [0, 0.001, 0.003, 0.028, 0.062, 0.115, 0.215, 0.401, 0.747, 1.39],
        "pga_values_cm/s2": [0, 0.5, 2.9, 27.5, 60.8, 112.8, 210.9, 393.4, 732.8, 1363.6],
        "pgv_values_cm/s": [0, 0.02, 0.1, 1.4, 4.7, 9.6, 20, 41, 86, 178],
        "sa_1_values_%g": [0, 0.02, 0.1, 1, 4.6, 10, 23, 50, 110, 244],
        "sa_1_values_g": [0, 0.0002, 0.001, 0.01, 0.046, 0.1, 0.23, 0.5, 1.1, 2.44],
        "sa_1_values_cm/s^2": [0, 0.2, 1, 9.8, 45.1, 98.1, 225.6, 490.5, 1079.1, 2393.6],
        "intensity_values_mmi": [0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10]
    }
    ems_table = {
        "pga_values_%g": [0, 0.1, 0.31, 0.51, 1.02, 2.05, 5.62, 15.42, 42.34, 116.24, 319.11, 876.08, 2405.13],
        "pga_values_g": [0, 0.001, 0.0031, 0.0051, 0.0102, 0.0205, 0.0562, 0.1542, 0.4234, 1.1624, 3.1911, 8.7608, 24.0513],
        "pga_values_cm/s^2": [0, 1, 3, 5, 10, 20.07, 55.11, 151.29, 415.36, 1140.3, 3130.5, 8594.3, 23594.3],
        "pgv_values_cm/s": [0, 1, 3, 5, 8, 13, 25, 56.64, 234.62, 971.97, 4026.6, 16681.01, 69104.48],
        "intensity_values_ems": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }
    table = usgs_table if scale_type == "usgs" else ems_table

    # Set default units
    usgs_default_units = {
        "PGA": "%g",
        "PGV": "cm/s",
        "SA_1": "%g",
        "MMI": "MMI"
    }
    ems_default_units = {
        "PGA": "cm/s^2",
        "PGV": "cm/s",
        "EMS": "EMS"
    }
    default_units = usgs_default_units if scale_type == "usgs" else ems_default_units

    # Set labels and scale labels
    labels = {
        "%g": "%g",
        "g": "g",
        "cm/s^2": "${cm/s}^2$",
        "cm/s": "cm/s",
        "MMI": "MMI",
        "EMS": "EMS"
    }
    usgs_scale_labels = {
        "PGA": "Peak Ground Acceleration",
        "PGV": "Peak Ground Velocity",
        "SA_1": "Spectral Acceleration $Sa_{1s}$",
        "MMI": "Modified Mercalli Intensity Scale"
    }
    ems_scale_labels = {
        "PGA": "Peak Ground Acceleration",
        "PGV": "Peak Ground Velocity",
        "EMS": "European Macroseismic Scale"
    }
    scale_labels = usgs_scale_labels if scale_type == "usgs" else ems_scale_labels

    # Determine the correct units to use
    if units is None:
        units = default_units.get(pgm_type, "%g")

    if scale_type == "ems" and pgm_type == "EMS":
        key = "intensity_values_ems"
    elif scale_type == "usgs" and pgm_type == "MMI":
        key = "intensity_values_mmi"
    else:
        key = f"{pgm_type.lower()}_values_{units.lower()}"

    if key not in table:
        raise ValueError(f"Invalid combination of PGM type '{pgm_type}' and units '{units}'")

    bounds = table[key]
    ticks = bounds
    used_scale = f'{scale_labels[pgm_type]} ({labels[units]})'

    # Create color bar
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(bottom=0.5)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                   norm=norm,
                                   boundaries=[0] + bounds + [bounds[-1] + 1],
                                   ticks=ticks,
                                   spacing='uniform',
                                   orientation='horizontal')
    cb.set_label(used_scale)


    plt.close()

    return cmap, bounds, ticks, norm, used_scale
    
    
    
    
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







class SiteAmplificationConverter:
    """
    SiteAmplificationConverter provides methods to convert ground‐motion measures
    (PGA, PGV) or macroseismic intensity (MMI) between different site conditions
    characterized by Vs30, using either USGS ShakeMap (Borcherdt‐style)
    amplification or EMS‑98 macroseismic amplification rules.

    1) USGS ShakeMap site amplification:
       - ShakeMap predicts ground motions on a “rock” reference condition
         (NEHRP Site Class C, Vs30 ≈ 760 m/s) using GMPEs (e.g.
         Pankow & Pechmann 2004; Spudich et al. 1999).
       - Site correction factors F_soil are derived from Borcherdt (1994),
         eq. 7a (short‐period ≃ PGA) and eq. 7b (mid‐period ≃ PGV),
         with V_ref = 910 m/s. Values for Vs30 bins (150–2197 m/s) and PGA
         cutoffs (150, 250, 350 gal) are tabulated in Table 2.3 of the
         ShakeMap Technical Manual :contentReference[oaicite:0]{index=0}.
       - To convert from soil to rock:
           PGA_rock = PGA_soil / F_soil(PGA_soil, Vs30_source)
       - To convert from rock to soil:
           PGA_target = PGA_rock * F_soil(PGA_rock, Vs30_target)
       - To convert directly from soil A to soil B, multiply by F_B / F_A.

    2) EMS‑98 macroseismic amplification (ΔIm) based on Vs30:
       - EMS‑98 defines amplification in terms of intensity residuals ΔIm
         (observed minus predicted) on geological/tectonic soil classes.
       - Fäh et al. (2011) calibrated ΔIm versus Vs30 for Switzerland,
         finding a reference soil Vs30 ≃ 640 ± 58 m/s (ΔIm = 0) and a
         rock reference Vs30 = 1105 m/s requiring a constant offset
         +0.47 intensity units :contentReference[oaicite:1]{index=1}.
       - ΔIm spans –0.31 to +1.05 intensity units across classes.
       - To convert intensity between sites:
           I_target = I_source + ΔIm(Vs30_target) – ΔIm(Vs30_source)

    3) Eurocode 8 PGA amplification:
       - EC8 anchors elastic spectra to S·ag_R on “Type A” ground (rock) and
         modifies by a “soil factor” S for ground types A–E.
       - Table 3.2 (Type 1 spectrum) and Table 3.3 (Type 2 spectrum) give S for
         classes A–E :contentReference[oaicite:1]{index=1}.
       - For Vs30-based classification (EC8 § 3.1.2, Table 1), sites are approximated
         as:
            • A: Vs30 > 800 m/s
            • B: 360 ≤ Vs30 ≤ 800 m/s
            • C: 180 ≤ Vs30 < 360 m/s
            • D: Vs30 < 180 m/s
         (Class E requires additional info on shallow alluvium thickness) :contentReference[oaicite:2]{index=2}.
       - Conversion is:
           PGA_rock = PGA_source / S_source
           PGA_target = PGA_rock * S_target
    """

    # USGS and EMS‑98 tables (unchanged) …

    # Eurocode‑8 soil factors S
    _EUROCODE_S = {
        'TYPE1': {'A': 1.00, 'B': 1.20, 'C': 1.15, 'D': 1.35, 'E': 1.40},
        'TYPE2': {'A': 1.00, 'B': 1.35, 'C': 1.50, 'D': 1.80, 'E': 1.60},
    }

    @staticmethod
    def _vs30_to_eurocode_class(vs30):
        """
        Approximate EC8 ground type by Vs30:
        A: >800, B: [360–800], C: [180–360), D: <180 (E needs stratigraphy info).
        """
        if vs30 > 800:
            return 'A'
        elif vs30 >= 360:
            return 'B'
        elif vs30 >= 180:
            return 'C'
        else:
            return 'D'

    def convert(self, value, vs30_source, vs30_target,
                measure='PGA', method='USGS', eurocode_type='Type1'):
        """
        Convert a ground‐motion or intensity value from one Vs30 to another.

        Parameters
        ----------
        value : float
            PGA (gal) or PGV (cm/s) or MMI at the source site.
        vs30_source : float
            Vs30 at the source site (m/s).
        vs30_target : float
            Vs30 at the target site (m/s).
        measure : str
            'PGA', 'PGV', or 'MMI'.
        method : str
            'USGS', 'EMS-98', or 'EC8' (Eurocode 8 PGA).
        eurocode_type : str
            'Type1' or 'Type2' spectrum (only for method='EC8').

        Returns
        -------
        float
            Converted value at the target Vs30.
        """
        m = measure.upper()
        M = method.upper()

        if M == 'USGS':
            # … (unchanged) …
            F_src = self._usgs_Fsoil(m, vs30_source, value)
            rock_val = value / F_src
            F_tgt = self._usgs_Fsoil(m, vs30_target, rock_val)
            return rock_val * F_tgt

        elif M == 'EMS-98':
            # … (unchanged) …
            if m != 'MMI':
                raise ValueError("EMS-98 only supports MMI")
            d_src = self._ems98_deltaI(vs30_source)
            d_tgt = self._ems98_deltaI(vs30_target)
            return value + (d_tgt - d_src)

        elif M in ('EC8', 'EUROCODE'):
            if m != 'PGA':
                raise ValueError("Eurocode method supports only PGA")
            t = eurocode_type.upper()
            if t not in ('TYPE1', 'TYPE2'):
                raise ValueError("Choose eurocode_type='Type1' or 'Type2'")
            cls_src = self._vs30_to_eurocode_class(vs30_source)
            cls_tgt = self._vs30_to_eurocode_class(vs30_target)
            S_src = self._EUROCODE_S[t][cls_src]
            S_tgt = self._EUROCODE_S[t][cls_tgt]
            # de‐amplify to rock, then re‐amplify
            rock_val = value / S_src
            return rock_val * S_tgt

        else:
            raise ValueError("Unknown method – use 'USGS', 'EMS-98', or 'EC8'")




