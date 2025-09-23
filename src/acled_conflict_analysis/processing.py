import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import h3
import numpy as np

def convert_to_h3_grid(gdf, resolution=7, sampling_factor=200):
    """
    Convert a GeoDataFrame to H3 hexagonal grid cells with improved coverage.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Input spatial data
    resolution : int
        H3 resolution (0-15, where higher numbers mean smaller cells)
        Resolution 7 has cells ~5km across
    sampling_factor : int
        Number of points to sample in each dimension (higher = more complete coverage)
        
    Returns:
    --------
    GeoDataFrame with H3 hexagons
    """
    # Get the bounds of the input GeoDataFrame
    bounds = gdf.total_bounds
    
    # Create a finer grid of points covering the area
    x_range = np.arange(bounds[0], bounds[2], (bounds[2] - bounds[0]) / sampling_factor)
    y_range = np.arange(bounds[1], bounds[3], (bounds[3] - bounds[1]) / sampling_factor)
    
    # Create H3 indices for the grid of points
    h3_indices = set()
    
    # First, get H3 cells for each vertex of each polygon
    if hasattr(gdf, 'geometry') and gdf.geometry is not None:
        for geom in gdf.geometry:
            if geom.geom_type == 'Polygon':
                for x, y in geom.exterior.coords:
                    h3_indices.add(h3.latlng_to_cell(y, x, resolution))
                for interior in geom.interiors:
                    for x, y in interior.coords:
                        h3_indices.add(h3.latlng_to_cell(y, x, resolution))
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    for x, y in poly.exterior.coords:
                        h3_indices.add(h3.latlng_to_cell(y, x, resolution))
                    for interior in poly.interiors:
                        for x, y in interior.coords:
                            h3_indices.add(h3.latlng_to_cell(y, x, resolution))
    
    # Then sample points throughout the area
    for x in x_range:
        for y in y_range:
            # Check if the point is within any polygon in the GeoDataFrame
            point = Point(x, y)
            if any(geom.contains(point) for geom in gdf.geometry):
                # Get the H3 index for this point
                h3_index = h3.latlng_to_cell(y, x, resolution)
                h3_indices.add(h3_index)
    
    # Fill in gaps using k-ring neighbors
    # Create a copy to iterate over while modifying the original set
    initial_indices = list(h3_indices)
    for h3_index in initial_indices:
        # Add immediate neighbors (k=1)
        neighbors = h3.grid_disk(h3_index, k=1)
        for neighbor in neighbors:
            # For each neighbor, check if it's mostly within the polygon
            boundary = h3.cell_to_boundary(neighbor)
            # Convert to shapely polygon
            hex_poly = Polygon([(lng, lat) for lat, lng in boundary])
            # Check if hexagon center is within any polygon in the gdf
            center = h3.cell_to_latlng(neighbor)
            center_point = Point(center[1], center[0])  # lng, lat
            if any(geom.contains(center_point) for geom in gdf.geometry):
                h3_indices.add(neighbor)
    
    # Create hexagons for each H3 index
    hexagons = []
    for h3_index in h3_indices:
        # Get the boundary of the hexagon
        boundaries = h3.cell_to_boundary(h3_index)
        # Convert to shapely polygon (note: h3 returns [lat, lng] but Shapely expects [lng, lat])
        polygon = Polygon([(lng, lat) for lat, lng in boundaries])
        hexagons.append({'h3_index': h3_index, 'geometry': polygon})
    
    # Create a GeoDataFrame with the hexagons
    h3_gdf = gpd.GeoDataFrame(hexagons, crs=gdf.crs)
    
    return h3_gdf

def data_type_conversion(data):
    """
    Convert specific columns in the dataset to appropriate data types.
    
    This function converts latitude and longitude to float64, fatalities to integer,
    and event_date to datetime format. This ensures proper data typing for
    geospatial analysis and time-based operations.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame containing ACLED conflict data with columns for latitude,
        longitude, fatalities, and event_date.
        
    Returns:
    --------
    None
        The function modifies the DataFrame in place.
    """
    data["latitude"] = data["latitude"].astype("float64")
    data["longitude"] = data["longitude"].astype("float64")
    data["fatalities"] = data["fatalities"].astype("int")
    data['event_date'] = pd.to_datetime(data['event_date'])


def convert_to_gdf(df):
    """
    Convert a pandas DataFrame with latitude and longitude columns to a GeoDataFrame.
    
    This function creates Point geometries from latitude and longitude columns
    and returns a GeoDataFrame with the WGS84 coordinate reference system (EPSG:4326).
    If the DataFrame already has a geometry column, it will use that instead.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing latitude and longitude columns.
        
    Returns:
    --------
    geopandas.GeoDataFrame
        A GeoDataFrame with Point geometries created from latitude and longitude
        columns, using the WGS84 (EPSG:4326) coordinate reference system.
        
    Notes:
    ------
    There appears to be a potential bug in the original code where 'geometry' is 
    referenced but not defined in the else branch. The corrected version would 
    likely need to extract geometry from the existing 'geometry' column.
    """
    if 'geometry' not in df.columns:
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)
    else:
        gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=geometry)
        # Note: The original code has a bug here as 'geometry' is used but not defined
        # in this branch. It should probably be df['geometry'] instead.

    return gdf

def get_acled_by_group(
        data,
        columns = ['latitude', 'longitude'],
        freq=None,
        date_column='event_date'
):
    """
    Group ACLED conflict data by specified columns and optionally by time frequency.
    
    This function aggregates conflict data by specified geographic columns and
    optionally by time frequency, calculating the sum of fatalities and count of events
    for each group.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame containing ACLED conflict data.
    columns : list, default=['latitude', 'longitude']
        The columns to group by, typically geographic identifiers.
    freq : str, optional
        Frequency string for time-based grouping (e.g., 'MS' for month start,
        'W' for weekly). If None, no time-based grouping is performed.
    date_column : str, default='event_date'
        The column name containing datetime information for time-based grouping.
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame grouped by the specified columns (and time frequency if provided),
        containing the sum of fatalities ('nrFatalities') and count of events ('nrEvents')
        for each group.
        
    Notes:
    ------
    Common freq values: 'D' (daily), 'W' (weekly), 'MS' (month start), 
    'QS' (quarter start), 'YS' (year start)
    """
    conflict_grouped = data
    if freq:
        conflict_grouped = (
        conflict_grouped.groupby([pd.Grouper(key="event_date", freq=freq)]+columns)["fatalities"]
        .agg(["sum", "count"])
        .reset_index()
    )
    else:
        conflict_grouped = (
        conflict_grouped.groupby(columns)["fatalities"]
        .agg(["sum", "count"])
        .reset_index()
    )

    conflict_grouped.rename(
        columns={"sum": "nrFatalities", "count": "nrEvents"}, inplace=True
)
    return conflict_grouped


def get_acled_by_admin(
    adm,
    acled,
    columns=["ADM4_EN", "ADM3_EN", "ADM2_EN", "ADM1_EN"],
    nearest=False,
    event_date="event_date",
    fatalities="fatalities",
    freq="MS",
):
    """
    Join ACLED conflict data with administrative boundaries and aggregate by admin levels.
    
    This function performs a spatial join between conflict data and administrative 
    boundaries, then aggregates the results by administrative levels and time frequency.
    It can use either a standard spatial join or a nearest neighbor join with a 
    maximum distance threshold.
    
    Parameters:
    -----------
    adm : geopandas.GeoDataFrame
        GeoDataFrame containing administrative boundaries with columns specified in 'columns'.
    acled : pandas.DataFrame
        DataFrame containing ACLED conflict data with latitude, longitude, event_date,
        and fatalities columns.
    columns : list, default=["ADM4_EN", "ADM3_EN", "ADM2_EN", "ADM1_EN"]
        Administrative level columns to group by, from most detailed to least detailed.
    nearest : bool, default=False
        If True, use sjoin_nearest with a maximum distance of 2000 (presumably meters).
        If False, use standard spatial join (point-in-polygon).
    event_date : str, default="event_date"
        Column name containing datetime information for time-based grouping.
    fatalities : str, default="fatalities"
        Column name containing the number of fatalities to aggregate.
    freq : str, default="MS"
        Frequency string for time-based grouping (e.g., 'MS' for month start).
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame grouped by administrative levels and time frequency,
        containing the sum of fatalities ('nrFatalities') and count of events ('nrEvents')
        for each group.
        
    Notes:
    ------
    - ADM1 typically represents the first-level administrative divisions (e.g., states/provinces)
    - ADM2 typically represents second-level divisions (e.g., counties/districts)
    - ADM3 and ADM4 represent more detailed administrative divisions
    - The function preserves the index after aggregation with reset_index()
    """
    acled_adm = convert_to_gdf(acled)
    if nearest == True:
        acled_adm = (
            adm.sjoin_nearest(acled_adm, max_distance=2000)[
                [event_date, fatalities] + columns
            ]
            .groupby([pd.Grouper(key=event_date, freq=freq)] + columns)[fatalities]
            .agg(["sum", "count"])
            .reset_index()
        )
    else:
        acled_adm = (
            adm.sjoin(acled_adm)[[event_date, fatalities] + columns]
            .groupby([pd.Grouper(key=event_date, freq=freq)] + columns)[fatalities]
            .agg(["sum", "count"])
            .reset_index()
        )
    acled_adm.rename(columns={"sum": "nrFatalities", "count": "nrEvents"}, inplace=True)

    return acled_adm.reset_index()

def calculate_conflict_index(df: pd.DataFrame, arbitrary_constant: float = 1.0) -> pd.DataFrame:
    """
    Calculates the Conflict Intensity Index for a DataFrame and adds it as a new column.

    The conflict intensity index is calculated as the geometric mean of conflict events
    and fatalities, with an adjustment to handle zero values.

    Formula:
    $$ \\text{Conflict Intensity Index} = \\sqrt{(\\text{nrEvents}) \\times (\\text{nrFatalities} + \\text{arbitrary_constant})} $$

    Where:
    - $\\text{nrEvents}$ is the number of conflict events in a given period and location.
    - $\\text{nrFatalities}$ is the number of fatalities from conflicts in the same period and location.
    - $\\text{arbitrary_constant}$ is a constant added to fatalities to ensure the index is defined
      even when fatalities are zero. This is arbitrary and is done just to account for 0 values of fatalities.

    This index provides a balanced measure that accounts for both the frequency of conflicts
    and their severity. Compared to arithmetic means, the geometric mean reduces the
    influence of extreme values in either component (conflict events + fatalities).
    Areas with both high events and high fatalities will have higher index values
    than areas with many events but few fatalities or vice versa.

    Conflict index is calculated at the location and then average is taken over time
    (across the three time periods). This is to preserve the integrity of the conflict
    index in that specific location.

    Args:
        df (pd.DataFrame): A DataFrame expected to contain 'nrEvents' and 'nrFatalities' columns.
        arbitrary_constant (float): A constant to be added to 'nrFatalities' before
                                    calculating the geometric mean. Defaults to 1.0.

    Returns:
        pd.DataFrame: The original pandas DataFrame with a new column named
                      'conflict_intensity_index' containing the calculated values.
                      Returns an empty DataFrame if 'nrEvents' or 'nrFatalities' columns are missing.
    """
    # Create a copy of the DataFrame to avoid modifying the original DataFrame directly
    df_copy = df.copy()

    # Check if the required columns exist in the DataFrame
    if 'nrEvents' not in df_copy.columns or 'nrFatalities' not in df_copy.columns:
        print("Error: DataFrame must contain 'nrEvents' and 'nrFatalities' columns.")
        return pd.DataFrame() # Return an empty DataFrame if columns are missing

    # Calculate the conflict intensity index
    df_copy['conflict_intensity_index'] = np.sqrt(
        df_copy['nrEvents'] * (df_copy['nrFatalities'] + arbitrary_constant)
    )

    return df_copy
