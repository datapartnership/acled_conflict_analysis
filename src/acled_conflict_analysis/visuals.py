"""
ACLED Conflict Analysis Visualization Library
============================================

This module provides visualization tools for analyzing conflict data from the Armed Conflict Location & Event Data 
Project (ACLED) or similar conflict datasets. It includes functions for creating bar charts, stacked bar charts,
line plots, and interactive maps to visualize conflict events, fatalities, and related metrics.

Functions:
---------
- get_bar_chart: Creates a bar chart to visualize time-series event data
- get_stacked_bar_chart: Creates a stacked bar chart for comparing categories over time
- get_line_plot: Creates a line plot for comparing trends across different regions
- get_animated_map: Creates a static map with toggleable time-period layers
- get_cumulative_animated_map: Creates an animated timeline map

Dependencies:
------------
- bokeh: For creating static visualizations (bar charts and line plots)
- folium: For creating interactive maps
- pandas: For data manipulation
"""

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Legend, Span, Label
from bokeh.layouts import column
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import EMPTY_LAYOUT
import folium
from folium.plugins import TimestampedGeoJson
from folium import FeatureGroup
import pandas as pd
import importlib.resources as pkg_resources
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Silence the EMPTY_LAYOUT warning from Bokeh
silence(EMPTY_LAYOUT, True)

# Standard color palette for visualization consistency
COLOR_PALETTE = [
    "#002244",  # Blue
    "#F05023",  # Orange
    "#2EB1C2",  # Red
    "#009CA7",  # Teal
    "#00AB51",  # Green
    "#FDB714",  # Yellow
    "#872B90",  # Purple
    "#F78D28",  # Light Orange
    "#00A996",  # Teal-Ish Green
    "#A6192E",  # Dark Red
    "#004C97",  # Navy Blue
    "#FFD100",  # Bright Yellow
    "#7A5195",  # Lavender Purple
    "#EF5675",  # Coral Red
    "#955196",  # Light Purple
    "#003F5C",  # Dark Navy
    "#FFA600",  # Bright Orange
    "#58B947",  # Lime Green
    "#8D230F",  # Brick Red
    "#FFB400",  # Gold
    "#24693D",  # Forest Green
    "#CC2525",  # Bright Red
    "#6A4C93",  # Violet
    "#1C3144",  # Dark Slate Blue
    "#C7EFCF",  # Mint Green
]


def get_bar_chart(
    dataframe,
    title,
    source,
    subtitle=None,
    measure="nrEvents",
    category=None,
    color_code=None,
    category_value=None,
    events_dict=None,
    width=86400000 * 1.5
):
    """
    Create a bar chart to visualize time-series event data.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing event data with event_date column.
    title : str
        Main title for the chart.
    source : str
        Source information to display at the bottom.
    subtitle : str, optional
        Subtitle for the chart, displayed as the plot title.
    measure : str, optional
        Column name for the measure to plot on y-axis, defaults to "nrEvents".
    category : str, optional
        Column name for filtering the data.
    color_code : str, optional
        Color for the bars.
    category_value : any, optional
        Value to filter by if category is specified.
    events_dict : dict, optional
        Dictionary of {datetime: label} for marking significant events.
    width : int, optional
        Width of bars in milliseconds, defaults to 1.5 days.
        
    Returns
    -------
    bokeh.layouts.column
        A layout containing the title, bar chart, and source information.
    """
    # Initialize the figure
    p2 = figure(x_axis_type="datetime", width=750, height=400, toolbar_location="above")
    p2.add_layout(Legend(), "right")

    # Filter data if category is provided
    if category:
        category_df = dataframe[dataframe[category] == category_value].copy()
        category_df.sort_values(by="event_date", inplace=True)
        category_source = ColumnDataSource(category_df)
    else:
        category_df = dataframe.copy()
        category_source = ColumnDataSource(dataframe)

    # Plot the bars
    p2.vbar(
        x="event_date",
        top=measure,
        width=width,
        source=category_source,
        color=color_code,
    )

    # Configure legend
    p2.legend.click_policy = "hide"
    p2.legend.location = "top_right"

    # Set the subtitle as the title of the plot if it exists
    if subtitle:
        p2.title.text = subtitle

    # Create title and subtitle text using separate figures
    title_fig = figure(title=title, toolbar_location=None, width=750, height=40)
    title_fig.title.align = "left"
    title_fig.title.text_font_size = "14pt"
    title_fig.border_fill_alpha = 0
    title_fig.outline_line_color = None

    sub_title_fig = figure(title=source, toolbar_location=None, width=750, height=80)
    sub_title_fig.title.align = "left"
    sub_title_fig.title.text_font_size = "10pt"
    sub_title_fig.title.text_font_style = "normal"
    sub_title_fig.border_fill_alpha = 0
    sub_title_fig.outline_line_color = None

    # Add event markers if provided
    if events_dict:
        used_y_positions = []
        
        for index, (event_date, label) in enumerate(events_dict.items()):
            # Add vertical line marker
            span = Span(
                location=event_date,
                dimension="height",
                line_color='#C6C6C6',
                line_width=2,
                line_dash=(4, 4)
            )
            p2.renderers.append(span)

            # Determine label position to avoid overlap
            base_y = max(category_df[measure])
            y_position = base_y

            while y_position in used_y_positions:
                y_position -= max(category_df[measure])/20

            used_y_positions.append(y_position)

            # Add event label
            event_label = Label(
                x=event_date,
                y=y_position,
                text=label,
                text_color="black",
                text_font_size="10pt",
                background_fill_color="grey",
                background_fill_alpha=0.2,
            )
            p2.add_layout(event_label)

    # Combine the title, plot, and subtitle into a single layout
    layout = column(title_fig, p2, sub_title_fig)

    return layout


def get_stacked_bar_chart(
    dataframe,
    title,
    source_text,
    subtitle=None,
    date_column="date",
    categories=None,
    colors=None,
    events_dict=None,
    category_column="event_type",
    measure="nrEvents",
    width=1000,
    height=400
):
    """
    Create a stacked bar chart for comparing categories over time.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing event data.
    title : str
        Main title for the chart.
    source_text : str
        Source information to display at the bottom.
    subtitle : str, optional
        Subtitle for the chart, displayed as the plot title.
    date_column : str, optional
        Column name for date values, defaults to "date".
    categories : list, optional
        List of category names to include in the stacked bars.
    colors : list, optional
        List of colors for each category.
    events_dict : dict, optional
        Dictionary of {datetime: label} for marking significant events.
    category_column : str, optional
        Column name for category grouping, defaults to "event_type".
    measure : str, optional
        Column name for the measure to plot, defaults to "nrEvents".
    width : int, optional
        Width of the chart in pixels, defaults to 1000.
    height : int, optional
        Height of the chart in pixels, defaults to 400.
        
    Returns
    -------
    bokeh.layouts.column
        A layout containing the title, stacked bar chart, and source information.
    """
    # Create pivot table for stacked bars
    df_pivot = dataframe.pivot_table(
        index=date_column, columns=category_column, values=measure, fill_value=0
    ).reset_index()

    # Initialize the figure
    p2 = figure(x_axis_type="datetime", width=width, height=height, toolbar_location="above")

    # Convert dataframe to ColumnDataSource
    source = ColumnDataSource(df_pivot)

    # Stack bars
    renderers = p2.vbar_stack(
        stackers=categories,
        x=date_column,
        width=86400000 * 3,  # 3 days width
        color=colors,
        source=source,
    )

    # Configure legend
    legend = Legend(
        items=[
            (category, [renderer]) for category, renderer in zip(categories, renderers)
        ],
        location=(0, -30),
    )
    p2.add_layout(legend, "right")
    p2.legend.click_policy = "hide"
    p2.legend.location = "top_right"

    if subtitle:
        p2.title.text = subtitle

    # Create title and subtitle text using separate figures
    title_fig = figure(title=title, toolbar_location=None, width=width, height=40)
    title_fig.title.align = "left"
    title_fig.title.text_font_size = "14pt"
    title_fig.border_fill_alpha = 0
    title_fig.outline_line_color = None

    sub_title_fig = figure(title=source_text, toolbar_location=None, width=width, height=80)
    sub_title_fig.title.align = "left"
    sub_title_fig.title.text_font_size = "10pt"
    sub_title_fig.title.text_font_style = "normal"
    sub_title_fig.border_fill_alpha = 0
    sub_title_fig.outline_line_color = None

    # Add event markers if provided
    if events_dict:
        used_y_positions = []
        
        for index, (event_date, label) in enumerate(events_dict.items()):
            # Add vertical line marker
            span = Span(
                location=event_date,
                dimension="height",
                line_color='#C6C6C6',
                line_width=2,
                line_dash=(4, 4)
            )
            p2.renderers.append(span)

            # Determine label position to avoid overlap
            base_y = dataframe[measure].max()
            y_position = base_y

            while y_position in used_y_positions:
                y_position -= max(dataframe[measure])/20

            used_y_positions.append(y_position)

            # Add event label
            event_label = Label(
                x=event_date,
                y=y_position,
                text=label,
                text_color="black",
                text_font_size="10pt",
                background_fill_color="grey",
                background_fill_alpha=0.2,
            )
            p2.add_layout(event_label)

    # Combine everything into a single layout
    layout = column(title_fig, p2, sub_title_fig)

    return layout


def get_line_plot(
    dataframe,
    title,
    source,
    subtitle=None,
    measure="conflictIndex",
    category="DT",
    event_date="event_date",
    events_dict=None
):
    """
    Create a line plot for comparing trends across different regions or categories.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing time-series data.
    title : str
        Main title for the chart.
    source : str
        Source information to display at the bottom.
    subtitle : str, optional
        Subtitle for the chart, displayed as the plot title.
    measure : str, optional
        Column name for the measure to plot on y-axis, defaults to "conflictIndex".
    category : str, optional
        Column name for grouping the data into different lines, defaults to "DT".
    event_date : str, optional
        Column name for date values, defaults to "event_date".
    events_dict : dict, optional
        Dictionary of {datetime: label} for marking significant events.
        
    Returns
    -------
    bokeh.layouts.column
        A layout containing the title, line plot, and source information.
    """
    # Initialize the figure
    p2 = figure(x_axis_type="datetime", width=800, height=500, toolbar_location="above")
    p2.add_layout(Legend(), "right")

    # Create a line for each category
    for id, adm2 in enumerate(dataframe[category].unique()):
        df = dataframe[dataframe[category] == adm2][
            [event_date, measure]
        ].reset_index(drop=True)
        
        p2.line(
            df[event_date],
            df[measure],
            line_width=2,
            line_color=COLOR_PALETTE[id % len(COLOR_PALETTE)],  # Cycle colors if needed
            legend_label=adm2,
        )

    # Configure legend
    p2.legend.click_policy = "hide"
    
    if subtitle is not None:
        p2.title = subtitle

    # Create title figure
    title_fig = figure(
        title=title,
        toolbar_location=None,
        width=800,
        height=40,
    )
    title_fig.title.align = "left"
    title_fig.title.text_font_size = "14pt"
    title_fig.border_fill_alpha = 0
    title_fig.outline_line_width = 0

    # Create subtitle figure
    sub_title = figure(
        title=source,
        toolbar_location=None,
        width=800,
        height=40,
    )
    sub_title.title.align = "left"
    sub_title.title.text_font_size = "10pt"
    sub_title.title.text_font_style = "normal"
    sub_title.border_fill_alpha = 0
    sub_title.outline_line_width = 0

    # Add event markers if provided
    if events_dict:
        used_y_positions = []
        
        for index, (event_date_value, label_text) in enumerate(events_dict.items()):
            # Add vertical line marker
            span = Span(
                location=event_date_value,
                dimension="height",
                line_color='#C6C6C6',
                line_width=2,
                line_dash=(4, 4)
            )
            p2.renderers.append(span)

            # Determine label position to avoid overlap
            base_y = max(dataframe[measure])
            y_position = base_y

            while y_position in used_y_positions:
                y_position -= max(dataframe[measure])/20

            used_y_positions.append(y_position)

            # Add event label
            event_label = Label(
                x=event_date,
                y=y_position,
                text=label_text,
                text_color="black",
                text_font_size="10pt",
                background_fill_color="grey",
                background_fill_alpha=0.2,
            )
            p2.add_layout(event_label)

    # Combine into a single layout
    layout = column(title_fig, p2, sub_title)

    return layout


def load_country_centroids():
    """
    Load country centroid data for map visualizations.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing country names and their geographic centroids.
    """
    # Access the file from the package using importlib.resources
    with pkg_resources.open_text('acled_conflict_analysis.data', 'countries_centroids.csv') as file:
        country_centroids = pd.read_csv(file)
    return country_centroids


def get_animated_map(
    data, 
    country='India', 
    threshold=100, 
    measure='nrFatalities', 
    animation_period='P1Y', 
    fill_color='red', 
    outline_color='black'
):
    """
    Create a Folium map with different layers for each time period.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing event data with 'latitude', 'longitude', and 'event_date' columns.
    country : str, optional
        Name of the country to center the map, defaults to 'India'.
    threshold : int, optional
        The threshold for scaling marker size, defaults to 100.
    measure : str, optional
        The measure to visualize ('nrFatalities' or 'nrEvents'), defaults to 'nrFatalities'.
    animation_period : str, optional
        Period for grouping data ('P1Y' for year or 'P1M' for month), defaults to 'P1Y'.
    fill_color : str, optional
        Color for filling the markers, defaults to 'red'.
    outline_color : str, optional
        Color for marker outlines, defaults to 'black'.
        
    Returns
    -------
    folium.Map
        Folium map with toggleable time-period layers.
    """
    # Set measure name for UI display
    if measure == 'nrFatalities':
        measure_name = 'Fatalities'
    elif measure == 'nrEvents':
        measure_name = 'Events'

    # Get country centroid for map centering
    country_centroids = load_country_centroids()
    country_centroid = list(country_centroids[country_centroids['COUNTRY'] == country][['latitude', 'longitude']].iloc[0])

    # Create the base map
    m = folium.Map(
        location=country_centroid, 
        zoom_start=5, 
        tiles="CartoDB positron", 
        attr="Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL."
    )

    # Define bubble size scaling parameters
    max_radius = 20  # Maximum size for large numbers
    min_radius = 2   # Minimum size for small numbers

    # Scaling function for bubble size
    def scale_bubble_size(value, max_radius):
        if value > threshold:
            return max_radius
        elif value > 25:  # Between 25 and threshold
            return max_radius / 2
        else:
            return min_radius

    # Group the data by selected time period
    if animation_period == 'P1Y':
        data['year'] = data['event_date'].dt.year
        grouped = data.groupby('year')
    else:
        data['month'] = data['event_date'].dt.to_period('M')
        grouped = data.groupby('month')

    # Create FeatureGroups for each time period
    time_layers = {}

    for period, group in grouped:
        feature_group = FeatureGroup(name=str(period))

        for _, row in group.iterrows():
            scaled_radius = scale_bubble_size(row[measure], max_radius)

            # Create marker for each event
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=scaled_radius,
                color=outline_color,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.6,
                popup=f"{measure_name}: {row[measure]}<br>Date: {row['event_date'].strftime('%Y-%m-%d')}"
            ).add_to(feature_group)

        # Add the feature group (layer) to the map
        feature_group.add_to(m)
        time_layers[str(period)] = feature_group
    
    # Add a LayerControl for toggling between time layers
    folium.LayerControl().add_to(m)

    return m



def create_comparative_maps(data, title, measures=None, aggregation='h3', 
                           categories=None, cmaps=None, plot_type='color',
                           boundary_gdf=None, figsize=(15, 5), color_dict=None):
    """
    Creates comparative maps based on specified measures, aggregation levels, and categories.
    
    Parameters:
    -----------
    data : GeoDataFrame
        The spatial data to plot
    title : str
        The title for the figure
    measures : str, list, or dict
        - If str: single measure to plot
        - If list: multiple measures to plot with different colors
        - If dict: {measure_name: {options}} for advanced configuration
    aggregation : str
        Type of spatial aggregation: 'h3', 'latlon', or 'admin'
    categories : list or None
        List of category values to create separate maps for (e.g., time periods)
        If None, no category separation is applied
    cmaps : str, list, or dict
        Color maps to use for different measures:
        - If str: single colormap for all measures
        - If list: list of colormaps matching measures list
        - If dict: {measure_name: colormap} for specific mapping
    plot_type : str
        'color' for choropleth maps with quartiles
        'size' for bubble maps where size indicates value
        'both' for maps with both size and color coding
    boundary_gdf : GeoDataFrame or None
        GeoDataFrame containing boundary lines to plot (e.g., country borders)
    figsize : tuple
        Figure size as (width, height)
    color_dict : dict or None
        User-defined colors for specific measures, e.g., {'nrEvents': 'Blues', 'custom_measure': 'Greens'}
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    
    # Default values
    if measures is None:
        measures = data.select_dtypes(include=np.number).columns[0]
    
    # Convert measures to standardized format (always a dictionary)
    if isinstance(measures, str):
        measures = {measures: {}}
    elif isinstance(measures, list):
        measures = {m: {} for m in measures}
    
    # Set default categories if none provided (single map)
    if categories is None:
        categories = ['All Data']
        if 'category' not in data.columns:
            data['category'] = 'All Data'
    
    # Set up default colormaps
    default_cmaps = {
        'nrEvents': 'Blues',
        'nrFatalities': 'Reds',
    }
    
    # Process user-provided color dict (overrides the defaults)
    if color_dict is not None:
        default_cmaps.update(color_dict)
    
    # Set up measure-specific options
    for measure_name in measures.keys():
        # Set default colormap if not specified
        if 'cmap' not in measures[measure_name]:
            if cmaps is not None:
                if isinstance(cmaps, str):
                    measures[measure_name]['cmap'] = cmaps
                elif isinstance(cmaps, list) and len(cmaps) == len(measures):
                    measures[measure_name]['cmap'] = cmaps[list(measures.keys()).index(measure_name)]
                elif isinstance(cmaps, dict) and measure_name in cmaps:
                    measures[measure_name]['cmap'] = cmaps[measure_name]
                else:
                    # Use default from color_dict, fall back to default_cmaps, or use 'Purples' as the fallback
                    measures[measure_name]['cmap'] = default_cmaps.get(measure_name, 'Purples')
            else:
                measures[measure_name]['cmap'] = default_cmaps.get(measure_name, 'Purples')
        
        # Set default alpha if not specified
        if 'alpha' not in measures[measure_name]:
            measures[measure_name]['alpha'] = 0.7
        
        # Set default size factor for sizes - adjusted for better scaling
        if 'size_factor' not in measures[measure_name]:
            # Default size factor depends on the plot type and data characteristics
            if plot_type == 'size' or plot_type == 'both':
                measures[measure_name]['size_factor'] = 100  # Base factor, will be scaled by data range later
            else:
                measures[measure_name]['size_factor'] = 5
    
    # Create figure and axes
    fig, axes = plt.subplots(1, len(categories), figsize=figsize)
    
    # Handle single subplot case
    if len(categories) == 1:
        axes = [axes]
    
    # Create a deep copy of the dataframe to avoid SettingWithCopyWarning
    plot_data = data.copy(deep=True)
    
    # Process each measure for quartiles if using color mapping
    if plot_type in ['color', 'both']:
        for measure_name in measures.keys():
            # Filter out NaN values for calculations
            non_nan_data = plot_data[plot_data[measure_name].notna()]
            
            if len(non_nan_data) == 0:
                print(f"Warning: No valid data for measure '{measure_name}'")
                continue
                
            try:
                # Calculate quantiles for 4 equal-sized groups
                q_values = [0, 0.25, 0.5, 0.75, 1.0]
                quantiles = non_nan_data[measure_name].quantile(q_values).tolist()
                
                # Handle the case where there are duplicate values at quantile boundaries
                unique_quantiles = []
                for q in quantiles:
                    if q not in unique_quantiles:
                        unique_quantiles.append(q)
                        
                # If all values are the same, add a small increment to create distinctions
                if len(unique_quantiles) == 1:
                    epsilon = 1e-10
                    unique_quantiles = [unique_quantiles[0] - epsilon, 
                                       unique_quantiles[0],
                                       unique_quantiles[0] + epsilon,
                                       unique_quantiles[0] + 2*epsilon,
                                       unique_quantiles[0] + 3*epsilon]
                                       
                # If we have fewer than 5 edges (needed for 4 bins), add intermediates
                while len(unique_quantiles) < 5:
                    for i in range(len(unique_quantiles)-1):
                        mid = (unique_quantiles[i] + unique_quantiles[i+1]) / 2
                        if mid not in unique_quantiles:
                            unique_quantiles.insert(i+1, mid)
                            break
                    
                # Ensure we have exactly 5 quantiles for 4 bins
                unique_quantiles = sorted(unique_quantiles[:5])
                
                # Create quartile categories using pd.cut() with our manual bins
                quartile_col = f"{measure_name}_quartile"
                quartile_categories = pd.cut(
                    plot_data[measure_name],
                    bins=unique_quantiles,
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],
                    include_lowest=True
                )
                
                # Store the bin edges for the legend
                measures[measure_name]['bin_edges'] = unique_quantiles
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Issue computing quartiles for {measure_name}: {e}")
                # Fallback to hardcoded bins if we still have issues
                min_val = non_nan_data[measure_name].min()
                max_val = non_nan_data[measure_name].max()
                step = (max_val - min_val) / 4
                
                # Create evenly spaced bins
                bin_edges = [min_val + i * step for i in range(5)]
                if bin_edges[0] == bin_edges[1]:  # Avoid duplicate lower bound
                    bin_edges[0] -= 0.0001
                    
                # Create quartile categories
                quartile_col = f"{measure_name}_quartile"
                quartile_categories = pd.cut(
                    plot_data[measure_name],
                    bins=bin_edges,
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],
                    include_lowest=True
                )
                
                # Store bin edges
                measures[measure_name]['bin_edges'] = bin_edges
            
            # Convert to string to avoid dtype incompatibility
            plot_data[quartile_col] = quartile_categories.astype(str)
    
    # Find global min and max for each measure for consistent sizing/coloring
    for measure_name in measures.keys():
        non_nan_data = plot_data[plot_data[measure_name].notna()]
        
        if len(non_nan_data) > 0:
            min_val = non_nan_data[measure_name].min()
            max_val = non_nan_data[measure_name].max()
            data_range = max_val - min_val
            
            measures[measure_name]['vmin'] = min_val
            measures[measure_name]['vmax'] = max_val
            
            # Adjust size_factor based on data range
            if plot_type in ['size', 'both']:
                # Scale size factor based on the data range to ensure consistent visuals
                # For smaller ranges, use larger factors; for larger ranges, use smaller factors
                if data_range > 0:
                    base_size = measures[measure_name]['size_factor']
                    
                    if aggregation == 'h3':
                        # H3 cells are polygons, so we use a different sizing approach
                        measures[measure_name]['size_factor'] = base_size / (10 * np.log10(data_range + 1))
                    else:
                        # For point data, scale inversely with the data range
                        measures[measure_name]['size_factor'] = base_size / np.sqrt(data_range)
                else:
                    # If all values are the same, use a fixed size
                    measures[measure_name]['size_factor'] = 10
        else:
            print(f"Warning: No valid data for measure '{measure_name}'")
            measures[measure_name]['vmin'] = 0
            measures[measure_name]['vmax'] = 1
    
    # Plot each category
    for idx, category in enumerate(categories):
        ax = axes[idx]
        
        # Plot boundary if provided
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax, color='lightgrey', alpha=0.5, linewidth=1)
        
        # Filter data for this category
        if category != 'All Data' or 'category' in plot_data.columns:
            category_data = plot_data[plot_data['category'] == category]
        else:
            category_data = plot_data
        
        # Plot each measure according to the specified plot type
        for i, (measure_name, measure_opts) in enumerate(measures.items()):
            vmin = measure_opts['vmin']
            vmax = measure_opts['vmax']
            cmap = measure_opts['cmap']
            alpha = measure_opts['alpha']
            size_factor = measure_opts['size_factor']
            
            # Skip if no data (prevents empty plots)
            if len(category_data) == 0 or not category_data[measure_name].notna().any():
                continue
            
            if plot_type == 'color' or (plot_type == 'both' and aggregation == 'h3'):
                # Plot with choropleth colors based on quartiles
                quartile_col = f"{measure_name}_quartile"
                category_data.plot(
                    ax=ax,
                    column=quartile_col,
                    categorical=True,
                    cmap=cmap,
                    alpha=alpha,
                    legend=False
                )
                
            elif plot_type == 'size' or (plot_type == 'both' and aggregation in ['latlon', 'admin']):
                # Calculate marker sizes with a better scaling approach
                sizes = np.sqrt((category_data[measure_name] - vmin) / (vmax - vmin + 1e-10) + 0.1) * size_factor
                
                # Plot with bubble sizes
                category_data.plot(
                    ax=ax,
                    color=plt.cm.get_cmap(cmap)(0.6),  # Use a fixed color from the colormap
                    alpha=alpha,
                    markersize=sizes * 20  # Scale for better visibility
                )
                
            elif plot_type == 'both':
                # Both size and color for a single measure
                # Create a normalized color mapping based on values
                norm = Normalize(vmin=vmin, vmax=vmax)
                
                # For combined plots, we need to handle sizing better
                if hasattr(category_data.geometry.iloc[0], 'centroid'):
                    # For polygon data (like H3 cells), we'll plot centroids with sized markers
                    for _, row in category_data.iterrows():
                        if pd.notna(row[measure_name]):
                            value = row[measure_name]
                            normalized_value = (value - vmin) / (vmax - vmin + 1e-10)
                            color = plt.cm.get_cmap(cmap)(normalized_value)
                            
                            # Get the centroid for the marker placement
                            centroid = row.geometry.centroid
                            
                            # Scale marker size based on normalized value
                            marker_size = np.sqrt(normalized_value + 0.1) * size_factor * 20
                            
                            ax.plot(
                                centroid.x, centroid.y,
                                'o',
                                color=color,
                                alpha=alpha,
                                markersize=marker_size
                            )
                else:
                    # For point data, use direct plotting with sized markers
                    for _, row in category_data.iterrows():
                        if pd.notna(row[measure_name]):
                            value = row[measure_name]
                            normalized_value = (value - vmin) / (vmax - vmin + 1e-10)
                            color = plt.cm.get_cmap(cmap)(normalized_value)
                            
                            # Scale marker size based on normalized value
                            marker_size = np.sqrt(normalized_value + 0.1) * size_factor * 20
                            
                            ax.plot(
                                row.geometry.x, row.geometry.y,
                                'o',
                                color=color,
                                alpha=alpha,
                                markersize=marker_size
                            )
        
        # Add legend on the last subplot
        if idx == len(categories) - 1:
            legend_items = []
            
            for measure_name, measure_opts in measures.items():
                if plot_type == 'color' or (plot_type == 'both' and aggregation == 'h3'):
                    # Create color legend for quartiles
                    cmap = plt.cm.get_cmap(measure_opts['cmap'])
                    colors = cmap(np.linspace(0.2, 0.8, 4))
                    
                    # Check if bin_edges exists
                    if 'bin_edges' in measure_opts:
                        bin_edges = measure_opts['bin_edges']
                        
                        for i in range(4):
                            if i < len(bin_edges) - 1:  # Ensure we have enough bin edges
                                legend_items.append(
                                    Patch(
                                        facecolor=colors[i],
                                        edgecolor='none',
                                        alpha=measure_opts['alpha'],
                                        label=f"{measure_name} Q{i+1} ({bin_edges[i]:.2f}-{bin_edges[i+1]:.2f})"
                                    )
                                )
                
                if plot_type == 'size' or (plot_type == 'both' and aggregation != 'h3'):
                    # Create size legend with better size representation
                    vmin = measure_opts['vmin']
                    vmax = measure_opts['vmax']
                    size_factor = measure_opts['size_factor']
                    
                    # Create evenly spaced values between min and max
                    size_values = np.linspace(vmin, vmax, 4)
                    size_labels = [f"{measure_name}: {val:.2f}" for val in size_values]
                    
                    # Calculate marker sizes consistently with the plotting approach
                    marker_sizes = [np.sqrt((val - vmin) / (vmax - vmin + 1e-10) + 0.1) * size_factor * 20 
                                   for val in size_values]
                    
                    for ms, label in zip(marker_sizes, size_labels):
                        legend_items.append(
                            Line2D(
                                [0], [0],
                                marker='o',
                                color='w',
                                markerfacecolor=plt.cm.get_cmap(measure_opts['cmap'])(0.6),
                                markersize=ms/2,  # Adjust for legend display
                                alpha=measure_opts['alpha'],
                                label=label
                            )
                        )
            
            # Add legend with appropriate positioning
            if legend_items:
                legend_cols = min(4, len(legend_items))
                if len(legend_items) <= 4:
                    legend = ax.legend(
                        handles=legend_items,
                        loc='lower right',
                        frameon=False,
                        ncol=legend_cols
                    )
                else:
                    # For many legend items, place at bottom of figure
                    legend = fig.legend(
                        handles=legend_items,
                        loc='lower center',
                        frameon=False,
                        ncol=legend_cols,
                        bbox_to_anchor=(0.5, 0.01)
                    )
        
        # Set title and clean up axes
        ax.set_title(category)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Set main title
    plt.suptitle(title, fontsize=16)
    
    # Adjust layout
    if legend_items and len(legend_items) > 4:
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for legend and title
    else:
        plt.tight_layout()
    
    return fig, axes