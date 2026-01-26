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
- get_comparative_maps: Creates comparative maps based on specified measures and categories
- get_h3_maps: Generates H3 hexagon maps for visualizing conflict data
- bivariate_choropleth: Creates a bivariate choropleth map for visualizing two variables

Dependencies:
------------
- bokeh: For creating static visualizations (bar charts and line plots)
- folium: For creating interactive maps
- pandas: For data manipulation
"""

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Legend, Span, Label, HoverTool
from bokeh.palettes import Category10
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

    # Add HoverTool for interactivity
    hover = HoverTool(tooltips=[
        ("Date", "@event_date{%F}"),
        (measure.replace('_', ' ').title(), f"@{measure}")
    ], formatters={'@event_date': 'datetime'})
    p2.add_tools(hover)

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

    # Add HoverTool for interactivity
    hover = HoverTool(tooltips=[
        ("Date", f"@{date_column}{{%F}}"),
        ("Category", "$name"),
        ("Value", "$~{0.0}")
    ], formatters={f'@{date_column}': 'datetime'})
    p2.add_tools(hover)

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
    events_dict=None,
    plot_width=900,
    plot_height=550
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
        Subtitle for the chart, combined with the main title.
    measure : str, optional
        Column name for the measure to plot on y-axis, defaults to "conflictIndex".
    category : str, optional
        Column name for grouping the data into different lines, defaults to "DT".
    event_date : str, optional
        Column name for date values, defaults to "event_date".
    events_dict : dict, optional
        Dictionary of {datetime: label} for marking significant events.
    plot_width : int, optional
        Width of the main plot in pixels. Defaults to 900.
    plot_height : int, optional
        Height of the main plot in pixels. Defaults to 550.
        
    Returns
    -------
    bokeh.plotting.figure
        The Bokeh figure object containing the line plot, title, and source information.
    """
    # Ensure event_date column is datetime type
    dataframe[event_date] = pd.to_datetime(dataframe[event_date])

    # Determine data range for Y-axis
    valid_measure_data = dataframe[measure].dropna()
    if not valid_measure_data.empty:
        min_measure = valid_measure_data.min()
        max_measure = valid_measure_data.max()
    else:
        min_measure, max_measure = 0, 1 # Default range if no data

    # Calculate initial Y-axis padding for data
    data_y_range_padding = (max_measure - min_measure) * 0.1 if max_measure != min_measure else 0.1
    initial_y_start = min_measure - data_y_range_padding
    initial_y_end = max_measure + data_y_range_padding

    # Create the main plot figure
    p2 = figure(
        x_axis_type="datetime",
        width=plot_width,
        height=plot_height,
        toolbar_location="above",
        title_location="above",
        background_fill_color="white",
        background_fill_alpha=1.0, # Fully opaque white
        y_range=(initial_y_start, initial_y_end)
    )

    # --- Unified Title Setup ---
    full_title_text = title
    if subtitle:
        full_title_text = f"{title}\n{subtitle}"
    p2.title.text = full_title_text
    p2.title.align = "left"
    p2.title.text_font_size = "16pt"
    p2.title.text_font_style = "bold"
    p2.title.text_color = "#333333"

    # --- Axis Labels ---
    p2.xaxis.axis_label = "Date"
    p2.yaxis.axis_label = measure.replace('_', ' ').title()
    p2.xaxis.axis_label_text_font_size = "12pt"
    p2.yaxis.axis_label_text_font_size = "12pt"
    p2.xaxis.major_label_text_font_size = "10pt"
    p2.yaxis.major_label_text_font_size = "10pt"
    p2.xaxis.axis_line_color = "gray"
    p2.yaxis.axis_line_color = "gray"
    p2.xaxis.major_tick_line_color = "gray"
    p2.yaxis.major_tick_line_color = "gray"

    # --- Grid Lines (Prettification) ---
    p2.xgrid.grid_line_alpha = 0.3 # Subtle vertical grid lines
    p2.ygrid.grid_line_alpha = 0.5 # Slightly more prominent horizontal grid lines
    p2.xgrid.grid_line_dash = [6, 4]
    p2.ygrid.grid_line_dash = [6, 4]

    # Add HoverTool for interactivity
    hover = HoverTool(tooltips=[
        ("Date", "@{%s}{%%F}" % event_date),
        (measure.replace('_', ' ').title(), f"@{measure}{{0.00}}"),
        (category.replace('_', ' ').title(), f"@{category}")
    ])
    p2.add_tools(hover)

    # Get unique categories to assign colors
    unique_categories = dataframe[category].unique()
    # Use Category10 palette, ensuring enough colors by cycling if more than 10 categories
    colors = Category10[10] # Category10 has 10 distinct colors

    # Create a line for each category
    for id, adm2 in enumerate(unique_categories): # Iterate through unique categories
        df_category = dataframe[dataframe[category] == adm2].copy()
        df_category = df_category.sort_values(by=event_date).reset_index(drop=True)

        if df_category.empty:
            print(f"Warning: No data for category '{adm2}'. Skipping line plot.")
            continue

        plot_x, plot_y = df_category[event_date].values, df_category[measure].values
        
        # Create a ColumnDataSource for each line to ensure proper linking for hover and legend
        source_category = ColumnDataSource(data={
            event_date: plot_x,
            measure: plot_y,
            category: [str(adm2)] * len(plot_x) # Ensure category column is in source
        })

        p2.line(
            x=event_date,
            y=measure,
            source=source_category,
            line_width=2.5,
            line_color=colors[id % len(colors)], # Assign color from Category10 palette
            legend_label=str(adm2), # Set legend label for each line
        )

    # Configure legend
    p2.legend.click_policy = "hide"
    p2.legend.orientation = "vertical"
    p2.legend.location = "top_left"
    p2.legend.background_fill_alpha = 0.8
    p2.legend.border_line_color = None

    # Add event markers if provided
    max_label_y_position = initial_y_end
    if events_dict:
        sorted_events = sorted(events_dict.items())

        plot_y_min_current, plot_y_max_current = p2.y_range.start, p2.y_range.end
        current_data_range = max_measure - min_measure
        
        approx_label_height_data_units = (plot_y_max_current - plot_y_min_current) * 0.04 
        
        label_starting_y = max_measure + (current_data_range * 0.20)
        
        occupied_y_intervals = [] 
        
        for index, (event_date_value, label_text) in enumerate(sorted_events):
            span = Span(
                location=event_date_value,
                dimension="height",
                line_color='#888888',
                line_width=1.5,
                line_dash="dashed",
                line_alpha=0.7
            )
            p2.renderers.append(span)

            proposed_y = label_starting_y
            max_attempts = 30
            attempt_count = 0
            
            while attempt_count < max_attempts:
                label_y_start = proposed_y - (approx_label_height_data_units / 2)
                label_y_end = proposed_y + (approx_label_height_data_units / 2)
                
                overlap = False
                for occupied_start, occupied_end in occupied_y_intervals:
                    if not (label_y_end < occupied_start or label_y_start > occupied_end):
                        overlap = True
                        break
                
                if not overlap:
                    buffer_factor = 0.15 
                    occupied_y_intervals.append(
                        (label_y_start - approx_label_height_data_units * buffer_factor,
                         label_y_end + approx_label_height_data_units * buffer_factor)
                    )
                    break 
                else:
                    proposed_y -= approx_label_height_data_units * 1.8 
                    attempt_count += 1
                    
                    if proposed_y < p2.y_range.start + approx_label_height_data_units:
                        break 

            final_y_position = proposed_y 

            if final_y_position > max_label_y_position:
                max_label_y_position = final_y_position
            
            final_y_position = max(final_y_position, p2.y_range.start + approx_label_height_data_units/2)

            event_label = Label(
                x=event_date_value,
                y=final_y_position,
                x_offset=5,
                y_offset=0,
                text=label_text,
                text_color="#555555",
                text_font_size="9pt",
                background_fill_color="#eeeeee",
                background_fill_alpha=0.7,
                border_line_color="#cccccc",
                border_line_alpha=0.5,
            )
            p2.add_layout(event_label)
        
        final_top_buffer_for_labels = approx_label_height_data_units * 2 
        
        p2.y_range.end = max(initial_y_end, max_label_y_position + final_top_buffer_for_labels)

    # --- Source Information as Label Annotation ---
    # Place it at the bottom left of the plot area
    source_label_x_pos = p2.x_range.start # Aligned with left axis
    # The y-position should be just above the very bottom edge of the plot's data range
    source_label_y_pos = p2.y_range.start + (p2.y_range.end - p2.y_range.start) * 0.005 # A tiny bit from bottom

    source_label = Label(
        x=source_label_x_pos,
        y=source_label_y_pos,
        x_units='data',
        y_units='data',
        text=f"Source: {source} (Generated: {datetime.now().strftime('%Y-%m-%d')})",
        text_color="#888888",
        text_font_size="8pt",
        text_font_style="italic",
        background_fill_alpha=0,
        border_line_alpha=0,
        x_offset=5,
        y_offset=5
    )
    p2.add_layout(source_label)

    return p2



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
    from datetime import datetime

    font_choice = 'Arial'
    
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
                
        # Set default label name if not specified
        if 'label_name' not in measures[measure_name]:
            measures[measure_name]['label_name'] = measure_name
    
    # Create figure and axes with gridspec for equal distribution
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, len(categories), wspace=0.05, hspace=0.05)
    axes = []
    
    # Create axes with equal size and distribution
    for i in range(len(categories)):
        axes.append(fig.add_subplot(gs[0, i]))
        # Immediately disable all ticks and spines for each subplot
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in axes[i].spines.values():
            spine.set_visible(False)
        
    # Handle single subplot case
    if len(categories) == 1:
        axes = [axes[0]]
    
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
        
        # Ensure all spines are hidden for all plots
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Plot boundary if provided
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax, color='lightgrey', alpha=0.7, linewidth=1)
        
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
            label_name = measure_opts.get('label_name', measure_name)
            
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
                # Use a more controlled approach to sizing that works better with outliers
                # Apply a log-based transformation to handle large value ranges
                if vmin < vmax:  # Only if we have a valid range
                    # Add a small constant to handle zeros
                    epsilon = (vmax - vmin) * 0.01 if vmax > vmin else 0.1
                    
                    # Get log-transformed values
                    log_vals = np.log1p((category_data[measure_name] - vmin) + epsilon)
                    log_max = np.log1p((vmax - vmin) + epsilon)
                    
                    # Scale to a reasonable marker size range (3 to 15)
                    sizes = 3 + (log_vals / log_max) * 12
                else:
                    # Fallback for when all values are the same
                    sizes = np.ones(len(category_data)) * 5
                
                # Plot with bubble sizes
                category_data.plot(
                    ax=ax,
                    color=plt.cm.get_cmap(cmap)(0.6),  # Use a fixed color from the colormap
                    alpha=alpha,
                    markersize=sizes
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
                            
                            # Apply log-based sizing for better distribution of sizes
                            epsilon = (vmax - vmin) * 0.01 if vmax > vmin else 0.1
                            log_val = np.log1p((value - vmin) + epsilon)
                            log_max = np.log1p((vmax - vmin) + epsilon)
                            marker_size = 3 + (log_val / log_max) * 12
                            
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
                            
                            # Apply log-based sizing for better distribution of sizes
                            epsilon = (vmax - vmin) * 0.01 if vmax > vmin else 0.1
                            log_val = np.log1p((value - vmin) + epsilon)
                            log_max = np.log1p((vmax - vmin) + epsilon)
                            marker_size = 3 + (log_val / log_max) * 12
                            
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
                                        label=f"{label_name} Q{i+1} ({bin_edges[i]:.2f}-{bin_edges[i+1]:.2f})"
                                    )
                                )
                
                if plot_type == 'size' or (plot_type == 'both' and aggregation != 'h3'):
                    # Create size legend with better size representation
                    vmin = measure_opts['vmin']
                    vmax = measure_opts['vmax']
                    
                    # Create evenly spaced values between min and max for a better legend
                    size_values = np.linspace(vmin, vmax, 4)
                    size_labels = [f"{measure_name}: {val:.2f}" for val in size_values]
                    
                    # Calculate marker sizes consistently with the plotting approach
                    epsilon = (vmax - vmin) * 0.01 if vmax > vmin else 0.1
                    log_values = [np.log1p((val - vmin) + epsilon) for val in size_values]
                    log_max = np.log1p((vmax - vmin) + epsilon)
                    marker_sizes = [3 + (log_val / log_max) * 12 for log_val in log_values]
                    
                    for ms, label in zip(marker_sizes, size_labels):
                        legend_items.append(
                            Line2D(
                                [0], [0],
                                marker='o',
                                color='w',
                                markerfacecolor=plt.cm.get_cmap(measure_opts['cmap'])(0.6),
                                markersize=ms,
                                alpha=measure_opts['alpha'],
                                label=label
                            )
                        )

        # Set title and clean up axes - position the title at the top
        ax.set_title(category, y=1.0, pad=10, fontfamily=font_choice)
        
        # Ensure all ticks and spines are completely removed
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, right=False, bottom=False, top=False,
                    labelleft=False, labelright=False, labelbottom=False, labeltop=False)
        
        # Hide all spines
        for spine in ax.spines.values():
            spine.set_visible(False)
    
            
    # Add legend with appropriate positioning
    if legend_items:
        legend_cols = min(4, len(legend_items))
        # Place legend at the bottom of the figure
        legend = fig.legend(
            handles=legend_items,
            loc='lower center',
            frameon=False,
            ncol=legend_cols,
            bbox_to_anchor=(0.5, 0.12)  # Moved up slightly to make room for source text
        )
        
    # Set main title
    fig.suptitle(title, fontsize=16, y=0.95, fontfamily = font_choice, fontweight='bold')
    
    # Add source text box at the bottom
    today = datetime.now().strftime("%B %d, %Y")
    source_text = f"Source: ACLED. Accessed: {today}"
    fig.text(0.3, 0.02, source_text, ha='center', fontsize=9, fontfamily=font_choice
             )
    
    # Adjust layout to ensure equal spacing and size
    # Leave more space at bottom for legend and source text
    plt.subplots_adjust(bottom=0.25, top=0.85, wspace=0.05)
    
    return fig, axes


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# --- Helper Functions ---

def _calculate_h3_quartiles(data_gdf, measure):
    """
    Calculates tercile bin edges and assigns tercile categories to the data.
    Uses qcut for true terciles (equal number of observations in each bin).

    Parameters:
    -----------
    data_gdf : GeoDataFrame
        The full dataset to calculate terciles from.
    measure : str
        The numeric column to base terciles on.

    Returns:
    --------
    tuple: (bin_edges, plot_data_with_quartiles, norm)
        - bin_edges (list): The calculated tercile bin edges.
        - plot_data_with_quartiles (GeoDataFrame): The input GeoDataFrame
          with an added 'quartile' column and 'quartile_numeric' column.
        - norm (Normalize): Matplotlib Normalizer for consistent colormapping.
    """
    plot_data = data_gdf.copy(deep=True)
    non_nan_data = plot_data[plot_data[measure].notna()]

    # Handle cases where there is no data or not enough unique values for 3 terciles
    if non_nan_data.empty or non_nan_data[measure].nunique() < 3:
        min_val = non_nan_data[measure].min() if not non_nan_data.empty else 0
        max_val = non_nan_data[measure].max() if not non_nan_data.empty else 1
        
        if min_val == max_val: # Avoid division by zero for step if all values are the same
            min_val -= 0.0001
            max_val += 0.0001
        
        step = (max_val - min_val) / 3
        bin_edges = [min_val + i * step for i in range(4)]
        
        quartile_categories = pd.cut(
            plot_data[measure],
            bins=bin_edges,
            labels=['Q1', 'Q2', 'Q3'],
            include_lowest=True
        )
    else:
        # Calculate percentiles for terciles
        q_values = [0, 0.33333, 0.66667, 1.0]
        percentile_edges = non_nan_data[measure].quantile(q_values).tolist()
        
        # Ensure unique bin edges by adding tiny increments if needed
        bin_edges = []
        for i, edge in enumerate(percentile_edges):
            if i == 0 or edge > bin_edges[-1]:
                bin_edges.append(edge)
            else:
                # Add a tiny increment to make it unique
                bin_edges.append(bin_edges[-1] + 1e-9)
        
        # Use cut to assign data to bins
        quartile_categories = pd.cut(
            plot_data[measure],
            bins=bin_edges,
            labels=['Q1', 'Q2', 'Q3'],
            include_lowest=True
        )
    
    plot_data = plot_data.assign(quartile=quartile_categories.astype(str))
    
    quartile_map = {'Q1': 0, 'Q2': 1, 'Q3': 2}
    plot_data['quartile_numeric'] = plot_data['quartile'].map(quartile_map).fillna(-1)
    
    norm = Normalize(vmin=0, vmax=2) # 3 terciles, mapped to 0-2 for norm

    return bin_edges, plot_data, norm

def _plot_h3_on_ax(ax, data_subset_gdf, cmap, norm, country_boundary=None, admin1_boundary=None, subplot_title=None, date_text=None, subtitle_text=None, basemap_choice=None, basemap_alpha=0.5, hexagon_alpha=0.7, zoom=8):
    """
    Plots H3 grid data on a given Matplotlib axes object.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    data_subset_gdf : GeoDataFrame
        The subset of data to plot. Must contain 'quartile_numeric' column.
    cmap : matplotlib.colors.Colormap
        The colormap to use.
    norm : matplotlib.colors.Normalize
        The normalizer for the colormap.
    country_boundary : GeoDataFrame, optional
        Country boundary (admin0) to plot behind H3 grids.
    admin1_boundary : GeoDataFrame, optional
        Admin1 boundaries to plot on top of H3 grids.
    subplot_title : str, optional
        Title for this specific subplot.
    subtitle_text : str, optional
        An optional subtitle for this specific subplot, displayed in the bottom-left corner.
    basemap_choice : str, optional
        Basemap to use. Options: 'osm', 'carto_light', 'carto_dark'. If None, no basemap.
    basemap_alpha : float, optional
        Transparency of the basemap (0=invisible, 1=opaque). Default is 0.5.
    hexagon_alpha : float, optional
        Transparency of the hexagons (0=invisible, 1=opaque). Default is 0.7.
    zoom : int, optional
        Zoom level for the basemap. Default is 8.
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from shapely.geometry import box
    
    # Get country geometry and bounds
    if country_boundary is not None:
        country_bounds = country_boundary.total_bounds
        country_geom = country_boundary.union_all()
        
        # Set axis limits
        ax.set_xlim(country_bounds[0], country_bounds[2])
        ax.set_ylim(country_bounds[1], country_bounds[3])
    
    # Plot country border behind H3 grids (like bivariate map)
    if country_boundary is not None:
        country_boundary.boundary.plot(ax=ax, color='black', alpha=0.75, linewidth=1.5, zorder=14)
    
    # Add basemap FIRST (if needed) so axis limits are set correctly
    if basemap_choice is not None:
        import contextily as ctx
        basemap_dict = {
            'osm': ctx.providers.OpenStreetMap.Mapnik,
            'carto_light': ctx.providers.CartoDB.Positron,
            'carto_dark': ctx.providers.CartoDB.DarkMatter
        }
        
        # Check if basemap_choice is a custom URL or a predefined choice
        if basemap_choice.startswith('http'):
            # Custom tile URL provided
            basemap_source = basemap_choice
        else:
            # Use predefined basemap
            basemap_source = basemap_dict.get(basemap_choice, ctx.providers.CartoDB.Positron)
        
        try:
            ctx.add_basemap(ax, source=basemap_source, crs=data_subset_gdf.crs.to_string(), 
                          alpha=basemap_alpha, zoom=zoom, zorder=1)
        except Exception as e:
            print(f"Warning: Could not add basemap. Error: {e}")
    
    # Plot H3 hexagons with colors (filter out invalid quartiles) - ON TOP of basemap
    valid_data = data_subset_gdf[data_subset_gdf['quartile_numeric'] >= 0].copy()
    
    if len(valid_data) > 0:
        valid_data.plot(
            ax=ax,
            color=cmap(norm(valid_data['quartile_numeric'])),
            alpha=hexagon_alpha,
            edgecolor='white',
            linewidth=0.5,
            legend=False,
            zorder=10
        )
    else:
        print(f"Warning: No valid data to plot after filtering")
    
    # Plot admin1 boundaries on top of everything
    if admin1_boundary is not None:
        admin1_boundary.boundary.plot(ax=ax, color='#666666', alpha=0.8, linewidth=1, zorder=15)
    
    # Create mask for areas outside country (like bivariate map)
    if country_boundary is not None:
        margin = 60.0
        bbox = box(country_bounds[0] - margin, country_bounds[1] - margin, 
                   country_bounds[2] + margin, country_bounds[3] + margin)
        mask_geom = bbox.difference(country_geom)
        
        mask_patches = [mask_geom] if mask_geom.geom_type == 'Polygon' else list(mask_geom.geoms) if mask_geom.geom_type == 'MultiPolygon' else []
        
        for mask_poly in mask_patches:
            if mask_poly.exterior is not None:
                vertices, codes = [], []
                ext_coords = list(mask_poly.exterior.coords)
                vertices.extend(ext_coords)
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ext_coords) - 2) + [Path.CLOSEPOLY])
                
                for interior in mask_poly.interiors:
                    int_coords = list(interior.coords)
                    vertices.extend(int_coords)
                    codes.extend([Path.MOVETO] + [Path.LINETO] * (len(int_coords) - 2) + [Path.CLOSEPOLY])
                
                path = Path(vertices, codes)
                patch = PathPatch(path, facecolor='white', edgecolor='none', zorder=100)
                ax.add_patch(patch)
    
    if subplot_title:
        ax.set_title(subplot_title, fontsize=14, fontweight='bold', loc='left', pad=10)
    if date_text:
        ax.text(0.02, 0.98, date_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', color='#555', zorder=200)
    
    if subtitle_text:
        ax.text(0.02, 0.02, subtitle_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='bottom', color='#555', zorder=200)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

# --- Main Plotting Functions ---

def create_bivariate_conflict_map(
    conflict_data, title="Conflict Patterns: Events × Fatalities"
):
    # Set up figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plot_data = conflict_data.copy(deep=True)

    # Remove NaN values for calculations
    events_data = plot_data["nrEvents"].dropna()
    fatalities_data = plot_data["nrFatalities"].dropna()

    # Calculate quantiles and ensure unique bin edges
    def create_unique_bins(data, num_bins=3):  # Changed to 3 bins
        quantiles = [0, 0.33, 0.67, 1.0]  # Changed to 3 equal quantiles
        bins = data.quantile(quantiles).tolist()
        # Ensure bins are unique by adding small increments
        for i in range(1, len(bins)):
            if bins[i] <= bins[i - 1]:
                bins[i] = bins[i - 1] + 0.000001
        return bins

    events_bins = create_unique_bins(events_data)
    fatalities_bins = create_unique_bins(fatalities_data)

    # Create categories
    plot_data["events_category"] = pd.cut(
        plot_data["nrEvents"],
        bins=events_bins,
        labels=["Low", "Medium", "High"],  # Changed to 3 categories
        include_lowest=True,
    )

    plot_data["fatalities_category"] = pd.cut(
        plot_data["nrFatalities"],
        bins=fatalities_bins,
        labels=["Low", "Medium", "High"],  # Changed to 3 categories
        include_lowest=True,
    )

    # Create color dictionary with 3×3 matrix (9 colors)
    colors = {
        ("Low", "Low"): (0.85, 0.85, 0.85, 1),  # Light gray
        ("Low", "Medium"): (0.9, 0.6, 0.6, 1),  # Medium pink
        ("Low", "High"): (0.9, 0.2, 0.2, 1),  # Red
        ("Medium", "Low"): (0.6, 0.6, 0.9, 1),  # Medium blue
        ("Medium", "Medium"): (0.7, 0.5, 0.7, 1),  # Purple
        ("Medium", "High"): (0.8, 0.3, 0.5, 1),  # Dark pink
        ("High", "Low"): (0.2, 0.2, 0.9, 1),  # Blue
        ("High", "Medium"): (0.4, 0.2, 0.7, 1),  # Dark purple
        ("High", "High"): (0.4, 0.1, 0.4, 1),  # Very dark purple
    }

    # Assign colors
    plot_data["color"] = plot_data.apply(
        lambda row: colors.get(
            (row["events_category"], row["fatalities_category"]), (0.8, 0.8, 0.8, 0.3)
        )
        if not pd.isna(row["events_category"])
        and not pd.isna(row["fatalities_category"])
        else (0.8, 0.8, 0.8, 0.3),
        axis=1,
    )

    # Plot each period
    for idx, period in enumerate(["Before HTS", "Regime Change", "After Assad"]):
        # Plot the Syria administrative boundaries
        syria_adm1.boundary.plot(ax=ax[idx], color="lightgrey", alpha=0.5, linewidth=1)

        # Filter data for this period
        period_data = plot_data[plot_data["category"] == period]

        # Plot filled H3 grid cells
        for _, row in period_data.iterrows():
            if not pd.isna(row["color"]):
                # Plot the polygon geometry with fill color
                ax[idx].fill(*row.geometry.exterior.xy, color=row["color"], alpha=0.8)

        ax[idx].set_title(period, fontsize=14)
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        for spine in ax[idx].spines.values():
            spine.set_visible(False)

    # Add legend
    legend_ax = fig.add_axes([0.28, 0.05, 0.44, 0.15], frameon=True)
    legend_ax.set_facecolor("white")

    categories = ["Low", "Medium", "High"]  # Changed to 3 categories
    for i, event_cat in enumerate(categories):
        for j, fatal_cat in enumerate(categories):
            legend_ax.add_patch(
                plt.Rectangle((j, i), 1, 1, color=colors[(event_cat, fatal_cat)])
            )

    legend_ax.text(
        1.0, -0.5, "Fatalities →", ha="center", fontsize=12, fontweight="bold"
    )
    legend_ax.text(
        -0.5, 1.0, "Events →", va="center", rotation=90, fontsize=12, fontweight="bold"
    )

    for i, cat in enumerate(categories):
        legend_ax.text(i + 0.5, -0.2, cat, ha="center", fontsize=9)
        legend_ax.text(-0.2, i + 0.5, cat, va="center", fontsize=9)

    legend_ax.set_xlim(-0.7, 3.2)  # Adjusted for 3 categories
    legend_ax.set_ylim(-0.7, 3.2)  # Adjusted for 3 categories
    legend_ax.axis("off")

    plt.suptitle(title, fontsize=16, y=0.97)

    return fig, ax

def get_h3_maps(daily_mean_gdf, title, measure='nrEvents', category_list=None, cmap_name=None, custom_colors=None, figsize=None, subtitle=None, country_boundary=None, admin1_boundary=None, basemap_choice=None, basemap_alpha=0.5, hexagon_alpha=0.7, zoom=8, date_ranges=None, legend_title=None):
    """
    Plot H3 grids with color representing the specified measure divided into quartiles.
    Can create either a single map or multiple maps based on categories.

    Parameters:
    -----------
    daily_mean_gdf : GeoDataFrame
        The data to plot.
    title : str
        The main title for the figure (will be suptitle).
    measure : str
        The measure to plot on color scale (can be any numeric column).
    category_list : list, optional
        List of category values to create separate maps for. If None, creates a single map.
    cmap_name : str, optional
        The name of the colormap to use (e.g., 'Blues', 'Reds', 'Purples'). If None, automatically selected based on measure.
    custom_colors : list, optional
        List of 3 hex color strings for terciles (Q1 to Q3). If provided, overrides cmap_name.
    figsize : tuple, optional
        The size of the figure (width, height) in inches. If None, calculated based on number of categories.
    subtitle : str, optional
        An optional subtitle for the plot, displayed in the bottom-left corner. Defaults to None.
    country_boundary : GeoDataFrame, optional
        Country boundary (admin0) to plot behind H3 grids.
    admin1_boundary : GeoDataFrame, optional
        Admin1 boundaries to plot on top of H3 grids.
    basemap_choice : str, optional
        Basemap to use. Options: 'osm', 'carto_light', 'carto_dark'. If None, no basemap.
    basemap_alpha : float, optional
        Transparency of the basemap (0=invisible, 1=opaque). Default is 0.5.
    date_ranges : dict, optional
        Dictionary mapping category names to date range strings (e.g., {'Before': 'Nov 26, 2023 - Nov 27, 2024'}).
    legend_title : str, optional
        Title for the legend. If None, no title is displayed.
    """
    import matplotlib.font_manager as fm
    import os
    
    # Set professional font - load Open Sans directly if available (same as bivariate map)
    open_sans_path = os.path.expanduser("~/Library/Fonts/OpenSans-VariableFont_wdth,wght.ttf")
    if os.path.exists(open_sans_path):
        try:
            fm.fontManager.addfont(open_sans_path)
            plt.rcParams['font.family'] = 'Open Sans'
        except:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    else:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # Determine if we're creating multiple maps or a single map
    if category_list is None:
        num_maps = 1
        category_list = ['All Data']
        if 'category' not in daily_mean_gdf.columns:
            daily_mean_gdf = daily_mean_gdf.copy()
            daily_mean_gdf['category'] = 'All Data'
    else:
        num_maps = len(category_list)
    
    # Set default figure size if not provided
    if figsize is None:
        figsize = (5 * num_maps, 5)
    
    fig, ax = plt.subplots(1, num_maps, figsize=figsize, squeeze=False)
    ax = ax[0]  # Get the first row since we're using 1 row

    # Use custom colors or auto-select colormap based on measure
    if custom_colors is not None:
        from matplotlib.colors import ListedColormap
        if len(custom_colors) != 3:
            print(f"Warning: custom_colors must have exactly 3 colors. Got {len(custom_colors)}. Using default colormap.")
            custom_colors = None
    
    if custom_colors is not None:
        cmap = ListedColormap(custom_colors)
    else:
        if cmap_name is None:
            if measure == 'nrFatalities':
                cmap_name = 'Reds'
            elif measure == 'nrEvents':
                cmap_name = 'Blues'
            else:
                cmap_name = 'Purples'

        try:
            cmap = plt.colormaps[cmap_name]
        except KeyError:
            print(f"Warning: Colormap '{cmap_name}' not found. Falling back to 'Blues'.")
            cmap = plt.colormaps['Blues']

    # Calculate quartiles and prepare data for plotting (globally across all categories)
    bin_edges, plot_data_with_quartiles, norm = _calculate_h3_quartiles(daily_mean_gdf, measure)

    # Plot each category
    for idx, category in enumerate(category_list):
        current_ax = ax[idx] if num_maps > 1 else ax[0]
        
        # Filter data for this category
        if category == 'All Data':
            category_data = plot_data_with_quartiles
        else:
            category_data = plot_data_with_quartiles[plot_data_with_quartiles['category'] == category]
        
        # Plot on this axis
        date_text = date_ranges.get(category, '') if date_ranges else ''
        #print(f"date text is '{date_text}' for category '{category}'")
        _plot_h3_on_ax(current_ax, category_data, cmap, norm, country_boundary=country_boundary, admin1_boundary=admin1_boundary, subplot_title=category, date_text=date_text, subtitle_text=None, basemap_choice=basemap_choice, basemap_alpha=basemap_alpha, hexagon_alpha=hexagon_alpha, zoom=zoom)
    
    # Create custom legend elements
    legend_elements = []
    colors_for_legend = [cmap(norm(i)) for i in range(3)]
    
    # Ensure bin_edges has enough elements for the labels
    # This block ensures bin_edges is correctly padded for legend labels,
    # especially in edge cases where unique values are less than 3.
    if len(bin_edges) < 4:
        # Calculate a reasonable 'step' if not explicitly defined from tercile calculation
        if len(bin_edges) > 1:
            step_val = bin_edges[1] - bin_edges[0]
        else: # If bin_edges has 0 or 1 element, use a default step or infer from measure range
            if not daily_mean_gdf[measure].empty:
                data_range = daily_mean_gdf[measure].max() - daily_mean_gdf[measure].min()
                step_val = data_range / 3 if data_range > 0 else 1
            else:
                step_val = 1
        
        last_val = bin_edges[-1] if bin_edges else 0
        while len(bin_edges) < 4:
            bin_edges.append(last_val + step_val)
            last_val = bin_edges[-1]
            
    for i in range(3):
        try:
            label = f'Q{i+1} ({bin_edges[i]:.2f}-{bin_edges[i+1]:.2f})'
        except IndexError:
            label = f'Q{i+1}'
        
        legend_elements.append(
            Patch(
                facecolor=colors_for_legend[i], 
                edgecolor='none', 
                alpha=0.7, 
                label=label
            )
        )
    
    legend = fig.legend(handles=legend_elements, loc='lower right', ncol=1, bbox_to_anchor=(1, 0), frameon=False, prop={'family': 'Open Sans', 'size': 10})
    
    # Add legend title if provided with left alignment
    if legend_title:
        legend.set_title(legend_title, prop={'size': 10, 'weight': 'bold', 'family': 'Open Sans'})
        legend._legend_box.align = 'left'
    
    # Main title - bold and larger (matching bivariate map style)
    from matplotlib.font_manager import FontProperties
    bold_font = FontProperties(family='Open Sans', weight='bold', size=14.5)
    fig.text(0.05, 0.97, title, fontproperties=bold_font, ha='left')
    
    # Subtitle below title - regular weight, smaller, matches subplot subtitle color
    if subtitle:
        subtitle_font = FontProperties(family='Open Sans', size=10)
        fig.text(0.05, 0.93, subtitle, fontproperties=subtitle_font, ha='left', color='#555')
    
    # Add source text at bottom
    from datetime import datetime
    extraction_date = datetime.now().strftime('%b %d, %Y')
    source_font = FontProperties(family='Open Sans', size=8)
    fig.text(0.05, 0.02, f"Source: ACLED. Extracted {extraction_date}", 
             fontproperties=source_font, color='#666', ha='left')
    
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.06)
    
    return fig, ax

def plot_h3_maps_by_column_value(daily_mean_gdf, column_name, title, measure='nrEvents', cmap_name='Blues', subplot_figsize_per_map=(8, 6), subtitle_prefix="Category: ", boundary_gdfs_map=None):
    """
    Creates subplots of H3 maps, one for each unique value in a specified column.
    Quartile binning is consistent across all subplots. Each subplot can have a different boundary GeoDataFrame.

    Parameters:
    -----------
    daily_mean_gdf : GeoDataFrame
        The data to plot.
    column_name : str
        The name of the column to use for splitting data into subplots
        (e.g., 'category' for 'Before HTS', 'Regime Change', 'After Assad').
    title : str
        The main title for the entire figure (will be suptitle).
    measure : str, optional
        The measure to plot on color scale (can be any numeric column). Defaults to 'nrEvents'.
    cmap_name : str, optional
        The name of the colormap to use (e.g., 'Blues', 'Reds', 'Purples'). Defaults to 'Blues'.
    subplot_figsize_per_map : tuple, optional
        The approximate size (width, height) in inches for each individual subplot.
        The overall figure size will be calculated based on the number of subplots.
        Defaults to (8, 6).
    subtitle_prefix : str, optional
        A prefix for the subtitle of each subplot. The value from the `column_name`
        will be appended (e.g., "Category: Before HTS"). Defaults to "Category: ".
    boundary_gdfs_map : dict, optional
        A dictionary where keys are the unique values from `column_name` and values
        are the corresponding GeoDataFrames to use as boundaries for that subplot.
        If a category is not in the map, no boundary will be plotted for that subplot.
        Defaults to None (no boundaries will be plotted unless specified).

    Returns:
    --------
    tuple: (matplotlib.figure.Figure, numpy.ndarray)
        The figure and an array of axes objects for the subplots.
    """
    if boundary_gdfs_map is None:
        boundary_gdfs_map = {} # Ensure it's a dict even if None is passed

    unique_categories = daily_mean_gdf[column_name].dropna().unique()
    num_plots = len(unique_categories)

    if num_plots == 0:
        print("No data or unique categories to plot.")
        return None, None

    # Determine optimal grid size for subplots
    ncols = min(num_plots, 3) # Max 3 columns for reasonable layout
    nrows = int(np.ceil(num_plots / ncols))

    # Calculate overall figure size
    total_figsize = (subplot_figsize_per_map[0] * ncols, subplot_figsize_per_map[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=total_figsize, squeeze=False) # squeeze=False ensures axes is always 2D array

    # Get the colormap object
    try:
        cmap = plt.colormaps[cmap_name]
    except KeyError:
        print(f"Warning: Colormap '{cmap_name}' not found. Falling back to 'Blues'.")
        cmap = plt.colormaps['Blues']

    # Calculate quartiles and prepare data for plotting once for the entire dataset
    # This ensures consistent binning across all subplots
    bin_edges, full_plot_data, norm = _calculate_h3_quartiles(daily_mean_gdf, measure)

    # Plot each category
    for i, category in enumerate(unique_categories):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        subset_gdf = full_plot_data[full_plot_data[column_name] == category]
        
        # Get the specific boundary_gdf for the current category
        current_boundary_gdf = boundary_gdfs_map.get(category, None)

        _plot_h3_on_ax(
            ax, 
            subset_gdf, 
            cmap, 
            norm, 
            boundary_gdf=current_boundary_gdf, # Pass the specific boundary_gdf
            subplot_title=f"{subtitle_prefix}{category}" # Use category as subplot title
        )
    
    # Hide any unused subplots
    for i in range(num_plots, nrows * ncols):
        row = i // ncols
        col = i % ncols
        fig.delaxes(axes[row, col])

    # Create a unified legend for the entire figure
    legend_elements = []
    colors_for_legend = [cmap(norm(i)) for i in range(3)]

    # Ensure bin_edges has enough elements for the labels (similar logic as in get_h3_maps)
    if len(bin_edges) < 4:
        if len(bin_edges) > 1:
            step_val = bin_edges[1] - bin_edges[0]
        else:
            if not daily_mean_gdf[measure].empty:
                data_range = daily_mean_gdf[measure].max() - daily_mean_gdf[measure].min()
                step_val = data_range / 3 if data_range > 0 else 1
            else:
                step_val = 1
        
        last_val = bin_edges[-1] if bin_edges else 0
        while len(bin_edges) < 4:
            bin_edges.append(last_val + step_val)
            last_val = bin_edges[-1]
            
    for i in range(3):
        try:
            label = f'Q{i+1} ({bin_edges[i]:.2f}-{bin_edges[i+1]:.2f})'
        except IndexError:
            label = f'Q{i+1}'
        
        legend_elements.append(
            Patch(
                facecolor=colors_for_legend[i], 
                edgecolor='none', 
                alpha=0.7, 
                label=label
            )
        )
    
    fig.legend(handles=legend_elements, loc='lower right', ncol=1, bbox_to_anchor=(1, 0), frameon=False)
    
    plt.suptitle(title, fontsize=16) # Main figure title
    fig.set_constrained_layout(True)
    
    return fig, axes

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from matplotlib.patches import Patch

# --- Helper Functions ---

def _create_unique_bins(data, num_bins=3):
    """
    Calculates unique bin edges for given data based on quantiles.
    Ensures bins are strictly increasing by adding small increments if values are duplicated.
    """
    quantiles = np.linspace(0, 1, num_bins + 1).tolist() # Dynamically create quantiles based on num_bins
    bins = data.quantile(quantiles).tolist()
    for i in range(1, len(bins)):
        if bins[i] <= bins[i-1]:
            bins[i] = bins[i-1] + 0.000001
    return bins

def _get_bivariate_colors():
    """
    Defines the 3x3 color matrix for bivariate plotting.
    """
    return {
        ('Low', 'Low'): (0.85, 0.85, 0.85, 1),           # Light gray
        ('Low', 'Medium'): (0.9, 0.6, 0.6, 1),           # Medium pink
        ('Low', 'High'): (0.9, 0.2, 0.2, 1),             # Red
        
        ('Medium', 'Low'): (0.6, 0.6, 0.9, 1),           # Medium blue
        ('Medium', 'Medium'): (0.7, 0.5, 0.7, 1),        # Purple
        ('Medium', 'High'): (0.8, 0.3, 0.5, 1),          # Dark pink
        
        ('High', 'Low'): (0.2, 0.2, 0.9, 1),             # Blue
        ('High', 'Medium'): (0.4, 0.2, 0.7, 1),          # Dark purple
        ('High', 'High'): (0.4, 0.1, 0.4, 1)             # Very dark purple
    }

def _plot_bivariate_on_ax(ax, data_subset_gdf, measure1_col, measure2_col, 
                          measure1_bins, measure2_bins, colors_map, 
                          boundary_gdf=None, subplot_title=None, subtitle_text=None):
    """
    Plots a single bivariate H3 map on a given Matplotlib axes object.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    data_subset_gdf : GeoDataFrame
        The subset of data to plot.
    measure1_col : str
        Name of the first measure column.
    measure2_col : str
        Name of the second measure column.
    measure1_bins : list
        Pre-calculated bin edges for measure1.
    measure2_bins : list
        Pre-calculated bin edges for measure2.
    colors_map : dict
        Dictionary mapping (measure1_cat, measure2_cat) tuples to RGBA colors.
    boundary_gdf : GeoDataFrame, optional
        Geographical boundaries to plot on the map.
    subplot_title : str, optional
        Title for this specific subplot.
    subtitle_text : str, optional
        An optional subtitle for this specific subplot, displayed in the bottom-left corner.
    """
    plot_data_copy = data_subset_gdf.copy(deep=True) # Work on a copy

    # Create categories using globally consistent bins
    plot_data_copy[f'{measure1_col}_category'] = pd.cut(
        plot_data_copy[measure1_col], 
        bins=measure1_bins,
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    plot_data_copy[f'{measure2_col}_category'] = pd.cut(
        plot_data_copy[measure2_col], 
        bins=measure2_bins,
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    # Assign colors
    plot_data_copy['color'] = plot_data_copy.apply(
        lambda row: colors_map.get((row[f'{measure1_col}_category'], row[f'{measure2_col}_category']), 
                                   (0.8, 0.8, 0.8, 0.3)) # Default light grey for NaN/unmapped
        if not pd.isna(row[f'{measure1_col}_category']) and not pd.isna(row[f'{measure2_col}_category']) 
        else (0.8, 0.8, 0.8, 0.3), 
        axis=1
    )
    
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color='lightgrey', alpha=0.5, linewidth=1)
    
    # Plot filled H3 grid cells
    for _, row in plot_data_copy.iterrows():
        if not pd.isna(row['color']):
            ax.fill(*row.geometry.exterior.xy, color=row['color'], alpha=0.8)
    
    if subplot_title:
        ax.set_title(subplot_title)
    
    if subtitle_text:
        ax.text(0.01, 0.01, subtitle_text, ha='left', va='bottom', transform=ax.transAxes, fontsize=10)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# --- Main Plotting Functions ---

def create_single_bivariate_conflict_map(conflict_data_gdf, measure1_col, measure2_col, title="Conflict Patterns: Events × Fatalities", figsize=(8, 6), subtitle=None, boundary_gdf=None, show_legend=True):
    """
    Plot a single H3 grid map with color representing the combination of two specified measures.

    Parameters:
    -----------
    conflict_data_gdf : GeoDataFrame
        The data to plot.
    measure1_col : str
        The name of the first measure column (e.g., 'nrEvents').
    measure2_col : str
        The name of the second measure column (e.g., 'nrFatalities').
    title : str
        The main title for the plot (displayed at the top).
    figsize : tuple, optional
        The size of the figure (width, height) in inches. Defaults to (8, 6).
    subtitle : str, optional
        An optional subtitle for the plot, displayed in the bottom-left corner. Defaults to None.
    boundary_gdf : GeoDataFrame, optional
        Geographical boundaries to plot on the map.
    show_legend : bool, optional
        Whether to display the legend or not. Defaults to True.
    """
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Calculate global bins for consistency (even for a single plot)
    measure1_bins = _create_unique_bins(conflict_data_gdf[measure1_col].dropna(), num_bins=3)
    measure2_bins = _create_unique_bins(conflict_data_gdf[measure2_col].dropna(), num_bins=3)
    colors = _get_bivariate_colors()

    # Plot on the single axis using the helper
    _plot_bivariate_on_ax(
        ax, 
        conflict_data_gdf, 
        measure1_col, measure2_col, 
        measure1_bins, measure2_bins, colors, 
        boundary_gdf=boundary_gdf, 
        subplot_title=title, # Main title for this single plot
        subtitle_text=subtitle
    )
    
    # Add legend (as a grid in a separate axes, conditionally)
    if show_legend:
        # Adjusted bbox_to_anchor for bottom-right placement
        # Coordinates are [left, bottom, width, height] relative to the figure
        legend_ax = fig.add_axes([0.65, 0.05, 0.3, 0.15], frameon=True, facecolor='white')
        
        categories = ['Low', 'Medium', 'High']
        for i, measure1_cat in enumerate(categories):
            for j, measure2_cat in enumerate(categories):
                legend_ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, color=colors[(measure1_cat, measure2_cat)]
                ))
        
        legend_ax.text(1.0, -0.5, f"{measure2_col.replace('nr', '').capitalize()} →", ha='center', fontsize=10, fontweight='bold')
        legend_ax.text(-0.5, 1.0, f"{measure1_col.replace('nr', '').capitalize()} →", va='center', rotation=90, fontsize=10, fontweight='bold')
        
        for i, cat in enumerate(categories):
            legend_ax.text(i + 0.5, -0.2, cat, ha='center', fontsize=8)
            legend_ax.text(-0.2, i + 0.5, cat, va='center', fontsize=8)
        
        legend_ax.set_xlim(-0.7, 3.2)
        legend_ax.set_ylim(-0.7, 3.2)
        legend_ax.axis('off')
    
    fig.set_constrained_layout(True)
    
    return fig, ax

def plot_bivariate_maps_by_column_value(daily_mean_gdf, column_name, measure1_col, measure2_col, title="Conflict Patterns: Events × Fatalities", subplot_figsize_per_map=(8, 6), subtitle_prefix="Category: ", boundary_gdfs_map=None, show_legend=True):
    """
    Creates subplots of bivariate H3 maps, one for each unique value in a specified column.
    Binning for both measures is consistent across all subplots. Each subplot can have a different boundary GeoDataFrame.

    Parameters:
    -----------
    daily_mean_gdf : GeoDataFrame
        The data to plot.
    column_name : str
        The name of the column to use for splitting data into subplots
        (e.g., 'category' for 'Before HTS', 'Regime Change', 'After Assad').
    measure1_col : str
        The name of the first measure column (e.g., 'nrEvents').
    measure2_col : str
        The name of the second measure column (e.g., 'nrFatalities').
    title : str
        The main title for the entire figure (will be suptitle).
    subplot_figsize_per_map : tuple, optional
        The approximate size (width, height) in inches for each individual subplot.
        The overall figure size will be calculated based on the number of subplots.
        Defaults to (8, 6).
    subtitle_prefix : str, optional
        A prefix for the subtitle of each subplot. The value from the `column_name`
        will be appended (e.g., "Category: Before HTS"). Defaults to "Category: ".
    boundary_gdfs_map : dict, optional
        A dictionary where keys are the unique values from `column_name` and values
        are the corresponding GeoDataFrames to use as boundaries for that subplot.
        If a category is not in the map, no boundary will be plotted for that subplot.
        Defaults to None (no boundaries will be plotted unless specified).
    show_legend : bool, optional
        Whether to display the unified legend for the figure or not. Defaults to True.

    Returns:
    --------
    tuple: (matplotlib.figure.Figure, numpy.ndarray)
        The figure and an array of axes objects for the subplots.
    """
    if boundary_gdfs_map is None:
        boundary_gdfs_map = {}

    unique_categories = daily_mean_gdf[column_name].dropna().unique()
    num_plots = len(unique_categories)

    if num_plots == 0:
        print("No data or unique categories to plot.")
        return None, None

    # Determine optimal grid size for subplots
    ncols = min(num_plots, 3) # Max 3 columns for reasonable layout
    nrows = int(np.ceil(num_plots / ncols))

    # Calculate overall figure size
    total_figsize = (subplot_figsize_per_map[0] * ncols, subplot_figsize_per_map[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=total_figsize, squeeze=False)

    # Calculate global bins for consistency across all subplots
    measure1_bins = _create_unique_bins(daily_mean_gdf[measure1_col].dropna(), num_bins=3)
    measure2_bins = _create_unique_bins(daily_mean_gdf[measure2_col].dropna(), num_bins=3)
    colors_map = _get_bivariate_colors()

    # Plot each category
    for i, category in enumerate(unique_categories):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        subset_gdf = daily_mean_gdf[daily_mean_gdf[column_name] == category].copy(deep=True)
        
        # Get the specific boundary_gdf for the current category
        current_boundary_gdf = boundary_gdfs_map.get(category, None)

        _plot_bivariate_on_ax(
            ax, 
            subset_gdf, 
            measure1_col, measure2_col, 
            measure1_bins, measure2_bins, colors_map, 
            boundary_gdf=current_boundary_gdf, 
            subplot_title=f"{subtitle_prefix}{category}" # Use category as subplot title
        )
    
    # Hide any unused subplots
    for i in range(num_plots, nrows * ncols):
        row = i // ncols
        col = i % ncols
        fig.delaxes(axes[row, col])

    # Create a unified legend for the entire figure
    if show_legend:
        legend_ax = fig.add_axes([0.65, 0.05, 0.3, 0.15], frameon=True, facecolor='white') # [left, bottom, width, height]
        
        categories = ['Low', 'Medium', 'High']
        for i, measure1_cat in enumerate(categories):
            for j, measure2_cat in enumerate(categories):
                legend_ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, color=colors_map[(measure1_cat, measure2_cat)]
                ))
        
        legend_ax.text(1.0, -0.5, f"{measure2_col.replace('nr', '').capitalize()} →", ha='center', fontsize=10, fontweight='bold')
        legend_ax.text(-0.5, 1.0, f"{measure1_col.replace('nr', '').capitalize()} →", va='center', rotation=90, fontsize=10, fontweight='bold')
        
        for i, cat in enumerate(categories):
            legend_ax.text(i + 0.5, -0.2, cat, ha='center', fontsize=8)
            legend_ax.text(-0.2, i + 0.5, cat, va='center', fontsize=8)
        
        legend_ax.set_xlim(-0.7, 3.2)
        legend_ax.set_ylim(-0.7, 3.2)
        legend_ax.axis('off')
    
    plt.suptitle(title, fontsize=16) # Main figure title
    fig.set_constrained_layout(True)
    
    return fig, axes

def create_bivariate_conflict_map(conflict_data, category_list, boundary_gdf=None, title="Conflict Patterns: Events × Fatalities"):
    # Set up figure - dynamic number of subplots based on category_list
    num_categories = len(category_list)
    fig, ax = plt.subplots(1, num_categories, figsize=(5 * num_categories, 5))
    
    # Handle single category case (ax won't be array)
    if num_categories == 1:
        ax = [ax]
    
    plot_data = conflict_data.copy(deep=True)
    
    # Remove NaN values for calculations
    events_data = plot_data['nrEvents'].dropna()
    fatalities_data = plot_data['nrFatalities'].dropna()
    
    # Calculate quantiles and ensure unique bin edges
    def create_unique_bins(data, num_bins=3):
        quantiles = [0, 0.33, 0.67, 1.0]
        bins = data.quantile(quantiles).tolist()
        # Ensure bins are unique by adding small increments
        for i in range(1, len(bins)):
            if bins[i] <= bins[i-1]:
                bins[i] = bins[i-1] + 0.000001
        return bins
    
    events_bins = create_unique_bins(events_data)
    fatalities_bins = create_unique_bins(fatalities_data)
    
    # Create categories
    plot_data['events_category'] = pd.cut(
        plot_data['nrEvents'], 
        bins=events_bins,
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    plot_data['fatalities_category'] = pd.cut(
        plot_data['nrFatalities'], 
        bins=fatalities_bins,
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    # Create color dictionary with 3×3 matrix (9 colors)
    colors = {
        ('Low', 'Low'): (0.85, 0.85, 0.85, 1),           # Light gray
        ('Low', 'Medium'): (0.9, 0.6, 0.6, 1),           # Medium pink
        ('Low', 'High'): (0.9, 0.2, 0.2, 1),             # Red
        
        ('Medium', 'Low'): (0.6, 0.6, 0.9, 1),           # Medium blue
        ('Medium', 'Medium'): (0.7, 0.5, 0.7, 1),        # Purple
        ('Medium', 'High'): (0.8, 0.3, 0.5, 1),          # Dark pink
        
        ('High', 'Low'): (0.2, 0.2, 0.9, 1),             # Blue
        ('High', 'Medium'): (0.4, 0.2, 0.7, 1),          # Dark purple
        ('High', 'High'): (0.4, 0.1, 0.4, 1)             # Very dark purple
    }
    
    # Assign colors
    plot_data['color'] = plot_data.apply(
        lambda row: colors.get((row['events_category'], row['fatalities_category']), 
                              (0.8, 0.8, 0.8, 0.3)) 
        if not pd.isna(row['events_category']) and not pd.isna(row['fatalities_category']) 
        else (0.8, 0.8, 0.8, 0.3), 
        axis=1
    )
    
    # Plot each period from category_list
    for idx, period in enumerate(category_list):
        # Plot the boundary if provided
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax[idx], color='lightgrey', alpha=0.5, linewidth=1)
        
        # Filter data for this period
        period_data = plot_data[plot_data['category'] == period]
        
        # Plot filled H3 grid cells
        for _, row in period_data.iterrows():
            if not pd.isna(row['color']):
                # Plot the polygon geometry with fill color
                ax[idx].fill(*row.geometry.exterior.xy, color=row['color'], alpha=0.8)
        
        ax[idx].set_title(period, fontsize=14)
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        for spine in ax[idx].spines.values():
            spine.set_visible(False)
    
    # Add legend - adjust position based on number of categories
    if num_categories == 2:
        legend_left = 0.35
        legend_width = 0.3
    else:
        legend_left = 0.28
        legend_width = 0.44
        
    legend_ax = fig.add_axes([legend_left, 0.05, legend_width, 0.15], frameon=True)
    legend_ax.set_facecolor('white')
    
    bin_categories = ['Low', 'Medium', 'High']
    for i, event_cat in enumerate(bin_categories):
        for j, fatal_cat in enumerate(bin_categories):
            legend_ax.add_patch(plt.Rectangle(
                (j, i), 1, 1, color=colors[(event_cat, fatal_cat)]
            ))
    
    legend_ax.text(1.0, -0.5, "Fatalities →", ha='center', fontsize=12, fontweight='bold')
    legend_ax.text(-0.5, 1.0, "Events →", va='center', rotation=90, fontsize=12, fontweight='bold')
    
    for i, cat in enumerate(bin_categories):
        legend_ax.text(i + 0.5, -0.2, cat, ha='center', fontsize=9)
        legend_ax.text(-0.2, i + 0.5, cat, va='center', fontsize=9)
    
    legend_ax.set_xlim(-0.7, 3.2)
    legend_ax.set_ylim(-0.7, 3.2)
    legend_ax.axis('off')
    
    plt.suptitle(title, fontsize=16, y=0.97)
    
    return fig, ax


def create_bivariate_map_with_basemap(
    conflict_data,
    category_list,
    country_boundary,
    admin1_boundary=None,
    main_title="Conflict Patterns with Terrain",
    basemap_choice='carto_light',
    basemap_alpha=0.5,
    hexagon_alpha=0.5,
    extraction_date=None,
    date_ranges=None,
    bivariate_colors=None,
    legend_size=0.24,
    figsize_per_map=5
):
    """
    Create a bivariate choropleth map with basemap background for any country.
    
    This function creates a multi-panel map showing conflict events and fatalities 
    simultaneously using a 3×3 bivariate color scheme, with a configurable basemap.
    Dates are automatically extracted from the dataframe.
    
    Parameters
    ----------
    conflict_data : geopandas.GeoDataFrame
        DataFrame with H3 hexagon geometries and columns: 'nrEvents', 'nrFatalities', 'category', 'event_date'
    category_list : list of str
        List of category names to create separate map panels (e.g., ['Before', 'After'])
    country_boundary : geopandas.GeoDataFrame
        Country boundary (admin0) for masking areas outside the country
    admin1_boundary : geopandas.GeoDataFrame, optional
        Admin1 boundaries to overlay on the map
    main_title : str, default='Conflict Patterns with Terrain'
        Main title for the figure
    basemap_choice : str, default='carto_light'
        Basemap to use. Options: 'osm', 'carto_light', 'carto_dark'
    basemap_alpha : float, default=0.5
        Transparency of the basemap (0=invisible, 1=opaque)
    hexagon_alpha : float, default=0.5
        Transparency of the hexagons (0=invisible, 1=opaque). Lower values show more basemap terrain.
    extraction_date : str, optional
        Date string for source citation. If None, uses current date
    date_ranges : dict, optional
        Dictionary mapping category names to date range strings (e.g., {'Before': 'Nov 26, 2023 - Nov 27, 2024'}).
        If None, will attempt to extract from 'event_date' column in data
    bivariate_colors : dict, optional
        Custom color scheme dictionary. If None, uses default purple-blue-red scheme
    legend_size : float, default=0.24
        Size of legend as fraction of figure (doubled for multi-panel layouts)
    figsize_per_map : int, default=5
        Figure width per map panel in inches
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    ax : list of matplotlib.axes.Axes
        List of axes objects for each panel
    
    Examples
    --------
    >>> fig, ax = create_bivariate_map_with_basemap(
    ...     conflict_data=conflict_daily_h3_mean,
    ...     category_list=['Before Regime Change', 'After Regime Change'],
    ...     country_boundary=adm0,
    ...     admin1_boundary=syria_adm1,
    ...     main_title="Conflict Patterns in Syria",
    ...     basemap_choice='carto_light'
    ... )
    """
    import contextily as ctx
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from shapely.geometry import box
    import matplotlib.font_manager as fm
    import os
    
    # Set professional font - load Open Sans static fonts
    open_sans_regular = os.path.expanduser("~/Library/Fonts/OpenSans-Regular.ttf")
    open_sans_bold = os.path.expanduser("~/Library/Fonts/OpenSans-Bold.ttf")
    
    fonts_loaded = False
    if os.path.exists(open_sans_regular) and os.path.exists(open_sans_bold):
        try:
            fm.fontManager.addfont(open_sans_regular)
            fm.fontManager.addfont(open_sans_bold)
            plt.rcParams['font.family'] = 'Open Sans'
            fonts_loaded = True
        except:
            pass
    
    if not fonts_loaded:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # Set default extraction date if not provided
    if extraction_date is None:
        extraction_date = datetime.now().strftime('%b %d, %Y')
    
    # Default bivariate color scheme (blue for events, red for fatalities)
    # Rows (events): Low to High = lighter to darker blues
    # Columns (fatalities): Low to High = lighter to darker reds
    if bivariate_colors is None:
        bivariate_colors = {
            (1, 1): "#E8E8E8", (1, 2): "#E4ACAC", (1, 3): "#C85A5A",  # Low events
            (2, 1): "#ACB8E5", (2, 2): "#AD9EA5", (2, 3): "#985356",  # Medium events
            (3, 1): "#5698B9", (3, 2): "#627F8C", (3, 3): "#574249",  # High events
        }
    
    # Get country boundaries
    country_bounds = country_boundary.total_bounds
    country_geom = country_boundary.union_all()
    
    # Create figure with subplots - use 16:9 aspect ratio (World Bank standard)
    num_categories = len(category_list)
    fig_height = (figsize_per_map * num_categories) * (9/16)
    fig, ax = plt.subplots(1, num_categories, figsize=(figsize_per_map * num_categories, fig_height), dpi=150)
    if num_categories == 1:
        ax = [ax]
    
    # Calculate date ranges for each category from the data (if not provided)
    if date_ranges is None:
        date_ranges = {}
        for category in category_list:
            cat_data = conflict_data[conflict_data['category'] == category]
            if 'event_date' in cat_data.columns and not cat_data['event_date'].isna().all():
                min_date = cat_data['event_date'].min()
                max_date = cat_data['event_date'].max()
                date_ranges[category] = f"{min_date.strftime('%b %d, %Y')} - {max_date.strftime('%b %d, %Y')}"
            else:
                date_ranges[category] = ''
    else:
        # Ensure all categories have entries
        for category in category_list:
            if category not in date_ranges:
                date_ranges[category] = ''
    
    # Add basemap and mask to each subplot
    for idx, axis in enumerate(ax):
        # Don't set facecolor to allow basemap colors to show through properly
        axis.set_xlim(country_bounds[0], country_bounds[2])
        axis.set_ylim(country_bounds[1], country_bounds[3])
        
        # Add basemap
        try:
            basemap_urls = {
                'osm': ctx.providers.OpenStreetMap.Mapnik,
                'carto_light': ctx.providers.CartoDB.Positron,
                'carto_dark': ctx.providers.CartoDB.DarkMatter,
            }
            
            # Check if basemap_choice is a custom URL or a predefined choice
            if basemap_choice.startswith('http'):
                # Custom tile URL provided
                basemap_source = basemap_choice
            else:
                # Use predefined basemap
                basemap_source = basemap_urls.get(basemap_choice, ctx.providers.CartoDB.Positron)
            
            ctx.add_basemap(axis, crs=country_boundary.crs, source=basemap_source, 
                          alpha=basemap_alpha, zoom=8, zorder=-1)
        except Exception as e:
            print(f"⚠ Could not add basemap to map {idx + 1}: {e}")
        
        # Create mask for areas outside country
        margin = 60.0
        bbox = box(country_bounds[0] - margin, country_bounds[1] - margin, 
                   country_bounds[2] + margin, country_bounds[3] + margin)
        mask_geom = bbox.difference(country_geom)
        
        mask_patches = [mask_geom] if mask_geom.geom_type == 'Polygon' else list(mask_geom.geoms) if mask_geom.geom_type == 'MultiPolygon' else []
        
        for mask_poly in mask_patches:
            if mask_poly.exterior is not None:
                vertices, codes = [], []
                ext_coords = list(mask_poly.exterior.coords)
                vertices.extend(ext_coords)
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ext_coords) - 2) + [Path.CLOSEPOLY])
                
                for interior in mask_poly.interiors:
                    int_coords = list(interior.coords)
                    vertices.extend(int_coords)
                    codes.extend([Path.MOVETO] + [Path.LINETO] * (len(int_coords) - 2) + [Path.CLOSEPOLY])
                
                path = Path(vertices, codes)
                patch = PathPatch(path, facecolor='white', edgecolor='none', zorder=1, alpha=1.0)
                axis.add_patch(patch)
    
    # Prepare bivariate data
    plot_data = conflict_data.copy(deep=True)
    events_data = plot_data['nrEvents'].dropna()
    fatalities_data = plot_data['nrFatalities'].dropna()
    
    # Calculate quantiles
    def create_unique_bins(data):
        quantiles = [0, 0.33, 0.67, 1.0]
        bins = data.quantile(quantiles).tolist()
        for i in range(1, len(bins)):
            if bins[i] <= bins[i-1]:
                bins[i] = bins[i-1] + 0.000001
        return bins
    
    events_bins = create_unique_bins(events_data)
    fatalities_bins = create_unique_bins(fatalities_data)
    
    # Create categories
    plot_data['events_category'] = pd.cut(plot_data['nrEvents'], bins=events_bins, 
                                           labels=['Low', 'Medium', 'High'], include_lowest=True)
    plot_data['fatalities_category'] = pd.cut(plot_data['nrFatalities'], bins=fatalities_bins,
                                                labels=['Low', 'Medium', 'High'], include_lowest=True)
    
    # Create color dictionary with tuple keys for matplotlib
    colors = {
        ('Low', 'Low'): tuple(int(bivariate_colors[(1,1)].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,),
        ('Low', 'Medium'): tuple(int(bivariate_colors[(1,2)].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,),
        ('Low', 'High'): tuple(int(bivariate_colors[(1,3)].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,),
        ('Medium', 'Low'): tuple(int(bivariate_colors[(2,1)].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,),
        ('Medium', 'Medium'): tuple(int(bivariate_colors[(2,2)].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,),
        ('Medium', 'High'): tuple(int(bivariate_colors[(2,3)].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,),
        ('High', 'Low'): tuple(int(bivariate_colors[(3,1)].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,),
        ('High', 'Medium'): tuple(int(bivariate_colors[(3,2)].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,),
        ('High', 'High'): tuple(int(bivariate_colors[(3,3)].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,)
    }
    
    # Assign colors
    plot_data['color'] = plot_data.apply(
        lambda row: colors.get((row['events_category'], row['fatalities_category']), (0.8, 0.8, 0.8, 0.3)) 
        if not pd.isna(row['events_category']) and not pd.isna(row['fatalities_category']) 
        else (0.8, 0.8, 0.8, 0.3), axis=1
    )
    
    # Plot data on each subplot
    for idx, period in enumerate(category_list):
        period_data = plot_data[plot_data['category'] == period]
        
        # Plot country border in black (behind H3 grids)
        country_boundary.boundary.plot(ax=ax[idx], color='black', alpha=0.75, 
                                      linewidth=1.5, zorder=14)
        
        # Plot hexagons
        for _, row in period_data.iterrows():
            if not pd.isna(row['color']):
                patch = ax[idx].fill(*row.geometry.exterior.xy, color=row['color'], 
                                    alpha=hexagon_alpha, edgecolor='none')
                for p in patch:
                    p.set_zorder(15)
        
        # Plot admin1 boundaries
        if admin1_boundary is not None:
            admin1_boundary.boundary.plot(ax=ax[idx], color='darkgrey', alpha=0.7, 
                                         linewidth=0.8, zorder=20)
        
        # Add titles
        ax[idx].set_title(period, fontsize=14, fontweight='bold', loc='left', pad=10)
        if date_ranges.get(period):
            ax[idx].text(0.02, 0.98, date_ranges[period], transform=ax[idx].transAxes, 
                        fontsize=10, verticalalignment='top', color='#555')
        
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        for spine in ax[idx].spines.values():
            spine.set_visible(False)
    
    # Add legend
    legend_ax = fig.add_axes([0.82, 0.08, legend_size, legend_size], frameon=True)
    legend_ax.set_facecolor('white')
    for spine in legend_ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.5)
    
    # Draw 3x3 color grid
    bin_categories = ['Low', 'Medium', 'High']
    cell_size = 1
    for i, event_cat in enumerate(bin_categories):
        for j, fatal_cat in enumerate(bin_categories):
            legend_ax.add_patch(plt.Rectangle(
                (j * cell_size, i * cell_size), cell_size, cell_size, 
                facecolor=colors[(event_cat, fatal_cat)], 
                edgecolor='white', linewidth=1
            ))
    
    # Legend labels with arrows at the end
    # Fatalities at bottom (horizontal axis)
    legend_ax.text(1.5 * cell_size, -0.5 * cell_size, "Higher Fatalities →", 
                   ha='center', va='center', fontsize=9,
                   fontfamily='Open Sans')
    
    # Events on left (vertical axis) - arrow pointing up (use → which becomes ↑ when rotated)
    legend_ax.text(-0.5 * cell_size, 1.5 * cell_size, "Higher Events →", 
                   ha='center', va='center', fontsize=9, rotation=90,
                   fontfamily='Open Sans')
    
    legend_ax.set_xlim(-1.5 * cell_size, 3.9 * cell_size)
    legend_ax.set_ylim(-0.2 * cell_size, 4.5 * cell_size)
    legend_ax.set_aspect('equal')
    legend_ax.axis('off')
    
    # Add main title with Open Sans Bold
    from matplotlib.font_manager import FontProperties
    bold_font = FontProperties(family='Open Sans', weight='bold', size=14.5)
    fig.text(0.05, 0.97, main_title, fontproperties=bold_font, ha='left')
    
    # Add source text with Open Sans
    source_font = FontProperties(family='Open Sans', size=8)
    fig.text(0.05, 0.02, f"Source: ACLED. Extracted {extraction_date}", 
             fontproperties=source_font, color='#666', ha='left')
    
    # Adjust layout
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.06)
    
    return fig, ax