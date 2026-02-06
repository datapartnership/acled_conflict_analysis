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
- wbpyplot: For creating World Bank styled visualizations (bar charts and line plots)
- matplotlib: For data visualization
- folium: For creating interactive maps
- pandas: For data manipulation
"""

from wbpyplot import wb_plot
import pandas as pd
import importlib.resources as pkg_resources
from folium.plugins import TimestampedGeoJson
from folium import FeatureGroup
import folium
import contextily as ctx
import geopandas as gpd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import altair as alt

# World Bank color palette for visualization consistency
# Based on official World Bank Data Visualization Style Guide
# https://worldbank.github.io/data-visualization-style-guide/colors
COLOR_PALETTE = [
    "#34A7F2",  # cat1 - Blue
    "#FF9800",  # cat2 - Orange
    "#664AB6",  # cat3 - Purple
    "#4EC2C0",  # cat4 - Teal
    "#F3578E",  # cat5 - Pink
    "#081079",  # cat6 - Navy
    "#0C7C68",  # cat7 - Green
    "#AA0000",  # cat8 - Red
    "#DDDA21",  # cat9 - Yellow
]

# World Bank text colors
WB_TEXT_DARK = "#111111"
WB_TEXT_SUBTLE = "#666666"
WB_GREY_300 = "#8A969F"
WB_GREY_200 = "#CED4DE"
WB_GREY_100 = "#EBEEF4"


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
    width=None,
    figsize=(10, 6)
):
    """
    Create a bar chart to visualize time-series event data using wbpyplot.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing event data with event_date column.
    title : str
        Main title for the chart.
    source : str
        Source information to display at the bottom.
    subtitle : str, optional
        Subtitle for the chart.
    measure : str, optional
        Column name for the measure to plot on y-axis, defaults to "nrEvents".
    category : str, optional
        Column name for filtering the data.
    color_code : str, optional
        Color for the bars (defaults to World Bank blue).
    category_value : any, optional
        Value to filter by if category is specified.
    events_dict : dict, optional
        Dictionary of {datetime: label} for marking significant events.
    width : float, optional
        Width of bars in days, defaults to automatic.
    figsize : tuple, optional
        Figure size (width, height) in inches.
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.
    """
    # Filter data if category is provided
    if category:
        category_df = dataframe[dataframe[category] == category_value].copy()
        category_df.sort_values(by="event_date", inplace=True)
    else:
        category_df = dataframe.copy()
        category_df.sort_values(by="event_date", inplace=True)
    
    # Set default color if not provided
    if color_code is None:
        color_code = "#002244"  # World Bank blue
    
    # Create the plotting function with wb_plot decorator
    @wb_plot(
        title=title,
        subtitle=subtitle,
        note=[("Source:", source)]
    )
    def plot_bars(axs):
        ax = axs[0]
        
        # Calculate appropriate bar width based on date frequency
        if len(category_df) > 1:
            dates = pd.to_datetime(category_df['event_date'])
            date_diff = (dates.iloc[1] - dates.iloc[0]).days
            bar_width = max(0.8, date_diff * 0.8) if width is None else width
        else:
            bar_width = width if width else 1
        
        # Plot bars
        bars = ax.bar(
            category_df['event_date'],
            category_df[measure],
            width=bar_width,
            color=color_code,
            alpha=0.8
        )
        
        # Remove value labels from bars (wbpyplot adds them automatically)
        for text in ax.texts:
            text.set_visible(False)
        
        # Set y-axis label
        ax.set_ylabel(measure.replace('_', ' ').title())
        
        # Improve x-axis formatting for dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add event markers if provided
        if events_dict:
            max_val = category_df[measure].max()
            for event_date, label in events_dict.items():
                ax.axvline(x=event_date, color='#C6C6C6', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.text(event_date, max_val * 0.95, label, rotation=90, 
                       verticalalignment='top', fontsize=9, color='#555555')
        
        return ax
    
    # Call the plotting function
    fig = plot_bars()
    
    # Remove value labels that wbpyplot adds automatically
    if fig and hasattr(fig, 'axes'):
        for ax in fig.axes:
            for text in ax.texts[:]:  # Use slice to iterate over copy
                text.remove()
    
    return fig


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
    width=None,
    height=None,
    figsize=(12, 6)
):
    """
    Create a stacked bar chart for comparing categories over time using wbpyplot.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing event data.
    title : str
        Main title for the chart.
    source_text : str
        Source information to display at the bottom.
    subtitle : str, optional
        Subtitle for the chart.
    date_column : str, optional
        Column name for date values, defaults to "date".
    categories : list, optional
        List of category names to include in the stacked bars.
    colors : list, optional
        List of colors for each category (uses World Bank palette if None).
    events_dict : dict, optional
        Dictionary of {datetime: label} for marking significant events.
    category_column : str, optional
        Column name for category grouping, defaults to "event_type".
    measure : str, optional
        Column name for the measure to plot, defaults to "nrEvents".
    width : int, optional
        Deprecated, use figsize instead.
    height : int, optional
        Deprecated, use figsize instead.
    figsize : tuple, optional
        Figure size (width, height) in inches.
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.
    """
    # Create pivot table for stacked bars
    df_pivot = dataframe.pivot_table(
        index=date_column, columns=category_column, values=measure, fill_value=0
    ).reset_index()
    
    # Use default World Bank categorical colors if not provided
    if colors is None:
        from matplotlib import cm
        colors = COLOR_PALETTE[:len(categories)] if categories else None
    
    # Create the plotting function with wb_plot decorator
    @wb_plot(
        title=title,
        subtitle=subtitle,
        note=[("Source:", source_text)],
        palette="wb_categorical"
    )
    def plot_stacked_bars(axs):
        ax = axs[0]
        
        # Calculate appropriate bar width based on date frequency
        if len(df_pivot) > 1:
            dates = pd.to_datetime(df_pivot[date_column])
            date_diff = (dates.iloc[1] - dates.iloc[0]).days
            bar_width = max(0.8, date_diff * 0.8)  # 80% of the date interval
        else:
            bar_width = 1
        
        # Create stacked bar chart
        bottom = np.zeros(len(df_pivot))
        for i, cat in enumerate(categories):
            if cat in df_pivot.columns:
                color = colors[i] if colors else None
                ax.bar(
                    df_pivot[date_column],
                    df_pivot[cat],
                    bottom=bottom,
                    label=cat,
                    color=color,
                    width=bar_width,
                    alpha=0.85
                )
                bottom += df_pivot[cat].values
        
        # Remove value labels from bars (wbpyplot adds them automatically)
        for text in ax.texts:
            text.set_visible(False)
        
        # Set y-axis label
        ax.set_ylabel(measure.replace('_', ' ').title())
        ax.legend(loc='upper left', frameon=True, fancybox=False)
        
        # Improve x-axis formatting for dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add event markers if provided
        if events_dict:
            max_val = bottom.max()
            for event_date, label in events_dict.items():
                ax.axvline(x=event_date, color='#C6C6C6', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.text(event_date, max_val * 0.95, label, rotation=90,
                       verticalalignment='top', fontsize=9, color='#555555')
        
        return ax
    
    # Call the plotting function
    fig = plot_stacked_bars()
    
    # Remove value labels that wbpyplot adds automatically
    if fig and hasattr(fig, 'axes'):
        for ax in fig.axes:
            for text in ax.texts[:]:  # Use slice to iterate over copy
                text.remove()
    
    return fig


def get_line_plot(
    dataframe,
    title,
    source,
    subtitle=None,
    measure="conflictIndex",
    category="DT",
    event_date="event_date",
    events_dict=None,
    plot_width=None,
    plot_height=None,
    figsize=(12, 7)
):
    """
    Create a line plot for comparing trends across different regions or categories using wbpyplot.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing time-series data.
    title : str
        Main title for the chart.
    source : str
        Source information to display at the bottom.
    subtitle : str, optional
        Subtitle for the chart.
    measure : str or list of str, optional
        Column name(s) for the measure(s) to plot on y-axis. Can be a single string 
        (defaults to "conflictIndex") or a list of strings for multiple subplots.
        If a list is provided, creates a grid with 2 columns and as many rows as needed.
    category : str, optional
        Column name for grouping the data into different lines, defaults to "DT".
    event_date : str, optional
        Column name for date values, defaults to "event_date".
    events_dict : dict, optional
        Dictionary of {datetime: label} for marking significant events.
    plot_width : int, optional
        Deprecated, use figsize instead.
    plot_height : int, optional
        Deprecated, use figsize instead.
    figsize : tuple, optional
        Figure size (width, height) in inches. Automatically adjusted for multi-plot layouts.
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the line plot(s).
    """
    # Ensure event_date column is datetime type
    dataframe = dataframe.copy()
    dataframe[event_date] = pd.to_datetime(dataframe[event_date])
    
    # Get unique categories
    unique_categories = dataframe[category].unique()
    
    # Use World Bank categorical colors
    colors = COLOR_PALETTE[:len(unique_categories)]
    
    # Handle multiple measures
    if isinstance(measure, list):
        measures = measure
        n_measures = len(measures)
        n_cols = 2
        n_rows = (n_measures + n_cols - 1) // n_cols  # Ceiling division
        
        # Create the plotting function with wb_plot decorator for multi-plot layout
        # Calculate width and height in pixels (wb_plot uses pixels, not inches)
        width_px = int(figsize[0] * 100 * 1.5)
        height_px = int(figsize[1] * 100 * n_rows / 1.5)
        
        @wb_plot(
            title=title,
            subtitle=subtitle,
            note=[("Source:", source)],
            palette="wb_categorical",
            nrows=n_rows,
            ncols=n_cols,
            width=width_px,
            height=height_px
        )
        def plot_multi_lines(axs):
            # Flatten axes array for easier iteration
            if n_rows == 1 and n_cols == 1:
                axes = [axs[0]]
            else:
                axes = axs
            
            # Plot each measure in a separate subplot
            for idx, measure_col in enumerate(measures):
                ax = axes[idx]
                
                # Plot each category as a line
                for cat_idx, cat in enumerate(unique_categories):
                    df_category = dataframe[dataframe[category] == cat].copy()
                    df_category = df_category.sort_values(by=event_date).reset_index(drop=True)
                    
                    if df_category.empty:
                        print(f"Warning: No data for category '{cat}'. Skipping line plot.")
                        continue
                    
                    ax.plot(
                        df_category[event_date],
                        df_category[measure_col],
                        label=str(cat),
                        color=colors[cat_idx % len(colors)],
                        linewidth=2.5,
                        alpha=0.85
                    )
                
                # Set labels and title
                ax.set_ylabel(measure_col.replace('_', ' ').title())
                ax.set_title(f"{measure_col.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
                ax.legend(loc='upper left')
                
                # Add event markers if provided
                if events_dict:
                    y_min, y_max = ax.get_ylim()
                    for event_date_value, label_text in events_dict.items():
                        ax.axvline(x=event_date_value, color='#888888', linestyle='--', linewidth=1.5, alpha=0.7)
                        ax.text(event_date_value, y_max * 0.95, label_text, rotation=90,
                               verticalalignment='top', fontsize=9, color='#555555')
            
            # Hide empty subplots
            for idx in range(n_measures, len(axes)):
                if idx < len(axes):
                    axes[idx].set_visible(False)
            
            return axes
        
        # Call the plotting function
        fig = plot_multi_lines()
        
        return fig
    
    else:
        # Single measure - use original wbpyplot decorator approach
        @wb_plot(
            title=title,
            subtitle=subtitle,
            note=[("Source:", source)],
            palette="wb_categorical"
        )
        def plot_lines(axs):
            ax = axs[0]
            
            # Plot each category as a line
            for idx, cat in enumerate(unique_categories):
                df_category = dataframe[dataframe[category] == cat].copy()
                df_category = df_category.sort_values(by=event_date).reset_index(drop=True)
                
                if df_category.empty:
                    print(f"Warning: No data for category '{cat}'. Skipping line plot.")
                    continue
                
                ax.plot(
                    df_category[event_date],
                    df_category[measure],
                    label=str(cat),
                    color=colors[idx % len(colors)],
                    linewidth=2.5,
                    alpha=0.85
                )
            
            # Set y-axis label
            ax.set_ylabel(measure.replace('_', ' ').title())
            ax.legend(loc='upper left')
            
            # Add event markers if provided
            if events_dict:
                y_min, y_max = ax.get_ylim()
                for event_date_value, label_text in events_dict.items():
                    ax.axvline(x=event_date_value, color='#888888', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.text(event_date_value, y_max * 0.95, label_text, rotation=90,
                           verticalalignment='top', fontsize=9, color='#555555')
            
            return ax
        
        # Call the plotting function
        fig = plot_lines()
        
        return fig


def plot_conflict_trends_altair(
    dataframe,
    title,
    source,
    subtitle=None,
    measures=["nrEvents", "nrFatalities"],
    plot_type="bar",
    category=None,
    event_date="event_date",
    show_labels=False,
    date_filter=None,
    width=600,
    height=400,
    bar_width_ratio=0.9,
    share_y_axis=True,
    show_y_title=False
):
    """
    Create professional conflict trend visualizations using Altair with World Bank styling.
    
    This function creates side-by-side charts for multiple measures using Altair, with styling
    similar to World Bank publications. Supports both bar and line charts.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing time-series data.
    title : str
        Main title for the chart.
    source : str
        Source information to display.
    subtitle : str, optional
        Subtitle for the chart.
    measures : list of str, optional
        List of column names for measures to plot. Defaults to ["nrEvents", "nrFatalities"].
    plot_type : str, optional
        Type of plot: "bar" or "line". Defaults to "bar".
    category : str, optional
        Column name for grouping data (used for line plots with multiple lines).
    event_date : str, optional
        Column name for date values, defaults to "event_date".
    show_labels : bool, optional
        Whether to show value labels on bars. Defaults to False.
    date_filter : str or datetime, optional
        Filter data to dates >= this value.
    width : int, optional
        Width of each subplot in pixels. Defaults to 600.
    height : int, optional
        Height of each subplot in pixels. Defaults to 400.
    bar_width_ratio : float, optional
        Bar width as a ratio of the time interval (0-1). Defaults to 0.9.
    share_y_axis : bool, optional
        Whether to use the same y-axis scale for charts side-by-side. Defaults to True.
    show_y_title : bool, optional
        Whether to show y-axis title labels. Defaults to False.
        
    Returns
    -------
    altair.Chart
        The Altair chart object containing the visualizations.
        
    Examples
    --------
    >>> chart = plot_conflict_trends_altair(
    ...     conflict_yearly_national,
    ...     "Annual Conflict Trends",
    ...     "Source: ACLED. Accessed 2024-01-01",
    ...     measures=["nrEvents", "nrFatalities"],
    ...     plot_type="bar"
    ... )
    >>> chart.display()
    """
    df = dataframe.copy()
    
    # Apply date filter if provided
    if date_filter:
        df = df[df[event_date] >= date_filter]
    
    # Ensure date column is datetime
    df[event_date] = pd.to_datetime(df[event_date])
    
    # Configure Altair to use Open Sans font globally for this chart
    # Altair will use system fonts if available
    alt.themes.register('wb_theme', lambda: {
        'config': {
            'font': 'Open Sans',
            'title': {'font': 'Open Sans'},
            'axis': {'labelFont': 'Open Sans', 'titleFont': 'Open Sans'},
            'legend': {'labelFont': 'Open Sans', 'titleFont': 'Open Sans'},
            'header': {'labelFont': 'Open Sans', 'titleFont': 'Open Sans'},
            'mark': {'font': 'Open Sans'},
            'text': {'font': 'Open Sans'}
        }
    })
    alt.themes.enable('wb_theme')
    
    # Create color mapping
    color_map = {measures[i]: COLOR_PALETTE[i] for i in range(len(measures))}
    
    # Detect time frequency and determine appropriate formatting
    dates_sorted = df[event_date].sort_values()
    if len(dates_sorted) > 1:
        time_diffs = dates_sorted.diff().dropna()
        median_diff = time_diffs.median()
        
        # Determine frequency and format
        date_range_years = (dates_sorted.max() - dates_sorted.min()).days / 365
        
        if median_diff.days <= 1:
            # Daily data
            time_format = '%b %d, %Y'
            tick_interval = 'week'
            tick_step = 2
        elif median_diff.days <= 7:
            # Weekly data
            time_format = '%b %d'
            tick_interval = 'week'
            tick_step = 4
        elif median_diff.days <= 31:
            # Monthly data
            time_format = '%b %Y'
            tick_interval = 'month'
            # Show labels smartly based on data range
            if date_range_years > 5:
                tick_step = 6  # Every 6 months
            elif date_range_years > 3:
                tick_step = 3  # Every 3 months
            else:
                tick_step = 1  # Show all months
        elif median_diff.days <= 92:
            # Quarterly data
            time_format = 'Q%q %Y'
            tick_interval = 'month'
            tick_step = 3
        else:
            # Yearly or longer
            time_format = '%Y'
            tick_interval = 'year'
            tick_step = 1
        
        # Calculate bar width as percentage of time unit
        # bar_width_ratio should be 0-1, representing % of available space
        bar_width_days = median_diff.days * min(bar_width_ratio, 1.0)
    else:
        time_format = '%Y'
        tick_interval = 'year'
        tick_step = 1
        bar_width_days = 30
    
    # Calculate y-axis domain for shared scale across charts in same row
    y_max = 0
    if share_y_axis:
        for measure in measures:
            y_max = max(y_max, df[measure].max())
    
    charts = []
    
    for idx, measure in enumerate(measures):
        # Prepare data
        chart_df = df[[event_date, measure]].copy()
        
        # Format measure name for display
        measure_title = measure.replace('_', ' ').replace('nr', 'Number of ').title()
        
        # Determine if this chart should share y-axis with another
        # Charts in same row (i.e., side by side) should share y-axis
        is_left_chart = idx % 2 == 0
        use_shared_y = share_y_axis and len(measures) > 1 and (idx < len(measures) - 1 if is_left_chart else True)
        
        if plot_type == "bar":
            # Create x-axis encoding with smart label filtering
            x_axis_params = {
                'title': None,
                'labelAngle': 0,
                'labelFontWeight': 'normal',
                'format': time_format,
                'labelFontSize': 11,
                'tickCount': {'interval': tick_interval, 'step': tick_step}
            }
            
            # Create bar chart with time-aware bar width
            base = alt.Chart(chart_df).encode(
                x=alt.X(f'{event_date}:T', 
                       axis=alt.Axis(**x_axis_params),
                       scale=alt.Scale(domain=[chart_df[event_date].min(), chart_df[event_date].max()])),
                y=alt.Y(f'{measure}:Q', 
                       axis=alt.Axis(title=measure_title if show_y_title else None, 
                                    titleFontSize=11, 
                                    labelFontSize=10, grid=True, gridOpacity=0.3),
                       scale=alt.Scale(zero=True, domain=[0, y_max * 1.1] if use_shared_y else [0, chart_df[measure].max() * 1.1])),
                color=alt.value(color_map[measure])
            )
            
            # Use time-aware bar width
            bars = base.mark_bar(
                opacity=0.85,
                width={'band': min(bar_width_ratio, 1.0)}
            )
            
            if show_labels:
                text = base.mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-5,
                    fontSize=8
                ).encode(
                    text=alt.Text(f'{measure}:Q', format='.0f')
                )
                chart = (bars + text)
            else:
                chart = bars
                
        else:  # line chart
            if category and category in df.columns:
                # Multiple lines per category
                chart_df = df[[event_date, measure, category]].copy()
                chart = alt.Chart(chart_df).mark_line(
                    strokeWidth=2.5,
                    opacity=0.85
                ).encode(
                    x=alt.X(f'{event_date}:T', 
                           axis=alt.Axis(title=None, labelAngle=0, labelFontWeight='normal',
                                        labelFontSize=11)),
                    y=alt.Y(f'{measure}:Q', 
                           axis=alt.Axis(title=measure_title if show_y_title else None, 
                                        titleFontSize=11, 
                                        labelFontSize=10, grid=True, gridOpacity=0.3),
                           scale=alt.Scale(zero=True, domain=[0, y_max * 1.1] if use_shared_y else [0, chart_df[measure].max() * 1.1])),
                    color=alt.Color(f'{category}:N', 
                                   scale=alt.Scale(range=COLOR_PALETTE[:df[category].nunique()]),
                                   legend=alt.Legend(title=None, orient='top-left'))
                )
            else:
                # Single line
                chart = alt.Chart(chart_df).mark_line(
                    strokeWidth=2.5,
                    opacity=0.85,
                    color=color_map[measure]
                ).encode(
                    x=alt.X(f'{event_date}:T', 
                           axis=alt.Axis(title=None, labelAngle=0, labelFontWeight='normal',
                                        labelFontSize=11)),
                    y=alt.Y(f'{measure}:Q', 
                           axis=alt.Axis(title=measure_title if show_y_title else None, 
                                        titleFontSize=11, 
                                        labelFontSize=10, grid=True, gridOpacity=0.3),
                           scale=alt.Scale(zero=True, domain=[0, y_max * 1.1] if use_shared_y else [0, chart_df[measure].max() * 1.1]))
                )
        
        # Add subtitle as title (don't apply configure here - will do after concat)
        chart = chart.properties(
            width=width,
            height=height,
            title=alt.TitleParams(
                text=measure_title,
                fontSize=13,
                fontWeight='bold',
                anchor='start',
                align='left'
            )
        )
        
        charts.append(chart)
    
    # Combine charts side by side
    if len(charts) == 1:
        combined = charts[0]
    else:
        # Create rows with 2 columns
        rows = []
        for i in range(0, len(charts), 2):
            if i + 1 < len(charts):
                rows.append(alt.hconcat(charts[i], charts[i + 1], spacing=30))
            else:
                rows.append(charts[i])
        combined = alt.vconcat(*rows, spacing=20) if len(rows) > 1 else rows[0]
    
    # Build title with optional subtitle
    title_text = [title]
    if subtitle:
        title_text.append(subtitle)
    
    # Add overall title (no configuration yet)
    final_chart = combined.properties(
        title=alt.TitleParams(
            text=title_text,
            fontSize=16,
            fontWeight='bold',
            anchor='start',
            align='left',
            dy=-10,
            subtitleFontSize=12,
            subtitleFontWeight='normal'
        )
    )
    
    # Add source as a separate text annotation at the bottom
    source_text = source if source.startswith("Source:") else f"Source: {source}"
    source_chart = alt.Chart({'values': [{}]}).mark_text(
        align='left',
        baseline='top',
        fontSize=10,
        fontWeight='normal',
        color=WB_TEXT_SUBTLE,
        dx=-width * (2 if len(charts) > 1 else 1) / 2 - (15 if len(charts) > 1 else 0),
        dy=10
    ).encode(
        text=alt.value(source_text)
    ).properties(
        width=width * (2 if len(charts) > 1 else 1) + (30 if len(charts) > 1 else 0),
        height=20
    )
    
    # Combine chart with source and apply all configurations at the end
    final_with_source = alt.vconcat(
        final_chart, 
        source_chart, 
        spacing=5
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domainWidth=0,
        gridColor=WB_GREY_100,
        labelColor=WB_TEXT_SUBTLE,
        titleColor=WB_TEXT_DARK
    ).configure_title(
        color=WB_TEXT_DARK,
        subtitleColor=WB_TEXT_SUBTLE
    )
    
    return final_with_source


def plot_conflict_by_category_altair(
    dataframe,
    title,
    source,
    category_column,
    measure="nrEvents",
    subtitle=None,
    plot_type="bar",
    event_date="event_date",
    date_filter=None,
    width=350,
    height=200,
    bar_width_ratio=0.9,
    n_cols=2,
    color=None
):
    """
    Create faceted conflict visualizations broken down by a categorical variable.
    
    This function creates separate subplots for each value in the specified category column,
    using Altair with World Bank styling.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing time-series data.
    title : str
        Main title for the chart.
    source : str
        Source information to display.
    category_column : str
        Column name for the categorical variable to facet by (e.g., 'event_type', 'region').
    measure : str, optional
        Column name for the measure to plot. Defaults to "nrEvents".
    subtitle : str, optional
        Subtitle for the chart.
    plot_type : str, optional
        Type of plot: "bar" or "line". Defaults to "bar".
    event_date : str, optional
        Column name for date values, defaults to "event_date".
    date_filter : str or datetime, optional
        Filter data to dates >= this value.
    width : int, optional
        Width of each subplot in pixels. Defaults to 350.
    height : int, optional
        Height of each subplot in pixels. Defaults to 200.
    bar_width_ratio : float, optional
        Bar width as a ratio of the time interval (0-1). Defaults to 0.9.
    n_cols : int, optional
        Number of columns in the grid layout. Defaults to 2.
    color : str, optional
        Color to use for the bars/lines. If None, defaults to World Bank blue (#34A7F2) 
        for nrEvents and red (#AA0000) for nrFatalities.
        
    Returns
    -------
    altair.Chart
        The Altair chart object containing the faceted visualizations.
        
    Examples
    --------
    >>> chart = plot_conflict_by_category_altair(
    ...     conflict_event_type,
    ...     "Conflict Events by Type",
    ...     "ACLED. Accessed 2024-01-01",
    ...     category_column="event_type",
    ...     measure="nrEvents"
    ... )
    >>> chart.display()
    """
    df = dataframe.copy()
    
    # Apply date filter if provided
    if date_filter:
        df = df[df[event_date] >= date_filter]
    
    # Ensure date column is datetime
    df[event_date] = pd.to_datetime(df[event_date])
    
    # Configure Altair to use Open Sans font
    alt.themes.register('wb_theme', lambda: {
        'config': {
            'font': 'Open Sans',
            'title': {'font': 'Open Sans'},
            'axis': {'labelFont': 'Open Sans', 'titleFont': 'Open Sans'},
            'legend': {'labelFont': 'Open Sans', 'titleFont': 'Open Sans'},
            'header': {'labelFont': 'Open Sans', 'titleFont': 'Open Sans'},
            'mark': {'font': 'Open Sans'},
            'text': {'font': 'Open Sans'}
        }
    })
    alt.themes.enable('wb_theme')
    
    # Get unique categories - filter out any with no data
    df = df.dropna(subset=[measure])  # Remove rows with NaN in measure
    
    # Filter out categories that have zero or all-NaN values
    category_totals = df.groupby(category_column)[measure].sum().sort_values(ascending=False)
    categories = category_totals[category_totals > 0].index.tolist()
    n_categories = len(categories)
    
    if n_categories == 0:
        raise ValueError(f"No valid data found for measure '{measure}'")
    
    # Filter dataframe to only include categories with data
    df = df[df[category_column].isin(categories)]
    
    # Determine color based on measure if not specified
    if color is None:
        if 'events' in measure.lower():
            color = '#34A7F2'  # World Bank blue for events
        elif 'fatalit' in measure.lower():
            color = '#AA0000'  # World Bank red for fatalities
        else:
            color = COLOR_PALETTE[0]  # Default to first color in palette
    
    # Detect time frequency for formatting and calculate bar width using unique dates
    dates_sorted = df[event_date].drop_duplicates().sort_values()
    if len(dates_sorted) > 1:
        time_diffs = dates_sorted.diff().dropna()
        median_diff = time_diffs.median()
        date_range_years = (dates_sorted.max() - dates_sorted.min()).days / 365
        
        if median_diff.days <= 1:
            time_format = '%b %d, %Y'
            tick_interval = 'week'
            tick_step = 2
            bar_width_pixels = max(1, bar_width_ratio * 5)
        elif median_diff.days <= 7:
            time_format = '%b %d'
            tick_interval = 'week'
            tick_step = 4
            bar_width_pixels = max(1, bar_width_ratio * 10)
        elif median_diff.days <= 31:
            time_format = '%b %Y'
            tick_interval = 'month'
            if date_range_years > 5:
                tick_step = 6
            elif date_range_years > 3:
                tick_step = 3
            else:
                tick_step = 1
            bar_width_pixels = max(1, bar_width_ratio * 20)
        elif median_diff.days <= 92:
            time_format = 'Q%q %Y'
            tick_interval = 'month'
            tick_step = 3
            bar_width_pixels = max(1, bar_width_ratio * 40)
        else:
            time_format = '%Y'
            tick_interval = 'year'
            tick_step = 1
            bar_width_pixels = max(1, bar_width_ratio * 60)
    else:
        time_format = '%Y'
        tick_interval = 'year'
        tick_step = 1
        bar_width_pixels = max(1, bar_width_ratio * 60)
    
    # For yearly data, adjust tick step based on date range
    if tick_interval == 'year' and len(dates_sorted) > 1:
        date_range_years = (dates_sorted.max() - dates_sorted.min()).days / 365
        if date_range_years > 20:
            tick_step = 5
        elif date_range_years > 10:
            tick_step = 2
        else:
            tick_step = 1
    
    # Format measure name for display
    measure_title = measure.replace('_', ' ').replace('nr', 'Number of ').title()
    
    if plot_type == "bar":
        # Create bar chart with faceting
        # Note: For temporal axes, use fixed pixel width based on time interval
        base = alt.Chart(df).mark_bar(
            opacity=0.85,
            width=bar_width_pixels
        ).encode(
            x=alt.X(f'{event_date}:T',
                   axis=alt.Axis(
                       title=None,
                       labelAngle=0,
                       labelFontWeight='normal',
                       format=time_format,
                       labelFontSize=10,
                       tickCount={'interval': tick_interval, 'step': tick_step}
                   ),
                   scale=alt.Scale(
                       domain=[df[event_date].min(), df[event_date].max()],
                       padding=10
                   )),
            y=alt.Y(f'{measure}:Q',
                   axis=alt.Axis(
                       title=measure_title,
                       titleFontSize=10,
                       labelFontSize=9,
                       grid=True,
                       gridOpacity=0.3
                   ),
                   scale=alt.Scale(zero=True)),
            color=alt.value(color),
            facet=alt.Facet(
                f'{category_column}:N',
                columns=n_cols,
                header=alt.Header(
                    title=None,
                    labelFontSize=11,
                    labelFontWeight='bold',
                    labelAnchor='start'
                )
            )
        ).properties(
            width=width,
            height=height
        )
    else:  # line chart
        base = alt.Chart(df).mark_line(
            strokeWidth=2.5,
            opacity=0.85
        ).encode(
            x=alt.X(f'{event_date}:T',
                   axis=alt.Axis(
                       title=None,
                       labelAngle=0,
                       labelFontWeight='normal',
                       format=time_format,
                       labelFontSize=10,
                       tickCount={'interval': tick_interval, 'step': tick_step}
                   )),
            y=alt.Y(f'{measure}:Q',
                   axis=alt.Axis(
                       title=measure_title,
                       titleFontSize=10,
                       labelFontSize=9,
                       grid=True,
                       gridOpacity=0.3
                   ),
                   scale=alt.Scale(zero=True)),
            color=alt.value(color),
            facet=alt.Facet(
                f'{category_column}:N',
                columns=n_cols,
                header=alt.Header(
                    title=None,
                    labelFontSize=11,
                    labelFontWeight='bold',
                    labelAnchor='start'
                )
            )
        ).properties(
            width=width,
            height=height
        )
    
    # Build title with optional subtitle
    title_text = [title]
    if subtitle:
        title_text.append(subtitle)
    
    # Add overall title
    final_chart = base.properties(
        title=alt.TitleParams(
            text=title_text,
            fontSize=16,
            fontWeight='bold',
            anchor='start',
            align='left',
            dy=-10,
            subtitleFontSize=12,
            subtitleFontWeight='normal'
        )
    ).resolve_scale(
        y='independent',  # Each subplot gets its own y-scale
        x='independent'   # Each subplot gets its own x-scale and labels
    )
    
    # Add source as a separate text annotation at the bottom
    source_text = source if source.startswith("Source:") else f"Source: {source}"
    total_width = width * n_cols + 30 * (n_cols - 1)
    
    source_chart = alt.Chart({'values': [{}]}).mark_text(
        align='left',
        baseline='top',
        fontSize=10,
        fontWeight='normal',
        color=WB_TEXT_SUBTLE,
        dx=-total_width / 2,
        dy=10
    ).encode(
        text=alt.value(source_text)
    ).properties(
        width=total_width,
        height=20
    )
    
    # Combine chart with source
    final_with_source = alt.vconcat(
        final_chart,
        source_chart,
        spacing=5
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domainWidth=0,
        gridColor=WB_GREY_100,
        labelColor=WB_TEXT_SUBTLE,
        titleColor=WB_TEXT_DARK
    ).configure_title(
        color=WB_TEXT_DARK,
        subtitleColor=WB_TEXT_SUBTLE
    ).configure_header(
        labelColor=WB_TEXT_DARK
    )
    
    return final_with_source


def plot_lines_by_category_altair(
    dataframe,
    title,
    source,
    category_column,
    measure="nrEvents",
    subtitle=None,
    event_date="event_date",
    date_filter=None,
    width=800,
    height=400,
    colors=None
):
    """
    Create a single line chart with multiple lines, one for each category.
    
    This function creates a single plot with multiple colored lines representing
    different categories, using Altair with World Bank styling.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing time-series data.
    title : str
        Main title for the chart.
    source : str
        Source information to display.
    category_column : str
        Column name for the categorical variable (e.g., 'event_type', 'ADM1_EN').
    measure : str, optional
        Column name for the measure to plot. Defaults to "nrEvents".
    subtitle : str, optional
        Subtitle for the chart.
    event_date : str, optional
        Column name for date values, defaults to "event_date".
    date_filter : str or datetime, optional
        Filter data to dates >= this value.
    width : int, optional
        Width of the chart in pixels. Defaults to 800.
    height : int, optional
        Height of the chart in pixels. Defaults to 400.
    colors : list, optional
        List of colors to use for different categories. If None, uses World Bank palette.
        
    Returns
    -------
    altair.Chart
        The Altair chart object containing the multi-line visualization.
        
    Examples
    --------
    >>> chart = plot_lines_by_category_altair(
    ...     conflict_regional_yearly,
    ...     "Fatalities by Region",
    ...     "ACLED. Accessed 2024-01-01",
    ...     category_column="ADM1_EN",
    ...     measure="nrFatalities"
    ... )
    >>> chart.display()
    """
    df = dataframe.copy()
    
    # Apply date filter if provided
    if date_filter:
        df = df[df[event_date] >= date_filter]
    
    # Ensure date column is datetime
    df[event_date] = pd.to_datetime(df[event_date])
    
    # Filter out rows with NaN in measure or category
    df = df.dropna(subset=[measure, category_column])
    
    if len(df) == 0:
        raise ValueError(f"No valid data found for measure '{measure}' and category '{category_column}'")
    
    # Configure Altair to use Open Sans font
    alt.themes.register('wb_theme', lambda: {
        'config': {
            'font': 'Open Sans',
            'title': {'font': 'Open Sans'},
            'axis': {'labelFont': 'Open Sans', 'titleFont': 'Open Sans'},
            'legend': {'labelFont': 'Open Sans', 'titleFont': 'Open Sans'},
            'header': {'labelFont': 'Open Sans', 'titleFont': 'Open Sans'},
            'mark': {'font': 'Open Sans'},
            'text': {'font': 'Open Sans'}
        }
    })
    alt.themes.enable('wb_theme')
    
    # Get unique categories
    categories = sorted(df[category_column].unique())
    n_categories = len(categories)
    
    # Use custom colors or default to World Bank palette
    if colors is None:
        colors = COLOR_PALETTE[:n_categories]
    
    # Detect time frequency for formatting using unique dates
    dates_sorted = df[event_date].drop_duplicates().sort_values()
    if len(dates_sorted) > 1:
        time_diffs = dates_sorted.diff().dropna()
        median_diff = time_diffs.median()
        date_range_years = (dates_sorted.max() - dates_sorted.min()).days / 365
        
        if median_diff.days <= 1:
            time_format = '%b %d, %Y'
            tick_interval = 'week'
            tick_step = 2
        elif median_diff.days <= 7:
            time_format = '%b %d'
            tick_interval = 'week'
            tick_step = 4
        elif median_diff.days <= 31:
            time_format = '%b %Y'
            tick_interval = 'month'
            if date_range_years > 5:
                tick_step = 6
            elif date_range_years > 3:
                tick_step = 3
            else:
                tick_step = 1
        elif median_diff.days <= 92:
            time_format = 'Q%q %Y'
            tick_interval = 'month'
            tick_step = 3
        else:
            time_format = '%Y'
            tick_interval = 'year'
            tick_step = 1
    else:
        time_format = '%Y'
        tick_interval = 'year'
        tick_step = 1
    
    # For yearly data, adjust tick step based on date range
    if tick_interval == 'year' and len(dates_sorted) > 1:
        date_range_years = (dates_sorted.max() - dates_sorted.min()).days / 365
        if date_range_years > 20:
            tick_step = 5
        elif date_range_years > 10:
            tick_step = 2
        else:
            tick_step = 1
    
    # Format measure name for display
    measure_title = measure.replace('_', ' ').replace('nr', 'Number of ').title()
    
    # Create line chart with multiple lines
    lines = alt.Chart(df).mark_line(
        strokeWidth=2.5,
        opacity=0.85
    ).encode(
        x=alt.X(f'{event_date}:T',
               axis=alt.Axis(
                   title=None,
                   labelAngle=0,
                   labelFontWeight='normal',
                   format=time_format,
                   labelFontSize=11,
                   tickCount={'interval': tick_interval, 'step': tick_step}
               )),
        y=alt.Y(f'{measure}:Q',
               axis=alt.Axis(
                   title=measure_title,
                   titleFontSize=11,
                   labelFontSize=10,
                   grid=True,
                   gridOpacity=0.3
               ),
               scale=alt.Scale(zero=True)),
        color=alt.Color(
            f'{category_column}:N',
            scale=alt.Scale(range=colors),
            legend=alt.Legend(
                title=category_column.replace('_', ' ').title(),
                orient='right',
                labelFontSize=10,
                titleFontSize=11
            )
        )
    ).properties(
        width=width,
        height=height
    )
    
    # Build title with optional subtitle
    title_text = [title]
    if subtitle:
        title_text.append(subtitle)
    
    final_chart = lines.properties(
        title=alt.TitleParams(
            text=title_text,
            fontSize=16,
            fontWeight='bold',
            anchor='start',
            align='left',
            dy=-10,
            subtitleFontSize=12,
            subtitleFontWeight='normal'
        )
    )
    
    # Add source as a separate text annotation at the bottom
    source_text = source if source.startswith("Source:") else f"Source: {source}"
    
    source_chart = alt.Chart({'values': [{}]}).mark_text(
        align='left',
        baseline='top',
        fontSize=10,
        fontWeight='normal',
        color=WB_TEXT_SUBTLE,
        dx=-width / 2,
        dy=10
    ).encode(
        text=alt.value(source_text)
    ).properties(
        width=width,
        height=20
    )
    
    # Combine chart with source
    final_with_source = alt.vconcat(
        final_chart,
        source_chart,
        spacing=5
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domainWidth=0,
        gridColor=WB_GREY_100,
        labelColor=WB_TEXT_SUBTLE,
        titleColor=WB_TEXT_DARK
    ).configure_title(
        color=WB_TEXT_DARK,
        subtitleColor=WB_TEXT_SUBTLE
    ).configure_legend(
        labelColor=WB_TEXT_DARK,
        titleColor=WB_TEXT_DARK
    )
    
    return final_with_source


def plot_conflict_trends(
    dataframe,
    title,
    source,
    subtitle=None,
    measures=None,
    plot_type="bar",
    category=None,
    event_date="event_date",
    events_dict=None,
    colors=None,
    figsize=(14, 5),
    date_filter=None,
    show_labels=False
):
    """
    Create side-by-side plots for multiple measures using wbpyplot styling.
    Supports both line and bar plots with flexible time intervals.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing time-series data.
    title : str
        Main title for the chart.
    source : str
        Source information to display at the bottom.
    subtitle : str, optional
        Subtitle for the chart.
    measures : list of str, optional
        List of column names for measures to plot. Defaults to ["nrEvents", "nrFatalities"].
    plot_type : str, optional
        Type of plot: "bar" or "line". Defaults to "bar".
    category : str, optional
        Column name for grouping data (used for line plots with multiple lines).
    event_date : str, optional
        Column name for date values, defaults to "event_date".
    events_dict : dict, optional
        Dictionary of {datetime: label} for marking significant events with vertical lines.
    colors : list, optional
        List of colors for each measure. If None, uses World Bank colors.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    date_filter : str, optional
        Date string to filter data from (e.g., "2016-01-01").
    show_labels : bool, optional
        Whether to show value labels on bar charts. Defaults to False.
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the plots.
        
    Examples
    --------
    >>> # Bar chart with multiple measures
    >>> fig = plot_conflict_trends(
    ...     conflict_yearly_national,
    ...     "Annual Conflict Trends",
    ...     "Source: ACLED. Accessed 2024-01-01",
    ...     measures=["nrEvents", "nrFatalities"],
    ...     plot_type="bar"
    ... )
    
    >>> # Line chart with categories
    >>> fig = plot_conflict_trends(
    ...     conflict_monthly,
    ...     "Monthly Conflict Trends by Region",
    ...     "Source: ACLED. Accessed 2024-01-01",
    ...     measures=["nrEvents", "nrFatalities"],
    ...     plot_type="line",
    ...     category="admin1"
    ... )
    """
    # Set default measures
    if measures is None:
        measures = ["nrEvents", "nrFatalities"]
    
    # Set default colors
    if colors is None:
        default_colors = {
            "nrEvents": "#1AA1DB",
            "nrFatalities": "#F28E2B"
        }
        colors = [default_colors.get(m, COLOR_PALETTE[i % len(COLOR_PALETTE)]) 
                 for i, m in enumerate(measures)]
    
    # Prepare data
    df = dataframe.copy()
    df[event_date] = pd.to_datetime(df[event_date])
    
    # Apply date filter if provided
    if date_filter:
        df = df[df[event_date] >= date_filter]
    
    # Calculate layout
    n_measures = len(measures)
    n_cols = 2
    n_rows = (n_measures + n_cols - 1) // n_cols
    
    # Convert figsize to width/height in pixels for wb_plot
    width_px = int(figsize[0] * 100)
    height_px = int(figsize[1] * 100)
    
    # Use wbpyplot decorator
    @wb_plot(
        title=title,
        subtitle=subtitle,
        note=[("Source:", source)],
        nrows=n_rows,
        ncols=n_cols,
        width=width_px,
        height=height_px
    )
    def plot_trends(axs):
        # Handle single vs multiple axes
        if n_rows == 1 and n_cols == 1:
            axes = [axs[0]]
        else:
            axes = axs
        
        # Plot each measure
        for idx, measure in enumerate(measures):
            ax = axes[idx]
            
            if plot_type == "bar":
                # Bar plot - calculate appropriate bar width based on time interval
                dates = pd.to_datetime(df[event_date])
                if len(dates) > 1:
                    # Calculate the median time difference between consecutive dates
                    time_diffs = dates.sort_values().diff().dropna()
                    if len(time_diffs) > 0:
                        median_diff = time_diffs.median()
                        # Convert to days and use 80% of that as bar width
                        bar_width = median_diff.days * 0.8
                    else:
                        bar_width = 30  # Default to 30 days
                else:
                    bar_width = 30
                
                bars = ax.bar(df[event_date], df[measure], width=bar_width, 
                             color=colors[idx], edgecolor='none', alpha=0.85)
                
                # Add labels if requested
                if show_labels:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:  # Only label non-zero bars
                            ax.text(bar.get_x() + bar.get_width() / 2., height,
                                   f'{int(height)}',
                                   ha='center', va='bottom', 
                                   fontsize=8, fontweight='normal')
                
            elif plot_type == "line":
                # Line plot
                if category and category in df.columns:
                    # Multiple lines per category
                    unique_categories = df[category].unique()
                    cat_colors = COLOR_PALETTE[:len(unique_categories)]
                    
                    for cat_idx, cat in enumerate(unique_categories):
                        df_cat = df[df[category] == cat].sort_values(event_date)
                        ax.plot(
                            df_cat[event_date],
                            df_cat[measure],
                            label=str(cat),
                            color=cat_colors[cat_idx % len(cat_colors)],
                            linewidth=2.5,
                            alpha=0.85
                        )
                    ax.legend(loc='upper left', fontsize=9)
                else:
                    # Single line
                    df_sorted = df.sort_values(event_date)
                    ax.plot(
                        df_sorted[event_date],
                        df_sorted[measure],
                        color=colors[idx],
                        linewidth=2.5,
                        alpha=0.85
                    )
            
            # Set title and labels
            measure_title = measure.replace('_', ' ').replace('nr', 'Number of ')
            ax.set_title(measure_title.title(), fontsize=12, fontweight='bold', pad=10, loc='left')
            ax.set_ylabel(measure_title.title(), fontsize=10)
            
            # Add event markers if provided
            if events_dict:
                y_min, y_max = ax.get_ylim()
                for event_date_val, event_label in events_dict.items():
                    ax.axvline(x=event_date_val, color='red', linestyle='--', 
                             alpha=0.5, linewidth=1)
                    # Optionally add label
                    # ax.text(event_date_val, y_max * 0.95, event_label, 
                    #        rotation=90, va='top', fontsize=8, alpha=0.7)
            
            # Style the plot
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            # Don't set datavalues to None here - it breaks wbpyplot
            # We'll remove labels after the decorator finishes
        
        # Hide empty subplots
        for idx in range(n_measures, len(axes)):
            if idx < len(axes):
                axes[idx].set_visible(False)
        
        return axes[0].figure if len(axes) > 0 else None
    
    # Create and return the plot
    fig = plot_trends()
    
    # Additional cleanup after wb_plot returns: remove bar labels and fix x-axis
    if fig:
        for ax in fig.axes:
            if not ax.get_visible():
                continue
                
            # Remove ALL bar value label text objects if not showing labels
            if plot_type == "bar" and not show_labels:
                # Remove all text objects on the axes (these are the bar labels)
                # We need to be more aggressive here since wbpyplot adds them
                for txt in list(ax.texts):
                    txt.remove()
            
            # Fix x-axis formatting and limits
            ax.xaxis.set_tick_params(labelbottom=True)
            for label in ax.xaxis.get_ticklabels():
                label.set_visible(True)
                label.set_fontweight('normal')  # Remove bold
            
            # Set x-axis limits to match the actual data range
            dates = pd.to_datetime(df[event_date])
            if len(dates) > 0:
                min_date = dates.min()
                max_date = dates.max()
                ax.set_xlim(min_date, max_date)
    
    return fig


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

def _plot_h3_on_ax(ax, data_subset_gdf, cmap, norm, country_boundary=None, admin1_boundary=None, subplot_title=None, date_text=None, subtitle_text=None, basemap_choice=None, basemap_alpha=0.5, hexagon_alpha=0.7, zoom=8, title_pad=10):
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
    title_pad : float, optional
        Padding between the subplot title and the map content. Default is 10.
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
        ax.set_title(subplot_title, fontsize=14, fontweight='bold', loc='left', pad=title_pad)
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

def get_h3_maps(daily_mean_gdf, title, measure='nrEvents', category_list=None, cmap_name=None, custom_colors=None, figsize=None, subtitle=None, country_boundary=None, admin1_boundary=None, basemap_choice=None, basemap_alpha=0.5, hexagon_alpha=0.7, zoom=8, date_ranges=None, legend_title=None, subplot_titles=None, custom_bins=None, source_text=None, title_pad=15):
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
    subplot_titles : list, dict, or None, optional
        Controls subplot titles. Options:
        - None: Uses category names as titles (default behavior)
        - List: Custom titles for each subplot (must match length of category_list)
        - Dict: Maps category names to custom titles
        - False or empty list []: No subplot titles displayed
    custom_bins : list, optional
        Custom bin edges for classification. Must have exactly 4 values (edges for 3 bins).
        Example: [0, 10, 20, 30] creates bins [0-10), [10-20), [20-30].
        If None, bins are calculated automatically using terciles.
    source_text : str, optional
        Custom source text to display at the bottom of the figure.
        If None, defaults to "Source: ACLED. Extracted {current_date}".
        Set to empty string '' or False to hide source text.
    title_pad : float, optional
        Padding between subplot titles and map content in points. Default is 15.
        Increase this value if titles appear too close to the maps.
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
    if custom_bins is not None:
        # Validate custom bins
        if not isinstance(custom_bins, (list, tuple)) or len(custom_bins) != 4:
            raise ValueError("custom_bins must be a list or tuple with exactly 4 values (bin edges)")
        
        # Ensure bins are strictly increasing
        for i in range(1, len(custom_bins)):
            if custom_bins[i] <= custom_bins[i-1]:
                raise ValueError(f"custom_bins must be strictly increasing. Found {custom_bins[i]} <= {custom_bins[i-1]}")
        
        # Use custom bins to create quartiles
        plot_data_with_quartiles = daily_mean_gdf.copy(deep=True)
        quartile_categories = pd.cut(
            plot_data_with_quartiles[measure],
            bins=custom_bins,
            labels=['Q1', 'Q2', 'Q3'],
            include_lowest=True
        )
        plot_data_with_quartiles['quartile'] = quartile_categories.astype(str)
        quartile_map = {'Q1': 0, 'Q2': 1, 'Q3': 2}
        plot_data_with_quartiles['quartile_numeric'] = plot_data_with_quartiles['quartile'].map(quartile_map).fillna(-1)
        norm = Normalize(vmin=0, vmax=2)
        bin_edges = list(custom_bins)
    else:
        # Use automatic tercile calculation
        bin_edges, plot_data_with_quartiles, norm = _calculate_h3_quartiles(daily_mean_gdf, measure)
    
    # Process subplot_titles parameter
    subplot_title_map = {}
    if subplot_titles is False or (isinstance(subplot_titles, list) and len(subplot_titles) == 0):
        # No titles
        for category in category_list:
            subplot_title_map[category] = None
    elif isinstance(subplot_titles, dict):
        # Dictionary mapping categories to titles
        subplot_title_map = subplot_titles
    elif isinstance(subplot_titles, list):
        # List of titles - must match length of category_list
        if len(subplot_titles) != len(category_list):
            raise ValueError(f"subplot_titles list length ({len(subplot_titles)}) must match category_list length ({len(category_list)})")
        subplot_title_map = {cat: title for cat, title in zip(category_list, subplot_titles)}
    elif subplot_titles is None:
        # Default: use category names as titles
        for category in category_list:
            subplot_title_map[category] = category
    else:
        raise ValueError("subplot_titles must be None, False, a list, or a dict")

    # Plot each category
    for idx, category in enumerate(category_list):
        current_ax = ax[idx] if num_maps > 1 else ax[0]
        
        # Filter data for this category
        if category == 'All Data':
            category_data = plot_data_with_quartiles
        else:
            category_data = plot_data_with_quartiles[plot_data_with_quartiles['category'] == category]
        
        # Get subplot title for this category
        subplot_title = subplot_title_map.get(category, category)
        
        # Plot on this axis
        date_text = date_ranges.get(category, '') if date_ranges else ''
        #print(f"date text is '{date_text}' for category '{category}'")
        _plot_h3_on_ax(current_ax, category_data, cmap, norm, country_boundary=country_boundary, admin1_boundary=admin1_boundary, subplot_title=subplot_title, date_text=date_text, subtitle_text=None, basemap_choice=basemap_choice, basemap_alpha=basemap_alpha, hexagon_alpha=hexagon_alpha, zoom=zoom, title_pad=title_pad)
    
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
    if source_text is not False and source_text != '':
        from datetime import datetime
        if source_text is None:
            # Default source text
            extraction_date = datetime.now().strftime('%b %d, %Y')
            source_text = f"Source: ACLED. Extracted {extraction_date}"
        
        source_font = FontProperties(family='Open Sans', size=8)
        fig.text(0.05, 0.02, source_text, 
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