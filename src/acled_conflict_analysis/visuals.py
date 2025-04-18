from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Legend, Span, Label
from bokeh.layouts import column
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import EMPTY_LAYOUT
import folium
from folium.plugins import TimestampedGeoJson
import pandas as pd
import importlib.resources as pkg_resources



# Use the silence function to ignore the EMPTY_LAYOUT warning
silence(EMPTY_LAYOUT, True)


color_palette = [
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
    # Initialize the figure
    p2 = figure(x_axis_type="datetime", width=750, height=400, toolbar_location="above")
    p2.add_layout(Legend(), "right")


    if category:
        category_df = dataframe[dataframe[category] == category_value].copy()
        category_df.sort_values(
            by="event_date", inplace=True
        )  # Ensure the DataFrame is sorted by date
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

    if events_dict:

        used_y_positions = []

        
        for index, (event_date, label) in enumerate(events_dict.items()):
            span = Span(
                location=event_date,
                dimension="height",
                line_color='#C6C6C6',
                line_width=2,
                line_dash=(4, 4)
            )
            p2.renderers.append(span)

            # Determine a base y position
            base_y = max(category_df[measure])  # Adjust for visibility above the plot
            # Find an appropriate y position that doesn't overlap
            y_position = base_y  # Default position

            # Adjust y_position if it overlaps with previous labels
            while y_position in used_y_positions:
                y_position -= max(category_df[measure])/20  # Move down until it's free

            used_y_positions.append(y_position)  # Store the used position

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
        width=86400000 * 3,
        color=colors,
        source=source,
    )

    legend = Legend(
        items=[
            (category, [renderer]) for category, renderer in zip(categories, renderers)
        ],
        location=(0, -30),
    )
    p2.add_layout(legend, "right")

    # Configure legend
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

    # Add dates of important events if provided
    if events_dict:
        used_y_positions = []

        
        for index, (event_date, label) in enumerate(events_dict.items()):
            span = Span(
                location=event_date,
                dimension="height",
                line_color='#C6C6C6',
                line_width=2,
                line_dash=(4, 4)
            )
            p2.renderers.append(span)

            # Determine a base y position
            base_y = dataframe[measure].max()  # Adjust for visibility above the plot
            # Find an appropriate y position that doesn't overlap
            y_position = base_y  # Default position

            # Adjust y_position if it overlaps with previous labels
            while y_position in used_y_positions:
                y_position -= max(dataframe[measure])/20  # Move down until it's free

            used_y_positions.append(y_position)  # Store the used position

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

    # # Set subtitles and titles
    # title_fig = _create_title_figure(title, width=750, height=50, font_size="14pt")
    # sub_title_fig = _create_title_figure(subtitle if subtitle else source, width=750, height=30, font_size="10pt", font_style="normal")

    # Combine everything into a single layout
    layout = column(title_fig,p2, sub_title_fig)



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
    p2 = figure(x_axis_type="datetime", width=800, height=500, toolbar_location="above")

    p2.add_layout(Legend(), "right")

    for id, adm2 in enumerate(dataframe[category].unique()):
        df = dataframe[dataframe[category] == adm2][
            [event_date, measure]
        ].reset_index(drop=True)
        p2.line(
            df[event_date],
            df[measure],
            line_width=2,
            line_color=color_palette[id],
            legend_label=adm2,
        )

    p2.legend.click_policy = "hide"
    if subtitle is not None:
        p2.title = subtitle

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

    # with silence(MISSING_RENDERERS):
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

    if events_dict:

        used_y_positions = []

            
        for index, (event_date_value, label_text) in enumerate(events_dict.items()):
            span = Span(
                location=event_date_value,
                dimension="height",
                line_color='#C6C6C6',
                line_width=2,
                line_dash=(4, 4)
            )
            p2.renderers.append(span)

            # Determine a base y position
            base_y = max(dataframe[measure])  # Adjust for visibility above the plot
            # Find an appropriate y position that doesn't overlap
            y_position = base_y  # Default position

            # Adjust y_position if it overlaps with previous labels
            while y_position in used_y_positions:
                y_position -= max(dataframe[measure])/20  # Move down until it's free

            used_y_positions.append(y_position)  # Store the used position

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

    layout = column(title_fig, p2, sub_title)


    return layout

import folium
from folium import FeatureGroup
import pandas as pd
from datetime import datetime

def get_animated_map(data, country='India', threshold=100, measure='nrFatalities', animation_period='P1Y', fill_color = 'red', outline_color = 'black'):
    """
    Create a Folium map with different layers for each month or year, with radio buttons to toggle between layers.
    
    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame containing the data with 'latitude', 'longitude', and event information (including 'event_date').
    
    country: str
        Name of the country to center the map (defaults to 'India').
    
    threshold: int
        The threshold for scaling the marker size (defaults to 100).
    
    measure: str
        The measure to show on the map ('nrFatalities' or 'nrEvents').
    
    animation_period: str
        Period of time to group by ('P1Y' for year or 'P1M' for month).
    
    Returns
    -------
    folium.Map
        Folium map with layers for each period (month or year).
    """
    
    if measure == 'nrFatalities':
        measure_name = 'Fatalities'
    elif measure == 'nrEvents':
        measure_name = 'Events'

    # Default location (for India)
    country_centroids = load_country_centroids()
    country_centroid = list(country_centroids[country_centroids['COUNTRY'] == country][['latitude', 'longitude']].iloc[0])

    # Create the base map
    m = folium.Map(location=country_centroid, zoom_start=5, tiles="CartoDB positron", 
                attr="Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.")

    max_radius = 20  # Maximum size for large numbers
    min_radius = 2   # Minimum size for small numbers
    threshold = threshold   # Beyond this number, bubble size stops growing

    # Scaling function for bubble size
    def scale_bubble_size(value, max_radius):
        if value > threshold:
            return max_radius
        elif value > 25:  # Between 25 and threshold
            return max_radius / 2
        else:
            return min_radius

    # Group the data by month or year
    if animation_period == 'P1Y':
        data['year'] = data['event_date'].dt.year
        grouped = data.groupby('year')
    else:
        data['month'] = data['event_date'].dt.to_period('M')
        grouped = data.groupby('month')

    # Create FeatureGroups for each time period (month/year)
    time_layers = {}

    for period, group in grouped:
        feature_group = FeatureGroup(name=str(period))  # Name the layer by period (e.g., Year or Month)

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
    
    # Add a LayerControl so users can toggle between time layers
    folium.LayerControl().add_to(m)

    return m


def load_country_centroids():
    # Access the file from the package using importlib.resources
    with pkg_resources.open_text('acled_conflict_analysis.data', 'countries_centroids.csv') as file:
        country_centroids = pd.read_csv(file)
    return country_centroids

def get_cumulative_animated_map(data, country='India', threshold=100, measure='nrFatalities', animation_period='P1Y'):

    if measure == 'nrFatalities':
        measure_name = 'Fatalities'
    elif measure == 'nrEvents':
        measure_name = 'Events'
    
    country_centroids = load_country_centroids()
    country_centroid = list(country_centroids[country_centroids['COUNTRY'] == country][['latitude', 'longitude']].iloc[0])

    # Create the base map
    m = folium.Map(location=country_centroid, zoom_start=5, tiles="CartoDB positron", 
                attr="Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.")

    # Create a list to hold the geojson features
    features = []

    max_radius = 20  # Maximum size for large numbers
    min_radius = 2   # Minimum size for small numbers
    threshold = threshold   # Beyond this number, bubble size stops growing

    # Scaling function for bubble size
    def scale_bubble_size(value, max_radius):
        if value > threshold:
            return max_radius
        elif value > 25:  # Between 25 and threshold
            return max_radius / 2
        else:
            return min_radius

    # Create features for all events
    for _, row in data.iterrows():
        scaled_radius = scale_bubble_size(row[measure], max_radius)

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['longitude'], row['latitude']],
            },
            'properties': {
                'time': row['event_date'].isoformat(),  # Ensure time is in ISO format
                'popup': f"Fatalities: {row['nrFatalities']}<br>Events: {row['nrEvents']}",
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': 'red',
                    'fillOpacity': 0.6,
                    'stroke': 'false',
                    'color': None,
                    'radius': scaled_radius  # Scale marker size by 'nrFatalities'
                }
            }
        }
        features.append(feature)

    # Create the TimestampedGeoJson layer
    timestamped_geojson = TimestampedGeoJson(
        {
            'type': 'FeatureCollection',
            'features': features
        },
        period=animation_period,  # One month between timestamps
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options='YYYY-MM-DD',
        time_slider_drag_update=True
    )

    # Add the TimestampedGeoJson to the map
    timestamped_geojson.add_to(m)

    # Custom HTML Legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 300px; left: 500px; width: 220px; height: auto; 
        background-color: white; border:0px solid grey; z-index:9999; font-size:14px; padding: 10px;">
        <table>
            <tr>
                <td style="padding: 5px;">
                    <svg width="30" height="30">
                        <circle cx="15" cy="15" r="2" fill="red" />
                    </svg>
                </td>
                <td style="padding-left: 10px;">{measure_name} <= {threshold/2} </td>
            </tr>
            <tr>
                <td style="padding: 5px;">
                    <svg width="30" height="30">
                        <circle cx="15" cy="15" r="10" fill="red" />
                    </svg>
                </td>
                <td style="padding-left: 10px;">{measure_name} > {threshold/2} and <= {threshold}</td>
            </tr>
            <tr>
                <td style="padding: 5px;">
                    <svg width="30" height="30">
                        <circle cx="15" cy="15" r="15" fill="red" />
                    </svg>
                </td>
                <td style="padding-left: 10px;">{measure_name} > {threshold}</td>
            </tr>
        </table>
    </div>
    """

    # Add the custom legend to the map
    m.get_root().html.add_child(folium.Element(legend_html))

    # Show the map
    return m




