import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

def data_type_conversion(data):
    data["latitude"] = data["latitude"].astype("float64")
    data["longitude"] = data["longitude"].astype("float64")
    data["fatalities"] = data["fatalities"].astype("int")
    data['event_date'] = pd.to_datetime(data['event_date'])


def convert_to_gdf(df):
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)

    return gdf

def get_acled_by_group(
        data,
        columns = ['latitude', 'longitude'],
        freq=None,
        date_column='event_date'
):
    conflict_grouped = data[
    [
        "country",
        "latitude",
        "longitude",
        "event_type",
        "sub_event_type",
        "location",
        "event_date",
        "fatalities",
    ]
]
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
