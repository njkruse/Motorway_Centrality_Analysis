import pandas as pd
import osmnx as ox
import numpy as np
from pyproj import Geod
from . import config


def import_network(network_path="data/nrw_detector.graphml"):
    """Load the GraphML network file."""
    return ox.load_graphml(network_path)

def coord_distance(p1, p2):
    """Calculate the distance between two shapely Points in meters."""
    geod = Geod(ellps="WGS84")
    _, _, distance_m = geod.inv(p1.x, p1.y, p2.x, p2.y)
    return distance_m

def get_detector_name(ids, G):
    """Retrieve detector names from the graph using node IDs."""
    return [G.nodes[detid].get("name", "Unknown") for detid in ids]

def get_nodelist(path="data/legend.csv"):
    """Load the node list/legend as a DataFrame."""
    return pd.read_csv(path, header=None, names=[
        "dummy_id", "name", "spur", "lat", "lon", "autobahn", "maxspeed"
    ])

def get_timeseries(detectors, day, var, h5f,node_list=None):
    """
    Extract time series data for a list of detectors from an HDF5 file.
    """
    if node_list is None:
        node_list = get_nodelist()

    valid_detectors = [d for d in detectors if d in node_list["name"].values]
    if not valid_detectors:
        return np.array([])

    # Retrieve node info while maintaining the order of valid_detectors
    used_nodes = node_list.set_index("name").loc[valid_detectors].reset_index()

    timeseries_list = []
    for _, node in used_nodes.iterrows():
        h5_path = f"{node['autobahn']}/{node['name']}/{day}/{var}"
        if h5_path in h5f:
            timeseries_list.append(h5f[h5_path][:])
        else:
            timeseries_list.append(np.full(1440, np.nan))  # 1440 minutes = 24h

    return np.stack(timeseries_list)

def GetDays(dayTypes, holiday, vacation, startDate="2016-12-02", endDate="2017-12-01",
            holiday_list=None, vacation_list=None, special_nonholidays=None):
    """
    Generate and filter dates based on day type, holidays, and vacations.
    """
    holiday_list = holiday_list if holiday_list is not None else config.HOLIDAYS_2017
    vacation_list = vacation_list if vacation_list is not None else config.VACATIONS_2017
    special_nonholidays = special_nonholidays if special_nonholidays is not None else config.SPECIAL_NON_HOLIDAYS

    # Generate full date range and map day of week (0=Monday, ..., 6=Sunday)
    dates = pd.date_range(start=startDate, end=endDate, freq="D")
    weekday_map = {0: "MO", 1: "DI", 2: "MI", 3: "DO", 4: "FR", 5: "SA", 6: "SO"}
    weekdays = dates.dayofweek.map(weekday_map)

    dayData = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "weekday": weekdays
    })

    day_mapping = {
        "weekend": ["SA", "SO"],
        "weekday": ["MO", "DI", "MI", "DO", "FR"],
        "mo": ["MO"], "monday": ["MO"], "montag": ["MO"],
        "di": ["DI"], "tuesday": ["DI"], "dienstag": ["DI"],
        "mi": ["MI"], "wednesday": ["MI"], "mittwoch": ["MI"],
        "do": ["DO"], "thursday": ["DO"], "donnerstag": ["DO"],
        "fr": ["FR"], "friday": ["FR"], "freitag": ["FR"],
        "sa": ["SA"], "saturday": ["SA"], "samstag": ["SA"],
        "so": ["SO"], "sunday": ["SO"], "sonntag": ["SO"],
        "all": ["MO", "DI", "MI", "DO", "FR", "SA", "SO"],
        "modo": ["MO", "DI", "MI", "DO"]
    }
    acceptedDays = day_mapping.get(dayTypes.lower(), [])

    # Filter holidays
    if holiday == "yes":
        acceptedDates = dayData["date"]
    elif holiday == "no":
        acceptedDates = dayData["date"][~dayData["date"].isin(holiday_list)]
    elif holiday == "only":
        acceptedDates = dayData["date"][dayData["date"].isin(holiday_list)]
    elif holiday == "no_extended":
        acceptedDates = dayData["date"][
            ~dayData["date"].isin(holiday_list) & ~dayData["date"].isin(special_nonholidays)
        ]
    else:
        acceptedDates = pd.Series(dtype=str)

    # Filter vacations
    if vacation == "no":
        acceptedDates = acceptedDates[~acceptedDates.isin(vacation_list)]
    elif vacation == "only":
        acceptedDates = acceptedDates[acceptedDates.isin(vacation_list)]

    # Filter by selected weekdays and return as list
    final_dates = dayData["date"][
        dayData["date"].isin(acceptedDates) & dayData["weekday"].isin(acceptedDays)
    ]

    return final_dates.tolist()