from src import functions as f
import numpy as np
import networkx as nx
import pandas as pd
import osmnx as ox

from shapely.geometry import Point
from tqdm import tqdm
from collections import Counter
import h5py



def centrality_analysis(
    min_dist_km: float,
    G: nx.MultiDiGraph,
    dijkstra_weight: str = "travel_time",
    hdf5_path: str = "data/DetektorDaten.h5",
) -> pd.DataFrame:
    """Analyze topological centrality of motorway junctions and compare it

    with empirical traffic volume data from detector stations.
    """
    # Extract node GeoDataFrame and group by junction names
    gdf_nodes, _ = ox.graph_to_gdfs(G, nodes=True, edges=True)
    junction_nodes = gdf_nodes[gdf_nodes["highway"] == "motorway_junction"]

    df_junctions = (
        junction_nodes.reset_index()
        .groupby("name", dropna=True)
        .agg(
            count=("name", "size"),
            mean_x=("x", "mean"),
            mean_y=("y", "mean"),
            node_ids=("osmid", list),
        )
        .reset_index()
    )
    df_junctions = df_junctions[df_junctions["name"] != "nan"]

    detector_nodes = {
        n for n, d in G.nodes(data=True) if d.get("highway") == "detector"
    }
    detector_counter = Counter()
    detector_counter_dist = Counter()

    # Shortest path calculation between all junction pairs
    for _, row_origin in tqdm(
        df_junctions.iterrows(),
        total=len(df_junctions),
        desc="Routing junctions",
    ):
        origin_nodes = row_origin["node_ids"]
        network_distances = {}
        network_paths = {}

        # Compute single-source shortest paths from all nodes of the origin junction
        for start_node in origin_nodes:
            lengths, paths = nx.single_source_dijkstra(
                G, source=start_node, weight=dijkstra_weight
            )
            for target_node, dist in lengths.items():
                if (
                    target_node not in network_distances
                    or dist < network_distances[target_node]
                ):
                    network_distances[target_node] = dist
                    network_paths[target_node] = paths[target_node]

        # Evaluate paths to all destination junctions
        for _, row_dest in df_junctions.iterrows():
            geo_dist_m = f.coord_distance(
                Point(row_origin["mean_x"], row_origin["mean_y"]),
                Point(row_dest["mean_x"], row_dest["mean_y"]),
            )

            min_dist = float("inf")
            best_path = None

            for target_id in row_dest["node_ids"]:
                if target_id in network_distances:
                    dist = network_distances[target_id]
                    if dist < min_dist:
                        min_dist = dist
                        best_path = network_paths[target_id]

            found_detectors = []
            if min_dist < float("inf") and best_path is not None:
                found_detectors = [
                    node for node in best_path if node in detector_nodes
                ]
                detector_counter.update({det: 1 for det in found_detectors})

            # Check distance threshold for long-distance routes
            if geo_dist_m >= (min_dist_km * 1000) and found_detectors:
                detector_counter_dist.update(
                    {det: 1 for det in found_detectors}
                )

    # Build base DataFrame for analyzed detector stations
    df_detectors = pd.DataFrame(
        detector_counter.items(), columns=["node_id", "centrality"]
    )
    df_detectors["centrality_dist"] = df_detectors["node_id"].map(
        detector_counter_dist
    ).map(detector_counter_dist).fillna(0)

    # Retrieve valid evaluation dates
    accepted_dates = f.GetDays(
        dayTypes="all",
        holiday="yes",
        vacation="yes",
        startDate="2016-12-02",
        endDate="2017-12-01",
    )
    accepted_dates_daytime = f.GetDays(
        dayTypes="weekday",
        holiday="no",
        vacation="yes",
        startDate="2016-12-02",
        endDate="2017-12-01",
    )

    detector_names = f.get_detector_name(df_detectors["node_id"], G)
    node_list = f.get_nodelist()

    # Initialize traffic metric columns
    df_detectors["traffic_volume"] = 0.0
    df_detectors["traffic_volume_day"] = 0.0
    df_detectors["measurements_count"] = 0
    df_detectors["measurements_count_day"] = 0

    # Aggregate time series data from HDF5
    with h5py.File(hdf5_path, "r") as h5f:
        for day in tqdm(accepted_dates, desc="Processing HDF5 time series"):
            timeseries = f.get_timeseries(
                detectors=detector_names,
                day=day,
                var="q",
                h5f=h5f,
                node_list=node_list,
            )

            # Daily total (24h)
            vol_sum = np.nansum(timeseries, axis=1)
            valid_points = (~np.isnan(timeseries)).sum(axis=1)
            df_detectors["traffic_volume"] += vol_sum
            df_detectors["measurements_count"] += valid_points

            # Daytime window (07:00 - 19:00 = minute 420 to 1140)
            if day in accepted_dates_daytime:
                daytime_slice = timeseries[:, 7 * 60 : 19 * 60]
                df_detectors["traffic_volume_day"] += np.nansum(
                    daytime_slice, axis=1
                )
                df_detectors["measurements_count_day"] += (
                    ~np.isnan(daytime_slice)
                ).sum(axis=1)

    # Normalize traffic volume and filter out low-data/inactive detectors
    df_detectors.loc[
        df_detectors["traffic_volume"] < 48800, "traffic_volume"
    ] = 0.0
    df_detectors["traffic_volume"] = df_detectors["traffic_volume"] / (
        df_detectors["measurements_count"] + 0.001
    )
    df_detectors.loc[
        df_detectors["measurements_count"] < 14400, "traffic_volume"
    ] = np.nan
    df_detectors.loc[df_detectors["traffic_volume"] == 0, "traffic_volume"] = (
        np.nan
    )

    return df_detectors