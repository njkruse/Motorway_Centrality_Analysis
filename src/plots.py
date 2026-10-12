import numpy as np
import networkx as nx
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def plot_network_metric(
    df_detectors: pd.DataFrame,
    G: nx.MultiDiGraph,
    direction: str = "NO",
    metric_col: str = "centrality",
    labels: dict = None,
    filename: str = "nrw_centrality.pdf",
    output_dir: str = "images",
    normalized: bool = False,
) -> None:
    """Plot spatial distribution of detector metrics (e.g., centrality or traffic volume)

    on the North Rhine-Westphalia motorway network.
    """
    if labels is None:
        labels = {
            "title": f"Centrality Direction {direction}",
            "cbar": r"$Z_i$",
        }

    def is_in_direction(node_id) -> bool:
        if node_id in G:
            name_attr = G.nodes[node_id].get("name", "")
            name_str = (
                " ".join(name_attr)
                if isinstance(name_attr, list)
                else str(name_attr)
            )
            return direction in name_str
        return False

    # Filter detectors by driving direction
    df_filtered = df_detectors[
        df_detectors["node_id"].apply(is_in_direction)
    ].copy()

    x_coords = [G.nodes[n]["x"] for n in df_filtered["node_id"]]
    y_coords = [G.nodes[n]["y"] for n in df_filtered["node_id"]]
    metric_values = df_filtered[metric_col]

    # Scale values and define colorbar limits
    if normalized:
        metric_values = metric_values / np.nanmean(metric_values)
        vmin, vmax = 0.0, 4.0
    else:
        vmin = np.nanmin(metric_values)
        vmax = np.nanmax(metric_values)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot NRW state boundary and road network graph
    nrw_gdf = ox.geocode_to_gdf("North Rhine-Westphalia, Germany")
    minx, miny, maxx, maxy = nrw_gdf.total_bounds

    ox.plot_footprints(
        nrw_gdf,
        ax=ax,
        color="#f0f0f0",
        edge_color="#000000",
        alpha=0.7,
        show=False,
        close=False,
        edge_linewidth=0.5,
    )
    ox.plot_graph(
        G,
        ax=ax,
        edge_linewidth=0.7,
        edge_color="#444444",
        node_size=0,
        bgcolor="white",
        show=False,
        close=False,
        bbox=(minx, miny, maxx, maxy),
    )

    # Overlay detector locations colored by metric
    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=metric_values,
        cmap="viridis",
        s=9,
        zorder=3,
        edgecolors="black",
        linewidths=0.1,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(scatter, ax=ax, orientation="vertical", shrink=0.9)
    cbar.set_label(labels.get("cbar", ""))
    ax.set_title(labels.get("title", ""))

    # Save figure to relative output directory if filename is provided
    if filename:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path / filename, bbox_inches="tight", dpi=300)

    plt.show()

def plot_metric_correlation(
    df: pd.DataFrame,
    x_col: str = "centrality",
    y_col: str = "traffic_volume",
    norm_x: bool = True,
    norm_y: bool = False,
    labels: dict = None,
    filename: str = None,
    output_dir: str = "images",
) -> None:
    """Plot a scatter plot comparing two detector metrics and display their

    Pearson correlation coefficient in the title.
    """
    if labels is None:
        labels = {
            "xlabel": r"Centrality $\tilde{Z}_i$",
            "ylabel": r"Traffic Flow $Q_i$ (veh/min)",
            "corr_vars": r"Q_i, Z_i",
        }

    # Prepare data and apply mean normalization if requested
    x_data = df[x_col].copy()
    y_data = df[y_col].copy()

    if norm_x:
        x_data = x_data / np.nanmean(x_data)
    if norm_y:
        y_data = y_data / np.nanmean(y_data)

    # Calculate Pearson correlation between the selected columns
    corr_val = df[x_col].corr(df[y_col])

    fig, ax = plt.subplots(figsize=(4, 4))

    # Use okabe_ito colormap with a safe fallback for standard matplotlib environments
    try:
        color = plt.get_cmap("okabe_ito")(5)
    except ValueError:
        color = "#0072B2"  # Standard Okabe-Ito blue

    ax.scatter(x_data, y_data, alpha=0.4, color=color, s=16)
    ax.set_box_aspect(1)

    ax.set_xlabel(labels.get("xlabel", x_col))
    ax.set_ylabel(labels.get("ylabel", y_col))

    corr_vars = labels.get("corr_vars", f"{y_col}, {x_col}")
    ax.set_title(rf"corr(${corr_vars}$) = {corr_val:.2f}")

    # Save figure if a filename is provided
    if filename:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path / filename, bbox_inches="tight", dpi=300)

    plt.show()