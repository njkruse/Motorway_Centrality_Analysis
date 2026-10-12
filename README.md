# Topological Centrality & Traffic Flow Analysis of the NRW Motorway Network

This repository contains a network-science-driven, systematic analysis of road network topology, shortest-path routing dynamics, and empirical traffic loads. Instead of relying on complex socio-economic traffic simulations, this project explores how accurately pure graph-theoretical centrality measures can predict real-world traffic flow across the motorway network of North Rhine-Westphalia (NRW), Germany.

## Key Concepts Explored

- **Topological Centrality & Shortest-Path Dynamics:** Modeling the NRW motorway system as a directed graph using OpenStreetMap (OSM) data and evaluating node relevance ($Z_i$) via all-pairs Dijkstra shortest-path routing (`travel_time`) between motorway junctions.

- **Spatial Constraints & Structural Robustness:** Investigating the stability of normalized centrality distributions ($\tilde{Z}_i = Z_i / \langle Z_i \rangle$) under spatial filtering (e.g., long-distance transit routes $\ge 10\text{ km}$, $\tilde{Z}_i^{\mathrm{(Dist)}}$), demographic weighting ($\tilde{Z}_i^{\mathrm{(Pop)}}$), and subordinate road network integration ($\tilde{Z}_i^{\mathrm{(Road)}}$).

- **Empirical Time-Series Aggregation:** Processing high-resolution loop detector measurements ($Q_i$ in vehicles/minute) from an HDF5 database over a full 12-month period, accounting for weekday regimes, public holidays, school vacations, daytime windows (07:00–19:00), and data-completeness thresholds.

- **Structural Load vs. Parallel Redundancy:** Quantifying the linear relationship ($r = 0.52$) between theoretical network centrality and actual traffic volume, identifying how dense metropolitan network clusters distribute empirical load across parallel relief corridors (e.g., vs. the primary SW–NE axis of the A1).

## Repository Structure

```text
.
├── data/                             # GraphML networks and pre-computed datasets
│   ├── nrw_detector.graphml          # OSM motorway graph with mapped detector nodes
│   ├── Legende.csv                   # Detector metadata (lanes, coordinates, speed limits)
│   └── sample_results.parquet        # Pre-calculated centrality and traffic metrics
├── images/                           # Publication-ready vector graphics (PDF/SVG)
│   ├── map_centrality_norm.pdf       # Spatial distribution of normalized centrality
│   ├── map_centrality_dist.pdf       # Long-distance centrality map (>= 10 km)
│   ├── map_traffic_volume.pdf        # Empirical traffic flow map (veh/min)
│   ├── scatter_centrality_vs_dist.pdf # Correlation scatter plot (centrality vs. long-distance centrality)
│   └── scatter_centrality_vs_traffic.pdf # Correlation scatter plot (Q_i vs. Z_i)
├── notebooks/
│   └── 01_centrality_showcase.ipynb  # Main analytical showcase and visualization
├── src/                              # Modular Python source code
│   ├── config.py                     # Calendar constants (NRW holidays, vacations)
│   ├── functions.py                  # Graph loading, geodesic distance, and HDF5 I/O
│   ├── centrality_analysis.py        # Dijkstra routing and traffic aggregation pipeline
│   └── plots.py                      # Standardized geospatial and statistical plotting
└── requirements.txt                  # Python package dependencies