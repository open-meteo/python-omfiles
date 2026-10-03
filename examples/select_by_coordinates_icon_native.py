#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[fsspec]>=1.2.0",  # x-release-please-version
#     "scipy",
#     "matplotlib",
# ]
# ///

"""Plot nearest-cell forecasts on ICON's native icosahedral grid.

Run with: uv run examples/select_by_coordinates_icon_native.py
Edit LOCATIONS and VARIABLE below to choose points and a forecast variable.
The first run reads all static coordinates and builds a KD-tree in memory;
allow a few hundred MB of RAM. Reuse the tree for further location queries.
Only the selected cells' forecast chunks are fetched, not the global forecast.

Data organization: https://github.com/open-meteo/open-data#data-organization
"""

import datetime as dt
import json

import fsspec
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from omfiles import OmFileReader
from s3fs import S3FileSystem
from scipy.spatial import cKDTree

MODEL_DOMAIN = "dwd_icon_global_native"
VARIABLE = "temperature_2m"
LOCATIONS = {
    "Zurich": (47.3769, 8.5417),
    "Paris": (48.8647, 2.3490),
}
COORDINATES_URI = f"s3://openmeteo/data/{MODEL_DOMAIN}/static/coordinates.om"
RUN_PREFIX = f"s3://openmeteo/data_run/{MODEL_DOMAIN}"
OUTPUT_PATH = f"{MODEL_DOMAIN}_{VARIABLE}_timeseries.png"


def unit_sphere(latitude, longitude):
    """Convert degrees to Cartesian coordinates on the unit sphere."""
    lat = np.deg2rad(np.asarray(latitude, dtype=np.float64))
    lon = np.deg2rad(np.asarray(longitude, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)), axis=-1)


def main():
    backend = fsspec.open(
        f"filecache::{COORDINATES_URI}",
        mode="rb",
        s3={"anon": True},
        # A domain's grid is fixed, so its coordinates are immutable too
        filecache={"cache_storage": "cache/icon_native/filecache", "check_files": False},
    )
    with OmFileReader(backend) as coordinates:
        # Coordinates have shape (1, n_cells); remove the singleton dimension.
        latitude = coordinates.get_child_by_name("lat")[0, :]
        longitude = coordinates.get_child_by_name("lon")[0, :]

    # Euclidean chord distance on the unit sphere has the same nearest neighbour
    # as great-circle distance. Using degrees directly fails at poles/date line.
    print(f"Building KD-tree for {latitude.size:,} cells...", flush=True)
    tree = cKDTree(unit_sphere(latitude, longitude))
    targets = np.asarray(list(LOCATIONS.values()))
    chord_distances, cell_indices = tree.query(unit_sphere(targets[:, 0], targets[:, 1]))

    # Discover an available run instead of assuming a particular date exists.
    # Fetch the manifest afresh; do not cache a mutable 'latest' pointer.
    fs = S3FileSystem(anon=True)
    latest = json.loads(fs.cat_file(f"{RUN_PREFIX}/latest.json"))
    run_time = dt.datetime.fromisoformat(latest["reference_time"].replace("Z", "+00:00"))
    forecast_uri = f"{RUN_PREFIX}/{run_time:%Y/%m/%d/%H%MZ}/{VARIABLE}.om"
    print(f"Reading forecast: {forecast_uri}", flush=True)

    backend = fsspec.open(
        f"blockcache::{forecast_uri}",
        mode="rb",
        s3={"anon": True, "default_block_size": 65536},
        blockcache={"cache_storage": "cache/icon_native/blockcache", "check_files": False},
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    with OmFileReader(backend) as forecast:
        # Forecasts have shape (1, n_cells, n_times), with the same cell order
        # as the static coordinates. The root array holds VARIABLE's data.
        if len(forecast.shape) != 3 or forecast.shape[:2] != (1, latitude.size):
            raise ValueError(f"Forecast shape {forecast.shape} does not match the native coordinates")
        times = forecast.get_child_by_name("time")[:].astype("datetime64[s]").ravel()
        unit = forecast.get_child_by_name("unit").read_scalar()

        for name, cell, chord in zip(LOCATIONS, cell_indices, chord_distances):
            # Index before loading: this fetches only chunks covering this cell.
            values = forecast[0, int(cell), :]
            distance_km = 2 * 6371.229 * np.arcsin(np.clip(chord / 2, 0, 1))
            print(
                f"\n{name}: cell {cell}, lat={latitude[cell]:.5f}, lon={longitude[cell]:.5f}, "
                f"distance={distance_km:.2f} km"
            )
            ax.plot(times, values, label=name)

    ax.set_title(f"{MODEL_DOMAIN} {VARIABLE}\nRun: {run_time:%Y-%m-%d %H:%M} UTC")
    ax.set_xlabel("Forecast time (UTC)")
    ax.set_ylabel(f"{VARIABLE} ({unit})")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
