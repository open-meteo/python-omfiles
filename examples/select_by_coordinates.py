#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[proj,fsspec,xarray] @ /home/fred/dev/terraputix/python-omfiles",
#     "matplotlib",
# ]
# ///

from datetime import datetime
from typing import Tuple

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from fsspec.implementations.cache_mapper import BasenameCacheMapper
from fsspec.implementations.cached import CachingFileSystem
from omfiles import OmFileReader
from omfiles.om_grid import OmGrid
from omfiles.om_meta import OmMetaChunks
from s3fs import S3FileSystem
from xarray import Dataset


def load_variable_dimensions(
    chunk_index: int, domain_name: str, variable_name: str, fs: fsspec.AbstractFileSystem
) -> Tuple[int, int, int]:
    s3_path = f"openmeteo/data/{domain_name}/{variable_name}/chunk_{chunk_index}.om"
    with OmFileReader.from_fsspec(fs, s3_path) as reader:
        return reader.shape
    raise ValueError(f"Failed to load variable dimensions for chunk {chunk_index}")


def load_chunk_data(
    chunk_index: int,
    domain_name: str,
    variable_name: str,
    grid_coords: Tuple[int, int],
    fs: fsspec.AbstractFileSystem,
    start_date: np.datetime64,
    end_date: np.datetime64,
    meta: OmMetaChunks,
):
    """
    Load data for a specific chunk and grid coordinates.

    Args:
        chunk_index (int): Index of the chunk to load.
        domain_name (str): Name of the domain.
        variable_name (str): Name of the variable to fetch.
        grid_coords (Tuple[int, int]): Grid coordinates (x, y) to extract.
        fs (fsspec.AbstractFileSystem): Filesystem to use for loading data.
        start_date (np.datetime64): Start of requested date range.
        end_date (np.datetime64): End of requested date range.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing (time_array, data_array).
    """
    x, y = grid_coords
    s3_path = f"openmeteo/data/{domain_name}/{variable_name}/chunk_{chunk_index}.om"

    # Generate time array and check if any times are in our range
    chunk_times = meta.get_chunk_time_range(chunk_index)
    time_mask = (chunk_times >= start_date) & (chunk_times <= end_date)
    if not np.any(time_mask):
        return np.array([], dtype="datetime64[s]"), np.array([], dtype=float)

    # Create reader and read data of interest
    with OmFileReader.from_fsspec(fs, s3_path) as reader:
        indices = np.where(time_mask)[0]
        time_slice = slice(indices[0], indices[-1] + 1)  # +1 to include the end
        data = reader[y, x, time_slice]
        return chunk_times[time_mask], data

    raise ValueError("Unreachable")  # Make Pyright happy...


def get_data_for_coordinates(
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
    domain_name: str = "ecmwf_ifs025",
    variable_name: str = "temperature_2m",
) -> Dataset:
    """
    Fetch weather data for specific coordinates across a date range, merging multiple files as needed.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        domain_name (str): Name of the domain to use (must have data on AWS S3 under openmeteo/data/{domain_name}/).
        variable_name (str): Name of the variable to fetch.
        start_date (datetime): Start date for the data.
        end_date (datetime): End date for the data.

    Returns:
        xr.Dataset: Dataset containing the requested variable at the specified location.
    """
    meta = OmMetaChunks.from_s3_json_path(f"openmeteo/data/{domain_name}/static/meta.json", FS)

    start_timestamp = np.datetime64(start_date)
    end_timestamp = np.datetime64(end_date)

    # Find all chunks needed for this date range
    chunk_indices = meta.chunks_for_date_range(start_timestamp, end_timestamp)
    print(f"Need to fetch {len(chunk_indices)} chunks: {chunk_indices}")

    # get dimensions of the variable
    num_y, num_x, num_t = load_variable_dimensions(chunk_indices[0], domain_name, variable_name, FS)
    grid = OmGrid(meta.crs_wkt, (num_y, num_x))

    # Find grid coordinates for geographical coordinates
    grid_point = grid.find_point_xy(lat, lon)
    if grid_point is None:
        raise ValueError(f"Coordinates ({lat}, {lon}) not found in grid of {domain_name}")

    x, y = grid_point
    print(f"Found grid point {grid_point} for coordinates ({lat}, {lon})")
    print(f"Fetching data from {start_date} to {end_date}")

    start_timestamp = np.datetime64(start_date)
    end_timestamp = np.datetime64(end_date)

    # Find all chunks needed for this date range
    chunk_indices = meta.chunks_for_date_range(start_timestamp, end_timestamp)
    print(f"Need to fetch {len(chunk_indices)} chunks: {chunk_indices}")

    # Load data from all chunks
    all_times = []
    all_data = []

    for chunk_idx in chunk_indices:
        times, data = load_chunk_data(
            chunk_idx, domain_name, variable_name, (x, y), FS, start_timestamp, end_timestamp, meta
        )
        if len(times) > 0:
            all_times.append(times)
            all_data.append(data)

    # Concatenate all data
    if not all_times:
        raise ValueError("Failed to load any data for the specified date range")

    time_array = np.concatenate(all_times)
    data_array = np.concatenate(all_data)

    # Create xarray dataset
    ds = xr.Dataset(
        data_vars={
            variable_name: (["time"], data_array),
        },
        coords={
            "time": time_array,
            "latitude": lat,
            "longitude": lon,
        },
        attrs={
            "domain": domain_name,
            "grid_indices": grid_point,
        },
    )
    return ds


# We load data from this Cached Fs-Spec Filesystem
FS = CachingFileSystem(
    fs=S3FileSystem(anon=True, default_block_size=256, default_cache_type="none"),
    # TODO: we'd need to verify files do not change on the remote if they still could change
    cache_check=60,
    block_size=256,
    cache_storage="cache",
    check_files=False,
)
LATITUDE, LONGITUDE = 48.864716, 2.349014  # Paris
START_DATE = datetime(2025, 4, 25, 12, 0)  # 25-04-2025'T'12:00
END_DATE = datetime(2025, 5, 18, 12, 0)  # 18-05-2025'T'12:00
VARIABLE = "temperature_2m"
DOMAINS = [
    "dwd_icon",
    "dwd_icon_eu",
    "dwd_icon_d2",
    "ecmwf_ifs025",
    "ecmwf_ifs",
    "meteofrance_arpege_europe",
    "meteofrance_arpege_world025",
    "meteofrance_arome_france0025",
    "meteofrance_arome_france_hd",
    "meteofrance_arome_france_hd_15min",
    "cmc_gem_gdps",
    "cmc_gem_rdps",
    "cmc_gem_hrdps",
]

print(f"Fetching {VARIABLE} data for coordinates: {LATITUDE}N, {LONGITUDE}E")
print(f"Date range: {START_DATE} to {END_DATE}")


# Collect data from each domain
domain_data: dict[str, xr.Dataset] = {}

# Loop through all domains in the main function
for domain_name in DOMAINS:
    try:
        print(f"\nTrying to fetch data from domain: {domain_name}")
        ds = get_data_for_coordinates(
            lat=LATITUDE,
            lon=LONGITUDE,
            start_date=START_DATE,
            end_date=END_DATE,
            domain_name=domain_name,
            variable_name=VARIABLE,
        )
        domain_data[domain_name] = ds
        print(f"Successfully fetched data from {domain_name}")
    except Exception as e:
        print(f"Could not fetch data from {domain_name}: {e}")

print(f"\nSuccessfully fetched data from {len(domain_data)} domains: {domain_data.keys()}")

if len(domain_data) == 0:
    print("No data could be fetched from any domain. Exiting.")
    exit(1)

# Domain colors for consistent line colors
colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(domain_data)))

plt.figure(figsize=(12, 6))

# Plot data from each domain
for i, (domain_name, ds) in enumerate(domain_data.items()):
    plt.plot(ds["time"].values, ds[VARIABLE].values, label=domain_name, color=colors[i], linewidth=2)

# Enhance the plot
plt.title(f"{VARIABLE.replace('_', ' ').title()} at {LATITUDE:.2f}N, {LONGITUDE:.2f}E")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)" if VARIABLE == "temperature_2m" else VARIABLE)
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()

# Save and show the figure
plt.savefig(f"{VARIABLE}_comparison.png", dpi=150)
plt.show()
