"""
Example showing how to select data from specific coordinates in an Open-Meteo file stored in S3.

This script demonstrates how to:
1. Use the OmDomain class to work with weather model domains
2. Find the correct grid point for specific latitude/longitude coordinates
3. Load data from S3 using fsspec
4. Convert the data to an xarray Dataset for analysis
5. Extract time series for the selected coordinates across multiple files
6. Merge timeseries data from multiple chunks

Usage:
    python examples/select_by_coordinates.py

Requirements:
    - fsspec
    - s3fs
    - xarray
    - numpy
    - matplotlib (for plotting)
    - omfiles
"""

from datetime import datetime
from typing import Tuple

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from fsspec.implementations.cache_mapper import BasenameCacheMapper
from fsspec.implementations.cached import CachingFileSystem
from omfiles import OmFilePyReader
from omfiles.om_domains import DOMAINS
from s3fs import S3FileSystem
from xarray import Dataset

# We load data from this Cached Fs-Spec Filesystem
FS = CachingFileSystem(
    fs=S3FileSystem(anon=True, default_block_size=256, default_cache_type="none"),
    # we keep the cache_check short: If files are modified on the remote,
    # but we cache parts of these files locally, we potentially run into crashes/UB
    cache_check=60,
    block_size=256,
    cache_storage="cache",
    check_files=False,
    cache_mapper=BasenameCacheMapper(directory_levels=3)
)

def load_chunk_data(
    chunk_index: int,
    domain_name: str,
    variable_name: str,
    grid_coords: Tuple[int, int],
    fs: fsspec.AbstractFileSystem,
    start_date: np.datetime64,
    end_date: np.datetime64
):
    """
    Load data for a specific chunk and grid coordinates.

    Parameters:
    -----------
    chunk_index : int
        Index of the chunk to load
    domain_name : str
        Name of the domain
    variable_name : str
        Name of the variable to fetch
    grid_coords : Tuple[int, int]
        Grid coordinates (x, y) to extract
    fs : fsspec.AbstractFileSystem
        Filesystem to use for loading data
    start_date : np.datetime64
        Start of requested date range
    end_date : np.datetime64
        End of requested date range

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (time_array, data_array)
    """
    domain = DOMAINS[domain_name]
    x, y = grid_coords
    s3_path = f"openmeteo/data/{domain.name}/{variable_name}/chunk_{chunk_index}.om"

    # Generate time array and check if any times are in our range
    chunk_times = domain.get_chunk_time_range(chunk_index)
    time_mask = (chunk_times >= start_date) & (chunk_times <= end_date)
    if not np.any(time_mask):
        return np.array([], dtype='datetime64[s]'), np.array([], dtype=float)

    # Create reader and read data of interest
    with OmFilePyReader.from_fsspec(fs, s3_path) as reader:
        indices = np.where(time_mask)[0]
        time_slice = slice(indices[0], indices[-1] + 1)  # +1 to include the end
        data = reader[y, x, time_slice]
        return chunk_times[time_mask], data

    raise ValueError("Unreachable") # Make Pyright happy...


def get_data_for_coordinates(
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
    domain_name: str = 'ecmwf_ifs025',
    variable_name: str = 'temperature_2m',
) -> Dataset:
    """
    Fetch weather data for specific coordinates across a date range, merging multiple files as needed.

    Parameters:
    -----------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    domain_name : str
        Name of the domain to use (must be in omfiles.om_domains.DOMAINS)
    variable_name : str
        Name of the variable to fetch
    start_date : datetime
        Start date for the data
    end_date : datetime
        End date for the data

    Returns:
    --------
    xr.Dataset
        Dataset containing the requested variable at the specified location
    """
    # Get the domain configuration
    if domain_name not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain_name}. Available domains: {list(DOMAINS.keys())}")

    domain = DOMAINS[domain_name]

    # Find grid coordinates for geographical coordinates
    grid_point = domain.grid.findPointXy(lat, lon)
    if grid_point is None:
        raise ValueError(f"Coordinates ({lat}, {lon}) not found in grid of {domain_name}")

    x, y = grid_point
    print(f"Found grid point {grid_point} for coordinates ({lat}, {lon})")
    print(f"Fetching data from {start_date} to {end_date}")

    start_timestamp = np.datetime64(start_date)
    end_timestamp = np.datetime64(end_date)

    # Find all chunks needed for this date range
    chunk_indices = domain.chunks_for_date_range(start_timestamp, end_timestamp)
    print(f"Need to fetch {len(chunk_indices)} chunks: {chunk_indices}")

    # Load data from all chunks
    all_times = []
    all_data = []

    for chunk_idx in chunk_indices:
        times, data = load_chunk_data(
            chunk_idx,
            domain_name,
            variable_name,
            (x, y),
            FS,
            start_timestamp,
            end_timestamp
            )
        if len(times) > 0:
            all_times.append(times)
            all_data.append(data)

    # Concatenate all data
    if not all_times:
        raise ValueError("Failed to load any data for the specified date range")

    time_array = np.concatenate(all_times)
    data_array = np.concatenate(all_data)

    # Create the xarray dataset
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
        }
    )
    return ds


if __name__ == "__main__":
    # Example coordinates: Paris
    latitude = 48.864716
    longitude = 2.349014

    # Define a date range
    start_date = datetime(2025, 4, 16, 12, 0)  # 16-04-2025'T'12:00
    end_date = datetime(2025, 5, 18, 12, 0)    # 18-05-2025'T'12:00

    # Create a common figure for both plots
    plt.figure(figsize=(12, 6))

    # Fetch and plot Meteofrance Arpege data
    arpege_ds = get_data_for_coordinates(
        lat=latitude,
        lon=longitude,
        start_date=start_date,
        end_date=end_date,
        domain_name='meteofrance_arpege_europe',
        variable_name='temperature_2m',
    )
    arpege_ds.temperature_2m.plot(label='METEOFRANCE ARPEGE (Europe)')

    # Fetch and plot ICON D2 data
    icon_ds = get_data_for_coordinates(
        lat=latitude,
        lon=longitude,
        start_date=start_date,
        end_date=end_date,
        domain_name='dwd_icon_d2',
        variable_name='temperature_2m',
    )
    icon_ds.temperature_2m.plot(label='DWD ICON D2 (Central Europe)')

    # Plot the temperature series
    plt.title(f"2m Temperature at {latitude:.2f}N, {longitude:.2f}E")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("temperature_comparison.png")
    plt.show()
