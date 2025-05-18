"""
Example showing how to select data from multiple domains in Open-Meteo files stored in S3.

This script demonstrates how to:
1. Use the OmDomain class to work with weather model domains
2. Find the correct grid point for specific latitude/longitude coordinates
3. Load data from S3 using fsspec
4. Convert the data to an xarray Dataset for analysis
5. Extract time series for the selected coordinates across multiple files
6. Merge timeseries data from multiple chunks
7. Plot data from multiple domains in a single figure

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

    # Variable to fetch
    variable = 'temperature_2m'

    print(f"Fetching {variable} data for coordinates: {latitude}N, {longitude}E")
    print(f"Date range: {start_date} to {end_date}")

    # Domain display names for nicer legends
    domains_and_display_names = {
        'dwd_icon': 'DWD ICON (Global)',
        'dwd_icon_eu': 'DWD ICON (Europe)',
        'dwd_icon_d2': 'DWD ICON D2 (Central Europe)',
        'ecmwf_ifs025': 'ECMWF IFS (Global)',
        'meteofrance_arpege_europe': 'Météo-France ARPEGE (Europe)',
        'meteofrance_arpege_world025': 'Météo-France ARPEGE (Global)',
        'meteofrance_arome_france0025': 'Météo-France AROME (France)',
        'meteofrance_arome_france_hd': 'Météo-France AROME HD (France)',
        'meteofrance_arome_france_hd_15min': 'Météo-France AROME HD 15min (France)',
    }

    # Collect data from each domain
    domain_data = {}
    successful_domains = []

    # Loop through all domains in the main function
    for domain_name in domains_and_display_names.keys():
        try:
            print(f"\nTrying to fetch data from domain: {domain_name}")
            ds = get_data_for_coordinates(
                lat=latitude,
                lon=longitude,
                start_date=start_date,
                end_date=end_date,
                domain_name=domain_name,
                variable_name=variable
            )
            domain_data[domain_name] = ds
            successful_domains.append(domain_name)
            print(f"Successfully fetched data from {domain_name}")
        except Exception as e:
            print(f"Could not fetch data from {domain_name}: {e}")
            domain_data[domain_name] = None

    print(f"\nSuccessfully fetched data from {len(successful_domains)} domains: {successful_domains}")

    if not successful_domains:
        print("No data could be fetched from any domain. Exiting.")
        exit(1)

    # Domain colors for consistent line colors
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(successful_domains)))

    plt.figure(figsize=(12, 6))

    # Plot data from each domain
    for i, domain_name in enumerate(successful_domains):
        ds = domain_data[domain_name]
        label = domains_and_display_names[domain_name]
        ds[variable].plot(label=label, color=colors[i], linewidth=2)

    # Enhance the plot
    plt.title(f"{variable.replace('_', ' ').title()} at {latitude:.2f}N, {longitude:.2f}E")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)" if variable == 'temperature_2m' else variable)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    # Save and show the figure
    plt.savefig(f"{variable}_comparison.png", dpi=150)
    plt.show()
