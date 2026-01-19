#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[fsspec, grids]>=1.1.0",
#     "matplotlib",
#     "cartopy",
#     "scipy",
# ]
# ///

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import fsspec
import matplotlib.pyplot as plt
import numpy as np
from omfiles import OmFileReader
from omfiles.grids import OmGrid
from scipy.interpolate import griddata

MODEL_DOMAIN = "ncep_hrrr_conus"
VARIABLE = "temperature_2m"
# Example: URI for a spatial data file in the `data_spatial` S3 bucket
# See data organization details: https://github.com/open-meteo/open-data?tab=readme-ov-file#data-organization
# Note: Spatial data is only retained for 7 days. The example file below may no longer exist.
# Please update the URI to match a currently available file.
S3_URI = f"s3://openmeteo/data_spatial/{MODEL_DOMAIN}/2026/01/17/0000Z/2026-01-19T0000.om"

LON_MIN = -130.0
LON_MAX = -100.0
LAT_MIN = 30.0
LAT_MAX = 50.0
RESOLUTION_DEGREES = 0.1
TARGET_LONS = np.arange(LON_MIN, LON_MAX, RESOLUTION_DEGREES, dtype=np.float32)
TARGET_LATS = np.arange(LAT_MIN, LAT_MAX, RESOLUTION_DEGREES, dtype=np.float32)


def find_boundary_grid_indices(grid: OmGrid, lons: np.ndarray, lats: np.ndarray):
    """
    Iterates over lons and lats to find the boundary grid indices.

    If the selection is fully within the grid, it could be done way more efficiently
    by only iterating over the top, bottom, left and right boundaries. If the
    boundary-lines are not covered by the grid, however, we would not find the
    correct grid indices. Therefore, this is more of a brute-force approach.
    """
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    for lon in lons:
        for lat in lats:
            grid_point = grid.find_point_xy(float(lat), float(lon))
            if grid_point is None:
                continue
            xmin = min(xmin, grid_point.x)
            xmax = max(xmax, grid_point.x)
            ymin = min(ymin, grid_point.y)
            ymax = max(ymax, grid_point.y)
    return xmin, xmax, ymin, ymax


backend = fsspec.open(
    f"blockcache::{S3_URI}",
    mode="rb",
    s3={"anon": True, "default_block_size": 65536},
    blockcache={"cache_storage": "cache"},
)
with OmFileReader(backend) as reader:
    print("reader.is_group", reader.is_group)
    child = reader.get_child_by_name(VARIABLE)

    # Create coordinate arrays
    num_y, num_x = child.shape
    grid = OmGrid(reader.get_child_by_name("crs_wkt").read_scalar(), (num_y, num_x))
    source_crs = grid.crs
    if source_crs is None:
        raise ValueError("CRS is None, this should only happen for gaussian grids")

    xmin, xmax, ymin, ymax = find_boundary_grid_indices(grid, TARGET_LONS, TARGET_LATS)

    print(f"Grid selection: ({ymin}, {ymax}) x ({xmin}, {xmax})")
    # Get meshgrid - already in geographic coordinates (WGS84)
    lon_grid, lat_grid = grid.get_meshgrid()
    print(f"Original grid shape: {lon_grid.shape}")
    lon_grid = lon_grid[ymin:ymax, xmin:xmax]
    lat_grid = lat_grid[ymin:ymax, xmin:xmax]
    print(f"Trimmed grid shape: {lon_grid.shape}")

    print("child.name", child.name)
    print("child.shape", child.shape)
    print("child.chunks", child.chunks)
    # Only read the relevant data
    data = child[ymin:ymax, xmin:xmax]
    print(f"Data shape: {data.shape}")
    print(f"Data range: {np.nanmin(data)} to {np.nanmax(data)}")
    print(f"Longitude range: {np.nanmin(lon_grid)} to {np.nanmax(lon_grid)}")
    print(f"Latitude range: {np.nanmin(lat_grid)} to {np.nanmax(lat_grid)}")

    target_lon_grid, target_lat_grid = np.meshgrid(TARGET_LONS, TARGET_LATS)

    print(f"Target grid shape: {target_lon_grid.shape}")

    # Flatten source coordinates and data for interpolation
    source_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    source_values = data.ravel()

    # # Remove NaN values
    # valid_mask = ~np.isnan(source_values)
    # source_points = source_points[valid_mask]
    # source_values = source_values[valid_mask]

    # Interpolate to regular grid using scipy.griddata
    # Options: 'linear', 'nearest', 'cubic'
    regridded_data = griddata(source_points, source_values, (target_lon_grid, target_lat_grid), method="linear")

    print(f"Regridded shape: {regridded_data.shape}")
    print(f"Regridded data range: {np.nanmin(regridded_data)} to {np.nanmax(regridded_data)}")

    # Create plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.3)

    # Plot the data
    im = ax.contourf(target_lon_grid, target_lat_grid, regridded_data, levels=20, cmap="coolwarm")
    ax.gridlines(draw_labels=True, alpha=0.3)
    plt.colorbar(im, ax=ax, orientation="vertical", pad=0.05, aspect=40, shrink=0.55, label=VARIABLE)
    plt.title(f"{MODEL_DOMAIN} {VARIABLE} Regridded to 0.1° Map", fontsize=12, fontweight="bold", pad=16)
    plt.tight_layout()

    output_filename = f"map_{MODEL_DOMAIN}_{VARIABLE}_regridded.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {output_filename}")
    plt.close()
