#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[fsspec]>=1.1.0",
#     "matplotlib",
#     "cartopy",
#     "earthkit-regrid==0.5.0",
# ]
# ///

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import fsspec
import matplotlib.pyplot as plt
import numpy as np
from earthkit.regrid import interpolate
from omfiles import OmFileReader

MODEL_DOMAIN = "ecmwf_ifs"
VARIABLE = "temperature_2m"
# Example: URI for a spatial data file in the `data_spatial` S3 bucket
# See data organization details: https://github.com/open-meteo/open-data?tab=readme-ov-file#data-organization
# Note: Spatial data is only retained for 7 days. The example file below may no longer exist.
# Please update the URI to match a currently available file.
S3_URI = f"s3://openmeteo/data_spatial/{MODEL_DOMAIN}/2026/01/10/0000Z/2026-01-12T0000.om"

backend = fsspec.open(
    f"blockcache::{S3_URI}",
    mode="rb",
    s3={"anon": True, "default_block_size": 65536},
    blockcache={"cache_storage": "cache"},
)
with OmFileReader(backend) as reader:
    print("reader.is_group", reader.is_group)

    child = reader.get_child_by_name(VARIABLE)
    print("child.name", child.name)

    # Get the full data array
    print("child.shape", child.shape)
    print("child.chunks", child.chunks)
    data = child[:]
    print(f"Data shape: {data.shape}")
    print(f"Data range: {np.nanmin(data)} to {np.nanmax(data)}")

    # We are using earthkit-regrid for regridding: https://earthkit-regrid.readthedocs.io/en/stable/interpolate.html#interpolate
    # with linear interpolation. Nearest neighbor interpolation can be obtained with`method="nearest-neighbour"`
    regridded = interpolate(data, in_grid={"grid": "O1280"}, out_grid={"grid": [0.1, 0.1]}, method="linear")
    print(f"Regridded shape: {regridded.shape}")

    # Create plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.3)

    # Create coordinate arrays
    # These bounds need to match the output grid of the regridding!
    height, width = regridded.shape
    lon = np.linspace(0, 360, width, endpoint=False)
    lat = np.linspace(90, -90, height)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Plot the data
    im = ax.contourf(lon_grid, lat_grid, regridded, levels=20, cmap="coolwarm")
    ax.set_global()
    ax.gridlines(draw_labels=True, alpha=0.3)
    plt.colorbar(im, ax=ax, orientation="vertical", pad=0.05, aspect=40, shrink=0.55, label=VARIABLE)
    plt.title(f"{MODEL_DOMAIN} {VARIABLE} Regridded to 0.1° Map", fontsize=12, fontweight="bold", pad=16)
    plt.tight_layout()

    output_filename = f"map_{MODEL_DOMAIN}_{VARIABLE}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {output_filename}")
    plt.close()
