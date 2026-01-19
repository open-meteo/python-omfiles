#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[fsspec,grids]>=1.1.0",
#     "matplotlib",
#     "cartopy",
# ]
# ///

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import fsspec
import matplotlib.pyplot as plt
import numpy as np
from omfiles import OmFileReader
from omfiles.grids import OmGrid

MODEL_DOMAIN = "dmi_harmonie_arome_europe"
VARIABLE = "relative_humidity_2m"
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
    print("child.shape", child.shape)
    print("child.chunks", child.chunks)
    # Get the full data array
    data = child[:]
    print(f"Data shape: {data.shape}")
    print(f"Data range: {np.nanmin(data)} to {np.nanmax(data)}")

    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.3)

    # Create coordinate arrays
    num_y, num_x = data.shape
    grid = OmGrid(reader.get_child_by_name("crs_wkt").read_scalar(), (num_y, num_x))
    lon_grid, lat_grid = grid.get_meshgrid()
    crs = grid.crs
    if crs is None:
        raise ValueError("CRS is None, this should only happen for gaussian grids")

    # Plot the data
    im = ax.contourf(lon_grid, lat_grid, data, cmap="coolwarm")
    ax.gridlines(draw_labels=True, alpha=0.3)
    ax.set_aspect("equal")
    # ax.set_global()
    plt.colorbar(im, ax=ax, orientation="vertical", pad=0.05, aspect=40, shrink=0.55, label=VARIABLE)
    plt.title(f"{MODEL_DOMAIN} {VARIABLE} Map\nCRS: {crs.name}", fontsize=12, fontweight="bold", pad=16)
    plt.tight_layout()

    output_filename = f"map_{VARIABLE}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {output_filename}")
    plt.close()
