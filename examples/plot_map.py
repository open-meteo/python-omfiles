#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[grids] @ /home/fred/dev/terraputix/python-omfiles",
#     "fsspec>=2025.7.0",
#     "s3fs",
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
from omfiles.om_grid import OmGrid

MODEL_DOMAIN = "dmi_harmonie_arome_europe"
# Example: URI for a spatial data file in the `data_spatial` S3 bucket
# See data organization details: https://github.com/open-meteo/open-data?tab=readme-ov-file#data-organization
# Note: Spatial data is only retained for 7 days. The example file below may no longer exist.
# Please update the URI to match a currently available file.
s3_uri = f"s3://openmeteo/data_spatial/{MODEL_DOMAIN}/2026/01/10/0000Z/2026-01-12T0000.om"

# The following two incantations are equivalent
#
# from fsspec.implementations.cached import CachingFileSystem
# from s3fs import S3FileSystem
# s3_fs = S3FileSystem(anon=True, default_block_size=65536, default_cache_type="none")
# backend = CachingFileSystem(
#     fs=s3_fs, cache_check=3600, block_size=65536, cache_storage="cache", check_files=False, same_names=True
# )
# with OmFileReader.from_fsspec(backend, s3_uri) as reader:

backend = fsspec.open(
    f"blockcache::{s3_uri}",
    mode="rb",
    s3={"anon": True, "default_block_size": 65536},
    blockcache={"cache_storage": "cache"},
)
with OmFileReader(backend) as reader:
    print("reader.is_group", reader.is_group)

    child = reader.get_child_by_name("relative_humidity_2m")
    print("child.name", child.name)

    # Get the full data array
    print("child.shape", child.shape)
    print("child.chunks", child.chunks)
    data = child[:]
    print(f"Data shape: {data.shape}")
    print(f"Data range: {np.nanmin(data)} to {np.nanmax(data)}")

    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())  # use PlateCarree projection

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.3)

    # Create coordinate arrays
    num_y, num_x = child.shape
    grid = OmGrid(reader.get_child_by_name("crs_wkt").read_scalar(), (num_y, num_x))
    lon_grid, lat_grid = grid.get_meshgrid()

    # Plot the data
    im = ax.contourf(lon_grid, lat_grid, data, levels=20, transform=ccrs.PlateCarree(), cmap="viridis")
    plt.colorbar(im, ax=ax, shrink=0.6, label=child.name)
    ax.gridlines(draw_labels=True, alpha=0.3)
    plt.title(f"2D Map: {child.name}")
    # ax.set_global()
    plt.tight_layout()

    output_filename = f"map_{child.name.replace('/', '_')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {output_filename}")
    plt.close()
