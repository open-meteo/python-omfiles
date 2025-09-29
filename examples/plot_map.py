#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles==0.1.0",
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

# The following two incantations are equivalent
#
# from fsspec.implementations.cached import CachingFileSystem
# from s3fs import S3FileSystem
# s3_path = "s3://openmeteo/data_spatial/dwd_icon/2025/09/23/0000Z/2025-09-30T0000.om"
# s3_fs = S3FileSystem(anon=True, default_block_size=65536, default_cache_type="none")
# backend = CachingFileSystem(
#     fs=s3_fs, cache_check=3600, block_size=65536, cache_storage="cache", check_files=False, same_names=True
# )
# with OmFileReader.from_fsspec(backend, s3_path) as reader:

backend = fsspec.open(
    f"blockcache::s3://openmeteo/data_spatial/dwd_icon/2025/09/23/0000Z/2025-09-30T0000.om",
    mode="rb",
    s3={"anon": True, "default_block_size": 65536},
    blockcache={"cache_storage": "cache", "same_names": True},
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
    # Currently, the files don't contain any information about the spatial coordinates,
    # so you need to provide these coordinate arrays manually.
    height, width = data.shape
    lon = np.linspace(-180, 180, width)  # Adjust these bounds
    lat = np.linspace(-90, 90, height)  # Adjust these bounds
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Plot the data
    im = ax.contourf(lon_grid, lat_grid, data, levels=20, transform=ccrs.PlateCarree(), cmap="viridis")
    plt.colorbar(im, ax=ax, shrink=0.6, label=child.name)
    ax.gridlines(draw_labels=True, alpha=0.3)
    plt.title(f"2D Map: {child.name}")
    ax.set_global()
    plt.tight_layout()

    output_filename = f"map_{child.name.replace('/', '_')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {output_filename}")
    plt.close()
