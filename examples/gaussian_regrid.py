#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles>=1.0.0",
#     "fsspec>=2025.7.0",
#     "s3fs",
#     "matplotlib",
#     "cartopy",
#     "earthkit-regrid",
# ]
# ///

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import fsspec
import matplotlib.pyplot as plt
import numpy as np
from earthkit.regrid import interpolate
from omfiles import OmFileReader

ifs_spatial_file = f"openmeteo/data_spatial/ecmwf_ifs/2025/10/01/0000Z/2025-10-01T0000.om"
backend = fsspec.open(
    f"blockcache::s3://{ifs_spatial_file}",
    mode="rb",
    s3={"anon": True, "default_block_size": 65536},
    blockcache={"cache_storage": "cache", "same_names": True},
)
with OmFileReader(backend) as reader:
    print("reader.is_group", reader.is_group)

    child = reader.get_child_by_name("temperature_2m")
    print("child.name", child.name)

    # Get the full data array
    print("child.shape", child.shape)
    print("child.chunks", child.chunks)
    data = child[:]
    print(f"Data shape: {data.shape}")
    print(f"Data range: {np.nanmin(data)} to {np.nanmax(data)}")

    regridded = interpolate(data, in_grid={"grid": "O1280"}, out_grid={"grid": [0.1, 0.1]}, method="linear")
    print(regridded.shape)

    # Create plot
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
    height, width = regridded.shape
    lon = np.linspace(0, 360, width, endpoint=False)  # Adjust these bounds
    lat = np.linspace(90, -90, height)  # Adjust these bounds
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Plot the data
    im = ax.contourf(lon_grid, lat_grid, regridded, levels=20, transform=ccrs.PlateCarree(), cmap="viridis")
    plt.colorbar(im, ax=ax, shrink=0.6, label=child.name)
    ax.gridlines(draw_labels=True, alpha=0.3)
    plt.title(f"2D Map: {child.name}")
    ax.set_global()
    plt.tight_layout()

    output_filename = f"map_{child.name.replace('/', '_')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {output_filename}")
    plt.close()
