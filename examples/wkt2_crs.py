#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[fsspec,proj] @ /home/fred/dev/terraputix/python-omfiles",
#     "matplotlib",
# ]
# ///

import fsspec
import matplotlib.pyplot as plt
import numpy as np
from omfiles import OmFileReader
from omfiles.om_grid import OmGrid
from pyproj import CRS

# Example: URI for a spatial data file in the `data_spatial` S3 bucket
# See data organization details: https://github.com/open-meteo/open-data?tab=readme-ov-file#data-organization
# Note: Spatial data is only retained for 7 days. The example file below may no longer exist.
# Please update the URI to match a currently available file.
# Other models to test: ncep_gfs013 cmc_gem_hrdps cmc_gem_rdps ukmo_uk_deterministic_2km dmi_harmonie_aroma_europe meteofrance_arome_france0025
s3_uri = "s3://openmeteo/data_spatial/meteoswiss_icon_ch1/2026/01/06/0000Z/2026-01-06T0000.om"

# Note: This code does not support ECMWF IFS HRES grids (Reduced Gaussian O1280)!

backend = fsspec.open(
    f"blockcache::{s3_uri}",
    mode="rb",
    s3={"anon": True, "default_block_size": 65536},
    blockcache={"cache_storage": "cache"},
)
with OmFileReader(backend) as reader:
    # Get the full data array and read into regular "data" array
    child = reader.get_child_by_name("temperature_2m")
    print("child.shape", child.shape)
    print("child.chunks", child.chunks)
    data = child[:]

    # Setup projection
    crs_wkt = reader.get_child_by_name("crs_wkt").read_scalar()
    grid = OmGrid(crs_wkt, shape=data.shape)
    # Get coordinate meshgrid for plotting
    lon_grid, lat_grid = grid.get_meshgrid()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    crs = CRS.from_wkt(crs_wkt)

    # Simple plot without cartopy
    c = ax.pcolormesh(lon_grid, lat_grid, data, cmap="coolwarm", shading="auto")

    # Add colorbar
    plt.colorbar(c, ax=ax, orientation="vertical", label="Temperature (°C)")

    # Add labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Temperature Data\nCRS: {crs.name}")

    # Set aspect ratio
    ax.set_aspect("equal")

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("temperature_map_pyproj.png", dpi=150)
    print("Plot saved to temperature_map_pyproj.png")
