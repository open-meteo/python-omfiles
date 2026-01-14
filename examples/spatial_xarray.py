#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[grids,fsspec,xarray]@ /home/fred/dev/terraputix/python-omfiles",
#     "matplotlib",
#     "cartopy",
# ]
# ///

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from omfiles.om_grid import OmGrid

PLOT_VARIABLE = "temperature_2m"
MODEL_DOMAIN = "dwd_icon"

# Example: URI for a spatial data file in the `data_spatial` S3 bucket
# See data organization details: https://github.com/open-meteo/open-data?tab=readme-ov-file#data-organization
# Note: Spatial data is only retained for 7 days. The example file below may no longer exist.
# Please update the URI to match a currently available file.
s3_run = f"s3://openmeteo/data_spatial/{MODEL_DOMAIN}/2026/01/10/0000Z/"
s3_uri = f"{s3_run}2026-01-12T0000.om"

backend = fsspec.open(
    f"blockcache::{s3_uri}",
    mode="rb",
    s3={"anon": True, "default_block_size": 65536},
    blockcache={"cache_storage": "cache"},
)

ds = xr.open_dataset(backend, engine="om")  # type: ignore
print(ds.attrs)
print(ds.variables.keys())  # any of these keys can be used for plotting

fig = plt.figure(figsize=(12, 8))
ax = ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.3)
ax.add_feature(cfeature.LAND, alpha=0.3)
ax.add_feature(cfeature.LAKES, alpha=0.3)
ax.add_feature(cfeature.RIVERS, alpha=0.3)

plot_data = ds[PLOT_VARIABLE]  # shape: (lat, lon)
# Use OmGrid with the crs_wkt attribute to get the lat/lon grid
grid = OmGrid(ds.attrs["crs_wkt"], ds[PLOT_VARIABLE].shape)
lon2d, lat2d = grid.get_meshgrid()

min = int(plot_data.min().values)
max = int(plot_data.max().values)
stepsize = int((max - min) / 30)

c = ax.contourf(
    lon2d,
    lat2d,
    plot_data,
    levels=np.arange(min, max, stepsize),
    cmap="Spectral_r",  # or "RdYlBu_r"
    vmin=min,
    vmax=max,
    transform=ccrs.PlateCarree(),
    extend="both",
)
cb = plt.colorbar(c, ax=ax, orientation="vertical", pad=0.02, aspect=40, shrink=0.8)
cb.set_label(PLOT_VARIABLE, fontsize=14)
plt.title(f"{MODEL_DOMAIN} {PLOT_VARIABLE}", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("xarray_map.png", dpi=300, bbox_inches="tight")
