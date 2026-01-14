#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[fsspec,grids,xarray]>=1.1.0",
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
from omfiles.grids import OmGrid

MODEL_DOMAIN = "dwd_icon"
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

ds = xr.open_dataset(backend, engine="om")  # type: ignore
print(ds.attrs)
print(ds.variables.keys())  # any of these keys can be used for plotting

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.3)
ax.add_feature(cfeature.LAND, alpha=0.3)

data = ds[VARIABLE]  # shape: (lat, lon)
# Use OmGrid with the crs_wkt attribute to get the lat/lon grid
grid = OmGrid(ds.attrs["crs_wkt"], ds[VARIABLE].shape)
lon2d, lat2d = grid.get_meshgrid()

min = int(data.min().values)
max = int(data.max().values)
stepsize = int((max - min) / 30)

im = ax.contourf(
    lon2d,
    lat2d,
    data,
    levels=np.arange(min, max, stepsize),
    cmap="Spectral_r",
    vmin=min,
    vmax=max,
    extend="both",
)
ax.gridlines(draw_labels=True, alpha=0.3)
plt.colorbar(im, ax=ax, orientation="vertical", pad=0.05, aspect=40, shrink=0.55, label=VARIABLE)
plt.title(f"{MODEL_DOMAIN} {VARIABLE}", fontsize=12, fontweight="bold", pad=16)
plt.tight_layout()
plt.savefig("xarray_map.png", dpi=300, bbox_inches="tight")
