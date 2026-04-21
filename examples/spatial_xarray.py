#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[fsspec,grids,xarray] @ /home/fred/dev/terraputix/python-omfiles",  # x-release-please-version
#     "matplotlib",
#     "cartopy",
# ]
# ///

import datetime as dt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from omfiles.grids import OmGrid

MODEL_DOMAIN = "dwd_icon"
VARIABLE = ""

# Example: URI for a spatial data file in the `data_spatial` S3 bucket
# See data organization details: https://github.com/open-meteo/open-data?tab=readme-ov-file#data-organization
# Note: Spatial data is only retained for 7 days. The script uses one file within this period.
date_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)
S3_URI = (
    f"s3://openmeteo/data_run/{MODEL_DOMAIN}/{date_time.year}/{date_time.month:02}/{date_time.day:02}/0000Z/rain.om"
)
print(f"Using om file: {S3_URI}")

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

data = ds[VARIABLE][:, :, 2]  # shape: (lat, lon)
# Use OmGrid with the crs_wkt attribute to get the lat/lon grid
grid = OmGrid(ds.attrs["crs_wkt"], data.shape)
lon2d, lat2d = grid.get_meshgrid()

min_val = int(data.min().values)
max_val = int(data.max().values)
stepsize = int((max_val - min_val) / min(max_val - min_val, 30))

im = ax.contourf(
    lon2d,
    lat2d,
    data,
    levels=np.arange(min_val, max_val, stepsize),
    cmap="Spectral_r",
    vmin=min_val,
    vmax=max_val,
    extend="both",
)
ax.gridlines(draw_labels=True, alpha=0.3)
plt.colorbar(im, ax=ax, orientation="vertical", pad=0.05, aspect=40, shrink=0.55, label=VARIABLE)
plt.title(f"{MODEL_DOMAIN} {VARIABLE}", fontsize=12, fontweight="bold", pad=16)
plt.tight_layout()
plt.savefig("xarray_map.png", dpi=300, bbox_inches="tight")
