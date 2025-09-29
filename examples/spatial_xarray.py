#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles==0.1.0",
#     "fsspec>=2025.7.0",
#     "s3fs",
#     "xarray",
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
from omfiles import OmFileReader

PLOT_VARIABLE = "temperature_2m"
MODEL = "dwd_icon"

s3_spatial_file = f"openmeteo/data_spatial/{MODEL}/2025/09/23/0000Z/2025-09-30T0000.om"
backend = fsspec.open(
    f"blockcache::s3://{s3_spatial_file}",
    mode="rb",
    s3={"anon": True, "default_block_size": 65536},
    blockcache={"cache_storage": "cache", "same_names": True},
)

ds = xr.open_dataset(backend, engine="om")  # type: ignore
print(ds.variables.keys())  # any of these keys can be used for plotting
ds = ds.rename_dims({"dim0": "lat", "dim1": "lon"})
# You need to know what exactly the dimensions of the specified domain is referring to.
# Icon is using a global regular grid:
# https://github.com/open-meteo/open-meteo/blob/a4cdae1ad139f9dfa6dd2552c0636c7e572dcb52/Sources/App/Icon/Icon.swift#L146
ds["lat"] = np.linspace(-90, 90, ds.sizes["lat"], endpoint=True)
ds["lon"] = np.linspace(-180, 180, num=ds.sizes["lon"], endpoint=False)

fig = plt.figure(figsize=(12, 8))
ax = ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.3)
ax.add_feature(cfeature.LAND, alpha=0.3)
ax.add_feature(cfeature.LAKES, alpha=0.3)
ax.add_feature(cfeature.RIVERS, alpha=0.3)

plot_data = ds[PLOT_VARIABLE]  # shape: (lat, lon)
lon2d, lat2d = np.meshgrid(ds["lon"].values, ds["lat"].values)

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
plt.title(f"{MODEL} {PLOT_VARIABLE}", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("xarray_map.png", dpi=300, bbox_inches="tight")
