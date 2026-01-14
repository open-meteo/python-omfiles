#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[grids,fsspec] @ /home/fred/dev/terraputix/python-omfiles",
#     "matplotlib",
# ]
# ///

from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from fsspec.implementations.cached import CachingFileSystem
from omfiles import OmFileReader
from omfiles.chunk_reader import OmFileChunkReader
from omfiles.om_grid import OmGrid
from omfiles.om_meta import OmMetaChunks
from s3fs import S3FileSystem

# We load data from this Cached Fs-Spec Filesystem
FS = CachingFileSystem(
    fs=S3FileSystem(anon=True, default_block_size=256, default_cache_type="none"),
    # TODO: we'd need to verify files do not change on the remote if they still could change
    cache_check=60,
    block_size=256,
    cache_storage="cache",
    check_files=False,
)
LATITUDE, LONGITUDE = 48.864716, 2.349014  # Paris
START_DATE = np.datetime64(datetime(2025, 4, 25, 12, 0))  # 25-04-2025'T'12:00
END_DATE = np.datetime64(datetime(2025, 5, 18, 12, 0))  # 18-05-2025'T'12:00
VARIABLE = "temperature_2m"
DOMAINS = [
    "dwd_icon",
    "dwd_icon_eu",
    "dwd_icon_d2",
    "ecmwf_ifs025",
    "ecmwf_ifs",
    "meteofrance_arpege_europe",
    "meteofrance_arpege_world025",
    "meteofrance_arome_france0025",
    "meteofrance_arome_france_hd",
    "meteofrance_arome_france_hd_15min",
    "cmc_gem_gdps",
    "cmc_gem_rdps",
    "cmc_gem_hrdps",
]

print(f"Fetching {VARIABLE} data for coordinates: {LATITUDE}N, {LONGITUDE}E")
print(f"Date range: {START_DATE} to {END_DATE}")


# Collect data from each domain
domain_data: dict[str, Tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]] = {}

for domain_name in DOMAINS:
    try:
        print(f"\nTrying to fetch data from domain: {domain_name}")
        meta = OmMetaChunks.from_s3_json_path(f"openmeteo/data/{domain_name}/static/meta.json", FS)
        chunk_reader = OmFileChunkReader(
            meta, FS, f"s3://openmeteo/data/{domain_name}/{VARIABLE}", START_DATE, END_DATE
        )
        first = next(chunk_reader.iter_files(), None)
        if first is None:
            print(f"No data found for domain {domain_name}")
            continue
        _, s3_path = first
        grid: OmGrid | None = None
        with OmFileReader.from_fsspec(FS, s3_path) as reader:
            grid = meta.get_grid(reader)

        assert grid is not None, "Grid not found"
        indices = grid.find_point_xy(LATITUDE, LONGITUDE)
        assert indices is not None, "Indices not found"
        times, data = chunk_reader.load_chunked_data(indices)
        domain_data[domain_name] = times, data
        print(f"Successfully fetched data from {domain_name}")
    except Exception as e:
        print(f"Could not fetch data from {domain_name}: {e}")

print(f"\nSuccessfully fetched data from {len(domain_data)} domains: {domain_data.keys()}")

if len(domain_data) == 0:
    print("No data could be fetched from any domain. Exiting.")
    exit(1)

# Domain colors for consistent line colors
colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(domain_data)))

plt.figure(figsize=(12, 6))

# Plot data from each domain
for i, (domain_name, (times, data)) in enumerate(domain_data.items()):
    plt.plot(times, data, label=domain_name, color=colors[i], linewidth=2)

# Enhance the plot
plt.title(f"{VARIABLE.replace('_', ' ').title()} at {LATITUDE:.2f}N, {LONGITUDE:.2f}E")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)" if VARIABLE == "temperature_2m" else VARIABLE)
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()

# Save and show the figure
plt.savefig(f"{VARIABLE}_comparison.png", dpi=150)
plt.show()
