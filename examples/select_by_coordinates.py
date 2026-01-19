#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[grids,fsspec]>=1.1.0",
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
from omfiles.chunk_reader import OmChunkFileReader
from omfiles.grids import OmGrid
from omfiles.meta import OmChunksMeta
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
domain_data: dict[str, Tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float32]]] = {}

for domain_name in DOMAINS:
    try:
        print(f"\nTrying to fetch data from domain: {domain_name}")
        meta = OmChunksMeta.from_s3_json_path(f"openmeteo/data/{domain_name}/static/meta.json", FS)
        chunk_reader = OmChunkFileReader(
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

        if grid is None:
            print(f"Grid not found for domain {domain_name}")
            continue

        indices = grid.find_point_xy(LATITUDE, LONGITUDE)
        if indices is None:
            print(f"Indices not found for domain {domain_name}")
            continue

        times, data = chunk_reader.load_data(indices)
        domain_data[domain_name] = times, data
        print(f"Successfully fetched data from {domain_name}")
    except Exception as e:
        print(f"Could not fetch data from {domain_name}: {e}")

print(f"\nSuccessfully fetched data from {len(domain_data)} domains: {domain_data.keys()}")

if len(domain_data) == 0:
    print("No data could be fetched from any domain. Exiting.")
    exit(1)

plt.figure(figsize=(12, 6))

# Plot data from each domain
for domain_name, (times, data) in domain_data.items():
    plt.plot(times, data, label=domain_name, linewidth=1)

# Enhance the plot
plt.title(f"{VARIABLE} at {LATITUDE:.2f}N, {LONGITUDE:.2f}E")
plt.xlabel("Time")
plt.ylabel(VARIABLE)
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()

# Save the figure
plt.savefig(f"{VARIABLE}_comparison.png", dpi=300)
