#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[fsspec]>=1.2.1",  # x-release-please-version
#     "matplotlib",
#     "scipy",
# ]
# ///

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
from fsspec.implementations.cached import CachingFileSystem
from omfiles import OmFileReader
from omfiles.chunk_reader import OmChunkFileReader
from omfiles.meta import OmChunksMeta
from s3fs import S3FileSystem
from scipy.spatial import cKDTree

# We load data from this Cached Fs-Spec Filesystem
FS = CachingFileSystem(
    fs=S3FileSystem(anon=True, default_block_size=256, default_cache_type="none"),
    # TODO: we'd need to verify files do not change on the remote if they still could change
    cache_check=60,
    block_size=256,
    cache_storage="cache",
    check_files=False,
)
LOCATIONS = {
    "London": (51.5074, -0.1278),
    "Paris": (48.864716, 2.349014),
}
START_DATE = np.datetime64(datetime.now(timezone.utc).date())
END_DATE = START_DATE + np.timedelta64(2, "D")
VARIABLE = "temperature_2m"
DOMAIN = "dwd_icon_global_native"


def unit_sphere(latitude, longitude):
    """Convert degrees to Cartesian coordinates on the unit sphere."""
    lat = np.deg2rad(np.asarray(latitude, dtype=np.float64))
    lon = np.deg2rad(np.asarray(longitude, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)), axis=-1)


print(f"Fetching {VARIABLE} data for {', '.join(LOCATIONS)}")
print(f"Date range: {START_DATE} to {END_DATE}")

meta = OmChunksMeta.from_s3_json_path(f"openmeteo/data/{DOMAIN}/static/meta.json", FS)
chunk_reader = OmChunkFileReader(meta, FS, f"s3://openmeteo/data/{DOMAIN}/{VARIABLE}", START_DATE, END_DATE)

with OmFileReader.from_fsspec(FS, f"s3://openmeteo/data/{DOMAIN}/static/coordinates.om") as coordinates:
    # Coordinates have shape (1, n_cells), in the same cell order as the data.
    latitude = coordinates.get_child_by_name("lat")[0, :]
    longitude = coordinates.get_child_by_name("lon")[0, :]

# Search on the unit sphere so distances work across the date line and near poles.
# Build the tree once and reuse it when querying additional locations.
tree = cKDTree(unit_sphere(latitude, longitude))

plt.figure(figsize=(12, 6))
for name, (target_latitude, target_longitude) in LOCATIONS.items():
    _, cell_index = tree.query(unit_sphere(target_latitude, target_longitude))
    print(f"{name}: cell {cell_index}, lat={latitude[cell_index]:.5f}, lon={longitude[cell_index]:.5f}")

    # Native chunk files have shape (1, n_cells, time). load_data takes (x, y).
    indices = (int(cell_index), 0)
    times, data = chunk_reader.load_data(indices)
    plt.plot(times, data, label=name, linewidth=1)

plt.title(f"{DOMAIN}: {VARIABLE}")
plt.xlabel("Time")
plt.ylabel(VARIABLE)
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()

output_filename = f"{DOMAIN}_{VARIABLE}_timeseries.png"
plt.savefig(output_filename, dpi=300)
print(f"\nPlot saved as: {output_filename}")
