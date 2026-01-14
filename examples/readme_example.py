#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "omfiles[fsspec]>=1.1.0",
# ]
# ///

import fsspec
import numpy as np
from omfiles import OmFileReader

MODEL_DOMAIN = "dwd_icon"
VARIABLE = "temperature_2m"
# Example: URI for a spatial data file in the `data_spatial` S3 bucket
# See data organization details: https://github.com/open-meteo/open-data?tab=readme-ov-file#data-organization
# Note: Spatial data is only retained for 7 days. The example file below may no longer exist.
# Please update the URI to match a currently available file.
S3_URI = f"s3://openmeteo/data_spatial/{MODEL_DOMAIN}/2026/01/10/0000Z/2026-01-12T0000.om"

# Create and open filesystem, wrapping it in a blockcache
backend = fsspec.open(
    f"blockcache::{S3_URI}",
    mode="rb",
    s3={"anon": True, "default_block_size": 65536},  # s3 settings
    blockcache={"cache_storage": "cache"},  # blockcache settings
)
# Create reader from the fsspec file object using a context manager.
# This will automatically close the file when the block is exited.
with OmFileReader(backend) as root:
    # We are at the root of the data hierarchy!
    # What type of node is this?
    print(f"root.is_array: {root.is_array}")  # False
    print(f"root.is_scalar: {root.is_scalar}")  # False
    print(f"root.is_group: {root.is_group}")  # True

    temperature_reader = root.get_child_by_name(VARIABLE)
    print(f"temperature_reader.is_array: {temperature_reader.is_array}")  # True
    print(f"temperature_reader.is_scalar: {temperature_reader.is_scalar}")  # False
    print(f"temperature_reader.is_group: {temperature_reader.is_group}")  # False

    # What shape does the stored array have?
    print(f"temperature_reader.shape: {temperature_reader.shape}")  # (1441, 2879)

    # Read all data from the array
    temperature_data = temperature_reader.read_array((...))
    print(f"temperature_data.shape: {temperature_data.shape}")  # (1441, 2879)

    # It's also possible to read any subset of the array
    temperature_data_subset1 = temperature_reader.read_array((slice(0, 10), slice(0, 10)))
    print(temperature_data_subset1)
    print(f"temperature_data_subset1.shape: {temperature_data_subset1.shape}")  # (10, 10)

    # Numpy basic indexing is supported for direct access if the reader is an array.
    temperature_data_subset2 = temperature_reader[0:10, 0:10]
    print(temperature_data_subset2)
    print(f"temperature_data_subset2.shape: {temperature_data_subset2.shape}")  # (10, 10)

    # Compare the two temperature subsets and verify that they are the same
    are_equal = np.array_equal(temperature_data_subset1, temperature_data_subset2, equal_nan=True)
    print(f"Are the two temperature subsets equal? {are_equal}")
