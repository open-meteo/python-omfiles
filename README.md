# Python Bindings for Open Meteo File Format

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/omfiles.svg)](https://pypi.org/project/omfiles/)
[![Build and Test](https://github.com/open-meteo/python-omfiles/actions/workflows/build-test.yml/badge.svg)](https://github.com/open-meteo/python-omfiles/actions/workflows/build-test.yml)

## Features

- Read Open-Meteo (`.om`) files directly from cloud storage using Python
- Traverse the hierarchical data structure
- Arrays/array slices are returned directly as [NumPy](https://github.com/numpy/numpy) arrays
- Support for [fsspec](https://github.com/fsspec/filesystem_spec) and [xarray](https://github.com/pydata/xarray)
- Chunked data access behind the scenes

## Installation

```bash
pip install omfiles
```

### Pre-Built Wheels & Platform Support

We provide pre-built wheels for the following platforms:

- Linux x86_64 (`manylinux_2_28`)
- Linux aarch64 (`manylinux_2_28`)
- Linux musl x86_64 (`musllinux_1_2`)
- Windows x64
- macOS x86_64
- macOS ARM64 (Apple Silicon)


### Stability Notice

**This project is now stable as of version 1.0.0.**
The public API is considered stable and will follow [semantic versioning](https://semver.org/).
Breaking changes will not be introduced in 1.x releases without a major version bump.


## Reading

### Reading Files without Hierarchy

OM files are [structured like a tree of variables](https://github.com/open-meteo/om-file-format?tab=readme-ov-file#data-hierarchy-model).
The following example assumes that the file `test_file.om` contains an array variable as a root variable which has a dimensionality greater than 2 and a size of at least 2x100:

```python
from omfiles import OmFileReader

reader = OmFileReader("test_file.om")
data = reader[0:2, 0:100, ...]
reader.close() # Close the reader to release resources
```

### Reading Hierarchical Files, e.g. S3 Spatial Files

```python
import fsspec
import numpy as np
from omfiles import OmFileReader

# URI of the file on S3
s3_uri = "s3://openmeteo/data_spatial/dwd_icon/2025/09/23/0000Z/2025-09-30T0000.om"
# Create and open filesystem, wrapping it in a blockcache
backend = fsspec.open(
    f"blockcache::{s3_uri}",
    mode="rb",
    s3={"anon": True, "default_block_size": 65536},  # s3 settings
    blockcache={"cache_storage": "cache", "same_names": True},  # blockcache settings
)
# Create reader from the fsspec file object using a context manager.
# This will automatically close the file when the block is exited.
with OmFileReader(backend) as root:
    # We are at the root of the data hierarchy!
    # What type of node is this?
    print(f"root.is_array: {root.is_array}")  # False
    print(f"root.is_scalar: {root.is_scalar}")  # False
    print(f"root.is_group: {root.is_group}")  # True

    temperature_reader = root.get_child_by_name("temperature_2m")
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
```

## Writing

### Single Array
```python
import numpy as np
from omfiles import OmFileWriter

# Create sample data
data = np.random.rand(100, 100).astype(np.float32)

# Initialize writer
writer = OmFileWriter("simple.om")

# Write array with compression
array_variable = writer.write_array(
    data,
    chunks=[50, 50],
    scale_factor=1.0,
    add_offset=0.0,
    compression="pfor_delta_2d",
    name="data"
)

# Finalize the file using array_variable as entry-point
writer.close(array_variable)
```

### Hierarchical Structure
```python
import numpy as np
from omfiles import OmFileWriter

# Create sample data
features = np.random.rand(1000, 64).astype(np.float32)
labels = np.random.randint(0, 10, size=(1000,), dtype=np.int32)

# Initialize writer
writer = OmFileWriter("hierarchical.om")

# Write child arrays first
features_var = writer.write_array(features, chunks=[100, 64], name="features", compression="pfor_delta_2d")
labels_var = writer.write_array(labels, chunks=[100], name="labels")
metadata_var = writer.write_scalar(42, name="metadata")

# Create root group with children
root_var = writer.write_group(
    name="root",
    children=[features_var, labels_var, metadata_var],
)
# Finalize the file using root_var as entry-point into the hierarchy
writer.close(root_var)
```


## Development

```bash
# install the required dependencies in .venv directory
uv sync
# to run the tests
uv run pytest tests/
# to build the wheels
uv run build
# or to trigger maturin directly:
# maturin develop
```

### Tests

```bash
cargo test
```

runs rust tests.

```bash
uv run pytest tests/
```

runs Python tests.

### Python Type Stubs

Can be generated from the rust doc comments via

```bash
cargo run stub_gen
```
