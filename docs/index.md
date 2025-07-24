# OM-Files Python & Rust Package

<rewrite_this>
!!! warning
    This package is under active development and may change without notice.
</rewrite_this>

Welcome to the documentation for the `omfiles` Python package, an interface for reading and writing `.om` files.

The [OM file format](https://github.com/open-meteo/om-file-format) is designed for efficient storage and access of multi-dimensional scientific data, supporting hierarchical structures, chunked arrays, and compression.
The Python bindings are powered by [PyO3](https://pyo3.rs/), leveraging the [rust-omfiles](https://github.com/open-meteo/rust-omfiles) package.

---

## Features

- **Read and write `.om` files** from Python and Rust.
- **Hierarchical data model**: groups, datasets, and attributes.
- **Chunked and compressed arrays** for efficient storage.
- **NumPy-style indexing** and async access in Python.
- **Context manager support** for safe resource handling.
- **Integration with fsspec** for remote file access.

---

## Installation

Install the Python package (requires Python 3.8+):

```bash
pip install omfiles
```

Or build from source (see [GitHub repository](https://github.com/open-meteo/python-omfiles) for details):

```bash
git clone https://github.com/open-meteo/python-omfiles.git
cd omfiles
pip install .
```

---

## Quickstart

### Reading an OM file

```python
from omfiles import OmFilePyReader

with OmFilePyReader.from_path("test_file.om") as reader:
    arr = reader[0:2, 0:100, ...]
    print(arr.shape)
```

### Writing an OM file

```python
import numpy as np
from omfiles import OmFilePyWriter

# Create sample data
data = np.random.rand(100, 100).astype(np.float32)

# Initialize writer
writer = OmFilePyWriter("simple.om")

# Write array with compression
variable = writer.write_array(
    data,
    chunks=[50, 50],
    scale_factor=1.0,
    add_offset=0.0,
    compression="pfor_delta_2d",
    name="data"
)

# Finalize the file. This writes the trailer and flushes the buffers.
writer.close(variable)
```

---

## API Reference

See the [API documentation](api.md) for details on all classes and methods.

---

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/open-meteo/python-omfiles) for guidelines and issue tracking.

---

## License

OM-Files is released under the GPLv2 license.

---

## Acknowledgements

- [PyO3](https://pyo3.rs/)
- [NumPy](https://numpy.org/)
- [fsspec](https://filesystem-spec.readthedocs.io/en/latest/)
