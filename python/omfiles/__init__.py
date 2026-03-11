"""Provides classes and utilities for reading, writing, and manipulating OM files."""

from . import types
from ._rust import OmFileReader, OmFileReaderAsync, OmFileWriter, OmVariable, _check_cpu_features

_check_cpu_features()

__all__ = [
    "OmFileReader",
    "OmFileReaderAsync",
    "OmFileWriter",
    "OmVariable",
    "types",
]

try:
    from .xarray import write_dataset

    __all__.append("write_dataset")
except ImportError:
    pass
