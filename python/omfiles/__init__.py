"""Provides classes and utilities for reading, writing, and manipulating OM files."""

from omfiles import types
from omfiles._rust import OmFileReader, OmFileReaderAsync, OmFileWriter, OmVariable, _check_cpu_features

_check_cpu_features()

__all__ = [
    "OmFileReader",
    "OmFileReaderAsync",
    "OmFileWriter",
    "OmVariable",
    "types",
]
