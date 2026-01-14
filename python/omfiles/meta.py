"""Representation of Open-Meteo meta.json files."""

import json
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from omfiles.grids import OmGrid

from omfiles import OmFileReader

try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python < 3.11


import fsspec
import numpy as np

EPOCH = np.datetime64(0, "s")


@dataclass
class OmMetaBase:
    """Base class for Open-Meteo metadata."""

    crs_wkt: str  # Coordinate Reference System in Well-Known Text format

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create instance from dictionary, ignoring extra keys."""
        # fields(cls) correctly identifies fields in the subclass
        class_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in class_fields}
        return cls(**filtered_data)

    @classmethod
    def from_metajson_string(cls, metajson_str: str) -> Self:
        """Create instance from metajson string."""
        return cls.from_dict(json.loads(metajson_str))

    @classmethod
    def from_s3_json_path(cls, s3_json_path: str, fs: fsspec.AbstractFileSystem) -> Self:
        """Create instance from S3 JSON path."""
        meta_dict = json.loads(fs.cat_file(s3_json_path))
        return cls.from_dict(meta_dict)

    def get_grid(self, reader: OmFileReader) -> "OmGrid":
        """Create grid from metadata."""
        try:
            from omfiles.grids import OmGrid
        except ImportError:
            raise ImportError("omfiles[grids] is required for grid operations")
        """Create grid from metadata."""
        if len(reader.shape) == 2:
            return OmGrid(self.crs_wkt, reader.shape)
        elif len(reader.shape) == 3:
            return OmGrid(self.crs_wkt, reader.shape[:2])
        else:
            raise ValueError("Reader shape must be 2D or 3D")


@dataclass
class OmSpatialMeta(OmMetaBase):
    """Representation of the meta.json for spatial datasets."""

    last_modified_time: str  # ISO8601 for last modification
    reference_time: str  # ISO8601 for reference time
    valid_times: List[str]  # List of valid times in ISO8601 format
    variables: List[str]  # List of variables in the dataset


@dataclass
class OmChunksMeta(OmMetaBase):
    """Representation of the meta.json for time oriented chunks."""

    chunk_time_length: int  # Number of time steps per chunk (file_length)
    data_end_time: int  # Unix timestamp for when data ends
    last_run_availability_time: int  # Unix timestamp for last availability
    last_run_initialisation_time: int  # Unix timestamp for last initialization
    last_run_modification_time: int  # Unix timestamp for last modification
    temporal_resolution_seconds: int  # Time resolution in seconds
    update_interval_seconds: int  # How often data is updated

    def time_to_chunk_index(self, timestamp: np.datetime64) -> int:
        """
        Convert a timestamp to a chunk index.

        This depends on the file_length and the temporal_resolution_seconds of the domain.

        Args:
            timestamp (np.datetime64): The timestamp to convert.

        Returns:
            int: The chunk index containing the timestamp.
        """
        seconds_since_epoch = (timestamp - EPOCH) / np.timedelta64(1, "s")
        chunk_index = int(seconds_since_epoch / (self.chunk_time_length * self.temporal_resolution_seconds))
        return chunk_index

    def chunks_for_date_range(
        self,
        start_timestamp: np.datetime64,
        end_timestamp: np.datetime64,
    ) -> List[int]:
        """
        Find all chunk indices that contain data within the given date range.

        Args:
            start_timestamp (np.datetime64): Start timestamp for the data range.
            end_timestamp (np.datetime64): End timestamp for the data range.

        Returns:
            List[int]: List of chunk indices containing data within the date range.
        """
        # Get chunk indices for start and end dates
        start_chunk = self.time_to_chunk_index(start_timestamp)
        end_chunk = self.time_to_chunk_index(end_timestamp)

        # Generate list of all chunks between start and end (inclusive)
        return list(range(start_chunk, end_chunk + 1))

    def get_chunk_time_range(self, chunk_index: int):
        """
        Get the time range covered by a specific chunk.

        Args:
            chunk_index (int): Index of the chunk.

        Returns:
            np.ndarray: Array of datetime64 objects representing the time points in the chunk.
        """
        chunk_start_seconds = chunk_index * self.chunk_time_length * self.temporal_resolution_seconds
        start_time = EPOCH + np.timedelta64(chunk_start_seconds, "s")

        # Generate timestamps at regular intervals from the start time
        time_delta = np.timedelta64(self.temporal_resolution_seconds, "s")
        # Note: better type inference via list comprehension here
        timestamps = np.array([start_time + i * time_delta for i in range(self.chunk_time_length)])
        return timestamps
