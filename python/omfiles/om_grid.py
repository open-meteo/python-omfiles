"""An OmGrid provides utilities to transform between geographic coordinates and grid indices."""

import json
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pyproj import CRS, Transformer

from omfiles.grids.gaussian import GaussianGrid
from omfiles.grids.regular import RegularGrid

EPOCH = np.datetime64(0, "s")


@dataclass
class OmMetaJson:
    """Class to decode Open-Meteo metadata JSON files."""

    chunk_time_length: int  # Number of time steps per chunk (file_length)
    crs_wkt: str  # Coordinate Reference System in Well-Known Text format
    # data_end_time: int  # Unix timestamp for when data ends
    # last_run_availability_time: int  # Unix timestamp for last availability
    # last_run_initialisation_time: int  # Unix timestamp for last initialization
    # last_run_modification_time: int  # Unix timestamp for last modification
    temporal_resolution_seconds: int  # Time resolution in seconds
    # update_interval_seconds: int  # How often data is updated

    @classmethod
    def from_dict(cls, data: dict) -> "OmMetaJson":
        """Create instance from dictionary, ignoring extra keys."""
        # Get the names of all fields defined in the dataclass
        class_fields = {f.name for f in fields(cls)}

        # Filter the input dictionary
        filtered_data = {k: v for k, v in data.items() if k in class_fields}

        return cls(**filtered_data)

    @classmethod
    def from_metajson_string(cls, metajson_str: str) -> "OmMetaJson":
        """Create instance from metajson string."""
        return cls.from_dict(json.loads(metajson_str))

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


def _is_gaussian_grid(crs_wkt: str) -> bool:
    """Check if WKT string represents a Gaussian grid."""
    return "Reduced Gaussian Grid" in crs_wkt or "Gaussian Grid" in crs_wkt


class OmGrid:
    """Wrapper for grid implementations - automatically delegates to appropriate grid type."""

    def __init__(self, crs_wkt: str, shape: Tuple[int, int]):
        """
        Initialize grid from WKT projection string and data shape.

        Args:
            crs_wkt: Coordinate Reference System in Well-Known Text format
            shape: Grid shape as (ny, nx)
        """
        # Detect grid type and create appropriate implementation
        if _is_gaussian_grid(crs_wkt):
            self._grid = GaussianGrid(crs_wkt, shape)
        else:
            self._grid = RegularGrid(crs_wkt, shape)

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as (ny, nx)."""
        return self._grid.shape

    @property
    def latitude(self) -> npt.NDArray[np.float64]:
        """Get array of latitude coordinates for all grid points."""
        return self._grid.latitude

    @property
    def longitude(self) -> npt.NDArray[np.float64]:
        """Get array of longitude coordinates for all grid points."""
        return self._grid.longitude

    def find_point_xy(self, lat: float, lon: float) -> Optional[Tuple[int, int]]:
        """Find grid point indices for given lat/lon coordinates."""
        return self._grid.find_point_xy(lat, lon)

    def get_coordinates(self, x: int, y: int) -> Tuple[float, float]:
        """Get lat/lon coordinates for given grid point indices."""
        return self._grid.get_coordinates(x, y)

    def get_meshgrid(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get meshgrid of coordinates."""
        return self._grid.get_meshgrid()

    @property
    def is_gaussian(self) -> bool:
        """Check if this is a Gaussian grid."""
        return isinstance(self._grid, GaussianGrid)

    @property
    def crs(self) -> CRS | None:
        """Get the Coordinate Reference System."""
        if isinstance(self._grid, GaussianGrid):
            return None
        return self._grid.crs
