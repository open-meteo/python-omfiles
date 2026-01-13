"""An OmGrid provides utilities to transform between geographic coordinates and grid indices."""

import json
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pyproj import CRS, Transformer

from omfiles._utils import EPOCH


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


class OmGrid:
    """Latitude and longitude grid based on cartopy and wkt projection strings."""

    # lon_grid: npt.NDArray[np.float64]
    # lat_grid: npt.NDArray[np.float64]

    def __init__(self, crs_wkt: str, shape: Tuple[int, int]):
        """
        Initialize grid from WKT projection string and data shape.

        Args:
            crs_wkt: Coordinate Reference System in Well-Known Text format
            shape: Grid shape as (ny, nx) - number of points in y and x directions
        """
        self.crs = CRS.from_wkt(crs_wkt)
        self.wgs84 = CRS.from_epsg(4326)
        self.ny, self.nx = shape

        # TODO: Special case for gaussian grids!

        # Transformers for coordinate conversions
        self.to_projection = Transformer.from_crs(self.wgs84, self.crs, always_xy=True)
        self.to_wgs84 = Transformer.from_crs(self.crs, self.wgs84, always_xy=True)

        # Get projection bounds from area of use
        area = self.crs.area_of_use
        if area is None:
            raise ValueError("CRS does not have an area of use defined")

        # Transform WGS84 bounds to projection space
        xmin, ymin = self.to_projection.transform(area.west, area.south)
        xmax, ymax = self.to_projection.transform(area.east, area.north)

        self.bounds = (xmin, xmax, ymin, ymax)
        self.origin = (xmin, ymin)

        if self.nx <= 1 or self.ny <= 1:
            raise ValueError("Invalid grid shape")

        # Calculate grid spacing
        self.dx = (xmax - xmin) / (self.nx - 1)
        self.dy = (ymax - ymin) / (self.ny - 1)

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as (ny, nx)."""
        return (self.ny, self.nx)

    @property
    def latitude(self) -> npt.NDArray[np.float64]:
        """
        Get 2D array of latitude coordinates for all grid points.

        Returns:
            Array of shape (ny, nx) with latitude values
        """
        if not hasattr(self, "_latitude"):
            self._compute_coordinates()
        return self._latitude

    @property
    def longitude(self) -> npt.NDArray[np.float64]:
        """
        Get 2D array of longitude coordinates for all grid points.

        Returns:
            Array of shape (ny, nx) with longitude values
        """
        if not hasattr(self, "_longitude"):
            self._compute_coordinates()
        return self._longitude

    def _compute_coordinates(self) -> None:
        """Compute and cache latitude/longitude arrays for all grid points."""
        # Create meshgrid of projection coordinates
        x_coords = np.linspace(self.origin[0], self.origin[0] + self.dx * (self.nx - 1), self.nx)
        y_coords = np.linspace(self.origin[1], self.origin[1] + self.dy * (self.ny - 1), self.ny)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Transform to WGS84
        lon_grid, lat_grid = self.to_wgs84.transform(x_grid, y_grid)

        self._longitude = lon_grid
        self._latitude = lat_grid

    def find_point_xy(self, lat: float, lon: float) -> Optional[Tuple[int, int]]:
        """
        Find grid point indices (x, y) for given lat/lon coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            (x, y) grid indices if point is in grid bounds, None otherwise
        """
        # Transform to projection coordinates
        x_proj, y_proj = self.to_projection.transform(lon, lat)

        # Calculate grid indices
        x_idx = int(round((x_proj - self.origin[0]) / self.dx))
        y_idx = int(round((y_proj - self.origin[1]) / self.dy))

        # Validate indices
        if not (0 <= x_idx < self.nx and 0 <= y_idx < self.ny):
            return None

        return (x_idx, y_idx)

    def get_coordinates(self, x: int, y: int) -> Tuple[float, float]:
        """
        Get lat/lon coordinates for given grid point indices.

        Args:
            x: Grid x index
            y: Grid y index

        Returns:
            (latitude, longitude) in degrees
        """
        # Calculate projection coordinates
        x_proj = self.origin[0] + x * self.dx
        y_proj = self.origin[1] + y * self.dy

        # Transform to WGS84
        lon, lat = self.to_wgs84.transform(x_proj, y_proj)

        return (float(lat), float(lon))

    def get_meshgrid(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get meshgrid of projection coordinates.

        Useful for plotting with matplotlib/cartopy.

        Returns:
            (lon_grid, lat_grid) arrays of shape (ny, nx) in projection coordinates
        """
        x_coords: npt.NDArray[np.float64] = np.linspace(
            self.origin[0], self.origin[0] + self.dx * (self.nx - 1), self.nx
        )
        y_coords: npt.NDArray[np.float64] = np.linspace(
            self.origin[1], self.origin[1] + self.dy * (self.ny - 1), self.ny
        )
        return np.meshgrid(x_coords, y_coords)
