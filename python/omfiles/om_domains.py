from abc import ABC, abstractmethod
from datetime import datetime
from functools import cached_property
from typing import List, Optional, Tuple

import numpy as np

from omfiles.utils import EPOCH, _modulo_positive


class AbstractGrid(ABC):
    """
    Abstract base class for weather model grid definitions.

    This defines the interface that all grid implementations must follow.
    """

    @property
    @abstractmethod
    def grid_type(self) -> str:
        """Return the grid type identifier."""
        pass

    @cached_property
    @abstractmethod
    def latitude(self) -> np.ndarray:
        """
        Return the latitude coordinates array.

        Uses cached_property to ensure the array is computed only once.
        """
        pass

    @cached_property
    @abstractmethod
    def longitude(self) -> np.ndarray:
        """
        Return the longitude coordinates array.

        Uses cached_property to ensure the array is computed only once.
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Return the grid shape as (n_lat, n_lon)."""
        pass

    @abstractmethod
    def findPointXy(self, lat: float, lon: float) -> Optional[Tuple[int, int]]:
        """
        Find grid point indices (x, y) for given lat/lon coordinates.
        """
        pass

    @abstractmethod
    def getCoordinates(self, x: int, y: int) -> Tuple[float, float]:
        """
        Get lat/lon coordinates for a given grid point indices.
        """
        pass


class RegularLatLonGrid(AbstractGrid):
    """
    Regular latitude-longitude grid implementation.

    This represents a standard equirectangular grid with uniform spacing.
    """

    def __init__(
        self,
        lat_start: float,
        lat_steps: int,
        lat_step_size: float,
        lon_start: float,
        lon_steps: int,
        lon_step_size: float,
    ):
        """
        Initialize a regular lat/lon grid.

        Parameters:
        -----------
        lat_start : float
            Starting latitude value
        lat_steps : int
            Number of latitude points
        lat_step_size : float
            Spacing between latitude points
        lon_start : float
            Starting longitude value
        lon_steps : int
            Number of longitude points
        lon_step_size : float
            Spacing between longitude points
        """
        self._lat_start = lat_start
        self._lat_steps = lat_steps
        self._lat_step_size = lat_step_size
        self._lon_start = lon_start
        self._lon_steps = lon_steps
        self._lon_step_size = lon_step_size

    @property
    def grid_type(self) -> str:
        return "regular_latlon"

    @cached_property
    def latitude(self) -> np.ndarray:
        """
        Lazily compute and cache the latitude coordinate array.
        """
        return np.linspace(
            self._lat_start,
            self._lat_start + self._lat_step_size * self._lat_steps,
            self._lat_steps,
            endpoint=False
        )

    @cached_property
    def longitude(self) -> np.ndarray:
        """
        Lazily compute and cache the longitude coordinate array.
        """
        return np.linspace(
            self._lon_start,
            self._lon_start + self._lon_step_size * self._lon_steps,
            self._lon_steps,
            endpoint=False
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._lat_steps, self._lon_steps)

    def findPointXy(self, lat: float, lon: float) -> Optional[Tuple[int, int]]:
        """
        Find grid point indices (x, y) for given lat/lon coordinates.

        Parameters:
        -----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns:
        --------
        tuple or None
            (x, y) grid indices if point is in grid, None otherwise
        """
        # Calculate raw x and y indices
        x = int(round((lon - self._lon_start) / self._lon_step_size))
        y = int(round((lat - self._lat_start) / self._lat_step_size))

        # Handle wrapping for global grids
        xx = _modulo_positive(x, self._lon_steps) if (self._lon_steps * self._lon_step_size) >= 359 else x
        yy = _modulo_positive(y, self._lat_steps) if (self._lat_steps * self._lat_step_size) >= 179 else y

        # Check if point is within grid bounds
        if yy < 0 or xx < 0 or yy >= self._lat_steps or xx >= self._lon_steps:
            return None

        return (xx, yy)

    def getCoordinates(self, x: int, y: int) -> Tuple[float, float]:
        """
        Get lat/lon coordinates for a given grid point indices.

        Parameters:
        -----------
        x : longitude index
        y : latitude index

        Returns:
        --------
        tuple
            (latitude, longitude) coordinates
        """

        lat = self._lat_start + float(y) * self._lat_step_size
        lon = self._lon_start + float(x) * self._lon_step_size

        return (lat, lon)

class OmDomain:
    """
    Class representing a domain configuration for a weather model.

    This class provides metadata and configuration for different
    weather model grids used in Open-Meteo.
    """

    def __init__(
        self,
        name: str,
        grid: AbstractGrid,
        file_length: int,
        temporal_resolution_seconds: int = 3600
    ):
        """
        Initialize a domain configuration.

        Parameters:
        -----------
        name : str
            Name of the domain
        grid : AbstractGrid
            Grid implementation for this domain
        file_length : int
            Number of time steps in each file chunk
        temporal_resolution_seconds : int, optional
            Time resolution in seconds (default: 3600 = 1 hour)
        """
        self.name = name
        self.grid = grid
        self.file_length = file_length
        self.temporal_resolution_seconds = temporal_resolution_seconds

    def time_to_chunk_index(self, timestamp: np.datetime64) -> int:
        """
        Convert a timestamp to a chunk index. This depends on the file_length
        and the temporal_resolution_seconds of the domain.

        Parameters:
        -----------
        timestamp : np.datetime64
            The timestamp to convert

        Returns:
        --------
        int
            The chunk index containing the timestamp
        """
        seconds_since_epoch = (timestamp - EPOCH) / np.timedelta64(1, 's')
        chunk_index = int(seconds_since_epoch / (self.file_length * self.temporal_resolution_seconds))
        return chunk_index

    def chunks_for_date_range(
        self,
        start_timestamp: np.datetime64,
        end_timestamp: np.datetime64,
    ) -> List[int]:
        """
        Find all chunk indices that contain data within the given date range.

        Parameters:
        -----------
        start_date : datetime
            Start date for the data range
        end_date : datetime
            End date for the data range
        Returns:
        --------
        List[int]
            List of chunk indices containing data within the date range
        """
        # Get chunk indices for start and end dates
        start_chunk = self.time_to_chunk_index(start_timestamp)
        end_chunk = self.time_to_chunk_index(end_timestamp)

        # Generate list of all chunks between start and end (inclusive)
        return list(range(start_chunk, end_chunk +1))

    def get_chunk_time_range(self, chunk_index: int):
        """
        Get the time range covered by a specific chunk.

        Parameters:
        -----------
        chunk_index : int
            Index of the chunk

        Returns:
        --------
        np.ndarray
            Array of datetime64 objects representing the time points in the chunk
        """
        chunk_start_seconds = chunk_index * self.file_length * self.temporal_resolution_seconds
        start_time = EPOCH + np.timedelta64(chunk_start_seconds, 's')

        # Generate timestamps at regular intervals from the start time
        time_delta = np.timedelta64(self.temporal_resolution_seconds, 's')
        # Note: better type inference via list comprehension here
        timestamps = np.array([start_time + i * time_delta for i in range(self.file_length)])
        return timestamps

# - MARK: Create grid instances for supported domains

# DWD ICON D2 is regularized during download to nx: 1215, ny: 746 points
# https://github.com/open-meteo/open-meteo/blob/1753ebb4966d05f61b17dd5bdf59700788d4a913/Sources/App/Icon/Icon.swift#L154
_dwd_icon_d2_grid = RegularLatLonGrid(
    lat_start=43.18,
    lat_steps=746,
    lat_step_size=0.02,
    lon_start=-3.94,
    lon_steps=1215,
    lon_step_size=0.02
)

# ECMWF IQFS grid is a regular global lat/lon grid, nx: 1440, ny: 721 points
# https://github.com/open-meteo/open-meteo/blob/1753ebb4966d05f61b17dd5bdf59700788d4a913/Sources/App/Ecmwf/EcmwfDomain.swift#L107
_ecmwf_ifs025_grid = RegularLatLonGrid(
    lat_start=-90,
    lat_steps=721,
    lat_step_size=360/1440,
    lon_start=-180,
    lon_steps=1440,
    lon_step_size=180/(721-1)
)

# https://github.com/open-meteo/open-meteo/blob/1753ebb4966d05f61b17dd5bdf59700788d4a913/Sources/App/MeteoFrance/MeteoFranceDomain.swift#L348
_meteofrance_arpege_europe_grid = RegularLatLonGrid(
    lat_start=20,
    lat_steps=521,
    lat_step_size=0.1,
    lon_start=-32,
    lon_steps=741,
    lon_step_size=0.1
)

DOMAINS: dict[str, OmDomain] = {
    'dwd_icon_d2': OmDomain(
        name='dwd_icon_d2',
        grid=_dwd_icon_d2_grid,
        file_length=121,
        temporal_resolution_seconds=3600
    ),
    'ecmwf_ifs025': OmDomain(
        name='ecmwf_ifs025',
        grid=_ecmwf_ifs025_grid,
        file_length=104,
        temporal_resolution_seconds=3600*3
    ),
    'meteofrance_arpege_europe': OmDomain(
        name='meteofrance_arpege_europe',
        grid=_meteofrance_arpege_europe_grid,
        file_length=114+3*24,
        temporal_resolution_seconds=3600
    )
    # Additional domains can be added here
}
