from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional, Tuple

import numpy as np

from omfiles.utils import _modulo_positive


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
