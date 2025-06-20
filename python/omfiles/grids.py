from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt

from omfiles.utils import _modulo_positive, _normalize_longitude


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
            self._lat_start, self._lat_start + self._lat_step_size * self._lat_steps, self._lat_steps, endpoint=False
        )

    @cached_property
    def longitude(self) -> np.ndarray:
        """
        Lazily compute and cache the longitude coordinate array.
        """
        return np.linspace(
            self._lon_start, self._lon_start + self._lon_step_size * self._lon_steps, self._lon_steps, endpoint=False
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


# Type aliases for clarity
FloatType = Union[float, np.floating]
ArrayType = npt.NDArray[np.floating]
CoordType = Union[float, ArrayType]
ReturnUnionType = Union[tuple[ArrayType, ArrayType], tuple[float, float]]


# Abstract base class instead of Protocol
class AbstractProjection(ABC):
    """Base class for projection implementations."""

    @abstractmethod
    def forward(self, latitude: CoordType, longitude: CoordType) -> ReturnUnionType:
        """
        Transform from lat/lon coordinates to projected x/y coordinates.

        Handles both scalar and array inputs transparently.
        """
        pass

    @abstractmethod
    def inverse(self, x: CoordType, y: CoordType) -> ReturnUnionType:
        """
        Transform from projected x/y coordinates back to lat/lon.

        Handles both scalar and array inputs transparently.
        """
        pass


class RotatedLatLonProjection(AbstractProjection):
    """
    Rotated lat/lon projection implementation.

    This implements the transformation between regular lat/lon coordinates and
    rotated lat/lon coordinates where the pole is shifted to a specified location.
    Based on: https://github.com/open-meteo/open-meteo/blob/main/Sources/App/Domains/RotatedLatLon.swift
    """

    def __init__(self, lat_origin: float, lon_origin: float):
        """
        Initialize a rotated lat/lon projection.

        Parameters:
        -----------
        lat_origin : float
            Latitude of origin in degrees
        lon_origin : float
            Longitude of origin in degrees
        """
        # θ: Rotation around y-axis
        self.theta = np.radians(90.0 + lat_origin)
        # ϕ: Rotation around z-axis
        self.phi = np.radians(lon_origin)

    def forward(self, latitude: CoordType, longitude: CoordType) -> ReturnUnionType:
        """
        Transform from regular lat/lon to rotated lat/lon coordinates.

        Parameters:
        -----------
        latitude : float or array
            Latitude in degrees
        longitude : float or array
            Longitude in degrees

        Returns:
        --------
        tuple
            (rotated_lat, rotated_lon) in degrees
        """
        scalar_input = np.isscalar(latitude) and np.isscalar(longitude)

        lat_arr = np.asarray(latitude, dtype=np.float32)
        lon_arr = np.asarray(longitude, dtype=np.float32)

        # Convert to radians
        lat_rad = np.radians(lat_arr)
        lon_rad = np.radians(lon_arr)

        # Convert to cartesian coordinates
        x = np.cos(lon_rad) * np.cos(lat_rad)
        y = np.sin(lon_rad) * np.cos(lat_rad)
        z = np.sin(lat_rad)

        # Apply rotation
        x2 = (
            np.cos(self.theta) * np.cos(self.phi) * x
            + np.cos(self.theta) * np.sin(self.phi) * y
            + np.sin(self.theta) * z
        )
        y2 = -np.sin(self.phi) * x + np.cos(self.phi) * y
        z2 = (
            -np.sin(self.theta) * np.cos(self.phi) * x
            - np.sin(self.theta) * np.sin(self.phi) * y
            + np.cos(self.theta) * z
        )

        # Convert back to spherical coordinates
        rot_lon = np.degrees(np.arctan2(y2, x2))
        rot_lat = np.degrees(np.arcsin(z2))

        if scalar_input:
            return float(rot_lon.item()), float(rot_lat.item())

        return rot_lon, rot_lat

    def inverse(self, x: CoordType, y: CoordType) -> ReturnUnionType:
        """
        Transform from rotated lat/lon back to regular lat/lon coordinates.

        Parameters:
        -----------
        x : float or array
            Rotated longitude in degrees
        y : float or array
            Rotated latitude in degrees

        Returns:
        --------
        tuple
            (latitude, longitude) in degrees
        """
        scalar_input = np.isscalar(x) and np.isscalar(y)

        rot_lon = np.radians(np.asarray(x, dtype=np.float32))
        rot_lat = np.radians(np.asarray(y, dtype=np.float32))

        theta_neg = -self.theta
        phi_neg = -self.phi

        # Quick solution without conversion in cartesian space
        lat_rad = np.arcsin(np.cos(theta_neg) * np.sin(rot_lat) - np.cos(rot_lon) * np.sin(theta_neg) * np.cos(rot_lat))

        lon_rad = (
            np.arctan2(np.sin(rot_lon), np.tan(rot_lat) * np.sin(theta_neg) + np.cos(rot_lon) * np.cos(theta_neg))
            - phi_neg
        )

        lat2 = np.degrees(lat_rad)
        lon2 = np.degrees(lon_rad)

        if scalar_input:
            return float(lat2.item()), float(lon2.item())

        return lat2, lon2


class StereographicProjection(AbstractProjection):
    """
    Stereographic projection implementation.

    This implements the equations for the stereographic projection
    which projects a sphere onto a plane.
    https://mathworld.wolfram.com/StereographicProjection.html
    """

    def __init__(self, latitude: float, longitude: float, radius: float = 6371000.0):
        """
        Initialize a stereographic projection.

        Parameters:
        -----------
        latitude : float
            Central latitude in degrees
        longitude : float
            Central longitude in degrees
        radius : float
            Radius of Earth in meters (default: 6371000.0)
        """
        self.lambda_0: npt.NDArray[np.float32] = np.radians(longitude)
        self.sin_phi_1: npt.NDArray[np.float32] = np.sin(np.radians(latitude))
        self.cos_phi_1: npt.NDArray[np.float32] = np.cos(np.radians(latitude))
        self.R = radius

    def forward(self, latitude: CoordType, longitude: CoordType) -> ReturnUnionType:
        """
        Transform from lat/lon coordinates to projected x/y coordinates.

        Parameters:
        -----------
        latitude : float or array
            Latitude in degrees
        longitude : float or array
            Longitude in degrees

        Returns:
        --------
        tuple
            (x, y) coordinates in the projection
        """
        scalar_input = np.isscalar(latitude) and np.isscalar(longitude)

        lat_arr = np.asarray(latitude, dtype=np.float32)
        lon_arr = np.asarray(longitude, dtype=np.float32)

        phi = np.radians(lat_arr)
        lambda_ = np.radians(lon_arr)
        k = (
            2
            * self.R
            / (1 + self.sin_phi_1 * np.sin(phi) + self.cos_phi_1 * np.cos(phi) * np.cos(lambda_ - self.lambda_0))
        )
        x = k * np.cos(phi) * np.sin(lambda_ - self.lambda_0)
        y = k * (self.cos_phi_1 * np.sin(phi) - self.sin_phi_1 * np.cos(phi) * np.cos(lambda_ - self.lambda_0))

        if scalar_input:
            return float(x.item()), float(y.item())

        return x, y

    def inverse(self, x: CoordType, y: CoordType) -> ReturnUnionType:
        """
        Transform from projected x/y coordinates back to lat/lon.

        Parameters:
        -----------
        x : float or array
            X coordinate in the projection
        y : float or array
            Y coordinate in the projection

        Returns:
        --------
        tuple
            (latitude, longitude) in degrees
        """
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        p = np.sqrt(x_arr * x_arr + y_arr * y_arr)

        # Initialize output arrays
        phi = np.zeros_like(p)
        lambda_ = np.zeros_like(p)

        c = 2 * np.arctan2(p, 2 * self.R)
        phi = np.arcsin(np.cos(c) * self.sin_phi_1 + (y_arr * np.sin(c) * self.cos_phi_1) / p)
        lambda_ = self.lambda_0 + np.arctan2(
            x_arr * np.sin(c), p * self.cos_phi_1 * np.cos(c) - y_arr * self.sin_phi_1 * np.sin(c)
        )

        return np.degrees(phi), np.degrees(lambda_)


class LambertAzimuthalEqualAreaProjection(AbstractProjection):
    """
    Lambert Azimuthal Equal-Area projection implementation.

    This implements the equations for the Lambert Azimuthal Equal-Area projection
    which preserves area but not angles or distances.
    https://mathworld.wolfram.com/LambertAzimuthalEqual-AreaProjection.html
    """

    def __init__(self, lambda_0: float, phi_1: float, radius: float = 6371229.0):
        """
        Initialize a Lambert Azimuthal Equal-Area projection.

        Parameters:
        -----------
        lambda_0 : float
            Central longitude in degrees
        phi_1 : float
            Standard parallel in degrees
        radius : float
            Radius of Earth in meters (default: 6371229.0)
        """
        self.lambda_0 = np.radians(lambda_0)
        self.phi_1 = np.radians(phi_1)
        self.R = radius

    def forward(self, latitude: CoordType, longitude: CoordType) -> ReturnUnionType:
        """
        Transform from lat/lon coordinates to projected x/y coordinates.

        Parameters:
        -----------
        latitude : float or array
            Latitude in degrees
        longitude : float or array
            Longitude in degrees

        Returns:
        --------
        tuple
            (x, y) coordinates in the projection
        """
        scalar_input = np.isscalar(latitude) and np.isscalar(longitude)

        lat_arr = np.asarray(latitude, dtype=np.float64)
        lon_arr = np.asarray(longitude, dtype=np.float64)

        lambda_ = np.radians(lon_arr)
        phi = np.radians(lat_arr)

        k = np.sqrt(
            2
            / (
                1
                + np.sin(self.phi_1) * np.sin(phi)
                + np.cos(self.phi_1) * np.cos(phi) * np.cos(lambda_ - self.lambda_0)
            )
        )

        x = self.R * k * np.cos(phi) * np.sin(lambda_ - self.lambda_0)
        y = (
            self.R
            * k
            * (np.cos(self.phi_1) * np.sin(phi) - np.sin(self.phi_1) * np.cos(phi) * np.cos(lambda_ - self.lambda_0))
        )

        if scalar_input:
            return float(x.item()), float(y.item())

        return x, y

    def inverse(self, x: CoordType, y: CoordType) -> ReturnUnionType:
        """
        Transform from projected x/y coordinates back to lat/lon.

        Parameters:
        -----------
        x : float or array
            X coordinate in the projection
        y : float or array
            Y coordinate in the projection

        Returns:
        --------
        tuple
            (latitude, longitude) in degrees
        """
        scalar_input = np.isscalar(x) and np.isscalar(y)

        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        x_norm = x_arr / self.R
        y_norm = y_arr / self.R
        p = np.sqrt(x_norm * x_norm + y_norm * y_norm)

        # Handle the case where p is zero (projection center)
        zero_p = p == 0
        p = np.where(zero_p, np.finfo(np.float32).eps, p)  # Avoid division by zero

        c = 2 * np.arcsin(0.5 * p)
        phi = np.arcsin(np.cos(c) * np.sin(self.phi_1) + (y_norm * np.sin(c) * np.cos(self.phi_1)) / p)
        lambda_ = self.lambda_0 + np.arctan2(
            x_norm * np.sin(c), p * np.cos(self.phi_1) * np.cos(c) - y_norm * np.sin(self.phi_1) * np.sin(c)
        )
        lat = np.degrees(phi)
        lon = np.degrees(lambda_)

        if scalar_input:
            return float(lat.item()), float(lon.item())

        return lat, lon


class LambertConformalConicProjection(AbstractProjection):
    """
    Lambert Conformal Conic projection implementation.

    This implements the equations for the Lambert Conformal Conic projection,
    which preserves angles but not areas or distances.
    https://mathworld.wolfram.com/LambertConformalConicProjection.html
    https://pubs.usgs.gov/pp/1395/report.pdf page 104
    """

    def __init__(self, lambda_0: float, phi_0: float, phi_1: float, phi_2: float, radius: float = 6370997):
        """
        Initialize a Lambert Conformal Conic projection.

        Parameters:
        -----------
        lambda_0 : float
            Reference longitude in degrees (LoVInDegrees in grib)
        phi_0 : float
            Reference latitude in degrees (LaDInDegrees in grib)
        phi_1 : float
            First standard parallel in degrees (Latin1InDegrees in grib)
        phi_2 : float
            Second standard parallel in degrees (Latin2InDegrees in grib)
        radius : float
            Radius of Earth in meters (default: 6370997)
        """
        # Normalize lambda_0 to [-180, 180] range
        lambda_0_normalized = _normalize_longitude(lambda_0)
        self.lambda_0 = np.radians(lambda_0_normalized)

        phi_0_rad = np.radians(phi_0)
        phi_1_rad = np.radians(phi_1)
        phi_2_rad = np.radians(phi_2)

        if phi_1 == phi_2:
            self.n = np.sin(phi_1_rad)
        else:
            self.n = np.log(np.cos(phi_1_rad) / np.cos(phi_2_rad)) / np.log(
                np.tan(np.pi / 4 + phi_2_rad / 2) / np.tan(np.pi / 4 + phi_1_rad / 2)
            )

        self.F = (np.cos(phi_1_rad) * np.power(np.tan(np.pi / 4 + phi_1_rad / 2), self.n)) / self.n

        self.rho_0 = self.F / np.power(np.tan(np.pi / 4 + phi_0_rad / 2), self.n)

        # Earth radius
        self.R = radius

    def forward(self, latitude: CoordType, longitude: CoordType) -> ReturnUnionType:
        """
        Transform from lat/lon coordinates to projected x/y coordinates.

        Parameters:
        -----------
        latitude : float or array
            Latitude in degrees
        longitude : float or array
            Longitude in degrees

        Returns:
        --------
        tuple
            (x, y) coordinates in the projection
        """
        scalar_input = np.isscalar(latitude) and np.isscalar(longitude)

        phi = np.radians(np.asarray(latitude, dtype=np.float64))
        lambda_ = np.radians(np.asarray(longitude, dtype=np.float64))

        # If (λ - λ0) exceeds the range:±: 180°, 360° should be added or subtracted.
        theta = self.n * (lambda_ - self.lambda_0)

        rho = self.F / np.power(np.tan(np.pi / 4 + phi / 2), self.n)
        x = self.R * rho * np.sin(theta)
        y = self.R * (self.rho_0 - rho * np.cos(theta))

        if scalar_input:
            return float(x.item()), float(y.item())

        return x, y

    def inverse(self, x: CoordType, y: CoordType) -> ReturnUnionType:
        """
        Transform from projected x/y coordinates back to lat/lon.

        Parameters:
        -----------
        x : float or array
            X coordinate in the projection
        y : float or array
            Y coordinate in the projection

        Returns:
        --------
        tuple
            (latitude, longitude) in degrees
        """
        scalar_input = np.isscalar(x) and np.isscalar(y)

        x_scaled = np.asarray(x, dtype=np.float64) / self.R
        y_scaled = np.asarray(y, dtype=np.float64) / self.R

        theta = np.where(
            self.n >= 0, np.arctan2(x_scaled, self.rho_0 - y_scaled), np.arctan2(-x_scaled, y_scaled - self.rho_0)
        )

        sign = np.where(self.n > 0, 1, -1)
        rho = sign * np.sqrt(np.square(x_scaled) + np.square(self.rho_0 - y_scaled))

        phi = 2 * np.arctan(np.power(self.F / rho, 1 / self.n)) - np.pi / 2
        lambda_ = self.lambda_0 + theta / self.n

        lat = np.degrees(phi)
        lon = np.degrees(lambda_)

        lon = np.where(lon > 180, lon - 360, lon)

        if scalar_input:
            return float(lat.item()), float(lon.item())

        return lat, lon


P = TypeVar("P", bound=AbstractProjection)


class ProjectionGrid(AbstractGrid, Generic[P]):
    """
    Grid implementation using a projection.

    This represents a grid in a projected coordinate system.
    """

    def __init__(self, projection: P, nx: int, ny: int, origin: Tuple[float, float], dx: float, dy: float):
        """
        Initialize a projection grid with all parameters.

        Parameters:
        -----------
        projection : Projectable
            Projection implementation
        nx : int
            Number of grid points in x direction
        ny : int
            Number of grid points in y direction
        origin : Tuple[float, float]
            Origin coordinates (x, y) of the grid in projection space
        dx : float
            Grid spacing in x direction
        dy : float
            Grid spacing in y direction
        """
        self.projection = projection
        self.nx = nx
        self.ny = ny
        self.origin = origin
        self.dx = dx
        self.dy = dy

    @classmethod
    def from_bounds(
        cls, nx: int, ny: int, lat_range: Tuple[float, float], lon_range: Tuple[float, float], projection: P
    ) -> "ProjectionGrid[P]":
        """
        Create a projection grid from geographic bounds.

        Parameters:
        -----------
        nx : int
            Number of grid points in x direction
        ny : int
            Number of grid points in y direction
        lat_range : Tuple[float, float]
            Latitude range (min, max) in degrees
        lon_range : Tuple[float, float]
            Longitude range (min, max) in degrees
        projection : Projectable
            Projection implementation

        Returns:
        --------
        ProjectionGrid
            New grid instance
        """
        sw = projection.forward(lat_range[0], lon_range[0])
        ne = projection.forward(lat_range[1], lon_range[1])
        origin = cast(tuple[float, float], sw)
        dx = (ne[0] - sw[0]) / (nx - 1)
        dy = (ne[1] - sw[1]) / (ny - 1)
        return cls(projection, nx, ny, origin, float(dx), float(dy))

    @classmethod
    def from_center(
        cls, nx: int, ny: int, center_lat: float, center_lon: float, dx: float, dy: float, projection: P
    ) -> "ProjectionGrid[P]":
        """
        Create a projection grid centered at a geographic location.

        Parameters:
        -----------
        nx : int
            Number of grid points in x direction
        ny : int
            Number of grid points in y direction
        center_lat : float
            Center latitude in degrees
        center_lon : float
            Center longitude in degrees
        dx : float
            Grid spacing in x direction in meters
        dy : float
            Grid spacing in y direction in meters
        projection : Projectable
            Projection implementation

        Returns:
        --------
        ProjectionGrid
            New grid instance
        """
        center = cast(tuple[float, float], projection.forward(center_lat, center_lon))
        return cls(projection, nx, ny, center, dx, dy)

    @property
    def grid_type(self) -> str:
        return "projection"

    @cached_property
    def _coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lazily compute and cache both latitude and longitude arrays.
        """
        # Create meshgrid of coordinates
        y_indices, x_indices = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")

        # Convert to projected coordinates
        x_coords = x_indices * self.dx + self.origin[0]
        y_coords = y_indices * self.dy + self.origin[1]

        # Convert to lat/lon using vectorized inverse method
        lat, lon = cast(tuple[ArrayType, ArrayType], self.projection.inverse(x_coords, y_coords))
        return lat, lon

    @property
    def latitude(self) -> np.ndarray:  # type: ignore
        """
        Get the latitude coordinate array.
        """
        return self._coordinates[0]

    @property
    def longitude(self) -> np.ndarray:  # type: ignore
        """
        Get the longitude coordinate array.
        """
        return self._coordinates[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.ny, self.nx)

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
        pos = cast(tuple[float, float], self.projection.forward(lat, lon))
        x = int(round((pos[0] - self.origin[0]) / self.dx))
        y = int(round((pos[1] - self.origin[1]) / self.dy))

        if y < 0 or x < 0 or y >= self.ny or x >= self.nx:
            return None

        return (x, y)

    def getCoordinates(self, x: int, y: int) -> Tuple[float, float]:
        """
        Get lat/lon coordinates for a given grid point indices.

        Parameters:
        -----------
        x : int
            X index
        y : int
            Y index

        Returns:
        --------
        tuple
            (latitude, longitude) coordinates
        """
        xcord = float(x) * self.dx + self.origin[0]
        ycord = float(y) * self.dy + self.origin[1]
        lat, lon = cast(tuple[float, float], self.projection.inverse(xcord, ycord))
        # Normalize longitude to -180 to 180 range
        lon = _normalize_longitude(lon)
        return (lat, lon)

    def get_true_north_direction(self) -> np.ndarray:
        """
        Calculate angle towards true north for every grid point.

        Returns:
        --------
        numpy.ndarray
            Array of angles in degrees, 0 = points towards north pole
        """
        pos = self.projection.forward(90, 0)  # North pole
        north_pole_x = (pos[0] - self.origin[0]) / self.dx
        north_pole_y = (pos[1] - self.origin[1]) / self.dy

        # Create grid of x, y coordinates
        y_indices, x_indices = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")

        # Vectorized calculation of angles
        true_north = np.degrees(np.arctan2(north_pole_x - x_indices, north_pole_y - y_indices))

        return true_north

    def find_box(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> np.ndarray:
        """
        Find indices of grid points within a geographic bounding box.

        Parameters:
        -----------
        lat_min : float
            Minimum latitude
        lat_max : float
            Maximum latitude
        lon_min : float
            Minimum longitude
        lon_max : float
            Maximum longitude

        Returns:
        --------
        numpy.ndarray
            Array of grid point indices within the box
        """
        sw = self.findPointXy(lat_min, lon_min)
        se = self.findPointXy(lat_min, lon_max)
        nw = self.findPointXy(lat_max, lon_min)
        ne = self.findPointXy(lat_max, lon_max)

        if not all([sw, se, nw, ne]):
            return np.array([], dtype=int)

        # Type casting to inform pyright that these variables are not None
        sw = cast(Tuple[int, int], sw)
        se = cast(Tuple[int, int], se)
        nw = cast(Tuple[int, int], nw)
        ne = cast(Tuple[int, int], ne)

        x_min = min(sw[0], nw[0])
        x_max = max(se[0], ne[0]) + 1
        y_min = min(sw[1], se[1])
        y_max = max(nw[1], ne[1]) + 1

        # Create meshgrid of indices
        y_indices, x_indices = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing="ij")

        # Convert to flat indices
        return np.ravel_multi_index((y_indices.flatten(), x_indices.flatten()), (self.ny, self.nx))


class ProjProjection(AbstractProjection):
    """A projection that wraps a proj projection"""

    def __init__(self, proj_string: str):
        """Initialize with a proj string or EPSG code

        Parameters
        ----------
        proj_string : str
            The proj string (e.g. "+proj=lcc +lat_0=50...") or
            EPSG code (e.g. "EPSG:4326")
        """
        import pyproj

        # Create transformer from lat/lon to projection coordinates
        self.crs_proj = pyproj.CRS(proj_string)
        self.crs_latlon = pyproj.CRS("EPSG:4326")  # WGS84
        self.forward_transformer = pyproj.Transformer.from_crs(
            self.crs_latlon,
            self.crs_proj,
            always_xy=True,  # This ensures lon/lat -> x/y order
        )
        self.inverse_transformer = pyproj.Transformer.from_crs(self.crs_proj, self.crs_latlon, always_xy=True)

    def forward(self, latitude: CoordType, longitude: CoordType) -> ReturnUnionType:
        """Transform from latitude/longitude to projection coordinates

        Parameters
        ----------
        latitude : float
            Latitude in degrees
        longitude : float
            Longitude in degrees

        Returns
        -------
        tuple[float, float]
            The (x, y) coordinates in the projection
        """
        x, y = self.forward_transformer.transform(longitude, latitude)
        return x, y

    def inverse(self, x: CoordType, y: CoordType) -> ReturnUnionType:
        """Transform from projection coordinates to latitude/longitude

        Parameters
        ----------
        x : float
            X coordinate in the projection
        y : float
            Y coordinate in the projection

        Returns
        -------
        tuple[float, float]
            The (latitude, longitude) coordinates in degrees
        """
        lon, lat = self.inverse_transformer.transform(x, y)
        return lat, lon
