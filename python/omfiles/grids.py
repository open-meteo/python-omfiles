from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt

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
        k = 2 * self.R / (1 + self.sin_phi_1 * np.sin(phi) +
                         self.cos_phi_1 * np.cos(phi) * np.cos(lambda_ - self.lambda_0))
        x = k * np.cos(phi) * np.sin(lambda_ - self.lambda_0)
        y = k * (self.cos_phi_1 * np.sin(phi) -
               self.sin_phi_1 * np.cos(phi) * np.cos(lambda_ - self.lambda_0))

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
        # Convert inputs to numpy arrays for uniform handling
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        # Calculate distance from origin
        p = np.sqrt(x_arr*x_arr + y_arr*y_arr)

        # Initialize output arrays
        phi = np.zeros_like(p)
        lambda_ = np.zeros_like(p)

        # Handle the origin case
        origin = (p == 0)
        phi[origin] = np.arcsin(self.sin_phi_1)
        lambda_[origin] = self.lambda_0

        # Handle non-origin points
        non_origin = ~origin
        if np.any(non_origin):
            c = 2 * np.arctan2(p[non_origin], 2*self.R)
            phi[non_origin] = np.arcsin(np.cos(c) * self.sin_phi_1 +
                           (y_arr[non_origin] * np.sin(c) * self.cos_phi_1) / p[non_origin])
            lambda_[non_origin] = self.lambda_0 + np.arctan2(
                x_arr[non_origin] * np.sin(c),
                p[non_origin] * self.cos_phi_1 * np.cos(c) - y_arr[non_origin] * self.sin_phi_1 * np.sin(c)
            )

        # Convert to degrees
        lat = np.degrees(phi)
        lon = np.degrees(lambda_)

        return lat, lon

P = TypeVar('P', bound=AbstractProjection)


class ProjectionGrid(AbstractGrid, Generic[P]):
    """
    Grid implementation using a projection.

    This represents a grid in a projected coordinate system.
    """

    def __init__(
        self,
        projection: P,
        nx: int,
        ny: int,
        origin: Tuple[float, float],
        dx: float,
        dy: float
    ):
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
            Grid spacing in x direction in meters
        dy : float
            Grid spacing in y direction in meters
        """
        self.projection = projection
        self.nx = nx
        self.ny = ny
        self.origin = origin
        self.dx = dx
        self.dy = dy

    @classmethod
    def from_bounds(
        cls,
        nx: int,
        ny: int,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        projection: P
    ) -> 'ProjectionGrid[P]':
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
        dx = (ne[0] - sw[0]) / (nx-1)
        dy = (ne[1] - sw[1]) / (ny-1)
        return cls(projection, nx, ny, origin, float(dx), float(dy))

    @classmethod
    def from_center(
        cls,
        nx: int,
        ny: int,
        center_lat: float,
        center_lon: float,
        dx: float,
        dy: float,
        projection: P
    ) -> 'ProjectionGrid[P]':
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
        origin_x = center[0] - dx * (nx // 2)
        origin_y = center[1] - dy * (ny // 2)
        return cls(projection, nx, ny, (origin_x, origin_y), dx, dy)

    @property
    def grid_type(self) -> str:
        return "projection"

    @cached_property
    def latitude(self) -> np.ndarray:
        """
        Lazily compute and cache the latitude coordinate array.
        """
        # Create meshgrid of coordinates
        y_indices, x_indices = np.meshgrid(
            np.arange(self.ny),
            np.arange(self.nx),
            indexing='ij'
        )

        # Convert to projected coordinates
        x_coords = x_indices * self.dx + self.origin[0]
        y_coords = y_indices * self.dy + self.origin[1]

        # Convert to lat/lon using vectorized inverse method
        lat, _ = cast(tuple[ArrayType, ArrayType], self.projection.inverse(x_coords, y_coords))
        return lat

    @cached_property
    def longitude(self) -> np.ndarray:
        """
        Lazily compute and cache the longitude coordinate array.
        """
        # Create meshgrid of coordinates
        y_indices, x_indices = np.meshgrid(
            np.arange(self.ny),
            np.arange(self.nx),
            indexing='ij'
        )

        # Convert to projected coordinates
        x_coords = x_indices * self.dx + self.origin[0]
        y_coords = y_indices * self.dy + self.origin[1]

        # Convert to lat/lon using vectorized inverse method
        _, lon =  cast(tuple[ArrayType, ArrayType], self.projection.inverse(x_coords, y_coords))
        return lon

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
        lon = ((lon + 180.0) % 360.0) - 180.0
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
        y_indices, x_indices = np.meshgrid(
            np.arange(self.ny),
            np.arange(self.nx),
            indexing='ij'
        )

        # Vectorized calculation of angles
        true_north = np.degrees(np.arctan2(
            north_pole_x - x_indices,
            north_pole_y - y_indices
        ))

        return true_north

    def find_box(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> np.ndarray:
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
        y_indices, x_indices = np.meshgrid(
            np.arange(y_min, y_max),
            np.arange(x_min, x_max),
            indexing='ij'
        )

        # Convert to flat indices
        return np.ravel_multi_index(
            (y_indices.flatten(), x_indices.flatten()),
            (self.ny, self.nx)
        )
