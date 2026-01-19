"""Regular latitude/longitude or projected grid."""

from typing import NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pyproj import CRS, Transformer

from omfiles.types import LatLon, XYIndex


class RegularGrid:
    """Regular latitude/longitude or projected grid."""

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

    def find_point_xy(self, lat: float, lon: float) -> Optional[XYIndex]:
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

        return XYIndex(x_idx, y_idx)

    def get_coordinates(self, x: int, y: int) -> LatLon:
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

        return LatLon(float(lat), float(lon))

    def get_meshgrid(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get meshgrid of geographic coordinates.

        Useful for plotting with matplotlib/cartopy.

        Returns:
            (lon_grid, lat_grid) arrays of shape (ny, nx) in geographic coordinates
        """
        return (self.longitude, self.latitude)
