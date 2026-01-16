"""An OmGrid provides utilities to transform between geographic coordinates and grid indices."""

from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from omfiles.types import LatLon, XYIndex

try:
    from pyproj import CRS
except ImportError:
    raise ImportError("omfiles[grids] is required for OmGrid functionality")

from omfiles.grids.gaussian import GaussianGrid
from omfiles.grids.regular import RegularGrid

EPOCH = np.datetime64(0, "s")


def _is_gaussian_grid(crs_wkt: str) -> bool:
    """Check if WKT string represents a Gaussian grid."""
    return "Reduced Gaussian Grid" in crs_wkt or "Gaussian Grid" in crs_wkt


class OmGrid:
    """Wrapper for grid implementations - automatically delegates to appropriate grid type."""

    def __init__(self, crs_wkt: str, shape: tuple[int, ...]):
        """
        Initialize grid from WKT projection string and data shape.

        Args:
            crs_wkt: Coordinate Reference System in Well-Known Text format
            shape: Grid shape as (ny, nx)
        """
        if not isinstance(shape, tuple) or not all(isinstance(dim, int) for dim in shape) or not len(shape) == 2:
            raise ValueError("shape must be a tuple of two integers")
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

    def find_point_xy(self, lat: float, lon: float) -> Optional[XYIndex]:
        """Find grid point indices for given lat/lon coordinates."""
        return self._grid.find_point_xy(lat, lon)

    def get_coordinates(self, x: int, y: int) -> LatLon:
        """Get lat/lon coordinates for given grid point indices."""
        return self._grid.get_coordinates(x, y)

    def get_meshgrid(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get meshgrid of geographic coordinates."""
        return self._grid.get_meshgrid()

    @property
    def is_gaussian(self) -> bool:
        """Check if this is a Gaussian grid."""
        return isinstance(self._grid, GaussianGrid)

    @property
    def crs(self) -> Union[CRS, None]:
        """Get the Coordinate Reference System."""
        if isinstance(self._grid, GaussianGrid):
            return None
        return self._grid.crs
