"""Gaussian grid implementation for reduced Gaussian grids like ECMWF IFS."""

from typing import NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt

from omfiles.types import LatLon, XYIndex


class GaussianGrid:
    """
    Implementation of reduced Gaussian grids (O1280, O320, N320, N160).

    These grids have varying numbers of longitude points at each latitude line,
    with more points near the equator and fewer near the poles.
    """

    # Lookup tables for N-type grids
    N320_COUNT_PER_LINE = [
        18,
        25,
        36,
        40,
        45,
        50,
        60,
        64,
        72,
        72,
        75,
        81,
        90,
        96,
        100,
        108,
        120,
        120,
        125,
        135,
        144,
        144,
        150,
        160,
        180,
        180,
        180,
        192,
        192,
        200,
        216,
        216,
        216,
        225,
        240,
        240,
        240,
        250,
        256,
        270,
        270,
        288,
        288,
        288,
        300,
        300,
        320,
        320,
        320,
        324,
        360,
        360,
        360,
        360,
        360,
        360,
        375,
        375,
        384,
        384,
        400,
        400,
        405,
        432,
        432,
        432,
        432,
        450,
        450,
        450,
        480,
        480,
        480,
        480,
        480,
        486,
        500,
        500,
        500,
        512,
        512,
        540,
        540,
        540,
        540,
        540,
        576,
        576,
        576,
        576,
        576,
        576,
        600,
        600,
        600,
        600,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        648,
        648,
        675,
        675,
        675,
        675,
        720,
        720,
        720,
        720,
        720,
        720,
        720,
        720,
        720,
        729,
        750,
        750,
        750,
        750,
        768,
        768,
        768,
        768,
        800,
        800,
        800,
        800,
        800,
        800,
        810,
        810,
        864,
        864,
        864,
        864,
        864,
        864,
        864,
        864,
        864,
        864,
        864,
        900,
        900,
        900,
        900,
        900,
        900,
        900,
        900,
        960,
        960,
        960,
        960,
        960,
        960,
        960,
        960,
        960,
        960,
        960,
        960,
        960,
        960,
        972,
        972,
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        1024,
        1024,
        1024,
        1024,
        1024,
        1024,
        1080,
        1080,
        1080,
        1080,
        1080,
        1080,
        1080,
        1080,
        1080,
        1080,
        1080,
        1080,
        1080,
        1080,
        1125,
        1125,
        1125,
        1125,
        1125,
        1125,
        1125,
        1125,
        1125,
        1125,
        1125,
        1125,
        1125,
        1125,
        1152,
        1152,
        1152,
        1152,
        1152,
        1152,
        1152,
        1152,
        1152,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1200,
        1215,
        1215,
        1215,
        1215,
        1215,
        1215,
        1215,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
        1280,
    ]

    N160_COUNT_PER_LINE = [
        18,
        25,
        36,
        40,
        45,
        50,
        60,
        64,
        72,
        72,
        80,
        90,
        90,
        96,
        108,
        120,
        120,
        125,
        128,
        135,
        144,
        150,
        160,
        160,
        180,
        180,
        180,
        192,
        192,
        200,
        216,
        216,
        225,
        225,
        240,
        240,
        243,
        250,
        256,
        270,
        270,
        288,
        288,
        288,
        300,
        300,
        320,
        320,
        320,
        320,
        324,
        360,
        360,
        360,
        360,
        360,
        360,
        375,
        375,
        375,
        384,
        384,
        400,
        400,
        400,
        405,
        432,
        432,
        432,
        432,
        432,
        450,
        450,
        450,
        450,
        480,
        480,
        480,
        480,
        480,
        480,
        480,
        500,
        500,
        500,
        500,
        500,
        512,
        512,
        540,
        540,
        540,
        540,
        540,
        540,
        540,
        540,
        576,
        576,
        576,
        576,
        576,
        576,
        576,
        576,
        576,
        576,
        600,
        600,
        600,
        600,
        600,
        600,
        600,
        600,
        600,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
    ]

    def __init__(self, crs_wkt: str, shape: Tuple[int, int]):
        """
        Initialize Gaussian grid from WKT projection string and shape.

        Args:
            crs_wkt: WKT string containing grid type information in REMARK field
            shape: Grid shape as (ny, nx) where ny=1 and nx is total grid point count
        """
        self.ny, self.nx = shape

        if self.ny != 1:
            raise ValueError(f"Gaussian grid must have ny=1, got {self.ny}")

        # Parse grid type from WKT REMARK field
        self.grid_type = self._parse_grid_type(crs_wkt)
        self.latitude_lines = self._get_latitude_lines()

        # Validate grid point count
        expected_count = self._calculate_total_points()
        if self.nx != expected_count:
            raise ValueError(f"Grid point count mismatch: expected {expected_count}, got {self.nx}")

        # Pre-calculate cumulative sums for faster lookups
        self._build_integral_table()

    def _parse_grid_type(self, crs_wkt: str) -> str:
        """Extract grid type (O1280, O320, N320, N160) from WKT string."""
        if "O1280" in crs_wkt or self.nx == 6599680:
            return "O1280"
        elif "O320" in crs_wkt or self.nx == 421120:
            return "O320"
        elif "N320" in crs_wkt or self.nx == 542080:
            return "N320"
        elif "N160" in crs_wkt or self.nx == 138346:
            return "N160"
        else:
            raise ValueError(f"Unknown Gaussian grid type with {self.nx} points")

    def _get_latitude_lines(self) -> int:
        """Get number of latitude lines from pole to equator."""
        if self.grid_type == "O1280":
            return 1280
        elif self.grid_type == "O320":
            return 320
        elif self.grid_type == "N320":
            return 320
        elif self.grid_type == "N160":
            return 160
        else:
            raise ValueError(f"Unknown grid type: {self.grid_type}")

    def _calculate_total_points(self) -> int:
        """Calculate total number of grid points."""
        if self.grid_type in ["O1280", "O320"]:
            return 4 * self.latitude_lines * (self.latitude_lines + 9)
        elif self.grid_type == "N320":
            return 542080
        elif self.grid_type == "N160":
            return 138346
        else:
            raise ValueError(f"Unknown grid type: {self.grid_type}")

    def _nx_of_y(self, y: int) -> int:
        """Get number of longitude points at latitude line y."""
        if self.grid_type in ["O1280", "O320"]:
            # O-type grids have analytical formula
            if y < self.latitude_lines:
                return 20 + y * 4
            else:
                return (2 * self.latitude_lines - y - 1) * 4 + 20
        else:
            # N-type grids use lookup table
            count_per_line = self.N320_COUNT_PER_LINE if self.grid_type == "N320" else self.N160_COUNT_PER_LINE
            if y < self.latitude_lines:
                return count_per_line[y]
            else:
                return count_per_line[2 * len(count_per_line) - y - 1]

    def _build_integral_table(self):
        """Pre-calculate cumulative sums for faster grid point lookups."""
        self._integral_table = [0]
        for y in range(2 * self.latitude_lines):
            self._integral_table.append(self._integral_table[-1] + self._nx_of_y(y))

    def _integral(self, y: int) -> int:
        """Get cumulative number of grid points up to (but not including) latitude line y."""
        return self._integral_table[y]

    def _get_pos(self, gridpoint: int) -> Tuple[int, int, int]:
        """
        Find latitude line (y) and longitude index (x) for given grid point.

        This matches the Swift getPos method exactly.

        Returns:
            (y, x, nx) where nx is number of points on this latitude line
        """
        if gridpoint < 0 or gridpoint >= self.nx:
            raise ValueError(f"Grid point {gridpoint} out of range [0, {self.nx})")

        if self.grid_type in ["O1280", "O320"]:
            # O-type grids use analytical formula
            count = self.nx
            half_count = count // 2

            if gridpoint < half_count:
                # Northern hemisphere (including equator)
                # Solve: gridpoint = 2*y*y + 18*y
                y = int((np.sqrt(2 * gridpoint + 81) - 9) / 2)
            else:
                # Southern hemisphere
                # Mirror from the other side
                gridpoint_from_end = count - gridpoint - 1
                y_from_end = int((np.sqrt(2 * gridpoint_from_end + 81) - 9) / 2)
                y = 2 * self.latitude_lines - 1 - y_from_end

            x = gridpoint - self._integral(y)
            nx = self._nx_of_y(y)
            return (y, x, nx)
        else:
            # N-type grids use lookup
            count_per_line = self.N320_COUNT_PER_LINE if self.grid_type == "N320" else self.N160_COUNT_PER_LINE

            # Search in northern hemisphere first
            cumsum = 0
            for y, n in enumerate(count_per_line):
                cumsum += n
                if gridpoint < cumsum:
                    return (y, gridpoint - (cumsum - n), n)

            # Search in southern hemisphere
            for y, n in enumerate(reversed(count_per_line)):
                cumsum += n
                if gridpoint < cumsum:
                    actual_y = y + len(count_per_line)
                    return (actual_y, gridpoint - (cumsum - n), n)

            raise ValueError(f"Grid point {gridpoint} not found")

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as (ny, nx)."""
        return (self.ny, self.nx)

    def get_coordinates(self, x: int, y: int) -> LatLon:
        """
        Get lat/lon coordinates for grid point index.

        For Gaussian grids, y is always 0, and x is the flat grid point index.

        Args:
            x: Grid point index (0 to nx-1)
            y: Must be 0 for Gaussian grids

        Returns:
            (latitude, longitude) in degrees
        """
        if y != 0:
            raise ValueError(f"Gaussian grid only has y=0, got y={y}")

        return self._get_coordinates_from_gridpoint(x)

    def _get_coordinates_from_gridpoint(self, gridpoint: int) -> LatLon:
        """Get coordinates from flat grid point index."""
        y, x, nx = self._get_pos(gridpoint)

        # Calculate latitude
        dy = 180.0 / (2 * self.latitude_lines + 0.5)
        lat = (self.latitude_lines - y - 1) * dy + dy / 2

        # Calculate longitude
        dx = 360.0 / nx
        lon = x * dx

        # Normalize longitude to [-180, 180)
        if lon >= 180:
            lon -= 360

        return LatLon(float(lat), float(lon))

    def find_point_xy(self, lat: float, lon: float) -> Optional[XYIndex]:
        """
        Find grid point index for given lat/lon coordinates.

        For Gaussian grids, returns (gridpoint, 0) to match the interface.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            (gridpoint, 0) if point is valid, None otherwise
        """
        x_idx, y_idx = self._find_point_xy(lat, lon)

        # Convert to flat grid point index
        gridpoint = self._integral(y_idx) + x_idx
        return XYIndex(gridpoint, 0)

    def _find_point_xy(self, lat: float, lon: float) -> XYIndex:
        # Calculate latitude line
        dy = 180.0 / (2.0 * self.latitude_lines + 0.5)

        # Note: Limited by -2 because later we add +1
        y_float = self.latitude_lines - 1.0 - ((lat - dy / 2.0) / dy)
        y = max(0, min(2 * self.latitude_lines - 2, int(y_float)))
        y_upper = y + 1

        # Get number of longitude points on both lines
        nx = self._nx_of_y(y)
        nx_upper = self._nx_of_y(y_upper)

        dx = 360.0 / nx
        dx_upper = 360.0 / nx_upper

        # Find closest x on both lines
        x = int(round(lon / dx))
        x_upper = int(round(lon / dx_upper))

        # Calculate actual coordinates
        point_lat = (self.latitude_lines - y - 1) * dy + dy / 2.0
        point_lon = x * dx
        point_lat_upper = (self.latitude_lines - y_upper - 1) * dy + dy / 2.0
        point_lon_upper = x_upper * dx_upper

        # Calculate squared distances
        distance = (point_lat - lat) ** 2 + (point_lon - lon) ** 2
        distance_upper = (point_lat_upper - lat) ** 2 + (point_lon_upper - lon) ** 2

        # Return closest point with proper wrapping
        if distance < distance_upper:
            return XYIndex((x + nx) % nx, y)
        else:
            return XYIndex((x_upper + nx_upper) % nx_upper, y_upper)

    @property
    def latitude(self) -> npt.NDArray[np.float64]:
        """Get 1D array of latitude coordinates for all grid points."""
        if not hasattr(self, "_latitude"):
            self._compute_coordinates()
        return self._latitude

    @property
    def longitude(self) -> npt.NDArray[np.float64]:
        """Get 1D array of longitude coordinates for all grid points."""
        if not hasattr(self, "_longitude"):
            self._compute_coordinates()
        return self._longitude

    def _compute_coordinates(self) -> None:
        """Compute and cache latitude/longitude arrays for all grid points."""
        lats = np.zeros(self.nx, dtype=np.float64)
        lons = np.zeros(self.nx, dtype=np.float64)

        for gridpoint in range(self.nx):
            lat, lon = self._get_coordinates_from_gridpoint(gridpoint)
            lats[gridpoint] = lat
            lons[gridpoint] = lon

        self._latitude = lats
        self._longitude = lons

    def get_meshgrid(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Meshgrids are not meaningful for Gaussian grids, because the grid points are not evenly spaced.
        """
        raise NotImplementedError(
            "Meshgrids are not meaningful for Gaussian grids. Use earthkit.regrid to regrid to a regular grid."
        )
