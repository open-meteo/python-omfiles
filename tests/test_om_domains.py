import numpy as np
from omfiles.om_domains import DOMAINS, RegularLatLonGrid


def test_regular_grid_findPointXy_inside():
    """Test finding grid points inside the domain."""
    grid = RegularLatLonGrid(
        lat_start=0.0,
        lat_steps=10,
        lat_step_size=1.0,
        lon_start=0.0,
        lon_steps=20,
        lon_step_size=1.0
    )

    # Test exact grid points
    assert grid.findPointXy(5.0, 10.0) == (10, 5)
    assert grid.findPointXy(0.0, 0.0) == (0, 0)
    assert grid.findPointXy(9.0, 19.0) == (19, 9)

    # Test points that should round to grid points
    assert grid.findPointXy(5.1, 10.2) == (10, 5)
    assert grid.findPointXy(0.4, 0.4) == (0, 0)


def test_regular_grid_findPointXy_outside():
    """Test finding grid points outside the domain."""
    grid = RegularLatLonGrid(
        lat_start=0.0,
        lat_steps=10,
        lat_step_size=1.0,
        lon_start=0.0,
        lon_steps=20,
        lon_step_size=1.0
    )

    # Test points outside of grid
    assert grid.findPointXy(-1.0, 10.0) is None
    assert grid.findPointXy(5.0, -1.0) is None
    assert grid.findPointXy(10.0, 10.0) is None
    assert grid.findPointXy(5.0, 20.0) is None


def test_global_grid_wrapping():
    """Test that global grids wrap around at the edges."""
    # Create a global grid (360° longitude, 180° latitude coverage)
    global_grid = RegularLatLonGrid(
        lat_start=-90.0,
        lat_steps=180,
        lat_step_size=1.0,
        lon_start=-180.0,
        lon_steps=360,
        lon_step_size=1.0
    )

    # Test wrapping around the longitude
    # Point at longitude 180 should be the same as -180
    assert global_grid.findPointXy(0.0, 180.0) == (0, 90)
    assert global_grid.findPointXy(0.0, -180.0) == (0, 90)

    # Test a point beyond the normal range
    assert global_grid.findPointXy(0.0, 540.0) == (0, 90)


def test_ecmwf_grid():
    """Test the ECMWF IFS grid specifically."""
    ecmwf_grid = DOMAINS["ecmwf_ifs025"].grid

    # Test some known points on the grid
    # Point at the prime meridian and equator
    assert ecmwf_grid.findPointXy(0.0, 0.0) == (720, 360)

    # Point at the North Pole
    assert ecmwf_grid.findPointXy(90.0, 0.0) == (720, 720)

    # Test some edge points (ensure they are properly handled)
    assert ecmwf_grid.findPointXy(-90.0, -180.0) == (0, 0)
    assert ecmwf_grid.findPointXy(90.0, 180.0) == (0, 720)

    # Test wrapping for global grid
    # A point at longitude 181 should wrap to longitude -179
    point1 = ecmwf_grid.findPointXy(0.0, 181.0)
    point2 = ecmwf_grid.findPointXy(0.0, -179.0)
    assert point1 == point2


def test_grid_coordinates():
    """Test getting coordinates from grid indices."""
    grid = RegularLatLonGrid(
        lat_start=10.0,
        lat_steps=5,
        lat_step_size=2.0,
        lon_start=100.0,
        lon_steps=10,
        lon_step_size=5.0
    )

    # Test exact grid points
    assert grid.getCoordinates(0, 0) == (10.0, 100.0)
    assert grid.getCoordinates(5, 2) == (14.0, 125.0)

    # Test round-trip conversion
    lat, lon = 14.0, 125.0
    result = grid.findPointXy(lat, lon)
    assert result is not None, f"Could not find grid point for ({lat}, {lon})"
    x, y = result
    assert grid.getCoordinates(x, y) == (lat, lon)


def test_dwd_icon_d2_grid_points():
    """Test specific points in the DWD ICON D2 grid."""
    dwd_grid = DOMAINS["dwd_icon_d2"].grid

    # Test a point known to be in the domain (Central Europe)
    # Berlin coordinates: approx. 52.52°N, 13.40°E
    berlin = dwd_grid.findPointXy(52.52, 13.40)
    assert berlin is not None

    # Test a point outside the domain (should return None)
    # New York coordinates: approx. 40.71°N, -74.01°E
    new_york = dwd_grid.findPointXy(40.71, -74.01)
    assert new_york is None

    # Test gridpoint to coordinate conversion
    if berlin is not None:
        x, y = berlin
        lat, lon = dwd_grid.getCoordinates(x, y)
        # Check that we get close to the original coordinates
        assert abs(lat - 52.52) < 0.05
        assert abs(lon - 13.40) < 0.05


def test_cached_property_computation():
    """Test that latitude and longitude arrays are lazily computed."""
    grid = RegularLatLonGrid(
        lat_start=0.0,
        lat_steps=10,
        lat_step_size=1.0,
        lon_start=0.0,
        lon_steps=20,
        lon_step_size=1.0
    )

    # Access latitude array
    lat1 = grid.latitude
    # Access it again, should be the cached value
    lat2 = grid.latitude

    # Check that we get the same array (same memory)
    assert lat1 is lat2


def test_time_to_chunk_index():
    """Test conversion from timestamp to chunk index."""
    domain = DOMAINS["dwd_icon_d2"]

    # Create test timestamp (2023-01-01 12:00:00 UTC)
    timestamp = np.datetime64('2023-01-01T12:00:00')

    # Calculate expected chunk index
    # Seconds since epoch = (2023-01-01 12:00:00 - 1970-01-01 00:00:00) seconds
    # chunk_index = seconds_since_epoch / (file_length * temporal_resolution_seconds)
    epoch = np.datetime64('1970-01-01T00:00:00')
    seconds_since_epoch = (timestamp - epoch) / np.timedelta64(1, 's')
    expected_chunk = int(seconds_since_epoch / (domain.file_length * domain.temporal_resolution_seconds))

    # Test the time_to_chunk_index function
    chunk_index = domain.time_to_chunk_index(timestamp)
    assert chunk_index == expected_chunk


def test_get_chunk_time_range():
    """Test getting time range for a specific chunk."""
    domain = DOMAINS["dwd_icon_d2"]

    # Test chunk 1000
    chunk_index = 1000
    time_range = domain.get_chunk_time_range(chunk_index)

    # Check that we get the expected number of time points
    assert len(time_range) == domain.file_length

    # Check that time points are evenly spaced
    time_diff = time_range[1] - time_range[0]
    assert time_diff == np.timedelta64(domain.temporal_resolution_seconds, 's')
