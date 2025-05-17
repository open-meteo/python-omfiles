import numpy as np
import pytest
from omfiles.grids import ProjectionGrid, StereographicProjection
from omfiles.om_domains import RegularLatLonGrid

# Fixtures for grids

@pytest.fixture
def local_regular_lat_lon_grid():
    return RegularLatLonGrid(
        lat_start=0.0,
        lat_steps=10,
        lat_step_size=1.0,
        lon_start=0.0,
        lon_steps=20,
        lon_step_size=1.0
    )

@pytest.fixture
def stereographic_projection():
    projection = StereographicProjection(90.0, 249.0, 6371229.0)
    return ProjectionGrid.from_bounds(
        nx=935,
        ny=824,
        lat_range=(18.14503, 45.405453),
        lon_range=(217.10745, 349.8256),
        projection=projection
    )

def test_regular_grid_findPointXy_inside(local_regular_lat_lon_grid):
    # Test exact grid points
    assert local_regular_lat_lon_grid.findPointXy(5.0, 10.0) == (10, 5)
    assert local_regular_lat_lon_grid.findPointXy(0.0, 0.0) == (0, 0)
    assert local_regular_lat_lon_grid.findPointXy(9.0, 19.0) == (19, 9)

    # Test points that should round to grid points
    assert local_regular_lat_lon_grid.findPointXy(5.1, 10.2) == (10, 5)
    assert local_regular_lat_lon_grid.findPointXy(0.4, 0.4) == (0, 0)


def test_regular_grid_findPointXy_outside(local_regular_lat_lon_grid):
    # Test points outside of grid
    assert local_regular_lat_lon_grid.findPointXy(-1.0, 10.0) is None
    assert local_regular_lat_lon_grid.findPointXy(5.0, -1.0) is None
    assert local_regular_lat_lon_grid.findPointXy(10.0, 10.0) is None
    assert local_regular_lat_lon_grid.findPointXy(5.0, 20.0) is None


def test_global_grid_wrapping():
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

def test_grid_coordinates(local_regular_lat_lon_grid):
    # Test exact grid points
    assert local_regular_lat_lon_grid.getCoordinates(0, 0) == (0.0, 0.0)
    assert local_regular_lat_lon_grid.getCoordinates(5, 2) == (2.0, 5.0)

    # Test round-trip conversion
    lat, lon = 8.0, 15.0
    result = local_regular_lat_lon_grid.findPointXy(lat, lon)
    assert result is not None, f"Could not find grid point for ({lat}, {lon})"
    x, y = result
    assert local_regular_lat_lon_grid.getCoordinates(x, y) == (lat, lon)


def test_cached_property_computation(local_regular_lat_lon_grid):
    lat1 = local_regular_lat_lon_grid.latitude
    lat2 = local_regular_lat_lon_grid.latitude

    # Check that we get the same array (same memory)
    assert lat1 is lat2


def test_stereographic(stereographic_projection):
    #https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Tests/AppTests/DataTests.swift#L248
    pos_x, pos_y = stereographic_projection.findPointXy(lat=64.79836, lon=241.40111)

    assert pos_x == 420
    assert pos_y == 468

    # Get the coordinates back
    lat, lon = stereographic_projection.getCoordinates(pos_x, pos_y)
    assert abs(lat - 64.79836) < 1e-4
    assert np.mod(abs(lon - 241.40111), 360) < 1e-4

def test_grid_properties(stereographic_projection):
    assert stereographic_projection.shape == (824, 935)
    assert stereographic_projection.grid_type == "projection"

def test_out_of_bounds(stereographic_projection):
    far_point = stereographic_projection.findPointXy(30.0, 120.0)
    assert far_point is None

def test_latitude_longitude_arrays(stereographic_projection):
    # Get latitude and longitude arrays
    lats = stereographic_projection.latitude
    lons = stereographic_projection.longitude

    # Check shapes match the grid
    assert lats.shape == (824, 935)
    assert lons.shape == (824, 935)
