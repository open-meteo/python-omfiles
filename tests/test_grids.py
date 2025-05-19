import numpy as np
import pytest
from omfiles.grids import (
    LambertAzimuthalEqualAreaProjection,
    ProjectionGrid,
    RotatedLatLonProjection,
    StereographicProjection,
)
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

@pytest.fixture
def hrdps_projection():
    return RotatedLatLonProjection(lat_origin=-36.0885, lon_origin=245.305)

@pytest.fixture
def hrdps_grid(hrdps_projection):
    from omfiles.grids import ProjectionGrid
    return ProjectionGrid.from_bounds(
        nx=2540,
        ny=1290,
        lat_range=(39.626034, 47.876457),
        lon_range=(-133.62952, -40.708557),
        projection=hrdps_projection
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

def test_hrdps_grid(hrdps_grid):
    """Test the HRDPS Continental grid with a modified approach"""
    test_points = [
        # lat, lon, expected_x, expected_y
        (39.626034, -133.62952, 0, 0),          # Bottom-left
        (27.284597, -66.96642, 2539, 0),        # Bottom-right
        (38.96126, -73.63256, 2032, 283),       # Middle point
        (47.876457, -40.708557, 2539, 1289),    # Top-right
    ]

    for lat, lon, expected_x, expected_y in test_points:
        # Test finding grid point
        pos = hrdps_grid.findPointXy(lat=lat, lon=lon)
        assert pos is not None, f"Could not find point for {lat}, {lon}"

        x, y = pos
        assert x == expected_x, f"X mismatch: got {x}, expected {expected_x}"
        assert y == expected_y, f"Y mismatch: got {y}, expected {expected_y}"

        lat2, lon2 = hrdps_grid.getCoordinates(x, y)
        assert abs(lat2 - lat) < 0.001, f"latitude mismatch: got {lat2}, expected {lat}"
        assert abs(lon2 - lon) < 0.001, f"longitude mismatch: got {lon2}, expected {lon}"


def test_lambert_azimuthal_equal_area_projection():
    """
    Test the Lambert Azimuthal Equal-Area projection.
    https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Tests/AppTests/DataTests.swift#L189
    """
    proj = LambertAzimuthalEqualAreaProjection(lambda_0=-2.5, phi_1=54.9, radius=6371229)
    grid = ProjectionGrid(
        projection=proj,
        nx=1042,
        ny=970,
        origin=(-1158000, -1036000),
        dx=2000,
        dy=2000
    )

    test_lon = 10.620785
    test_lat = 57.745566
    x, y = proj.forward(latitude=test_lat, longitude=test_lon)
    assert abs(x - 773650.5058) < 0.0001 # TODO: There are numerical differences with the Swift test case
    assert abs(y - 389820.1483) < 0.0001 # TODO: There are numerical differences with the Swift test case

    lat, lon = proj.inverse(x=773650.5, y=389820.06)
    assert abs(lon - test_lon) < 0.0001
    assert abs(lat - test_lat) < 0.0001

    point_xy = grid.findPointXy(lat=test_lat, lon=test_lon)
    assert point_xy is not None, "Point not found in grid"
    x_idx, y_idx = point_xy
    assert x_idx == 966, f"Expected x index to be 966, got {x_idx}"
    assert y_idx == 713, f"Expected y index to be 713, got {y_idx}"

    lat2, lon2 = grid.getCoordinates(x_idx, y_idx)
    assert abs(lon2 - 10.6271515) < 0.0001, f"Expected longitude to be 10.6271515, got {lon2}"
    assert abs(lat2 - 57.746563) < 0.0001, f"Expected latitude to be 57.746563, got {lat2}"
