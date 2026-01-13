from typing import cast

import numpy as np
import pyproj
import pytest
from omfiles._utils import _normalize_longitude
from omfiles.grids import (
    LambertAzimuthalEqualAreaProjection,
    LambertConformalConicProjection,
    ProjectionGrid,
    ProjProjection,
    RotatedLatLonProjection,
    StereographicProjection,
)
from omfiles.om_grid import OmGrid


@pytest.fixture
def stereographic_projection():
    projection = StereographicProjection(90.0, 249.0, 6371229.0)
    return ProjectionGrid.from_bounds(
        nx=935, ny=824, lat_range=(18.14503, 45.405453), lon_range=(217.10745, 349.8256), projection=projection
    )


# Fixtures for grids
@pytest.fixture
def icon_global_grid():
    wkt = 'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563]],CS[ellipsoidal,2],AXIS["latitude",north],AXIS["longitude",east],ANGLEUNIT["degree",0.0174532925199433]USAGE[SCOPE["grid"],BBOX[-90.0,-180.0,90.0,179.75]]]'
    return OmGrid(wkt, (1441, 2879))


@pytest.fixture
def hrdps_grid():
    wkt = 'GEOGCRS["Rotated Lat/Lon",BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563]]],DERIVINGCONVERSION["Rotated Lat/Lon",METHOD["PROJ ob_tran o_proj=longlat"],PARAMETER["o_lon_p",0],PARAMETER["o_lat_p",36.0885],PARAMETER["lon_0",245.305]]CS[ellipsoidal,2],AXIS["latitude",north],AXIS["longitude",east],ANGLEUNIT["degree",0.0174532925199433],USAGE[SCOPE["grid"],BBOX[39.626034,-133.62952,47.87646,-40.708527]]]'
    return OmGrid(wkt, (1290, 2540))


@pytest.fixture
def ukmo2_wkt():
    return 'PROJCRS["Lambert Azimuthal Equal-Area",\n    BASEGEOGCRS["GCS_Sphere",DATUM["D_Sphere",ELLIPSOID["Sphere",6371229.0,0.0]]],\n    CONVERSION["Lambert Azimuthal Equal-Area",\n        METHOD["Lambert Azimuthal Equal-Area"],\n        PARAMETER["Latitude of natural origin", 54.9],\n        PARAMETER["Longitude of natural origin", -2.5],\n        PARAMETER["False easting", 0.0],\n        PARAMETER["False northing", 0.0]],\n    CS[Cartesian,2],\n        AXIS["easting",east],\n        AXIS["northing",north],\n        LENGTHUNIT["metre",1.0],\n    USAGE[\n        SCOPE["grid"],\n        BBOX[44.508755,-17.152863,61.92511,15.352753]]]'


@pytest.fixture
def ukmo2_grid(ukmo2_wkt):
    return OmGrid(ukmo2_wkt, (970, 1042))


@pytest.fixture
def gfs_nam_conus_grid():
    wkt = 'PROJCRS["Lambert Conic Conformal",\n    BASEGEOGCRS["GCS_Sphere",DATUM["D_Sphere",ELLIPSOID["Sphere",6371229.0,0.0]]],\n    CONVERSION["Lambert Conic Conformal",\n        METHOD["Lambert Conic Conformal (2SP)"],\n        PARAMETER["Latitude of 1st standard parallel",38.5],\n        PARAMETER["Latitude of 2nd standard parallel",38.5],\n        PARAMETER["Latitude of false origin",0.0],\n        PARAMETER["Longitude of false origin",-97.5]],\n    CS[Cartesian,2],\n        AXIS["easting",east],\n        AXIS["northing",north],\n        LENGTHUNIT["metre",1],\n    USAGE[\n        SCOPE["grid"],\n        BBOX[21.137995,-122.72,47.842403,-60.918]]]'
    return OmGrid(wkt, (1059, 1799))


def test_regular_grid(icon_global_grid: OmGrid):
    assert icon_global_grid.find_point_xy(-90, -180) == (0, 0)
    assert icon_global_grid.find_point_xy(-90, 179.75) == (2878, 0)
    assert icon_global_grid.find_point_xy(90, -180) == (0, 1440)
    assert icon_global_grid.find_point_xy(90, 179.75) == (2878, 1440)
    assert icon_global_grid.find_point_xy(0, 0) == (1440, 720)


def test_regular_grid_roundtrip(icon_global_grid: OmGrid):
    lat, lon = 8.0, 15.0
    result = icon_global_grid.find_point_xy(lat, lon)
    assert result is not None, f"Could not find grid point for ({lat}, {lon})"
    x, y = result
    assert icon_global_grid.get_coordinates(x, y) == (lat, lon)


def test_cached_property_computation(icon_global_grid: OmGrid):
    lat1 = icon_global_grid.latitude
    lat2 = icon_global_grid.latitude

    # Check that we get the same array (same memory)
    assert lat1 is lat2


# def test_stereographic(stereographic_projection):
#     # https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Tests/AppTests/DataTests.swift#L248
#     pos_x, pos_y = stereographic_projection.findPointXy(lat=64.79836, lon=241.40111)

#     assert pos_x == 420
#     assert pos_y == 468

#     # Get the coordinates back
#     lat, lon = stereographic_projection.getCoordinates(pos_x, pos_y)
#     assert abs(lat - 64.79836) < 1e-4
#     assert np.mod(abs(lon - 241.40111), 360) < 1e-4


# def test_grid_properties(stereographic_projection):
#     assert stereographic_projection.shape == (824, 935)
#     assert stereographic_projection.grid_type == "projection"


# def test_out_of_bounds(stereographic_projection):
#     far_point = stereographic_projection.findPointXy(30.0, 120.0)
#     assert far_point is None


# def test_latitude_longitude_arrays(stereographic_projection):
#     # Get latitude and longitude arrays
#     lats = stereographic_projection.latitude
#     lons = stereographic_projection.longitude

#     # Check shapes match the grid
#     assert lats.shape == (824, 935)
#     assert lons.shape == (824, 935)


def test_hrdps_grid(hrdps_grid: OmGrid):
    """Test the HRDPS Continental grid with a modified approach"""
    test_points = [
        # lat, lon, expected_x, expected_y
        (39.626034, -133.62952, 0, 0),  # Bottom-left
        # FIXME: Bottom-right point is not valid for HRDPS grid
        # (27.284597, -66.96642, 2539, 0),  # Bottom-right
        (38.96126, -73.63256, 2032, 283),  # Middle point
        (47.876457, -40.708557, 2539, 1289),  # Top-right
    ]

    for lat, lon, expected_x, expected_y in test_points:
        # Test finding grid point
        pos = hrdps_grid.find_point_xy(lat=lat, lon=lon)
        assert pos is not None, f"Could not find point for {lat}, {lon}"

        x, y = pos
        assert x == expected_x, f"X mismatch: got {x}, expected {expected_x}"
        assert y == expected_y, f"Y mismatch: got {y}, expected {expected_y}"

        lat2, lon2 = hrdps_grid.get_coordinates(x, y)
        assert abs(lat2 - lat) < 0.001, f"latitude mismatch: got {lat2}, expected {lat}"
        assert abs(lon2 - lon) < 0.001, f"longitude mismatch: got {lon2}, expected {lon}"


def test_lambert_azimuthal_equal_area_projection(ukmo2_grid: OmGrid):
    """
    Test the Lambert Azimuthal Equal-Area projection.
    https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Tests/AppTests/DataTests.swift#L189
    """
    test_lon = 10.620785
    test_lat = 57.745566

    point_xy = ukmo2_grid.find_point_xy(lat=test_lat, lon=test_lon)
    assert point_xy is not None, "Point not found in grid"
    x_idx, y_idx = point_xy
    assert x_idx == 966
    assert y_idx == 713

    lat2, lon2 = ukmo2_grid.get_coordinates(x_idx, y_idx)
    assert abs(lon2 - 10.6271515) < 0.0001
    assert abs(lat2 - 57.746563) < 0.0001


def test_lambert_conformal(gfs_nam_conus_grid: OmGrid):
    """
    Based on Based on: https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Tests/AppTests/DataTests.swift#L128
    """

    point_xy = gfs_nam_conus_grid.find_point_xy(lat=34, lon=-118)
    assert point_xy is not None
    x_idx, y_idx = point_xy
    flat_idx = y_idx * gfs_nam_conus_grid.nx + x_idx
    assert flat_idx == 777441

    lat2, lon2 = gfs_nam_conus_grid.get_coordinates(x_idx, y_idx)
    assert abs(lat2 - 34) < 0.01
    assert abs(lon2 - (-118)) < 0.1

    # Test reference grid points
    reference_points = [
        (21.137999999999987, 237.28 - 360, 0),
        (24.449714395051082, 265.54789437771944 - 360, 10000),
        (22.73382904757237, 242.93190409785294 - 360, 20000),
        (24.37172305316154, 271.6307003393202 - 360, 30000),
        (24.007414634071907, 248.77817290935954 - 360, 40000),
    ]

    for lat, lon, expected_idx in reference_points:
        point_xy = gfs_nam_conus_grid.find_point_xy(lat=lat, lon=lon)
        assert point_xy is not None
        x_idx, y_idx = point_xy
        flat_idx = y_idx * gfs_nam_conus_grid.nx + x_idx
        assert flat_idx == expected_idx

        lat2, lon2 = gfs_nam_conus_grid.get_coordinates(x_idx, y_idx)
        assert abs(lat2 - lat) < 0.001
        assert abs(lon2 - lon) < 0.001


def test_nbm_grid():
    """
    Test the NBM (National Blend of Models) grid using Lambert Conformal Conic projection.
    https://vlab.noaa.gov/web/mdl/nbm-grib2-v4.0
    https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Tests/AppTests/DataTests.swift#L94
    """
    # Create projection with appropriate parameters
    proj = LambertConformalConicProjection(lambda_0=265 - 360, phi_0=0, phi_1=25, phi_2=25, radius=6371200)

    # Create grid
    grid = ProjectionGrid.from_center(
        projection=proj, nx=2345, ny=1597, center_lat=19.229, center_lon=233.723 - 360, dx=2539.7, dy=2539.7
    )

    # Test forward projection of grid origin
    x, y = proj.forward(latitude=19.229, longitude=233.723 - 360)
    assert abs(x - (-3271192.6)) < 0.1
    assert abs(y - 2604269.4) < 0.1

    # Test grid point lookup
    point_xy = grid.findPointXy(lat=19.229, lon=233.723 - 360)
    assert point_xy is not None
    assert point_xy[0] == 0
    assert point_xy[1] == 0

    # Test reference grid points directly from grib files
    reference_points = [
        (21.137999999999987, 237.28 - 360, 117411),
        (24.449714395051082, 265.54789437771944 - 360, 188910),
        (22.73382904757237, 242.93190409785294 - 360, 180965),
        (24.37172305316154, 271.6307003393202 - 360, 196187),
        (24.007414634071907, 248.77817290935954 - 360, 232796),
    ]

    for lat, lon, expected_idx in reference_points:
        point_xy = grid.findPointXy(lat=lat, lon=lon)
        assert point_xy is not None
        x_idx, y_idx = point_xy
        flat_idx = y_idx * grid.nx + x_idx
        assert flat_idx == expected_idx

    # Test grid coordinate lookup for specific indices
    reference_coords = [
        (0, 19.228992, -126.27699),
        (10000, 21.794254, -111.44652),
        (20000, 22.806227, -96.18898),
        (30000, 22.222015, -80.87921),
        (40000, 20.274399, -123.18192),
    ]

    for idx, expected_lat, expected_lon in reference_coords:
        y_idx = idx // grid.nx
        x_idx = idx % grid.nx
        lat, lon = grid.getCoordinates(x_idx, y_idx)
        assert abs(lat - expected_lat) < 0.001
        assert abs(lon - expected_lon) < 0.001


def test_lambert_conformal_conic_projection():
    """
    Test the Lambert Conformal Conic projection.
    Based on: https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Tests/AppTests/DataTests.swift#L163
    """
    proj = LambertConformalConicProjection(lambda_0=352, phi_0=55.5, phi_1=55.5, phi_2=55.5, radius=6371229)

    center_lat = 39.671
    center_lon = -25.421997

    grid = ProjectionGrid.from_center(
        nx=1906, ny=1606, center_lat=center_lat, center_lon=center_lon, dx=2000, dy=2000, projection=proj
    )

    # Test forward projection
    origin_x, origin_y = proj.forward(latitude=center_lat, longitude=center_lon)
    assert abs(origin_x - (-1527524.624)) < 0.001
    assert abs(origin_y - (-1588681.042)) < 0.001
    lat, lon = proj.inverse(origin_x, origin_y)
    assert abs(center_lat - lat) < 0.0001
    assert abs(center_lon - lon) < 0.0001

    # Test another point
    test_lat = 39.675304
    test_lon = -25.400146
    x1, y1 = proj.forward(latitude=test_lat, longitude=test_lon)
    assert abs(origin_x - x1 - (-1998.358)) < 0.001
    assert abs(origin_y - y1 - (-0.187)) < 0.001
    lat, lon = proj.inverse(x1, y1)
    assert abs(test_lat - lat) < 0.0001
    assert abs(test_lon - lon) < 0.0001

    # Point at index 1
    lat, lon = grid.getCoordinates(1, 0)
    assert abs(lat - test_lat) < 0.001
    assert abs(lon - test_lon) < 0.001
    point_idx = grid.findPointXy(lat=test_lat, lon=test_lon)
    assert point_idx == (1, 0)

    # Coords(i: 122440, x: 456, y: 64, latitude: 42.18604, longitude: -15.30127)
    lat, lon = grid.getCoordinates(456, 64)
    assert abs(lat - 42.18604) < 0.001
    assert abs(lon - (-15.30127)) < 0.001
    point_idx = grid.findPointXy(lat=lat, lon=lon)
    assert point_idx == (456, 64)

    # Coords(i: 2999780, x: 1642, y: 1573, latitude: 64.943695, longitude: 30.711975)
    lat, lon = grid.getCoordinates(1642, 1573)
    assert abs(lat - 64.943695) < 0.001
    assert abs(lon - 30.711975) < 0.001
    point_idx = grid.findPointXy(lat=lat, lon=lon)
    assert point_idx == (1642, 1573)


def test_rotated_latlon_against_proj():
    # Create our custom projection
    lat_origin = -36.0885
    lon_origin = 245.305
    custom_proj = RotatedLatLonProjection(lat_origin=lat_origin, lon_origin=lon_origin)

    # Create equivalent PROJ projection
    proj_string = (
        f"+proj=ob_tran +o_proj=longlat +o_lat_p={-lat_origin} "
        f"+o_lon_p=0.0 +lon_0={lon_origin} +datum=WGS84 +no_defs +type=crs"
    )
    proj_proj = pyproj.Proj(proj_string)

    # Test points covering different regions
    test_points = [
        (0, 0),  # Origin
        (45, 45),  # Mid-latitude point
        (-45, -45),  # Mid-latitude point (southern hemisphere)
        (10, 50),  # Europe
        (40, -100),  # North America
        (50, -170),  # Pacific
        (-30, 170),  # South Pacific
    ]

    for lat, lon in test_points:
        # Forward transformation using our implementation
        custom_x, custom_y = custom_proj.forward(latitude=lat, longitude=lon)

        # Forward transformation using PROJ
        # Note: PROJ expects (lon, lat) order, not (lat, lon)
        # proj_x, proj_y = proj_proj(np.radians(lon), np.radians(lat))
        proj_x, proj_y = proj_proj(lon, lat)
        # The following fix should be available in proj, but something is weird
        # with radians/degrees with ob_tran....
        # https://github.com/OSGeo/PROJ/issues/2804
        proj_x = np.degrees(proj_x)
        proj_y = np.degrees(proj_y)

        # Compare results - allowing for small differences due to floating point math
        # Convert to radians for comparison since our implementation works in radians
        assert abs(custom_x - proj_x) < 1e-5, f"X mismatch for ({lat}, {lon}): custom={custom_x}, proj={proj_x}"
        assert abs(custom_y - proj_y) < 1e-5, f"Y mismatch for ({lat}, {lon}): custom={custom_y}, proj={proj_y}"

        # Test inverse transformation
        custom_lat, custom_lon = custom_proj.inverse(x=custom_x, y=custom_y)
        # PROJ expects inverse=True for inverse transform
        proj_lon, proj_lat = proj_proj(np.radians(proj_x), np.radians(proj_y), inverse=True)

        # Compare results
        assert abs(custom_lat - proj_lat) < 1e-5, (
            f"Lat mismatch for ({custom_x}, {custom_y}): custom={custom_lat}, proj={proj_lat}"
        )
        assert abs(np.mod(custom_lon - proj_lon + 180, 360) - 180) < 1e-5, (
            f"Lon mismatch for ({custom_x}, {custom_y}): custom={custom_lon}, proj={proj_lon}"
        )


def test_stereographic_against_proj():
    # Create our custom projection
    latitude = 90.0  # North pole
    longitude = 249.0
    radius = 6371229.0
    custom_proj = StereographicProjection(latitude=latitude, longitude=longitude, radius=radius)

    # Create equivalent PROJ projection
    proj_string = f"+proj=stere +lat_0={latitude} +lon_0={longitude} +k=1 +x_0=0 +y_0=0 +R={radius} +units=m +no_defs"
    proj_proj = pyproj.Proj(proj_string)

    # Test points - staying away from singular points (poles)
    test_points = [
        (0, 0),  # Equator
        (45, 45),  # Mid-latitude
        (60, -120),  # Northern regions
        (45, 249),  # Along the central meridian
        (70, 249),  # Along the central meridian
        (80, 249),  # Along the central meridian
    ]

    for lat, lon in test_points:
        # Forward transformation using our implementation
        custom_x, custom_y = custom_proj.forward(latitude=lat, longitude=lon)

        # Forward transformation using PROJ
        # PROJ uses (lon, lat) order
        proj_x, proj_y = proj_proj(lon, lat)

        # Compare results (allowing some tolerance due to potential differences in algorithms)
        # Stereographic projections can have larger errors for points far from the center
        tolerance = 1  # tolerance in meters
        assert abs(custom_x - proj_x) < tolerance, f"X mismatch for ({lat}, {lon}): custom={custom_x}, proj={proj_x}"
        assert abs(custom_y - proj_y) < tolerance, f"Y mismatch for ({lat}, {lon}): custom={custom_y}, proj={proj_y}"

        # Test inverse transformation
        custom_lat, custom_lon = custom_proj.inverse(x=custom_x, y=custom_y)

        proj_lon, proj_lat = proj_proj(proj_x, proj_y, inverse=True)

        # Compare results
        assert abs(custom_lat - proj_lat) < 1e-5, f"Lat mismatch: custom={custom_lat}, proj={proj_lat}"
        custom_lon = _normalize_longitude(custom_lon)
        assert abs(custom_lon - proj_lon) < 1e-4, f"Lon mismatch: custom={custom_lon}, proj={proj_lon}"


def test_lambert_azimuthal_equal_area_against_proj(ukmo2_wkt):
    # Create our custom projection
    lambda_0 = -2.5  # Central longitude in degrees
    phi_1 = 54.9  # Standard parallel/latitude in degrees
    radius = 6371229.0  # Earth radius in meters
    custom_proj = LambertAzimuthalEqualAreaProjection(lambda_0=lambda_0, phi_1=phi_1, radius=radius)

    # Create equivalent PROJ projection
    # For Lambert Azimuthal Equal Area, we use lat_0 for the standard parallel and lon_0 for central longitude
    proj_string = f"+proj=laea +lat_0={phi_1} +lon_0={lambda_0} +x_0=0 +y_0=0 +R={radius} +units=m +no_defs +type=crs"
    proj_proj = pyproj.Proj(proj_string)
    print(proj_proj)
    proj_proj = pyproj.Proj(ukmo2_wkt)
    print(proj_proj)

    # Test points covering different regions
    test_points = [
        (0, 0),  # Origin
        (54.9, -2.5),  # Projection center (should map to 0,0)
        (45, 45),  # Mid-latitude point
        (-45, -45),  # Mid-latitude point (southern hemisphere)
        (10, 50),  # Europe
        (40, -100),  # North America
        (50, -170),  # Pacific
        (-30, 170),  # South Pacific
        # Test point from the existing test
        (57.745566, 10.620785),
    ]

    for lat, lon in test_points:
        # Forward transformation using our implementation
        custom_x, custom_y = custom_proj.forward(latitude=lat, longitude=lon)

        # Forward transformation using PROJ
        # Note: PROJ expects (lon, lat) order, not (lat, lon)
        proj_x, proj_y = proj_proj(lon, lat)

        # Compare results - Lambert projections can have larger differences due to algorithmic differences
        # Use a reasonable tolerance (e.g., 0.1 meter for a 6.3 million meter radius)
        tolerance = 0.1
        assert abs(custom_x - proj_x) < tolerance, f"X mismatch for ({lat}, {lon}): custom={custom_x}, proj={proj_x}"
        assert abs(custom_y - proj_y) < tolerance, f"Y mismatch for ({lat}, {lon}): custom={custom_y}, proj={proj_y}"

        # Test inverse transformation (skip points very close to the poles where inverse can be unstable)
        if abs(lat) < 89:
            custom_lat, custom_lon = custom_proj.inverse(x=custom_x, y=custom_y)
            # PROJ expects inverse=True for inverse transform
            proj_lon, proj_lat = proj_proj(proj_x, proj_y, inverse=True)

            # Compare results with appropriate tolerance
            # For inverse transformations, angular differences can be larger
            angular_tolerance = 1e-5  # roughly 0.00001 degrees
            assert abs(custom_lat - proj_lat) < angular_tolerance, (
                f"Lat mismatch for ({custom_x}, {custom_y}): custom={custom_lat}, proj={proj_lat}"
            )

            # Handle longitude wraparound for comparison
            lon_diff = np.mod(abs(custom_lon - proj_lon), 360)
            assert min(lon_diff, 360 - lon_diff) < angular_tolerance, (
                f"Lon mismatch for ({custom_x}, {custom_y}): custom={custom_lon}, proj={proj_lon}"
            )


def test_lambert_conformal_conic_against_proj():
    # Create our custom projection with parameters from the existing test
    lambda_0 = 352  # Reference longitude in degrees
    phi_0 = 55.5  # Reference latitude in degrees
    phi_1 = 55.5  # First standard parallel in degrees
    phi_2 = 55.5  # Second standard parallel in degrees
    radius = 6371229.0  # Earth radius in meters

    custom_proj = LambertConformalConicProjection(
        lambda_0=lambda_0, phi_0=phi_0, phi_1=phi_1, phi_2=phi_2, radius=radius
    )

    lambda_0_norm = _normalize_longitude(lambda_0)
    # Create equivalent PROJ projection
    # For Lambert Conformal Conic, we use lat_0, lon_0, lat_1, lat_2 parameters
    proj_string = (
        f"+proj=lcc +lat_0={phi_0} +lon_0={lambda_0_norm} +lat_1={phi_1} +lat_2={phi_2} "
        f"+x_0=0 +y_0=0 +R={radius} +units=m +no_defs +type=crs"
    )
    proj_proj = pyproj.Proj(proj_string)

    # Test points from the existing test
    center_lat = 39.671
    center_lon = -25.421997
    test_points = [
        (center_lat, center_lon),  # Center point
        (39.675304, -25.400146),  # Near the center
        (42.18604, -15.30127),  # Point from the test (x=456, y=64)
        (64.943695, 30.711975),  # Point from the test (x=1642, y=1573)
        # Additional test points for broader coverage
        (0, 0),  # Origin
        (phi_0, lambda_0_norm),  # Projection origin
        (45, 0),  # Mid-latitude point
        (-45, -45),  # Southern hemisphere
        (10, 50),  # Europe
        (40, -100),  # North America
        (50, -170),  # Pacific
        (-30, 170),  # South Pacific
    ]

    for lat, lon in test_points:
        # Forward transformation using our implementation
        custom_x, custom_y = custom_proj.forward(latitude=lat, longitude=lon)

        # Forward transformation using PROJ
        # Note: PROJ expects (lon, lat) order, not (lat, lon)
        proj_x, proj_y = proj_proj(lon, lat)
        tolerance = 0.1  # 0.1 meters for a 6.3 million meter radius is a reasonable precision
        assert abs(custom_x - proj_x) < tolerance, f"X mismatch for ({lat}, {lon}): custom={custom_x}, proj={proj_x}"
        assert abs(custom_y - proj_y) < tolerance, f"Y mismatch for ({lat}, {lon}): custom={custom_y}, proj={proj_y}"

        # Test inverse transformation
        custom_lat, custom_lon = custom_proj.inverse(x=custom_x, y=custom_y)
        # PROJ expects inverse=True for inverse transform
        proj_lon, proj_lat = proj_proj(proj_x, proj_y, inverse=True)
        angular_tolerance = 1e-5  # approximately 0.00001 degrees
        assert abs(custom_lat - proj_lat) < angular_tolerance, (
            f"Lat mismatch for ({custom_x}, {custom_y}): custom={custom_lat}, proj={proj_lat}"
        )

        # Handle longitude wraparound for comparison
        lon_diff = np.mod(abs(custom_lon - proj_lon), 360)
        assert min(lon_diff, 360 - lon_diff) < angular_tolerance, (
            f"Lon mismatch for ({custom_x}, {custom_y}): custom={custom_lon}, proj={proj_lon}"
        )


# def test_regular_lat_lon_grid_against_proj():
#     """Test that RegularLatLonGrid operations match proj equivalent operations"""
#     # Create a regular lat-lon grid with 1-degree steps
#     grid = RegularLatLonGrid(
#         lat_start=-90,
#         lat_steps=181,  # -90 to 90
#         lat_step_size=1.0,
#         lon_start=-180,
#         lon_steps=360,  # -180 to 180
#         lon_step_size=1.0,
#     )

#     # Create proj objects for WGS84 lat/lon
#     proj_wgs84 = pyproj.Proj(proj="latlong", datum="WGS84")

#     # Test points covering different scenarios
#     test_points: list[tuple[float, float]] = [
#         (0, 0),  # Origin
#         (45, 45),  # NE quadrant
#         (-45, -45),  # SW quadrant
#         (45, -45),  # SE quadrant
#         (-45, 45),  # NW quadrant
#         (89, 0),  # Near North pole
#         (-89, 0),  # Near South pole
#         (0, 179),  # Near date line (east)
#         (0, -179),  # Near date line (west)
#         (10, 20),  # Random point
#         (-33, 151),  # Sydney
#         (37, -122),  # San Francisco
#     ]

#     for lat, lon in test_points:
#         # Get grid coordinates using our implementation
#         grid_sel = grid.findPointXy(lat, lon)
#         assert type(grid_sel) is tuple
#         grid_x, grid_y = grid_sel
#         result_lat, result_lon = grid.getCoordinates(grid_x, grid_y)

#         # For a lat/lon grid, proj just keeps the same coordinates
#         proj_x, proj_y = proj_wgs84(lon, lat)  # Note: proj uses (lon, lat) order
#         proj_lat, proj_lon = proj_wgs84(proj_y, proj_x, inverse=True)  # Get back lat/lon from proj

#         # We'll check that our forward and inverse transformations are consistent
#         # and match with proj's (which just returns the original coordinates for this projection)

#         # Check roundtrip accuracy
#         assert abs(result_lat - lat) < 1e-9, f"Lat roundtrip error: original={lat}, result={result_lat}"

#         # Normalize longitudes before comparison due to -180/180 wrapping
#         lon_norm = _normalize_longitude(lon)
#         result_lon_norm = _normalize_longitude(result_lon)
#         assert abs(result_lon_norm - lon_norm) < 1e-9, f"Lon roundtrip error: original={lon}, result={result_lon}"

#         # Verify agreement with proj
#         assert abs(lat - proj_lat) < 1e-9
#         assert abs(lon_norm - _normalize_longitude(proj_lon)) < 1e-9, f"Lon mismatch with proj at ({lat}, {lon})"

#     # Test longitude wrapping behavior
#     wrap_test_points = [
#         (0, 185),  # Should wrap to (0, -175)
#         (0, -185),  # Should wrap to (0, 175)
#         (0, 361),  # Should wrap to (0, 1)
#         (0, -361),  # Should wrap to (0, -1)
#         (45, 540),  # Should wrap to (45, -180)
#     ]

#     for lat, lon in wrap_test_points:
#         # Find grid coordinates for the wrapped point
#         result = grid.findPointXy(lat, lon)
#         assert result is not None, "Point not found in grid"
#         grid_x, grid_y = result

#         # Find grid coordinates for the normalized longitude
#         norm_lon = cast(float, _normalize_longitude(lon))
#         result = grid.findPointXy(lat=lat, lon=norm_lon)
#         assert result is not None, "Point not found in grid"
#         norm_grid_x, norm_grid_y = result
#         assert grid_x == norm_grid_x, f"Grid X mismatch for wrapped lon: {lon} vs {norm_lon}"
#         assert grid_y == norm_grid_y, f"Grid Y mismatch for wrapped lon: {lon} vs {norm_lon}"


def test_proj_projection_grid():
    """Test that ProjProjection correctly wraps proj transformations"""
    # Test with a Lambert Conformal Conic projection
    proj_string = "+proj=lcc +lat_0=55.5 +lon_0=352 +lat_1=55.5 +lat_2=55.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"

    # Create our projection wrapper
    proj = ProjProjection(proj_string)

    # Create a grid using this projection
    grid = ProjectionGrid.from_bounds(
        nx=100, ny=100, lat_range=(39.67, 64.94), lon_range=(-25.42, 30.71), projection=proj
    )

    # Test points
    test_points = [
        (39.671, -25.421997),  # Lower left
        (64.943695, 30.711975),  # Upper right
        (50.0, 0.0),  # Middle-ish
    ]

    # Create the raw proj transformer for comparison
    raw_proj = pyproj.Proj(proj_string)

    for lat, lon in test_points:
        # Forward transformation
        grid_x, grid_y = proj.forward(lat, lon)
        raw_x, raw_y = raw_proj(lon, lat)  # Note: raw proj expects (lon, lat)

        # Compare forward results
        assert abs(grid_x - raw_x) < 1e-8, f"X mismatch: {grid_x} vs {raw_x}"
        assert abs(grid_y - raw_y) < 1e-8, f"Y mismatch: {grid_y} vs {raw_y}"

        # Inverse transformation
        back_lat, back_lon = proj.inverse(grid_x, grid_y)
        raw_lon, raw_lat = raw_proj(raw_x, raw_y, inverse=True)

        # Compare inverse results
        assert abs(back_lat - raw_lat) < 1e-8, f"Lat mismatch: {back_lat} vs {raw_lat}"
        assert abs(back_lon - raw_lon) < 1e-8, f"Lon mismatch: {back_lon} vs {raw_lon}"

        # Test roundtrip through the grid
        grid_coords = grid.findPointXy(lat=lat, lon=lon)
        assert grid_coords is not None, f"Grid coordinates not found for lat={lat}, lon={lon}"
        result_lat, result_lon = grid.getCoordinates(*grid_coords)

        # Results should match within grid resolution
        assert abs(result_lat - lat) < grid.dy, f"Grid lat error: {result_lat} vs {lat}"
        assert abs(result_lon - lon) < grid.dx, f"Grid lon error: {result_lon} vs {lon}"


def test_grid_equivalence_lcc():
    """Test that a proj-based grid matches the original implementation"""
    # Create original LambertConformalConic projection
    original_proj = LambertConformalConicProjection(lambda_0=352, phi_0=55.5, phi_1=55.5, phi_2=55.5, radius=6371229.0)

    # Create equivalent proj-based projection
    proj_proj = ProjProjection(
        "+proj=lcc +lat_0=55.5 +lon_0=352 +lat_1=55.5 +lat_2=55.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
    )

    # Create grids with both projections
    grid_bounds = {
        "nx": 100,
        "ny": 100,
        "lat_range": (39.67, 64.94),
        "lon_range": (-25.42, 30.71),
    }

    original_grid = ProjectionGrid.from_bounds(projection=original_proj, **grid_bounds)
    proj_grid = ProjectionGrid.from_bounds(projection=proj_proj, **grid_bounds)

    # Test points
    test_points = [
        (39.671, -25.421997),  # Lower left
        (64.943695, 30.711975),  # Upper right
        (50.0, 0.0),  # Middle-ish
        (45.0, -10.0),  # Random point
        (60.0, 20.0),  # Random point
    ]

    for lat, lon in test_points:
        # Compare projection results
        orig_x, orig_y = original_proj.forward(lat, lon)
        proj_x, proj_y = proj_proj.forward(lat, lon)

        # Results should match within reasonable tolerance
        assert abs(orig_x - proj_x) < 1e-3, f"X mismatch: {orig_x} vs {proj_x}"
        assert abs(orig_y - proj_y) < 1e-3, f"Y mismatch: {orig_y} vs {proj_y}"

        # Compare grid results
        orig_grid_xy = original_grid.findPointXy(lat=lat, lon=lon)
        proj_grid_xy = proj_grid.findPointXy(lat=lat, lon=lon)

        # Both should find the point or both should not find it
        assert (orig_grid_xy is None) == (proj_grid_xy is None), f"Inconsistent point finding for ({lat}, {lon})"
        if orig_grid_xy is None or proj_grid_xy is None:
            return

        # Grid coordinates should match exactly
        assert abs(orig_grid_xy[0] - proj_grid_xy[0]) < 1e-8, f"Grid X mismatch: {orig_grid_xy[0]} vs {proj_grid_xy[0]}"
        assert abs(orig_grid_xy[1] - proj_grid_xy[1]) < 1e-8, f"Grid Y mismatch: {orig_grid_xy[1]} vs {proj_grid_xy[1]}"


# def test_grid_equivalence_regular_latlon():
#     """Test that a proj-based regular lat-lon grid matches the original implementation"""
#     # Create a regular lat-lon grid using RegularLatLonGrid
#     original_grid = RegularLatLonGrid(
#         lat_start=10.0, lat_steps=100, lat_step_size=0.5, lon_start=-30.0, lon_steps=120, lon_step_size=0.5
#     )

#     # Create equivalent proj-based regular lat-lon projection
#     proj_proj = ProjProjection("+proj=longlat +datum=WGS84 +no_defs")
#     proj_grid = ProjectionGrid(projection=proj_proj, nx=120, ny=100, origin=(-30.0, 10.0), dx=0.5, dy=0.5)

#     # Test points covering various areas within the grid
#     test_points = [
#         (10.0, -30.0),  # Lower left corner
#         (59.5, 29.5),  # Upper right corner
#         (35.0, 0.0),  # Middle-ish
#         (20.0, -15.0),  # Random point
#         (50.0, 20.0),  # Random point
#         (15.25, -25.75),  # Point between grid cells
#     ]

#     for lat, lon in test_points:
#         # Get grid points using both implementations
#         orig_grid_xy = original_grid.findPointXy(lat=lat, lon=lon)
#         proj_grid_xy = proj_grid.findPointXy(lat=lat, lon=lon)

#         # Both should find the point
#         assert orig_grid_xy is not None and proj_grid_xy is not None, f"Point not found for ({lat}, {lon})"

#         # If point is found, coordinates should match exactly
#         if orig_grid_xy is not None:
#             assert abs(orig_grid_xy[0] - proj_grid_xy[0]) < 1e-8, (
#                 f"Grid X mismatch: {orig_grid_xy[0]} vs {proj_grid_xy[0]}"
#             )
#             assert abs(orig_grid_xy[1] - proj_grid_xy[1]) < 1e-8, (
#                 f"Grid Y mismatch: {orig_grid_xy[1]} vs {proj_grid_xy[1]}"
#             )

#         # Test the inverse transformation (getCoordinates)
#         if orig_grid_xy is not None:
#             x, y = orig_grid_xy
#             orig_lat, orig_lon = original_grid.getCoordinates(x, y)
#             proj_lat, proj_lon = proj_grid.getCoordinates(x, y)

#             # Results should match exactly
#             assert abs(orig_lat - proj_lat) < 1e-8, f"Latitude mismatch: {orig_lat} vs {proj_lat}"
#             assert abs(orig_lon - proj_lon) < 1e-8, f"Longitude mismatch: {orig_lon} vs {proj_lon}"


def test_grid_equivalence_rotated_latlon():
    """Test that a proj-based rotated lat-lon grid matches the original implementation"""
    lat_origin = -36.0885
    lon_origin = 245.305

    original_proj = RotatedLatLonProjection(lat_origin=lat_origin, lon_origin=lon_origin)
    proj_proj = ProjProjection(
        f"+proj=ob_tran +o_proj=longlat +o_lat_p={-lat_origin} +o_lon_p=0.0 +lon_0={lon_origin} +datum=WGS84 +no_defs +type=crs"
    )

    # Grid bounds could be adjusted to something used in the open-meteo backend
    grid_bounds = {
        "nx": 100,
        "ny": 80,
        "lat_range": (-40.0, 40.0),
        "lon_range": (200.0, 300.0),
    }

    original_grid = ProjectionGrid.from_bounds(projection=original_proj, **grid_bounds)
    proj_grid = ProjectionGrid.from_bounds(projection=proj_proj, **grid_bounds)

    # Test points covering various areas within the grid
    test_points = [
        (0, 0),  # Origin
        (45, 45),  # Mid-latitude
        (-45, -45),  # Southern hemisphere
        (10, 50),  # Europe
        (40, -100),  # North America
        (50, -170),  # Pacific
        (-30, 170),  # South Pacific
    ]

    for lat, lon in test_points:
        # Compare projection results
        orig_x, orig_y = original_proj.forward(lat, lon)
        proj_x, proj_y = proj_proj.forward(lat, lon)
        assert abs(orig_x - proj_x) < 1e-3, f"X mismatch: {orig_x} vs {proj_x}"
        assert abs(orig_y - proj_y) < 1e-3, f"Y mismatch: {orig_y} vs {proj_y}"

        # Compare grid results
        orig_grid_xy = original_grid.findPointXy(lat=lat, lon=lon)
        proj_grid_xy = proj_grid.findPointXy(lat=lat, lon=lon)
        assert orig_grid_xy == proj_grid_xy, f"Grid index mismatch: {orig_grid_xy} vs {proj_grid_xy}"

        # Test the inverse transformation (getCoordinates)
        if orig_grid_xy is not None:
            x, y = orig_grid_xy
            orig_lat, orig_lon = original_grid.getCoordinates(x, y)
            proj_lat, proj_lon = proj_grid.getCoordinates(x, y)
            assert abs(orig_lat - proj_lat) < 1e-4, f"Latitude mismatch: {orig_lat} vs {proj_lat}"
            assert abs(orig_lon - proj_lon) < 1e-4, f"Longitude mismatch: {orig_lon} vs {proj_lon}"
