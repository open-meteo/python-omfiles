import pyproj
import pytest
from omfiles.grids import GaussianGrid, OmGrid


# Fixtures for grids
@pytest.fixture
def icon_global_grid():
    wkt = 'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563]],CS[ellipsoidal,2],AXIS["latitude",north],AXIS["longitude",east],ANGLEUNIT["degree",0.0174532925199433]USAGE[SCOPE["grid"],BBOX[-90.0,-180.0,90.0,179.75]]]'
    return OmGrid(wkt, (1441, 2879))


@pytest.fixture
def gem_hrdps_grid():
    wkt = 'GEOGCRS["Rotated Lat/Lon",BASEGEOGCRS["GCS_Sphere",DATUM["D_Sphere",ELLIPSOID["Sphere",6371229.0,0.0]]],DERIVINGCONVERSION["Rotated Lat/Lon",METHOD["PROJ ob_tran o_proj=longlat"],PARAMETER["o_lon_p",0],PARAMETER["o_lat_p",36.0885],PARAMETER["lon_0",245.305]]CS[ellipsoidal,2],AXIS["latitude",north],AXIS["longitude",east],ANGLEUNIT["degree",0.0174532925199433],USAGE[SCOPE["grid"],BBOX[39.626034,-133.62952,47.87646,-40.708527]]]'
    return OmGrid(wkt, (1290, 2540))


@pytest.fixture
def gem_regional_grid():
    wkt = 'PROJCRS["Stereographic",\n    BASEGEOGCRS["GCS_Sphere",DATUM["D_Sphere",ELLIPSOID["Sphere",6371229.0,0.0]]],\n    CONVERSION["Stereographic",\n        METHOD["Stereographic"],\n        PARAMETER["Latitude of natural origin", 90.0],\n        PARAMETER["Longitude of natural origin", 249.0],\n        PARAMETER["Scale factor at natural origin", 1.0],\n        PARAMETER["False easting", 0.0],\n        PARAMETER["False northing", 0.0]],\n    CS[Cartesian,2],\n        AXIS["easting",east],\n        AXIS["northing",north],\n        LENGTHUNIT["metre",1.0],\n    USAGE[\n        SCOPE["grid"],\n        BBOX[18.145027,-142.89252,45.40545,-10.174438]]]'
    return OmGrid(wkt, (824, 935))


@pytest.fixture
def ukmo2_wkt():
    return 'PROJCRS["Lambert Azimuthal Equal-Area",\n    BASEGEOGCRS["GCS_Sphere",DATUM["D_Sphere",ELLIPSOID["Sphere",6371229.0,0.0]]],\n    CONVERSION["Lambert Azimuthal Equal-Area",\n        METHOD["Lambert Azimuthal Equal-Area"],\n        PARAMETER["Latitude of natural origin", 54.9],\n        PARAMETER["Longitude of natural origin", -2.5],\n        PARAMETER["False easting", 0.0],\n        PARAMETER["False northing", 0.0]],\n    CS[Cartesian,2],\n        AXIS["easting",east],\n        AXIS["northing",north],\n        LENGTHUNIT["metre",1.0],\n    USAGE[\n        SCOPE["grid"],\n        BBOX[44.508755,-17.152863,61.92511,15.352753]]]'


@pytest.fixture
def ukmo2_grid(ukmo2_wkt):
    return OmGrid(ukmo2_wkt, (970, 1042))


@pytest.fixture
def gfs_nam_conus_wkt():
    return 'PROJCRS["Lambert Conic Conformal",\n    BASEGEOGCRS["GCS_Sphere",DATUM["D_Sphere",ELLIPSOID["Sphere",6371229.0,0.0]]],\n    CONVERSION["Lambert Conic Conformal",\n        METHOD["Lambert Conic Conformal (2SP)"],\n        PARAMETER["Latitude of 1st standard parallel",38.5],\n        PARAMETER["Latitude of 2nd standard parallel",38.5],\n        PARAMETER["Latitude of false origin",0.0],\n        PARAMETER["Longitude of false origin",-97.5]],\n    CS[Cartesian,2],\n        AXIS["easting",east],\n        AXIS["northing",north],\n        LENGTHUNIT["metre",1],\n    USAGE[\n        SCOPE["grid"],\n        BBOX[21.137995,-122.72,47.842403,-60.918]]]'


@pytest.fixture
def gfs_nam_conus_grid(gfs_nam_conus_wkt):
    return OmGrid(gfs_nam_conus_wkt, (1059, 1799))


@pytest.fixture
def nbm_conus_grid():
    wkt = 'PROJCRS["Lambert Conic Conformal",\n    BASEGEOGCRS["GCS_Sphere",DATUM["D_Sphere",ELLIPSOID["Sphere",6371229.0,0.0]]],\n    CONVERSION["Lambert Conic Conformal",\n        METHOD["Lambert Conic Conformal (2SP)"],\n        PARAMETER["Latitude of 1st standard parallel",25.0],\n        PARAMETER["Latitude of 2nd standard parallel",25.0],\n        PARAMETER["Latitude of false origin",0.0],\n        PARAMETER["Longitude of false origin",-95.0]],\n    CS[Cartesian,2],\n        AXIS["easting",east],\n        AXIS["northing",north],\n        LENGTHUNIT["metre",1],\n    USAGE[\n        SCOPE["grid"],\n        BBOX[19.228985,-126.27699,54.372913,-59.042786]]]'
    return OmGrid(wkt, (1597, 2345))


@pytest.fixture
def dmi_harmoni_europe_wkt():
    return 'PROJCRS["Lambert Conic Conformal",\n    BASEGEOGCRS["GCS_Sphere",DATUM["D_Sphere",ELLIPSOID["Sphere",6371229.0,0.0]]],\n    CONVERSION["Lambert Conic Conformal",\n        METHOD["Lambert Conic Conformal (2SP)"],\n        PARAMETER["Latitude of 1st standard parallel",55.5],\n        PARAMETER["Latitude of 2nd standard parallel",55.5],\n        PARAMETER["Latitude of false origin",55.5],\n        PARAMETER["Longitude of false origin",352.0]],\n    CS[Cartesian,2],\n        AXIS["easting",east],\n        AXIS["northing",north],\n        LENGTHUNIT["metre",1],\n    USAGE[\n        SCOPE["grid"],\n        BBOX[39.670998,-25.421997,62.667618,40.069855]]]'


@pytest.fixture
def ecmwf_ifs_wkt():
    return 'GEOGCRS["Reduced Gaussian Grid",\n    DATUM["World Geodetic System 1984",\n        ELLIPSOID["WGS 84",6378137,298.257223563]],\n    CS[ellipsoidal,2],\n        AXIS["latitude",north],\n        AXIS["longitude",east],\n        ANGLEUNIT["degree",0.0174532925199433],\n    REMARK["Reduced Gaussian Grid O1280 (ECMWF)"],\n    USAGE[\n        SCOPE["grid"],\n        BBOX[-90,-180.0,90,180]]]'


@pytest.fixture
def ecmwf_ifs_grid(ecmwf_ifs_wkt):
    return OmGrid(ecmwf_ifs_wkt, (1, 6599680))._grid


@pytest.fixture
def dmi_harmoni_europe_grid(dmi_harmoni_europe_wkt):
    return OmGrid(dmi_harmoni_europe_wkt, (1606, 1906))


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


def test_stereographic(gem_regional_grid: OmGrid):
    # https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Tests/AppTests/DataTests.swift#L248
    indices = gem_regional_grid.find_point_xy(lat=64.79836, lon=241.40111)

    assert indices is not None
    pos_x, pos_y = indices

    assert pos_x == 420
    assert pos_y == 468

    # Get the coordinates back
    lat, lon = gem_regional_grid.get_coordinates(pos_x, pos_y)
    assert abs(lat - 64.79836) < 1e-4
    assert abs(abs(lon - 241.40111) - 360) < 1e-4


def test_stereographic_out_of_bounds(gem_regional_grid: OmGrid):
    far_point = gem_regional_grid.find_point_xy(lat=30.0, lon=120.0)
    assert far_point is None


def test_stereographic_latitude_longitude_arrays(gem_regional_grid: OmGrid):
    # Get latitude and longitude arrays
    lats = gem_regional_grid.latitude
    lons = gem_regional_grid.longitude

    # Check shapes match the grid
    assert lats.shape == (824, 935)
    assert lons.shape == (824, 935)


def test_hrdps_grid(gem_hrdps_grid: OmGrid):
    """Test the HRDPS Continental grid with a modified approach"""
    test_points = [
        # lat, lon, expected_x, expected_y
        (39.626034, -133.62952, 0, 0),  # Bottom-left
        # FIXME: Bottom-right point is not valid for HRDPS grid
        (27.284597, -66.96642, 2539, 0),  # Bottom-right
        (38.96126, -73.63256, 2032, 283),  # Middle point
        (47.876457, -40.708557, 2539, 1289),  # Top-right
    ]

    for lat, lon, expected_x, expected_y in test_points:
        # Test finding grid point
        pos = gem_hrdps_grid.find_point_xy(lat=lat, lon=lon)
        assert pos is not None, f"Could not find point for {lat}, {lon}"

        x, y = pos
        assert x == expected_x, f"X mismatch: got {x}, expected {expected_x}"
        assert y == expected_y, f"Y mismatch: got {y}, expected {expected_y}"

        lat2, lon2 = gem_hrdps_grid.get_coordinates(x, y)
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
    assert x_idx == 273
    assert y_idx == 432

    lat2, lon2 = gfs_nam_conus_grid.get_coordinates(x_idx, y_idx)
    assert abs(lat2 - 34) < 0.01
    assert abs(lon2 - (-118)) < 0.1

    # Test reference grid points
    reference_points = [
        (21.137999999999987, 237.28 - 360, 0, 0),
        (24.449714395051082, 265.54789437771944 - 360, 1005, 5),
        (22.73382904757237, 242.93190409785294 - 360, 211, 11),
        (24.37172305316154, 271.6307003393202 - 360, 1216, 16),
        (24.007414634071907, 248.77817290935954 - 360, 422, 22),
    ]

    for lat, lon, expected_x, expected_y in reference_points:
        point_xy = gfs_nam_conus_grid.find_point_xy(lat=lat, lon=lon)
        assert point_xy is not None
        x_idx, y_idx = point_xy
        assert x_idx == expected_x
        assert y_idx == expected_y

        lat2, lon2 = gfs_nam_conus_grid.get_coordinates(x_idx, y_idx)
        assert abs(lat2 - lat) < 0.001
        assert abs(lon2 - lon) < 0.001


def test_nbm_grid(nbm_conus_grid: OmGrid):
    """
    Test the NBM (National Blend of Models) grid using Lambert Conformal Conic projection.
    https://vlab.noaa.gov/web/mdl/nbm-grib2-v4.0
    https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Tests/AppTests/DataTests.swift#L94
    """
    # # Create projection with appropriate parameters
    # proj = LambertConformalConicProjection(lambda_0=265 - 360, phi_0=0, phi_1=25, phi_2=25, radius=6371200)

    # # Test forward projection of grid origin
    # x, y = proj.forward(latitude=19.229, longitude=233.723 - 360)
    # assert abs(x - (-3271192.6)) < 0.1
    # assert abs(y - 2604269.4) < 0.1

    # Test grid point lookup
    point_xy = nbm_conus_grid.find_point_xy(lat=19.229, lon=233.723 - 360)
    assert point_xy is not None
    assert point_xy[0] == 0
    assert point_xy[1] == 0

    # Test reference grid points directly from grib files
    reference_points = [
        (21.137999999999987, 237.28 - 360, 161, 50),
        (24.449714395051082, 265.54789437771944 - 360, 1310, 80),
        (22.73382904757237, 242.93190409785294 - 360, 400, 77),
        (24.37172305316154, 271.6307003393202 - 360, 1552, 83),
        (24.007414634071907, 248.77817290935954 - 360, 641, 99),
    ]

    for lat, lon, expected_x, expected_y in reference_points:
        point_xy = nbm_conus_grid.find_point_xy(lat=lat, lon=lon)
        assert point_xy is not None
        x_idx, y_idx = point_xy
        assert x_idx == expected_x
        assert y_idx == expected_y

    # Test grid coordinate lookup for specific indices
    reference_coords = [
        (0, 0, 19.228992, -126.27699),
        (4, 620, 21.794254, -111.44652),
        (8, 1240, 22.806227, -96.18898),
        (12, 1860, 22.222015, -80.87921),
        (17, 135, 20.274399, -123.18192),
    ]

    for y_idx, x_idx, expected_lat, expected_lon in reference_coords:
        lat, lon = nbm_conus_grid.get_coordinates(x_idx, y_idx)
        assert abs(lat - expected_lat) < 0.001
        assert abs(lon - expected_lon) < 0.001


def test_lambert_conformal_conic_projection(dmi_harmoni_europe_wkt: str, dmi_harmoni_europe_grid: OmGrid):
    """
    Test the Lambert Conformal Conic projection.
    Based on: https://github.com/open-meteo/open-meteo/blob/7eb49a5dd41e66ac5cf386023a0527eead3104b4/Tests/AppTests/DataTests.swift#L352
    """

    proj = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), pyproj.CRS.from_wkt(dmi_harmoni_europe_wkt))
    inverse_proj = pyproj.Transformer.from_crs(pyproj.CRS.from_wkt(dmi_harmoni_europe_wkt), pyproj.CRS.from_epsg(4326))

    center_lat = 39.671
    center_lon = -25.421997
    # Test forward projection
    origin_x, origin_y = proj.transform(center_lat, center_lon)
    assert abs(origin_x - (-1527524.624)) < 0.001
    assert abs(origin_y - (-1588681.042)) < 0.001
    lat, lon = inverse_proj.transform(origin_x, origin_y)
    assert abs(center_lat - lat) < 0.0001
    assert abs(center_lon - lon) < 0.0001

    # Test another point
    test_lat = 39.675304
    test_lon = -25.400146
    x1, y1 = proj.transform(test_lat, test_lon)
    assert abs(origin_x - x1 - (-1998.358)) < 0.001
    assert abs(origin_y - y1 - (-0.187)) < 0.001
    lat, lon = inverse_proj.transform(x1, y1)
    assert abs(test_lat - lat) < 0.0001
    assert abs(test_lon - lon) < 0.0001

    # Point at index 1
    lat, lon = dmi_harmoni_europe_grid.get_coordinates(1, 0)
    assert abs(lat - test_lat) < 0.001
    assert abs(lon - test_lon) < 0.001
    point_idx = dmi_harmoni_europe_grid.find_point_xy(lat=test_lat, lon=test_lon)
    assert point_idx == (1, 0)

    # Coords(i: 122440, x: 456, y: 64, latitude: 42.18604, longitude: -15.30127)
    lat, lon = dmi_harmoni_europe_grid.get_coordinates(456, 64)
    assert abs(lat - 42.18604) < 0.001
    assert abs(lon - (-15.30127)) < 0.001
    point_idx = dmi_harmoni_europe_grid.find_point_xy(lat=lat, lon=lon)
    assert point_idx == (456, 64)

    # Coords(i: 2999780, x: 1642, y: 1573, latitude: 64.943695, longitude: 30.711975)
    lat, lon = dmi_harmoni_europe_grid.get_coordinates(1642, 1573)
    assert abs(lat - 64.943695) < 0.001
    assert abs(lon - 30.711975) < 0.001
    point_idx = dmi_harmoni_europe_grid.find_point_xy(lat=lat, lon=lon)
    assert point_idx == (1642, 1573)


def test_ecmwf_grid(ecmwf_ifs_grid: GaussianGrid):
    # https://github.com/open-meteo/open-meteo/blob/7eb49a5dd41e66ac5cf386023a0527eead3104b4/Tests/AppTests/DataTests.swift#L614
    assert ecmwf_ifs_grid._find_point_xy(53.63797, 45) == (261, 517)
    assert ecmwf_ifs_grid._find_point_xy(19.229, 233.723 - 360) == (2625, 1006)
    assert ecmwf_ifs_grid._find_point_xy(91.0, 342) == (19, 0)
    assert ecmwf_ifs_grid._find_point_xy(-91, 342) == (19, 2559)
    assert ecmwf_ifs_grid._find_point_xy(-19.229, 233.723 - 360) == (2625, 1553)

    flat_grid_coords = ecmwf_ifs_grid.find_point_xy(89.94619, 0)
    assert flat_grid_coords is not None, "Failed to find point"
    assert flat_grid_coords == (0, 0)
    position = ecmwf_ifs_grid.get_coordinates(flat_grid_coords[1], flat_grid_coords[0])
    assert abs(position[0] - 89.94619) < 0.005
    assert abs(position[1] - 0) < 0.005
