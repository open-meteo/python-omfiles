# import numpy as np
# from omfiles.om_domains import DOMAINS


# def test_dwd_icon_d2_grid_points():
#     """Test specific points in the DWD ICON D2 grid."""
#     dwd_grid = DOMAINS["dwd_icon_d2"].grid

#     # Test a point known to be in the domain (Central Europe)
#     # Berlin coordinates: approx. 52.52°N, 13.40°E
#     berlin = dwd_grid.findPointXy(52.52, 13.40)
#     assert berlin is not None

#     # Test a point outside the domain (should return None)
#     # New York coordinates: approx. 40.71°N, -74.01°E
#     new_york = dwd_grid.findPointXy(40.71, -74.01)
#     assert new_york is None

#     # Test gridpoint to coordinate conversion
#     if berlin is not None:
#         x, y = berlin
#         lat, lon = dwd_grid.getCoordinates(x, y)
#         # Check that we get close to the original coordinates
#         assert abs(lat - 52.52) < 0.05
#         assert abs(lon - 13.40) < 0.05


# def test_ecmwf_grid():
#     """Test the ECMWF IFS grid specifically."""
#     ecmwf_grid = DOMAINS["ecmwf_ifs025"].grid

#     # Test some known points on the grid
#     # Point at the prime meridian and equator
#     assert ecmwf_grid.findPointXy(0.0, 0.0) == (720, 360)

#     # Point at the North Pole
#     assert ecmwf_grid.findPointXy(90.0, 0.0) == (720, 720)

#     # Test some edge points (ensure they are properly handled)
#     assert ecmwf_grid.findPointXy(-90.0, -180.0) == (0, 0)
#     assert ecmwf_grid.findPointXy(90.0, 180.0) == (0, 720)

#     # Test wrapping for global grid
#     # A point at longitude 181 should wrap to longitude -179
#     point1 = ecmwf_grid.findPointXy(0.0, 181.0)
#     point2 = ecmwf_grid.findPointXy(0.0, -179.0)
#     assert point1 == point2


# def test_time_to_chunk_index():
#     """Test conversion from timestamp to chunk index."""
#     domain = DOMAINS["dwd_icon_d2"]

#     # Create test timestamp (2023-01-01 12:00:00 UTC)
#     timestamp = np.datetime64("2023-01-01T12:00:00")

#     # Calculate expected chunk index
#     # Seconds since epoch = (2023-01-01 12:00:00 - 1970-01-01 00:00:00) seconds
#     # chunk_index = seconds_since_epoch / (file_length * temporal_resolution_seconds)
#     epoch = np.datetime64("1970-01-01T00:00:00")
#     seconds_since_epoch = (timestamp - epoch) / np.timedelta64(1, "s")
#     expected_chunk = int(seconds_since_epoch / (domain.file_length * domain.temporal_resolution_seconds))

#     # Test the time_to_chunk_index function
#     chunk_index = domain.time_to_chunk_index(timestamp)
#     assert chunk_index == expected_chunk


# def test_get_chunk_time_range():
#     """Test getting time range for a specific chunk."""
#     domain = DOMAINS["dwd_icon_d2"]

#     # Test chunk 1000
#     chunk_index = 1000
#     time_range = domain.get_chunk_time_range(chunk_index)

#     # Check that we get the expected number of time points
#     assert len(time_range) == domain.file_length

#     # Check that time points are evenly spaced
#     time_diff = time_range[1] - time_range[0]
#     assert time_diff == np.timedelta64(domain.temporal_resolution_seconds, "s")
