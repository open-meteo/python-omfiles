import numpy as np
import pytest
from omfiles.om_grid import OmMetaJson


@pytest.fixture
def icon_d2_meta_json() -> str:
    # return meta_str
    return '{"chunk_time_length":121,"crs_wkt":"GEOGCRS[\\"WGS 84\\",\\n    DATUM[\\"World Geodetic System 1984\\",\\n        ELLIPSOID[\\"WGS 84\\",6378137,298.257223563]],\\n    CS[ellipsoidal,2],\\n        AXIS[\\"latitude\\",north],\\n        AXIS[\\"longitude\\",east],\\n        ANGLEUNIT[\\"degree\\",0.0174532925199433]\\n    USAGE[\\n        SCOPE[\\"grid\\"],\\n        BBOX[43.18,-3.94,58.08,20.339998]]]","data_end_time":1768503600,"last_run_availability_time":1768332519,"last_run_initialisation_time":1768327200,"last_run_modification_time":1768332519,"temporal_resolution_seconds":3600,"update_interval_seconds":10800}'


@pytest.fixture
def icon_d2_meta(icon_d2_meta_json: str) -> OmMetaJson:
    return OmMetaJson.from_metajson_string(icon_d2_meta_json)


def test_meta_json_creation(icon_d2_meta_json: str):
    """Test creation of OmMetaJson object from JSON string."""
    meta = OmMetaJson.from_metajson_string(icon_d2_meta_json)
    assert meta.chunk_time_length == 121


def test_time_to_chunk_index(icon_d2_meta: OmMetaJson):
    """Test conversion from timestamp to chunk index."""

    # Create test timestamp (2023-01-01 12:00:00 UTC)
    timestamp = np.datetime64("2023-01-01T12:00:00")

    # Calculate expected chunk index
    # Seconds since epoch = (2023-01-01 12:00:00 - 1970-01-01 00:00:00) seconds
    # chunk_index = seconds_since_epoch / (file_length * temporal_resolution_seconds)
    epoch = np.datetime64("1970-01-01T00:00:00")
    seconds_since_epoch = (timestamp - epoch) / np.timedelta64(1, "s")
    expected_chunk = int(
        seconds_since_epoch / (icon_d2_meta.chunk_time_length * icon_d2_meta.temporal_resolution_seconds)
    )

    # Test the time_to_chunk_index function
    chunk_index = icon_d2_meta.time_to_chunk_index(timestamp)
    assert chunk_index == expected_chunk


def test_get_chunk_time_range(icon_d2_meta: OmMetaJson):
    """Test getting time range for a specific chunk."""

    # Test chunk 1000
    chunk_index = 1000
    time_range = icon_d2_meta.get_chunk_time_range(chunk_index)

    # Check that we get the expected number of time points
    assert len(time_range) == icon_d2_meta.chunk_time_length

    # Check that time points are evenly spaced
    time_diff = time_range[1] - time_range[0]
    assert time_diff == np.timedelta64(icon_d2_meta.temporal_resolution_seconds, "s")
