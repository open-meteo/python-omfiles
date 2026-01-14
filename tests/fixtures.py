import pytest


@pytest.fixture
def icon_d2_meta_json() -> str:
    # return meta_str
    return '{"chunk_time_length":121,"crs_wkt":"GEOGCRS[\\"WGS 84\\",\\n    DATUM[\\"World Geodetic System 1984\\",\\n        ELLIPSOID[\\"WGS 84\\",6378137,298.257223563]],\\n    CS[ellipsoidal,2],\\n        AXIS[\\"latitude\\",north],\\n        AXIS[\\"longitude\\",east],\\n        ANGLEUNIT[\\"degree\\",0.0174532925199433]\\n    USAGE[\\n        SCOPE[\\"grid\\"],\\n        BBOX[43.18,-3.94,58.08,20.339998]]]","data_end_time":1768503600,"last_run_availability_time":1768332519,"last_run_initialisation_time":1768327200,"last_run_modification_time":1768332519,"temporal_resolution_seconds":3600,"update_interval_seconds":10800}'
