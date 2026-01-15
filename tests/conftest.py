import os
import tempfile

import numpy as np
import numpy.typing as npt
import pytest
from omfiles.meta import OmChunksMeta
from omfiles.omfiles import OmFileWriter

from .test_utils import create_test_om_file


@pytest.fixture
def temp_om_file():
    """
    Fixture that creates a temporary OM file filled with some data.
    Returns a path to the temporary file.
    """

    dtype: npt.DTypeLike = np.float32
    shape: tuple = (5, 5)

    # On Windows a file cannot be opened twice, so we need to close it first
    # and take care of deleting it ourselves
    with tempfile.NamedTemporaryFile(suffix=".om", delete=False) as temp_file:
        create_test_om_file(temp_file.name, shape=shape, dtype=dtype)
        temp_file.close()
        filename = temp_file.name

    yield filename

    if os.path.exists(filename):
        try:
            os.remove(filename)
        except (PermissionError, OSError) as e:
            import warnings

            warnings.warn(f"Failed to remove temporary file {filename}: {e}")


@pytest.fixture
def temp_hierarchical_om_file():
    """
    Fixture that creates a hierarchical temporary OM file filled with some data.
    Returns a path to the temporary file.
    """

    with tempfile.NamedTemporaryFile(suffix=".om", delete=False) as temp_file:
        dtype1: npt.DTypeLike = np.float32
        dtype2: npt.DTypeLike = np.int64
        shape1: tuple = (5, 5)
        shape2: tuple = (50, 5)
        test_data1 = np.arange(np.prod(shape1), dtype=dtype1).reshape(shape1)
        test_data2 = np.arange(np.prod(shape2), dtype=dtype2).reshape(shape2) * 2

        writer = OmFileWriter(temp_file.name)
        variable1 = writer.write_array(test_data1, chunks=[5, 5], name="variable1")
        variable2 = writer.write_array(test_data2, chunks=[2, 2], name="variable2")

        group = writer.write_group("root_group", children=[variable1, variable2])
        writer.close(group)
        temp_file.close()
        filename = temp_file.name

    yield filename

    if os.path.exists(filename):
        try:
            os.remove(filename)
        except (PermissionError, OSError) as e:
            import warnings

            warnings.warn(f"Failed to remove temporary file {filename}: {e}")


@pytest.fixture
def empty_temp_om_file():
    """
    Fixture that creates a temporary empty OM file.
    Returns a path to the temporary file.
    """

    # On Windows a file cannot be opened twice, so we need to close it first
    # and take care of deleting it ourselves
    with tempfile.NamedTemporaryFile(suffix=".om", delete=False) as temp_file:
        temp_file.close()
        filename = temp_file.name

    yield filename

    if os.path.exists(filename):
        try:
            os.remove(filename)
        except (PermissionError, OSError) as e:
            import warnings

            warnings.warn(f"Failed to remove temporary file {filename}: {e}")


@pytest.fixture
def icon_d2_meta_json() -> str:
    # return meta_str
    return '{"chunk_time_length":121,"crs_wkt":"GEOGCRS[\\"WGS 84\\",\\n    DATUM[\\"World Geodetic System 1984\\",\\n        ELLIPSOID[\\"WGS 84\\",6378137,298.257223563]],\\n    CS[ellipsoidal,2],\\n        AXIS[\\"latitude\\",north],\\n        AXIS[\\"longitude\\",east],\\n        ANGLEUNIT[\\"degree\\",0.0174532925199433]\\n    USAGE[\\n        SCOPE[\\"grid\\"],\\n        BBOX[43.18,-3.94,58.08,20.339998]]]","data_end_time":1768503600,"last_run_availability_time":1768332519,"last_run_initialisation_time":1768327200,"last_run_modification_time":1768332519,"temporal_resolution_seconds":3600,"update_interval_seconds":10800}'


@pytest.fixture
def icond2_om_chunks_meta(icon_d2_meta_json: str):
    return OmChunksMeta.from_metajson_string(icon_d2_meta_json)
