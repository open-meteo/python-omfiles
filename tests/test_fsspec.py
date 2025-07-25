import os
import tempfile

import numpy as np
import numpy.typing as npt
import omfiles
import pytest
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from s3fs import S3FileSystem

from .test_utils import filter_numpy_size_warning

# --- Fixtures ---


@pytest.fixture
def memory_fs():
    return MemoryFileSystem()


@pytest.fixture
def local_fs():
    return LocalFileSystem()


@pytest.fixture
def s3_test_file():
    return "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"


@pytest.fixture
def s3_backend():
    return S3FileSystem(anon=True, default_block_size=256, default_cache_type="none")


@pytest.fixture
def s3_backend_with_cache():
    s3_fs = S3FileSystem(anon=True, default_block_size=256, default_cache_type="none")
    from fsspec.implementations.cached import CachingFileSystem

    return CachingFileSystem(
        fs=s3_fs, cache_check=3600, block_size=256, cache_storage="cache", check_files=False, same_names=True
    )


@pytest.fixture
async def s3_backend_async():
    return S3FileSystem(anon=True, asynchronous=True, default_block_size=256, default_cache_type="none")


# --- Helpers ---


def create_test_data(shape=(10, 10), dtype: npt.DTypeLike = np.float32) -> np.ndarray:
    return np.arange(np.prod(shape)).reshape(shape).astype(dtype)


def write_simple_omfile(writer, data, name="test_data"):
    metadata = writer.write_scalar("Test data", name="description")
    variable = writer.write_array(
        data, chunks=[max(1, data.shape[0] // 2), max(1, data.shape[1] // 2)], name=name, children=[metadata]
    )
    writer.close(variable)


def assert_file_exists(fs, path):
    assert fs.exists(path)
    assert fs.size(path) > 0


# --- Tests ---


def test_local_read(local_fs, temp_om_file):
    reader = omfiles.OmFilePyReader.from_fsspec(local_fs, temp_om_file)
    data = reader[0:5, 0:5]
    np.testing.assert_array_equal(data, np.arange(25).reshape(5, 5))


def test_s3_read(s3_backend, s3_test_file):
    reader = omfiles.OmFilePyReader.from_fsspec(s3_backend, s3_test_file)
    data = reader[57812:60000, 0:100]
    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[0, :10], expected)


def test_s3_read_with_cache(s3_backend_with_cache, s3_test_file):
    reader = omfiles.OmFilePyReader.from_fsspec(s3_backend_with_cache, s3_test_file)
    data = reader[57812:60000, 0:100]
    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[0, :10], expected)


@pytest.mark.asyncio
async def test_s3_concurrent_read(s3_backend_async, s3_test_file):
    reader = await omfiles.OmFilePyReaderAsync.from_fsspec(s3_backend_async, s3_test_file)
    data = await reader.read_concurrent((slice(57812, 60000), slice(0, 100)))
    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[0, :10], expected)


@filter_numpy_size_warning
@pytest.mark.xfail(reason="Om Files on S3 currently have no names assigned for the variables")
def test_s3_xarray(s3_backend_with_cache):
    ds = xr.open_dataset(s3_backend_with_cache, engine="om")
    assert any(ds.variables.keys())


def test_fsspec_reader_close(local_fs, temp_om_file):
    with local_fs.open(temp_om_file, "rb") as f:
        reader = omfiles.OmFilePyReader(f)
        assert reader.shape == (5, 5)
        assert reader.chunks == (5, 5)
        assert not reader.closed
        data = reader[0:4, 0:4]
        assert data.dtype == np.float32
        assert data.shape == (4, 4)
        reader.close()
        assert reader.closed
        with pytest.raises(ValueError):
            _ = reader[0:4, 0:4]
    with local_fs.open(temp_om_file, "rb") as f:
        with omfiles.OmFilePyReader(f) as reader:
            ctx_data = reader[0:4, 0:4]
            np.testing.assert_array_equal(ctx_data, data)
        assert reader.closed

    # Data obtained before closing should still be valid
    expected = [
        [0.0, 1.0, 2.0, 3.0],
        [5.0, 6.0, 7.0, 8.0],
        [10.0, 11.0, 12.0, 13.0],
        [15.0, 16.0, 17.0, 18.0],
    ]
    np.testing.assert_array_equal(data, expected)


def test_fsspec_file_actually_closes(local_fs, temp_om_file):
    reader = omfiles.OmFilePyReader.from_fsspec(local_fs, temp_om_file)
    assert reader.shape == (5, 5)
    assert reader.chunks == (5, 5)
    assert reader.dtype == np.float32
    reader.close()
    assert reader.closed
    with pytest.raises((ValueError, OSError)):
        reader[0:5]


def test_write_memory_fsspec(memory_fs):
    data = create_test_data()
    writer = omfiles.OmFilePyWriter.from_fsspec(memory_fs, "test_memory.om")
    write_simple_omfile(writer, data)
    assert_file_exists(memory_fs, "test_memory.om")


def test_write_local_fsspec(local_fs):
    data = create_test_data(shape=(20, 15), dtype=np.float64)
    with tempfile.NamedTemporaryFile(suffix=".om", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        writer = omfiles.OmFilePyWriter.from_fsspec(local_fs, tmp_path)
        write_simple_omfile(writer, data, name="local_test_data")
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        reader = omfiles.OmFilePyReader.from_fsspec(local_fs, tmp_path)
        np.testing.assert_array_equal(reader[:], data)

    finally:
        os.unlink(tmp_path)


def test_write_hierarchical_fsspec(memory_fs):
    temperature = create_test_data(shape=(5, 5, 10))
    humidity = create_test_data(shape=(5, 5, 10))
    writer = omfiles.OmFilePyWriter.from_fsspec(memory_fs, "hierarchical_test.om")
    temp_var = writer.write_array(temperature, chunks=[5, 5, 5], name="temperature", scale_factor=100.0)
    humid_var = writer.write_array(humidity, chunks=[5, 5, 5], name="humidity", scale_factor=100.0)
    temp_units = writer.write_scalar("celsius", name="units")
    temp_desc = writer.write_scalar("Surface temperature", name="description")
    temp_dims = writer.write_scalar("lat,lon,time", name="_ARRAY_DIMENSIONS")
    humid_units = writer.write_scalar("percent", name="units")
    humid_desc = writer.write_scalar("Relative humidity", name="description")
    humid_dims = writer.write_scalar("lat,lon,time", name="_ARRAY_DIMENSIONS")
    temp_metadata = writer.write_group("temp_metadata", [temp_units, temp_desc, temp_dims])
    humid_metadata = writer.write_group("humid_metadata", [humid_units, humid_desc, humid_dims])
    root_group = writer.write_group("weather_data", [temp_var, humid_var, temp_metadata, humid_metadata])
    writer.close(root_group)
    assert_file_exists(memory_fs, "hierarchical_test.om")


def test_fsspec_roundtrip(memory_fs):
    # Write
    data = create_test_data(shape=(8, 8), dtype=np.float32)
    writer = omfiles.OmFilePyWriter.from_fsspec(memory_fs, "roundtrip.om")
    # fpx_xor_2d is a lossless compression
    variable = writer.write_array(data, chunks=[4, 4], name="roundtrip_data", compression="fpx_xor_2d")
    writer.close(variable)
    assert_file_exists(memory_fs, "roundtrip.om")
    # Read
    reader = omfiles.OmFilePyReader.from_fsspec(memory_fs, "roundtrip.om")
    read_data = reader[:]
    np.testing.assert_array_equal(data, read_data)
    reader.close()
