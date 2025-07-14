import os
import tempfile

import numpy as np
import omfiles
import pytest
import xarray as xr
from fsspec.implementations.cached import CachingFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from s3fs import S3FileSystem

from .test_utils import filter_numpy_size_warning


@pytest.fixture
def s3_test_file():
    yield "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"


@pytest.fixture
def s3_backend():
    fs = S3FileSystem(anon=True, default_block_size=256, default_cache_type="none")
    yield fs


@pytest.fixture
def s3_backend_with_cache():
    s3_fs = S3FileSystem(anon=True, default_block_size=256, default_cache_type="none")
    fs = CachingFileSystem(
        fs=s3_fs, cache_check=3600, block_size=256, cache_storage="cache", check_files=False, same_names=True
    )
    yield fs


@pytest.fixture
def local_fs():
    fs = LocalFileSystem()
    yield fs


@pytest.fixture
async def s3_backend_async():
    fs = S3FileSystem(anon=True, asynchronous=True, default_block_size=256, default_cache_type="none")
    yield fs


def test_local_read(local_fs, temp_om_file):
    reader = omfiles.OmFilePyReader.from_fsspec(local_fs, temp_om_file)
    data = reader[0:5, 0:5]

    np.testing.assert_array_equal(
        data,
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
        ],
    )


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
    """Test that closing a reader with fsspec file object works correctly."""
    # Test explicit closure
    with local_fs.open(temp_om_file, "rb") as f:
        reader = omfiles.OmFilePyReader(f)

        # Check properties before closing
        assert reader.shape == (5, 5)
        assert reader.chunks == (5, 5)
        assert not reader.closed

        # Get data and verify
        data = reader[0:4, 0:4]
        assert data.dtype == np.float32
        assert data.shape == (4, 4)

        # Close and verify
        reader.close()
        assert reader.closed

        # Operations should fail after close
        try:
            _ = reader[0:4, 0:4]
            assert False, "Should fail on closed reader"
        except ValueError:
            pass

    # Test context manager
    with local_fs.open(temp_om_file, "rb") as f:
        with omfiles.OmFilePyReader(f) as reader:
            ctx_data = reader[0:4, 0:4]
            np.testing.assert_array_equal(ctx_data, data)

        # Should be closed after context
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
    """Test that the underlying fsspec file is actually closed."""

    # Create, verify and close reader
    reader = omfiles.OmFilePyReader.from_fsspec(local_fs, temp_om_file)
    assert reader.shape == (5, 5)
    assert reader.chunks == (5, 5)
    dtype = reader.dtype
    assert dtype == np.float32
    reader.close()

    assert reader.closed

    # Reader should be closed - verify by trying to read from it
    try:
        reader[0:5]
        assert False, "File should be closed"
    except (ValueError, OSError):
        pass


def test_memory_filesystem():
    """Test writing to an in-memory filesystem."""
    print("Testing memory filesystem...")

    # Create a memory filesystem
    fs = MemoryFileSystem()

    # Create test data
    data = np.random.rand(10, 10).astype(np.float32)

    # Create writer using fsspec
    writer = omfiles.OmFilePyWriter.from_fsspec(fs, "test_memory.om")
    metadata = writer.write_scalar("Test data from memory filesystem", name="description")
    variable = writer.write_array(data, chunks=[5, 5], name="test_data", children=[metadata])
    writer.close(variable)

    # Debug: List files in memory filesystem
    print(f"Files in memory filesystem: {fs.ls('/')}")

    # Verify the file exists in memory
    if not fs.exists("test_memory.om"):
        print("ERROR: File does not exist in memory filesystem")
        print(f"Available files: {fs.ls('/')}")
        return None, None

    file_size = fs.size("test_memory.om")
    print(f"Memory filesystem test passed! File size: {file_size} bytes")

    return fs, "test_memory.om"


def test_local_filesystem():
    """Test writing to local filesystem using fsspec."""
    print("Testing local filesystem...")

    # Create a local filesystem
    fs = LocalFileSystem()

    # Create test data
    data = np.random.rand(20, 15).astype(np.float64)

    # Use a temporary file
    with tempfile.NamedTemporaryFile(suffix=".om", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create writer using fsspec
        writer = omfiles.OmFilePyWriter.from_fsspec(fs, tmp_path)

        # Add metadata
        units = writer.write_scalar("meters", name="units")
        description = writer.write_scalar("Test data from local filesystem", name="description")

        # Create a group with children
        metadata_group = writer.write_group("metadata", [units, description])

        # Write array with different parameters
        variable = writer.write_array(
            data,
            chunks=[10, 5],
            scale_factor=100.0,
            add_offset=10.0,
            compression="pfor_delta_2d",
            name="local_test_data",
            children=[metadata_group],
        )

        # Close with root variable
        writer.close(variable)

        # Verify the file exists
        assert os.path.exists(tmp_path)
        file_size = os.path.getsize(tmp_path)
        print(f"Local filesystem test passed! File size: {file_size} bytes")

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_hierarchical_data():
    """Test writing hierarchical data structure using fsspec."""
    print("Testing hierarchical data structure...")

    # Create a memory filesystem
    fs = MemoryFileSystem()

    # Create test data
    temperature = np.random.rand(5, 5, 10).astype(np.float32)
    humidity = np.random.rand(5, 5, 10).astype(np.float32)

    # Create writer
    writer = omfiles.OmFilePyWriter.from_fsspec(fs, "hierarchical_test.om")

    # Write temperature data
    temp_var = writer.write_array(temperature, chunks=[5, 5, 5], name="temperature", scale_factor=100.0)

    # Write humidity data
    humid_var = writer.write_array(humidity, chunks=[5, 5, 5], name="humidity", scale_factor=100.0)

    # Add metadata for temperature
    temp_units = writer.write_scalar("celsius", name="units")
    temp_desc = writer.write_scalar("Surface temperature", name="description")
    temp_dims = writer.write_scalar("lat,lon,time", name="_ARRAY_DIMENSIONS")

    # Add metadata for humidity
    humid_units = writer.write_scalar("percent", name="units")
    humid_desc = writer.write_scalar("Relative humidity", name="description")
    humid_dims = writer.write_scalar("lat,lon,time", name="_ARRAY_DIMENSIONS")

    # Create metadata groups
    temp_metadata = writer.write_group("temp_metadata", [temp_units, temp_desc, temp_dims])
    humid_metadata = writer.write_group("humid_metadata", [humid_units, humid_desc, humid_dims])

    # Create root group
    root_group = writer.write_group("weather_data", [temp_var, humid_var, temp_metadata, humid_metadata])

    # Close with root group
    writer.close(root_group)

    # Verify
    assert fs.exists("hierarchical_test.om")
    file_size = fs.size("hierarchical_test.om")
    print(f"Hierarchical data test passed! File size: {file_size} bytes")


def test_different_data_types():
    """Test writing different data types using fsspec."""
    print("Testing different data types...")

    # Create a memory filesystem
    fs = MemoryFileSystem()

    # Create writer
    writer = omfiles.OmFilePyWriter.from_fsspec(fs, "datatypes_test.om")

    # Test different data types
    int32_data = np.random.randint(0, 100, size=(8, 8), dtype=np.int32)
    float64_data = np.random.rand(6, 6).astype(np.float64)
    int8_data = np.random.randint(-128, 127, size=(4, 4), dtype=np.int8)

    # Write arrays
    int32_var = writer.write_array(int32_data, chunks=[4, 4], name="int32_data")
    float64_var = writer.write_array(float64_data, chunks=[3, 3], name="float64_data")
    int8_var = writer.write_array(int8_data, chunks=[2, 2], name="int8_data")

    # Write scalar values of different types
    str_scalar = writer.write_scalar("test string", name="string_value")
    int_scalar = writer.write_scalar(42, name="int_value")
    float_scalar = writer.write_scalar(3.14159, name="float_value")

    # Create root group
    root_group = writer.write_group(
        "mixed_data", [int32_var, float64_var, int8_var, str_scalar, int_scalar, float_scalar]
    )

    # Close with root group
    writer.close(root_group)

    # Verify
    assert fs.exists("datatypes_test.om")
    file_size = fs.size("datatypes_test.om")
    print(f"Different data types test passed! File size: {file_size} bytes")
