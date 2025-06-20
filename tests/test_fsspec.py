import numpy as np
import omfiles
import pytest
import xarray as xr
from fsspec.implementations.cached import CachingFileSystem
from fsspec.implementations.local import LocalFileSystem
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
        assert reader.shape == [5, 5]
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
    assert reader.shape == [5, 5]
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
