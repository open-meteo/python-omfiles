import fsspec
import numpy as np
import omfiles
import pytest
import xarray as xr
from omfiles.async_fsspec_wrapper import AsyncFsSpecWrapper
from s3fs import S3FileSystem

from .test_utils import filter_numpy_size_warning


@pytest.fixture
def s3_backend():
    s3_test_file = "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"
    fs = S3FileSystem(anon=True)
    file = fs.open(s3_test_file, mode="rb", block_size=256)
    yield file

# @pytest.fixture
# async def s3_backend_async():
#     s3_test_file = "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"
#     fs = S3FileSystem(anon=True, asynchronous=True)
#     session = await fs.set_session()
#     s3_backend = await fs.open_async(path=s3_test_file, mode="rb", block_size=256)
#     yield s3_backend
#     await session.close()

@pytest.fixture
def s3_backend_with_cache():
    s3_test_file = "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"
    fs = fsspec.filesystem(protocol="s3", anon=True)
    yield fs.open(s3_test_file, mode="rb", cache_type="mmap", block_size=256, cache_options={"location": "cache"})

@pytest.fixture
def local_file(temp_om_file):
    fs = fsspec.filesystem("file")
    yield fs.open(temp_om_file, mode="rb")


def test_local_read(local_file):
    reader = omfiles.OmFilePyReader(local_file)
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

# @pytest.mark.asyncio
# async def test_local_read_concurrent(local_file):
#     reader = await omfiles.OmFilePyReader.from_fsspec(local_file)
#     data = await reader.read_concurrent((slice(0, 5), slice(0, 5)))

#     np.testing.assert_array_equal(
#         data,
#         [
#             [0.0, 1.0, 2.0, 3.0, 4.0],
#             [5.0, 6.0, 7.0, 8.0, 9.0],
#             [10.0, 11.0, 12.0, 13.0, 14.0],
#             [15.0, 16.0, 17.0, 18.0, 19.0],
#             [20.0, 21.0, 22.0, 23.0, 24.0],
#         ],
#     )


def test_s3_read(s3_backend):
    reader = omfiles.OmFilePyReader(s3_backend)
    data = reader[57812:60000, 0:100]

    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[0, :10], expected)

@pytest.mark.asyncio
async def test_s3_concurrent_read(s3_backend):
    # await s3_backend_async.read()
    s3_backend_async = AsyncFsSpecWrapper(s3_backend)
    reader = await omfiles.OmFilePyReaderAsync.from_fsspec(s3_backend_async)
    data = await reader.read_concurrent((slice(57812, 60000), slice(0, 100)))

    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[0, :10], expected)


def test_s3_read_with_cache(s3_backend_with_cache):
    reader = omfiles.OmFilePyReader(s3_backend_with_cache)
    data = reader[57812:57813, 0:100]

    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[:10], expected)


@filter_numpy_size_warning
@pytest.mark.xfail(reason="Om Files on S3 currently have no names assigned for the variables")
def test_s3_xarray(s3_backend_with_cache):
    ds = xr.open_dataset(s3_backend_with_cache, engine="om")
    assert any(ds.variables.keys())


def test_fsspec_reader_close(temp_om_file):
    """Test that closing a reader with fsspec file object works correctly."""
    fs = fsspec.filesystem("file")

    # Test explicit closure
    with fs.open(temp_om_file, "rb") as f:
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
    with fs.open(temp_om_file, "rb") as f:
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


def test_fsspec_file_actually_closes(temp_om_file):
    """Test that the underlying fsspec file is actually closed."""
    fs = fsspec.filesystem("file")
    f = fs.open(temp_om_file, "rb")

    # Create, verify and close reader
    reader = omfiles.OmFilePyReader(f)
    assert reader.shape == [5, 5]
    dtype = reader.dtype
    assert dtype == np.float32
    reader.close()

    # File should be closed - verify by trying to read from it
    try:
        f.read(1)
        assert False, "File should be closed"
    except (ValueError, OSError):
        pass
