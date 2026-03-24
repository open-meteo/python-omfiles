import tempfile
import tracemalloc

import dask.array as da
import numpy as np
import pytest
from omfiles import OmFileReader, OmFileWriter
from omfiles.dask import write_dask_array


@pytest.fixture
def dask_array_2d():
    np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
    return da.from_array(np_data, chunks=(5, 10))


@pytest.fixture
def dask_array_3d():
    np_data = np.arange(192, dtype=np.int32).reshape(4, 6, 8)
    return da.from_array(np_data, chunks=(2, 3, 4))


def test_dask_roundtrip_2d(dask_array_2d):
    expected = dask_array_2d.compute()

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        var = write_dask_array(writer, dask_array_2d, scale_factor=10000.0)
        writer.close(var)

        reader = OmFileReader(f.name)
        result = reader[:]
        reader.close()

        np.testing.assert_array_almost_equal(result, expected, decimal=4)


def test_dask_roundtrip_3d(dask_array_3d):
    expected = dask_array_3d.compute()

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        var = write_dask_array(writer, dask_array_3d)
        writer.close(var)

        reader = OmFileReader(f.name)
        result = reader[:]
        reader.close()

        np.testing.assert_array_equal(result, expected)


def test_dask_boundary_chunks():
    np_data = np.arange(91, dtype=np.float32).reshape(7, 13)
    darr = da.from_array(np_data, chunks=(4, 5))

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        var = write_dask_array(writer, darr, scale_factor=10000.0)
        writer.close(var)

        reader = OmFileReader(f.name)
        result = reader[:]
        reader.close()

        np.testing.assert_array_almost_equal(result, np_data, decimal=4)


def test_dask_custom_name(dask_array_2d):
    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        var = write_dask_array(writer, dask_array_2d, scale_factor=10000.0, name="temperature")
        assert var.name == "temperature"
        writer.close(var)


def test_dask_non_multiple_chunks_raises():
    """Dask chunks that aren't multiples of OM chunks should raise."""
    np_data = np.arange(30, dtype=np.float32).reshape(6, 5)
    darr = da.from_array(np_data, chunks=(3, 5))

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        with pytest.raises(ValueError, match="not a multiple"):
            write_dask_array(writer, darr, chunks=[2, 5])


def test_dask_larger_chunks_than_om_2d():
    """Dask blocks spanning multiple OM chunks along dim 1 (full trailing dim)."""
    np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
    darr = da.from_array(np_data, chunks=(10, 20))

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        var = write_dask_array(writer, darr, chunks=[5, 10], scale_factor=10000.0)
        writer.close(var)

        reader = OmFileReader(f.name)
        result = reader[:]
        reader.close()

        np.testing.assert_array_almost_equal(result, np_data, decimal=4)


def test_dask_larger_chunks_than_om_3d():
    """Dask blocks with full trailing dims, multiple OM chunks in dim 0."""
    np_data = np.arange(192, dtype=np.int32).reshape(4, 6, 8)
    darr = da.from_array(np_data, chunks=(4, 6, 8))

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        var = write_dask_array(writer, darr, chunks=[2, 3, 4])
        writer.close(var)

        reader = OmFileReader(f.name)
        result = reader[:]
        reader.close()

        np.testing.assert_array_equal(result, np_data)


def test_dask_single_om_chunk_per_slow_dim():
    """Dask blocks with 1 OM chunk in dim 0, partial trailing dim coverage."""
    np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
    darr = da.from_array(np_data, chunks=(5, 10))

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        var = write_dask_array(writer, darr, chunks=[5, 5], scale_factor=10000.0)
        writer.close(var)

        reader = OmFileReader(f.name)
        result = reader[:]
        reader.close()

        np.testing.assert_array_almost_equal(result, np_data, decimal=4)


def test_dask_misaligned_trailing_dims_raises():
    """Dask blocks with multi-chunk dim 0 but partial trailing dim raises."""
    np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
    darr = da.from_array(np_data, chunks=(10, 10))

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        with pytest.raises(ValueError, match="not fully covered"):
            write_dask_array(writer, darr, chunks=[5, 5])


def test_dask_not_a_dask_array_raises():
    np_data = np.arange(20, dtype=np.float32).reshape(4, 5)
    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        with pytest.raises(TypeError, match="Expected a dask array"):
            write_dask_array(writer, np_data)


def test_dask_streaming_memory_stays_bounded():
    """Peak memory during a dask streaming write stays well below the full dataset size."""
    # ~16 MB dataset (2048 x 2048 x float32), written in 256x256 chunks (~256 KB each)
    side = 2048
    chunk = 256
    dtype = np.float32
    dataset_bytes = side * side * np.dtype(dtype).itemsize

    darr = da.random.random((side, side), chunks=(chunk, chunk)).astype(dtype)

    tracemalloc.start()

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        var = write_dask_array(writer, darr)
        writer.close(var)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak Python memory should be a fraction of the total dataset size,
    # proving that chunks are streamed rather than fully materialized.
    assert peak < dataset_bytes, (
        f"Peak traced memory ({peak / 1024 / 1024:.1f} MB) should be less than "
        f"the dataset size ({dataset_bytes / 1024 / 1024:.1f} MB)"
    )
