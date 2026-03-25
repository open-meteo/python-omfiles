import dask.array as da
import numpy as np
import pytest
from omfiles import OmFileReader, OmFileWriter
from omfiles.dask import write_dask_array


@pytest.fixture
def dask_array_2d():
    np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
    return da.from_array(np_data, chunks=(5, 10))  # type: ignore[arg-type]


@pytest.fixture
def dask_array_3d():
    np_data = np.arange(192, dtype=np.int32).reshape(4, 6, 8)
    return da.from_array(np_data, chunks=(2, 3, 4))  # type: ignore[arg-type]


def test_dask_roundtrip_2d(empty_temp_om_file, dask_array_2d):
    expected = dask_array_2d.compute()

    writer = OmFileWriter(empty_temp_om_file)
    var = write_dask_array(writer, dask_array_2d, scale_factor=10000.0)
    writer.close(var)

    reader = OmFileReader(empty_temp_om_file)
    result = reader[:]
    reader.close()

    np.testing.assert_array_almost_equal(result, expected, decimal=4)


def test_dask_roundtrip_3d(empty_temp_om_file, dask_array_3d):
    expected = dask_array_3d.compute()

    writer = OmFileWriter(empty_temp_om_file)
    var = write_dask_array(writer, dask_array_3d)
    writer.close(var)

    reader = OmFileReader(empty_temp_om_file)
    result = reader[:]
    reader.close()

    np.testing.assert_array_equal(result, expected)


def test_dask_boundary_chunks(empty_temp_om_file):
    np_data = np.arange(91, dtype=np.float32).reshape(7, 13)
    darr = da.from_array(np_data, chunks=(4, 5))  # type: ignore[arg-type]

    writer = OmFileWriter(empty_temp_om_file)
    var = write_dask_array(writer, darr, scale_factor=10000.0)
    writer.close(var)

    reader = OmFileReader(empty_temp_om_file)
    result = reader[:]
    reader.close()

    np.testing.assert_array_almost_equal(result, np_data, decimal=4)


def test_dask_custom_name(empty_temp_om_file, dask_array_2d):
    writer = OmFileWriter(empty_temp_om_file)
    var = write_dask_array(writer, dask_array_2d, scale_factor=10000.0, name="temperature")
    assert var.name == "temperature"
    writer.close(var)


def test_dask_non_multiple_chunks_raises(empty_temp_om_file):
    """Dask chunks that aren't multiples of OM chunks should raise."""
    np_data = np.arange(30, dtype=np.float32).reshape(6, 5)
    darr = da.from_array(np_data, chunks=(3, 5))  # type: ignore[arg-type]

    writer = OmFileWriter(empty_temp_om_file)
    with pytest.raises(ValueError, match="not a multiple"):
        write_dask_array(writer, darr, chunks=[2, 5])


def test_dask_larger_chunks_than_om_2d(empty_temp_om_file):
    """Dask blocks spanning multiple OM chunks along dim 1 (full trailing dim)."""
    np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
    darr = da.from_array(np_data, chunks=(10, 20))  # type: ignore[arg-type]

    writer = OmFileWriter(empty_temp_om_file)
    var = write_dask_array(writer, darr, chunks=[5, 10], scale_factor=10000.0)
    writer.close(var)

    reader = OmFileReader(empty_temp_om_file)
    result = reader[:]
    reader.close()

    np.testing.assert_array_almost_equal(result, np_data, decimal=4)


def test_dask_larger_chunks_than_om_3d(empty_temp_om_file):
    """Dask blocks with full trailing dims, multiple OM chunks in dim 0."""
    np_data = np.arange(192, dtype=np.int32).reshape(4, 6, 8)
    darr = da.from_array(np_data, chunks=(4, 6, 8))  # type: ignore[arg-type]

    writer = OmFileWriter(empty_temp_om_file)
    var = write_dask_array(writer, darr, chunks=[2, 3, 4])
    writer.close(var)

    reader = OmFileReader(empty_temp_om_file)
    result = reader[:]
    reader.close()

    np.testing.assert_array_equal(result, np_data)


def test_dask_single_om_chunk_per_slow_dim(empty_temp_om_file):
    """Dask blocks with 1 OM chunk in dim 0, partial trailing dim coverage."""
    np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
    darr = da.from_array(np_data, chunks=(5, 10))  # type: ignore[arg-type]

    writer = OmFileWriter(empty_temp_om_file)
    var = write_dask_array(writer, darr, chunks=[5, 5], scale_factor=10000.0)
    writer.close(var)

    reader = OmFileReader(empty_temp_om_file)
    result = reader[:]
    reader.close()

    np.testing.assert_array_almost_equal(result, np_data, decimal=4)


def test_dask_misaligned_trailing_dims_raises(empty_temp_om_file):
    np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
    darr = da.from_array(np_data, chunks=(10, 10))  # type: ignore[arg-type]

    writer = OmFileWriter(empty_temp_om_file)
    with pytest.raises(ValueError, match="not fully covered"):
        write_dask_array(writer, darr, chunks=[5, 5])


def test_dask_not_a_dask_array_raises(empty_temp_om_file):
    np_data = np.arange(20, dtype=np.float32).reshape(4, 5)
    writer = OmFileWriter(empty_temp_om_file)
    with pytest.raises(TypeError, match="Expected a dask array"):
        write_dask_array(writer, np_data)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_chunks",
    [
        pytest.param([5], id="too_few"),
        pytest.param([5, 10, 4], id="too_many"),
    ],
)
def test_dask_chunk_ndim_mismatch_raises(empty_temp_om_file, bad_chunks):
    np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
    darr = da.from_array(np_data, chunks=(5, 10))  # type: ignore[arg-type]

    writer = OmFileWriter(empty_temp_om_file)
    with pytest.raises(ValueError, match=r"chunks has \d+ element"):
        write_dask_array(writer, darr, chunks=bad_chunks)


def test_dask_irregular_chunks_misaligned_raises(empty_temp_om_file):
    """
    Non-first dask block spans multiple OM chunks while trailing dim is not
    fully covered.

    Array (12, 16), dask chunks ((4, 8), (8, 8)), OM chunks [4, 8]:
      block (1,0) shape (8, 8) → 2 OM rows but only 1 of 2 OM columns.
    """
    np_data = np.arange(192, dtype=np.float32).reshape(12, 16)
    darr = da.from_array(np_data, chunks=((4, 8), (8, 8)))  # type: ignore[arg-type]

    writer = OmFileWriter(empty_temp_om_file)
    with pytest.raises(ValueError, match="not fully covered"):
        write_dask_array(writer, darr, chunks=[4, 8])


def test_dask_irregular_chunks_valid_roundtrip(empty_temp_om_file):
    """
    Non-first dask block spans multiple OM chunks but trailing dim IS fully
    covered — this configuration is valid and must produce correct output.

    Array (12, 16), dask chunks ((4, 8), (16,)), OM chunks [4, 8]:
      block (1,0) shape (8, 16) → 2 OM rows and all OM columns — safe.
    """
    np_data = np.arange(192, dtype=np.float32).reshape(12, 16)
    darr = da.from_array(np_data, chunks=((4, 8), (16,)))  # type: ignore[arg-type]

    writer = OmFileWriter(empty_temp_om_file)
    var = write_dask_array(writer, darr, chunks=[4, 8], scale_factor=10000.0)
    writer.close(var)

    reader = OmFileReader(empty_temp_om_file)
    result = reader[:]
    reader.close()

    np.testing.assert_array_almost_equal(result, np_data, decimal=4)
