import tempfile

import numpy as np
import omfiles
import pytest


def test_write_om_roundtrip(temp_om_file):
    reader = omfiles.OmFileReader(temp_om_file)
    data = reader[0:5, 0:5]
    reader.close()

    assert data.shape == (5, 5)
    assert data.dtype == np.float32
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


def test_round_trip_array_datatypes():
    shape = (5, 5, 5, 2)
    chunks = [2, 2, 2, 1]
    test_cases = [
        (np.random.rand(*shape).astype(np.float32), "float32"),
        (np.random.rand(*shape).astype(np.float64), "float64"),
        (np.random.randint(-128, 127, size=shape, dtype=np.int8), "int8"),
        (np.random.randint(-32768, 32767, size=shape, dtype=np.int16), "int16"),
        (np.random.randint(-2147483648, 2147483647, size=shape, dtype=np.int32), "int32"),
        (np.random.randint(-9223372036854775808, 9223372036854775807, size=shape, dtype=np.int64), "int64"),
        (np.random.randint(0, 255, size=shape, dtype=np.uint8), "uint8"),
        (np.random.randint(0, 65535, size=shape, dtype=np.uint16), "uint16"),
        (np.random.randint(0, 4294967295, size=shape, dtype=np.uint32), "uint32"),
        (np.random.randint(0, 18446744073709551615, size=shape, dtype=np.uint64), "uint64"),
    ]

    for test_data, dtype in test_cases:
        with tempfile.NamedTemporaryFile(suffix=".om") as temp_file:
            writer = omfiles.OmFileWriter(temp_file.name)
            variable = writer.write_array(test_data, chunks=chunks, scale_factor=10000.0, add_offset=0.0)
            writer.close(variable)

            # Read data back
            reader = omfiles.OmFileReader(temp_file.name)
            read_data = reader[:]
            reader.close()

            # Verify data
            assert read_data.dtype == test_data.dtype
            assert read_data.shape == test_data.shape
            # use assert_array_almost_equal since our floating point values are compressed lossy
            np.testing.assert_array_almost_equal(read_data, test_data, decimal=4)


def test_write_hierarchical_file(empty_temp_om_file):
    # Create test data
    root_data = np.random.rand(10, 10).astype(np.float32)
    child1_data = np.random.rand(5, 5).astype(np.float32)
    child2_data = np.random.rand(3, 3).astype(np.float32)

    # Write hierarchical structure
    writer = omfiles.OmFileWriter(empty_temp_om_file)

    # Write child2 array
    child2_var = writer.write_array(child2_data, chunks=[1, 1], name="child2", scale_factor=100000.0)

    # Write attributes and get their variables
    meta1_var = writer.write_scalar(42.0, name="metadata1")
    meta2_var = writer.write_scalar(123, name="metadata2")
    meta3_var = writer.write_scalar("blub", name="metadata3")

    # Write child1 array with attribute children
    child1_var = writer.write_array(
        child1_data, chunks=[2, 2], name="child1", scale_factor=100000.0, children=[meta1_var, meta2_var, meta3_var]
    )

    # Write root array with children
    root_var = writer.write_array(
        root_data, chunks=[5, 5], name="root", scale_factor=100000.0, children=[child1_var, child2_var]
    )

    # Finalize the file
    writer.close(root_var)

    # Read and verify the data using OmFileReader
    reader = omfiles.OmFileReader(empty_temp_om_file)

    # Verify root data
    read_root = reader[:]
    np.testing.assert_array_almost_equal(read_root, root_data, decimal=4)
    assert read_root.shape == (10, 10)
    assert read_root.dtype == np.float32

    # Get child readers
    child_metadata = reader._get_flat_variable_metadata()

    # Verify child1 data
    child1_reader = reader._init_from_variable(child_metadata["/root/child1"])
    read_child1 = child1_reader[:]
    np.testing.assert_array_almost_equal(read_child1, child1_data, decimal=4)
    assert read_child1.shape == (5, 5)
    assert read_child1.dtype == np.float32

    # Verify child2 data
    child2_reader = reader._init_from_variable(child_metadata["/root/child2"])
    read_child2 = child2_reader[:]
    np.testing.assert_array_almost_equal(read_child2, child2_data, decimal=4)
    assert read_child2.shape == (3, 3)
    assert read_child2.dtype == np.float32

    # Verify metadata attributes
    metadata_reader0 = child1_reader.get_child_by_index(0)
    metadata = metadata_reader0.read_scalar()
    assert metadata == 42.0
    assert metadata_reader0.dtype == np.float64

    metadata_reader2 = child1_reader.get_child_by_index(2)
    metadata = metadata_reader2.read_scalar()
    assert metadata == "blub"
    assert metadata_reader2.dtype == str

    reader.close()
    child1_reader.close()
    child2_reader.close()
    metadata_reader0.close()
    metadata_reader2.close()


@pytest.mark.asyncio
async def test_read_async(temp_om_file):
    with await omfiles.OmFileReaderAsync.from_path(temp_om_file) as reader:
        # Test basic async read
        data = await reader.read_array((slice(0, 5), ...))
        assert data.shape == (5, 5)
        assert data.dtype == np.float32
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

    # Test that not awaiting results before closing the reader is safe
    with await omfiles.OmFileReaderAsync.from_path(temp_om_file) as reader_we_dont_await:
        for _ in range(100):
            data = reader_we_dont_await.read_array((...))

    reader = await omfiles.OmFileReaderAsync.from_path(temp_om_file)
    reader.close()
    # Test read_array with a closed reader
    try:
        _ = await reader.read_array((slice(0, 5), ...))
        assert False, "Expected ValueError when reading from closed reader"
    except ValueError as e:
        assert "closed" in str(e).lower()


def test_reader_close(temp_om_file):
    reader = omfiles.OmFileReader(temp_om_file)

    # Verify we can read data
    data = reader[0:5, 0:5]
    assert data.shape == (5, 5)
    assert data.dtype == np.float32

    # Test context manager
    with omfiles.OmFileReader(temp_om_file) as ctx_reader:
        ctx_data = ctx_reader[0:5, 0:5]
        assert ctx_data.shape == (5, 5)
        # Reader should be valid inside context
        assert not ctx_reader.closed
    # Reader should be closed after context
    assert ctx_reader.closed

    # Explicitly close the reader
    reader.close()

    # Verify that the reader reports as closed
    assert reader.closed

    # Verify that operations on a closed reader fail
    try:
        _ = reader[0:5, 0:5]
        assert False, "Expecting an error when accessing a closed reader"
    except ValueError as e:
        assert "closed" in str(e).lower()

    try:
        _ = reader._get_flat_variable_metadata()
        assert False, "Expecting an error when calling methods on a closed reader"
    except ValueError as e:
        assert "closed" in str(e).lower()

    # Test double-close is safe
    reader.close()  # This should not raise an exception
    assert reader.closed

    # Test we can still use the data after closing the reader
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


def test_child_traversal(temp_hierarchical_om_file):
    reader = omfiles.OmFileReader(temp_hierarchical_om_file)

    assert reader.num_children == 2
    assert reader.dtype == type(None)
    with pytest.raises(ValueError):
        _ = reader.compression_name
    with pytest.raises(ValueError):
        _ = reader.read_scalar()
    with pytest.raises(ValueError):
        _ = reader[:]
    with pytest.raises(ValueError):
        # only 0 and 1 are valid indices
        _ = reader.get_child_by_index(2)

    with reader.get_child_by_index(0) as var1_reader:
        assert var1_reader.shape == (5, 5)
        assert var1_reader.name == "variable1"
        assert var1_reader.compression_name == "pfor_delta_2d"
        assert var1_reader.dtype == np.float32

    var2_reader = reader.get_child_by_index(1)
    # verify that closing reader is safe for var2_reader
    reader.close()
    assert var2_reader.shape == (50, 5)
    assert var2_reader.name == "variable2"
    assert var2_reader.dtype == np.int64
    np.testing.assert_array_equal(var2_reader[:], np.arange(50 * 5).reshape(50, 5) * 2)
