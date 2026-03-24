import tempfile

import numpy as np
import pytest
from omfiles import OmFileReader, OmFileWriter


def test_streaming_single_chunk():
    shape = (10, 20)
    chunks = [10, 20]
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)

        def chunk_iter():
            yield data

        var = writer.write_array_streaming(
            dimensions=list(shape),
            chunks=chunks,
            chunk_iterator=chunk_iter(),
            dtype="float32",
            scale_factor=10000.0,
        )
        writer.close(var)

        reader = OmFileReader(f.name)
        result = reader[:]
        reader.close()

        np.testing.assert_array_almost_equal(result, data, decimal=4)


def test_streaming_multiple_chunks_2d():
    shape = (10, 20)
    chunks = [5, 10]
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)

        def chunk_iter():
            for i in range(0, 10, 5):
                for j in range(0, 20, 10):
                    yield data[i : i + 5, j : j + 10].copy()

        var = writer.write_array_streaming(
            dimensions=list(shape),
            chunks=chunks,
            chunk_iterator=chunk_iter(),
            dtype="float32",
            scale_factor=10000.0,
        )
        writer.close(var)

        reader = OmFileReader(f.name)
        result = reader[:]
        reader.close()

        np.testing.assert_array_almost_equal(result, data, decimal=4)


def test_streaming_all_dtypes():
    shape = (6, 8)
    chunks = [3, 4]
    dtypes = [
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]

    for dt in dtypes:
        if np.issubdtype(dt, np.floating):
            data = np.random.rand(*shape).astype(dt)
        elif np.issubdtype(dt, np.signedinteger):
            info = np.iinfo(dt)
            data = np.random.randint(max(info.min, -1000), min(info.max, 1000), size=shape, dtype=dt)
        else:
            info = np.iinfo(dt)
            data = np.random.randint(0, min(info.max, 1000), size=shape, dtype=dt)

        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)

            def chunk_iter(d=data):
                for i in range(0, shape[0], chunks[0]):
                    for j in range(0, shape[1], chunks[1]):
                        ie = min(i + chunks[0], shape[0])
                        je = min(j + chunks[1], shape[1])
                        yield d[i:ie, j:je].copy()

            var = writer.write_array_streaming(
                dimensions=list(shape),
                chunks=chunks,
                chunk_iterator=chunk_iter(),
                dtype=np.dtype(dt).name,
                scale_factor=10000.0,
            )
            writer.close(var)

            reader = OmFileReader(f.name)
            result = reader[:]
            reader.close()

            assert result.dtype == dt, f"dtype mismatch for {dt}"
            if np.issubdtype(dt, np.floating):
                np.testing.assert_array_almost_equal(result, data, decimal=4)
            else:
                np.testing.assert_array_equal(result, data)


def test_streaming_3d_array():
    shape = (4, 6, 8)
    chunks = [2, 3, 4]
    data = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)

        def chunk_iter():
            for i in range(0, shape[0], chunks[0]):
                for j in range(0, shape[1], chunks[1]):
                    for k in range(0, shape[2], chunks[2]):
                        ie = min(i + chunks[0], shape[0])
                        je = min(j + chunks[1], shape[1])
                        ke = min(k + chunks[2], shape[2])
                        yield data[i:ie, j:je, k:ke].copy()

        var = writer.write_array_streaming(
            dimensions=list(shape),
            chunks=chunks,
            chunk_iterator=chunk_iter(),
            dtype="int32",
        )
        writer.close(var)

        reader = OmFileReader(f.name)
        result = reader[:]
        reader.close()

        np.testing.assert_array_equal(result, data)


def test_streaming_boundary_chunks():
    shape = (7, 13)
    chunks = [4, 5]
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)

        def chunk_iter():
            for i in range(0, shape[0], chunks[0]):
                for j in range(0, shape[1], chunks[1]):
                    ie = min(i + chunks[0], shape[0])
                    je = min(j + chunks[1], shape[1])
                    yield data[i:ie, j:je].copy()

        var = writer.write_array_streaming(
            dimensions=list(shape),
            chunks=chunks,
            chunk_iterator=chunk_iter(),
            dtype="float32",
            scale_factor=10000.0,
        )
        writer.close(var)

        reader = OmFileReader(f.name)
        result = reader[:]
        reader.close()

        np.testing.assert_array_almost_equal(result, data, decimal=4)


def test_streaming_matches_write_array():
    shape = (10, 20)
    chunks = [5, 10]
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    with tempfile.NamedTemporaryFile(suffix=".om") as f1:
        writer1 = OmFileWriter(f1.name)
        var1 = writer1.write_array(data, chunks=chunks, scale_factor=10000.0)
        writer1.close(var1)
        reader1 = OmFileReader(f1.name)
        result1 = reader1[:]
        reader1.close()

    with tempfile.NamedTemporaryFile(suffix=".om") as f2:
        writer2 = OmFileWriter(f2.name)

        def chunk_iter():
            for i in range(0, shape[0], chunks[0]):
                for j in range(0, shape[1], chunks[1]):
                    ie = min(i + chunks[0], shape[0])
                    je = min(j + chunks[1], shape[1])
                    yield data[i:ie, j:je].copy()

        var2 = writer2.write_array_streaming(
            dimensions=list(shape),
            chunks=chunks,
            chunk_iterator=chunk_iter(),
            dtype="float32",
            scale_factor=10000.0,
        )
        writer2.close(var2)
        reader2 = OmFileReader(f2.name)
        result2 = reader2[:]
        reader2.close()

    np.testing.assert_array_equal(result1, result2)


def test_streaming_unsupported_dtype_raises():
    with tempfile.NamedTemporaryFile(suffix=".om") as f:
        writer = OmFileWriter(f.name)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            writer.write_array_streaming(
                dimensions=[10],
                chunks=[5],
                chunk_iterator=iter([]),
                dtype="complex128",
            )
