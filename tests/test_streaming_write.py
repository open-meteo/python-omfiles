import tempfile
import tracemalloc

import numpy as np
import pytest
from omfiles import OmFileReader, OmFileWriter


class TestWriteArrayStreaming:
    def test_streaming_single_chunk(self):
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

    def test_streaming_multiple_chunks_2d(self):
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

    def test_streaming_all_dtypes(self):
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
                np.testing.assert_array_almost_equal(result, data, decimal=4)

    def test_streaming_3d_array(self):
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

    def test_streaming_boundary_chunks(self):
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

    def test_streaming_matches_write_array(self):
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

    def test_streaming_unsupported_dtype_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            with pytest.raises(ValueError, match="Unsupported dtype"):
                writer.write_array_streaming(
                    dimensions=[10],
                    chunks=[5],
                    chunk_iterator=iter([]),
                    dtype="complex128",
                )


class TestWriteDaskArray:
    @pytest.fixture(autouse=True)
    def _import_dask(self):
        pytest.importorskip("dask.array")
        from omfiles.dask import write_dask_array

        self.write_dask_array = write_dask_array

    @pytest.fixture
    def dask_array_2d(self):
        import dask.array as da

        np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
        return da.from_array(np_data, chunks=(5, 10))

    @pytest.fixture
    def dask_array_3d(self):
        import dask.array as da

        np_data = np.arange(192, dtype=np.int32).reshape(4, 6, 8)
        return da.from_array(np_data, chunks=(2, 3, 4))

    def test_dask_roundtrip_2d(self, dask_array_2d):
        expected = dask_array_2d.compute()

        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            var = self.write_dask_array(
                writer,
                dask_array_2d,
                scale_factor=10000.0,
            )
            writer.close(var)

            reader = OmFileReader(f.name)
            result = reader[:]
            reader.close()

            np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_dask_roundtrip_3d(self, dask_array_3d):
        expected = dask_array_3d.compute()

        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            var = self.write_dask_array(writer, dask_array_3d)
            writer.close(var)

            reader = OmFileReader(f.name)
            result = reader[:]
            reader.close()

            np.testing.assert_array_equal(result, expected)

    def test_dask_boundary_chunks(self):
        import dask.array as da

        np_data = np.arange(91, dtype=np.float32).reshape(7, 13)
        darr = da.from_array(np_data, chunks=(4, 5))

        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            var = self.write_dask_array(writer, darr, scale_factor=10000.0)
            writer.close(var)

            reader = OmFileReader(f.name)
            result = reader[:]
            reader.close()

            np.testing.assert_array_almost_equal(result, np_data, decimal=4)

    def test_dask_custom_name(self, dask_array_2d):
        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            var = self.write_dask_array(
                writer,
                dask_array_2d,
                scale_factor=10000.0,
                name="temperature",
            )
            assert var.name == "temperature"
            writer.close(var)

    def test_dask_non_multiple_chunks_raises(self):
        """Dask chunks that aren't multiples of OM chunks should raise."""
        import dask.array as da

        np_data = np.arange(30, dtype=np.float32).reshape(6, 5)
        darr = da.from_array(np_data, chunks=(3, 5))

        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            with pytest.raises(ValueError, match="not a multiple"):
                self.write_dask_array(writer, darr, chunks=[2, 5])

    def test_dask_larger_chunks_than_om_2d(self):
        """Dask blocks spanning multiple OM chunks along dim 1 (full trailing dim)."""
        import dask.array as da

        np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
        darr = da.from_array(np_data, chunks=(10, 20))

        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            var = self.write_dask_array(
                writer,
                darr,
                chunks=[5, 10],
                scale_factor=10000.0,
            )
            writer.close(var)

            reader = OmFileReader(f.name)
            result = reader[:]
            reader.close()

            np.testing.assert_array_almost_equal(result, np_data, decimal=4)

    def test_dask_larger_chunks_than_om_3d(self):
        """Dask blocks with full trailing dims, multiple OM chunks in dim 0."""
        import dask.array as da

        np_data = np.arange(192, dtype=np.int32).reshape(4, 6, 8)
        darr = da.from_array(np_data, chunks=(4, 6, 8))

        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            var = self.write_dask_array(writer, darr, chunks=[2, 3, 4])
            writer.close(var)

            reader = OmFileReader(f.name)
            result = reader[:]
            reader.close()

            np.testing.assert_array_equal(result, np_data)

    def test_dask_single_om_chunk_per_slow_dim(self):
        """Dask blocks with 1 OM chunk in dim 0, partial trailing dim coverage."""
        import dask.array as da

        np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
        darr = da.from_array(np_data, chunks=(5, 10))

        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            var = self.write_dask_array(
                writer,
                darr,
                chunks=[5, 5],
                scale_factor=10000.0,
            )
            writer.close(var)

            reader = OmFileReader(f.name)
            result = reader[:]
            reader.close()

            np.testing.assert_array_almost_equal(result, np_data, decimal=4)

    def test_dask_misaligned_trailing_dims_raises(self):
        """Dask blocks with multi-chunk dim 0 but partial trailing dim raises."""
        import dask.array as da

        np_data = np.arange(200, dtype=np.float32).reshape(10, 20)
        darr = da.from_array(np_data, chunks=(10, 10))

        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            with pytest.raises(ValueError, match="not fully covered"):
                self.write_dask_array(writer, darr, chunks=[5, 5])

    def test_dask_not_a_dask_array_raises(self):
        np_data = np.arange(20, dtype=np.float32).reshape(4, 5)
        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            with pytest.raises(TypeError, match="Expected a dask array"):
                self.write_dask_array(writer, np_data)

    def test_dask_streaming_memory_stays_bounded(self):
        """Peak memory during a dask streaming write stays well below the full dataset size."""
        # ~16 MB dataset (2048 x 2048 x float32), written in 256x256 chunks (~256 KB each)
        side = 2048
        chunk = 256
        dtype = np.float32
        dataset_bytes = side * side * np.dtype(dtype).itemsize

        import dask.array as da

        darr = da.random.random((side, side), chunks=(chunk, chunk)).astype(dtype)

        tracemalloc.start()

        with tempfile.NamedTemporaryFile(suffix=".om") as f:
            writer = OmFileWriter(f.name)
            var = self.write_dask_array(writer, darr)
            writer.close(var)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak Python memory should be a fraction of the total dataset size,
        # proving that chunks are streamed rather than fully materialized.
        assert peak < dataset_bytes, (
            f"Peak traced memory ({peak / 1024 / 1024:.1f} MB) should be less than "
            f"the dataset size ({dataset_bytes / 1024 / 1024:.1f} MB)"
        )
