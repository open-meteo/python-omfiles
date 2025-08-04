import sys

import numpy as np
import pytest

pytest.importorskip("numcodecs.zarr3")
from numcodecs.zarr3 import Delta
from omfiles._zarr3 import PforCodec, PforSerializer
from zarr import create_array
from zarr.abc.store import Store
from zarr.storage import LocalStore, MemoryStore, StorePath


@pytest.fixture
def store(request):
    if request.param == "memory":
        yield MemoryStore()
    elif request.param == "local":
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()
        store = LocalStore(temp_dir)
        yield store
        shutil.rmtree(temp_dir)
    else:
        raise ValueError(f"Unknown store type: {request.param}")


test_dtypes = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python >= 3.11")
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("dtype", test_dtypes)
async def test_pfordelta_roundtrip(store: Store, dtype: np.dtype) -> None:
    """Test roundtrip encoding/decoding similar to the Rust test."""

    path = "pfordelta_roundtrip"
    spath = StorePath(store, path)
    assert await store.is_empty("")

    data = np.array(
        [
            [10, 22, 23, 24, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 12, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 12, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        ],
        dtype=np.dtype(dtype),
    )

    chunk_shape = (1, 10)  # NOTE: chunks are no clean divisor of data.shape

    # Create array with our codec
    z = create_array(
        spath,
        shape=data.shape,
        chunks=chunk_shape,
        dtype=data.dtype,
        fill_value=0,
        filters=[],
        # Codec is used as a byte-byte-compressor here
        compressors=PforCodec(),
    )

    bytes_before = z.nbytes_stored()

    assert not await store.is_empty("")

    # Write the test data
    z[:] = data
    bytes_after = z.nbytes_stored()
    assert bytes_after > bytes_before

    np.testing.assert_array_equal(z[:], data)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python >= 3.11")
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("dtype", test_dtypes)
async def test_pfor_serializer_roundtrip(store: Store, dtype: np.dtype) -> None:
    """Test PforSerializer as an array-to-bytes codec (serializer)."""

    path = "pfor_serializer_roundtrip"
    spath = StorePath(store, path)
    assert await store.is_empty("")

    data = np.array(
        [
            [10, 22, 23, 24, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 12, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 12, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        ],
        dtype=np.dtype(dtype),
    )

    # Different chunk than above
    chunk_shape = (2, 5)

    z = create_array(
        spath,
        shape=data.shape,
        chunks=chunk_shape,
        dtype=data.dtype,
        fill_value=0,
        # PforSerializer can serialize all integer types as an array-to-bytes codec
        serializer=PforSerializer(),
        filters=[],
        compressors=[],
    )

    bytes_before = z.nbytes_stored()
    assert not await store.is_empty("")

    # Write the test data
    z[:] = np.ascontiguousarray(data)
    bytes_after = z.nbytes_stored()
    assert bytes_after > bytes_before

    # Verify data matches after roundtrip
    np.testing.assert_array_equal(z[:], data)
