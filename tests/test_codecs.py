import numpy as np
import pytest
from numcodecs.zarr3 import Delta
from omfiles.omfiles_zarr_codecs import PforCodec, PforSerializer
from zarr import create_array
from zarr.abc.store import Store
from zarr.storage import LocalStore, MemoryStore, StorePath

delta_config = {
    'float32': '<f4',
    'float64': '<f8',
    'int8': '<i1',
    'uint8': '<u1',
    'int16': '<i2',
    'uint16': '<u2',
    'int32': '<i4',
    'uint32': '<u4',
    'int64': '<i8',
    'uint64': '<u8'
}


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

test_dtypes = [
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64
]

@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("dtype", test_dtypes)
async def test_pfordelta_roundtrip(store: Store, dtype: np.dtype) -> None:
    """Test roundtrip encoding/decoding similar to the Rust test."""

    path = "pfordelta_roundtrip"
    spath = StorePath(store, path)
    assert await store.is_empty("")

    # Create data similar to the Rust test
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
        dtype=np.dtype(dtype))

    chunk_shape = (1, 10) # NOTE: chunks are no clean divisor of data.shape

    delta_filter = Delta(dtype=delta_config[dtype.__name__])
    # Create array with our codec
    z = create_array(
        spath,
        shape=data.shape,
        chunks=chunk_shape,
        dtype=data.dtype,
        fill_value=0,
        filters=[delta_filter],
        # Codec is used as a byte-byte-compressor here, therefore dtype is set like this.
        # We should rather use it as a serializer, i.e. ByteArrayCompressor
        compressors=PforCodec()
    )

    bytes_before = z.nbytes_stored()

    assert not await store.is_empty("")

    # Write the test data
    z[:] = data
    bytes_after = z.nbytes_stored()
    assert bytes_after > bytes_before


    # Verify data matches
    np.testing.assert_array_equal(z[:], data)

    # Check original and compressed sizes
    original_size = data.nbytes

    # For detailed analysis, we could trace the actual compression ratio
    # In a real scenario, you might want to capture metrics about the compression

    print(f"Successfully round-tripped data. Original size: {original_size} bytes, Compressed size: {bytes_after - bytes_before} bytes")

@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("dtype", test_dtypes)
async def test_pfor_serializer_roundtrip(store: Store, dtype: np.dtype) -> None:
    """Test PforSerializer as an array-to-bytes codec (serializer)."""

    path = "pfor_serializer_roundtrip"
    spath = StorePath(store, path)
    assert await store.is_empty("")

    # Create test data
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
        dtype=np.dtype(dtype))

    chunk_shape = (2, 5)  # Different chunk shape for variety

    # Create array with PforSerializer as the serializer
    z = create_array(
        spath,
        shape=data.shape,
        chunks=chunk_shape,
        dtype=data.dtype,
        fill_value=0,
        serializer=PforSerializer(),  # Use PforSerializer as the array-to-bytes codec
        filters=[],  # No filters needed, can use empty list
        compressors=[]  # No additional compression
    )

    bytes_before = z.nbytes_stored()
    assert not await store.is_empty("")

    # Write the test data
    z[:] = data
    bytes_after = z.nbytes_stored()
    assert bytes_after > bytes_before

    # Verify data matches after roundtrip
    np.testing.assert_array_equal(z[:], data)

    # Check original and compressed sizes
    original_size = data.nbytes
    print(f"PforSerializer test: Original size: {original_size} bytes, Stored size: {bytes_after - bytes_before} bytes")
