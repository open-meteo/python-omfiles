import numpy as np
import pytest
from numcodecs.zarr3 import Delta

# from numcodecs.delta import Delta
from omfiles.omfiles_numcodecs import PyPforDelta2dCodec
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

float_test_dtypes = [
    np.float32,
    np.float64
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

    delta_filter = Delta(dtype=delta_config[dtype.__name__])
    # Create array with our codec
    z = create_array(
        spath,
        shape=data.shape,
        chunks=(1,10), # NOTE: chunks are no clean divisor of data.shape
        dtype=data.dtype,
        fill_value=0,
        filters=[delta_filter],
        compressors=PyPforDelta2dCodec(dtype=dtype.__name__)
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


# @pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
# @pytest.mark.parametrize("dtype", float_test_dtypes)
# async def test_fpxxor_roundtrip(store: Store, dtype: np.dtype) -> None:
#     """Test roundtrip encoding/decoding for FPX XOR codec with floating point data."""

#     path = "fpxxor_roundtrip"
#     spath = StorePath(store, path)
#     assert await store.is_empty("")

#     # Create floating point test data with similar patterns
#     data = np.array(
#         [
#             [10.5, 22.3, 23.1, 24.9, 29.4, 30.6, 31.2, 32.0, 33.7, 34.1],
#             [25.2, 26.8, 27.3, 12.4, 29.5, 30.2, 31.9, 32.3, 33.8, 34.2],
#             [25.3, 26.7, 27.2, 12.5, 29.6, 30.3, 31.8, 32.5, 33.9, 34.3],
#             [25.4, 26.6, 27.1, 28.6, 29.7, 30.4, 31.7, 32.6, 33.0, 34.4],
#             [25.5, 26.5, 27.0, 28.7, 29.8, 30.5, 31.6, 32.7, 33.1, 34.5],
#             [25.6, 26.4, 27.9, 28.8, 29.9, 30.1, 31.5, 32.8, 33.2, 34.6],
#             [25.7, 26.3, 27.8, 28.9, 29.0, 30.7, 31.4, 32.9, 33.3, 34.7],
#         ],
#         dtype=np.dtype(dtype))

#     print(delta_config[dtype.__name__])
#     delta_filter = Delta(dtype=delta_config[dtype.__name__])
#     quantize_filter = Quantize(digits=1, dtype=delta_config[dtype.__name__])
#     # Create array with FPX XOR codec
#     z = create_array(
#         spath,
#         shape=data.shape,
#         chunks=(1,10),
#         dtype=data.dtype,
#         fill_value=0,
#         filters=[delta_filter, quantize_filter],
#         compressors=PyFpxXor2dCodec(dtype=dtype.__name__)
#     )

#     bytes_before = z.nbytes_stored()

#     assert not await store.is_empty("")

#     # Write the test data
#     z[:] = data
#     bytes_after = z.nbytes_stored()
#     assert bytes_after > bytes_before

#     # Verify data matches
#     np.testing.assert_array_almost_equal(z[:], data, decimal=1)

#     # Check original and compressed sizes
#     original_size = data.nbytes

#     print(f"Successfully round-tripped floating point data ({dtype.__name__}). Original size: {original_size} bytes, Compressed size: {bytes_after - bytes_before} bytes")
