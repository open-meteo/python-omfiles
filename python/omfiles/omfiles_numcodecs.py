import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Self

import numpy as np
import numpy.typing as npt
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray
from numcodecs.registry import register_codec
from numcodecs.zarr3 import ArrayBytesCodec, BufferPrototype, NDBuffer, _NumcodecsCodec
from zarr.abc.codec import BytesBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer
from zarr.core.common import JSON, parse_named_configuration

from .omfiles import (
    # PyPforDelta2dInt16Codec as RustPforDelta2dInt16Codec,
    # PyPforDelta2dInt16LogarithmicCodec as RustPforDelta2dInt16LogarithmicCodec,
    FpxXor2dCodec as RustFpxXor2dCodec,  # type: ignore[arg-type]
)
from .omfiles import (
    PforDelta2dCodec as RustPforDelta2dCodec,  # type: ignore[arg-type]
)


@dataclass(frozen=True)
class PyFpxXor2dCodec(BytesBytesCodec):
    """FPX XOR 2D compression for floating-point data."""
    codec_id = 'fpx_xor_2d'
    is_fixed_size = False
    dtype: str = "float32"

    def __init__(self, dtype: str = "float32") -> None:
        dtype_parsed = parse_dtype(dtype)
        object.__setattr__(self, "dtype", dtype_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "fpx_xor_2d")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": "fpx_xor_2d",
            "configuration": {
                "dtype": self.dtype,
            },
        }

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        # Possibly we could auto-adapt the dtype based on the array_spec
        # For now, just return self since we don't adapt
        return self

    @property
    def _impl(self) -> RustFpxXor2dCodec:
        return RustFpxXor2dCodec(self.dtype)

    async def _decode_single(
        self,
        chunk_data: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await asyncio.to_thread(
                lambda bytes_data: chunk_spec.prototype.buffer.from_bytes(
                    self._impl.decode(bytes_data.to_bytes(), np.prod(chunk_spec.shape))
                ),
                chunk_data
            )

    async def _encode_single(
        self,
        chunk_data: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return await asyncio.to_thread(
            lambda chunk: chunk_spec.prototype.buffer.from_bytes(
                self._impl.encode(chunk.as_numpy_array().tobytes())
            ),
            chunk_data,
        )

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        # FPX XOR doesn't have a guaranteed size formula, so we provide a conservative estimate
        return input_byte_length * 2  # A conservative estimate


def parse_dtype(data: JSON) -> str:
    if not isinstance(data, str):
        raise TypeError(f"Value must be a string. Got {type(data)} instead.")
    valid_dtypes = ["int8", "uint8", "int16", "uint16", "int32", "uint32",
                    "int64", "uint64", "float32", "float64"]
    if data not in valid_dtypes:
        raise ValueError(f"dtype must be one of {valid_dtypes}. Got '{data}'")
    return data

# @dataclass(frozen=True)
# class PyPforDelta2dCodec(BytesBytesCodec):
#     """PFor-Delta 2D compression for various data types."""
#     codec_id: ClassVar[str] = "pfor_delta_2d"
#     is_fixed_size = False
#     dtype: str = "int16"
#     _impl_cache: RustPforDelta2dCodec = field(init=False, repr=False, compare=False, default=None)
#     _buffer_cache: dict = field(init=False, repr=False, compare=False, default_factory=dict)
#     _lock: asyncio.Lock = field(init=False, repr=False, compare=False, default_factory=asyncio.Lock)

#     def __init__(self, dtype: str = "int16") -> None:
#         dtype_parsed = parse_dtype(dtype)
#         object.__setattr__(self, "dtype", dtype_parsed)
#         object.__setattr__(self, "_impl_cache", RustPforDelta2dCodec(dtype_parsed))
#         object.__setattr__(self, "_buffer_cache", {})
#         object.__setattr__(self, "_lock", asyncio.Lock())

#     @classmethod
#     def from_dict(cls, data: dict[str, JSON]) -> Self:
#         _, configuration_parsed = parse_named_configuration(data, "pfor_delta_2d")
#         return cls(**configuration_parsed)  # type: ignore[arg-type]

#     def to_dict(self) -> dict[str, JSON]:
#         return {
#             "name": "pfor_delta_2d",
#             "configuration": {
#                 "dtype": self.dtype,
#             },
#         }

#     def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
#         # Possibly we could auto-adapt the dtype based on the array_spec
#         # For now, just return self since we don't adapt
#         return self

#     @property
#     def _impl(self) -> RustPforDelta2dCodec:
#         return self._impl_cache

#     async def _decode_single(
#         self,
#         chunk_data: Buffer,
#         chunk_spec: ArraySpec,
#     ) -> Buffer:
#         # Use a thread pool executor to avoid blocking the event loop
#         loop = asyncio.get_running_loop()

#         # Get the expected output size
#         output_size = np.prod(chunk_spec.shape) * np.dtype(self.dtype).itemsize

#         # Get or create an appropriately sized output buffer
#         async with self._lock:
#             buffer_key = f"decode_{output_size}"
#             if buffer_key not in self._buffer_cache:
#                 # Create a new output buffer for this size
#                 self._buffer_cache[buffer_key] = np.frombuffer(bytearray(output_size), dtype=np.int8)

#             # Get the reusable buffer
#             output_buffer = self._buffer_cache[buffer_key]

#         # Run the decode operation in a thread
#         return await loop.run_in_executor(
#             None,
#             self._decode_sync,
#             chunk_data,
#             chunk_spec,
#             output_buffer
#         )

#     def _decode_sync(self, chunk_data: Buffer, chunk_spec: ArraySpec, output_buffer=None) -> Buffer:
#         # Synchronous decode function to run in a thread
#         if output_buffer is None:
#             # No reusable buffer provided, fallback to standard decoding
#             decoded = self._impl.decode(chunk_data.to_bytes(), np.prod(chunk_spec.shape))
#             return chunk_spec.prototype.buffer.from_bytes(decoded)
#         else:
#             print("output buffer out type", output_buffer.dtype)
#             # Use the reusable buffer
#             bytes_written = self._impl.decode_array(
#                 chunk_data.as_array_like(),
#                 output_buffer
#             )

#             # Create a Buffer from the appropriate slice of the output buffer
#             return chunk_spec.prototype.buffer.from_bytes(bytes(output_buffer[:bytes_written]))

#     async def _encode_single(
#         self,
#         chunk_data: Buffer,
#         chunk_spec: ArraySpec,
#     ) -> Buffer | None:
#         # Use a thread pool executor to avoid blocking the event loop
#         loop = asyncio.get_running_loop()

#         # For encoding, we generally can't know the output size in advance
#         # so we won't try to reuse buffers for the output
#         return await loop.run_in_executor(
#             None,
#             self._encode_sync,
#             chunk_data,
#             chunk_spec
#         )

#     def _encode_sync(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
#         # Synchronous encode function to run in a thread
#         encoded = self._impl.encode_array(chunk_data.as_array_like())
#         # encoded = self._impl.encode(chunk_data.to_bytes())
#         return chunk_spec.prototype.buffer.from_bytes(encoded)

#     def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
#         # PFor-Delta doesn't have a guaranteed size formula, so we can't implement this precisely
#         # Return double the size to be safe
#         return input_byte_length * 2 # A conservative estimate



@dataclass(frozen=True)
class PyPforDelta2d(Codec):
    """PFor-Delta 2D compression for various data types."""
    codec_id: ClassVar[str] = "pfor"
    is_fixed_size = False
    dtype: str = "int16"
    _impl_cache: Any = field(init=False, repr=False, compare=False, default=None)
    _buffer_pool: dict = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "_impl_cache", RustPforDelta2dCodec(self.dtype))
        object.__setattr__(self, "_buffer_pool", {})

    @property
    def _impl(self) -> RustPforDelta2dCodec:
        return self._impl_cache

    @classmethod
    def from_config(cls, config) -> Self:
        """Create a codec instance from a configuration dict.

        Parameters
        ----------
        config : dict
            Configuration parameters for this codec.

        Returns
        -------
        codec : PyPforDelta2d
            A codec instance.
        """
        # Extract the configuration parameters
        dtype = config.get('dtype', 'int16')

        # Create and return the codec instance
        return cls(dtype=dtype)

    def encode(self, buf):
        # Convert input to contiguous numpy array
        buf = ensure_contiguous_ndarray(buf)

        # Use our optimized array-based encoding when possible
        try:
            return self._impl_cache.encode_array(buf)
        except (TypeError, AttributeError):
            # Fall back to byte encoding if array method not available
            return self._impl_cache.encode(buf.tobytes())


    def decode(self, buf, out=None, length: int|None = None):
        """Decode (decompress) data.

        Parameters
        ----------
        buf : bytes-like
            Data to be decoded.
        out : array-like, optional
            Array to store the decoded results. If provided, its size determines
            how many elements to decode.

        Returns
        -------
        out : numpy.ndarray
            Decoded data as a numpy array.
        """
        print("out", out)
        if out is not None:
            # Case 1: Output array provided - use its size
            print("type(out)", type(out))
            print("self.dtype", self.dtype)
            print("out.dtype", out.dtype)
            print("out.shape", out.shape)
            out = ensure_contiguous_ndarray(out)

            # Convert input to numpy array
            if not isinstance(buf, np.ndarray):
                buf_array = np.frombuffer(buf, dtype=np.int8)
            else:
                buf_array = buf

            # Use array-based decoding
            bytes_written = self._impl_cache.decode_array(buf_array, out)
            return out

        else:
            # Case 2: No output array - we need to determine size
            # create output buffer
            print("length", length)
            # Convert input to numpy array
            if not isinstance(buf, np.ndarray):
                buf_array = np.frombuffer(buf, dtype=np.int8)
            else:
                buf_array = buf
            out = np.frombuffer(bytearray(length), dtype=self.dtype)
            print("out.__len__()", out.__len__())
            bytes_written = self._impl_cache.decode_array(buf_array, out)
            return out


# Register the codecs with numcodecs
register_codec(PyPforDelta2d)
# register_codec("pfor_delta_2d", PyPforDelta2dCodec)


class _NumcodecsArrayBytesCodecWithShapeLength(_NumcodecsCodec, ArrayBytesCodec):
    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_bytes = chunk_data.to_bytes()
        chunk_length = np.prod(chunk_spec.shape) * np.dtype(chunk_spec.dtype).itemsize
        out = await asyncio.to_thread(self._codec.decode, chunk_bytes, None, chunk_length)

        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.buffer.from_bytes(out)

CODEC_PREFIX = "numcodecs."

def _make_array_bytes_codec_with_length(codec_name: str, cls_name: str) -> type[_NumcodecsArrayBytesCodecWithShapeLength]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _Codec(_NumcodecsArrayBytesCodecWithShapeLength):
        codec_name = _codec_name

        def __init__(self, **codec_config: JSON) -> None:
            super().__init__(**codec_config)

    _Codec.__name__ = cls_name
    return _Codec


def as_numpy_array_wrapper(
    func: Callable[[npt.NDArray[Any]], bytes],
    buf: Buffer,
    prototype: BufferPrototype,
    length: int
) -> Buffer:
    """Converts the input of `func` to a numpy array and the output back to `Buffer`.

    This function is useful when calling a `func` that only support host memory such
    as `GZip.decode` and `Blosc.decode`. In this case, use this wrapper to convert
    the input `buf` to a Numpy array and convert the result back into a `Buffer`.

    Parameters
    ----------
    func
        The callable that will be called with the converted `buf` as input.
        `func` must return bytes, which will be converted into a `Buffer`
        before returned.
    buf
        The buffer that will be converted to a Numpy array before given as
        input to `func`.
    prototype
        The prototype of the output buffer.

    Returns
    -------
        The result of `func` converted to a `Buffer`
    """
    return prototype.buffer.from_bytes(func(buf.as_numpy_array(), None, length))

class _NumcodecsBytesBytesCodecWithShapeLength(_NumcodecsCodec, BytesBytesCodec):
    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        chunk_length = int(np.prod(chunk_spec.shape) * np.dtype(chunk_spec.dtype).itemsize)
        print("chunk_length", chunk_length)
        return await asyncio.to_thread(
            as_numpy_array_wrapper,
            self._codec.decode,
            chunk_data,
            chunk_spec.prototype,
            chunk_length
        )

    def _encode(self, chunk_bytes: Buffer, prototype: BufferPrototype) -> Buffer:
        encoded = self._codec.encode(chunk_bytes.as_array_like())
        if isinstance(encoded, np.ndarray):  # Required for checksum codecs
            return prototype.buffer.from_bytes(encoded.tobytes())
        return prototype.buffer.from_bytes(encoded)

    async def _encode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(self._encode, chunk_data, chunk_spec.prototype)

def _make_bytes_bytes_codec_with_length(codec_name: str, cls_name: str) -> type[_NumcodecsBytesBytesCodecWithShapeLength]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _Codec(_NumcodecsBytesBytesCodecWithShapeLength):
        codec_name = _codec_name

        def __init__(self, **codec_config: JSON) -> None:
            super().__init__(**codec_config)

    _Codec.__name__ = cls_name
    return _Codec



PyPforDelta2dSerializer = _make_array_bytes_codec_with_length("pfor", "PyPforDelta2d")

PyPforDelta2dCodec = _make_bytes_bytes_codec_with_length("pfor", "PyPforDelta2d")
