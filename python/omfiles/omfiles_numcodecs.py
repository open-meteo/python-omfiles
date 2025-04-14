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
from zarr.core.common import JSON

from .omfiles import (
    PforDelta2dCodec as RustPforDelta2dCodec,  # type: ignore[arg-type]
)


def parse_dtype(data: JSON) -> str:
    if not isinstance(data, str):
        raise TypeError(f"Value must be a string. Got {type(data)} instead.")
    valid_dtypes = ["int8", "uint8", "int16", "uint16", "int32", "uint32",
                    "int64", "uint64", "float32", "float64"]
    if data not in valid_dtypes:
        raise ValueError(f"dtype must be one of {valid_dtypes}. Got '{data}'")
    return data


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
        return self._impl_cache.encode_array(buf)


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
        if out is not None:
            # Case 1: Output array provided - use its size
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
            # Convert input to numpy array
            if not isinstance(buf, np.ndarray):
                buf_array = np.frombuffer(buf, dtype=np.int8)
            else:
                buf_array = buf
            out = np.frombuffer(bytearray(length), dtype=np.int8) # FIXME: dtype here is set for bytes-bytes-codec
            bytes_written = self._impl_cache.decode_array(buf_array, out)
            return out


# Register the codecs with numcodecs
register_codec(PyPforDelta2d)
# register_codec("pfor_delta_2d", PyPforDelta2dCodec)


class _NumcodecsArrayBytesCodecWithShapeLength(_NumcodecsCodec, ArrayBytesCodec):
    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_bytes = chunk_data.as_array_like()

        chunk_length = np.prod(chunk_spec.shape) * np.dtype(chunk_spec.dtype).itemsize
        out = np.frombuffer(bytearray(chunk_length), dtype=chunk_spec.dtype)
        await asyncio.to_thread(self._codec.decode, chunk_bytes, out, chunk_length)

        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        out = await asyncio.to_thread(self._codec.encode, chunk_data.as_ndarray_like())
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
    return prototype.buffer.from_bytes(func(buf.as_array_like(), None, length))

class _NumcodecsBytesBytesCodecWithShapeLength(_NumcodecsCodec, BytesBytesCodec):
    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        chunk_length = int(np.prod(chunk_spec.shape) * np.dtype(chunk_spec.dtype).itemsize)
        return await asyncio.to_thread(
            as_numpy_array_wrapper,
            self._codec.decode,
            chunk_data,
            chunk_spec.prototype,
            chunk_length
        )

    def _encode(self, chunk_bytes: Buffer, prototype: BufferPrototype) -> Buffer:
        encoded = self._codec.encode(chunk_bytes.as_array_like())
        # if isinstance(encoded, np.ndarray):  # Required for checksum codecs
        #     return prototype.buffer.from_bytes(encoded.tobytes())
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
