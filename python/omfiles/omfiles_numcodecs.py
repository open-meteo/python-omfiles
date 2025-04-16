from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import numpy as np
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray
from numcodecs.registry import register_codec
from numcodecs.zarr3 import (
    _make_array_bytes_codec,
    _make_bytes_bytes_codec,
)
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
    length: int|None = None
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
        length = config.get('length')

        # Create and return the codec instance
        return cls(dtype=dtype, length=length)

    def encode(self, buf):
        # Convert input to contiguous numpy array
        buf = ensure_contiguous_ndarray(buf)
        return self._impl_cache.encode_array(buf)


    def decode(self, buf, out=None):
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
        print("decode with self.length ", self.length)
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
            out = np.zeros(shape=self.length, dtype=self.dtype)
            # out = np.frombuffer(bytearray(length), dtype=np.int8) # FIXME: dtype here is set for bytes-bytes-codec
            bytes_written = self._impl_cache.decode_array(buf_array, out)
            return out


# Register the codecs with numcodecs
register_codec(PyPforDelta2d)
# register_codec("pfor_delta_2d", PyPforDelta2dCodec)

PyPforDelta2dSerializer = _make_array_bytes_codec("pfor", "PyPforDelta2d")
PyPforDelta2dCodec = _make_bytes_bytes_codec("pfor", "PyPforDelta2d")
