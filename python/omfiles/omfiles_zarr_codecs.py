import asyncio
from dataclasses import dataclass, field

import numpy as np

from .omfiles import (
    PforDelta2dCodec as RustPforCodec,  # type: ignore[arg-type]
)

try:
    from typing import Any, Dict, Self

    import numcodecs.abc

    @dataclass(frozen=True)
    class TurboPfor(numcodecs.abc.Codec):
        codec_id = "turbo_pfor"
        dtype: str = "int16"
        chunk_elements: int | None = None

        @classmethod
        def from_config(cls, config) -> Self:
            """
            Create a codec instance from a configuration dict.
            """
            dtype = config.get("dtype", "int16")
            chunk_elements = config.get("chunk_elements")
            return cls(dtype=dtype, chunk_elements=chunk_elements)

        @property
        def _impl(self) -> RustPforCodec:
            return RustPforCodec()

        def encode(self, buf):
            return self._impl.encode_array(buf, np.dtype(self.dtype))

        def decode(self, buf, out=None):
            if out is not None:
                raise ValueError("Output array not supported")

            if isinstance(buf, np.ndarray):
                buf = buf.tobytes()
            else:
                buf = buf

            return self._impl.decode_array(buf, np.dtype(self.dtype), self.chunk_elements)

    numcodecs.register_codec(TurboPfor)
except ImportError:

    class TurboPfor:
        def __init__(self, *args, **kwargs):
            raise ImportError("numcodecs must be installed to use TurboPfor codec.")


try:
    import zarr
    from packaging import version
    from zarr.abc.codec import ArrayBytesCodec, BytesBytesCodec
    from zarr.abc.metadata import Metadata
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer.core import Buffer, NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.common import JSON, ChunkCoords

    ZARR_VERSION_AFTER_3_1_0 = version.parse(zarr.__version__) >= version.parse("3.1.0")

    @dataclass(frozen=True)
    class PforSerializer(ArrayBytesCodec, Metadata):
        """Array-to-bytes codec for PFor-Delta 2D compression."""

        name: str = "omfiles.pfor_serializer"
        config: dict[str, JSON] = field(default_factory=dict)

        # _impl_cache: Any = field(init=False, repr=False, compare=False, default=None)

        # def __post_init__(self):
        #     object.__setattr__(self, "_impl_cache", RustPforCodec(self.dtype))

        @property
        def _impl(self) -> RustPforCodec:
            return RustPforCodec()

        async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
            """Encode a single array chunk to bytes."""
            if ZARR_VERSION_AFTER_3_1_0:
                numpy_dtype = chunk_spec.dtype.to_native_dtype()
            else:
                numpy_dtype = chunk_spec.dtype

            out = await asyncio.to_thread(
                self._impl.encode_array, np.ascontiguousarray(chunk_data.as_numpy_array()), numpy_dtype
            )
            return chunk_spec.prototype.buffer.from_bytes(out)

        async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
            """Decode a single byte chunk to an array."""
            chunk_bytes = chunk_data.to_bytes()
            if ZARR_VERSION_AFTER_3_1_0:
                numpy_dtype = chunk_spec.dtype.to_native_dtype()
            else:
                numpy_dtype = chunk_spec.dtype
            out = await asyncio.to_thread(self._impl.decode_array, chunk_bytes, numpy_dtype, np.prod(chunk_spec.shape))
            return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

        def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
            # PFor compression is variable-size
            raise NotImplementedError("PFor compression produces variable-sized output")

        def validate(self, *, shape: ChunkCoords, dtype: np.dtype[Any], chunk_grid: ChunkGrid) -> None:
            """Validate codec compatibility with the array spec."""
            pass
            # if dtype != np.dtype(self.dtype):
            #     raise ValueError(f"Array dtype {dtype} doesn't match codec dtype {self.dtype}")

        @classmethod
        def from_config(cls, config: Dict[str, Any]) -> Self:
            """Create codec instance from configuration."""
            return cls()
            # dtype = config.get('dtype', 'int16')
            # length = config.get('length')
            # return cls(dtype=dtype, length=length)

    @dataclass(frozen=True)
    class PforCodec(BytesBytesCodec, Metadata):
        """Bytes-to-bytes codec for PFor-Delta 2D compression."""

        name: str = "omfiles.pfor"
        config: dict[str, JSON] = field(default_factory=dict)

        # _impl_cache: Any = field(init=False, repr=False, compare=False, default=None)

        # def __post_init__(self):
        #     object.__setattr__(self, "_impl_cache", RustPforCodec(self.dtype))

        @property
        def _impl(self) -> RustPforCodec:
            return RustPforCodec()

        async def _encode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
            """Encode a single bytes chunk."""
            out = await asyncio.to_thread(self._impl.encode_array, chunk_data.as_array_like(), np.dtype("uint8"))
            return chunk_spec.prototype.buffer.from_bytes(out)

        async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
            """Decode a single bytes chunk."""
            out = await asyncio.to_thread(
                self._impl.decode_array,
                chunk_data.to_bytes(),
                np.dtype("uint8"),
                np.prod(chunk_spec.shape) * chunk_spec.dtype.item_size,
            )
            return chunk_spec.prototype.buffer.from_bytes(out)

        def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
            # PFor compression is variable-size
            raise NotImplementedError("PFor compression produces variable-sized output")

        @classmethod
        def from_config(cls, config: Dict[str, Any]) -> Self:
            """Create codec instance from configuration."""
            return cls()
            # dtype = config.get('dtype', 'int16')
            # length = config.get('length')
            # return cls(dtype=dtype, length=length)
except ImportError:

    class PforSerializer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Zarr must be installed to use omfiles.pfor_serializer codec.")

    class PforCodec:
        def __init__(self, *args, **kwargs):
            raise ImportError("Zarr must be installed to use omfiles.pfor codec.")
