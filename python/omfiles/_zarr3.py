"""ArrayBytesCodec and BytesBytesCodec for TurboPFor."""

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Self

import numpy as np

if TYPE_CHECKING:
    from zarr.core.dtype import TBaseDType, TBaseScalar, ZDType
else:
    TBaseDType = object
    TBaseScalar = object
    ZDType = object

try:
    import zarr
except ImportError as e:
    raise ImportError(
        "The omfiles.zarr module requires the 'zarr' package. Install it with: pip install omfiles[codec]"
    ) from e

from importlib.metadata import version

from packaging.version import Version
from zarr.abc.codec import ArrayBytesCodec, BytesBytesCodec
from zarr.abc.metadata import Metadata
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer.core import Buffer, NDBuffer
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.common import JSON, BytesLike, ChunkCoords

from .omfiles import RustPforCodec


def _from_zarr_dtype(dtype: Any) -> np.dtype:
    """
    Get a numpy data type from an array spec, depending on the zarr version.
    """
    if Version(version("zarr")) >= Version("3.1.0"):
        if hasattr(dtype, "to_native_dtype"):
            return dtype.to_native_dtype()
        elif hasattr(dtype, "type"):
            return np.dtype(dtype.type)

    return dtype


_SUPPORTED_DATATYPES = [
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    np.dtype(np.uint64),
    np.dtype(np.int8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.int64),
]


@dataclass(frozen=True)
class PforSerializer(ArrayBytesCodec, Metadata):
    """Array-to-bytes codec for PFor-Delta 2D compression."""

    _impl = RustPforCodec()
    name: str = "omfiles.pfor_serializer"
    config: dict[str, JSON] = field(default_factory=dict)

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        """Encode a single array chunk to bytes."""
        numpy_dtype = _from_zarr_dtype(chunk_spec.dtype)

        out = await asyncio.to_thread(
            self._impl.encode_array, np.ascontiguousarray(chunk_data.as_numpy_array()), numpy_dtype
        )
        return chunk_spec.prototype.buffer.from_bytes(out)

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        """Decode a single byte chunk to an array."""
        chunk_bytes = chunk_data.to_bytes()
        numpy_dtype = _from_zarr_dtype(chunk_spec.dtype)
        out = await asyncio.to_thread(self._impl.decode_array, chunk_bytes, numpy_dtype, int(np.prod(chunk_spec.shape)))
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))  # type: ignore

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Self:
        """Create codec instance from configuration."""
        return cls()


@dataclass(frozen=True)
class PforCodec(BytesBytesCodec, Metadata):
    """Bytes-to-bytes codec for PFor-Delta 2D compression."""

    _impl = RustPforCodec()
    name: str = "omfiles.pfor"
    config: dict[str, JSON] = field(default_factory=dict)

    async def _encode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        """Encode a single bytes chunk."""
        out = await asyncio.to_thread(self._impl.encode_array, chunk_data.as_array_like(), np.dtype("uint8"))  # type: ignore
        return chunk_spec.prototype.buffer.from_bytes(out)

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        """Decode a single bytes chunk."""
        out = (
            await asyncio.to_thread(
                self._impl.decode_array,
                chunk_data.to_bytes(),
                np.dtype("uint8"),
                int(np.prod(chunk_spec.shape)) * chunk_spec.dtype.to_native_dtype().itemsize,
            )
        ).tobytes()
        return chunk_spec.prototype.buffer.from_bytes(out)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Self:
        """Create codec instance from configuration."""
        return cls()
