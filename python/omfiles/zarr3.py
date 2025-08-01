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

from packaging import version
from zarr.abc.codec import ArrayBytesCodec, BytesBytesCodec
from zarr.abc.metadata import Metadata
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer.core import Buffer, NDBuffer
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.common import JSON, ChunkCoords

from .omfiles import (
    PforDelta2dCodec as RustPforCodec,  # type: ignore[arg-type]
)

ZARR_VERSION_AFTER_3_1_0 = version.parse(zarr.__version__) >= version.parse("3.1.0")


@dataclass(frozen=True)
class PforSerializer(ArrayBytesCodec, Metadata):
    """Array-to-bytes codec for PFor-Delta 2D compression."""

    _impl = RustPforCodec()
    name: str = "omfiles.pfor_serializer"
    config: dict[str, JSON] = field(default_factory=dict)

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

    def validate(self, *, shape: ChunkCoords, dtype: "ZDType[TBaseDType, TBaseScalar]", chunk_grid: ChunkGrid) -> None:
        """Validate codec compatibility with the array spec."""
        pass
        # if dtype != np.dtype(self.dtype):
        #     raise ValueError(f"Array dtype {dtype} doesn't match codec dtype {self.dtype}")

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
        out = await asyncio.to_thread(self._impl.encode_array, chunk_data.as_array_like(), np.dtype("uint8"))
        return chunk_spec.prototype.buffer.from_bytes(out)

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        """Decode a single bytes chunk."""
        out = await asyncio.to_thread(
            self._impl.decode_array,
            chunk_data.to_bytes(),
            np.dtype("uint8"),
            np.prod(chunk_spec.shape) * chunk_spec.dtype.to_native_dtype().itemsize,
        )
        return chunk_spec.prototype.buffer.from_bytes(out)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Self:
        """Create codec instance from configuration."""
        return cls()
