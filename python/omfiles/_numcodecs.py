"""TurboPfor codec implementation as numcodecs.abc.Codec."""

from dataclasses import dataclass
from typing import Self

try:
    import numcodecs.abc
except ImportError as e:
    raise ImportError(
        "The omfiles.numcodecs module requires the 'numcodecs' package. Install it with: pip install omfiles[codec]"
    ) from e
import numpy as np

from .omfiles import RustPforCodec


@dataclass
class TurboPfor(numcodecs.abc.Codec):
    """TurboPfor codec for compressing integer arrays."""

    codec_id = "turbo_pfor"
    dtype: str = "int16"
    chunk_elements: int | None = None
    _impl = RustPforCodec()

    @classmethod
    def from_config(cls, config) -> Self:
        """
        Create a codec instance from a configuration dict.
        """
        dtype = config.get("dtype", "int16")
        chunk_elements = config.get("chunk_elements")
        return cls(dtype=dtype, chunk_elements=chunk_elements)

    def encode(self, buf):  # type: ignore
        """Encode a numpy array using TurboPfor."""
        return self._impl.encode_array(buf, np.dtype(self.dtype))

    def decode(self, buf, out=None):  # type: ignore
        """Decode a buffer using TurboPfor."""
        if out is not None:
            raise ValueError("Output array not supported")

        assert self.chunk_elements is not None, "chunk_elements must be set"

        if isinstance(buf, np.ndarray):
            buf = buf.tobytes()
        else:
            buf = buf

        return self._impl.decode_array(buf, np.dtype(self.dtype), self.chunk_elements)


numcodecs.register_codec(TurboPfor)
