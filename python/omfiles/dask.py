"""Dask array integration for writing to OM files."""

import math
from typing import Iterator, Optional, Sequence

import numpy as np

from omfiles._rust import OmFileWriter, OmVariable

try:
    import dask.array as da
except ImportError:
    raise ImportError("omfiles[dask] is required for dask functionality")


def _validate_chunk_alignment(
    data_chunks: tuple,
    om_chunks: list[int],
    array_shape: tuple,
) -> None:
    """
    Validate dask chunks are compatible with OM chunks for block-level streaming.

    Every non-last dask chunk along each dimension must be an exact multiple
    of the corresponding OM chunk size (the last chunk may be smaller).
    Additionally, for the leftmost dimension where a dask block contains more
    than one OM chunk, every trailing dimension must be fully covered by each
    dask block. This ensures the local chunk traversal inside a block matches
    the global file order.
    """
    ndim = len(om_chunks)

    for d in range(ndim):
        dim_chunks = data_chunks[d]
        for i, c in enumerate(dim_chunks[:-1]):
            if c % om_chunks[d] != 0:
                raise ValueError(
                    f"Dask chunk size {c} along dimension {d} (block {i}) "
                    f"is not a multiple of the OM chunk size {om_chunks[d]}."
                )

    first_multi = None
    for d in range(ndim):
        local_n = math.ceil(data_chunks[d][0] / om_chunks[d])
        if local_n > 1:
            first_multi = d
            break

    if first_multi is not None:
        for d in range(first_multi + 1, ndim):
            local_n = math.ceil(data_chunks[d][0] / om_chunks[d])
            global_n = math.ceil(array_shape[d] / om_chunks[d])
            if local_n != global_n:
                raise ValueError(
                    f"Dask blocks have multiple OM chunks in dimension {first_multi}, "
                    f"but dimension {d} is not fully covered by each dask block "
                    f"(dask chunk {data_chunks[d][0]} vs array size {array_shape[d]}). "
                    f"Rechunk so trailing dimensions are fully covered."
                )


def _dask_block_iterator(dask_array: da.Array) -> Iterator[np.ndarray]:
    """
    Yield computed numpy arrays from a dask array in C-order block traversal.

    The OM file format requires chunks to be written in sequential order
    corresponding to a row-major (C-order) traversal of the chunk grid.
    ndindex does this: the last dimension is iterated over first.
    """
    for block_indices in np.ndindex(*dask_array.numblocks):
        yield dask_array.blocks[block_indices].compute()


def write_dask_array(
    writer: OmFileWriter,
    data: da.Array,
    chunks: Optional[Sequence[int]] = None,
    scale_factor: float = 1.0,
    add_offset: float = 0.0,
    compression: str = "pfor_delta_2d",
    name: str = "data",
    children: Optional[Sequence[OmVariable]] = None,
) -> OmVariable:
    """
    Write a dask array to an OM file using streaming/incremental writes.

    Iterates over the blocks of the dask array, computing each block
    on-the-fly, and streams them to the OM file writer. Only one block
    is held in memory at a time.

    The dask array's chunk structure is used to determine the OM file's
    chunk dimensions by default. Dask chunks must be multiples of the OM
    chunk sizes (except the last chunk along each dimension which may be
    smaller). When a dask block contains more than one OM chunk in a
    dimension, all trailing dimensions must be fully covered by each block.

    Performance: write speed depends on the number of dask tasks, not just
    data size. For best performance, use dask chunks much larger than the
    OM chunk sizes — ideally covering the full extent of trailing dimensions.
    For example, with OM chunks of (124, 124) on an (8192, 8192) array,
    dask chunks of (124, 8192) will write ~10x faster than (124, 124).

    Args:
        writer: An open OmFileWriter instance.
        data: A dask array to write.
        chunks: OM file chunk sizes per dimension. If None, uses the dask
                array's chunk sizes. Dask chunks must be multiples of these.
        scale_factor: Scale factor for float compression (default: 1.0).
        add_offset: Offset for float compression (default: 0.0).
        compression: Compression algorithm (default: "pfor_delta_2d").
        name: Variable name (default: "data").
        children: Child variables (default: None).

    Returns:
        OmVariable representing the written array.

    Raises:
        TypeError: If data is not a dask array.
        ValueError: If dask chunks are incompatible with OM chunks.
    """
    if not isinstance(data, da.Array):
        raise TypeError(f"Expected a dask array, got {type(data)}")

    om_chunks = list(chunks) if chunks is not None else [int(c[0]) for c in data.chunks]
    _validate_chunk_alignment(data.chunks, om_chunks, data.shape)

    return writer.write_array_streaming(
        dimensions=[int(d) for d in data.shape],
        chunks=om_chunks,
        chunk_iterator=_dask_block_iterator(data),
        dtype=data.dtype,
        scale_factor=scale_factor,
        add_offset=add_offset,
        compression=compression,
        name=name,
        children=list(children) if children is not None else [],
    )
