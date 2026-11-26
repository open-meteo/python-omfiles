import math


def _validate_chunk_alignment(
    data_chunks: tuple,
    om_chunks: list[int],
    array_shape: tuple,
) -> None:
    """
    Validate chunked array blocks for compatibility with OM chunk streaming.

    Every non-last data chunk along each dimension must be an exact multiple
    of the corresponding OM chunk size (the last chunk may be smaller).
    Additionally, for the leftmost dimension where a data block contains more
    than one OM chunk, every trailing dimension must be fully covered by each
    data block. This ensures the local chunk traversal inside a block matches
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
        local_n = max(math.ceil(c / om_chunks[d]) for c in data_chunks[d])
        if local_n > 1:
            first_multi = d
            break

    if first_multi is not None:
        for d in range(first_multi + 1, ndim):
            dim_chunks = data_chunks[d]
            if not (len(dim_chunks) == 1 and dim_chunks[0] == array_shape[d]):
                raise ValueError(
                    f"Dask blocks have multiple OM chunks in dimension {first_multi}, "
                    f"but dimension {d} is not fully covered by each dask block "
                    f"(dask chunks {dim_chunks} vs array size {array_shape[d]}). "
                    f"Rechunk so trailing dimensions are fully covered."
                )
