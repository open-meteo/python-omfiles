"""OmFileReader backend for Xarray."""
# ruff: noqa: D101, D102, D105, D107

import itertools
import os
import warnings
from typing import Any, Generator

import numpy as np

from omfiles.dask import _validate_chunk_alignment

try:
    from xarray.core import indexing
except ImportError:
    raise ImportError("omfiles[xarray] is required for Xarray functionality")

from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    _normalize_path,
)
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.dataset import Dataset
from xarray.core.utils import FrozenDict
from xarray.core.variable import Variable

from ._rust import OmFileReader, OmFileWriter, OmVariable

# need some special secret attributes to tell us the dimensions
DIMENSION_KEY = "coordinates"


class OmXarrayEntrypoint(BackendEntrypoint):
    def guess_can_open(self, filename_or_obj):
        return isinstance(filename_or_obj, str) and filename_or_obj.endswith(".om")

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
    ) -> Dataset:
        filename_or_obj = _normalize_path(filename_or_obj)
        with OmFileReader(filename_or_obj) as root_variable:
            store = OmDataStore(root_variable)
            store_entrypoint = StoreBackendEntrypoint()
            ds = store_entrypoint.open_dataset(
                store,
                drop_variables=drop_variables,
            )
            coord_attr = "_COORDINATE_VARIABLES"
            if coord_attr in ds.attrs:
                coord_names = [c for c in ds.attrs[coord_attr].split(",") if c in ds]
                ds = ds.set_coords(coord_names)
                ds.attrs = {k: v for k, v in ds.attrs.items() if k != coord_attr}
            return ds
        raise ValueError("Failed to open dataset")

    description = "Use .om files in Xarray"

    url = "https://github.com/open-meteo/om-file-format/"


class OmDataStore(AbstractDataStore):
    root_variable: OmFileReader
    variables_store: dict[str, OmVariable]

    def __init__(self, root_variable: OmFileReader):
        self.root_variable = root_variable
        self.variables_store = self.root_variable._get_flat_variable_metadata()

    def get_variables(self):
        datasets = self._get_datasets(self.root_variable)
        # Remove all leading slashes from keys
        datasets_no_leading_slash = {(k.lstrip("/")): v for k, v in datasets.items()}
        return FrozenDict(datasets_no_leading_slash)

    def get_attrs(self):
        # Global attributes are attributes directly under the root variable.
        return FrozenDict(self._get_attributes_for_variable(self.root_variable, f"/{self.root_variable.name}"))

    def _get_attributes_for_variable(self, reader: OmFileReader, path: str):
        attrs = {}
        direct_children = self._find_direct_children_in_store(path)
        for k, variable in direct_children.items():
            child_reader = reader._init_from_variable(variable)
            if child_reader.is_scalar:
                # Skip scalars that have _ARRAY_DIMENSIONS — they are 0-d
                # coordinate variables, not plain attributes.
                dim_key = path + "/" + k + "/" + DIMENSION_KEY
                if dim_key in self.variables_store:
                    continue
                attrs[k] = child_reader.read_scalar()
        return attrs

    def _find_direct_children_in_store(self, path: str):
        prefix = path + "/"

        return {
            key[len(prefix) :]: variable
            for key, variable in self.variables_store.items()
            if key.startswith(prefix) and key != path and "/" not in key[len(prefix) :]
        }

    def _is_group(self, variable):
        return self.root_variable._init_from_variable(variable).is_group

    def _get_known_arrays(self):
        arrays = {}
        for var_key, var in self.variables_store.items():
            reader = self.root_variable._init_from_variable(var)
            if reader.is_array:
                arrays[var_key] = var
        return arrays

    def _get_known_dimensions(self):
        """
        Get a set of all dimension names used in the dataset.

        This scans all variables for their _ARRAY_DIMENSIONS attribute.
        """
        dimensions = set()

        # Scan all variables for dimension names
        for var_key in self.variables_store:
            var = self.variables_store[var_key]
            reader = self.root_variable._init_from_variable(var)
            if reader is None or reader.is_group or reader.is_scalar:
                continue

            attrs = self._get_attributes_for_variable(reader, var_key)
            if DIMENSION_KEY in attrs:
                dim_names = attrs[DIMENSION_KEY]
                if isinstance(dim_names, str):
                    dimensions.update(dim_names.split())
                elif isinstance(dim_names, list):
                    dimensions.update(dim_names)

        return dimensions

    def _get_datasets(self, reader: OmFileReader):
        datasets = {}

        for var_key, variable in self._get_known_arrays().items():
            child_reader = reader._init_from_variable(variable)
            backend_array = OmBackendArray(reader=child_reader)
            shape = backend_array.reader.shape

            # Get attributes to check for dimension information
            attrs = self._get_attributes_for_variable(child_reader, var_key)
            attrs_for_var = {attr_k: attr_v for attr_k, attr_v in attrs.items() if attr_k != DIMENSION_KEY}

            # Look for dimension names in the _ARRAY_DIMENSIONS attribute
            if DIMENSION_KEY in attrs:
                dim_names = attrs[DIMENSION_KEY]
                if isinstance(dim_names, str):
                    # Dimensions are stored as white space separated string, split them
                    #
                    # If sep is not specified or is None, a different splitting algorithm is applied:
                    # runs of consecutive whitespace are regarded as a single separator, and the result will
                    # contain no empty strings at the start or end if the string has leading or trailing
                    # whitespace. Consequently, splitting an empty string or a string consisting of just
                    # whitespace with a None separator returns [].
                    dim_names = dim_names.split()
            else:
                # Default to generic dimension names if not specified
                dim_names = [f"dim{i}" for i in range(len(shape))]

            # Check if this variable is itself a dimension variable
            variable_name = var_key.split("/")[-1]
            if len(shape) == 1 and variable_name in self._get_known_dimensions():
                dim_names = [variable_name]

            data = indexing.LazilyIndexedArray(backend_array)
            datasets[var_key] = Variable(dims=dim_names, data=data, attrs=attrs_for_var, encoding=None, fastpath=True)

        # Handle 0-d (scalar) variables that have _ARRAY_DIMENSIONS metadata.
        # These are scalar coordinates written by write_dataset.
        for var_key, var in self.variables_store.items():
            if var_key in datasets:
                continue
            child_reader = reader._init_from_variable(var)
            if not child_reader.is_scalar:
                continue
            dim_path = var_key + "/" + DIMENSION_KEY
            if dim_path not in self.variables_store:
                continue
            dim_reader = reader._init_from_variable(self.variables_store[dim_path])
            dim_names_str = dim_reader.read_scalar()
            if isinstance(dim_names_str, str) and dim_names_str == "":
                dim_names = ()
            elif isinstance(dim_names_str, str):
                dim_names = tuple(dim_names_str.split(","))
            else:
                dim_names = ()
            scalar_value = child_reader.read_scalar()
            attrs = self._get_attributes_for_variable(child_reader, var_key)
            attrs_for_var = {k: v for k, v in attrs.items() if k != DIMENSION_KEY}
            datasets[var_key] = Variable(dims=dim_names, data=np.array(scalar_value))

        return datasets

    def close(self):
        self.root_variable.close()


class OmBackendArray(BackendArray):
    """OmBackendArray is an xarray backend implementation for the OmFileReader."""

    def __init__(self, reader: OmFileReader):
        self.reader = reader

    @property
    def shape(self):
        return self.reader.shape

    @property
    def dtype(self):
        return self.reader.dtype

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        """Retrieve data from the OmFileReader using the provided key."""
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self.reader.__getitem__,
        )


def _write_scalar_safe(writer: OmFileWriter, value: Any, name: str) -> OmVariable | None:
    """Write a scalar, returning None and warning if the type is unsupported."""
    try:
        return writer.write_scalar(value, name=name)
    except (ValueError, TypeError) as e:
        warnings.warn(
            f"Skipping attribute '{name}' with value {value!r}: {e}",
            UserWarning,
            stacklevel=3,
        )
        return None


def _chunked_block_iterator(data: Any) -> Generator[np.ndarray, None, None]:
    """
    Yield numpy arrays from a chunked array in C-order block traversal.

    Works with any array that exposes ``.numblocks``, ``.blocks[idx]``,
    and ``.compute()`` (e.g. dask arrays).  No dask import required.
    """
    block_index_ranges = [range(n) for n in data.numblocks]
    for block_indices in itertools.product(*block_index_ranges):
        block = data.blocks[block_indices]
        if hasattr(block, "compute"):
            yield block.compute()
        else:
            yield np.asarray(block)


def _resolve_chunks_for_variable(
    var_name: str,
    var: Variable,
    encoding: dict[str, dict[str, Any]] | None,
    global_chunks: dict[str, int] | None,
    data_chunks: tuple | None = None,
) -> list[int]:
    """Resolve chunk sizes for a variable using the priority chain."""
    if encoding and var_name in encoding and "chunks" in encoding[var_name]:
        return list(encoding[var_name]["chunks"])

    if global_chunks is not None:
        return [global_chunks.get(dim, min(size, 512)) for dim, size in zip(var.dims, var.shape)]

    if data_chunks is not None:
        return [int(c[0]) for c in data_chunks]

    return [min(size, 512) for size in var.shape]


def _resolve_encoding_for_variable(
    var_name: str,
    encoding: dict[str, dict[str, Any]] | None,
    global_scale_factor: float,
    global_add_offset: float,
    global_compression: str,
) -> tuple[float, float, str]:
    """Resolve compression parameters for a variable."""
    var_enc = (encoding or {}).get(var_name, {})
    sf = var_enc.get("scale_factor", global_scale_factor)
    ao = var_enc.get("add_offset", global_add_offset)
    comp = var_enc.get("compression", global_compression)
    return sf, ao, comp


def write_dataset(
    ds: Dataset,
    path: str | os.PathLike,
    *,
    fs: Any | None = None,
    encoding: dict[str, dict[str, Any]] | None = None,
    chunks: dict[str, int] | None = None,
    scale_factor: float = 1.0,
    add_offset: float = 0.0,
    compression: str = "pfor_delta_2d",
) -> None:
    """
    Write an xarray Dataset to an OM file.

    The resulting file can be read back with ``xr.open_dataset(path, engine="om")``.

    Args:
        ds: The xarray Dataset to write.
        path: Output file path (local path or path within the fsspec filesystem).
        fs: Optional fsspec filesystem object. When provided, the file is written
            via ``OmFileWriter.from_fsspec(fs, path)`` instead of the default
            local-file writer.
        encoding: Per-variable overrides. Keys per variable: ``"chunks"``,
            ``"scale_factor"``, ``"add_offset"``, ``"compression"``.
        chunks: Global default chunk sizes as ``{dim_name: chunk_size}``.
        scale_factor: Global default scale factor for float compression.
        add_offset: Global default offset for float compression.
        compression: Global default compression algorithm.
    """
    path = str(path)
    if fs is not None:
        writer = OmFileWriter.from_fsspec(fs, path)
    else:
        writer = OmFileWriter(path)
    all_children: list[OmVariable] = []

    def _write_variable(name: str, var: Variable, is_dim_coord: bool) -> None:
        """Write a single variable (data var or non-dimension coordinate)."""
        if np.issubdtype(var.dtype, np.datetime64) or np.issubdtype(var.dtype, np.timedelta64):
            raise TypeError(
                f"Variable '{name}' has dtype {var.dtype}. "
                "OM files do not support datetime64/timedelta64 natively. "
                "Convert to a numeric type before writing."
            )

        var_children: list[OmVariable] = []

        if not is_dim_coord:
            dim_str = ",".join(var.dims)
            dim_var = writer.write_scalar(dim_str, name=DIMENSION_KEY)
            var_children.append(dim_var)

            for attr_name, attr_value in var.attrs.items():
                scalar = _write_scalar_safe(writer, attr_value, attr_name)
                if scalar is not None:
                    var_children.append(scalar)

        if var.ndim == 0:
            om_var = writer.write_scalar(
                var.values[()],
                name=name,
                children=var_children if var_children else None,
            )
            all_children.append(om_var)
            return

        data = var.data
        is_chunked = not is_dim_coord and hasattr(data, "chunks") and data.chunks is not None

        if is_dim_coord:
            resolved_chunks = [var.shape[0]]
        else:
            resolved_chunks = _resolve_chunks_for_variable(
                name,
                var,
                encoding,
                chunks,
                data_chunks=data.chunks if is_chunked else None,
            )

        sf, ao, comp = _resolve_encoding_for_variable(name, encoding, scale_factor, add_offset, compression)

        if is_chunked:
            _validate_chunk_alignment(data.chunks, resolved_chunks, var.shape)
            om_var = writer.write_array_streaming(
                dimensions=[int(d) for d in var.shape],
                chunks=[int(c) for c in resolved_chunks],
                chunk_iterator=_chunked_block_iterator(data),
                dtype=var.dtype,
                scale_factor=sf,
                add_offset=ao,
                compression=comp,
                name=name,
                children=var_children if var_children else None,
            )
        else:
            om_var = writer.write_array(
                var.values,
                chunks=resolved_chunks,
                scale_factor=sf,
                add_offset=ao,
                compression=comp,
                name=name,
                children=var_children if var_children else None,
            )
        all_children.append(om_var)

    for var_name in ds.data_vars:
        _write_variable(var_name, ds[var_name].variable, is_dim_coord=False)

    non_dim_coords: list[str] = []
    for coord_name in ds.coords:
        if coord_name in ds.data_vars:
            continue
        coord = ds.coords[coord_name]
        is_dim_coord = coord.ndim == 1 and coord.dims[0] == coord_name
        if not is_dim_coord:
            non_dim_coords.append(coord_name)
        _write_variable(coord_name, coord.variable, is_dim_coord=is_dim_coord)

    # Write list of non-dimension coordinates so the reader can restore them
    if non_dim_coords:
        coord_list_var = writer.write_scalar(",".join(non_dim_coords), name="_COORDINATE_VARIABLES")
        all_children.append(coord_list_var)

    for attr_name, attr_value in ds.attrs.items():
        scalar = _write_scalar_safe(writer, attr_value, attr_name)
        if scalar is not None:
            all_children.append(scalar)

    root_var = writer.write_group(name="", children=all_children)
    writer.close(root_var)
