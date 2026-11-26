"""OmFileReader backend for Xarray."""
# ruff: noqa: D101, D102, D105, D107

import itertools
import os
import warnings
from typing import Any, Generator

import numpy as np

from omfiles._chunk_utils import _validate_chunk_alignment

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

# Special metadata child used to declare array dimension names.
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
            return store_entrypoint.open_dataset(
                store,
                drop_variables=drop_variables,
            )
        raise ValueError("Failed to open dataset")

    description = "Use .om files in Xarray"

    url = "https://github.com/open-meteo/om-file-format/"


class OmDataStore(AbstractDataStore):
    root_variable: OmFileReader
    variables_store: dict[str, OmVariable]
    _direct_children_store: dict[str, dict[str, OmVariable]]
    _attributes_store: dict[str, dict[str, Any]]
    _known_arrays_store: dict[str, OmVariable] | None
    _dimension_declarations_store: dict[str, tuple[str, ...] | None]

    def __init__(self, root_variable: OmFileReader):
        self.root_variable = root_variable
        self.variables_store = self.root_variable._get_flat_variable_metadata()
        self._direct_children_store = {}
        for var_key, variable in self.variables_store.items():
            parent_path, _, child_name = var_key.rpartition("/")
            self._direct_children_store.setdefault(parent_path, {})[child_name] = variable
        self._attributes_store = {}
        self._known_arrays_store = None
        self._dimension_declarations_store = {}

    def get_variables(self):
        datasets = self._get_datasets(self.root_variable)
        # Remove all leading slashes from keys
        datasets_no_leading_slash = {(k.lstrip("/")): v for k, v in datasets.items()}
        return FrozenDict(datasets_no_leading_slash)

    def get_attrs(self):
        # Global attributes are attributes directly under the root variable.
        return FrozenDict(self._get_attributes_for_variable(self.root_variable, f"/{self.root_variable.name}"))

    def _get_attributes_for_variable(self, reader: OmFileReader, path: str):
        if path in self._attributes_store:
            return self._attributes_store[path]

        attrs = {}
        direct_children = self._find_direct_children_in_store(path)
        for k, variable in direct_children.items():
            if k == DIMENSION_KEY:
                continue
            child_reader = reader._init_from_variable(variable)
            if child_reader.is_scalar:
                # Skip scalars that have dimension metadata — they are 0-d coordinate variables,
                # not plain attributes.
                dim_key = path + "/" + k + "/" + DIMENSION_KEY
                if dim_key in self.variables_store:
                    continue
                attrs[k] = child_reader.read_scalar()
        self._attributes_store[path] = attrs
        return attrs

    def _find_direct_children_in_store(self, path: str):
        return self._direct_children_store.get(path, {})

    def _get_known_arrays(self):
        if self._known_arrays_store is not None:
            return self._known_arrays_store

        arrays = {}
        for var_key, var in self.variables_store.items():
            reader = self.root_variable._init_from_variable(var)
            if reader.is_array:
                arrays[var_key] = var
        self._known_arrays_store = arrays
        return arrays

    @staticmethod
    def _display_path(path: str) -> str:
        return "/" + path.lstrip("/")

    def _get_dimension_declaration(self, path: str) -> tuple[str, ...] | None:
        if path in self._dimension_declarations_store:
            return self._dimension_declarations_store[path]

        dimension_variable = self._find_direct_children_in_store(path).get(DIMENSION_KEY)
        if dimension_variable is None:
            self._dimension_declarations_store[path] = None
            return None

        dimension_path = f"{self._display_path(path).rstrip('/')}/{DIMENSION_KEY}"
        dimension_reader = self.root_variable._init_from_variable(dimension_variable)
        if not dimension_reader.is_scalar:
            raise ValueError(f"Invalid dimension metadata at '{dimension_path}': expected a scalar string.")

        value = dimension_reader.read_scalar()
        if not isinstance(value, str):
            raise ValueError(
                f"Invalid dimension metadata at '{dimension_path}': expected a string, got {type(value).__name__}."
            )

        declaration = tuple(value.split())
        self._dimension_declarations_store[path] = declaration
        return declaration

    def _get_parent_reader(self, path: str) -> OmFileReader | None:
        if path == f"/{self.root_variable.name}":
            return self.root_variable

        variable = self.variables_store.get(path)
        if variable is None:
            return None
        return self.root_variable._init_from_variable(variable)

    def _validate_array_dimensions(self, path: str, declaration: tuple[str, ...], ndim: int) -> None:
        if len(declaration) == ndim:
            return

        dimension_path = f"{self._display_path(path).rstrip('/')}/{DIMENSION_KEY}"
        raise ValueError(
            f"Invalid dimension metadata at '{dimension_path}' for array '{self._display_path(path)}': "
            f"declared {len(declaration)} dimension(s) {declaration}, but the array has {ndim}."
        )

    def _get_datasets(self, reader: OmFileReader):
        datasets = {}
        known_arrays = self._get_known_arrays()
        arrays = []
        sibling_dimensions: dict[str, set[str]] = {}

        for var_key, variable in known_arrays.items():
            child_reader = reader._init_from_variable(variable)
            backend_array = OmBackendArray(reader=child_reader)
            shape = backend_array.reader.shape
            attrs = self._get_attributes_for_variable(child_reader, var_key)
            declaration = self._get_dimension_declaration(var_key)
            if declaration is not None:
                self._validate_array_dimensions(var_key, declaration, len(shape))
                parent_path = var_key.rpartition("/")[0]
                sibling_dimensions.setdefault(parent_path, set()).update(declaration)
            arrays.append((var_key, backend_array, shape, attrs, declaration))

        parent_declarations: dict[str, tuple[str, ...] | None] = {}
        for var_key, backend_array, shape, attrs, declaration in arrays:
            if declaration is not None:
                dim_names = declaration
            else:
                parent_path, _, variable_name = var_key.rpartition("/")
                if parent_path not in parent_declarations:
                    parent_declarations[parent_path] = self._get_dimension_declaration(parent_path)
                parent_declaration = parent_declarations[parent_path]
                known_parent_dimensions = sibling_dimensions.get(parent_path, set())
                if parent_declaration is not None:
                    known_parent_dimensions = known_parent_dimensions.union(parent_declaration)

                parent_reader = self._get_parent_reader(parent_path)
                if len(shape) == 1 and variable_name in known_parent_dimensions:
                    dim_names = (variable_name,)
                elif (
                    parent_reader is not None
                    and parent_reader.is_group
                    and parent_declaration is not None
                    and len(parent_declaration) == len(shape)
                ):
                    dim_names = parent_declaration
                else:
                    dim_names = tuple(f"dim{i}" for i in range(len(shape)))

            data = indexing.LazilyIndexedArray(backend_array)
            datasets[var_key] = Variable(dims=dim_names, data=data, attrs=attrs, encoding=None, fastpath=True)

        # Handle 0-d (scalar) variables that have dimension metadata.
        for var_key, var in self.variables_store.items():
            if var_key in datasets:
                continue
            child_reader = reader._init_from_variable(var)
            if not child_reader.is_scalar:
                continue
            declaration = self._get_dimension_declaration(var_key)
            if declaration is None:
                continue
            self._validate_array_dimensions(var_key, declaration, 0)
            scalar_value = child_reader.read_scalar()
            attrs = self._get_attributes_for_variable(child_reader, var_key)
            datasets[var_key] = Variable(dims=(), data=np.array(scalar_value), attrs=attrs)

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


def _validate_om_name(name: Any, description: str) -> None:
    """Validate a name that will become an OM hierarchy child or dimension token."""
    if not isinstance(name, str):
        raise ValueError(f"{description} must be a string, got {type(name).__name__}.")
    if not name:
        raise ValueError(f"{description} must not be empty.")
    if "/" in name:
        raise ValueError(f"{description} '{name}' must not contain '/'.")
    if any(character.isspace() for character in name):
        raise ValueError(f"{description} '{name}' must not contain whitespace.")


def _validate_dataset_for_writing(ds: Dataset) -> None:
    """Validate that a dataset can be represented without extending the Open-Meteo convention."""
    variable_names = set(ds.variables)

    for dimension_name in ds.dims:
        _validate_om_name(dimension_name, "Dimension name")

    for variable_name, variable in ds.variables.items():
        _validate_om_name(variable_name, "Variable name")
        if DIMENSION_KEY in variable.attrs:
            raise ValueError(
                f"Variable '{variable_name}' attribute '{DIMENSION_KEY}' conflicts with OM dimension metadata."
            )
        for attribute_name in variable.attrs:
            _validate_om_name(attribute_name, f"Attribute name on variable '{variable_name}'")

        if np.issubdtype(variable.dtype, np.datetime64) or np.issubdtype(variable.dtype, np.timedelta64):
            raise TypeError(
                f"Variable '{variable_name}' has dtype {variable.dtype}. "
                "OM files do not support datetime64/timedelta64 natively. "
                "Convert to a numeric type before writing."
            )

    for coordinate_name, coordinate in ds.coords.items():
        if coordinate.ndim != 1 or coordinate.dims != (coordinate_name,):
            raise ValueError(
                f"Coordinate '{coordinate_name}' with dimensions {coordinate.dims} is not supported. "
                "OM dataset writing supports only one-dimensional dimension coordinates."
            )

    for attribute_name in ds.attrs:
        _validate_om_name(attribute_name, "Global attribute name")
        if attribute_name == DIMENSION_KEY:
            raise ValueError(f"Global attribute '{DIMENSION_KEY}' conflicts with OM dimension metadata.")
        if attribute_name in variable_names:
            raise ValueError(f"Global attribute '{attribute_name}' conflicts with a dataset variable of the same name.")


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

    Only one-dimensional dimension coordinates are supported. Auxiliary and
    scalar coordinates cannot be represented by the Open-Meteo coordinate
    convention and are rejected before the output file is created.

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
    _validate_dataset_for_writing(ds)
    path = str(path)
    if fs is not None:
        writer = OmFileWriter.from_fsspec(fs, path)
    else:
        writer = OmFileWriter(path)
    all_children: list[OmVariable] = []

    def _write_variable(name: str, var: Variable, is_dim_coord: bool) -> None:
        """Write a data variable or dimension coordinate."""
        dim_var = writer.write_scalar(" ".join(var.dims), name=DIMENSION_KEY)
        var_children: list[OmVariable] = [dim_var]

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

    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        _write_variable(coord_name, coord.variable, is_dim_coord=True)

    for attr_name, attr_value in ds.attrs.items():
        scalar = _write_scalar_safe(writer, attr_value, attr_name)
        if scalar is not None:
            all_children.append(scalar)

    root_var = writer.write_group(name="", children=all_children)
    writer.close(root_var)
