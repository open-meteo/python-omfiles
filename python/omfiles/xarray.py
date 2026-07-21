"""OmFileReader backend for Xarray."""
# ruff: noqa: D101, D102, D105, D107

from __future__ import annotations

import numpy as np

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

from ._rust import OmFileReader, OmVariable

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

    def __init__(self, root_variable: OmFileReader):
        self.root_variable = root_variable
        self.variables_store = self.root_variable._get_flat_variable_metadata()
        self._children_by_parent: dict[str, dict[str, OmVariable]] = {}
        self._attributes_by_path: dict[str, dict] = {}
        self._known_arrays: dict[str, OmVariable] | None = None

        for path, variable in self.variables_store.items():
            parent_path, _, child_name = path.rpartition("/")
            self._children_by_parent.setdefault(parent_path, {})[child_name] = variable

    def get_variables(self):
        datasets = self._get_datasets(self.root_variable)
        # Remove all leading slashes from keys
        datasets_no_leading_slash = {(k.lstrip("/")): v for k, v in datasets.items()}
        return FrozenDict(datasets_no_leading_slash)

    def get_attrs(self):
        # Global attributes are attributes directly under the root variable.
        return FrozenDict(self._get_attributes_for_variable(self.root_variable, f"/{self.root_variable.name}"))

    def _get_attributes_for_variable(self, reader: OmFileReader, path: str):
        cached = self._attributes_by_path.get(path)
        if cached is not None:
            return cached

        attrs = {}
        direct_children = self._find_direct_children_in_store(path)
        for k, variable in direct_children.items():
            child_reader = reader._init_from_variable(variable)
            if child_reader.is_scalar:
                attrs[k] = child_reader.read_scalar()

        self._attributes_by_path[path] = attrs
        return attrs

    def _find_direct_children_in_store(self, path: str):
        return self._children_by_parent.get(path, {})

    def _get_known_arrays(self):
        if self._known_arrays is not None:
            return self._known_arrays

        arrays = {}
        for var_key, variable in self.variables_store.items():
            if self.root_variable._init_from_variable(variable).is_array:
                arrays[var_key] = variable

        self._known_arrays = arrays
        return arrays

    def _get_known_dimensions(self, arrays: dict[str, OmVariable]):
        """
        Get a set of all dimension names used in the dataset.

        This scans all array variables for their dimension metadata.
        """
        dimensions = set()

        # Scan all array variables for dimension names.
        for var_key, variable in arrays.items():
            reader = self.root_variable._init_from_variable(variable)
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
        arrays = self._get_known_arrays()
        known_dimensions = self._get_known_dimensions(arrays)

        for var_key, variable in arrays.items():
            child_reader = reader._init_from_variable(variable)
            backend_array = OmBackendArray(reader=child_reader)
            shape = backend_array.reader.shape

            # Get attributes to check for dimension information.
            attrs = self._get_attributes_for_variable(child_reader, var_key)
            attrs_for_var = {attr_k: attr_v for attr_k, attr_v in attrs.items() if attr_k != DIMENSION_KEY}

            # Look for dimension names in the dimension metadata.
            if DIMENSION_KEY in attrs:
                dim_names = attrs[DIMENSION_KEY]
                if isinstance(dim_names, str):
                    # With no explicit separator, split() treats consecutive whitespace as one separator and
                    # does not produce empty names for leading or trailing whitespace.
                    dim_names = dim_names.split()
            else:
                # Default to generic dimension names if not specified.
                dim_names = [f"dim{i}" for i in range(len(shape))]

            # Check if this variable is itself a dimension variable.
            variable_name = var_key.split("/")[-1]
            if len(shape) == 1 and variable_name in known_dimensions:
                dim_names = [variable_name]

            data = indexing.LazilyIndexedArray(backend_array)
            datasets[var_key] = Variable(dims=dim_names, data=data, attrs=attrs_for_var, encoding=None, fastpath=True)
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
