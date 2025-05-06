from __future__ import annotations

import numpy as np
from xarray.backends.common import BackendArray, BackendEntrypoint, WritableCFDataStore, _normalize_path
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.dataset import Dataset
from xarray.core.utils import FrozenDict
from xarray.core.variable import Variable

from .omfiles import OmFilePyReader, OmVariable

# need some special secret attributes to tell us the dimensions
DIMENSION_KEY = "_ARRAY_DIMENSIONS"

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
        with OmFilePyReader(filename_or_obj) as root_variable:
            store = OmDataStore(root_variable)
            store_entrypoint = StoreBackendEntrypoint()
            return store_entrypoint.open_dataset(
                store,
                drop_variables=drop_variables,
            )
        raise ValueError("Failed to open dataset")

    description = "Use .om files in Xarray"

    url = "https://github.com/open-meteo/om-file-format/"


class OmDataStore(WritableCFDataStore):
    root_variable: OmFilePyReader
    variables_store: dict[str, OmVariable]

    def __init__(self, root_variable: OmFilePyReader):
        self.root_variable = root_variable
        self.variables_store = self.root_variable.get_flat_variable_metadata()

    def get_variables(self):
        return FrozenDict(self._get_datasets_for_variable(self.root_variable))

    def get_attrs(self):
        # Global attributes are attributes directly under the root variable.
        return FrozenDict(self._get_attributes_for_variable(self.root_variable, self.root_variable.name))

    def _get_attributes_for_variable(self, reader: OmFilePyReader, path: str):
        attrs = {}
        direct_children = self._find_direct_children_in_store(path)
        for k, variable in direct_children.items():
            child_reader = reader.init_from_variable(variable)
            if child_reader.is_scalar:
                attrs[k] = child_reader.get_scalar()
        return attrs

    def _find_direct_children_in_store(self, path: str):
        prefix = "" if path == "" or path is None else path + "/"

        return {
            key[len(prefix):]: variable
            for key, variable in self.variables_store.items()
            if key.startswith(prefix) and key != path and "/" not in key[len(prefix):]
        }

    def _is_group(self, variable):
        return self.root_variable.init_from_variable(variable).is_group

    def _get_known_dimensions(self):
        """
        Get a set of all dimension names used in the dataset.
        This scans all variables for their _ARRAY_DIMENSIONS attribute.
        """
        dimensions = set()

        # Scan all variables for dimension names
        for var_key in self.variables_store:
            var = self.variables_store[var_key]
            reader = self.root_variable.init_from_variable(var)
            if reader is None or reader.is_group or reader.is_scalar:
                continue

            attrs = self._get_attributes_for_variable(reader, var_key)
            if DIMENSION_KEY in attrs:
                dim_names = attrs[DIMENSION_KEY]
                if isinstance(dim_names, str):
                    dimensions.update(dim_names.split(','))
                elif isinstance(dim_names, list):
                    dimensions.update(dim_names)

        return dimensions

    def _get_datasets_for_variable(self, reader: OmFilePyReader):
        datasets = {}
        direct_children = self._find_direct_children_in_store("")

        for k, variable in direct_children.items():
            child_reader = reader.init_from_variable(variable)
            if not (child_reader.is_scalar or child_reader.is_group):
                # This is an array variable
                backend_array = OmBackendArray(reader=child_reader)
                shape = backend_array.reader.shape

                # Get attributes to check for dimension information
                attrs = self._get_attributes_for_variable(child_reader, k)
                attrs_for_var = {attr_k: attr_v for attr_k, attr_v in attrs.items() if attr_k != DIMENSION_KEY}

                # Look for dimension names in the _ARRAY_DIMENSIONS attribute
                if DIMENSION_KEY in attrs:
                    dim_names = attrs[DIMENSION_KEY]
                    if isinstance(dim_names, str):
                        # Dimensions are stored as a comma-separated string, split them
                        dim_names = dim_names.split(',')
                else:
                    # Default to generic dimension names if not specified
                    dim_names = [f"dim{i}" for i in range(len(shape))]

                # Check if this variable is itself a dimension variable
                variable_name = k.split('/')[-1]  # Get the actual name without parent path

                # If this variable is a 1D array and its name matches a dimension name, use its own name
                if len(shape) == 1 and variable_name in self._get_known_dimensions():
                    dim_names = [variable_name]

                data = indexing.LazilyIndexedArray(backend_array)
                datasets[k] = Variable(dims=dim_names, data=data, attrs=attrs_for_var, encoding=None, fastpath=True)

        return datasets

    def close(self):
        self.root_variable.close()


class OmBackendArray(BackendArray):
    def __init__(self, reader: OmFilePyReader):
        self.reader = reader

    @property
    def shape(self):
        return self.reader.shape

    @property
    def dtype(self):
        return self.reader.dtype

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self.reader.__getitem__,
        )
