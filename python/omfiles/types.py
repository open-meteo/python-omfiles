import numpy as np
import numpy.typing as npt

try:
    from types import EllipsisType
except ImportError:
    EllipsisType = type(Ellipsis)
from typing import Tuple, Union

# This is from https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/core/indexing.py#L38C1-L40C87

BasicSelector = Union[int, slice, EllipsisType]
"""A single index selector for an array dimension: integer, slice, or ellipsis."""
BasicSelection = Union[BasicSelector, Tuple[Union[int, slice, EllipsisType], ...]]
"""A selection for an array: either a single selector or a tuple of selectors (also used for BlockIndex)."""

# Type aliases for grids for clarity
FloatType = Union[float, np.floating]
ArrayType = npt.NDArray[np.floating]
CoordType = Union[float, ArrayType]
ReturnUnionType = Union[tuple[ArrayType, ArrayType], tuple[float, float]]
