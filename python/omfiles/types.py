"""Types used by omfiles."""

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
