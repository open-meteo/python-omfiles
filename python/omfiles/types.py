"""Types used throughout the library."""

try:
    from types import EllipsisType
except ImportError:
    EllipsisType = type(Ellipsis)
from typing import NamedTuple, Tuple, Union

# This is from https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/core/indexing.py#L38C1-L40C87

BasicSelector = Union[int, slice, EllipsisType]
BasicSelection = Union[BasicSelector, Tuple[Union[int, slice, EllipsisType], ...]]


class XYIndex(NamedTuple):
    """Represents a 2D index in a grid."""

    x: int
    y: int


class LatLon(NamedTuple):
    """Represents a latitude and longitude pair."""

    lat: float
    lon: float
