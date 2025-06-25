import numpy as np

from omfiles.types import ArrayType

EPOCH = np.datetime64(0, "s")


def _modulo_positive(value: int, modulo: int) -> int:
    """
    Calculate modulo that always returns positive value.

    Parameters:
    -----------
    value : int
        Value to calculate modulo for
    modulo : int
        Modulo value

    Returns:
    --------
    int
        Positive modulo result
    """
    return ((value % modulo) + modulo) % modulo


def _normalize_longitude(lon: ArrayType | float) -> ArrayType | float:
    return ((lon + 180.0) % 360.0) - 180.0
