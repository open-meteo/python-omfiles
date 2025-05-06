import os
import tempfile

import numpy as np
import numpy.typing as npt
import pytest

from .test_utils import create_test_om_file


@pytest.fixture
def temp_om_file():
    """
    Fixture that creates a temporary OM file filled with some data.
    Returns a path to the temporary file.
    """

    dtype: npt.DTypeLike = np.float32
    shape: tuple = (5, 5)

    # On Windows a file cannot be opened twice, so we need to close it first
    # and take care of deleting it ourselves
    with tempfile.NamedTemporaryFile(suffix=".om", delete=False) as temp_file:
        create_test_om_file(temp_file.name, shape=shape, dtype=dtype)
        temp_file.close()
        filename = temp_file.name

    yield filename

    if os.path.exists(filename):
        try:
            os.remove(filename)
        except (PermissionError, OSError) as e:
            import warnings
            warnings.warn(f"Failed to remove temporary file {filename}: {e}")

@pytest.fixture
def empty_temp_om_file():
    """
    Fixture that creates a temporary empty OM file.
    Returns a path to the temporary file.
    """

    # On Windows a file cannot be opened twice, so we need to close it first
    # and take care of deleting it ourselves
    with tempfile.NamedTemporaryFile(suffix=".om", delete=False) as temp_file:
        temp_file.close()
        filename = temp_file.name

    yield filename

    if os.path.exists(filename):
        try:
            os.remove(filename)
        except (PermissionError, OSError) as e:
            import warnings
            warnings.warn(f"Failed to remove temporary file {filename}: {e}")
