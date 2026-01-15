from typing import Tuple
from unittest.mock import MagicMock, Mock, patch

import fsspec
import numpy as np
import pytest
from omfiles.chunk_reader import OmChunkFileReader
from omfiles.meta import OmChunksMeta


@pytest.fixture
def mock_fs():
    return Mock()


@pytest.fixture
def date_range():
    start_date = np.datetime64("2024-01-01T00:00:00")
    end_date = np.datetime64("2024-01-31T23:00:00")
    return start_date, end_date


@pytest.fixture
def chunk_reader(
    icond2_om_chunks_meta: OmChunksMeta,
    mock_fs: fsspec.AbstractFileSystem,
    date_range: Tuple[np.datetime64, np.datetime64],
):
    start_date, end_date = date_range
    return OmChunkFileReader(
        chunks_meta=icond2_om_chunks_meta,
        fs=mock_fs,
        chunk_files_path="s3://bucket/path",
        start_date=start_date,
        end_date=end_date,
    )


def test_init_success(
    icond2_om_chunks_meta: OmChunksMeta,
    mock_fs: fsspec.AbstractFileSystem,
    date_range: Tuple[np.datetime64, np.datetime64],
):
    start_date, end_date = date_range
    reader = OmChunkFileReader(
        chunks_meta=icond2_om_chunks_meta,
        fs=mock_fs,
        chunk_files_path="s3://bucket/path",
        start_date=start_date,
        end_date=end_date,
    )

    assert reader.chunks_meta == icond2_om_chunks_meta
    assert reader.fs == mock_fs
    assert reader.chunk_files_path == "s3://bucket/path"
    assert reader.start_date == start_date
    assert reader.end_date == end_date
    assert reader.chunk_indices == [3912, 3913, 3914, 3915, 3916, 3917, 3918]


def test_init_invalid_date_range(icond2_om_chunks_meta: OmChunksMeta, mock_fs: fsspec.AbstractFileSystem):
    start_date = np.datetime64("2024-01-31")
    end_date = np.datetime64("2024-01-01")

    with pytest.raises(ValueError, match="start_date must be <= end_date"):
        OmChunkFileReader(
            chunks_meta=icond2_om_chunks_meta,
            fs=mock_fs,
            chunk_files_path="s3://bucket/path",
            start_date=start_date,
            end_date=end_date,
        )


def test_iter_files(chunk_reader: OmChunkFileReader):
    files = list(chunk_reader.iter_files())

    assert len(files) == 7
    assert files[0] == (3912, "s3://bucket/path/chunk_3912.om")
    assert files[1] == (3913, "s3://bucket/path/chunk_3913.om")
    assert files[-1] == (3918, "s3://bucket/path/chunk_3918.om")


def test_load_data_success(chunk_reader: OmChunkFileReader, icond2_om_chunks_meta: OmChunksMeta):
    mock_reader_instance = MagicMock()
    mock_reader_instance.__enter__ = Mock(return_value=mock_reader_instance)
    mock_reader_instance.__exit__ = Mock(return_value=False)

    # Calculate expected data length for each chunk
    # For Jan 1-31, 2024, we have chunks 3912-3917
    expected_data = []
    for chunk_idx in chunk_reader.chunk_indices:
        chunk_times = icond2_om_chunks_meta.get_chunk_time_range(chunk_idx)
        time_mask = (chunk_times >= chunk_reader.start_date) & (chunk_times <= chunk_reader.end_date)
        num_points = np.sum(time_mask)
        # Create mock data with the correct length
        expected_data.append(np.arange(num_points, dtype=np.float32))

    mock_reader_instance.__getitem__ = Mock(side_effect=expected_data)

    with patch("omfiles.chunk_reader.OmFileReader.from_fsspec", return_value=mock_reader_instance):
        times, data = chunk_reader.load_data((10, 20))

    # Times and data should have the same length
    assert len(times) == len(data)
    # Should have data for January for 31 days * 24 hours
    assert len(times) == 744
    assert data.dtype == np.float32
