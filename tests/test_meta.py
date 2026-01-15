import numpy as np
import pytest
from omfiles.meta import OmChunksMeta


def test_meta_json_creation(icon_d2_meta_json: str):
    meta = OmChunksMeta.from_metajson_string(icon_d2_meta_json)
    assert meta.chunk_time_length == 121


def test_time_to_chunk_index(icond2_om_chunks_meta: OmChunksMeta):
    timestamp = np.datetime64("2023-01-01T12:00:00")
    chunk_index = icond2_om_chunks_meta.time_to_chunk_index(timestamp)
    assert chunk_index == 3839


def test_get_chunk_time_range(icond2_om_chunks_meta: OmChunksMeta):
    chunk_index = 1000
    time_range = icond2_om_chunks_meta.get_chunk_time_range(chunk_index)

    # Check that we get the expected number of time points
    assert len(time_range) == icond2_om_chunks_meta.chunk_time_length
    # Check that time points are evenly spaced
    time_diff = time_range[1] - time_range[0]
    assert time_diff == np.timedelta64(icond2_om_chunks_meta.temporal_resolution_seconds, "s")
