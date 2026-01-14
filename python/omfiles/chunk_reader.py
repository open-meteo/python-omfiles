"""Utility class to iterate over chunks of data."""

from typing import List, Tuple, Union

try:
    import fsspec
except ImportError:
    raise ImportError("omfiles[fsspec] is required for using the chunk reader.")

import numpy as np
import numpy.typing as npt

from omfiles import OmFileReader
from omfiles.meta import OmChunksMeta


class OmChunkFileReader:
    """Utility class to iterate over chunks of data."""

    def __init__(
        self,
        om_meta: OmChunksMeta,
        fs: "fsspec.AbstractFileSystem",
        s3_path_to_chunk_files: str,
        start_date: np.datetime64,
        end_date: np.datetime64,
    ) -> None:
        """
        Initialize the chunk reader.

        Args:
            om_meta (OmChunksMeta): Metadata for the OM files.
            fs (fsspec.AbstractFileSystem): Filesystem for accessing the OM files.
            s3_path_to_chunk_files (str): Path to the chunk files.
            start_date (np.datetime64): Start date of the data to load.
            end_date (np.datetime64): End date of the data to load.
        """
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")

        self.om_meta = om_meta
        self.fs = fs

        self.s3_path_to_chunk_files = s3_path_to_chunk_files
        self.start_date = start_date
        self.end_date = end_date
        self.chunk_indices = self.om_meta.chunks_for_date_range(start_date, end_date)

    def iter_files(self):
        """
        Iterate over chunk files.

        Yields:
            Tuple[int, str]: Chunk index and path to the chunk file.
        """
        for chunk_index in self.chunk_indices:
            yield chunk_index, f"{self.s3_path_to_chunk_files}/chunk_{chunk_index}.om"

    def load_chunked_data(
        self, spatial_index: Union[Tuple[int, int], Tuple[slice, slice]]
    ) -> Tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float32]]:
        """
        Load data from all chunks for a given spatial index.

        Args:
            spatial_index (Tuple[int, int]): Spatial index (x, y) of the data to load.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time array and data array.

        Raises:
            FileNotFoundError: If chunk file doesn't exist.
        """
        if not self.chunk_indices:
            return np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.float32)

        all_times: List[npt.NDArray[np.datetime64]] = []
        all_data: List[npt.NDArray[np.float32]] = []

        for chunk_index, s3_path in self.iter_files():
            times, data = self._load_chunk_data(chunk_index, s3_path, spatial_index)
            if len(times) > 0:
                all_times.append(times)
                all_data.append(data)

        time_array = np.concatenate(all_times)
        data_array = np.concatenate(all_data)
        return time_array, data_array

    def _load_chunk_data(
        self,
        chunk_index: int,
        s3_path: str,
        spatial_index: Union[Tuple[int, int], Tuple[slice, slice]],
    ) -> Tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float32]]:
        """
        Load data from a single chunk.

        Args:
            chunk_index (int): Index of the chunk.
            s3_path (str): Path to the chunk file.
            spatial_index (Tuple[int, int]): Spatial index (x, y).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time array and data array for this chunk.
        """
        chunk_times = self.om_meta.get_chunk_time_range(chunk_index)
        time_mask = (chunk_times >= self.start_date) & (chunk_times <= self.end_date)

        if not np.any(time_mask):
            return np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.float32)

        try:
            with OmFileReader.from_fsspec(self.fs, s3_path) as reader:
                indices = np.where(time_mask)[0]
                time_slice = slice(indices[0], indices[-1] + 1)  # +1 to include the end
                x, y = spatial_index
                data = reader[y, x, time_slice].astype(np.float32)
                times = chunk_times[time_mask]
                assert len(times) == len(data), f"Expected {len(times)} timestamps but got {len(data)}"
                return times, data
            raise RuntimeError("Unreachable Error")
        except FileNotFoundError:
            raise FileNotFoundError(f"Chunk file not found: {s3_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading chunk {chunk_index} from {s3_path}: {e}")
