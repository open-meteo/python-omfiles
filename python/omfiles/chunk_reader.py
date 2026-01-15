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
    """Utility class to iterate over chunked om files."""

    def __init__(
        self,
        chunks_meta: OmChunksMeta,
        fs: "fsspec.AbstractFileSystem",
        chunk_files_path: str,
        start_date: np.datetime64,
        end_date: np.datetime64,
    ) -> None:
        """
        Initialize the chunk reader.

        Args:
            chunks_meta (OmChunksMeta): Metadata for the OM chunk files.
            fs (fsspec.AbstractFileSystem): Filesystem for accessing the OM chunk files.
            chunk_files_path (str): Path to the directory containing the chunk files compatible with the provided fs.
            start_date (np.datetime64): Start date of the data to load.
            end_date (np.datetime64): End date of the data to load (inclusive).
        """
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")

        self.chunks_meta = chunks_meta
        self.fs = fs

        self.chunk_files_path = chunk_files_path
        self.start_date = start_date
        self.end_date = end_date
        self.chunk_indices = self.chunks_meta.chunks_for_date_range(start_date, end_date)

    def iter_files(self):
        """
        Iterate over chunk files.

        Yields:
            Tuple[int, str]: Chunk index and path to the chunk file.
        """
        for chunk_index in self.chunk_indices:
            yield chunk_index, f"{self.chunk_files_path}/chunk_{chunk_index}.om"

    def load_data(
        self, spatial_index: Union[Tuple[int, int], Tuple[slice, slice]]
    ) -> Tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float32]]:
        """
        Load data from all chunks for a given spatial index.

        Args:
            spatial_index (Union[Tuple[int, int], Tuple[slice, slice]]): Spatial index (x, y) or slice ranges for the data to load.

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

        time_array = np.concatenate(all_times, axis=-1)
        data_array = np.concatenate(all_data, axis=-1)
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
            spatial_index (Union[Tuple[int, int], Tuple[slice, slice]]): Spatial index (x, y) or slice ranges for the data to load.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time array and data array for this chunk.
        """
        chunk_times = self.chunks_meta.get_chunk_time_range(chunk_index)
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
                if times.shape[-1] != 1 and times.shape[-1] != data.shape[-1]:
                    raise RuntimeError(f"Expected {times.shape[-1]} timestamps but got {data.shape[-1]}")
                return times, data
            raise RuntimeError("Unreachable Error")
        except FileNotFoundError:
            raise FileNotFoundError(f"Chunk file not found: {s3_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading chunk {chunk_index} from {s3_path}: {e}")
