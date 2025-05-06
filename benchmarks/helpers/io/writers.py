import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import h5py
import netCDF4 as nc
import omfiles as om
import zarr
from zarr.core.buffer import NDArrayLike


class BaseWriter(ABC):
    def __init__(self, filename: str):
        self.filename = Path(filename)

    @abstractmethod
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        raise NotImplementedError("The write method must be implemented by subclasses")

    def get_file_size(self) -> int:
        """Get the size of a file in bytes."""
        path = Path(self.filename)

        # For directories (like Zarr stores), calculate total size recursively
        if path.is_dir():
            total_size = 0
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = Path(dirpath) / f
                    if fp.is_file():
                        total_size += fp.stat().st_size
            return total_size
        # For regular files
        elif path.is_file():
            return path.stat().st_size
        else:
            return 0


class HDF5Writer(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        with h5py.File(self.filename, "w") as f:
            f.create_dataset("dataset", data=data, chunks=chunk_size)


class ZarrWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        import numcodecs
        compressors = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
        # zarr.create_array(str(self.filename), name="arr_0", data=data, chunks=chunk_size, compressors=compressors, zarr_format=2, overwrite=True)
        root = zarr.open(str(self.filename), mode="w", zarr_format=2)
        # Ensure root is a Group and not an Array (for type checker)
        if not isinstance(root, zarr.Group):
            raise TypeError("Expected root to be a zarr.hierarchy.Group")
        arr_0 = root.create_array("arr_0", shape=data.shape, chunks=chunk_size, dtype="f4", compressor=compressors)
        arr_0[:] = data


class NetCDFWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        with nc.Dataset(self.filename, "w", format="NETCDF4") as ds:
            dimension_names = tuple(f"dim{i}" for i in range(data.ndim))
            for dim, size in zip(dimension_names, data.shape):
                ds.createDimension(dim, size)

            var = ds.createVariable("dataset", data.dtype, dimension_names, chunksizes=chunk_size)
            var[:] = data


class OMWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        writer = om.OmFilePyWriter(str(self.filename))
        variable = writer.write_array(data.__array__(), chunk_size, 100, 0)
        writer.close(variable)
