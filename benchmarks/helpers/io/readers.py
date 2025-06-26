from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
import omfiles as om
import tensorstore as ts
import xarray as xr
import zarr
from omfiles.types import BasicSelection


class BaseReader(ABC):
    filename: Path

    def __init__(self, filename: str):
        self.filename = Path(filename)

    @abstractmethod
    def read(self, index: BasicSelection) -> np.ndarray:
        raise NotImplementedError("The read method must be implemented by subclasses")

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError("The close method must be implemented by subclasses")


class HDF5Reader(BaseReader):
    h5_reader: h5py.Dataset

    def __init__(self, filename: str):
        super().__init__(filename)
        # Disable chunk caching by setting cache properties
        # Parameters: (chunk_cache_mem_size, chunk_cache_nslots, chunk_cache_w0)
        # Setting size to 0 effectively disables the cache
        file = h5py.File(self.filename, "r", rdcc_nbytes=0, rdcc_nslots=0, rdcc_w0=0)
        dataset = file["dataset"]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError("Expected a h5py Dataset")
        self.h5_reader = dataset

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.h5_reader[index]

    def close(self) -> None:
        self.h5_reader.file.close()


class HDF5HidefixReader(BaseReader):
    h5_reader: xr.Dataset

    def __init__(self, filename: str):
        super().__init__(filename)
        self.h5_reader = xr.open_dataset(self.filename, engine="hidefix")

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.h5_reader["dataset"][index].values

    def close(self) -> None:
        self.h5_reader.close()


class ZarrReader(BaseReader):
    zarr_reader: zarr.Array

    def __init__(self, filename: str):
        super().__init__(filename)
        z = zarr.open(str(self.filename), mode="r")
        if not isinstance(z, zarr.Group):
            raise TypeError("Expected a zarr Group")
        array = z["arr_0"]
        if not isinstance(array, zarr.Array):
            raise TypeError("Expected a zarr Array")

        self.zarr_reader = array

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.zarr_reader[index].__array__()

    def close(self) -> None:
        self.zarr_reader.store.close()


class TensorStoreZarrReader(BaseReader):
    ts_reader: ts.TensorStore  # type: ignore

    def __init__(self, filename: str):
        super().__init__(filename)
        # Open the Zarr file using TensorStore
        self.ts_reader = ts.open(
            {  # type: ignore
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": str(self.filename),
                },
                "path": "arr_0",
                "open": True,
            }
        ).result()

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.ts_reader[index].read().result()

    def close(self) -> None:
        pass

class ZarrsCodecsZarrReader(BaseReader):
    zarr_reader: zarr.Array

    def __init__(self, filename: str):
        import zarrs  # noqa: F401
        zarr.config.set({
            # "threading.num_workers": None,
            # "array.write_empty_chunks": False,
            "codec_pipeline": {
                "path": "zarrs.ZarrsCodecPipeline",
                # "validate_checksums": True,
                # "store_empty_chunks": False,
                # "chunk_concurrent_minimum": 4,
                # "chunk_concurrent_maximum": None,
                "batch_size": 1,
            }
        })
        super().__init__(filename)
        z = zarr.open(str(self.filename), mode="r")
        if not isinstance(z, zarr.Group):
            raise TypeError("Expected a zarr Group")
        array = z["arr_0"]
        if not isinstance(array, zarr.Array):
            raise TypeError("Expected a zarr Array")

        self.zarr_reader = array

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.zarr_reader[index].__array__()

    def close(self) -> None:
        self.zarr_reader.store.close()


class NetCDFReader(BaseReader):
    nc_reader: nc.Dataset

    def __init__(self, filename: str):
        super().__init__(filename)
        # disable netcdf caching: https://www.unidata.ucar.edu/software/netcdf/workshops/2012/nc4chunking/Cache.html
        nc.set_chunk_cache(0, 0, 0)

        self.nc_reader = nc.Dataset(self.filename, "r")

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.nc_reader.variables["dataset"][index]

    def close(self) -> None:
        self.nc_reader.close()


class OMReader(BaseReader):
    om_reader: om.OmFilePyReader

    def __init__(self, filename: str):
        super().__init__(filename)
        self.om_reader = om.OmFilePyReader(str(self.filename))

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.om_reader[index]

    def close(self) -> None:
        pass
        # TODO: Implement close method
        # self.om_reader.close()
