from .io.readers import (
    BaseReader,
    HDF5HidefixReader,
    HDF5Reader,
    NetCDFReader,
    OMReader,
    TensorStoreZarrReader,
    ZarrReader,
)
from .io.writers import BaseWriter, HDF5Writer, NetCDFWriter, OMWriter, ZarrWriter


class FormatFactory:
    # fmt: off
    writers = {
        "h5": HDF5Writer,
        "zarr": ZarrWriter,
        "nc": NetCDFWriter,
        "om": OMWriter
    }

    readers = {
        "h5": HDF5Reader,
        "h5hidefix": HDF5HidefixReader,
        "zarr": ZarrReader,
        "zarrTensorStore": TensorStoreZarrReader,
        "nc": NetCDFReader,
        "om": OMReader
    }

    @classmethod
    def create_writer(cls, format_name: str, filename: str) -> BaseWriter:
        if format_name not in cls.writers:
            raise ValueError(f"Unknown format: {format_name}")
        return cls.writers[format_name](filename)

    @classmethod
    def create_reader(cls, format_name: str, filename: str) -> BaseReader:
        if format_name not in cls.readers:
            raise ValueError(f"Unknown format: {format_name}")
        return cls.readers[format_name](filename)
