use crate::{
    array_index::ArrayIndex, errors::convert_omfilesrs_error, fsspec_backend::AsyncFsSpecBackend,
    reader::get_shape_vec, typed_array::OmFileTypedArray,
};
use async_lock::RwLock;
use delegate::delegate;
use num_traits::Zero;
use numpy::{
    ndarray::{self},
    Element,
};
use omfiles_rs::{
    backend::{
        backends::OmFileReaderBackendAsync,
        mmapfile::{MmapFile, Mode},
    },
    core::data_types::{DataType, OmFileArrayDataType},
    io::reader::OmFileReader,
};
use pyo3::prelude::*;
use std::{fs::File, ops::Range, sync::Arc};

enum AsyncBackendImpl {
    FsSpec(AsyncFsSpecBackend),
    Mmap(MmapFile),
}

impl OmFileReaderBackendAsync for AsyncBackendImpl {
    delegate! {
        to match self {
            AsyncBackendImpl::Mmap(backend) => backend,
            AsyncBackendImpl::FsSpec(backend) => backend,
        } {
            fn count_async(&self) -> usize;
            async fn get_bytes_async(&self, offset: u64, count: u64) -> Result<Vec<u8>, omfiles_rs::errors::OmFilesRsError>;
        }
    }
}

#[pyclass]
pub struct OmFilePyReaderAsync {
    /// The reader is stored in an Option to be able to properly close it,
    /// particularly when working with memory-mapped files.
    /// The RwLock is used to allow multiple readers to access the reader
    /// concurrently, but only one writer to close it.
    reader: RwLock<Option<OmFileReader<AsyncBackendImpl>>>,
    #[pyo3(get)]
    shape: Vec<u64>,
}

#[pymethods]
impl OmFilePyReaderAsync {
    #[staticmethod]
    async fn from_fsspec(file_obj: PyObject) -> PyResult<Self> {
        let backend = Python::with_gil(|py| {
            let bound_object = file_obj.bind(py);

            if !bound_object.hasattr("read_bytes")? || !bound_object.hasattr("fs")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Input must be a valid fsspec file object with read_bytes and fs attribute",
                ));
            }

            Ok(AsyncFsSpecBackend::new(file_obj)?)
        })?;
        let backend = AsyncBackendImpl::FsSpec(backend);
        let reader = OmFileReader::async_new(Arc::new(backend))
            .await
            .map_err(convert_omfilesrs_error)?;

        let shape = get_shape_vec(&reader);
        Ok(Self {
            reader: RwLock::new(Some(reader)),
            shape,
        })
    }

    #[staticmethod]
    async fn from_path(file_path: String) -> PyResult<Self> {
        let file_handle = File::open(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let backend = AsyncBackendImpl::Mmap(MmapFile::new(file_handle, Mode::ReadOnly)?);
        let reader = OmFileReader::async_new(Arc::new(backend))
            .await
            .map_err(convert_omfilesrs_error)?;
        let shape = get_shape_vec(&reader);

        Ok(Self {
            reader: RwLock::new(Some(reader)),
            shape,
        })
    }

    async fn read_concurrent<'py>(&self, ranges: ArrayIndex) -> PyResult<OmFileTypedArray> {
        let io_size_max = None;
        let io_size_merge = None;
        // Convert the Python ranges to Rust ranges
        let read_ranges = ranges.to_read_range(&self.shape)?;

        let guard = self.reader.try_read().unwrap();

        let reader = if let Some(reader) = &*guard {
            Ok(reader)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "I/O operation on closed reader or file",
            ))
        }?;

        // Get the data type and a cloned backend from the reader
        let data_type = reader.data_type();
        let result = match data_type {
            DataType::Int8Array => {
                let array = read_squeezed_typed_array::<i8>(
                    reader,
                    &read_ranges,
                    io_size_max,
                    io_size_merge,
                )
                .await?;
                Ok(OmFileTypedArray::Int8(array))
            }
            DataType::Int16Array => {
                let array = read_squeezed_typed_array::<i16>(
                    reader,
                    &read_ranges,
                    io_size_max,
                    io_size_merge,
                )
                .await?;
                Ok(OmFileTypedArray::Int16(array))
            }
            DataType::Int32Array => {
                let array = read_squeezed_typed_array::<i32>(
                    reader,
                    &read_ranges,
                    io_size_max,
                    io_size_merge,
                )
                .await?;
                Ok(OmFileTypedArray::Int32(array))
            }
            DataType::Int64Array => {
                let array = read_squeezed_typed_array::<i64>(
                    reader,
                    &read_ranges,
                    io_size_max,
                    io_size_merge,
                )
                .await?;
                Ok(OmFileTypedArray::Int64(array))
            }
            DataType::Uint8Array => {
                let array = read_squeezed_typed_array::<u8>(
                    reader,
                    &read_ranges,
                    io_size_max,
                    io_size_merge,
                )
                .await?;
                Ok(OmFileTypedArray::Uint8(array))
            }
            DataType::Uint16Array => {
                let array = read_squeezed_typed_array::<u16>(
                    reader,
                    &read_ranges,
                    io_size_max,
                    io_size_merge,
                )
                .await?;
                Ok(OmFileTypedArray::Uint16(array))
            }
            DataType::Uint32Array => {
                let array = read_squeezed_typed_array::<u32>(
                    reader,
                    &read_ranges,
                    io_size_max,
                    io_size_merge,
                )
                .await?;
                Ok(OmFileTypedArray::Uint32(array))
            }
            DataType::Uint64Array => {
                let array = read_squeezed_typed_array::<u64>(
                    reader,
                    &read_ranges,
                    io_size_max,
                    io_size_merge,
                )
                .await?;
                Ok(OmFileTypedArray::Uint64(array))
            }
            DataType::FloatArray => {
                let array = read_squeezed_typed_array::<f32>(
                    reader,
                    &read_ranges,
                    io_size_max,
                    io_size_merge,
                )
                .await?;
                Ok(OmFileTypedArray::Float(array))
            }
            DataType::DoubleArray => {
                let array = read_squeezed_typed_array::<f64>(
                    reader,
                    &read_ranges,
                    io_size_max,
                    io_size_merge,
                )
                .await?;
                Ok(OmFileTypedArray::Double(array))
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "Invalid data type: {:?}",
                    data_type
                )));
            }
        };
        result
    }

    fn close(&self) -> PyResult<()> {
        // Need write access to take the reader
        let mut guard = match self.reader.try_write() {
            Some(guard) => guard,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Could not acquire write lock",
                ))
            }
        };

        // takes the reader, leaving None in the RwLock
        if let Some(reader) = guard.take() {
            // Extract the backend before dropping reader
            if let Ok(backend) = Arc::try_unwrap(reader.backend) {
                match backend {
                    AsyncBackendImpl::FsSpec(fs_backend) => {
                        fs_backend.close()?;
                    }
                    AsyncBackendImpl::Mmap(_) => {
                        // Will be dropped automatically
                    }
                }
            }
            // The reader is dropped here when it goes out of scope
        }

        Ok(())
    }
}

async fn read_squeezed_typed_array<T>(
    reader: &OmFileReader<AsyncBackendImpl>,
    read_ranges: &[Range<u64>],
    io_size_max: Option<u64>,
    io_size_merge: Option<u64>,
) -> PyResult<ndarray::ArrayD<T>>
where
    T: Element + OmFileArrayDataType + Clone + Zero + Send + Sync + 'static,
{
    // Just do the Rust async operation
    let array = reader
        .read_async::<T>(read_ranges, io_size_max, io_size_merge)
        .await
        .map_err(convert_omfilesrs_error)?;

    Ok(array.squeeze())
}
