use crate::{
    array_index::ArrayIndex, errors::convert_omfilesrs_error, fsspec_backend::AsyncFsSpecBackend,
    typed_array::OmFileTypedArray,
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
    io::reader_async::OmFileReaderAsync,
};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::{fs::File, ops::Range, sync::Arc};

/// A reader for OM files with async access.
///
/// This class provides asynchronous access to multi-dimensional array data stored
/// in OM files. It supports reading from local files via memory mapping or
/// from remote files through fsspec compatibility.
#[gen_stub_pyclass]
#[pyclass(module = "omfiles.omfiles")]
pub struct OmFilePyReaderAsync {
    /// The reader is stored in an Option to be able to properly close it,
    /// particularly when working with memory-mapped files.
    /// The RwLock is used to allow multiple readers to access the reader
    /// concurrently, but only one writer to close it.
    reader: RwLock<Option<OmFileReaderAsync<AsyncBackendImpl>>>,
    /// Shape of the array data in the file (read-only property)
    #[pyo3(get)]
    shape: Vec<u64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl OmFilePyReaderAsync {
    /// Create a new async reader from an fsspec file object.
    ///
    /// Parameters
    /// ----------
    /// file_obj : fsspec.core.OpenFile
    ///     An fsspec file object with read_bytes method and fs attribute.
    ///
    /// Returns
    /// -------
    /// OmFilePyReaderAsync
    ///     A new reader instance
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the provided file object is not a valid fsspec file
    /// IOError
    ///     If there's an error reading the file
    #[staticmethod]
    async fn from_fsspec(fs_obj: PyObject, path: String) -> PyResult<Self> {
        Python::with_gil(|py| {
            let bound_object = fs_obj.bind(py);

            if !bound_object.hasattr("_cat_file")? && !bound_object.hasattr("_size")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Input must be a valid fsspec file object with `_cat_file` and `_size` methods",
                ));
            }
            Ok(())
        })?;
        let backend = AsyncFsSpecBackend::new(fs_obj, path).await?;
        let backend = AsyncBackendImpl::FsSpec(backend);
        let reader = OmFileReaderAsync::new(Arc::new(backend))
            .await
            .map_err(convert_omfilesrs_error)?;

        let shape = get_shape_vec(&reader);
        Ok(Self {
            reader: RwLock::new(Some(reader)),
            shape,
        })
    }

    /// Create a new async reader from a local file path.
    ///
    /// Parameters
    /// ----------
    /// file_path : str
    ///     Path to the OM file to read
    ///
    /// Returns
    /// -------
    /// OmFilePyReaderAsync
    ///     A new reader instance
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the file cannot be opened or read
    #[staticmethod]
    async fn from_path(file_path: String) -> PyResult<Self> {
        let file_handle = File::open(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let backend = AsyncBackendImpl::Mmap(MmapFile::new(file_handle, Mode::ReadOnly)?);
        let reader = OmFileReaderAsync::new(Arc::new(backend))
            .await
            .map_err(convert_omfilesrs_error)?;
        let shape = get_shape_vec(&reader);

        Ok(Self {
            reader: RwLock::new(Some(reader)),
            shape,
        })
    }

    /// Read data from the array concurrently based on specified ranges.
    ///
    /// Parameters
    /// ----------
    /// ranges : ArrayIndex
    ///     Index or slice object specifying the ranges to read
    ///
    /// Returns
    /// -------
    /// OmFileTypedArray
    ///     Array data of the appropriate numpy type
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the reader is closed
    /// TypeError
    ///     If the data type is not supported
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

    /// Close the reader and release any resources.
    ///
    /// This method properly closes the underlying file resources.
    ///
    /// Returns
    /// -------
    /// None
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the reader cannot be closed due to concurrent access
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
    reader: &OmFileReaderAsync<AsyncBackendImpl>,
    read_ranges: &[Range<u64>],
    io_size_max: Option<u64>,
    io_size_merge: Option<u64>,
) -> PyResult<ndarray::ArrayD<T>>
where
    T: Element + OmFileArrayDataType + Clone + Zero + Send + Sync + 'static,
{
    // Just do the Rust async operation
    let array = reader
        .read::<T>(read_ranges, io_size_max, io_size_merge)
        .await
        .map_err(convert_omfilesrs_error)?;

    Ok(array.squeeze())
}

/// Small helper function to get the correct shape of the data. We need to
/// be careful with scalars and groups!
fn get_shape_vec<Backend>(reader: &OmFileReaderAsync<Backend>) -> Vec<u64> {
    let dtype = reader.data_type();
    if dtype == DataType::None {
        // "groups"
        return vec![];
    } else if (dtype as u8) < (DataType::Int8Array as u8) {
        // scalars
        return vec![];
    }
    reader.get_dimensions().to_vec()
}

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
