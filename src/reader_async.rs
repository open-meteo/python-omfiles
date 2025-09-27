use crate::{
    array_index::ArrayIndex, compression::PyCompressionType, data_type::describe_dtype,
    errors::convert_omfilesrs_error, fsspec_backend::AsyncFsSpecBackend,
    typed_array::OmFileTypedArray,
};
use async_lock::RwLock;
use delegate::delegate;
use num_traits::Zero;
use numpy::{
    ndarray::{self, Array0},
    Element, PyArray0,
};
use omfiles_rs::{
    reader_async::OmFileReaderAsync as OmFileReaderAsyncRs,
    traits::{
        OmArrayVariable, OmFileArrayDataType, OmFileAsyncReadable, OmFileReaderBackendAsync,
        OmFileScalarDataType, OmFileVariable, OmScalarVariable,
    },
    FileAccessMode, MmapFile, OmDataType,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyTuple,
    BoundObject,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::{fs::File, ops::Range, sync::Arc};

/// An OmFileReaderAsync class for reading .om files asynchronously.
///
/// An OmFileReaderAsync object can represent a multidimensional array variable, a scalar variable (an attribute), or a group.
/// An OmFileReaderAsync can have an arbitrary number of child readers, each representing a child node in a tree-hierarchy.
/// Supports reading from local files via memory mapping or from remote files through fsspec compatibility.
///
/// Variables in OM-Files do not have named dimensions! That means you have to know
/// what the dimensions represent in advance or you need to explicitly encode them as
/// some kind of attribute.
#[gen_stub_pyclass]
#[pyclass(module = "omfiles.omfiles")]
pub struct OmFileReaderAsync {
    /// The reader is stored in an Option to be able to properly close it,
    /// particularly when working with memory-mapped files.
    /// The RwLock is used to allow multiple readers to access the reader
    /// concurrently, but only one writer to close it.
    reader: RwLock<Option<Arc<OmFileReaderAsyncRs<AsyncReaderBackendImpl>>>>,
    /// Shape of the array data in the file (read-only property)
    shape: Vec<u64>,
}

impl OmFileReaderAsync {
    fn from_reader(reader: OmFileReaderAsyncRs<AsyncReaderBackendImpl>) -> PyResult<Self> {
        let shape = get_shape_vec(&reader);

        Ok(Self {
            reader: RwLock::new(Some(Arc::new(reader))),
            shape,
        })
    }

    async fn from_backend(backend: AsyncReaderBackendImpl) -> PyResult<Self> {
        let reader = OmFileReaderAsyncRs::new(Arc::new(backend))
            .await
            .map_err(convert_omfilesrs_error)?;
        Self::from_reader(reader)
    }

    fn lock_error() -> PyErr {
        PyErr::new::<PyRuntimeError, _>("Failed to acquire lock on reader")
    }

    fn closed_error() -> PyErr {
        PyErr::new::<PyValueError, _>("I/O operation on closed reader")
    }

    fn only_arrays_error() -> PyErr {
        PyErr::new::<PyValueError, _>("Only arrays are supported")
    }

    fn only_scalars_error() -> PyErr {
        PyErr::new::<PyValueError, _>("Only scalars are supported")
    }

    fn with_reader<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&OmFileReaderAsyncRs<AsyncReaderBackendImpl>) -> PyResult<R>,
    {
        let guard = self
            .reader
            .try_read()
            .map_or_else(|| Err(Self::lock_error()), |reader| Ok(reader))?;
        match &*guard {
            Some(reader) => f(reader),
            None => Err(Self::closed_error()),
        }
    }

    async fn with_reader_async<F, R>(&self, f: F) -> PyResult<R>
    where
        F: for<'a> AsyncFnOnce(&Arc<OmFileReaderAsyncRs<AsyncReaderBackendImpl>>) -> PyResult<R>,
    {
        let guard = self
            .reader
            .try_read()
            .map_or_else(|| Err(Self::lock_error()), |reader| Ok(reader))?;
        match &*guard {
            Some(reader) => f(reader).await,
            None => Err(Self::closed_error()),
        }
    }

    fn read_string_scalar(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.with_reader(|reader| {
            let scalar_reader = reader
                .expect_scalar()
                .map_err(|_| Self::only_scalars_error())?;

            let value = scalar_reader.read_scalar::<String>();

            value
                .into_pyobject(py)
                .map(BoundObject::into_any)
                .map(BoundObject::unbind)
                .map_err(Into::into)
        })
    }

    fn read_numeric_scalar<'py, T: Element + Clone>(&self, py: Python<'py>) -> PyResult<Py<PyAny>>
    where
        T: OmFileScalarDataType + IntoPyObject<'py>,
    {
        self.with_reader(|reader| {
            let scalar_reader = reader
                .expect_scalar()
                .map_err(|_| Self::only_scalars_error())?;

            let value = scalar_reader.read_scalar::<T>();
            let array_base = Array0::from_elem([], value.unwrap());
            let py_scalar = PyArray0::from_owned_array(py, array_base);

            return Ok(py_scalar.into_any().unbind());
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl OmFileReaderAsync {
    /// Create a new async reader from an fsspec fs object.
    ///
    /// Args:
    ///     fs_obj (fsspec.spec.AbstractFileSystem): A fsspec file system object which needs to have the async methods `_cat_file` and `_size`.
    ///     path (str): The path to the file within the file system.
    ///
    /// Returns:
    ///     OmFileReaderAsync: A new reader instance.
    ///
    /// Raises:
    ///     TypeError: If the provided file object is not a valid fsspec file.
    ///     IOError: If there's an error reading the file.
    #[staticmethod]
    async fn from_fsspec(fs_obj: Py<PyAny>, path: String) -> PyResult<Self> {
        Python::attach(|py| {
            let bound_object = fs_obj.bind(py);

            if !bound_object.hasattr("_cat_file")? && !bound_object.hasattr("_size")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Input must be a valid fsspec file object with `_cat_file` and `_size` methods",
                ));
            }
            Ok(())
        })?;
        let backend = AsyncFsSpecBackend::new(fs_obj, path).await?;
        let backend = AsyncReaderBackendImpl::FsSpec(backend);
        Self::from_backend(backend).await
    }

    /// Create a new async reader from a local file path.
    ///
    /// Args:
    ///     file_path (str): Path to the OM file to read.
    ///
    /// Returns:
    ///     OmFileReaderAsync: A new reader instance.
    ///
    /// Raises:
    ///     IOError: If the file cannot be opened or read.
    #[staticmethod]
    async fn from_path(file_path: String) -> PyResult<Self> {
        let file_handle = File::open(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let backend =
            AsyncReaderBackendImpl::Mmap(MmapFile::new(file_handle, FileAccessMode::ReadOnly)?);
        Self::from_backend(backend).await
    }

    /// Enter a context manager block.
    ///
    /// Returns:
    ///     OmFileReaderAsync: Self for use in context manager.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Exit a context manager block, closing the reader.
    ///
    /// Args:
    ///     _exc_type (type, optional): The exception type, if an exception was raised.
    ///     _exc_value (Exception, optional): The exception value, if an exception was raised.
    ///     _traceback (traceback, optional): The traceback, if an exception was raised.
    ///
    /// Returns:
    ///     bool: False (exceptions are not suppressed).
    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        _exc_type: Option<Py<PyAny>>,
        _exc_value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }

    /// Check if the reader is closed.
    ///
    /// Returns:
    ///     bool: True if the reader is closed, False otherwise.
    #[getter]
    fn closed(&self) -> PyResult<bool> {
        let guard = self
            .reader
            .try_read()
            .map_or_else(|| Err(Self::lock_error()), |reader| Ok(reader))?;
        Ok(guard.is_none())
    }

    /// Close the reader and release any resources.
    ///
    /// Properly closes the underlying file resources.
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     RuntimeError: If the reader cannot be closed due to concurrent access.
    fn close(&self) -> PyResult<()> {
        // Need write access to take the reader
        let mut guard = self
            .reader
            .try_write()
            .map_or_else(|| Err(Self::lock_error()), |reader| Ok(reader))?;

        // takes the reader, leaving None in the RwLock
        if let Some(reader) = guard.take() {
            // Extract the backend before dropping reader
            match &*reader.backend {
                AsyncReaderBackendImpl::FsSpec(fs_backend) => {
                    fs_backend.close()?;
                }
                AsyncReaderBackendImpl::Mmap(_) => {
                    // Will be dropped automatically
                }
            }
            // The reader is dropped here when it goes out of scope
        }

        Ok(())
    }

    /// The shape of the variable.
    ///
    /// Returns:
    ///     tuple[int, …]: The shape of the variable as a tuple.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyTuple>> {
        let tup = PyTuple::new(py, &self.shape)?;
        Ok(tup)
    }

    /// The chunk shape of the variable.
    ///
    /// Returns:
    ///     tuple[int, …]: The chunk shape of the variable as a tuple.
    #[getter]
    fn chunks<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyTuple>> {
        self.with_reader(|reader| {
            let chunks = get_chunk_shape(reader);
            let tup = PyTuple::new(py, chunks)?;
            Ok(tup)
        })
    }

    /// Check if the variable is an array.
    ///
    /// Returns:
    ///     bool: True if the variable is an array, False otherwise.
    #[getter]
    fn is_array(&self) -> PyResult<bool> {
        self.with_reader(|reader| Ok(reader.data_type().is_array()))
    }

    /// Check if the variable is a scalar.
    ///
    /// Returns:
    ///     bool: True if the variable is a scalar, False otherwise.
    #[getter]
    fn is_scalar(&self) -> PyResult<bool> {
        self.with_reader(|reader| Ok(reader.data_type().is_scalar()))
    }

    /// Check if the variable is a group (a variable with data type None).
    ///
    /// Returns:
    ///     bool: True if the variable is a group, False otherwise.
    #[getter]
    fn is_group(&self) -> PyResult<bool> {
        self.with_reader(|reader| Ok(reader.data_type() == OmDataType::None))
    }

    /// Get the data type of the data stored in the .om file.
    ///
    /// Returns:
    ///     numpy.dtype | type: Data type of the data.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.with_reader(|reader| describe_dtype(py, &reader.data_type()))
    }

    /// Get the name of the variable stored in the .om file.
    ///
    /// Returns:
    ///     str: Name of the variable or an empty string if not available.
    #[getter]
    fn name(&self) -> PyResult<String> {
        self.with_reader(|reader| Ok(reader.name().to_string()))
    }

    /// Get the compression type of the variable.
    ///
    /// Returns:
    ///     str: Compression type of the variable.
    #[getter]
    fn compression_name(&self) -> PyResult<PyCompressionType> {
        self.with_reader(|reader| {
            Ok(PyCompressionType::from_omfilesrs(
                reader
                    .expect_array()
                    .map_err(|_| Self::only_arrays_error())?
                    .compression(),
            ))
        })
    }

    /// Number of children of the variable.
    ///
    /// Returns:
    ///     int: Number of children of the variable.
    #[getter]
    fn num_children(&self) -> PyResult<u32> {
        self.with_reader(|reader| Ok(reader.number_of_children()))
    }

    /// Get a child reader at the specified index.
    ///
    /// Returns:
    ///     OmFileReaderAsync: Child reader at the specified index if exists.
    async fn get_child_by_index(&self, index: u32) -> PyResult<Self> {
        self.with_reader_async(
            |reader: &Arc<OmFileReaderAsyncRs<AsyncReaderBackendImpl>>| {
                let reader = reader.clone();
                async move {
                    match reader.get_child_by_index(index).await {
                        Some(child) => Self::from_reader(child),
                        None => Err(PyValueError::new_err(format!(
                            "Child at index {} does not exist",
                            index
                        ))),
                    }
                }
            },
        )
        .await
    }

    /// Get a child reader by name.
    ///
    /// Returns:
    ///     OmFileReaderAsync: Child reader with the specified name if exists.
    async fn get_child_by_name(&self, name: String) -> PyResult<Self> {
        self.with_reader_async(
            |reader: &Arc<OmFileReaderAsyncRs<AsyncReaderBackendImpl>>| {
                let reader = reader.clone();
                async move {
                    match reader.get_child_by_name(&name).await {
                        Some(child) => Self::from_reader(child),
                        None => Err(PyValueError::new_err(format!(
                            "Child with name '{}' does not exist",
                            name
                        ))),
                    }
                }
            },
        )
        .await
    }

    /// Read data from the array concurrently based on specified ranges.
    ///
    /// Args:
    ///     ranges (:py:data:`omfiles.types.BasicSelection`): Index or slice object specifying the ranges to read.
    ///
    /// Returns:
    ///     OmFileTypedArray: Array data of the appropriate numpy type.
    ///
    /// Raises:
    ///     ValueError: If the reader is closed.
    ///     TypeError: If the data type is not supported.
    async fn read_array<'py>(&self, ranges: ArrayIndex) -> PyResult<OmFileTypedArray> {
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
            OmDataType::Int8Array => {
                let array = read_squeezed_typed_array::<i8>(reader, &read_ranges).await?;
                Ok(OmFileTypedArray::Int8(array))
            }
            OmDataType::Int16Array => {
                let array = read_squeezed_typed_array::<i16>(reader, &read_ranges).await?;
                Ok(OmFileTypedArray::Int16(array))
            }
            OmDataType::Int32Array => {
                let array = read_squeezed_typed_array::<i32>(reader, &read_ranges).await?;
                Ok(OmFileTypedArray::Int32(array))
            }
            OmDataType::Int64Array => {
                let array = read_squeezed_typed_array::<i64>(reader, &read_ranges).await?;
                Ok(OmFileTypedArray::Int64(array))
            }
            OmDataType::Uint8Array => {
                let array = read_squeezed_typed_array::<u8>(reader, &read_ranges).await?;
                Ok(OmFileTypedArray::Uint8(array))
            }
            OmDataType::Uint16Array => {
                let array = read_squeezed_typed_array::<u16>(reader, &read_ranges).await?;
                Ok(OmFileTypedArray::Uint16(array))
            }
            OmDataType::Uint32Array => {
                let array = read_squeezed_typed_array::<u32>(reader, &read_ranges).await?;
                Ok(OmFileTypedArray::Uint32(array))
            }
            OmDataType::Uint64Array => {
                let array = read_squeezed_typed_array::<u64>(reader, &read_ranges).await?;
                Ok(OmFileTypedArray::Uint64(array))
            }
            OmDataType::FloatArray => {
                let array = read_squeezed_typed_array::<f32>(reader, &read_ranges).await?;
                Ok(OmFileTypedArray::Float(array))
            }
            OmDataType::DoubleArray => {
                let array = read_squeezed_typed_array::<f64>(reader, &read_ranges).await?;
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

    /// Read the scalar value of the variable.
    ///
    /// Returns:
    ///     object: The scalar value as a Python object (str, int, or float).
    ///
    /// Raises:
    ///     ValueError: If the variable is not a scalar.
    fn read_scalar(&self) -> PyResult<Py<PyAny>> {
        self.with_reader(|reader| {
            Python::attach(|py| match reader.data_type() {
                OmDataType::Int8 => self.read_numeric_scalar::<i8>(py),
                OmDataType::Uint8 => self.read_numeric_scalar::<u8>(py),
                OmDataType::Int16 => self.read_numeric_scalar::<i16>(py),
                OmDataType::Uint16 => self.read_numeric_scalar::<u16>(py),
                OmDataType::Int32 => self.read_numeric_scalar::<i32>(py),
                OmDataType::Uint32 => self.read_numeric_scalar::<u32>(py),
                OmDataType::Int64 => self.read_numeric_scalar::<i64>(py),
                OmDataType::Uint64 => self.read_numeric_scalar::<u64>(py),
                OmDataType::Float => self.read_numeric_scalar::<f32>(py),
                OmDataType::Double => self.read_numeric_scalar::<f64>(py),
                OmDataType::String => self.read_string_scalar(py),
                _ => Err(Self::only_scalars_error()),
            })
        })
    }
}

async fn read_squeezed_typed_array<T>(
    reader: &OmFileReaderAsyncRs<AsyncReaderBackendImpl>,
    read_ranges: &[Range<u64>],
) -> PyResult<ndarray::ArrayD<T>>
where
    T: Element + OmFileArrayDataType + Clone + Zero + Send + Sync + 'static,
{
    let reader = reader
        .expect_array_with_io_sizes(65536, 512)
        .map_err(convert_omfilesrs_error)?;
    let array = reader
        .read::<T>(read_ranges)
        .await
        .map_err(convert_omfilesrs_error)?;

    Ok(array.squeeze())
}

/// Small helper function to get the correct shape of the data. We need to
/// be careful with scalars and groups!
fn get_shape_vec(reader: &OmFileReaderAsyncRs<AsyncReaderBackendImpl>) -> Vec<u64> {
    let reader = reader.expect_array();
    match reader {
        Ok(reader) => reader.get_dimensions().to_vec(),
        Err(_) => return vec![],
    }
}

fn get_chunk_shape(reader: &OmFileReaderAsyncRs<AsyncReaderBackendImpl>) -> Vec<u64> {
    let reader = reader.expect_array();
    match reader {
        Ok(reader) => reader.get_chunk_dimensions().to_vec(),
        Err(_) => return vec![],
    }
}

enum AsyncReaderBackendImpl {
    FsSpec(AsyncFsSpecBackend),
    Mmap(MmapFile),
}

impl OmFileReaderBackendAsync for AsyncReaderBackendImpl {
    delegate! {
        to match self {
            AsyncReaderBackendImpl::Mmap(backend) => backend,
            AsyncReaderBackendImpl::FsSpec(backend) => backend,
        } {
            fn count_async(&self) -> usize;
            async fn get_bytes_async(&self, offset: u64, count: u64) -> Result<Vec<u8>, omfiles_rs::OmFilesError>;
        }
    }
}
