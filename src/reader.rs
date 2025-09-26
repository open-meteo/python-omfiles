use crate::{
    array_index::ArrayIndex, compression::PyCompressionType, data_type::describe_dtype,
    errors::convert_omfilesrs_error, fsspec_backend::FsSpecBackend, hierarchy::OmVariable,
    typed_array::OmFileTypedArray,
};
use delegate::delegate;
use num_traits::Zero;
use numpy::{
    ndarray::{self, Array0},
    Element, PyArray0,
};
use omfiles_rs::{
    reader::{OmFileArray as OmFileArrayRs, OmFileReader as OmFileReaderRs},
    traits::{
        OmArrayVariable, OmFileArrayDataType, OmFileReadable, OmFileReaderBackend,
        OmFileScalarDataType, OmFileVariable, OmFileVariableMetadataTree, OmScalarVariable,
    },
    FileAccessMode, MmapFile, OmDataType, OmFilesError,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyTuple,
    BoundObject,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::{
    borrow::Cow,
    collections::HashMap,
    fs::File,
    ops::Range,
    sync::{Arc, RwLock},
};

/// An OmFileReader class for reading .om files synchronously.
///
/// An OmFileReader object can represent a multidimensional array variable, a scalar variable (an attribute), or a group.
/// An OmFileReader can have an arbitrary number of child readers, each representing a child node in a tree-hierarchy.
/// Supports reading from local files via memory mapping or from remote files through fsspec compatibility.
///
/// Variables in OM-Files do not have named dimensions! That means you have to know
/// what the dimensions represent in advance or you need to explicitly encode them as
/// some kind of attribute.
#[gen_stub_pyclass]
#[pyclass(module = "omfiles.omfiles")]
pub struct OmFileReader {
    /// The reader is stored in an Option to be able to properly close it,
    /// particularly when working with memory-mapped files.
    /// The RwLock is used to allow multiple readers to access the reader
    /// concurrently, but only one writer to close it.
    reader: RwLock<Option<OmFileReaderRs<ReaderBackendImpl>>>,
    /// Get the shape of the data stored in the .om file.
    ///
    /// Returns:
    ///     list: List containing the dimensions of the data.
    shape: Vec<u64>,
}

impl OmFileReader {
    fn from_reader(reader: OmFileReaderRs<ReaderBackendImpl>) -> PyResult<Self> {
        let shape = get_shape_vec(&reader);

        Ok(Self {
            reader: RwLock::new(Some(reader)),
            shape,
        })
    }

    fn from_backend(backend: ReaderBackendImpl) -> PyResult<Self> {
        let reader = OmFileReaderRs::new(Arc::new(backend)).map_err(convert_omfilesrs_error)?;
        Self::from_reader(reader)
    }

    fn lock_error<T>(e: std::sync::TryLockError<T>) -> PyErr {
        PyErr::new::<PyRuntimeError, _>(format!("Failed to acquire lock on reader: {}", e))
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
        F: FnOnce(&OmFileReaderRs<ReaderBackendImpl>) -> PyResult<R>,
    {
        let guard = self.reader.try_read().map_err(|e| Self::lock_error(e))?;
        match &*guard {
            Some(reader) => f(reader),
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
impl OmFileReader {
    /// Initialize an OmFileReader from a file path or fsspec file object.
    ///
    /// Args:
    ///     source (str or fsspec.core.OpenFile): Path to the .om file to read or a fsspec file object.
    ///
    /// Raises:
    ///     ValueError: If the file cannot be opened or is invalid.
    #[new]
    fn new(source: Py<PyAny>) -> PyResult<Self> {
        Python::attach(|py| {
            if let Ok(path) = source.extract::<String>(py) {
                // If source is a string, treat it as a file path
                Self::from_path(&path)
            } else {
                let obj = source.bind(py);
                if obj.hasattr("path")? && obj.hasattr("fs")? {
                    let fs = obj.getattr("fs")?.unbind();
                    let path = obj.getattr("path")?.extract::<String>()?;
                    // If source has fsspec-like attributes, treat it as a fsspec file object
                    Self::from_fsspec(fs, path)
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Input must be either a file path string or an fsspec.core.OpenFile object",
                    ))
                }
            }
        })
    }

    /// Create an OmFileReader from a file path.
    ///
    /// Args:
    ///     file_path (str): Path to the .om file to read.
    ///
    /// Returns:
    ///     OmFileReader: OmFileReader instance.
    #[staticmethod]
    fn from_path(file_path: &str) -> PyResult<Self> {
        let file_handle = File::open(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let backend =
            ReaderBackendImpl::Mmap(MmapFile::new(file_handle, FileAccessMode::ReadOnly)?);
        Self::from_backend(backend)
    }

    /// Create an OmFileReader from a fsspec fs object.
    ///
    /// Args:
    ///     fs_obj (fsspec.spec.AbstractFileSystem): A fsspec file system object which needs to have the methods `cat_file` and `size`.
    ///     path (str): The path to the file within the file system.
    ///
    /// Returns:
    ///     OmFileReader: A new reader instance.
    #[staticmethod]
    fn from_fsspec(fs_obj: Py<PyAny>, path: String) -> PyResult<Self> {
        Python::attach(|py| {
            let bound_object = fs_obj.bind(py);

            if !bound_object.hasattr("cat_file")? || !bound_object.hasattr("size")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Input must be a valid fsspec file object with read, seek methods and fs attribute",
                    ));
            }

            let backend = ReaderBackendImpl::FsSpec(FsSpecBackend::new(fs_obj, path)?);
            Self::from_backend(backend)
        })
    }

    /// Get a mapping of variable names to their file offsets and sizes.
    ///
    /// Returns:
    ///     dict: Dictionary mapping variable names to their metadata.
    fn _get_flat_variable_metadata(&self) -> PyResult<HashMap<String, OmVariable>> {
        self.with_reader(|reader| {
            let metadata = reader._get_flat_variable_metadata();
            Ok(metadata
                .into_iter()
                .map(|(key, offset_size)| {
                    (
                        key.clone(),
                        OmVariable {
                            name: key,
                            offset: offset_size.offset,
                            size: offset_size.size,
                        },
                    )
                })
                .collect())
        })
    }

    /// Initialize a new OmFileReader from a child variable.
    ///
    /// Args:
    ///     variable (OmVariable): Variable metadata to create a new reader from.
    ///
    /// Returns:
    ///     OmFileReader: A new reader for the specified variable.
    fn _init_from_variable(&self, variable: OmVariable) -> PyResult<Self> {
        self.with_reader(|reader| {
            let child_reader = reader
                ._init_child_from_offset_size(variable.into())
                .map_err(convert_omfilesrs_error)?;

            let shape = get_shape_vec(&child_reader);
            Ok(Self {
                reader: RwLock::new(Some(child_reader)),
                shape,
            })
        })
    }

    /// Enter a context manager block.
    ///
    /// Returns:
    ///     OmFileReader: Self for use in context manager.
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
        let guard = self.reader.try_read().map_err(|e| Self::lock_error(e))?;
        Ok(guard.is_none())
    }

    /// Close the reader and release resources.
    ///
    /// This method releases all resources associated with the reader.
    /// After closing, any operation on the reader will raise a ValueError.
    ///
    /// It is safe to call this method multiple times.
    fn close(&self) -> PyResult<()> {
        // Need write access to take the reader
        let mut guard = self.reader.try_write().map_err(|e| Self::lock_error(e))?;

        // takes the reader, leaving None in the RwLock
        if let Some(reader) = guard.take() {
            // Extract the backend before dropping reader
            if let Ok(backend) = Arc::try_unwrap(reader.backend) {
                match backend {
                    ReaderBackendImpl::FsSpec(fs_backend) => {
                        fs_backend.close()?;
                    }
                    ReaderBackendImpl::Mmap(_) => {
                        // Will be dropped automatically
                    }
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
    ///     OmFileReader: Child reader at the specified index if exists.
    fn get_child_by_index(&self, index: u32) -> PyResult<Self> {
        self.with_reader(|reader| match reader.get_child_by_index(index) {
            Some(child) => Self::from_reader(child),
            None => Err(PyValueError::new_err(format!(
                "Child at index {} does not exist",
                index
            ))),
        })
    }

    /// Get a child reader by name.
    ///
    /// Returns:
    ///     OmFileReader: Child reader with the specified name if exists.
    fn get_child_by_name(&self, name: &str) -> PyResult<Self> {
        self.with_reader(|reader| match reader.get_child_by_name(name) {
            Some(child) => Self::from_reader(child),
            None => Err(PyValueError::new_err(format!(
                "Child with name '{}' does not exist",
                name
            ))),
        })
    }

    /// Read data from the open variable.om file using numpy-style indexing.
    ///
    /// Currently only slices with step 1 are supported.
    ///
    /// The returned array will have singleton dimensions removed (squeezed).
    /// For example, if you index a 3D array with [1,:,2], the result will
    /// be a 1D array since dimensions 0 and 2 have size 1.
    ///
    /// Args:
    ///     ranges (:py:data:`omfiles.types.BasicSelection`): Index expression to select data from the array.
    ///         Supports basic numpy indexing.
    ///
    /// Returns:
    ///     numpy.typing.NDArray[numpy.int8 | numpy.int16 | numpy.int32 | numpy.int64 | numpy.uint8 | numpy.uint16 | numpy.uint32 | numpy.uint64 | numpy.float32 | numpy.float64]: NDArray containing the requested data with squeezed singleton dimensions.
    ///
    /// Raises:
    ///     ValueError: If the requested ranges are invalid or if there's an error reading the data.
    fn read_array<'py>(&self, py: Python<'_>, ranges: ArrayIndex) -> PyResult<OmFileTypedArray> {
        py.detach(|| {
            self.with_reader(|reader| {
                let array_reader = reader
                    .expect_array_with_io_sizes(65536, 512)
                    .map_err(|_| Self::only_arrays_error())?;
                let read_ranges = ranges.to_read_range(&self.shape)?;
                let dtype = array_reader.data_type();

                let untyped_py_array_or_error = match dtype {
                    OmDataType::None
                    | OmDataType::Int8
                    | OmDataType::Uint8
                    | OmDataType::Int16
                    | OmDataType::Uint16
                    | OmDataType::Int32
                    | OmDataType::Uint32
                    | OmDataType::Int64
                    | OmDataType::Uint64
                    | OmDataType::Float
                    | OmDataType::Double
                    | OmDataType::String => Err(Self::only_arrays_error()),
                    OmDataType::Int8Array => {
                        let array = read_squeezed_typed_array::<i8>(&array_reader, &read_ranges)?;
                        Ok(OmFileTypedArray::Int8(array))
                    }
                    OmDataType::Uint8Array => {
                        let array = read_squeezed_typed_array::<u8>(&array_reader, &read_ranges)?;
                        Ok(OmFileTypedArray::Uint8(array))
                    }
                    OmDataType::Int16Array => {
                        let array = read_squeezed_typed_array::<i16>(&array_reader, &read_ranges)?;
                        Ok(OmFileTypedArray::Int16(array))
                    }
                    OmDataType::Uint16Array => {
                        let array = read_squeezed_typed_array::<u16>(&array_reader, &read_ranges)?;
                        Ok(OmFileTypedArray::Uint16(array))
                    }
                    OmDataType::Int32Array => {
                        let array = read_squeezed_typed_array::<i32>(&array_reader, &read_ranges)?;
                        Ok(OmFileTypedArray::Int32(array))
                    }
                    OmDataType::Uint32Array => {
                        let array = read_squeezed_typed_array::<u32>(&array_reader, &read_ranges)?;
                        Ok(OmFileTypedArray::Uint32(array))
                    }
                    OmDataType::Int64Array => {
                        let array = read_squeezed_typed_array::<i64>(&array_reader, &read_ranges)?;
                        Ok(OmFileTypedArray::Int64(array))
                    }
                    OmDataType::Uint64Array => {
                        let array = read_squeezed_typed_array::<u64>(&array_reader, &read_ranges)?;
                        Ok(OmFileTypedArray::Uint64(array))
                    }
                    OmDataType::FloatArray => {
                        let array = read_squeezed_typed_array::<f32>(&array_reader, &read_ranges)?;
                        Ok(OmFileTypedArray::Float(array))
                    }
                    OmDataType::DoubleArray => {
                        let array = read_squeezed_typed_array::<f64>(&array_reader, &read_ranges)?;
                        Ok(OmFileTypedArray::Double(array))
                    }
                    OmDataType::StringArray => {
                        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "String Arrays not currently supported",
                        ))
                    }
                };

                let untyped_py_array = untyped_py_array_or_error?;

                return Ok(untyped_py_array);
            })
        })
    }

    fn __getitem__<'py>(&self, py: Python<'_>, ranges: ArrayIndex) -> PyResult<OmFileTypedArray> {
        self.read_array(py, ranges)
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

fn read_squeezed_typed_array<T: Element + OmFileArrayDataType + Clone + Zero>(
    reader: &OmFileArrayRs<impl OmFileReaderBackend>,
    read_ranges: &[Range<u64>],
) -> PyResult<ndarray::ArrayD<T>> {
    let array = reader
        .read::<T>(read_ranges)
        .map_err(convert_omfilesrs_error)?
        .squeeze();
    Ok(array)
}

/// Small helper function to get the correct shape of the data. We need to
/// be careful with scalars and groups!
fn get_shape_vec<Backend: OmFileReaderBackend>(reader: &OmFileReaderRs<Backend>) -> Vec<u64> {
    let reader = reader.expect_array();
    match reader {
        Ok(reader) => reader.get_dimensions().to_vec(),
        Err(_) => return vec![],
    }
}

fn get_chunk_shape<Backend: OmFileReaderBackend>(reader: &OmFileReaderRs<Backend>) -> Vec<u64> {
    let reader = reader.expect_array();
    match reader {
        Ok(reader) => reader.get_chunk_dimensions().to_vec(),
        Err(_) => return vec![],
    }
}

/// Concrete wrapper type for the backend implementation, delegating to the appropriate backend
enum ReaderBackendImpl {
    Mmap(MmapFile),
    FsSpec(FsSpecBackend),
}

impl OmFileReaderBackend for ReaderBackendImpl {
    // `Cow` can hold either a borrowed slice or an owned Vec, and it
    // also implements `Deref<Target=[u8]>`, `Send`, and `Sync`,
    // satisfying all our trait bounds.
    type Bytes<'a> = Cow<'a, [u8]>;

    // We must implement `get_bytes` manually to handle the type unification.
    fn get_bytes(&self, offset: u64, count: u64) -> Result<Self::Bytes<'_>, OmFilesError> {
        match self {
            ReaderBackendImpl::Mmap(backend) => {
                // The mmap backend returns a `&[u8]`. We wrap it in `Cow::Borrowed`.
                let slice = backend.get_bytes(offset, count)?;
                Ok(Cow::Borrowed(slice))
            }
            ReaderBackendImpl::FsSpec(backend) => {
                // The fsspec backend returns a `Vec<u8>`. We wrap it in `Cow::Owned`.
                let vec = backend.get_bytes(offset, count)?;
                Ok(Cow::Owned(vec))
            }
        }
    }

    delegate! {
        to match self {
            ReaderBackendImpl::Mmap(backend) => backend,
            ReaderBackendImpl::FsSpec(backend) => backend,
        } {
            fn count(&self) -> usize;
            fn prefetch_data(&self, offset: usize, count: usize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_index::IndexType;
    use crate::create_test_binary_file;

    #[test]
    fn test_read_simple_v3_data() -> Result<(), Box<dyn std::error::Error>> {
        create_test_binary_file!("read_test.om")?;
        let file_path = "test_files/read_test.om";
        Python::initialize();

        let reader = OmFileReader::from_path(file_path).unwrap();
        let ranges = ArrayIndex(vec![
            IndexType::Slice {
                start: Some(0),
                stop: Some(5),
                step: None,
            },
            IndexType::Slice {
                start: Some(0),
                stop: Some(5),
                step: None,
            },
        ]);
        let data = Python::attach(|py| {
            let data = reader.read_array(py, ranges).expect("Could not get item!");
            let data = match data {
                OmFileTypedArray::Float(data) => data,
                _ => panic!("Unexpected data type"),
            };
            data
        });

        assert_eq!(data.shape(), [5, 5]);

        let data = data.as_slice().expect("Could not convert to slice!");
        let expected_data = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ];
        assert_eq!(data, expected_data);

        Ok(())
    }
}
