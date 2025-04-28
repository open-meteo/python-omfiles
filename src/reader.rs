use crate::{
    array_index::ArrayIndex, data_type::get_numpy_dtype, errors::convert_omfilesrs_error,
    fsspec_backend::FsSpecBackend, hierarchy::OmVariable, typed_array::OmFileTypedArray,
};
use delegate::delegate;
use num_traits::Zero;
use numpy::{
    ndarray::{self},
    Element, PyArrayDescr,
};
use omfiles_rs::{
    backend::{
        backends::OmFileReaderBackend,
        mmapfile::{MmapFile, Mode},
    },
    core::data_types::{DataType, OmFileArrayDataType, OmFileScalarDataType},
    io::reader::OmFileReader,
};
use pyo3::{prelude::*, BoundObject};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::{
    collections::HashMap,
    fs::File,
    ops::Range,
    sync::{Arc, RwLock},
};

/// An OmFilePyReader class for reading .om files.
///
/// A reader object can have an arbitrary number of child readers, each representing
/// a multidimensional variable or a scalar variable (an attribute). Thus, this class
/// implements a tree-like structure for multi-dimensional data access.
///
/// Variables in OM-Files do not have named dimensions! That means you have to know
/// what the dimensions represent in advance or you need to explicitly encode them as
/// some kind of attribute.
///
/// Most likely we will adopt the xarray convention which is implemented for zarr
/// which requires multi-dimensional variables to have an attribute called
/// _ARRAY_DIMENSIONS that contains a list of dimension names.
/// These dimension names should be encoded somewhere in the .om file hierarchy
/// as attributes.
///
/// Therefore, it might be useful to differentiate in some way between
/// hdf5-like groups and datasets/n-dim arrays in an om-file.
///
/// Group: Can contain datasets/arrays, attributes, and other groups.
/// Dataset: Data-array, might have associated attributes.
/// Attribute: A named data value associated with a group or dataset.
#[gen_stub_pyclass]
#[pyclass(module = "omfiles.omfiles")]
pub struct OmFilePyReader {
    /// The reader is stored in an Option to be able to properly close it,
    /// particularly when working with memory-mapped files.
    /// The RwLock is used to allow multiple readers to access the reader
    /// concurrently, but only one writer to close it.
    reader: RwLock<Option<OmFileReader<BackendImpl>>>,
    /// Get the shape of the data stored in the .om file.
    ///
    /// Returns
    /// -------
    /// list
    ///     List containing the dimensions of the data
    #[pyo3(get)]
    shape: Vec<u64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl OmFilePyReader {
    /// Initialize an OmFilePyReader from a file path or fsspec file object.
    ///
    /// Parameters
    /// ----------
    /// source : str or fsspec.core.OpenFile
    ///     Path to the .om file to read or a fsspec file object
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the file cannot be opened or is invalid
    #[new]
    fn new(source: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
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

    /// Create an OmFilePyReader from a file path.
    ///
    /// Parameters
    /// ----------
    /// file_path : str
    ///     Path to the .om file to read
    ///
    /// Returns
    /// -------
    /// OmFilePyReader
    ///     OmFilePyReader instance
    #[staticmethod]
    fn from_path(file_path: &str) -> PyResult<Self> {
        let file_handle = File::open(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let backend = BackendImpl::Mmap(MmapFile::new(file_handle, Mode::ReadOnly)?);
        let reader = OmFileReader::new(Arc::new(backend)).map_err(convert_omfilesrs_error)?;
        let shape = get_shape_vec(&reader);

        Ok(Self {
            reader: RwLock::new(Some(reader)),
            shape,
        })
    }

    /// Create an OmFilePyReader from a fsspec fs object.
    ///
    /// Parameters
    /// ----------
    /// fs_obj : fsspec.spec.AbstractFileSystem
    ///     A fsspec file system object which needs to have the methods `cat_file` and `size`.
    /// path : str
    ///     The path to the file within the file system.
    ///
    /// Returns
    /// -------
    /// OmFilePyReader
    ///     A new reader instance
    #[staticmethod]
    fn from_fsspec(fs_obj: PyObject, path: String) -> PyResult<Self> {
        Python::with_gil(|py| {
            let bound_object = fs_obj.bind(py);

            if !bound_object.hasattr("cat_file")? || !bound_object.hasattr("size")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Input must be a valid fsspec file object with read, seek methods and fs attribute",
                    ));
            }

            let backend = BackendImpl::FsSpec(FsSpecBackend::new(fs_obj, path)?);
            let reader = OmFileReader::new(Arc::new(backend)).map_err(convert_omfilesrs_error)?;
            let shape = get_shape_vec(&reader);

            Ok(Self {
                reader: RwLock::new(Some(reader)),
                shape,
            })
        })
    }

    /// Get a mapping of variable names to their file offsets and sizes.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary mapping variable names to their metadata
    fn get_flat_variable_metadata(&self) -> PyResult<HashMap<String, OmVariable>> {
        self.with_reader(|reader| {
            let metadata = reader.get_flat_variable_metadata();
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

    /// Initialize a new OmFilePyReader from a child variable.
    ///
    /// Parameters
    /// ----------
    /// variable : OmVariable
    ///     Variable metadata to create a new reader from
    ///
    /// Returns
    /// -------
    /// OmFilePyReader
    ///     A new reader for the specified variable
    fn init_from_variable(&self, variable: OmVariable) -> PyResult<Self> {
        self.with_reader(|reader| {
            let child_reader = reader
                .init_child_from_offset_size(variable.into())
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
    /// Returns
    /// -------
    /// OmFilePyReader
    ///     Self for use in context manager
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the reader is already closed
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Exit a context manager block, closing the reader.
    ///
    /// Parameters
    /// ----------
    /// _exc_type : type, optional
    ///     The exception type, if an exception was raised
    /// _exc_value : Exception, optional
    ///     The exception value, if an exception was raised
    /// _traceback : traceback, optional
    ///     The traceback, if an exception was raised
    ///
    /// Returns
    /// -------
    /// bool
    ///     False (exceptions are not suppressed)
    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        _exc_type: Option<PyObject>,
        _exc_value: Option<PyObject>,
        _traceback: Option<PyObject>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }

    /// Check if the reader is closed.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the reader is closed, False otherwise
    #[getter]
    fn closed(&self) -> PyResult<bool> {
        let guard = self.reader.try_read().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock error: {}", e))
        })?;

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
        let mut guard = self.reader.try_write().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock error: {}", e))
        })?;

        // takes the reader, leaving None in the RwLock
        if let Some(reader) = guard.take() {
            // Extract the backend before dropping reader
            if let Ok(backend) = Arc::try_unwrap(reader.backend) {
                match backend {
                    BackendImpl::FsSpec(fs_backend) => {
                        fs_backend.close()?;
                    }
                    BackendImpl::Mmap(_) => {
                        // Will be dropped automatically
                    }
                }
            }
            // The reader is dropped here when it goes out of scope
        }

        Ok(())
    }

    /// Check if the variable is a scalar.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the variable is a scalar, False otherwise
    #[getter]
    fn is_scalar(&self) -> PyResult<bool> {
        self.with_reader(|reader| {
            let data_type = reader.data_type() as u8;
            Ok(data_type > (DataType::None as u8) && data_type < (DataType::Int8Array as u8))
        })
    }

    /// Check if the variable is a group (a variable with data type None).
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the variable is a group, False otherwise
    #[getter]
    fn is_group(&self) -> PyResult<bool> {
        self.with_reader(|reader| Ok(reader.data_type() == DataType::None))
    }

    /// Get the data type of the data stored in the .om file.
    ///
    /// Returns
    /// -------
    /// numpy.dtype
    ///     Numpy data type of the data
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDescr>> {
        self.with_reader(|reader| get_numpy_dtype(py, &reader.data_type()))
    }

    /// Get the name of the variable stored in the .om file.
    ///
    /// Returns
    /// -------
    /// str
    ///     Name of the variable or an empty string if not available
    #[getter]
    fn name(&self) -> PyResult<String> {
        self.with_reader(|reader| Ok(reader.get_name().unwrap_or("".to_string())))
    }

    /// Read data from the open variable.om file using numpy-style indexing.
    /// Currently only slices with step 1 are supported.
    ///
    /// The returned array will have singleton dimensions removed (squeezed).
    /// For example, if you index a 3D array with [1,:,2], the result will
    /// be a 1D array since dimensions 0 and 2 have size 1.
    ///
    /// Parameters
    /// ----------
    /// ranges : array-like
    ///     Index expression that can be either a single slice/integer
    ///     or a tuple of slices/integers for multi-dimensional access.
    ///     Supports NumPy basic indexing including:
    ///     - Integers (e.g., a[1,2])
    ///     - Slices (e.g., a[1:10])
    ///     - Ellipsis (...)
    ///     - None/newaxis
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     NDArray containing the requested data with squeezed singleton dimensions.
    ///     The data type of the array matches the data type stored in the file
    ///     (int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, or float64).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the requested ranges are invalid or if there's an error reading the data
    fn __getitem__<'py>(&self, ranges: ArrayIndex) -> PyResult<OmFileTypedArray> {
        let io_size_max = None;
        let io_size_merge = None;
        let read_ranges = ranges.to_read_range(&self.shape)?;

        self.with_reader(|reader| {
            let dtype = reader.data_type();

            let scalar_error = PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Scalar data types are not supported",
            );

            let untyped_py_array_or_error = match dtype {
                DataType::None => Err(scalar_error),
                DataType::Int8 => Err(scalar_error),
                DataType::Uint8 => Err(scalar_error),
                DataType::Int16 => Err(scalar_error),
                DataType::Uint16 => Err(scalar_error),
                DataType::Int32 => Err(scalar_error),
                DataType::Uint32 => Err(scalar_error),
                DataType::Int64 => Err(scalar_error),
                DataType::Uint64 => Err(scalar_error),
                DataType::Float => Err(scalar_error),
                DataType::Double => Err(scalar_error),
                DataType::String => Err(scalar_error),
                DataType::Int8Array => {
                    let array = read_squeezed_typed_array::<i8>(
                        &reader,
                        &read_ranges,
                        io_size_max,
                        io_size_merge,
                    )?;
                    Ok(OmFileTypedArray::Int8(array))
                }
                DataType::Uint8Array => {
                    let array = read_squeezed_typed_array::<u8>(
                        &reader,
                        &read_ranges,
                        io_size_max,
                        io_size_merge,
                    )?;
                    Ok(OmFileTypedArray::Uint8(array))
                }
                DataType::Int16Array => {
                    let array = read_squeezed_typed_array::<i16>(
                        &reader,
                        &read_ranges,
                        io_size_max,
                        io_size_merge,
                    )?;
                    Ok(OmFileTypedArray::Int16(array))
                }
                DataType::Uint16Array => {
                    let array = read_squeezed_typed_array::<u16>(
                        &reader,
                        &read_ranges,
                        io_size_max,
                        io_size_merge,
                    )?;
                    Ok(OmFileTypedArray::Uint16(array))
                }
                DataType::Int32Array => {
                    let array = read_squeezed_typed_array::<i32>(
                        &reader,
                        &read_ranges,
                        io_size_max,
                        io_size_merge,
                    )?;
                    Ok(OmFileTypedArray::Int32(array))
                }
                DataType::Uint32Array => {
                    let array = read_squeezed_typed_array::<u32>(
                        &reader,
                        &read_ranges,
                        io_size_max,
                        io_size_merge,
                    )?;
                    Ok(OmFileTypedArray::Uint32(array))
                }
                DataType::Int64Array => {
                    let array = read_squeezed_typed_array::<i64>(
                        &reader,
                        &read_ranges,
                        io_size_max,
                        io_size_merge,
                    )?;
                    Ok(OmFileTypedArray::Int64(array))
                }
                DataType::Uint64Array => {
                    let array = read_squeezed_typed_array::<u64>(
                        &reader,
                        &read_ranges,
                        io_size_max,
                        io_size_merge,
                    )?;
                    Ok(OmFileTypedArray::Uint64(array))
                }
                DataType::FloatArray => {
                    let array = read_squeezed_typed_array::<f32>(
                        &reader,
                        &read_ranges,
                        io_size_max,
                        io_size_merge,
                    )?;
                    Ok(OmFileTypedArray::Float(array))
                }
                DataType::DoubleArray => {
                    let array = read_squeezed_typed_array::<f64>(
                        &reader,
                        &read_ranges,
                        io_size_max,
                        io_size_merge,
                    )?;
                    Ok(OmFileTypedArray::Double(array))
                }
                DataType::StringArray => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "String Arrays not currently supported",
                )),
            };

            let untyped_py_array = untyped_py_array_or_error?;

            return Ok(untyped_py_array);
        })
    }

    /// Get the scalar value of the variable.
    ///
    /// Returns
    /// -------
    /// object
    ///     The scalar value as a Python object (str, int, or float)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the variable is not a scalar
    fn get_scalar(&self) -> PyResult<PyObject> {
        self.with_reader(|reader| {
            Python::with_gil(|py| match reader.data_type() {
                DataType::Int8 => self.read_scalar_value::<i8>(py),
                DataType::Uint8 => self.read_scalar_value::<u8>(py),
                DataType::Int16 => self.read_scalar_value::<i16>(py),
                DataType::Uint16 => self.read_scalar_value::<u16>(py),
                DataType::Int32 => self.read_scalar_value::<i32>(py),
                DataType::Uint32 => self.read_scalar_value::<u32>(py),
                DataType::Int64 => self.read_scalar_value::<i64>(py),
                DataType::Uint64 => self.read_scalar_value::<u64>(py),
                DataType::Float => self.read_scalar_value::<f32>(py),
                DataType::Double => self.read_scalar_value::<f64>(py),
                DataType::String => self.read_scalar_value::<String>(py),
                _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Data type is not scalar",
                )),
            })
        })
    }
}

impl OmFilePyReader {
    fn with_reader<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&OmFileReader<BackendImpl>) -> PyResult<R>,
    {
        let guard = self.reader.try_read().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Trying to read from a reader which is being closed",
            )
        })?;
        if let Some(reader) = &*guard {
            f(reader)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "I/O operation on closed reader or file",
            ))
        }
    }

    fn read_scalar_value<'py, T>(&self, py: Python<'py>) -> PyResult<PyObject>
    where
        T: OmFileScalarDataType + IntoPyObject<'py>,
    {
        self.with_reader(|reader| {
            let value = reader.read_scalar::<T>();

            value
                .into_pyobject(py)
                .map(BoundObject::into_any)
                .map(BoundObject::unbind)
                .map_err(Into::into)
        })
    }
}

fn read_squeezed_typed_array<T: Element + OmFileArrayDataType + Clone + Zero>(
    reader: &OmFileReader<impl OmFileReaderBackend>,
    read_ranges: &[Range<u64>],
    io_size_max: Option<u64>,
    io_size_merge: Option<u64>,
) -> PyResult<ndarray::ArrayD<T>> {
    let array = reader
        .read::<T>(read_ranges, io_size_max, io_size_merge)
        .map_err(convert_omfilesrs_error)?
        .squeeze();
    Ok(array)
}

/// Small helper function to get the correct shape of the data. We need to
/// be careful with scalars and groups!
fn get_shape_vec<Backend>(reader: &OmFileReader<Backend>) -> Vec<u64> {
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

/// Concrete wrapper type for the backend implementation, delegating to the appropriate backend
enum BackendImpl {
    Mmap(MmapFile),
    FsSpec(FsSpecBackend),
}

impl OmFileReaderBackend for BackendImpl {
    delegate! {
        to match self {
            BackendImpl::Mmap(backend) => backend,
            BackendImpl::FsSpec(backend) => backend,
        } {
            fn count(&self) -> usize;
            fn needs_prefetch(&self) -> bool;
            fn prefetch_data(&self, offset: usize, count: usize);
            fn pre_read(&self, offset: usize, count: usize) -> Result<(), omfiles_rs::errors::OmFilesRsError>;
            fn get_bytes(&self, offset: u64, count: u64) -> Result<&[u8], omfiles_rs::errors::OmFilesRsError>;
            fn get_bytes_owned(&self, offset: u64, count: u64) -> Result<Vec<u8>, omfiles_rs::errors::OmFilesRsError>;
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
        pyo3::prepare_freethreaded_python();

        let reader = OmFilePyReader::from_path(file_path).unwrap();
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
        let data = reader.__getitem__(ranges).expect("Could not get item!");
        let data = match data {
            OmFileTypedArray::Float(data) => data,
            _ => panic!("Unexpected data type"),
        };

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
