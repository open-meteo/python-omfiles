use crate::{
    compression::PyCompressionType, errors::convert_omfilesrs_error,
    fsspec_backend::FsSpecWriterBackend, hierarchy::OmVariable,
};
use delegate::delegate;
use numpy::{
    dtype, Element, PyArrayDescr, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods,
};
use omfiles_rs::{
    traits::{OmFileArrayDataType, OmFileScalarDataType, OmFileWriterBackend},
    writer::{OmFileWriter as OmFileWriterRs, OmFileWriterArray},
    OmCompressionType, OmFilesError, OmOffsetSize,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyStopIteration, PyValueError},
    prelude::*,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::{
    fs::File,
    sync::{Mutex, PoisonError},
};

/// Helper to convert OmOffsetSize to OmVariable
fn to_variable(name: &str, os: OmOffsetSize) -> OmVariable {
    OmVariable {
        name: name.to_string(),
        offset: os.offset,
        size: os.size,
    }
}

/// All array element types supported by the writer.
enum OmElementType {
    Float32,
    Float64,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Uint64,
}

impl OmElementType {
    /// Resolve from a numpy `PyArrayDescr` (used by `write_array`).
    fn from_numpy_dtype(py: Python<'_>, d: &Bound<'_, PyArrayDescr>) -> PyResult<Self> {
        if d.is_equiv_to(&dtype::<f32>(py)) {
            Ok(Self::Float32)
        } else if d.is_equiv_to(&dtype::<f64>(py)) {
            Ok(Self::Float64)
        } else if d.is_equiv_to(&dtype::<i8>(py)) {
            Ok(Self::Int8)
        } else if d.is_equiv_to(&dtype::<u8>(py)) {
            Ok(Self::Uint8)
        } else if d.is_equiv_to(&dtype::<i16>(py)) {
            Ok(Self::Int16)
        } else if d.is_equiv_to(&dtype::<u16>(py)) {
            Ok(Self::Uint16)
        } else if d.is_equiv_to(&dtype::<i32>(py)) {
            Ok(Self::Int32)
        } else if d.is_equiv_to(&dtype::<u32>(py)) {
            Ok(Self::Uint32)
        } else if d.is_equiv_to(&dtype::<i64>(py)) {
            Ok(Self::Int64)
        } else if d.is_equiv_to(&dtype::<u64>(py)) {
            Ok(Self::Uint64)
        } else {
            Err(OmFileWriter::unsupported_array_type_error(d.clone()))
        }
    }

    /// Resolve from a dtype string like `"float32"` (used by `write_array_streaming`).
    fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            "int8" => Ok(Self::Int8),
            "uint8" => Ok(Self::Uint8),
            "int16" => Ok(Self::Int16),
            "uint16" => Ok(Self::Uint16),
            "int32" => Ok(Self::Int32),
            "uint32" => Ok(Self::Uint32),
            "int64" => Ok(Self::Int64),
            "uint64" => Ok(Self::Uint64),
            _ => Err(PyValueError::new_err(format!("Unsupported dtype: {}", s))),
        }
    }
}

/// Abstracts over the two write strategies (full array vs streaming iterator).
enum Feeder<'a, 'py> {
    /// Feeds an entire numpy array in a single `write_data` call.
    Full {
        data: &'a Bound<'py, PyUntypedArray>,
    },
    /// Feeds data chunk-by-chunk from a Python iterator.
    Streaming {
        py: Python<'py>,
        iter: Bound<'py, PyAny>,
    },
}

impl<'a, 'py> Feeder<'a, 'py> {
    fn feed<T: Element + OmFileArrayDataType>(
        self,
        w: &mut OmFileWriterArray<'_, T, WriterBackendImpl>,
    ) -> PyResult<()> {
        match self {
            Feeder::Full { data } => {
                let array = data.cast::<PyArrayDyn<T>>()?.readonly();
                w.write_data(array.as_array(), None, None)
                    .map_err(convert_omfilesrs_error)
            }
            Feeder::Streaming { py, iter } => loop {
                match iter.call_method0("__next__") {
                    Ok(item) => {
                        let array: PyReadonlyArrayDyn<'_, T> = item.extract()?;
                        w.write_data(array.as_array(), None, None)
                            .map_err(convert_omfilesrs_error)?;
                    }
                    Err(err) if err.is_instance_of::<PyStopIteration>(py) => break Ok(()),
                    Err(err) => break Err(err),
                }
            },
        }
    }
}

/// Resolved parameters shared by both `write_array` and `write_array_streaming`.
struct WriteArrayParams<'a> {
    name: &'a str,
    children: Vec<OmOffsetSize>,
    scale_factor: f32,
    add_offset: f32,
    compression: OmCompressionType,
}

impl<'a> WriteArrayParams<'a> {
    fn from_options(
        name: Option<&'a str>,
        children: Option<Vec<OmVariable>>,
        scale_factor: Option<f32>,
        add_offset: Option<f32>,
        compression: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            name: name.unwrap_or("data"),
            children: children
                .unwrap_or_default()
                .iter()
                .map(Into::into)
                .collect(),
            scale_factor: scale_factor.unwrap_or(1.0),
            add_offset: add_offset.unwrap_or(0.0),
            compression: compression
                .map(|s| PyCompressionType::from_str(s))
                .transpose()?
                .unwrap_or(PyCompressionType::PforDelta2d)
                .to_omfilesrs(),
        })
    }
}

/// A Python wrapper for the Rust OmFileWriter implementation.
#[gen_stub_pyclass]
#[pyclass]
pub struct OmFileWriter {
    writer: Mutex<Option<OmFileWriterRs<WriterBackendImpl>>>,
}

impl OmFileWriter {
    fn lock_error<T>(e: PoisonError<T>) -> PyErr {
        PyErr::new::<PyRuntimeError, _>(format!("Failed to acquire lock on writer: {}", e))
    }

    fn closed_error() -> PyErr {
        PyErr::new::<PyValueError, _>("I/O operation on closed writer")
    }

    fn unsupported_array_type_error(dtype: Bound<'_, PyArrayDescr>) -> PyErr {
        let type_name = dtype
            .typeobj()
            .name()
            .map(|s| s.to_string())
            .unwrap_or("unknown type".to_string());
        PyErr::new::<PyValueError, _>(format!("Unsupported array data type: {}", type_name))
    }

    fn unsupported_scalar_type_error(dtype: Bound<'_, pyo3::types::PyType>) -> PyErr {
        let type_name = dtype
            .name()
            .map(|s| s.to_string())
            .unwrap_or("unknown type".to_string());
        PyErr::new::<PyValueError, _>(format!("Unsupported scalar data type: {}", type_name))
    }

    /// Helper method for safe writer access.
    fn with_writer<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut OmFileWriterRs<WriterBackendImpl>) -> PyResult<R>,
    {
        let mut guard = self.writer.lock().map_err(|e| Self::lock_error(e))?;
        match guard.as_mut() {
            Some(writer) => f(writer),
            None => Err(Self::closed_error()),
        }
    }

    /// Unified 10-way type dispatch.
    ///
    /// Resolves `element_type` to a concrete `T`, prepares a typed array writer,
    /// feeds data, finalizes, and registers the result as a named variable.
    /// Both `write_array` and `write_array_streaming` delegate here.
    fn write_array_dispatched(
        &self,
        element_type: OmElementType,
        dimensions: Vec<u64>,
        chunks: Vec<u64>,
        params: &WriteArrayParams<'_>,
        feeder: Feeder<'_, '_>,
    ) -> PyResult<OmVariable> {
        self.with_writer(|writer| {
            macro_rules! dispatch {
                ($($variant:ident => $T:ty),+ $(,)?) => {
                    match element_type {
                        $(OmElementType::$variant => {
                            let mut w = writer
                                .prepare_array::<$T>(
                                    dimensions, chunks, params.compression,
                                    params.scale_factor, params.add_offset,
                                )
                                .map_err(convert_omfilesrs_error)?;
                            feeder.feed::<$T>(&mut w)?;
                            w.finalize()
                        }),+
                    }
                };
            }

            let array_meta = dispatch! {
                Float32 => f32,
                Float64 => f64,
                Int8    => i8,
                Uint8   => u8,
                Int16   => i16,
                Uint16  => u16,
                Int32   => i32,
                Uint32  => u32,
                Int64   => i64,
                Uint64  => u64,
            };

            writer
                .write_array(array_meta, params.name, &params.children)
                .map_err(convert_omfilesrs_error)
                .map(|os| to_variable(params.name, os))
        })
    }

    fn store_scalar<T: OmFileScalarDataType + 'static>(
        &self,
        value: T,
        name: &str,
        children: &[OmOffsetSize],
    ) -> PyResult<OmVariable> {
        self.with_writer(|writer| {
            writer
                .write_scalar(value, name, children)
                .map_err(convert_omfilesrs_error)
                .map(|os| to_variable(name, os))
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl OmFileWriter {
    /// Initialize an OmFileWriter.
    ///
    /// Args:
    ///     file_path: Path where the .om file will be created
    #[new]
    fn new(file_path: &str) -> PyResult<Self> {
        Self::at_path(file_path)
    }

    /// Initialize an OmFileWriter to write to a file at the specified path.
    ///
    /// Args:
    ///     path: Path where the .om file will be created
    ///
    /// Returns:
    ///     OmFileWriter: A new writer instance
    #[staticmethod]
    fn at_path(path: &str) -> PyResult<Self> {
        let file_handle = WriterBackendImpl::File(File::create(path)?);
        let writer = OmFileWriterRs::new(file_handle, 8 * 1024);
        Ok(Self {
            writer: Mutex::new(Some(writer)),
        })
    }

    /// Create an OmFileWriter from a fsspec filesystem object.
    ///
    /// Args:
    ///     fs_obj: A fsspec filesystem object that supports write operations
    ///     path: The path to the file within the file system
    ///
    /// Returns:
    ///     OmFileWriter: A new writer instance
    #[staticmethod]
    fn from_fsspec(fs_obj: Py<PyAny>, path: String) -> PyResult<Self> {
        let fsspec_backend = WriterBackendImpl::FsSpec(FsSpecWriterBackend::new(fs_obj, path)?);
        let writer = OmFileWriterRs::new(fsspec_backend, 8 * 1024);
        Ok(Self {
            writer: Mutex::new(Some(writer)),
        })
    }

    /// Finalize and close the .om file by writing the trailer with the root variable.
    ///
    /// Args:
    ///     root_variable (:py:data:`omfiles.OmVariable`): The OmVariable that serves as the root/entry point of the file hierarchy.
    ///                    All other variables should be accessible through this root variable.
    ///
    /// Returns:
    ///     None on success.
    ///
    /// Raises:
    ///     ValueError: If the writer has already been closed
    ///     RuntimeError: If a thread lock error occurs or if there's an error writing to the file
    fn close(&mut self, root_variable: OmVariable) -> PyResult<()> {
        let mut guard = self.writer.lock().map_err(|e| Self::lock_error(e))?;

        if let Some(writer) = guard.as_mut() {
            writer
                .write_trailer(root_variable.into())
                .map_err(convert_omfilesrs_error)?;
            // Take ownership and drop to ensure proper file closure
            guard.take();
        } else {
            return Err(Self::closed_error());
        }

        Ok(())
    }

    /// Check if the writer is closed.
    #[getter]
    fn closed(&self) -> PyResult<bool> {
        let guard = self.writer.lock().map_err(|e| Self::lock_error(e))?;
        Ok(guard.is_none())
    }

    /// Write a numpy array to the .om file with specified chunking and scaling parameters.
    ///
    /// ``scale_factor`` and ``add_offset`` are only respected and required for float32
    /// and float64 data types. Recommended compression is "pfor_delta_2d" as it achieves
    /// best compression ratios (on spatio-temporally correlated data), but it will be lossy
    /// when applied to floating-point data types because of the scale-offset encoding applied
    /// to convert float values to integer values.
    ///
    /// Args:
    ///     data: Input array to be written. Supported dtypes are:
    ///           float32, float64, int8, uint8, int16, uint16, int32, uint32, int64, uint64,
    ///     chunks: Chunk sizes for each dimension of the array
    ///     scale_factor: Scale factor for data compression (default: 1.0)
    ///     add_offset: Offset value for data compression (default: 0.0)
    ///     compression: Compression algorithm to use (default: "pfor_delta_2d")
    ///                  Supported values: "pfor_delta_2d", "fpx_xor_2d", "pfor_delta_2d_int16", "pfor_delta_2d_int16_logarithmic"
    ///     name: Name of the variable to be written (default: "data")
    ///     children: List of child variables (default: [])
    ///
    /// Returns:
    ///     :py:data:`omfiles.OmVariable` representing the written group in the file structure
    ///
    /// Raises:
    ///     ValueError: If the data type is unsupported or if parameters are invalid
    #[pyo3(
        text_signature = "(data, chunks, scale_factor=1.0, add_offset=0.0, compression='pfor_delta_2d', name='data', children=[])",
        signature = (data, chunks, scale_factor=None, add_offset=None, compression=None, name=None, children=None)
    )]
    fn write_array(
        &mut self,
        data: &Bound<'_, PyUntypedArray>,
        chunks: Vec<u64>,
        scale_factor: Option<f32>,
        add_offset: Option<f32>,
        compression: Option<&str>,
        name: Option<&str>,
        children: Option<Vec<OmVariable>>,
    ) -> PyResult<OmVariable> {
        let params =
            WriteArrayParams::from_options(name, children, scale_factor, add_offset, compression)?;
        let element_type = OmElementType::from_numpy_dtype(data.py(), &data.dtype())?;
        let dimensions = data.shape().iter().map(|x| *x as u64).collect();
        let feeder = Feeder::Full { data };

        self.write_array_dispatched(element_type, dimensions, chunks, &params, feeder)
    }

    /// Write an array to the .om file by streaming chunks from a Python iterator.
    ///
    /// This method is designed for writing large arrays that do not fit in memory.
    /// Instead of providing the full array, you provide the full array dimensions
    /// and an iterator that yields numpy array chunks.
    ///
    /// Chunks MUST be yielded in row-major order (C-order) of the chunk grid.
    /// Each chunk's shape determines how many internal file chunks it covers.
    ///
    /// Args:
    ///     dimensions: Shape of the full array (e.g., [1000, 2000])
    ///     chunks: Chunk sizes for each dimension (e.g., [100, 200])
    ///     chunk_iterator: Python iterable yielding numpy arrays, one per chunk region
    ///     dtype: String name of the numpy dtype (e.g., "float32", "int64")
    ///     scale_factor: Scale factor for data compression (default: 1.0)
    ///     add_offset: Offset value for data compression (default: 0.0)
    ///     compression: Compression algorithm to use (default: "pfor_delta_2d")
    ///     name: Name of the variable (default: "data")
    ///     children: List of child variables (default: [])
    ///
    /// Returns:
    ///     :py:data:`omfiles.OmVariable` representing the written array in the file structure
    ///
    /// Raises:
    ///     ValueError: If the dtype is unsupported or parameters are invalid
    ///     RuntimeError: If there's an error during compression or I/O
    #[pyo3(
        text_signature = "(dimensions, chunks, chunk_iterator, dtype, scale_factor=1.0, add_offset=0.0, compression='pfor_delta_2d', name='data', children=[])",
        signature = (dimensions, chunks, chunk_iterator, dtype, scale_factor=None, add_offset=None, compression=None, name=None, children=None)
    )]
    fn write_array_streaming(
        &mut self,
        py: Python<'_>,
        dimensions: Vec<u64>,
        chunks: Vec<u64>,
        chunk_iterator: &Bound<'_, PyAny>,
        dtype: &str,
        scale_factor: Option<f32>,
        add_offset: Option<f32>,
        compression: Option<&str>,
        name: Option<&str>,
        children: Option<Vec<OmVariable>>,
    ) -> PyResult<OmVariable> {
        let params =
            WriteArrayParams::from_options(name, children, scale_factor, add_offset, compression)?;
        let element_type = OmElementType::from_str(dtype)?;
        let iter = chunk_iterator.call_method0("__iter__")?;
        let feeder = Feeder::Streaming { py, iter };

        self.write_array_dispatched(element_type, dimensions, chunks, &params, feeder)
    }

    /// Write a scalar value to the .om file.
    ///
    /// Args:
    ///     value: Scalar value to write. Supported types are:
    ///            int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, String
    ///     name: Name of the scalar variable
    ///     children: List of child variables (default: None)
    ///
    /// Returns:
    ///     :py:data:`omfiles.OmVariable` representing the written scalar in the file structure
    ///
    /// Raises:
    ///     ValueError: If the value type is unsupported (e.g., booleans)
    ///     RuntimeError: If there's an error writing to the file
    #[pyo3(
        text_signature = "(value, name, children=None)",
        signature = (value, name, children=None)
    )]
    fn write_scalar(
        &mut self,
        value: &Bound<PyAny>,
        name: &str,
        children: Option<Vec<OmVariable>>,
    ) -> PyResult<OmVariable> {
        let children: Vec<OmOffsetSize> = children
            .unwrap_or_default()
            .iter()
            .map(Into::into)
            .collect();

        let py = value.py();

        // make an instance check against numpy scalar types
        macro_rules! check_numpy_type {
            ($numpy:expr, $type_name:literal, $rust_type:ty) => {
                if let Ok(numpy_type) = $numpy.getattr($type_name) {
                    if value.is_instance(&numpy_type)? {
                        let scalar_value: $rust_type = value.call_method0("item")?.extract()?;
                        return self.store_scalar(scalar_value, name, &children);
                    }
                }
            };
        }

        // Try to import numpy and check for numpy scalar types
        if let Ok(numpy) = py.import("numpy") {
            check_numpy_type!(numpy, "int8", i8);
            check_numpy_type!(numpy, "uint8", u8);
            check_numpy_type!(numpy, "int16", i16);
            check_numpy_type!(numpy, "uint16", u16);
            check_numpy_type!(numpy, "int32", i32);
            check_numpy_type!(numpy, "uint32", u32);
            check_numpy_type!(numpy, "int64", i64);
            check_numpy_type!(numpy, "uint64", u64);
            check_numpy_type!(numpy, "float32", f32);
            check_numpy_type!(numpy, "float64", f64);
        }

        // Fall back to Python built-in types
        let result = if let Ok(_value) = value.extract::<String>() {
            self.store_scalar(value.to_string(), name, &children)?
        } else if let Ok(value) = value.extract::<f64>() {
            self.store_scalar(value, name, &children)?
        } else if let Ok(value) = value.extract::<f32>() {
            self.store_scalar(value, name, &children)?
        } else if let Ok(value) = value.extract::<i64>() {
            self.store_scalar(value, name, &children)?
        } else if let Ok(value) = value.extract::<i32>() {
            self.store_scalar(value, name, &children)?
        } else if let Ok(value) = value.extract::<i16>() {
            self.store_scalar(value, name, &children)?
        } else if let Ok(value) = value.extract::<i8>() {
            self.store_scalar(value, name, &children)?
        } else if let Ok(value) = value.extract::<u64>() {
            self.store_scalar(value, name, &children)?
        } else if let Ok(value) = value.extract::<u32>() {
            self.store_scalar(value, name, &children)?
        } else if let Ok(value) = value.extract::<u16>() {
            self.store_scalar(value, name, &children)?
        } else if let Ok(value) = value.extract::<u8>() {
            self.store_scalar(value, name, &children)?
        } else {
            return Err(Self::unsupported_scalar_type_error(value.get_type()));
        };
        Ok(result)
    }

    /// Create a new group in the .om file.
    ///
    /// This is essentially a variable with no data, which serves as a container for other variables.
    ///
    /// Args:
    ///     name: Name of the group
    ///     children: List of child variables
    ///
    /// Returns:
    ///     :py:data:`omfiles.OmVariable` representing the written group in the file structure
    ///
    /// Raises:
    ///     RuntimeError: If there's an error writing to the file
    fn write_group(&mut self, name: &str, children: Vec<OmVariable>) -> PyResult<OmVariable> {
        let children: Vec<OmOffsetSize> = children.iter().map(Into::into).collect();

        self.with_writer(|writer| {
            writer
                .write_none(name, &children)
                .map_err(convert_omfilesrs_error)
                .map(|os| to_variable(name, os))
        })
    }
}

/// Concrete wrapper type for the backend implementation, delegating to the appropriate backend.
enum WriterBackendImpl {
    File(File),
    FsSpec(FsSpecWriterBackend),
}

impl OmFileWriterBackend for WriterBackendImpl {
    delegate! {
        to match self {
            WriterBackendImpl::File(backend) => backend,
            WriterBackendImpl::FsSpec(backend) => backend,
        } {
            fn write(&mut self, data: &[u8]) -> Result<(), OmFilesError>;
            fn synchronize(&self) -> Result<(), OmFilesError>;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::ArrayD, PyArrayDyn, PyArrayMethods};
    use std::fs;

    #[test]
    fn test_write_array() -> Result<(), Box<dyn std::error::Error>> {
        Python::initialize();

        Python::attach(|py| -> Result<(), Box<dyn std::error::Error>> {
            // numpy is not happy if we import it when modifying the PYTHONPATH to directly include numpy
            // because of broken handling of virtual environments in pyo3, we skip the test on import failure
            if let Err(e) = py.import("numpy") {
                eprintln!(
                    "Skipping test_write_array: could not import numpy ({:?})",
                    e
                );
                return Ok(()); // Skip the test
            }

            // Test parameters
            let file_path = "test_data.om";
            let dimensions = vec![10, 20];
            let chunks = vec![5u64, 5];

            // Create test data
            let data = ArrayD::from_shape_fn(dimensions, |idx| (idx[0] + idx[1]) as f32);
            let py_array = PyArrayDyn::from_array(py, &data);

            let mut file_writer = OmFileWriter::new(file_path).unwrap();

            // Write data
            let result = file_writer.write_array(
                py_array.as_untyped(),
                chunks,
                None,
                None,
                None,
                None,
                None,
            );

            assert!(result.is_ok());
            assert!(fs::metadata(file_path).is_ok());

            // Clean up
            fs::remove_file(file_path).unwrap();
            Ok(())
        })?;

        Ok(())
    }

    #[test]
    fn test_fsspec_writer() -> Result<(), Box<dyn std::error::Error>> {
        Python::initialize();

        Python::attach(|py| -> Result<(), Box<dyn std::error::Error>> {
            let fsspec = py.import("fsspec")?;
            let fs = fsspec.call_method1("filesystem", ("memory",))?;

            let _writer = OmFileWriter::from_fsspec(fs.into(), "test_file.om".to_string())?;

            Ok(())
        })?;

        Ok(())
    }
}
