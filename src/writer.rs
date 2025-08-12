use crate::{
    compression::PyCompressionType, errors::convert_omfilesrs_error,
    fsspec_backend::FsSpecWriterBackend, hierarchy::OmVariable,
};
use numpy::{
    dtype, Element, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn,
    PyUntypedArray, PyUntypedArrayMethods,
};
use omfiles_rs::{
    core::{
        compression::CompressionType,
        data_types::{OmFileArrayDataType, OmFileScalarDataType},
    },
    errors::OmFilesRsError,
    io::writer::{OmFileWriter as OmFileWriterRs, OmFileWriterArrayFinalized, OmOffsetSize},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::{fs::File, sync::Mutex};

enum WriterBackend {
    File(OmFileWriterRs<File>),
    FsSpec(OmFileWriterRs<FsSpecWriterBackend>),
}

#[gen_stub_pyclass]
#[pyclass(module = "omfiles.omfiles")]
/// A Python wrapper for the Rust OmFileWriter implementation.
pub struct OmFileWriter {
    writer: Mutex<Option<WriterBackend>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl OmFileWriter {
    #[new]
    /// Initialize an OmFileWriter.
    ///
    /// Args:
    ///     file_path: Path where the .om file will be created
    ///
    /// Raises:
    /// OSError: If the file cannot be created
    fn new(file_path: &str) -> PyResult<Self> {
        let file_handle = File::create(file_path)?;
        let writer = OmFileWriterRs::new(file_handle, 8 * 1024);
        Ok(Self {
            writer: Mutex::new(Some(WriterBackend::File(writer))),
        })
    }

    #[staticmethod]
    #[pyo3(
        text_signature = "(fs_obj, path, /)",
        signature = (fs_obj, path)
    )]
    /// Create an OmFileWriter from a fsspec filesystem object.
    ///
    /// Args:
    ///     fs_obj: A fsspec filesystem object that supports write operations
    ///     path: The path to the file within the file system
    ///
    /// Returns:
    ///     OmFileWriter: A new writer instance
    fn from_fsspec(fs_obj: PyObject, path: String) -> PyResult<Self> {
        let fsspec_backend = FsSpecWriterBackend::new(fs_obj, path)?;
        let writer = OmFileWriterRs::new(fsspec_backend, 8 * 1024);
        Ok(Self {
            writer: Mutex::new(Some(WriterBackend::FsSpec(writer))),
        })
    }

    #[pyo3(
            text_signature = "(root_variable, /)",
            signature = (root_variable)
        )]
    /// Finalize and close the .om file by writing the trailer with the root variable.
    ///
    /// Args:
    ///     root_variable: The OmVariable that serves as the root/entry point of the file hierarchy.
    ///                    All other variables should be accessible through this root variable.
    ///
    /// Returns:
    ///     None on success.
    ///
    /// Raises:
    ///     ValueError: If the writer has already been closed
    ///     RuntimeError: If a thread lock error occurs or if there's an error writing to the file
    fn close(&mut self, root_variable: OmVariable) -> PyResult<()> {
        let mut guard = self.writer.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock error: {}", e))
        })?;

        if let Some(writer) = guard.as_mut() {
            let result = match writer {
                WriterBackend::File(w) => w.write_trailer(root_variable.into()),
                WriterBackend::FsSpec(w) => w.write_trailer(root_variable.into()),
            };
            result.map_err(convert_omfilesrs_error)?;
            // Take ownership and drop to ensure proper file closure
            guard.take();
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "I/O operation on closed writer or file",
            ));
        }

        Ok(())
    }

    #[getter]
    /// Check if the writer is closed.
    fn closed(&self) -> PyResult<bool> {
        let guard = self.writer.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock error: {}", e))
        })?;

        Ok(guard.is_none())
    }

    #[pyo3(
            text_signature = "(data, chunks, scale_factor=1.0, add_offset=0.0, compression='pfor_delta_2d', name='data', children=[])",
            signature = (data, chunks, scale_factor=None, add_offset=None, compression=None, name=None, children=None)
        )]
    /// Write a numpy array to the .om file with specified chunking and scaling parameters.
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
    ///     OmVariable representing the written group in the file structure
    ///
    /// Raises:
    ///     ValueError: If the data type is unsupported or if parameters are invalid
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
        let name = name.unwrap_or("data");
        let children: Vec<OmOffsetSize> = children
            .unwrap_or_default()
            .iter()
            .map(Into::into)
            .collect();

        let element_type = data.dtype();
        let py = data.py();

        let scale_factor = scale_factor.unwrap_or(1.0);
        let add_offset = add_offset.unwrap_or(0.0);
        let compression = compression
            .map(|s| PyCompressionType::from_str(s))
            .transpose()?
            .unwrap_or(PyCompressionType::PforDelta2d)
            .to_omfilesrs();

        let array_meta = if element_type.is_equiv_to(&dtype::<f32>(py)) {
            let array = data.downcast::<PyArrayDyn<f32>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<f64>(py)) {
            let array = data.downcast::<PyArrayDyn<f64>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<i32>(py)) {
            let array = data.downcast::<PyArrayDyn<i32>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<i64>(py)) {
            let array = data.downcast::<PyArrayDyn<i64>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<u32>(py)) {
            let array = data.downcast::<PyArrayDyn<u32>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<u64>(py)) {
            let array = data.downcast::<PyArrayDyn<u64>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<i8>(py)) {
            let array = data.downcast::<PyArrayDyn<i8>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<u8>(py)) {
            let array = data.downcast::<PyArrayDyn<u8>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<i16>(py)) {
            let array = data.downcast::<PyArrayDyn<i16>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<u16>(py)) {
            let array = data.downcast::<PyArrayDyn<u16>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else {
            Err(OmFilesRsError::InvalidDataType).map_err(convert_omfilesrs_error)
        }?;

        self.with_writer(|writer| {
            let offset_size = match writer {
                WriterBackend::File(w) => w.write_array(array_meta, name, &children),
                WriterBackend::FsSpec(w) => w.write_array(array_meta, name, &children),
            }
            .map_err(convert_omfilesrs_error)?;

            Ok(OmVariable {
                name: name.to_string(),
                offset: offset_size.offset,
                size: offset_size.size,
            })
        })
    }

    #[pyo3(
        text_signature = "(value, name, children=None)",
        signature = (value, name, children=None)
    )]
    /// Write a scalar value to the .om file.
    ///
    /// Args:
    ///     value: Scalar value to write. Supported types are:
    ///            int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, String
    ///     name: Name of the scalar variable
    ///     children: List of child variables (default: None)
    ///
    /// Returns:
    ///     OmVariable representing the written scalar in the file structure
    ///
    /// Raises:
    ///     ValueError: If the value type is unsupported (e.g., booleans)
    ///     RuntimeError: If there's an error writing to the file
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
            return Err(PyValueError::new_err(format!(
                    "Unsupported attribute type for name '{}'. Supported types are: String, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64",
                    name
                )));
        };
        Ok(result)
    }

    #[pyo3(
        text_signature = "(name, children)",
        signature = (name, children)
    )]
    /// Create a new group in the .om file.
    ///
    /// This is essentially a variable with no data, which serves as a container for other variables.
    ///
    /// Args:
    ///     name: Name of the group
    ///     children: List of child variables
    ///
    /// Returns:
    ///     OmVariable representing the written group in the file structure
    ///
    /// Raises:
    ///     RuntimeError: If there's an error writing to the file
    fn write_group(&mut self, name: &str, children: Vec<OmVariable>) -> PyResult<OmVariable> {
        let children: Vec<OmOffsetSize> = children.iter().map(Into::into).collect();

        self.with_writer(|writer| {
            let offset_size = match writer {
                WriterBackend::File(w) => w.write_none(name, &children),
                WriterBackend::FsSpec(w) => w.write_none(name, &children),
            }
            .map_err(convert_omfilesrs_error)?;

            Ok(OmVariable {
                name: name.to_string(),
                offset: offset_size.offset,
                size: offset_size.size,
            })
        })
    }
}

impl OmFileWriter {
    // Helper method for safe writer access
    fn with_writer<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut WriterBackend) -> PyResult<R>,
    {
        let mut guard = self.writer.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock error: {}", e))
        })?;

        match guard.as_mut() {
            Some(writer) => f(writer),
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "I/O operation on closed writer or file",
            )),
        }
    }

    fn write_array_internal<'py, T>(
        &mut self,
        data: PyReadonlyArrayDyn<'py, T>,
        chunks: Vec<u64>,
        scale_factor: f32,
        add_offset: f32,
        compression: CompressionType,
    ) -> PyResult<OmFileWriterArrayFinalized>
    where
        T: Element + OmFileArrayDataType,
    {
        let dimensions = data
            .shape()
            .into_iter()
            .map(|x| *x as u64)
            .collect::<Vec<u64>>();

        self.with_writer(|writer| match writer {
            WriterBackend::File(w) => {
                let mut array_writer = w
                    .prepare_array::<T>(dimensions, chunks, compression, scale_factor, add_offset)
                    .map_err(convert_omfilesrs_error)?;

                array_writer
                    .write_data(data.as_array(), None, None)
                    .map_err(convert_omfilesrs_error)?;

                let variable_meta = array_writer.finalize();
                Ok(variable_meta)
            }
            WriterBackend::FsSpec(w) => {
                let mut array_writer = w
                    .prepare_array::<T>(dimensions, chunks, compression, scale_factor, add_offset)
                    .map_err(convert_omfilesrs_error)?;

                array_writer
                    .write_data(data.as_array(), None, None)
                    .map_err(convert_omfilesrs_error)?;

                let variable_meta = array_writer.finalize();
                Ok(variable_meta)
            }
        })
    }

    fn store_scalar<T: OmFileScalarDataType + 'static>(
        &mut self,
        value: T,
        name: &str,
        children: &[OmOffsetSize],
    ) -> PyResult<OmVariable> {
        self.with_writer(|writer| {
            let offset_size = match writer {
                WriterBackend::File(w) => w.write_scalar(value, name, children),
                WriterBackend::FsSpec(w) => w.write_scalar(value, name, children),
            }
            .map_err(convert_omfilesrs_error)?;

            Ok(OmVariable {
                name: name.to_string(),
                offset: offset_size.offset,
                size: offset_size.size,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::ArrayD, PyArrayDyn, PyArrayMethods};
    use std::fs;

    #[test]
    fn test_write_array() -> Result<(), Box<dyn std::error::Error>> {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
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
        });

        Ok(())
    }

    #[test]
    fn test_fsspec_writer() -> Result<(), Box<dyn std::error::Error>> {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| -> Result<(), Box<dyn std::error::Error>> {
            let fsspec = py.import("fsspec")?;
            let fs = fsspec.call_method1("filesystem", ("memory",))?;

            let _writer = OmFileWriter::from_fsspec(fs.into(), "test_file.om".to_string())?;

            Ok(())
        })?;

        Ok(())
    }
}
