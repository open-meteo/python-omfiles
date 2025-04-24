use crate::{
    array_index::ArrayIndex, data_type::get_numpy_dtype, errors::convert_omfilesrs_error,
    fsspec_backend::AsyncFsSpecBackend, reader::get_shape_vec,
};
use async_lock::RwLock;
use delegate::delegate;
use num_traits::Zero;
use numpy::{
    ndarray::{self, ArrayD},
    Element, IntoPyArray, PyArray, PyArrayDescr, PyArrayMethods, PyUntypedArray, ToPyArray,
};
use omfiles_rs::{
    backend::{
        backends::OmFileReaderBackendAsync,
        mmapfile::{MmapFile, Mode},
    },
    core::data_types::{DataType, OmFileArrayDataType, OmFileScalarDataType},
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

#[derive(Debug)] // Add Debug for potential logging/errors
enum RustArrayData {
    Float(ArrayD<f32>),
    // Int8(ArrayD<i8>),
    // Uint8(ArrayD<u8>),
    // Int16(ArrayD<i16>),
    // Uint16(ArrayD<u16>),
    // Int32(ArrayD<i32>),
    // Uint32(ArrayD<u32>),
    // Int64(ArrayD<i64>),
    // Uint64(ArrayD<u64>),
    // Double(ArrayD<f64>),
    // Add other variants as you uncomment them
}

impl<'py> IntoPyObject<'py> for RustArrayData {
    // The inner Python type this will convert to
    type Target = PyArray<f32, ndarray::Dim<ndarray::IxDynImpl>>;
    // The output is a Bound to the target type
    type Output = Bound<'py, Self::Target>;
    // The error type for conversion failures
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            RustArrayData::Float(arr) => {
                // Handle the array squeeze operation
                let squeezed = arr.squeeze();
                // Convert to a numpy array in Python
                let py_array = squeezed.into_pyarray(py);
                // let py_array = py_array.as_any(); // Convert to untyped array if needed

                // Return the bound PyAny
                Ok(py_array)
            } // Uncomment and implement other variants as needed:
              // RustArrayData::Int8(arr) => {
              //     let squeezed = arr.squeeze();
              //     let py_array = squeezed.into_pyarray(py).as_untyped();
              //     Ok(py_array.into())
              // },
              // ... other variants ...
        }
    }
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

    async fn read_concurrent<'py>(&self, ranges: ArrayIndex) -> PyResult<RustArrayData> {
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

        println!("Data type {:?}", data_type);
        let result = match data_type {
            DataType::FloatArray => {
                let array = reader
                    .read_async::<f32>(&read_ranges, io_size_max, io_size_merge)
                    .await
                    .map_err(convert_omfilesrs_error)?;
                Ok(RustArrayData::Float(array))
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Invalid data type",
                ));
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

    // /// Read data asynchronously
    // fn read_async<'py>(
    //     &self,
    //     py: Python<'py>,
    //     ranges: ArrayIndex,
    //     io_size_max: Option<u64>,
    //     io_size_merge: Option<u64>,
    // ) -> PyResult<Bound<'py, PyAny>> {
    //     // Convert the Python ranges to Rust ranges
    //     let read_ranges = ranges.to_read_range(&self.shape)?;

    //     // Get the data type and a cloned backend from the reader
    //     let (data_type, backend) =
    //         self.with_reader(|reader| Ok((reader.data_type(), reader.backend.clone())))?;

    //     let temp_reader = OmFileReader::new(backend).map_err(convert_omfilesrs_error)?;

    //     let blub = match data_type {
    //         DataType::Int8Array => local_future_into_py_with_locals(
    //             py,
    //             pyo3_async_runtimes::async_std::get_current_locals(py)?,
    //             async move {
    //                 let array_result =
    //                     read_async_untyped_array::<i8>(&temp_reader, &read_ranges).await?;
    //                 Python::with_gil(|py| Ok(array_result.into_pyarray(py).as_untyped().to_owned()))
    //             },
    //         ),
    //         _ => Err(PyErr::new::<PyValueError, _>(
    //             "Data type not supported for async reading",
    //         )),
    //     }?;
    //     Ok(blub)

    //     // // Create a future that will execute the async read
    //     // future_into_py(py, async move {
    //     //     // Create a temporary reader for this async operation
    //     //     let temp_reader = OmFileReader::new(backend).map_err(convert_omfilesrs_error)?;

    //     //     // Perform the appropriate async read based on data type
    //     //     match data_type {
    //     //         DataType::Int8Array => {
    //     //             let array_result =
    //     //                 read_async_untyped_array::<i8>(&temp_reader, &read_ranges).await;
    //     //             Ok(array_result?.into_pyarray(py).as_untyped().to_owned())
    //     //         }
    //     //         DataType::Uint8Array => {
    //     //             let array_result =
    //     //                 read_async_untyped_array::<u8>(&temp_reader, &read_ranges).await;
    //     //             Ok(array_result?.into_pyarray(py).as_untyped().to_owned())
    //     //         }
    //     //         DataType::Int16Array => {
    //     //             let array_result =
    //     //                 read_async_untyped_array::<i16>(&temp_reader, &read_ranges).await;
    //     //             Ok(array_result?.into_pyarray(py).as_untyped().to_owned())
    //     //         }
    //     //         DataType::Uint16Array => {
    //     //             let array_result =
    //     //                 read_async_untyped_array::<u16>(&temp_reader, &read_ranges).await;
    //     //             Ok(array_result?.into_pyarray(py).as_untyped().to_owned())
    //     //         }
    //     //         DataType::Int32Array => {
    //     //             let array_result =
    //     //                 read_async_untyped_array::<i32>(&temp_reader, &read_ranges).await;
    //     //             Ok(array_result?.into_pyarray(py).as_untyped().to_owned())
    //     //         }
    //     //         DataType::Uint32Array => {
    //     //             let array_result =
    //     //                 read_async_untyped_array::<u32>(&temp_reader, &read_ranges).await;
    //     //             Ok(array_result?.into_pyarray(py).as_untyped().to_owned())
    //     //         }
    //     //         DataType::Int64Array => {
    //     //             let array_result =
    //     //                 read_async_untyped_array::<i64>(&temp_reader, &read_ranges).await;
    //     //             Ok(array_result?.into_pyarray(py).as_untyped().to_owned())
    //     //         }
    //     //         DataType::Uint64Array => {
    //     //             let array_result =
    //     //                 read_async_untyped_array::<u64>(&temp_reader, &read_ranges).await;
    //     //             Ok(array_result?.into_pyarray(py).as_untyped().to_owned())
    //     //         }
    //     //         DataType::FloatArray => {
    //     //             let array_result =
    //     //                 read_async_untyped_array::<f32>(&temp_reader, &read_ranges).await;
    //     //             Ok(array_result?.into_pyarray(py).as_untyped().to_owned())
    //     //         }
    //     //         DataType::DoubleArray => {
    //     //             let array_result =
    //     //                 read_async_untyped_array::<f64>(&temp_reader, &read_ranges).await;
    //     //             Ok(array_result?.into_pyarray(py).as_untyped().to_owned())
    //     //         }
    //     //         _ => {
    //     //             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
    //     //                 "Data type not supported for async reading",
    //     //             ))
    //     //         }
    //     //     }
    //     // })

    //     // blub
    //     // let blob = blub?;
    //     // blob
    // }
}

// impl OmFilePyReaderAsync {
//     async fn with_reader_async<F, Fut, R>(&self, f: F) -> PyResult<R>
//     where
//         F: FnOnce(&OmFileReader<AsyncFsSpecBackend>) -> Fut,
//         Fut: Future<Output = PyResult<R>> + Send,
//     {
//         let guard = self.reader.try_read().map_err(|_| {
//             PyErr::new::<pyo3::exceptions::PyValueError, _>(
//                 "Trying to read from a reader which is being closed",
//             )
//         })?;
//         if let Some(reader) = &*guard {
//             f(reader).await
//         } else {
//             Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
//                 "I/O operation on closed reader or file",
//             ))
//         }
//     }
// }

async fn read_async_untyped_array<T>(
    reader: &OmFileReader<AsyncFsSpecBackend>,
    ranges: &[Range<u64>],
) -> PyResult<ndarray::ArrayD<T>>
where
    T: Element + OmFileArrayDataType + Clone + Zero + Send + Sync + 'static,
{
    // Just do the Rust async operation
    let array = reader
        .read_async::<T>(ranges, None, None)
        .await
        .map_err(convert_omfilesrs_error)?;

    Ok(array.squeeze())
}
