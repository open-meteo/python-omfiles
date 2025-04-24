use omfiles_rs::backend::backends::OmFileReaderBackend;
use omfiles_rs::backend::backends::OmFileReaderBackendAsync;
use omfiles_rs::errors::OmFilesRsError;
use pyo3::prelude::*;
use pyo3::Python;
use pyo3_async_runtimes::async_std::into_future;

/// An asynchronous backend for reading files using fsspec.
pub struct AsyncFsSpecBackend {
    py_file: PyObject,
    file_size: u64,
}

impl AsyncFsSpecBackend {
    /// Create a new asynchronous backend for reading files using fsspec.
    /// This init will fetch the file size via the `size` attribute or
    /// the `size()` method of the parent fs object.
    pub fn new(open_file: PyObject) -> PyResult<Self> {
        // Get file size synchronously - usually fast enough.
        let size = Python::with_gil(|py| -> PyResult<u64> {
            // Assuming 'open_file' is the result of fsspec.open(...)
            if let Ok(size_attr) = open_file.bind(py).getattr("size") {
                size_attr.extract::<u64>()
            } else {
                let fs = open_file.bind(py).getattr("fs")?;
                let path = open_file.bind(py).getattr("path")?;
                fs.call_method1("size", (path,))?.extract::<u64>()
            }
        })?;

        Ok(Self {
            py_file: open_file,
            file_size: size,
        })
    }

    // Consider making close async as well if the Python close can block
    pub fn close(&self) -> PyResult<()> {
        Python::with_gil(|py| {
            // Ensure close exists and call it
            if let Ok(close_method) = self.py_file.bind(py).getattr("close") {
                if close_method.is_callable() {
                    close_method.call0()?;
                }
            }
            Ok(())
        })
    }
}

impl OmFileReaderBackendAsync for AsyncFsSpecBackend {
    fn count_async(&self) -> usize {
        self.file_size as usize
    }

    // This function calls an async read_bytes method on the Python file object
    // and transforms it into a future that can be awaited
    // This allows us to execute multiple asynchronous operations concurrently
    async fn get_bytes_async(&self, offset: u64, count: u64) -> Result<Vec<u8>, OmFilesRsError> {
        let fut = Python::with_gil(|py| {
            let bound_file = self.py_file.bind(py);
            let coroutine = bound_file.call_method1("read_bytes", (offset, count))?;
            into_future(coroutine)
        })
        .map_err(|e| OmFilesRsError::DecoderError(format!("Python I/O error {}", e)))?;

        let bytes_obj = fut
            .await
            .map_err(|e| OmFilesRsError::DecoderError(format!("Python I/O error {}", e)))?;

        let bytes = Python::with_gil(|py| bytes_obj.extract::<Vec<u8>>(py))
            .map_err(|e| OmFilesRsError::DecoderError(format!("Python I/O error: {}", e)));
        bytes
    }
}

pub struct FsSpecBackend {
    py_file: PyObject,
    file_size: u64,
}

impl FsSpecBackend {
    pub fn new(open_file: PyObject) -> PyResult<Self> {
        let size = Python::with_gil(|py| -> PyResult<u64> {
            let fs = open_file.bind(py).getattr("fs")?;
            let path = open_file.bind(py).getattr("path")?;
            let size = fs.call_method1("size", (path,))?.extract::<u64>()?;
            Ok(size)
        })?;

        Ok(Self {
            py_file: open_file.into(),
            file_size: size,
        })
    }

    pub fn close(&self) -> PyResult<()> {
        Python::with_gil(|py| {
            let file_obj = &self.py_file;
            if file_obj.bind(py).hasattr("close")? {
                file_obj.bind(py).call_method0("close")?;
            }
            Ok(())
        })
    }
}

impl OmFileReaderBackend for FsSpecBackend {
    fn count(&self) -> usize {
        self.file_size as usize
    }

    fn needs_prefetch(&self) -> bool {
        false
    }

    fn prefetch_data(&self, _offset: usize, _count: usize) {
        // No-op for now
    }

    fn pre_read(
        &self,
        _offset: usize,
        _count: usize,
    ) -> Result<(), omfiles_rs::errors::OmFilesRsError> {
        Ok(())
    }

    /// This is a blocking operation that reads bytes from the file!
    fn get_bytes_owned(
        &self,
        offset: u64,
        count: u64,
    ) -> Result<Vec<u8>, omfiles_rs::errors::OmFilesRsError> {
        Python::with_gil(|py| {
            self.py_file.call_method1(py, "seek", (offset,))?;
            let bytes = self.py_file.call_method1(py, "read", (count,))?;
            bytes.extract::<Vec<u8>>(py)
        })
        .map_err(|e| OmFilesRsError::DecoderError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use crate::create_test_binary_file;

    use super::*;
    use std::error::Error;

    #[test]
    fn test_fsspec_backend() -> Result<(), Box<dyn Error>> {
        let file_name = "test_fsspec_backend.om";
        let file_path = format!("test_files/{}", file_name);
        create_test_binary_file!(file_name)?;
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| -> Result<(), Box<dyn Error>> {
            let fsspec = py.import("fsspec")?;
            let fs = fsspec.call_method1("filesystem", ("file",))?;
            let open_file = fs.call_method1("open", (file_path,))?;

            let backend = FsSpecBackend::new(open_file.into())?;
            assert_eq!(backend.file_size, 144);

            let bytes = backend.get_bytes_owned(0, 44)?;
            assert_eq!(
                &bytes,
                &[
                    79, 77, 3, 0, 4, 130, 0, 2, 3, 34, 0, 4, 194, 2, 10, 4, 178, 0, 12, 4, 242, 0,
                    14, 197, 17, 20, 194, 2, 22, 194, 2, 24, 3, 3, 228, 200, 109, 1, 0, 0, 20, 0,
                    4, 0
                ]
            );

            Ok(())
        })?;

        Ok(())
    }
}
