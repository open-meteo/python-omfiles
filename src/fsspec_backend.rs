use omfiles_rs::backend::backends::OmFileReaderBackend;
use omfiles_rs::backend::backends::OmFileReaderBackendAsync;
use omfiles_rs::errors::OmFilesRsError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;
use pyo3_async_runtimes::async_std::into_future;

/// An asynchronous backend for reading files using fsspec.
pub struct AsyncFsSpecBackend {
    fs: PyObject,
    path: String,
    file_size: u64,
}

impl AsyncFsSpecBackend {
    /// Create a new asynchronous backend for reading files using fsspec.
    /// This init expects any AbstractFileSystem as a fs object and a path
    /// to the file to be read.
    pub async fn new(fs: PyObject, path: String) -> PyResult<Self> {
        let fut = Python::with_gil(|py| {
            let bound_fs = fs.bind(py);
            let coroutine = bound_fs.call_method1("_size", (path.clone(),))?;
            into_future(coroutine)
        })?;
        let size_result = fut.await?;

        let size = Python::with_gil(|py| size_result.bind(py).extract::<u64>())?;

        Ok(Self {
            fs,
            path,
            file_size: size,
        })
    }

    // Consider making close async as well if the Python close can block
    pub fn close(&self) -> PyResult<()> {
        // fs object does not need to be closed
        Ok(())
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
            let bound_fs = self.fs.bind(py);
            // We only use named parameters here, because positional arguments can
            // be different between different implementations of the super class!
            let kwargs = PyDict::new(py);
            kwargs.set_item("start", offset)?;
            kwargs.set_item("end", offset + count)?;
            kwargs.set_item("path", &self.path)?;
            let coroutine = bound_fs.call_method("_cat_file", (), Some(&kwargs))?;
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
    fs: PyObject,
    path: String,
    file_size: u64,
}

impl FsSpecBackend {
    pub fn new(fs: PyObject, path: String) -> PyResult<Self> {
        let size = Python::with_gil(|py| {
            let bound_fs = fs.bind(py);
            bound_fs
                .call_method1("size", (path.clone(),))?
                .extract::<u64>()
        })?;

        Ok(Self {
            fs,
            path,
            file_size: size,
        })
    }

    pub fn close(&self) -> PyResult<()> {
        // fs object does not need to be closed
        Ok(())
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
        let bytes = Python::with_gil(|py| {
            let bound_fs = self.fs.bind(py);
            // We only use named parameters here, because positional arguments can
            // be different between different implementations of the super class!
            let kwargs = PyDict::new(py);
            kwargs.set_item("start", offset)?;
            kwargs.set_item("end", offset + count)?;
            kwargs.set_item("path", &self.path)?;
            bound_fs
                .call_method("cat_file", (), Some(&kwargs))?
                .extract::<Vec<u8>>()
        })
        .map_err(|e| OmFilesRsError::DecoderError(format!("Python I/O error {}", e)))?;
        Ok(bytes)
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

            let backend = FsSpecBackend::new(fs.into(), file_path)?;
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
