use omfiles_rs::traits::{OmFileReaderBackend, OmFileReaderBackendAsync, OmFileWriterBackend};
use omfiles_rs::OmFilesError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;
use pyo3_async_runtimes::async_std::into_future;

/// An asynchronous backend for reading files using fsspec.
pub struct AsyncFsSpecBackend {
    fs: Py<PyAny>,
    path: String,
    file_size: u64,
}

impl AsyncFsSpecBackend {
    /// Create a new asynchronous backend for reading files using fsspec.
    /// This init expects any AbstractFileSystem as a fs object and a path
    /// to the file to be read.
    pub async fn new(fs: Py<PyAny>, path: String) -> PyResult<Self> {
        let fut = Python::attach(|py| {
            let bound_fs = fs.bind(py);
            let coroutine = bound_fs.call_method1("_size", (path.clone(),))?;
            into_future(coroutine)
        })?;
        let size_result = fut.await?;

        let size = Python::attach(|py| size_result.bind(py).extract::<u64>())?;

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
    async fn get_bytes_async(&self, offset: u64, count: u64) -> Result<Vec<u8>, OmFilesError> {
        let fut = Python::attach(|py| {
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
        .map_err(|e| OmFilesError::DecoderError(format!("Python I/O error {}", e)))?;

        let bytes_obj = fut
            .await
            .map_err(|e| OmFilesError::DecoderError(format!("Python I/O error {}", e)))?;

        let bytes = Python::attach(|py| bytes_obj.extract::<Vec<u8>>(py))
            .map_err(|e| OmFilesError::DecoderError(format!("Python I/O error: {}", e)));
        bytes
    }
}

pub struct FsSpecBackend {
    fs: Py<PyAny>,
    path: String,
    file_size: u64,
}

impl FsSpecBackend {
    pub fn new(fs: Py<PyAny>, path: String) -> PyResult<Self> {
        let size = Python::attach(|py| {
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
    type Bytes<'a> = Vec<u8>;

    fn count(&self) -> usize {
        self.file_size as usize
    }

    fn prefetch_data(&self, _offset: usize, _count: usize) {
        // No-op for now
    }

    /// This is a blocking operation that reads bytes from the file!
    fn get_bytes(
        &self,
        offset: u64,
        count: u64,
    ) -> Result<Self::Bytes<'_>, omfiles_rs::OmFilesError> {
        let bytes = Python::attach(|py| {
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
        .map_err(|e| OmFilesError::DecoderError(format!("Python I/O error {}", e)))?;
        Ok(bytes)
    }
}

/// An fsspec writer backend that implements OmFileWriterBackend.
pub struct FsSpecWriterBackend {
    _fs: Py<PyAny>,
    open_fs: Py<PyAny>,
}

impl FsSpecWriterBackend {
    /// Create a new fsspec writer backend.
    pub fn new(fs: Py<PyAny>, path: String) -> PyResult<Self> {
        let open_fs = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let bound_fs = fs.bind(py);
            let open_file = bound_fs.call_method1("open", (path, "wb"))?.unbind();
            Ok(open_file)
        })?;
        Ok(Self { _fs: fs, open_fs })
    }
}

impl OmFileWriterBackend for FsSpecWriterBackend {
    fn write(&mut self, data: &[u8]) -> Result<(), OmFilesError> {
        Python::attach(|py| {
            let bound_file = self.open_fs.bind(py);
            let py_bytes = pyo3::types::PyBytes::new(py, data);
            // We need to write to the open_fs. Otherwise fsspec does not
            // provide an API which correctly buffers the data.
            bound_file.call_method1("write", (py_bytes,))?;
            Ok(())
        })
        .map_err(|e: pyo3::PyErr| {
            OmFilesError::DecoderError(format!("Failed to write to fsspec backend: {}", e))
        })
    }

    fn synchronize(&self) -> Result<(), OmFilesError> {
        // Fsspec operations are typically synchronized upon completion of the write call.
        // If a specific fs has an explicit sync method, it would be called here.
        // For many backends, calling `flush` on an opened file handle might be equivalent.
        // Without a specific `synchronize` method on fsspec's abstract interface,
        // we rely on the underlying backend's behavior.
        // For now, we'll no-op, assuming writes are flushed by the fsspec implementation.
        Ok(())
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
        Python::initialize();

        Python::attach(|py| -> Result<(), Box<dyn Error>> {
            let fsspec = py.import("fsspec")?;
            let fs = fsspec.call_method1("filesystem", ("file",))?;

            let backend = FsSpecBackend::new(fs.into(), file_path)?;
            assert_eq!(backend.file_size, 144);

            let bytes = backend.get_bytes(0, 44)?;
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

    #[test]
    fn test_fsspec_writer_backend() -> Result<(), Box<dyn Error>> {
        Python::initialize();

        Python::attach(|py| -> Result<(), Box<dyn Error>> {
            let memory_module = py.import("fsspec.implementations.memory")?;
            let fs = memory_module.call_method0("MemoryFileSystem")?;

            let mut write_backend =
                FsSpecWriterBackend::new(fs.into(), "test_file.om".to_string())?;

            // Test writing
            write_backend.write(b"Hello, World!")?;
            write_backend.synchronize()?;

            write_backend.write(b"fsspec")?;
            write_backend.synchronize()?;

            // Read back using the FsSpecBackend for reading
            let fs = memory_module.call_method0("MemoryFileSystem")?;
            let read_backend = FsSpecBackend::new(fs.into(), "test_file.om".to_string())?;
            let bytes = read_backend.get_bytes(0, 44)?;
            assert_eq!(
                &bytes,
                &[
                    72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33, 102, 115, 115, 112,
                    101, 99
                ]
            );

            Ok(())
        })?;

        Ok(())
    }
}
