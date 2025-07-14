use omfiles_rs::backend::backends::{
    OmFileReaderBackend, OmFileReaderBackendAsync, OmFileWriterBackend,
};
use omfiles_rs::errors::OmFilesRsError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;
use pyo3_async_runtimes::async_std::into_future;
use std::io::{self, Seek, SeekFrom, Write};

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

/// An fsspec writer backend that implements Write, Seek, and OmFileWriterBackend traits.
pub struct FsSpecWriterBackend {
    fs: PyObject,
    path: String,
    buffer: Vec<u8>,
    position: u64,
}

impl FsSpecWriterBackend {
    /// Create a new fsspec writer backend.
    pub fn new(fs: PyObject, path: String) -> PyResult<Self> {
        Ok(Self {
            fs,
            path,
            buffer: Vec::new(),
            position: 0,
        })
    }

    /// Flush the buffer to the fsspec file system.
    pub fn flush_to_fs(&self) -> PyResult<()> {
        Python::with_gil(|py| {
            let bound_fs = self.fs.bind(py);
            let bytes = pyo3::types::PyBytes::new(py, &self.buffer);
            bound_fs.call_method1("write_bytes", (self.path.clone(), bytes))?;
            Ok(())
        })
    }
}

impl Write for FsSpecWriterBackend {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Extend the buffer to accommodate the new data at the current position
        let end_position = self.position as usize + buf.len();
        if end_position > self.buffer.len() {
            self.buffer.resize(end_position, 0);
        }

        // Write the data to the buffer at the current position
        let start_pos = self.position as usize;
        self.buffer[start_pos..end_position].copy_from_slice(buf);
        self.position += buf.len() as u64;

        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_to_fs().map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to flush to fsspec: {}", e),
            )
        })
    }
}

impl Seek for FsSpecWriterBackend {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_position = match pos {
            SeekFrom::Start(pos) => pos,
            SeekFrom::End(offset) => {
                let end = self.buffer.len() as u64;
                if offset < 0 {
                    end.checked_sub((-offset) as u64).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "Seek before start")
                    })?
                } else {
                    end + offset as u64
                }
            }
            SeekFrom::Current(offset) => {
                if offset < 0 {
                    self.position.checked_sub((-offset) as u64).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "Seek before start")
                    })?
                } else {
                    self.position + offset as u64
                }
            }
        };

        self.position = new_position;
        Ok(self.position)
    }
}

impl OmFileWriterBackend for FsSpecWriterBackend {
    fn write(&mut self, data: &[u8]) -> Result<(), OmFilesRsError> {
        self.write_all(data).map_err(|e| {
            OmFilesRsError::DecoderError(format!("Failed to write to fsspec backend: {}", e))
        })?;
        // Immediately flush to fsspec after each write to ensure data persistence
        self.flush_to_fs()
            .map_err(|e| OmFilesRsError::DecoderError(format!("Failed to flush to fsspec: {}", e)))
    }

    fn write_at(&mut self, data: &[u8], offset: usize) -> Result<(), OmFilesRsError> {
        // Seek to the offset position
        self.seek(std::io::SeekFrom::Start(offset as u64))
            .map_err(|e| {
                OmFilesRsError::DecoderError(format!("Failed to seek in fsspec backend: {}", e))
            })?;

        // Write the data (this will auto-flush due to our write implementation)
        OmFileWriterBackend::write(self, data)
    }

    fn synchronize(&self) -> Result<(), OmFilesRsError> {
        // Flush the buffer to the fsspec file system
        self.flush_to_fs().map_err(|e| {
            OmFilesRsError::DecoderError(format!("Failed to synchronize fsspec backend: {}", e))
        })
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

    #[test]
    fn test_fsspec_writer_backend() -> Result<(), Box<dyn Error>> {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| -> Result<(), Box<dyn Error>> {
            let fsspec = py.import("fsspec")?;
            let fs = fsspec.call_method1("filesystem", ("memory",))?;

            let mut backend = FsSpecWriterBackend::new(fs.into(), "test_file.om".to_string())?;

            // Test writing
            std::io::Write::write(&mut backend, b"Hello, World!")?;
            backend.flush()?;

            // Test seeking
            backend.seek(SeekFrom::Start(7))?;
            std::io::Write::write(&mut backend, b"fsspec")?;
            backend.flush()?;

            Ok(())
        })?;

        Ok(())
    }
}
