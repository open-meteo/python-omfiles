use pyo3::PyErr;

/// Utility function to convert an OmFilesError to a PyRuntimeError
pub fn convert_omfilesrs_error(e: omfiles_rs::OmFilesError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
}
