use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
mod array_index;
mod compression;
mod data_type;
mod errors;
mod fsspec_backend;
mod hierarchy;
mod reader;
mod test_utils;
mod writer;

/// A Python module implemented in Rust.
#[pymodule]
fn omfiles<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<reader::OmFilePyReader>()?;
    m.add_class::<writer::OmFilePyWriter>()?;
    m.add_class::<hierarchy::OmVariable>()?;

    Ok(())
}

define_stub_info_gatherer!(stub_info);
