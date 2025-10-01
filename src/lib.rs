use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
mod array_index;
mod codecs;
mod compression;
mod cpu_info;
mod data_type;
mod errors;
mod fsspec_backend;
mod hierarchy;
mod reader;
mod reader_async;
mod test_utils;
mod typed_array;
mod writer;

#[pymodule(gil_used = false)]
fn omfiles<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<reader::OmFileReader>()?;
    m.add_class::<writer::OmFileWriter>()?;
    m.add_class::<reader_async::OmFileReaderAsync>()?;
    m.add_class::<hierarchy::OmVariable>()?;
    m.add_class::<codecs::RustPforCodec>()?;
    m.add_function(wrap_pyfunction!(cpu_info::_check_cpu_features, m)?)?;

    Ok(())
}

define_stub_info_gatherer!(stub_info);
