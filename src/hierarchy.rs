use omfiles_rs::OmOffsetSize;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass(module = "omfiles.omfiles")]
#[derive(Clone)]
/// Represents a variable in an OM file.
pub(crate) struct OmVariable {
    #[pyo3(get)]
    /// The name of the variable.
    pub name: String,
    #[pyo3(get)]
    /// The offset of the variable in the OM file.
    pub offset: u64,
    #[pyo3(get)]
    /// The size of the variable in bytes in the OM file.
    pub size: u64,
}

#[gen_stub_pymethods]
#[pymethods]
impl OmVariable {
    fn __repr__(&self) -> String {
        format!(
            "OmVariable(name='{}', offset={}, size={})",
            self.name, self.offset, self.size
        )
    }
}

impl Into<OmOffsetSize> for &OmVariable {
    fn into(self) -> OmOffsetSize {
        OmOffsetSize {
            offset: self.offset,
            size: self.size,
        }
    }
}

impl Into<OmOffsetSize> for OmVariable {
    fn into(self) -> OmOffsetSize {
        OmOffsetSize {
            offset: self.offset,
            size: self.size,
        }
    }
}
