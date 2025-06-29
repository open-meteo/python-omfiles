use std::convert::Infallible;

use omfiles_rs::core::compression::CompressionType;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyString};
use pyo3_stub_gen::impl_stub_type;

#[derive(Clone)]
pub enum PyCompressionType {
    PforDelta2dInt16,
    FpxXor2d,
    PforDelta2d,
    PforDelta2dInt16Logarithmic,
    None,
}

impl_stub_type!(PyCompressionType = PyResult<String>);

impl PyCompressionType {
    pub fn to_omfilesrs(&self) -> CompressionType {
        match self {
            PyCompressionType::PforDelta2dInt16 => CompressionType::PforDelta2dInt16,
            PyCompressionType::FpxXor2d => CompressionType::FpxXor2d,
            PyCompressionType::PforDelta2d => CompressionType::PforDelta2d,
            PyCompressionType::PforDelta2dInt16Logarithmic => {
                CompressionType::PforDelta2dInt16Logarithmic
            }
            PyCompressionType::None => CompressionType::None,
        }
    }

    pub fn from_omfilesrs(compression_type: CompressionType) -> Self {
        match compression_type {
            CompressionType::PforDelta2dInt16 => PyCompressionType::PforDelta2dInt16,
            CompressionType::FpxXor2d => PyCompressionType::FpxXor2d,
            CompressionType::PforDelta2d => PyCompressionType::PforDelta2d,
            CompressionType::PforDelta2dInt16Logarithmic => {
                PyCompressionType::PforDelta2dInt16Logarithmic
            }
            CompressionType::None => PyCompressionType::None,
        }
    }

    pub fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "pfor_delta_2d_int16" => Ok(PyCompressionType::PforDelta2dInt16),
            "fpx_xor_2d" => Ok(PyCompressionType::FpxXor2d),
            "pfor_delta_2d" => Ok(PyCompressionType::PforDelta2d),
            "pfor_delta_2d_int16_logarithmic" => Ok(PyCompressionType::PforDelta2dInt16Logarithmic),
            "none" => Ok(PyCompressionType::None),
            _ => Err(PyValueError::new_err(format!(
                "Unsupported compression type: {}",
                s
            ))),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            PyCompressionType::PforDelta2dInt16 => "pfor_delta_2d_int16",
            PyCompressionType::FpxXor2d => "fpx_xor_2d",
            PyCompressionType::PforDelta2d => "pfor_delta_2d",
            PyCompressionType::PforDelta2dInt16Logarithmic => "pfor_delta_2d_int16_logarithmic",
            PyCompressionType::None => "none",
        }
    }
}

impl<'py> IntoPyObject<'py> for PyCompressionType {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.as_str().into_pyobject(py)
    }
}
