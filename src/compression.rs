use omfiles_rs::core::compression::CompressionType;
use pyo3::{exceptions::PyValueError, prelude::*};

#[derive(Clone)]
pub enum PyCompressionType {
    PforDelta2dInt16,
    FpxXor2d,
    PforDelta2d,
    PforDelta2dInt16Logarithmic,
}

impl PyCompressionType {
    pub fn to_omfilesrs(&self) -> CompressionType {
        match self {
            PyCompressionType::PforDelta2dInt16 => CompressionType::PforDelta2dInt16,
            PyCompressionType::FpxXor2d => CompressionType::FpxXor2d,
            PyCompressionType::PforDelta2d => CompressionType::PforDelta2d,
            PyCompressionType::PforDelta2dInt16Logarithmic => {
                CompressionType::PforDelta2dInt16Logarithmic
            }
        }
    }

    pub fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "pfor_delta_2d_int16" => Ok(PyCompressionType::PforDelta2dInt16),
            "fpx_xor_2d" => Ok(PyCompressionType::FpxXor2d),
            "pfor_delta_2d" => Ok(PyCompressionType::PforDelta2d),
            "pfor_delta_2d_int16_logarithmic" => Ok(PyCompressionType::PforDelta2dInt16Logarithmic),
            _ => Err(PyValueError::new_err(format!(
                "Unsupported compression type: {}",
                s
            ))),
        }
    }
}
