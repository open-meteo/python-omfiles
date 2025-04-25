use numpy::{ndarray::ArrayD, IntoPyArray};
use pyo3::prelude::*;

pub enum OmFileTypedArray {
    Int8(ArrayD<i8>),
    Uint8(ArrayD<u8>),
    Int16(ArrayD<i16>),
    Uint16(ArrayD<u16>),
    Int32(ArrayD<i32>),
    Uint32(ArrayD<u32>),
    Int64(ArrayD<i64>),
    Uint64(ArrayD<u64>),
    Float(ArrayD<f32>),
    Double(ArrayD<f64>),
}

impl<'py> IntoPyObject<'py> for OmFileTypedArray {
    // The inner Python type this will convert to
    type Target = PyAny;
    // The output is a Bound to the target type
    type Output = Bound<'py, Self::Target>;
    // The error type for conversion failures
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            OmFileTypedArray::Int8(arr) => Ok(arr.into_pyarray(py).into_any()),
            OmFileTypedArray::Uint8(arr) => Ok(arr.into_pyarray(py).into_any()),
            OmFileTypedArray::Int16(arr) => Ok(arr.into_pyarray(py).into_any()),
            OmFileTypedArray::Uint16(arr) => Ok(arr.into_pyarray(py).into_any()),
            OmFileTypedArray::Int32(arr) => Ok(arr.into_pyarray(py).into_any()),
            OmFileTypedArray::Uint32(arr) => Ok(arr.into_pyarray(py).into_any()),
            OmFileTypedArray::Int64(arr) => Ok(arr.into_pyarray(py).into_any()),
            OmFileTypedArray::Uint64(arr) => Ok(arr.into_pyarray(py).into_any()),
            OmFileTypedArray::Float(arr) => Ok(arr.into_pyarray(py).into_any()),
            OmFileTypedArray::Double(arr) => Ok(arr.into_pyarray(py).into_any()),
        }
    }
}
