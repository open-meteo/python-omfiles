use numpy::{dtype, PyArrayDescr};
use omfiles_rs::OmDataType;
use pyo3::{exceptions::PyTypeError, Bound, PyResult, Python};

/// Get NumPy dtype, only for numeric types
pub fn get_numpy_dtype<'py>(
    py: Python<'py>,
    type_enum: &OmDataType,
) -> PyResult<Bound<'py, PyArrayDescr>> {
    match type_enum {
        OmDataType::Int8 | OmDataType::Int8Array => Ok(dtype::<i8>(py)),
        OmDataType::Uint8 | OmDataType::Uint8Array => Ok(dtype::<u8>(py)),
        OmDataType::Int16 | OmDataType::Int16Array => Ok(dtype::<i16>(py)),
        OmDataType::Uint16 | OmDataType::Uint16Array => Ok(dtype::<u16>(py)),
        OmDataType::Int32 | OmDataType::Int32Array => Ok(dtype::<i32>(py)),
        OmDataType::Uint32 | OmDataType::Uint32Array => Ok(dtype::<u32>(py)),
        OmDataType::Int64 | OmDataType::Int64Array => Ok(dtype::<i64>(py)),
        OmDataType::Uint64 | OmDataType::Uint64Array => Ok(dtype::<u64>(py)),
        OmDataType::Float | OmDataType::FloatArray => Ok(dtype::<f32>(py)),
        OmDataType::Double | OmDataType::DoubleArray => Ok(dtype::<f64>(py)),
        _ => Err(PyTypeError::new_err(
            "Type cannot be converted to NumPy dtype",
        )),
    }
}
