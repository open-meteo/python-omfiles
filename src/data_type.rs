use numpy::{dtype, PyArrayDescr};
use omfiles_rs::core::data_types::DataType;
use pyo3::{exceptions::PyTypeError, Bound, PyResult, Python};

/// Get NumPy dtype, only for numeric types
pub fn get_numpy_dtype<'py>(
    py: Python<'py>,
    type_enum: &DataType,
) -> PyResult<Bound<'py, PyArrayDescr>> {
    match type_enum {
        DataType::Int8 | DataType::Int8Array => Ok(dtype::<i8>(py)),
        DataType::Uint8 | DataType::Uint8Array => Ok(dtype::<u8>(py)),
        DataType::Int16 | DataType::Int16Array => Ok(dtype::<i16>(py)),
        DataType::Uint16 | DataType::Uint16Array => Ok(dtype::<u16>(py)),
        DataType::Int32 | DataType::Int32Array => Ok(dtype::<i32>(py)),
        DataType::Uint32 | DataType::Uint32Array => Ok(dtype::<u32>(py)),
        DataType::Int64 | DataType::Int64Array => Ok(dtype::<i64>(py)),
        DataType::Uint64 | DataType::Uint64Array => Ok(dtype::<u64>(py)),
        DataType::Float | DataType::FloatArray => Ok(dtype::<f32>(py)),
        DataType::Double | DataType::DoubleArray => Ok(dtype::<f64>(py)),
        _ => Err(PyTypeError::new_err(
            "Type cannot be converted to NumPy dtype",
        )),
    }
}
