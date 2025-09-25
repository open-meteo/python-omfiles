use numpy::dtype;
use omfiles_rs::OmDataType;
use pyo3::{
    exceptions::PyTypeError,
    types::{PyNone, PyString},
    Bound, PyAny, PyResult, PyTypeInfo, Python,
};

/// Describe the dtype of an OmVariable, i.e. translate it to the closest Python dtype.
///
/// Returns numpy dtypes for numeric variables (scalar and array),
/// type(str) for Strings, and NoneType for None.
pub fn describe_dtype<'py>(py: Python<'py>, type_enum: &OmDataType) -> PyResult<Bound<'py, PyAny>> {
    match type_enum {
        OmDataType::Int8 | OmDataType::Int8Array => Ok(dtype::<i8>(py).into_any()),
        OmDataType::Uint8 | OmDataType::Uint8Array => Ok(dtype::<u8>(py).into_any()),
        OmDataType::Int16 | OmDataType::Int16Array => Ok(dtype::<i16>(py).into_any()),
        OmDataType::Uint16 | OmDataType::Uint16Array => Ok(dtype::<u16>(py).into_any()),
        OmDataType::Int32 | OmDataType::Int32Array => Ok(dtype::<i32>(py).into_any()),
        OmDataType::Uint32 | OmDataType::Uint32Array => Ok(dtype::<u32>(py).into_any()),
        OmDataType::Int64 | OmDataType::Int64Array => Ok(dtype::<i64>(py).into_any()),
        OmDataType::Uint64 | OmDataType::Uint64Array => Ok(dtype::<u64>(py).into_any()),
        OmDataType::Float | OmDataType::FloatArray => Ok(dtype::<f32>(py).into_any()),
        OmDataType::Double | OmDataType::DoubleArray => Ok(dtype::<f64>(py).into_any()),
        OmDataType::String => Ok(PyString::type_object(py).into_any()),
        OmDataType::None => Ok(PyNone::type_object(py).into_any()),
        _ => Err(PyTypeError::new_err(
            "Type cannot be converted to NumPy dtype",
        )),
    }
}
