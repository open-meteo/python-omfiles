use numpy::{
    PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use om_file_format_sys::{OmCompression_t, OmDataType_t};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::ffi::c_uchar;

const CODEC_ID_PFOR_DELTA_2D: &str = "pfor_delta_2d";

// Helper to get OmDataType and element size from string
fn get_dtype_info(dtype_str: &str) -> PyResult<(OmDataType_t, usize)> {
    match dtype_str {
        "int8" => Ok((OmDataType_t::DATA_TYPE_INT8_ARRAY, 1)),
        "uint8" => Ok((OmDataType_t::DATA_TYPE_UINT8_ARRAY, 1)),
        "int16" => Ok((OmDataType_t::DATA_TYPE_INT16_ARRAY, 2)),
        "uint16" => Ok((OmDataType_t::DATA_TYPE_UINT16_ARRAY, 2)),
        "int32" => Ok((OmDataType_t::DATA_TYPE_INT32_ARRAY, 4)),
        "uint32" => Ok((OmDataType_t::DATA_TYPE_UINT32_ARRAY, 4)),
        "int64" => Ok((OmDataType_t::DATA_TYPE_INT64_ARRAY, 8)),
        "uint64" => Ok((OmDataType_t::DATA_TYPE_UINT64_ARRAY, 8)),
        "float32" => Ok((OmDataType_t::DATA_TYPE_FLOAT_ARRAY, 4)),
        "float64" => Ok((OmDataType_t::DATA_TYPE_DOUBLE_ARRAY, 8)),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported dtype: {}",
            dtype_str
        ))),
    }
}

fn get_dtype_string(dtype: OmDataType_t) -> Option<&'static str> {
    match dtype {
        OmDataType_t::DATA_TYPE_INT8_ARRAY => Some("int8"),
        OmDataType_t::DATA_TYPE_UINT8_ARRAY => Some("uint8"),
        OmDataType_t::DATA_TYPE_INT16_ARRAY => Some("int16"),
        OmDataType_t::DATA_TYPE_UINT16_ARRAY => Some("uint16"),
        OmDataType_t::DATA_TYPE_INT32_ARRAY => Some("int32"),
        OmDataType_t::DATA_TYPE_UINT32_ARRAY => Some("uint32"),
        OmDataType_t::DATA_TYPE_INT64_ARRAY => Some("int64"),
        OmDataType_t::DATA_TYPE_UINT64_ARRAY => Some("uint64"),
        OmDataType_t::DATA_TYPE_FLOAT_ARRAY => Some("float32"),
        OmDataType_t::DATA_TYPE_DOUBLE_ARRAY => Some("float64"),
        _ => None,
    }
}

#[pyclass(module = "omfiles_numcodecs._omfiles_rs_bindings", dict)]
#[derive(Debug, Clone)]
pub struct PforDelta2dCodec {
    // Requires dtype (various int/float) at init time
    dtype: OmDataType_t,
    element_size: usize,
    compression: OmCompression_t,
}

#[pymethods]
impl PforDelta2dCodec {
    #[new]
    #[pyo3(signature = (dtype="int16"))]
    fn new(dtype: &str) -> PyResult<Self> {
        let (om_dtype, element_size) = get_dtype_info(dtype)?;
        // Check if dtype is supported by this specific compression in C
        match om_dtype {
            OmDataType_t::DATA_TYPE_INT8_ARRAY
            | OmDataType_t::DATA_TYPE_INT16_ARRAY
            | OmDataType_t::DATA_TYPE_INT32_ARRAY
            | OmDataType_t::DATA_TYPE_INT64_ARRAY
            | OmDataType_t::DATA_TYPE_UINT8_ARRAY
            | OmDataType_t::DATA_TYPE_UINT16_ARRAY
            | OmDataType_t::DATA_TYPE_UINT32_ARRAY
            | OmDataType_t::DATA_TYPE_UINT64_ARRAY => { /* Supported */ }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "{} does not support dtype '{}'",
                    CODEC_ID_PFOR_DELTA_2D, dtype
                )))
            }
        }

        Ok(PforDelta2dCodec {
            dtype: om_dtype,
            element_size,
            compression: OmCompression_t::COMPRESSION_PFOR_DELTA2D,
        })
    }

    #[getter]
    fn codec_id(&self) -> &'static str {
        CODEC_ID_PFOR_DELTA_2D
    }

    #[getter]
    fn dtype(&self) -> PyResult<String> {
        get_dtype_string(self.dtype)
            .map(String::from)
            .ok_or_else(|| PyValueError::new_err("Internal error: Invalid dtype stored"))
    }

    fn get_config(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("id", self.codec_id())?;
        dict.set_item("dtype", self.dtype()?)?;
        Ok(dict.into())
    }

    #[pyo3(signature = (array))]
    fn encode_array<'py>(&self, array: &Bound<'py, PyUntypedArray>) -> PyResult<Py<PyBytes>> {
        let py = array.py();
        let dtype = array.dtype();
        // let dtype = self.dtype();

        // Allocate output buffer with reasonable sizing
        let mut output_buffer: Vec<u8> = vec![0u8; array.len() * self.element_size + 1024];
        let output_ptr = output_buffer.as_mut_ptr() as *mut c_uchar;

        // Get contiguous data from numpy array
        let bytes_written = if dtype.is_equiv_to(&numpy::dtype::<i8>(py)) {
            let array = array.downcast::<PyArrayDyn<i8>>()?;
            let encoded_size = unsafe {
                om_file_format_sys::p4nzenc8(
                    array.as_slice_mut()?.as_mut_ptr() as *mut u8,
                    array.len(),
                    output_ptr,
                )
            };
            encoded_size
        } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
            let array = array.downcast::<PyArrayDyn<i16>>()?;
            let encoded_size = unsafe {
                om_file_format_sys::p4nzenc128v16(
                    array.as_slice_mut()?.as_mut_ptr() as *mut u16,
                    array.len(),
                    output_ptr,
                )
            };
            encoded_size
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
            let array = array.downcast::<PyArrayDyn<i32>>()?;
            let encoded_size = unsafe {
                om_file_format_sys::p4nzenc128v32(
                    array.as_slice_mut()?.as_mut_ptr() as *mut u32,
                    array.len(),
                    output_ptr,
                )
            };
            encoded_size
        } else if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
            let array = array.downcast::<PyArrayDyn<i64>>()?;
            let encoded_size = unsafe {
                om_file_format_sys::p4nzenc64(
                    array.as_slice_mut()?.as_mut_ptr() as *mut u64,
                    array.len(),
                    output_ptr,
                )
            };
            encoded_size
        } else if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
            println!("encode u8");
            let array = array.downcast::<PyArrayDyn<u8>>()?;
            let encoded_size = unsafe {
                om_file_format_sys::p4ndenc8(
                    array.as_slice_mut()?.as_mut_ptr(),
                    array.len(),
                    output_ptr,
                )
            };
            encoded_size
        } else if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
            let array = array.downcast::<PyArrayDyn<u16>>()?;
            let encoded_size = unsafe {
                om_file_format_sys::p4ndenc128v16(
                    array.as_slice_mut()?.as_mut_ptr(),
                    array.len(),
                    output_ptr,
                )
            };
            encoded_size
        } else if dtype.is_equiv_to(&numpy::dtype::<u32>(py)) {
            let array = array.downcast::<PyArrayDyn<u32>>()?;
            let encoded_size = unsafe {
                om_file_format_sys::p4ndenc128v32(
                    array.as_slice_mut()?.as_mut_ptr(),
                    array.len(),
                    output_ptr,
                )
            };
            encoded_size
        } else if dtype.is_equiv_to(&numpy::dtype::<u64>(py)) {
            let array = array.downcast::<PyArrayDyn<u64>>()?;
            let encoded_size = unsafe {
                om_file_format_sys::p4ndenc64(
                    array.as_slice_mut()?.as_mut_ptr(),
                    array.len(),
                    output_ptr,
                )
            };
            encoded_size
        } else {
            return Err(PyTypeError::new_err(format!(
                "Unsupported array dtype: {}",
                array.getattr("dtype")?
            )));
        };

        // Set the actual length and return PyBytes
        unsafe {
            output_buffer.set_len(bytes_written as usize);
        }

        Ok(PyBytes::new(py, &output_buffer).into())
    }

    #[pyo3(signature = (data, output_array))]
    fn decode_array<'py>(
        &self,
        data: &Bound<'py, PyArray1<i8>>,          // Compressed data
        output_array: Bound<'py, PyUntypedArray>, // Output buffer to store decompressed data
    ) -> PyResult<usize> {
        // Get the raw pointers to work with
        let input_ptr = unsafe { data.as_slice()? }.as_ptr();
        let input_size = data.len();
        // Empty data check
        if input_size == 0 {
            return Ok(0);
        }

        let py = data.py();
        let dtype = output_array.dtype();

        let bytes_decoded = if dtype.is_equiv_to(&numpy::dtype::<i8>(py)) {
            let array = output_array.downcast::<PyArrayDyn<i8>>()?;
            let _encoded_size = unsafe {
                om_file_format_sys::p4nzdec8(
                    input_ptr as *mut u8,
                    array.len(),
                    array.as_slice_mut()?.as_mut_ptr() as *mut u8,
                )
            };
            array.len()
        } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
            let array = output_array.downcast::<PyArrayDyn<i16>>()?;
            let _encoded_size = unsafe {
                om_file_format_sys::p4nzdec128v16(
                    input_ptr as *mut u8,
                    array.len(),
                    array.as_slice_mut()?.as_mut_ptr() as *mut u16,
                )
            };
            array.len()
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
            let array = output_array.downcast::<PyArrayDyn<i32>>()?;
            let _encoded_size = unsafe {
                om_file_format_sys::p4nzdec128v32(
                    input_ptr as *mut u8,
                    array.len(),
                    array.as_slice_mut()?.as_mut_ptr() as *mut u32,
                )
            };
            array.len()
        } else if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
            let array = output_array.downcast::<PyArrayDyn<i64>>()?;
            let _encoded_size = unsafe {
                om_file_format_sys::p4nzdec64(
                    input_ptr as *mut u8,
                    array.len(),
                    array.as_slice_mut()?.as_mut_ptr() as *mut u64,
                )
            };
            array.len()
        } else if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
            let array = output_array.downcast::<PyArrayDyn<u8>>()?;
            let _encoded_size = unsafe {
                om_file_format_sys::p4nddec8(
                    input_ptr as *mut u8,
                    array.len(),
                    array.as_slice_mut()?.as_mut_ptr() as *mut u8,
                )
            };
            array.len()
        } else if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
            let array = output_array.downcast::<PyArrayDyn<u16>>()?;
            let _encoded_size = unsafe {
                om_file_format_sys::p4nddec128v16(
                    input_ptr as *mut u8,
                    array.len(),
                    array.as_slice_mut()?.as_mut_ptr() as *mut u16,
                )
            };
            array.len()
        } else if dtype.is_equiv_to(&numpy::dtype::<u32>(py)) {
            let array = output_array.downcast::<PyArrayDyn<u32>>()?;
            let _encoded_size = unsafe {
                om_file_format_sys::p4nddec128v32(
                    input_ptr as *mut u8,
                    array.len(),
                    array.as_slice_mut()?.as_mut_ptr() as *mut u32,
                )
            };
            array.len()
        } else if dtype.is_equiv_to(&numpy::dtype::<u64>(py)) {
            let array = output_array.downcast::<PyArrayDyn<u64>>()?;
            let _encoded_size = unsafe {
                om_file_format_sys::p4nddec64(
                    input_ptr as *mut u8,
                    array.len(),
                    array.as_slice_mut()?.as_mut_ptr() as *mut u64,
                )
            };
            array.len()
        } else {
            return Err(PyTypeError::new_err(format!(
                "Unsupported array dtype: {}",
                output_array.getattr("dtype")?
            )));
        };

        Ok(bytes_decoded)
    }
}
