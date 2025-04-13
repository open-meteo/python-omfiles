use numpy::{
    PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use om_file_format_sys::{OmCompression_t, OmDataType_t};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyByteArrayMethods, PyBytes, PyDict};
use std::ffi::c_void;

const CODEC_ID_PFOR_DELTA_2D_INT16: &str = "pfor_delta_2d_int16";
const CODEC_ID_FPX_XOR_2D: &str = "fpx_xor_2d";
const CODEC_ID_PFOR_DELTA_2D: &str = "pfor_delta_2d";
const CODEC_ID_PFOR_DELTA_2D_INT16_LOG: &str = "pfor_delta_2d_int16_logarithmic";

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

// --- Base trait helper (optional, for common logic) ---
// We won't use a trait here to keep individual classes explicit

// --- Codec Implementations ---

#[pyclass(module = "omfiles_numcodecs._omfiles_rs_bindings", dict)]
#[derive(Debug, Clone)]
pub struct PforDelta2dInt16Codec {
    // This codec specifically uses COMPRESSION_PFOR_DELTA2D_INT16
    // The C code indicates this expects DATA_TYPE_FLOAT_ARRAY but calls p4nzenc128v16 (uint16).
    // We'll assume the user MUST provide data interpretable as uint16.
    dtype: OmDataType_t,
    element_size: usize,
    compression: OmCompression_t,
}

#[pymethods]
impl PforDelta2dInt16Codec {
    #[new]
    fn new() -> Self {
        // Hardcoded based on the C function used (p4nzenc128v16)
        PforDelta2dInt16Codec {
            dtype: OmDataType_t::DATA_TYPE_UINT16_ARRAY, // Assuming uint16 based on C function
            element_size: 2,
            compression: OmCompression_t::COMPRESSION_PFOR_DELTA2D_INT16,
        }
    }

    #[getter]
    fn codec_id(&self) -> &'static str {
        CODEC_ID_PFOR_DELTA_2D_INT16
    }

    fn get_config(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("id", self.codec_id())?;
        Ok(dict.into())
    }

    // encode/decode methods (see implementation below, shared logic)
    fn encode<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Py<PyBytes>> {
        encode_via_ffi(py, data, self.dtype, self.element_size, self.compression)
    }

    #[pyo3(signature=(data, len, out=None))]
    fn decode<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
        len: usize,
        out: Option<Bound<'py, PyByteArray>>,
    ) -> PyResult<PyObject> {
        decode_via_ffi(
            py,
            data,
            len,
            self.dtype,
            self.element_size,
            self.compression,
            out,
        )
    }
}

#[pyclass(module = "omfiles_numcodecs._omfiles_rs_bindings", dict)]
#[derive(Debug, Clone)]
pub struct PforDelta2dInt16LogarithmicCodec {
    // Similar assumption as above
    dtype: OmDataType_t,
    element_size: usize,
    compression: OmCompression_t,
}

#[pymethods]
impl PforDelta2dInt16LogarithmicCodec {
    #[new]
    fn new() -> Self {
        PforDelta2dInt16LogarithmicCodec {
            dtype: OmDataType_t::DATA_TYPE_UINT16_ARRAY, // Assuming uint16 based on C function
            element_size: 2,
            compression: OmCompression_t::COMPRESSION_PFOR_DELTA2D_INT16_LOGARITHMIC,
        }
    }

    #[getter]
    fn codec_id(&self) -> &'static str {
        CODEC_ID_PFOR_DELTA_2D_INT16_LOG
    }

    fn get_config(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("id", self.codec_id())?;
        Ok(dict.into())
    }

    // encode/decode methods
    fn encode<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Py<PyBytes>> {
        encode_via_ffi(py, data, self.dtype, self.element_size, self.compression)
    }

    #[pyo3(signature=(data, len, out=None))]
    fn decode<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
        len: usize,
        out: Option<Bound<'py, PyByteArray>>,
    ) -> PyResult<PyObject> {
        decode_via_ffi(
            py,
            data,
            len,
            self.dtype,
            self.element_size,
            self.compression,
            out,
        )
    }
}

#[pyclass(module = "omfiles_numcodecs._omfiles_rs_bindings", dict)]
#[derive(Debug, Clone)]
pub struct FpxXor2dCodec {
    // Requires dtype (float32 or float64) at init time
    dtype: OmDataType_t,
    element_size: usize,
    compression: OmCompression_t,
}

#[pymethods]
impl FpxXor2dCodec {
    #[new]
    #[pyo3(signature = (dtype))]
    fn new(dtype: &str) -> PyResult<Self> {
        let (om_dtype, element_size) = get_dtype_info(dtype)?;
        if om_dtype != OmDataType_t::DATA_TYPE_FLOAT_ARRAY
            && om_dtype != OmDataType_t::DATA_TYPE_DOUBLE_ARRAY
        {
            return Err(PyValueError::new_err(format!(
                "{} only supports 'float32' or 'float64' dtypes, got '{}'",
                CODEC_ID_FPX_XOR_2D, dtype
            )));
        }
        Ok(FpxXor2dCodec {
            dtype: om_dtype,
            element_size,
            compression: OmCompression_t::COMPRESSION_FPX_XOR2D,
        })
    }

    #[getter]
    fn codec_id(&self) -> &'static str {
        CODEC_ID_FPX_XOR_2D
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

    // encode/decode methods
    fn encode<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Py<PyBytes>> {
        encode_via_ffi(py, data, self.dtype, self.element_size, self.compression)
    }

    #[pyo3(signature=(data, len, out=None))]
    fn decode<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
        len: usize,
        out: Option<Bound<'py, PyByteArray>>,
    ) -> PyResult<PyObject> {
        decode_via_ffi(
            py,
            data,
            len,
            self.dtype,
            self.element_size,
            self.compression,
            out,
        )
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

    // encode/decode methods
    fn encode<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Py<PyBytes>> {
        encode_via_ffi(py, data, self.dtype, self.element_size, self.compression)
    }

    #[pyo3(signature = (array))]
    fn encode_array<'py>(&self, array: &Bound<'py, PyArray1<i8>>) -> PyResult<Py<PyBytes>> {
        let py = array.py();
        // Get contiguous data from numpy array
        let data_ptr = unsafe { array.as_slice()? }.as_ptr() as *const c_void;
        let data_size = array.len();

        // Ensure the data is correctly aligned
        if data_size % self.element_size != 0 {
            return Err(PyValueError::new_err(format!(
                "Input array size ({}) is not a multiple of element size ({})",
                data_size, self.element_size
            )));
        }

        // Calculate element count
        let count = (data_size / self.element_size) as u64;

        // Allocate output buffer with reasonable sizing
        let mut output_buffer: Vec<u8> = vec![0u8; data_size * 2 + 1024]; // Same sizing logic
        let output_ptr = output_buffer.as_mut_ptr() as *mut c_void;

        // Call FFI function
        let bytes_written = unsafe {
            om_file_format_sys::om_encode_compress(
                self.dtype,
                self.compression,
                data_ptr,
                count,
                output_ptr,
            )
        };

        // Handle possible errors
        if bytes_written == 0 && count > 0 && self.compression != OmCompression_t::COMPRESSION_NONE
        {
            println!(
                "Warning: Compression returned 0 bytes for {} elements",
                count
            );
        }

        if bytes_written as usize > output_buffer.capacity() {
            return Err(PyValueError::new_err(format!(
                "FFI compression wrote {} bytes, exceeding buffer capacity {}",
                bytes_written,
                output_buffer.capacity()
            )));
        }

        // Set the actual length and return PyBytes
        unsafe {
            output_buffer.set_len(bytes_written as usize);
        }

        Ok(PyBytes::new(py, &output_buffer).into())
    }

    #[pyo3(signature=(data, len, out=None))]
    fn decode<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
        len: usize,
        out: Option<Bound<'py, PyByteArray>>,
    ) -> PyResult<PyObject> {
        decode_via_ffi(
            py,
            data,
            len,
            self.dtype,
            self.element_size,
            self.compression,
            out,
        )
    }

    #[pyo3(signature = (data, output_array))]
    fn decode_array<'py>(
        &self,
        data: &Bound<'py, PyArray1<i8>>,          // Compressed data
        output_array: Bound<'py, PyUntypedArray>, // Output buffer to store decompressed data
    ) -> PyResult<usize> {
        // Get the raw pointers to work with
        let input_ptr = unsafe { data.as_slice()? }.as_ptr() as *const c_void;
        let input_size = data.len();
        // Empty data check
        if input_size == 0 {
            return Ok(0);
        }

        let py = data.py();
        let dtype = output_array.dtype();

        let (output_ptr, output_elements) = if dtype.is_equiv_to(&numpy::dtype::<i8>(py)) {
            let array = output_array.downcast::<PyArrayDyn<i8>>()?;
            (
                unsafe { array.as_slice_mut()?.as_mut_ptr() as *mut c_void },
                array.len() / self.element_size, // FIXME!!!
            )
        } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
            let array = output_array.downcast::<PyArrayDyn<i16>>()?;
            (
                unsafe { array.as_slice_mut()?.as_mut_ptr() as *mut c_void },
                array.len(),
            )
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
            let array = output_array.downcast::<PyArrayDyn<i32>>()?;
            (
                unsafe { array.as_slice_mut()?.as_mut_ptr() as *mut c_void },
                array.len(),
            )
        } else if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
            let array = output_array.downcast::<PyArrayDyn<i64>>()?;
            (
                unsafe { array.as_slice_mut()?.as_mut_ptr() as *mut c_void },
                array.len(),
            )
        } else if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
            let array = output_array.downcast::<PyArrayDyn<u8>>()?;
            (
                unsafe { array.as_slice_mut()?.as_mut_ptr() as *mut c_void },
                array.len(),
            )
        } else if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
            let array = output_array.downcast::<PyArrayDyn<u16>>()?;
            (
                unsafe { array.as_slice_mut()?.as_mut_ptr() as *mut c_void },
                array.len(),
            )
        } else if dtype.is_equiv_to(&numpy::dtype::<u32>(py)) {
            let array = output_array.downcast::<PyArrayDyn<u32>>()?;
            (
                unsafe { array.as_slice_mut()?.as_mut_ptr() as *mut c_void },
                array.len(),
            )
        } else if dtype.is_equiv_to(&numpy::dtype::<u64>(py)) {
            let array = output_array.downcast::<PyArrayDyn<u64>>()?;
            (
                unsafe { array.as_slice_mut()?.as_mut_ptr() as *mut c_void },
                array.len(),
            )
        } else {
            return Err(PyTypeError::new_err(format!(
                "Unsupported array dtype: {}",
                output_array.getattr("dtype")?
            )));
        };

        // Call the C FFI decode function
        unsafe {
            om_file_format_sys::om_decode_decompress(
                self.dtype,
                self.compression,
                input_ptr,
                output_elements as u64,
                output_ptr,
            )
        };

        // Return the actual number of bytes decoded
        let bytes_decoded = output_elements * self.element_size;
        Ok(bytes_decoded)
    }
}

// --- Shared FFI Wrapper Logic ---
#[inline(always)]
fn encode_via_ffi<'py>(
    py: Python<'py>,
    data: &[u8],
    dtype: OmDataType_t,
    element_size: usize,
    compression: OmCompression_t,
) -> PyResult<Py<PyBytes>> {
    if element_size == 0 {
        return Err(PyValueError::new_err("Invalid element size (0)"));
    }
    if data.len() % element_size != 0 {
        return Err(PyValueError::new_err(format!(
            "Input data length ({}) is not a multiple of element size ({}) for dtype {:?}",
            data.len(),
            element_size,
            dtype
        )));
    }
    let count = (data.len() / element_size) as u64;
    if count == 0 && data.len() == 0 {
        // Handle empty input gracefully
        return Ok(PyBytes::new(py, &[]).into());
    }
    if count == 0 {
        // Non-empty data but zero elements? Should be caught by modulo check, but safety first.
        return Err(PyValueError::new_err(
            "Invalid input size leading to zero element count",
        ));
    }

    // Allocate output buffer: These compressors can potentially expand data.
    // A common strategy is input size + some overhead, or a multiple.
    // Let's start with 2 * input size + a fixed overhead (e.g., 1KB) as a guess.
    // This might need tuning based on worst-case behavior of PFor/FPX.
    let mut output_buffer: Vec<u8> =
        vec![0u8; data.len().saturating_mul(element_size).saturating_add(1024)];
    let output_ptr = output_buffer.as_mut_ptr() as *mut c_void;
    let input_ptr = data.as_ptr() as *const c_void;

    // Call the C FFI function
    let bytes_written = unsafe {
        // We need to ensure the vec capacity is actually available for writing by C
        // This is inherently unsafe if the C code writes beyond the vec's *initialized* length,
        // but standard practice for FFI often involves passing capacity and letting C write into it.
        // After the call, we'll set the length based on the return value.
        om_file_format_sys::om_encode_compress(dtype, compression, input_ptr, count, output_ptr)
    };

    if bytes_written == 0 && count > 0 {
        // Consider 0 bytes written for non-empty input an error, unless COMPRESSION_NONE
        if compression != OmCompression_t::COMPRESSION_NONE {
            println!(
                "Warning: om_encode_compress returned 0 bytes written for count={}",
                count
            );
            // It might be valid for some inputs, but could indicate an issue.
            // Depending on C API, 0 might be an error code. Check C API docs.
            // For now, let's proceed but maybe log.
        }
    }

    if bytes_written as usize > output_buffer.capacity() {
        // This should ideally not happen if the capacity was sufficient, but check anyway.
        return Err(PyValueError::new_err(format!(
            "FFI compression wrote {} bytes, exceeding allocated buffer capacity {}",
            bytes_written,
            output_buffer.capacity()
        )));
    }

    // Set the actual length of the vector based on what the C function wrote
    unsafe {
        output_buffer.set_len(bytes_written as usize);
    }

    Ok(PyBytes::new(py, &output_buffer).into())
}

#[inline(always)]
fn decode_via_ffi<'py>(
    py: Python<'py>,
    data: &[u8],            // Compressed data
    output_elements: usize, // Expected elements to be decoded
    dtype: OmDataType_t,
    element_size: usize,
    compression: OmCompression_t,
    out: Option<Bound<'py, PyByteArray>>,
) -> PyResult<PyObject> {
    if element_size == 0 {
        return Err(PyValueError::new_err("Invalid element size (0)"));
    }
    let input_size = data.len() as u64;
    let input_ptr = data.as_ptr() as *const c_void;

    if input_size == 0 {
        // Decompressing empty data should result in empty data
        if let Some(output_buffer) = out {
            if output_buffer.len() != 0 {
                return Err(PyValueError::new_err(
                    "Output buffer must be empty when decompressing empty data",
                ));
            }
            return Ok(0usize.to_object(py)); // 0 bytes written
        } else {
            return Ok(PyBytes::new(py, &[]).to_object(py));
        }
    }

    match out {
        Some(output_buffer) => {
            // Decode directly into the provided PyByteArray
            let output_ptr = unsafe { output_buffer.as_bytes_mut().as_mut_ptr() as *mut c_void };

            // Call the C FFI decode function
            unsafe {
                om_file_format_sys::om_decode_decompress(
                    dtype,
                    compression,
                    input_ptr,
                    output_elements as u64,
                    output_ptr,
                )
            };

            let bytes_decoded = output_elements as usize * element_size;

            Ok(bytes_decoded.to_object(py))
        }
        None => {
            // Decode into a new buffer. We need to know the output size.
            // Problem: The C API `om_decode_decompress` seems to require `max_count`.
            // Can we call it with NULL output to get the size? Assume NO for now.
            // Strategy: Overallocate, decode, then copy the correct slice.

            // Guess initial allocation size. E.g., compressed size * typical ratio (e.g., 5x?) + overhead
            // This is inefficient and potentially insufficient.
            // **A BETTER C API would provide a way to get the decoded size first.**
            // let mut estimated_elements =
            //     (input_size as usize).saturating_mul(6) / element_size + 1024; // Wild guess
            // if estimated_elements == 0 {
            //     estimated_elements = 1024
            // }; // Minimum guess
            let mut output_buffer: Vec<u8> = vec![0u8; output_elements * element_size];
            let output_ptr = output_buffer.as_mut_ptr() as *mut c_void;

            // Call the C FFI decode function
            unsafe {
                // Safety: C must not write past capacity.
                om_file_format_sys::om_decode_decompress(
                    dtype,
                    compression,
                    input_ptr,
                    output_elements as u64,
                    output_ptr,
                )
            };

            Ok(PyBytes::new(py, &output_buffer).to_object(py))
        }
    }
}
