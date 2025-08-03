use numpy::{
    ndarray, IntoPyArray, PyArray1, PyArrayDescr, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::ffi::c_uchar;

fn get_dtype_size(dtype_str: &str) -> PyResult<usize> {
    match dtype_str {
        "int8" => Ok(1),
        "uint8" => Ok(1),
        "int16" => Ok(2),
        "uint16" => Ok(2),
        "int32" => Ok(4),
        "uint32" => Ok(4),
        "int64" => Ok(8),
        "uint64" => Ok(8),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported dtype: {}",
            dtype_str
        ))),
    }
}

/// RustPforCodec codec for compressing and decompressing integer arrays.
///
/// Supports numpy arrays of dtype: int8, int16, int32, int64, uint8, uint16, uint32, uint64.
#[gen_stub_pyclass]
#[pyclass(module = "omfiles.omfiles")]
#[derive(Debug, Clone)]
pub struct RustPforCodec {}

#[gen_stub_pymethods]
#[pymethods]
impl RustPforCodec {
    /// Create a new RustPforCodec codec instance.
    #[new]
    fn new() -> PyResult<Self> {
        Ok(RustPforCodec {})
    }

    /// Compress a numpy integer array using RustPforCodec.
    ///
    /// Args:
    ///     array: Input numpy array (integer dtype).
    ///     dtype: Numpy dtype of the array.
    ///
    /// Returns:
    ///     Compressed bytes.
    #[pyo3(signature = (array, dtype))]
    fn encode_array<'py>(
        &self,
        array: &Bound<'py, PyUntypedArray>,
        dtype: &Bound<'py, PyArrayDescr>,
    ) -> PyResult<Py<PyBytes>> {
        let py = array.py();

        // Allocate output buffer with reasonable sizing
        let element_size = get_dtype_size(&dtype.to_string())?;
        let mut output_buffer: Vec<u8> = vec![0u8; array.len() * element_size * 2 + 1024];
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

    /// Decompress RustPforCodec-compressed bytes into a numpy array.
    ///
    /// Args:
    ///     data: Compressed bytes.
    ///     dtype: Numpy dtype of the output array.
    ///     length: Number of elements in the output array.
    ///
    /// Returns:
    ///     Decompressed numpy array.
    #[pyo3(signature = (data, dtype, length))]
    fn decode_array<'py>(
        &self,
        data: &Bound<'py, PyBytes>, // Compressed data,
        dtype: &Bound<'py, PyArrayDescr>,
        length: usize,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        // Get the raw pointers to work with
        let input_ptr = data.as_bytes().as_ptr();
        let py = data.py();

        // Empty data check
        let input_size = data.len()?;
        if input_size == 0 {
            return Ok(PyArray1::<i8>::from_vec(py, vec![]).as_untyped().to_owned());
        }

        let untyped_array = if dtype.is_equiv_to(&numpy::dtype::<i8>(py)) {
            let mut array =
                ndarray::Array1::<i8>::from_shape_vec((length,), vec![0; length]).unwrap();
            let _encoded_size = unsafe {
                om_file_format_sys::p4nzdec8(
                    input_ptr as *mut u8,
                    length,
                    array.as_slice_mut().unwrap().as_mut_ptr() as *mut u8,
                )
            };
            array.into_pyarray(py).as_untyped().to_owned()
        } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
            let mut array =
                ndarray::Array1::<i16>::from_shape_vec((length,), vec![0; length]).unwrap();
            let _encoded_size = unsafe {
                om_file_format_sys::p4nzdec128v16(
                    input_ptr as *mut u8,
                    length,
                    array.as_slice_mut().unwrap().as_mut_ptr() as *mut u16,
                )
            };
            array.into_pyarray(py).as_untyped().to_owned()
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
            let mut array =
                ndarray::Array1::<i32>::from_shape_vec((length,), vec![0; length]).unwrap();
            let _encoded_size = unsafe {
                om_file_format_sys::p4nzdec128v32(
                    input_ptr as *mut u8,
                    length,
                    array.as_slice_mut().unwrap().as_mut_ptr() as *mut u32,
                )
            };
            array.into_pyarray(py).as_untyped().to_owned()
        } else if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
            let mut array =
                ndarray::Array1::<i64>::from_shape_vec((length,), vec![0; length]).unwrap();
            let _encoded_size = unsafe {
                om_file_format_sys::p4nzdec64(
                    input_ptr as *mut u8,
                    length,
                    array.as_slice_mut().unwrap().as_mut_ptr() as *mut u64,
                )
            };
            array.into_pyarray(py).as_untyped().to_owned()
        } else if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
            let mut array =
                ndarray::Array1::<u8>::from_shape_vec((length,), vec![0; length]).unwrap();
            let _encoded_size = unsafe {
                om_file_format_sys::p4nddec8(
                    input_ptr as *mut u8,
                    length,
                    array.as_slice_mut().unwrap().as_mut_ptr() as *mut u8,
                )
            };
            array.into_pyarray(py).as_untyped().to_owned()
        } else if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
            let mut array =
                ndarray::Array1::<u16>::from_shape_vec((length,), vec![0; length]).unwrap();
            let _encoded_size = unsafe {
                om_file_format_sys::p4nddec128v16(
                    input_ptr as *mut u8,
                    length,
                    array.as_slice_mut().unwrap().as_mut_ptr() as *mut u16,
                )
            };
            array.into_pyarray(py).as_untyped().to_owned()
        } else if dtype.is_equiv_to(&numpy::dtype::<u32>(py)) {
            let mut array =
                ndarray::Array1::<u32>::from_shape_vec((length,), vec![0; length]).unwrap();
            let _encoded_size = unsafe {
                om_file_format_sys::p4nddec128v32(
                    input_ptr as *mut u8,
                    length,
                    array.as_slice_mut().unwrap().as_mut_ptr() as *mut u32,
                )
            };
            array.into_pyarray(py).as_untyped().to_owned()
        } else if dtype.is_equiv_to(&numpy::dtype::<u64>(py)) {
            let mut array =
                ndarray::Array1::<u64>::from_shape_vec((length,), vec![0; length]).unwrap();
            let _encoded_size = unsafe {
                om_file_format_sys::p4nddec64(
                    input_ptr as *mut u8,
                    length,
                    array.as_slice_mut().unwrap().as_mut_ptr() as *mut u64,
                )
            };
            array.into_pyarray(py).as_untyped().to_owned()
        } else {
            return Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {}",
                dtype
            )));
        };

        Ok(untyped_array)
    }
}
