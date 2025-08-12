use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

#[gen_stub_pyfunction(module = "omfiles.omfiles")]
#[pyfunction]
/// Check if the CPU has the required features for SIMD compression operations.
pub fn _check_cpu_features() -> PyResult<()> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !std::arch::is_x86_feature_detected!("ssse3") {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "This library requires SSSE3 support on x86 CPUs.",
            ));
        }
    }
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "This library requires NEON support on ARM CPUs.",
            ));
        }
    }
    Ok(())
}
