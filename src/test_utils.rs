#[cfg(test)]
pub use utils::*;

#[cfg(test)]
mod utils {
    use std::fs;
    use std::path::Path;

    use pyo3::types::PyAnyMethods;

    /// needed because venv dependencies are not supported out of the box in pyo3
    /// https://github.com/PyO3/pyo3/discussions/3726#discussioncomment-9061697
    pub fn pyo3_venv_path_hack(py: pyo3::Python<'_>) -> pyo3::PyResult<()> {
        // Try Unix-style first: .venv/lib/pythonX.Y/site-packages
        let unix_lib_path = Path::new(".venv/lib");
        let mut site_packages: Option<String> = None;

        if let Ok(entries) = fs::read_dir(unix_lib_path) {
            for entry in entries.flatten() {
                let path = entry.path().join("site-packages");
                if path.exists() {
                    if let Some(p) = path.to_str() {
                        site_packages = Some(p.to_string());
                        break;
                    }
                }
            }
        }

        // If not found, try Windows-style: .venv/Lib/site-packages
        if site_packages.is_none() {
            let win_path = Path::new(".venv/Lib/site-packages");
            if win_path.exists() {
                if let Some(p) = win_path.to_str() {
                    site_packages = Some(p.to_string());
                }
            }
        }

        // Add to sys.path if found
        if let Some(site_packages) = site_packages {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            path.call_method1("append", (site_packages,))?;
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "site-packages not found in .venv",
            ))
        }
    }

    // Helper function to ensure test directory exists
    pub fn ensure_test_dir() -> std::io::Result<()> {
        fs::create_dir_all("test_files")?;
        Ok(())
    }

    // Generate a simple binary file with specified bytes
    pub fn create_binary_file(filename: &str, data: &[u8]) -> std::io::Result<()> {
        ensure_test_dir()?;
        let path = Path::new("test_files").join(filename);
        fs::write(path, data)?;
        Ok(())
    }

    #[macro_export]
    macro_rules! create_test_binary_file {
        ($filename:expr) => {
            crate::test_utils::create_binary_file(
                $filename,
                &[
                    79, 77, 3, 0, 4, 130, 0, 2, 3, 34, 0, 4, 194, 2, 10, 4, 178, 0, 12, 4, 242, 0,
                    14, 197, 17, 20, 194, 2, 22, 194, 2, 24, 3, 3, 228, 200, 109, 1, 0, 0, 20, 0,
                    4, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
                    0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 100, 97, 116, 97, 0,
                    0, 0, 0, 79, 77, 3, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 0, 0,
                    0, 0,
                ],
            )
        };
    }
}
