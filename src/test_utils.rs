#[cfg(test)]
pub use utils::*;

#[cfg(test)]
mod utils {
    use std::fs;
    use std::path::Path;

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
