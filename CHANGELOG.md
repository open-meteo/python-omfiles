# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added Changelog
- Added example for a latitude-longitude plot
- Added Async Reader

### Fixed

- Fix type hint for shape property of OmFilePyReader
- Improved tests to use `pytest` fixtures
- Fix xarray contained attributes as variables
- Improve benchmarks slightly

## [0.0.2] - 2025-03-10

### Fixed

- Properly close reader and writer

## [0.0.1] - 2025-03-07

### Added
- Initial release of omfiles
- Support for reading .om files
- Integration with NumPy arrays
- xarray compatibility layer
