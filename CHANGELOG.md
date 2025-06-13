# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0](https://github.com/open-meteo/python-omfiles/compare/v0.0.2...v0.1.0) (2025-06-13)


### Features

* Async Python Reader Interface ([#26](https://github.com/open-meteo/python-omfiles/issues/26)) ([d714ffe](https://github.com/open-meteo/python-omfiles/commit/d714ffee782baeeee01c2c59a5efc5759cfea9a8))
* fenerated type stubs ([#28](https://github.com/open-meteo/python-omfiles/issues/28)) ([effb29d](https://github.com/open-meteo/python-omfiles/commit/effb29d1ace5fcc86264df55d7280538a8deefbc))


### Bug Fixes

* type hint for OmFilePyReader.shape ([a03e581](https://github.com/open-meteo/python-omfiles/commit/a03e581bc1da260411c70299237da1cf2babc947))
* wrong usage of zarr create_array compressor ([2f221c4](https://github.com/open-meteo/python-omfiles/commit/2f221c4fe5f10cd3b9ad56550a4b545f130bad0a))
* xarray contained attributes as variables ([#23](https://github.com/open-meteo/python-omfiles/issues/23)) ([8fac64d](https://github.com/open-meteo/python-omfiles/commit/8fac64d0a208cb3775533637e3767e916260bd32))

## [Unreleased]

### Added

- Added Changelog
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
