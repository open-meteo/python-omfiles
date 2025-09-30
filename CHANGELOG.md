# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0](https://github.com/open-meteo/python-omfiles/compare/v0.1.1...v1.0.0) (2025-09-30)


### Miscellaneous Chores

* release 1.0.0 ([1141aca](https://github.com/open-meteo/python-omfiles/commit/1141aca22717dc6bd083dc6b1c94148600f357ab))

## [0.1.1](https://github.com/open-meteo/python-omfiles/compare/v0.1.0...v0.1.1) (2025-09-29)


### Bug Fixes

* add more metadata to pyproject.toml ([1344631](https://github.com/open-meteo/python-omfiles/commit/1344631247f10a130f94d819eddacfb6c9dc7d87))
* missing readme on pypi because not included in sdist ([294295f](https://github.com/open-meteo/python-omfiles/commit/294295fd9636586c3e99319cf2117310cf0bc2bc))

## [0.1.0](https://github.com/open-meteo/python-omfiles/compare/v0.0.2...v0.1.0) (2025-09-27)


### Features

* add compression property to reader ([#44](https://github.com/open-meteo/python-omfiles/issues/44)) ([150f6a1](https://github.com/open-meteo/python-omfiles/commit/150f6a1f8b6f6b1e93de3712681e54c4db23a545))
* add pfor codecs ([#25](https://github.com/open-meteo/python-omfiles/issues/25)) ([7bfeb2c](https://github.com/open-meteo/python-omfiles/commit/7bfeb2c7229c29aea777ce96a07bead0dda67104))
* Async Python Reader Interface ([#26](https://github.com/open-meteo/python-omfiles/issues/26)) ([d714ffe](https://github.com/open-meteo/python-omfiles/commit/d714ffee782baeeee01c2c59a5efc5759cfea9a8))
* bump dependencies ([#56](https://github.com/open-meteo/python-omfiles/issues/56)) ([7ecfc09](https://github.com/open-meteo/python-omfiles/commit/7ecfc0907edde12e30c703a577d06e796801ee82))
* check availability of cpu features on import ([#52](https://github.com/open-meteo/python-omfiles/issues/52)) ([a116604](https://github.com/open-meteo/python-omfiles/commit/a116604c1d50e22036c213e9fd6b0f6d774c00e2))
* docs ([#46](https://github.com/open-meteo/python-omfiles/issues/46)) ([bd0470b](https://github.com/open-meteo/python-omfiles/commit/bd0470bb6919e15d727b67dd933d93343809d63f))
* fenerated type stubs ([#28](https://github.com/open-meteo/python-omfiles/issues/28)) ([effb29d](https://github.com/open-meteo/python-omfiles/commit/effb29d1ace5fcc86264df55d7280538a8deefbc))
* fsspec support for writer ([#45](https://github.com/open-meteo/python-omfiles/issues/45)) ([a429a30](https://github.com/open-meteo/python-omfiles/commit/a429a303ccdec40ce8dd407f768107ef514881b0))
* optional dependencies ([#47](https://github.com/open-meteo/python-omfiles/issues/47)) ([c9f9252](https://github.com/open-meteo/python-omfiles/commit/c9f92524f71931aebb35eaeb9bae0172bc626bff))
* public API and documentation improvements ([#55](https://github.com/open-meteo/python-omfiles/issues/55)) ([1daf92e](https://github.com/open-meteo/python-omfiles/commit/1daf92eef97d057563abae7371f80865980ec936))
* release gil during array read ([#48](https://github.com/open-meteo/python-omfiles/issues/48)) ([0d346c0](https://github.com/open-meteo/python-omfiles/commit/0d346c0941d82996229aabef8ea6ee2d5c68eb94))
* update readme and docs ([#57](https://github.com/open-meteo/python-omfiles/issues/57)) ([c1ef3fe](https://github.com/open-meteo/python-omfiles/commit/c1ef3fedb9137fe9e69341f397395b9f34de4c89))


### Bug Fixes

* ci cache improvements ([#58](https://github.com/open-meteo/python-omfiles/issues/58)) ([edd687a](https://github.com/open-meteo/python-omfiles/commit/edd687ad8516ad5bcf08a3d9c39fdb4a88a064c0))
* codec registration ([0a6e572](https://github.com/open-meteo/python-omfiles/commit/0a6e572942b60b5677760f5d7176c5832aae2b87))
* minor improvement in README.md ([e01df5e](https://github.com/open-meteo/python-omfiles/commit/e01df5e824d3ef2aa2e7dda2bc7c91c805c735e0))
* run docs on push on main branch ([00df254](https://github.com/open-meteo/python-omfiles/commit/00df25483348507cd98d4d3c43d6d5e81ee14ef3))
* shape should be a tuple ([#43](https://github.com/open-meteo/python-omfiles/issues/43)) ([8546752](https://github.com/open-meteo/python-omfiles/commit/85467520c29198f8958cb2f997d7179d5216b8fe))
* type hint for OmFilePyReader.shape ([a03e581](https://github.com/open-meteo/python-omfiles/commit/a03e581bc1da260411c70299237da1cf2babc947))
* wrong usage of zarr create_array compressor ([2f221c4](https://github.com/open-meteo/python-omfiles/commit/2f221c4fe5f10cd3b9ad56550a4b545f130bad0a))
* xarray contained attributes as variables ([#23](https://github.com/open-meteo/python-omfiles/issues/23)) ([8fac64d](https://github.com/open-meteo/python-omfiles/commit/8fac64d0a208cb3775533637e3767e916260bd32))
* zarr codec behavior for different zarr versions ([#50](https://github.com/open-meteo/python-omfiles/issues/50)) ([8c6e516](https://github.com/open-meteo/python-omfiles/commit/8c6e5161826f1989e18b4f010b3019eecb66e86c))

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
