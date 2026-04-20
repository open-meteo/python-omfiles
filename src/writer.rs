use crate::{
    compression::PyCompressionType, errors::convert_omfilesrs_error,
    fsspec_backend::FsSpecWriterBackend, hierarchy::OmWriterVariable,
};
use delegate::delegate;
use numpy::{
    dtype, Element, PyArrayDescr, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods,
};
use omfiles_rs::{
    traits::{OmFileArrayDataType, OmFileScalarDataType, OmFileWriterBackend},
    writer::{OmFileWriter as OmFileWriterRs, OmFileWriterArray, OmFileWriterArrayFinalized},
    OmCompressionType, OmFilesError, OmOffsetSize,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyStopIteration, PyValueError},
    prelude::*,
    types::PyIterator,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::{
    collections::HashMap,
    fs::File,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex, PoisonError,
    },
};

static NEXT_WRITER_ID: AtomicU64 = AtomicU64::new(1);

fn next_writer_id() -> u64 {
    NEXT_WRITER_ID.fetch_add(1, Ordering::Relaxed)
}

fn to_writer_variable(name: &str, writer_id: u64, variable_id: u64) -> OmWriterVariable {
    OmWriterVariable {
        name: name.to_string(),
        writer_id,
        variable_id,
    }
}

/// All array element types supported by the writer.
enum OmElementType {
    Float32,
    Float64,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Uint64,
}

impl OmElementType {
    /// Resolve from a numpy `PyArrayDescr` (used by `write_array`).
    fn from_numpy_dtype(d: &Bound<'_, PyArrayDescr>) -> PyResult<Self> {
        let py = d.py();
        if d.is_equiv_to(&dtype::<f32>(py)) {
            Ok(Self::Float32)
        } else if d.is_equiv_to(&dtype::<f64>(py)) {
            Ok(Self::Float64)
        } else if d.is_equiv_to(&dtype::<i8>(py)) {
            Ok(Self::Int8)
        } else if d.is_equiv_to(&dtype::<u8>(py)) {
            Ok(Self::Uint8)
        } else if d.is_equiv_to(&dtype::<i16>(py)) {
            Ok(Self::Int16)
        } else if d.is_equiv_to(&dtype::<u16>(py)) {
            Ok(Self::Uint16)
        } else if d.is_equiv_to(&dtype::<i32>(py)) {
            Ok(Self::Int32)
        } else if d.is_equiv_to(&dtype::<u32>(py)) {
            Ok(Self::Uint32)
        } else if d.is_equiv_to(&dtype::<i64>(py)) {
            Ok(Self::Int64)
        } else if d.is_equiv_to(&dtype::<u64>(py)) {
            Ok(Self::Uint64)
        } else {
            Err(OmFileWriter::unsupported_array_type_error(d.clone()))
        }
    }
}

/// Abstracts over the two write strategies (full array vs streaming iterator).
enum Feeder<'a, 'py> {
    /// Feeds an entire numpy array in a single `write_data` call.
    Full {
        data: &'a Bound<'py, PyUntypedArray>,
    },
    /// Feeds data chunk-by-chunk from a Python iterator.
    Streaming {
        py: Python<'py>,
        iter: Bound<'py, PyAny>,
    },
}

impl<'a, 'py> Feeder<'a, 'py> {
    fn feed<T: Element + OmFileArrayDataType>(
        self,
        w: &mut OmFileWriterArray<'_, T, WriterBackendImpl>,
    ) -> PyResult<()> {
        match self {
            Feeder::Full { data } => {
                let array = data.cast::<PyArrayDyn<T>>()?.readonly();
                w.write_data(array.as_array(), None, None)
                    .map_err(convert_omfilesrs_error)
            }
            Feeder::Streaming { py, iter } => loop {
                match iter.call_method0("__next__") {
                    Ok(item) => {
                        let array: PyReadonlyArrayDyn<'_, T> = item.extract()?;
                        w.write_data(array.as_array(), None, None)
                            .map_err(convert_omfilesrs_error)?;
                    }
                    Err(err) if err.is_instance_of::<PyStopIteration>(py) => break Ok(()),
                    Err(err) => break Err(err),
                }
            },
        }
    }
}

/// Resolved parameters shared by both `write_array` and `write_array_streaming`.
struct WriteArrayParams<'a> {
    name: &'a str,
    children: Vec<OmWriterVariable>,
    scale_factor: f32,
    add_offset: f32,
    compression: OmCompressionType,
}

impl<'a> WriteArrayParams<'a> {
    fn from_options(
        name: Option<&'a str>,
        children: Option<Vec<OmWriterVariable>>,
        scale_factor: Option<f32>,
        add_offset: Option<f32>,
        compression: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            name: name.unwrap_or("data"),
            children: children.unwrap_or_default(),
            scale_factor: scale_factor.unwrap_or(1.0),
            add_offset: add_offset.unwrap_or(0.0),
            compression: compression
                .map(PyCompressionType::from_str)
                .transpose()?
                .unwrap_or(PyCompressionType::PforDelta2d)
                .to_omfilesrs(),
        })
    }
}

enum DeferredVariableKind {
    Resolved,
    Scalar {
        value: DeferredScalarValue,
        children: Vec<u64>,
    },
    Array {
        array: Option<OmFileWriterArrayFinalized>,
        children: Vec<u64>,
    },
    Group {
        children: Vec<u64>,
    },
}

#[derive(Clone)]
enum DeferredScalarValue {
    String(String),
    Float64(f64),
    Float32(f32),
    Int64(i64),
    Int32(i32),
    Int16(i16),
    Int8(i8),
    Uint64(u64),
    Uint32(u32),
    Uint16(u16),
    Uint8(u8),
}

struct DeferredVariable {
    name: String,
    kind: DeferredVariableKind,
    resolved: Option<OmOffsetSize>,
}

struct WriterState {
    writer: OmFileWriterRs<WriterBackendImpl>,
    next_variable_id: u64,
    variables: HashMap<u64, DeferredVariable>,
}

impl WriterState {
    fn new(writer: OmFileWriterRs<WriterBackendImpl>) -> Self {
        Self {
            writer,
            next_variable_id: 1,
            variables: HashMap::new(),
        }
    }

    fn allocate_variable_id(&mut self) -> u64 {
        let variable_id = self.next_variable_id;
        self.next_variable_id += 1;
        variable_id
    }

    fn register_resolved(
        &mut self,
        name: &str,
        kind: DeferredVariableKind,
        offset_size: OmOffsetSize,
    ) -> u64 {
        let variable_id = self.allocate_variable_id();
        self.variables.insert(
            variable_id,
            DeferredVariable {
                name: name.to_string(),
                kind,
                resolved: Some(offset_size),
            },
        );
        variable_id
    }

    fn register_deferred(&mut self, name: &str, kind: DeferredVariableKind) -> u64 {
        let variable_id = self.allocate_variable_id();
        self.variables.insert(
            variable_id,
            DeferredVariable {
                name: name.to_string(),
                kind,
                resolved: None,
            },
        );
        variable_id
    }

    fn ensure_children_exist(&self, children: &[u64]) -> Result<(), OmFilesError> {
        for child in children {
            if !self.variables.contains_key(child) {
                return Err(OmFilesError::GenericError(format!(
                    "Unknown child variable id {}",
                    child
                )));
            }
        }
        Ok(())
    }

    fn resolved_children_inline(
        &self,
        children: &[u64],
    ) -> Result<Vec<OmOffsetSize>, OmFilesError> {
        let mut resolved = Vec::with_capacity(children.len());
        for child in children {
            let variable = self.variables.get(child).ok_or_else(|| {
                OmFilesError::GenericError(format!("Unknown child variable id {}", child))
            })?;
            let offset_size = variable.resolved.clone().ok_or_else(|| {
                OmFilesError::GenericError(format!(
                    "Child variable '{}' is not yet resolved for inline metadata placement",
                    variable.name
                ))
            })?;
            resolved.push(offset_size);
        }
        Ok(resolved)
    }

    fn resolve_variable(
        &mut self,
        variable_id: u64,
        resolving: &mut Vec<u64>,
    ) -> Result<OmOffsetSize, OmFilesError> {
        if resolving.contains(&variable_id) {
            return Err(OmFilesError::GenericError(
                "Cycle detected in deferred variable hierarchy".to_string(),
            ));
        }

        if let Some(variable) = self.variables.get(&variable_id) {
            if let Some(offset_size) = &variable.resolved {
                return Ok(offset_size.clone());
            }
        } else {
            return Err(OmFilesError::GenericError(format!(
                "Unknown variable id {}",
                variable_id
            )));
        };

        // Temporarily take ownership of the variable by removing it
        let mut variable = self.variables.remove(&variable_id).unwrap();
        resolving.push(variable_id);

        let child_ids = match &variable.kind {
            DeferredVariableKind::Scalar { children, .. } => children,
            DeferredVariableKind::Array { children, .. } => children,
            DeferredVariableKind::Group { children } => children,
            DeferredVariableKind::Resolved => unreachable!("Resolved variables return early"),
        };

        let mut resolved_children = Vec::with_capacity(child_ids.len());
        for &child_id in child_ids {
            resolved_children.push(self.resolve_variable(child_id, resolving)?);
        }

        let resolved = match &mut variable.kind {
            DeferredVariableKind::Scalar { value, .. } => match value {
                DeferredScalarValue::String(v) => {
                    self.writer
                        .write_scalar(v.clone(), &variable.name, &resolved_children)?
                }
                DeferredScalarValue::Float64(v) => {
                    self.writer
                        .write_scalar(*v, &variable.name, &resolved_children)?
                }
                DeferredScalarValue::Float32(v) => {
                    self.writer
                        .write_scalar(*v, &variable.name, &resolved_children)?
                }
                DeferredScalarValue::Int64(v) => {
                    self.writer
                        .write_scalar(*v, &variable.name, &resolved_children)?
                }
                DeferredScalarValue::Int32(v) => {
                    self.writer
                        .write_scalar(*v, &variable.name, &resolved_children)?
                }
                DeferredScalarValue::Int16(v) => {
                    self.writer
                        .write_scalar(*v, &variable.name, &resolved_children)?
                }
                DeferredScalarValue::Int8(v) => {
                    self.writer
                        .write_scalar(*v, &variable.name, &resolved_children)?
                }
                DeferredScalarValue::Uint64(v) => {
                    self.writer
                        .write_scalar(*v, &variable.name, &resolved_children)?
                }
                DeferredScalarValue::Uint32(v) => {
                    self.writer
                        .write_scalar(*v, &variable.name, &resolved_children)?
                }
                DeferredScalarValue::Uint16(v) => {
                    self.writer
                        .write_scalar(*v, &variable.name, &resolved_children)?
                }
                DeferredScalarValue::Uint8(v) => {
                    self.writer
                        .write_scalar(*v, &variable.name, &resolved_children)?
                }
            },
            DeferredVariableKind::Array { array, .. } => {
                let finalized = array.take().ok_or_else(|| {
                    OmFilesError::GenericError(
                        "Deferred array metadata was already consumed".to_string(),
                    )
                })?;
                self.writer
                    .write_array(finalized, &variable.name, &resolved_children)?
            }
            DeferredVariableKind::Group { .. } => {
                self.writer.write_none(&variable.name, &resolved_children)?
            }
            DeferredVariableKind::Resolved => unreachable!("Resolved variables return early"),
        };

        // Update the resolved state and put the variable back into the map
        variable.resolved = Some(resolved.clone());
        self.variables.insert(variable_id, variable);

        resolving.pop();
        Ok(resolved)
    }
}

/// A Python wrapper for the Rust OmFileWriter implementation.
#[gen_stub_pyclass]
#[pyclass]
pub struct OmFileWriter {
    writer: Mutex<Option<WriterState>>,
    writer_id: u64,
    deferred_tail_metadata: bool,
    explicitly_closed: AtomicBool,
}

impl OmFileWriter {
    fn lock_error<T>(e: PoisonError<T>) -> PyErr {
        PyErr::new::<PyRuntimeError, _>(format!("Failed to acquire lock on writer: {}", e))
    }

    fn closed_error() -> PyErr {
        PyErr::new::<PyValueError, _>("I/O operation on closed writer")
    }

    fn unsupported_array_type_error(dtype: Bound<'_, PyArrayDescr>) -> PyErr {
        let type_name = dtype
            .typeobj()
            .name()
            .map(|s| s.to_string())
            .unwrap_or("unknown type".to_string());
        PyErr::new::<PyValueError, _>(format!("Unsupported array data type: {}", type_name))
    }

    fn unsupported_scalar_type_error(dtype: Bound<'_, pyo3::types::PyType>) -> PyErr {
        let type_name = dtype
            .name()
            .map(|s| s.to_string())
            .unwrap_or("unknown type".to_string());
        PyErr::new::<PyValueError, _>(format!("Unsupported scalar data type: {}", type_name))
    }

    fn invalid_metadata_placement_error(value: &str) -> PyErr {
        PyErr::new::<PyValueError, _>(format!("Unsupported metadata placement: {}", value))
    }

    fn validate_metadata_placement(metadata_placement: Option<&str>) -> PyResult<String> {
        let placement = metadata_placement.unwrap_or("tail");
        match placement {
            "inline" | "tail" => Ok(placement.to_string()),
            other => Err(Self::invalid_metadata_placement_error(other)),
        }
    }

    fn validate_writer_variable(&self, variable: &OmWriterVariable) -> PyResult<()> {
        if variable.writer_id != self.writer_id {
            return Err(PyErr::new::<PyValueError, _>(
                "Variable handle belongs to a different writer",
            ));
        }
        Ok(())
    }

    fn validate_writer_variables(&self, variables: &[OmWriterVariable]) -> PyResult<()> {
        for variable in variables {
            self.validate_writer_variable(variable)?;
        }
        Ok(())
    }

    fn child_ids(&self, children: &[OmWriterVariable]) -> PyResult<Vec<u64>> {
        self.validate_writer_variables(children)?;
        Ok(children.iter().map(|child| child.variable_id).collect())
    }

    fn with_state<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut WriterState) -> PyResult<R>,
    {
        let mut guard = self.writer.lock().map_err(Self::lock_error)?;
        match guard.as_mut() {
            Some(state) => f(state),
            None => Err(Self::closed_error()),
        }
    }

    /// Unified 10-way type dispatch.
    ///
    /// Resolves `element_type` to a concrete `T`, prepares a typed array writer,
    /// feeds data, finalizes, and registers the result as a named variable.
    /// Both `write_array` and `write_array_streaming` delegate here.
    fn write_array_dispatched(
        &self,
        element_type: OmElementType,
        dimensions: Vec<u64>,
        chunks: Vec<u64>,
        params: &WriteArrayParams<'_>,
        feeder: Feeder<'_, '_>,
    ) -> PyResult<OmWriterVariable> {
        let child_ids = self.child_ids(&params.children)?;
        self.with_state(|state| {
            state
                .ensure_children_exist(&child_ids)
                .map_err(convert_omfilesrs_error)?;

            macro_rules! dispatch {
                ($($variant:ident => $T:ty),+ $(,)?) => {
                    match element_type {
                        $(OmElementType::$variant => {
                            let mut w = state
                                .writer
                                .prepare_array::<$T>(
                                    dimensions,
                                    chunks,
                                    params.compression,
                                    params.scale_factor,
                                    params.add_offset,
                                )
                                .map_err(convert_omfilesrs_error)?;
                            feeder.feed::<$T>(&mut w)?;
                            let finalized = w.finalize();
                            if self.deferred_tail_metadata {
                                let variable_id = state.register_deferred(
                                    params.name,
                                    DeferredVariableKind::Array {
                                        array: Some(finalized),
                                        children: child_ids,
                                    },
                                );
                                Ok(to_writer_variable(params.name, self.writer_id, variable_id))
                            } else {
                                let resolved_children = state
                                    .resolved_children_inline(&child_ids)
                                    .map_err(convert_omfilesrs_error)?;
                                let offset_size = state
                                    .writer
                                    .write_array(finalized, params.name, &resolved_children)
                                    .map_err(convert_omfilesrs_error)?;
                                let variable_id = state.register_resolved(
                                    params.name,
                                    DeferredVariableKind::Resolved,
                                    offset_size,
                                );
                                Ok(to_writer_variable(params.name, self.writer_id, variable_id))
                            }
                        }),+
                    }
                };
            }

            dispatch! {
                Float32 => f32,
                Float64 => f64,
                Int8    => i8 ,
                Uint8   => u8 ,
                Int16   => i16,
                Uint16  => u16,
                Int32   => i32,
                Uint32  => u32,
                Int64   => i64,
                Uint64  => u64
            }
        })
    }

    /// Store a scalar immediately for inline metadata placement, or defer its
    /// metadata emission until close-time for tail metadata placement.
    fn store_scalar<T: OmFileScalarDataType + 'static>(
        &self,
        value: T,
        name: &str,
        children: &[OmWriterVariable],
        deferred_value: DeferredScalarValue,
    ) -> PyResult<OmWriterVariable> {
        let child_ids = self.child_ids(children)?;
        self.with_state(|state| {
            state
                .ensure_children_exist(&child_ids)
                .map_err(convert_omfilesrs_error)?;

            if self.deferred_tail_metadata {
                let variable_id = state.register_deferred(
                    name,
                    DeferredVariableKind::Scalar {
                        value: deferred_value,
                        children: child_ids,
                    },
                );
                Ok(to_writer_variable(name, self.writer_id, variable_id))
            } else {
                let resolved_children = state
                    .resolved_children_inline(&child_ids)
                    .map_err(convert_omfilesrs_error)?;
                let offset_size = state
                    .writer
                    .write_scalar(value, name, &resolved_children)
                    .map_err(convert_omfilesrs_error)?;
                let variable_id =
                    state.register_resolved(name, DeferredVariableKind::Resolved, offset_size);
                Ok(to_writer_variable(name, self.writer_id, variable_id))
            }
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl OmFileWriter {
    /// Initialize an OmFileWriter.
    ///
    /// Args:
    ///     file_path: Path where the .om file will be created
    ///     metadata_placement: (optional) Where to emit metadata; either "inline" to write
    ///                         metadata entries immediately, or "tail" to defer emission
    ///                         until close() so metadata is consolidated near the end of
    ///                         the file (default: "tail").
    #[new]
    #[pyo3(signature = (file_path, metadata_placement=None))]
    fn new(file_path: &str, metadata_placement: Option<&str>) -> PyResult<Self> {
        Self::at_path(file_path, metadata_placement)
    }

    /// Initialize an OmFileWriter to write to a file at the specified path.
    ///
    /// Args:
    ///     path: Path where the .om file will be created
    ///     metadata_placement: (optional) Where to emit metadata; either "inline" or
    ///                         "tail" (see description in `__new__`).
    ///
    /// Returns:
    ///     OmFileWriter: A new writer instance
    #[staticmethod]
    #[pyo3(signature = (path, metadata_placement=None))]
    fn at_path(path: &str, metadata_placement: Option<&str>) -> PyResult<Self> {
        let metadata_placement = Self::validate_metadata_placement(metadata_placement)?;
        let deferred_tail_metadata = metadata_placement == "tail";
        let file_handle = WriterBackendImpl::File(File::create(path)?);
        let writer = OmFileWriterRs::new(file_handle, 8 * 1024);
        Ok(Self {
            writer: Mutex::new(Some(WriterState::new(writer))),
            writer_id: next_writer_id(),
            deferred_tail_metadata,
            explicitly_closed: AtomicBool::new(false),
        })
    }

    /// Create an OmFileWriter from a fsspec filesystem object.
    ///
    /// Args:
    ///     fs_obj: A fsspec filesystem object that supports write operations
    ///     path: The path to the file within the file system
    ///     metadata_placement: (optional) Where to emit metadata; either "inline" or
    ///                         "tail" (see description in `__new__`).
    ///
    /// Returns:
    ///     OmFileWriter: A new writer instance
    #[staticmethod]
    #[pyo3(signature = (fs_obj, path, metadata_placement=None))]
    fn from_fsspec(
        fs_obj: Py<PyAny>,
        path: String,
        metadata_placement: Option<&str>,
    ) -> PyResult<Self> {
        let metadata_placement = Self::validate_metadata_placement(metadata_placement)?;
        let deferred_tail_metadata = metadata_placement == "tail";
        let fsspec_backend = WriterBackendImpl::FsSpec(FsSpecWriterBackend::new(fs_obj, path)?);
        let writer = OmFileWriterRs::new(fsspec_backend, 8 * 1024);
        Ok(Self {
            writer: Mutex::new(Some(WriterState::new(writer))),
            writer_id: next_writer_id(),
            deferred_tail_metadata,
            explicitly_closed: AtomicBool::new(false),
        })
    }

    /// Finalize and close the .om file by writing the trailer with the resolved root variable.
    ///
    /// In ``metadata_placement="tail"`` mode, metadata for arrays, scalars, and
    /// groups is resolved and emitted during ``close()`` so that metadata is
    /// consolidated near the end of the file. In ``metadata_placement="inline"``
    /// mode, metadata is written immediately and child handles must already refer
    /// to resolved variables from the same writer.
    ///
    /// Args:
    ///     root_variable (:py:data:`omfiles.OmWriterVariable`): The writer handle
    ///                    that serves as the root/entry point of the file hierarchy.
    ///
    /// Returns:
    ///     None on success.
    ///
    /// Raises:
    ///     ValueError: If the writer has already been closed or the handle belongs
    ///                 to a different writer.
    ///     RuntimeError: If there is an error resolving deferred metadata or
    ///                   writing the trailer.
    fn close(&mut self, root_variable: OmWriterVariable) -> PyResult<()> {
        self.validate_writer_variable(&root_variable)?;
        let mut guard = self.writer.lock().map_err(Self::lock_error)?;

        let Some(state) = guard.as_mut() else {
            return Err(Self::closed_error());
        };

        let root_offset_size = if self.deferred_tail_metadata {
            let mut resolving = Vec::new();
            state
                .resolve_variable(root_variable.variable_id, &mut resolving)
                .map_err(convert_omfilesrs_error)?
        } else {
            let variable = state
                .variables
                .get(&root_variable.variable_id)
                .ok_or_else(|| PyErr::new::<PyValueError, _>("Unknown root variable handle"))?;
            variable
                .resolved
                .clone()
                .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("Root variable was not resolved"))?
        };

        state
            .writer
            .write_trailer(root_offset_size)
            .map_err(convert_omfilesrs_error)?;
        guard.take();
        self.explicitly_closed.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Check if the writer is closed.
    #[getter]
    fn closed(&self) -> PyResult<bool> {
        let guard = self.writer.lock().map_err(Self::lock_error)?;
        Ok(guard.is_none())
    }

    /// Write a numpy array to the .om file with specified chunking and scaling parameters.
    ///
    /// ``scale_factor`` and ``add_offset`` are only respected and required for float32
    /// and float64 data types. Recommended compression is "pfor_delta_2d" as it achieves
    /// best compression ratios (on spatio-temporally correlated data), but it will be lossy
    /// when applied to floating-point data types because of the scale-offset encoding applied
    /// to convert float values to integer values.
    ///
    /// Args:
    ///     data: Input array to be written. Supported dtypes are:
    ///           float32, float64, int8, uint8, int16, uint16, int32, uint32, int64, uint64,
    ///     chunks: Chunk sizes for each dimension of the array
    ///     scale_factor: Scale factor for data compression (default: 1.0)
    ///     add_offset: Offset value for data compression (default: 0.0)
    ///     compression: Compression algorithm to use (default: "pfor_delta_2d")
    ///                  Supported values: "pfor_delta_2d", "fpx_xor_2d", "pfor_delta_2d_int16", "pfor_delta_2d_int16_logarithmic"
    ///     name: Name of the variable to be written (default: "data")
    ///     children: List of child variables (default: [])
    ///
    /// ``write_array`` returns an :py:data:`omfiles.OmWriterVariable`, which is a
    /// write-time handle used to build hierarchy relationships and to select the
    /// root variable passed to ``close()``. It is not the same as
    /// :py:data:`omfiles.OmVariable`, which represents already-materialized
    /// metadata when reading.
    ///
    /// Returns:
    ///     :py:data:`omfiles.OmWriterVariable` representing the written group in the file structure
    ///
    /// Raises:
    ///     ValueError: If the data type is unsupported or if parameters are invalid
    #[pyo3(
        text_signature = "(data, chunks, scale_factor=1.0, add_offset=0.0, compression='pfor_delta_2d', name='data', children=[])",
        signature = (data, chunks, scale_factor=None, add_offset=None, compression=None, name=None, children=None)
    )]
    fn write_array(
        &mut self,
        data: &Bound<'_, PyUntypedArray>,
        chunks: Vec<u64>,
        scale_factor: Option<f32>,
        add_offset: Option<f32>,
        compression: Option<&str>,
        name: Option<&str>,
        children: Option<Vec<OmWriterVariable>>,
    ) -> PyResult<OmWriterVariable> {
        let params =
            WriteArrayParams::from_options(name, children, scale_factor, add_offset, compression)?;
        let element_type = OmElementType::from_numpy_dtype(&data.dtype())?;
        let dimensions = data.shape().iter().map(|x| *x as u64).collect();
        let feeder = Feeder::Full { data };

        self.write_array_dispatched(element_type, dimensions, chunks, &params, feeder)
    }

    /// Write an array to the .om file by streaming chunks from a Python iterator.
    ///
    /// This method is designed for writing large arrays that do not fit in memory.
    /// Instead of providing the full array, you provide the full array dimensions
    /// and an iterator that yields numpy array chunks.
    ///
    /// Chunks MUST be yielded in row-major order (C-order) of the chunk grid.
    /// Each chunk's shape determines how many internal file chunks it covers.
    ///
    /// Args:
    ///     dimensions: Shape of the full array (e.g., [1000, 2000])
    ///     chunks: Chunk sizes for each dimension (e.g., [100, 200])
    ///     chunk_iterator: Python iterable yielding numpy arrays, one per chunk region
    ///     dtype: Numpy dtype of the array (e.g., np.dtype(np.float32))
    ///     scale_factor: Scale factor for data compression (default: 1.0)
    ///     add_offset: Offset value for data compression (default: 0.0)
    ///     compression: Compression algorithm to use (default: "pfor_delta_2d")
    ///     name: Name of the variable (default: "data")
    ///     children: List of child variables (default: [])
    ///
    /// Returns:
    ///     :py:data:`omfiles.OmWriterVariable` representing the written array in the file structure
    ///
    /// Raises:
    ///     ValueError: If the dtype is unsupported or parameters are invalid
    ///     RuntimeError: If there's an error during compression or I/O
    #[pyo3(
        text_signature = "(dimensions, chunks, chunk_iterator, dtype, scale_factor=1.0, add_offset=0.0, compression='pfor_delta_2d', name='data', children=[])",
        signature = (dimensions, chunks, chunk_iterator, dtype, scale_factor=None, add_offset=None, compression=None, name=None, children=None)
    )]
    fn write_array_streaming<'py>(
        &mut self,
        py: Python<'_>,
        dimensions: Vec<u64>,
        chunks: Vec<u64>,
        #[gen_stub(override_type(type_repr="typing.Iterator", imports=("typing")))]
        chunk_iterator: &Bound<'_, PyIterator>,
        dtype: &Bound<'py, PyArrayDescr>,
        scale_factor: Option<f32>,
        add_offset: Option<f32>,
        compression: Option<&str>,
        name: Option<&str>,
        children: Option<Vec<OmWriterVariable>>,
    ) -> PyResult<OmWriterVariable> {
        let params =
            WriteArrayParams::from_options(name, children, scale_factor, add_offset, compression)?;
        let element_type = OmElementType::from_numpy_dtype(dtype)?;
        let iter = chunk_iterator.call_method0("__iter__")?;
        let feeder = Feeder::Streaming { py, iter };

        self.write_array_dispatched(element_type, dimensions, chunks, &params, feeder)
    }

    /// Write a scalar value to the .om file.
    ///
    /// Args:
    ///     value: Scalar value to write. Supported types are:
    ///            int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, String
    ///     name: Name of the scalar variable
    ///     children: List of child variables (default: None)
    ///
    /// Child handles must come from the same writer. In ``metadata_placement="inline"``
    /// mode they must already be resolved because metadata is emitted immediately.
    /// In ``metadata_placement="tail"`` mode they may be resolved later during
    /// ``close()``.
    ///
    /// Returns:
    ///     :py:data:`omfiles.OmWriterVariable` representing the written group in the file structure
    ///
    /// Raises:
    ///     ValueError: If the value type is unsupported (e.g., booleans)
    ///     RuntimeError: If there's an error writing to the file
    #[pyo3(
        text_signature = "(value, name, children=None)",
        signature = (value, name, children=None)
    )]
    fn write_scalar(
        &mut self,
        value: &Bound<PyAny>,
        name: &str,
        children: Option<Vec<OmWriterVariable>>,
    ) -> PyResult<OmWriterVariable> {
        let children = children.unwrap_or_default();
        let py = value.py();

        macro_rules! check_numpy_type {
            ($numpy:expr, $type_name:literal, $rust_type:ty, $variant:ident) => {
                if let Ok(numpy_type) = $numpy.getattr($type_name) {
                    if value.is_instance(&numpy_type)? {
                        let scalar_value: $rust_type = value.call_method0("item")?.extract()?;
                        return self.store_scalar(
                            scalar_value,
                            name,
                            &children,
                            DeferredScalarValue::$variant(scalar_value),
                        );
                    }
                }
            };
        }

        // Try to import numpy and check for numpy scalar types
        if let Ok(numpy) = py.import("numpy") {
            check_numpy_type!(numpy, "int8", i8, Int8);
            check_numpy_type!(numpy, "uint8", u8, Uint8);
            check_numpy_type!(numpy, "int16", i16, Int16);
            check_numpy_type!(numpy, "uint16", u16, Uint16);
            check_numpy_type!(numpy, "int32", i32, Int32);
            check_numpy_type!(numpy, "uint32", u32, Uint32);
            check_numpy_type!(numpy, "int64", i64, Int64);
            check_numpy_type!(numpy, "uint64", u64, Uint64);
            check_numpy_type!(numpy, "float32", f32, Float32);
            check_numpy_type!(numpy, "float64", f64, Float64);
        }

        if let Ok(value) = value.extract::<String>() {
            self.store_scalar(
                value.clone(),
                name,
                &children,
                DeferredScalarValue::String(value),
            )
        } else if let Ok(value) = value.extract::<f64>() {
            self.store_scalar(value, name, &children, DeferredScalarValue::Float64(value))
        } else if let Ok(value) = value.extract::<f32>() {
            self.store_scalar(value, name, &children, DeferredScalarValue::Float32(value))
        } else if let Ok(value) = value.extract::<i64>() {
            self.store_scalar(value, name, &children, DeferredScalarValue::Int64(value))
        } else if let Ok(value) = value.extract::<i32>() {
            self.store_scalar(value, name, &children, DeferredScalarValue::Int32(value))
        } else if let Ok(value) = value.extract::<i16>() {
            self.store_scalar(value, name, &children, DeferredScalarValue::Int16(value))
        } else if let Ok(value) = value.extract::<i8>() {
            self.store_scalar(value, name, &children, DeferredScalarValue::Int8(value))
        } else if let Ok(value) = value.extract::<u64>() {
            self.store_scalar(value, name, &children, DeferredScalarValue::Uint64(value))
        } else if let Ok(value) = value.extract::<u32>() {
            self.store_scalar(value, name, &children, DeferredScalarValue::Uint32(value))
        } else if let Ok(value) = value.extract::<u16>() {
            self.store_scalar(value, name, &children, DeferredScalarValue::Uint16(value))
        } else if let Ok(value) = value.extract::<u8>() {
            self.store_scalar(value, name, &children, DeferredScalarValue::Uint8(value))
        } else {
            Err(Self::unsupported_scalar_type_error(value.get_type()))
        }
    }

    /// Create a new group in the .om file.
    ///
    /// This is essentially a variable with no data, which serves as a container
    /// for other variables.
    ///
    /// Args:
    ///     name: Name of the group
    ///     children: List of child variables from the same writer
    ///
    /// Returns:
    ///     :py:data:`omfiles.OmWriterVariable` representing the written group in the file structure
    ///
    /// Raises:
    ///     ValueError: If a child handle belongs to a different writer
    ///     RuntimeError: If inline metadata placement is requested before child
    ///                   metadata has been resolved, or if there is an I/O error
    fn write_group(
        &mut self,
        name: &str,
        children: Vec<OmWriterVariable>,
    ) -> PyResult<OmWriterVariable> {
        let child_ids = self.child_ids(&children)?;
        self.with_state(|state| {
            state
                .ensure_children_exist(&child_ids)
                .map_err(convert_omfilesrs_error)?;

            if self.deferred_tail_metadata {
                let variable_id = state.register_deferred(
                    name,
                    DeferredVariableKind::Group {
                        children: child_ids,
                    },
                );
                Ok(to_writer_variable(name, self.writer_id, variable_id))
            } else {
                let resolved_children = state
                    .resolved_children_inline(&child_ids)
                    .map_err(convert_omfilesrs_error)?;
                let offset_size = state
                    .writer
                    .write_none(name, &resolved_children)
                    .map_err(convert_omfilesrs_error)?;
                let variable_id =
                    state.register_resolved(name, DeferredVariableKind::Resolved, offset_size);
                Ok(to_writer_variable(name, self.writer_id, variable_id))
            }
        })
    }
}

impl Drop for OmFileWriter {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.writer.lock() {
            if guard.is_some() && !self.explicitly_closed.load(Ordering::Relaxed) {
                eprintln!(
                    "Warning: OmFileWriter was dropped without calling close(); the OM file may be incomplete"
                );
            }
            guard.take();
        }
    }
}

/// Concrete wrapper type for the backend implementation, delegating to the appropriate backend.
enum WriterBackendImpl {
    File(File),
    FsSpec(FsSpecWriterBackend),
}

impl OmFileWriterBackend for WriterBackendImpl {
    delegate! {
        to match self {
            WriterBackendImpl::File(backend) => backend,
            WriterBackendImpl::FsSpec(backend) => backend,
        } {
            fn write(&mut self, data: &[u8]) -> Result<(), OmFilesError>;
            fn synchronize(&self) -> Result<(), OmFilesError>;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::reader::OmFileReader;

    use super::*;
    use numpy::{ndarray::ArrayD, PyArrayDyn, PyArrayMethods};
    use std::fs;

    #[test]
    fn test_write_array() -> Result<(), Box<dyn std::error::Error>> {
        Python::initialize();

        Python::attach(|py| -> Result<(), Box<dyn std::error::Error>> {
            // numpy is not happy if we import it when modifying the PYTHONPATH to directly include numpy
            // because of broken handling of virtual environments in pyo3, we skip the test on import failure
            if let Err(e) = py.import("numpy") {
                eprintln!(
                    "Skipping test_write_array: could not import numpy ({:?})",
                    e
                );
                return Ok(()); // Skip the test
            }

            // Test parameters
            let file_path = "test_write_array.om";
            let dimensions = vec![10, 20];
            let chunks = vec![5u64, 5];

            // Create test data
            let data = ArrayD::from_shape_fn(dimensions, |idx| (idx[0] + idx[1]) as f32);
            let py_array = PyArrayDyn::from_array(py, &data);

            let mut file_writer = OmFileWriter::new(file_path, None).unwrap();

            // Write data
            let result = file_writer.write_array(
                py_array.as_untyped(),
                chunks,
                None,
                None,
                None,
                None,
                None,
            );

            assert!(result.is_ok());

            let root = result.unwrap();
            file_writer.close(root)?;

            let reader = OmFileReader::from_path(file_path);
            assert!(reader.is_ok());

            // Clean up
            fs::remove_file(file_path).unwrap();
            Ok(())
        })?;

        Ok(())
    }

    #[test]
    fn test_fsspec_writer() -> Result<(), Box<dyn std::error::Error>> {
        Python::initialize();

        Python::attach(|py| -> Result<(), Box<dyn std::error::Error>> {
            let fsspec = py.import("fsspec")?;
            let fs = fsspec.call_method1("filesystem", ("memory",))?;
            let fs_py_any: Py<PyAny> = fs.into();

            let file_path = "test_fsspec_writer.om";

            let mut writer =
                OmFileWriter::from_fsspec(fs_py_any.clone_ref(py), file_path.to_string(), None)?;
            let value = 0i32.into_pyobject(py)?;
            let root = writer.write_scalar(&value, "zero_root", None)?;
            writer.close(root)?;

            let reader = OmFileReader::from_fsspec(fs_py_any, file_path.to_string());
            assert!(reader.is_ok());

            Ok(())
        })?;

        Ok(())
    }

    #[test]
    fn test_resolve_variable_detects_cycle() {
        let file_path = "test_cycle_detection.om";
        let file_handle = WriterBackendImpl::File(File::create(file_path).unwrap());
        let writer = OmFileWriterRs::new(file_handle, 8 * 1024);
        let mut state = WriterState::new(writer);

        let child_id = state.register_deferred(
            "child",
            DeferredVariableKind::Group {
                children: Vec::new(),
            },
        );
        let root_id = state.register_deferred(
            "root",
            DeferredVariableKind::Group {
                children: vec![child_id],
            },
        );

        if let Some(variable) = state.variables.get_mut(&child_id) {
            variable.kind = DeferredVariableKind::Group {
                children: vec![root_id],
            };
        }

        let err = state
            .resolve_variable(root_id, &mut Vec::new())
            .unwrap_err();
        assert!(
            matches!(err, OmFilesError::GenericError(message) if message == "Cycle detected in deferred variable hierarchy")
        );

        let _ = fs::remove_file(file_path);
    }
}
