use pyo3::prelude::*;
use std::{collections::HashMap, ops::Range};

/// Type annotation support via pyo3_stub_gen
impl pyo3_stub_gen::PyStubType for ArrayIndex {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        let mut import = std::collections::HashSet::new();
        import.insert("omfiles".into());
        import.insert("omfiles.types".into());
        pyo3_stub_gen::TypeInfo {
            name: "types.BasicSelection".into(),
            source_module: None,
            import,
            type_refs: HashMap::new(),
        }
    }
}

/// A simplified numpy-like array basic indexing implementation.
/// Compare https://numpy.org/doc/stable/user/basics.indexing.html.
/// Supports integer, slice and ellipsis indexing.
/// Slice indexing is also currently limited to step size 1!
#[derive(Debug)]
pub enum IndexType {
    Int(i64),
    Slice {
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    },
    Ellipsis,
}

#[derive(Debug)]
pub struct ArrayIndex(pub Vec<IndexType>);

impl<'py> FromPyObject<'_, 'py> for ArrayIndex {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        fn parse_index(item: &Bound<'_, PyAny>) -> PyResult<IndexType> {
            if item.is_instance_of::<pyo3::types::PySlice>() {
                let slice = item.cast::<pyo3::types::PySlice>()?;
                let start = slice.getattr("start")?.extract()?;
                let stop = slice.getattr("stop")?.extract()?;
                let step = slice.getattr("step")?.extract()?;
                Ok(IndexType::Slice { start, stop, step })
            } else if item.is_instance_of::<pyo3::types::PyEllipsis>() {
                Ok(IndexType::Ellipsis)
            } else {
                match item.extract() {
                    Ok(index) => Ok(IndexType::Int(index)),
                    Err(_) => {
                        let item_type = item.get_type().repr()?.extract::<String>()?;
                        Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                            "unsupported selection item for basic indexing; expected integer or slice, got {item_type}"
                        )))
                    }
                }
            }
        }

        if let Ok(tuple) = ob.cast::<pyo3::types::PyTuple>() {
            let indices = tuple
                .iter()
                .map(|idx| parse_index(&idx))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(ArrayIndex(indices))
        } else {
            Ok(ArrayIndex(vec![parse_index(&ob)?]))
        }
    }
}

impl ArrayIndex {
    pub fn get_ranges_and_output_shape(
        &self,
        shape: &[u64],
    ) -> PyResult<(Vec<Range<u64>>, Vec<usize>)> {
        // Each explicit index (integer or slice) applies to one input dimension.
        let explicit_dims: usize = self
            .0
            .iter()
            .filter(|&x| !matches!(x, IndexType::Ellipsis))
            .count();
        if explicit_dims > shape.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Too many indices for array",
            ));
        }

        let mut ranges = Vec::new();
        let mut output_shape = Vec::new();

        let mut shape_idx = 0;
        let mut ellipsis_seen = false;
        let ellipsis_dims = shape.len().saturating_sub(explicit_dims);

        for idx in &self.0 {
            match idx {
                IndexType::Ellipsis => {
                    if ellipsis_seen {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Only one ellipsis allowed in index",
                        ));
                    }
                    for _ in 0..ellipsis_dims {
                        ranges.push(Range {
                            start: 0,
                            end: shape[shape_idx],
                        });
                        output_shape.push(shape[shape_idx] as usize);
                        shape_idx += 1;
                    }
                    ellipsis_seen = true;
                }
                IndexType::Int(i) => {
                    let normalized_idx = Self::normalize_index(*i, shape[shape_idx])?;
                    ranges.push(Range {
                        start: normalized_idx,
                        end: normalized_idx + 1,
                    });

                    shape_idx += 1;
                }
                IndexType::Slice { start, stop, step } => {
                    if let Some(step) = step {
                        if *step != 1 {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Slice step must be 1",
                            ));
                        }
                    }
                    let dim_size = shape[shape_idx];

                    let start_idx = start
                        .map(|index| Self::normalize_slice_bound(index, dim_size))
                        .unwrap_or(0);
                    let stop_idx = stop
                        .map(|index| Self::normalize_slice_bound(index, dim_size))
                        .unwrap_or(dim_size)
                        .max(start_idx);

                    ranges.push(Range {
                        start: start_idx,
                        end: stop_idx,
                    });
                    output_shape.push((stop_idx - start_idx) as usize);

                    shape_idx += 1;
                }
            }
        }

        // Handle remaining dimensions if any
        while shape_idx < shape.len() {
            ranges.push(Range {
                start: 0,
                end: shape[shape_idx],
            });
            output_shape.push(shape[shape_idx] as usize);
            shape_idx += 1;
        }

        Ok((ranges, output_shape))
    }

    fn normalize_index(idx: i64, dim_size: u64) -> PyResult<u64> {
        let normalized = if idx < 0 {
            dim_size.checked_sub(idx.unsigned_abs())
        } else {
            let idx = idx as u64;
            (idx < dim_size).then_some(idx)
        };

        normalized.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} is out of bounds for axis with size {}",
                idx, dim_size
            ))
        })
    }

    fn normalize_slice_bound(idx: i64, dim_size: u64) -> u64 {
        if idx < 0 {
            dim_size.saturating_sub(idx.unsigned_abs())
        } else {
            (idx as u64).min(dim_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::{types::PySlice, BoundObject};

    #[test]
    fn test_numpy_indexing() {
        Python::initialize();

        Python::attach(|py| {
            // Test basic slicing
            let slice = PySlice::new(py, 1, 5, 1);
            let single_slice_tuple = pyo3::types::PyTuple::new(py, [&slice]).unwrap();
            let slice_index =
                ArrayIndex::extract(single_slice_tuple.as_any().as_borrowed()).unwrap();
            match &slice_index.0[0] {
                IndexType::Slice { start, stop, step } => {
                    assert_eq!(*start, Some(1));
                    assert_eq!(*stop, Some(5));
                    assert_eq!(*step, Some(1));
                }
                _ => panic!("Expected Slice type"),
            }

            // Test integer indexing
            let int_value = 42i64.into_pyobject(py).unwrap();
            let single_int_tuple = pyo3::types::PyTuple::new(py, [&int_value]).unwrap();
            let int_index = ArrayIndex::extract(single_int_tuple.as_any().as_borrowed()).unwrap();
            match int_index.0[0] {
                IndexType::Int(val) => assert_eq!(val, 42),
                _ => panic!("Expected Int type"),
            }

            // Test combination of different types
            let mixed_tuple = pyo3::types::PyTuple::new(
                py,
                [
                    &slice.into_any(),                            // slice
                    &42i64.into_pyobject(py).unwrap().into_any(), // integer
                ],
            )
            .unwrap();
            let mixed_index = ArrayIndex::extract(mixed_tuple.as_any().as_borrowed()).unwrap();

            // Verify the types in order
            match &mixed_index.0[0] {
                IndexType::Slice { start, stop, step } => {
                    assert_eq!(*start, Some(1));
                    assert_eq!(*stop, Some(5));
                    assert_eq!(*step, Some(1));
                }
                _ => panic!("Expected Slice type"),
            }

            match mixed_index.0[1] {
                IndexType::Int(val) => assert_eq!(val, 42),
                _ => panic!("Expected Int type"),
            }

            // Test slice with None values (open-ended slices)
            let open_slice = PySlice::full(py);
            let open_slice_tuple = pyo3::types::PyTuple::new(py, [&open_slice]).unwrap();
            let open_slice_index =
                ArrayIndex::extract(open_slice_tuple.as_any().as_borrowed()).unwrap();
            match &open_slice_index.0[0] {
                IndexType::Slice { start, stop, step } => {
                    assert_eq!(*start, None);
                    assert_eq!(*stop, None);
                    assert_eq!(*step, None);
                }
                _ => panic!("Expected Slice type"),
            }
        });
    }

    #[test]
    fn test_negative_indexing() {
        Python::initialize();

        Python::attach(|py| {
            let shape = vec![5];

            // Test negative integer index
            let neg_idx = (-2i64).into_pyobject(py).unwrap();
            let neg_tuple = pyo3::types::PyTuple::new(py, [&neg_idx]).unwrap();
            let index = ArrayIndex::extract(neg_tuple.as_any().as_borrowed()).unwrap();
            let ranges = index
                .get_ranges_and_output_shape(&shape)
                .expect("Could not convert to read_range!")
                .0;
            assert_eq!(ranges[0].start, 3); // -2 should map to index 3 in size 5

            // Test negative slice indices
            let slice = PySlice::new(py, -3, -1, 1);
            let slice_tuple = pyo3::types::PyTuple::new(py, [&slice]).unwrap();
            let index = ArrayIndex::extract(slice_tuple.as_any().as_borrowed()).unwrap();
            let ranges = index
                .get_ranges_and_output_shape(&shape)
                .expect("Could not convert to read_range!")
                .0;
            assert_eq!(ranges[0].start, 2); // -3 should map to index 2
            assert_eq!(ranges[0].end, 4); // -1 should map to index 4
        });
    }

    #[test]
    fn test_ellipsis() {
        Python::initialize();

        Python::attach(|py| {
            let shape = vec![2, 3, 4, 5];
            let ellipsis = pyo3::types::PyEllipsis::get(py).into_any();
            let integer = 1i64.into_pyobject(py).unwrap().into_any();

            // Test ..., 1
            let tuple = pyo3::types::PyTuple::new(py, [&ellipsis, &integer]).unwrap();
            let index = ArrayIndex::extract(tuple.as_any().as_borrowed()).unwrap();
            let (ranges, output_shape) = index.get_ranges_and_output_shape(&shape).unwrap();
            assert_eq!(ranges.len(), 4);
            assert_eq!(ranges[0], Range { start: 0, end: 2 });
            assert_eq!(ranges[1], Range { start: 0, end: 3 });
            assert_eq!(ranges[2], Range { start: 0, end: 4 });
            assert_eq!(ranges[3], Range { start: 1, end: 2 });
            assert_eq!(output_shape, vec![2, 3, 4]);

            // Test 1, ..., 2
            let tuple = pyo3::types::PyTuple::new(
                py,
                [
                    &1i64.into_pyobject(py).unwrap().into_any(),
                    &ellipsis,
                    &2i64.into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let index = ArrayIndex::extract(tuple.as_any().as_borrowed()).unwrap();
            let (ranges, output_shape) = index.get_ranges_and_output_shape(&shape).unwrap();

            assert_eq!(ranges.len(), 4);
            assert_eq!(ranges[0], Range { start: 1, end: 2 });
            assert_eq!(ranges[1], Range { start: 0, end: 3 });
            assert_eq!(ranges[2], Range { start: 0, end: 4 });
            assert_eq!(ranges[3], Range { start: 2, end: 3 });
            assert_eq!(output_shape, vec![3, 4]);
        });
    }

    #[test]
    fn test_unsupported_selection() {
        Python::initialize();

        Python::attach(|py| {
            let none = py.None();
            let none_value = none.bind(py);
            let tuple = pyo3::types::PyTuple::new(py, [none_value]).unwrap();
            let error = ArrayIndex::extract(tuple.as_any().as_borrowed()).unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyIndexError>(py));
            assert_eq!(
                error.to_string(),
                "IndexError: unsupported selection item for basic indexing; expected integer or slice, got <class 'NoneType'>"
            );
        });
    }

    #[test]
    fn test_integer_bounds() {
        let shape = vec![5];

        let first = ArrayIndex(vec![IndexType::Int(-5)])
            .get_ranges_and_output_shape(&shape)
            .unwrap();
        assert_eq!(first.0, vec![Range { start: 0, end: 1 }]);

        assert!(ArrayIndex(vec![IndexType::Int(5)])
            .get_ranges_and_output_shape(&shape)
            .is_err());
        assert!(ArrayIndex(vec![IndexType::Int(-6)])
            .get_ranges_and_output_shape(&shape)
            .is_err());
    }

    #[test]
    fn test_slice_clipping_and_empty_ranges() {
        let shape = vec![5];
        let cases = [
            ((Some(-99), None), Range { start: 0, end: 5 }),
            ((None, Some(99)), Range { start: 0, end: 5 }),
            ((Some(99), None), Range { start: 5, end: 5 }),
            ((None, Some(-99)), Range { start: 0, end: 0 }),
            ((Some(3), Some(1)), Range { start: 3, end: 3 }),
        ];

        for ((start, stop), expected) in cases {
            let resolved = ArrayIndex(vec![IndexType::Slice {
                start,
                stop,
                step: None,
            }])
            .get_ranges_and_output_shape(&shape)
            .unwrap();
            assert_eq!(resolved.0, vec![expected]);
        }

        assert!(ArrayIndex(vec![IndexType::Slice {
            start: None,
            stop: None,
            step: Some(2),
        }])
        .get_ranges_and_output_shape(&shape)
        .is_err());
    }
}
