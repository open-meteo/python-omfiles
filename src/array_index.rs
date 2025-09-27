use pyo3::prelude::*;
use std::ops::Range;

/// Type annotation support via pyo3_stub_gen
impl pyo3_stub_gen::PyStubType for ArrayIndex {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        let mut import = std::collections::HashSet::new();
        import.insert("omfiles".into());
        pyo3_stub_gen::TypeInfo {
            name: "omfiles.types.BasicSelection".into(),
            import,
        }
    }
}

/// A simplified numpy-like array basic indexing implementation.
/// Compare https://numpy.org/doc/stable/user/basics.indexing.html.
/// Supports integer, slice, newaxis and ellipsis indexing.
/// Slice indexing is also currently limited to step size 1!
#[derive(Debug)]
pub enum IndexType {
    Int(i64),
    Slice {
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    },
    NewAxis,
    Ellipsis,
}

#[derive(Debug)]
pub struct ArrayIndex(pub Vec<IndexType>);

impl<'py> FromPyObject<'py> for ArrayIndex {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        fn parse_index(item: &Bound<'_, PyAny>) -> PyResult<IndexType> {
            if item.is_instance_of::<pyo3::types::PySlice>() {
                let slice = item.downcast::<pyo3::types::PySlice>()?;
                let start = slice.getattr("start")?.extract()?;
                let stop = slice.getattr("stop")?.extract()?;
                let step = slice.getattr("step")?.extract()?;
                Ok(IndexType::Slice { start, stop, step })
            } else if item.is_instance_of::<pyo3::types::PyEllipsis>() {
                Ok(IndexType::Ellipsis)
            } else if item.is_none() {
                Ok(IndexType::NewAxis)
            } else {
                Ok(IndexType::Int(item.extract()?))
            }
        }

        if let Ok(tuple) = ob.downcast::<pyo3::types::PyTuple>() {
            let indices = tuple
                .iter()
                .map(|idx| parse_index(&idx))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(ArrayIndex(indices))
        } else {
            Ok(ArrayIndex(vec![parse_index(ob)?]))
        }
    }
}

impl ArrayIndex {
    pub fn to_read_range(&self, shape: &Vec<u64>) -> PyResult<Vec<Range<u64>>> {
        // Input validation
        if self.0.len() > shape.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Too many indices for array",
            ));
        }

        let mut ranges = Vec::new();
        let mut shape_idx = 0;
        let mut ellipsis_seen = false;
        let explicit_dims: usize = self
            .0
            .iter()
            .filter(|&x| !matches!(x, IndexType::Ellipsis))
            .count();
        let ellipsis_dims = shape.len().saturating_sub(explicit_dims);

        for (idx, &dim_size) in self.0.iter().zip(shape.iter()) {
            match idx {
                IndexType::Ellipsis => {
                    if ellipsis_seen {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Only one ellipsis allowed in index",
                        ));
                    }
                    // Add full ranges for all dimensions represented by the ellipsis
                    for _ in 0..ellipsis_dims {
                        ranges.push(Range {
                            start: 0,
                            end: shape[shape_idx],
                        });
                        shape_idx += 1;
                    }
                    ellipsis_seen = true;
                }
                IndexType::Int(i) => {
                    let normalized_idx = Self::normalize_index(*i, dim_size)?;
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

                    let start_idx = match start {
                        Some(s) => {
                            let normalized = Self::normalize_index(*s, dim_size)?;
                            if normalized > dim_size {
                                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                                    format!(
                                        "Index {} is out of bounds for axis with size {}",
                                        s, dim_size
                                    ),
                                ));
                            }
                            normalized
                        }
                        None => 0,
                    };

                    let stop_idx = match stop {
                        Some(s) => {
                            let normalized = Self::normalize_index(*s, dim_size)?;
                            if normalized > dim_size {
                                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                                    format!(
                                        "Index {} is out of bounds for axis with size {}",
                                        s, dim_size
                                    ),
                                ));
                            }
                            normalized
                        }
                        None => dim_size,
                    };

                    if stop_idx <= start_idx {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "omfiles currently do not support reversed ranges.",
                        ));
                    }

                    ranges.push(Range {
                        start: start_idx,
                        end: stop_idx,
                    });
                    shape_idx += 1;
                }
                IndexType::NewAxis => {
                    ranges.push(Range {
                        start: 0,
                        end: dim_size,
                    });
                }
            }
        }

        // Handle remaining dimensions if any
        while shape_idx < shape.len() {
            ranges.push(Range {
                start: 0,
                end: shape[shape_idx],
            });
            shape_idx += 1;
        }

        Ok(ranges)
    }

    fn normalize_index(idx: i64, dim_size: u64) -> PyResult<u64> {
        let dim_size_i64 = dim_size as i64;
        let normalized = if idx < 0 { idx + dim_size_i64 } else { idx };

        if normalized < 0 || normalized > dim_size_i64 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} is out of bounds for axis with size {}",
                idx, dim_size
            )));
        }

        Ok(normalized as u64)
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
            let slice_index = ArrayIndex::extract_bound(&single_slice_tuple).unwrap();
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
            let int_index = ArrayIndex::extract_bound(&single_int_tuple).unwrap();
            match int_index.0[0] {
                IndexType::Int(val) => assert_eq!(val, 42),
                _ => panic!("Expected Int type"),
            }

            // Test None (NewAxis)
            let none_value = py.None();
            let single_none_tuple = pyo3::types::PyTuple::new(py, [&none_value]).unwrap();
            let none_index = ArrayIndex::extract_bound(&single_none_tuple).unwrap();
            match none_index.0[0] {
                IndexType::NewAxis => (),
                _ => panic!("Expected NewAxis type"),
            }

            // Test combination of different types
            let mixed_tuple = pyo3::types::PyTuple::new(
                py,
                [
                    &slice.into_any(),                            // slice
                    &42i64.into_pyobject(py).unwrap().into_any(), // integer
                    py.None().bind(py),                           // NewAxis
                ],
            )
            .unwrap();
            let mixed_index = ArrayIndex::extract_bound(&mixed_tuple).unwrap();

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

            match mixed_index.0[2] {
                IndexType::NewAxis => (),
                _ => panic!("Expected NewAxis type"),
            }

            // Test slice with None values (open-ended slices)
            let open_slice = PySlice::full(py);
            let open_slice_tuple = pyo3::types::PyTuple::new(py, [&open_slice]).unwrap();
            let open_slice_index = ArrayIndex::extract_bound(&open_slice_tuple).unwrap();
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
            let index = ArrayIndex::extract_bound(&neg_tuple).unwrap();
            let ranges = index
                .to_read_range(&shape)
                .expect("Could not convert to read_range!");
            assert_eq!(ranges[0].start, 3); // -2 should map to index 3 in size 5

            // Test negative slice indices
            let slice = PySlice::new(py, -3, -1, 1);
            let slice_tuple = pyo3::types::PyTuple::new(py, [&slice]).unwrap();
            let index = ArrayIndex::extract_bound(&slice_tuple).unwrap();
            let ranges = index
                .to_read_range(&shape)
                .expect("Could not convert to read_range!");
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
            let index = ArrayIndex::extract_bound(&tuple).unwrap();
            let ranges = index.to_read_range(&shape).unwrap();
            assert_eq!(ranges.len(), 4);
            assert_eq!(ranges[0], Range { start: 0, end: 2 });
            assert_eq!(ranges[1], Range { start: 0, end: 3 });
            assert_eq!(ranges[2], Range { start: 0, end: 4 });
            assert_eq!(ranges[3], Range { start: 1, end: 2 });

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
            let index = ArrayIndex::extract_bound(&tuple).unwrap();
            let ranges = index.to_read_range(&shape).unwrap();

            assert_eq!(ranges.len(), 4);
            assert_eq!(ranges[0], Range { start: 1, end: 2 });
            assert_eq!(ranges[1], Range { start: 0, end: 3 });
            assert_eq!(ranges[2], Range { start: 0, end: 4 });
            assert_eq!(ranges[3], Range { start: 2, end: 3 });
        });
    }

    #[test]
    #[should_panic]
    fn test_invalid_input() {
        Python::initialize();

        Python::attach(|py| {
            let invalid_value = "not_an_index"
                .into_pyobject(py)
                .expect("Failed to create object");
            let invalid_tuple =
                pyo3::types::PyTuple::new(py, [&invalid_value]).expect("Failed to create tuple");
            let _should_fail = ArrayIndex::extract_bound(&invalid_tuple).unwrap();
        });
    }

    #[test]
    #[should_panic]
    fn test_invalid_negative_index() {
        Python::initialize();

        Python::attach(|py| {
            let shape = vec![5];
            let neg_idx = (-6i64).into_pyobject(py).unwrap();
            let neg_tuple = pyo3::types::PyTuple::new(py, [&neg_idx]).unwrap();
            let index = ArrayIndex::extract_bound(&neg_tuple).unwrap();
            let _should_fail = index.to_read_range(&shape).unwrap();
        });
    }

    #[test]
    #[should_panic]
    fn test_invalid_slice() {
        Python::initialize();

        Python::attach(|py| {
            let shape = vec![5];
            let neg_idx = (-6i64).into_pyobject(py).unwrap();
            let neg_tuple = pyo3::types::PyTuple::new(py, [&neg_idx]).unwrap();
            let index = ArrayIndex::extract_bound(&neg_tuple).unwrap();
            let _should_fail = index.to_read_range(&shape).unwrap();
        });
    }
}
