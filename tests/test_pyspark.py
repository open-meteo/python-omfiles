"""Tests for the PySpark custom data source integration.

These tests verify the core logic of the OmFileDataSource without requiring
a running Spark cluster.  They mock the PySpark base classes so that the
module can be imported and the read-path logic exercised locally.
"""

import os
import tempfile

import numpy as np
import pytest
from omfiles import OmFileWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_flat_om(path: str, data: np.ndarray) -> None:
    """Write *data* as a flat (non-hierarchical) .om file."""
    writer = OmFileWriter(path)
    var = writer.write_array(data, chunks=[min(5, s) for s in data.shape], scale_factor=1.0, add_offset=0.0)
    writer.close(var)


def _create_hierarchical_om(path: str) -> dict[str, np.ndarray]:
    """Write a hierarchical .om file with two variables and a crs_wkt scalar."""
    temp = np.arange(25, dtype=np.float32).reshape(5, 5)
    wind = (np.arange(25, dtype=np.float32).reshape(5, 5) * 0.5)

    writer = OmFileWriter(path)
    v_temp = writer.write_array(temp, chunks=[5, 5], scale_factor=1.0, add_offset=0.0, name="temperature_2m")
    v_wind = writer.write_array(wind, chunks=[5, 5], scale_factor=1.0, add_offset=0.0, name="wind_speed_10m")
    root = writer.write_group("root", children=[v_temp, v_wind])
    writer.close(root)
    return {"temperature_2m": temp, "wind_speed_10m": wind}


# ---------------------------------------------------------------------------
# Skip if pyspark is not installed
# ---------------------------------------------------------------------------

pyspark = pytest.importorskip("pyspark", reason="PySpark not installed — skipping PySpark datasource tests")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOmFileDataSource:
    """Unit tests for OmFileDataSource schema inference and reader logic."""

    def test_schema_inference_flat_array(self, tmp_path):
        """Schema for a flat array should have dim columns + value."""
        from omfiles.pyspark import OmFileDataSource

        path = str(tmp_path / "flat.om")
        data = np.arange(20, dtype=np.float32).reshape(4, 5)
        _create_flat_om(path, data)

        ds = OmFileDataSource.__new__(OmFileDataSource)
        ds.options = {"path": path}

        schema = ds.schema()
        field_names = [f.name for f in schema.fields]
        assert "dim0" in field_names
        assert "value" in field_names

    def test_schema_inference_hierarchical(self, tmp_path):
        """Schema for a hierarchical file should contain lat, lon, and variable columns."""
        from omfiles.pyspark import OmFileDataSource

        path = str(tmp_path / "hier.om")
        _create_hierarchical_om(path)

        ds = OmFileDataSource.__new__(OmFileDataSource)
        ds.options = {"path": path, "include_coordinates": "false"}

        schema = ds.schema()
        field_names = [f.name for f in schema.fields]
        assert "temperature_2m" in field_names
        assert "wind_speed_10m" in field_names

    def test_schema_inference_hierarchical_with_coords(self, tmp_path):
        """With include_coordinates=true (default), lat/lon columns should appear."""
        from omfiles.pyspark import OmFileDataSource

        path = str(tmp_path / "hier_coords.om")
        _create_hierarchical_om(path)

        ds = OmFileDataSource.__new__(OmFileDataSource)
        ds.options = {"path": path}

        schema = ds.schema()
        field_names = [f.name for f in schema.fields]
        assert "latitude" in field_names
        assert "longitude" in field_names
        assert "temperature_2m" in field_names

    def test_schema_inference_selected_variables(self, tmp_path):
        """Only selected variables should appear in the schema."""
        from omfiles.pyspark import OmFileDataSource

        path = str(tmp_path / "selected.om")
        _create_hierarchical_om(path)

        ds = OmFileDataSource.__new__(OmFileDataSource)
        ds.options = {"path": path, "variables": "temperature_2m", "include_coordinates": "false"}

        schema = ds.schema()
        field_names = [f.name for f in schema.fields]
        assert "temperature_2m" in field_names
        assert "wind_speed_10m" not in field_names


class TestOmFileDataSourceReader:
    """Tests for the DataSourceReader read path."""

    def test_read_flat_array(self, tmp_path):
        """Reading a flat array should yield one row per element."""
        from omfiles.pyspark import OmFileDataSourceReader, OmVariablePartition

        path = str(tmp_path / "flat.om")
        data = np.arange(20, dtype=np.float32).reshape(4, 5)
        _create_flat_om(path, data)

        reader = OmFileDataSourceReader.__new__(OmFileDataSourceReader)
        reader.options = {"path": path}
        reader.schema = None  # not used in the read path directly

        partition = OmVariablePartition("__array__")
        rows = list(reader.read(partition))
        assert len(rows) == 20
        # First row should be (dim0_index, value)
        assert rows[0] == (0, 0.0)
        # Last row
        assert rows[-1] == (3, 19.0)

    def test_read_spatial_variable_no_coords(self, tmp_path):
        """Reading a spatial variable without coordinates yields (value,) tuples."""
        from omfiles.pyspark import OmFileDataSourceReader, OmVariablePartition

        path = str(tmp_path / "hier.om")
        expected = _create_hierarchical_om(path)

        reader = OmFileDataSourceReader.__new__(OmFileDataSourceReader)
        reader.options = {"path": path, "include_coordinates": "false"}
        reader.schema = None

        partition = OmVariablePartition("temperature_2m")
        rows = list(reader.read(partition))
        assert len(rows) == 25  # 5x5 grid
        # Values should match the written data (row-major)
        values = [r[0] for r in rows]
        np.testing.assert_array_almost_equal(values, expected["temperature_2m"].flatten(), decimal=1)

    def test_read_spatial_variable_with_coords(self, tmp_path):
        """Reading with coordinates should yield (lat, lon, value) tuples."""
        from omfiles.pyspark import OmFileDataSourceReader, OmVariablePartition

        path = str(tmp_path / "hier_coords.om")
        _create_hierarchical_om(path)

        reader = OmFileDataSourceReader.__new__(OmFileDataSourceReader)
        reader.options = {"path": path, "include_coordinates": "true"}
        reader.schema = None

        partition = OmVariablePartition("temperature_2m")
        rows = list(reader.read(partition))
        assert len(rows) == 25
        # Each row should be a 3-tuple (lat, lon, value)
        assert len(rows[0]) == 3

    def test_partitions_hierarchical(self, tmp_path):
        """Partitions should be one per variable for hierarchical files."""
        from omfiles.pyspark import OmFileDataSourceReader, OmVariablePartition

        path = str(tmp_path / "parts.om")
        _create_hierarchical_om(path)

        reader = OmFileDataSourceReader.__new__(OmFileDataSourceReader)
        reader.options = {"path": path, "variables": "temperature_2m,wind_speed_10m"}
        reader.schema = None

        partitions = reader.partitions()
        assert len(partitions) == 2
        names = {p.variable_name for p in partitions}
        assert names == {"temperature_2m", "wind_speed_10m"}

    def test_partitions_flat(self, tmp_path):
        """Flat array files should have a single __array__ partition."""
        from omfiles.pyspark import OmFileDataSourceReader, OmVariablePartition

        path = str(tmp_path / "flat.om")
        _create_flat_om(path, np.arange(10, dtype=np.float32).reshape(2, 5))

        reader = OmFileDataSourceReader.__new__(OmFileDataSourceReader)
        reader.options = {"path": path}
        reader.schema = None

        partitions = reader.partitions()
        assert len(partitions) == 1
        assert partitions[0].variable_name == "__array__"

    def test_nan_becomes_none(self, tmp_path):
        """NaN values in float arrays should become None in yielded rows."""
        from omfiles.pyspark import OmFileDataSourceReader, OmVariablePartition

        path = str(tmp_path / "nan.om")
        data = np.array([[1.0, float("nan")], [3.0, 4.0]], dtype=np.float32)
        writer = OmFileWriter(path)
        v = writer.write_array(data, chunks=[2, 2], scale_factor=1.0, add_offset=0.0, name="temp")
        root = writer.write_group("root", children=[v])
        writer.close(root)

        reader = OmFileDataSourceReader.__new__(OmFileDataSourceReader)
        reader.options = {"path": path, "variables": "temp", "include_coordinates": "false"}
        reader.schema = None

        partition = OmVariablePartition("temp")
        rows = list(reader.read(partition))
        values = [r[0] for r in rows]
        assert values[0] == pytest.approx(1.0, abs=0.01)
        assert values[1] is None  # NaN → None
