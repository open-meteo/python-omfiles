"""
PySpark custom data source for reading Open-Meteo .om files.

This module provides a PySpark DataSource implementation that allows reading .om files
directly into Spark DataFrames, enabling efficient distributed processing in Databricks
and other Spark environments.

Requires PySpark (Databricks Runtime 15.2+) and ``omfiles[fsspec,grids]``.

Example usage::

    from omfiles.pyspark import OmFileDataSource

    spark.dataSource.register(OmFileDataSource)

    # Read a spatial .om file from S3
    df = (
        spark.read.format("om")
        .option("path", "s3://openmeteo/data_spatial/dwd_icon/2026/03/01/0000Z/2026-03-01T0000.om")
        .option("variables", "temperature_2m,wind_speed_10m")
        .option("s3_anon", "true")
        .load()
    )
    df.show()

    # Save as Delta table for fast reuse
    df.write.format("delta").saveAsTable("weather.temperature")
"""

from __future__ import annotations

from typing import Iterator, Sequence, Tuple

from pyspark.sql.datasource import DataSource, DataSourceReader, InputPartition
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


def _numpy_dtype_to_spark(np_dtype) -> DoubleType | FloatType | IntegerType | LongType:
    """Map a numpy dtype to the corresponding PySpark type."""
    import numpy as np

    name = np.dtype(np_dtype).name
    mapping = {
        "float32": FloatType(),
        "float64": DoubleType(),
        "int8": IntegerType(),
        "int16": IntegerType(),
        "int32": IntegerType(),
        "int64": LongType(),
        "uint8": IntegerType(),
        "uint16": IntegerType(),
        "uint32": IntegerType(),
        "uint64": LongType(),
    }
    return mapping.get(name, DoubleType())


class OmVariablePartition(InputPartition):
    """One partition per variable in the .om file."""

    def __init__(self, variable_name: str):
        self.variable_name = variable_name


class OmFileDataSource(DataSource):
    """
    PySpark DataSource for reading Open-Meteo ``.om`` files.

    Options:
        path (str): Path to the ``.om`` file.  Can be a local path or an S3 URI
            (e.g. ``s3://openmeteo/data_spatial/…``).
        variables (str, optional): Comma-separated list of variable names to read.
            When omitted, all array children of the root group are read.
        s3_anon (str, optional): Set to ``"true"`` for anonymous S3 access (default ``"true"``).
        s3_block_size (str, optional): S3 read block size in bytes (default ``"65536"``).
        cache_storage (str, optional): Local directory for fsspec block-cache (default ``""`` = no caching).
        include_coordinates (str, optional): ``"true"`` (default) to add ``latitude`` /
            ``longitude`` columns derived from the grid's CRS.
        row_chunk_size (str, optional): Number of latitude rows per partition (default ``"64"``).
    """

    @classmethod
    def name(cls) -> str:
        return "om"

    def schema(self) -> StructType:
        """
        Infer the Spark schema by inspecting the .om file metadata.

        For spatial files (root is a group) the schema is:
            - ``latitude``  DOUBLE
            - ``longitude`` DOUBLE
            - one column per selected variable (FLOAT / DOUBLE / INT / …)

        For flat array files (root is an array) the schema is:
            - one column ``value`` with the array's dtype
        """
        import numpy as np

        reader = self._open_reader()
        try:
            if reader.is_group:
                fields: list[StructField] = []
                include_coords = self.options.get("include_coordinates", "true").lower() == "true"
                if include_coords:
                    fields.append(StructField("latitude", DoubleType(), nullable=False))
                    fields.append(StructField("longitude", DoubleType(), nullable=False))

                variable_names = self._resolve_variable_names(reader)
                for var_name in variable_names:
                    child = reader.get_child_by_name(var_name)
                    spark_type = _numpy_dtype_to_spark(child.dtype)
                    fields.append(StructField(var_name, spark_type, nullable=True))
                return StructType(fields)
            elif reader.is_array:
                spark_type = _numpy_dtype_to_spark(reader.dtype)
                dims = len(reader.shape)
                fields = []
                for i in range(dims - 1):
                    fields.append(StructField(f"dim{i}", LongType(), nullable=False))
                fields.append(StructField("value", spark_type, nullable=True))
                return StructType(fields)
            else:
                raise ValueError("Root of .om file is a scalar — cannot be read as a table.")
        finally:
            reader.close()

    def reader(self, schema: StructType) -> DataSourceReader:
        return OmFileDataSourceReader(schema, self.options)

    # ------------------------------------------------------------------
    # Internal helpers (only used in the *driver* to infer schema)
    # ------------------------------------------------------------------

    def _open_reader(self):
        """Open an OmFileReader from the configured path."""
        from omfiles import OmFileReader

        path: str = self.options.get("path", "")
        if not path:
            raise ValueError("The 'path' option is required.")

        if path.startswith("s3://"):
            return self._open_s3_reader(path)
        else:
            return OmFileReader(path)

    def _open_s3_reader(self, path: str):
        """Open an OmFileReader backed by fsspec / S3."""
        import fsspec

        from omfiles import OmFileReader

        s3_anon = self.options.get("s3_anon", "true").lower() == "true"
        block_size = int(self.options.get("s3_block_size", "65536"))
        cache_storage = self.options.get("cache_storage", "")

        if cache_storage:
            uri = f"blockcache::{path}"
            backend = fsspec.open(
                uri,
                mode="rb",
                s3={"anon": s3_anon, "default_block_size": block_size},
                blockcache={"cache_storage": cache_storage},
            )
        else:
            backend = fsspec.open(
                path,
                mode="rb",
                s3={"anon": s3_anon, "default_block_size": block_size},
            )
        return OmFileReader(backend)

    def _resolve_variable_names(self, reader) -> list[str]:
        """Return the list of variable names to read."""
        variables_opt = self.options.get("variables", "")
        if variables_opt:
            return [v.strip() for v in variables_opt.split(",") if v.strip()]
        # Auto-discover all array children
        names: list[str] = []
        for i in range(reader.num_children):
            child = reader.get_child_by_index(i)
            if child.is_array:
                names.append(child.name)
        return names


class OmFileDataSourceReader(DataSourceReader):
    """
    Reads .om file data and yields rows to Spark.

    Each variable is read as a separate partition to enable parallelism across
    the Spark cluster.  Within each partition the full (lat × lon) grid is read
    for that single variable and yielded row-by-row.
    """

    def __init__(self, schema: StructType, options: dict):
        self.schema = schema
        self.options = options

    def partitions(self) -> Sequence[InputPartition]:
        """Create one partition per variable for spatial files, or one partition for flat arrays."""
        from omfiles import OmFileReader

        path = self.options.get("path", "")
        reader = self._open_reader()
        try:
            if reader.is_group:
                variable_names = self._resolve_variable_names(reader)
                return [OmVariablePartition(name) for name in variable_names]
            else:
                # Flat array — single partition
                return [OmVariablePartition("__array__")]
        finally:
            reader.close()

    def read(self, partition: InputPartition) -> Iterator[Tuple]:
        """Read data for a single variable partition and yield rows."""
        import numpy as np

        reader = self._open_reader()
        try:
            if isinstance(partition, OmVariablePartition) and partition.variable_name == "__array__":
                yield from self._read_flat_array(reader)
            elif isinstance(partition, OmVariablePartition):
                yield from self._read_spatial_variable(reader, partition.variable_name)
            else:
                raise ValueError(f"Unexpected partition type: {type(partition)}")
        finally:
            reader.close()

    # ------------------------------------------------------------------
    # Flat array reading
    # ------------------------------------------------------------------

    def _read_flat_array(self, reader) -> Iterator[Tuple]:
        """Yield one row per element in a flat array (with dimension indices)."""
        import numpy as np

        data = reader[...]
        it = np.nditer(data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            val = float(it[0]) if np.isfinite(it[0]) else None
            yield (*idx[:-1], val) if len(idx) > 1 else (val,)
            it.iternext()

    # ------------------------------------------------------------------
    # Spatial (hierarchical) variable reading
    # ------------------------------------------------------------------

    def _read_spatial_variable(self, reader, variable_name: str) -> Iterator[Tuple]:
        """Read a single spatial variable and yield (lat, lon, value) rows."""
        import numpy as np

        include_coords = self.options.get("include_coordinates", "true").lower() == "true"
        child = reader.get_child_by_name(variable_name)
        data = child[...]  # shape: (ny, nx) or higher-dimensional

        if include_coords:
            try:
                from omfiles.grids import OmGrid

                # Try to read crs_wkt from file attributes
                crs_wkt = self._get_crs_wkt(reader)
                grid = OmGrid(crs_wkt, data.shape[:2])
                lon2d, lat2d = grid.get_meshgrid()
            except Exception:
                # Fallback: use integer indices as coordinates
                ny, nx = data.shape[:2]
                lat2d = np.arange(ny, dtype=np.float64).reshape(ny, 1) * np.ones((1, nx))
                lon2d = np.arange(nx, dtype=np.float64).reshape(1, nx) * np.ones((ny, 1))

        ny, nx = data.shape[:2]
        for y in range(ny):
            for x in range(nx):
                val = data[y, x]
                # Convert numpy scalar to Python; NaN → None
                if hasattr(val, "item"):
                    val = val.item()
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    val = None
                if include_coords:
                    yield (float(lat2d[y, x]), float(lon2d[y, x]), val)
                else:
                    yield (val,)

    def _get_crs_wkt(self, reader) -> str:
        """Try to retrieve crs_wkt from the .om file attributes."""
        try:
            crs_reader = reader.get_child_by_name("crs_wkt")
            if crs_reader.is_scalar:
                return crs_reader.read_scalar()
        except Exception:
            pass
        raise ValueError("Could not find crs_wkt in file attributes")

    # ------------------------------------------------------------------
    # Helpers (duplicated from DataSource — these must be serializable
    # and run on executors, not on the driver)
    # ------------------------------------------------------------------

    def _open_reader(self):
        """Open an OmFileReader from the configured path."""
        from omfiles import OmFileReader

        path: str = self.options.get("path", "")
        if not path:
            raise ValueError("The 'path' option is required.")

        if path.startswith("s3://"):
            return self._open_s3_reader(path)
        else:
            return OmFileReader(path)

    def _open_s3_reader(self, path: str):
        """Open an OmFileReader backed by fsspec / S3."""
        import fsspec

        from omfiles import OmFileReader

        s3_anon = self.options.get("s3_anon", "true").lower() == "true"
        block_size = int(self.options.get("s3_block_size", "65536"))
        cache_storage = self.options.get("cache_storage", "")

        if cache_storage:
            uri = f"blockcache::{path}"
            backend = fsspec.open(
                uri,
                mode="rb",
                s3={"anon": s3_anon, "default_block_size": block_size},
                blockcache={"cache_storage": cache_storage},
            )
        else:
            backend = fsspec.open(
                path,
                mode="rb",
                s3={"anon": s3_anon, "default_block_size": block_size},
            )
        return OmFileReader(backend)

    def _resolve_variable_names(self, reader) -> list[str]:
        """Return the list of variable names to read."""
        variables_opt = self.options.get("variables", "")
        if variables_opt:
            return [v.strip() for v in variables_opt.split(",") if v.strip()]
        names: list[str] = []
        for i in range(reader.num_children):
            child = reader.get_child_by_index(i)
            if child.is_array:
                names.append(child.name)
        return names
