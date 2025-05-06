import numpy as np
import omfiles.omfiles as om
import omfiles.xarray_backend as om_xarray
import pytest
import xarray as xr
from xarray.core import indexing

from .test_utils import create_test_om_file, filter_numpy_size_warning

test_dtypes = [
    np.int8, np.uint8, np.int16, np.uint16, np.int32,
    np.uint32, np.int64, np.uint64, np.float32, np.float64
]

@pytest.mark.parametrize('dtype', test_dtypes, ids=[f"{dtype.__name__}" for dtype in test_dtypes])
def test_om_backend_xarray_dtype(dtype, empty_temp_om_file):
    dtype = np.dtype(dtype)

    create_test_om_file(empty_temp_om_file, shape=(5,5), dtype=dtype)

    reader = om.OmFilePyReader(empty_temp_om_file)
    backend_array = om_xarray.OmBackendArray(reader=reader)

    assert isinstance(backend_array.dtype, np.dtype)
    assert backend_array.dtype == dtype

    data = xr.Variable(dims=["x", "y"], data=indexing.LazilyIndexedArray(backend_array))
    assert data.dtype == dtype

    reader.close()


@filter_numpy_size_warning
def test_xarray_backend(temp_om_file):
    ds = xr.open_dataset(temp_om_file, engine="om")
    variable = ds["data"]

    data = variable.values
    assert data.shape == (5, 5)
    assert data.dtype == np.float32
    np.testing.assert_array_equal(
        data,
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
        ],
    )

@filter_numpy_size_warning
def test_xarray_hierarchical_file(empty_temp_om_file):
    # Create test data
    # temperature: lat, lon, alt, time
    temperature_data = np.random.rand(5, 5, 5, 10).astype(np.float32)
    # precipitation: lat, lon, time
    precipitation_data = np.random.rand(5, 5, 10).astype(np.float32)

    # Write hierarchical structure
    writer = om.OmFilePyWriter(empty_temp_om_file)

    # dimensionality metadata
    temperature_dimension_var = writer.write_scalar("LATITUDE,LONGITUDE,ALTITUDE,TIME", name="_ARRAY_DIMENSIONS")
    temp_units = writer.write_scalar("celsius", name="units")
    temp_metadata = writer.write_scalar("Surface temperature", name="description")

    # Write child2 array
    temperature_var = writer.write_array(
        temperature_data,
        chunks=[2, 2, 1, 10],
        name="temperature",
        scale_factor=100000.0,
        children=[temperature_dimension_var, temp_units, temp_metadata]
    )

    # dimensionality metadata
    precipitation_dimension_var = writer.write_scalar("LATITUDE,LONGITUDE,TIME", name="_ARRAY_DIMENSIONS")
    precip_units = writer.write_scalar("mm", name="units")
    precip_metadata = writer.write_scalar("Precipitation", name="description")

    # Write child1 array with attribute children
    precipitation_var = writer.write_array(
        precipitation_data,
        chunks=[2, 2, 10],
        name="precipitation",
        scale_factor=100000.0,
        children=[precipitation_dimension_var, precip_units, precip_metadata]
    )

    # Write dimensions
    lat = writer.write_array(name="LATITUDE", data=np.arange(5).astype(np.float32), chunks=[5])
    lon = writer.write_array(name="LONGITUDE", data=np.arange(5).astype(np.float32), chunks=[5])
    alt = writer.write_array(name="ALTITUDE", data=np.arange(5).astype(np.float32), chunks=[5])
    time = writer.write_array(name="TIME", data=np.arange(10).astype(np.float32), chunks=[10])

    global_attr = writer.write_scalar("This is a hierarchical OM File", name="description")

    # Write root array with children
    root_var = writer.write_group(
        name="",
        children=[temperature_var, precipitation_var, lat, lon, alt, time, global_attr]
    )

    # Finalize the file
    writer.close(root_var)

    ds = xr.open_dataset(empty_temp_om_file, engine="om")
    # Check coords are correctly set
    assert ds.coords["LATITUDE"].values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert ds.coords["LONGITUDE"].values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert ds.coords["ALTITUDE"].values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert ds.coords["TIME"].values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    # Check the global attribute
    assert ds.attrs["description"] == "This is a hierarchical OM File"
    # Check the variables
    assert set(ds.variables) == {"temperature", "precipitation", "LATITUDE", "LONGITUDE", "ALTITUDE", "TIME"}

    # Check temperature data
    temp = ds["temperature"]
    np.testing.assert_array_almost_equal(temp.values, temperature_data, decimal=4)
    assert temp.shape == (5, 5, 5, 10)
    assert temp.dtype == np.float32
    assert temp.dims == ("LATITUDE", "LONGITUDE", "ALTITUDE", "TIME")
    # Check attributes
    assert temp.attrs["description"] == "Surface temperature"
    assert temp.attrs["units"] == "celsius"

    # Check precipitation data
    precip = ds["precipitation"]
    np.testing.assert_array_almost_equal(precip.values, precipitation_data, decimal=4)
    assert precip.shape == (5, 5, 10)
    assert precip.dtype == np.float32
    assert precip.dims == ("LATITUDE", "LONGITUDE", "TIME")
    # Check attributes
    assert precip.attrs["description"] == "Precipitation"
    assert precip.attrs["units"] == "mm"

    # Check that dimensions are correctly assigned to dimensions variables
    assert ds["LATITUDE"].dims == ("LATITUDE",)
    assert ds["LONGITUDE"].dims == ("LONGITUDE",)
    assert ds["ALTITUDE"].dims == ("ALTITUDE",)
    assert ds["TIME"].dims == ("TIME",)

    # Test some xarray operations to ensure everything works as expected
    # Try selecting a subset
    subset = ds.sel(TIME=slice(0, 5))
    assert subset["temperature"].shape == (5, 5, 5, 6)
    assert subset["precipitation"].shape == (5, 5, 6)

    # Try computing mean over a dimension
    mean_temp = ds["temperature"].mean(dim="TIME")
    assert mean_temp.shape == (5, 5, 5)
    assert mean_temp.dims == ("LATITUDE", "LONGITUDE", "ALTITUDE")
