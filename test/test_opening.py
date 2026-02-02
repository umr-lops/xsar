"""
Test for low resolution dataset loading.
Ensures that all satellite types can be opened with resolution parameter.
"""
import pytest
import xsar
import xarray as xr


def test_open_dataset_sentinel1():
    """
    Test that open_dataset works with resolution='1000m' for Sentinel-1.

    This test verifies that:
    - The dataset can be opened successfully
    - The resolution parameter is accepted
    - Basic dataset properties are present
    """
    # Get the test file
    satellite_file = "S1B_IW_GRDH_1SDV_20181013T062322_20181013T062347_013130_018428_Z010.SAFE"
    file_path = xsar.get_test_file(satellite_file)

    # Open dataset at low resolution
    ds = xsar.open_dataset(file_path, resolution='1000m')

    # Verify dataset is loaded
    assert ds is not None, "Dataset should be loaded"

    # Verify it's an xarray Dataset
    assert isinstance(ds, xr.Dataset), "Result should be an xarray Dataset"

    # Verify dataset has some expected dimensions
    assert len(ds.dims) > 0, "Dataset should have dimensions"

    # Verify dataset has some data variables
    assert len(ds.data_vars) > 0, "Dataset should have data variables"

    # Close the dataset
    ds.close()


def test_open_dataset_radarsat2():
    """
    Test that open_dataset works with resolution='1000m' for RadarSat-2.

    This test verifies that:
    - The dataset can be opened successfully
    - The resolution parameter is accepted
    - Basic dataset properties are present
    """
    # Get the test file
    satellite_file = "RS2_OK135107_PK1187782_DK1151894_SCWA_20220407_182127_VV_VH_SGF"
    file_path = xsar.get_test_file(satellite_file)

    # Open dataset at low resolution
    ds = xsar.open_dataset(file_path, resolution='1000m')

    # Verify dataset is loaded
    assert ds is not None, "Dataset should be loaded"

    # Verify it's an xarray Dataset
    assert isinstance(ds, xr.Dataset), "Result should be an xarray Dataset"

    # Verify dataset has some expected dimensions
    assert len(ds.dims) > 0, "Dataset should have dimensions"

    # Verify dataset has some data variables
    assert len(ds.data_vars) > 0, "Dataset should have data variables"

    # Close the dataset
    ds.close()


def test_open_dataset_rcm():
    """
    Test that open_dataset works with resolution='1000m' for RCM.

    This test verifies that:
    - The dataset can be opened successfully
    - The resolution parameter is accepted
    - Basic dataset properties are present
    """
    # Get the test file
    satellite_file = "RCM1_OK2460179_PK2462841_1_SCLND_20230301_072431_VV_VH_GRD"
    file_path = xsar.get_test_file(satellite_file)

    # Open dataset at low resolution
    ds = xsar.open_dataset(file_path, resolution='1000m')

    # Verify dataset is loaded
    assert ds is not None, "Dataset should be loaded"

    # Verify it's an xarray Dataset
    assert isinstance(ds, xr.Dataset), "Result should be an xarray Dataset"

    # Verify dataset has some expected dimensions
    assert len(ds.dims) > 0, "Dataset should have dimensions"

    # Verify dataset has some data variables
    assert len(ds.data_vars) > 0, "Dataset should have data variables"

    # Close the dataset
    ds.close()
