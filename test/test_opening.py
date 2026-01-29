"""
Test for low resolution dataset loading.
Ensures that all satellite types can be opened with resolution parameter.
"""
import pytest
import xsar


@pytest.mark.parametrize(
    "satellite_file",
    [
        pytest.param(
            "S1B_IW_GRDH_1SDV_20181013T062322_20181013T062347_013130_018428_Z010.SAFE",
            id="sentinel1"
        ),
        pytest.param(
            "RS2_OK135107_PK1187782_DK1151894_SCWA_20220407_182127_VV_VH_SGF",
            id="radarsat2"
        ),
        pytest.param(
            "RCM1_OK1050603_PK1050605_1_SC50MB_20200214_115905_HH_HV_Z010",
            id="rcm"
        ),
    ],
)
def test_open_dataset(satellite_file):
    """
    Test that open_dataset works with resolution='1000m' for all satellite types.

    This test verifies that:
    - The dataset can be opened successfully
    - The resolution parameter is accepted
    - Basic dataset properties are present
    """
    # Get the test file
    file_path = xsar.get_test_file(satellite_file)

    # Open dataset at low resolution
    ds = xsar.open_dataset(file_path, resolution='1000m')

    # Verify dataset is loaded
    assert ds is not None, "Dataset should be loaded"

    # Verify it's an xarray Dataset
    import xarray as xr
    assert isinstance(ds, xr.Dataset), "Result should be an xarray Dataset"

    # Verify dataset has some expected dimensions
    assert len(ds.dims) > 0, "Dataset should have dimensions"

    # Verify dataset has some data variables
    assert len(ds.data_vars) > 0, "Dataset should have data variables"

    # Close the dataset
    ds.close()
