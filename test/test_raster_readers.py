from xsar.raster_readers import _to_lon180
import pytest
import xarray as xr
import copy
import numpy as np

val_lon_180s = [[120.,121.,122.,123.],[120.6,121.7,122.8,123.9]]
lat = xr.DataArray(np.array([[40.,41.,42.,43.],[40.5,41.5,42.5,43.5]]), dims=('y','x'),
                   coords={'x':[1,2,3,4], 'y':[5,6]})
lon = xr.DataArray(np.array(val_lon_180s), dims=('y','x'),
                   coords={'x':[1,2,3,4], 'y':[5,6]})
lon_acheval = xr.DataArray(np.array([[179.,179.2,179.5,179.9],[-179.4,-179.7,-178.8,-179.1]]), dims=('y','x'),
                   coords={'x':[1,2,3,4], 'y':[5,6]})
lon_0_360 = xr.DataArray(np.array(val_lon_180s)+200., dims=('y','x'),
                   coords={'x':[1,2,3,4], 'y':[5,6]})
lon_0_360_treated = xr.DataArray(np.array(val_lon_180s)+200.-360., dims=('y','x'),
                   coords={'x':[1,2,3,4], 'y':[5,6]})
# latwithNan = copy.copy(lat)
ds = xr.Dataset()
ds['lon'] = lon
ds['lat'] = lat

ds_on_antimeridian = xr.Dataset()
ds_on_antimeridian['lon'] = lon_acheval
ds_on_antimeridian['lat'] = lat

ds_0_360 = xr.Dataset()
ds_0_360['lon'] = lon_0_360
ds_0_360['lat'] = lat

ds_0_360_expected = xr.Dataset()
ds_0_360_expected['lon'] = lon_0_360_treated
ds_0_360_expected['lat'] = lat
@pytest.mark.parametrize(
    ["ds", "expected"],
    (
        pytest.param(ds, ds, id="180_180"),
        pytest.param(ds_0_360, ds_on_antimeridian, id="0_360"),
        pytest.param(ds_on_antimeridian, ds_0_360_expected, id="180_180_a_cheval"),
    ),
)
def test_to_lon180(ds, expected):
    actual_ds = _to_lon180(ds)
    assert actual_ds == expected