from xsar.raster_readers import _to_lon180
import pytest
import xarray as xr
import numpy as np
import pandas as pd

import types
import yaml
import mapraster

from xsar.base_dataset import BaseDataset


class DummyDataset(BaseDataset):
    def __init__(self, sar_meta):
        super().__init__()
        self.sar_meta = sar_meta
        self._dataset = xr.Dataset()


val_lon_180s = [[120., 121., 122., 123.], [120.6, 121.7, 122.8, 123.9]]
lat = xr.DataArray(np.array([[40., 41., 42., 43.], [40.5, 41.5, 42.5, 43.5]]), dims=('y', 'x'),
                   coords={'x': [1, 2, 3, 4], 'y': [5, 6]})
lon = xr.DataArray(np.array(val_lon_180s), dims=('y', 'x'),
                   coords={'x': [1, 2, 3, 4], 'y': [5, 6]})
lon_acheval = xr.DataArray(np.array([[179., 179.2, 179.5, 179.9], [-179.4, -179.7, -178.8, -179.1]]), dims=('y', 'x'),
                           coords={'x': [1, 2, 3, 4], 'y': [5, 6]})
lon_0_360 = xr.DataArray(np.array(val_lon_180s)+200., dims=('y', 'x'),
                         coords={'x': [1, 2, 3, 4], 'y': [5, 6]})
lon_0_360_treated = xr.DataArray(np.array(val_lon_180s)+200.-360., dims=('y', 'x'),
                                 coords={'x': [1, 2, 3, 4], 'y': [5, 6]})
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
        pytest.param(ds_on_antimeridian, ds_0_360_expected,
                     id="180_180_a_cheval"),
    ),
)
def test_to_lon180(ds, expected):
    actual_ds = _to_lon180(ds)
    assert actual_ds == expected


@pytest.mark.parametrize(
    "cross_antimeridian, expected_to180",
    [
        (False, True),
        (True, False),
    ],
)
def test_load_rasters_vars(monkeypatch, cross_antimeridian, expected_to180):
    """ 
    Test assets from load_rasters_vars
    - check that footprint and cross_antimeridian are correctly passed to map_raster
    - check the new vars in merged dataset
    - check "history" attr 
    - check to180 kwargs in function of cross_antimeridian
    """

    read_calls = []
    map_calls = []

    def fake_get_raster(resource, **kwargs):
        return kwargs["date"], "fake/path"

    def fake_read_raster(path, **kwargs):
        # capture to180 and return two components like ECMWF U10/V10
        read_calls.append(kwargs)
        return xr.Dataset(
            {"U10": (("x", "y"), np.ones((1, 1))),
             "V10": (("x", "y"), np.ones((1, 1)))},
            coords={"x": [0], "y": [0]},
        )

    def fake_map_raster(ds, target_ds, footprint=None, cross_antimeridian=None):
        map_calls.append({"footprint": footprint, "cross": cross_antimeridian})
        return ds

    monkeypatch.setattr(mapraster, "map_raster", fake_map_raster)

    sar_meta = types.SimpleNamespace(
        rasters=pd.DataFrame(
            [{"resource": "fake_path", "read_function": fake_read_raster,
                "get_function": fake_get_raster}],
            index=["rastername_1h"],
        ),
        start_date="2020-01-01 00:00:00.000000",
        footprint="fp",
        cross_antimeridian=cross_antimeridian,
    )

    dataset = DummyDataset(sar_meta)
    merged = dataset._load_rasters_vars()

    # - check that footprint and cross_antimeridian are correctly passed to map_raster

    assert map_calls == [{"footprint": "fp", "cross": cross_antimeridian}]

    # - check the new vars in merged dataset

    assert set(merged.data_vars) == {"rastername_1h_U10", "rastername_1h_V10"}

    #  - check "history" attr
    for var in ("rastername_1h_U10", "rastername_1h_V10"):
        history = yaml.safe_load(merged[var].attrs["history"])
        assert history == {
            var: {"resource": "fake/path", "resource_decoded": "fake/path"}}
    # - check to180 kwargs in function of cross_antimeridian
    assert read_calls[0]["to180"] is expected_to180
