
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xsar.base_meta import BaseMeta


class DummyMeta(BaseMeta):
    @property
    def footprint(self):
        return None

    @property
    def _dict_coords2ll(self):
        return {}

    @property
    def approx_transform(self):
        return None

    @property
    def _get_time_range(self):
        # minimal placeholder to satisfy BaseMeta
        return pd.Interval(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02"))


@pytest.mark.parametrize(
    "longitudes, expected",
    [
        ([0, 10, 20], False),
        ([179, -179], True),
        ([0, 180], False),
        ([170, -175], True),
    ],
)
def test_cross_antimeridian_from_geoloc(longitudes, expected):
    meta = DummyMeta()
    meta.geoloc = xr.Dataset({"longitude": (("x",), np.array(longitudes))})
    assert meta.cross_antimeridian is expected
