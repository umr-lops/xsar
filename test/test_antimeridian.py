import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box
from shapely.ops import unary_union

from xsar.base_meta import BaseMeta, _ShapefileFeature
from xsar.utils import geometry_to_lon180, geometry_to_lon360

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


@pytest.mark.parametrize(
    "longitudes, expected_flag, expected_longitudes",
    [
        ([175.0, 179.0, -179.0], True, [175.0, 179.0, 181.0]),
        ([-10.0, 0.0, 10.0], False, [-10.0, 0.0, 10.0]),
    ],
)
def test_continuous_longitude(longitudes, expected_flag, expected_longitudes):
    meta = DummyMeta()
    lon = xr.DataArray(np.array(longitudes), dims="x", attrs={"units": "degrees"})
    continuous = meta._continuous_longitude(lon)
    assert meta.cross_antimeridian is expected_flag
    np.testing.assert_allclose(continuous.values, expected_longitudes)
    assert continuous.attrs == {"units": "degrees"}


def test_geometry_to_lon360_and_back():
    # a footprint crossing the antimeridian, given in two parts in [-180, 180]
    split = unary_union([box(170, -45, 180, -40), box(-180, -45, -175, -40)])

    continuous = geometry_to_lon360(split)
    assert continuous.geom_type == "Polygon"
    assert continuous.bounds == (170.0, -45.0, 185.0, -40.0)
    assert continuous.area == pytest.approx(split.area)
    # already continuous: unchanged
    assert geometry_to_lon360(continuous) is continuous

    back = geometry_to_lon180(continuous)
    assert back.geom_type == "MultiPolygon"
    assert back.bounds == (-180.0, -45.0, 180.0, -40.0)
    assert back.area == pytest.approx(split.area)
    # already in [-180, 180]: unchanged
    assert geometry_to_lon180(back) is back

    # a geometry far from the antimeridian is never modified
    europe = box(-5, 45, 10, 50)
    assert geometry_to_lon180(europe) is europe
    assert geometry_to_lon360(box(5, 45, 10, 50)).bounds == (5.0, 45.0, 10.0, 50.0)


class DummyMetaWithFootprint(DummyMeta):
    def __init__(self, footprint):
        super().__init__()
        self._footprint = footprint
        self._mask_features = {}
        self._mask_intersecting_geometries = {}
        self._mask_geometry = {}

    @property
    def footprint(self):
        return self._footprint


def test_get_mask_on_both_sides_of_antimeridian():
    # footprint crossing the antimeridian, in the continuous [0, 360] range
    meta = DummyMetaWithFootprint(box(176, -42, 183, -37))
    # land on both sides, as given by a [-180, 180] coastline file
    west_of_180 = box(178, -40, 179, -39)
    east_of_180 = box(-179, -40, -178, -39)  # i.e. [181, 182]
    outside = box(-70, -40, -69, -39)
    meta.set_mask_feature(
        "land_test",
        _ShapefileFeature(gpd.GeoSeries([west_of_180, east_of_180, outside])),
    )

    mask = meta.get_mask("land_test")

    assert mask.is_valid
    assert mask.area == pytest.approx(2.0)
    assert mask.bounds == (178.0, -40.0, 182.0, -39.0)
