"""
Sentinel1Meta on a geolocation grid crossing the antimeridian, without any SAFE.

The grid covers about 179E..181E (written 179 ... 179.8, -180 ... -179 as in the product),
40S..38S, tilted like a real orbit. Before the fix: footprint was a band around the globe,
approx_transform was meaningless, cross_antimeridian changed after the first coords2ll call,
the land mask was empty and the ground heading was wrong by about 100 deg.
"""
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from xsar.base_meta import _ShapefileFeature
from xsar.sentinel1_meta import Sentinel1Meta


class _FakeNode:
    """Stands for the 'geolocationGrid' node of the datatree."""

    def __init__(self, ds):
        self.ds = ds

    def to_dataset(self):
        return self.ds.copy(deep=True)


@pytest.fixture
def meta():
    lines = np.arange(0.0, 1001.0, 100.0)
    samples = np.arange(0.0, 1001.0, 100.0)
    L, S = np.meshgrid(lines, samples, indexing="ij")
    lon = 179.0 + S * 0.002 - L * 0.0005
    lon = (lon + 180) % 360 - 180  # as given by the product
    lat = -40.0 + L * 0.002
    grid = xr.Dataset(
        {
            "longitude": (("line", "sample"), lon),
            "latitude": (("line", "sample"), lat),
            "height": (("line", "sample"), np.zeros_like(lon)),
        },
        coords={"line": lines, "sample": samples},
    )
    m = Sentinel1Meta.__new__(Sentinel1Meta)
    m.dt = {"geolocationGrid": _FakeNode(grid)}
    m.multidataset = False
    m.xsd_definitions = {}
    m._geoloc = None
    m._mask_features, m._mask_intersecting_geometries, m._mask_geometry = {}, {}, {}
    # island astride 180, stored in two pieces as in a [-180, 180] coastline file
    island = [box(179.8, -39.2, 180.0, -38.8), box(-180.0, -39.2, -179.8, -38.8)]
    m.set_mask_feature("island", _ShapefileFeature(gpd.GeoSeries(island)))
    return m


def test_footprint_and_approx_transform(meta):
    assert meta.cross_antimeridian is True
    assert meta.footprint.area == pytest.approx(4.0)
    lon, _ = meta.approx_transform * (0, 1000)
    assert (lon - 181 + 180) % 360 - 180 == pytest.approx(0, abs=1e-6)


def test_flag_does_not_change(meta):
    lon, _ = meta.coords2ll(0, 1000)
    assert lon == pytest.approx(-179.0)
    assert meta.cross_antimeridian is True


def test_ll2coords(meta):
    line, sample = meta.ll2coords(-179.75, -39.0)
    assert (line, sample) == (pytest.approx(500), pytest.approx(750))


def test_land_mask_astride_antimeridian(meta):
    assert meta.get_mask("island").area == pytest.approx(0.16)


def test_ground_heading(meta):
    heading = meta.coords2heading(np.array([500.0]), np.array([500.0]), approx=True)
    assert float(np.ravel(heading)[0]) == pytest.approx(-11.0, abs=0.05)
