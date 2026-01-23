"""
Test for handling invalid geometries in masks.
This reproduces the GSHHG self-intersection issue that caused TopologyException.
ISSUE : https://github.com/umr-lops/xsar/issues/275
PULL REQUEST : https://github.com/umr-lops/xsar/pull/276
"""
import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.validation import explain_validity
import cartopy

from xsar.base_meta import BaseMeta


class DummyMeta(BaseMeta):
    """Dummy meta class for testing with mocked footprint"""

    def __init__(self, footprint_geom):
        super().__init__()
        self._footprint_geom = footprint_geom
        self._mask_features = {}
        self._mask_intersecting_geometries = {}
        self._mask_geometry = {}

    @property
    def footprint(self):
        return self._footprint_geom

    @property
    def _dict_coords2ll(self):
        return {}

    @property
    def approx_transform(self):
        return None

    @property
    def _get_time_range(self):
        return pd.Interval(pd.Timestamp("2023-10-15"), pd.Timestamp("2023-10-15"))


def get_problematic_gshhs_polygon():
    """
    Returns the actual GSHHG polygon that caused TopologyException.
    This polygon has a self-intersection at (-69.753676, 43.774717).

    Source: GSHHS_h_L1.shp index 2244 "Self-intersection[-69.7536763036464 43.7747178403855]"
    Error: TopologyException: side location conflict at -69.753676303646373 43.774717840385456
    """
    coords = [
        (-69.761848, 43.756428), (-69.761642, 43.756477), (-69.765282, 43.752621),
        (-69.767799, 43.764641), (-69.771133, 43.755108), (-69.775627, 43.754505),
        (-69.776154, 43.764874), (-69.771271, 43.771881), (-69.769539, 43.769875),
        (-69.772346, 43.7747), (-69.769539, 43.778896), (-69.773384, 43.777336),
        (-69.775543, 43.789516), (-69.781868, 43.79213), (-69.782799, 43.795891),
        (-69.78006, 43.805584), (-69.774368, 43.808651), (-69.778091, 43.810043),
        (-69.773178, 43.816193), (-69.772911, 43.826752), (-69.768478, 43.82748),
        (-69.772125, 43.827564), (-69.77037, 43.833866), (-69.764488, 43.838039),
        (-69.767654, 43.83699), (-69.764946, 43.842857), (-69.761269, 43.843067),
        (-69.7658, 43.842403), (-69.76783, 43.848705), (-69.769188, 43.84412),
        (-69.77182, 43.850552), (-69.768463, 43.856533), (-69.761352, 43.852623),
        (-69.765602, 43.855373), (-69.762428, 43.856178), (-69.768082, 43.857018),
        (-69.75956, 43.857472), (-69.763031, 43.860344), (-69.757156, 43.862385),
        (-69.758484, 43.868206), (-69.764137, 43.871925), (-69.758354, 43.874088),
        (-69.755257, 43.882225), (-69.752152, 43.863876), (-69.746628, 43.8685),
        (-69.744186, 43.879242), (-69.741242, 43.875717), (-69.739731, 43.878868),
        (-69.733673, 43.863682), (-69.73613, 43.858295), (-69.731201, 43.860325),
        (-69.727821, 43.858559), (-69.730232, 43.856117), (-69.735992, 43.857193),
        (-69.73307, 43.85249), (-69.735283, 43.845657), (-69.747917, 43.825764),
        (-69.743355, 43.827442), (-69.744988, 43.823101), (-69.750427, 43.821388),
        (-69.747299, 43.816975), (-69.750679, 43.809708), (-69.747696, 43.81007),
        (-69.754562, 43.799053), (-69.753563, 43.796574), (-69.747749, 43.804718),
        (-69.750359, 43.787693), (-69.744385, 43.79892), (-69.746429, 43.808655),
        (-69.732018, 43.836842), (-69.732346, 43.845276), (-69.723991, 43.851276),
        (-69.719284, 43.847069), (-69.716667, 43.849129), (-69.716995, 43.836624),
        (-69.712997, 43.838634), (-69.709442, 43.8326), (-69.713242, 43.829716),
        (-69.707344, 43.828537), (-69.714943, 43.810127), (-69.718567, 43.820664),
        (-69.723267, 43.819828), (-69.717194, 43.7924), (-69.721069, 43.785225),
        (-69.725395, 43.787739), (-69.727814, 43.782673), (-69.731857, 43.782322),
        (-69.727455, 43.781174), (-69.725342, 43.785461), (-69.720924, 43.782417),
        (-69.73999, 43.770763), (-69.735962, 43.775669), (-69.739502, 43.782654),
        (-69.737541, 43.787086), (-69.742104, 43.788853), (-69.738442, 43.787361),
        (-69.740044, 43.781384), (-69.73661, 43.775352), (-69.749199, 43.761128),
        (-69.747024, 43.758758), (-69.756699, 43.752232), (-69.754898, 43.761948),
        (-69.75106, 43.7602), (-69.752434, 43.769447), (-69.749283, 43.772919),
        (-69.75396, 43.774834), (-69.752235, 43.779602), (-69.754715, 43.771198),
        (-69.757889, 43.771946), (-69.760155, 43.765842), (-69.766136, 43.764671),
        (-69.761848, 43.756428)
    ]
    return Polygon(coords)


@pytest.fixture
def sentinel1_footprint():
    """
    Footprint from S1A_EW_GRDM_1SDV_20231015T181242_20231015T181346_050779_061EA4_1F77
    Bounds: (176.74864974310074, 42.97978252061119, 182.9591742035645, 47.453072721813584)
    """
    return box(176.74865, 42.97978, 182.95917, 47.45307)


def test_gshhs_self_intersection_polygon():
    """
    Test that the problematic GSHHG polygon is indeed invalid.
    This verifies our test data is correct.
    """
    poly = get_problematic_gshhs_polygon()

    assert not poly.is_valid, "The test polygon should be invalid"

    explanation = explain_validity(poly)
    assert "Self-intersection" in explanation, f"Expected self-intersection, got: {explanation}"

    # Verify the self-intersection is at the expected location
    assert "-69.753676" in explanation and "43.774717" in explanation


def test_get_mask_handles_self_intersecting_geometry(sentinel1_footprint):
    """
    Test that get_mask correctly handles the GSHHG self-intersecting polygon
    that previously caused: TopologyException: side location conflict at
    -69.753676303646373 43.774717840385456

    This is a regression test for the fix using make_valid().
    """
    meta = DummyMeta(sentinel1_footprint)

    # Get the problematic polygon
    invalid_poly = get_problematic_gshhs_polygon()

    # Create a mock cartopy feature with the invalid geometry
    geoseries = gpd.GeoSeries([invalid_poly])
    mock_feature = cartopy.feature.ShapelyFeature(
        geoseries,
        cartopy.crs.PlateCarree()
    )

    # Set the mask feature
    meta.set_mask_feature("gshhs_test", mock_feature)

    # This should NOT raise TopologyException thanks to make_valid()
    result = meta.get_mask("gshhs_test")

    # Verify the result is valid
    assert result is not None
    assert isinstance(result, (Polygon, MultiPolygon))
    assert result.is_valid, "Result geometry should be valid after make_valid()"


def test_get_mask_with_multiple_invalid_coastal_geometries(sentinel1_footprint):
    """
    Test union_all() with multiple invalid geometries similar to the GSHHG case.

    In the real scenario, there were 708 polygons near the crash point,
    with 1 invalid geometry causing the union_all() to fail.
    """
    meta = DummyMeta(sentinel1_footprint)

    # Create multiple geometries: some valid, one invalid
    valid_poly1 = Polygon([
        (-69.8, 43.8), (-69.7, 43.8), (-69.7, 43.9), (-69.8, 43.9), (-69.8, 43.8)
    ])
    valid_poly2 = Polygon([
        (-69.72, 43.82), (-69.71, 43.82), (-69.71,
                                           43.83), (-69.72, 43.83), (-69.72, 43.82)
    ])
    invalid_poly = get_problematic_gshhs_polygon()

    # Create a mock feature with mixed geometries
    geoseries = gpd.GeoSeries([valid_poly1, invalid_poly, valid_poly2])
    mock_feature = cartopy.feature.ShapelyFeature(
        geoseries,
        cartopy.crs.PlateCarree()
    )

    # Set the mask feature
    meta.set_mask_feature("gshhs_multiple", mock_feature)

    # This should NOT raise TopologyException
    result = meta.get_mask("gshhs_multiple")

    # Verify the result
    assert result is not None
    assert result.is_valid, "Result should be valid"
    # Result might be empty if geometries don't intersect the footprint
    assert isinstance(result, (Polygon, MultiPolygon))


def test_make_valid_fixes_gshhs_polygon():
    """
    Direct test that make_valid() fixes the problematic polygon.
    """
    from shapely.validation import make_valid

    invalid_poly = get_problematic_gshhs_polygon()
    assert not invalid_poly.is_valid

    # Apply make_valid
    fixed_poly = make_valid(invalid_poly)

    # Verify it's now valid
    assert fixed_poly.is_valid, "make_valid() should produce a valid geometry"
    assert isinstance(fixed_poly, (Polygon, MultiPolygon))

    # The fixed polygon should have similar bounds
    orig_bounds = invalid_poly.bounds
    fixed_bounds = fixed_poly.bounds

    # Bounds should be close (within tolerance)
    for i in range(4):
        assert abs(orig_bounds[i] - fixed_bounds[i]) < 0.01, \
            "Fixed geometry should have similar bounds"


def test_get_mask_caches_fixed_geometry(sentinel1_footprint):
    """
    Test that the fixed geometry is cached and reused.
    """
    meta = DummyMeta(sentinel1_footprint)

    invalid_poly = get_problematic_gshhs_polygon()
    geoseries = gpd.GeoSeries([invalid_poly])
    mock_feature = cartopy.feature.ShapelyFeature(
        geoseries,
        cartopy.crs.PlateCarree()
    )

    meta.set_mask_feature("gshhs_cache", mock_feature)

    # First call
    result1 = meta.get_mask("gshhs_cache")

    # Second call should return cached result
    result2 = meta.get_mask("gshhs_cache")

    # Should be the same object (cached)
    assert result1 is result2, "Second call should return cached geometry"


def test_union_all_fails_without_make_valid():
    """
    Test that demonstrates union_all() fails WITHOUT make_valid() on invalid geometries.
    This proves the fix is necessary.

    This test directly uses shapely's union_all() to show the TopologyException
    would occur without the make_valid() fix in get_mask().
    """
    from shapely import union_all, GEOSException

    # Create multiple geometries including the invalid one
    valid_poly1 = Polygon([
        (-69.8, 43.8), (-69.7, 43.8), (-69.7, 43.9), (-69.8, 43.9), (-69.8, 43.8)
    ])
    valid_poly2 = Polygon([
        (-69.72, 43.82), (-69.71, 43.82), (-69.71,
                                           43.83), (-69.72, 43.83), (-69.72, 43.82)
    ])
    invalid_poly = get_problematic_gshhs_polygon()

    # Create GeoSeries with invalid geometry (mimics the GSHHG case)
    geoseries = gpd.GeoSeries([valid_poly1, invalid_poly, valid_poly2])

    # Verify that union_all fails on the invalid geometries
    with pytest.raises((GEOSException, Exception)) as exc_info:
        union_all(geoseries.values)

    # Check that it's the expected TopologyException
    error_msg = str(exc_info.value)
    assert "TopologyException" in error_msg or "side location conflict" in error_msg, \
        f"Expected TopologyException, got: {error_msg}"


def test_union_all_succeeds_with_make_valid():
    """
    Test that union_all() succeeds AFTER applying make_valid() to invalid geometries.
    This demonstrates the fix works.
    """
    from shapely import union_all
    from shapely.validation import make_valid

    # Create multiple geometries including the invalid one
    valid_poly1 = Polygon([
        (-69.8, 43.8), (-69.7, 43.8), (-69.7, 43.9), (-69.8, 43.9), (-69.8, 43.8)
    ])
    valid_poly2 = Polygon([
        (-69.72, 43.82), (-69.71, 43.82), (-69.71,
                                           43.83), (-69.72, 43.83), (-69.72, 43.82)
    ])
    invalid_poly = get_problematic_gshhs_polygon()

    # Create GeoSeries and fix invalid geometries
    geoseries = gpd.GeoSeries([valid_poly1, invalid_poly, valid_poly2])

    # Apply make_valid to invalid geometries (this is what get_mask does)
    geoseries_fixed = geoseries.apply(
        lambda g: make_valid(g) if not g.is_valid else g
    )

    # Now union_all should succeed
    result = union_all(geoseries_fixed.values)

    # Verify the result is valid
    assert result.is_valid, "Union result should be valid after make_valid()"
    assert isinstance(result, (Polygon, MultiPolygon))
