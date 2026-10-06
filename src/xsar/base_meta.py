import copy
import logging
import warnings

import rasterio
import shapely
from shapely.geometry import Polygon
from shapely.validation import make_valid
import numpy as np
from datetime import datetime

from abc import abstractmethod

from .raster_readers import available_rasters
from .base_dataset import BaseDataset
import geopandas as gpd

from .utils import (
    class_or_instancemethod,
    to_lon180,
    geometry_to_lon180,
    geometry_to_lon360,
    haversine,
    config,
)

logger = logging.getLogger("xsar.base_meta")
logger.addHandler(logging.NullHandler())

# we know tiff as no geotransform : ignore warning
warnings.filterwarnings(
    "ignore", category=rasterio.errors.NotGeoreferencedWarning)

# allow nan without warnings
# some dask warnings are still non filtered: https://github.com/dask/dask/issues/3245
np.errstate(invalid="ignore")


# Default 'land' mask: shapefile path from the 'land_mask' key of the config.
_DEFAULT_LAND = config.get("land_mask")


class _ShapefileFeature:
    """Minimal feature exposing ``.geometries()``, built from shapefile geometries (OSM, GSHHG...)."""

    def __init__(self, geometries):
        self._geometries = list(geometries)

    def geometries(self):
        return iter(self._geometries)


class BaseMeta(BaseDataset):
    """
    Abstract class that defines necessary common functions for the computation of different SAR metadata
    (Radarsat2, Sentinel1, RCM...).
    This also permit a better maintenance, because these functions aren't redefined many times.
    """

    # default mask feature (see self.set_mask_feature and cls.set_mask_feature)
    _mask_features_raw = {
        "land": _DEFAULT_LAND
    }

    _mask_features = {}
    _mask_intersecting_geometries = {}
    _mask_geometry = {}
    _geoloc = None
    _cross_antimeridian = None
    _rasterized_masks = None
    manifest_attrs = None
    _time_range = None
    name = None
    multidataset = None
    short_name = None
    path = None
    product = None
    manifest = None
    subdatasets = None
    dsid = None
    safe = None
    geoloc = None

    def __init__(self):
        self.rasters = available_rasters.iloc[0:0].copy()

    def _get_mask_feature(self, name):
        # internal method that returns a feature (with a .geometries() method) from a mask name
        if self._mask_features[name] is None:
            feature = self._mask_features_raw[name]
            if feature is None:
                raise ValueError(
                    f"No feature set for mask '{name}'. Set 'land_mask' in "
                    f"~/.xsar/config.yml, or call set_mask_feature('{name}', '/path/to.shp')."
                )
            if isinstance(feature, str):
                # feature is a shapefile (e.g. OSM land-polygons, GSHHG).
                # we get the crs from the shapefile to be able to transform the footprint to this crs_in
                # (so we can use `mask=` in gpd.read_file)
                import fiona
                import pyproj
                from shapely.ops import transform

                with fiona.open(feature) as fshp:
                    try:
                        # proj6 give a " FutureWarning: '+init=<authority>:<code>' syntax is deprecated. "
                        # '<authority>:<code>' is the preferred initialization method"
                        crs_in = fshp.crs["init"]
                    except KeyError:
                        crs_in = fshp.crs
                    crs_in = pyproj.CRS(crs_in)
                proj_transform = pyproj.Transformer.from_crs(
                    pyproj.CRS("EPSG:4326"), crs_in, always_xy=True
                ).transform
                # the shapefile is in [-180, 180]: a footprint crossing the antimeridian
                # (continuous [0, 360] longitudes) is split in two parts
                footprint_crs = transform(proj_transform, geometry_to_lon180(self.footprint))

                with warnings.catch_warnings():
                    # ignore "RuntimeWarning: Sequential read of iterator was interrupted. Resetting iterator."
                    warnings.simplefilter("ignore", RuntimeWarning)
                    # wrap the shapefile geometries in a minimal feature exposing .geometries()
                    feature = _ShapefileFeature(
                        gpd.read_file(feature, mask=footprint_crs)
                        .to_crs(epsg=4326)
                        .geometry
                    )
            if not hasattr(feature, "geometries"):
                raise TypeError(
                    "Expected a feature with a .geometries() method "
                    "(a shapefile path, or any feature exposing .geometries())")
            self._mask_features[name] = feature

        return self._mask_features[name]

    @class_or_instancemethod
    def set_mask_feature(self_or_cls, name, feature):
        """
        Set a named mask from a shapefile path.

        Parameters
        ----------
        name: str
            mask name
        feature: str
            path to a shapefile (or any file readable with fiona).
            Any object exposing a ``.geometries()`` method is also accepted.

        Examples
        --------
            Override the default 'land' mask with your own shapefile, at class level
            (ie as default mask for every new meta):
            ```
            >>> xsar.RadarSat2Meta.set_mask_feature("land", "/path/to/land_polygons.shp")
            >>> xsar.Sentinel1Meta.set_mask_feature("land", "/path/to/land_polygons.shp")
            ```

            Add an extra named mask at instance level (only for this meta instance):
            ```
            >>> meta.set_mask_feature("gshhg", "/path/to/GSHHS_h_L1.shp")
            ```


            High resolution land shapefiles can be found from openstreetmap.
            It is recommended to use WGS84 with large polygons split from https://osmdata.openstreetmap.de/
            The default 'land' mask is read from the 'land_mask' key of ~/.xsar/config.yml.

        See Also
        --------
        xsar.BaseMeta.get_mask
        """

        # see https://stackoverflow.com/a/28238047/5988771 for self_or_cls

        self_or_cls._mask_features_raw[name] = feature

        if not isinstance(self_or_cls, type):
            # self (instance, not class)
            self_or_cls._mask_intersecting_geometries[name] = None
            self_or_cls._mask_geometry[name] = None
            self_or_cls._mask_features[name] = None

    def get_mask(self, name, describe=False):
        """
        Get mask from `name` (e.g. 'land') as a shapely Polygon.
        The resulting polygon is contained in the footprint.

        Parameters
        ----------
        name: str

        Returns
        -------
        shapely.geometry.Polygon

        """
        if describe:
            descr = self._mask_features_raw.get(name)

            # 1) if descr is a str (shapefile path, e.g. OSM land-polygons / GSHHG):
            #    describe the mask by its source path -> ends up in the mask 'history' attr
            if isinstance(descr, str):
                return descr

            # 2) if descr is None
            if descr is None:
                return f"Unknown mask feature: {name}"

            # 3) otherwise a feature object: nice repr like 'module.Class name'
            module = getattr(descr, "__module__", "unknown_module")
            cls = descr.__class__.__name__
            feat_name = getattr(descr, "name",  f"{module}.{cls}")
            return "%s.%s %s" % (
                    module,
                    cls,
                    feat_name,
            )
        if self._mask_geometry[name] is None:
            intersecting_geoms = self._get_mask_intersecting_geometries(name)

            # Fix invalid geometries before union
            if any(~intersecting_geoms.is_valid):
                logging.warning(
                    f"Fixing {sum(~intersecting_geoms.is_valid)} invalid geometries in mask '{name}'")

                if isinstance(intersecting_geoms, gpd.GeoDataFrame):
                    intersecting_geoms['geometry'] = intersecting_geoms['geometry'].apply(
                        lambda g: make_valid(g) if not g.is_valid else g
                    )
                else:
                    intersecting_geoms = intersecting_geoms.apply(
                        lambda g: make_valid(g) if not g.is_valid else g
                    )

            union_geom = intersecting_geoms.union_all()

            # Fix invalid geometry after union
            if not union_geom.is_valid:
                union_geom = make_valid(union_geom)

            if union_geom and not union_geom.is_empty:
                footprint = self.footprint
                if footprint.bounds[2] > 180:
                    # footprint crossing the antimeridian: it is in the continuous
                    # [0, 360] longitude range, so the mask is expressed in the same range
                    union_geom = geometry_to_lon360(union_geom)
                poly = union_geom.intersection(footprint)
            else:
                poly = Polygon()

            if poly.is_empty:
                poly = Polygon()

            self._mask_geometry[name] = poly
        return self._mask_geometry[name]

    def _get_mask_intersecting_geometries(self, name):
        """

        :param name: str(eg land)
        :return:
        """
        if self._mask_intersecting_geometries[name] is None:
            gseries = gpd.GeoSeries(self._get_mask_feature(name).geometries())
            # gseries = gpd.GeoSeries(self._get_mask_feature(name)
            #                         .intersecting_geometries(self.footprint))
            if len(gseries) == 0:
                # no intersection with mask, but we want at least one geometry in the serie (an empty one)
                gseries = gpd.GeoSeries([Polygon()])
            self._mask_intersecting_geometries[name] = gseries
        return self._mask_intersecting_geometries[name]

    @property
    @abstractmethod
    def footprint(self):
        pass

    def _continuous_longitude(self, longitude):
        """
        Set `self.cross_antimeridian` from the longitudes of the geolocation grid, as given
        by the product ([-180, 180]), and return them in a continuous range.

        This has to be done once, when the geolocation grid is read: everything derived from
        this grid (footprint, approx_transform, interpolators) needs continuous longitudes.

        Parameters
        ----------
        longitude: xarray.DataArray or numpy.ndarray
            longitudes of the geolocation grid, in [-180, 180] range

        Returns
        -------
        xarray.DataArray or numpy.ndarray
            longitudes in [0, 360] range if the footprint cross antimeridian, unchanged otherwise.
        """
        self._cross_antimeridian = ((np.max(longitude) - np.min(longitude)) > 180).item()
        if self._cross_antimeridian:
            attrs = getattr(longitude, "attrs", {})
            longitude = longitude % 360
            if attrs:
                longitude.attrs.update(attrs)
        return longitude

    @property
    def cross_antimeridian(self):
        """True if footprint cross antimeridian"""
        if self._cross_antimeridian is None:
            # reading the geolocation grid sets the flag (see `_continuous_longitude`)
            geoloc = self.geoloc
            if self._cross_antimeridian is None:
                self._continuous_longitude(geoloc["longitude"])
        return self._cross_antimeridian

    @property
    def swath(self):
        """string like 'EW', 'IW', 'WV', etc ..."""
        return self.manifest_attrs["swath_type"]

    @property
    @abstractmethod
    def _dict_coords2ll(self):
        pass

    @property
    @abstractmethod
    def approx_transform(self):
        pass

    @property
    def mask_names(self):
        """

        Returns
        -------
        list of str
            mask names
        """
        return self._mask_features.keys()

    def coords2ll(self, *args, to_grid=False, approx=False):
        """
        convert `lines`, `samples` arrays to `longitude` and `latitude` arrays.
        or a shapely object in `lines`, `samples` coordinates to `longitude` and `latitude`.

        Parameters
        ----------
        *args: lines, samples  or a shapely geometry
            lines, samples are iterables or scalar

        to_grid: bool, default False
            If True, `lines` and `samples` must be 1D arrays. The results will be 2D array of shape (lines.size, samples.size).

        Returns
        -------
        tuple of np.array or tuple of float
            (longitude, latitude) , with shape depending on `to_grid` keyword.

        Notes
        -----
        longitudes are in [-180, 180] range. For a shapely object, if `self.cross_antimeridian` is True,
        longitudes are in the continuous [0, 360] range (a geometry can't jump from 180 to -180).

        See Also
        --------
        xsar.BaseMeta.ll2coords
        xsar.BaseDataset.ll2coords

        """
        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self._coords2ll_shapely(args[0])

        lines, samples = args

        scalar = True
        if hasattr(lines, "__iter__"):
            scalar = False

        if approx:
            if to_grid:
                samples2D, lines2D = np.meshgrid(samples, lines)
                lon, lat = self.approx_transform * (lines2D, samples2D)
                pass
            else:
                lon, lat = self.approx_transform * (lines, samples)
        else:
            dict_coords2ll = self._dict_coords2ll
            if to_grid:
                lon = dict_coords2ll["longitude"](lines, samples)
                lat = dict_coords2ll["latitude"](lines, samples)
            else:
                lon = dict_coords2ll["longitude"].ev(lines, samples)
                lat = dict_coords2ll["latitude"].ev(lines, samples)

        if self.cross_antimeridian:
            # go back to [-180, 180]
            lon = to_lon180(lon)

        if scalar and hasattr(lon, "__iter__"):
            lon = lon.item()
            lat = lat.item()

        if hasattr(lon, "__iter__") and type(lon) is not type(lines):
            lon = type(lines)(lon)
            lat = type(lines)(lat)

        return lon, lat

    def _ll2coords_shapely(self, shape, approx=False):
        if approx:
            (xoff, a, b, yoff, d, e) = (~self.approx_transform).to_gdal()
            return shapely.affinity.affine_transform(shape, (a, b, d, e, xoff, yoff))
        else:
            return shapely.ops.transform(self.ll2coords, shape)

    def _coords2ll_shapely(self, shape, approx=False):
        if approx:
            (xoff, a, b, yoff, d, e) = self.approx_transform.to_gdal()
            return shapely.affinity.affine_transform(shape, (a, b, d, e, xoff, yoff))
        else:

            def coords2ll_continuous(lines, samples):
                lon, lat = self.coords2ll(np.asarray(lines), np.asarray(samples))
                if self.cross_antimeridian:
                    # keep the geometry in one piece
                    lon = lon % 360
                return lon, lat

            return shapely.ops.transform(coords2ll_continuous, shape)

    def ll2coords(self, *args):
        """
        Get `(lines, samples)` from `(lon, lat)`,
        or convert a lon/lat shapely object to line/sample coordinates.

        Parameters
        ----------
        *args: lon, lat or shapely object
            lon and lat might be iterables or scalars

        Returns
        -------
        tuple of np.array or tuple of float (lines, samples) , or a shapely object

        Examples
        --------
            get nearest (line,sample) from (lon,lat) = (84.81, 21.32) in ds, without bounds checks

            >>> (line, sample) = self.ll2coords(84.81, 21.32)  # (lon, lat)
            >>> (line, sample)
            (9752.766349989339, 17852.571322887554)

        See Also
        --------
        xsar.BaseMeta.coords2ll
        xsar.BaseDataset.coords2ll

        """

        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self._ll2coords_shapely(args[0])

        lon, lat = args
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        
        cross_antimeridian = self.cross_antimeridian
        if cross_antimeridian:
            # define approx transform continuous longitudes
            lon = lon % 360

        # approximation with global inaccurate transform
        line_approx, sample_approx = ~self.approx_transform * (
            lon,
            lat
        )

        # Theoretical identity. It should be the same, but the difference show the error.
        lon_identity, lat_identity = self.coords2ll(
            line_approx, sample_approx, to_grid=False
        )
        if cross_antimeridian:
            lon_identity = lon_identity % 360
        line_identity, sample_identity = ~self.approx_transform * (
            lon_identity,
            lat_identity,
        )

        # we are now able to compute the error, and make a correction
        line_error = line_identity - line_approx
        sample_error = sample_identity - sample_approx

        line = line_approx - line_error
        sample = sample_approx - sample_error

        return line, sample

    def coords2heading(self, lines, samples, to_grid=False, approx=True):
        """
        Get image heading (lines increasing direction) at coords `lines`, `samples`.

        Parameters
        ----------
        lines: np.array or scalar
        samples: np.array or scalar
        to_grid: bool
            If True, `lines` and `samples` must be 1D arrays. The results will be 2D array of shape (lines.size, samples.size).

        Returns
        -------
        np.array or float
            `heading` , with shape depending on `to_grid` keyword.

        """

        lon1, lat1 = self.coords2ll(
            lines - 1, samples, to_grid=to_grid, approx=approx)
        lon2, lat2 = self.coords2ll(
            lines + 1, samples, to_grid=to_grid, approx=approx)
        _, heading = haversine(lon1, lat1, lon2, lat2)
        return heading

    @property
    @abstractmethod
    def _get_time_range(self):
        pass

    @property
    def time_range(self):
        """time range as pd.Interval"""
        if self._time_range is None:
            self._time_range = self._get_time_range()
        return self._time_range

    @property
    def start_date(self):
        """start date, as datetime.datetime"""
        out_format = "%Y-%m-%d %H:%M:%S.%f"
        date = self.time_range.left
        try:
            return "%s" % datetime.strptime("%s" % date, out_format)
        except ValueError:
            return "%s" % date.strftime(out_format)

    @property
    def stop_date(self):
        """stop date, as datetime.datetime"""
        out_format = "%Y-%m-%d %H:%M:%S.%f"
        date = self.time_range.right
        try:
            return "%s" % datetime.strptime("%s" % date, out_format)
        except ValueError:
            return "%s" % date.strftime(out_format)

    @class_or_instancemethod
    def set_raster(self_or_cls, name, resource, read_function=None, get_function=None):
        # get defaults if exists
        default = available_rasters.loc[name:name]

        # set from params, or from default
        self_or_cls.rasters.loc[name, "resource"] = (
            resource or default.loc[name, "resource"]
        )
        self_or_cls.rasters.loc[name, "read_function"] = (
            read_function or default.loc[name, "read_function"]
        )
        self_or_cls.rasters.loc[name, "get_function"] = (
            get_function or default.loc[name, "get_function"]
        )

        return

    @property
    def dict(self):
        # return a minimal dictionary that can be used with Sentinel1Meta.from_dict() or pickle (see __reduce__)
        # to reconstruct another instance of self
        #
        minidict = {
            "name": self.name,
            "_mask_features_raw": self._mask_features_raw,
            "_mask_features": {},
            "_mask_intersecting_geometries": {},
            "_mask_geometry": {},
            "rasters": self.rasters,
        }
        for name in minidict["_mask_features_raw"].keys():
            minidict["_mask_intersecting_geometries"][name] = None
            minidict["_mask_geometry"][name] = None
            minidict["_mask_features"][name] = None
        return minidict

    @classmethod
    def from_dict(cls, minidict):
        # like copy constructor, but take a dict from Sentinel1Meta.dict
        # https://github.com/umr-lops/xsar/issues/23
        for name in minidict["_mask_features_raw"].keys():
            assert minidict["_mask_geometry"][name] is None
            assert minidict["_mask_features"][name] is None
        minidict = copy.copy(minidict)
        new = cls(minidict["name"])
        new.__dict__.update(minidict)
        return new
