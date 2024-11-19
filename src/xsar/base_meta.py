import copy
import logging
import warnings

import cartopy
import rasterio
import shapely
from shapely.geometry import Polygon
import numpy as np
from datetime import datetime

from abc import abstractmethod

from .raster_readers import available_rasters
from .base_dataset import BaseDataset
import geopandas as gpd

from .utils import class_or_instancemethod, to_lon180, haversine

logger = logging.getLogger("xsar.base_meta")
logger.addHandler(logging.NullHandler())

# we know tiff as no geotransform : ignore warning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# allow nan without warnings
# some dask warnings are still non filtered: https://github.com/dask/dask/issues/3245
np.errstate(invalid="ignore")


class BaseMeta(BaseDataset):
    """
    Abstract class that defines necessary common functions for the computation of different SAR metadata
    (Radarsat2, Sentinel1, RCM...).
    This also permit a better maintenance, because these functions aren't redefined many times.
    """

    # default mask feature (see self.set_mask_feature and cls.set_mask_feature)
    _mask_features_raw = {
        "land": cartopy.feature.NaturalEarthFeature("physical", "land", "10m")
    }

    _mask_features = {}
    _mask_intersecting_geometries = {}
    _mask_geometry = {}
    _geoloc = None
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
        # internal method that returns a cartopy feature from a mask name
        if self._mask_features[name] is None:
            feature = self._mask_features_raw[name]
            if isinstance(feature, str):
                # feature is a shapefile.
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
                footprint_crs = transform(proj_transform, self.footprint)

                with warnings.catch_warnings():
                    # ignore "RuntimeWarning: Sequential read of iterator was interrupted. Resetting iterator."
                    warnings.simplefilter("ignore", RuntimeWarning)
                    feature = cartopy.feature.ShapelyFeature(
                        gpd.read_file(feature, mask=footprint_crs)
                        .to_crs(epsg=4326)
                        .geometry,
                        cartopy.crs.PlateCarree(),
                    )
            if not isinstance(feature, cartopy.feature.Feature):
                raise TypeError("Expected a cartopy.feature.Feature type")
            self._mask_features[name] = feature

        return self._mask_features[name]

    @class_or_instancemethod
    def set_mask_feature(self_or_cls, name, feature):
        """
        Set a named mask from a shapefile or a cartopy feature.

        Parameters
        ----------
        name: str
            mask name
        feature: str or cartopy.feature.Feature
            if str, feature is a path to a shapefile or whatever file readable with fiona.
            It is recommended to use str, as the serialization of cartopy feature might be big.

        Examples
        --------
            Add an 'ocean' mask at class level (ie as default mask):
            ```
            >>> xsar.RadarSat2Meta.set_mask_feature("ocean", cartopy.feature.OCEAN)
            >>> xsar.Sentinel1Meta.set_mask_feature("ocean", cartopy.feature.OCEAN)
            ```

            Add an 'ocean' mask at instance level (ie only for this self Sentinel1Meta (or RadarSat2Meta instance):
            ```
            >>> xsar.RadarSat2Meta.set_mask_feature("ocean", cartopy.feature.OCEAN)
            >>> xsar.Sentinel1Meta.set_mask_feature("ocean", cartopy.feature.OCEAN)
            ```


            High resoltion shapefiles can be found from openstreetmap.
            It is recommended to use WGS84 with large polygons split from https://osmdata.openstreetmap.de/

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
            descr = self._mask_features_raw[name]
            try:
                # nice repr for a class (like 'cartopy.feature.NaturalEarthFeature land')
                descr = "%s.%s %s" % (
                    descr.__module__,
                    descr.__class__.__name__,
                    descr.name,
                )
            except AttributeError:
                pass
            return descr

        if self._mask_geometry[name] is None:
            if self._get_mask_intersecting_geometries(name).unary_union:
                poly = self._get_mask_intersecting_geometries(
                    name
                ).unary_union.intersection(self.footprint)

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

    @property
    def cross_antemeridian(self):
        """True if footprint cross antemeridian"""
        return (
            (np.max(self.geoloc["longitude"]) - np.min(self.geoloc["longitude"])) > 180
        ).item()

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

        if self.cross_antemeridian:
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
            return shapely.ops.transform(self.coords2ll, shape)

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

        # approximation with global inaccurate transform
        line_approx, sample_approx = ~self.approx_transform * (
            np.asarray(lon),
            np.asarray(lat),
        )

        # Theoretical identity. It should be the same, but the difference show the error.
        lon_identity, lat_identity = self.coords2ll(
            line_approx, sample_approx, to_grid=False
        )
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

        lon1, lat1 = self.coords2ll(lines - 1, samples, to_grid=to_grid, approx=approx)
        lon2, lat2 = self.coords2ll(lines + 1, samples, to_grid=to_grid, approx=approx)
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
