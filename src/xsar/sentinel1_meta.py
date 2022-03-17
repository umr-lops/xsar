# -*- coding: utf-8 -*-
import cartopy.feature
import logging
import warnings
import copy
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import rasterio
from scipy.interpolate import RectBivariateSpline,interp1d
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely
from shapely.geometry import box
from .utils import to_lon180, haversine, timing, class_or_instancemethod
from .raster_readers import available_rasters
from . import sentinel1_xml_mappings
from .xml_parser import XmlParser
from affine import Affine
import os
from datetime import datetime
from collections import OrderedDict
from .ipython_backends import repr_mimebundle

logger = logging.getLogger('xsar.sentinel1_meta')
logger.addHandler(logging.NullHandler())


class Sentinel1Meta:
    """
    Handle dataset metadata.
    A `xsar.Sentinel1Meta` object can be used with `xsar.open_dataset`,
    but it can be used as itself: it contains usefull attributes and methods.

    Parameters
    ----------
    name: str
        path or gdal identifier like `'SENTINEL1_DS:%s:WV_001' % path`

    """

    # default mask feature (see self.set_mask_feature and cls.set_mask_feature)
    _mask_features_raw = {
        'land': cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
    }

    rasters = available_rasters.iloc[0:0].copy()


    # class attributes are needed to fetch instance attribute (ie self.name) with dask actors
    # ref http://distributed.dask.org/en/stable/actors.html#access-attributes
    # FIXME: not needed if @property, so it might be a good thing to have getter for those attributes
    multidataset = None
    xml_parser = None
    name = None
    short_name = None
    safe = None
    path = None
    product = None
    manifest = None
    subdatasets = None
    dsid = None
    manifest_attrs = None


    @timing
    def __init__(self, name, _xml_parser=None):

        if _xml_parser is None:
            self.xml_parser = XmlParser(
                xpath_mappings=sentinel1_xml_mappings.xpath_mappings,
                compounds_vars=sentinel1_xml_mappings.compounds_vars,
                namespaces=sentinel1_xml_mappings.namespaces
            )
        else:
            self.xml_parser = _xml_parser

        if not name.startswith('SENTINEL1_DS:'):
            name = 'SENTINEL1_DS:%s:' % name
        self.name = name
        """Gdal dataset name"""
        name_parts = self.name.split(':')
        if len(name_parts) > 3:
            # windows might have semicolon in path ('c:\...')
            name_parts[1] = ':'.join(name_parts[1:-1])
            del name_parts[2:-1]
        name_parts[1] = os.path.basename(name_parts[1])
        self.short_name = ':'.join(name_parts)
        """Like name, but without path"""
        self.path = ':'.join(self.name.split(':')[1:-1])
        """Dataset path"""
        self.safe = os.path.basename(self.path)
        """Safe file name"""
        # there is no information on resolution 'F' 'H' or 'M' in the manifest, so we have to extract it from filename
        self.product = os.path.basename(self.path).split('_')[2]
        """Product type, like 'GRDH', 'SLC', etc .."""
        self.manifest = os.path.join(self.path, 'manifest.safe')
        self.manifest_attrs = self.xml_parser.get_compound_var(self.manifest, 'safe_attributes')
        self._safe_files = None
        self.multidataset = False
        """True if multi dataset"""
        self.subdatasets = []
        """Subdatasets list (empty if single dataset)"""
        datasets_names = list(self.safe_files['dsid'].sort_index().unique())
        if self.name.endswith(':') and len(datasets_names) == 1:
            self.name = datasets_names[0]
        self.dsid = self.name.split(':')[-1]
        """Dataset identifier (like 'WV_001', 'IW1', 'IW'), or empty string for multidataset"""
        if self.short_name.endswith(':'):
            self.short_name = self.short_name + self.dsid
        if self.files.empty:
            self.subdatasets = datasets_names
            self.multidataset = True

        self.platform = self.manifest_attrs['mission'] + self.manifest_attrs['satellite']
        """Mission platform"""
        self._gcps = None
        self._time_range = None
        self._mask_features_raw = {}
        self._mask_features = {}
        self._mask_intersecting_geometries = {}
        self._mask_geometry = {}

        # get defaults masks from class attribute
        for name, feature in self.__class__._mask_features_raw.items():
            self.set_mask_feature(name, feature)
        self._geoloc = None
        self.rasters = self.__class__.rasters.copy()
        """pandas dataframe for rasters (see `xsar.Sentinel1Meta.set_raster`)"""

    def __del__(self):
        logger.debug('__del__')

    def have_child(self, name):
        """
        Check if dataset `name` belong to this Sentinel1Meta object.

        Parameters
        ----------
        name: str
            dataset name

        Returns
        -------
        bool
        """
        return name == self.name or name in self.subdatasets

    def _get_gcps(self):
        rio = rasterio.open(self.files['measurement'].iloc[0])
        rio_gcps, crs = rio.get_gcps()

        gcps_xtracks, gcps_atracks = [
            np.array(arr) for arr in zip(
                *[(g.col, g.row) for g in rio_gcps]
            )
        ]

        gcps_xtracks, gcps_atracks = [np.array(arr) for arr in [gcps_xtracks, gcps_atracks]]

        (xtracks, xtracks_gcps_idx), (atracks, atracks_gcps_idx) = [
            np.unique(arr, return_inverse=True) for arr in [gcps_xtracks, gcps_atracks]
        ]

        # assert regularly gridded
        assert xtracks.size * atracks.size == len(rio_gcps)

        da_list = []

        # grid gcps index as an xarray(atrack,xtrack)
        np_gcps_idx = (xtracks_gcps_idx + atracks_gcps_idx * len(xtracks)).reshape(atracks.size, xtracks.size)
        gcps_idx = xr.DataArray(
            np_gcps_idx,
            coords={'atrack': atracks, 'xtrack': xtracks},
            dims=['atrack', 'xtrack'],
            name='index'
        )
        da_list.append(gcps_idx)

        # grid gcps with similar xarray
        np_gcps = np.array([rio_gcps[i] for i in np_gcps_idx.flat]).reshape(np_gcps_idx.shape)
        gcps = xr.DataArray(
            np_gcps,
            coords={'atrack': atracks, 'xtrack': xtracks},
            dims=['atrack', 'xtrack'],
            name='gcp'
        ).astype(object)
        da_list.append(gcps)

        # same for 'longitude', 'latitude', 'altitude'
        gcps_mappings = {
            'longitude': 'x',
            'latitude': 'y',
            'altitude': 'z'
        }

        for var_name, gcp_attr in gcps_mappings.items():
            np_arr = np.array([getattr(rio_gcps[i], gcp_attr) for i in np_gcps_idx.flat]).reshape(np_gcps_idx.shape)
            da_arr = xr.DataArray(
                np_arr,
                coords={'atrack': atracks, 'xtrack': xtracks},
                dims=['atrack', 'xtrack'],
                name=var_name
            )
            da_list.append(da_arr)

        gcps_ds = xr.merge(da_list)

        # add attributes

        attrs = gcps_ds.attrs

        # approx transform, from all gcps (inaccurate)
        approx_transform = rasterio.transform.from_gcps(rio_gcps)

        # affine parameters are swaped, to be compatible with xsar (atrack, xtrack) coordinates ordering
        attrs['approx_transform'] = approx_transform * Affine.permutation()

        footprint_dict = {}
        for ll in ['longitude', 'latitude']:
            footprint_dict[ll] = [
                gcps_ds[ll].isel(atrack=a, xtrack=x).values for a, x in [(0, 0), (0, -1), (-1, -1), (-1, 0)]
            ]
        # compute attributes (footprint, coverage, pixel_size)
        corners = list(zip(footprint_dict['longitude'], footprint_dict['latitude']))
        attrs['footprint'] = Polygon(corners)
        # compute acquisition size/resolution in meters
        # first vector is on xtrack
        acq_xtrack_meters, _ = haversine(*corners[0], *corners[1])
        # second vector is on atrack
        acq_atrack_meters, _ = haversine(*corners[1], *corners[2])
        pix_xtrack_meters = acq_xtrack_meters / rio.width
        pix_atrack_meters = acq_atrack_meters / rio.height
        attrs['coverage'] = "%dkm * %dkm (atrack * xtrack )" % (
            acq_atrack_meters / 1000, acq_xtrack_meters / 1000)
        attrs['pixel_xtrack_m'] = int(np.round(pix_xtrack_meters * 10)) / 10
        attrs['pixel_atrack_m'] = int(np.round(pix_atrack_meters * 10)) / 10

        gcps_ds.attrs = attrs
        return gcps_ds

    def _get_time_range(self):
        if self.multidataset:
            time_range = [self.manifest_attrs['start_date'], self.manifest_attrs['stop_date']]
        else:
            time_range = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.atrack_time_range')
        return pd.Interval(left=pd.Timestamp(time_range[0]), right=pd.Timestamp(time_range[-1]), closed='both')

    def to_dict(self, keys='minimal'):

        info_keys = {
            'minimal': ['ipf', 'platform', 'swath', 'product', 'pols']
        }
        info_keys['all'] = info_keys['minimal'] + ['name', 'start_date', 'stop_date', 'footprint', 'coverage',
                                                   'pixel_atrack_m', 'pixel_xtrack_m', 'orbit_pass', 'platform_heading']

        if isinstance(keys, str):
            keys = info_keys[keys]

        res_dict = {}
        for k in keys:
            if hasattr(self, k):
                res_dict[k] = getattr(self, k)
            elif k in self.manifest_attrs.keys():
                res_dict[k] = self.manifest_attrs[k]
            else:
                raise KeyError('Unable to find key/attr "%s" in Sentinel1Meta' % k)
        return res_dict

    @property
    def orbit_pass(self):
        """
        Orbit pass, i.e 'Ascending' or 'Descending'
        """

        if self.multidataset:
            return None  # not defined for multidataset

        return self.orbit.attrs['orbit_pass']

    @property
    def platform_heading(self):
        """
        Platform heading, relative to north
        """

        if self.multidataset:
            return None  # not defined for multidataset

        return self.orbit.attrs['platform_heading']

    @property
    def rio(self):
        raise DeprecationWarning(
            'Sentinel1Meta.rio is deprecated. '
            'Use `rasterio.open` on files in `Sentinel1Meta..files["measurement"] instead`'
        )

    @property
    def safe_files(self):
        """
        Files and polarizations for whole SAFE.
        The index is the file number, extracted from the filename.
        To get files in official SAFE order, the resulting dataframe should be sorted by polarization or index.

        Returns
        -------
        pandas.Dataframe
            with columns:
                * index         : file number, extracted from the filename.
                * dsid          : dataset id, compatible with gdal sentinel1 driver ('SENTINEL1_DS:/path/file.SAFE:WV_012')
                * polarization  : polarization name.
                * annotation    : xml annotation file.
                * calibration   : xml calibration file.
                * noise         : xml noise file.
                * measurement   : tiff measurement file.

        See Also
        --------
        xsar.Sentinel1Meta.files

        """
        if self._safe_files is None:
            files = self.xml_parser.get_compound_var(self.manifest, 'files')
            # add path
            for f in ['annotation', 'measurement', 'noise', 'calibration']:
                files[f] = files[f].map(lambda f: os.path.join(self.path, f))

            # set "polarization" as a category, so sorting dataframe on polarization
            # will return the dataframe in same order as self._safe_attributes['polarizations']
            files["polarization"] = files.polarization.astype('category').cat.reorder_categories(
                self.manifest_attrs['polarizations'], ordered=True)
            # replace 'dsid' with full path, compatible with gdal sentinel1 driver
            files['dsid'] = files['dsid'].map(lambda dsid: "SENTINEL1_DS:%s:%s" % (self.path, dsid))
            files.sort_values('polarization', inplace=True)
            self._safe_files = files
        return self._safe_files

    @property
    def files(self):
        """
        Files for current dataset. (Empty for multi datasets)

        See Also
        --------
        xsar.Sentinel1Meta.safe_files
        """
        return self.safe_files[self.safe_files['dsid'] == self.name]

    @property
    def gcps(self):
        """
        get gcps from rasterio.

        Returns
        -------
        xarray.DataArray
             xarray.DataArray with atracks/xtracks coordinates, and gcps as values.
             attrs is a dict with keys ['footprint', 'coverage', 'pixel_atrack_m', 'pixel_xtrack_m' ]
        """
        if self._gcps is None:
            self._gcps = self._get_gcps()
        return self._gcps

    @property
    def footprint(self):
        """footprint, as a shapely polygon or multi polygon"""
        if self.multidataset:
            return unary_union(self._footprints)
        return self.gcps.attrs['footprint']

    @property
    def geometry(self):
        """alias for footprint"""
        return self.footprint

    @property
    def geoloc(self):
        """
        xarray.Dataset with `['longitude', 'latitude', 'height', 'azimuth_time', 'slant_range_time','incidence','elevation' ]` variables
        and `['atrack', 'xtrack'] coordinates, at the geolocation grid
        """
        if self.multidataset:
            raise TypeError('geolocation_grid not available for multidataset')
        if self._geoloc is None:
            xml_annotation = self.files['annotation'].iloc[0]
            da_var_list = []
            for var_name in ['longitude', 'latitude', 'height', 'azimuth_time', 'slant_range_time','incidence','elevation']:
                # TODO: we should use dask.array.from_delayed so xml files are read on demand
                da_var = self.xml_parser.get_compound_var(xml_annotation, var_name)
                da_var.name = var_name
                da_var.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0],
                                                                                 var_name,
                                                                                 describe=True)
                da_var_list.append(da_var)

            self._geoloc = xr.merge(da_var_list)

            self._geoloc.attrs = {}
            # compute attributes (footprint, coverage, pixel_size)
            footprint_dict = {}
            for ll in ['longitude', 'latitude']:
                footprint_dict[ll] = [
                    self._geoloc[ll].isel(atrack=a, xtrack=x).values for a, x in [(0, 0), (0, -1), (-1, -1), (-1, 0)]
                ]
            corners = list(zip(footprint_dict['longitude'], footprint_dict['latitude']))
            p = Polygon(corners)
            logger.debug('polyon : %s', p)
            self._geoloc.attrs['footprint'] = p

        return self._geoloc

    @property
    def _footprints(self):
        """footprints as list. should len 1 for single meta, or len(self.subdatasets) for multi meta"""
        return self.manifest_attrs['footprints']

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
            >>> xsar.Sentinel1Meta.set_mask_feature('ocean', cartopy.feature.OCEAN)
            ```

            Add an 'ocean' mask at instance level (ie only for this self Sentinel1Meta instance):
            ```
            >>> xsar.Sentinel1Meta.set_mask_feature('ocean', cartopy.feature.OCEAN)
            ```


            High resoltion shapefiles can be found from openstreetmap.
            It is recommended to use WGS84 with large polygons split from https://osmdata.openstreetmap.de/

        See Also
        --------
        xsar.Sentinel1Meta.get_mask
        """

        # see https://stackoverflow.com/a/28238047/5988771 for self_or_cls

        self_or_cls._mask_features_raw[name] = feature

        if not isinstance(self_or_cls, type):
            # self (instance, not class)
            self_or_cls._mask_intersecting_geometries[name] = None
            self_or_cls._mask_geometry[name] = None
            self_or_cls._mask_features[name] = None


    @property
    def mask_names(self):
        """

        Returns
        -------
        list of str
            mask names
        """
        return self._mask_features.keys()

    @timing
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
                descr = '%s.%s %s' % (descr.__module__, descr.__class__.__name__ , descr.name)
            except AttributeError:
                pass
            return descr


        if self._mask_geometry[name] is None:
            poly = self._get_mask_intersecting_geometries(name)\
                .unary_union.intersection(self.footprint)

            if poly.is_empty:
                poly = Polygon()

            self._mask_geometry[name] = poly
        return self._mask_geometry[name]

    def _get_mask_intersecting_geometries(self, name):
        if self._mask_intersecting_geometries[name] is None:
            gseries = gpd.GeoSeries(self._get_mask_feature(name).intersecting_geometries(self.footprint.bounds))
            if len(gseries) == 0:
                # no intersection with mask, but we want at least one geometry in the serie (an empty one)
                gseries = gpd.GeoSeries([Polygon()])
            self._mask_intersecting_geometries[name] = gseries
        return self._mask_intersecting_geometries[name]

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
                        # proj6 give a " FutureWarning: '+init=<authority>:<code>' syntax is deprecated.
                        # '<authority>:<code>' is the preferred initialization method"
                        crs_in = fshp.crs['init']
                    except KeyError:
                        crs_in = fshp.crs
                    crs_in = pyproj.CRS(crs_in)
                proj_transform = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), crs_in, always_xy=True).transform
                footprint_crs = transform(proj_transform, self.footprint)

                with warnings.catch_warnings():
                    # ignore "RuntimeWarning: Sequential read of iterator was interrupted. Resetting iterator."
                    warnings.simplefilter("ignore", RuntimeWarning)
                    feature = cartopy.feature.ShapelyFeature(
                        gpd.read_file(feature, mask=footprint_crs).to_crs(epsg=4326).geometry,
                        cartopy.crs.PlateCarree()
                    )
            if not isinstance(feature, cartopy.feature.Feature):
                raise TypeError('Expected a cartopy.feature.Feature type')
            self._mask_features[name] = feature

        return self._mask_features[name]

    @class_or_instancemethod
    def set_raster(self_or_cls, name, resource, read_function=None, get_function=None):
        # get defaults if exists
        default = available_rasters.loc[name:name]

        # set from params, or from default
        self_or_cls.rasters.loc[name, 'resource'] = resource or default.loc[name, 'resource']
        self_or_cls.rasters.loc[name, 'read_function'] = read_function or default.loc[name, 'read_function']
        self_or_cls.rasters.loc[name, 'get_function'] = get_function or default.loc[name, 'get_function']

        return

    @property
    def coverage(self):
        """coverage, as a string like '251km * 170km (xtrack * atrack )'"""
        if self.multidataset:
            return None  # not defined for multidataset
        return self.gcps.attrs['coverage']

    @property
    def pixel_atrack_m(self):
        """pixel atrack spacing, in meters (at sensor level)"""
        k = '%s_%s' % (self.swath, self.product)
        if self.multidataset:
            return None  # not defined for multidataset
        return self.gcps.attrs['pixel_atrack_m']

    @property
    def pixel_xtrack_m(self):
        """pixel xtrack spacing, in meters (at sensor level)"""
        if self.multidataset:
            return None  # not defined for multidataset
        return self.gcps.attrs['pixel_xtrack_m']

    @property
    def time_range(self):
        """time range as pd.Interval"""
        if self._time_range is None:
            self._time_range = self._get_time_range()
        return self._time_range

    @property
    def start_date(self):
        """start date, as datetime.datetime"""
        return self.time_range.left

    @property
    def stop_date(self):
        """stort date, as datetime.datetime"""
        return self.time_range.right

    @property
    def denoised(self):
        """dict with pol as key, and bool as values (True is DN is predenoised at L1 level)"""
        if self.multidataset:
            return None  # not defined for multidataset
        else:
            return dict(
                [self.xml_parser.get_compound_var(f, 'denoised') for f in self.files['annotation']])

    @property
    def ipf(self):
        """ipf version"""
        return self.manifest_attrs['ipf_version']

    @property
    def swath(self):
        """string like 'EW', 'IW', 'WV', etc ..."""
        return self.manifest_attrs['swath_type']

    @property
    def pols(self):
        """polarisations strings, separated by spaces """
        return " ".join(self.manifest_attrs['polarizations'])

    @property
    def cross_antemeridian(self):
        """True if footprint cross antemeridian"""
        return ((np.max(self.gcps['longitude']) - np.min(self.gcps['longitude'])) > 180).item()

    @property
    def orbit(self):
        """
        orbit, as a geopandas.GeoDataFrame, with columns:
          - 'velocity' : shapely.geometry.Point with velocity in x, y, z direction
          - 'geometry' : shapely.geometry.Point with position in x, y, z direction

        crs is set to 'geocentric'

        attrs keys:
          - 'orbit_pass': 'Ascending' or 'Descending'
          - 'platform_heading': in degrees, relative to north

        Notes
        -----
        orbit is longer than the SAFE, because it belongs to all datatakes, not only this slice

        """
        if self.multidataset:
            return None  # not defined for multidataset
        gdf_orbit = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'orbit')
        gdf_orbit.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'orbit', describe=True)
        return gdf_orbit

    @property
    def image(self):
        if self.multidataset:
            return None
        img_dict = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'image')
        img_dict['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'image', describe=True)
        return img_dict

    @property
    def azimuth_fmrate(self):
        """
        xarray.Dataset
            Frequency Modulation rate annotations such as t0 (azimuth time reference) and polynomial coefficients: Azimuth FM rate = c0 + c1(tSR - t0) + c2(tSR - t0)^2
        """
        fmrates = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'azimuth_fmrate')
        fmrates.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'azimuth_fmrate', describe=True)
        return fmrates

    @property
    def _dict_coords2ll(self):
        """
        dict with keys ['longitude', 'latitude'] with interpolation function (RectBivariateSpline) as values.

        Examples:
        ---------
            get longitude at atrack=100 and xtrack=200:
            ```
            >>> self._dict_coords2ll['longitude'].ev(100,200)
            array(-66.43947434)
            ```
        Notes:
        ------
            if self.cross_antemeridian is True, 'longitude' will be in range [0, 360]
        """
        resdict = {}
        gcps = self.gcps
        if self.cross_antemeridian:
            gcps['longitude'] = gcps['longitude'] % 360

        idx_xtrack = np.array(gcps.xtrack)
        idx_atrack = np.array(gcps.atrack)

        for ll in ['longitude', 'latitude']:
            resdict[ll] = RectBivariateSpline(idx_atrack, idx_xtrack, np.asarray(gcps[ll]), kx=1, ky=1)

        return resdict

    def _coords2ll_shapely(self, shape, approx=False):
        if approx:
            (xoff, a, b, yoff, d, e) = self.approx_transform.to_gdal()
            return shapely.affinity.affine_transform(shape, (a, b, d, e, xoff, yoff))
        else:
            return shapely.ops.transform(self.coords2ll, shape)

    def _ll2coords_shapely(self, shape, approx=False):
        if approx:
            (xoff, a, b, yoff, d, e) = (~self.approx_transform).to_gdal()
            return shapely.affinity.affine_transform(shape, (a, b, d, e, xoff, yoff))
        else:
            return shapely.ops.transform(self.ll2coords, shape)

    def coords2ll(self, *args, to_grid=False, approx=False):
        """
        convert `atracks`, `xtracks` arrays to `longitude` and `latitude` arrays.
        or a shapely object in `atracks`, `xtracks` coordinates to `longitude` and `latitude`.

        Parameters
        ----------
        *args: atracks, xtracks  or a shapely geometry
            atracks, xtracks are iterables or scalar

        to_grid: bool, default False
            If True, `atracks` and `xtracks` must be 1D arrays. The results will be 2D array of shape (atracks.size, xtracks.size).

        Returns
        -------
        tuple of np.array or tuple of float
            (longitude, latitude) , with shape depending on `to_grid` keyword.

        See Also
        --------
        xsar.Sentinel1Meta.ll2coords
        xsar.Sentinel1Dataset.ll2coords

        """

        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self._coords2ll_shapely(args[0])

        atracks, xtracks = args

        scalar = True
        if hasattr(atracks, '__iter__'):
            scalar = False

        if approx:
            if to_grid:
                xtracks2D, atracks2D = np.meshgrid(xtracks, atracks)
                lon, lat = self.approx_transform * (atracks2D, xtracks2D)
                pass
            else:
                lon, lat = self.approx_transform * (atracks, xtracks)
        else:
            dict_coords2ll = self._dict_coords2ll
            if to_grid:
                lon = dict_coords2ll['longitude'](atracks, xtracks)
                lat = dict_coords2ll['latitude'](atracks, xtracks)
            else:
                lon = dict_coords2ll['longitude'].ev(atracks, xtracks)
                lat = dict_coords2ll['latitude'].ev(atracks, xtracks)

        if self.cross_antemeridian:
            lon = to_lon180(lon)

        if scalar and hasattr(lon, '__iter__'):
            lon = lon.item()
            lat = lat.item()

        if hasattr(lon, '__iter__') and type(lon) is not type(atracks):
            lon = type(atracks)(lon)
            lat = type(atracks)(lat)

        return lon, lat

    def ll2coords(self, *args, dataset=None):
        """
        Get `(atracks, xtracks)` from `(lon, lat)`,
        or convert a lon/lat shapely shapely object to atrack/xtrack coordinates.

        Parameters
        ----------
        *args: lon, lat or shapely object
            lon and lat might be iterables or scalars

        Returns
        -------
        tuple of np.array or tuple of float (atracks, xtracks) , or a shapely object

        Examples
        --------
            get nearest (atrack,xtrack) from (lon,lat) = (84.81, 21.32) in ds, without bounds checks

            >>> (atrack, xtrack) = meta.ll2coords(84.81, 21.32) # (lon, lat)
            >>> (atrack, xtrack)
            (9752.766349989339, 17852.571322887554)

        See Also
        --------
        xsar.Sentinel1Meta.coords2ll
        xsar.Sentinel1Dataset.coords2ll

        """

        if dataset is not None:
            ### FIXME remove deprecation
            warnings.warn("dataset kw is deprecated. See xsar.Sentinel1Dataset.ll2coords")

        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self._ll2coords_shapely(args[0])

        lon, lat = args

        # approximation with global inaccurate transform
        atrack_approx, xtrack_approx = ~self.approx_transform * (np.asarray(lon), np.asarray(lat))

        # Theoretical identity. It should be the same, but the difference show the error.
        lon_identity, lat_identity = self.coords2ll(atrack_approx, xtrack_approx, to_grid=False)
        atrack_identity, xtrack_identity = ~self.approx_transform * (lon_identity, lat_identity)

        # we are now able to compute the error, and make a correction
        atrack_error = atrack_identity - atrack_approx
        xtrack_error = xtrack_identity - xtrack_approx

        atrack = atrack_approx - atrack_error
        xtrack = xtrack_approx - xtrack_error

        if hasattr(lon, '__iter__'):
            scalar = False
        else:
            scalar = True

        if dataset is not None:
            # xtrack, atrack are float coordinates.
            # try to convert them to the nearest coordinates in dataset
            if not self.have_child(dataset.attrs['name']):
                raise ValueError("dataset %s is not a child of meta %s" % (dataset.attrs['name'], self.name))
            tolerance = np.max([np.percentile(np.diff(dataset[c].values), 90) / 2 for c in ['atrack', 'xtrack']]) + 1
            try:
                # select the nearest valid pixel in ds
                ds_nearest = dataset.sel(atrack=atrack, xtrack=xtrack, method='nearest', tolerance=tolerance)
                if scalar:
                    (atrack, xtrack) = (ds_nearest.atrack.values.item(), ds_nearest.xtrack.values.item())
                else:
                    (atrack, xtrack) = (ds_nearest.atrack.values, ds_nearest.xtrack.values)
            except KeyError:
                # out of bounds, because of `tolerance` keyword
                # if ds is resampled, tolerance should be computed from resolution/2 (for ex 5 for a resolution of 10)
                (atrack, xtrack) = (atrack * np.nan, xtrack * np.nan)

        return atrack, xtrack

    def coords2heading(self, atracks, xtracks, to_grid=False, approx=True):
        """
        Get image heading (atracks increasing direction) at coords `atracks`, `xtracks`.

        Parameters
        ----------
        atracks: np.array or scalar
        xtracks: np.array or scalar
        to_grid: bool
            If True, `atracks` and `xtracks` must be 1D arrays. The results will be 2D array of shape (atracks.size, xtracks.size).

        Returns
        -------
        np.array or float
            `heading` , with shape depending on `to_grid` keyword.

        """

        lon1, lat1 = self.coords2ll(atracks - 1, xtracks, to_grid=to_grid, approx=approx)
        lon2, lat2 = self.coords2ll(atracks + 1, xtracks, to_grid=to_grid, approx=approx)
        _, heading = haversine(lon1, lat1, lon2, lat2)
        return heading

    @property
    def _bursts(self):
        if self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.number_of_bursts') > 0:
            bursts = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'bursts')
            bursts.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'bursts', describe=True)
            return bursts
        else:
            # no burst, return empty dataset
            return xr.Dataset({'azimuthTime':('burst',[])})

    @property
    def approx_transform(self):
        """
        Affine transfom from gcps.

        This is an inaccurate transform, with errors up to 600 meters.
        But it's fast, and may fit some needs, because the error is stable localy.
        See `xsar.Sentinel1Meta.coords2ll` `xsar.Sentinel1Meta.ll2coords` for accurate methods.

        Examples
        --------
            get `longitude` and `latitude` from tuple `(atrack, xtrack)`:

            >>> longitude, latitude = self.approx_transform * (atrack, xtrack)

            get `atrack` and `xtrack` from tuple `(longitude, latitude)`

            >>> atrack, xtrack = ~self.approx_transform * (longitude, latitude)

        See Also
        --------
        xsar.Sentinel1Meta.coords2ll
        xsar.Sentinel1Meta.ll2coords`

        """
        return self.gcps.attrs['approx_transform']

    def __repr__(self):
        if self.multidataset:
            meta_type = "multi (%d)" % len(self.subdatasets)
        else:
            meta_type = "single"
        return "<Sentinel1Meta %s object>" % meta_type

    def _repr_mimebundle_(self, include=None, exclude=None):
        return repr_mimebundle(self, include=include, exclude=exclude)

    def __reduce__(self):
        # make self serializable with pickle
        # https://docs.python.org/3/library/pickle.html#object.__reduce__

        return self.__class__, (self.name,), self.dict

    @property
    def dict(self):
        # return a minimal dictionary that can be used with Sentinel1Meta.from_dict() or pickle (see __reduce__)
        # to reconstruct another instance of self
        #
        # TODO: find a way to get self.footprint and self.gcps. ( speed optimisation )
        minidict = {
            'name': self.name,
            '_mask_features_raw': self._mask_features_raw,
            '_mask_features': {},
            '_mask_intersecting_geometries': {},
            '_mask_geometry': {},
            'rasters': self.rasters
        }
        for name in minidict['_mask_features_raw'].keys():
            minidict['_mask_intersecting_geometries'][name] = None
            minidict['_mask_geometry'][name] = None
            minidict['_mask_features'][name] = None
        return minidict

    @classmethod
    def from_dict(cls, minidict):
        # like copy constructor, but take a dict from Sentinel1Meta.dict
        # https://github.com/umr-lops/xsar/issues/23
        for name in minidict['_mask_features_raw'].keys():
            assert minidict['_mask_geometry'][name] is None
            assert minidict['_mask_features'][name] is None
        minidict = copy.copy(minidict)
        new = cls(minidict['name'])
        new.__dict__.update(minidict)
        return new


    @property
    def _doppler_estimate(self):
        """
        xarray.Dataset
            with Doppler Centroid Estimates from annotations such as geo_polynom,data_polynom or frequency
        """
        dce = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'doppler_estimate')
        dce.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'doppler_estimate',
                                                                describe=True)
        return dce

    def burst_azitime(self, atrack_values,return_all=False):
        """
        Get azimuth time for bursts (TOPS SLC).
        To be used for locations of interpolation since line indices will not handle
        properly overlap.
        Parameters
        ----------
        atrack_values: np.array or scalar or xarray
        """
        # For consistency, azimuth time is derived from the one given in
        # geolocation grid (the one given in burst_list do not always perfectly
        # match).
        burst_nlines = self._bursts.attrs['lines_per_burst']
        azi_time_int = self.image['azimuth_time_interval']
        azi_time_int = np.timedelta64(int(azi_time_int*1e12),'ps') #turn this interval float/seconds into timedelta/picoseconds
        geoloc_line = self.geoloc['atrack'].values
        geoloc_iburst = np.floor(geoloc_line / float(burst_nlines)).astype('int32') # find the indice of the bursts in the geolocation grid
        iburst = np.floor(atrack_values / float(burst_nlines)).astype('int32') # find the indices of the bursts in the high resolution grid
        ind = np.searchsorted(geoloc_iburst, iburst, side='left') # find the indices of the burst transitions
        print('ind',ind)
        #n_pixels = int((self.geoloc.attrs['npixels'] - 1) / 2)
        n_pixels = int((len(self.geoloc['atrack']) - 1 ) / 2)
        geoloc_azitime = self.geoloc['azimuth_time'].values[:, n_pixels]
        if ind.max() >= len(geoloc_azitime): #security check for unrealistic atrack_values exceeding the image extent
            print('security ind',ind)
            ind[ind>=len(geoloc_azitime)] = len(geoloc_azitime)-1
        azitime = geoloc_azitime[ind] + (atrack_values - geoloc_line[ind]) * azi_time_int.astype('<m8[ns]') #compute the azimuth time by adding a step function (first term) and a growing term (second term)
        logger.debug('azitime %s %s %s',azitime,type(azitime),azitime.dtype)
        if return_all:
            return azitime,ind,geoloc_azitime[ind],geoloc_iburst
        else:
            return azitime
    def get_bursts_polygons(self,atracks, xtracks, name):
        """
        idea of this method is to prepare a geopandas containing boxes (shapely) for the 10 bursts of a subswath
        on which we perform interp1d to get the high resolution fields from geolocationGrid annotations.
        this method is meant to replace the map_block_coords() from utils.py used so far in the
        first implementation for SLC TOPS products
        the philosophy of this method is also to mimick method noise_lut_azi() in sentinel1_xml_mappings.py
        steps:
        1) get the indice of start and stop of each burst in axtrack low resolution geometry
        2) compute the 10 bursts interp functions+ area in image xtrack/atrack geometry
        3) store them in a geopandas
         Parameters
        ----------
        atracks : np.ndarray
            1D array of atrack coordinates at high resolution image
        xtracks: np.ndarray
            1D array of xtrack coordinates at high resolution image
        name: str
            name of the geoloc

        Returns
        -------
        geopandas.GeoDataframe
            noise range geometry.
            'geometry' is the polygon where 'variable_f' is defined.
            attrs['type'] set to 'xtrack'
        """
        if self._bursts['burst'].size==0:
            res = gpd.GeoDataFrame()
        else:
            geoloc_vars_LR = self.geoloc[name]
            if atracks.max()>self.image['shape'][0]:
                logger.warning('image max atrack coordinate is %s<%s',atracks.max(),self.image['shape'][0])
                logger.warning('xsar limit the computation to the image extent.')
                atracks = atracks[atracks<self.image['shape'][0]]
            if xtracks.max()>self.image['shape'][1]:
                logger.warning('image max xtrack coordinate is %s<%s',xtracks.max(),self.image['shape'][1])
                logger.warning('xsar limit the computation to the image extent.')
                xtracks = xtracks[xtracks < self.image['shape'][1]]
            class box_burst:
                def __init__(self, atrack_hr,xtrack_hr,a_start, a_stop, x, l):
                    """
                    atrack_hr : atrack values of the burst considered HR 1D
                    xtrack_hr : xtrack values of the burst considered HR 2D
                    a_start: atrack start burst aztime at low resolution (annotation geoloc grid)
                    a_stop: atrack stop burst aztime at low resolution (annotation geoloc grid)
                    x: geoloc['xtrack']  indices (1D vector) at low resolution (annotation geoloc grid)
                    l: values of one of the variable defined in the geolocation grid (xml annotations)
                    azimuth_time_lr : azimuth time at low resolution from geolocation grid (full matric 2D 10x20)
                    """
                    #self.atracks = np.arange(a_start, a_stop)
                    # interpolation needs to be done on azimuth time not on atrack (contrarily to the intersection of burst and box)
                    #self.azitimes = azimuth_time_lr[a_start:a_stop,x[0]:x[-1]]
                    self.azitimes = np.array([a_start.values,a_stop]).astype(float)
                    tiled_xtrack_lr = np.tile(x, (len(np.unique(inds_burst)), 1))
                    self.xtracks = tiled_xtrack_lr
                    self.area = box(atrack_hr[0], xtrack_hr[0], atrack_hr[-1], xtrack_hr[-1]) #keep atrack xtrack geometry
                    # self.variable_interp_f = interp1d(x, l, kind='linear', fill_value=np.nan, assume_sorted=True,
                    #                       bounds_error=False)
                    print('interp func def %s %s %s',self.azitimes.shape,x.shape,l.shape)
                    self.variable_interp_f = RectBivariateSpline(self.azitimes[:, np.newaxis],tiled_xtrack_lr[np.newaxis,:],l,kx=1,ky=1)
                def __call__(self, azitime, xtracks):
                    """
                    azitime: azimuth time 2D matrix at high resolution
                    xtracks:  azimuth time 2D matrix at high resolution
                    """
                    azaz,rara = np.meshgrid(azitime,xtracks)
                    varhr = self.variable_interp_f(azaz,rara,grid=False).T
                    return varhr

            bursts = []
            # atracks is where lut is defined. compute atracks interval validity
            #atracks_start = (atracks - np.diff(atracks, prepend=0) / 2).astype(int)
            #azimuth_time_start = (azimuth_time_LR-np.diff(azimuth_time_LR, prepend=0) / 2)
            azitime_hr, inds_burst, _, geoloc_iburst = self.burst_azitime(atracks,return_all=True)
            geoloc_iburst = geoloc_iburst.astype(int) #for indexing
            #azimuth_time_start_bursts = self.geoloc.azimuth_time[geoloc_iburst]
            azimuth_time_start_bursts = self._bursts.azimuthTime # #TODO check if it is same values than the one given by the variable start_burst_time below
            # atracks_stop = np.ceil(
            #     atracks + np.diff(atracks, append=atracks[-1] + 1) / 2
            # ).astype(int)  # end is not included in the interval
            #azimuth_time_stop_bursts = np.ceil(azimuth_time_LR + np.diff(azimuth_time_LR, append=azimuth_time_LR[-1] + 1) / 2)
            #azimuth_time_stop_bursts = self.geoloc.azimuth_time[geoloc_iburst+1]

            bursts_az_inds = {}
            end_burst_time = []
            start_burst_time = []
            burst_variable_matrix_lr = []
            print('go for burst definition',geoloc_iburst)
            for uu in np.unique(inds_burst):
                inds_one_val = np.where(inds_burst == uu)[0]
                bursts_az_inds[uu] = inds_one_val
                #mini_burst_azi = inds_one_val.min()
                #maxi_burst_azi = inds_one_val.max()
                end_burst_time.append(inds_one_val.max())
                start_burst_time.append(inds_one_val.min())
                logger.debug('geoloc_iburst %s',geoloc_iburst)
                burst_variable_matrix_lr.append(geoloc_vars_LR[geoloc_iburst[uu]:geoloc_iburst[uu]+2])
                logger.debug('variable shape lr: %s',len(burst_variable_matrix_lr))
            azimuth_time_stop_bursts = azitime_hr[np.array(end_burst_time)]
            logger.info('azimuth_time_stop_bursts %s',azimuth_time_stop_bursts.shape)
            logger.info('azimuth_time_start_bursts %s',azimuth_time_start_bursts.shape)
            #azimuth_time_start_bursts = azitime_hr[np.array(start_burst_time)]
            #atracks_stop[-1] = 65535  # be sure to include all image if last azimuth line, is not last azimuth image
            #TODO Fix this security to include the whole image
            cpt_burst = 0
            print("self.geoloc.xtrack,",self.geoloc.xtrack,)
            #tiled_xtrack_lr = np.tile(self.geoloc.xtrack.values, (len(np.unique(inds_burst)), 1))
            #print('tiled_xtrack_lr',tiled_xtrack_lr.shape)
            for a_start, a_stop, xx, l in zip(azimuth_time_start_bursts, azimuth_time_stop_bursts, self.geoloc.xtrack.values, burst_variable_matrix_lr):
                print('a_start ',a_start,type(a_start))
                variable_f = box_burst(bursts_az_inds[cpt_burst],xtracks,a_start, a_stop, xx, l)
                burst = pd.Series(dict([
                    ('variable_interp_f', variable_f),
                    ('geometry', variable_f.area)]))
                bursts.append(burst)
                cpt_burst += 1
            # to geopandas
            blocks = pd.concat(bursts, axis=1).T
            blocks = gpd.GeoDataFrame(blocks)
            res = _HRvariablesFromGeoloc(blocks)
        return res


    def extent_burst(self, burst, valid=True):
        """Get extent for a SAR image burst.
        copy pasted from sarimage.py ODL
        """
        nbursts = self._bursts['burst'].size
        if nbursts == 0:
            raise Exception('No bursts in SAR image')
        if burst < 0 or burst >= nbursts:
            raise Exception('Invalid burst index number')
        if valid is True:
            burst_list = self._bursts
            extent = np.copy(burst_list['valid_location'].values[burst, :])
        else:
            extent = self._extent_max()
            nlines = self._bursts.attrs['lines_per_burst']
            extent[0:3:2] = [nlines*burst, nlines*(burst+1)-1]
        return extent


    def _extent_max(self):
        """Get extent for the whole SAR image.
        copy/pasted from cerbere
        """
        return np.array((0, 0, self.number_of_lines-1, #TODO see whether it is still needed if gcp a set on integer index (instead of x.5 index)
                         self.number_of_samples-1))




class _HRvariablesFromGeoloc:
    """small internal class that return a function(atracks, xtracks) defined on all the image, from blocks in the image"""

    def __init__(self, bursts):
        """
        bursts defined in get_bursts_polygons()
        """
        self.bursts = bursts

    def __call__(self, atracks, xtracks,azimuthtimeHR):
        """
        atracks values in atrack HR on which I want a specific variable
        xtracks values in xtrack HR on which I want a specific variable
        azimuthtimeHR azimuth time at high resolution (from burst_azitime()) on the same atracks coordinates
        """
        """ return noise[a.size,x.size], by finding the intersection with bursts and calling the corresponding block.lut_f"""
        if len(self.bursts) == 0:
            return 1 # for GRD products
        else:
            # the array to be returned
            variable = xr.DataArray(
                np.ones((atracks.size, xtracks.size)) * np.nan,
                dims=('atrack', 'xtrack'),
                coords={'atrack': atracks, 'xtrack': xtracks}
            )
            # find bursts that intersects with asked_box
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # the box coordinates of the returned array
                # asked_box = box(max(0, atracks[0] - 0.5), max(0, xtracks[0] - 0.5), atracks[-1] + 0.5,
                #                 xtracks[-1] + 0.5)
                asked_box = box(max(0, atracks[0] - 0.5), max(0, xtracks[0] - 0.5), atracks[-1] + 0.5,
                                xtracks[-1] + 0.5)
                # set match_bursts as the non empty intersection with asked_box
                match_bursts = self.bursts.copy()
                match_bursts.geometry = self.bursts.geometry.intersection(asked_box)
                match_bursts = match_bursts[~match_bursts.is_empty]
            for i, block in match_bursts.iterrows():
                (sub_a_min, sub_x_min, sub_a_max, sub_x_max) = map(int, block.geometry.bounds)
                sub_a_final = atracks[(atracks >= sub_a_min) & (atracks <= sub_a_max)]
                sub_a = azimuthtimeHR[(atracks >= sub_a_min) & (atracks <= sub_a_max)]
                sub_x = xtracks[(xtracks >= sub_x_min) & (xtracks <= sub_x_max)]
                print('sub_a',sub_a.shape)
                print('sub_x ,',sub_x.shape)
                tmptmp = block.variable_interp_f(sub_a, sub_x)
                logger.debug('i %s tmptmp: %s',i,tmptmp.shape)
                variable.loc[dict(atrack=sub_a_final, xtrack=sub_x)] = tmptmp

        # values returned as np array
        return variable.values
