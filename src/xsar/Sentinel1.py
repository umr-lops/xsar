# -*- coding: utf-8 -*-
import cartopy.feature
import logging
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import dask
import rasterio
import re
from scipy.interpolate import RectBivariateSpline, interp1d
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely
from .utils import timing, to_lon180, haversine, map_blocks_coords, rioread, \
    rioread_fromfunction
from . import sentinel1_xml_mappings
from .xml_parser import XmlParser
from numpy import asarray
from affine import Affine
from functools import partial
import datetime
import os

logger = logging.getLogger('xsar.Sentinel1')
logger.addHandler(logging.NullHandler())

# we know tiff as no geotransform : ignore warning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def product_info(path, columns='minimal', include_multi=False, driver='GTiff', _xml_parser=None):
    """

    Parameters
    ----------
    path: str or iterable of str
        path or gdal url.
    columns: list of str or str, optional
        'minimal' by default: only include columns from attributes found in manifest.safe.
        Use 'spatial' to have 'time_range' and 'geometry'.
        Might be a list of properties from `xsar.SentinelMeta`
    include_multi: bool, optional
        False by default: don't include multi datasets

    Returns
    -------
    geopandas.GeoDataFrame
      One dataset per lines, with info as columns

    See Also
    --------
    `xsar.SentinelMeta`

    """

    info_keys = {
        'minimal': ['name', 'ipf', 'platform', 'swath', 'product', 'pols', 'meta']
    }
    info_keys['spatial'] = info_keys['minimal'] + ['time_range', 'geometry']

    if isinstance(columns, str):
        columns = info_keys[columns]

    # 'meta' column is not a SentinelMeta attribute
    real_cols = [c for c in columns if c != 'meta']
    add_cols = []
    if 'path' not in real_cols:
        add_cols.append('path')
    if 'dsid' not in real_cols:
        add_cols.append('dsid')

    def _meta2df(meta):
        df = pd.Series(data=meta.to_dict(add_cols + real_cols)).to_frame().T
        if 'meta' in columns:
            df['meta'] = meta
        return df

    if isinstance(path, str):
        path = [path]

    if _xml_parser is None:
        _xml_parser = XmlParser(
            xpath_mappings=sentinel1_xml_mappings.xpath_mappings,
            compounds_vars=sentinel1_xml_mappings.compounds_vars,
            namespaces=sentinel1_xml_mappings.namespaces)

    df_list = []
    for p in path:
        s1meta = SentinelMeta(p, driver=driver)
        if s1meta.multidataset and include_multi:
            df_list.append(_meta2df(s1meta))
        elif not s1meta.multidataset:
            df_list.append(_meta2df(s1meta))
        if s1meta.multidataset:
            for n in s1meta.subdatasets:
                s1meta = SentinelMeta(n, driver=driver)
                df_list.append(_meta2df(s1meta))
    df = pd.concat(df_list).reset_index(drop=True)
    if 'geometry' in df:
        df = gpd.GeoDataFrame(df).set_crs(epsg=4326)

    df = df.set_index(['path', 'dsid'], drop=False)
    if add_cols:
        df = df.drop(columns=add_cols)

    return df


# hardcoded sensor pixel spacing from https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/resolutions
# if key is not found in this dict, values must be computed from footprint and image size.
# (but values will differ from one image to another)
# values are [ atrack, xtrack ], in meters
_sensor_pixel_spacing = {
    'IW_GRDH': [10, 10],
    'IW_GRDM': [40, 40],
    'EW_GRDM': [40, 40],
    'EW_GRDH': [25, 25],
    'WV_GRDM': [25, 25],
}


class SentinelMeta:
    """
    Handle dataset metadata.
    A `xsar.SentinelMeta` object can be used with `xsar.open_dataset`,
    but it can be used as itself: it contain usefull attributes and methods.

    Parameters
    ----------
    name: str
        path or gdal identifier like `'SENTINEL1_DS:%s:WV_001' % path`

    Returns
    -------
    `xsar.SentinelMeta`

    """

    def __init__(self, name, xml_parser=None, driver='GTiff'):
        if xml_parser is None:
            xml_parser = XmlParser(
                xpath_mappings=sentinel1_xml_mappings.xpath_mappings,
                compounds_vars=sentinel1_xml_mappings.compounds_vars,
                namespaces=sentinel1_xml_mappings.namespaces)
        self.xml_parser = xml_parser
        self.driver = driver
        """GDAL driver used. ('auto' for SENTINEL1, or 'tiff')"""
        if not name.startswith('SENTINEL1_DS:'):
            name = 'SENTINEL1_DS:%s:' % name
        self.name = name
        """Gdal dataset name"""
        name_parts = self.name.split(':')
        name_parts[1] = os.path.basename(name_parts[1])
        self.short_name = ":".join(name_parts)
        """Like name, but without path"""
        self.path = self.name.split(':')[1]
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
        if self.files.empty:
            self.subdatasets = datasets_names
            self.multidataset = True
        self.dsid = self.name.split(':')[2]
        """Dataset identifier (like 'WV_001', 'IW1', 'IW'), or empty string for multidataset"""
        self.platform = self.manifest_attrs['mission'] + self.manifest_attrs['satellite']
        """Mission platform"""
        self._gcps = None
        self._time_range = None
        self._mask_features = {}
        self._mask_intersecting_geometries = {}
        self._mask_geometry = {}
        self.set_mask_feature('land', cartopy.feature.LAND)
        self._orbit_pass = None
        self._platform_heading = None

    def _get_gcps(self):
        rio_gcps, crs = self.rio.get_gcps()

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
        rio = self.rio
        pix_xtrack_meters = acq_xtrack_meters / rio.width
        pix_atrack_meters = acq_atrack_meters / rio.height
        attrs['coverage'] = "%dkm * %dkm (xtrack * atrack )" % (
            acq_xtrack_meters / 1000, acq_atrack_meters / 1000)
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
                raise KeyError('Unable to find key/attr "%s" in SentinelMeta' % k)
        return res_dict

    @property
    def orbit_pass(self):
        """Orbit pass, i.e 'Ascending' or 'Descending'"""

        if self.multidataset:
            return None  # not defined for multidataset
        if self._orbit_pass is None:
            self._orbit_pass = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.pass')
        return self._orbit_pass

    @property
    def platform_heading(self):
        """Platform heading, relative to north"""

        if self.multidataset:
            return None  # not defined for multidataset
        if self._platform_heading is None:
            self._platform_heading = self.xml_parser.get_var(self.files['annotation'].iloc[0],
                                                             'annotation.platform_heading')
        return self._platform_heading

    @property
    def rio(self):
        """
        get `rasterio.io.DatasetReader` object.
        This object is usefull to get image height, width, shape, etc...
        See rasterio doc for more infos.

        See Also
        --------
        `rasterio.io.DatasetReader <https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetReader>`_
        """

        # rio object not stored as self.rio attribute because of pickling problem
        # (https://github.com/dymaxionlabs/dask-rasterio/issues/3)
        if self.driver == 'GTiff':
            name = self.files['measurement'].iloc[0]
        else:
            name = self.name
        return rasterio.open(name)

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
        xsar.SentinelMeta.files

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
        xsar.SentinelMeta.safe_files
        """
        return self.safe_files[self.safe_files['dsid'] == self.name]

    @property
    def gcps(self):
        """
        get gcps from self.rio

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
    def _footprints(self):
        """footprints as list. should len 1 for single meta, or len(self.subdatasets) for multi meta"""
        return self.manifest_attrs['footprints']

    def set_mask_feature(self, name, feature):
        """
        Set a named mask from a shapefile or a cartopy feature.

        Parameters
        ----------
        name: str
            mask name
        feature: str or cartopy.feature.Feature
            if str, feature is a path to a shapefile.

        Examples
        --------
            Add an 'ocean' mask:
            ```
            >>> self.set_mask_feature('ocean', cartopy.feature.OCEAN)
            ```

        See Also
        --------
        xsar.SentinelMeta.get_mask
        """
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
        # reset variable cache
        self._mask_intersecting_geometries[name] = None
        self._mask_geometry[name] = None

    def get_mask(self, name):
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
        if self._mask_geometry[name] is None:
            self._mask_geometry[name] = self._get_mask_intersecting_geometries(name).unary_union.intersection(
                self.footprint)
        return self._mask_geometry[name]

    def _get_mask_intersecting_geometries(self, name):
        if self._mask_intersecting_geometries[name] is None:
            self._mask_intersecting_geometries[name] = gpd.GeoSeries(
                self._mask_features[name].intersecting_geometries(self.footprint.bounds))
        return self._mask_intersecting_geometries[name]

    @property
    def coverage(self):
        """coverage, as a string like '251km * 170km (xtrack * atrack )'"""
        if self.multidataset:
            return None  # not defined for multidataset
        return self.gcps.attrs['coverage']

    @property
    def pixel_atrack_m(self):
        """pixel atrack size, in meters (at sensor level)"""
        k = '%s_%s' % (self.swath, self.product)
        if k in _sensor_pixel_spacing:
            return _sensor_pixel_spacing[k][0]
        # hard-coded value not present, have to compute it from gcps
        if self.multidataset:
            return None  # not defined for multidataset
        return self.gcps.attrs['pixel_atrack_m']

    @property
    def pixel_xtrack_m(self):
        """pixel xtrack size, in meters (at sensor level)"""
        k = '%s_%s' % (self.swath, self.product)
        if k in _sensor_pixel_spacing:
            return _sensor_pixel_spacing[k][1]
        # hard-coded value not present, have to compute it from gcps
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
    def cross_antimeridian(self):
        """True if footprint cross antimeridian"""
        return (np.max(self.gcps['longitude']) - np.min(self.gcps['longitude'])) > 180

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
            if self.cross_antimeridian is True, 'longitude' will be in range [0, 360]
        """
        resdict = {}
        gcps = self.gcps
        if self.cross_antimeridian:
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

    def coords2ll(self, *args, to_grid=False):
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
        xsar.SentinelMeta.ll2coords

        """

        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self._coords2ll_shapely(args[0])

        atracks, xtracks = args

        scalar = True
        if hasattr(atracks, '__iter__'):
            scalar = False

        dict_coords2ll = self._dict_coords2ll
        if to_grid:
            lon = dict_coords2ll['longitude'](atracks, xtracks)
            lat = dict_coords2ll['latitude'](atracks, xtracks)
        else:
            lon = dict_coords2ll['longitude'].ev(atracks, xtracks)
            lat = dict_coords2ll['latitude'].ev(atracks, xtracks)

        if self.cross_antimeridian:
            lon = to_lon180(lon)

        if scalar and hasattr(lon, '__iter__'):
            lon = lon.item()
            lat = lat.item()

        return lon, lat

    def ll2coords(self, *args, dataset=None):
        """
        Get `(atracks, xtracks)` from `(lon, lat)`,
        or convert a lon/lat shapely shapely object to atrack/xtrack coordinates

        Parameters
        ----------
        *args: lon, lat or shapely object
            lon and lat might be iterables or scalars

        dataset: xsar dataset, or None (default)
            if a dataset is provided, it must be derived from the same meta object.
            (atracks, xtracks) will be the existing nearest coordinates in dataset, or np.nan if out of bounds


        Returns
        -------
        tuple of np.array or tuple of float (atracks, xtracks) , or a shapely object

        Examples
        --------
            get nearest (atrack,xtrack) from (lon,lat) = (84.81, 21.32) in ds, without bounds checks

            >>> (atrack, xtrack) = meta.ll2coords(84.81, 21.32) # (lon, lat)
            >>> (atrack, xtrack)
            (9752.766349989339, 17852.571322887554)

            same as above, but with bounds checks, and nearest coordinates in dataset
            (note that dataset and meta must have the same identifier)

            >>> (atrack, xtrack) = meta.ll2coords(84.81, 21.32, dataset=dataset) # (lon, lat)
            >>> (atrack, xtrack)
            (9752.5, 17852.5)  # those coordinates exists in dataset

        Notes
        ------
            if dataset is provided, and only one coordinate is out of bounds,
            **all** coordinates will be set to np.nan.

        See Also
        --------
        xsar.SentinelMeta.coords2ll

        """

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
            if dataset.attrs['name'] != self.name:
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

    def coords2heading(self, atracks, xtracks, to_grid=False):
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
        # FIXME use approx_transform (faster)
        lon1, lat1 = self.coords2ll(atracks - 1, xtracks, to_grid=to_grid)
        lon2, lat2 = self.coords2ll(atracks + 1, xtracks, to_grid=to_grid)
        _, heading = haversine(lon1, lat1, lon2, lat2)
        return heading

    @property
    def approx_transform(self):
        """
        Affine transfom from gcps.

        This is an inaccurate transform, with errors up to 600 meters.
        But it's fast, and may fit some needs, because the error is stable localy.
        See `xsar.SentinelMeta.coords2ll` `xsar.SentinelMeta.ll2coords` for accurate methods.

        Examples
        --------
            get `longitude` and `latitude` from tuple `(atrack, xtrack)`:

            >>> longitude, latitude = self.transform * (atrack, xtrack)

            get `atrack` and `xtrack` from tuple `(longitude, latitude)`

            >>> atrack, xtrack = ~self.transform * (longitude, latitude)

        See Also
        --------
        xsar.SentinelMeta.coords2ll
        xsar.SentinelMeta.ll2coords`

        """
        return self.gcps.attrs['approx_transform']

    def __str__(self):
        if self.multidataset:
            meta_type = "multi (%d)" % len(self.subdatasets)
        else:
            meta_type = "single"
        return "%s SentinelMeta object" % meta_type

    def _repr_mimebundle_(self, include=None, exclude=None):
        """html output for notebook"""
        import cartopy
        import geoviews as gv
        import geoviews.feature as gf
        import jinja2
        gv.extension('bokeh', logo=False)

        template = jinja2.Template(
            """
            <div align="left">
                <h5>{{ intro }}</h5>
                <table style="width:100%">
                    <thead>
                        <tr>
                            <th colspan="2">{{ short_name }}</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>
                                <table>
                                    {% for key, value in properties.items() %}
                                     <tr>
                                         <th> {{ key }} </th>
                                         <td> {{ value }} </td>
                                     </tr>
                                    {% endfor %}
                                </table>
                            </td>
                            <td>{{ location }}</td>
                        </tr>
                    </tbody>
                </table>
    
            </div>

            """
        )

        crs = cartopy.crs.PlateCarree()

        world = gv.operation.resample_geometry(gf.land.geoms('10m')).opts(color='khaki', projection=crs, alpha=0.5)

        center = self.footprint.centroid
        xlim = (center.x - 20, center.x + 20)
        ylim = (center.y - 20, center.y + 20)

        if self.multidataset and \
                len(self.subdatasets) == len(
            self._footprints):  # checks len because SAFEs like IW_SLC has only one footprint for 3 subdatasets
            dsid = [s.split(':')[2] for s in self.subdatasets]
            footprint = self._footprints
        else:
            dsid = [self.dsid]
            footprint = [self.footprint]

        footprints_df = gpd.GeoDataFrame(
            {
                'dsid': dsid,
                'geometry': footprint
            }
        )

        footprint = gv.Polygons(footprints_df).opts(projection=crs, xlim=xlim, ylim=ylim, alpha=0.5, tools=['hover'])

        location = (world * footprint).opts(width=400, height=400, title='Map')

        data, metadata = location._repr_mimebundle_(include=include, exclude=exclude)

        properties = self.to_dict()
        properties['orbit_pass'] = self.orbit_pass
        if self.pixel_atrack_m is not None:
            properties['pixel size'] = "%.1f * %.1f meters (xtrack * atrack)" % (
                self.pixel_xtrack_m, self.pixel_atrack_m)
        properties['coverage'] = self.coverage
        properties['start_date'] = self.start_date
        properties['stop_date'] = self.stop_date
        if len(self.subdatasets) > 0:
            properties['subdatasets'] = "list of %d subdatasets" % len(self.subdatasets)
        properties = {k: v for k, v in properties.items() if v is not None}

        if self.multidataset:
            intro = "Multi (%d) dataset" % len(self.subdatasets)
        else:
            intro = "Single dataset"
            properties['dsid'] = self.dsid

        if 'text/html' in data:
            data['text/html'] = template.render(
                intro=intro,
                short_name=self.short_name,
                properties=properties,
                location=data['text/html']
            )

        return data, metadata


class SentinelDataset:
    """
    Handle a SAFE subdataset.
    A dataset might contain several tiff files (multiples polarizations), but all tiff files must share the same footprint.

    The main attribute usefull to the end-user is `self.dataset` (`xarray.Dataset` , withl all variables parsed from xml and tiff files.)

    Parameters
    ----------
    dataset_id: str or SentinelMeta object
        if str, it can be a path, or a gdal dataset identifier like `'SENTINEL1_DS:%s:WV_001' % filename`)

        Note:SentinelMeta object or a full gdal string is mandatory if the SAFE has multiples subdatasets.
    resolution: None or dict
        see `xsar.open_dataset`
    resampling: rasterio.enums.Resampling
        see `xsar.open_dataset`
    luts: bool
        If True, `self.luts` will be merged in `self.dataset`. (False by default)
    pol_dim: bool,
        If True, self.dataset will not have 'pol' dimension. var names will be duplicated, ie 'sigma0_raw_vv' ans 'sigma0_raw_vh'.
        False by default.
    chunks: dict
        passed to `xarray.open_rasterio`.
    dtypes: dict
        Optional. Specify the variable dtypes. ex : dtypes = { 'latitude' : 'float32', 'nesz': 'float32' }

    Notes
    -----
    End user doesn't have to call this class, as it's done by `xsar.open_dataset`

    See Also
    --------
    xsar.open_dataset
    """

    def __init__(self, dataset_id, resolution=None,
                 resampling=rasterio.enums.Resampling.average,
                 luts=False, pol_dim=True, chunks=None,
                 dtypes=None):

        # default dtypes (TODO: find defaults, so science precision is not affected)
        self.dtypes = {
            'latitude': 'f4',
            'longitude': 'f4',
            'incidence': 'f4',
            'elevation': 'f4',
            'nesz': None,
            'negz': None,
            'sigma0_raw': None,
            'gamma0_raw': None,
            'noise_lut': 'f4',
            'noise_lut_range': 'f4',
            'noise_lut_azi': 'f4',
            'sigma0_lut': 'f8',
            'gamma0_lut': 'f8',
            'digital_number': None
        }
        if dtypes is not None:
            self.dtypes.update(dtypes)

        # max chunks size, in bytes (for dn dtype)
        self.block_size_limit = 2.5e8  # 250M

        # default meta for map_blocks output.
        # as asarray is imported from numpy, it's a numpy array.
        # but if later we decide to import asarray from cupy, il will be a cupy.array (gpu)
        self._default_meta = asarray([], dtype='f8')

        if not isinstance(dataset_id, SentinelMeta):
            xml_parser = XmlParser(
                xpath_mappings=sentinel1_xml_mappings.xpath_mappings,
                compounds_vars=sentinel1_xml_mappings.compounds_vars,
                namespaces=sentinel1_xml_mappings.namespaces)
            self.s1meta = SentinelMeta(dataset_id, xml_parser=xml_parser)
        else:
            self.s1meta = dataset_id

        if self.s1meta.multidataset:
            raise IndexError(
                """Can't open an multi-dataset. Use `xsar.SentinelMeta('%s').subdatasets` to show availables ones""" % self.s1meta.path
            )

        self._dataset = self.load_digital_number(resolution=resolution, resampling=resampling, chunks=chunks)

        # set time(atrack) from s1meta.time_range
        time_range = self.s1meta.time_range
        atrack_time = interp1d(self._dataset.atrack[[0, -1]],
                               [time_range.left.to_datetime64(), time_range.right.to_datetime64()])(
            self._dataset.atrack)
        self._dataset = self._dataset.assign(time=("atrack", pd.to_datetime(atrack_time)))

        # dataset no-pol template for function evaluation on coordinates (*no* values used)
        # what's matter here is the shape of the image, not the values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            self._da_tmpl = xr.DataArray(
                dask.array.empty_like(
                    self._dataset.digital_number.isel(pol=0).drop('pol'),
                    dtype=np.int8, name="empty_var_tmpl-%s" % dask.base.tokenize(self.s1meta.name)),
                dims=('atrack', 'xtrack'),
                coords={'atrack': self._dataset.digital_number.atrack,
                        'xtrack': self._dataset.digital_number.xtrack}
            )

        self._dataset.attrs.update(self.s1meta.to_dict("all"))

        # load subswath geometry
        # self.geometry = self.load_geometry(self.files['noise'].iloc[0])
        # self._dataset.attrs['geometry'] = self.geometry

        # dict mapping for variables names to create by applying specified lut on digital_number
        self._map_var_lut = {
            'sigma0_raw': 'sigma0_lut',
            'gamma0_raw': 'gamma0_lut',
        }

        # dict mapping for lut names to file type (from self.files columns)
        self._map_lut_files = {
            'sigma0_lut': 'calibration',
            'gamma0_lut': 'calibration',
            'noise_lut_range': 'noise',
            'noise_lut_azi': 'noise',
            'incidence': 'annotation',
            'elevation': 'annotation'
        }

        # dict mapping specifying if the variable has 'pol' dimension
        self._vars_with_pol = {
            'sigma0_lut': True,
            'gamma0_lut': True,
            'noise_lut_range': True,
            'noise_lut_azi': True,
            'incidence': False,
            'elevation': False
        }

        # variables not returned to the user (unless luts=True)
        self._hidden_vars = ['sigma0_lut', 'gamma0_lut', 'noise_lut', 'noise_lut_range', 'noise_lut_azi']

        self._luts = self.lazy_load_luts(self._map_lut_files.keys())

        # noise_lut is noise_lut_range * noise_lut_azi
        if 'noise_lut_range' in self._luts.keys() and 'noise_lut_azi' in self._luts.keys():
            self._luts = self._luts.assign(noise_lut=self._luts.noise_lut_range * self._luts.noise_lut_azi)

        lon_lat = self.load_lon_lat()

        ds_merge_list = [self._dataset, lon_lat] + [self._luts[v] for v in self._luts.keys() if
                                                    v not in self._hidden_vars]
        if luts:
            ds_merge_list.append(self._luts[self._hidden_vars])
        attrs = self._dataset.attrs
        self._dataset = xr.merge(ds_merge_list)
        self._dataset.attrs = attrs

        for var_name, lut_name in self._map_var_lut.items():
            if lut_name in self._luts:
                # merge var_name into dataset (not denoised)
                self._dataset = self._dataset.merge(self.apply_calibration_lut(var_name))
                # merge noise equivalent for var_name (named 'ne%sz' % var_name[0)
                self._dataset = self._dataset.merge(self.get_noise(var_name))
            else:
                logger.debug("Skipping variable '%s' ('%s' lut is missing)" % (var_name, lut_name))

        self._dataset = self.add_denoised(self._dataset)

        if not pol_dim:
            # remove 'pol' dim
            self.dataset = self._remove_pol_dim(self._dataset)
        else:
            self.dataset = self._dataset

    @timing
    def lazy_load_luts(self, luts_names):
        """
        lazy load luts from xml files
        Parameters
        ----------
        luts_names: list of str


        Returns
        -------
        xarray.Dataset with variables from `luts_names`.

        """

        luts_list = []
        # dask array template from digital_number (no pol)
        lut_tmpl = self._dataset.digital_number.isel(pol=0)

        for lut_name in luts_names:
            xml_type = self._map_lut_files[lut_name]
            xml_files = self.s1meta.files.copy()
            # polarization is a category. we use codes (ie index),
            # to have well ordered polarizations in latter combine_by_coords
            xml_files['pol_code'] = xml_files['polarization'].cat.codes
            xml_files = xml_files.set_index('pol_code')[xml_type]

            if not self._vars_with_pol[lut_name]:
                # luts are identical in all pols: take the fist one
                xml_files = xml_files.iloc[[0]]

            for pol_code, xml_file in xml_files.iteritems():
                pol = self.s1meta.files['polarization'].cat.categories[pol_code]
                if self._vars_with_pol[lut_name]:
                    name = "%s_%s" % (lut_name, pol)
                else:
                    name = lut_name
                lut_f_delayed = dask.delayed(self.s1meta.xml_parser.get_compound_var)(
                    xml_file, lut_name, dask_key_name="xml_%s-%s" % (name, dask.base.tokenize(xml_file)))
                # 1s faster, but lut_f will be loaded, even if not used
                # lut_f_delayed = lut_f_delayed.persist()

                lut = map_blocks_coords(self._da_tmpl.astype(self.dtypes[lut_name]), lut_f_delayed,
                                        name='blocks_%s' % name)

                # needs to add pol dim ?
                if self._vars_with_pol[lut_name]:
                    lut = lut.assign_coords(pol_code=pol_code).expand_dims('pol_code')

                lut = lut.to_dataset(name=lut_name)
                luts_list.append(lut)
            luts = xr.combine_by_coords(luts_list)

            # convert pol_code to string
            pols = self.s1meta.files['polarization'].cat.categories[luts.pol_code.values.tolist()]
            luts = luts.rename({'pol_code': 'pol'}).assign_coords({'pol': pols})
        return luts

    @timing
    def load_digital_number(self, resolution=None, chunks=None, resampling=rasterio.enums.Resampling.average):
        """
        load digital_number from self.s1meta.rio, as an `xarray.Dataset`.

        Parameters
        ----------
        resolution: None or dict
            see `xsar.open_dataset`
        resampling: rasterio.enums.Resampling
            see `xsar.open_dataset`

        Returns
        -------
        xarray.Dataset
            dataset (possibly dual-pol), with basic coords/dims naming convention
        """

        map_dims = {
            'pol': 'band',
            'atrack': 'y',
            'xtrack': 'x'
        }

        rio = self.s1meta.rio
        chunks['pol'] = 1
        # sort chunks keys like map_dims
        chunks = dict(sorted(chunks.items(), key=lambda pair: list(map_dims.keys()).index(pair[0])))
        chunks_rio = {map_dims[d]: chunks[d] for d in map_dims.keys()}

        if resolution is None:
            if self.s1meta.driver == 'auto':
                # using sentinel1 driver: all pols are read in one shot
                dn = xr.open_rasterio(
                    self.s1meta.name,
                    chunks=chunks_rio,
                    parse_coordinates=False)  # manual coordinates parsing because we're going to pad
            elif self.s1meta.driver == 'GTiff':
                # using tiff driver: need to read individual tiff and concat them
                # self.s1meta.files['measurement'] is ordered like self.s1meta.manifest_attrs['polarizations']
                dn = xr.concat(
                    [
                        xr.open_rasterio(
                            f, chunks=chunks_rio, parse_coordinates=False
                        ) for f in self.s1meta.files['measurement']
                    ], 'band'
                ).assign_coords(band=np.arange(len(self.s1meta.manifest_attrs['polarizations'])) + 1)
            else:
                raise NotImplementedError('Unhandled driver %s' % self.s1meta.driver)

            # set dimensions names
            dn = dn.rename(dict(zip(map_dims.values(), map_dims.keys())))

            # create coordinates from dimension index (because of parse_coordinates=False)
            # 0.5 is for pixel center (geotiff standard)
            dn = dn.assign_coords({'atrack': dn.atrack + 0.5, 'xtrack': dn.xtrack + 0.5})
        else:
            # resample the DN at gdal level, before feeding it to the dataset
            out_shape = (
                rio.height // resolution['atrack'],
                rio.width // resolution['xtrack']
            )
            if self.s1meta.driver == 'auto':
                out_shape_pol = (rio.count,) + out_shape
            else:
                out_shape_pol = (1,) + out_shape
            # both following methods produce same result,
            # but we can choose one of them by comparing out_shape and chunks size
            if all([b - s >= 0 for b, s in zip(chunks.values(), out_shape_pol)][1:]):
                # read resampled array in one chunk, and rechunk
                # winsize is the maximum full image size that can be divided  by resolution (int)
                winsize = (0, 0, rio.width // resolution['xtrack'] * resolution['xtrack'],
                           rio.height // resolution['atrack'] * resolution['atrack'])

                if self.s1meta.driver == 'GTiff':
                    resampled = [
                        xr.DataArray(
                            dask.array.from_delayed(
                                dask.delayed(rioread)(f, out_shape_pol, winsize, resampling=resampling),
                                out_shape_pol, dtype=np.dtype(rio.dtypes[0])
                            ),
                            dims = tuple(map_dims.keys()), coords = {'pol': [pol]}
                        ) for f, pol in
                        zip(self.s1meta.files['measurement'], self.s1meta.manifest_attrs['polarizations'])
                    ]
                    dn = xr.concat(resampled, 'pol').chunk(chunks)
                else:
                    resampled = dask.array.from_delayed(
                        dask.delayed(rioread)(self.s1meta.name, out_shape_pol, winsize, resampling=resampling),
                        out_shape_pol, dtype=np.dtype(rio.dtypes[0]))
                    dn = xr.DataArray(resampled, dims=tuple(map_dims.keys())).chunk(chunks)
            else:
                # read resampled array chunk by chunk
                # TODO: there is no way to specify dask graph name with fromfunction: => open github issue ?
                if self.s1meta.driver == 'GTiff':
                    resampled = [
                        xr.DataArray(
                            dask.array.fromfunction(
                                partial(rioread_fromfunction, f),
                                shape=out_shape_pol,
                                chunks=tuple(chunks.values()),
                                dtype=np.dtype(rio.dtypes[0]),
                                resolution=resolution, resampling=resampling
                            ),
                            dims=tuple(map_dims.keys()), coords={'pol': [pol]}
                        ) for f, pol in
                        zip(self.s1meta.files['measurement'], self.s1meta.manifest_attrs['polarizations'])
                    ]
                    dn = xr.concat(resampled, 'pol')
                else:
                    chunks['pol'] = 2
                    resampled = dask.array.fromfunction(
                        partial(rioread_fromfunction, self.s1meta.name),
                        shape=out_shape_pol,
                        chunks=tuple(chunks.values()),
                        dtype=np.dtype(rio.dtypes[0]),
                        resolution=resolution, resampling=resampling)
                    dn = xr.DataArray(resampled.rechunk({0: 1}), dims=tuple(map_dims.keys()))

            # create coordinates at box center (+0.5 for pixel center, -1 because last index not included)
            translate = Affine.translation((resolution['xtrack'] - 1) / 2 + 0.5, (resolution['atrack'] - 1) / 2 + 0.5)
            scale = Affine.scale(
                rio.width // resolution['xtrack'] * resolution['xtrack'] / out_shape[1],
                rio.height // resolution['atrack'] * resolution['atrack'] / out_shape[0])
            xtrack, _ = translate * scale * (dn.xtrack, 0)
            _, atrack = translate * scale * (0, dn.atrack)
            dn = dn.assign_coords({'atrack': atrack, 'xtrack': xtrack})

        if self.s1meta.driver == 'auto':
            # pols are ordered as self.rio.files,
            # and we may have to reorder them in the same order as the manifest
            dn_pols = [
                self.s1meta.xml_parser.get_var(f, "annotation.polarization")
                for f in rio.files
                if re.search(r'annotation.*\.xml', f)]
            dn = dn.assign_coords(pol=dn_pols)
            dn = dn.reindex(pol=self.s1meta.manifest_attrs['polarizations'])
        else:
            # for GTiff driver, pols are already ordered. just rename them
            dn = dn.assign_coords(pol=self.s1meta.manifest_attrs['polarizations'])

        dn.attrs = {}
        var_name = 'digital_number'
        ds = dn.to_dataset(name=var_name)

        astype = self.dtypes.get(var_name)
        if astype is not None:
            ds = ds.astype(self.dtypes[var_name])

        return ds

    @timing
    def load_lon_lat(self):
        """
        Load longitude and latitude using `self.s1meta.gcps`.

        Returns
        -------
        tuple xarray.Dataset
            xarray.Dataset:
                dataset with `longitude` and `latitude` variables, with same shape as mono-pol digital_number.
        """

        def coords2ll(*args):
            # *args[1:] to skip dummy 'll' dimension
            return np.stack(self.s1meta.coords2ll(*args[1:], to_grid=True))

        ll_coords = ['longitude', 'latitude']
        # ll_tmpl is like self._da_tmpl stacked 2 times (for both longitude and latitude)
        ll_tmpl = self._da_tmpl.expand_dims({'ll': 2}).assign_coords(ll=ll_coords).astype(self.dtypes['longitude'])
        ll_ds = map_blocks_coords(ll_tmpl, coords2ll, name='blocks_lonlat')
        # remove ll_coords to have two separate variables longitude and latitude
        ll_ds = xr.merge([ll_ds.sel(ll=ll).drop('ll').rename(ll) for ll in ll_coords])

        return ll_ds

    def get_lut(self, var_name):
        """
        Get lut for `var_name`

        Parameters
        ----------
        var_name: str

        Returns
        -------
        xarray.Dataarray
            lut for `var_name`
        """
        try:
            lut_name = self._map_var_lut[var_name]
        except KeyError:
            raise ValueError("can't find lut name for var '%s'" % var_name)
        try:
            lut = self._luts[lut_name]
        except KeyError:
            raise ValueError("can't find lut from name '%s' for variable '%s' " % (lut_name, var_name))
        return lut

    def apply_calibration_lut(self, var_name):
        """
        Apply calibration lut to `digital_number` to compute `var_name`.
        see https://sentinel.esa.int/web/sentinel/radiometric-calibration-of-level-1-products

        Parameters
        ----------
        var_name: str
            Variable name to compute by applying lut. Must exist in `self._map_var_lut`` to be able to get the corresponding lut.

        Returns
        -------
        xarray.Dataset
            with one variable named by `var_name`
        """
        lut = self.get_lut(var_name)
        res = (np.abs(self._dataset.digital_number) ** 2. / (lut ** 2))
        # dn default value is 0: convert to Nan
        res = res.where(res > 0)
        astype = self.dtypes.get(var_name)
        if astype is not None:
            res = res.astype(astype)
        return res.to_dataset(name=var_name)

    def reverse_calibration_lut(self, ds_var):
        """
        TODO: replace ds_var by var_name
        Inverse of `apply_calibration_lut` : from `var_name`, reverse apply lut, to get digital_number.
        See `official ESA documentation <https://sentinel.esa.int/web/sentinel/radiometric-calibration-of-level-1-products>`_ .
        > Level-1 products provide four calibration Look Up Tables (LUTs) to produce ß0i, σ0i and γi
        > or to return to the Digital Number (DN)

        A warning message may be issued if original complex 'digital_number' is converted to module during this operation.

        Parameters
        ----------
        ds_var: xarray.Dataset
            with only one variable name that must exist in `self._map_var_lut` to be able to reverse the lut to get digital_number

        Returns
        -------
        xarray.Dataset
            with one variable named 'digital_number'.
        """
        var_names = list(ds_var.keys())
        assert len(var_names) == 1
        var_name = var_names[0]
        if var_name not in self._map_var_lut:
            raise ValueError(
                "Unable to find lut for var '%s'. Allowed : %s" % (var_name, str(self._map_var_lut.keys())))
        da_var = ds_var[var_name]
        lut = self._luts[self._map_var_lut[var_name]]

        # resize lut with same a/xtrack as da_var
        lut = lut.sel(atrack=da_var.atrack, xtrack=da_var.xtrack, method='nearest')
        # as we used 'nearest', force exact coords
        lut['atrack'] = da_var.atrack
        lut['xtrack'] = da_var.xtrack
        # waiting for https://github.com/pydata/xarray/pull/4155
        # lut = lut.interp(atrack=da_var.atrack, xtrack=da_var.xtrack)

        # revert lut to get dn
        dn = np.sqrt(da_var * lut ** 2)

        if self._dataset.digital_number.dtype == np.complex and dn.dtype != np.complex:
            warnings.warn(
                "Unable to retrieve 'digital_number' as dtype '%s'. Fallback to '%s'"
                % (str(self._dataset.digital_number.dtype), str(dn.dtype))
            )

        name = 'digital_number'
        ds = dn.to_dataset(name=name)

        return ds

    def get_noise(self, var_name):
        """
        Get noise equivalent for  `var_name`.
        see https://sentinel.esa.int/web/sentinel/radiometric-calibration-of-level-1-products

        Parameters
        ----------
        var_name: str
            Variable name to compute. Must exist in `self._map_var_lut` to be able to get the corresponding lut.

        Returns
        -------
        xarray.Dataset
            with one variable named by `'ne%sz' % var_name[0]` (ie 'nesz' for 'sigma0', 'nebz' for 'beta0', etc...)
        """
        noise_lut = self._luts['noise_lut']
        lut = self.get_lut(var_name)
        dataarr = noise_lut / lut ** 2
        name = 'ne%sz' % var_name[0]
        astype = self.dtypes.get(name)
        if astype is not None:
            dataarr = dataarr.astype(astype)
        return dataarr.to_dataset(name=name)

    @timing
    def _remove_pol_dim(self, ds):
        """
        remove 'pol' dim from `ds`, by renaming vars with polarisation name
        Returns
        -------
        xarray.Dataset
            dataset with 'pol' dimension removed, and correponding variables renamed

        """
        ds_no_pol = xr.Dataset()
        ds_no_pol.attrs = ds.attrs
        for var in ds:
            if 'pol' in ds[var].dims:
                for pol in list(ds.pol.data):
                    ds_no_pol = ds_no_pol.merge(
                        ds[var].sel(pol=pol).to_dataset(name='%s_%s' % (var, pol.lower())).drop('pol'))
            else:
                # no pol dim : simple copy
                ds_no_pol = ds_no_pol.merge(ds[var])
        return ds_no_pol

    def add_denoised(self, ds, clip=True, vars=None):
        """add denoised vars to dataset

        Parameters
        ----------
        ds : xarray.DataSet
            dataset with non denoised vars, named `%s_raw`.
        clip : bool, optional
            If True, negative signal will be clipped to 0. (default to True )
        vars : list, optional
            variables names to add, by default `['sigma0' , 'beta0' , 'gamma0']`

        Returns
        -------
        xarray.DataSet
            dataset with denoised vars
        """
        if vars is None:
            vars = ['sigma0', 'beta0', 'gamma0']
        for varname in vars:
            varname_raw = varname + '_raw'
            noise = 'ne%sz' % varname[0]
            if varname_raw not in ds:
                continue
            if all(self.s1meta.denoised.values()):
                # already denoised, just add an alias
                ds[varname] = ds[varname_raw]
            elif len(set(self.s1meta.denoised.values())) != 1:
                # TODO: to be implemented
                raise NotImplementedError("semi denoised products not yet implemented")
            else:
                denoised = ds[varname_raw] - ds[noise]
                if clip:
                    denoised = denoised.clip(min=0)
                ds[varname] = denoised
        return ds
