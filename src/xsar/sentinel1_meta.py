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
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely
from .utils import to_lon180, haversine, timing
from . import sentinel1_xml_mappings
from .xml_parser import XmlParser
from affine import Affine
import os
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
        self.set_mask_feature('land', cartopy.feature.NaturalEarthFeature('physical', 'land', '10m'))
        self._orbit_pass = None
        self._platform_heading = None

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
        if self._orbit_pass is None:
            self._orbit_pass = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.pass')
        return self._orbit_pass

    @property
    def platform_heading(self):
        """
        Platform heading, relative to north
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._platform_heading is None:
            self._platform_heading = self.xml_parser.get_var(self.files['annotation'].iloc[0],
                                                             'annotation.platform_heading')
        return self._platform_heading

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
            if str, feature is a path to a shapefile or whatever file readable with fiona.
            It is recommended to use str, as the serialization of cartopy feature might be big.

        Examples
        --------
            Add an 'ocean' mask:
            ```
            >>> self.set_mask_feature('ocean', cartopy.feature.OCEAN)
            ```

            High resoltion shapefiles can be found from openstreetmap.
            It is recommended to use WGS84 with large polygons split from https://osmdata.openstreetmap.de/

        See Also
        --------
        xsar.Sentinel1Meta.get_mask
        """
        self._mask_features_raw[name] = feature
        # reset variable cache
        self._mask_intersecting_geometries[name] = None
        self._mask_geometry[name] = None
        self._mask_features[name] = None

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
                self._get_mask_feature(name).intersecting_geometries(self.footprint.bounds))
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
        minidict = copy.copy(minidict)
        new = cls(minidict['name'])
        new.__dict__.update(minidict)
        return new

