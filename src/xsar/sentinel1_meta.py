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
from .utils import to_lon180, haversine, timing, class_or_instancemethod
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
        self._orbit_pass = None
        self._platform_heading = None
        self._number_of_bursts = None
        self._number_of_lines = None
        self._number_of_samples = None
        self._lines_per_burst = None
        self._samples_per_burst = None
        self._radar_frequency = None
        self._azimuth_time_interval = None
        self._npoints_geolocgrid = None
        self._orbit_state_vectors = None
        self._geoloc = None
        self._ground_spacing = None
        self._swathtiming = None
        self._nb_state_vector = None
        self._nb_dcestimate = None
        self._nb_dataDcPoly = None
        self._nb_geoDcPoly = None
        self._azimuth_steering_rate = None
        self._nb_fineDce = None
        self._dopplercentroid = None
        self._range_sampling_rate = None
        self._azimuthfmrate = None
        self._nb_fmrate = None
        self._slant_range_time = None

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
    def nb_geoDcPoly(self):
        """
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._nb_geoDcPoly is None:
            self._nb_geoDcPoly = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.nb_geoDcPoly')
        return self._nb_geoDcPoly


    @property
    def nb_dataDcPoly(self):
        """
        """
        if self.multidataset:
            return None  # not defined for multidataset
        if self._nb_dataDcPoly is None:
            self._nb_dataDcPoly = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.nb_dataDcPoly')
        return self._nb_dataDcPoly

    @property
    def nb_fineDce(self):
        """
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._nb_fineDce is None:
            self._nb_fineDce = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.nb_fineDce')
        return self._nb_fineDce


    @property
    def azimuth_steering_rate(self):
        """
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._azimuth_steering_rate is None:
            self._azimuth_steering_rate = self.xml_parser.get_var(self.files['annotation'].iloc[0],
                                                                  'annotation.azimuth_steering_rate')
        return self._azimuth_steering_rate

    @property
    def geoloc(self):
        """
        xarray.Dataset with `['longitude', 'latitude', 'height', 'azimuth_time', 'slant_range_time_lr']` variables
        and `['atrack', 'xtrack'] coordinates, at the geolocation grid
        """
        # TODO: this function should be merged with self.gcps ('longitude', 'latitude', 'height' are the same)
        if self.multidataset:
            raise TypeError('geolocation_grid not available for multidataset')
        if self._geoloc is None:
            xml_annotation = self.files['annotation'].iloc[0]
            da_var_list = []
            for var_name in ['longitude', 'latitude', 'height', 'azimuth_time', 'slant_range_time_lr']:
                # TODO: we should use dask.array.from_delayed so xml files are read on demand
                da_var = self.xml_parser.get_compound_var(xml_annotation, var_name)
                da_var.name = var_name
                # FIXME: waiting for merge from upstream
                # da_var['history'] = self.xml_parser.get_compound_var(xml_annotation, var_name, describe=True)
                da_var_list.append(da_var)

            self._geoloc = xr.merge(da_var_list)

            self._geoloc.attrs = {
                'pixel_xtrack_m': self.xml_parser.get_var(xml_annotation, 'annotation.azimuthPixelSpacing'),
                'pixel_atrack_m': self.xml_parser.get_var(xml_annotation, 'annotation.rangePixelSpacing')
            }
        return self._geoloc

    @property
    def bursts(self):
        return self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'bursts')



    @property
    def number_of_lines(self):
        """
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._number_of_lines is None:
            self._number_of_lines = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.number_of_lines')
        return self._number_of_lines


    @property
    def range_sampling_rate(self):
        """
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._range_sampling_rate is None:
            self._range_sampling_rate = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.range_sampling_rate')
        logger.debug('range sampling rate %s %s',self._range_sampling_rate,type(self._range_sampling_rate))
        logger.debug('range sampling rate %s %s',self._range_sampling_rate,type(self._range_sampling_rate))
        return self._range_sampling_rate

    @property
    def incidence_angle_mid_swath(self):
        """
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._incidence_angle_mid_swath is None:
            self._incidence_angle_mid_swath = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.incidence_angle_mid_swath')
        return self._incidence_angle_mid_swath

    @property
    def number_of_samples(self):
        """
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._number_of_samples is None:
            self._number_of_samples = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.number_of_samples')
        return self._number_of_samples

    @property
    def slant_range_time(self):
        """
        /product/imageAnnotation/imageInformation/slantRangeTime
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._slant_range_time is None:
            self._slant_range_time = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.slant_range_time')
        return self._slant_range_time

    @property
    def nb_dcestimate(self):
        """
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._nb_dcestimate is None:
            self._nb_dcestimate = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.nb_dcestimate')
        return self._nb_dcestimate


    @property
    def azimuth_time_interval(self):
        """
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._azimuth_time_interval is None:
            self._azimuth_time_interval = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.azimuth_time_interval')
        return self._azimuth_time_interval

    @property
    def radar_frequency(self):
        """
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._radar_frequency is None:
            self._radar_frequency = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.radar_frequency')
        return self._radar_frequency


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
        for name in minidict['_mask_features_raw'].keys():
            assert minidict['_mask_geometry'][name] is None
            assert minidict['_mask_features'][name] is None
        minidict = copy.copy(minidict)
        new = cls(minidict['name'])
        new.__dict__.update(minidict)
        return new

    # ajout temporaire agrouaze
    def burst_azitime(self, line):
        """
        Get azimuth time for bursts (TOPS SLC).
        To be used for locations of interpolation since line indices will not handle
        properly overlap.
        """
        ## burst_nlines = self.read_global_attribute('lines_per_burst')
        ## azi_time_int = self.read_global_attribute('azimuth_time_interval')
        ## burst_list = self.read_global_attribute('burst_list')
        ## index_burst = np.floor(line / float(burst_nlines)).astype('int32')
        ## azitime = burst_list['azimuth_time'][index_burst] + \
        ##           (line - index_burst * burst_nlines) * azi_time_int
        # For consistency, azimuth time is derived from the one given in
        # geolocation grid (the one given in burst_list do not always perfectly
        # match).
        burst_nlines = self.lines_per_burst
        azi_time_int = self.azimuth_time_interval
        #geoloc = self._get_geolocation_grid()
        #geoloc = self.geoloc
        geoloc_line = self.geoloc['line'][:, int((self.geoloc['npixels'] - 1) / 2)]
        geoloc_iburst = np.floor(geoloc_line / float(burst_nlines)).astype('int32')
        iburst = np.floor(line / float(burst_nlines)).astype('int32')
        ind = np.searchsorted(geoloc_iburst, iburst, side='left')
        geoloc_azitime = self.geoloc['azimuth_time'][:, int((self.geoloc['npixels'] - 1) / 2)]
        azitime = geoloc_azitime[ind] + (line - geoloc_line[ind]) * azi_time_int
        return azitime


    @property
    def dopplercentroid(self):
        """
        copy pasted from safegeotifffile for cross spectra estimation
        """
        if self._dopplercentroid is None:
            dce = OrderedDict()
            #estimates = pads.findall('./dopplerCentroid/dcEstimateList/dcEstimate')
            dce['nlines'] = self.nb_dcestimate#len(estimates)
            #dce['npixels'] = int(estimates[0].find('fineDceList').get('count'))
            dce['npixels'] = self.nb_fineDce
            # dce['ngeocoeffs'] = \
            #     int(estimates[0].find('geometryDcPolynomial').get('count'))
            dce['ngeocoeffs'] = self.nb_geoDcPoly
            # dce['ndatacoeffs'] = \
            #     int(estimates[0].find('dataDcPolynomial').get('count'))
            dce['ndatacoeffs'] = self.nb_dataDcPoly
            dims = (dce['nlines'], dce['npixels'])
            dce['azimuth_time'] = np.empty(dims, dtype='float64')
            dce['t0'] = np.empty(dce['nlines'], dtype='float64')
            dce['geo_polynom'] = np.empty((dce['nlines'], dce['ngeocoeffs']),
                                          dtype='float32')
            dce['data_polynom'] = np.empty((dce['nlines'], dce['ndatacoeffs']),
                                           dtype='float32')
            dce['data_rms'] = np.empty(dce['nlines'], dtype='float32')
            #dce['data_rms_threshold'] =
            dce['azimuth_time_start'] = np.empty(dce['nlines'], dtype='float64')
            dce['azimuth_time_stop'] = np.empty(dce['nlines'], dtype='float64')
            dce['slant_range_time'] = np.empty(dims, dtype='float64')
            dce['frequency'] = np.empty(dims, dtype='float32')
            #for iline, estimate in enumerate(estimates):
            tmp_dce_data = {}
            for vv in ['dc_azimuth_time','dc_t0','dc_geoDcPoly','dc_dataDcPoly','dc_rmserr','dc_rmserrAboveThres'
                       ,'dc_azstarttime','dc_azstoptime','dc_slantRangeTime','dc_frequency']:
                tmp_dce_data[vv] = self.xml_parser.get_var(self._safe_files['annotation'].iloc[0], 'annotation.%s'%vv)

            for iline in range(self.nb_dcestimate):
                #strtime = estimate.find('./azimuthTime').text
                strtime = tmp_dce_data['dc_azimuth_time'][iline]
                dce['azimuth_time'][iline, :] = self._strtime2numtime(strtime)
                #dce['t0'][iline] = estimate.find('./t0').text
                dce['t0'][iline] = tmp_dce_data['dc_t0'][iline]
                # dce['geo_polynom'][iline, :] = \
                #     estimate.find('./geometryDcPolynomial').text.split()
                dce['geo_polynom'][iline, :] = tmp_dce_data['dc_geoDcPoly'][iline,:]
                # dce['data_polynom'][iline, :] = \
                #     estimate.find('./dataDcPolynomial').text.split()
                dce['data_polynom'][iline, :] = tmp_dce_data['dc_dataDcPoly'][iline,:]
                #dce['data_rms'][iline] = estimate.find('./dataDcRmsError').text
                dce['data_rms'][iline] = tmp_dce_data['dc_rmserr'][iline]
                #dce['data_rms_threshold'] =
                #strtime = estimate.find('./fineDceAzimuthStartTime').text
                strtime = tmp_dce_data['dc_azstarttime'][iline]
                dce['azimuth_time_start'][iline] = self._strtime2numtime(strtime)
                #strtime = estimate.find('./fineDceAzimuthStopTime').text
                strtime = tmp_dce_data['dc_azstoptime'][iline]
                dce['azimuth_time_stop'][iline] = self._strtime2numtime(strtime)
                #finedces = estimate.findall('./fineDceList/fineDce')
                #for ipixel, finedce in enumerate(finedces):
                dce['slant_range_time'] = tmp_dce_data['dc_slantRangeTime'].reshape((self.nb_dcestimate,self.nb_fineDce))
                dce['frequency'] = tmp_dce_data['dc_frequency'].reshape((self.nb_dcestimate,self.nb_fineDce))
                # for ipixel in  range(self.nb_fineDce):
                #     # dce['slant_range_time'][iline, ipixel] = \
                #     #     finedce.find('./slantRangeTime').text
                #     dce['slant_range_time'][iline, ipixel] = tmp_dce_data['dc_slantRangeTime'][iline,ipixel]
                #     # dce['frequency'][iline, ipixel] = \
                #     #     finedce.find('./frequency').text
                #     dce['frequency'][iline, ipixel] = tmp_dce_data['dc_frequency'][iline,ipixel]
            self._dopplercentroid = dce
        else:
            dce = self._dopplercentroid
            #dic['doppler_centroid_estimates'] = dce
        logger.debug('doppler centroid estimete slant range time %s',dce['slant_range_time'].shape)
        return dce

    def _strtime2numtime(self, strtime, fmt='%Y-%m-%dT%H:%M:%S.%f'):
        """
        Convert string time to numeric time.
        """
        dtime = datetime.strptime(strtime, fmt)
        #numtime = date2num(dtime, self.read_field('time').units)
        TIMEUNITS = 'seconds since 1990-01-01T00:00:00'
        # 'seconds since 2014-01-01 00:00:00'
        numtime = date2num(dtime,TIMEUNITS )
        return numtime

    @property
    def swathtiming(self):
        """
        read information from annotations files containing bursts timing
        """
        res_dict = None
        if self._swathtiming is None:
            res_dict = {}
            #ads = pads.find('./swathTiming')
            res_dict['lines_per_burst'] = self.lines_per_burst
            res_dict['samples_per_burst'] = self.samples_per_burst
            #ads = ads.find('./burstList')
            #dic['number_of_bursts'] = int(ads.get('count'))
            res_dict['number_of_bursts'] = self.number_of_bursts
            burstlist = OrderedDict()
            if res_dict['number_of_bursts']  != 0:
                # bursts = ads.findall('./burst')
                #bursts = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.all_bursts')
                #nbursts = len(bursts)
                nbursts = int(res_dict['number_of_bursts'])
                burstlist['nbursts'] = nbursts
                burstlist['azimuth_time'] = np.empty(nbursts, dtype='float64')
                burstlist['azimuth_anx_time'] = np.empty(nbursts, dtype='float64')
                burstlist['sensing_time'] = np.empty(nbursts, dtype='float64')
                burstlist['byte_offset'] = np.empty(nbursts, dtype='uint64')
                nlines = res_dict['lines_per_burst']
                shp = (nbursts, nlines)
                burstlist['first_valid_sample'] = np.empty(shp, dtype='int32')
                burstlist['last_valid_sample'] = np.empty(shp, dtype='int32')
                burstlist['valid_location'] = np.empty((nbursts, 4), dtype='int32')
                tmp_data = {}
                for vv in ['azimuthTime','azimuthAnxTime','sensingTime','byteOffset','firstValidSample','lastValidSample']:
                    tmp_data[vv] = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.burst_'+vv)
                    logger.debug('tmp_data %s : %s %s',vv,tmp_data[vv],tmp_data[vv].shape)
                #for ibur, burst in enumerate(bursts):
                for ibur in range(nbursts):
                    #strtime = burst.find('./azimuthTime').text
                    strtime = tmp_data['azimuthTime'][ibur]
                    burstlist['azimuth_time'][ibur] = self._strtime2numtime(strtime)
                    # burstlist['azimuth_anx_time'][ibur] = \
                    #     np.float64(burst.find('./azimuthAnxTime').text)
                    burstlist['azimuth_anx_time'][ibur] = \
                        np.float64(tmp_data['azimuthAnxTime'][ibur])
                    #strtime = burst.find('./sensingTime').text
                    strtime = tmp_data['sensingTime'][ibur]
                    burstlist['sensing_time'][ibur] = self._strtime2numtime(strtime)
                    # burstlist['byte_offset'][ibur] = \
                    #     np.uint64(burst.find('./byteOffset').text)
                    burstlist['byte_offset'][ibur] = \
                        np.uint64(tmp_data['byteOffset'][ibur])
                    # fvs = np.int32(burst.find('./firstValidSample').text.split())
                    fvs = np.int32(tmp_data['firstValidSample'][ibur])
                    burstlist['first_valid_sample'][ibur, :] = fvs
                    # lvs = np.int32(burst.find('./lastValidSample').text.split())
                    lvs = np.int32(tmp_data['lastValidSample'][ibur])
                    burstlist['last_valid_sample'][ibur, :] = lvs
                    valind = np.where((fvs != -1) | (lvs != -1))[0]
                    valloc = [ibur*nlines+valind.min(), fvs[valind].min(),
                              ibur*nlines+valind.max(), lvs[valind].max()]
                    burstlist['valid_location'][ibur, :] = valloc
            res_dict['burst_list'] = burstlist
            self._swathtiming = res_dict
        else:
            logger.debug('swath timing already filled')
            res_dict = self._swathtiming
        return res_dict

    def extent_burst(self, burst, valid=True):
        """Get extent for a SAR image burst.
        copy pasted from sarimage.py ODL
        """
        nbursts = self.number_of_bursts
        if nbursts == 0:
            raise Exception('No bursts in SAR image')
        if burst < 0 or burst >= nbursts:
            raise Exception('Invalid burst index number')
        if valid is True:
            burst_list = self.swathtiming['burst_list']
            extent = np.copy(burst_list['valid_location'][burst, :])
        else:
            extent = self._extent_max()
            nlines = self.lines_per_burst
            extent[0:3:2] = [nlines*burst, nlines*(burst+1)-1]
        return extent


    def _extent_max(self):
        """Get extent for the whole SAR image.
        copy/pasted from cerbere
        """
        return np.array((0, 0, self.number_of_lines-1,
                         self.number_of_samples-1))

    @property
    def nb_state_vector(self):
        """
        Platform heading, relative to north
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._nb_state_vector is None:
            self._nb_state_vector = self.xml_parser.get_var(self.files['annotation'].iloc[0],
                                                             'annotation.nb_state_vector')
        return self._nb_state_vector

    @property
    def nb_fmrate(self):
        """
        azimuthFmRateList annotations
        """

        if self.multidataset:
            return None  # not defined for multidataset
        if self._nb_fmrate is None:
            self._nb_fmrate = self.xml_parser.get_var(self.files['annotation'].iloc[0],
                                                             'annotation.nb_fmrate')
        return self._nb_fmrate

    @property
    def orbit_state_vectors(self):
        if self._orbit_state_vectors is None:
            res = {}
            #vectors = pads.findall('./generalAnnotation/orbitList/orbit')
            #nvect = len(vectors)
            nvect = self.nb_state_vector
            osv = OrderedDict()
            osv['nlines'] = nvect
            osv['time'] = np.empty(nvect, dtype='float64')
            osv['frame'] = []
            osv['position'] = np.empty((nvect, 3), dtype='float32')
            osv['velocity'] = np.empty((nvect, 3), dtype='float32')
            #for ivect, vector in enumerate(vectors):
            tmpdata = {}
            for vv in ['orbit_time','orbit_frame','orbit_pos_x','orbit_pos_y','orbit_pos_z',
                       'orbit_vel_x','orbit_vel_y','orbit_vel_z']:
                tmpdata[vv] = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.%s'%vv)
            for ivect in range(nvect):
                #strtime = vector.find('./time').text
                strtime = tmpdata['orbit_time'][ivect]
                osv['time'][ivect] = self._strtime2numtime(strtime)
                #osv['frame'].append(vector.find('./frame').text)
                osv['frame'].append(tmpdata['orbit_frame'][ivect])
                #osv['position'][ivect, 0] = vector.find('./position/x').text
                osv['position'][ivect, 0] = tmpdata['orbit_pos_x'][ivect]
                osv['position'][ivect, 1] = tmpdata['orbit_pos_y'][ivect]
                osv['position'][ivect, 2] = tmpdata['orbit_pos_z'][ivect]
                osv['velocity'][ivect, 0] = tmpdata['orbit_vel_x'][ivect]
                osv['velocity'][ivect, 1] = tmpdata['orbit_vel_y'][ivect]
                osv['velocity'][ivect, 2] = tmpdata['orbit_vel_z'][ivect]
            # osv['total_velocity'] = np.sqrt(osv['velocity'][:, 0]**2 + \
            #                                 osv['velocity'][:, 1]**2 + \
            #                                 osv['velocity'][:, 2]**2)
            res['orbit_state_vectors'] = osv
            res['orbit_state_position'] = osv['position'][nvect // 2, :]
            res['orbit_state_velocity'] = osv['velocity'][nvect // 2, :]
        else:
            res = self._orbit_state_vectors
        return res

    @property
    def azimuthfmrate(self):
        """
        /generalAnnotation/azimuthFmRateList/azimuthFmRate
        """
        if self._azimuthfmrate is None:
            ncoeff = self.nb_fmrate
            afr = OrderedDict()
            afr['nlines'] = ncoeff
            afr['azimuth_time'] = np.empty(ncoeff, dtype='float64')
            afr['t0'] = np.empty(ncoeff, dtype='float32')
            afr['c0'] = np.empty(ncoeff, dtype='float32')
            afr['c1'] = np.empty(ncoeff, dtype='float32')
            afr['c2'] = np.empty(ncoeff, dtype='float32')
            tmp_fmrates = {}
            for vv in ['fmrate_azimuthtime','fmrate_c0','fmrate_c1','fmrate_c2','fmrate_t0','fmrate_azimuthFmRatePolynomial']:
                tmp_fmrates[vv] = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.%s'%vv)
            for icoeff in range(ncoeff):
                strtime = tmp_fmrates["fmrate_azimuthtime"][icoeff]
                afr['azimuth_time'][icoeff] = self._strtime2numtime(strtime)
                afr['t0'][icoeff] = tmp_fmrates['fmrate_t0'][icoeff]
                if tmp_fmrates['fmrate_c0'] != []:
                    poly1 = [tmp_fmrates[cname][icoeff] for cname in ['fmrate_c0', 'fmrate_c1', 'fmrate_c2']]
                else:
                    poly1 = [None,None,None]
                #poly2 = coeff.find('./azimuthFmRatePolynomial')
                poly2 = tmp_fmrates['fmrate_azimuthFmRatePolynomial'][icoeff]
                if all([p is not None for p in poly1]): # old annotation
                    polycoeff = [p.text for p in poly1]
                elif poly2 is not None: # new annotation (if not bug)
                    #polycoeff = poly2.text.split(' ')
                    polycoeff = poly2
                else:
                    raise Exception('Could not find azimuth FM rate polynomial coefficients')
                afr['c0'][icoeff] = polycoeff[0]
                afr['c1'][icoeff] = polycoeff[1]
                afr['c2'][icoeff] = polycoeff[2]
            self._azimuthfmrate = afr
        else:
            afr = self._azimuthfmrate
        return afr


