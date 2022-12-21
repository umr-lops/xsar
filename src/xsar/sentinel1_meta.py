# -*- coding: utf-8 -*-
import cartopy.feature
import logging
import warnings
import copy
import numpy as np
import xarray
import xarray as xr
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.control import GroundControlPoint
from scipy.interpolate import RectBivariateSpline, interp1d
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
    platform = None
    short_name = None
    safe = None
    path = None
    product = None
    platform = None
    manifest = None
    subdatasets = None
    dsid = None
    manifest_attrs = None
    xsd_definitions = None

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
        try:
            self.product = os.path.basename(self.path).split('_')[2]
        except:
            print("path: %s" % self.path)
            self.product = "XXX"
        """Product type, like 'GRDH', 'SLC', etc .."""
        self.manifest = os.path.join(self.path, 'manifest.safe')
        self.manifest_attrs = self.xml_parser.get_compound_var(self.manifest, 'safe_attributes')

        self._safe_files = None
        self.multidataset = False
        """True if multi dataset"""
        self.subdatasets = gpd.GeoDataFrame(geometry=[], index=[])
        """Subdatasets as GeodataFrame (empty if single dataset)"""
        datasets_names = list(self.safe_files['dsid'].sort_index().unique())
        self.xsd_definitions = self.get_annotation_definitions()
        if self.name.endswith(':') and len(datasets_names) == 1:
            self.name = datasets_names[0]
        self.dsid = self.name.split(':')[-1]
        """Dataset identifier (like 'WV_001', 'IW1', 'IW'), or empty string for multidataset"""
        # submeta is a list of submeta objects if multidataset and TOPS
        # this list will remain empty for _WV__SLC because it will be time-consuming to process them
        self._submeta = []
        if self.short_name.endswith(':'):
            self.short_name = self.short_name + self.dsid
        if self.files.empty:
            try:
                self.subdatasets = gpd.GeoDataFrame(geometry=self.manifest_attrs['footprints'], index=datasets_names)
            except ValueError:
                # not as many footprints than subdatasets count. (probably TOPS product)
                self._submeta = [ Sentinel1Meta(subds) for subds in datasets_names ]
                sub_footprints = [ submeta.footprint for submeta in self._submeta ]
                self.subdatasets = gpd.GeoDataFrame(geometry=sub_footprints, index=datasets_names)
            self.multidataset = True

        self.platform = self.manifest_attrs['mission'] + self.manifest_attrs['satellite']
        """Mission platform"""
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
        return name == self.name or name in self.subdatasets.index


    def _get_time_range(self):
        if self.multidataset:
            time_range = [self.manifest_attrs['start_date'], self.manifest_attrs['stop_date']]
        else:
            time_range = self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.line_time_range')
        return pd.Interval(left=pd.Timestamp(time_range[0]), right=pd.Timestamp(time_range[-1]), closed='both')

    def to_dict(self, keys='minimal'):

        info_keys = {
            'minimal': ['ipf', 'platform', 'swath', 'product', 'pols']
        }
        info_keys['all'] = info_keys['minimal'] + ['name', 'start_date', 'stop_date', 'footprint', 'coverage',
                                                   'orbit_pass', 'platform_heading'] #  'pixel_line_m', 'pixel_sample_m',

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
    def footprint(self):
        """footprint, as a shapely polygon or multi polygon"""
        if self.multidataset:
            return unary_union(self._footprints)
        return self.geoloc.attrs['footprint']

    @property
    def geometry(self):
        """alias for footprint"""
        return self.footprint

    @property
    def geoloc(self):
        """
        xarray.Dataset with `['longitude', 'latitude', 'altitude', 'azimuth_time', 'slant_range_time','incidence','elevation' ]` variables
        and `['line', 'sample']` coordinates, at the geolocation grid
        """
        if self.multidataset:
            raise TypeError('geolocation_grid not available for multidataset')
        if self._geoloc is None:
            xml_annotation = self.files['annotation'].iloc[0]
            da_var_list = []
            for var_name in ['longitude', 'latitude', 'height', 'azimuthTime', 'slantRangeTime', 'incidenceAngle',
                             'elevationAngle']:
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
                    self._geoloc[ll].isel(line=a, sample=x).values for a, x in [(0, 0), (0, -1), (-1, -1), (-1, 0)]
                ]
            corners = list(zip(footprint_dict['longitude'], footprint_dict['latitude']))
            p = Polygon(corners)
            self._geoloc.attrs['footprint'] = p

            # compute acquisition size/resolution in meters
            # first vector is on sample
            acq_sample_meters, _ = haversine(*corners[0], *corners[1])
            # second vector is on line
            acq_line_meters, _ = haversine(*corners[1], *corners[2])
            self._geoloc.attrs['coverage'] = "%dkm * %dkm (line * sample )" % (
                acq_line_meters / 1000, acq_sample_meters / 1000)
            
            # compute self._geoloc.attrs['approx_transform'], from gcps
            # we need to convert self._geoloc to  a list of GroundControlPoint
            def _to_rio_gcp(pt_geoloc):
                # convert a point from self._geoloc grid to rasterio GroundControlPoint
                return GroundControlPoint(
                    x=pt_geoloc.longitude.item(),
                    y=pt_geoloc.latitude.item(),
                    z=pt_geoloc.height.item(),
                    col=pt_geoloc.line.item(),
                    row=pt_geoloc.sample.item()
                )

            gcps = [
                _to_rio_gcp(self._geoloc.sel(line=line, sample=sample))
                for line in  self._geoloc.line for sample in self._geoloc.sample
            ]
            # approx transform, from all gcps (inaccurate)
            self._geoloc.attrs['approx_transform'] = rasterio.transform.from_gcps(gcps)
            for vv in self._geoloc:
                if vv in self.xsd_definitions:
                    self._geoloc[vv].attrs['definition'] = str(self.xsd_definitions[vv])


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
                descr = '%s.%s %s' % (descr.__module__, descr.__class__.__name__, descr.name)
            except AttributeError:
                pass
            return descr

        if self._mask_geometry[name] is None:
            poly = self._get_mask_intersecting_geometries(name) \
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
        """coverage, as a string like '251km * 170km (sample * line )'"""
        if self.multidataset:
            return None  # not defined for multidataset
        return self.geoloc.attrs['coverage']

    @property
    def pixel_line_m(self):
        """pixel line spacing, in meters (at sensor level)"""
        if self.multidataset:
            res = None  # not defined for multidataset
        else:
            res = self.image['azimuthPixelSpacing']
        return res

    @property
    def pixel_sample_m(self):
        """pixel sample spacing, in meters (at sensor level)"""
        if self.multidataset:
            res = None  # not defined for multidataset
        else:
            res = self.image['groundRangePixelSpacing']
        return res

    @property
    def time_range(self):
        """time range as pd.Interval"""
        if self._time_range is None:
            self._time_range = self._get_time_range()
        return self._time_range

    @property
    def start_date(self):
        """start date, as datetime.datetime"""
        return '%s'%self.time_range.left

    @property
    def stop_date(self):
        """stort date, as datetime.datetime"""
        return '%s'%self.time_range.right

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
        return ((np.max(self.geoloc['longitude']) - np.min(self.geoloc['longitude'])) > 180).item()

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
        for vv in gdf_orbit:
            if vv in self.xsd_definitions:
                gdf_orbit[vv].attrs['definition'] = self.xsd_definitions[vv]
        gdf_orbit.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'orbit',
                                                                      describe=True)
        return gdf_orbit

    @property
    def image(self) -> xarray.Dataset:
        if self.multidataset:
            return None
        img_dict = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'image')
        img_dict['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'image', describe=True)
        for vv in img_dict:
            if vv in self.xsd_definitions:
                img_dict[vv].attrs['definition'] = self.xsd_definitions[vv]
        return img_dict

    @property
    def azimuth_fmrate(self):
        """
        xarray.Dataset
            Frequency Modulation rate annotations such as t0 (azimuth time reference) and polynomial coefficients: Azimuth FM rate = c0 + c1(tSR - t0) + c2(tSR - t0)^2
        """
        fmrates = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'azimuth_fmrate')
        fmrates.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'azimuth_fmrate',
                                                                    describe=True)
        for vv in fmrates:
            if vv in self.xsd_definitions:
                fmrates[vv].attrs['definition'] = self.xsd_definitions[vv]
        return fmrates

    @property
    def _dict_coords2ll(self):
        """
        dict with keys ['longitude', 'latitude'] with interpolation function (RectBivariateSpline) as values.

        Examples:
        ---------
            get longitude at line=100 and sample=200:
            ```
            >>> self._dict_coords2ll['longitude'].ev(100,200)
            array(-66.43947434)
            ```
        Notes:
        ------
            if self.cross_antemeridian is True, 'longitude' will be in range [0, 360]
        """
        resdict = {}
        geoloc = self.geoloc
        if self.cross_antemeridian:
            geoloc['longitude'] = geoloc['longitude'] % 360

        idx_sample = np.array(geoloc.sample)
        idx_line = np.array(geoloc.line)

        for ll in ['longitude', 'latitude']:
            resdict[ll] = RectBivariateSpline(idx_line, idx_sample, np.asarray(geoloc[ll]), kx=1, ky=1)

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
        xsar.Sentinel1Meta.ll2coords
        xsar.Sentinel1Dataset.ll2coords

        """

        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self._coords2ll_shapely(args[0])

        lines, samples = args

        scalar = True
        if hasattr(lines, '__iter__'):
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
                lon = dict_coords2ll['longitude'](lines, samples)
                lat = dict_coords2ll['latitude'](lines, samples)
            else:
                lon = dict_coords2ll['longitude'].ev(lines, samples)
                lat = dict_coords2ll['latitude'].ev(lines, samples)

        if self.cross_antemeridian:
            lon = to_lon180(lon)

        if scalar and hasattr(lon, '__iter__'):
            lon = lon.item()
            lat = lat.item()

        if hasattr(lon, '__iter__') and type(lon) is not type(lines):
            lon = type(lines)(lon)
            lat = type(lines)(lat)

        return lon, lat

    def ll2coords(self, *args):
        """
        Get `(lines, samples)` from `(lon, lat)`,
        or convert a lon/lat shapely shapely object to line/sample coordinates.

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

            >>> (line, sample) = meta.ll2coords(84.81, 21.32) # (lon, lat)
            >>> (line, sample)
            (9752.766349989339, 17852.571322887554)

        See Also
        --------
        xsar.Sentinel1Meta.coords2ll
        xsar.Sentinel1Dataset.coords2ll

        """

        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self._ll2coords_shapely(args[0])

        lon, lat = args

        # approximation with global inaccurate transform
        line_approx, sample_approx = ~self.approx_transform * (np.asarray(lon), np.asarray(lat))

        # Theoretical identity. It should be the same, but the difference show the error.
        lon_identity, lat_identity = self.coords2ll(line_approx, sample_approx, to_grid=False)
        line_identity, sample_identity = ~self.approx_transform * (lon_identity, lat_identity)

        # we are now able to compute the error, and make a correction
        line_error = line_identity - line_approx
        sample_error = sample_identity - sample_approx

        line = line_approx - line_error
        sample = sample_approx - sample_error

        if hasattr(lon, '__iter__'):
            scalar = False
        else:
            scalar = True

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
    def _bursts(self):
        if self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.number_of_bursts') > 0:
            bursts = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'bursts')
            for vv in bursts:
                if vv in self.xsd_definitions:
                    bursts[vv].attrs['definition'] = self.xsd_definitions[vv]
            bursts.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'bursts',
                                                                       describe=True)
            return bursts
        else:
            bursts = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'bursts_grd')
            bursts.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'bursts_grd',
                                                                       describe=True)
            return bursts

    @property
    def approx_transform(self):
        """
        Affine transfom from geoloc.

        This is an inaccurate transform, with errors up to 600 meters.
        But it's fast, and may fit some needs, because the error is stable localy.
        See `xsar.Sentinel1Meta.coords2ll` `xsar.Sentinel1Meta.ll2coords` for accurate methods.

        Examples
        --------
            get `longitude` and `latitude` from tuple `(line, sample)`:

            >>> longitude, latitude = self.approx_transform * (line, sample)

            get `line` and `sample` from tuple `(longitude, latitude)`

            >>> line, sample = ~self.approx_transform * (longitude, latitude)

        See Also
        --------
        xsar.Sentinel1Meta.coords2ll
        xsar.Sentinel1Meta.ll2coords`

        """
        return self.geoloc.attrs['approx_transform']

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
        for vv in dce:
            if vv in self.xsd_definitions:
                dce[vv].attrs['definition'] = self.xsd_definitions[vv]
        dce.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'doppler_estimate',
                                                                describe=True)
        return dce



    def get_annotation_definitions(self):
        """
        
        :return:
        """
        final_dict = {}
        ds_path_xsd = self.xml_parser.get_compound_var(self.manifest, 'xsd_files')
        full_path_xsd = os.path.join(self.path, ds_path_xsd['xsd_product'].values[0])
        if os.path.exists(full_path_xsd):
            rootxsd = self.xml_parser.getroot(full_path_xsd)
            mypath = '/xsd:schema/xsd:complexType/xsd:sequence/xsd:element'

            for lulu, uu in enumerate(rootxsd.xpath(mypath, namespaces=sentinel1_xml_mappings.namespaces)):
                mykey = uu.values()[0]
                if uu.getchildren() != []:
                    myvalue = uu.getchildren()[0].getchildren()[0]
                else:
                    myvalue = None
                final_dict[mykey] = myvalue

        return final_dict
