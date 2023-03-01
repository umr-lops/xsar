# -*- coding: utf-8 -*-
import logging
import warnings
import numpy as np
import xarray
import xarray as xr
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.control import GroundControlPoint
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .base_meta import BaseMeta
from .utils import haversine, timing, class_or_instancemethod
from .raster_readers import available_rasters
from . import sentinel1_xml_mappings
from .xml_parser import XmlParser
import os
from .ipython_backends import repr_mimebundle

logger = logging.getLogger('xsar.sentinel1_meta')
logger.addHandler(logging.NullHandler())


class Sentinel1Meta(BaseMeta):
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
    xml_parser = None
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
        for name, feature in self.__class__._mask_features_raw.items():
            self.set_mask_feature(name, feature)
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
                for line in self._geoloc.line for sample in self._geoloc.sample
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
    def pols(self):
        """polarisations strings, separated by spaces """
        return " ".join(self.manifest_attrs['polarizations'])

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

    def get_calibration_luts(self):
        """
        """
        #sigma0_lut = self.xml_parser.get_var(self.files['calibration'].iloc[0], 'calibration.sigma0_lut',describe=True)
        luts = self.xml_parser.get_compound_var(self.files['calibration'].iloc[0],'luts_raw')
        return luts

    def get_noise_azi_raw(self):
        tmp = []
        pols = []
        for pol_code, xml_file in self.files['noise'].items():
            #pol = self.files['polarization'].cat.categories[pol_code-1]
            pol = os.path.basename(xml_file).split('-')[4].lower()
            pols.append(pol)
            if self.product == 'SLC':
                noise_lut_azi_raw_ds = self.xml_parser.get_compound_var(xml_file,'noise_lut_azi_raw_slc')
            else:
                noise_lut_azi_raw_ds = self.xml_parser.get_compound_var(xml_file, 'noise_lut_azi_raw_grd')
            for vari in noise_lut_azi_raw_ds:
                if 'noiseLut_' in vari:
                    varitmp = 'noiseLut'
                    hihi = self.xml_parser.get_var(self.files['noise'].iloc[0], 'noise.azi.%s' % varitmp,
                                                   describe=True)
                elif vari == 'noiseLut' and self.product=='WV': #WV case
                    hihi = 'dummy variable, noise is not defined in azimuth for WV acquisitions'
                else:
                    varitmp = vari
                    hihi = self.xml_parser.get_var(self.files['noise'].iloc[0], 'noise.azi.%s' % varitmp,
                                                   describe=True)

                noise_lut_azi_raw_ds[vari].attrs['description'] = hihi
            tmp.append(noise_lut_azi_raw_ds)
        ds = xr.concat(tmp,pd.Index(pols, name="pol"))
        return ds

    def get_noise_range_raw(self):
        tmp = []
        pols = []
        for pol_code, xml_file in self.files['noise'].items():
            #pol = self.files['polarization'].cat.categories[pol_code - 1]
            pol = os.path.basename(xml_file).split('-')[4].lower()
            pols.append(pol)
            noise_lut_range_raw_ds = self.xml_parser.get_compound_var(xml_file, 'noise_lut_range_raw')
            for vari in noise_lut_range_raw_ds:
                hihi = self.xml_parser.get_var(self.files['noise'].iloc[0], 'noise.range.%s' % vari,
                                               describe=True)
                noise_lut_range_raw_ds[vari].attrs['description'] = hihi
            tmp.append(noise_lut_range_raw_ds)
        ds = xr.concat(tmp, pd.Index(pols, name="pol"))
        return ds
