# -*- coding: utf-8 -*-
import logging
import warnings
import numpy as np
import xarray
from scipy.interpolate import RectBivariateSpline
import xarray as xr
import dask
import rasterio
import rasterio.features
from rasterio.control import GroundControlPoint
import rioxarray
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, box
import shapely
from .utils import timing, haversine, map_blocks_coords, bbox_coords, BlockingActorProxy, merge_yaml, get_glob, \
    to_lon180
from numpy import asarray
from affine import Affine
from .sentinel1_meta import Sentinel1Meta
from .ipython_backends import repr_mimebundle
import yaml
import datatree
import pandas as pd
import geopandas as gpd

logger = logging.getLogger('xsar.sentinel1_dataset')
logger.addHandler(logging.NullHandler())

# we know tiff as no geotransform : ignore warning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# allow nan without warnings
# some dask warnings are still non filtered: https://github.com/dask/dask/issues/3245
np.errstate(invalid='ignore')


# noinspection PyTypeChecker
class Sentinel1Dataset:
    """
    Handle a SAFE subdataset.
    A dataset might contain several tiff files (multiples polarizations), but all tiff files must share the same footprint.

    The main attribute useful to the end-user is `self.dataset` (`xarray.Dataset` , with all variables parsed from xml and tiff files.)

    Parameters
    ----------
    dataset_id: str or Sentinel1Meta object

        if str, it can be a path, or a gdal dataset identifier like `'SENTINEL1_DS:%s:WV_001' % filename`)

    resolution: dict, number or string, optional
        resampling dict like `{'line': 20, 'sample': 20}` where 20 is in pixels.

        if a number, dict will be constructed from `{'line': number, 'sample': number}`

        if str, it must end with 'm' (meters), like '100m'. dict will be computed from sensor pixel size.

    resampling: rasterio.enums.Resampling or str, optional

        Only used if `resolution` is not None.

        ` rasterio.enums.Resampling.rms` by default. `rasterio.enums.Resampling.nearest` (decimation) is fastest.

    luts: bool, optional

        if `True` return also luts as variables (ie `sigma0_lut`, `gamma0_lut`, etc...). False by default.

    chunks: dict, optional

        dict with keys ['pol','line','sample'] (dask chunks).

    dtypes: None or dict, optional

        Specify the data type for each variable.

    patch_variable: bool, optional

        activate or not variable pathching ( currently noise lut correction for IPF2.9X)

    """

    def __init__(self, dataset_id, resolution=None,
                 resampling=rasterio.enums.Resampling.rms,
                 luts=False, chunks={'line': 5000, 'sample': 5000},
                 dtypes=None, patch_variable=True):

        # miscellaneous attributes that are not know from xml files
        attrs_dict = {
            'pol': {
                'comment': 'ordered polarizations (copol, crosspol)'
            },
            'line': {
                'units': '1',
                'comment': 'azimuth direction, in pixels from full resolution tiff'
            },
            'sample': {
                'units': '1',
                'comment': 'cross track direction, in pixels from full resolution tiff'
            },
            'sigma0_raw': {
                'units': 'linear'
            },
            'gamma0_raw': {
                'units': 'linear'
            },
            'nesz': {
                'units': 'linear',
                'comment': 'sigma0 noise'
            },
            'negz': {
                'units': 'linear',
                'comment': 'beta0 noise'
            },
        }

        # default dtypes
        self._dtypes = {
            'latitude': 'f4',
            'longitude': 'f4',
            'incidence': 'f4',
            'elevation': 'f4',
            'altitude': 'f4',
            'ground_heading': 'f4',
            'nesz': None,
            'negz': None,
            'sigma0_raw': None,
            'gamma0_raw': None,
            'noise_lut': 'f4',
            'noise_lut_range': 'f4',
            'noise_lut_azi': 'f4',
            'sigma0_lut': 'f8',
            'gamma0_lut': 'f8',
            'azimuth_time': np.datetime64,
            'slant_range_time': None
        }
        if dtypes is not None:
            self._dtypes.update(dtypes)

        # default meta for map_blocks output.
        # as asarray is imported from numpy, it's a numpy array.
        # but if later we decide to import asarray from cupy, il will be a cupy.array (gpu)
        self._default_meta = asarray([], dtype='f8')

        self.s1meta = None
        """`xsar.Sentinel1Meta` object"""

        if not isinstance(dataset_id, Sentinel1Meta):
            self.s1meta = BlockingActorProxy(Sentinel1Meta, dataset_id)
            # check serializable
            # import pickle
            # s1meta = pickle.loads(pickle.dumps(self.s1meta))
            # assert isinstance(s1meta.coords2ll(100, 100),tuple)
        else:
            # we want self.s1meta to be a dask actor on a worker
            self.s1meta = BlockingActorProxy(Sentinel1Meta.from_dict, dataset_id.dict)
        del dataset_id

        if self.s1meta.multidataset:
            raise IndexError(
                """Can't open an multi-dataset. Use `xsar.Sentinel1Meta('%s').subdatasets` to show availables ones""" % self.s1meta.path
            )
        self.datatree = datatree.DataTree.from_dict({'high_resolution_dataset':self._load_digital_number(resolution=resolution, resampling=resampling, chunks=chunks)})

        #self.datatree['high_resolution_dataset'].ds = .from_dict({'high_resolution_dataset':self._load_digital_number(resolution=resolution, resampling=resampling, chunks=chunks)
        self._dataset = self.datatree['high_resolution_dataset'].ds #the two variables should be linken then.
        self._dataset = xr.merge([xr.Dataset({'time': self.get_burst_azitime}), self._dataset])

        # dataset no-pol template for function evaluation on coordinates (*no* values used)
        # what's matter here is the shape of the image, not the values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            if self.s1meta._bursts['burst'].size != 0:
                # SLC TOPS, tune the high res grid because of bursts overlapping
                #line_time = self._burst_azitime
                line_time = self.get_burst_azitime
                self._da_tmpl = xr.DataArray(
                    dask.array.empty_like(
                        np.empty((len(line_time), len(self._dataset.digital_number.sample))),
                        dtype=np.int8, name="empty_var_tmpl-%s" % dask.base.tokenize(self.s1meta.name)),
                    dims=('line', 'sample'),
                    coords={
                        'line': self._dataset.digital_number.line,
                        'sample': self._dataset.digital_number.sample,
                        'line_time': line_time.astype(float),
                    },
                )
            else:

                self._da_tmpl = xr.DataArray(
                    dask.array.empty_like(
                        self._dataset.digital_number.isel(pol=0).drop('pol'),
                        dtype=np.int8, name="empty_var_tmpl-%s" % dask.base.tokenize(self.s1meta.name)),
                    dims=('line', 'sample'),
                    coords={'line': self._dataset.digital_number.line,
                            'sample': self._dataset.digital_number.sample}
                )

        # FIXME possible memory leak
        # when calling a self.s1meta method, an ActorFuture is returned.
        # But this seems to break __del__ methods from both Sentinel1Meta and XmlParser
        # Is it a memory leak ?
        # see https://github.com/dask/distributed/issues/5610
        # tmp_f = self.s1meta.to_dict("all")
        # del tmp_f
        # return

        self._dataset.attrs.update(self.s1meta.to_dict("all"))

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
        }

        # dict mapping specifying if the variable has 'pol' dimension
        self._vars_with_pol = {
            'sigma0_lut': True,
            'gamma0_lut': True,
            'noise_lut_range': True,
            'noise_lut_azi': True,
            'incidence': False,
            'elevation': False,
            'altitude': False,
            'azimuth_time': False,
            'slant_range_time': False,
            'longitude': False,
            'latitude': False
        }

        # variables not returned to the user (unless luts=True)
        self._hidden_vars = ['sigma0_lut', 'gamma0_lut', 'noise_lut', 'noise_lut_range', 'noise_lut_azi']
        # attribute to activate correction on variables, if available
        self._patch_variable = patch_variable

        self._luts = self._lazy_load_luts(self._map_lut_files.keys())

        # noise_lut is noise_lut_range * noise_lut_azi
        if 'noise_lut_range' in self._luts.keys() and 'noise_lut_azi' in self._luts.keys():
            self._luts = self._luts.assign(noise_lut=self._luts.noise_lut_range * self._luts.noise_lut_azi)
            self._luts.noise_lut.attrs['history'] = merge_yaml(
                [self._luts.noise_lut_range.attrs['history'] + self._luts.noise_lut_azi.attrs['history']],
                section='noise_lut'
            )

        #self._rasterized_masks = self.load_rasterized_masks()

        ds_merge_list = [self._dataset, self._load_ground_heading(),  # lon_lat
                         self._luts.drop_vars(self._hidden_vars, errors='ignore')]

        if luts:
            ds_merge_list.append(self._luts[self._hidden_vars])
        attrs = self._dataset.attrs
        self._dataset = xr.merge(ds_merge_list)
        self._dataset.attrs = attrs

        for var_name, lut_name in self._map_var_lut.items():
            if lut_name in self._luts:
                # merge var_name into dataset (not denoised)
                self._dataset = self._dataset.merge(self._apply_calibration_lut(var_name))
                # merge noise equivalent for var_name (named 'ne%sz' % var_name[0)
                self._dataset = self._dataset.merge(self._get_noise(var_name))
            else:
                logger.debug("Skipping variable '%s' ('%s' lut is missing)" % (var_name, lut_name))

        self._dataset = self._dataset.merge(self._load_from_geoloc(['altitude', 'azimuth_time', 'slant_range_time',
                                                                    'incidence', 'elevation', 'longitude', 'latitude']))

        rasters = self._load_rasters_vars()
        if rasters is not None:
            self._dataset = xr.merge([self._dataset, rasters])

        self._dataset = self._dataset.merge(self._get_sensor_velocity())
        self._dataset = self._dataset.merge(self._range_ground_spacing())
        self._dataset = self._add_denoised(self._dataset)
        self.recompute_attrs()

        # set miscellaneous attrs
        for var, attrs in attrs_dict.items():
            try:
                self._dataset[var].attrs.update(attrs)
            except KeyError:
                pass

        self.sliced = False
        """True if dataset is a slice of original L1 dataset"""

        self.resampled = resolution is not None
        """True if dataset is not a sensor resolution"""

        # save original bbox
        self._bbox_coords_ori = self._bbox_coords

        # build datatree
        ### geoloc
        geoloc = self.s1meta.geoloc
        geoloc.attrs['history'] = 'annotations'
        ### bursts
        bu = self.s1meta._bursts
        bu.attrs['history'] = 'annotations'

        # azimuth fm rate
        FM = self.s1meta.azimuth_fmrate
        FM.attrs['history'] = 'annotations'
        # dataset principal
        self._dataset['sampleSpacing'] = xarray.DataArray(self.s1meta.image['slantRangePixelSpacing'],
                                                            attrs={'units': 'm', 'referential': 'slant'})
        self._dataset['lineSpacing'] = xarray.DataArray(self.s1meta.image['azimuthPixelSpacing'],
                                                          attrs={'units': 'm'})
        # doppler
        dop = self.s1meta._doppler_estimate
        dop.attrs['history'] = 'annotations'

        self.datatree = datatree.DataTree.from_dict({'high_resolution_dataset': self._dataset, 'geolocation_annotation': geoloc,
                                                'bursts': bu, 'FMrate': FM, 'doppler_estimate': dop,
                                                # 'image_information':
                                                'orbit': self.s1meta.orbit
                                                })
        self.datatree.attrs = xr.Dataset(self.s1meta.image)

    def __del__(self):
        logger.debug('__del__')

    @property
    def dataset(self):
        """
        `xarray.Dataset` representation of this `xsar.Sentinel1Dataset` object.
        This property can be set with a new dataset, if the dataset was computed from the original dataset.
        """
        #return self._dataset
        return self.datatree['high_resolution_dataset'].ds

    @dataset.setter
    def dataset(self, ds):
        if self.s1meta.name == ds.attrs['name']:
            # check if new ds has changed coordinates
            if not self.sliced:
                self.sliced = any(
                    [list(ds[d].values) != list(self._dataset[d].values) for d in ['line', 'sample']])
            self._dataset = ds
            #self._dataset = self.datatree['high_resolution_dataset'].ds
            self.recompute_attrs()
        else:
            raise ValueError("dataset must be same kind as original one.")

    @dataset.deleter
    def dataset(self):
        logger.debug('deleter dataset')

    @property
    def _bbox_coords(self):
        """
        Dataset bounding box, in line/sample coordinates
        """
        bbox_ext = bbox_coords(self.dataset.line.values, self.dataset.sample.values)
        return bbox_ext

    @property
    def _bbox_ll(self):
        """Dataset bounding box, lon/lat"""
        return self.s1meta.coords2ll(*zip(*self._bbox_coords))

    @property
    def geometry(self):
        """
        geometry of this dataset, as a `shapely.geometry.Polygon` (lon/lat coordinates)
        """
        return Polygon(zip(*self._bbox_ll))

    @property
    def footprint(self):
        """alias for `xsar.Sentinel1Dataset.geometry`"""
        return self.geometry

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

        Notes
        -----
        The difference with `xsar.Sentinel1Meta.ll2coords` is that coordinates are rounded to the nearest dataset coordinates.

        See Also
        --------
        xsar.Sentinel1Meta.ll2coords

        """
        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self.s1meta.ll2coords_shapely(args[0].intersection(self.geometry))

        line, sample = self.s1meta.ll2coords(*args)

        if hasattr(args[0], '__iter__'):
            scalar = False
        else:
            scalar = True

        tolerance = np.max([np.percentile(np.diff(self.dataset[c].values), 90) / 2 for c in ['line', 'sample']]) + 1
        try:
            # select the nearest valid pixel in ds
            ds_nearest = self.dataset.sel(line=line, sample=sample, method='nearest', tolerance=tolerance)
            if scalar:
                (line, sample) = (ds_nearest.line.values.item(), ds_nearest.sample.values.item())
            else:
                (line, sample) = (ds_nearest.line.values, ds_nearest.sample.values)
        except KeyError:
            # out of bounds, because of `tolerance` keyword
            (line, sample) = (line * np.nan, sample * np.nan)

        return line, sample

    def coords2ll(self, *args, **kwargs):
        """
         Alias for `xsar.Sentinel1Meta.coords2ll`

         See Also
         --------
         xsar.Sentinel1Meta.coords2ll
        """
        return self.s1meta.coords2ll(*args, **kwargs)

    @property
    def len_line_m(self):
        """line length, in meters"""
        bbox_ll = list(zip(*self._bbox_ll))
        len_m, _ = haversine(*bbox_ll[1], *bbox_ll[2])
        return len_m

    @property
    def len_sample_m(self):
        """sample length, in meters """
        bbox_ll = list(zip(*self._bbox_ll))
        len_m, _ = haversine(*bbox_ll[0], *bbox_ll[1])
        return len_m

    @property
    def pixel_line_m(self):
        """line pixel spacing, in meters (relative to dataset)"""
        return self.s1meta.pixel_line_m * np.unique(np.round(np.diff(self._dataset['line'].values), 1))[0]

    @property
    def pixel_sample_m(self):
        """sample pixel spacing, in meters (relative to dataset)"""
        return self.s1meta.pixel_sample_m * np.unique(np.round(np.diff(self._dataset['sample'].values), 1))[0]

    @property
    def coverage(self):
        """coverage string"""
        return "%dkm * %dkm (line * sample )" % (self.len_line_m / 1000, self.len_sample_m / 1000)

    @property
    def _regularly_spaced(self):
        return max(
            [np.unique(np.round(np.diff(self._dataset[dim].values), 1)).size for dim in ['line', 'sample']]) == 1

    def _set_rio(self, ds):
        # set .rio accessor for ds. ds must be same kind a self._dataset (not checked!)
        gcps = self._local_gcps

        want_dataset = True
        if isinstance(ds, xarray.DataArray):
            # temporary convert to dataset
            try:
                ds = ds.to_dataset()
            except ValueError:
                ds = ds.to_dataset(name='_tmp_rio')
            want_dataset = False

        for v in ds:
            if set(['line', 'sample']).issubset(set(ds[v].dims)):
                ds[v] = ds[v].set_index({'sample': 'sample', 'line': 'line'})
                ds[v] = ds[v].rio.write_gcps(
                    gcps, 'epsg:4326', inplace=True
                ).rio.set_spatial_dims(
                    'sample', 'line', inplace=True
                ).rio.write_coordinate_system(
                    inplace=True)
                # remove/reset some incorrect attrs set by rio
                # (long_name is 'latitude', but it's incorrect for line axis ...)
                for ax in ['line', 'sample']:
                    [ds[v][ax].attrs.pop(k, None) for k in ['long_name', 'standard_name']]
                    ds[v][ax].attrs['units'] = '1'

        if not want_dataset:
            # convert back to dataarray
            ds = ds[v]
            if ds.name == '_tmp_rio':
                ds.name = None
        return ds

    def recompute_attrs(self):
        """
        Recompute dataset attributes. It's automaticaly called if you assign a new dataset, for example

        >>> xsar_obj.dataset = xsar_obj.dataset.isel(line=slice(1000,5000))
        >>> #xsar_obj.recompute_attrs() # not needed

        This function must be manually called before using the `.rio` accessor of a variable

        >>> xsar_obj.recompute_attrs()
        >>> xsar_obj.dataset['sigma0'].rio.reproject(...)

        See Also
        --------
            [rioxarray information loss](https://corteva.github.io/rioxarray/stable/getting_started/manage_information_loss.html)

        """
        if not self._regularly_spaced:
            warnings.warn(
                "Irregularly spaced dataset (probably multiple selection). Some attributes will be incorrect.")
        attrs = self._dataset.attrs
        attrs['pixel_sample_m'] = self.pixel_sample_m
        attrs['pixel_line_m'] = self.pixel_line_m
        attrs['coverage'] = self.coverage
        attrs['footprint'] = self.footprint

        self.dataset.attrs.update(attrs)

        self._dataset = self._set_rio(self._dataset)

        return None

    def _patch_lut(self, lut):
        """
        patch proposed by MPC Sentinel-1 : https://jira-projects.cls.fr/browse/MPCS-2007 for noise vectors of WV SLC IPF2.9X products
        adjustement proposed by BAE are the same for HH and VV, and suppose to work for both old and new WV2 EAP
        they were estimated using WV image with very low NRCS (black images) and computing std(sigma0).
        Parameters
        ----------
        lut xarray.Dataset

        Returns
        -------
        lut xarray.Dataset
        """
        if self.s1meta.swath == 'WV':
            if lut.name in ['noise_lut_azi'] and self.s1meta.ipf in [2.9, 2.91] and \
                    self.s1meta.platform in ['SENTINEL-1A', 'SENTINEL-1B']:
                noise_calibration_cst_pp1 = {
                    'SENTINEL-1A':
                        {'WV1': -38.13,
                         'WV2': -36.84
                         },
                    'SENTINEL-1B':
                        {'WV1': -39.30,
                         'WV2': -37.44,
                         }
                }
                cst_db = noise_calibration_cst_pp1[self.s1meta.platform][self.s1meta.image['swath_subswath']]
                cst_lin = 10 ** (cst_db / 10)
                lut = lut * cst_lin
                lut.attrs['comment'] = 'patch on the noise_lut_azi : %s dB' % cst_db
        return lut

    @timing
    def _lazy_load_luts(self, luts_names):
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
        luts = None
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

                # get the lut function. As it takes some time to parse xml, make it delayed
                lut_f_delayed = dask.delayed(self.s1meta.xml_parser.get_compound_var)(xml_file, lut_name)
                lut = map_blocks_coords(
                    self._da_tmpl.astype(self._dtypes[lut_name]),
                    lut_f_delayed,
                    name='blocks_%s' % name
                )

                # needs to add pol dim ?
                if self._vars_with_pol[lut_name]:
                    lut = lut.assign_coords(pol_code=pol_code).expand_dims('pol_code')

                # set xml file and xpath used as history
                histo = self.s1meta.xml_parser.get_compound_var(xml_file, lut_name,
                                                                describe=True)
                lut.name = lut_name
                if self._patch_variable:
                    lut = self._patch_lut(lut)
                lut.attrs['history'] = histo
                lut = lut.to_dataset()

                luts_list.append(lut)
            luts = xr.combine_by_coords(luts_list)

            # convert pol_code to string
            pols = self.s1meta.files['polarization'].cat.categories[luts.pol_code.values.tolist()]
            luts = luts.rename({'pol_code': 'pol'}).assign_coords({'pol': pols})
        return luts

    @timing
    def _load_digital_number(self, resolution=None, chunks=None, resampling=rasterio.enums.Resampling.rms):
        """
        load digital_number from self.s1meta.files['measurement'], as an `xarray.Dataset`.

        Parameters
        ----------
        resolution: None, number, str or dict
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
            'line': 'y',
            'sample': 'x'
        }

        if resolution is not None:
            comment = 'resampled at "%s" with %s.%s.%s' % (
                resolution, resampling.__module__, resampling.__class__.__name__, resampling.name)
        else:
            comment = 'read at full resolution'

        # arbitrary rio object, to get shape, etc ... (will not be used to read data)
        rio = rasterio.open(self.s1meta.files['measurement'].iloc[0])

        chunks['pol'] = 1
        # sort chunks keys like map_dims
        chunks = dict(sorted(chunks.items(), key=lambda pair: list(map_dims.keys()).index(pair[0])))
        chunks_rio = {map_dims[d]: chunks[d] for d in map_dims.keys()}

        if resolution is None:
            # using tiff driver: need to read individual tiff and concat them
            # riofiles['rio'] is ordered like self.s1meta.manifest_attrs['polarizations']

            dn = xr.concat(
                [
                    rioxarray.open_rasterio(
                        f, chunks=chunks_rio, parse_coordinates=False
                    ) for f in self.s1meta.files['measurement']
                ], 'band'
            ).assign_coords(band=np.arange(len(self.s1meta.manifest_attrs['polarizations'])) + 1)

            # set dimensions names
            dn = dn.rename(dict(zip(map_dims.values(), map_dims.keys())))

            # create coordinates from dimension index (because of parse_coordinates=False)
            dn = dn.assign_coords({'line': dn.line, 'sample': dn.sample})
            dn = dn.drop_vars('spatial_ref', errors='ignore')
        else:
            if not isinstance(resolution, dict):
                if isinstance(resolution, str) and resolution.endswith('m'):
                    resolution = float(resolution[:-1])
                resolution = dict(line=resolution / self.s1meta.pixel_line_m,
                                  sample=resolution / self.s1meta.pixel_sample_m)

            # resample the DN at gdal level, before feeding it to the dataset
            out_shape = (
                int(rio.height / resolution['line']),
                int(rio.width / resolution['sample'])
            )
            out_shape_pol = (1,) + out_shape
            # read resampled array in one chunk, and rechunk
            # this doesn't optimize memory, but total size remain quite small

            if isinstance(resolution['line'], int):
                # legacy behaviour: winsize is the maximum full image size that can be divided  by resolution (int)
                winsize = (0, 0, rio.width // resolution['sample'] * resolution['sample'],
                           rio.height // resolution['line'] * resolution['line'])
                window = rasterio.windows.Window(*winsize)
            else:
                window = None

            dn = xr.concat(
                [
                    xr.DataArray(
                        dask.array.from_array(
                            rasterio.open(f).read(
                                out_shape=out_shape_pol,
                                resampling=resampling,
                                window=window
                            ),
                            chunks=chunks_rio
                        ),
                        dims=tuple(map_dims.keys()), coords={'pol': [pol]}
                    ) for f, pol in
                    zip(self.s1meta.files['measurement'], self.s1meta.manifest_attrs['polarizations'])
                ],
                'pol'
            ).chunk(chunks)

            # create coordinates at box center
            translate = Affine.translation((resolution['sample'] - 1) / 2, (resolution['line'] - 1) / 2)
            scale = Affine.scale(
                rio.width // resolution['sample'] * resolution['sample'] / out_shape[1],
                rio.height // resolution['line'] * resolution['line'] / out_shape[0])
            sample, _ = translate * scale * (dn.sample, 0)
            _, line = translate * scale * (0, dn.line)
            dn = dn.assign_coords({'line': line, 'sample': sample})

        # for GTiff driver, pols are already ordered. just rename them
        dn = dn.assign_coords(pol=self.s1meta.manifest_attrs['polarizations'])

        if not all(self.s1meta.denoised.values()):
            descr = 'denoised'
        else:
            descr = 'not denoised'
        var_name = 'digital_number'

        dn.attrs = {
            'comment': '%s digital number, %s' % (descr, comment),
            'history': yaml.safe_dump(
                {
                    var_name: get_glob(
                        [p.replace(self.s1meta.path + '/', '') for p in self.s1meta.files['measurement']])
                }
            )
        }
        ds = dn.to_dataset(name=var_name)
        astype = self._dtypes.get(var_name)
        if astype is not None:
            ds = ds.astype(self._dtypes[var_name])

        return ds

    @timing
    def _load_from_geoloc(self, varnames):
        """
        Interpolate (with RectBiVariateSpline) variables from `self.s1meta.geoloc` to `self._dataset`

        Parameters
        ----------
        varnames: list of str
            subset of variables names in `self.s1meta.geoloc`

        Returns
        -------
        xarray.Dataset
            With interpolated vaiables

        """
        mapping_dataset_geoloc = {'latitude': 'latitude',
            'longitude': 'longitude',
            'incidence': 'incidenceAngle',
            'elevation': 'elevationAngle',
            'altitude': 'height',
            'azimuth_time': 'azimuthTime',
            'slant_range_time': 'slantRangeTime'}
        da_list = []

        def interp_func_slc(vect1dazti, vect1dxtrac, **kwargs):
            """

            Parameters
            ----------
            vect1dazti (np.ndarray) : azimuth times at high resolution
            vect1dxtrac (np.ndarray): range coords

            Returns
            -------

            """
            # exterieur de boucle
            rbs = kwargs['rbs']

            def wrapperfunc(*args, **kwargs):
                rbs2 = args[2]
                return rbs2(args[0], args[1], grid=False)

            return wrapperfunc(vect1dazti[:, np.newaxis], vect1dxtrac[np.newaxis, :], rbs)

        for varname in varnames:
            varname_in_geoloc = mapping_dataset_geoloc[varname]
            if varname in ['azimuth_time']:
                z_values = self.s1meta.geoloc[varname_in_geoloc].astype(float)
            elif varname == 'longitude':
                z_values = self.s1meta.geoloc[varname_in_geoloc]
                if self.s1meta.cross_antemeridian:
                    logger.debug('translate longitudes between 0 and 360')
                    z_values = z_values % 360
            else:
                z_values = self.s1meta.geoloc[varname_in_geoloc]
            if self.s1meta._bursts['burst'].size != 0:
                # TOPS SLC
                rbs = RectBivariateSpline(
                    self.s1meta.geoloc.azimuthTime[:, 0].astype(float),
                    self.s1meta.geoloc.sample,
                    z_values,
                    kx=1, ky=1,
                )
                interp_func = interp_func_slc
            else:
                rbs = None
                interp_func = RectBivariateSpline(
                    self.s1meta.geoloc.line,
                    self.s1meta.geoloc.sample,
                    z_values,
                    kx=1, ky=1
                )
            # the following take much cpu and memory, so we want to use dask
            # interp_func(self._dataset.line, self.dataset.sample)
            typee = self.s1meta.geoloc[varname_in_geoloc].dtype
            if self.s1meta._bursts['burst'].size != 0:
                datemplate = self._da_tmpl.astype(typee).copy()
                # replace the line coordinates by line_time coordinates
                datemplate = datemplate.assign_coords({'line': datemplate.coords['line_time']})
                da_var = map_blocks_coords(
                    datemplate,
                    interp_func,
                    func_kwargs={"rbs": rbs}
                )
                # put back the real line coordinates
                da_var = da_var.assign_coords({'line': self._dataset.digital_number.line})
            else:
                da_var = map_blocks_coords(
                    self._da_tmpl.astype(typee),
                    interp_func
                )
            if varname == 'longitude':
                if self.s1meta.cross_antemeridian:
                    da_var.data = da_var.data.map_blocks(to_lon180)

            da_var.name = varname

            # copy history
            try:
                da_var.attrs['history'] = self.s1meta.geoloc[varname_in_geoloc].attrs['history']
            except KeyError:
                pass

            da_list.append(da_var)

        return xr.merge(da_list)

    @timing
    def _load_ground_heading(self):
        def coords2heading(lines, samples):
            return self.s1meta.coords2heading(lines, samples, to_grid=True, approx=True)

        gh = map_blocks_coords(
            self._da_tmpl.astype(self._dtypes['ground_heading']),
            coords2heading,
            name='ground_heading'
        )

        gh.attrs = {
            'comment': 'at ground level, computed from lon/lat in line direction'
        }

        return gh.to_dataset(name='ground_heading')

    @timing
    def _load_rasterized_masks(self):

        def _rasterize_mask_by_chunks(line, sample, mask='land'):
            chunk_coords = bbox_coords(line, sample, pad=None)
            # chunk footprint polygon, in dataset coordinates (with buffer, to enlarge a little the footprint)
            chunk_footprint_coords = Polygon(chunk_coords).buffer(10)
            # chunk footprint polygon, in lon/lat
            chunk_footprint_ll = self.s1meta.coords2ll(chunk_footprint_coords)

            # get vector mask over chunk, in lon/lat
            vector_mask_ll = self.s1meta.get_mask(mask).intersection(chunk_footprint_ll)

            if vector_mask_ll.is_empty:
                # no intersection with mask, return zeros
                return np.zeros((line.size, sample.size))

            # vector mask, in line/sample coordinates
            vector_mask_coords = self.s1meta.ll2coords(vector_mask_ll)

            # shape of the returned chunk
            out_shape = (line.size, sample.size)

            # transform * (x, y) -> (line, sample)
            # (where (x, y) are index in out_shape)
            # Affine.permutation() is used because (line, sample) is transposed from geographic

            transform = Affine.translation(*chunk_coords[0]) * Affine.scale(
                *[np.unique(np.diff(c))[0] for c in [line, sample]]) * Affine.permutation()

            raster_mask = rasterio.features.rasterize(
                [vector_mask_coords],
                out_shape=out_shape,
                all_touched=False,
                transform=transform
            )
            return raster_mask

        da_list = []
        for mask in self.s1meta.mask_names:
            da_mask = map_blocks_coords(
                self._da_tmpl,
                _rasterize_mask_by_chunks,
                func_kwargs={'mask': mask}
            )
            name = '%s_mask' % mask
            da_mask.attrs['history'] = yaml.safe_dump({name: self.s1meta.get_mask(mask, describe=True)})
            da_list.append(da_mask.to_dataset(name=name))

        return xr.merge(da_list)

    def add_rasterized_masks(self):
        self._rasterized_masks = self._load_rasterized_masks()
        self.datatree['high_resolution_dataset'].ds = xr.merge([self.datatree['high_resolution_dataset'].ds,self._rasterized_masks])

    @timing
    def map_raster(self, raster_ds):
        """
        Map a raster onto xsar grid

        Parameters
        ----------
        raster_ds: xarray.Dataset or xarray.DataArray
            The dataset we want to project onto xsar grid. The `raster_ds.rio` accessor must be valid.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The projected dataset, with 'line' and 'sample' coordinate (same size as xsar dataset), and with valid `.rio` accessor.


        """
        if not raster_ds.rio.crs.is_geographic:
            raster_ds = raster_ds.rio.reproject(4326)

        if self.s1meta.cross_antemeridian:
            raise NotImplementedError('Antimeridian crossing not yet checked')

        # get lon/lat box for xsar dataset
        lon1, lat1, lon2, lat2 = self.s1meta.footprint.exterior.bounds
        lon_range = [lon1, lon2]
        lat_range = [lat1, lat2]

        # ensure dims ordering
        raster_ds = raster_ds.transpose('y', 'x')

        # ensure coords are increasing ( for RectBiVariateSpline )
        for coord in ['x', 'y']:
            if raster_ds[coord].values[-1] < raster_ds[coord].values[0]:
                raster_ds = raster_ds.reindex({coord : raster_ds[coord][::-1]})

        # from lon/lat box in xsar dataset, get the corresponding box in raster_ds (by index)
        ilon_range = [
            np.searchsorted(raster_ds.x.values, lon_range[0]),
            np.searchsorted(raster_ds.x.values, lon_range[1])
        ]
        ilat_range = [
            np.searchsorted(raster_ds.y.values, lat_range[0]),
            np.searchsorted(raster_ds.y.values, lat_range[1])
        ]
        # enlarge the raster selection range, for correct interpolation
        ilon_range, ilat_range = [[rg[0] - 1, rg[1] + 1] for rg in (ilon_range, ilat_range)]

        # select the xsar box in the raster
        raster_ds = raster_ds.isel(x=slice(*ilon_range), y=slice(*ilat_range))

        # upscale coordinates, in original projection
        # 1D array of lons/lats, trying to have same spacing as dataset (if not to high)
        num = min((self._dataset.sample.size + self._dataset.line.size) // 2, 1000)
        lons = np.linspace(*lon_range, num=num)
        lats = np.linspace(*lat_range, num=num)

        name = None
        if isinstance(raster_ds, xr.DataArray):
            # convert to temporary dataset
            name = raster_ds.name or '_tmp_name'
            raster_ds = raster_ds.to_dataset(name=name)

        mapped_ds_list = []
        for var in raster_ds:
            raster_da = raster_ds[var].chunk(raster_ds[var].shape)
            # upscale in original projection using interpolation
            # in most cases, RectBiVariateSpline give better results, but can't handle Nans
            if np.any(np.isnan(raster_da)):
                upscaled_da = raster_da.interp(x=lons, y=lats)
            else:
                upscaled_da = map_blocks_coords(
                    xr.DataArray(dims=['y', 'x'], coords={'x': lons, 'y': lats}).chunk(1000),
                    RectBivariateSpline(
                        raster_da.y.values,
                        raster_da.x.values,
                        raster_da.values,
                        kx=3, ky=3
                    )
                )
            upscaled_da.name = var
            # interp upscaled_da on sar grid
            mapped_ds_list.append(
                upscaled_da.interp(
                    x=self._dataset.longitude,
                    y=self._dataset.latitude
                ).drop_vars(['x', 'y'])
            )
        mapped_ds = xr.merge(mapped_ds_list)

        if name is not None:
            # convert back to dataArray
            mapped_ds = mapped_ds[name]
            if name == '_tmp_name':
                mapped_ds.name = None
        return self._set_rio(mapped_ds)

    @timing
    def _load_rasters_vars(self):
        # load and map variables from rasterfile (like ecmwf) on dataset
        if self.s1meta.rasters.empty:
            return None
        else:
            logger.warning('Raster variable are experimental')

        if self.s1meta.cross_antemeridian:
            raise NotImplementedError('Antimeridian crossing not yet checked')

        # will contain xr.DataArray to merge
        da_var_list = []

        for name, infos in self.s1meta.rasters.iterrows():
            # read the raster file using helpers functions
            read_function = infos['read_function']
            get_function = infos['get_function']
            resource = infos['resource']

            kwargs = {
                's1meta': self,
                'date': self.s1meta.start_date,
                'footprint': self.s1meta.footprint
            }

            logger.debug('adding raster "%s" from resource "%s"' % (name, str(resource)))
            if get_function is not None:
                try:
                    resource_dec = get_function(resource, **kwargs)
                except TypeError:
                    resource_dec = get_function(resource)

            if read_function is None:
                raster_ds = xr.open_dataset(resource_dec, chunk=1000)
            else:
                # read_function should return a chunked dataset (so it's fast)
                raster_ds = read_function(resource_dec)

            # add globals raster attrs to globals dataset attrs
            hist_res = {'resource': resource}
            if get_function is not None:
                hist_res.update({'resource_decoded': resource_dec})

            reprojected_ds = self.map_raster(raster_ds).rename({v: '%s_%s' % (name, v) for v in raster_ds})

            for v in reprojected_ds:
                reprojected_ds[v].attrs['history'] = yaml.safe_dump({v: hist_res})

            da_var_list.append(reprojected_ds)
        return xr.merge(da_var_list)

    def _get_lut(self, var_name):
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

    def _apply_calibration_lut(self, var_name):
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
        lut = self._get_lut(var_name)
        res = (np.abs(self._dataset.digital_number) ** 2. / (lut ** 2))
        # dn default value is 0: convert to Nan
        res = res.where(res > 0)
        astype = self._dtypes.get(var_name)
        if astype is not None:
            res = res.astype(astype)

        res.attrs.update(lut.attrs)
        res.attrs['history'] = merge_yaml([lut.attrs['history']], section=var_name)
        res.attrs['references'] = 'https://sentinel.esa.int/web/sentinel/radiometric-calibration-of-level-1-products'

        return res.to_dataset(name=var_name)

    def reverse_calibration_lut(self, ds_var):
        """
        TODO: replace ds_var by var_name
        Inverse of `_apply_calibration_lut` : from `var_name`, reverse apply lut, to get digital_number.
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

        # resize lut with same a/sample as da_var
        lut = lut.sel(line=da_var.line, sample=da_var.sample, method='nearest')
        # as we used 'nearest', force exact coords
        lut['line'] = da_var.line
        lut['sample'] = da_var.sample
        # waiting for https://github.com/pydata/xarray/pull/4155
        # lut = lut.interp(line=da_var.line, sample=da_var.sample)

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

    def _get_noise(self, var_name):
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
        lut = self._get_lut(var_name)
        dataarr = noise_lut / lut ** 2
        name = 'ne%sz' % var_name[0]
        astype = self._dtypes.get(name)
        if astype is not None:
            dataarr = dataarr.astype(astype)
        dataarr.attrs['history'] = merge_yaml([lut.attrs['history'], noise_lut.attrs['history']], section=name)
        return dataarr.to_dataset(name=name)

    def _add_denoised(self, ds, clip=False, vars=None):
        """add denoised vars to dataset

        Parameters
        ----------
        ds : xarray.DataSet
            dataset with non denoised vars, named `%s_raw`.
        clip : bool, optional
            If True, negative signal will be clipped to 0. (default to False )
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
                denoised.attrs['history'] = merge_yaml(
                    [ds[varname_raw].attrs['history'], ds[noise].attrs['history']],
                    section=varname
                )
                if clip:
                    denoised = denoised.clip(min=0)
                    denoised.attrs['comment'] = 'clipped, no values <0'
                else:
                    denoised.attrs['comment'] = 'not clipped, some values can be <0'
                ds[varname] = denoised
        return ds

    @property
    def get_burst_azitime(self):
        """
        Get azimuth time at high resolution.

        Returns
        -------
        xarray.DataArray
            the high resolution azimuth time vector interpolated at the middle of the sub-swath
        """
        #azitime = self.s1meta._burst_azitime()
        azitime = self._burst_azitime()
        iz = np.searchsorted(azitime.line, self._dataset.line)
        azitime = azitime.isel({'line': iz})
        azitime = azitime.assign_coords({"line": self._dataset.line})
        return azitime

    def _get_sensor_velocity(self):
        """
        Interpolated sensor velocity
        Returns
        -------
        xarray.Dataset()
            containing a single variable velocity
        """

        azimuth_times = self.get_burst_azitime
        orbstatevect = self.s1meta.orbit
        azi_times = orbstatevect['time'].values
        velos = np.array([orbstatevect['velocity_x'] ** 2.,orbstatevect['velocity_y'] ** 2.,orbstatevect['velocity_z'] ** 2.])
        vels = np.sqrt(np.sum(velos, axis=0))
        interp_f = interp1d(azi_times.astype(float), vels)
        _vels = interp_f(azimuth_times.astype(float))
        res = xr.DataArray(_vels, dims=['line'], coords={'line': self.dataset.line})
        return xr.Dataset({'velocity': res})

    def _range_ground_spacing(self):
        """
        Get SAR image range ground spacing.

        Parameters
        ----------
        Returns
        -------
        range_ground_spacing_vect : xarray.DataArray
            range ground spacing (sample coordinates)

        Notes
        -----
        For GRD products is it the same same value along sample axis
        """
        ground_spacing = np.array([self.s1meta.image['azimuthPixelSpacing'],self.s1meta.image['slantRangePixelSpacing']])
        if self.s1meta.product == 'SLC':
            line_tmp = self._dataset['line']
            sample_tmp = self._dataset['sample']
            # get the incidence at the middle of line dimension of the part of image selected
            inc = self._dataset['incidence'].isel({'line': int(len(line_tmp) / 2),
                                                   })
            range_ground_spacing_vect = ground_spacing[1] / np.sin(np.radians(inc))
            range_ground_spacing_vect.attrs['history'] = ''

        else:  # GRD
            valuess = np.ones((len(self._dataset['sample']))) * ground_spacing[1]
            range_ground_spacing_vect = xr.DataArray(valuess, coords={'sample': self._dataset['sample']},
                                                     dims=['sample'])
        return xr.Dataset({'range_ground_spacing': range_ground_spacing_vect})

    @property
    def _local_gcps(self):
        # get local gcps, for rioxarray.reproject (row and col are *index*, not coordinates)
        local_gcps = []
        for line in self.dataset.line.values[::int(self.dataset.line.size / 20)+1]:
            for sample in self.dataset.sample.values[::int(self.dataset.sample.size / 20)+1]:
                irow = np.argmin(np.abs(self.dataset.line.values - line))
                icol = np.argmin(np.abs(self.dataset.sample.values - sample))
                lon, lat = self.s1meta.coords2ll(line, sample)
                gcp = GroundControlPoint(
                    x=lon,
                    y=lat,
                    z=0,
                    col=icol,
                    row=irow
                )
                local_gcps.append(gcp)
        return local_gcps

    def __repr__(self):
        if self.sliced:
            intro = "sliced"
        else:
            intro = "full covevage"
        return "<Sentinel1Dataset %s object>" % intro

    def _repr_mimebundle_(self, include=None, exclude=None):
        return repr_mimebundle(self, include=include, exclude=exclude)

    def get_burst_valid_location(self):
        """
        add a field 'valid_location' in the bursts sub-group of the datatree
        :return:
        """
        nbursts = len(self.datatree['bursts'].ds['burst'])
        burst_firstValidSample = self.datatree['bursts'].ds['firstValidSample'].values
        burst_lastValidSample = self.datatree['bursts'].ds['lastValidSample'].values
        valid_locations = np.empty((nbursts, 4), dtype='int32')
        line_per_burst = len(self.datatree['bursts'].ds['line'])
        for ibur in range(nbursts):
            fvs = burst_firstValidSample[ibur, :]
            lvs = burst_lastValidSample[ibur, :]
            # valind = np.where((fvs != -1) | (lvs != -1))[0]
            valind = np.where(np.isfinite(fvs) | np.isfinite(lvs))[0]
            valloc = [ibur * line_per_burst + valind.min(), fvs[valind].min(),
                      ibur * line_per_burst + valind.max(), lvs[valind].max()]
            valid_locations[ibur, :] = valloc
        tmpda = xr.DataArray(dims=['burst', 'limits'],
                                    coords={'burst':self.datatree['bursts'].ds['burst'].values,'limits':np.arange(4)},
                                    data=valid_locations,
                                    attrs={
                           'description': 'start line index, start sample index, stop line index, stop sample index'})
        self.datatree['bursts'].ds['valid_location'] = tmpda

    def get_bursts_polygons(self, only_valid_location=True):
        """
        get the polygons of radar bursts in the image geometry

        Parameters
        ----------
        only_valid_location : bool
            [True] -> polygons of the TOPS SLC bursts are cropped using valid location index
            False -> polygons of the TOPS SLC bursts are aligned with azimuth time start/stop index

        Returns
        -------
        geopandas.GeoDataframe
            polygons of the burst in the image (ie line/sample) geometry
            'geometry' is the polygon

        """
        if self.s1meta.multidataset:
            blocks_list = []
            # for subswath in self.subdatasets.index:
            for submeta in self.s1meta._submeta:
                block = submeta.bursts(only_valid_location=only_valid_location)
                block['subswath'] = submeta.dsid
                block = block.set_index('subswath', append=True).reorder_levels(['subswath', 'burst'])
                blocks_list.append(block)
            blocks = pd.concat(blocks_list)
        else:
            #burst_list = self._bursts
            self.get_burst_valid_location()
            burst_list = self.datatree['bursts'].ds
            nb_samples = self.datatree.attrs['numberOfSamples']
            if burst_list['burst'].size == 0:
                blocks = gpd.GeoDataFrame()
            else:
                bursts = []
                bursts_az_inds = {}
                inds_burst, geoloc_azitime, geoloc_iburst, geoloc_line = self._get_indices_bursts()
                for burst_ind, uu in enumerate(np.unique(inds_burst)):
                    if only_valid_location:
                        extent = np.copy(burst_list['valid_location'].values[burst_ind, :])
                        area = box(extent[0], extent[1], extent[2], extent[3])

                    else:
                        inds_one_val = np.where(inds_burst == uu)[0]
                        bursts_az_inds[uu] = inds_one_val
                        area = box(bursts_az_inds[burst_ind][0], 0, bursts_az_inds[burst_ind][-1], nb_samples)
                    burst = pd.Series(dict([
                        ('geometry_image', area)]))
                    bursts.append(burst)
                # to geopandas
                blocks = pd.concat(bursts, axis=1).T
                blocks = gpd.GeoDataFrame(blocks)
                blocks['geometry'] = blocks['geometry_image'].apply(self.coords2ll)
                blocks.index.name = 'burst'
        return blocks

    def _get_indices_bursts(self):
        """

        Returns
        -------
        ind np.array
            index of the burst start in the line coordinates
        geoloc_azitime np.array
            azimuth time at the middle of the image from geolocation grid (low resolution)
        geoloc_iburst np.array

        """
        ind = None
        geoloc_azitime = None
        geoloc_iburst = None
        geoloc_line = None
        if self.s1meta.product == 'SLC' and 'WV' not in self.s1meta.swath:
        #if self.datatree.attrs['product'] == 'SLC' and 'WV' not in self.datatree.attrs['swath']:
            burst_nlines = self.s1meta._bursts.attrs['line_per_burst']
            #burst_nlines = self.datatree['bursts'].ds['line'].size

            geoloc_line = self.s1meta.geoloc['line'].values
            #geoloc_line = self.datatree['geolocation_annotation'].ds['line'].values
            # find the indice of the bursts in the geolocation grid
            geoloc_iburst = np.floor(geoloc_line / float(burst_nlines)).astype('int32')
            # find the indices of the bursts in the high resolution grid
            line = np.arange(0, self.s1meta.image['numberOfLines'])
            #line = np.arange(0, self.datatree.attrs['numberOfLines'])
            iburst = np.floor(line / float(burst_nlines)).astype('int32')
            # find the indices of the burst transitions
            ind = np.searchsorted(geoloc_iburst, iburst, side='left')
            n_pixels = int((len(self.s1meta.geoloc['sample']) - 1) / 2)
            geoloc_azitime = self.s1meta.geoloc['azimuthTime'].values[:, n_pixels]
            # security check for unrealistic line_values exceeding the image extent
            if ind.max() >= len(geoloc_azitime):
                ind[ind >= len(geoloc_azitime)] = len(geoloc_azitime) - 1
        return ind, geoloc_azitime, geoloc_iburst, geoloc_line


    def _burst_azitime(self):
        """
        Get azimuth time at high resolution on the full image shape

        Returns
        -------
        np.ndarray
            the high resolution azimuth time vector interpolated at the midle of the subswath
        """
        line = np.arange(0, self.s1meta.image['numberOfLines'])
        #line = np.arange(0,self.datatree.attrs['numberOfLines'])
        if self.s1meta.product == 'SLC' and 'WV' not in self.s1meta.swath:
        #if self.datatree.attrs['product'] == 'SLC' and 'WV' not in self.datatree.attrs['swath']:
            azi_time_int = self.s1meta.image['azimuthTimeInterval']
            #azi_time_int = self.datatree.attrs['azimuthTimeInterval']
            # turn this interval float/seconds into timedelta/picoseconds
            azi_time_int = np.timedelta64(int(azi_time_int * 1e12), 'ps')
            ind, geoloc_azitime, geoloc_iburst, geoloc_line = self._get_indices_bursts()
            # compute the azimuth time by adding a step function (first term) and a growing term (second term)
            azitime = geoloc_azitime[ind] + (line - geoloc_line[ind]) * azi_time_int.astype('<m8[ns]')
        else:  # GRD* cases
            # n_pixels = int((len(self.datatree['geolocation_annotation'].ds['sample']) - 1) / 2)
            # geoloc_azitime = self.datatree['geolocation_annotation'].ds['azimuth_time'].values[:, n_pixels]
            # geoloc_line = self.datatree['geolocation_annotation'].ds['line'].values
            n_pixels = int((len(self.s1meta.geoloc['sample']) - 1) / 2)
            geoloc_azitime = self.s1meta.geoloc['azimuthTime'].values[:, n_pixels]
            geoloc_line = self.s1meta.geoloc['line'].values
            finterp = interp1d(geoloc_line, geoloc_azitime.astype(float))
            azitime = finterp(line)
            azitime = azitime.astype('<m8[ns]')
        azitime = xr.DataArray(azitime, coords={'line': line}, dims=['line'],
                               attrs={
                                   'description': 'azimuth times interpolated along line dimension at the middle of range dimension'})

        return azitime


