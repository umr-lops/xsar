# -*- coding: utf-8 -*-
import logging
import warnings
import xarray.core.formatting as formatting
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
        resampling dict like `{'atrack': 20, 'xtrack': 20}` where 20 is in pixels.

        if a number, dict will be constructed from `{'atrack': number, 'xtrack': number}`

        if str, it must end with 'm' (meters), like '100m'. dict will be computed from sensor pixel size.

    resampling: rasterio.enums.Resampling or str, optional

        Only used if `resolution` is not None.

        ` rasterio.enums.Resampling.rms` by default. `rasterio.enums.Resampling.nearest` (decimation) is fastest.

    luts: bool, optional

        if `True` return also luts as variables (ie `sigma0_lut`, `gamma0_lut`, etc...). False by default.

    chunks: dict, optional

        dict with keys ['pol','atrack','xtrack'] (dask chunks).

    dtypes: None or dict, optional

        Specify the data type for each variable.

    patch_variable: bool, optional

        activate or not variable pathching ( currently noise lut correction for IPF2.9X)

    """

    def __init__(self, dataset_id, resolution=None,
                 resampling=rasterio.enums.Resampling.rms,
                 luts=False, chunks={'atrack': 5000, 'xtrack': 5000},
                 dtypes=None, patch_variable=True):

        # miscellaneous attributes that are not know from xml files
        attrs_dict = {
            'pol': {
                'comment': 'ordered polarizations (copol, crosspol)'
            },
            'atrack': {
                'units': '1',
                'comment': 'azimuth direction, in pixels from full resolution tiff'
            },
            'xtrack': {
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

        self._dataset = self._load_digital_number(resolution=resolution, resampling=resampling, chunks=chunks)
        self._dataset = xr.merge([xr.Dataset({'time': self._burst_azitime}), self._dataset])

        # dataset no-pol template for function evaluation on coordinates (*no* values used)
        # what's matter here is the shape of the image, not the values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            if self.s1meta._bursts['burst'].size != 0:
                # SLC TOPS, tune the high res grid because of bursts overlapping
                atrack_time = self._burst_azitime
                self._da_tmpl = xr.DataArray(
                    dask.array.empty_like(
                        np.empty((len(atrack_time), len(self._dataset.digital_number.xtrack))),
                        dtype=np.int8, name="empty_var_tmpl-%s" % dask.base.tokenize(self.s1meta.name)),
                    dims=('atrack', 'xtrack'),
                    coords={
                        'atrack': self._dataset.digital_number.atrack,
                        'xtrack': self._dataset.digital_number.xtrack,
                        'atrack_time': atrack_time.astype(float),
                    },
                )
            else:

                self._da_tmpl = xr.DataArray(
                    dask.array.empty_like(
                        self._dataset.digital_number.isel(pol=0).drop('pol'),
                        dtype=np.int8, name="empty_var_tmpl-%s" % dask.base.tokenize(self.s1meta.name)),
                    dims=('atrack', 'xtrack'),
                    coords={'atrack': self._dataset.digital_number.atrack,
                            'xtrack': self._dataset.digital_number.xtrack}
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

        self._rasterized_masks = self._load_rasterized_masks()

        ds_merge_list = [self._dataset, self._rasterized_masks, self._load_ground_heading(),  # lon_lat
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
        self._dataset = self._add_denoised(self._dataset,clip=True)
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

    def __del__(self):
        logger.debug('__del__')

    @property
    def dataset(self):
        """
        `xarray.Dataset` representation of this `xsar.Sentinel1Dataset` object.
        This property can be set with a new dataset, if the dataset was computed from the original dataset.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, ds):
        if self.s1meta.name == ds.attrs['name']:
            # check if new ds has changed coordinates
            if not self.sliced:
                self.sliced = any(
                    [list(ds[d].values) != list(self._dataset[d].values) for d in ['atrack', 'xtrack']])
            self._dataset = ds
            self.recompute_attrs()
        else:
            raise ValueError("dataset must be same kind as original one.")

    @dataset.deleter
    def dataset(self):
        logger.debug('deleter dataset')

    @property
    def _bbox_coords(self):
        """
        Dataset bounding box, in atrack/xtrack coordinates
        """
        bbox_ext = bbox_coords(self.dataset.atrack.values, self.dataset.xtrack.values)
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
        Get `(atracks, xtracks)` from `(lon, lat)`,
        or convert a lon/lat shapely shapely object to atrack/xtrack coordinates.

        Parameters
        ----------
        *args: lon, lat or shapely object

            lon and lat might be iterables or scalars

        Returns
        -------
        tuple of np.array or tuple of float (atracks, xtracks) , or a shapely object

        Notes
        -----
        The difference with `xsar.Sentinel1Meta.ll2coords` is that coordinates are rounded to the nearest dataset coordinates.

        See Also
        --------
        xsar.Sentinel1Meta.ll2coords

        """
        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self.s1meta.ll2coords_shapely(args[0].intersection(self.geometry))

        atrack, xtrack = self.s1meta.ll2coords(*args)

        if hasattr(args[0], '__iter__'):
            scalar = False
        else:
            scalar = True

        tolerance = np.max([np.percentile(np.diff(self.dataset[c].values), 90) / 2 for c in ['atrack', 'xtrack']]) + 1
        try:
            # select the nearest valid pixel in ds
            ds_nearest = self.dataset.sel(atrack=atrack, xtrack=xtrack, method='nearest', tolerance=tolerance)
            if scalar:
                (atrack, xtrack) = (ds_nearest.atrack.values.item(), ds_nearest.xtrack.values.item())
            else:
                (atrack, xtrack) = (ds_nearest.atrack.values, ds_nearest.xtrack.values)
        except KeyError:
            # out of bounds, because of `tolerance` keyword
            (atrack, xtrack) = (atrack * np.nan, xtrack * np.nan)

        return atrack, xtrack

    def coords2ll(self, *args, **kwargs):
        """
         Alias for `xsar.Sentinel1Meta.coords2ll`

         See Also
         --------
         xsar.Sentinel1Meta.coords2ll
        """
        return self.s1meta.coords2ll(*args, **kwargs)

    @property
    def len_atrack_m(self):
        """atrack length, in meters"""
        bbox_ll = list(zip(*self._bbox_ll))
        len_m, _ = haversine(*bbox_ll[1], *bbox_ll[2])
        return len_m

    @property
    def len_xtrack_m(self):
        """xtrack length, in meters """
        bbox_ll = list(zip(*self._bbox_ll))
        len_m, _ = haversine(*bbox_ll[0], *bbox_ll[1])
        return len_m

    @property
    def pixel_atrack_m(self):
        """atrack pixel spacing, in meters (relative to dataset)"""
        return self.s1meta.pixel_atrack_m * np.unique(np.round(np.diff(self._dataset['atrack'].values), 1))[0]

    @property
    def pixel_xtrack_m(self):
        """xtrack pixel spacing, in meters (relative to dataset)"""
        return self.s1meta.pixel_xtrack_m * np.unique(np.round(np.diff(self._dataset['xtrack'].values), 1))[0]

    @property
    def coverage(self):
        """coverage string"""
        return "%dkm * %dkm (atrack * xtrack )" % (self.len_atrack_m / 1000, self.len_xtrack_m / 1000)

    @property
    def _regularly_spaced(self):
        return max(
            [np.unique(np.round(np.diff(self._dataset[dim].values), 1)).size for dim in ['atrack', 'xtrack']]) == 1

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
            if set(['atrack', 'xtrack']).issubset(set(ds[v].dims)):
                ds[v] = ds[v].set_index({'xtrack': 'xtrack', 'atrack': 'atrack'})
                ds[v] = ds[v].rio.write_gcps(
                    gcps, 'epsg:4326', inplace=True
                ).rio.set_spatial_dims(
                    'xtrack', 'atrack', inplace=True
                ).rio.write_coordinate_system(
                    inplace=True)
                # remove/reset some incorrect attrs set by rio
                # (long_name is 'latitude', but it's incorrect for atrack axis ...)
                for ax in ['atrack', 'xtrack']:
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

        >>> xsar_obj.dataset = xsar_obj.dataset.isel(atrack=slice(1000,5000))
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
        attrs['pixel_xtrack_m'] = self.pixel_xtrack_m
        attrs['pixel_atrack_m'] = self.pixel_atrack_m
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
            'atrack': 'y',
            'xtrack': 'x'
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
            dn = dn.assign_coords({'atrack': dn.atrack, 'xtrack': dn.xtrack})
            dn = dn.drop_vars('spatial_ref', errors='ignore')
        else:
            if not isinstance(resolution, dict):
                if isinstance(resolution, str) and resolution.endswith('m'):
                    resolution = float(resolution[:-1])
                resolution = dict(atrack=resolution / self.s1meta.pixel_atrack_m,
                                  xtrack=resolution / self.s1meta.pixel_xtrack_m)

            # resample the DN at gdal level, before feeding it to the dataset
            out_shape = (
                int(rio.height / resolution['atrack']),
                int(rio.width / resolution['xtrack'])
            )
            out_shape_pol = (1,) + out_shape
            # read resampled array in one chunk, and rechunk
            # this doesn't optimize memory, but total size remain quite small

            if isinstance(resolution['atrack'], int):
                # legacy behaviour: winsize is the maximum full image size that can be divided  by resolution (int)
                winsize = (0, 0, rio.width // resolution['xtrack'] * resolution['xtrack'],
                           rio.height // resolution['atrack'] * resolution['atrack'])
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
            translate = Affine.translation((resolution['xtrack'] - 1) / 2, (resolution['atrack'] - 1) / 2)
            scale = Affine.scale(
                rio.width // resolution['xtrack'] * resolution['xtrack'] / out_shape[1],
                rio.height // resolution['atrack'] * resolution['atrack'] / out_shape[0])
            xtrack, _ = translate * scale * (dn.xtrack, 0)
            _, atrack = translate * scale * (0, dn.atrack)
            dn = dn.assign_coords({'atrack': atrack, 'xtrack': xtrack})

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
            if varname in ['azimuth_time']:
                z_values = self.s1meta.geoloc[varname].astype(float)
            elif varname == 'longitude':
                z_values = self.s1meta.geoloc[varname]
                if self.s1meta.cross_antemeridian:
                    logger.debug('translate longitudes between 0 and 360')
                    z_values = z_values % 360
            else:
                z_values = self.s1meta.geoloc[varname]
            if self.s1meta._bursts['burst'].size != 0:
                # TOPS SLC
                rbs = RectBivariateSpline(
                    self.s1meta.geoloc.azimuth_time[:, 0].astype(float),
                    self.s1meta.geoloc.xtrack,
                    z_values,
                    kx=1, ky=1,
                )
                interp_func = interp_func_slc
            else:
                rbs = None
                interp_func = RectBivariateSpline(
                    self.s1meta.geoloc.atrack,
                    self.s1meta.geoloc.xtrack,
                    z_values,
                    kx=1, ky=1
                )
            # the following take much cpu and memory, so we want to use dask
            # interp_func(self._dataset.atrack, self.dataset.xtrack)
            typee = self.s1meta.geoloc[varname].dtype
            if self.s1meta._bursts['burst'].size != 0:
                datemplate = self._da_tmpl.astype(typee).copy()
                # replace the atrack coordinates by atrack_time coordinates
                datemplate = datemplate.assign_coords({'atrack': datemplate.coords['atrack_time']})
                da_var = map_blocks_coords(
                    datemplate,
                    interp_func,
                    func_kwargs={"rbs": rbs}
                )
                # put back the real atrack coordinates
                da_var = da_var.assign_coords({'atrack': self._dataset.digital_number.atrack})
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
                da_var.attrs['history'] = self.s1meta.geoloc[varname].attrs['history']
            except KeyError:
                pass

            da_list.append(da_var)

        return xr.merge(da_list)

    @timing
    def _load_ground_heading(self):
        def coords2heading(atracks, xtracks):
            return self.s1meta.coords2heading(atracks, xtracks, to_grid=True, approx=True)

        gh = map_blocks_coords(
            self._da_tmpl.astype(self._dtypes['ground_heading']),
            coords2heading,
            name='ground_heading'
        )

        gh.attrs = {
            'comment': 'at ground level, computed from lon/lat in atrack direction'
        }

        return gh.to_dataset(name='ground_heading')

    @timing
    def _load_rasterized_masks(self):

        def _rasterize_mask_by_chunks(atrack, xtrack, mask='land'):
            chunk_coords = bbox_coords(atrack, xtrack, pad=None)
            # chunk footprint polygon, in dataset coordinates (with buffer, to enlarge a little the footprint)
            chunk_footprint_coords = Polygon(chunk_coords).buffer(10)
            # chunk footprint polygon, in lon/lat
            chunk_footprint_ll = self.s1meta.coords2ll(chunk_footprint_coords)

            # get vector mask over chunk, in lon/lat
            vector_mask_ll = self.s1meta.get_mask(mask).intersection(chunk_footprint_ll)

            if vector_mask_ll.is_empty:
                # no intersection with mask, return zeros
                return np.zeros((atrack.size, xtrack.size))

            # vector mask, in atrack/xtrack coordinates
            vector_mask_coords = self.s1meta.ll2coords(vector_mask_ll)

            # shape of the returned chunk
            out_shape = (atrack.size, xtrack.size)

            # transform * (x, y) -> (atrack, xtrack)
            # (where (x, y) are index in out_shape)
            # Affine.permutation() is used because (atrack, xtrack) is transposed from geographic

            transform = Affine.translation(*chunk_coords[0]) * Affine.scale(
                *[np.unique(np.diff(c))[0] for c in [atrack, xtrack]]) * Affine.permutation()

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
            The projected dataset, with 'atrack' and 'xtrack' coordinate (same size as xsar dataset), and with valid `.rio` accessor.


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
        num = min((self._dataset.xtrack.size + self._dataset.atrack.size) // 2, 1000)
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
        #res = (np.abs(self._dataset.digital_number) ** 2. / (abs(lut) ** 2)) # test agrouaze
        # dn default value is 0: convert to Nan
        #res = res.where(res > 0)
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
        #dataarr = noise_lut / lut ** 2
        dataarr = abs(noise_lut) / lut ** 2 # test agrouaze
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
    def _burst_azitime(self):
        """
        Get azimuth time at high resolution.

        Returns
        -------
        xarray.DataArray
            the high resolution azimuth time vector interpolated at the middle of the sub-swath
        """
        azitime = self.s1meta._burst_azitime()
        iz = np.searchsorted(azitime.atrack, self._dataset.atrack)
        azitime = azitime.isel({'atrack': iz})
        azitime = azitime.assign_coords({"atrack": self._dataset.atrack})
        return azitime

    def _get_sensor_velocity(self):
        """
        Interpolated sensor velocity
        Returns
        -------
        xarray.Dataset()
            containing a single variable velocity
        """

        azimuth_times = self._burst_azitime
        orbstatevect = self.s1meta.orbit
        azi_times = orbstatevect.index.values
        velos = np.array([[uu.x ** 2., uu.y ** 2., uu.z ** 2.] for uu in orbstatevect['velocity'].values])
        vels = np.sqrt(np.sum(velos, axis=1))
        interp_f = interp1d(azi_times.astype(float), vels)
        _vels = interp_f(azimuth_times.astype(float))
        res = xr.DataArray(_vels, dims=['atrack'], coords={'atrack': self.dataset.atrack})
        return xr.Dataset({'velocity': res})

    def _range_ground_spacing(self):
        """
        Get SAR image range ground spacing.

        Parameters
        ----------
        Returns
        -------
        range_ground_spacing_vect : xarray.DataArray
            range ground spacing (xtrack coordinates)

        Notes
        -----
        For GRD products is it the same same value along xtrack axis
        """
        ground_spacing = np.array(self.s1meta.image['slant_pixel_spacing'])
        if self.s1meta.product == 'SLC':
            atrack_tmp = self._dataset['atrack']
            xtrack_tmp = self._dataset['xtrack']
            # get the incidence at the middle of atrack dimension of the part of image selected
            inc = self._dataset['incidence'].isel({'atrack': int(len(atrack_tmp) / 2),
                                                   })
            range_ground_spacing_vect = ground_spacing[1] / np.sin(np.radians(inc))
            range_ground_spacing_vect.attrs['history'] = ''

        else:  # GRD
            valuess = np.ones((len(self._dataset['xtrack']))) * ground_spacing[1]
            range_ground_spacing_vect = xr.DataArray(valuess, coords={'xtrack': self._dataset['xtrack']},
                                                     dims=['xtrack'])
        return xr.Dataset({'range_ground_spacing': range_ground_spacing_vect})

    @property
    def _local_gcps(self):
        # get local gcps, for rioxarray.reproject (row and col are *index*, not coordinates)
        local_gcps = []
        for atrack in self.dataset.atrack.values[::int(self.dataset.atrack.size / 20)+1]:
            for xtrack in self.dataset.xtrack.values[::int(self.dataset.xtrack.size / 20)+1]:
                irow = np.argmin(np.abs(self.dataset.atrack.values - atrack))
                icol = np.argmin(np.abs(self.dataset.xtrack.values - xtrack))
                lon, lat = self.s1meta.coords2ll(atrack, xtrack)
                gcp = GroundControlPoint(
                    x=lon,
                    y=lat,
                    z=0,
                    col=icol,
                    row=irow
                )
                local_gcps.append(gcp)
        return local_gcps


    def __repr__(self):# -> str:
    #     # if self.sliced:
    #     #     intro = "sliced"
    #     # else:
    #     #     intro = "full covevage"
    #     # intt = "<Sentinel1Dataset %s object>" % intro
    #     from IPython.core.display import HTML
    #     full_repr = f"<p>Sentinel1Dataset is a class to read Level-1 Sentinel-1 data\n <\p>" +self.dataset._repr_html_()
    #     #             "it contains:\n \
    #     #                -a dataset;:%s\n \
    #     #                -a meta: %s
    #     #             %s
    #     #             """%(self.dataset.__repr__,self.s1meta.__repr__,intt)
    #     #full_repr = formatting.data_vars_repr(self)
    #     #full_repr = self.dataset._repr_html_

        full_repr = self._repr_html_()
        return full_repr

    def _repr_html_(self):
        from IPython.core.display import HTML
        from xarray.core.formatting_html import collapsible_section, summarize_attrs
        collapsible = collapsible_section(
            "tututu",
            # details={'radis':3,'viande':7}
            details=summarize_attrs({'radis': 3, 'viande': 7, 'pain': 56.8})
            , n_items=3, enabled=True, collapsed=True
        )
        # collap_geoloc = collapsible_section(
        # "geoloc",
        #               details= summarize_attrs(self.s1meta.geoloc._repr_html_())
        #     ,n_items=1, enabled=True, collapsed=True
        # )
        list_meta = f"""<ul>
                       <li>s1meta.image</li>
                       <li>s1meta.azimuth_fmrate</li>
                       <li>s1meta._bursts</li>
                       <li>s1meta.orbit</li>
                       <li>s1meta.geoloc</li>
                       <li>s1meta.doppler_estimate</li>
                       </ul>
                       """  # .format(collap_geoloc)
        full_repr = f"<p>Sentinel1Dataset is a class to read Level-1 Sentinel-1 data <br /> {self.dataset._repr_html_()}" + \
                    " <b>Sentinel-1 META</b> <br>" + list_meta + '<br>'  # +collapsible
        #full_repr = self.__repr__()
        return full_repr

    def __str__(self):
        return 'toto'

    # def _repr_mimebundle_(self, include=None, exclude=None):
    #     return repr_mimebundle(self, include=include, exclude=exclude)
