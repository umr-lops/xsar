# -*- coding: utf-8 -*-
import logging
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import dask
import rasterio
import rasterio.features
import rioxarray
from scipy.interpolate import interp1d, RectBivariateSpline
from shapely.geometry import Polygon
import shapely
from .utils import timing, haversine, map_blocks_coords, bbox_coords, BlockingActorProxy, merge_yaml, get_glob
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

    See Also
    --------
    xsar.open_dataset
    """

    def __init__(self, dataset_id, resolution=None,
                 resampling=rasterio.enums.Resampling.rms,
                 luts=False, chunks={'atrack': 5000, 'xtrack': 5000},
                 dtypes=None):

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
                'units': 'm2/m2'
            },
            'gamma0_raw': {
                'units': 'm2/m2'
            },
            'nesz': {
                'units': 'm2/m2',
                'comment': 'sigma0 noise'
            },
            'negz': {
                'units': 'm2/m2',
                'comment': 'beta0 noise'
            },
        }

        # default dtypes
        self._dtypes = {
            'latitude': 'f4',
            'longitude': 'f4',
            'incidence': 'f4',
            'elevation': 'f4',
            'height': 'f4',
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
            'digital_number': None,
            'azimuth_time':None,
            'slant_range_time':None
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
            #import pickle
            #s1meta = pickle.loads(pickle.dumps(self.s1meta))
            #assert isinstance(s1meta.coords2ll(100, 100),tuple)
        else:
            # we want self.s1meta to be a dask actor on a worker
            self.s1meta = BlockingActorProxy(Sentinel1Meta.from_dict, dataset_id.dict)
        del dataset_id

        if self.s1meta.multidataset:
            raise IndexError(
                """Can't open an multi-dataset. Use `xsar.Sentinel1Meta('%s').subdatasets` to show availables ones""" % self.s1meta.path
            )

        self._dataset = self._load_digital_number(resolution=resolution, resampling=resampling, chunks=chunks)

        # set time(atrack) from s1meta.time_range
        time_range = self.s1meta.time_range
        atrack_time = interp1d(
            self._dataset.atrack[[0, -1]],
            [time_range.left.to_datetime64(), time_range.right.to_datetime64()]
        )(self._dataset.atrack)

        self._dataset = self._dataset.assign(time=("atrack", pd.to_datetime(atrack_time)))
        self._dataset['time'].attrs['comment'] = 'Simple interpolated time between start_date and stop_date'

        # dataset no-pol template for function evaluation on coordinates (*no* values used)
        # what's matter here is the shape of the image, not the values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            if self.s1meta.bursts.nbursts != 0:
                # SLC TOPS, tune the high res grid because of bursts overlapping
                xint = self.s1meta.burst_azitime(self._dataset.digital_number.atrack.values)
                self._da_tmpl = xr.DataArray(
                    dask.array.empty_like(
                        np.empty((len(xint),len(self._dataset.digital_number.xtrack))),
                        dtype=np.int8, name="empty_var_tmpl-%s" % dask.base.tokenize(self.s1meta.name)),
                    dims=('atrack', 'xtrack'),
                    coords={'atrack': self._dataset.digital_number.atrack,
                            'xtrack': self._dataset.digital_number.xtrack},
                    #attrs={'xint':xint}
                )
                self._da_tmpl['xint'] = xr.DataArray(xint,dims=['atrack'])
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
            #'incidence': 'annotation',
            #'elevation': 'annotation',
        }

        # dict mapping specifying if the variable has 'pol' dimension
        self._vars_with_pol = {
            'sigma0_lut': True,
            'gamma0_lut': True,
            'noise_lut_range': True,
            'noise_lut_azi': True,
            'incidence': False,
            'elevation': False,
            'height': False,
            'azimuth_time': False,
            'slant_range_time': False,
            'longitude': False,
            'latitude': False
        }

        # variables not returned to the user (unless luts=True)
        self._hidden_vars = ['sigma0_lut', 'gamma0_lut', 'noise_lut', 'noise_lut_range', 'noise_lut_azi']

        self._luts = self._lazy_load_luts(self._map_lut_files.keys())

        # noise_lut is noise_lut_range * noise_lut_azi
        if 'noise_lut_range' in self._luts.keys() and 'noise_lut_azi' in self._luts.keys():
            self._luts = self._luts.assign(noise_lut=self._luts.noise_lut_range * self._luts.noise_lut_azi)
            self._luts.noise_lut.attrs['history'] = merge_yaml(
                [self._luts.noise_lut_range.attrs['history'] + self._luts.noise_lut_azi.attrs['history']],
                section='noise_lut'
            )

        #lon_lat = self._load_lon_lat()

        self._rasterized_masks = self._load_rasterized_masks()

        ds_merge_list = [self._dataset, self._rasterized_masks, self._load_ground_heading(), #lon_lat
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

        rasters = self._load_rasters_vars()
        if rasters is not None:
            self._dataset = xr.merge([self._dataset, rasters])

        self._dataset = self._dataset.merge(self._load_from_geoloc(['height', 'azimuth_time', 'slant_range_time',
                                                                    'incidence','elevation','longitude','latitude']))
        self._dataset = self._add_denoised(self._dataset)
        self._dataset.attrs = self._recompute_attrs()

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
                    [list(ds[d].values) != list(self._dataset[d].values) for d in ['atrack', 'xtrack', 'pol']])
            self._dataset = ds
            self._dataset.attrs = self._recompute_attrs()
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
        return self.len_atrack_m / self.dataset.atrack.size

    @property
    def pixel_xtrack_m(self):
        """xtrack pixel spacing, in meters (relative to dataset)"""
        return self.len_xtrack_m / self.dataset.xtrack.size

    @property
    def coverage(self):
        """coverage string"""
        return "%dkm * %dkm (atrack * xtrack )" % (self.len_atrack_m / 1000, self.len_xtrack_m / 1000)

    @property
    def _regularly_spaced(self):
        return max([np.unique(np.round(np.diff(self._dataset[dim].values),1)).size for dim in ['atrack', 'xtrack']]) == 1

    def _recompute_attrs(self):
        if not self._regularly_spaced:
            warnings.warn(
                "Irregularly spaced dataset (probably multiple selection). Some attributes will be incorrect.")
        attrs = self._dataset.attrs
        attrs['pixel_xtrack_m'] = self.pixel_xtrack_m
        attrs['pixel_atrack_m'] = self.pixel_atrack_m
        attrs['coverage'] = self.coverage
        attrs['footprint'] = self.footprint
        return attrs

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
                lut.attrs['history'] = self.s1meta.xml_parser.get_compound_var(xml_file, lut_name, describe=True)

                lut = lut.to_dataset(name=lut_name)
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
            comment = 'resampled at "%s" with %s.%s.%s' % (resolution, resampling.__module__, resampling.__class__.__name__, resampling.name)
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
            # 0.5 is for pixel center (geotiff standard)
            dn = dn.assign_coords({'atrack': dn.atrack + 0.5, 'xtrack': dn.xtrack + 0.5})
            dn = dn.drop_vars('spatial_ref', errors='ignore')
        else:
            if not isinstance(resolution, dict):
                if isinstance(resolution, str) and resolution.endswith('m'):
                    resolution = float(resolution[:-1])
                resolution = dict(atrack=resolution / self.s1meta.pixel_atrack_m, xtrack=resolution / self.s1meta.pixel_xtrack_m)

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
            )



            # create coordinates at box center (+0.5 for pixel center, -1 because last index not included)
            translate = Affine.translation((resolution['xtrack'] - 1) / 2 + 0.5, (resolution['atrack'] - 1) / 2 + 0.5)
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
                    var_name: get_glob([p.replace(self.s1meta.path+'/', '') for p in self.s1meta.files['measurement'] ])
                }
            )
        }
        ds = dn.to_dataset(name=var_name)

        astype = self._dtypes.get(var_name)
        if astype is not None:
            ds = ds.astype(self._dtypes[var_name])

        return ds

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
        for varname in varnames:

            if self.s1meta.bursts.nbursts!=0:
                # TOPS SLC
                logger.debug(' x %s y %s z %s',self.s1meta.geoloc.azimuth_time.shape,self.s1meta.geoloc.xtrack.shape,self.s1meta.geoloc[varname].shape)
                interp_func = RectBivariateSpline(
                    self.s1meta.geoloc.azimuth_time[:,0],
                    self.s1meta.geoloc.xtrack,
                    self.s1meta.geoloc[varname],
                    kx=1, ky=1
                )
            else:
                interp_func = RectBivariateSpline(
                    self.s1meta.geoloc.atrack,
                    self.s1meta.geoloc.xtrack,
                    self.s1meta.geoloc[varname],
                    kx=1, ky=1
                )
            logger.debug('%s %s',varname,interp_func)
            # the following take much cpu and memory, so we want to use dask
            # interp_func(self._dataset.atrack, self.dataset.xtrack)
            if self.s1meta.bursts.nbursts!=0:
                da_var = map_blocks_coords(
                    self._da_tmpl.astype(self.s1meta.geoloc[varname].dtype),
                    interp_func,
                    withburst=True,func_kwargs={'grid':False},
                )
            else:
                da_var = map_blocks_coords(
                    self._da_tmpl.astype(self.s1meta.geoloc[varname].dtype),
                    interp_func
                )
            da_var.name = varname

            # copy history
            try:
                da_var.attrs['history'] = self.s1meta.geoloc[varname].attrs['history']
            except KeyError:
                pass

            da_list.append(da_var)

        return xr.merge(da_list)

    @timing
    def _load_lon_lat(self):
        """
        Load longitude and latitude using `self.s1meta.gcps`.
        #TODO deprecated func
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

        ll_tmpl = self._da_tmpl.expand_dims({'ll': 2}).assign_coords(ll=ll_coords).astype(self._dtypes['longitude'])
        ll_ds = map_blocks_coords(ll_tmpl, coords2ll, name='blocks_lonlat')
        # remove ll_coords to have two separate variables longitude and latitude
        ll_ds = xr.merge([ll_ds.sel(ll=ll).drop_vars(['ll']).rename(ll) for ll in ll_coords])

        ll_ds.longitude.attrs = {
            'long_name': 'longitude',
            'units': 'degrees_east',
            'standard_name': 'longitude'
        }

        ll_ds.latitude.attrs = {
            'long_name': 'latitude',
            'units': 'degrees_north',
            'standard_name': 'latitude'
        }
        return ll_ds

    @timing
    def _load_ground_heading(self):
        def coords2heading(atracks, xtracks):
            return self.s1meta.coords2heading(atracks, xtracks, to_grid=True, approx=False)

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
        def _test(atrack, xtrack, mask=None):
            chunk_coords = bbox_coords(atrack, xtrack, pad=None)
            # chunk footprint polygon, in dataset coordinates (with buffer, to enlarge a little the footprint)
            chunk_footprint_coords = Polygon(chunk_coords).buffer(10)
            tmp = self.s1meta.name
            #vector_mask_ll = s1meta.get_mask(mask) #
            return np.meshgrid(xtrack, atrack)[0] * 0

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
    def _load_rasters_vars(self):
        # load and map variables from rasterfile (like ecmwf) on dataset
        if self.s1meta.rasters.empty:
            return None
        else:
            logger.warning('Raster variable are experimental')

        if self.s1meta.cross_antemeridian:
            raise NotImplementedError('Antimeridian crossing not yet checked')

        # get lon/lat box for xsar dataset
        lons, lats = list(zip(*self.s1meta.footprint.exterior.coords))
        lon_range = [min(lons), max(lons)]
        lat_range = [min(lats), max(lats)]

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


            if not raster_ds.rio.crs.is_geographic:
                raise NotImplementedError("Non geographic crs not implemented")

            # ensure dim ordering
            raster_ds = raster_ds.transpose('y', 'x')
            if np.all(raster_ds.y.diff('y') <= 0):
                # sort y (lat) ascending (for RectBiVariateSpline)
                raster_ds = raster_ds.reindex(y=raster_ds.y[::-1])

            # from lon/lat box in xsar dataset, get the corresponding box in raster_ds (by index)
            ilon_range = [
                np.searchsorted(raster_ds.x.values, lon_range[0], side='right'),
                np.searchsorted(raster_ds.x.values, lon_range[1], side='left')
            ]
            ilat_range = [
                np.searchsorted(raster_ds.y.values, lat_range[0], side='right'),
                np.searchsorted(raster_ds.y.values, lat_range[1], side='left')
            ]
            # select the xsar box in the raster
            raster_ds = raster_ds.isel(x=slice(*ilon_range), y=slice(*ilat_range))

            # 1D array of lons/lats, trying to have same spacing as dataset (if not to high)
            num = min((self._dataset.xtrack.size + self._dataset.atrack.size) // 2, 1000)
            lons = np.linspace(*lon_range, num=num)
            lats = np.linspace(*lat_range, num=num)

            @dask.delayed
            def _map_raster2xsar(da):
                # map the 'da' dataarray variable from the raster to xsar dataset
                da = da.drop_vars(['spatial_ref', 'crs'], errors='ignore')

                upscaled_da = map_blocks_coords(
                    xr.DataArray(dims=['y', 'x'], coords={'x': lons, 'y': lats}).chunk(3000),
                    RectBivariateSpline(da.y.values, da.x.values, da.values)
                )

                reprojected_da = upscaled_da.interp(
                    x=self._dataset.longitude,
                    y=self._dataset.latitude
                )
                reprojected_da = reprojected_da.drop_vars(['x', 'y', 'spatial_ref', 'crs'], errors='ignore')
                reprojected_da.attrs.update(da.attrs)

                reprojected_da.name = '%s_%s' % (name, da.name)

                # reprojected_da has same shape as other variables is xsar dataset, with optional 3rd dim
                return reprojected_da


            for var in raster_ds:
                var_name = '%s_%s' % (name, raster_ds[var].name)
                da_var = xr.DataArray(
                    dask.array.from_delayed(
                        _map_raster2xsar(raster_ds[var]),
                        self._da_tmpl.shape,
                        dtype='f8',
                        name='%s' % var_name
                    ),
                    dims=['atrack', 'xtrack'],
                    coords={'atrack': self._da_tmpl.atrack, 'xtrack': self._da_tmpl.xtrack},
                    attrs=raster_ds[var].attrs
                )
                da_var.attrs['history'] = yaml.safe_dump({var_name: hist_res})
                logger.debug('adding variable "%s" from raster "%s"' % (var_name, name))
                da_var_list.append(da_var)

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

    def __repr__(self):
        if self.sliced:
            intro = "sliced"
        else:
            intro = "full covevage"
        return "<Sentinel1Dataset %s object>" % intro

    def _repr_mimebundle_(self, include=None, exclude=None):
        return repr_mimebundle(self, include=include, exclude=exclude)



    def ground_spacing(self):
        """Get SAR image ground spacing.

        Parameters
        ----------
        Returns
        -------
        ground_spacing : ndarray
            [azimuth, range] ground spacing in meters.

        Notes
        -----
        For GRD products, range_index and extent are ignored.
        """
        logger.debug('pixel_xtrack_m : %s',self.s1meta.geoloc.attrs['pixel_xtrack_m'])
        ground_spacing = np.array((self.s1meta.geoloc.attrs['pixel_atrack_m'],self.s1meta.geoloc.attrs['pixel_xtrack_m']))
        if self.s1meta.product == 'SLC':
            atrack_tmp = self._dataset['atrack']
            xtrack_tmp = self._dataset['xtrack']
            # get the incidence at the center of the part of image selected
            logger.debug('inc da : %s',self._dataset['incidence'])
            inc = self._dataset['incidence'].isel({'atrack': int(len(atrack_tmp) / 2),
                                                   'xtrack': int(len(xtrack_tmp) / 2)
                                                   }).values
            logger.debug('inc : %s',inc)
            ground_spacing[1] /= np.sin(inc * np.pi / 180)
        return ground_spacing

    def get_sensor_velocity(self, azimuth_time):
        """Interpolate sensor velocity at given azimuth time
        Parameters
        ----------
            azimuth_time (int):
        Returns
        -------
        """
        orbstatevect = self.s1meta.orbit
        azi_times = orbstatevect['time'].values
        vels = np.sqrt(np.sum(orbstatevect['velocity'].values ** 2., axis=1))
        iv = np.clip(np.searchsorted(azi_times, azimuth_time) - 1, 0, azi_times.size - 2)
        _vels = vels[iv] + (azimuth_time - azi_times[iv]) * \
                (vels[iv + 1] - vels[iv]) / (azi_times[iv + 1] - azi_times[iv])
        return _vels


    
