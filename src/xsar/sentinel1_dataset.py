# -*- coding: utf-8 -*-
import cartopy.feature
import logging
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import dask
import rasterio
import rasterio.features
import re
from scipy.interpolate import interp1d
from shapely.geometry import Polygon
import shapely
from .utils import timing, haversine, map_blocks_coords, rioread, rioread_fromfunction, bbox_coords, bind
from . import sentinel1_xml_mappings
from .xml_parser import XmlParser
from numpy import asarray
from affine import Affine
from functools import partial
from .sentinel1_meta import Sentinel1Meta
from .ipython_backends import repr_mimebundle

logger = logging.getLogger('xsar.sentinel1_dataset')
logger.addHandler(logging.NullHandler())

# we know tiff as no geotransform : ignore warning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class Sentinel1Dataset:
    """
    Handle a SAFE subdataset.
    A dataset might contain several tiff files (multiples polarizations), but all tiff files must share the same footprint.

    The main attribute useful to the end-user is `self.dataset` (`xarray.Dataset` , with all variables parsed from xml and tiff files.)

    Parameters
    ----------
    dataset_id: str or Sentinel1Meta object
        if str, it can be a path, or a gdal dataset identifier like `'SENTINEL1_DS:%s:WV_001' % filename`)
    resolution: dict, optional
        resampling dict like `{'atrack': 20, 'xtrack': 20}` where 20 is in pixels.
    resampling: rasterio.enums.Resampling or str, optional
        Only used if `resolution` is not None.
        ` rasterio.enums.Resampling.average` by default. `rasterio.enums.Resampling.nearest` (decimation) is fastest.
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
                 resampling=rasterio.enums.Resampling.average,
                 luts=False, chunks={'atrack': 5000, 'xtrack': 5000},
                 dtypes=None):

        # default dtypes (TODO: find defaults, so science precision is not affected)
        self._dtypes = {
            'latitude': 'f4',
            'longitude': 'f4',
            'incidence': 'f4',
            'elevation': 'f4',
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
            'digital_number': None
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
            xml_parser = XmlParser(
                xpath_mappings=sentinel1_xml_mappings.xpath_mappings,
                compounds_vars=sentinel1_xml_mappings.compounds_vars,
                namespaces=sentinel1_xml_mappings.namespaces)
            self.s1meta = Sentinel1Meta(dataset_id, xml_parser=xml_parser)
        else:
            self.s1meta = dataset_id

        if self.s1meta.multidataset:
            raise IndexError(
                """Can't open an multi-dataset. Use `xsar.Sentinel1Meta('%s').subdatasets` to show availables ones""" % self.s1meta.path
            )

        self._dataset = self._load_digital_number(resolution=resolution, resampling=resampling, chunks=chunks)

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

        self._luts = self._lazy_load_luts(self._map_lut_files.keys())

        # noise_lut is noise_lut_range * noise_lut_azi
        if 'noise_lut_range' in self._luts.keys() and 'noise_lut_azi' in self._luts.keys():
            self._luts = self._luts.assign(noise_lut=self._luts.noise_lut_range * self._luts.noise_lut_azi)

        lon_lat = self._load_lon_lat()

        self._raster_masks = self._load_raster_masks()

        ds_merge_list = [self._dataset, lon_lat, self._raster_masks, self._load_ground_heading(),
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

        self._dataset = self._add_denoised(self._dataset)
        self._dataset.attrs = self._recompute_attrs()

        self.coords2ll = self.s1meta.coords2ll
        """
        Alias for `xsar.Sentinel1Meta.coords2ll`

        See Also
        --------
        xsar.Sentinel1Meta.coords2ll
        """

        self.sliced = False
        """True if dataset is a slice of original L1 dataset"""

        self.resampled = resolution is not None
        """True if dataset is not a sensor resolution"""

        # save original bbox
        self._bbox_coords_ori = self._bbox_coords

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
        return max([np.unique(np.diff(self._dataset[dim].values)).size for dim in ['atrack', 'xtrack']]) == 1

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

                lut = map_blocks_coords(self._da_tmpl.astype(self._dtypes[lut_name]), lut_f_delayed,
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
    def _load_digital_number(self, resolution=None, chunks=None, resampling=rasterio.enums.Resampling.average):
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
        riot_dtyp = rio.dtypes[0]
        if riot_dtyp == 'complex_int16' :
            riot_dtyp = np.complex_
        else :
            pass
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
                    logging.info('rio dtypes zero: %s',rio.dtypes[0])

                    resampled = [
                        xr.DataArray(
                            dask.array.from_delayed(
                                dask.delayed(rioread)(f, out_shape_pol, winsize, resampling=resampling),
                                out_shape_pol, dtype=np.dtype(riot_dtyp)
                            ),
                            dims=tuple(map_dims.keys()), coords={'pol': [pol]}
                        ) for f, pol in
                        zip(self.s1meta.files['measurement'], self.s1meta.manifest_attrs['polarizations'])
                    ]
                    dn = xr.concat(resampled, 'pol').chunk(chunks)
                else:
                    resampled = dask.array.from_delayed(
                        dask.delayed(rioread)(self.s1meta.name, out_shape_pol, winsize, resampling=resampling),
                        out_shape_pol, dtype=np.dtype(riot_dtyp))
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
                                dtype=np.dtype(riot_dtyp),
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
                        dtype=np.dtype(riot_dtyp),
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

        astype = self._dtypes.get(var_name)
        if astype is not None:
            ds = ds.astype(self._dtypes[var_name])

        return ds

    @timing
    def _load_lon_lat(self):
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
        ll_tmpl = self._da_tmpl.expand_dims({'ll': 2}).assign_coords(ll=ll_coords).astype(self._dtypes['longitude'])
        ll_ds = map_blocks_coords(ll_tmpl, coords2ll, name='blocks_lonlat')
        # remove ll_coords to have two separate variables longitude and latitude
        ll_ds = xr.merge([ll_ds.sel(ll=ll).drop_vars(['ll']).rename(ll) for ll in ll_coords])

        return ll_ds

    @timing
    def _load_ground_heading(self):
        coords2heading = bind(self.s1meta.coords2heading, ..., ..., to_grid=True, approx=False)
        gh = map_blocks_coords(self._da_tmpl.astype(self._dtypes['ground_heading']), coords2heading,
                               name='ground_heading')
        return gh.to_dataset(name='ground_heading')

    @timing
    def _load_raster_masks(self):
        def _rasterize_mask_by_chunks(atrack, xtrack, mask='land'):
            chunk_coords = bbox_coords(atrack, xtrack, pad=None)
            # chunk footprint polygon, in dataset coordinates (with buffer, to enlarge a little the footprint)
            chunk_footprint_coords = Polygon(chunk_coords).buffer(10)
            # chunk footprint polygon, in lon/lat
            chunk_footprint_ll = self.s1meta.coords2ll(chunk_footprint_coords)

            # get vector mask over chunk
            # FIXME: speedup if get_mask is first called outside worker
            vector_mask_ll = self.s1meta.get_mask(mask).intersection(chunk_footprint_ll)

            if vector_mask_ll.is_empty:
                # no intersection with mask, return zeros
                return np.zeros((atrack.size, xtrack.size))

            # vector mask, in atrack/xtrack coordinates
            vector_mask_coords = self.s1meta.ll2coords(vector_mask_ll)

            raster_mask = rasterio.features.rasterize(
                [vector_mask_coords],
                out_shape=(xtrack.size, atrack.size),
                all_touched=False,
                transform=Affine.translation(*chunk_coords[0]) * Affine.scale(*[np.unique(np.diff(c)).item() for c in [xtrack,atrack]])
            ).T
            return raster_mask

        da_list = [
            map_blocks_coords(
                self._da_tmpl,
                _rasterize_mask_by_chunks,
                func_kwargs={'mask': mask}
            ).to_dataset(name='%s_mask' % mask) for mask in self.s1meta._mask_features.keys()
        ]
        return xr.merge(da_list)

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
        return dataarr.to_dataset(name=name)

    def _add_denoised(self, ds, clip=True, vars=None):
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

    def __str__(self):
        if self.sliced:
            intro = "sliced"
        else:
            intro = "full covevage"
        return "%s Sentinel1Dataset object" % intro

    def _repr_mimebundle_(self, include=None, exclude=None):
        return repr_mimebundle(self, include=include, exclude=exclude)


class SentinelDataset(Sentinel1Dataset):
    def __init__(self, *args, **kwargs):
        warnings.warn("SentinelDataset is deprecated. Please update your code to use 'Sentinel1Dataset'")
        super().__init__(*args, **kwargs)
