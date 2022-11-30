# -*- coding: utf-8 -*-
import logging
import warnings
from .radarsat2_meta import RadarSat2Meta
from .utils import timing, haversine, map_blocks_coords, bbox_coords, BlockingActorProxy, merge_yaml, get_glob, \
    to_lon180
import numpy as np
import rasterio
import rasterio.features
from numpy import asarray
from xradarsat2 import load_digital_number, rs2_reader
import xarray as xr
from scipy.interpolate import RectBivariateSpline
import dask

logger = logging.getLogger('xsar.sentinel1_dataset')
logger.addHandler(logging.NullHandler())

# we know tiff as no geotransform : ignore warning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# allow nan without warnings
# some dask warnings are still non filtered: https://github.com/dask/dask/issues/3245
np.errstate(invalid='ignore')


class RadarSat2Dataset:

    def __init__(self, dataset_id, resolution=None,
                 resampling=rasterio.enums.Resampling.rms,
                 luts=False, chunks={'line': 5000, 'sample': 5000},
                 dtypes=None, patch_variable=True):

        # default meta for map_blocks output.
        # as asarray is imported from numpy, it's a numpy array.
        # but if later we decide to import asarray from cupy, il will be a cupy.array (gpu)
        self._default_meta = asarray([], dtype='f8')

        self.rs2meta = None
        """`xsar.RadarSat2Meta` object"""

        if not isinstance(dataset_id, RadarSat2Meta):
            self.rs2meta = BlockingActorProxy(RadarSat2Meta, dataset_id)
            # check serializable
            # import pickle
            # s1meta = pickle.loads(pickle.dumps(self.s1meta))
            # assert isinstance(s1meta.coords2ll(100, 100),tuple)
        else:
            # we want self.s1meta to be a dask actor on a worker
            self.rs2meta = BlockingActorProxy(RadarSat2Meta.from_dict, dataset_id.dict)
        del dataset_id

        if self.rs2meta.multidataset:
            raise IndexError(
                """Can't open an multi-dataset. Use `xsar.Sentinel1Meta('%s').subdatasets` to show availables ones""" % self.rs2.path
            )

        self._dataset = load_digital_number(self.rs2meta.dt, resolution=resolution,
                                            resampling=resampling, chunks=chunks)['digital_numbers'].ds
        # self._dataset = xr.merge([xr.Dataset({'time': self._burst_azitime}), self._dataset])
        # dataset no-pol template for function evaluation on coordinates (*no* values used)
        # what's matter here is the shape of the image, not the values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            # SLC TOPS, tune the high res grid because of bursts overlapping
            self._da_tmpl = xr.DataArray(
                dask.array.empty_like(
                    np.empty((len(self._dataset.line))),
                    dtype=np.int8, name="empty_var_tmpl-%s" % dask.base.tokenize(self.rs2meta.name)),
                dims=('line', 'sample'),
                coords={
                    'line': self._dataset.line,
                    'pixel': self._dataset.sample,
                },
            )
        self._dataset.attrs.update(self.rs2meta.to_dict("all"))

        # dict mapping for variables names to create by applying specified lut on digital_number
        self._map_var_lut = {
            'sigma0_raw': 'sigma0_lut',
            'gamma0_raw': 'gamma0_lut',
            'beta0_raw': 'beta0_lut'

        }

        # dict mapping for lut names to file type (from self.files columns)
        self._map_lut_files = {
            'sigma0_lut': 'calibration',
            'gamma0_lut': 'calibration',
            'beta0_lut': 'calibration',
            'noise_lut_range': 'noise',
            'noise_lut_azi': 'noise'
        }

        # dict mapping specifying if the variable has 'pol' dimension
        self._vars_with_pol = {
            'sigma0_lut': False,
            'gamma0_lut': False,
            'beta0_lut': False,
            'noise_lut_range': False,
            'noise_lut_azi': False,
            #'incidence': False,
            #'elevation': False,
            #'altitude': False,
            #'azimuth_time': False,
            #'slant_range_time': False,
            'longitude': False,
            'latitude': False,
            'height': False
        }

        # variables not returned to the user (unless luts=True)
        self._hidden_vars = ['sigma0_lut', 'gamma0_lut', 'noise_lut', 'noise_lut_range', 'noise_lut_azi']
        # attribute to activate correction on variables, if available
        self._patch_variable = patch_variable

        self._luts = self.rs2meta.dt['lut'].ds

        """# noise_lut is noise_lut_range * noise_lut_azi
        if 'noise_lut_range' in self._luts.keys() and 'noise_lut_azi' in self._luts.keys():
            self._luts = self._luts.assign(noise_lut=self._luts.noise_lut_range * self._luts.noise_lut_azi)
            self._luts.noise_lut.attrs['history'] = merge_yaml(
                [self._luts.noise_lut_range.attrs['history'] + self._luts.noise_lut_azi.attrs['history']],
                section='noise_lut'
            )"""
        #TODO : continue __init__


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

        def interp_func_sgf(line, pixel, **kwargs):

            # exterieur de boucle
            rbs = kwargs['rbs']

            def wrapperfunc(*args, **kwargs):
                rbs2 = args[2]
                return rbs2(args[0], args[1], grid=False)

            return wrapperfunc(line[:, np.newaxis], pixel[np.newaxis, :], rbs)

        for varname in varnames:
            if varname == 'longitude':
                z_values = self.rs2.dt['geolocationGrid'][varname]
                if self.rs2meta.cross_antemeridian:
                    logger.debug('translate longitudes between 0 and 360')
                    z_values = z_values % 360
            else:
                z_values = self.rs2.dt['geolocationGrid'][varname]

            rbs = RectBivariateSpline(
                self.rs2.dt['geolocationGrid']['pixel'][:, 0],
                self.rs2.dt['geolocationGrid']['line'],
                z_values,
                kx=1, ky=1,
            )
            interp_func = interp_func_sgf

            # the following take much cpu and memory, so we want to use dask
            # interp_func(self._dataset.atrack, self.dataset.xtrack)
            typee = self.rs2.dt['geolocationGrid'][varname].dtype
            datemplate = self._da_tmpl.astype(typee).copy()
            # replace the atrack coordinates by atrack_time coordinates
            #datemplate = datemplate.assign_coords({'atrack': datemplate.coords['atrack_time']})
            da_var = map_blocks_coords(
                datemplate,
                interp_func,
                func_kwargs={"rbs": rbs}
            )
            # put back the real atrack coordinates
            da_var = da_var.assign_coords({'atrack': self._dataset.digital_number.atrack})
            if varname == 'longitude':
                if self.rs2meta.cross_antemeridian:
                    da_var.data = da_var.data.map_blocks(to_lon180)

            da_var.name = varname

            # copy history
            try:
                da_var.attrs['history'] = self.rs2meta.dt['geolocationGrid'][varname].attrs['xpath']
            except KeyError:
                pass

            da_list.append(da_var)

        return xr.merge(da_list)