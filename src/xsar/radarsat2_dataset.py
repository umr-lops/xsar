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
                 luts=False, chunks={'atrack': 5000, 'xtrack': 5000},
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

        self._dataset = load_digital_number(self.rs2meta.dt, resolution=resolution, resampling=resampling, chunks=chunks)
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

        def interp_func_slc(line, pixel, **kwargs):

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
            #TODO : make equivalent of _burst
            if self.rs2meta._bursts['burst'].size != 0:
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