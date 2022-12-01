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
import datatree

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
        self.geoloc_tree = None
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
                """Can't open an multi-dataset. Use `xsar.RadarSat2Meta('%s').subdatasets` to show availables ones""" % self.rs2.path
            )

        # build datatree
        DN_tmp = load_digital_number(self.rs2meta.dt, resolution=resolution,
                                     resampling=resampling, chunks=chunks)['digital_numbers'].ds
        ### geoloc
        geoloc = self.rs2meta.geoloc
        geoloc.attrs['history'] = 'annotations'

        ### orbitAndAttitude
        orbit_and_attitude = self.rs2meta.orbit_and_attitude
        orbit_and_attitude.attrs['history'] = 'annotations'

        ### dopplerCentroid
        doppler_centroid = self.rs2meta.doppler_centroid
        doppler_centroid.attrs['history'] = 'annotations'

        ### dopplerRateValues
        doppler_rate_values = self.rs2meta.doppler_rate_values
        doppler_rate_values.attrs['history'] = 'annotations'

        ### chirp
        chirp = self.rs2meta.chirp
        chirp.attrs['history'] = 'annotations'

        ### radarParameters
        radar_parameters = self.rs2meta.radar_parameters
        radar_parameters.attrs['history'] = 'annotations'

        ### lookUpTables
        lut = self.rs2meta.lut
        lut.attrs['history'] = 'annotations'

        self.datatree = datatree.DataTree.from_dict({'measurement': DN_tmp, 'geolocation_annotation': geoloc
                                                     })

        self._dataset = self.datatree['measurement'].to_dataset()
        for att in ['name', 'short_name', 'product', 'safe', 'swath', 'multidataset']:
            if att not in self.datatree.attrs:
                #tmp = xr.DataArray(self.s1meta.__getattr__(att),attrs={'source':'filename decoding'})
                self.datatree.attrs[att] = self.rs2meta.__getattr__(att)
                self._dataset.attrs[att] = self.rs2meta.__getattr__(att)

        value_res_line = self.rs2meta.geoloc.line.attrs['rasterAttributes_sampledLineSpacing_value']
        value_res_sample = self.rs2meta.geoloc.pixel.attrs['rasterAttributes_sampledPixelSpacing_value']
        refe_spacing = 'slant'
        if resolution is not None:
            refe_spacing = 'ground' # if the data sampling changed it means that the quantities are projected on ground
            if isinstance(resolution, str):
                value_res_sample = float(resolution.replace('m',''))
                value_res_line = value_res_sample
            elif isinstance(resolution, dict):
                value_res_sample = self.rs2meta.geoloc.pixel.attrs['rasterAttributes_sampledPixelSpacing_value']\
                                   *resolution['sample']
                value_res_line = self.rs2meta.geoloc.line.attrs['rasterAttributes_sampledLineSpacing_value']\
                                 *resolution['line']
            else:
                logger.warning('resolution type not handle (%s) should be str or dict -> sampleSpacing'
                               ' and lineSpacing are not correct', type(resolution))
        self._dataset['sampleSpacing'] = xr.DataArray(value_res_sample, attrs={'units': 'm', 'referential': refe_spacing})
        self._dataset['lineSpacing'] = xr.DataArray(value_res_line, attrs={'units': 'm'})

        # dataset no-pol template for function evaluation on coordinates (*no* values used)
        # what's matter here is the shape of the image, not the values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            # SLC TOPS, tune the high res grid because of bursts overlapping
            self._da_tmpl = xr.DataArray(
                dask.array.empty_like(
                    np.empty((len(self._dataset.line), len(self._dataset.sample))),
                    ################# NOT SURE   ##################
                    dtype=np.int8, name="empty_var_tmpl-%s" % dask.base.tokenize(self.rs2meta.name)),
                dims=('line', 'sample'),
                coords={
                    'line': self._dataset.line,
                    'pixel': self._dataset.sample,
                },
            )
        self._dataset.attrs.update(self.rs2meta.to_dict("all"))
        self.datatree.attrs.update(self.rs2meta.to_dict("all"))


        self._luts = self.rs2meta.dt['lut'].ds

        """# noise_lut is noise_lut_range * noise_lut_azi
        if 'noise_lut_range' in self._luts.keys() and 'noise_lut_azi' in self._luts.keys():
            self._luts = self._luts.assign(noise_lut=self._luts.noise_lut_range * self._luts.noise_lut_azi)
            self._luts.noise_lut.attrs['history'] = merge_yaml(
                [self._luts.noise_lut_range.attrs['history'] + self._luts.noise_lut_azi.attrs['history']],
                section='noise_lut'
            )"""
        # TODO : continue __init__

    @timing
    def _load_from_geoloc(self, varnames):
        """
        Interpolate (with RectBiVariateSpline) variables from `self.s1meta.geoloc` to `self._dataset`

        Parameters
        ----------
        varnames: list of str
            subset of variables names in `self.rs2meta.geoloc`

        Returns
        -------
        xarray.Dataset
            With interpolated vaiables

        """
        mapping_dataset_geoloc = {'latitude': 'latitude',
                                  'longitude': 'longitude',
                                  # 'incidence': 'incidenceAngle',
                                  # 'elevation': 'elevationAngle',
                                  'altitude': 'height',
                                  # 'azimuth_time': 'azimuthTime',
                                  # 'slant_range_time': 'slantRangeTime'
                                  }
        da_list = []

        def interp_func_sgf(vect1line, vect1pixel, **kwargs):

            # exterieur de boucle
            rbs = kwargs['rbs']

            def wrapperfunc(*args, **kwargs):
                rbs2 = args[2]
                return rbs2(args[0], args[1], grid=False)

            return wrapperfunc(vect1line[:, np.newaxis], vect1pixel[np.newaxis, :], rbs)

        for varname in varnames:
            varname_in_geoloc = mapping_dataset_geoloc[varname]
            if varname == 'longitude':
                z_values = self.rs2meta.geoloc[varname]
                if self.rs2meta.cross_antemeridian:
                    logger.debug('translate longitudes between 0 and 360')
                    z_values = z_values % 360
            else:
                z_values = self.rs2meta.geoloc[varname]

            rbs = RectBivariateSpline(
                self.rs2meta.geoloc.line[:, 0],
                self.rs2meta.geoloc.pixel,
                z_values,
                kx=1, ky=1,
            )
            interp_func = interp_func_sgf

            # the following take much cpu and memory, so we want to use dask
            # interp_func(self._dataset.atrack, self.dataset.xtrack)
            typee = self.rs2meta.geoloc[varname].dtype
            datemplate = self._da_tmpl.astype(typee).copy()
            # replace the atrack coordinates by atrack_time coordinates
            # datemplate = datemplate.assign_coords({'atrack': datemplate.coords['atrack_time']})
            da_var = map_blocks_coords(
                datemplate,
                interp_func,
                func_kwargs={"rbs": rbs}
            )
            # put back the real line coordinates
            da_var = da_var.assign_coords({'line': self._dataset.digital_number.line})
            if varname == 'longitude':
                if self.rs2meta.cross_antemeridian:
                    da_var.data = da_var.data.map_blocks(to_lon180)

            da_var.name = varname

            # copy history
            try:
                da_var.attrs['history'] = self.rs2meta.geoloc[varname].attrs['xpath']
            except KeyError:
                pass

            da_list.append(da_var)

        return xr.merge(da_list)

    # TODO : add to_dict()
