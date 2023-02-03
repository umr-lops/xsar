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
from xradarsat2 import load_digital_number
import xarray as xr
from scipy.interpolate import RectBivariateSpline
import dask
import datatree
# import numpy_groupies as npg
from scipy import ndimage
from .sentinel1_xml_mappings import signal_lut

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
        self.resolution = resolution
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

        self.DN_without_res = load_digital_number(self.rs2meta.dt, resampling=resampling, chunks=chunks)[
            'digital_numbers'].ds
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

        # dict mapping for variable names to create by applying specified lut on digital numbers

        self._map_var_lut_noise = {
            'sigma0': 'noiseLevelValues_SigmaNought',
            'gamma0': 'noiseLevelValues_Gamma',
            'beta0': 'noiseLevelValues_BetaNought',
        }

        self._map_var_lut = {
            'sigma0': 'lutSigma',
            'gamma0': 'lutGamma',
            'beta0': 'lutBeta',
        }

        for att in ['name', 'short_name', 'product', 'safe', 'swath', 'multidataset']:
            if att not in self.datatree.attrs:
                # tmp = xr.DataArray(self.s1meta.__getattr__(att),attrs={'source':'filename decoding'})
                self.datatree.attrs[att] = self.rs2meta.__getattr__(att)
                self._dataset.attrs[att] = self.rs2meta.__getattr__(att)

        value_res_line = self.rs2meta.geoloc.line.attrs['rasterAttributes_sampledLineSpacing_value']
        value_res_sample = self.rs2meta.geoloc.pixel.attrs['rasterAttributes_sampledPixelSpacing_value']
        # self._load_incidence_from_lut()
        refe_spacing = 'slant'
        if resolution is not None:
            refe_spacing = 'ground'  # if the data sampling changed it means that the quantities are projected on ground
            if isinstance(resolution, str):
                value_res_sample = float(resolution.replace('m', ''))
                value_res_line = value_res_sample
            elif isinstance(resolution, dict):
                value_res_sample = self.rs2meta.geoloc.pixel.attrs['rasterAttributes_sampledPixelSpacing_value'] \
                                   * resolution['sample']
                value_res_line = self.rs2meta.geoloc.line.attrs['rasterAttributes_sampledLineSpacing_value'] \
                                 * resolution['line']
            else:
                logger.warning('resolution type not handle (%s) should be str or dict -> sampleSpacing'
                               ' and lineSpacing are not correct', type(resolution))
        self._dataset['sampleSpacing'] = xr.DataArray(value_res_sample,
                                                      attrs={'units': 'm', 'referential': refe_spacing})
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
        self._luts = self.rs2meta.dt['lut'].ds.rename({'pixels': 'sample'})
        self.apply_calibration_and_denoising()
        self._dataset = xr.merge([self._load_from_geoloc(['latitude', 'longitude', 'altitude',
                                                          'incidence', 'elevation'
                                                          ]
                                                         ), self._dataset])
        a = self._dataset.copy()
        self._dataset = self.flip_sample_da(a)
        self.datatree['measurement'] = self.datatree['measurement'].assign(self._dataset)
        a = self._dataset.copy()
        self._dataset = self.flip_line_da(a)
        self.datatree['measurement'] = self.datatree['measurement'].assign(self._dataset)
        self.datatree = datatree.DataTree.from_dict(
            {'measurement': self.datatree['measurement'],
             'geolocation_annotation': self.datatree['geolocation_annotation'],
             'reader': self.rs2meta.dt})

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
                                  'incidence': 'incidenceAngle',
                                  'elevation': 'elevationAngle',
                                  'altitude': 'height',
                                  }
        da_list = []

        def interp_func_agnostic_satellite(vect1line, vect1pixel, **kwargs):

            # exterieur de boucle
            rbs = kwargs['rbs']

            def wrapperfunc(*args, **kwargs):
                rbs2 = args[2]
                return rbs2(args[0], args[1], grid=False)

            return wrapperfunc(vect1line[:, np.newaxis], vect1pixel[np.newaxis, :], rbs)

        for varname in varnames:
            varname_in_geoloc = mapping_dataset_geoloc[varname]
            if varname == 'incidence':
                da = self._load_incidence_from_lut()
                da.name = varname
                da_list.append(da)
            elif varname == 'elevation':
                da = self._load_elevation_from_lut()
                da.name = varname
                da_list.append(da)
            else:
                if varname == 'longitude':
                    z_values = self.rs2meta.geoloc[varname]
                    if self.rs2meta.cross_antemeridian:
                        logger.debug('translate longitudes between 0 and 360')
                        z_values = z_values % 360
                else:
                    z_values = self.rs2meta.geoloc[varname_in_geoloc]
                # interp_func = interp_func_agnostic_satellite
                rbs = None
                interp_func = RectBivariateSpline(
                    self.rs2meta.geoloc.line,
                    self.rs2meta.geoloc.pixel,
                    z_values,
                    kx=1, ky=1
                )

                da_val = interp_func(self._dataset.digital_number.line, self._dataset.digital_number.sample)
                da_var = xr.DataArray(data=da_val, dims=['line', 'sample'],
                                      coords={'line': self._dataset.digital_number.line,
                                              'sample': self._dataset.digital_number.sample})
                if varname == 'longitude':
                    if self.rs2meta.cross_antemeridian:
                        da_var.data = da_var.data.map_blocks(to_lon180)

                da_var.name = varname

                # copy history
                try:
                    da_var.attrs['history'] = self.rs2meta.geoloc[varname_in_geoloc].attrs['xpath']
                except KeyError:
                    pass

                da_list.append(da_var)
        ds = xr.merge(da_list)
        return ds

    @timing
    def _load_incidence_from_lut(self):
        beta = self._dataset.beta0_raw[0]
        gamma = self._dataset.gamma0_raw[0]
        incidence_pre = gamma / beta
        i_angle = np.degrees(np.arctan(incidence_pre))
        # i_angle_hd = np.tile(i_angle, (len(self._dataset.digital_number.line), 1))
        return xr.DataArray(data=i_angle, dims=['line', 'sample'], coords={
            'line': self._dataset.digital_number.line,
            'sample': self._dataset.digital_number.sample})

    @timing
    def _resample_lut_values(self, lut):
        resolution = self.resolution
        sample_spacing = self.rs2meta.dt["geolocationGrid"]["pixel"].attrs[
            "rasterAttributes_sampledPixelSpacing_value"]
        if isinstance(resolution, str) and resolution.endswith('m'):
            resolution = float(self.resolution[:-1])
            out_sample_resolution = resolution / sample_spacing
        elif isinstance(resolution, dict):
            out_sample_resolution = resolution['sample']
        else:
            raise ValueError("There is a problem with the resolution format")
        data = lut.values
        # filter the signal with a gaussian kernel 10 m
        sig = 0.5 * out_sample_resolution
        trunc = 4
        res = ndimage.gaussian_filter1d(data, sigma=sig, mode='mirror', truncate=trunc)
        res_da = xr.DataArray(res, coords={'sample': np.arange(lut.sample.shape[0])}, dims=['sample'])
        # bb_da = xr.DataArray(data, coords={'sample': np.arange(lut.sample.shape[0])}, dims=['sample'])
        # nperseg = {'sample': int(out_sample_resolution)}
        # define the posting of the resampled coordinates using https://github.com/umr-lops/xsar_slc
        # posting = xtiling(bb_da, nperseg, noverlap=0, centering=False, side='left', prefix='')
        posting = self._dataset.digital_number.sample.values
        # resampled_signal = res_da.isel(sample=posting['sample'].sample)
        resampled_signal = res_da.isel(sample=posting.astype(int))
        """# remove last samples which can't be averaged
        new_size = int(int(data.shape[0]/out_sample_resolution) * out_sample_resolution)
        data = data[:new_size]

        group_idx = np.floor(np.arange(new_size) / out_sample_resolution).astype(int)
        resampled_lut_values = npg.aggregate(group_idx, data, func='mean')
        return xr.DataArray(data=resampled_lut_values, dims=['sample'],
                            coords={'sample': self._dataset.digital_number.sample})"""
        return resampled_signal.assign_coords({"sample": self._dataset.digital_number.sample})

    """@timing
    def _resample_lut_values(self, lut):
        hr_line_nb = self.rs2meta.dt['geolocationGrid'].ds.line[-1].values + 1
        lut_2d = xr.DataArray(data=np.tile(lut, (hr_line_nb, 1)),
                              coords={'line': np.arange(hr_line_nb), 'sample': lut.sample},
                              dims=['line', 'sample'])
        signal_lut_2d = signal_lut(lut_2d.line, lut_2d.sample, lut)"""

    @timing
    def _load_elevation_from_lut(self):
        satellite_height = self.rs2meta.dt.attrs['satelliteHeight']
        earth_radius = 6.371e6
        incidence = self._load_incidence_from_lut()
        angle_rad = np.sin(np.radians(incidence))
        inside = angle_rad * earth_radius / (earth_radius + satellite_height)
        return np.degrees(np.arcsin(inside))

    @timing
    def _get_lut(self, var_name):
        """
        Get lut for var_name

        Parameters
        ----------
        var_name: str

        Returns
        -------
        xarray.DataArray
            lut for `var_name`
        """
        try:
            lut_name = self._map_var_lut[var_name]
        except KeyError:
            raise ValueError("can't find lut name for var '%s'" % var_name)
        try:
            lut = self._luts[lut_name]
        except KeyError:
            raise ValueError("can't find lut from name '%s' for variable '%s'" % (lut_name, var_name))
        return lut

    @timing
    def _get_lut_noise(self, var_name):
        """
        Get noise lut in the reader for var_name

        Parameters
        ----------
        var_name: str

        Returns
        -------
        xarray.DataArray
            noise lut for `var_name`
        """
        try:
            lut_name = self._map_var_lut_noise[var_name]
        except KeyError:
            raise ValueError("can't find noise lut name for var '%s'" % var_name)
        try:
            lut = self.rs2meta.dt['radarParameters'][lut_name]
        except KeyError:
            raise ValueError("can't find noise lut from name '%s' for variable '%s'" % (lut_name, var_name))
        return lut

    @timing
    def _interpolate_for_noise_lut(self, var_name):
        """
        Interpolate the noise level values (from the reader) and resample it to create a noise lut

        Parameters
        ----------
        var_name : str
            Variable name to compute by applying lut. Must exist in `self._map_var_lut_noise` to be able to get the corresponding lut.

        Returns
        -------
        xarray.DataArray
            Noise level values interpolated and resampled
        """
        initial_lut = self._get_lut_noise(var_name)
        first_pix = initial_lut.attrs['pixelFirstNoiseValue']
        step = initial_lut.attrs['stepSize']
        #noise_values = (10 ** (initial_lut / 10)).values
        noise_values = initial_lut.values
        lines = np.arange(self.rs2meta.geoloc.line.values[-1] + 1)
        noise_values_2d = np.tile(noise_values, (lines.shape[0], 1))
        indexes = [first_pix + step * i for i in range(0, noise_values.shape[0])]
        inter_func = RectBivariateSpline(x=lines, y=indexes, z=noise_values_2d, kx=1, ky=1)
        var = inter_func(self._dataset.digital_number.line, self._dataset.digital_number.sample)
        da_var = xr.DataArray(data=var, dims=['line', 'sample'],
                              coords={'line': self._dataset.digital_number.line,
                                      'sample': self._dataset.digital_number.sample})
        return da_var

    @timing
    def _get_noise(self, var_name):
        """
            Get noise equivalent for  `var_name`.

            Parameters
            ----------
            var_name: str Variable name to compute. Must exist in `self._map_var_lut` and
            `self._map_var_lut_noise` to be able to get the corresponding lut.

            Returns
            -------
            xarray.Dataset
                with one variable named by `'ne%sz' % var_name[0]` (ie 'nesz' for 'sigma0', 'nebz' for 'beta0', etc...)
        """
        lut = self._get_lut(var_name)
        offset = lut.attrs['offset']
        if self.resolution is not None:
            lut = dask.delayed(self._resample_lut_values)(lut)
        lut_noise = dask.delayed(self._interpolate_for_noise_lut(var_name))
        name = 'ne%sz' % var_name[0]
        res = ((lut_noise.compute() ** 2) + offset) / lut.compute()
        return res.to_dataset(name=name)

    @timing
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
            else:
                denoised = ds[varname_raw] - ds[noise]

                if clip:
                    denoised = denoised.clip(min=0)
                    denoised.attrs['comment'] = 'clipped, no values <0'
                else:
                    denoised.attrs['comment'] = 'not clipped, some values can be <0'
                ds[varname] = denoised
        return ds

    @timing
    def _apply_calibration_lut(self, var_name):
        """
            Apply calibration lut to `digital_number` to compute `var_name`.

            Parameters
            ----------
            var_name: str
                Variable name to compute by applying lut. Must exist in `self._map_var_lut` to be able to get the corresponding lut.

            Returns
            -------
            xarray.Dataset
                with one variable named by `var_name`
        """
        lut = self._get_lut(var_name)
        offset = lut.attrs['offset']
        if self.resolution is not None:
            lut = dask.delayed(self._resample_lut_values)(lut)
        res = ((self._dataset.digital_number ** 2.) + offset) / lut.compute()
        res = res.where(res > 0).compute()
        res.attrs.update(lut.compute().attrs)
        return res.to_dataset(name=var_name + '_raw')

    @timing
    def apply_calibration_and_denoising(self):
        """
        apply calibration and denoising functions to get high resolution sigma0 , beta0 and gamma0 + variables *_raw
        :return:
        """
        for var_name, lut_name in self._map_var_lut.items():
            if lut_name in self._luts:
                # merge var_name into dataset (not denoised)
                self._dataset = xr.merge([self._apply_calibration_lut(var_name), self._dataset])
                # merge noise equivalent for var_name (named 'ne%sz' % var_name[0)
                self._dataset = xr.merge([self._get_noise(var_name), self._dataset])
            else:
                logger.debug("Skipping variable '%s' ('%s' lut is missing)" % (var_name, lut_name))
        self._dataset = self._add_denoised(self._dataset)
        self.datatree['measurement'] = self.datatree['measurement'].assign(self._dataset)

        return

    @timing
    def flip_sample_da(self, ds):
        """
        When a product is flipped, flip back data arrays (from a dataset) sample dimensions to respect the xsar
        convention (increasing incidence values)

        Parameters
        ----------
        ds : xarray.Dataset
            Contains dataArrays which depends on `sample` dimension

        Returns
        -------
        xarray.Dataset
            Flipped back, respecting the xsar convention
        """
        antenna_pointing = self.rs2meta.dt['radarParameters'].attrs['antennaPointing']
        pass_direction = self.rs2meta.dt.attrs['passDirection']
        flipped_cases = [('Left', 'Ascending'), ('Right', 'Descending')]
        if (antenna_pointing, pass_direction) in flipped_cases:
            new_ds = ds.isel(sample=slice(None, None, -1)).assign_coords(sample=ds.sample)
            new_ds.attrs['samples_flipped'] = 'xsar convention : increasing incidence values along samples axis'
        else:
            new_ds = ds
        return new_ds

    @timing
    def flip_line_da(self, ds):
        """
        Flip dataArrays (from a dataset) that depend on line dimension when a product is ascending, in order to
        respect the xsar convention (increasing time along line axis, whatever ascending or descending product).
        Reference : `schemas/rs2prod_burstAttributes.xsd:This corresponds to the top-left pixel in a coordinate
        system where the range increases to the right and the zero-Doppler time increases downward. Note that this is
        not necessarily the top-left pixel of the image block in the final product.`

        Parameters
        ----------
        ds : xarray.Dataset
            Contains dataArrays which depends on `line` dimension

        Returns
        -------
        xarray.Dataset
            Flipped back, respecting the xsar convention
        """
        pass_direction = self.rs2meta.dt.attrs['passDirection']
        if pass_direction == 'Ascending':
            new_ds = ds.copy().isel(line=slice(None, None, -1)).assign_coords(line=ds.line)
            new_ds.attrs['lines_flipped'] = 'xsar convention : increasing time along line axis (whatever ascending or '\
                                            'descending pass direction)'
        else:
            new_ds = ds.copy()
        return new_ds

    @property
    def dataset(self):
        """
        `xarray.Dataset` representation of this `xsar.Sentinel1Dataset` object.
        This property can be set with a new dataset, if the dataset was computed from the original dataset.
        """
        # return self._dataset
        res = self.datatree['measurement'].to_dataset()
        res.attrs = self.datatree.attrs
        return res

    @dataset.setter
    def dataset(self, ds):
        if self.rs2meta.name == ds.attrs['name']:
            # check if new ds has changed coordinates
            if not self.sliced:
                self.sliced = any(
                    [list(ds[d].values) != list(self._dataset[d].values) for d in ['line', 'sample']])
            self._dataset = ds
            # self._dataset = self.datatree['measurement'].ds
            self.recompute_attrs()
        else:
            raise ValueError("dataset must be same kind as original one.")

    @dataset.deleter
    def dataset(self):
        logger.debug('deleter dataset')

