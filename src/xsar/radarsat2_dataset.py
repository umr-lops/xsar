# -*- coding: utf-8 -*-
import logging
import warnings


from .radarsat2_meta import RadarSat2Meta
from .utils import timing, map_blocks_coords, BlockingActorProxy, to_lon180
import numpy as np
import rasterio.features
import xarray as xr
from scipy.interpolate import RectBivariateSpline, interp1d
import dask
from .base_dataset import BaseDataset

logger = logging.getLogger("xsar.radarsat2_dataset")
logger.addHandler(logging.NullHandler())

# we know tiff as no geotransform : ignore warning
warnings.filterwarnings(
    "ignore", category=rasterio.errors.NotGeoreferencedWarning)

# allow nan without warnings
# some dask warnings are still non filtered: https://github.com/dask/dask/issues/3245
np.errstate(invalid="ignore")


class RadarSat2Dataset(BaseDataset):
    """
    Handle a SAFE subdataset.
    A dataset might contain several tiff files (multiples polarizations), but all tiff files must share the same footprint.

    The main attribute useful to the end-user is `self.dataset` (`xarray.Dataset` , with all variables parsed from xml and tiff files.)

    Parameters
    ----------
    dataset_id: str or RadarSat2Meta object

        if str, it can be a path, or a gdal dataset identifier)

    resolution: dict, number or string, optional
        resampling dict like `{'line': 20, 'sample': 20}` where 20 is in pixels.

        if a number, dict will be constructed from `{'line': number, 'sample': number}`

        if str, it must end with 'm' (meters), like '100m'. dict will be computed from sensor pixel size.

    resampling: rasterio.enums.Resampling or str, optional

        Only used if `resolution` is not None.

        ` rasterio.enums.Resampling.rms` by default. `rasterio.enums.Resampling.nearest` (decimation) is fastest.

    chunks: dict, optional

        dict with keys ['pol','line','sample'] (dask chunks).

    dtypes: None or dict, optional

        Specify the data type for each variable.

    lazyloading: bool, optional
        activate or not the lazy loading of the high resolution fields

    """

    def __init__(
        self,
        dataset_id,
        resolution=None,
        resampling=rasterio.enums.Resampling.rms,
        chunks={"line": 5000, "sample": 5000},
        dtypes=None,
        lazyloading=True,
        skip_variables=None,
    ):
        if skip_variables is None:
            skip_variables = []
        # default dtypes
        if dtypes is not None:
            self._dtypes.update(dtypes)

        # default meta for map_blocks output.
        # as asarray is imported from numpy, it's a numpy array.
        # but if later we decide to import asarray from cupy, il will be a cupy.array (gpu)
        self.sar_meta = None
        self.resolution = resolution

        if not isinstance(dataset_id, RadarSat2Meta):
            self.sar_meta = BlockingActorProxy(RadarSat2Meta, dataset_id)
            # check serializable
            # import pickle
            # s1meta = pickle.loads(pickle.dumps(self.s1meta))
            # assert isinstance(sar_meta.coords2ll(100, 100),tuple)
        else:
            # we want self.sar_meta to be a dask actor on a worker
            self.sar_meta = BlockingActorProxy(
                RadarSat2Meta.from_dict, dataset_id.dict)
        del dataset_id
        if self.sar_meta.multidataset:
            raise IndexError(
                """Can't open an multi-dataset. Use `xsar.RadarSat2Meta('%s').subdatasets` to show availables ones"""
                % self.sar_meta.path
            )

        from xradarsat2 import load_digital_number

        # build datatree
        DN_tmp = load_digital_number(
            self.sar_meta.dt,
            resolution=resolution,
            resampling=resampling,
            chunks=chunks,
        )["digital_numbers"].ds
        # In order to respect xsar convention, lines and samples have been flipped in the metadata when necessary.
        # `load_digital_number` uses these metadata but rio creates new coords without keeping the flipping done.
        # So we have to flip again a time digital numbers to respect xsar convention
        DN_tmp = self.flip_sample_da(DN_tmp)
        DN_tmp = self.flip_line_da(DN_tmp)

        # geoloc
        geoloc = self.sar_meta.geoloc
        geoloc.attrs["history"] = "annotations"

        # orbitAndAttitude
        orbit_and_attitude = self.sar_meta.orbit_and_attitude
        orbit_and_attitude.attrs["history"] = "annotations"

        # dopplerCentroid
        doppler_centroid = self.sar_meta.doppler_centroid
        doppler_centroid.attrs["history"] = "annotations"

        # dopplerRateValues
        doppler_rate_values = self.sar_meta.doppler_rate_values
        doppler_rate_values.attrs["history"] = "annotations"

        # chirp
        chirp = self.sar_meta.chirp
        chirp.attrs["history"] = "annotations"

        # radarParameters
        radar_parameters = self.sar_meta.radar_parameters
        radar_parameters.attrs["history"] = "annotations"

        # lookUpTables
        lut = self.sar_meta.lut
        lut.attrs["history"] = "annotations"

        self.datatree = xr.DataTree.from_dict(
            {"measurement": DN_tmp, "geolocation_annotation": geoloc}
        )

        self._dataset = self.datatree["measurement"].to_dataset()

        # dict mapping for variable names to create by applying specified lut on digital numbers

        self._map_var_lut_noise = {
            "sigma0": "noiseLevelValues_SigmaNought",
            "gamma0": "noiseLevelValues_Gamma",
            "beta0": "noiseLevelValues_BetaNought",
        }

        self._map_var_lut = {
            "sigma0": "lutSigma",
            "gamma0": "lutGamma",
            "beta0": "lutBeta",
        }
        geoloc_vars = ["latitude", "longitude",
                       "altitude", "incidence", "elevation"]
        for vv in skip_variables:
            if vv in geoloc_vars:
                geoloc_vars.remove(vv)

        for att in ["name", "short_name", "product", "safe", "swath", "multidataset"]:
            if att not in self.datatree.attrs:
                # tmp = xr.DataArray(self.s1meta.__getattr__(att),attrs={'source':'filename decoding'})
                self.datatree.attrs[att] = self.sar_meta.__getattr__(att)
                self._dataset.attrs[att] = self.sar_meta.__getattr__(att)

        value_res_line = self.sar_meta.geoloc.line.attrs[
            "rasterAttributes_sampledLineSpacing_value"
        ]
        value_res_sample = self.sar_meta.geoloc.pixel.attrs[
            "rasterAttributes_sampledPixelSpacing_value"
        ]
        # self._load_incidence_from_lut()
        refe_spacing = "slant"
        if resolution is not None:
            # if the data sampling changed it means that the quantities are projected on ground
            refe_spacing = "ground"
            if isinstance(resolution, str):
                value_res_sample = float(resolution.replace("m", ""))
                value_res_line = value_res_sample
            elif isinstance(resolution, dict):
                value_res_sample = (
                    self.sar_meta.geoloc.pixel.attrs[
                        "rasterAttributes_sampledPixelSpacing_value"
                    ]
                    * resolution["sample"]
                )
                value_res_line = (
                    self.sar_meta.geoloc.line.attrs[
                        "rasterAttributes_sampledLineSpacing_value"
                    ]
                    * resolution["line"]
                )
            else:
                logger.warning(
                    "resolution type not handle (%s) should be str or dict -> sampleSpacing"
                    " and lineSpacing are not correct",
                    type(resolution),
                )
        self._dataset["sampleSpacing"] = xr.DataArray(
            value_res_sample, attrs={"units": "m", "referential": refe_spacing}
        )
        self._dataset["lineSpacing"] = xr.DataArray(
            value_res_line, attrs={"units": "m"}
        )

        # dataset no-pol template for function evaluation on coordinates (*no* values used)
        # what's matter here is the shape of the image, not the values.
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore", np.ComplexWarning)
            self._da_tmpl = xr.DataArray(
                dask.array.empty_like(
                    self._dataset.digital_number.isel(pol=0).drop("pol"),
                    dtype=np.int8,
                    name="empty_var_tmpl-%s" % dask.base.tokenize(
                        self.sar_meta.name),
                ),
                dims=("line", "sample"),
                coords={
                    "line": self._dataset.digital_number.line,
                    "sample": self._dataset.digital_number.sample,
                },
            )
        # Add vars to define if lines or samples have been flipped to respect xsar convention
        self._dataset = xr.merge(
            [
                xr.DataArray(
                    data=self.sar_meta.samples_flipped,
                    attrs={
                        "meaning": "xsar convention : increasing incidence values along samples axis"
                    },
                ).to_dataset(name="samples_flipped"),
                self._dataset,
            ]
        )
        self._dataset = xr.merge(
            [
                xr.DataArray(
                    data=self.sar_meta.lines_flipped,
                    attrs={
                        "meaning": "xsar convention : increasing time along line axis "
                        "(whatever ascending or descending pass direction)"
                    },
                ).to_dataset(name="lines_flipped"),
                self._dataset,
            ]
        )

        self._luts = self.lazy_load_luts()
        self.apply_calibration_and_denoising()
        self._dataset = xr.merge(
            [
                self.load_from_geoloc(geoloc_vars, lazy_loading=lazyloading),
                self._dataset,
            ]
        )

        # compute offboresight in self._dataset
        self._get_offboresight_from_elevation()
        rasters = self._load_rasters_vars()
        if rasters is not None:
            self._dataset = xr.merge([self._dataset, rasters])
        self._dataset = xr.merge([self.interpolate_times, self._dataset])
        if "ground_heading" not in skip_variables:
            self._dataset = xr.merge(
                [self.load_ground_heading(), self._dataset])
        if "velocity" not in skip_variables:
            self._dataset = xr.merge(
                [self.get_sensor_velocity(), self._dataset])
        self._rasterized_masks = self.load_rasterized_masks()
        self._dataset = xr.merge([self._rasterized_masks, self._dataset])
        """a = self._dataset.copy()
        self._dataset = self.flip_sample_da(a)
        self.datatree['measurement'] = self.datatree['measurement'].assign(self._dataset)
        a = self._dataset.copy()
        self._dataset = self.flip_line_da(a)"""
        self.datatree["measurement"] = self.datatree["measurement"].assign(
            self._dataset
        )
        """self.datatree = datatree.DataTree.from_dict(
            {'measurement': self.datatree['measurement'],
             'geolocation_annotation': self.datatree['geolocation_annotation'],
             'reader': self.sar_meta.dt})"""
        self._reconfigure_reader_datatree()
        self._dataset.attrs.update(self.sar_meta.to_dict("all"))
        self.datatree.attrs.update(self.sar_meta.to_dict("all"))

        self.resampled = resolution is not None

    def lazy_load_luts(self):
        """
        Lazy load luts from the reader as delayed

        Returns
        -------
        xarray.Dataset
            Contains delayed dataArrays of luts
        """
        luts_ds = self.sar_meta.dt["lut"].ds.rename({"pixel": "sample"})
        merge_list = []
        for lut_name in luts_ds:
            lut_f_delayed = dask.delayed()(luts_ds[lut_name])
            ar = dask.array.from_delayed(
                lut_f_delayed.data,
                (luts_ds[lut_name].data.size,),
                luts_ds[lut_name].dtype,
            )
            da = xr.DataArray(
                data=ar,
                dims=["sample"],
                coords={"sample": luts_ds[lut_name].sample},
                attrs=luts_ds[lut_name].attrs,
            )
            ds_lut_f_delayed = da.to_dataset(name=lut_name)
            merge_list.append(ds_lut_f_delayed)
        return xr.combine_by_coords(merge_list)

    @timing
    def _get_offboresight_from_elevation(self):
        """
        Compute offboresight angle.

        Returns
        -------

        """
        self._dataset["offboresight"] = (
            self._dataset.elevation
            - (
                30.1833947 * self._dataset.latitude**0
                + 0.0082998714 * self._dataset.latitude**1
                - 0.00031181534 * self._dataset.latitude**2
                - 0.0943533e-07 * self._dataset.latitude**3
                + 3.0191435e-08 * self._dataset.latitude**4
                + 4.968415428e-12 * self._dataset.latitude**5
                - 9.315371305e-13 * self._dataset.latitude**6
            )
            + 29.45
        )
        self._dataset["offboresight"].attrs[
            "comment"
        ] = "built from elevation angle and latitude"

    @timing
    def load_from_geoloc(self, varnames, lazy_loading=True):
        """
        Interpolate (with RectBiVariateSpline) variables from `self.sar_meta.geoloc` to `self._dataset`

        Parameters
        ----------
        varnames: list of str
            subset of variables names in `self.sar_meta.geoloc`

        Returns
        -------
        xarray.Dataset
            With interpolated variables

        """
        mapping_dataset_geoloc = {
            "latitude": "latitude",
            "longitude": "longitude",
            "incidence": "incidenceAngle",
            "elevation": "elevationAngle",
            "altitude": "height",
        }
        da_list = []

        for varname in varnames:
            varname_in_geoloc = mapping_dataset_geoloc[varname]
            if varname == "incidence":
                da = self._load_incidence_from_lut()
                da.name = varname
                da_list.append(da)
            elif varname == "elevation":
                da = self._load_elevation_from_lut()
                da.name = varname
                da_list.append(da)
            else:
                if varname == "longitude":
                    z_values = self.sar_meta.geoloc[varname]
                    if self.sar_meta.cross_antemeridian:
                        logger.debug("translate longitudes between 0 and 360")
                        z_values = z_values % 360
                else:
                    z_values = self.sar_meta.geoloc[varname_in_geoloc]
                interp_func = RectBivariateSpline(
                    self.sar_meta.geoloc.line,
                    self.sar_meta.geoloc.pixel,
                    z_values,
                    kx=1,
                    ky=1,
                )
                typee = self.sar_meta.geoloc[varname_in_geoloc].dtype
                if lazy_loading:
                    da_var = map_blocks_coords(
                        self._da_tmpl.astype(typee), interp_func)
                else:
                    da_val = interp_func(
                        self._dataset.digital_number.line,
                        self._dataset.digital_number.sample,
                    )
                    da_var = xr.DataArray(
                        data=da_val,
                        dims=["line", "sample"],
                        coords={
                            "line": self._dataset.digital_number.line,
                            "sample": self._dataset.digital_number.sample,
                        },
                    )
                if varname == "longitude":
                    if self.sar_meta.cross_antemeridian:
                        da_var.data = da_var.data.map_blocks(to_lon180)

                da_var.name = varname

                # copy history
                try:
                    da_var.attrs["history"] = self.sar_meta.geoloc[
                        varname_in_geoloc
                    ].attrs["xpath"]
                except KeyError:
                    pass

                da_list.append(da_var)
        ds = xr.merge(da_list)
        return ds

    @timing
    def _load_incidence_from_lut(self):
        """
        load incidence thanks to lut. In the formula to have the incidence, we understand that it is calculated
        thanks to lut. But we can ask ourselves if we consider the denoised ones or not. In this case we have chosen to
        take the not denoised luts.
        Usually look up tables depends on luts, but we already have information about look up tables, so we determine
        incidence thanks to these.
        Reference : `Radarsat2 product format definition` 7.2

        Returns
        -------
        xarray.DataArray
            DataArray of incidence (expressed in degrees)
        """
        beta = self._dataset.beta0_raw[0]
        gamma = self._dataset.gamma0_raw[0]
        incidence_pre = gamma / beta
        i_angle = np.degrees(np.arctan(incidence_pre))
        return xr.DataArray(
            data=i_angle,
            dims=["line", "sample"],
            coords={
                "line": self._dataset.digital_number.line,
                "sample": self._dataset.digital_number.sample,
            },
        )

    @timing
    def _resample_lut_values(self, lut):
        lines = self.sar_meta.geoloc.line
        samples = np.arange(lut.shape[0])
        lut_values_2d = dask.delayed(np.tile)(lut, (lines.shape[0], 1))
        interp_func = dask.delayed(RectBivariateSpline)(
            x=lines, y=samples, z=lut_values_2d, kx=1, ky=1
        )
        """var = inter_func(self._dataset.digital_number.line, self._dataset.digital_number.sample)
        da_var = xr.DataArray(data=var, dims=['line', 'sample'],
                              coords={'line': self._dataset.digital_number.line,
                                      'sample': self._dataset.digital_number.sample})"""
        da_var = map_blocks_coords(
            self._da_tmpl.astype(lut.dtype), interp_func)
        return da_var

    @timing
    def _load_elevation_from_lut(self):
        """
        Load elevation from lut.
        Formula reference : `RSI-GS-026 RS-1 Data Products Specifications` 5.3.3.2.
        this formula needs the orbit altitude. But 2 variables look like this one : `satelliteHeight` and `Altitude`.
        We considered the satelliteHeight.

        Returns
        -------

        """
        satellite_height = self.sar_meta.dt.attrs["satelliteHeight"]
        earth_radius = 6.371e6
        incidence = self._load_incidence_from_lut()
        angle_rad = np.sin(np.radians(incidence))
        inside = angle_rad * earth_radius / (earth_radius + satellite_height)
        return np.degrees(np.arcsin(inside))

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
            raise ValueError(
                "can't find noise lut name for var '%s'" % var_name)
        try:
            lut = self.sar_meta.dt["radarParameters"][lut_name]
        except KeyError:
            raise ValueError(
                "can't find noise lut from name '%s' for variable '%s'"
                % (lut_name, var_name)
            )
        return lut

    @timing
    def _interpolate_for_noise_lut(self, var_name):
        """
        Interpolate the noise level values (from the reader) and resample it to create a noise lut.
        Initial values are at low resolution, and the high resolution range is made from the pixel first noise
        level value and the step. Then, an interpolation with RectBivariateSpline permit having a full resolution
        and extrapolate the first pixels; getting by the end resampled noise values.
        Nb : Noise Level Values extracted from the reader are already calibrated, and expressed in dB
        (so they are converted in linear).

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
        first_pix = initial_lut.attrs["pixelFirstNoiseValue"]
        step = initial_lut.attrs["stepSize"]
        noise_values = 10 ** (initial_lut / 10)
        lines = np.arange(self.sar_meta.geoloc.line[-1] + 1)
        noise_values_2d = np.tile(noise_values, (lines.shape[0], 1))
        indexes = [first_pix + step *
                   i for i in range(0, noise_values.shape[0])]
        interp_func = dask.delayed(RectBivariateSpline)(
            x=lines, y=indexes, z=noise_values_2d, kx=1, ky=1
        )
        da_var = map_blocks_coords(
            self._da_tmpl.astype(self._dtypes["noise_lut"]), interp_func
        )
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
        name = "ne%sz" % var_name[0]
        concat_list = []
        # add pol dim
        for pol in self._dataset.pol:
            lut_noise = (
                self._interpolate_for_noise_lut(var_name)
                .assign_coords(pol=pol)
                .expand_dims("pol")
            )
            concat_list.append(lut_noise)
        return xr.concat(concat_list, dim="pol").to_dataset(name=name)

    def apply_calibration_and_denoising(self):
        """
        apply calibration and denoising functions to get high resolution sigma0 , beta0 and gamma0 + variables *_raw

        Returns:
        --------

        """
        for var_name, lut_name in self._map_var_lut.items():
            if lut_name in self._luts:
                # merge var_name into dataset (not denoised)
                self._dataset = self._dataset.merge(
                    self._apply_calibration_lut(var_name)
                )
                # merge noise equivalent for var_name (named 'ne%sz' % var_name[0)
                self._dataset = self._dataset.merge(self._get_noise(var_name))
            else:
                logger.debug(
                    "Skipping variable '%s' ('%s' lut is missing)"
                    % (var_name, lut_name)
                )
        self._dataset = self._add_denoised(self._dataset)

        for var_name, lut_name in self._map_var_lut.items():
            var_name_raw = var_name + "_raw"
            if var_name_raw in self._dataset:
                self._dataset[var_name_raw] = self._dataset[var_name_raw].where(
                    self._dataset[var_name_raw] > 0, 0)
            else:
                logger.debug(
                    "Skipping variable '%s' ('%s' lut is missing)"
                    % (var_name, lut_name)
                )

        self.datatree["measurement"] = self.datatree["measurement"].assign(
            self._dataset
        )
        # self._dataset = self.datatree[
        #     'measurement'].to_dataset()  # test oct 22 to see if then I can modify variables of the dt
        return

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
            vars = ["sigma0", "beta0", "gamma0"]
        for varname in vars:
            varname_raw = varname + "_raw"
            noise = "ne%sz" % varname[0]
            if varname_raw not in ds:
                continue
            else:
                denoised = ds[varname_raw] - ds[noise]

                if clip:
                    denoised = denoised.clip(min=0)
                    denoised.attrs["comment"] = "clipped, no values <0"
                else:
                    denoised.attrs["comment"] = "not clipped, some values can be <0"
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
        offset = lut.attrs["offset"]
        # if self.resolution is not None:
        lut = self._resample_lut_values(lut)
        res = ((self._dataset.digital_number**2.0) + offset) / lut
        res.attrs.update(lut.attrs)
        return res.to_dataset(name=var_name + "_raw")

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
        antenna_pointing = self.sar_meta.dt["radarParameters"].attrs["antennaPointing"]
        pass_direction = self.sar_meta.dt.attrs["passDirection"]
        flipped_cases = [("Left", "Ascending"), ("Right", "Descending")]
        if (antenna_pointing, pass_direction) in flipped_cases:
            new_ds = (
                ds.copy()
                .isel(sample=slice(None, None, -1))
                .assign_coords(sample=ds.sample)
            )
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
        pass_direction = self.sar_meta.dt.attrs["passDirection"]
        if pass_direction == "Ascending":
            new_ds = (
                ds.copy().isel(line=slice(None, None, -1)).assign_coords(line=ds.line)
            )
        else:
            new_ds = ds.copy()
        return new_ds

    @property
    def interpolate_times(self):
        """
        Apply interpolation with RectBivariateSpline to the azimuth time extracted from `self.sar_meta.geoloc`

        Returns
        -------
        xarray.Dataset
            Contains the time as delayed at the good resolution and expressed as type datetime64[ns]
        """
        times = self.sar_meta.get_azitime
        lines = self.sar_meta.geoloc.line
        samples = self.sar_meta.geoloc.pixel
        time_values_2d = np.tile(times, (samples.shape[0], 1)).transpose()
        interp_func = RectBivariateSpline(
            x=lines, y=samples, z=time_values_2d.astype(float), kx=1, ky=1
        )
        da_var = map_blocks_coords(
            self._da_tmpl.astype("datetime64[ns]"), interp_func)
        return da_var.isel(sample=0).to_dataset(name="time")

    def get_sensor_velocity(self):
        """
        Interpolated sensor velocity
        Returns
        -------
        xarray.Dataset()
            containing a single variable velocity
        """
        azimuth_times = self.sar_meta.get_azitime
        interp_times = self.interpolate_times["time"]
        orbstatevect = self.sar_meta.orbit_and_attitude
        orbstatevect2 = xr.Dataset()
        # number of values must be the same as number of lines
        for var in orbstatevect:
            da = orbstatevect[var]
            interp_func = interp1d(y=da, x=da.timeStamp.astype("float"))
            interp_var = interp_func(azimuth_times.astype("float"))
            orbstatevect2[var] = xr.DataArray(
                data=interp_var,
                dims=["timeStamp"],
                coords={"timeStamp": azimuth_times},
                attrs=da.attrs,
            )
        orbstatevect2.attrs = orbstatevect.attrs
        velos = np.array(
            [
                orbstatevect2["xVelocity"] ** 2.0,
                orbstatevect2["yVelocity"] ** 2.0,
                orbstatevect2["zVelocity"] ** 2.0,
            ]
        )
        vels = np.sqrt(np.sum(velos, axis=0))
        interp_f = interp1d(azimuth_times.astype(float), vels)
        _vels = interp_f(interp_times.astype(float))
        res = xr.DataArray(_vels, dims=["line"], coords={
                           "line": self.dataset.line})
        return xr.Dataset({"velocity": res})

    def _reconfigure_reader_datatree(self):
        """
        Modify the structure of self.datatree to merge the reader's one and the measurement dataset.
        Modify the structure of the reader datatree for a better user experience.
        Include attributes from the reader (concerning the satellite) in the attributes of self.datatree

        Returns
        -------
        xarray.Datatree
        """

        dic = {
            "measurement": self.datatree["measurement"],
            "geolocation_annotation": self.datatree["geolocation_annotation"],
        }

        def del_items_in_dt(dt, list_items):
            for item in list_items:
                dt.to_dataset().__delitem__(item)
            return

        def get_list_keys_delete(dt, list_keys, inside=True):
            dt_keys = dt.keys()
            final_list = []
            for item in dt_keys:
                if inside:
                    if item in list_keys:
                        final_list.append(item)
                else:
                    if item not in list_keys:
                        final_list.append(item)
            return final_list

        exclude = ["geolocationGrid", "lut", "radarParameters"]
        rename_lut = {
            "lutBeta": "beta0_lut",
            "lutGamma": "gamma0_lut",
            "lutSigma": "sigma0_lut",
        }
        rename_radarParameters = {
            "noiseLevelValues_BetaNought": "beta0_noise_lut",
            "noiseLevelValues_SigmaNought": "sigma0_noise_lut",
            "noiseLevelValues_Gamma": "gamma0_noise_lut",
        }
        new_dt = xr.DataTree.from_dict(dic)

        dt = self.sar_meta.dt

        # rename lut
        new_dt["lut"] = dt["lut"].ds.rename(rename_lut)

        # extract noise_lut, rename and put these in a dataset
        new_dt["noise_lut"] = dt["radarParameters"].ds.rename(
            rename_radarParameters)
        new_dt["noise_lut"].attrs = {}  # reset attributes
        delete_list = get_list_keys_delete(
            new_dt["noise_lut"], rename_radarParameters.values(), inside=False
        )
        del_items_in_dt(new_dt["noise_lut"], delete_list)

        # Create a dataset for radar parameters without the noise luts
        new_dt["radarParameters"] = dt["radarParameters"]
        delete_list = get_list_keys_delete(
            new_dt["radarParameters"], rename_radarParameters.keys()
        )
        del_items_in_dt(new_dt["radarParameters"], delete_list)

        # extract others dataset
        copy_dt = dt.copy()
        for key in dt.copy():
            if key not in exclude:
                if key == "imageGenerationParameters":
                    new_dt[key] = xr.DataTree(children=copy_dt[key])
                else:
                    new_dt[key] = copy_dt[key]
        self.datatree = new_dt
        self.datatree.attrs.update(self.sar_meta.dt.attrs)
        return

    def __repr__(self):
        if self.sliced:
            intro = "sliced"
        else:
            intro = "full coverage"
        return "<RadarSat2Dataset %s object>" % intro

    @property
    def dataset(self):
        """
        `xarray.Dataset` representation of this `xsar.RadarSat2Dataset` object.
        This property can be set with a new dataset, if the dataset was computed from the original dataset.
        """
        # return self._dataset
        res = self.datatree["measurement"].to_dataset()
        res.attrs = self.datatree.attrs
        return res

    @dataset.setter
    def dataset(self, ds):
        if self.sar_meta.name == ds.attrs["name"]:
            # check if new ds has changed coordinates
            if not self.sliced:
                self.sliced = any(
                    [
                        list(ds[d].values) != list(self._dataset[d].values)
                        for d in ["line", "sample"]
                    ]
                )
            self._dataset = ds
            # self._dataset = self.datatree['measurement'].ds
            self.recompute_attrs()
        else:
            raise ValueError("dataset must be same kind as original one.")

    @dataset.deleter
    def dataset(self):
        logger.debug("deleter dataset")

    @property
    def rs2meta(self):
        logger.warning("Please use `sar_meta` to call the sar meta object")
        return self.sar_meta
