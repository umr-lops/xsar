# -*- coding: utf-8 -*-
import glob
import logging
import os
import warnings

import rasterio
import yaml
from affine import Affine

from .rcm_meta import RcmMeta
import numpy as np
from .utils import timing, map_blocks_coords, BlockingActorProxy, to_lon180, get_glob
from scipy.interpolate import RectBivariateSpline, interp1d
import dask
import xarray as xr
import rioxarray
from .base_dataset import BaseDataset

logger = logging.getLogger("xsar.radarsat2_dataset")
logger.addHandler(logging.NullHandler())

# we know tiff as no geotransform : ignore warning
warnings.filterwarnings(
    "ignore", category=rasterio.errors.NotGeoreferencedWarning)

# allow nan without warnings
# some dask warnings are still non filtered: https://github.com/dask/dask/issues/3245
np.errstate(invalid="ignore")


class RcmDataset(BaseDataset):
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
            # TODO : Remove `velocity` from the skip_variable when the problem is resolved
            #  (must understand link between time and lines)
            skip_variables = ["velocity"]
        # default dtypes
        if dtypes is not None:
            self._dtypes.update(dtypes)

        # default meta for map_blocks output.
        # as asarray is imported from numpy, it's a numpy array.
        # but if later we decide to import asarray from cupy, il will be a cupy.array (gpu)
        self.sar_meta = None
        self.resolution = resolution

        if not isinstance(dataset_id, RcmMeta):
            self.sar_meta = BlockingActorProxy(RcmMeta, dataset_id)
            # check serializable
            # import pickle
            # s1meta = pickle.loads(pickle.dumps(self.s1meta))
            # assert isinstance(rs2meta.coords2ll(100, 100),tuple)
        else:
            # we want self.rs2meta to be a dask actor on a worker
            self.sar_meta = BlockingActorProxy(
                RcmMeta.from_dict, dataset_id.dict)
        del dataset_id

        if self.sar_meta.multidataset:
            raise IndexError(
                """Can't open an multi-dataset. Use `xsar.RadarSat2Meta('%s').subdatasets` to show availables ones"""
                % self.sar_meta.path
            )

        # build datatree
        DN_tmp = self.load_digital_number(
            resolution=resolution, resampling=resampling, chunks=chunks
        )
        DN_tmp = self.flip_sample_da(DN_tmp)
        DN_tmp = self.flip_line_da(DN_tmp)

        # geoloc
        geoloc = self.sar_meta.geoloc
        geoloc.attrs["history"] = "annotations"

        # orbitInformation
        orbit = self.sar_meta.orbit
        orbit.attrs["history"] = "annotations"

        self.datatree = xr.DataTree.from_dict(
            {"measurement": DN_tmp, "geolocation_annotation": geoloc}
        )

        self._dataset = self.datatree["measurement"].to_dataset()

        # merge the datatree with the reader
        for group in self.sar_meta.dt:
            self.datatree[group] = self.sar_meta.dt[group]

        # dict mapping for calibration type in the reader
        self._map_calibration_type = {
            "sigma0": "Sigma Nought",
            "gamma0": "Gamma",
            "beta0": "Beta Nought",
        }

        self._map_var_lut = {
            "sigma0": "sigma0",
            "gamma0": "gamma0",
            "beta0": "beta0",
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

        value_res_line = self.sar_meta.pixel_line_m
        value_res_sample = self.sar_meta.pixel_sample_m
        # self._load_incidence_from_lut()
        refe_spacing = "slant"
        if resolution is not None:
            # if the data sampling changed it means that the quantities are projected on ground
            refe_spacing = "ground"
            if isinstance(resolution, str):
                value_res_sample = float(resolution.replace("m", ""))
                value_res_line = value_res_sample
            elif isinstance(resolution, dict):
                value_res_sample = self.sar_meta.pixel_sample_m * \
                    resolution["sample"]
                value_res_line = self.sar_meta.pixel_line_m * \
                    resolution["line"]
            else:
                logger.warning(
                    "resolution type not handle (%s) should be str or dict -> sampleSpacing"
                    " and lineSpacing are not correct",
                    type(resolution),
                )
        self._dataset["sampleSpacing"] = xr.DataArray(
            value_res_sample,
            attrs={"referential": refe_spacing}
            | self.sar_meta.dt["imageReferenceAttributes/rasterAttributes"][
                "sampledPixelSpacing"
            ].attrs,
        )
        self._dataset["lineSpacing"] = xr.DataArray(
            value_res_line,
            attrs=self.sar_meta.dt["imageReferenceAttributes/rasterAttributes"][
                "sampledLineSpacing"
            ].attrs,
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
        self._noise_luts = self.lazy_load_noise_luts()
        self._noise_luts = self._noise_luts.drop(
            ["pixelFirstNoiseValue", "stepSize"])
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
        if "ground_heading" not in skip_variables:
            self._dataset = xr.merge(
                [self.load_ground_heading(), self._dataset])
        if "velocity" not in skip_variables:
            self._dataset = xr.merge(
                [self.get_sensor_velocity(), self._dataset])
        self._rasterized_masks = self.load_rasterized_masks()
        self._dataset = xr.merge([self._rasterized_masks, self._dataset])
        self.datatree["measurement"] = self.datatree["measurement"].assign(
            self._dataset
        )
        # merge the datatree with the reader

        self.reconfigure_reader_datatree()
        self._dataset.attrs.update(self.sar_meta.to_dict("all"))
        self.datatree.attrs.update(self.sar_meta.to_dict("all"))

    def lazy_load_luts(self):
        """
        Lazy load luts from the reader as delayed

        Returns
        -------
        xarray.Dataset
            Contains delayed dataArrays of luts
        """
        merge_list = []
        for key, value in self._map_calibration_type.items():
            list_da = []
            for pola in self.sar_meta.lut.lookup_tables.pole:
                lut = self.sar_meta.lut.lookup_tables.sel(
                    sarCalibrationType=value, pole=pola
                ).rename(
                    {
                        "pixel": "sample",
                    }
                )
                values_nb = lut.attrs["numberOfValues"]
                lut_f_delayed = dask.delayed()(lut)
                ar = dask.array.from_delayed(
                    lut_f_delayed.data, (values_nb,), lut.dtype
                )
                da = xr.DataArray(
                    data=ar,
                    dims=["sample"],
                    coords={"sample": lut.sample},
                    attrs=lut.attrs,
                )
                da = (
                    self._interpolate_var(da)
                    .assign_coords(pole=pola)
                    .rename({"pole": "pol"})
                )
                list_da.append(da)
            full_da = xr.concat(list_da, dim="pol")
            full_da.attrs = lut.attrs
            ds_lut_f_delayed = full_da.to_dataset(name=key)
            merge_list.append(ds_lut_f_delayed)
        return xr.merge(merge_list)

    def lazy_load_noise_luts(self):
        """
        Lazy load noise luts from the reader as delayed

        Returns
        -------
        xarray.Dataset
            Contains delayed dataArrays of luts
        """
        merge_list = []
        for key, value in self._map_calibration_type.items():
            list_da = []
            values_nb = self.sar_meta.noise_lut.attrs["numberOfValues"]
            for pola in self.sar_meta.noise_lut.noiseLevelValues.pole:
                lut = self.sar_meta.noise_lut.noiseLevelValues.sel(
                    sarCalibrationType=value, pole=pola
                ).rename(
                    {
                        "pixel": "sample",
                    }
                )
                lut_f_delayed = dask.delayed()(lut)
                ar = dask.array.from_delayed(
                    lut_f_delayed.data, (values_nb,), lut.dtype
                )
                da = xr.DataArray(
                    data=ar,
                    dims=["sample"],
                    coords={"sample": lut.sample},
                    attrs=lut.attrs,
                )
                da = (
                    self._interpolate_var(da, type="noise")
                    .assign_coords(pole=pola)
                    .rename({"pole": "pol"})
                )
                list_da.append(da)
            full_da = xr.concat(list_da, dim="pol")
            ds_lut_f_delayed = full_da.to_dataset(name=key)
            merge_list.append(ds_lut_f_delayed)
        return xr.merge(merge_list)

    @timing
    def _interpolate_var(self, var, type="lut"):
        """
        Interpolate look up table (from the reader) or another variable and resample it.
        Initial values are at low resolution, and the high resolution range is made from the pixel first noise
        level value and the step. Then, an interpolation with RectBivariateSpline permit having a full resolution
        and extrapolate the first pixels; getting by the end resampled look up tables.

        Parameters
        ----------
        var: xarray.DataArray
            variable we want to interpolate (lut extracted from the reader :noise or calibration ; incidence; elevation)

        type: str
            type of variable we want to interpolate. Can be "lut", "noise", "incidence"

        Returns
        -------
        xarray.DataArray
            Variable interpolated and resampled
        """
        accepted_types = ["lut", "noise", "incidence"]
        if type not in accepted_types:
            raise ValueError(
                "Please enter a type accepted ('lut', 'noise', 'incidence')"
            )
        lines = self.sar_meta.geoloc.line
        samples = var.sample
        var_type = None
        if type == "noise":
            # Give the good saving type and convert the noise values to linear
            var_type = self._dtypes["noise_lut"]
            var = 10 ** (var / 10)
        elif type == "lut":
            var_type = self._dtypes["sigma0_lut"]
        elif type == "incidence":
            var_type = self._dtypes[type]
        var_2d = np.tile(var, (lines.shape[0], 1))
        interp_func = dask.delayed(RectBivariateSpline)(
            x=lines, y=samples, z=var_2d, kx=1, ky=1
        )
        da_var = map_blocks_coords(self._da_tmpl.astype(var_type), interp_func)
        return da_var

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
        res = ((self._dataset.digital_number**2.0) + offset) / lut
        res.attrs.update(lut.attrs)
        return res.to_dataset(name=var_name + "_raw")

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
        try:
            lut_name = self._map_var_lut[var_name]
        except KeyError:
            raise ValueError("can't find lut name for var '%s'" % var_name)
        try:
            lut = self._noise_luts[lut_name]
        except KeyError:
            raise ValueError(
                "can't find noise lut from name '%s' for variable '%s' "
                % (lut_name, var_name)
            )
        return lut.to_dataset(name=name)

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
            self._dataset)

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
        already_denoised = self.datatree["imageGenerationParameters"][
            "sarProcessingInformation"
        ].attrs["noiseSubtractionPerformed"]

        if vars is None:
            vars = ["sigma0", "beta0", "gamma0"]
        for varname in vars:
            varname_raw = varname + "_raw"
            noise = "ne%sz" % varname[0]
            if varname_raw not in ds:
                continue
            else:

                if not already_denoised:
                    denoised = ds[varname_raw] - ds[noise]
                    if clip:
                        denoised = denoised.clip(min=0)
                        denoised.attrs["comment"] = "clipped, no values <0"
                    else:
                        denoised.attrs["comment"] = "not clipped, some values can be <0"
                    ds[varname] = denoised
                    
                    ds[varname_raw].attrs[
                        "denoising information"
                    ] = f"product was not denoised"

                else:
                    logging.debug(
                        "product was already denoised (noiseSubtractionPerformed = True), noise added back"
                    )
                    denoised = ds[varname_raw]
                    denoised.attrs["denoising information"] = (
                        "already denoised by Canadian Space Agency"
                    )
                    if clip:
                        denoised = denoised.clip(min=0)
                        denoised.attrs["comment"] = "clipped, no values <0"
                    else:
                        denoised.attrs["comment"] = "not clipped, some values can be <0"
                    ds[varname] = denoised

                    ds[varname_raw] = denoised + ds[noise]
                    ds[varname_raw].attrs[
                        "denoising information"
                    ] = "product was already denoised by Canadian Space Agency, noise added back"

        return ds

    def _load_incidence_from_lut(self):
        """
        Load incidence from the reader as delayed

        Returns
        -------
        xarray.Dataset
            Contains delayed dataArrays of incidence
        """
        incidence = self.sar_meta.incidence.rename({"pixel": "sample"})
        angles = incidence.angles
        values_nb = incidence.attrs["numberOfValues"]
        lut_f_delayed = dask.delayed()(angles)
        ar = dask.array.from_delayed(
            lut_f_delayed.data, (values_nb,), self._dtypes["incidence"]
        )
        da = xr.DataArray(
            data=ar,
            dims=["sample"],
            coords={"sample": angles.sample},
            attrs=angles.attrs,
        )
        da = self._interpolate_var(da, type="incidence")
        # ds_lut_f_delayed = da.to_dataset(name='incidence')
        # ds_lut_f_delayed.attrs = incidence.attrs
        return da

    @ timing
    def _load_elevation_from_lut(self):
        """
        Load elevation from lut.
        this formula needs the orbit altitude. But 2 variables look like this one : `satelliteHeight` and `Altitude`.
        We considered the satelliteHeight.

        Returns
        -------

        """
        satellite_height = (
            self.sar_meta.dt["imageGenerationParameters/sarProcessingInformation"]
            .to_dataset()
            .satelliteHeight
        )
        earth_radius = 6.371e6
        incidence = self._load_incidence_from_lut()
        angle_rad = np.sin(np.radians(incidence))
        inside = angle_rad * earth_radius / (earth_radius + satellite_height)
        return np.degrees(np.arcsin(inside))

    @ timing
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

    @ timing
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

    @ property
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
        orbstatevect = self.sar_meta.orbit
        velos = np.array(
            [
                orbstatevect["xVelocity"] ** 2.0,
                orbstatevect["yVelocity"] ** 2.0,
                orbstatevect["zVelocity"] ** 2.0,
            ]
        )
        vels = np.sqrt(np.sum(velos, axis=0))
        interp_f = interp1d(azimuth_times.astype(float), vels)
        _vels = interp_f(self.interpolate_times["time"].astype(float))
        res = xr.DataArray(_vels, dims=["line"], coords={
            "line": self.dataset.line})
        return xr.Dataset({"velocity": res})

    @ timing
    def load_digital_number(
        self, resolution=None, chunks=None, resampling=rasterio.enums.Resampling.rms
    ):
        """
        load digital_number from tiff files, as an `xarray.Dataset`.

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

        def _list_tiff_files():
            """
            Return a list that contains all tiff files paths

            Returns
            -------
            List[str]
                List of Tiff file paths located in a folder
            """

            return glob.glob(os.path.join(self.sar_meta.path, "imagery", "*.tif"))

        def _sort_list_files_and_get_pols(list_tiff):
            """
            From a list of tiff files, sort it to get the co polarization tiff file as the first element, and extract pols

            Parameters
            ----------
            list_tiff: List[str]
                List of tiff files

            Returns
            -------
            (List[str], List[str])
                Tuple that contains the tiff files list sorted and the polarizations
            """
            pols = []
            if len(list_tiff) > 1:
                first_base = os.path.basename(list_tiff[0]).split(".")[0]
                first_pol = first_base[-2:]
                if first_pol[0] != first_pol[1]:
                    list_tiff.reverse()
            for file in list_tiff:
                base = os.path.basename(file).split(".")[0]
                pol = base[-2:]
                pols.append(pol)
            return list_tiff, pols

        map_dims = {"pol": "band", "line": "y", "sample": "x"}
        tiff_files = _list_tiff_files()
        tiff_files, pols = _sort_list_files_and_get_pols(tiff_files)
        if resolution is not None:
            comment = 'resampled at "%s" with %s.%s.%s' % (
                resolution,
                resampling.__module__,
                resampling.__class__.__name__,
                resampling.name,
            )
        else:
            comment = "read at full resolution"

        # arbitrary rio object, to get shape, etc ... (will not be used to read data)
        rio = rasterio.open(tiff_files[0])

        chunks["pol"] = 1
        # sort chunks keys like map_dims
        chunks = dict(
            sorted(
                chunks.items(), key=lambda pair: list(map_dims.keys()).index(pair[0])
            )
        )
        chunks_rio = {map_dims[d]: chunks[d] for d in map_dims.keys()}
        self.resolution = None
        if resolution is None:
            # using tiff driver: need to read individual tiff and concat them
            # riofiles['rio'] is ordered like self.s1meta.manifest_attrs['polarizations']

            dn = xr.concat(
                [
                    rioxarray.open_rasterio(
                        f, chunks=chunks_rio, parse_coordinates=False
                    )
                    for f in tiff_files
                ],
                "band",
            ).assign_coords(band=np.arange(len(pols)) + 1)

            # set dimensions names
            dn = dn.rename(dict(zip(map_dims.values(), map_dims.keys())))

            # create coordinates from dimension index (because of parse_coordinates=False)
            dn = dn.assign_coords({"line": dn.line, "sample": dn.sample})
            dn = dn.drop_vars("spatial_ref", errors="ignore")
        else:
            if not isinstance(resolution, dict):
                if isinstance(resolution, str) and resolution.endswith("m"):
                    resolution = float(resolution[:-1])
                    self.resolution = resolution
                resolution = dict(
                    line=resolution / self.sar_meta.pixel_line_m,
                    sample=resolution / self.sar_meta.pixel_sample_m,
                )
                # resolution = dict(line=resolution / self.dataset['sampleSpacing'].values,
                #                   sample=resolution / self.dataset['lineSpacing'].values)

            # resample the DN at gdal level, before feeding it to the dataset
            out_shape = (
                int(rio.height / resolution["line"]),
                int(rio.width / resolution["sample"]),
            )
            out_shape_pol = (1,) + out_shape
            # read resampled array in one chunk, and rechunk
            # this doesn't optimize memory, but total size remain quite small

            if isinstance(resolution["line"], int):
                # legacy behaviour: winsize is the maximum full image size that can be divided  by resolution (int)
                winsize = (
                    0,
                    0,
                    rio.width // resolution["sample"] * resolution["sample"],
                    rio.height // resolution["line"] * resolution["line"],
                )
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
                                window=window,
                            ),
                            chunks=chunks_rio,
                        ),
                        dims=tuple(map_dims.keys()),
                        coords={"pol": [pol]},
                    )
                    for f, pol in zip(tiff_files, pols)
                ],
                "pol",
            ).chunk(chunks)

            # create coordinates at box center
            translate = Affine.translation(
                (resolution["sample"] - 1) / 2, (resolution["line"] - 1) / 2
            )
            scale = Affine.scale(
                rio.width // resolution["sample"] *
                resolution["sample"] / out_shape[1],
                rio.height // resolution["line"] *
                resolution["line"] / out_shape[0],
            )
            sample, _ = translate * scale * (dn.sample, 0)
            _, line = translate * scale * (0, dn.line)
            dn = dn.assign_coords({"line": line, "sample": sample})

        # for GTiff driver, pols are already ordered. just rename them
        dn = dn.assign_coords(pol=pols)

        descr = "not denoised"
        var_name = "digital_number"

        dn.attrs = {
            "comment": "%s digital number, %s" % (descr, comment),
            "history": yaml.safe_dump(
                {
                    var_name: get_glob(
                        [p.replace(self.sar_meta.path + "/", "")
                         for p in tiff_files]
                    )
                }
            ),
        }
        ds = dn.to_dataset(name=var_name)
        astype = self._dtypes.get(var_name)
        if astype is not None:
            ds = ds.astype(self._dtypes[var_name])

        return ds

    def __repr__(self):
        if self.sliced:
            intro = "sliced"
        else:
            intro = "full coverage"
        return "<RcmDataset %s object>" % intro

    @ timing
    def flip_sample_da(self, ds):
        """
        When a product is flipped, flip back data arrays (from a dataset) sample dimensions to respect the xsar
        convention (increasing incidence values).
        Documentation reference : RCM Image Product Format Definition (4.2.1)

        Parameters
        ----------
        ds : xarray.Dataset
            Contains dataArrays which depends on `sample` dimension

        Returns
        -------
        xarray.Dataset
            Flipped back, respecting the xsar convention
        """
        antenna_pointing = self.sar_meta.dt["sourceAttributes/radarParameters"].attrs[
            "antennaPointing"
        ]
        pass_direction = self.sar_meta.dt[
            "sourceAttributes/orbitAndAttitude/orbitInformation"
        ].attrs["passDirection"]
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

    @ timing
    def flip_line_da(self, ds):
        """
        Flip dataArrays (from a dataset) that depend on line dimension when a product is ascending, in order to
        respect the xsar convention (increasing time along line axis, whatever ascending or descending product).
        Documentation reference : RCM Image Product Format Definition (4.2.1)

        Parameters
        ----------
        ds : xarray.Dataset
            Contains dataArrays which depends on `line` dimension

        Returns
        -------
        xarray.Dataset
            Flipped back, respecting the xsar convention
        """
        pass_direction = self.sar_meta.dt[
            "sourceAttributes/orbitAndAttitude/orbitInformation"
        ].attrs["passDirection"]
        if pass_direction == "Ascending":
            new_ds = (
                ds.copy().isel(line=slice(None, None, -1)).assign_coords(line=ds.line)
            )
        else:
            new_ds = ds.copy()
        return new_ds

    def reconfigure_reader_datatree(self):
        """
        Merge attributes of the reader's datatree in the attributes of self.datatree
        """
        self.datatree.attrs |= self.sar_meta.dt.attrs
        return

    @ property
    def dataset(self):
        """
        `xarray.Dataset` representation of this `xsar.RcmDataset` object.
        This property can be set with a new dataset, if the dataset was computed from the original dataset.
        """
        # return self._dataset
        res = self.datatree["measurement"].to_dataset()
        res.attrs = self.datatree.attrs
        return res

    @ dataset.setter
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

    @ dataset.deleter
    def dataset(self):
        logger.debug("deleter dataset")
