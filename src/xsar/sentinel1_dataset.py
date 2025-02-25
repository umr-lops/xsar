# -*- coding: utf-8 -*-

import logging
import warnings
import numpy as np
import xarray
from scipy.interpolate import RectBivariateSpline
import xarray as xr
import dask
import rasterio.features
from scipy.interpolate import interp1d
from shapely.geometry import box

from .utils import (
    timing,
    map_blocks_coords,
    BlockingActorProxy,
    merge_yaml,
    to_lon180,
    config,

    get_path_aux_cal,
    get_path_aux_pp1,
    get_geap_gains,
    get_gproc_gains,
)
from .sentinel1_meta import Sentinel1Meta
from .ipython_backends import repr_mimebundle
from .base_dataset import BaseDataset
import pandas as pd
import geopandas as gpd

import os


logger = logging.getLogger("xsar.sentinel1_dataset")
logger.addHandler(logging.NullHandler())

# we know tiff as no geotransform : ignore warning
warnings.filterwarnings(
    "ignore", category=rasterio.errors.NotGeoreferencedWarning)

# allow nan without warnings
# some dask warnings are still non filtered: https://github.com/dask/dask/issues/3245
np.errstate(invalid="ignore")

mapping_dataset_geoloc = {
    "latitude": "latitude",
    "longitude": "longitude",
    "incidence": "incidenceAngle",
    "elevation": "elevationAngle",
    "altitude": "height",
    "azimuth_time": "azimuthTime",
    "slant_range_time": "slantRangeTime",
    "offboresight": "offboresightAngle",
}


# noinspection PyTypeChecker
class Sentinel1Dataset(BaseDataset):
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

    lazyloading: bool, optional
        activate or not the lazy loading of the high resolution fields

    """

    def __init__(
        self,
        dataset_id,
        resolution=None,
        resampling=rasterio.enums.Resampling.rms,
        luts=False,
        chunks={"line": 5000, "sample": 5000},
        dtypes=None,
        patch_variable=True,
        lazyloading=True,
        recalibration=False,
    ):
        # default dtypes
        if dtypes is not None:
            self._dtypes.update(dtypes)
        # default meta for map_blocks output.
        # as asarray is imported from numpy, it's a numpy array.
        # but if later we decide to import asarray from cupy, il will be a cupy.array (gpu)
        self.sar_meta = None
        self.interpolation_func_slc = {}
        """`xsar.Sentinel1Meta` object"""

        if not isinstance(dataset_id, Sentinel1Meta):
            self.sar_meta = BlockingActorProxy(Sentinel1Meta, dataset_id)
            # check serializable
            # import pickle
            # sar_meta = pickle.loads(pickle.dumps(self.sar_meta))
            # assert isinstance(sar_meta.coords2ll(100, 100),tuple)
        else:
            # we want self.sar_meta to be a dask actor on a worker
            self.sar_meta = BlockingActorProxy(
                Sentinel1Meta.from_dict, dataset_id.dict)
        del dataset_id

        if self.sar_meta.multidataset:
            raise IndexError(
                """Can't open an multi-dataset. Use `xsar.Sentinel1Meta('%s').subdatasets` to show availables ones"""
                % self.sar_meta.path
            )
        # security to prevent using resolution argument with SLC
        if (
            self.sar_meta.product == "SLC"
            and resolution is not None
            and self.sar_meta.swath in ["IW", "EW"]
        ):
            # we tolerate resampling for WV since image width is only 20 km
            logger.error(
                "xsar is not handling resolution change for SLC TOPS products."
            )
            raise Exception(
                "xsar is not handling resolution change for SLC TOPS products."
            )

        # build datatree
        self.resolution, DN_tmp = self.sar_meta.reader.load_digital_number(
            resolution=resolution, resampling=resampling, chunks=chunks
        )

        # geoloc
        geoloc = self.sar_meta.geoloc
        geoloc.attrs["history"] = "annotations"

        #  offboresight angle
        geoloc["offboresightAngle"] = (
            geoloc.elevationAngle
            - (
                30.1833947 * geoloc.latitude**0
                + 0.0082998714 * geoloc.latitude**1
                - 0.00031181534 * geoloc.latitude**2
                - 0.0943533e-07 * geoloc.latitude**3
                + 3.0191435e-08 * geoloc.latitude**4
                + 4.968415428e-12 * geoloc.latitude**5
                - 9.315371305e-13 * geoloc.latitude**6
            )
            + 29.45
        )
        geoloc["offboresightAngle"].attrs[
            "comment"
        ] = "built from elevation angle and latitude"

        # bursts
        bu = self.sar_meta._bursts
        bu.attrs["history"] = "annotations"

        # azimuth fm rate
        FM = self.sar_meta.azimuth_fmrate
        FM.attrs["history"] = "annotations"

        # doppler
        dop = self.sar_meta._doppler_estimate
        dop.attrs["history"] = "annotations"

        # calibration LUTs
        ds_luts = self.sar_meta.get_calibration_luts
        ds_luts.attrs["history"] = "calibration"

        # noise levels LUTs
        ds_noise_range = self.sar_meta.get_noise_range_raw
        ds_noise_azi = self.sar_meta.get_noise_azi_raw

        # antenna pattern
        ds_antenna_pattern = self.sar_meta.get_antenna_pattern

        # swath merging
        ds_swath_merging = self.sar_meta.get_swath_merging

        if self.sar_meta.swath == "WV":
            # since WV noise is not defined on azimuth we apply the patch on range noise
            # ds_noise_azi['noise_lut'] = self._patch_lut(ds_noise_azi[
            #                                                 'noise_lut'])  # patch applied here is distinct to same patch applied on interpolated noise LUT
            ds_noise_range["noise_lut"] = self._patch_lut(
                ds_noise_range["noise_lut"]
            )  # patch applied here is distinct to same patch applied on interpolated noise LUT

        self.datatree = xr.DataTree.from_dict(
            {
                "measurement": DN_tmp,
                "geolocation_annotation": geoloc,
                "bursts": bu,
                "FMrate": FM,
                "doppler_estimate": dop,
                # 'image_information':
                "orbit": self.sar_meta.orbit,
                "image": self.sar_meta.image,
                "calibration": ds_luts,
                "noise_range": ds_noise_range,
                "noise_azimuth": ds_noise_azi,
                "antenna_pattern": ds_antenna_pattern,
                "swath_merging": ds_swath_merging,
            }
        )

        # apply recalibration ?
        self.apply_recalibration = recalibration
        if self.apply_recalibration and (
            self.sar_meta.swath != "EW" and self.sar_meta.swath != "IW"
        ):
            self.apply_recalibration = False
            raise ValueError(
                f"Recalibration in only done for EW/IW modes. You have '{self.sar_meta.swath}'. apply_recalibration is set to False."
            )

        if self.apply_recalibration and np.all(
            np.isnan(self.datatree.antenna_pattern["roll"].values)
        ):
            self.apply_recalibration = False
            raise ValueError(
                f"Recalibration can't be done without roll angle. You probably work with an old file for which roll angle is not in auxiliary file.")

        # self.datatree['measurement'].ds = .from_dict({'measurement':self._load_digital_number(resolution=resolution, resampling=resampling, chunks=chunks)
        # self._dataset = self.datatree['measurement'].ds #the two variables should be linked then.
        self._dataset = self.datatree[
            "measurement"
        ].to_dataset()  # test oct 22 to see if then I can modify variables of the dt

        # create a datatree for variables used in recalibration
        self.datatree["recalibration"] = xr.DataTree()
        self._dataset_recalibration = xr.Dataset(
            coords=self.datatree["measurement"].coords
        )

        for att in ["name", "short_name", "product", "safe", "swath", "multidataset"]:
            if att not in self.datatree.attrs:
                # tmp = xr.DataArray(self.sar_meta.__getattr__(att),attrs={'source':'filename decoding'})
                self.datatree.attrs[att] = self.sar_meta.__getattr__(att)
                self._dataset.attrs[att] = self.sar_meta.__getattr__(att)

        self._dataset = xr.merge(
            [xr.Dataset({"time": self.get_burst_azitime}), self._dataset]
        )
        value_res_sample = self.sar_meta.image["slantRangePixelSpacing"]
        value_res_line = self.sar_meta.image["azimuthPixelSpacing"]
        refe_spacing = "slant"
        if resolution is not None:
            # if the data sampling changed it means that the quantities are projected on ground
            refe_spacing = "ground"
            if isinstance(resolution, str):
                value_res_sample = float(resolution.replace("m", ""))
                value_res_line = value_res_sample
            elif isinstance(resolution, dict):
                value_res_sample = (
                    self.sar_meta.image["slantRangePixelSpacing"] *
                    resolution["sample"]
                )
                value_res_line = (
                    self.sar_meta.image["azimuthPixelSpacing"] *
                    resolution["line"]
                )
            else:
                logger.warning(
                    "resolution type not handle (%s) should be str or dict -> sampleSpacing"
                    " and lineSpacing are not correct",
                    type(resolution),
                )
        self._dataset["sampleSpacing"] = xarray.DataArray(
            value_res_sample, attrs={"units": "m", "referential": refe_spacing}
        )
        self._dataset["lineSpacing"] = xarray.DataArray(
            value_res_line, attrs={"units": "m"}
        )
        # dataset principal

        # dataset no-pol template for function evaluation on coordinates (*no* values used)
        # what's matter here is the shape of the image, not the values.
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore", np.ComplexWarning) # deprecated in numpy>=2.0.0
            if self.sar_meta._bursts["burst"].size != 0:
                # SLC TOPS, tune the high res grid because of bursts overlapping
                # line_time = self._burst_azitime
                line_time = self.get_burst_azitime
                self._da_tmpl = xr.DataArray(
                    dask.array.empty_like(
                        np.empty(
                            (len(line_time), len(self._dataset.digital_number.sample))
                        ),
                        dtype=np.int8,
                        name="empty_var_tmpl-%s"
                        % dask.base.tokenize(self.sar_meta.name),
                    ),
                    dims=("line", "sample"),
                    coords={
                        "line": self._dataset.digital_number.line,
                        "sample": self._dataset.digital_number.sample,
                        "line_time": line_time.astype(float),
                    },
                )
            else:

                self._da_tmpl = xr.DataArray(
                    dask.array.empty_like(
                        self._dataset.digital_number.isel(pol=0).drop("pol"),
                        dtype=np.int8,
                        name="empty_var_tmpl-%s"
                        % dask.base.tokenize(self.sar_meta.name),
                    ),
                    dims=("line", "sample"),
                    coords={
                        "line": self._dataset.digital_number.line,
                        "sample": self._dataset.digital_number.sample,
                    },
                )
        # FIXME possible memory leak
        # when calling a self.sar_meta method, an ActorFuture is returned.
        # But this seems to break __del__ methods from both Sentinel1Meta and XmlParser
        # Is it a memory leak ?
        # see https://github.com/dask/distributed/issues/5610
        # tmp_f = self.sar_meta.to_dict("all")
        # del tmp_f
        # return
        self._dataset.attrs.update(self.sar_meta.to_dict("all"))
        self.datatree["measurement"] = self.datatree["measurement"].assign(
            self._dataset
        )
        self.datatree.attrs.update(self.sar_meta.to_dict("all"))

        # load land_mask by default for GRD products

        if "GRD" in str(self.datatree.attrs["product"]):
            self.add_high_resolution_variables(
                patch_variable=patch_variable, luts=luts, lazy_loading=lazyloading
            )
            if self.apply_recalibration:
                self.select_gains()
            self.apply_calibration_and_denoising()

        # added 6 fev 23, to fill  empty attrs
        self.datatree["measurement"].attrs = self.datatree.attrs
        if (
            self.sar_meta.product == "SLC" and "WV" not in self.sar_meta.swath
        ):  # TOPS cases
            tmp = self.corrected_range_noise_lut(self.datatree)
            # the corrcted noise_range dataset shold now contain an attrs 'corrected_range_noise_lut'
            self.datatree["noise_range"].ds = tmp
        self.sliced = False
        """True if dataset is a slice of original L1 dataset"""

        self.resampled = resolution is not None
        """True if dataset is not a sensor resolution"""

        # save original bbox
        self._bbox_coords_ori = self._bbox_coords

    def corrected_range_noise_lut(self, dt):
        """
        Patch F.Nouguier see https://jira-projects.cls.fr/browse/MPCS-3581 and https://github.com/umr-lops/xsar_slc/issues/175
        Return range noise lut with corrected line numbering. This function should be used only on the full SLC dataset dt
        Args:
            dt (xarray.datatree) : datatree returned by xsar corresponding to one subswath
        Return:
            (xarray.dataset) : range noise lut with corrected line number
        """
        # Detection of azimuthTime jumps (linked to burst changes). Burst sensingTime should NOT be used since they have erroneous value too !
        line_shift = 0
        tt = dt["measurement"]["time"]
        i_jump = np.ravel(
            np.argwhere(np.diff(tt) < np.timedelta64(0)) + 1
        )  # index of jumps
        # line number of jumps
        line_jump_meas = dt["measurement"]["line"][i_jump]
        # line_jump_noise = np.ravel(dt['noise_range']['line'][1:-1].data) # annotated line number of burst begining, this one is corrupted for some S1 TOPS product
        # annoted line number of burst begining
        line_jump_noise = np.ravel(
            dt["noise_range"]["line"][1: 1 + len(line_jump_meas)].data
        )
        burst_first_lineshift = line_jump_meas - line_jump_noise
        if len(np.unique(burst_first_lineshift)) == 1:
            line_shift = int(np.unique(burst_first_lineshift)[0])
            logging.debug("line_shift: %s", line_shift)
        else:
            raise ValueError(
                "Inconsistency in line shifting : {}".format(
                    burst_first_lineshift)
            )
        res = dt["noise_range"].ds.assign_coords(
            {"line": dt["noise_range"]["line"] + line_shift}
        )
        if line_shift == 0:
            res.attrs["corrected_range_noise_lut"] = "no change"
        else:
            res.attrs["corrected_range_noise_lut"] = "shift : %i lines" % line_shift
        return res

    def select_gains(self):
        """
        attribution of the good swath gain by getting the swath number of each pixel

        Returns:
        --------

        """

        def get_gains(ds, var_template):
            if self.sar_meta.swath == "EW":
                resultat = xr.where(
                    ds["swath_number"] == 1,
                    ds[var_template + "_1"],
                    xr.where(
                        ds["swath_number"] == 2,
                        ds[var_template + "_2"],
                        xr.where(
                            ds["swath_number"] == 3,
                            ds[var_template + "_3"],
                            xr.where(
                                ds["swath_number"] == 4,
                                ds[var_template + "_4"],
                                xr.where(
                                    ds["swath_number"] == 5,
                                    ds[var_template + "_5"],
                                    np.nan,
                                ),
                            ),
                        ),
                    ),
                )
                return resultat
            elif self.sar_meta.swath == "IW":
                resultat = xr.where(
                    ds["swath_number"] == 1,
                    ds[var_template + "_1"],
                    xr.where(
                        ds["swath_number"] == 2,
                        ds[var_template + "_2"],
                        xr.where(
                            ds["swath_number"] == 3, ds[var_template + "_3"], np.nan
                        ),
                    ),
                )
                return resultat
            else:
                raise ValueError(
                    f"Recalibration in only done for EW/IW modes. You have '{self.sar_meta.swath}'"
                )

        for var_template in ["old_geap", "new_geap", "old_gproc", "new_gproc"]:
            self._dataset_recalibration[var_template] = get_gains(
                self._dataset_recalibration, var_template
            )
            # vars_to_drop.extend([f"{var_template}_{i}" for i in range(1, 6)])
        # self._dataset = self._dataset.drop_vars(vars_to_drop,errors = "ignore")

        self.datatree["recalibration"] = self.datatree["recalibration"].assign(
            self._dataset_recalibration
        )

    def add_high_resolution_variables(
        self,
        luts=False,
        patch_variable=True,
        skip_variables=None,
        load_luts=True,
        lazy_loading=True,
    ):
        """
        Parameters
        ----------

        luts: bool, optional
            if `True` return also luts as variables (ie `sigma0_lut`, `gamma0_lut`, etc...). False by default.

        patch_variable: bool, optional
            activate or not variable pathching ( currently noise lut correction for IPF2.9X)

        skip_variables: list, optional
            list of strings eg ['land_mask','longitude'] to skip at rasterisation step

        load_luts: bool
            True -> load hiddens luts sigma0 beta0 gamma0, False -> no luts reading

        lazy_loading : bool
            True -> use map_blocks_coords() to have delayed rasterization on variables such as longitude, latitude, incidence,..., False -> directly compute RectBivariateSpline with memory usage
            (Currently the lazy_loading generate a memory leak)
        """
        if "longitude" in self.dataset:
            logger.debug(
                "the high resolution variable such as : longitude, latitude, incidence,.. are already visible in the dataset"
            )
        else:
            # miscellaneous attributes that are not know from xml files
            attrs_dict = {
                "pol": {"comment": "ordered polarizations (copol, crosspol)"},
                "line": {
                    "units": "1",
                    "comment": "azimuth direction, in pixels from full resolution tiff",
                },
                "sample": {
                    "units": "1",
                    "comment": "cross track direction, in pixels from full resolution tiff",
                },
                "sigma0_raw": {"units": "linear"},
                "gamma0_raw": {"units": "linear"},
                "nesz": {"units": "linear", "comment": "sigma0 noise"},
                "negz": {"units": "linear", "comment": "beta0 noise"},
            }
            # dict mapping for variables names to create by applying specified lut on digital_number
            self._map_var_lut = {
                "sigma0_raw": "sigma0_lut",
                "gamma0_raw": "gamma0_lut",
            }

            # dict mapping for lut names to file type (from self.files columns)
            self._map_lut_files = {
                "sigma0_lut": "calibration",
                "gamma0_lut": "calibration",
                "noise_lut_range": "noise",
                "noise_lut_azi": "noise",
            }

            # dict mapping specifying if the variable has 'pol' dimension
            self._vars_with_pol = {
                "sigma0_lut": True,
                "gamma0_lut": True,
                "noise_lut_range": True,
                "noise_lut_azi": True,
                "incidence": False,
                "elevation": False,
                "altitude": False,
                "azimuth_time": False,
                "slant_range_time": False,
                "longitude": False,
                "latitude": False,
            }
            if skip_variables is None:
                skip_variables = []
            # variables not returned to the user (unless luts=True)
            # self._hidden_vars = ['sigma0_lut', 'gamma0_lut', 'noise_lut', 'noise_lut_range', 'noise_lut_azi']
            self._hidden_vars = []
            # attribute to activate correction on variables, if available
            self._patch_variable = patch_variable
            if load_luts:
                self._luts = self._lazy_load_luts(self._map_lut_files.keys())

                # noise_lut is noise_lut_range * noise_lut_azi
                if (
                    "noise_lut_range" in self._luts.keys()
                    and "noise_lut_azi" in self._luts.keys()
                ):
                    self._luts = self._luts.assign(
                        noise_lut=self._luts.noise_lut_range * self._luts.noise_lut_azi
                    )
                    self._luts.noise_lut.attrs["history"] = merge_yaml(
                        [
                            self._luts.noise_lut_range.attrs["history"]
                            + self._luts.noise_lut_azi.attrs["history"]
                        ],
                        section="noise_lut",
                    )

                ds_merge_list = [
                    self._dataset,  # self.load_ground_heading(),  # lon_lat
                    self._luts.drop_vars(self._hidden_vars, errors="ignore"),
                ]
            else:
                ds_merge_list = [self._dataset]

            if "ground_heading" not in skip_variables:
                ds_merge_list.append(self.load_ground_heading())
            self._rasterized_masks = self.load_rasterized_masks()
            ds_merge_list.append(self._rasterized_masks)
            # self.add_rasterized_masks() #this method update the datatree while in this part of the code, the dataset is updated

            if luts:
                ds_merge_list.append(self._luts[self._hidden_vars])
            attrs = self._dataset.attrs
            self._dataset = xr.merge(ds_merge_list)
            self._dataset.attrs = attrs
            geoloc_vars = [
                "altitude",
                "azimuth_time",
                "slant_range_time",
                "incidence",
                "elevation",
                "longitude",
                "latitude",
                "offboresight",
            ]

            for vv in skip_variables:
                if vv in geoloc_vars:
                    geoloc_vars.remove(vv)

            self._dataset = self._dataset.merge(
                self._load_from_geoloc(geoloc_vars, lazy_loading=lazy_loading)
            )

            if "GRD" in str(self.datatree.attrs["product"]):
                self.add_swath_number()
                path_aux_cal_old = get_path_aux_cal(
                    self.sar_meta.manifest_attrs["aux_cal"]
                )

                path_aux_pp1_old = get_path_aux_pp1(
                    self.sar_meta.manifest_attrs["aux_pp1"]
                )

                if self.apply_recalibration == False:
                    new_cal = "None"
                    new_pp1 = "None"

                if self.apply_recalibration:
                    path_dataframe_aux = config["path_dataframe_aux"]
                    dataframe_aux = pd.read_csv(path_dataframe_aux)

                    sel_cal = dataframe_aux.loc[(dataframe_aux.sat_name == self.sar_meta.manifest_attrs['satellite']) &
                                                (dataframe_aux.aux_type == "CAL") &
                                                (dataframe_aux.icid == int(self.sar_meta.manifest_attrs['icid'])) &
                                                (dataframe_aux.validation_date <= self.sar_meta.start_date)]
                    # Check if sel_cal is empty
                    if sel_cal.empty:
                        sel_ = dataframe_aux.loc[(dataframe_aux.sat_name == self.sar_meta.manifest_attrs['satellite']) &
                                                 (dataframe_aux.aux_type == "CAL") &
                                                 (dataframe_aux.icid == int(self.sar_meta.manifest_attrs['icid']))]
                        if sel_.empty:
                            raise ValueError(
                                f"Cannot recalibrate - No matching CAL data found for your data : start_date : {self.sar_meta.start_date}.")
                        else:
                            raise ValueError(f"Cannot recalibrate - No matching CAL data found for your data: start_date: {self.sar_meta.start_date} & \
                                miniumum validation date in active aux files: {sel_.sort_values(by=['validation_date'], ascending=False).validation_date.values[0]}.")

                    sel_cal = sel_cal.sort_values(
                        by=["validation_date", "generation_date"], ascending=False)

                    new_cal = sel_cal.iloc[0].aux_path

                    sel_pp1 = dataframe_aux.loc[(dataframe_aux.sat_name == self.sar_meta.manifest_attrs['satellite']) &
                                                (dataframe_aux.aux_type == "PP1") &
                                                (dataframe_aux.icid == int(self.sar_meta.manifest_attrs['icid'])) &
                                                (dataframe_aux.validation_date <= self.sar_meta.start_date)]

                    # Check if sel_pp1 is empty
                    if sel_pp1.empty:
                        sel_ = dataframe_aux.loc[(dataframe_aux.sat_name == self.sar_meta.manifest_attrs['satellite']) &
                                                 (dataframe_aux.aux_type == "PP1") &
                                                 (dataframe_aux.icid == int(self.sar_meta.manifest_attrs['icid']))]
                        if sel_.empty:
                            raise ValueError(
                                f"Cannot recalibrate - No matching PP1 data found for your data : start_date : {self.sar_meta.start_date}.")
                        else:
                            raise ValueError(f"Cannot recalibrate - No matching PP1 data found for your data: start_date: {self.sar_meta.start_date} & \
                                miniumum validation date in active aux files: {sel_.sort_values(by=['validation_date'], ascending=False).validation_date.values[0]}.")
                    sel_pp1 = sel_pp1.sort_values(
                        by=["validation_date", "generation_date"], ascending=False
                    )
                    new_pp1 = sel_pp1.iloc[0].aux_path

                    path_aux_cal_new = get_path_aux_cal(
                        os.path.basename(new_cal))
                    path_aux_pp1_new = get_path_aux_pp1(
                        os.path.basename(new_pp1))

                    self.add_gains(path_aux_cal_new, path_aux_pp1_new,
                                   path_aux_cal_old, path_aux_pp1_old)

                self.datatree["recalibration"].attrs["aux_cal_new"] = os.path.basename(
                    new_cal)
                self.datatree["recalibration"].attrs["aux_pp1_new"] = os.path.basename(
                    new_pp1)

            rasters = self._load_rasters_vars()
            if rasters is not None:
                self._dataset = xr.merge([self._dataset, rasters])
            if "velocity" not in skip_variables:
                self._dataset = self._dataset.merge(self.get_sensor_velocity())
            if "range_ground_spacing" not in skip_variables:
                self._dataset = self._dataset.merge(
                    self._range_ground_spacing())

            # set miscellaneous attrs
            for var, attrs in attrs_dict.items():
                try:
                    self._dataset[var].attrs.update(attrs)
                except KeyError:
                    pass
            # self.datatree[
            #     'measurement'].ds = self._dataset  # last link to make sure all previous modifications are also in the datatree
            # need high resolution rasterised longitude and latitude , needed for .rio accessor
            self.recompute_attrs()
            self.datatree["measurement"] = self.datatree["measurement"].assign(
                self._dataset
            )
            if "land_mask" in skip_variables:
                self._dataset = self._dataset.drop("land_mask")
                self.datatree["measurement"] = self.datatree["measurement"].assign(
                    self._dataset
                )
            else:
                assert "land_mask" in self.datatree["measurement"]
                if (
                    self.sar_meta.product == "SLC" and "WV" not in self.sar_meta.swath
                ):  # TOPS cases
                    logger.debug("a TOPS product")
                    # self.land_mask_slc_per_bursts(
                    #    lazy_loading=lazy_loading)  # replace "GRD" like (Affine transform) land_mask by a burst-by-burst rasterised land mask
                else:
                    logger.debug(
                        "not a TOPS product -> land_mask already available.")
        return

    def add_swath_number(self):
        """
        add a DataArray with the swath number chosen for each pixel of the dataset ;
        also add a DataArray with a flag

        Returns:
        --------
        """
        swath_tab = xr.DataArray(
            np.full_like(self._dataset.elevation, np.nan, dtype=int),
            coords={
                "line": self._dataset.coords["line"],
                "sample": self._dataset.coords["sample"],
            },
        )
        flag_tab = xr.DataArray(
            np.zeros_like(self._dataset.elevation, dtype=int),
            coords={
                "line": self._dataset.coords["line"],
                "sample": self._dataset.coords["sample"],
            },
        )

        # Supposons que dataset.swaths ait 45 éléments comme mentionné
        for i in range(len(self.datatree["swath_merging"].ds["swaths"])):
            y_min, y_max = (
                self.datatree["swath_merging"]["firstAzimuthLine"][i],
                self.datatree["swath_merging"]["lastAzimuthLine"][i],
            )
            x_min, x_max = (
                self.datatree["swath_merging"]["firstRangeSample"][i],
                self.datatree["swath_merging"]["lastRangeSample"][i],
            )

            # Localisation des pixels appartenant à ce swath
            swath_index = int(self.datatree["swath_merging"].ds["swaths"][i])

            condition = (
                (self._dataset.line >= y_min)
                & (self._dataset.line <= y_max)
                & (self._dataset.sample >= x_min)
                & (self._dataset.sample <= x_max)
            )

            # Marquer les pixels déjà vus
            flag_tab = xr.where(
                (flag_tab == 1) & condition & (
                    swath_tab == swath_index), 2, flag_tab
            )

            # Affecter le swath actuel
            swath_tab = xr.where(condition, swath_index, swath_tab)

            # Marquer les premiers pixels vus
            flag_tab = xr.where((flag_tab == 0) & condition, 1, flag_tab)

        self._dataset_recalibration["swath_number"] = swath_tab
        self._dataset_recalibration["swath_number_flag"] = flag_tab
        self._dataset_recalibration["swath_number_flag"].attrs[
            "flag_info"
        ] = "0 : no swath \n1 : unique swath \n2 : undecided swath"

        self.datatree["recalibration"] = self.datatree["recalibration"].assign(
            self._dataset_recalibration
        )

    def add_gains(self, path_aux_cal_new, path_aux_pp1_new, path_aux_cal_old, path_aux_pp1_old):

        from scipy.interpolate import interp1d

        logger.debug(
            f"doing recalibration with AUX_CAL = {path_aux_cal_new} & AUX_PP1 = {path_aux_pp1_new}"
        )

        #  1 - compute offboresight angle
        roll = self.datatree["antenna_pattern"]["roll"]
        azimuthTime = self.datatree["antenna_pattern"]["azimuthTime"]
        interp_roll = interp1d(
            azimuthTime.values.flatten().astype(int),
            roll.values.flatten(),
            kind="linear",
            fill_value="extrapolate",
        )

        self._dataset_recalibration = self._dataset_recalibration.assign(
            rollAngle=(["line", "sample"], interp_roll(
                self._dataset.azimuth_time))
        )
        self._dataset_recalibration = self._dataset_recalibration.assign(
            offboresigthAngle=(
                ["line", "sample"],
                self._dataset["elevation"].data
                - self._dataset_recalibration["rollAngle"].data,
            )
        )

        # 2- get gains geap and map them
        dict_geap_old = get_geap_gains(
            path_aux_cal_old,
            mode=self.sar_meta.manifest_attrs["swath_type"],
            pols=self.sar_meta.manifest_attrs["polarizations"],
        )
        dict_geap_new = get_geap_gains(
            path_aux_cal_new,
            mode=self.sar_meta.manifest_attrs["swath_type"],
            pols=self.sar_meta.manifest_attrs["polarizations"],
        )

        for key, infos_geap in dict_geap_old.items():
            pol = key[-2:]
            number = key[2:-3]

            keyf = "old_geap_" + number

            if keyf not in self._dataset_recalibration:
                data_shape = (
                    len(self._dataset_recalibration.coords["line"]),
                    len(self._dataset_recalibration.coords["sample"]),
                    len(self._dataset_recalibration.coords["pol"]),
                )
                self._dataset_recalibration[keyf] = xr.DataArray(
                    np.full(data_shape, np.nan),
                    coords={
                        "line": self._dataset_recalibration.coords["line"],
                        "sample": self._dataset_recalibration.coords["sample"],
                        "pol": self._dataset_recalibration.coords["pol"],
                    },
                    dims=["line", "sample", "pol"],
                )  # coords=self._dataset.coords))

                self._dataset_recalibration[keyf].attrs["aux_path"] = os.path.join(
                    os.path.basename(
                        os.path.dirname(os.path.dirname(path_aux_cal_old))
                    ),
                    os.path.basename(os.path.dirname(path_aux_cal_old)),
                    os.path.basename(path_aux_cal_old),
                )

            interp = interp1d(
                infos_geap["offboresightAngle"], infos_geap["gain"], kind="linear"
            )
            self._dataset_recalibration[keyf].loc[:, :, pol] = interp(
                self._dataset_recalibration["offboresigthAngle"]
            )

        for key, infos_geap in dict_geap_new.items():
            pol = key[-2:]
            number = key[2:-3]

            keyf = "new_geap_" + number

            if keyf not in self._dataset_recalibration:
                data_shape = (
                    len(self._dataset_recalibration.coords["line"]),
                    len(self._dataset_recalibration.coords["sample"]),
                    len(self._dataset_recalibration.coords["pol"]),
                )
                self._dataset_recalibration[keyf] = xr.DataArray(
                    np.full(data_shape, np.nan),
                    coords={
                        "line": self._dataset_recalibration.coords["line"],
                        "sample": self._dataset_recalibration.coords["sample"],
                        "pol": self._dataset_recalibration.coords["pol"],
                    },
                    dims=["line", "sample", "pol"],
                )  # coords=self._dataset.coords))

                self._dataset_recalibration[keyf].attrs["aux_path"] = os.path.join(
                    os.path.basename(
                        os.path.dirname(os.path.dirname(path_aux_cal_new))
                    ),
                    os.path.basename(os.path.dirname(path_aux_cal_new)),
                    os.path.basename(path_aux_cal_new),
                )

            interp = interp1d(
                infos_geap["offboresightAngle"], infos_geap["gain"], kind="linear"
            )
            self._dataset_recalibration[keyf].loc[:, :, pol] = interp(
                self._dataset_recalibration["offboresigthAngle"]
            )

        # 3- get gains gproc and map them
        dict_gproc_old = get_gproc_gains(
            path_aux_pp1_old,
            mode=self.sar_meta.manifest_attrs["swath_type"],
            product=self.sar_meta.product,
        )
        dict_gproc_new = get_gproc_gains(
            path_aux_pp1_new,
            mode=self.sar_meta.manifest_attrs["swath_type"],
            product=self.sar_meta.product,
        )

        for idxpol, pol in enumerate(["HH", "HV", "VV", "VH"]):
            if pol in self.sar_meta.manifest_attrs["polarizations"]:
                valid_keys_indices = [
                    (key, idxpol, pol) for key, infos_gproc in dict_gproc_old.items()
                ]
                for key, idxpol, pol in valid_keys_indices:
                    sw_nb = str(key)[-1]
                    keyf = "old_gproc_" + sw_nb
                    if keyf not in self._dataset_recalibration:
                        self._dataset_recalibration[keyf] = xr.DataArray(
                            np.nan,
                            dims=["pol"],
                            coords={
                                "pol": self._dataset_recalibration.coords["pol"]},
                        )
                        self._dataset_recalibration[keyf].attrs["aux_path"] = (
                            os.path.join(
                                os.path.basename(
                                    os.path.dirname(
                                        os.path.dirname(path_aux_pp1_old))
                                ),
                                os.path.basename(
                                    os.path.dirname(path_aux_pp1_old)),
                                os.path.basename(path_aux_pp1_old),
                            )
                        )
                    self._dataset_recalibration[keyf].loc[..., pol] = dict_gproc_old[
                        key
                    ][idxpol]

        for idxpol, pol in enumerate(["HH", "HV", "VV", "VH"]):
            if pol in self.sar_meta.manifest_attrs["polarizations"]:
                valid_keys_indices = [
                    (key, idxpol, pol) for key, infos_gproc in dict_gproc_new.items()
                ]
                for key, idxpol, pol in valid_keys_indices:
                    sw_nb = str(key)[-1]
                    keyf = "new_gproc_" + sw_nb
                    if keyf not in self._dataset_recalibration:
                        self._dataset_recalibration[keyf] = xr.DataArray(
                            np.nan,
                            dims=["pol"],
                            coords={
                                "pol": self._dataset_recalibration.coords["pol"]},
                        )
                        self._dataset_recalibration[keyf].attrs["aux_path"] = (
                            os.path.join(
                                os.path.basename(
                                    os.path.dirname(
                                        os.path.dirname(path_aux_pp1_new))
                                ),
                                os.path.basename(
                                    os.path.dirname(path_aux_pp1_new)),
                                os.path.basename(path_aux_pp1_new),
                            )
                        )
                    self._dataset_recalibration[keyf].loc[..., pol] = dict_gproc_new[
                        key
                    ][idxpol]

        self.datatree["recalibration"] = self.datatree["recalibration"].assign(
            self._dataset_recalibration
        )

        # return self._dataset

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

        if self.apply_recalibration:
            interest_var = ["sigma0_raw", "gamma0_raw", "beta0_raw"]
            for var in interest_var:
                if var not in self._dataset:
                    continue

                var_dB = 10 * np.log10(self._dataset[var])

                corrected_dB = (
                    var_dB
                    + 10 * np.log10(self._dataset_recalibration["old_geap"])
                    - 10 * np.log10(self._dataset_recalibration["new_geap"])
                    - 2 * 10 *
                    np.log10(self._dataset_recalibration["old_gproc"])
                    + 2 * 10 *
                    np.log10(self._dataset_recalibration["new_gproc"])
                )

                self._dataset_recalibration[var + "__corrected"] = 10 ** (
                    corrected_dB / 10
                )
            self.datatree["recalibration"] = self.datatree["recalibration"].assign(
                self._dataset_recalibration
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

    def __del__(self):
        logger.debug("__del__")

    @property
    def dataset(self):
        """
        `xarray.Dataset` representation of this `xsar.Sentinel1Dataset` object.
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

    # @property
    # def pixel_line_m(self):
    #     """line pixel spacing, in meters (relative to dataset)"""
    #     return self.sar_meta.pixel_line_m * np.unique(np.round(np.diff(self._dataset['line'].values), 1))[0]

    # @property
    # def pixel_sample_m(self):
    #     """sample pixel spacing, in meters (relative to dataset)"""
    #     return self.sar_meta.pixel_sample_m * np.unique(np.round(np.diff(self._dataset['sample'].values), 1))[0]

    def _patch_lut(self, lut):
        """
        patch proposed by MPC Sentinel-1 : https://jira-projects.cls.fr/browse/MPCS-2007 for noise vectors of WV SLC
        IPF2.9X products adjustment proposed by BAE are the same for HH and VV, and suppose to work for both old and
        new WV2 EAP they were estimated using WV image with very low NRCS (black images) and computing std(sigma0).
        Parameters
        ----------
        lut xarray.Dataset

        Returns
        -------
        lut xarray.Dataset
        """
        if self.sar_meta.swath == "WV":
            if (
                lut.name in ["noise_lut_azi", "noise_lut"]
                and self.sar_meta.ipf_version in [2.9, 2.91]
                and self.sar_meta.platform in ["SENTINEL-1A", "SENTINEL-1B"]
            ):
                noise_calibration_cst_pp1 = {
                    "SENTINEL-1A": {"WV1": -38.13, "WV2": -36.84},
                    "SENTINEL-1B": {
                        "WV1": -39.30,
                        "WV2": -37.44,
                    },
                }
                subswath = str(self.sar_meta.image["swath_subswath"].values)
                cst_db = noise_calibration_cst_pp1[self.sar_meta.platform][subswath]
                cst_lin = 10 ** (cst_db / 10)
                lut = lut * cst_lin
                lut.attrs["comment"] = "patch on the noise_lut_azi : %s dB" % cst_db
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

        class _NoiseLut:
            """small internal class that return a lut function(lines, samples) defined on all the image, from blocks in the image"""

            def __init__(self, blocks):
                self.blocks = blocks

            def __call__(self, lines, samples):
                """return noise[a.size,x.size], by finding the intersection with blocks and calling the corresponding block.lut_f"""
                if len(self.blocks) == 0:
                    # no noise (ie no azi noise for ipf < 2.9)
                    return 1
                else:
                    # the array to be returned
                    noise = xr.DataArray(
                        np.ones((lines.size, samples.size)) * np.nan,
                        dims=("line", "sample"),
                        coords={"line": lines, "sample": samples},
                    )
                    # find blocks that intersects with asked_box
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # the box coordinates of the returned array
                        asked_box = box(
                            max(0, lines[0] - 0.5),
                            max(0, samples[0] - 0.5),
                            lines[-1] + 0.5,
                            samples[-1] + 0.5,
                        )
                        # set match_blocks as the non empty intersection with asked_box
                        match_blocks = self.blocks.copy()
                        match_blocks.geometry = self.blocks.geometry.intersection(
                            asked_box
                        )
                        match_blocks = match_blocks[~match_blocks.is_empty]
                    for i, block in match_blocks.iterrows():
                        (sub_a_min, sub_x_min, sub_a_max, sub_x_max) = map(
                            int, block.geometry.bounds
                        )
                        sub_a = lines[(lines >= sub_a_min) &
                                      (lines <= sub_a_max)]
                        sub_x = samples[(samples >= sub_x_min)
                                        & (samples <= sub_x_max)]
                        noise.loc[dict(line=sub_a, sample=sub_x)] = block.lut_f(
                            sub_a, sub_x
                        )

                # values returned as np array
                return noise.values

        def noise_lut_range(lut_range):
            """

            Parameters
            ----------
            lines: np.ndarray
                1D array of lines. lut is defined at each line
            samples: list of np.ndarray
                arrays of samples. list length is same as samples. each array define samples where lut is defined
            noiseLuts: list of np.ndarray
                arrays of luts. Same structure as samples.

            Returns
            -------
            geopandas.GeoDataframe
                noise range geometry.
                'geometry' is the polygon where 'lut_f' is defined.
                attrs['type'] set to 'sample'


            """

            class Lut_box_range:
                def __init__(self, a_start, a_stop, x, lll):
                    self.lines = np.arange(a_start, a_stop)
                    self.samples = x
                    self.area = box(a_start, x[0], a_stop, x[-1])
                    self.lut_f = interp1d(
                        x,
                        lll,
                        kind="linear",
                        fill_value=np.nan,
                        assume_sorted=True,
                        bounds_error=False,
                    )

                def __call__(self, lines, samples):
                    lut = np.tile(self.lut_f(samples), (lines.size, 1))
                    return lut

            blocks = []
            lines = lut_range.line
            samples = np.tile(lut_range.sample, (len(lines), 1))
            noiseLuts = lut_range.noise_lut
            # lines is where lut is defined. compute lines interval validity
            lines_start = (lines - np.diff(lines, prepend=0) / 2).astype(int)
            lines_stop = np.ceil(
                lines + np.diff(lines, append=lines[-1] + 1) / 2
            ).astype(
                int
            )  # end is not included in the interval
            # be sure to include all image if last azimuth line, is not last azimuth image
            lines_stop[-1] = 65535
            for a_start, a_stop, x, lll in zip(
                lines_start, lines_stop, samples, noiseLuts
            ):
                lut_f = Lut_box_range(a_start, a_stop, x, lll)
                block = pd.Series(
                    dict([("lut_f", lut_f), ("geometry", lut_f.area)]))
                blocks.append(block)

            # to geopandas
            blocks = pd.concat(blocks, axis=1).T
            blocks = gpd.GeoDataFrame(blocks)

            return _NoiseLut(blocks)

        def noise_lut_azi():
            """
            Parameters
            ----------
            line_azi
            line_azi_start
            line_azi_stop
            sample_azi_start
            sample_azi_stop
            noise_azi_lut
            swath

            Returns
            -------
            geopandas.GeoDataframe
                noise range geometry.
                'geometry' is the polygon where 'lut_f' is defined.
                attrs['type'] set to 'line'
            """

            class Lut_box_azi:
                def __init__(self, sw, a, a_start, a_stop, x_start, x_stop, lut):
                    self.lines = a
                    self.samples = np.arange(x_start, x_stop + 1)
                    self.area = box(
                        max(0, a_start - 0.5),
                        max(0, x_start - 0.5),
                        a_stop + 0.5,
                        x_stop + 0.5,
                    )
                    if len(lut) > 1:
                        self.lut_f = interp1d(
                            a,
                            lut,
                            kind="linear",
                            fill_value="extrapolate",
                            assume_sorted=True,
                            bounds_error=False,
                        )
                    else:
                        # not enought values to do interpolation
                        # noise will be constant on this box!
                        self.lut_f = lambda _a: lut

                def __call__(self, lines, samples):
                    return np.tile(self.lut_f(lines), (samples.size, 1)).T

            blocks = []

            (
                swath,
                lines,
                line_start,
                line_stop,
                sample_start,
                sample_stop,
                noise_lut,
            ) = self.sar_meta.reader.get_noise_azi_initial_parameters(pol)

            for sw, a, a_start, a_stop, x_start, x_stop, lut in zip(
                swath,
                lines,
                line_start,
                line_stop,
                sample_start,
                sample_stop,
                noise_lut,
            ):
                lut_f = Lut_box_azi(sw, a, a_start, a_stop,
                                    x_start, x_stop, lut)
                block = pd.Series(
                    dict([("lut_f", lut_f), ("geometry", lut_f.area)]))
                blocks.append(block)

            if len(blocks) == 0:
                # no azi noise (ipf < 2.9) or WV
                blocks.append(
                    pd.Series(
                        dict(
                            [
                                ("lines", np.array([])),
                                ("samples", np.array([])),
                                ("lut_f", lambda a, x: 1),
                                ("geometry", box(0, 0, 65535, 65535)),
                            ]
                        )
                    )
                )  # arbitrary large box (bigger than whole image)

            # to geopandas
            blocks = pd.concat(blocks, axis=1).T
            blocks = gpd.GeoDataFrame(blocks)

            return _NoiseLut(blocks)

        def signal_lut(lut):
            lut_f = RectBivariateSpline(lut.line, lut.sample, lut, kx=1, ky=1)
            return lut_f

        # get the lut in metadata. Lut name must be in self._map_lut_files.keys()
        _get_lut_meta = {
            "sigma0_lut": self.sar_meta.get_calibration_luts.sigma0_lut,
            "gamma0_lut": self.sar_meta.get_calibration_luts.gamma0_lut,
            "noise_lut_range": self.sar_meta.get_noise_range_raw,
            "noise_lut_azi": self.sar_meta.get_noise_azi_raw,
        }
        # map the func to apply for each lut. Lut name must be in self._map_lut_files.keys()
        _map_func = {
            "sigma0_lut": signal_lut,
            "gamma0_lut": signal_lut,
            "noise_lut_range": noise_lut_range,
            "noise_lut_azi": noise_lut_azi,
        }
        luts_list = []
        luts = None
        for lut_name in luts_names:
            raw_lut = _get_lut_meta[lut_name]
            for pol in raw_lut.pol.values:
                if self._vars_with_pol[lut_name]:
                    name = "%s_%s" % (lut_name, pol)
                else:
                    name = lut_name

                # get the lut function. As it takes some time to parse xml, make it delayed
                if lut_name == "noise_lut_azi":
                    # noise_lut_azi doesn't need the raw_lut
                    lut_f_delayed = dask.delayed(_map_func[lut_name])()
                else:
                    lut_f_delayed = dask.delayed(_map_func[lut_name])(
                        raw_lut.sel(pol=pol)
                    )
                lut = map_blocks_coords(
                    self._da_tmpl.astype(self._dtypes[lut_name]),
                    lut_f_delayed,
                    name="blocks_%s" % name,
                )
                # needs to add pol dim ?
                if self._vars_with_pol[lut_name]:
                    lut = lut.assign_coords(pol=pol).expand_dims("pol")

                # set xml file and xpath used as history
                histo = raw_lut.attrs["history"]
                lut.name = lut_name
                if self._patch_variable:
                    lut = self._patch_lut(lut)
                lut.attrs["history"] = histo
                lut = lut.to_dataset()

                luts_list.append(lut)
            luts = xr.combine_by_coords(luts_list)
        return luts

    @timing
    def _load_from_geoloc(self, varnames, lazy_loading=True):
        """
        Interpolate (with RectBiVariateSpline) variables from `self.sar_meta.geoloc` to `self._dataset`

        Parameters
        ----------
        varnames: list of str
            subset of variables names in `self.sar_meta.geoloc`

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
            rbs = kwargs["rbs"]

            def wrapperfunc(*args, **kwargs):
                rbs2 = args[2]
                return rbs2(args[0], args[1], grid=False)

            return wrapperfunc(
                vect1dazti[:, np.newaxis], vect1dxtrac[np.newaxis, :], rbs
            )

        for varname in varnames:
            varname_in_geoloc = mapping_dataset_geoloc[varname]
            if varname in ["azimuth_time"]:
                z_values = self.sar_meta.geoloc[varname_in_geoloc].astype(
                    float)
            elif varname == "longitude":
                z_values = self.sar_meta.geoloc[varname_in_geoloc]
                if self.sar_meta.cross_antemeridian:
                    logger.debug("translate longitudes between 0 and 360")
                    z_values = z_values % 360
            else:
                z_values = self.sar_meta.geoloc[varname_in_geoloc]
            if self.sar_meta._bursts["burst"].size != 0:
                # TOPS SLC
                rbs = RectBivariateSpline(
                    self.sar_meta.geoloc.azimuthTime[:, 0].astype(float),
                    self.sar_meta.geoloc.sample,
                    z_values,
                    kx=1,
                    ky=1,
                )
                interp_func = interp_func_slc
            else:
                rbs = None
                interp_func = RectBivariateSpline(
                    self.sar_meta.geoloc.line,
                    self.sar_meta.geoloc.sample,
                    z_values,
                    kx=1,
                    ky=1,
                )
            # the following take much cpu and memory, so we want to use dask
            # interp_func(self._dataset.line, self.dataset.sample)
            typee = self.sar_meta.geoloc[varname_in_geoloc].dtype

            if self.sar_meta._bursts["burst"].size != 0:
                datemplate = self._da_tmpl.astype(typee).copy()
                # replace the line coordinates by line_time coordinates
                datemplate = datemplate.assign_coords(
                    {"line": datemplate.coords["line_time"]}
                )
                if lazy_loading:
                    da_var = map_blocks_coords(
                        datemplate, interp_func, func_kwargs={"rbs": rbs}
                    )
                    # put back the real line coordinates
                    da_var = da_var.assign_coords(
                        {"line": self._dataset.digital_number.line}
                    )
                else:
                    line_time = self.get_burst_azitime
                    XX, YY = np.meshgrid(
                        line_time.astype(
                            float), self._dataset.digital_number.sample
                    )
                    da_var = rbs(XX, YY, grid=False)
                    da_var = xr.DataArray(
                        da_var.T,
                        coords={
                            "line": self._dataset.digital_number.line,
                            "sample": self._dataset.digital_number.sample,
                        },
                        dims=["line", "sample"],
                    )
            else:
                if lazy_loading:
                    da_var = map_blocks_coords(
                        self._da_tmpl.astype(typee), interp_func)
                else:
                    da_var = interp_func(
                        self._dataset.digital_number.line,
                        self._dataset.digital_number.sample,
                    )
            if varname == "longitude":
                if self.sar_meta.cross_antemeridian:
                    da_var.data = da_var.data.map_blocks(to_lon180)

            da_var.name = varname

            # copy history
            try:
                da_var.attrs["history"] = self.sar_meta.geoloc[varname_in_geoloc].attrs[
                    "history"
                ]
            except KeyError:
                pass

            da_list.append(da_var)

        return xr.merge(da_list)

    def get_ll_from_SLC_geoloc(self, line, sample, varname):
        """

        Parameters
        ----------
            line (np.ndarray) : azimuth times at high resolution
            sample (np.ndarray): range coords
            varname (str): e.g. longitude , latitude , incidence (any variable that is given in geolocation grid annotations
        Returns
        -------
            z_interp_value (np.ndarray):
        """
        varname_in_geoloc = mapping_dataset_geoloc[varname]
        if varname in ["azimuth_time"]:
            z_values = self.sar_meta.geoloc[varname_in_geoloc].astype(float)
        elif varname == "longitude":
            z_values = self.sar_meta.geoloc[varname_in_geoloc]
            # if self.sar_meta.cross_antemeridian:
            #     logger.debug('translate longitudes between 0 and 360')
            #     z_values = z_values % 360
        else:
            z_values = self.sar_meta.geoloc[varname_in_geoloc]
        if varname not in self.interpolation_func_slc:
            rbs = RectBivariateSpline(
                self.sar_meta.geoloc.azimuthTime[:, 0].astype(float),
                self.sar_meta.geoloc.sample,
                z_values,
                kx=1,
                ky=1,
            )
            self.interpolation_func_slc[varname] = rbs
        line_time = self.get_burst_azitime
        line_az_times_values = line_time.values[line]
        z_interp_value = self.interpolation_func_slc[varname](
            line_az_times_values, sample, grid=False
        )
        return z_interp_value

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
        res = np.abs(self._dataset.digital_number) ** 2.0 / (lut**2)
        astype = self._dtypes.get(var_name)
        if astype is not None:
            res = res.astype(astype)

        res.attrs.update(lut.attrs)
        res.attrs["history"] = merge_yaml(
            [lut.attrs["history"]], section=var_name)
        res.attrs["references"] = (
            "https://sentinel.esa.int/web/sentinel/radiometric-calibration-of-level-1-products"
        )

        return res.to_dataset(name=var_name)

    def reverse_calibration_lut(self, var_name):
        """
        ONLY MADE FOR GRD YET
        can't retrieve complex number for SLC

        Reverse the calibration Look Up Table (LUT) applied to `var_name` to retrieve the original digital number (DN).
        This is the inverse operation of `_apply_calibration_lut`.
        Refer to the official ESA documentation for more details on the radiometric calibration of Level-1 products:
        https://sentinel.esa.int/web/sentinel/radiometric-calibration-of-level-1-products

        Parameters
        ----------
        var_name: str
            The variable name from which the LUT should be reversed to get 'digital_number'. The variable must exist in `self._map_var_lut`.

        Returns
        -------
        xarray.Dataset
            A dataset with one variable named 'digital_number'.

        Raises
        ------
        ValueErro
            If `var_name` does not have an associated LUT in `self._map_var_lut`.
        """
        # Check if the variable has an associated LUT, raise ValueError if not
        if var_name not in self._map_var_lut:
            raise ValueError(
                f"Unable to find lut for var '{var_name}'. Allowed: {list(self._map_var_lut.keys())}"
            )

        if self.sar_meta.product != "GRD":
            raise ValueError(
                "SAR product must be GRD. Not implemented for SLC")

        # Retrieve the variable data array and corresponding LUT
        da_var = self._dataset[var_name]
        lut = self._luts[self._map_var_lut[var_name]]

        # Interpolate the LUT to match the variable's coordinates
        lut = lut.interp(line=da_var.line, sample=da_var.sample)

        # Reverse the LUT application to compute the squared digital number
        dn2 = da_var * (lut**2)

        # Where the variable data array is NaN, set the squared digital number to 0
        dn2 = xr.where(np.isnan(da_var), 0, dn2)

        # Apply square root to get the digital number
        dn = np.sqrt(dn2)

        # Check and warn if the dtype of the original 'digital_number' is not preserved
        if (
            self._dataset.digital_number.dtype == np.complex_
            and dn.dtype != np.complex_
        ):
            warnings.warn(
                f"Unable to retrieve 'digital_number' as dtype '{self._dataset.digital_number.dtype}'. "
                f"Fallback to '{dn.dtype}'"
            )

        # Create a dataset with the computed digital number
        name = "digital_number"
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
        noise_lut = self._luts["noise_lut"]
        lut = self._get_lut(var_name)
        dataarr = noise_lut / lut**2
        name = "ne%sz" % var_name[0]
        astype = self._dtypes.get(name)
        if astype is not None:
            dataarr = dataarr.astype(astype)
        dataarr.attrs["history"] = merge_yaml(
            [lut.attrs["history"], noise_lut.attrs["history"]], section=name
        )
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
            vars = ["sigma0", "beta0", "gamma0"]
        for varname in vars:
            varname_raw = varname + "_raw"
            noise = "ne%sz" % varname[0]
            if varname_raw not in ds:
                continue
            if all(self.sar_meta.denoised.values()):
                # already denoised, just add an alias
                ds[varname] = ds[varname_raw]
            elif len(set(self.sar_meta.denoised.values())) != 1:
                # TODO: to be implemented
                raise NotImplementedError(
                    "semi denoised products not yet implemented")
            else:
                varname_raw_corrected = varname_raw + "__corrected"
                if (self.apply_recalibration) & (
                    varname_raw_corrected in self._dataset_recalibration.variables
                ):
                    denoised = (
                        self._dataset_recalibration[varname_raw_corrected] - ds[noise]
                    )
                    denoised.attrs["history"] = merge_yaml(
                        [ds[varname_raw].attrs["history"],
                            ds[noise].attrs["history"]],
                        section=varname,
                    )
                    denoised.attrs["comment_recalibration"] = (
                        "kersten recalibration applied"
                    )
                else:
                    denoised = ds[varname_raw] - ds[noise]
                    denoised.attrs["history"] = merge_yaml(
                        [ds[varname_raw].attrs["history"],
                            ds[noise].attrs["history"]],
                        section=varname,
                    )
                    denoised.attrs["comment_recalibration"] = (
                        "kersten recalibration not applied"
                    )

                if clip:
                    denoised = denoised.clip(min=0)
                    denoised.attrs["comment"] = "clipped, no values <0"
                else:
                    denoised.attrs["comment"] = "not clipped, some values can be <0"
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
        # azitime = self.sar_meta._burst_azitime()
        azitime = self._burst_azitime()
        iz = np.searchsorted(azitime.line, self._dataset.line)
        azitime = azitime.isel({"line": iz})
        azitime = azitime.assign_coords({"line": self._dataset.line})
        return azitime

    def get_sensor_velocity(self):
        """
        Interpolated sensor velocity
        Returns
        -------
        xarray.Dataset()
            containing a single variable velocity
        """

        azimuth_times = self.get_burst_azitime
        orbstatevect = self.sar_meta.orbit
        azi_times = orbstatevect["time"].values
        velos = np.array(
            [
                orbstatevect["velocity_x"] ** 2.0,
                orbstatevect["velocity_y"] ** 2.0,
                orbstatevect["velocity_z"] ** 2.0,
            ]
        )
        vels = np.sqrt(np.sum(velos, axis=0))
        interp_f = interp1d(azi_times.astype(float), vels)
        _vels = interp_f(azimuth_times.astype(float))
        res = xr.DataArray(_vels, dims=["line"], coords={
                           "line": self.dataset.line})
        return xr.Dataset({"velocity": res})

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
        ground_spacing = np.array(
            [
                self.sar_meta.image["azimuthPixelSpacing"],
                self.sar_meta.image["slantRangePixelSpacing"],
            ]
        )
        if self.sar_meta.product == "SLC":
            line_tmp = self._dataset["line"]
            # get the incidence at the middle of line dimension of the part of image selected
            inc = self._dataset["incidence"].isel(
                {
                    "line": int(len(line_tmp) / 2),
                }
            )
            range_ground_spacing_vect = ground_spacing[1] / \
                np.sin(np.radians(inc))
            range_ground_spacing_vect.attrs["history"] = ""

        else:  # GRD
            valuess = np.ones(
                (len(self._dataset["sample"]))) * ground_spacing[1]
            range_ground_spacing_vect = xr.DataArray(
                valuess, coords={"sample": self._dataset["sample"]}, dims=["sample"]
            )
        return xr.Dataset({"range_ground_spacing": range_ground_spacing_vect})

    def __repr__(self):
        if self.sliced:
            intro = "sliced"
        else:
            intro = "full coverage"
        return "<Sentinel1Dataset %s object>" % intro

    def _repr_mimebundle_(self, include=None, exclude=None):
        return repr_mimebundle(self, include=include, exclude=exclude)

    def _burst_azitime(self):
        """
        Get azimuth time at high resolution on the full image shape

        Returns
        -------
        np.ndarray
            the high resolution azimuth time vector interpolated at the midle of the subswath
        """
        line = np.arange(0, self.sar_meta.image["numberOfLines"])
        # line = np.arange(0,self.datatree.attrs['numberOfLines'])
        if self.sar_meta.product == "SLC" and "WV" not in self.sar_meta.swath:
            # if self.datatree.attrs['product'] == 'SLC' and 'WV' not in self.datatree.attrs['swath']:
            azi_time_int = self.sar_meta.image["azimuthTimeInterval"]
            # azi_time_int = self.datatree.attrs['azimuthTimeInterval']
            # turn this interval float/seconds into timedelta/picoseconds
            azi_time_int = np.timedelta64(int(azi_time_int * 1e12), "ps")
            ind, geoloc_azitime, geoloc_iburst, geoloc_line = self._get_indices_bursts()
            # compute the azimuth time by adding a step function (first term) and a growing term (second term)
            azitime = geoloc_azitime[ind] + (
                line - geoloc_line[ind]
            ) * azi_time_int.astype("<m8[ns]")
        else:  # GRD* cases
            # n_pixels = int((len(self.datatree['geolocation_annotation'].ds['sample']) - 1) / 2)
            # geoloc_azitime = self.datatree['geolocation_annotation'].ds['azimuth_time'].values[:, n_pixels]
            # geoloc_line = self.datatree['geolocation_annotation'].ds['line'].values
            n_pixels = int((len(self.sar_meta.geoloc["sample"]) - 1) / 2)
            geoloc_azitime = self.sar_meta.geoloc["azimuthTime"].values[:, n_pixels]
            geoloc_line = self.sar_meta.geoloc["line"].values
            finterp = interp1d(geoloc_line, geoloc_azitime.astype(float))
            azitime = finterp(line)
            azitime = azitime.astype("<M8[ns]")
        azitime = xr.DataArray(
            azitime,
            coords={"line": line},
            dims=["line"],
            attrs={
                "description": "azimuth times interpolated along line dimension at the middle of range dimension"
            },
        )

        return azitime

    @property
    def s1meta(self):
        logger.warning("Please use `sar_meta` to call the sar meta object")
        return self.sar_meta
