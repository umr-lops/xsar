# -*- coding: utf-8 -*-
"""
BIOMASS dataset class for xsar.

Loads BIOMASS data through BiomassReader (via BiomassMeta), applies
calibration and denoising, and provides a standard xarray.Dataset
following the xsar conventions.
"""

import logging
import warnings

import dask
import dask.array
import numpy as np
import rasterio
import rasterio.features
import xarray
import xarray as xr
from scipy.ndimage import zoom

from .base_dataset import BaseDataset
from .biomass_meta import BiomassMeta
from .utils import (
    BlockingActorProxy,
    merge_yaml,
    timing,
)

logger = logging.getLogger("xsar.biomass_dataset")
logger.addHandler(logging.NullHandler())

# we know tiff has no geotransform: ignore warning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

np.errstate(invalid="ignore")

# Mapping from dataset variable names to geolocation grid variable names
_MAPPING_DATASET_GEOLOC = {
    "latitude": "latitude",
    "longitude": "longitude",
    "incidence": "incidenceAngle",
    "elevation": "elevationAngle",
    "altitude": "height",
}


def _resample_lut(lut_data, target_shape, order=1):
    """
    Resample a 2D LUT array to target_shape using scipy.ndimage.zoom.

    Parameters
    ----------
    lut_data : np.ndarray
        2D array at reduced resolution.
    target_shape : tuple of int
        (n_lines, n_samples) of the target grid.
    order : int
        Interpolation order (1=bilinear, 3=bicubic).

    Returns
    -------
    np.ndarray
        Resampled 2D array.
    """
    if hasattr(lut_data, "values"):
        data = lut_data.values
    else:
        data = np.asarray(lut_data)

    if data.shape == target_shape:
        return data.astype(np.float32)

    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])

    has_nan = np.any(np.isnan(data))
    if has_nan:
        valid_mask = ~np.isnan(data)
        from scipy.ndimage import distance_transform_edt

        _, indices = distance_transform_edt(
            ~valid_mask, return_distances=True, return_indices=True
        )
        data_filled = data[tuple(indices)]
        resampled = zoom(data_filled, zoom_factors, order=order)
        mask_resampled = zoom(valid_mask.astype(float), zoom_factors, order=0)
        resampled[mask_resampled < 0.5] = np.nan
    else:
        resampled = zoom(data, zoom_factors, order=order)

    return resampled.astype(np.float32)


class BiomassDataset(BaseDataset):
    """
    Handle a BIOMASS product dataset.

    The main attribute is ``self.dataset`` (``xarray.Dataset`` with all
    variables parsed from product files).

    Parameters
    ----------
    dataset_id : str or BiomassMeta
        Path to BIOMASS product directory, or a BiomassMeta object.
    resolution : dict, number, str, or None, optional
        Resampling resolution. ``None`` for full resolution.
    resampling : rasterio.enums.Resampling, optional
        Resampling method. Default ``rms``.
    luts : bool, optional
        If True, include raw LUT variables in the dataset.
    chunks : dict, optional
        Dask chunks for ``{'line': ..., 'sample': ...}``.
    dtypes : dict or None, optional
        Override data types for variables.
    lazyloading : bool, optional
        If True (default), use lazy dask-based interpolation for
        high-resolution variables.
    """

    def __init__(
        self,
        dataset_id,
        resolution=None,
        resampling=rasterio.enums.Resampling.rms,
        luts=False,
        chunks=None,
        dtypes=None,
        lazyloading=True,
    ):
        if chunks is None:
            chunks = {"line": 5000, "sample": 5000}

        if dtypes is not None:
            self._dtypes.update(dtypes)

        self.sar_meta = None

        if not isinstance(dataset_id, BiomassMeta):
            self.sar_meta = BlockingActorProxy(BiomassMeta, dataset_id)
        else:
            self.sar_meta = BlockingActorProxy(
                BiomassMeta.from_dict, dataset_id.dict
            )
        del dataset_id

        # Load digital number
        self.resolution, DN_tmp = self.sar_meta.reader.load_digital_number(
            resolution=resolution, resampling=resampling, chunks=chunks
        )

        # Geolocation
        geoloc = self.sar_meta.geoloc
        geoloc.attrs["history"] = "BIOMASS LUT geometry"

        # Calibration LUTs
        ds_cal = self.sar_meta.get_calibration_luts
        ds_cal.attrs["history"] = "BIOMASS radiometry LUT"

        # Noise LUTs
        ds_noise = self.sar_meta.get_noise_luts
        ds_noise.attrs["history"] = "BIOMASS denoising LUT"

        # Build datatree
        self.datatree = xr.DataTree.from_dict(
            {
                "measurement": DN_tmp,
                "geolocation_annotation": geoloc,
                "orbit": self.sar_meta.orbit,
                "image": self.sar_meta.image,
                "calibration": ds_cal,
                "noise": ds_noise,
            }
        )

        self._dataset = self.datatree["measurement"].to_dataset()

        # Add metadata attributes
        for att in ["name", "short_name", "product", "safe", "swath", "multidataset"]:
            if att not in self.datatree.attrs:
                val = self.sar_meta.__getattr__(att)
                self.datatree.attrs[att] = val
                self._dataset.attrs[att] = val

        # Pixel spacing
        value_res_sample = self.sar_meta.image["groundRangePixelSpacing"]
        value_res_line = self.sar_meta.image["azimuthPixelSpacing"]
        refe_spacing = "slant"
        if resolution is not None:
            refe_spacing = "ground"
            if isinstance(resolution, str):
                value_res_sample = float(resolution.replace("m", ""))
                value_res_line = value_res_sample
            elif isinstance(resolution, dict):
                value_res_sample = (
                    self.sar_meta.image["groundRangePixelSpacing"] * resolution["sample"]
                )
                value_res_line = (
                    self.sar_meta.image["azimuthPixelSpacing"] * resolution["line"]
                )

        self._dataset["sampleSpacing"] = xarray.DataArray(
            value_res_sample, attrs={"units": "m", "referential": refe_spacing}
        )
        self._dataset["lineSpacing"] = xarray.DataArray(
            value_res_line, attrs={"units": "m"}
        )

        # Template DataArray for coordinate-based operations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._da_tmpl = xr.DataArray(
                dask.array.empty_like(
                    self._dataset.digital_number.isel(pol=0).drop("pol"),
                    dtype=np.int8,
                    name="empty_var_tmpl-%s" % dask.base.tokenize(self.sar_meta.name),
                ),
                dims=("line", "sample"),
                coords={
                    "line": self._dataset.digital_number.line,
                    "sample": self._dataset.digital_number.sample,
                },
            )

        self._dataset.attrs.update(self.sar_meta.to_dict("all"))
        self.datatree["measurement"] = self.datatree["measurement"].assign(self._dataset)
        self.datatree.attrs.update(self.sar_meta.to_dict("all"))

        # Load high-resolution variables
        self._add_high_resolution_variables(luts=luts, lazy_loading=lazyloading)

        # Apply calibration and denoising
        self._apply_biomass_calibration()

        self.datatree["measurement"] = self.datatree["measurement"].assign(self._dataset)
        self.datatree["measurement"].attrs = self.datatree.attrs

        self.sliced = False
        self.resampled = resolution is not None
        self._bbox_coords_ori = self._bbox_coords

    def load_ground_heading(self):
        return super().load_ground_heading(approx=False)

    def _add_high_resolution_variables(self, luts=False, lazy_loading=True):
        """
        Add high-resolution variables: lat/lon from GCPs, LUT variables via resampling,
        ground heading and land mask.

        Lat/lon are generated from TIFF GCPs using GCPTransformer (accurate).
        LUT variables (incidence, elevation, altitude) are resampled using scipy.ndimage.zoom.
        """
        ds_merge_list = [self._dataset]

        # Ground heading
        ds_merge_list.append(self.load_ground_heading())

        # Land mask
        self._rasterized_masks = self.load_rasterized_masks()
        ds_merge_list.append(self._rasterized_masks)

        attrs = self._dataset.attrs
        self._dataset = xr.merge(ds_merge_list)
        self._dataset.attrs = attrs

        # 1. Generate lat/lon from GCPs using actual pixel coordinates.
        # The dataset's line/sample coords are in the original full-resolution
        # pixel space (set by the Affine transform in load_digital_number).
        # We must use these instead of np.arange(shape) so the GCP transformer
        # maps to the correct geographic extent at any resolution.
        from rasterio.transform import GCPTransformer
        gcps, _ = self.sar_meta.reader._read_gcps()
        transformer = GCPTransformer(gcps)
        line_vals = self._dataset.line.values
        sample_vals = self._dataset.sample.values
        rows, cols = np.meshgrid(line_vals, sample_vals, indexing='ij')
        lon_flat, lat_flat = transformer.xy(rows.ravel(), cols.ravel())
        lat = np.array(lat_flat).reshape(len(line_vals), len(sample_vals))
        lon = np.array(lon_flat).reshape(len(line_vals), len(sample_vals))

        self._dataset["latitude"] = xr.DataArray(
            lat.astype(np.float64),
            dims=("line", "sample"),
            coords={
                "line": self._dataset.line,
                "sample": self._dataset.sample,
            },
            attrs={"units": "degrees_north", "long_name": "Latitude"},
        )
        self._dataset["longitude"] = xr.DataArray(
            lon.astype(np.float64),
            dims=("line", "sample"),
            coords={
                "line": self._dataset.line,
                "sample": self._dataset.sample,
            },
            attrs={"units": "degrees_east", "long_name": "Longitude"},
        )

        # 2. Resample LUT variables (incidence, elevation, altitude, height) using scipy.ndimage.zoom
        lut_vars_mapping = [
            ("incidence", "incidenceAngle"),
            ("elevation", "elevationAngle"),
            ("altitude", "height"),
            ("height", "height"),  # Add height explicitly as requested
        ]

        for varname, lut_varname in lut_vars_mapping:
            if lut_varname in self.sar_meta.geoloc:
                lut_data = self.sar_meta.geoloc[lut_varname]
                resampled = _resample_lut(lut_data, full_shape, order=1)
                base_attrs = lut_data.attrs if hasattr(lut_data, 'attrs') else {}
                var_attrs = dict(base_attrs)
                if varname == "height":
                    var_attrs["units"] = "m"
                    var_attrs["long_name"] = "Terrain height above ellipsoid"
                self._dataset[varname] = xr.DataArray(
                    resampled,
                    dims=("line", "sample"),
                    coords={
                        "line": self._dataset.line,
                        "sample": self._dataset.sample,
                    },
                    attrs=var_attrs,
                )

        # Recompute spatial attrs
        self.recompute_attrs()
        self.datatree["measurement"] = self.datatree["measurement"].assign(self._dataset)

    def _apply_biomass_calibration(self):
        """
        Apply BIOMASS-specific calibration using radiometry and denoising LUTs.

        BIOMASS calibration formula (from calibration_biomass.md):
            σ⁰ = (Amplitude_MDS)² × (σ⁰_LUT)²
            σ⁰_noise = Noise_LUT × (σ⁰_LUT)²
            σ⁰_denoised = max(σ⁰ - σ⁰_noise, 0)
        """
        cal_ds = self.sar_meta.get_calibration_luts
        noise_ds = self.sar_meta.get_noise_luts

        n_lines = self._dataset.sizes["line"]
        n_samples = self._dataset.sizes["sample"]
        target_shape = (n_lines, n_samples)
        pols = self._dataset.pol.values

        # Look for sigma0/gamma0 LUT variables in calibration dataset
        sigma0_lut_var = None
        gamma0_lut_var = None
        for vname in cal_ds.data_vars:
            vl = vname.lower()
            if "sigma" in vl:
                sigma0_lut_var = vname
            elif "gamma" in vl:
                gamma0_lut_var = vname

        # Store resampled LUTs for reuse in denoising
        sigma0_lut_da = None
        gamma0_lut_da = None

        # Apply sigma0 calibration: σ⁰ = DN² × LUT²
        if sigma0_lut_var is not None:
            lut_data = cal_ds[sigma0_lut_var].values
            lut_resampled = _resample_lut(lut_data, target_shape)
            sigma0_lut_da = xr.DataArray(
                lut_resampled,
                dims=("line", "sample"),
                coords={
                    "line": self._dataset.line,
                    "sample": self._dataset.sample,
                },
            )
            # CORRECT: σ⁰ = DN² × LUT²
            sigma0_raw = np.abs(self._dataset.digital_number) ** 2 * (sigma0_lut_da ** 2)
            sigma0_raw.name = "sigma0_raw"
            sigma0_raw.attrs = {"units": "linear", "history": "BIOMASS σ⁰ = DN² × LUT²"}
            self._dataset["sigma0_raw"] = sigma0_raw
            self._dataset["sigma0_lut"] = sigma0_lut_da
            self._dataset["sigma0_lut"].attrs = {
                "units": "1",
                "history": "BIOMASS sigma0 calibration LUT (resampled)",
            }

        # Apply gamma0 calibration: γ⁰ = DN² × LUT²
        if gamma0_lut_var is not None:
            lut_data = cal_ds[gamma0_lut_var].values
            lut_resampled = _resample_lut(lut_data, target_shape)
            gamma0_lut_da = xr.DataArray(
                lut_resampled,
                dims=("line", "sample"),
                coords={
                    "line": self._dataset.line,
                    "sample": self._dataset.sample,
                },
            )
            # CORRECT: γ⁰ = DN² × LUT²
            gamma0_raw = np.abs(self._dataset.digital_number) ** 2 * (gamma0_lut_da ** 2)
            gamma0_raw.name = "gamma0_raw"
            gamma0_raw.attrs = {"units": "linear", "history": "BIOMASS γ⁰ = DN² × LUT²"}
            self._dataset["gamma0_raw"] = gamma0_raw
            self._dataset["gamma0_lut"] = gamma0_lut_da
            self._dataset["gamma0_lut"].attrs = {
                "units": "1",
                "history": "BIOMASS gamma0 calibration LUT (resampled)",
            }

        # Load noise LUTs (per polarization)
        noise_luts = {}
        for pol in pols:
            noise_var = None
            for vname in noise_ds.data_vars:
                vl = vname.lower()
                if pol.lower() in vl or f"denoising{pol.lower()}" in vl:
                    noise_var = vname
                    break

            if noise_var is not None:
                noise_data = noise_ds[noise_var].values
                noise_resampled = _resample_lut(noise_data, target_shape)
                noise_luts[pol] = xr.DataArray(
                    noise_resampled,
                    dims=("line", "sample"),
                    coords={
                        "line": self._dataset.line,
                        "sample": self._dataset.sample,
                    },
                )
                # Store raw noise LUT for reference
                self._dataset[f"noise_lut_{pol}"] = noise_luts[pol]
                self._dataset[f"noise_lut_{pol}"].attrs = {
                    "units": "power (linear)",
                    "history": "BIOMASS denoising LUT (raw)",
                }

        # Compute denoised sigma0: σ⁰_denoised = max(σ⁰ - σ⁰_noise, 0)
        if "sigma0_raw" in self._dataset and sigma0_lut_da is not None:
            denoised_list = []
            nesz_list = []
            for i, pol in enumerate(pols):
                sigma0_pol = self._dataset["sigma0_raw"].isel(pol=i)
                if pol in noise_luts:
                    # Convert noise to σ⁰: σ⁰_noise = noise_lut × LUT²
                    noise_sigma0 = noise_luts[pol] * (sigma0_lut_da ** 2)
                    # Store NESZ (Noise Equivalent Sigma Zero)
                    nesz_list.append(noise_sigma0)
                    # Denoise and clip to 0
                    denoised_pol = np.maximum(sigma0_pol - noise_sigma0, 0)
                else:
                    denoised_pol = sigma0_pol
                    nesz_list.append(xr.zeros_like(sigma0_pol))
                denoised_list.append(denoised_pol)

            # Combine into 3D array
            sigma0_denoised = xr.concat(denoised_list, dim="pol").assign_coords(pol=pols)
            sigma0_denoised.attrs = {
                "units": "linear",
                "comment": "denoised and clipped to 0",
                "history": "BIOMASS σ⁰_denoised = max(σ⁰ - σ⁰_noise, 0)",
            }
            self._dataset["sigma0"] = sigma0_denoised

            # Store NESZ
            nesz = xr.concat(nesz_list, dim="pol").assign_coords(pol=pols)
            nesz.attrs = {
                "units": "linear",
                "comment": "Noise Equivalent Sigma Zero",
                "history": "BIOMASS NESZ = noise_lut × LUT²",
            }
            self._dataset["nesz"] = nesz

        # Compute denoised gamma0: γ⁰_denoised = max(γ⁰ - γ⁰_noise, 0)
        if "gamma0_raw" in self._dataset and gamma0_lut_da is not None:
            denoised_list = []
            negz_list = []
            for i, pol in enumerate(pols):
                gamma0_pol = self._dataset["gamma0_raw"].isel(pol=i)
                if pol in noise_luts:
                    # Convert noise to γ⁰: γ⁰_noise = noise_lut × LUT²
                    noise_gamma0 = noise_luts[pol] * (gamma0_lut_da ** 2)
                    negz_list.append(noise_gamma0)
                    # Denoise and clip to 0
                    denoised_pol = np.maximum(gamma0_pol - noise_gamma0, 0)
                else:
                    denoised_pol = gamma0_pol
                    negz_list.append(xr.zeros_like(gamma0_pol))
                denoised_list.append(denoised_pol)

            gamma0_denoised = xr.concat(denoised_list, dim="pol").assign_coords(pol=pols)
            gamma0_denoised.attrs = {
                "units": "linear",
                "comment": "denoised and clipped to 0",
                "history": "BIOMASS γ⁰_denoised = max(γ⁰ - γ⁰_noise, 0)",
            }
            self._dataset["gamma0"] = gamma0_denoised

            # Store NEGZ
            negz = xr.concat(negz_list, dim="pol").assign_coords(pol=pols)
            negz.attrs = {
                "units": "linear",
                "comment": "Noise Equivalent Gamma Zero",
                "history": "BIOMASS NEGZ = noise_lut × LUT²",
            }
            self._dataset["negz"] = negz

    @property
    def dataset(self):
        """
        ``xarray.Dataset`` representation of this BiomassDataset object.
        """
        res = self.datatree["measurement"].to_dataset()
        res.attrs = self.datatree.attrs
        return res

    @dataset.setter
    def dataset(self, ds):
        if self.sar_meta.name == ds.attrs.get("name"):
            if not self.sliced:
                self.sliced = any(
                    [
                        list(ds[d].values) != list(self._dataset[d].values)
                        for d in ["line", "sample"]
                    ]
                )
            self._dataset = ds
            self.recompute_attrs()
        else:
            raise ValueError("dataset must be same kind as original one.")

    @dataset.deleter
    def dataset(self):
        logger.debug("deleter dataset")

    def __repr__(self):
        if self.sliced:
            intro = "sliced"
        else:
            intro = "full coverage"
        return "<BiomassDataset %s object>" % intro
