# -*- coding: utf-8 -*-
"""
BIOMASS metadata class for xsar.

Wraps the BiomassReader (from xarray-biomass package) and implements
the BaseMeta interface required by xsar.
"""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import GCPTransformer
from shapely.geometry import Polygon

from .base_meta import BaseMeta
from .utils import haversine, timing

logger = logging.getLogger("xsar.biomass_meta")
logger.addHandler(logging.NullHandler())


class BiomassMeta(BaseMeta):
    """
    Handle BIOMASS dataset metadata.

    A ``BiomassMeta`` object can be used with ``xsar.open_dataset``.
    It wraps a ``BiomassReader`` from the ``xarray-biomass`` package and
    adapts its interface to the xsar ``BaseMeta`` contract.

    Parameters
    ----------
    name : str
        Path to the BIOMASS product directory.
    """

    @timing
    def __init__(self, name):
        super().__init__()

        from biomass.reader import BiomassReader

        # Handle BIOMASS_DS: prefix (analogous to SENTINEL1_DS:)
        if isinstance(name, str) and name.startswith("BIOMASS_DS:"):
            parts = name.split(":")
            path = parts[1]
        else:
            path = name

        self.reader = BiomassReader(path)
        self.name = str(path)
        self.short_name = self.reader.short_name
        self.path = str(self.reader.path)
        self.safe = os.path.basename(self.path)
        self.product = self.reader.product
        self.dt = self.reader.datatree
        self.manifest_attrs = self.reader.manifest_attrs
        self.manifest = str(self.reader._mph_file) if self.reader._mph_file else None
        self.multidataset = False
        self.subdatasets = gpd.GeoDataFrame(geometry=[], index=[])
        self.dsid = ""
        self.platform = "BIOMASS"
        self._time_range = None
        self._geoloc = None
        self._gcps = None
        self._gcp_crs = None
        self._gcp_transformer = None

        for name_feat, feature in self.__class__._mask_features_raw.items():
            self.set_mask_feature(name_feat, feature)

    def _get_time_range(self):
        time_range = self.reader.time_range
        start = time_range[0] if time_range[0] else "1970-01-01T00:00:00"
        stop = time_range[1] if time_range[1] else start
        return pd.Interval(
            left=pd.Timestamp(start),
            right=pd.Timestamp(stop),
            closed="both",
        )

    @property
    def footprint(self):
        """Footprint as a shapely Polygon from geolocation grid corners."""
        geoloc = self.geoloc
        footprint_dict = {}
        for ll in ["longitude", "latitude"]:
            footprint_dict[ll] = [
                geoloc[ll].isel(line=a, sample=x).values
                for a, x in [(0, 0), (0, -1), (-1, -1), (-1, 0)]
            ]
        corners = list(
            zip(footprint_dict["longitude"], footprint_dict["latitude"])
        )
        return Polygon(corners)

    @property
    def geometry(self):
        """Alias for footprint."""
        return self.footprint

    @property
    def geoloc(self):
        """
        Geolocation grid with longitude, latitude, incidenceAngle,
        elevationAngle, height on (line, sample) coordinates.

        The grid is from the LUT NetCDF geometry group. Line/sample
        coordinates are scaled to map to the full-resolution image
        pixel indices.
        """
        if self._geoloc is not None:
            return self._geoloc

        self._geoloc = self.reader.geoloc.copy()
        self._geoloc.attrs = {}

        # Scale line/sample coordinates to full-resolution pixel indices
        image_info = self.reader.image
        n_lines = int(image_info["numberOfLines"].item())
        n_samples = int(image_info["numberOfSamples"].item())
        geoloc_n_lines = self._geoloc.sizes["line"]
        geoloc_n_samples = self._geoloc.sizes["sample"]

        # Map geoloc grid indices to full-resolution pixel indices
        line_coords = np.linspace(0, n_lines - 1, geoloc_n_lines)
        sample_coords = np.linspace(0, n_samples - 1, geoloc_n_samples)
        self._geoloc = self._geoloc.assign_coords(
            line=line_coords,
            sample=sample_coords,
        )

        # Compute footprint attribute
        footprint_dict = {}
        for ll in ["longitude", "latitude"]:
            footprint_dict[ll] = [
                self._geoloc[ll].isel(line=a, sample=x).values
                for a, x in [(0, 0), (0, -1), (-1, -1), (-1, 0)]
            ]
        corners = list(
            zip(footprint_dict["longitude"], footprint_dict["latitude"])
        )
        p = Polygon(corners)
        self._geoloc.attrs["footprint"] = p

        # Compute coverage
        acq_sample_meters, _ = haversine(*corners[0], *corners[1])
        acq_line_meters, _ = haversine(*corners[1], *corners[2])
        self._geoloc.attrs["coverage"] = "%dkm * %dkm (line * sample )" % (
            acq_line_meters / 1000,
            acq_sample_meters / 1000,
        )

        # Compute approx_transform from real TIFF GCPs
        gcps, _ = self.reader._read_gcps()
        self._geoloc.attrs["approx_transform"] = rasterio.transform.from_gcps(gcps)

        return self._geoloc

    @property
    def coverage(self):
        """Coverage string like '251km * 170km (sample * line )'."""
        return self.geoloc.attrs.get("coverage", "unknown")

    @property
    def gcp_transformer(self):
        """
        GCPTransformer for accurate pixel-to-geographic coordinate conversion.

        Uses the GCPs from the TIFF file for thin plate spline interpolation,
        which is more accurate than interpolating the reduced-resolution LUT
        lat/lon grids.
        """
        if self._gcp_transformer is None:
            gcps, gcp_crs = self.reader._read_gcps()
            if gcps:
                self._gcp_transformer = GCPTransformer(gcps)
                self._gcps = gcps
                self._gcp_crs = gcp_crs
            else:
                raise ValueError("No GCPs found in the measurement TIFF file.")
        return self._gcp_transformer

    @property
    def _dict_coords2ll(self):
        """
        Dict with 'longitude' and 'latitude' keys mapping to
        callables that transform pixel coordinates to geographic coordinates.

        Uses GCPTransformer for accurate transformation instead of
        interpolating reduced-resolution LUT lat/lon grids.

        The returned objects implement both __call__(line, sample) for grid
        interpolation (to_grid=True) and ev(line, sample) for point evaluation.
        """
        transformer = self.gcp_transformer
        cross_antimeridian = self.cross_antimeridian

        class GCPInterpolator:
            """Wrapper providing RectBivariateSpline-like interface for GCPTransformer."""

            def __init__(self, transformer, coord_type, cross_antimeridian):
                self.transformer = transformer
                self.coord_type = coord_type  # 'longitude' or 'latitude'
                self.cross_antimeridian = cross_antimeridian

            def __call__(self, line, sample):
                """
                Interpolate on a grid (for to_grid=True in coords2ll).

                Parameters
                ----------
                line : array-like
                    1D array of line coordinates.
                sample : array-like
                    1D array of sample coordinates.

                Returns
                -------
                result : ndarray
                    2D array with shape (len(line), len(sample)).
                """
                line = np.atleast_1d(line)
                sample = np.atleast_1d(sample)
                # Create meshgrid
                lines2D, samples2D = np.meshgrid(line, sample, indexing='ij')
                # GCPTransformer.xy() expects (row, col) and returns (x, y) = (lon, lat)
                lon, lat = self.transformer.xy(lines2D.ravel(), samples2D.ravel())
                if self.coord_type == 'longitude':
                    result = np.array(lon).reshape(lines2D.shape)
                    if self.cross_antimeridian:
                        result = result % 360
                else:
                    result = np.array(lat).reshape(lines2D.shape)
                return result

            def ev(self, line, sample):
                """
                Evaluate at specific points (for to_grid=False in coords2ll).

                Parameters
                ----------
                line : array-like or scalar
                    Line coordinates.
                sample : array-like or scalar
                    Sample coordinates.

                Returns
                -------
                result : ndarray or scalar
                    Coordinate values at the specified points.
                """
                line = np.atleast_1d(line)
                sample = np.atleast_1d(sample)
                # GCPTransformer.xy() expects (row, col) and returns (x, y) = (lon, lat)
                lon, lat = self.transformer.xy(line.ravel(), sample.ravel())
                if self.coord_type == 'longitude':
                    result = np.array(lon).reshape(line.shape)
                    if self.cross_antimeridian:
                        result = result % 360
                else:
                    result = np.array(lat).reshape(line.shape)
                # Return scalar if input was scalar
                if result.size == 1:
                    return result.item()
                return result

        return {
            "longitude": GCPInterpolator(transformer, 'longitude', cross_antimeridian),
            "latitude": GCPInterpolator(transformer, 'latitude', cross_antimeridian),
        }

    @property
    def approx_transform(self):
        """
        Affine transform from geolocation grid.

        This is an approximate transform. Use coords2ll / ll2coords for
        accurate coordinate conversions.
        """
        return self.geoloc.attrs["approx_transform"]

    def ll2coords(self, *args):
        """
        Get (lines, samples) from (lon, lat) using GCPTransformer.

        Overrides BaseMeta.ll2coords because the affine approx_transform
        is too inaccurate for BIOMASS geometry.

        Uses GCPTransformer.rowcol() which is the exact inverse of
        GCPTransformer.xy() used in coords2ll.

        Parameters
        ----------
        *args : lon, lat or shapely geometry
            lon and lat might be iterables or scalars

        Returns
        -------
        tuple of np.array or tuple of float
            (lines, samples), or a shapely object if input was shapely
        """
        import shapely.geometry.base

        # Handle shapely geometries
        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self._ll2coords_shapely(args[0])

        lon, lat = args

        # Handle scalar vs array input
        scalar = not hasattr(lon, '__iter__')
        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)

        # Use GCPTransformer.rowcol() - inverse of xy()
        # rowcol(x, y) returns (row, col) = (line, sample)
        transformer = self.gcp_transformer
        lines, samples = transformer.rowcol(lon, lat)

        lines = np.asarray(lines)
        samples = np.asarray(samples)

        if scalar:
            return float(lines.flat[0]), float(samples.flat[0])

        return lines, samples

    @property
    def swath(self):
        """Swath type (always 'SM' for BIOMASS stripmap)."""
        return self.manifest_attrs.get("swath_type", "SM")

    @property
    def orbit_pass(self):
        """Orbit pass direction (Ascending/Descending)."""
        return self.manifest_attrs.get("orbit_direction", "Unknown")

    @property
    def platform_heading(self):
        """
        Platform heading relative to north (degrees).

        Read from the annotation XML via the reader. The annotation XML
        contains the accurate platformHeading value from acquisition
        information.
        """
        # Primary source: annotation XML via manifest_attrs
        if "platformHeading" in self.manifest_attrs:
            return self.manifest_attrs["platformHeading"]
        # Fallback: orbit dataset attrs
        try:
            return self.orbit.attrs.get("platform_heading", 0.0)
        except Exception:
            return 0.0

    @property
    def orbit(self):
        """Orbit data as xarray.Dataset."""
        return self.dt["orbit"].to_dataset()

    @property
    def image(self):
        """Image information dataset."""
        return self.dt["image"].to_dataset()

    @property
    def pixel_line_m(self):
        """Pixel line spacing in meters."""
        return self.image["azimuthPixelSpacing"]

    @property
    def pixel_sample_m(self):
        """Pixel sample spacing in meters."""
        return self.image["groundRangePixelSpacing"]

    @property
    def pols(self):
        """Polarizations as a space-separated string."""
        return " ".join(self.manifest_attrs["polarizations"])

    @property
    def denoised(self):
        """Dict with pol as key and bool as value."""
        return self.reader.denoised

    @property
    def get_calibration_luts(self):
        """Calibration LUTs dataset."""
        return self.dt["calibration_luts"].to_dataset()

    @property
    def get_noise_luts(self):
        """Noise/denoising LUTs dataset."""
        return self.dt["noise_luts"].to_dataset()

    def to_dict(self, keys="minimal"):
        info_keys = {
            "minimal": ["platform", "swath", "product", "pols"],
        }
        info_keys["all"] = info_keys["minimal"] + [
            "name",
            "start_date",
            "stop_date",
            "footprint",
            "coverage",
            "orbit_pass",
            "platform_heading",
        ]

        if isinstance(keys, str):
            keys = info_keys.get(keys, info_keys["minimal"])

        res_dict = {}
        for k in keys:
            if hasattr(self, k):
                res_dict[k] = getattr(self, k)
            elif k in self.manifest_attrs:
                res_dict[k] = self.manifest_attrs[k]
        return res_dict

    def __repr__(self):
        return "<BiomassMeta %s object>" % self.product

    def __reduce__(self):
        return self.__class__, (self.name,), self.dict
