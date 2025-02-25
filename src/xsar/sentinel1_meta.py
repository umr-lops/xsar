# -*- coding: utf-8 -*-
import logging
import numpy as np
import xarray
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.control import GroundControlPoint
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .base_meta import BaseMeta
from .utils import haversine, timing
import os
from .ipython_backends import repr_mimebundle


logger = logging.getLogger("xsar.sentinel1_meta")
logger.addHandler(logging.NullHandler())


class Sentinel1Meta(BaseMeta):
    """
    Handle dataset metadata.
    A `xsar.Sentinel1Meta` object can be used with `xsar.open_dataset`,
    but it can be used as itself: it contains usefull attributes and methods.

    Parameters
    ----------
    name: str
        path or gdal identifier like `'SENTINEL1_DS:%s:WV_001' % path`

    """

    # class attributes are needed to fetch instance attribute (ie self.name) with dask actors
    # ref http://distributed.dask.org/en/stable/actors.html#access-attributes
    # FIXME: not needed if @property, so it might be a good thing to have getter for those attributes
    xsd_definitions = None

    @timing
    def __init__(self, name):
        super().__init__()

        try:
            from safe_s1.metadata import Sentinel1Reader
        except ImportError:
            from safe_s1.reader import Sentinel1Reader
        self.reader = Sentinel1Reader(name)

        if not name.startswith("SENTINEL1_DS:"):
            name = name.rstrip("/")  # remove trailing space
            name = "SENTINEL1_DS:%s:" % name
        else:
            name = name.replace("/:", ":")
        self.name = name
        """Gdal dataset name"""
        name_parts = self.name.split(":")
        if len(name_parts) > 3:
            # windows might have semicolon in path ('c:\...')
            name_parts[1] = ":".join(name_parts[1:-1])
            del name_parts[2:-1]
        name_parts[1] = os.path.basename(name_parts[1])
        self.short_name = ":".join(name_parts)
        """Like name, but without path"""
        self.path = ":".join(self.name.split(":")[1:-1])
        """Dataset path"""
        self.safe = os.path.basename(self.path)
        """Safe file name"""
        # there is no information on resolution 'F' 'H' or 'M' in the manifest, so we have to extract it from filename
        try:
            self.product = os.path.basename(self.path).split("_")[2]
        except ValueError:
            print("path: %s" % self.path)
            self.product = "XXX"
        """Product type, like 'GRDH', 'SLC', etc .."""

        self.dt = self.reader.datatree

        self.manifest_attrs = self.reader.manifest_attrs

        for attr in ['aux_cal', 'aux_pp1', 'aux_ins']:
            if attr not in self.manifest_attrs:
                self.manifest_attrs[attr] = None
            else:
                self.manifest_attrs[attr] = os.path.basename(
                    self.manifest_attrs[attr])

        self.multidataset = False
        """True if multi dataset"""
        self.subdatasets = gpd.GeoDataFrame(geometry=[], index=[])
        """Subdatasets as GeodataFrame (empty if single dataset)"""
        datasets_names = self.reader.datasets_names
        self.xsd_definitions = self.reader.get_annotation_definitions()
        if self.name.endswith(":") and len(datasets_names) == 1:
            self.name = datasets_names[0]
        self.dsid = self.name.split(":")[-1]
        """Dataset identifier (like 'WV_001', 'IW1', 'IW'), or empty string for multidataset"""
        # submeta is a list of submeta objects if multidataset and TOPS
        # this list will remain empty for _WV__SLC because it will be time-consuming to process them
        self._submeta = []
        if self.short_name.endswith(":"):
            self.short_name = self.short_name + self.dsid
        if self.reader.files.empty:
            try:
                self.subdatasets = gpd.GeoDataFrame(
                    geometry=self.manifest_attrs["footprints"], index=datasets_names
                )
            except ValueError:
                # not as many footprints than subdatasets count. (probably TOPS product)
                self._submeta = [Sentinel1Meta(subds)
                                 for subds in datasets_names]
                sub_footprints = [
                    submeta.footprint for submeta in self._submeta]
                self.subdatasets = gpd.GeoDataFrame(
                    geometry=sub_footprints, index=datasets_names
                )
            self.multidataset = True

        self.platform = (
            self.manifest_attrs["mission"] + self.manifest_attrs["satellite"]
        )
        """Mission platform"""
        self._time_range = None
        for name, feature in self.__class__._mask_features_raw.items():
            self.set_mask_feature(name, feature)
        """pandas dataframe for rasters (see `xsar.Sentinel1Meta.set_raster`)"""

    def __del__(self):
        logger.debug("__del__")

    def have_child(self, name):
        """
        Check if dataset `name` belong to this Sentinel1Meta object.

        Parameters
        ----------
        name: str
            dataset name

        Returns
        -------
        bool
        """
        return name == self.name or name in self.subdatasets.index

    def _get_time_range(self):
        if self.multidataset:
            time_range = [
                self.manifest_attrs["start_date"],
                self.manifest_attrs["stop_date"],
            ]
        else:
            time_range = self.reader.time_range
        return pd.Interval(
            left=pd.Timestamp(time_range[0]),
            right=pd.Timestamp(time_range[-1]),
            closed="both",
        )

    def to_dict(self, keys="minimal"):

        info_keys = {"minimal": [
            "ipf_version", "platform", "swath", "product", "pols"]}
        info_keys["all"] = info_keys["minimal"] + [
            "name",
            "start_date",
            "stop_date",
            "footprint",
            "coverage",
            "orbit_pass",
            "platform_heading",
            "icid",
            "aux_cal",
            "aux_pp1",
            "aux_ins",

        ]  # 'pixel_line_m', 'pixel_sample_m',

        if isinstance(keys, str):
            keys = info_keys[keys]

        res_dict = {}
        for k in keys:
            if hasattr(self, k):
                res_dict[k] = getattr(self, k)
            elif k in self.manifest_attrs.keys():
                res_dict[k] = self.manifest_attrs[k]
            else:
                raise KeyError(
                    'Unable to find key/attr "%s" in Sentinel1Meta' % k)

        return res_dict

    def annotation_angle(self, line, sample, angle):
        """Interpolate angle with RectBivariateSpline"""
        lut = angle.reshape(line.size, sample.size)
        lut_f = RectBivariateSpline(line, sample, lut, kx=1, ky=1)
        return lut_f

    @property
    def orbit_pass(self):
        """
        Orbit pass, i.e 'Ascending' or 'Descending'
        """

        if self.multidataset:
            return None  # not defined for multidataset

        return self.orbit.attrs["orbit_pass"]

    @property
    def platform_heading(self):
        """
        Platform heading, relative to north
        """

        if self.multidataset:
            return None  # not defined for multidataset

        return self.orbit.attrs["platform_heading"]

    @property
    def rio(self):
        raise DeprecationWarning(
            "Sentinel1Meta.rio is deprecated. "
            'Use `rasterio.open` on files in `Sentinel1Meta..files["measurement"] instead`'
        )

    @property
    def footprint(self):
        """footprint, as a shapely polygon or multi polygon"""
        if self.multidataset:
            return unary_union(self._footprints)
        return self.geoloc.attrs["footprint"]

    @property
    def geometry(self):
        """alias for footprint"""
        return self.footprint

    @property
    def geoloc(self):
        """
        xarray.Dataset with `['longitude', 'latitude', 'altitude', 'azimuth_time', 'slant_range_time','incidence','elevation' ]` variables
        and `['line', 'sample']` coordinates, at the geolocation grid
        """
        if self.multidataset:
            raise TypeError("geolocation_grid not available for multidataset")
        if self._geoloc is None:
            self._geoloc = self.dt["geolocationGrid"].to_dataset()
            self._geoloc.attrs = {}
            # compute attributes (footprint, coverage, pixel_size)
            footprint_dict = {}
            for ll in ["longitude", "latitude"]:
                footprint_dict[ll] = [
                    self._geoloc[ll].isel(line=a, sample=x).values
                    for a, x in [(0, 0), (0, -1), (-1, -1), (-1, 0)]
                ]
            corners = list(
                zip(footprint_dict["longitude"], footprint_dict["latitude"]))
            p = Polygon(corners)
            self._geoloc.attrs["footprint"] = p

            # compute acquisition size/resolution in meters
            # first vector is on sample
            acq_sample_meters, _ = haversine(*corners[0], *corners[1])
            # second vector is on line
            acq_line_meters, _ = haversine(*corners[1], *corners[2])
            self._geoloc.attrs["coverage"] = "%dkm * %dkm (line * sample )" % (
                acq_line_meters / 1000,
                acq_sample_meters / 1000,
            )

            # compute self._geoloc.attrs['approx_transform'], from gcps
            # we need to convert self._geoloc to  a list of GroundControlPoint
            def _to_rio_gcp(pt_geoloc):
                # convert a point from self._geoloc grid to rasterio GroundControlPoint
                return GroundControlPoint(
                    x=pt_geoloc.longitude.item(),
                    y=pt_geoloc.latitude.item(),
                    z=pt_geoloc.height.item(),
                    col=pt_geoloc.line.item(),
                    row=pt_geoloc.sample.item(),
                )

            gcps = [
                _to_rio_gcp(self._geoloc.sel(line=line, sample=sample))
                for line in self._geoloc.line
                for sample in self._geoloc.sample
            ]
            # approx transform, from all gcps (inaccurate)
            self._geoloc.attrs["approx_transform"] = rasterio.transform.from_gcps(
                gcps)
            for vv in self._geoloc:
                if vv in self.xsd_definitions:
                    self._geoloc[vv].attrs["definition"] = str(
                        self.xsd_definitions[vv])

        return self._geoloc

    @property
    def _footprints(self):
        """footprints as list. should len 1 for single meta, or len(self.subdatasets) for multi meta"""
        return self.manifest_attrs["footprints"]

    @property
    def coverage(self):
        """coverage, as a string like '251km * 170km (sample * line )'"""
        if self.multidataset:
            return None  # not defined for multidataset
        return self.geoloc.attrs["coverage"]

    @property
    def pixel_line_m(self):
        """pixel line spacing, in meters (at sensor level)"""
        if self.multidataset:
            res = None  # not defined for multidataset
        else:
            res = self.image["azimuthPixelSpacing"]
        return res

    @property
    def pixel_sample_m(self):
        """pixel sample spacing, in meters (at sensor level)"""
        if self.multidataset:
            res = None  # not defined for multidataset
        else:
            res = self.image["groundRangePixelSpacing"]
        return res

    @property
    def denoised(self):
        """dict with pol as key, and bool as values (True is DN is predenoised at L1 level)"""
        return self.reader.denoised

    @property
    def ipf_version(self):
        """ipf version"""
        return self.manifest_attrs["ipf_version"]

    @property
    def pols(self):
        """polarisations strings, separated by spaces"""
        return " ".join(self.manifest_attrs["polarizations"])

    @property
    def orbit(self):
        """
        orbit, as a geopandas.GeoDataFrame, with columns:
          - 'velocity' : shapely.geometry.Point with velocity in x, y, z direction
          - 'geometry' : shapely.geometry.Point with position in x, y, z direction

        crs is set to 'geocentric'

        attrs keys:
          - 'orbit_pass': 'Ascending' or 'Descending'
          - 'platform_heading': in degrees, relative to north

        Notes
        -----
        orbit is longer than the SAFE, because it belongs to all datatakes, not only this slice

        """
        return self.dt["orbit"].to_dataset()

    @property
    def image(self) -> xarray.Dataset:
        return self.dt["image"].to_dataset()

    @property
    def azimuth_fmrate(self):
        """
        xarray.Dataset
            Frequency Modulation rate annotations such as t0 (azimuth time reference) and polynomial coefficients: Azimuth FM rate = c0 + c1(tSR - t0) + c2(tSR - t0)^2
        """
        return self.dt["azimuth_fmrate"].to_dataset()

    @property
    def _dict_coords2ll(self):
        """
        dict with keys ['longitude', 'latitude'] with interpolation function (RectBivariateSpline) as values.

        Examples:
        ---------
            get longitude at line=100 and sample=200:
            ```
            >>> self._dict_coords2ll["longitude"].ev(100, 200)
            array(-66.43947434)
            ```
        Notes:
        ------
            if self.cross_antemeridian is True, 'longitude' will be in range [0, 360]
        """
        resdict = {}
        geoloc = self.geoloc
        if self.cross_antemeridian:
            geoloc["longitude"] = geoloc["longitude"] % 360

        idx_sample = np.array(geoloc.sample)
        idx_line = np.array(geoloc.line)

        for ll in ["longitude", "latitude"]:
            resdict[ll] = RectBivariateSpline(
                idx_line, idx_sample, np.asarray(geoloc[ll]), kx=1, ky=1
            )

        return resdict

    @property
    def _bursts(self):
        return self.dt["bursts"].to_dataset()

    @property
    def approx_transform(self):
        """
        Affine transfom from geoloc.

        This is an inaccurate transform, with errors up to 600 meters.
        But it's fast, and may fit some needs, because the error is stable localy.
        See `xsar.BaseMeta.coords2ll` `xsar.BaseMeta.ll2coords` for accurate methods.

        Examples
        --------
            get `longitude` and `latitude` from tuple `(line, sample)`:

            >>> longitude, latitude = self.approx_transform * (line, sample)

            get `line` and `sample` from tuple `(longitude, latitude)`

            >>> line, sample = ~self.approx_transform * (longitude, latitude)

        See Also
        --------
        xsar.BaseMeta.coords2ll
        xsar.BaseMeta.ll2coords
        xsar.BaseMeta.coords2ll
        xsar.BaseMeta.ll2coords

        """
        return self.geoloc.attrs["approx_transform"]

    def __repr__(self):
        if self.multidataset:
            meta_type = "multi (%d)" % len(self.subdatasets)
        else:
            meta_type = "single"
        return "<Sentinel1Meta %s object>" % meta_type

    def _repr_mimebundle_(self, include=None, exclude=None):
        return repr_mimebundle(self, include=include, exclude=exclude)

    def __reduce__(self):
        # make self serializable with pickle
        # https://docs.python.org/3/library/pickle.html#object.__reduce__

        return self.__class__, (self.name,), self.dict

    @property
    def _doppler_estimate(self):
        """
        xarray.Dataset
            with Doppler Centroid Estimates from annotations such as geo_polynom,data_polynom or frequency
        """
        return self.dt["doppler_estimate"].to_dataset()

    @property
    def get_calibration_luts(self):
        """
        get original (ie not interpolation) xr.Dataset sigma0 and gamma0 Look Up Tables to apply calibration

        """
        return self.dt["calibration_luts"].to_dataset()

    @property
    def get_noise_azi_raw(self):
        return self.dt["noise_azimuth_raw"].to_dataset()

    @property
    def get_noise_range_raw(self):
        return self.dt["noise_range_raw"].to_dataset()

    @property
    def get_antenna_pattern(self):
        return self.dt["antenna_pattern"].to_dataset()

    @property
    def get_swath_merging(self):
        return self.dt["swath_merging"].to_dataset()
