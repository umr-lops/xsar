import rasterio
from rasterio.control import GroundControlPoint
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon

from .utils import haversine, timing
from .base_meta import BaseMeta
import os
import numpy as np
import pandas as pd
import geopandas as gpd


class RcmMeta(BaseMeta):
    """
    Handle dataset metadata.
    A `xsar.RadarSat2Meta` object can be used with `xsar.open_dataset`,
    but it can be used as itself: it contains usefull attributes and methods.

    Parameters
    ----------
    name: str
        path or gdal identifier

    """

    dt = None

    @timing
    def __init__(self, name):
        super().__init__()

        from safe_rcm import api

        if ":" in name:
            self.dt = api.open_rcm(name.split(":")[1])
        else:
            self.dt = api.open_rcm(name)
        if not name.startswith("RCM_DS:"):
            name = "RCM_DS:%s:" % name
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
            self.product = os.path.basename(self.path).split("_")[9]
        except ValueError:
            self.product = "XXX"
        """Product type, like 'GRDH', 'SLC', etc .."""

        # define important xpath to easier access some dataset of the reader's datatree
        self._xpath = {
            "geolocationGrid": "imageReferenceAttributes/geographicInformation/geolocationGrid",
            "lut": "lookupTables/lookupTables",
            "noise_lut": "lookupTables/noiseLevels/referenceNoiseLevel",
            "incidence": "lookupTables/incidenceAngles",
            "attitude": "sourceAttributes/orbitAndAttitude/attitudeInformation",
            "orbit": "sourceAttributes/orbitAndAttitude/orbitInformation",
        }

        # replace `coefficients` dim by `pixel` in the datatree
        var_with_coeff = ["noise_lut", "lut", "incidence"]
        for var in var_with_coeff:
            self.dt[self._xpath[var]] = self.dt[self._xpath[var]].ds.rename(
                {"coefficients": "pixel"}
            )
        # build pixels coords
        self.assign_index(var_with_coeff)

        # flip line and samples when necessary to respect `xsar` convention
        self.samples_flipped = False
        self.lines_flipped = False
        self.flip_sample_da()
        self.flip_line_da()

        self._safe_files = None
        self.multidataset = False
        """True if multi dataset"""
        self.subdatasets = gpd.GeoDataFrame(geometry=[], index=[])
        """Subdatasets as GeodataFrame (empty if single dataset)"""
        self.geoloc = self.dt[self._xpath["geolocationGrid"]].to_dataset()

        self.orbit = self.dt[self._xpath["orbit"]].ds
        self.attitude = self.dt[self._xpath["attitude"]].ds
        self.lut = self.dt[self._xpath["lut"]].ds
        self.noise_lut = self.dt[self._xpath["noise_lut"]].ds
        self.incidence = self.dt[self._xpath["incidence"]].ds
        self.manifest_attrs = self._create_manifest_attrs()
        for name, feature in self.__class__._mask_features_raw.items():
            self.set_mask_feature(name, feature)

    def flip_sample_da(self):
        """
        When a product is flipped, flip back data arrays (from the reader datatree) sample dimensions to respect the xsar
        convention (increasing incidence values).
        Documentation reference : RCM Image Product Format Definition (4.2.1)
        """
        antenna_pointing = self.dt["sourceAttributes/radarParameters"].attrs[
            "antennaPointing"
        ]
        pass_direction = self.dt[
            "sourceAttributes/orbitAndAttitude/orbitInformation"
        ].attrs["passDirection"]
        flipped_cases = [("Left", "Ascending"), ("Right", "Descending")]
        samples_depending_ds = ["geolocationGrid"]
        if (antenna_pointing, pass_direction) in flipped_cases:
            for ds_name in samples_depending_ds:
                self.dt[self._xpath[ds_name]] = (
                    self.dt[self._xpath[ds_name]]
                    .ds.copy()
                    .isel(pixel=slice(None, None, -1))
                    .assign_coords(pixel=self.dt[self._xpath[ds_name]].ds.pixel)
                )
            self.samples_flipped = True
        return

    def flip_line_da(self):
        """
        Flip dataArrays (from the reader datatree) that depend on line dimension when a product is ascending, in order to
        respect the xsar convention (increasing time along line axis, whatever ascending or descending product).
        Documentation reference : RCM Image Product Format Definition (4.2.1)
        """
        pass_direction = self.dt[
            "sourceAttributes/orbitAndAttitude/orbitInformation"
        ].attrs["passDirection"]
        samples_depending_ds = ["geolocationGrid"]
        if pass_direction == "Ascending":
            for ds_name in samples_depending_ds:
                self.dt[self._xpath[ds_name]] = (
                    self.dt[self._xpath[ds_name]]
                    .ds.copy()
                    .isel(line=slice(None, None, -1))
                    .assign_coords(line=self.dt[self._xpath[ds_name]].ds.line)
                )
            self.lines_flipped = True
        return

    def assign_index(self, ds_list):
        """
        Generate a list of increasing indexes as a dataset coord. The change is made directly in the reader's datatree
        to all the datasets given in ds_list.
        The list of indexes is built thanks to a firstPixelValue, a step and a number of values.
        Documentation reference : RCM Image Product Format Definition (7.5.1)

        Parameters
        ----------
        ds_list : List[str]
            List of dataset that don't have the samples list built. The list can contain these values :
            ['lut', 'noise_lut', 'incidence'].

        Returns
        -------
        """

        def _assign_index_lut():
            xpath = self._xpath["lut"]
            lut = self.dt[xpath].ds.lookup_tables
            first_pix = lut.attrs["pixelFirstLutValue"]
            step = lut.attrs["stepSize"]
            values_nb = lut.attrs["numberOfValues"]
            indexes = [first_pix + step * i for i in range(0, values_nb)]
            # we want increasing indexes
            if step < 0:
                indexes = indexes[::-1]
            self.dt[xpath] = self.dt[xpath].ds.assign_coords(pixel=indexes)
            return

        def _assign_index_noise_lut():
            xpath = self._xpath["noise_lut"]
            noise_lut = self.dt[xpath].ds
            first_pix = noise_lut.pixelFirstNoiseValue
            step = noise_lut.stepSize
            values_nb = noise_lut.attrs["numberOfValues"]
            indexes = [first_pix + step * i for i in range(0, values_nb)]
            # we want increasing indexes
            if step < 0:
                indexes = indexes[::-1]
            self.dt[xpath] = self.dt[xpath].ds.assign_coords(pixel=indexes)
            return

        def _assign_index_incidence():
            xpath = self._xpath["incidence"]
            incidence = self.dt[xpath].ds
            first_pix = incidence.attrs["pixelFirstAnglesValue"]
            step = incidence.attrs["stepSize"]
            values_nb = incidence.attrs["numberOfValues"]
            indexes = [first_pix + step * i for i in range(0, values_nb)]
            # we want increasing indexes
            if step < 0:
                indexes = indexes[::-1]
            self.dt[xpath] = self.dt[xpath].ds.assign_coords(pixel=indexes)
            return

        if not all(item in ["lut", "noise_lut", "incidence"] for item in ds_list):
            raise ValueError(
                "Please enter accepted dataset names ('lut', 'noise_lut', 'incidence')"
            )
        if "lut" in ds_list:
            _assign_index_lut()
        if "noise_lut" in ds_list:
            _assign_index_noise_lut()
        if "incidence" in ds_list:
            _assign_index_incidence()
        return

    def _create_manifest_attrs(self):
        dic = dict()
        dic["swath_type"] = os.path.basename(self.path).split("_")[4]
        dic["polarizations"] = self.dt["sourceAttributes/radarParameters"][
            "pole"
        ].values
        dic["product_type"] = self.product
        dic["satellite"] = "RCM"
        dic["start_date"] = self.start_date
        dic["stop_date"] = self.stop_date
        # compute attributes (footprint, coverage, pixel_size)
        footprint_dict = {}
        for ll in ["longitude", "latitude"]:
            footprint_dict[ll] = [
                self.geoloc[ll].isel(line=a, pixel=x).values
                for a, x in [(0, 0), (0, -1), (-1, -1), (-1, 0)]
            ]
        corners = list(zip(footprint_dict["longitude"], footprint_dict["latitude"]))
        p = Polygon(corners)
        self.geoloc.attrs["footprint"] = p
        dic["footprints"] = p
        # compute acquisition size/resolution in meters
        # first vector is on sample
        acq_sample_meters, _ = haversine(*corners[0], *corners[1])
        # second vector is on line
        acq_line_meters, _ = haversine(*corners[1], *corners[2])
        dic["coverage"] = "%dkm * %dkm (line * sample )" % (
            acq_line_meters / 1000,
            acq_sample_meters / 1000,
        )

        def _to_rio_gcp(pt_geoloc):
            # convert a point from self._geoloc grid to rasterio GroundControlPoint
            return GroundControlPoint(
                x=pt_geoloc.longitude.item(),
                y=pt_geoloc.latitude.item(),
                z=pt_geoloc.height.item(),
                col=pt_geoloc.line.item(),
                row=pt_geoloc.pixel.item(),
            )

        gcps = [
            _to_rio_gcp(self.geoloc.sel(line=line, pixel=sample))
            for line in self.geoloc.line
            for sample in self.geoloc.pixel
        ]
        # approx transform, from all gcps (inaccurate)
        dic["approx_transform"] = rasterio.transform.from_gcps(gcps)
        return dic

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
        xsar.BaseDataset.coords2ll
        xsar.BaseDataset.ll2coords
        xsar.BaseMeta.coords2ll
        xsar.BaseMeta.ll2coords

        """
        return self.manifest_attrs["approx_transform"]

    @property
    def footprint(self):
        """footprint, as a shapely polygon or multi polygon"""
        return self.geoloc.attrs["footprint"]

    @property
    def pixel_line_m(self):
        """pixel line spacing, in meters (at sensor level)"""
        if self.multidataset:
            res = None  # not defined for multidataset
        else:
            res = self.dt["imageReferenceAttributes/rasterAttributes"][
                "sampledLineSpacing"
            ].values
        return res

    @property
    def pixel_sample_m(self):
        """pixel sample spacing, in meters (at sensor level)"""
        if self.multidataset:
            res = None  # not defined for multidataset
        else:
            res = self.dt["imageReferenceAttributes/rasterAttributes"][
                "sampledPixelSpacing"
            ].values
        return res

    @property
    def get_azitime(self):
        """
        Get time at low resolution

        Returns
        -------
        array[datetime64[ns]]
            times
        """
        return self.orbit.timeStamp.values

    @property
    def pols(self):
        """polarisations strings, separated by spaces"""
        return " ".join(self.manifest_attrs["polarizations"])

    def __reduce__(self):
        # make self serializable with pickle
        # https://docs.python.org/3/library/pickle.html#object.__reduce__

        return self.__class__, (self.name,), self.dict

    def __repr__(self):
        if self.multidataset:
            meta_type = "multi (%d)" % len(self.subdatasets)
        else:
            meta_type = "single"
        return "<RcmMeta %s object>" % meta_type

    def _get_time_range(self):
        if self.multidataset:
            time_range = [
                self.manifest_attrs["start_date"],
                self.manifest_attrs["stop_date"],
            ]
        else:
            time_range = self.orbit.timeStamp
        return pd.Interval(
            left=pd.Timestamp(time_range.values[0]),
            right=pd.Timestamp(time_range.values[-1]),
            closed="both",
        )

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

        idx_sample = np.array(geoloc.pixel)
        idx_line = np.array(geoloc.line)

        for ll in ["longitude", "latitude"]:
            resdict[ll] = RectBivariateSpline(
                idx_line, idx_sample, np.asarray(geoloc[ll]), kx=1, ky=1
            )

        return resdict

    def to_dict(self, keys="minimal"):

        info_keys = {
            "minimal": [
                # 'platform',
                "swath",
                "product",
                "pols",
            ]
        }
        info_keys["all"] = info_keys["minimal"] + [
            "name",
            "start_date",
            "stop_date",
            "footprint",
            "coverage",
            "pixel_line_m",
            "pixel_sample_m",
            "approx_transform",
            # 'orbit_pass',
            # 'platform_heading'
        ]

        if isinstance(keys, str):
            keys = info_keys[keys]

        res_dict = {}
        for k in keys:
            if hasattr(self, k):
                res_dict[k] = getattr(self, k)
            elif k in self.manifest_attrs.keys():
                res_dict[k] = self.manifest_attrs[k]
            else:
                raise KeyError('Unable to find key/attr "%s" in RcmMeta' % k)
        return res_dict
