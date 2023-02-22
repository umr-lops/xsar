import copy

import pandas as pd
import rasterio
from rasterio.control import GroundControlPoint
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon

from .utils import haversine, timing
from xradarsat2 import rs2_reader
import os
import geopandas as gpd
import numpy as np
from .base_meta import BaseMeta


class RadarSat2Meta(BaseMeta):
    """
        Handle dataset metadata.
        A `xsar.RadarSat2Meta` object can be used with `xsar.open_dataset`,
        but it can be used as itself: it contains usefull attributes and methods.

        Parameters
        ----------
        name: str
            path or gdal identifier like `'SENTINEL1_DS:%s:WV_001' % path`

    """
    # class attributes are needed to fetch instance attribute (ie self.name) with dask actors
    # ref http://distributed.dask.org/en/stable/actors.html#access-attributes
    # FIXME: not needed if @property, so it might be a good thing to have getter for those attributes
    multidataset = None
    name = None
    short_name = None
    path = None
    product = None
    manifest = None
    subdatasets = None
    dsid = None
    manifest_attrs = None
    dt = None
    safe = None

    @timing
    def __init__(self, name):
        self.dt = rs2_reader(name)
        if not name.startswith('RADARSAT2_DS:'):
            name = 'RADARSAT2_DS:%s:' % name
        self.name = name
        """Gdal dataset name"""
        name_parts = self.name.split(':')
        if len(name_parts) > 3:
            # windows might have semicolon in path ('c:\...')
            name_parts[1] = ':'.join(name_parts[1:-1])
            del name_parts[2:-1]
        name_parts[1] = os.path.basename(name_parts[1])
        self.short_name = ':'.join(name_parts)
        """Like name, but without path"""
        self.path = ':'.join(self.name.split(':')[1:-1])
        """Dataset path"""
        self.safe = os.path.basename(self.path)
        """Safe file name"""
        # there is no information on resolution 'F' 'H' or 'M' in the manifest, so we have to extract it from filename
        try:
            self.product = os.path.basename(self.path).split('_')[9]
        except:
            self.product = "XXX"
        """Product type, like 'GRDH', 'SLC', etc .."""

        # self.manifest = os.path.join(self.path, 'manifest.safe')

        self._safe_files = None
        self.multidataset = False
        """True if multi dataset"""
        self.subdatasets = gpd.GeoDataFrame(geometry=[], index=[])
        """Subdatasets as GeodataFrame (empty if single dataset)"""
        self.geoloc = self.dt['geolocationGrid'].to_dataset()

        self.orbit_and_attitude = self.dt['orbitAndAttitude'].ds
        self.doppler_centroid = self.dt['imageGenerationParameters']['doppler']['dopplerCentroid'].ds
        self.doppler_rate_values = self.dt['imageGenerationParameters']['doppler']['dopplerRateValues'].ds
        self.chirp = self.dt['imageGenerationParameters']['chirp'].ds
        self.radar_parameters = self.dt['radarParameters'].ds
        self.lut = self.dt['lut'].ds
        self.manifest_attrs = self._create_manifest_attrs()
        for name, feature in self.__class__._mask_features_raw.items():
            self.set_mask_feature(name, feature)
        #self.rasters = BaseMeta.__class__.rasters.copy()
        #self._mask_features_raw = BaseMeta.__class__._mask_features_raw
        #self._mask_features = BaseMeta.__class__._mask_features
        #self._mask_intersecting_geometries = BaseMeta.__class__._mask_intersecting_geometries
        #self._mask_geometry = BaseMeta.__class__._mask_geometry
        #self._geoloc = super()._geoloc

    def _create_manifest_attrs(self):
        dic = dict()
        dic["swath_type"] = os.path.basename(self.path).split('_')[4]
        dic["polarizations"] = self.dt["radarParameters"]["pole"].values
        dic["product_type"] = self.product
        dic['satellite'] = self.dt.attrs['satellite']
        dic['start_date'] = self.start_date
        dic['stop_date'] = self.stop_date
        # compute attributes (footprint, coverage, pixel_size)
        footprint_dict = {}
        for ll in ['longitude', 'latitude']:
            footprint_dict[ll] = [
                self.geoloc[ll].isel(line=a, pixel=x).values for a, x in [(0, 0), (0, -1), (-1, -1), (-1, 0)]
            ]
        corners = list(zip(footprint_dict['longitude'], footprint_dict['latitude']))
        p = Polygon(corners)
        self.geoloc.attrs['footprint'] = p
        dic["footprints"] = p
        # compute acquisition size/resolution in meters
        # first vector is on sample
        acq_sample_meters, _ = haversine(*corners[0], *corners[1])
        # second vector is on line
        acq_line_meters, _ = haversine(*corners[1], *corners[2])
        dic['coverage'] = "%dkm * %dkm (line * sample )" % (
            acq_line_meters / 1000, acq_sample_meters / 1000)

        def _to_rio_gcp(pt_geoloc):
            # convert a point from self._geoloc grid to rasterio GroundControlPoint
            return GroundControlPoint(
                x=pt_geoloc.longitude.item(),
                y=pt_geoloc.latitude.item(),
                z=pt_geoloc.height.item(),
                col=pt_geoloc.line.item(),
                row=pt_geoloc.pixel.item()
            )

        gcps = [
            _to_rio_gcp(self.geoloc.sel(line=line, pixel=sample))
            for line in self.geoloc.line for sample in self.geoloc.pixel
        ]
        # approx transform, from all gcps (inaccurate)
        dic['approx_transform'] = rasterio.transform.from_gcps(gcps)
        return dic

    @property
    def approx_transform(self):
        """
        Affine transfom from geoloc.

        This is an inaccurate transform, with errors up to 600 meters.
        But it's fast, and may fit some needs, because the error is stable localy.
        See `xsar.Sentinel1Meta.coords2ll` `xsar.RdarSat2Dataset.ll2coords` for accurate methods.

        Examples
        --------
            get `longitude` and `latitude` from tuple `(line, sample)`:

            >>> longitude, latitude = self.approx_transform * (line, sample)

            get `line` and `sample` from tuple `(longitude, latitude)`

            >>> line, sample = ~self.approx_transform * (longitude, latitude)

        See Also
        --------
        xsar.BaseDataset.coords2ll
        xsar.BaseDataset.ll2coords`

        """
        return self.manifest_attrs['approx_transform']

    @property
    def cross_antemeridian(self):
        """True if footprint cross antemeridian"""
        return ((np.max(self.geoloc['longitude']) - np.min(
            self.geoloc['longitude'])) > 180).item()

    def to_dict(self, keys='minimal'):

        info_keys = {
            'minimal': [
                #'platform',
                'swath', 'product', 'pols']
        }
        info_keys['all'] = info_keys['minimal'] + ['name', 'start_date', 'stop_date',
                                                   'footprint',
                                                   'coverage',
                                                   'pixel_line_m', 'pixel_sample_m',
                                                   'approx_transform',

                                                   #'orbit_pass',
                                                   #'platform_heading'
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
                raise KeyError('Unable to find key/attr "%s" in RadarSat2Meta' % k)
        return res_dict

    @property
    def swath(self):
        """string like 'EW', 'IW', 'WV', etc ..."""
        return self.manifest_attrs['swath_type']

    @property
    def pols(self):
        """polarisations strings, separated by spaces """
        return " ".join(self.manifest_attrs['polarizations'])

    @property
    def footprint(self):
        """footprint, as a shapely polygon or multi polygon"""
        return self.geoloc.attrs['footprint']

    @property
    def _dict_coords2ll(self):
        """
        dict with keys ['longitude', 'latitude'] with interpolation function (RectBivariateSpline) as values.

        Examples:
        ---------
            get longitude at line=100 and sample=200:
            ```
            >>> self._dict_coords2ll['longitude'].ev(100,200)
            array(-66.43947434)
            ```
        Notes:
        ------
            if self.cross_antemeridian is True, 'longitude' will be in range [0, 360]
        """
        resdict = {}
        geoloc = self.geoloc
        if self.cross_antemeridian:
            geoloc['longitude'] = geoloc['longitude'] % 360

        idx_sample = np.array(geoloc.pixel)
        idx_line = np.array(geoloc.line)

        for ll in ['longitude', 'latitude']:
            resdict[ll] = RectBivariateSpline(idx_line, idx_sample, np.asarray(geoloc[ll]), kx=1, ky=1)

        return resdict

    @property
    def pixel_line_m(self):
        """pixel line spacing, in meters (at sensor level)"""
        if self.multidataset:
            res = None  # not defined for multidataset
        else:
            res = self.geoloc.line.attrs['rasterAttributes_sampledLineSpacing_value']
        return res

    @property
    def pixel_sample_m(self):
        """pixel sample spacing, in meters (at sensor level)"""
        if self.multidataset:
            res = None  # not defined for multidataset
        else:
            res = self.geoloc.pixel.attrs['rasterAttributes_sampledPixelSpacing_value']
        return res

    def _get_time_range(self):
        if self.multidataset:
            time_range = [self.manifest_attrs['start_date'], self.manifest_attrs['stop_date']]
        else:
            time_range = self.orbit_and_attitude.timeStamp
        return pd.Interval(left=pd.Timestamp(time_range.values[0]), right=pd.Timestamp(time_range.values[-1]), closed='both')

    @property
    def get_azitime(self):
        """
        Get time at low resolution

        Returns
        -------
        array[datetime64[ns]]
            times
        """
        return self.orbit_and_attitude.timeStamp.values

    def __reduce__(self):
        # make self serializable with pickle
        # https://docs.python.org/3/library/pickle.html#object.__reduce__

        return self.__class__, (self.name,), self.dict

    @property
    def dict(self):
        # return a minimal dictionary that can be used with Sentinel1Meta.from_dict() or pickle (see __reduce__)
        # to reconstruct another instance of self
        #
        minidict = {
            'name': self.name,
            '_mask_features_raw': self._mask_features_raw,
            '_mask_features': {},
            '_mask_intersecting_geometries': {},
            '_mask_geometry': {},
            'rasters': self.rasters
        }
        for name in minidict['_mask_features_raw'].keys():
            minidict['_mask_intersecting_geometries'][name] = None
            minidict['_mask_geometry'][name] = None
            minidict['_mask_features'][name] = None
        return minidict

    @classmethod
    def from_dict(cls, minidict):
        # like copy constructor, but take a dict from Sentinel1Meta.dict
        # https://github.com/umr-lops/xsar/issues/23
        for name in minidict['_mask_features_raw'].keys():
            assert minidict['_mask_geometry'][name] is None
            assert minidict['_mask_features'][name] is None
        minidict = copy.copy(minidict)
        new = cls(minidict['name'])
        new.__dict__.update(minidict)
        return new



