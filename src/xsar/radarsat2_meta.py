import cartopy.feature
# from .raster_readers import available_rasters
from .utils import to_lon180, haversine, timing, class_or_instancemethod
from . import raster_readers
from xradarsat2 import rs2_reader
from xradarsat2.radarSat2_xarray_reader import xpath_get
import os
import geopandas as gpd
import xmltodict
import numpy as np


class RadarSat2Meta:
    """
        Handle dataset metadata.
        A `xsar.RadarSat2Meta` object can be used with `xsar.open_dataset`,
        but it can be used as itself: it contains usefull attributes and methods.

        Parameters
        ----------
        name: str
            path or gdal identifier like `'SENTINEL1_DS:%s:WV_001' % path`

        """
    # default mask feature (see self.set_mask_feature and cls.set_mask_feature)
    _mask_features_raw = {
        'land': cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
    }

    rasters = raster_readers.available_rasters.iloc[0:0].copy()

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
            print("path: %s" % self.path)
            self.product = "XXX"
        """Product type, like 'GRDH', 'SLC', etc .."""
        print(
            f"safe:{self.safe} \n name:{self.name} \n name_parts:{name_parts} \n path:{self.path} \n short name:{self.short_name} \n"
            f"product:{self.product} \n")

        # self.manifest = os.path.join(self.path, 'manifest.safe')
        self.manifest_attrs = self._create_manifest_attrs()
        self._safe_files = None
        self.multidataset = False
        """True if multi dataset"""
        self.subdatasets = gpd.GeoDataFrame(geometry=[], index=[])
        """Subdatasets as GeodataFrame (empty if single dataset)"""

    def _create_manifest_attrs(self):
        dic = dict()
        dic["swath_type"] = os.path.basename(self.path).split('_')[4]
        dic["polarizations"] = self.dt["radarParameters"]["pole"].values
        dic["product_type"] = self.product
        dic['satellite'] = self.dt.attrs['satellite']
        dic['start_date'] = self.dt.attrs['rawDataStartTime']
        return dic

    @property
    def cross_antemeridian(self):
        """True if footprint cross antemeridian"""
        return ((np.max(self.rs2.dt['geolocationGrid']['longitude']) - np.min(self.rs2.dt['geolocationGrid']['longitude'])) > 180).item()

    @property
    def _bursts(self):
        if self.xml_parser.get_var(self.files['annotation'].iloc[0], 'annotation.number_of_bursts') > 0:
            bursts = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'bursts')
            bursts.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'bursts',
                                                                       describe=True)
            return bursts
        else:
            bursts = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'bursts_grd')
            bursts.attrs['history'] = self.xml_parser.get_compound_var(self.files['annotation'].iloc[0], 'bursts_grd',
                                                                       describe=True)
            return bursts



"""if __name__ == "__main__":
    RadarSat2Meta("/home/datawork-cersat-public/cache/project/sarwing/data/RS2/L1/VV_VH/2020/234/"
                  "RS2_OK123649_PK1086794_DK1033082_SCWA_20200821_133300_VV_VH_SGF")"""
