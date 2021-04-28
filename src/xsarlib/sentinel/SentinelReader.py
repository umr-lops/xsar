"""
TODO: This docstring describe the SentinelReader module. It should be updated.
"""
import logging
import os

from lxml import objectify
from datetime import datetime

import numpy as np
import xarray as xr
import rasterio

from shapely.geometry import Polygon

from xsarlib.utils import timing, to_lon180, haversine, xpath_to_dict, dict_flatten, minigrid
from . import SentinelMetadata, SentinelImage

logger = logging.getLogger('xsar.SentinelReader')
logger.addHandler(logging.NullHandler())


class SentinelReader:
    """
    SentinelReader is a main class for read and parse sentinel SAFE data
    TODO: this docstring describe the SentinelReader class. it should be updated.
    """

    def __init__(self, dir_safe, chunks=None):
        """TODO: this docstring describe the SentinelReader constructor. it should be updated"""
        self._dir_safe = dir_safe

        if chunks is None:
            chunks = {'pol': 1, 'xtrack': 1000, 'atrack': 1000}
        self.chunks = chunks
        """TODO: docstring for chunks"""

        # xml namesspaces for xpath, find, etc...
        self._xml_namespaces = {
            'xfdu': "urn:ccsds:schema:xfdu:1",
            "s1sarl1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1",
            "s1sar": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar",
            "s1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
            "safe": "http://www.esa.int/safe/sentinel-1.0",
            "gml": "http://www.opengis.net/gml"}

        # manifest file, as a file path
        self.manifest_file = os.path.join(dir_safe, 'manifest.safe')

        # manifest, as an lxml objectify object
        self.manifest_root = None

        self.images_number = 0
        self.sentinel_image_list = []
        """Images list of SAFE"""
        self._pol = []
        self.safe_attribute = dict()
        self.__parseManifestFile__()

        self.rio = rasterio.open('SENTINEL1_DS:' + self._dir_safe + ':' + self.safe_attribute['swath_type'][0])

        self._polarization = []
        self.__find_xml_files_and_raster_files()
        self._parse_sentinel_images()

    @timing
    def _parse_sentinel_images(self):
        """
        _parse_sentinel_images is methode for complete the sentinel_image_list with a SentinelImage
        """

        if self.safe_attribute['swath_type'][0] == 'IW':
            self.sentinel_image_list.append(
                SentinelImage(self.rio, self.chunks, self._pol, self._polarization, self.safe_attribute))

    @timing
    def __find_xml_files_and_raster_files(self):

        safe_files = self.rio.files
        xml_ann_files = [file for file in safe_files if '.xml' in file]
        xml_cal_files = [file.replace('annotation/', 'annotation/calibration/calibration-') for file in xml_ann_files]
        xml_cal_noise_files = [file.replace('annotation/', 'annotation/calibration/noise-') for file in xml_ann_files]

        for xml_ann_file, xml_cal_file, xml_cal_noise_file in zip(xml_ann_files, xml_cal_files, xml_cal_noise_files):
            sigma0_lut, gamma0_lut = self._read_calibration_data_lut(xml_cal_file)
            # noise_lut = self.__read_noise_lut(xml_cal_noise_file[0])
            noise_lut = xr.DataArray()

            self._polarization.append(SentinelMetadata(annotation_file=xml_ann_file, calibration_file=xml_cal_file,
                                                       noise_calibration_file=xml_cal_noise_file, sigma0_lut=sigma0_lut,
                                                       gamma0_lut=gamma0_lut, noise_lut=noise_lut))

    @staticmethod
    @timing
    def _read_calibration_data_lut(calibration_file):
        xml_root = objectify.parse(calibration_file).getroot()
        number_of_vector = int(xml_root.find('./calibrationVectorList').attrib['count'])
        calibration_vector = xml_root.findall('./calibrationVectorList/calibrationVector')
        # transpose from line/pixel (ie atrack/xtrack to xtrack/atrack )
        atrack = np.array([int(c.find('./line').text.split()[0]) for c in calibration_vector])
        xtrack = np.array([int(r) for r in calibration_vector[0].find('./pixel').text.split()])

        sigma0_lut = np.array([[float(l) for l in v.find('./sigmaNought').text.split()] for v in calibration_vector]).T
        gamma0_lut = np.array([[float(l) for l in v.find('./gamma').text.split()] for v in calibration_vector]).T

        sigma0_lut_data = xr.DataArray(sigma0_lut, dims=('xtrack', 'atrack'),
                                       coords={'xtrack': xtrack, 'atrack': atrack})
        gamma0_lut_data = xr.DataArray(gamma0_lut, dims=('xtrack', 'atrack'),
                                       coords={'xtrack': xtrack, 'atrack': atrack})

        return sigma0_lut_data, gamma0_lut_data

    @timing
    def __read_noise_lut(self, calibration_file):
        xml_root = objectify.parse(calibration_file).getroot()

        if self.safe_attribute['ipf_version'][0] < 2.9:
            number_of_vector = int(xml_root.find('./noiseVectorList').attrib['count'])
            noise_vector = xml_root.findall('./noiseVectorList/noiseVector')
            # transpose from line/pixel (ie atrack/xtrack to xtrack/atrack )
            atrack = np.array([int(c.find('./line').text.split()[0]) for c in noise_vector])
            xtrack = np.array([int(r) for r in noise_vector[0].find('./pixel').text.split()])
            noise_lut = np.array([[float(l) for l in v.find('./noiseLut').text.split()] for v in noise_vector]).T
        else:
            number_of_vector = int(xml_root.find('./noiseRangeVectorList').attrib['count'])
            noise_vector = xml_root.findall('./noiseRangeVectorList/noiseRangeVector')
            noise_lut = np.array([[float(l) for l in v.find('./noiseRangeLut').text.split()] for v in noise_vector]).T
            xtrack = np.array([int(r) for r in noise_vector[0].find('./pixel').text.split()])

        noise_lut_data = xr.DataArray(noise_lut, dims=('xtrack', 'atrack'),
                                      coords={'xtrack': xtrack, 'atrack': atrack})

        return noise_lut_data

    def __parseManifestFile__(self):
        self.manifest_root = objectify.parse(self.manifest_file).getroot()
        self._xml_namespaces.update(self.manifest_root.nsmap)

        # dict key:xpath mapping to extract from self.manifest_root
        xpath_mapping = {
            'ipf_version': '//xmlData/safe:processing/safe:facility/safe:software/@version',
            'swath_type': '//s1sarl1:instrumentMode/s1sarl1:swath',
            'polarizations': '//s1sarl1:standAloneProductInformation/s1sarl1:transmitterReceiverPolarisation',
            'footprint': '//safe:frame/safe:footPrint/gml:coordinates',
            'product_info': {
                '.': '//s1sarl1:standAloneProductInformation',
                'product_class': 's1sarl1:productClass',
                'product_type': 's1sarl1:productType',
            },
            'platform': {
                '.': '//safe:platform',
                'mission': 'safe:familyName',
                'satellite': 'safe:number',

            },
            'dates': {
                '.': '//safe:acquisitionPeriod',
                'start_date': 'safe:startTime',
                'stop_date': 'safe:stopTime'
            }

        }
        xml_dict = xpath_to_dict(self.manifest_root, xpath_mapping, namespaces=self._xml_namespaces)

        # types conversion
        datetime_format = '%Y-%m-%dT%H:%M:%S.%f'
        xml_dict['dates'] = {k: [datetime.strptime(d[0], datetime_format)] for k, d in xml_dict['dates'].items()}
        xml_dict['ipf_version'] = [float(xml_dict['ipf_version'][0])]
        # footprint is "lat1,lon1 lat2,lon2 ...", and shapely need list of (lon,lat) tuples
        xml_dict['footprint'] = [Polygon(
            [(float(lon), float(lat)) for lat, lon in [latlon.split(",")
                                                       for latlon in xml_dict['footprint'][0].split(" ")]])]
        self.images_number = len(xml_dict['footprint'])

        # xml dict is a nested dict. a flat dict is easiest to handle
        self.safe_attribute = dict_flatten(xml_dict)

        self._pol = self.safe_attribute['polarizations']
