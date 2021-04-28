#!/usr/bin/env python
import unittest
import xsarlib as Sr

import dask.array as da
import numpy
import xarray as xr

from datetime import datetime
from shapely.geometry import Polygon

class ReadTestGRDH(unittest.TestCase):

    def setUp(self):
        chunks = {'pol': 1, 'xtrack': 500, 'atrack': 200}
        dir_safe = "/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/IW" \
                   "/S1A_IW_GRDH_1S/2017/250/S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_992F.SAFE"
        # dir_safe = "/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/IW/S1A_IW_GRDH_1S/2017/284/S1A_IW_GRDH_1SDV_20171011T173159_20171011T173228_018768_01FAD2_A8E2.SAFE"
        self.reader = Sr.SentinelReader(dir_safe, chunks)
        self.image = self.reader.sentinel_image_list[0].dataset

    def test_read_manifest_file(self):
        dims_values = self.image.coords.__getitem__('pol').data
        self.assertEqual(dims_values[0], 'VV')
        self.assertEqual(dims_values[1], 'VH')
        self.assertEqual(self.image.attrs['ipf_version'], 2.84)
        self.assertEqual(self.image.attrs['product_class'], 'S')
        self.assertEqual(self.image.attrs['product_type'], 'GRD')
        self.assertEqual(self.image.attrs['mission'], 'SENTINEL-1')
        self.assertEqual(self.image.attrs['satellite'], 'A')
        self.assertEqual(self.image.attrs['swath_type'], 'IW')
        self.assertEqual(self.image.attrs['start_date'], datetime(2017, 9, 7, 10, 30, 20, 936409))
        self.assertEqual(self.image.attrs['stop_date'], datetime(2017, 9, 7, 10, 30, 45, 935264))
        self.assertEqual(self.image.attrs['footprint'], Polygon([(-68.15836299999999, 19.215193), (-70.514343, 19.640442), (-70.221626, 21.147583), (-67.842209, 20.725643), (-68.15836299999999, 19.215193)]))

    def test_search_raster_and_xmlfile(self):
        self.assertEqual(self.reader._polarization[1].annotation_file, '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/IW/S1A_IW_GRDH_1S/2017/250/S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_992F.SAFE/annotation/s1a-iw-grd-vv-20170907t103020-20170907t103045-018268-01eb76-001.xml')
        self.assertEqual(self.reader._polarization[0].annotation_file, '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/IW/S1A_IW_GRDH_1S/2017/250/S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_992F.SAFE/annotation/s1a-iw-grd-vh-20170907t103020-20170907t103045-018268-01eb76-002.xml')
        self.assertEqual(self.reader._polarization[1].calibration_file, '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/IW/S1A_IW_GRDH_1S/2017/250/S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_992F.SAFE/annotation/calibration/calibration-s1a-iw-grd-vv-20170907t103020-20170907t103045-018268-01eb76-001.xml')
        self.assertEqual(self.reader._polarization[0].calibration_file, '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/IW/S1A_IW_GRDH_1S/2017/250/S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_992F.SAFE/annotation/calibration/calibration-s1a-iw-grd-vh-20170907t103020-20170907t103045-018268-01eb76-002.xml')
        self.assertEqual(self.reader._polarization[1].noise_calibration_file, '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/IW/S1A_IW_GRDH_1S/2017/250/S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_992F.SAFE/annotation/calibration/noise-s1a-iw-grd-vv-20170907t103020-20170907t103045-018268-01eb76-001.xml')
        self.assertEqual(self.reader._polarization[0].noise_calibration_file, '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/IW/S1A_IW_GRDH_1S/2017/250/S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_992F.SAFE/annotation/calibration/noise-s1a-iw-grd-vh-20170907t103020-20170907t103045-018268-01eb76-002.xml')

    def test_digitalNumber(self):
        self.assertEqual(type(self.image.digital_number), type(xr.DataArray()))
        self.assertEqual(self.image.digital_number.dims, ('pol', 'xtrack', 'atrack'))
        dims_pol_values = self.image.digital_number.coords.__getitem__('pol').data
        self.assertEqual(dims_pol_values[0], 'VV')
        self.assertEqual(dims_pol_values[1], 'VH')

        # TODO: test with cebere values

    def test_lon_lat(self):
        self.assertEqual(type(self.image.longitude), type(xr.DataArray()))
        self.assertEqual(self.image.longitude.dims, ('xtrack', 'atrack'))

        self.assertEqual(type(self.image.latitude), type(xr.DataArray()))
        self.assertEqual(self.image.latitude.dims, ('xtrack', 'atrack'))

        # TODO: test with cebere values

    def test_sigma0(self):
        self.assertEqual(type(self.image.sigma0), type(xr.DataArray()))

        # TODO: test with cebere values

    def test_gamma0(self):
        self.assertEqual(type(self.image.gamma0), type(xr.DataArray()))

        # TODO: test with cebere values

    def test_noise(self):
        self.assertTrue(True)

class ReadTestWV(unittest.TestCase):
    def setUp(self):
        dir_safe = "/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2020/001/S1A_WV_SLC__1SSV_20200101T115652_20200101T121907_030606_0381A9_63F1.SAFE"
        self.reader = Sr.SentinelReader(dir_safe)

    def test_read_manifest_file(self):
        dims_values = self.reader.coords.__getitem__('pol').data
        self.assertEqual(dims_values[0], 'VV')
        self.assertEqual(self.reader.attrs['product_class'], 'S')
        self.assertEqual(self.reader.attrs['product_type'], 'SLC')
        self.assertEqual(self.reader.attrs['mission'], 'SENTINEL-1')
        self.assertEqual(self.reader.attrs['satellite'], 'A')
        self.assertEqual(self.reader.attrs['instrument_mode'], 'WV')
        self.assertEqual(self.reader.attrs['instrument_swath'], 'WV1')
        self.assertEqual(self.reader.attrs['start_date_time'], datetime(2020, 1, 1, 11, 56, 52, 70953))
        self.assertEqual(self.reader.attrs['stop_date_time'], datetime(2020, 1, 1, 12, 19, 7, 897705))
        self.assertEqual(self.reader.attrs['coordinates'], Polygon(
            [(-63.44804, 113.217766), (-63.366947, 113.601585), (-63.531601, 113.774193), (-63.613148, 113.388412)]))

    def test_search_raster_and_xmlfile(self):
        self.assertEqual(self.reader._xml_files['VV'][0], '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2020/001/S1A_WV_SLC__1SSV_20200101T115652_20200101T121907_030606_0381A9_63F1.SAFE/annotation/s1a-wv1-slc-vv-20200101t115652-20200101t115655-030606-0381a9-001.xml')
        self.assertEqual(self.reader._xml_files['VV'][-1], '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2020/001/S1A_WV_SLC__1SSV_20200101T115652_20200101T121907_030606_0381A9_63F1.SAFE/annotation/s1a-wv2-slc-vv-20200101t121904-20200101t121907-030606-0381a9-092.xml')
        self.assertEqual(len(self.reader._xml_files['VV']), 92)

        self.assertEqual(self.reader._xml_files['VV_calibration'][0], '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2020/001/S1A_WV_SLC__1SSV_20200101T115652_20200101T121907_030606_0381A9_63F1.SAFE/annotation/calibration/calibration-s1a-wv1-slc-vv-20200101t115652-20200101t115655-030606-0381a9-001.xml')
        self.assertEqual(self.reader._xml_files['VV_calibration'][-1], '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2020/001/S1A_WV_SLC__1SSV_20200101T115652_20200101T121907_030606_0381A9_63F1.SAFE/annotation/calibration/calibration-s1a-wv2-slc-vv-20200101t121904-20200101t121907-030606-0381a9-092.xml')
        self.assertEqual(len(self.reader._xml_files['VV_calibration']), 92)

        self.assertEqual(self.reader._xml_files['VV_noise'][0], '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2020/001/S1A_WV_SLC__1SSV_20200101T115652_20200101T121907_030606_0381A9_63F1.SAFE/annotation/calibration/noise-s1a-wv1-slc-vv-20200101t115652-20200101t115655-030606-0381a9-001.xml')
        self.assertEqual(self.reader._xml_files['VV_noise'][-1], '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2020/001/S1A_WV_SLC__1SSV_20200101T115652_20200101T121907_030606_0381A9_63F1.SAFE/annotation/calibration/noise-s1a-wv2-slc-vv-20200101t121904-20200101t121907-030606-0381a9-092.xml')
        self.assertEqual(len(self.reader._xml_files['VV_noise']), 92)

        self.assertEqual(self.reader._raster_files['VV'][0], '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2020/001/S1A_WV_SLC__1SSV_20200101T115652_20200101T121907_030606_0381A9_63F1.SAFE/measurement/s1a-wv1-slc-vv-20200101t115652-20200101t115655-030606-0381a9-001.tiff')
        self.assertEqual(self.reader._raster_files['VV'][-1], '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2020/001/S1A_WV_SLC__1SSV_20200101T115652_20200101T121907_030606_0381A9_63F1.SAFE/measurement/s1a-wv2-slc-vv-20200101t121904-20200101t121907-030606-0381a9-092.tiff')
        self.assertEqual(len(self.reader._raster_files['VV']), 92)

    def test_digitalNumber(self):
        self.assertEqual(type(self.reader.data['VV']['digital_number'][0]), type(da.from_array([])))
        self.assertEqual(self.reader.data['VV']['digital_number'][0].shape, (4900, 5471))
        # TODO: maybe improve testing

    def test_sigma0(self):
        self.assertEqual(type(self.reader.data['VV']['sigma0'][0]), type(tuple()))
        self.assertEqual(type(self.reader.data['VV']['sigma0'][0][0]), type(da.from_array([])))
        self.assertEqual(self.reader.data['VV']['sigma0'][0][0].shape, (5, 138))
        self.assertEqual(type(self.reader.data['VV']['sigma0'][0][1]), type(da.from_array([])))
        self.assertEqual(self.reader.data['VV']['sigma0'][0][1].shape, (5, 138))
        self.assertEqual(type(self.reader.data['VV']['sigma0'][0][2]), type(da.from_array([])))
        self.assertEqual(self.reader.data['VV']['sigma0'][0][2].shape, (5, 138))

        # TODO: maybe improve testing

    def test_lon_lat(self):
        self.assertEqual(type(self.reader.data['VV']['lon_lat'][0]), type((da.from_array(([])), da.from_array([]))))
        self.assertEqual(self.reader.data['VV']['lon_lat'][0][0].shape, (4900, 5471))
        self.assertEqual(self.reader.data['VV']['lon_lat'][0][1].shape, (4900, 5471))

    def test_noise(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
