import xsar
import rasterio
import os
import logging

logging.basicConfig()
logging.getLogger('xsar').setLevel(logging.DEBUG)
logging.getLogger('xsar.utils').setLevel(logging.DEBUG)
logging.getLogger('xsar.xml_parser').setLevel(logging.DEBUG)
logging.captureWarnings(True)

logger = logging.getLogger('xsar_test')
logger.setLevel(logging.DEBUG)


logger.info('using %s as test_dir' % xsar.config['data_dir'])


meta = xsar.Sentinel1Meta(
    xsar.get_test_file('S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_Z010.SAFE'))
ds = xsar.open_dataset(meta, resolution={'atrack': 100, 'xtrack': 100}, resampling=rasterio.enums.Resampling.average)

ds.compute()

ds = xsar.open_dataset(meta).isel(pol=0,atrack=slice(0,100),xtrack=slice(0,100))
ds.compute()
