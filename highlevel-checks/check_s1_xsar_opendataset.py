import xsar
import rasterio
import os
import logging
import dill
import pickle


logging.basicConfig()
logging.getLogger('xsar').setLevel(logging.DEBUG)
logging.getLogger('xsar.utils').setLevel(logging.DEBUG)
logging.getLogger('xsar.xml_parser').setLevel(logging.DEBUG)
logging.captureWarnings(True)

logger = logging.getLogger('xsar_test')
logger.setLevel(logging.DEBUG)

# logger.info('using %s as test_dir' % xsar.config['data_dir'])


meta = xsar.Sentinel1Meta(
    xsar.get_test_file('S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_Z010.SAFE'))


def test_open_dataset():
    try:
        ds = xsar.open_dataset(meta, resolution={'sample': 100, 'line': 100}, resampling=rasterio.enums.Resampling.average)
        ds.compute()
        ds = xsar.open_dataset(meta).isel(pol=0, sample=slice(0, 100), line=slice(0, 100))
        ds.compute()
        assert True
    except:
        assert False


def test_serializable_s1_meta():
    s1meta = dill.loads(dill.dumps(meta))
    assert isinstance(s1meta.coords2ll(100, 100), tuple)
