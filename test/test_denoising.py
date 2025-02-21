import pdb

from xsar.sentinel1_dataset import add_denoised
import xsar
import logging
logging.basicConfig()
logging.getLogger('xsar').setLevel(logging.DEBUG)
logging.getLogger('xsar.utils').setLevel(logging.DEBUG)
logging.getLogger('xsar.xml_parser').setLevel(logging.DEBUG)
logging.captureWarnings(True)

logger = logging.getLogger('xsar_test')
logger.setLevel(logging.DEBUG)
def test_denoising():
    meta = xsar.Sentinel1Meta(
        xsar.get_test_file('S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_Z010.SAFE'))
    reader = xsar.Sentinel1Dataset(meta)
    ds = reader.datatree['measurement'].dataset #(meta)#.isel(pol=0, sample=slice(0, 100), line=slice(0, 100))
    ds = ds[['longitude','latitude','incidence','sigma0_raw','nesz']] # removed calibrated / denoised variables for the test
    print('ds input',ds)
    # pdb.set_trace()
    ds = add_denoised(ds,sar_meta_denoised={'VV':False},dataset_recalibration=reader._dataset_recalibration)
    print('ds output',ds)
    return True

if __name__ == '__main__':
    test_denoising()
