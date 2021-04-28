import xsar
import os
filename = xsar.get_test_file('S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_Z200.SAFE')
sar_ds = xsar.open_dataset(filename)
sar_ds

