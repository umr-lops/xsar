#!/usr/bin/env python

import logging
import argparse
import shutil
import os
import rasterio
import warnings
import xarray as xr
import numpy as np
import subprocess
import glob
from xsar.utils import compress_safe

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
logging.basicConfig()
logger = logging.getLogger(os.path.basename(__file__))

logging.getLogger('rasterio').setLevel(logging.CRITICAL)


def get_dir_size(path):
    """

    Parameters
    ----------
    path: str

    Returns
    -------
    float
        dir size in Mb

    """

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size / (1024 * 1024)


def generate_product_id(smooth, constant=None, initial_product_id='0000'):
    """

    Parameters
    ----------
    smooth: int
    constant: None or not None
    initial_product_id: str

    Returns
    -------
    str
        new 4 chars product id.
         - 1st char: 'Z' if smooth >= 1, or 'C' if constant is not Nonee
         - 2-4 chars : smooth size or 00 if constant, in decimal int


    """
    if constant is not None:
        res_id = 'C%0.3d' % constant
    else:
        res_id = 'Z%0.3d' % smooth

    return res_id


if __name__ == '__main__':
    # default smooth size
    smooth_size = {
        'IW': 10,
        'EW': 5,
        'WV': 10
    }

    parser = argparse.ArgumentParser(description='compress safe')
    parser.add_argument('safe_path')
    parser.add_argument('-d', action='store', help='out dir', default='./', required=False)
    parser.add_argument('-z', action='store',
                        help='compress format (see https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Compression)',
                        default='zstd', required=False)
    parser.add_argument('-c', action='store', help='fix constant value in band', required=False)
    parser.add_argument('-s', action='store', help='average dn by s*s box (in pixel). Use 0 to keep full resolution',
                        default='auto')
    args = parser.parse_args()

    safe_path = os.path.normpath(os.path.expanduser(args.safe_path))
    safe_name = os.path.basename(safe_path)
    safe_type = safe_name.split('_')[1]
    safe_product = os.path.splitext(safe_name)[0].split('_')[0]
    if args.s == 'auto':
        if safe_type in smooth_size:
            smooth = smooth_size[safe_type]
        else:
            raise NotImplementedError('no default smooth size for %s. Use explicit -s' % safe_type)
    else:
        smooth = int(args.s)

    # in "*_XXXX.SAFE", XXXX is the product id
    product_id_init = os.path.splitext(safe_name)[0].split('_')[-1]
    product_id_out = generate_product_id(smooth)

    safe_path_out = safe_name.replace(product_id_init, product_id_out)
    safe_path_out = os.path.join(args.d, safe_path_out)

    print("Compressing %s..." % os.path.basename(safe_path))
    safe_out = compress_safe(safe_path, safe_path_out, product=safe_product, smooth=smooth,
                             rasterio_kwargs={'compress': args.z})
    in_size = get_dir_size(safe_path)
    out_size = get_dir_size(safe_out)
    ratio = (out_size / in_size) * 100
    zip_file = '%s.zip' % os.path.basename(safe_out)
    zip_msg = subprocess.check_output('cd %s ; zip -r %s %s' % (args.d, zip_file, os.path.basename(safe_out)),
                                      shell=True)
    zip_path = os.path.join(args.d, zip_file)
    zip_size = os.path.getsize(zip_path) / (1024 ** 2)  # in Mb
    zip_ratio = (zip_size / out_size) * 100
    all_ratio = (zip_size / in_size) * 100
    print("%.0fM -> %.0fM ( %.1f%% ) (%s) -> %.0fM ( %.1f%% ) (zip)" % (
        in_size, out_size, ratio, args.z, zip_size, all_ratio))
    # clean temporary path
    subprocess.check_output('cd %s ; rm -fr %s' % (args.d, safe_out), shell=True)
    print(zip_path)
