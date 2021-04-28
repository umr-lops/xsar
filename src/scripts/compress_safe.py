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

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
logging.basicConfig()
logger = logging.getLogger(os.path.basename(__file__))


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


def compress_safe(safe_path, out_dir_prefix, constant=None, smooth=0, rasterio_kwargs={'compress': None}):
    """
    
    Parameters
    ----------
    safe_path: str
        input SAFE path
    out_dir_prefix: str
        output directory prefix
    rasterio_kwargs: dict
        passed to rasterio.open
    constant:
        all value will be set to constant

    Returns
    -------
    str
        wrotten output path

    """

    file_name = os.path.basename(safe_path)
    # in "*_XXXX.SAFE", XXXX is the product id
    product_id_init = os.path.splitext(file_name)[0].split('_')[-1]
    compress = rasterio_kwargs['compress']
    product_id_out = generate_product_id(smooth, constant=constant, initial_product_id=product_id_init)

    file_out = file_name.replace(product_id_init, product_id_out)
    file_out = os.path.join(out_dir_prefix, file_out)
    file_out_tmp = file_out + '.tmp'
    if os.path.exists(file_out):
        raise FileExistsError("%s already exists" % file_out)
    try:
        shutil.rmtree(file_out_tmp)
    except:
        pass
    os.mkdir(file_out_tmp)

    shutil.copytree(safe_path + "/annotation", file_out_tmp + "/annotation")
    shutil.copyfile(safe_path + "/manifest.safe", file_out_tmp + "/manifest.safe")

    os.mkdir(file_out_tmp + "/measurement")
    for tiff_file in glob.glob(os.path.join(safe_path, 'measurement', '*.tiff')):
        src = rasterio.open(tiff_file)
        open_kwargs = {
            'driver': 'GTiff',
            'height': src.shape[0],
            'width': src.shape[1],
            'count': src.count,
            'dtype': src.dtypes[0],
            'crs': src.crs
        }
        open_kwargs.update(rasterio_kwargs)
        if smooth > 1:
            reduced = xr.DataArray(
                src.read(
                    1, out_shape=(src.height // smooth, src.width // smooth),
                    resampling=rasterio.enums.Resampling.average))
            mean = reduced.mean().item()
            if not isinstance(mean,complex) and mean < 1:
                raise RuntimeError('rasterio returned empty band. Try to use smallest smooth size')
            reduced = reduced.assign_coords(
                dim_0=reduced.dim_0 * smooth + smooth / 2,
                dim_1=reduced.dim_1 * smooth + smooth / 2)
            band = reduced.interp(
                dim_0=np.arange(src.height),
                dim_1=np.arange(src.width),
                method='nearest').values.astype(src.dtypes[0])
        #            open_kwargs['blockxsize'] = smooth
        #            open_kwargs['blockysize'] = smooth
        else:
            band = src.read(1)

        if constant is not None:
            band = src.read(1)
            band[band >= 0] = int(constant)
        with rasterio.open(
                file_out_tmp + "/measurement/" + os.path.basename(tiff_file),
                'w',
                **open_kwargs
        ) as dst:
            dst.write(band, 1)

    os.rename(file_out_tmp, file_out)

    return file_out


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
    parser.add_argument('-s', action='store', help='average dn by s*s box (in pixel). Use 0 to keep full resolution', default='auto')
    args = parser.parse_args()

    safe_path = os.path.normpath(os.path.expanduser(args.safe_path))
    safe_name = os.path.basename(safe_path)
    safe_type = safe_name.split('_')[1]
    if args.s == 'auto':
        if safe_type in smooth_size:
            smooth = smooth_size[safe_type]
        else:
            raise NotImplementedError('no default smooth size for %s. Use explicit -s' % safe_type)
    else:
        smooth = int(args.s)
    print("Compressing %s..." % os.path.basename(safe_path))
    safe_out = compress_safe(safe_path, args.d, smooth=smooth, constant=args.c, rasterio_kwargs={'compress': args.z})
    in_size = get_dir_size(safe_path)
    out_size = get_dir_size(safe_out)
    ratio = (out_size / in_size) * 100
    zip_file = '%s.zip' % os.path.basename(safe_out)
    zip_msg = subprocess.check_output('cd %s ; zip -r %s %s' % (args.d, zip_file, safe_out), shell=True)
    zip_path = os.path.join(args.d, zip_file)
    zip_size = os.path.getsize(zip_path) / (1024 ** 2)  # in Mb
    zip_ratio = (zip_size / out_size) * 100
    all_ratio = (zip_size / in_size) * 100
    print("%.0fM -> %.0fM ( %.1f%% ) (%s) -> %.0fM ( %.1f%% ) (zip)" % (in_size, out_size, ratio, args.z, zip_size, all_ratio))
    # clean temporary path
    subprocess.check_output('cd %s ; rm -fr %s' % (args.d, safe_out), shell=True)
    print(zip_path)




