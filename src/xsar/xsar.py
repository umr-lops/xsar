"""
TODO: this docstring is the main xsar module documentation shown to the user. It's should be updated with some examples.
"""

from importlib.metadata import version

__version__ = version('xsar')

import logging
from .utils import timing
import rasterio
import numpy as np
import os
import numbers
import yaml
from importlib_resources import files
from pathlib import Path
import subprocess


def _load_config():
    """
    load config from default xsar/config.yml file or user ~/.xsar/config.yml
    Returns
    -------
    dict
    """
    user_config_file = Path('~/.xsar/config.yml').expanduser()
    default_config_file = files('xsar').joinpath('config.yml')

    if user_config_file.exists():
        config_file = user_config_file
    else:
        config_file = default_config_file

    config = yaml.load(
        config_file.open(),
        Loader=yaml.FullLoader)
    return config


config = _load_config()

from .Sentinel1 import SentinelDataset, SentinelMeta, product_info

logger = logging.getLogger('xsar')
"""
TODO: inform the user how to handle logging
"""
logger.addHandler(logging.NullHandler())

os.environ["GDAL_CACHEMAX"] = "128"


@timing
def open_dataset(dataset_id, resolution=None, resampling=rasterio.enums.Resampling.average, sub_datasets=None,
                 chunks={'xtrack': 120, 'atrack': 120}, pol_dim=True, luts=False, dtypes=None):
    """

    Parameters
    ----------
    dataset_id: str or SentinelMeta object
        if str, it can be a path, or a gdal dataset identifier like `'SENTINEL1_DS:%s:WV_001' % filename`)
    resolution: dict, optional
        resampling dict like `{'atrack': 20, 'xtrack': 20}` where 20 is in pixels.
    resampling: rasterio.enums.Resampling or str, optional
        Only used if `resolution` is not None.
        ` rasterio.enums.Resampling.rms` by default. `rasterio.enums.Resampling.nearest` (decimation) is fastest.
    pol_dim: bool, optional
        if `False`, datasets will not have 'pol' dimension, but several variables names (ie 'sigma0_vv' and 'sigma0_vh').
        (`True` by default).
    luts: bool, optional
        if `True` return also luts as variables (ie `sigma0_lut`, `gamma0_lut`, etc...). False by default.
    chunks: dict, optional
        dict with keys ['pol','atrack','xtrack'] (dask chunks).
        Chunks size will be adjusted so every chunks have the same size. (rechunking later is possible if not wanted)
    dtypes: None or dict, optional
        Specify the data type for each variable. Keys are assumed with `pol_dim=True` (ie no `_vv`).

    Returns
    -------
    xarray.Dataset

    Notes
    -----
      * for `dataset_id`, SentinelMeta object or a full gdal string is mandatory if the SAFE has multiples subdatasets.
      * for `resolution` and `resampling`:
        if `resampling` is `rasterio.enums.Resampling.nearest`, the result looks like:

        >>> res = [ 20 , 20 ] # note that in this case, res is in *pixels*.
        >>> ds.sel(atrack=slice(res[0]/2-1,None,res[0]),xtrack=slice(res[1]/2-1,None,res[1])

        but it's computed much faster.

    """
    # TODO: check product type (S1, RS2), and call specific reader
    if isinstance(dataset_id, SentinelMeta) or isinstance(dataset_id, str) and ".SAFE" in dataset_id:
        sar_obj = SentinelDataset(dataset_id, resolution=resolution, resampling=resampling, pol_dim=pol_dim,
                                  luts=luts, chunks=chunks, dtypes=dtypes)
    else:
        raise TypeError("Unknown dataset type from %s" % str(dataset_id))

    return apply_cf_convention(sar_obj.dataset)


@timing
def apply_cf_convention(dataset):
    """
    Apply `CF-1.7 convention <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html>`_ to a dataset

    Parameters
    ----------
    dataset

    Returns
    -------
    dataset wtih the cf convention
    """

    def to_cf(attr):
        if not isinstance(attr, (str, numbers.Number, np.ndarray, np.number, list, tuple)):
            return str(attr)
        else:
            return attr

    attr_dict = {
        'atrack': {
            'units': '1'
        },
        'xtrack': {
            'units': '1'
        },
        'longitude': {
            'standard_name': 'longitude',
            'units': 'degrees_east'
        },
        'latitude': {
            'standard_name': 'latitude',
            'units': 'degrees_north'
        },
        'sigma0_raw': {
            'units': 'm2/m2'
        },
        'gamma0_raw': {
            'units': 'm2/m2'
        },
    }

    for k, v in dataset.attrs.items():
        dataset.attrs[k] = to_cf(dataset.attrs[k])

    dataset.attrs['Conventions'] = 'CF-1.7'

    for key, attribute in attr_dict.items():
        for key_attr, value in attribute.items():
            if key in dataset:
                dataset[key].attrs[key_attr] = value

    return dataset


def get_test_file(fname):
    """
    get test file from  https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata/
    file is unziped and extracted to `config['data_dir']`

    Parameters
    ----------
    fname: str
        file name to get (without '.zip' extension)

    Returns
    -------
    str
        path to file, relative to `config['data_dir']`

    """

    res_path = os.path.join(config['data_dir'], fname)
    base_url = 'https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata'
    file_url = '%s/%s.zip' % (base_url, fname)
    if not os.path.exists(res_path):
        try:
            subprocess.check_output(
                'cd %s ; wget -N %s' % (config['data_dir'], file_url),
                stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            raise ConnectionError("Unable to fetch %s. Error is:\n %s" % (file_url, e.output.decode("utf-8")))

        try:
            subprocess.check_output(
                'cd %s ; unzip -n %s.zip' % (config['data_dir'], fname),
                stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            raise IOError("Unable to unzip %s.zip.  Error is:\n %s" % (fname, e.output.decode("utf-8")))

    return res_path