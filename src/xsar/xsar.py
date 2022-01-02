"""
TODO: this docstring is the main xsar module documentation shown to the user. It's should be updated with some examples.
"""
import warnings

try:
    from importlib import metadata
except ImportError: # for Python<3.8
    import importlib_metadata as metadata
__version__ = metadata.version('xsar')

import logging
from .utils import timing
import numpy as np
import os
import numbers
import yaml
from importlib_resources import files
from pathlib import Path
import fsspec
import aiohttp
import zipfile
from . import sentinel1_xml_mappings
from .xml_parser import XmlParser
import pandas as pd
import geopandas as gpd


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


from .sentinel1_meta import Sentinel1Meta
from .sentinel1_dataset import Sentinel1Dataset

logger = logging.getLogger('xsar')
"""
TODO: inform the user how to handle logging
"""
logger.addHandler(logging.NullHandler())

os.environ["GDAL_CACHEMAX"] = "128"


@timing
def open_dataset(*args, **kwargs):
    """
    Parameters
    ----------
    *args:
        Passed to `xsar.SentinelDataset`
    **kwargs:
        Passed to `xsar.SentinelDataset`

    Returns
    -------
    xarray.Dataset

    Notes
    -----
    xsar.open_dataset` is a simple wrapper to `xsar.SentinelDataset` that directly returns the `xarray.Dataset` object.

    >>> xsar.Sentinel1Dataset(*args, **kwargs).dataset

    See Also
    --------
    xsar.Sentinel1Dataset
    """
    dataset_id = args[0]
    # TODO: check product type (S1, RS2), and call specific reader
    if isinstance(dataset_id, Sentinel1Meta) or isinstance(dataset_id, str) and ".SAFE" in dataset_id:
        sar_obj = Sentinel1Dataset(*args, **kwargs)
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

def product_info(path, columns='minimal', include_multi=False, _xml_parser=None):
    """

    Parameters
    ----------
    path: str or iterable of str
        path or gdal url.
    columns: list of str or str, optional
        'minimal' by default: only include columns from attributes found in manifest.safe.
        Use 'spatial' to have 'time_range' and 'geometry'.
        Might be a list of properties from `xsar.Sentinel1Meta`
    include_multi: bool, optional
        False by default: don't include multi datasets

    Returns
    -------
    geopandas.GeoDataFrame
      One dataset per lines, with info as columns

    See Also
    --------
    xsar.Sentinel1Meta

    """

    info_keys = {
        'minimal': ['name', 'ipf', 'platform', 'swath', 'product', 'pols', 'meta']
    }
    info_keys['spatial'] = info_keys['minimal'] + ['time_range', 'geometry']

    if isinstance(columns, str):
        columns = info_keys[columns]

    # 'meta' column is not a Sentinel1Meta attribute
    real_cols = [c for c in columns if c != 'meta']
    add_cols = []
    if 'path' not in real_cols:
        add_cols.append('path')
    if 'dsid' not in real_cols:
        add_cols.append('dsid')

    def _meta2df(meta):
        df = pd.Series(data=meta.to_dict(add_cols + real_cols)).to_frame().T
        if 'meta' in columns:
            df['meta'] = meta
        return df

    if isinstance(path, str):
        path = [path]

    if _xml_parser is None:
        _xml_parser = XmlParser(
            xpath_mappings=sentinel1_xml_mappings.xpath_mappings,
            compounds_vars=sentinel1_xml_mappings.compounds_vars,
            namespaces=sentinel1_xml_mappings.namespaces)

    df_list = []
    for p in path:
        s1meta = Sentinel1Meta(p)
        if s1meta.multidataset and include_multi:
            df_list.append(_meta2df(s1meta))
        elif not s1meta.multidataset:
            df_list.append(_meta2df(s1meta))
        if s1meta.multidataset:
            for n in s1meta.subdatasets:
                s1meta = Sentinel1Meta(n)
                df_list.append(_meta2df(s1meta))
    df = pd.concat(df_list).reset_index(drop=True)
    if 'geometry' in df:
        df = gpd.GeoDataFrame(df).set_crs(epsg=4326)

    df = df.set_index(['path', 'dsid'], drop=False)
    if add_cols:
        df = df.drop(columns=add_cols)

    return df


def get_test_file(fname):
    """
    get test file from  https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata/
    file is unzipped and extracted to `config['data_dir']`

    Parameters
    ----------
    fname: str
        file name to get (without '.zip' extension)

    Returns
    -------
    str
        path to file, relative to `config['data_dir']`

    """
    config = _load_config()
    res_path = config['data_dir']
    base_url = 'https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata'
    file_url = '%s/%s.zip' % (base_url, fname)
    if not os.path.exists(os.path.join(res_path, fname)):
        warnings.warn("Downloading %s" % file_url)
        with fsspec.open(
                'filecache::%s' % file_url,
                https={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}},
                filecache={'cache_storage': os.path.join(os.path.join(config['data_dir'], 'fsspec_cache'))}
        ) as f:
            warnings.warn("Unzipping %s" % os.path.join(res_path, fname))
            with zipfile.ZipFile(f, 'r') as zip_ref:
                zip_ref.extractall(res_path)
    return os.path.join(res_path, fname)
