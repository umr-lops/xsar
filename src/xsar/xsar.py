"""
TODO: this docstring is the main xsar module documentation shown to the user. It's should be updated with some examples.
"""
import pdb

import warnings
import xarray

try:
    from importlib import metadata
except ImportError: # for Python<3.8
    import importlib_metadata as metadata
__version__ = metadata.version('xsar')

import logging
from .utils import timing, config, url_get
import os
import fsspec
import aiohttp
import zipfile
from . import sentinel1_xml_mappings
from .xml_parser import XmlParser
import pandas as pd
import geopandas as gpd
import xarray as xr


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
    ### geoloc
    geoloc = sar_obj.s1meta.geoloc
    geoloc.attrs['history'] = 'annotations'
    #geoloc = geoloc.rename({'xtrack':'sample_low_res','atrack':'line_low_res'})
    #geoloc = geoloc.rename({'xtrack': 'sample', 'atrack': 'line'})
    # for uu in geoloc:
    #     geoloc = geoloc.rename({uu:uu+'_low_resolution'})
    ### bursts
    bu = sar_obj.s1meta._bursts
    bu.attrs['history'] = 'annotations'
    #bu = bu.rename({'azimuthTime':'azimuthTimeBursts'})
    #bu = bu.rename({'xtrack': 'sample_burst'})

    #azimuth fm rate
    FM = sar_obj.s1meta.azimuth_fmrate
    FM.attrs['history'] = 'annotations'
    # FM = FM.rename({'azimuth_time':'line_FMRate'})
    # for uu in FM:
    #     FM[uu].attrs = {'group':'azimuth FM rate'}
    # dataset principal

    #ds['sample'].attrs = {'slant_spacing':sar_obj.s1meta.image['slantRangePixelSpacing'],'unit':'m'}
    #ds['line'].attrs = {'slant_spacing': sar_obj.s1meta.image['azimuthPixelSpacing'],'unit':'m'}
    sar_obj.dataset['sampleSpacing'] = xarray.DataArray(sar_obj.s1meta.image['slantRangePixelSpacing'],attrs={'unit':'m','referential':'slant'})
    sar_obj.dataset['lineSpacing'] = xarray.DataArray(sar_obj.s1meta.image['azimuthPixelSpacing'],
                                           attrs={'unit': 'm'})
    ds = sar_obj.dataset
    #ds = ds.rename({'atrack':'line','xtrack':'pixel'})
    #doppler
    dop = sar_obj.s1meta._doppler_estimate
    dop.attrs['history'] = 'annotations'
    #final_ds = xr.merge([ds, geoloc, bu,FM])
    import datatree
    final_ds = datatree.DataTree.from_dict({'high_resolution_dataset': ds, 'geolocation_annotation': geoloc,
                                            'bursts':bu,'FMrate':FM,'doppler_estimate':dop,#'image_information':
                                            'orbit':sar_obj.s1meta.orbit
                                            })
    final_ds.attrs=xr.Dataset(sar_obj.s1meta.image)
    return final_ds


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
            for n in s1meta.subdatasets.index:
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
    res_path = config['data_dir']
    base_url = 'https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata'
    file_url = '%s/%s.zip' % (base_url, fname)
    if not os.path.exists(os.path.join(res_path, fname)):
        warnings.warn("Downloading %s" % file_url)
        local_file = url_get(file_url)
        warnings.warn("Unzipping %s" % os.path.join(res_path, fname))
        with zipfile.ZipFile(local_file, 'r') as zip_ref:
            zip_ref.extractall(res_path)
    return os.path.join(res_path, fname)
