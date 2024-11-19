"""
TODO: this docstring is the main xsar module documentation shown to the user. It's should be updated with some examples.
"""

import warnings

from importlib import metadata

__version__ = metadata.version("xsar")

import logging
from .utils import timing, config, url_get
import os
import zipfile
import pandas as pd
import geopandas as gpd


from .sentinel1_meta import Sentinel1Meta
from .sentinel1_dataset import Sentinel1Dataset
from .radarsat2_meta import RadarSat2Meta
from .radarsat2_dataset import RadarSat2Dataset
from .rcm_meta import RcmMeta
from .rcm_dataset import RcmDataset

logger = logging.getLogger("xsar")
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
    >>> xsar.RadarSat2Dataset(*args, **kwargs).dataset
    >>> xsar.RcmDataset(*args, **kwargs).dataset

    See Also
    --------
    xsar.Sentinel1Dataset
    xsar.RadarSat2Dataset
    xsar.RcmDataset
    """
    dataset_id = args[0]
    # TODO: check product type (S1, RS2), and call specific reader
    if (
        isinstance(dataset_id, Sentinel1Meta)
        or isinstance(dataset_id, str)
        and "S1" in dataset_id
    ):
        sar_obj = Sentinel1Dataset(*args, **kwargs)
    elif (
        isinstance(dataset_id, RadarSat2Meta)
        or isinstance(dataset_id, str)
        and "RS2" in dataset_id
    ):
        sar_obj = RadarSat2Dataset(*args, **kwargs)
    elif (
        isinstance(dataset_id, RcmMeta)
        or isinstance(dataset_id, str)
        and "RCM" in dataset_id
    ):
        sar_obj = RcmDataset(*args, **kwargs)
    else:
        raise TypeError("Unknown dataset type from %s" % str(dataset_id))
    ds = sar_obj.dataset
    return ds


def open_datatree(*args, **kwargs):
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
    >>> xsar.RadarSat2Dataset(*args, **kwargs).dataset
    >>> xsar.RcmDataset(*args, **kwargs).dataset

    See Also
    --------
    xsar.Sentinel1Dataset
    xsar.RadarSat2Dataset
    xsar.RcmDataset
    """
    dataset_id = args[0]
    # Check product type (S1, RS2), and call specific reader
    if (
        isinstance(dataset_id, Sentinel1Meta)
        or isinstance(dataset_id, str)
        and "S1" in dataset_id
    ):
        sar_obj = Sentinel1Dataset(*args, **kwargs)
    elif (
        isinstance(dataset_id, RadarSat2Meta)
        or isinstance(dataset_id, str)
        and "RS2" in dataset_id
    ):
        sar_obj = RadarSat2Dataset(*args, **kwargs)
    elif (
        isinstance(dataset_id, RcmMeta)
        or isinstance(dataset_id, str)
        and "RCM" in dataset_id
    ):
        sar_obj = RcmDataset(*args, **kwargs)
    else:
        raise TypeError("Unknown dataset type from %s" % str(dataset_id))
    dt = sar_obj.datatree
    return dt


def product_info(path, columns="minimal", include_multi=False):
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
        "minimal": ["name", "ipf", "platform", "swath", "product", "pols", "meta"]
    }
    info_keys["spatial"] = info_keys["minimal"] + ["time_range", "geometry"]

    if isinstance(columns, str):
        columns = info_keys[columns]

    # 'meta' column is not a Sentinel1Meta attribute
    real_cols = [c for c in columns if c != "meta"]
    add_cols = []
    if "path" not in real_cols:
        add_cols.append("path")
    if "dsid" not in real_cols:
        add_cols.append("dsid")

    def _meta2df(meta):
        df = pd.Series(data=meta.to_dict(add_cols + real_cols)).to_frame().T
        if "meta" in columns:
            df["meta"] = meta
        return df

    if isinstance(path, str):
        path = [path]

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
    if "geometry" in df:
        df = gpd.GeoDataFrame(df).set_crs(epsg=4326)

    df = df.set_index(["path", "dsid"], drop=False)
    if add_cols:
        df = df.drop(columns=add_cols)

    return df


def get_test_file(
    fname, base_url="https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata"
):
    """
    get test file from base_url(https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata/)
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
    res_path = config["data_dir"]
    file_url = "%s/%s.zip" % (base_url, fname)
    if not os.path.exists(os.path.join(res_path, fname)):
        warnings.warn("Downloading %s" % file_url)
        local_file = url_get(file_url)
        warnings.warn("Unzipping %s" % os.path.join(res_path, fname))
        with zipfile.ZipFile(local_file, "r") as zip_ref:
            zip_ref.extractall(res_path)
    return os.path.join(res_path, fname)
