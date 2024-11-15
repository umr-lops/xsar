__all__ = [
    "open_dataset",
    "open_datatree",
    "product_info",
    "Sentinel1Meta",
    "Sentinel1Dataset",
    "RadarSat2Dataset",
    "RadarSat2Meta",
    "RcmMeta",
    "RcmDataset",
    "BaseDataset",
    "BaseMeta",
    "get_test_file",
]

from xsar.radarsat2_dataset import RadarSat2Dataset
from xsar.sentinel1_dataset import Sentinel1Dataset
from xsar.sentinel1_meta import Sentinel1Meta
from xsar.rcm_meta import RcmMeta
from xsar.radarsat2_meta import RadarSat2Meta
from xsar.rcm_dataset import RcmDataset
from xsar.base_dataset import BaseDataset
from xsar.base_meta import BaseMeta
from xsar.xsar import open_dataset,open_datatree,product_info
from xsar.xsar import get_test_file

import xsar
try:
    from importlib import metadata
except ImportError: # for Python<3.8
    import importlib_metadata as metadata
try:
    __version__ = metadata.version("xsar")
except Exception:
    __version__ = "999"
