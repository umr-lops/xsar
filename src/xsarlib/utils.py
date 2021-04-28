"""miscellaneous functions"""

from functools import wraps
import time
import os
import numpy as np
import logging
from scipy.interpolate import griddata
import xarray as xr

logger = logging.getLogger('xsar.utils')
logger.addHandler(logging.NullHandler())

mem_monitor = True

try:
    from psutil import Process
except ImportError:
    logger.warning("psutil module not found. Disabling memory monitor")
    mem_monitor = False


def timing(f):
    """provide a @timing decorator for functions, that log time spent in it"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        mem_str = ''
        process = None
        if mem_monitor:
            process = Process(os.getpid())
            startrss = process.memory_info().rss
        starttime = time.time()
        result = f(*args, **kwargs)
        endtime = time.time()
        if mem_monitor:
            endrss = process.memory_info().rss
            mem_str = 'mem: %+.1fMb' % ((endrss - startrss) / (1024 ** 2))
        logger.debug(
            'timing %s : %.1fs. %s' % (f.__name__, endtime - starttime, mem_str))
        return result

    return wrapper


def to_lon180(lon):
    """

    Parameters
    ----------
    lon: array_like of float
        longitudes in [0, 360] range

    Returns
    -------
    array_like
        longitude in [-180, 180] range

    """
    change = lon > 180
    lon[change] = lon[change] - 360
    return lon


def haversine(lon1, lat1, lon2, lat2):
    """
    Compute distance in meters, and bearing in degrees from point1 to point2, assuming spherical earth.

    Parameters
    ----------
    lon1: float
    lat1: float
    lon2: float
    lat2: float

    Returns
    -------
    tuple(float, float)
        distance in meters, and bearing in degrees

    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters.
    bearing = np.arctan2(np.sin(lon2 - lon1) * np.cos(lat2),
                         np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    return c * r, np.rad2deg(bearing)


def xpath_to_dict(element, mapping, namespaces=None):
    """
    use key,xpath from dict mapping to search in lxml element with xpath.

    if xpath is a dict, it will be processed recursively as another mapping.
    in this case, key "." must be used to specify the new xml element.

    Parameters
    ----------
    element: lxml.etree.Element

    mapping: dict
        dict of key, xpath ( xpath may also be a nested mapping dict )

    namespaces: dict
        passed to lxml.etree.xpath

    Returns
    -------
    dict
        dict with same structure as `mapping`, but with `xpath` decoded in a list.

    """
    cur_dict = {}
    for key, path in mapping.items():
        if isinstance(path, dict):
            # path is another element, we need some recursion
            subelement = element.xpath(path['.'], namespaces=namespaces)
            if len(subelement) > 1:
                raise NotImplementedError("sub xpath must return only one element")
            cur_dict[key] = xpath_to_dict(subelement[0], path, namespaces=namespaces)
        else:
            if key == '.':
                continue

            cur_dict[key] = [e.pyval if hasattr(e, 'pyval') else e for e in
                             element.xpath(path, namespaces=namespaces)]
    return cur_dict


def dict_flatten(nested_dict):
    """
    Flatten nested dict.
    Warning: no error will be raised on duplicate key.

    Parameters
    ----------
    nested_dict: dict
        dict with nested dicts

    Returns
    -------
    dict
        flattened dict
    """

    flat_dict = {}
    for k, v in nested_dict.items():
        if not isinstance(v, dict):
            flat_dict[k] = v
        else:
            flat_dict.update(dict_flatten(v))
    return flat_dict


def minigrid(x, y, z, method='linear', dims=['x', 'y']):
    """

    Parameters
    ----------
    x: 1D array_like
        x coordinates

    y: 1D array_like
        y coodinate

    z: 1D array_like
        value at [x, y] coordinates

    method: str
        default to 'linear'. passed to `scipy.interpolate.griddata`

    dims: list of str
        dimensions names for returned dataarray. default to `['x', 'y']`

    Returns
    -------
    xarray.DataArray
        2D grid of `z` interpolated values, with 1D coordinates `x` and `y`.
    """
    x_u = np.unique(np.sort(x))
    y_u = np.unique(np.sort(y))
    xx, yy = np.meshgrid(x_u, y_u, indexing='ij')
    ngrid = griddata((x, y), z, (xx, yy), method=method)
    return xr.DataArray(ngrid, dims=dims, coords={dims[0]: x_u, dims[1]: y_u})
