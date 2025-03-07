"""miscellaneous functions"""

import warnings
import time
import os

import numpy as np
import logging
from scipy.interpolate import griddata
import xarray as xr
import dask
from dask.distributed import get_client
from functools import wraps, partial
import rasterio
import shutil
import glob
import re
import datetime
import string
import pytz
import yaml
from importlib.resources import files
from pathlib import Path
import fsspec
import aiohttp
from lxml import objectify

logger = logging.getLogger("xsar.utils")
logger.addHandler(logging.NullHandler())

mem_monitor = True


try:
    from psutil import Process
except ImportError:
    logger.warning("psutil module not found. Disabling memory monitor")
    mem_monitor = False


def _load_config():
    """
    load config from default xsar/config.yml file or user ~/.xsar/config.yml
    Returns
    -------
    dict
    """
    user_config_file = Path("~/.xsar/config.yml").expanduser()
    default_config_file = files("xsar").joinpath("config.yml")

    if user_config_file.exists():
        config_file = user_config_file
    else:
        config_file = default_config_file

    config = yaml.load(config_file.open(), Loader=yaml.FullLoader)
    return config


global config
config = _load_config()


class bind(partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder
    https://stackoverflow.com/a/66274908
    """

    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)


class class_or_instancemethod(classmethod):
    # see https://stackoverflow.com/a/28238047/5988771
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


def timing(f):
    """provide a @timing decorator for functions, that log time spent in it"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        mem_str = ""
        process = None
        if mem_monitor:
            process = Process(os.getpid())
            startrss = process.memory_info().rss
        starttime = time.time()
        result = f(*args, **kwargs)
        endtime = time.time()
        if mem_monitor:
            endrss = process.memory_info().rss
            mem_str = "mem: %+.1fMb" % ((endrss - startrss) / (1024**2))
        logger.debug(
            "timing %s : %.2fs. %s" % (
                f.__name__, endtime - starttime, mem_str)
        )
        return result

    return wrapper


def to_lon180(lon):
    """

    Parameters
    ----------
    lon: array_like of float, or float
        longitudes in [0, 360] range

    Returns
    -------
    array_like, or float
        longitude in [-180, 180] range

    """
    change = lon > 180
    try:
        lon[change] = lon[change] - 360
    except TypeError:
        # scalar input
        if change:
            lon = lon - 360
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
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters.
    bearing = np.arctan2(
        np.sin(lon2 - lon1) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) *
        np.cos(lat2) * np.cos(lon2 - lon1),
    )
    return c * r, np.rad2deg(bearing)


def minigrid(x, y, z, method="linear", dims=["x", "y"]):
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
    xx, yy = np.meshgrid(x_u, y_u, indexing="ij")
    ngrid = griddata((x, y), z, (xx, yy), method=method)
    return xr.DataArray(ngrid, dims=dims, coords={dims[0]: x_u, dims[1]: y_u})


def map_blocks_coords(da, func, func_kwargs={}, **kwargs):
    """
    like `dask.map_blocks`, but `func` parameters are dimensions coordinates belonging to the block.

    Parameters
    ----------
    da: xarray.DataArray
        template (meta) of the output dataarray, with dask chunks, dimensions, coordinates and dtype
    func: function or future
        function that take gridded `numpy.array` atrack and xtrack, and return a `numpy.array`.
        (see `_evaluate_from_coords`)
    kwargs: dict
        passed to dask.array.map_blocks

    Returns
    -------
    xarray.DataArray
        dataarray with same coords/dims as self from `func(*dims)`.
    """

    def _evaluate_from_coords(block, f, coords, block_info=None, dtype=None):
        """
        evaluate 'f(x_coords,y_coords,...)' with x_coords, y_coords, ... extracted from dims

        Parameters
        ----------
        coords: iterable of numpy.array
            coordinates for each dimension block
        block: numpy.array
            the current block in the dataarray
        f: function
            function to evaluate with 'func(atracks_grid, xtracks_grid)'
        block_info: dict
            provided by 'xarray.DataArray.map_blocks', and used to get block location.

        Returns
        -------
        numpy.array
            result from 'f(*coords_sel)', where coords_sel are dimension coordinates for each dimensions

        Notes
        -----
        block values are not used.
        Unless manualy providing block_info,
        this function should be called from 'xarray.DataArray.map_blocks' with coords preset with functools.partial
        """

        # get loc ((dim_0_start,dim_0_stop),(dim_1_start,dim_1_stop),...)
        try:
            loc = block_info[None]["array-location"]
        except TypeError:
            # map_blocks is feeding us some dummy block data to check output type and shape
            # so we juste generate dummy coords to be able to call f
            # (Note : dummy coords are 0 sized if dummy block is empty)
            loc = tuple(zip((0,) * len(block.shape), block.shape))

        # use loc to get corresponding coordinates
        coords_sel = tuple(c[loc[i][0]: loc[i][1]]
                           for i, c in enumerate(coords))

        result = f(*coords_sel, **func_kwargs)

        if dtype is not None:
            result = result.astype(dtype)

        return result

    coords = {c: da[c].values for c in da.dims}
    if "name" not in kwargs:
        kwargs["name"] = dask.utils.funcname(func)

    meta = da.data
    dtype = meta.dtype

    from_coords = bind(_evaluate_from_coords, ..., ...,
                       coords.values(), dtype=dtype)

    daskarr = meta.map_blocks(from_coords, func, meta=meta, **kwargs)
    dataarr = xr.DataArray(daskarr, dims=da.dims, coords=coords)
    return dataarr


def bbox_coords(xs, ys, pad="extends"):
    """
    [(xs[0]-padx, ys[0]-pady), (xs[0]-padx, ys[-1]+pady), (xs[-1]+padx, ys[-1]+pady), (xs[-1]+padx, ys[0]-pady)]
    where padx and pady are xs and ys spacing/2
    """
    bbox_norm = [(0, 0), (0, -1), (-1, -1), (-1, 0)]
    if pad == "extends":
        xdiff, ydiff = [np.unique(np.diff(d))[0] for d in (xs, ys)]
        xpad = (-xdiff / 2, xdiff / 2)
        ypad = (-ydiff / 2, ydiff / 2)
    elif pad is None:
        xpad = (0, 0)
        ypad = (0, 0)
    else:
        xpad = (-pad[0], pad[0])
        ypad = (-pad[1], pad[1])
    # use apad and xpad to get surrounding box
    bbox_ext = [(xs[x] + xpad[x], ys[y] + ypad[y]) for x, y in bbox_norm]
    return bbox_ext


def compress_safe(
    safe_path_in,
    safe_path_out,
    product="S1",
    smooth=0,
    rasterio_kwargs={"compress": "zstd"},
):
    """

    Parameters
    ----------
    safe_path_in: str
        input SAFE path
    safe_path_out: str
        output SAFE path (be warned to keep good nomenclature)
    rasterio_kwargs: dict
        passed to rasterio.open

    Returns
    -------
    str
        wrotten output path

    """

    safe_path_out_tmp = safe_path_out + ".tmp"
    if os.path.exists(safe_path_out):
        raise FileExistsError("%s already exists" % safe_path_out)
    try:
        shutil.rmtree(safe_path_out_tmp)
    except IOError:
        pass
    os.mkdir(safe_path_out_tmp)
    if "S1" in product:
        shutil.copytree(safe_path_in + "/annotation",
                        safe_path_out_tmp + "/annotation")
        shutil.copyfile(
            safe_path_in + "/manifest.safe", safe_path_out_tmp + "/manifest.safe"
        )

        os.mkdir(safe_path_out_tmp + "/measurement")
        for tiff_file in glob.glob(os.path.join(safe_path_in, "measurement", "*.tiff")):
            src = rasterio.open(tiff_file)
            open_kwargs = src.profile
            open_kwargs.update(rasterio_kwargs)
            gcps, crs = src.gcps
            open_kwargs["gcps"] = gcps
            open_kwargs["crs"] = crs
            if smooth > 1:
                reduced = xr.DataArray(
                    src.read(
                        1,
                        out_shape=(src.height // smooth, src.width // smooth),
                        resampling=rasterio.enums.Resampling.rms,
                    )
                )
                mean = reduced.mean().item()
                if not isinstance(mean, complex) and mean < 1:
                    raise RuntimeError(
                        "rasterio returned empty band. Try to use smallest smooth size"
                    )
                reduced = reduced.assign_coords(
                    dim_0=reduced.dim_0 * smooth + smooth / 2,
                    dim_1=reduced.dim_1 * smooth + smooth / 2,
                )
                band = reduced.interp(
                    dim_0=np.arange(src.height),
                    dim_1=np.arange(src.width),
                    method="nearest",
                )
                try:
                    # convert to original datatype if possible
                    band = band.values.astype(src.dtypes[0])
                except TypeError:
                    pass
            else:
                band = src.read(1)

            with rasterio.open(
                safe_path_out_tmp + "/measurement/" +
                    os.path.basename(tiff_file),
                "w",
                **open_kwargs,
            ) as dst:
                dst.write(band, 1)
    elif "RCM" in product:
        shutil.copytree(safe_path_in + "/metadata",
                        safe_path_out_tmp + "/metadata")
        shutil.copytree(safe_path_in + "/support",
                        safe_path_out_tmp + "/support")
        shutil.copyfile(
            safe_path_in + "/manifest.safe", safe_path_out_tmp + "/manifest.safe"
        )

        os.mkdir(safe_path_out_tmp + "/imagery")
        for tiff_file in glob.glob(os.path.join(safe_path_in, "imagery", "*.tif")):
            src = rasterio.open(tiff_file)
            open_kwargs = src.profile
            open_kwargs.update(rasterio_kwargs)
            gcps, crs = src.gcps
            open_kwargs["gcps"] = gcps
            open_kwargs["crs"] = crs
            if smooth > 1:
                reduced = xr.DataArray(
                    src.read(
                        1,
                        out_shape=(src.height // smooth, src.width // smooth),
                        resampling=rasterio.enums.Resampling.rms,
                    )
                )
                mean = reduced.mean().item()
                if not isinstance(mean, complex) and mean < 1:
                    raise RuntimeError(
                        "rasterio returned empty band. Try to use smallest smooth size"
                    )
                reduced = reduced.assign_coords(
                    dim_0=reduced.dim_0 * smooth + smooth / 2,
                    dim_1=reduced.dim_1 * smooth + smooth / 2,
                )
                band = reduced.interp(
                    dim_0=np.arange(src.height),
                    dim_1=np.arange(src.width),
                    method="nearest",
                )
                try:
                    # convert to original datatype if possible
                    band = band.values.astype(src.dtypes[0])
                except TypeError:
                    pass
            else:
                band = src.read(1)

            with rasterio.open(
                safe_path_out_tmp + "/imagery/" + os.path.basename(tiff_file),
                "w",
                **open_kwargs,
            ) as dst:
                dst.write(band, 1)

    elif "RS2" in product:
        shutil.copytree(safe_path_in + "/schemas",
                        safe_path_out_tmp + "/schemas")
        for xml_file in glob.glob(os.path.join(safe_path_in, "*.xml")):
            shutil.copyfile(
                xml_file, os.path.join(
                    safe_path_out_tmp, os.path.basename(xml_file))
            )
        for tiff_file in glob.glob(os.path.join(safe_path_in, "*.tif")):
            src = rasterio.open(tiff_file)
            open_kwargs = src.profile
            open_kwargs.update(rasterio_kwargs)
            gcps, crs = src.gcps
            open_kwargs["gcps"] = gcps
            open_kwargs["crs"] = crs
            if smooth > 1:
                reduced = xr.DataArray(
                    src.read(
                        1,
                        out_shape=(src.height // smooth, src.width // smooth),
                        resampling=rasterio.enums.Resampling.rms,
                    )
                )
                mean = reduced.mean().item()
                if not isinstance(mean, complex) and mean < 1:
                    raise RuntimeError(
                        "rasterio returned empty band. Try to use smallest smooth size"
                    )
                reduced = reduced.assign_coords(
                    dim_0=reduced.dim_0 * smooth + smooth / 2,
                    dim_1=reduced.dim_1 * smooth + smooth / 2,
                )
                band = reduced.interp(
                    dim_0=np.arange(src.height),
                    dim_1=np.arange(src.width),
                    method="nearest",
                )
                try:
                    # convert to original datatype if possible
                    band = band.values.astype(src.dtypes[0])
                except TypeError:
                    pass
            else:
                band = src.read(1)

            with rasterio.open(
                os.path.join(safe_path_out_tmp, os.path.basename(tiff_file)),
                "w",
                **open_kwargs,
            ) as dst:
                dst.write(band, 1)

    os.rename(safe_path_out_tmp, safe_path_out)

    return safe_path_out


class BlockingActorProxy:
    # http://distributed.dask.org/en/stable/actors.html
    # like dask Actor, but no need to do .result() on methods
    # so the resulting instance is usable like the proxied instance
    def __init__(self, cls, *args, actor=True, **kwargs):

        # the class to be proxied  (ie Sentinel1Meta)
        self._cls = cls
        self._actor = None

        # save for unpickling
        self._args = args
        self._kwargs = kwargs

        self._dask_client = None
        if actor is True:
            try:
                self._dask_client = get_client()
            except ValueError:
                logger.info(
                    "BlockingActorProxy: Transparent proxy for %s" % self._cls.__name__
                )
        elif isinstance(actor, dask.distributed.actor.Actor):
            logger.debug("BlockingActorProxy: Reusing existing actor")
            self._actor = actor

        if self._dask_client is not None:
            logger.debug("submit new actor")
            self._actor_future = self._dask_client.submit(
                self._cls, *args, **kwargs, actors=True
            )
            self._actor = self._actor_future.result()
        elif self._actor is None:
            # transparent proxy: no future
            self._actor = self._cls(*args, **kwargs)

    def __repr__(self):
        return f"<BlockingActorProxy: {self._cls.__name__}>"

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(attr for attr in dir(self._cls) if not attr.startswith("_"))
        return sorted(o)

    def __getattr__(self, key):
        attr = getattr(self._actor, key)
        if not callable(attr):
            return attr
        else:

            @wraps(attr)
            def func(*args, **kwargs):
                res = attr(*args, **kwargs)
                if isinstance(res, dask.distributed.ActorFuture):
                    return res.result()
                else:
                    # transparent proxy
                    return res

            return func

    def __reduce__(self):
        # make self serializable with pickle
        # https://docs.python.org/3/library/pickle.html#object.__reduce__
        kwargs = self._kwargs
        kwargs["actor"] = self._actor
        return partial(BlockingActorProxy, **kwargs), (self._cls, *self._args)


def merge_yaml(yaml_strings_list, section=None):
    # merge a list of yaml strings in one string

    dict_like = yaml.safe_load("\n".join(yaml_strings_list))
    if section is not None:
        dict_like = {section: dict_like}

    return yaml.safe_dump(dict_like)


def get_glob(strlist):
    # from list of str, replace diff by '?'
    def _get_glob(st):
        stglob = "".join(
            [
                "?" if len(charlist) > 1 else charlist[0]
                for charlist in [list(set(charset)) for charset in zip(*st)]
            ]
        )
        return re.sub(r"\?+", "*", stglob)

    strglob = _get_glob(strlist)
    if strglob.endswith("*"):
        strglob += _get_glob(s[::-1] for s in strlist)[::-1]
        strglob = strglob.replace("**", "*")

    return strglob


def safe_dir(filename, path=".", only_exists=False):
    """
    get dir path from safe filename.

    Parameters
    ----------
    filename: str
        SAFE filename, with no dir, and valid nomenclature
    path: str or list of str
        path template
    only_exists: bool
        if True and path doesn't exists, return None.
        if False, return last path found

    Examples
    --------
    For datarmor at ifremer, path template should be:

    '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/${longmissionid}/L${LEVEL}/${BEAM}/${MISSIONID}_${BEAM}_${PRODUCT}${RESOLUTION}_${LEVEL}${CLASS}/${year}/${doy}/${SAFE}'

    For creodias, it should be:

    '/eodata/Sentinel-1/SAR/${PRODUCT}/${year}/${month}/${day}/${SAFE}'

    Returns
    -------
    str
        path from template

    """

    # this function is shared between sentinelrequest and xsar

    if "S1" in filename:
        regex = re.compile(
            "(...)_(..)_(...)(.)_(.)(.)(..)_(........T......)_(........T......)_(......)_(......)_(....).SAFE"
        )
        template = string.Template(
            "${MISSIONID}_${BEAM}_${PRODUCT}${RESOLUTION}_${LEVEL}${CLASS}${POL}_${STARTDATE}_${STOPDATE}_${ORBIT}_${TAKEID}_${PRODID}.SAFE"
        )
    elif "S2" in filename:
        # S2B_MSIL1C_20211026T094029_N0301_R036_T33SWU_20211026T115128.SAFE
        # YYYYMMDDHHMMSS: the datatake sensing start time
        # Nxxyy: the PDGS Processing Baseline number (e.g. N0204)
        # ROOO: Relative Orbit number (R001 - R143)
        # Txxxxx: Tile Number field*
        # second date if product discriminator
        regex = re.compile(
            "(...)_(MSI)(...)_(........T......)_N(....)_R(...)_T(.....)_(........T......).SAFE"
        )
        template = string.Template(
            "${MISSIONID}_${PRODUCT}${LEVEL}_${STARTDATE}_${PROCESSINGBL}_${ORBIT}_${TIlE}_${PRODID}.SAFE"
        )
    else:
        raise Exception("mission not handle")
    regroups = re.search(regex, filename)
    tags = {}
    for itag, tag in enumerate(
        re.findall(r"\$\{([\w]+)\}", template.template), start=1
    ):
        tags[tag] = regroups.group(itag)

    startdate = datetime.datetime.strptime(tags["STARTDATE"], "%Y%m%dT%H%M%S").replace(
        tzinfo=pytz.UTC
    )
    tags["SAFE"] = regroups.group(0)
    tags["missionid"] = tags["MISSIONID"][
        1:3
    ].lower()  # should be replaced by tags["MISSIONID"].lower()
    tags["longmissionid"] = "sentinel-%s" % tags["MISSIONID"][1:3].lower()
    tags["year"] = startdate.strftime("%Y")
    tags["month"] = startdate.strftime("%m")
    tags["day"] = startdate.strftime("%d")
    tags["doy"] = startdate.strftime("%j")
    if isinstance(path, str):
        path = [path]
    filepath = None
    for p in path:
        # deprecation warnings (see https://github.com/oarcher/sentinelrequest/issues/4)
        if "{missionid}" in p:
            warnings.warn(
                "{missionid} tag is deprecated. Update your path template to use {longmissionid}"
            )
        filepath = string.Template(p).substitute(tags)
        if not filepath.endswith(filename):
            filepath = os.path.join(filepath, filename)
        if only_exists:
            if not os.path.isfile(os.path.join(filepath, "manifest.safe")):
                filepath = None
            else:
                # a path was found. Stop iterating over path list
                break
    return filepath


def url_get(url, cache_dir=os.path.join(config["data_dir"], "fsspec_cache")):
    """
    Get fil from url, using caching.

    Parameters
    ----------
    url: str
    cache_dir: str
        Cache dir to use. default to `os.path.join(config['data_dir'], 'fsspec_cache')`

    Raises
    ------
    FileNotFoundError

    Returns
    -------
    filename: str
        The local file name

    Notes
    -----
    Due to fsspec, the returned filename won't match the remote one.
    """

    if "://" in url:
        with fsspec.open(
            "filecache::%s" % url,
            https={"client_kwargs": {
                "timeout": aiohttp.ClientTimeout(total=3600)}},
            filecache={
                "cache_storage": os.path.join(
                    os.path.join(config["data_dir"], "fsspec_cache")
                )
            },
        ) as f:
            fname = f.name
    else:
        fname = url

    return fname


def get_geap_gains(path_aux_cal, mode, pols):
    """
    Find gains Geap associated with mode product and slice number from AUX_CAL.

    DOC : `https://sentinel.esa.int/documents/247904/1877131/DI-MPC-PB-0241-3-10_Sentinel-1IPFAuxiliaryProductSpecification.pdf/ae025687-c3e3-6ab0-de8d-d9cf58657431?t=1669115416469`

    Parameters
    ----------
    path_aux_cal: str

    mode: str
        "IW" for example.

    pols : list
        ["VV","VH"] for example;


    Returns
    ----------
    dict
        return a dict for the given (mode+pols).
        this dictionnary contains a dict with offboresight angle values and associated gains values
    """
    with open(path_aux_cal, "rb") as file:
        xml = file.read()

    root_aux = objectify.fromstring(xml)
    dict_gains = {}

    for calibrationParams in root_aux.calibrationParamsList.getchildren():
        swath = calibrationParams.swath
        polarisation = calibrationParams.polarisation

        if mode in swath.text and polarisation in pols:
            dict_temp = {}

            increment = (
                calibrationParams.elevationAntennaPattern.elevationAngleIncrement
            )

            valuesIQ = np.array(
                [
                    float(e)
                    for e in calibrationParams.elevationAntennaPattern[
                        "values"
                    ].text.split(" ")
                ]
            )

            if valuesIQ.size == 1202:
                gain = np.sqrt(valuesIQ[::2] ** 2 + valuesIQ[1::2] ** 2)
            elif valuesIQ.size == 601:
                gain = 10 ** (valuesIQ / 10)
            else:
                raise ValueError(
                    "valuesIQ must be of size 601 (float) or 1202 (complex)"
                )
            # gain = np.sqrt(valuesIQ[::2]**2+valuesIQ[1::2]**2)

            count = gain.size
            ang = np.linspace(
                -((count - 1) / 2) * increment, ((count - 1) / 2) * increment, count
            )

            dict_temp["offboresightAngle"] = ang
            dict_temp["gain"] = gain
            dict_gains[swath + "_" + polarisation] = dict_temp

    return dict_gains


def get_gproc_gains(path_aux_pp1, mode, product):
    """
    Find gains Gproc associated with mode product and slice number from AUX_PP1.

    DOC : `https://sentinel.esa.int/documents/247904/1877131/DI-MPC-PB-0241-3-10_Sentinel-1IPFAuxiliaryProductSpecification.pdf/ae025687-c3e3-6ab0-de8d-d9cf58657431?t=1669115416469`

    Parameters
    ----------
    path_aux_pp1: str

    Returns
    ----------
    dict
        return a dict of 4 linear gain values for each (mode+product_type+slice_number)
        in this order : Gproc_HH, Gproc_HV, Gproc_VV, Gproc_VH
    """

    # Parse the XML file
    with open(path_aux_pp1, "rb") as file:
        xml = file.read()
    root_pp1 = objectify.fromstring(xml)
    dict_gains = {}
    for prd in root_pp1.productList.getchildren():
        for swathParams in prd.slcProcParams.swathParamsList.getchildren():
            if mode in swathParams.swath.text and product in prd.productId.text:
                gains = [float(g) for g in swathParams.gain.text.split(" ")]
                key = swathParams.swath
                dict_gains[key] = gains
    return dict_gains


def get_path_aux_cal(aux_cal_name):
    """
    Get full path to AUX_CAL file.

    Parameters
    ----------
    aux_cal_name: str
        name of the AUX_CAL file

    Returns
    -------
    str
        full path to the AUX_CAL file
    """
    path = os.path.join(
        config["auxiliary_dir"],
        aux_cal_name[0:3] + "_AUX_CAL",
        aux_cal_name,
        "data",
        aux_cal_name[0:3].lower() + "-aux-cal.xml",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File doesn't exist: {path}")
    return path


def get_path_aux_pp1(aux_pp1_name):
    """
    Get full path to AUX_PP1 file.

    Parameters
    ----------
    aux_pp1_name: str
        name of the AUX_PP1 file

    Returns
    -------
    str
        full path to the AUX_PP1 file
    """
    path = os.path.join(
        config["auxiliary_dir"],
        aux_pp1_name[0:3] + "_AUX_PP1",
        aux_pp1_name,
        "data",
        aux_pp1_name[0:3].lower() + "-aux-pp1.xml",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File doesn't exist: {path}")
    return path


def get_path_aux_ins(aux_ins_name):
    """
    Get full path to AUX_INS file.

    Parameters
    ----------
    aux_ins_name: str
        name of the AUX_INS file

    Returns
    -------
    str
        full path to the AUX_INS file
    """
    path = os.path.join(
        config["auxiliary_dir"],
        aux_ins_name[0:3] + "_AUX_INS",
        aux_ins_name,
        "data",
        aux_ins_name[0:3].lower() + "-aux-ins.xml",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File doesn't exist: {path}")
    return path
