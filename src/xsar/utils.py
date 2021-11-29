"""miscellaneous functions"""

from functools import wraps
import time
import os
import numpy as np
import logging
from scipy.interpolate import griddata
import xarray as xr
import dask
from functools import reduce, partial
import rasterio
import shutil
import glob

logger = logging.getLogger('xsar.utils')
logger.addHandler(logging.NullHandler())

mem_monitor = True

try:
    from psutil import Process
except ImportError:
    logger.warning("psutil module not found. Disabling memory monitor")
    mem_monitor = False


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
            'timing %s : %.2fs. %s' % (f.__name__, endtime - starttime, mem_str))
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


@timing
def gdal_rms(filename, out_shape, winsize=None):
    """
    Temporary function, waiting fadjuor a better solution on https://github.com/OSGeo/gdal/issues/3196 (gdal>=3.3)
    """
    from osgeo import gdal
    gdal.UseExceptions()
    sourcexml = '''
        <SimpleSource>
            <SourceFilename>{fname}</SourceFilename>
            <SourceBand>{band}</SourceBand>
        </SimpleSource>
        '''

    ds = gdal.Open(filename, gdal.GA_ReadOnly)

    if winsize is None:
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
    else:
        xsize = winsize[2]
        ysize = winsize[3]
    count = ds.RasterCount

    prefix = '/vsimem'
    dn2file = '%s/dn2.vrt' % prefix
    avgfile = '%s/dn2_avg.vrt' % prefix
    datfile = '%s/dn2_avd_dat.vrt' % prefix

    vrt_driver = gdal.GetDriverByName("VRT")
    ds_dn2 = vrt_driver.Create(dn2file, xsize=xsize, ysize=ysize, bands=0)

    # Create square band
    options = [
        'subClass=VRTDerivedRasterBand',
        'PixelFunctionType=intensity',
        'SourceTransferType=UInt32']
    for iband in range(1, ds.RasterCount + 1):
        ds_dn2.AddBand(gdal.GDT_UInt32, options)
        ds_dn2.GetRasterBand(iband).SetMetadata(
            {'source_0': sourcexml.format(fname=filename, band=iband)},
            'vrt_sources')

    ds_dn2.SetProjection(ds.GetProjection())

    ds = None

    ##Set up options for translation
    gdalTranslateOpts = gdal.TranslateOptions(
        format='VRT',
        width=out_shape[0], height=out_shape[1],
        srcWin=[0, 0, xsize, ysize],
        resampleAlg=gdal.gdalconst.GRIORA_Average)

    # translate using average

    ds_dn2_average = gdal.Translate(avgfile, ds_dn2, options=gdalTranslateOpts)
    ds_dn2 = None
    ds_dn2_average = None

    # Write from memory to VRT using pixel functions
    ds_dn2_average = gdal.OpenShared(avgfile)
    ds_dn2_average_data = vrt_driver.Create(datfile, out_shape[0], out_shape[1], 0)
    ds_dn2_average_data.SetProjection(ds_dn2_average.GetProjection())
    ds_dn2_average_data.SetGeoTransform(ds_dn2_average.GetGeoTransform())

    options = ['subClass=VRTDerivedRasterBand',
               'sourceTransferType=UInt32']

    options = ['subClass=VRTDerivedRasterBand',
               'SourceTransferType=UInt32']

    for iband in range(1, count + 1):
        ds_dn2_average_data.AddBand(gdal.GDT_UInt32, options)
        ds_dn2_average_data.GetRasterBand(iband).SetMetadata({'source_0': sourcexml.format(fname=avgfile, band=iband)},
                                                             'vrt_sources')

    dn2_arr = ds_dn2_average.ReadAsArray()

    dn_arr = dn2_arr ** 0.5
    ds_dn2_average_data = None

    gdal.Unlink(dn2file)
    gdal.Unlink(avgfile)
    gdal.Unlink(datfile)

    return dn_arr


def map_blocks_coords(da, func, func_kwargs={}, **kwargs):
    """
    like `dask.map_blocks`, but `func` parameters are dimensions coordinates belonging to the block.

    Parameters
    ----------
    da: xarray.DataArray
        template (meta) of the output dataarray, with dask chunks, dimensions, coordinates and dtype
    func: function
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
            loc = block_info[None]['array-location']
        except TypeError:
            # map_blocks is feeding us some dummy block data to check output type and shape
            # so we juste generate dummy coords to be able to call f
            # (Note : dummy coords are 0 sized if dummy block is empty)
            loc = tuple(zip((0,) * len(block.shape), block.shape))

        # use loc to get corresponding coordinates
        coords_sel = tuple(c[loc[i][0]:loc[i][1]] for i, c in enumerate(coords))

        result = f(*coords_sel, **func_kwargs)

        if dtype is not None:
            result = result.astype(dtype)

        return result

    coords = {c: da[c].values for c in da.dims}
    if 'name' not in kwargs:
        kwargs['name'] = dask.utils.funcname(func)

    meta = da.data
    dtype = meta.dtype

    from_coords = bind(_evaluate_from_coords, ..., ..., coords.values(), dtype=dtype)

    daskarr = meta.map_blocks(from_coords, func, meta=meta, **kwargs)
    dataarr = xr.DataArray(daskarr,
                           dims=da.dims,
                           coords=coords
                           )
    return dataarr


def rioread(subdataset, out_shape, winsize, resampling=rasterio.enums.Resampling.rms):
    """
    wrapper around rasterio.read, to replace self.rio.read and
    avoid 'TypeError: self._hds cannot be converted to a Python object for pickling'
    (see https://github.com/mapbox/rasterio/issues/1731)
    """
    with rasterio.open(subdataset) as rio:
        resampled = rio.read(out_shape=out_shape, resampling=resampling,
                             window=rasterio.windows.Window(*winsize))
        return resampled


def rioread_fromfunction(subdataset, bands, atracks, xtracks, resampling=None, resolution=None):
    """
    rioread version for dask.array.fromfunction
    """
    bounds = (xtracks.min() * resolution['xtrack'], atracks.min() * resolution['atrack'],
              (xtracks.max() + 1) * resolution['xtrack'], (atracks.max() + 1) * resolution['atrack'])
    winsize = (bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1])

    return rioread(subdataset, bands.shape, winsize, resampling=resampling)


def bbox_coords(xs, ys, pad='extends'):
    """
    [(xs[0]-padx, ys[0]-pady), (xs[0]-padx, ys[-1]+pady), (xs[-1]+padx, ys[-1]+pady), (xs[-1]+padx, ys[0]-pady)]
    where padx and pady are xs and ys spacing/2
    """
    bbox_norm = [(0, 0), (0, -1), (-1, -1), (-1, 0)]
    if pad == 'extends':
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
    bbox_ext = [
        (
            xs[x] + xpad[x],
            ys[y] + ypad[y]
        ) for x, y in bbox_norm
    ]
    return bbox_ext


def compress_safe(safe_path_in, safe_path_out, smooth=0, rasterio_kwargs={'compress': 'zstd'}):
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

    safe_path_out_tmp = safe_path_out + '.tmp'
    if os.path.exists(safe_path_out):
        raise FileExistsError("%s already exists" % safe_path_out)
    try:
        shutil.rmtree(safe_path_out_tmp)
    except:
        pass
    os.mkdir(safe_path_out_tmp)

    shutil.copytree(safe_path_in + "/annotation", safe_path_out_tmp + "/annotation")
    shutil.copyfile(safe_path_in + "/manifest.safe", safe_path_out_tmp + "/manifest.safe")

    os.mkdir(safe_path_out_tmp + "/measurement")
    for tiff_file in glob.glob(os.path.join(safe_path_in, 'measurement', '*.tiff')):
        src = rasterio.open(tiff_file)
        open_kwargs = src.profile
        open_kwargs.update(rasterio_kwargs)
        gcps, crs = src.gcps
        open_kwargs['gcps'] = gcps
        open_kwargs['crs'] = crs
        if smooth > 1:
            reduced = xr.DataArray(
                src.read(
                    1, out_shape=(src.height // smooth, src.width // smooth),
                    resampling=rasterio.enums.Resampling.rms))
            mean = reduced.mean().item()
            if not isinstance(mean, complex) and mean < 1:
                raise RuntimeError('rasterio returned empty band. Try to use smallest smooth size')
            reduced = reduced.assign_coords(
                dim_0=reduced.dim_0 * smooth + smooth / 2,
                dim_1=reduced.dim_1 * smooth + smooth / 2)
            band = reduced.interp(
                dim_0=np.arange(src.height),
                dim_1=np.arange(src.width),
                method='nearest').values.astype(src.dtypes[0])
        else:
            band = src.read(1)

        # if constant is not None:
        #    band = src.read(1)
        #    band[band >= 0] = int(constant)
        with rasterio.open(
                safe_path_out_tmp + "/measurement/" + os.path.basename(tiff_file),
                'w',
                **open_kwargs
        ) as dst:
            dst.write(band, 1)

    os.rename(safe_path_out_tmp, safe_path_out)

    return safe_path_out


class Memoize:
    # inspired from https://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/
    # to put cache in obj instance (unlike lru_cache that is global)
    def __init__(self, func):
        self.func = func
        self.memoize = False

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self.__call__, obj)

    def __call__(self, *args, **kwargs):
        if not self.memoize:
            return self.func(*args, **kwargs)
        obj = args[0]
        try:
            cache = obj._memoize_cache
        except AttributeError:
            cache = obj._memoize_cache = {}
        key = (self.func, args[1:], frozenset(kwargs.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kwargs)
        return res
