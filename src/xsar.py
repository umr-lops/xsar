"""
TODO: this docstring is the main xsar module documentation shown to the user. It's should be updated with some examples.
"""

import logging
import numpy as np
from pkg_resources import get_distribution
from xsarlib import SentinelReader
from xsarlib.utils import timing

__version__ = get_distribution('xsar').version

logger = logging.getLogger('xsar')
"""
TODO: inform the user how to handle logging
"""
logger.addHandler(logging.NullHandler())


@timing
def open_dataset(dataset_path, chunks={'pol': 2, 'xtrack': 1000, 'atrack': 1000}, resolution=None, units='pixels', coars_variables=['sigma0', 'gamma0']):
    """

    Parameters
    ----------
    dataset_path
    chunks
    resolution
    units
    coars_variables

    Returns
    -------

    """
    # TODO: check product type (S1, RS2), and call specific reader
    if ".SAFE" in dataset_path:
        sar_obj = SentinelReader(dataset_path, chunks=chunks)
    else:
        raise TypeError("Unknown dataset type from %s" % dataset_path)

    # get xarray dataset from sar_obj
    sar_ds = sar_obj.sentinel_image_list[0].dataset

    # coarsening, if needed
    sar_ds = coarsen(sar_ds, resolution=resolution, units=units, variables=coars_variables)

    return sar_ds


@timing
def coarsen(ds, resolution=None, units='pixels', variables=['sigma0', 'gamma0'], boundary='trim'):
    """
    Reduce ds resolution by applying mean on a moving box.

    Parameters
    ----------
    ds : xarray.Dataset
        sar dataset

    resolution: dict
        window size, ie `{'xtrack': 10, 'atrack': 10}`

    units: str
        resolution units : 'pixels' or 'meters'

    variables: list of str
        variables from `ds` to coarsen

    boundary
        passed to xarray.DataArray.coarsen

    Returns
    -------
    xarray.Dataset
        sar dataset at lower res
    """

    if resolution is None:
        return ds

    if units == 'meters':
        # compute resolution in pixels
        resolution['xtrack'] = int(np.round(resolution['xtrack'] / ds.attrs['pixel_xtrack_m']))
        resolution['atrack'] = int(np.round(resolution['atrack'] / ds.attrs['pixel_atrack_m']))
        logger.debug('coarsen convert meters to pixels : %s' % str(resolution))

    ds_coarsed = None
    logger.info("coarsing to resolution : %s" % str(resolution))
    for var in variables:
        logger.debug('coarsing var %s' % var)
        # coars var, by applying mean() on box of 'resolution' size
        da_coarsed = ds[var].coarsen(resolution, boundary=boundary).mean()
        if ds_coarsed is None:
            # build a new dataset, by selecting new x/atrack from the coarsed one
            # .sel is faster
            ds_coarsed = ds.sel(xtrack=da_coarsed.xtrack, atrack=da_coarsed.atrack, method='nearest')\
                .drop(variables)
            # .interp is slower, and as of 20200806, https://github.com/pydata/xarray/pull/4155 is not yet merged
            #ds_coarsed = ds.interp(xtrack=da_coarsed.xtrack, atrack=da_coarsed.atrack, method='linear', assume_sorted=True) \
            #    .drop(variables)
        # due to .sel  method='nearest', we have to be sure x/atrack exactly match
        # ( this is uneeded for .interp )
        da_coarsed['xtrack'] = ds_coarsed.xtrack
        da_coarsed['atrack'] = ds_coarsed.atrack
        ds_coarsed = ds_coarsed.assign({var: da_coarsed})
    return ds_coarsed
