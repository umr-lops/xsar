import xarray as xr
import datetime
import numpy as np
import glob
from .utils import bind
import pandas as pd


def resource_strftime(resource, **kwargs):
    """
    From a resource string like '%Y/%j/myfile_%Y%m%d%H%M.nc' and a date like 'Timestamp('2018-10-13 06:23:22.317102')',
    returns a string like '/2018/286/myfile_201810130600.nc'

    Parameters
    ----------
    resource: str

        resource string, with strftime template

    date: datetime

        date to be used

    step: int

        hour step between 2 files


    Returns
    -------
    str

    """

    date = kwargs['date']
    step = kwargs['step']

    delta = datetime.timedelta(hours=step) / 2
    date = date.replace(
        year=(date+delta).year,
        month=(date+delta).month,
        day=(date+delta).day,
        hour=(date+delta).hour // step * step,
        minute=0,
        second=0,
        microsecond=0
    )
    return date.strftime(resource)

def ecmwf_0100_1h(fname):
    """ecmwf 0.1 deg 1h reader (ECMWF_FORECAST_0100_202109091300_10U_10V.nc)"""
    ecmwf_ds = xr.open_dataset(fname).isel(time=0)
    ecmwf_ds.attrs['time'] = datetime.datetime.fromtimestamp(ecmwf_ds.time.item() // 1000000000)
    ecmwf_ds = ecmwf_ds.drop_vars('time').rename(
        {
            'Longitude': 'longitude',
            'Latitude': 'latitude',
            '10U': 'U10',
            '10V': 'V10'
        }
    )
    ecmwf_ds['WSPD'] = np.sqrt(ecmwf_ds.U10**2 + ecmwf_ds.V10**2)
    ecmwf_ds.attrs = {k: ecmwf_ds.attrs[k] for k in ['title', 'institution', 'time']}

    # dataset is lon [0, 360], make it [-180,180]
    ecmwf_ds = ecmwf_ds.roll(longitude=-ecmwf_ds.longitude.size // 2, roll_coords=True)
    ecmwf_ds['longitude'] = xr.where(ecmwf_ds['longitude'] >= 180, ecmwf_ds['longitude'] - 360, ecmwf_ds['longitude'])

    ecmwf_ds.rio.write_crs("EPSG:4326", inplace=True)

    return ecmwf_ds

def gebco(gebco_files):
    """gebco file reader (geotiff from https://www.gebco.net/data_and_products/gridded_bathymetry_data)"""
    return xr.combine_by_coords(
        [
            xr.open_dataset(
                f, chunks={'x': 1000, 'y': 1000}
            ).rename(x='longitude', y='latitude').isel(band=0).drop_vars('band') for f in gebco_files
        ]
    )


# defaults read_function
raster_readers = {
    'ecmwf_0100_1h': ecmwf_0100_1h,
    'gebco': gebco
}

# defaults get_function
raster_getters = {
    'ecmwf_0100_1h': bind(resource_strftime, ..., step=1),
    'gebco': glob.glob
}

# list available rasters as a pandas dataframe
available_rasters = pd.DataFrame(columns=['resource', 'read_function', 'get_function'])
available_rasters.loc['gebco'] = [None, gebco, glob.glob]
available_rasters.loc['ecmwf_0100_1h'] = [None, ecmwf_0100_1h, bind(resource_strftime, ..., step=1)]