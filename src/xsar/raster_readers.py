import xarray as xr
import datetime
import numpy as np

def ecmwf_01_1h_reader(fname):
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