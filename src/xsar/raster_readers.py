import xarray as xr
import datetime
import numpy as np
import glob
from .utils import bind, url_get
import pandas as pd


def resource_strftime(resource, **kwargs):
    """
    From a resource string like '%Y/%j/myfile_%Y%m%d%H%M.nc' and a date like 'Timestamp('2018-10-13 06:23:22.317102')',
    returns a tuple composed of the closer available date and string like '/2018/286/myfile_201810130600.nc'

    If ressource string is an url (ie 'ftp://ecmwf/%Y/%j/myfile_%Y%m%d%H%M.nc'), fsspec will be used to retreive the file locally.

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
    tuple : (datetime,str)

    """

    date = kwargs["date"]
    step = kwargs["step"]

    delta = datetime.timedelta(hours=step) / 2
    date = date.replace(
        year=(date + delta).year,
        month=(date + delta).month,
        day=(date + delta).day,
        hour=(date + delta).hour // step * step,
        minute=0,
        second=0,
        microsecond=0,
    )

    return date, url_get(date.strftime(resource))


def _to_lon180(ds):
    # roll [0, 360] to [-180, 180] on dim x
    ds = ds.roll(x=-np.searchsorted(ds.x, 180), roll_coords=True)
    ds["x"] = xr.where(ds["x"] >= 180, ds["x"] - 360, ds["x"])
    return ds


def ecmwf_0100_1h(fname, **kwargs):
    """
    ecmwf 0.1 deg 1h reader (ECMWF_FORECAST_0100_202109091300_10U_10V.nc)

    Parameters
    ----------
    fname: str

        hwrf filename

    Returns
    -------
    xarray.Dataset
    """
    ecmwf_ds = xr.open_dataset(
        fname, chunks={'time': 1, 'latitude': 901, 'longitude': 1800}).isel(time=0)
    ecmwf_ds.attrs['time'] = datetime.datetime.fromtimestamp(
        ecmwf_ds.time.item() // 1000000000)
    if 'time' in ecmwf_ds:
        ecmwf_ds = ecmwf_ds.drop("time")
    ecmwf_ds = ecmwf_ds[["Longitude", "Latitude", "10U", "10V"]].rename(
        {"Longitude": "x", "Latitude": "y", "10U": "U10", "10V": "V10"}
    )
    ecmwf_ds.attrs = {k: ecmwf_ds.attrs[k] for k in ["title", "institution", "time"]}

    # dataset is lon [0, 360], make it [-180,180]
    ecmwf_ds = _to_lon180(ecmwf_ds)

    ecmwf_ds.rio.write_crs("EPSG:4326", inplace=True)

    return ecmwf_ds


def ecmwf_0125_1h(fname, **kwargs):
    """
    ecmwf 0.125 deg 1h reader (ecmwf_201709071100.nc)

    Parameters
    ----------
    fname: str

        hwrf filename

    Returns
    -------
    xarray.Dataset
    """
    ecmwf_ds = xr.open_dataset(fname, chunks={"longitude": 1000, "latitude": 1000})

    ecmwf_ds = (
        ecmwf_ds.rename({"longitude": "x", "latitude": "y"})
        .rename({"Longitude": "x", "Latitude": "y", "U": "U10", "V": "V10"})
        .set_coords(["x", "y"])
    )

    ecmwf_ds["x"] = ecmwf_ds.x.compute()
    ecmwf_ds["y"] = ecmwf_ds.y.compute()

    # dataset is lon [0, 360], make it [-180,180]
    ecmwf_ds = _to_lon180(ecmwf_ds)

    ecmwf_ds.attrs["time"] = datetime.datetime.fromisoformat(ecmwf_ds.attrs["date"])

    ecmwf_ds.rio.write_crs("EPSG:4326", inplace=True)

    return ecmwf_ds


def hwrf_0015_3h(fname, **kwargs):
    """
    hwrf 0.015 deg 3h reader ()


    Parameters
    ----------
    fname: str

        hwrf filename

    Returns
    -------
    xarray.Dataset
    """
    hwrf_ds = xr.open_dataset(fname, chunks={'t': 1, 'LAT': 700, 'LON': 700})
    try:
        hwrf_ds = hwrf_ds[["U", "V", "LON", "LAT"]]
        hwrf_ds = hwrf_ds.squeeze("t", drop=True)

    except Exception:
        raise ValueError("date '%s' can't be find in %s " % (kwargs["date"], fname))

    hwrf_ds.attrs["time"] = datetime.datetime.strftime(
        kwargs["date"], "%Y-%m-%d %H:%M:%S"
    )

    hwrf_ds = (
        hwrf_ds.assign_coords(
            {"x": hwrf_ds.LON.values[0, :], "y": hwrf_ds.LAT.values[:, 0]}
        )
        .drop_vars(["LON", "LAT"])
        .rename({"U": "U10", "V": "V10"})
    )

    # hwrf_ds.attrs = {k: hwrf_ds.attrs[k] for k in ['institution', 'time']}
    hwrf_ds = _to_lon180(hwrf_ds)
    hwrf_ds.rio.write_crs("EPSG:4326", inplace=True)

    return hwrf_ds


def era5_0250_1h(fname, **kwargs):
    """
    era5 0.250 deg 1h reader ()


    Parameters
    ----------
    fname: str

        era5 filename

    Returns
    -------
    xarray.Dataset
    """

    ds_era5 = xr.open_dataset(
        fname, chunks={'time': 6, 'latitude025': 721, 'longitude025': 1440})
    ds_era5 = ds_era5[['u10', 'v10', 'latitude025', 'longitude025']]
    ds_era5 = ds_era5.sel(time=str(kwargs['date']), method="nearest")
    ds_era5 = ds_era5.drop('time')

    ds_era5 = ds_era5.rename(
        {"longitude025": "x", "latitude025": "y", "u10": "U10", "v10": "V10"}
    )

    ds_era5.attrs["time"] = kwargs["date"]
    ds_era5 = _to_lon180(ds_era5)
    ds_era5.rio.write_crs("EPSG:4326", inplace=True)
    return ds_era5


def gebco(gebco_files):
    """gebco file reader (geotiff from https://www.gebco.net/data_and_products/gridded_bathymetry_data)"""
    return xr.combine_by_coords(
        [
            xr.open_dataset(f, chunks={"x": 1000, "y": 1000})
            .isel(band=0)
            .drop_vars("band")
            for f in gebco_files
        ]
    )


# list available rasters as a pandas dataframe
available_rasters = pd.DataFrame(columns=["resource", "read_function", "get_function"])
available_rasters.loc["gebco"] = [None, gebco, glob.glob]
available_rasters.loc["ecmwf_0100_1h"] = [
    None,
    ecmwf_0100_1h,
    bind(resource_strftime, ..., step=1),
]
available_rasters.loc["ecmwf_0125_1h"] = [
    None,
    ecmwf_0125_1h,
    bind(resource_strftime, ..., step=1),
]
available_rasters.loc["hwrf_0015_3h"] = [
    None,
    hwrf_0015_3h,
    bind(resource_strftime, ..., step=3),
]
available_rasters.loc["era5_0250_1h"] = [
    None,
    era5_0250_1h,
    bind(resource_strftime, ..., step=1),
]
