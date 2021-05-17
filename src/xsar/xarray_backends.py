import xarray as xr
import xsar
import rasterio

class XsarXarrayBackend(xr.backends.common.BackendEntrypoint):
    def open_dataset(self,
                     dataset_id, resolution=None, resampling=rasterio.enums.Resampling.average,
                     luts=False, chunks={'atrack': 5000, 'xtrack': 5000}, dtypes=None, drop_variables=[]):
        ds = xsar.open_dataset(dataset_id, resolution=resolution, resampling=resampling, luts=luts, chunks=chunks, dtypes=dtypes)
        return ds.drop_vars(drop_variables, errors='ignore')

    def guess_can_open(self, filename_or_obj):
        if isinstance(filename_or_obj, xsar.Sentinel1Meta) or isinstance(filename_or_obj, xsar.SentinelMeta):
            return True
        if isinstance(filename_or_obj, str) and '.SAFE' in filename_or_obj:
            return True
        return False
