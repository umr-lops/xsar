import xarray as xr
import xsar
import rasterio
import warnings
import os


class XsarXarrayBackend(xr.backends.common.BackendEntrypoint):
    def open_dataset(
        self,
        dataset_id,
        resolution=None,
        resampling=rasterio.enums.Resampling.rms,
        luts=False,
        dtypes=None,
        drop_variables=[],
    ):
        ds = xsar.open_dataset(
            dataset_id,
            resolution=resolution,
            resampling=resampling,
            luts=luts,
            dtypes=dtypes,
        )
        if not list(ds.chunks):
            warnings.warn("Not using `chunks` kw is discouraged when openning SAFE")
        return ds.drop_vars(drop_variables, errors="ignore")

    def guess_can_open(self, filename_or_obj):
        if isinstance(filename_or_obj, xsar.Sentinel1Meta):
            return True
        if isinstance(filename_or_obj, str) and os.path.basename(
            filename_or_obj
        ).endswith(".SAFE"):
            return True
        return False
