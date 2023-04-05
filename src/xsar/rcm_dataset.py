# -*- coding: utf-8 -*-
import glob
import logging
import os
import warnings

import rasterio
import yaml
from affine import Affine

from .rcm_meta import RcmMeta
import numpy as np
from .utils import timing, map_blocks_coords, BlockingActorProxy, to_lon180, get_glob
from scipy.interpolate import RectBivariateSpline, interp1d
import dask
import datatree
import xarray as xr
import rioxarray
from .base_dataset import BaseDataset

logger = logging.getLogger('xsar.radarsat2_dataset')
logger.addHandler(logging.NullHandler())

# we know tiff as no geotransform : ignore warning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# allow nan without warnings
# some dask warnings are still non filtered: https://github.com/dask/dask/issues/3245
np.errstate(invalid='ignore')


class RcmDataset(BaseDataset):
    """
        Handle a SAFE subdataset.
        A dataset might contain several tiff files (multiples polarizations), but all tiff files must share the same footprint.

        The main attribute useful to the end-user is `self.dataset` (`xarray.Dataset` , with all variables parsed from xml and tiff files.)

        Parameters
        ----------
        dataset_id: str or RadarSat2Meta object

            if str, it can be a path, or a gdal dataset identifier)

        resolution: dict, number or string, optional
            resampling dict like `{'line': 20, 'sample': 20}` where 20 is in pixels.

            if a number, dict will be constructed from `{'line': number, 'sample': number}`

            if str, it must end with 'm' (meters), like '100m'. dict will be computed from sensor pixel size.

        resampling: rasterio.enums.Resampling or str, optional

            Only used if `resolution` is not None.

            ` rasterio.enums.Resampling.rms` by default. `rasterio.enums.Resampling.nearest` (decimation) is fastest.

        chunks: dict, optional

            dict with keys ['pol','line','sample'] (dask chunks).

        dtypes: None or dict, optional

            Specify the data type for each variable.

        lazyloading: bool, optional
            activate or not the lazy loading of the high resolution fields

    """
    def __init__(self, dataset_id, resolution=None,
                 resampling=rasterio.enums.Resampling.rms,
                 chunks={'line': 5000, 'sample': 5000},
                 dtypes=None, lazyloading=True, skip_variables=None):
        if skip_variables is None:
            skip_variables = []
        # default dtypes
        if dtypes is not None:
            self._dtypes.update(dtypes)

        # default meta for map_blocks output.
        # as asarray is imported from numpy, it's a numpy array.
        # but if later we decide to import asarray from cupy, il will be a cupy.array (gpu)
        self.rcmeta = None
        self.resolution = resolution

        if not isinstance(dataset_id, RcmMeta):
            self.rcmeta = BlockingActorProxy(RcmMeta, dataset_id)
            # check serializable
            # import pickle
            # s1meta = pickle.loads(pickle.dumps(self.s1meta))
            # assert isinstance(rs2meta.coords2ll(100, 100),tuple)
        else:
            # we want self.rs2meta to be a dask actor on a worker
            self.rcmeta = BlockingActorProxy(RcmMeta.from_dict, dataset_id.dict)
        del dataset_id
        self.objet_meta = self.rcmeta

        if self.rcmeta.multidataset:
            raise IndexError(
                """Can't open an multi-dataset. Use `xsar.RadarSat2Meta('%s').subdatasets` to show availables ones""" % self.rcmeta.path
            )

        # build datatree
        DN_tmp = self._load_digital_number(resolution=resolution, resampling=resampling, chunks=chunks)

        ### geoloc
        geoloc = self.rcmeta.geoloc
        geoloc.attrs['history'] = 'annotations'

        ### orbitInformation
        orbit = self.rcmeta.orbit
        orbit.attrs['history'] = 'annotations'

        self.datatree = datatree.DataTree.from_dict({'measurement': DN_tmp, 'geolocation_annotation': geoloc
                                                     })

        self._dataset = self.datatree['measurement'].to_dataset()

    def _list_tiff_files(self):
        """
        Return a list that contains all tiff files paths

        Returns
        -------
        List[str]
            List of Tiff file paths located in a folder
        """

        return glob.glob(os.path.join(self.rcmeta.path, "imagery", "*"))

    def _sort_list_files_and_get_pols(self, list_tiff):
        """
        From a list of tiff files, sort it to get the co polarization tiff file as the first element, and extract pols

        Parameters
        ----------
        list_tiff: List[str]
            List of tiff files

        Returns
        -------
        (List[str], List[str])
            Tuple that contains the tiff files list sorted and the polarizations
        """
        pols = []
        if len(list_tiff) > 1:
            first_base = os.path.basename(list_tiff[0]).split(".")[0]
            first_pol = first_base[-2:]
            if first_pol[0] != first_pol[1]:
                list_tiff.reverse()
        for file in list_tiff:
            base = os.path.basename(file).split(".")[0]
            pol = base[-2:]
            pols.append(pol)
        return list_tiff, pols

    @timing
    def _load_digital_number(self, resolution=None, chunks=None, resampling=rasterio.enums.Resampling.rms):
        """
        load digital_number from tiff files, as an `xarray.Dataset`.

        Parameters
        ----------
        resolution: None, number, str or dict
            see `xsar.open_dataset`
        resampling: rasterio.enums.Resampling
            see `xsar.open_dataset`

        Returns
        -------
        xarray.Dataset
            dataset (possibly dual-pol), with basic coords/dims naming convention
        """

        map_dims = {
            'pol': 'band',
            'line': 'y',
            'sample': 'x'
        }
        tiff_files = self._list_tiff_files()
        tiff_files, pols = self._sort_list_files_and_get_pols(tiff_files)
        if resolution is not None:
            comment = 'resampled at "%s" with %s.%s.%s' % (
                resolution, resampling.__module__, resampling.__class__.__name__, resampling.name)
        else:
            comment = 'read at full resolution'

        # arbitrary rio object, to get shape, etc ... (will not be used to read data)
        rio = rasterio.open(tiff_files[0])

        chunks['pol'] = 1
        # sort chunks keys like map_dims
        chunks = dict(sorted(chunks.items(), key=lambda pair: list(map_dims.keys()).index(pair[0])))
        chunks_rio = {map_dims[d]: chunks[d] for d in map_dims.keys()}
        self.resolution = None
        if resolution is None:
            # using tiff driver: need to read individual tiff and concat them
            # riofiles['rio'] is ordered like self.s1meta.manifest_attrs['polarizations']

            dn = xr.concat(
                [
                    rioxarray.open_rasterio(
                        f, chunks=chunks_rio, parse_coordinates=False
                    ) for f in tiff_files
                ], 'band'
            ).assign_coords(band=np.arange(len(pols)) + 1)

            # set dimensions names
            dn = dn.rename(dict(zip(map_dims.values(), map_dims.keys())))

            # create coordinates from dimension index (because of parse_coordinates=False)
            dn = dn.assign_coords({'line': dn.line, 'sample': dn.sample})
            dn = dn.drop_vars('spatial_ref', errors='ignore')
        else:
            if not isinstance(resolution, dict):
                if isinstance(resolution, str) and resolution.endswith('m'):
                    resolution = float(resolution[:-1])
                    self.resolution = resolution
                resolution = dict(line=resolution / self.rcmeta.pixel_line_m,
                                  sample=resolution / self.rcmeta.pixel_sample_m)
                # resolution = dict(line=resolution / self.dataset['sampleSpacing'].values,
                #                   sample=resolution / self.dataset['lineSpacing'].values)

            # resample the DN at gdal level, before feeding it to the dataset
            out_shape = (
                int(rio.height / resolution['line']),
                int(rio.width / resolution['sample'])
            )
            out_shape_pol = (1,) + out_shape
            # read resampled array in one chunk, and rechunk
            # this doesn't optimize memory, but total size remain quite small

            if isinstance(resolution['line'], int):
                # legacy behaviour: winsize is the maximum full image size that can be divided  by resolution (int)
                winsize = (0, 0, rio.width // resolution['sample'] * resolution['sample'],
                           rio.height // resolution['line'] * resolution['line'])
                window = rasterio.windows.Window(*winsize)
            else:
                window = None

            dn = xr.concat(
                [
                    xr.DataArray(
                        dask.array.from_array(
                            rasterio.open(f).read(
                                out_shape=out_shape_pol,
                                resampling=resampling,
                                window=window
                            ),
                            chunks=chunks_rio
                        ),
                        dims=tuple(map_dims.keys()), coords={'pol': [pol]}
                    ) for f, pol in
                    zip(tiff_files, pols)
                ],
                'pol'
            ).chunk(chunks)

            # create coordinates at box center
            translate = Affine.translation((resolution['sample'] - 1) / 2, (resolution['line'] - 1) / 2)
            scale = Affine.scale(
                rio.width // resolution['sample'] * resolution['sample'] / out_shape[1],
                rio.height // resolution['line'] * resolution['line'] / out_shape[0])
            sample, _ = translate * scale * (dn.sample, 0)
            _, line = translate * scale * (0, dn.line)
            dn = dn.assign_coords({'line': line, 'sample': sample})

        # for GTiff driver, pols are already ordered. just rename them
        dn = dn.assign_coords(pol=pols)

        descr = 'not denoised'
        var_name = 'digital_number'

        dn.attrs = {
            'comment': '%s digital number, %s' % (descr, comment),
            'history': yaml.safe_dump(
                {
                    var_name: get_glob(
                        [p.replace(self.rcmeta.path + '/', '') for p in tiff_files])
                }
            )
        }
        ds = dn.to_dataset(name=var_name)
        astype = self._dtypes.get(var_name)
        if astype is not None:
            ds = ds.astype(self._dtypes[var_name])

        return ds
