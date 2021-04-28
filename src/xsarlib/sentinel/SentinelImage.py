# author : alevieux
# -*- coding: utf-8 -*-
# date : $Date$
# usage : 

import logging

import numpy as np
import xarray as xr

from scipy.interpolate import RectBivariateSpline

from xsarlib.utils import timing, to_lon180, haversine, minigrid

logger = logging.getLogger('xsar.SentinelImage')
logger.addHandler(logging.NullHandler())

class SentinelImage:
    """
    SentinelImage
    """
    def __init__(self, rio, chunks, polarization, metadata, safe_attribute):
        self.rio = rio
        self.chunks = chunks
        self._polarization = polarization
        self._metadata = metadata
        self._safe_attribute = safe_attribute

        self.dataset = xr.Dataset()
        self.__init_attr_dataset()

        self.__parseFiles()

        self.dataset = self.dataset.chunk(self.chunks)

    def __init_attr_dataset(self):
        # polarizations are a dimension
        self.dataset = self.dataset.expand_dims(
            {'pol': self._safe_attribute['polarizations']}
        )
        del self._safe_attribute['polarizations']  # not to be put in self.dataset.attrs

        # all remaining keys are scalar, and to be put as is in dataset.attr
        for k, v in self._safe_attribute.items():
            self.dataset.attrs[k] = v[0]

    @timing
    def __parseFiles(self):
        """TODO: even if this method is internal, it should be documented"""
        self.__parseRasterFile()
        self.__parseLonLat()
        self.__parseCalibrationFile()
        # self.__parseNoise()

    @timing
    def __parseRasterFile(self):
        dn = xr.open_rasterio(self.rio, chunks={'x': self.chunks['xtrack'], 'y': self.chunks['atrack']})
        dn = dn.transpose('band', 'x', 'y')
        ds = dn.to_dataset(name='digital_number')

        ds.attrs = self.dataset.attrs
        ds = ds.assign_coords(band=self._polarization)
        self.dataset = ds.rename(band='pol', x='xtrack', y='atrack')

        # compute acquisition size/resolution in meters
        lons, lats = self.dataset.attrs['footprint'].exterior.coords.xy
        acq_xtrack_meters, _ = haversine(lons[0], lats[0], lons[1], lats[1])
        acq_atrack_meters, _ = haversine(lons[1], lats[1], lons[2], lats[2])
        pix_xtrack_meters = acq_xtrack_meters / self.dataset.xtrack.size
        pix_atrack_meters = acq_atrack_meters / self.dataset.atrack.size
        self.dataset.attrs['coverage'] = "%dkm * %dkm (xtrack * atrack )" % (
            acq_xtrack_meters / 1000, acq_atrack_meters / 1000)
        self.dataset.attrs['pixel_xtrack_m'] = int(np.round(pix_xtrack_meters * 10)) / 10
        self.dataset.attrs['pixel_atrack_m'] = int(np.round(pix_atrack_meters * 10)) / 10

    @timing
    def __parseLonLat(self):
        gcps, _ = self.rio.get_gcps()
        gcps_lin = np.array([[g.col, g.row, g.x, g.y] for g in gcps])

        gcps_lin_xtrack = gcps_lin[:, 0]
        gcps_lin_atrack = gcps_lin[:, 1]
        gcps_lin_lon = gcps_lin[:, 2]
        gcps_lin_lat = gcps_lin[:, 3]
        # map gcp to regular 2D grid
        # GCPs seems to be regularly gridded, but we won't assume it
        gcps_lut_lon = minigrid(gcps_lin_xtrack, gcps_lin_atrack, gcps_lin_lon, dims=['xtrack', 'atrack'])
        gcps_lut_lat = minigrid(gcps_lin_xtrack, gcps_lin_atrack, gcps_lin_lat, dims=['xtrack', 'atrack'])
        idx_xtrack = np.array(gcps_lut_lon.xtrack)
        idx_atrack = np.array(gcps_lut_lon.atrack)

        bounds = self.dataset.attrs['footprint'].bounds
        isAntimeridian = bounds[2] - bounds[0] > 180
        if isAntimeridian:
            gcps_lut_lon %= 360

        # lon/lat interp function from xtrack,atrack
        # dataarray digital_number is taken as a reference, but no value will be used
        # (just the block position)
        dn = self.dataset.digital_number.sel(pol=self._polarization[0])
        interp_lut = RectBivariateSpline(idx_xtrack, idx_atrack,
                                         np.asarray(gcps_lut_lon), kx=1, ky=1).ev
        longitude = dn.data.map_blocks(_get_interp_lut, interp_lut)
        interp_lut = RectBivariateSpline(idx_xtrack, idx_atrack,
                                         np.asarray(gcps_lut_lat), kx=1, ky=1).ev
        latitude = dn.data.map_blocks(_get_interp_lut, interp_lut)

        if isAntimeridian:
            longitude = longitude.map_blocks(to_lon180)

        longitude = xr.DataArray(longitude, dims=['xtrack', 'atrack'],
                                 coords={'xtrack': dn.xtrack.values, 'atrack': dn.atrack.values})

        latitude = xr.DataArray(latitude, dims=['xtrack', 'atrack'],
                                coords={'xtrack': dn.xtrack.values, 'atrack': dn.atrack.values})

        self.dataset['longitude'] = longitude
        self.dataset['latitude'] = latitude

    def __parseCalibrationFile(self):
        self.__parseSigma0()
        self.__parseGamma0()

    @timing
    def __parseSigma0(self):
        sigma0_bypol = []
        for metadata, pol in zip(self._metadata, self._polarization):
            dn = self.dataset.digital_number.sel(pol=pol)
            lut = metadata.sigma0_lut
            # getting x and y inside map_blocks is little hacky ..
            # Warning : do *not* pass lut as a dataset to a map_blocks function.
            # unexpected transpose can occur (to be investigated)
            interp_lut = RectBivariateSpline(np.asarray(lut.xtrack), np.asarray(lut.atrack),
                                             np.asarray(lut), kx=1, ky=1).ev
            sigma0_inter_lut = dn.data.map_blocks(_get_interp_lut, interp_lut)
            sigma0_inter_lut = xr.DataArray(sigma0_inter_lut, dims=['xtrack', 'atrack'],
                                            coords={'xtrack': dn.xtrack.values, 'atrack': dn.atrack.values})
            sigma0 = _apply_calibration_lut(dn, sigma0_inter_lut)
            sigma0_bypol.append(sigma0)
        # concat polarization
        self.dataset['sigma0'] = xr.concat(sigma0_bypol, 'pol')

    @timing
    def __parseGamma0(self):
        gamma0_bypol = []
        for metadata, pol in zip(self._metadata, self._polarization):
            dn = self.dataset.digital_number.sel(pol=pol)
            lut = metadata.gamma0_lut

            interp_lut = RectBivariateSpline(np.asarray(lut.xtrack), np.asarray(lut.atrack),
                                             np.asarray(lut), kx=1, ky=1).ev
            gamma0_interp_lut = dn.data.map_blocks(_get_interp_lut, interp_lut)
            gamma0_interp_lut = xr.DataArray(gamma0_interp_lut, dims=['xtrack', 'atrack'],
                                             coords={'xtrack': dn.xtrack.values, 'atrack': dn.atrack.values})
            gamma0 = _apply_calibration_lut(dn, gamma0_interp_lut)
            gamma0_bypol.append(gamma0)
        # concat polarization
        self.dataset['gamma0'] = xr.concat(gamma0_bypol, 'pol')

    @timing
    def __parseNoise(self):
        noise_bypol = []

        for pol, metadata in zip(self._metadata, self._polarization):
            dn = self.dataset.digital_number.sel(pol=pol)
            lut = metadata.noise_lut

            interp_lut = RectBivariateSpline(np.asarray(lut.xtrack), np.asarray(lut.atrack),
                                             np.asarray(lut), kx=1, ky=1).ev

            noise = dn.data.map_blocks(_get_interp_lut, interp_lut)

            noise = xr.DataArray(noise, dims=['xtrack', 'atrack'],
                                 coords={'xtrack': dn.xtrack.values, 'atrack': dn.atrack.values})
            noise_bypol.append(noise)
        # concat polarization
        self.dataset['noise'] = xr.concat(noise_bypol, 'pol')

def _get_interp_lut(block, interp_lut_f, block_info=None):
    if block_info is None or block.size == 0:
        # map_blocks is feeding us some dummy data to check output type and shape
        # we just return the dummy data as float or complex
        if block.dtype == np.complex:
            return block.astype(np.complex)
        else:
            return block.astype(np.float)

    ((start_xtrack, stop_xtrack), (start_atrack, stop_atrack)) = block_info[None]['array-location']
    xx, yy = np.mgrid[start_xtrack:stop_xtrack, start_atrack:stop_atrack]
    return interp_lut_f(xx, yy)


def _apply_calibration_lut(dn, interp_lut):
    # https://sentinel.esa.int/web/sentinel/radiometric-calibration-of-level-1-products
    return np.abs(dn) ** 2 / (interp_lut ** 2)
