import warnings
from abc import ABC
from datetime import datetime

import dask
import pandas as pd
import shapely
import xarray as xr
import yaml
from affine import Affine
from numpy import asarray
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon, box
import numpy as np
import logging
import rasterio
from rasterio.control import GroundControlPoint
from shapely.validation import make_valid
import geopandas as gpd
from scipy.spatial import KDTree
import time
import rasterio.features


from xsar.utils import bbox_coords, haversine, map_blocks_coords, timing

logger = logging.getLogger("xsar.base_dataset")
logger.addHandler(logging.NullHandler())

# we know tiff as no geotransform : ignore warning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# allow nan without warnings
# some dask warnings are still non filtered: https://github.com/dask/dask/issues/3245
np.errstate(invalid="ignore")


class BaseDataset(ABC):
    """
    Abstract class that defines necessary common functions for the computation of different SAR dataset variables
    (Radarsat2, Sentinel1, RCM...).
    This also permit a better maintenance, because these functions aren't redefined many times.
    """

    datatree = None
    _dataset = None
    name = None
    sliced = False
    sar_meta = None
    _rasterized_masks = None
    resolution = None
    _da_tmpl = None
    _luts = None
    _map_var_lut = None
    _dtypes = {
        "latitude": "f4",
        "longitude": "f4",
        "incidence": "f4",
        "elevation": "f4",
        "altitude": "f4",
        "ground_heading": "f4",
        "offboresight": "f4",
        "nesz": None,
        "negz": None,
        "sigma0_raw": None,
        "gamma0_raw": None,
        "noise_lut": "f4",
        "noise_lut_range": "f4",
        "noise_lut_azi": "f4",
        "sigma0_lut": "f8",
        "gamma0_lut": "f8",
        "azimuth_time": np.datetime64,
        "slant_range_time": None,
    }
    _default_meta = asarray([], dtype="f8")
    geoloc_tree = None

    @property
    def len_line_m(self):
        """line length, in meters"""
        _bbox_ll = list(zip(*self._bbox_ll))
        len_m, _ = haversine(*_bbox_ll[1], *_bbox_ll[2])
        return len_m

    @property
    def len_sample_m(self):
        """sample length, in meters"""
        _bbox_ll = list(zip(*self._bbox_ll))
        len_m, _ = haversine(*_bbox_ll[0], *_bbox_ll[1])
        return len_m

    @property
    def coverage(self):
        """coverage string"""
        return "%dkm * %dkm (line * sample )" % (
            self.len_line_m / 1000,
            self.len_sample_m / 1000,
        )

    @property
    def _regularly_spaced(self):
        return (
            max(
                [
                    np.unique(np.round(np.diff(self._dataset[dim].values), 1)).size
                    for dim in ["line", "sample"]
                ]
            )
            == 1
        )

    @property
    def _bbox_ll(self):
        """Dataset bounding box, lon/lat"""
        return self.sar_meta.coords2ll(*zip(*self._bbox_coords))

    @property
    def _bbox_coords(self):
        """
        Dataset bounding box, in line/sample coordinates
        """
        bbox_ext = bbox_coords(self.dataset.line.values, self.dataset.sample.values)
        return bbox_ext

    @property
    def geometry(self):
        """
        geometry of this dataset, as a `shapely.geometry.Polygon` (lon/lat coordinates)
        """
        return Polygon(zip(*self._bbox_ll))

    def load_ground_heading(self):
        """
        Load ground heading as delayed thanks to `BaseMeta.coords2heading`.

        Returns
        -------
        xarray.Dataset
            Contains the ground heading
        """

        def coords2heading(lines, samples):
            return self.sar_meta.coords2heading(
                lines, samples, to_grid=True, approx=True
            )

        gh = map_blocks_coords(
            self._da_tmpl.astype(self._dtypes["ground_heading"]),
            coords2heading,
            name="ground_heading",
        )

        gh.attrs = {
            "comment": "at ground level, computed from lon/lat in azimuth direction",
            "long_name": "Platform heading (azimuth from North)",
            "units": "Degrees",
        }

        return gh.to_dataset(name="ground_heading")

    def add_rasterized_masks(self):
        """
        add rasterized masks only (included in add_high_resolution_variables() for Sentinel-1)
        :return:
        """
        self._rasterized_masks = self.load_rasterized_masks()
        # self.datatree['measurement'].ds = xr.merge([self.datatree['measurement'].ds,self._rasterized_masks])
        self.datatree["measurement"] = self.datatree["measurement"].assign(
            xr.merge([self.datatree["measurement"].ds, self._rasterized_masks])
        )

    def recompute_attrs(self):
        """
        Recompute dataset attributes. It's automaticaly called if you assign a new dataset, for example

        >>> xsar_obj.dataset = xsar_obj.dataset.isel(line=slice(1000, 5000))
        >>> # xsar_obj.recompute_attrs() # not needed

        This function must be manually called before using the `.rio` accessor of a variable

        >>> xsar_obj.recompute_attrs()
        >>> xsar_obj.dataset["sigma0"].rio.reproject(...)

        See Also
        --------
            [rioxarray information loss](https://corteva.github.io/rioxarray/stable/getting_started/manage_information_loss.html)

        """
        if not self._regularly_spaced:
            warnings.warn(
                "Irregularly spaced dataset (probably multiple selection). Some attributes will be incorrect."
            )
        attrs = self._dataset.attrs
        # attrs['pixel_sample_m'] = self.pixel_sample_m
        # attrs['pixel_line_m'] = self.pixel_line_m
        attrs["coverage"] = self.coverage
        attrs["footprint"] = self.footprint

        self.dataset.attrs.update(attrs)

        self._dataset = self._set_rio(self._dataset)

        return None

    def coords2ll(self, *args, **kwargs):
        """
        Alias for `xsar.BaseMeta.coords2ll`
        See Also
        --------
        xsar.BaseMeta.coords2ll
        """
        return self.sar_meta.coords2ll(*args, **kwargs)

    def ll2coords(self, *args):
        """
        Get `(lines, samples)` from `(lon, lat)`,
        or convert a lon/lat shapely object to line/sample coordinates.

        Parameters
        ----------
        *args: lon, lat or shapely object

            lon and lat might be iterables or scalars

        Returns
        -------
        tuple of np.array or tuple of float (lines, samples) , or a shapely object

        Notes
        -----
        The difference with `xsar.BaseMeta.ll2coords` is that coordinates are rounded to the nearest dataset coordinates.

        See Also
        --------
        xsar.BaseMeta.ll2coords

        """
        if isinstance(args[0], shapely.geometry.base.BaseGeometry):
            return self.sar_meta._ll2coords_shapely(args[0].intersection(self.geometry))

        line, sample = self.sar_meta.ll2coords(*args)

        if hasattr(args[0], "__iter__"):
            scalar = False
        else:
            scalar = True

        tolerance = (
            np.max(
                [
                    np.percentile(np.diff(self.dataset[c].values), 90) / 2
                    for c in ["line", "sample"]
                ]
            )
            + 1
        )
        try:
            # select the nearest valid pixel in ds
            ds_nearest = self.dataset.sel(
                line=line, sample=sample, method="nearest", tolerance=tolerance
            )
            if scalar:
                (line, sample) = (
                    ds_nearest.line.values.item(),
                    ds_nearest.sample.values.item(),
                )
            else:
                (line, sample) = (ds_nearest.line.values, ds_nearest.sample.values)
        except KeyError:
            # out of bounds, because of `tolerance` keyword
            (line, sample) = (line * np.nan, sample * np.nan)

        return line, sample

    def _set_rio(self, ds):
        # set .rio accessor for ds. ds must be same kind a self._dataset (not checked!)
        gcps = self._local_gcps

        want_dataset = True
        if isinstance(ds, xr.DataArray):
            # temporary convert to dataset
            try:
                ds = ds.to_dataset()
            except ValueError:
                ds = ds.to_dataset(name="_tmp_rio")
            want_dataset = False

        for v in ds:
            if set(["line", "sample"]).issubset(set(ds[v].dims)):
                ds[v] = ds[v].set_index({"sample": "sample", "line": "line"})
                ds[v] = (
                    ds[v]
                    .rio.write_gcps(gcps, "epsg:4326", inplace=True)
                    .rio.set_spatial_dims("sample", "line", inplace=True)
                    .rio.write_coordinate_system(inplace=True)
                )
                # remove/reset some incorrect attrs set by rio
                # (long_name is 'latitude', but it's incorrect for line axis ...)
                for ax in ["line", "sample"]:
                    [
                        ds[v][ax].attrs.pop(k, None)
                        for k in ["long_name", "standard_name"]
                    ]
                    ds[v][ax].attrs["units"] = "1"

        if not want_dataset:
            # convert back to dataarray
            ds = ds[v]
            if ds.name == "_tmp_rio":
                ds.name = None
        return ds

    @property
    def _local_gcps(self):
        # get local gcps, for rioxarray.reproject (row and col are *index*, not coordinates)
        local_gcps = []
        line_decimated = self.dataset.line.values[
            :: int(self.dataset.line.size / 20) + 1
        ]
        sample_decimated = self.dataset.sample.values[
            :: int(self.dataset.sample.size / 20) + 1
        ]
        XX, YY = np.meshgrid(line_decimated, sample_decimated)
        if self.sar_meta.product == "SLC":
            logger.debug(
                "GCPs computed from affine transformations on SLC products can be strongly shifted in position, we advise against ds.rio.reproject()"
            )
        #     lon_s,lat_s = self.coords2ll_SLC(XX.ravel(order='F'),YY.ravel(order='F'))
        #     lon_s = lon_s.values
        #     lat_s = lat_s.values
        cpt = 0
        for line in line_decimated.astype(int):
            for sample in sample_decimated.astype(int):
                irow = np.argmin(np.abs(self.dataset.line.values - line))
                irow = int(irow)
                icol = np.argmin(np.abs(self.dataset.sample.values - sample))
                icol = int(icol)
                # if self.s1meta.product == 'SLC':
                #     #lon, lat = self.coords2ll_SLC(line,sample)
                #     lon = lon_s[cpt]
                #     lat = lat_s[cpt]
                #     if sample%0==0:
                #         print('#########')
                #         print('lon',lon)
                #         print('lat',lat)
                # else:
                lon, lat = self.sar_meta.coords2ll(line, sample)
                gcp = GroundControlPoint(x=lon, y=lat, z=0, col=icol, row=irow)
                local_gcps.append(gcp)
                cpt += 1
        return local_gcps

    def get_burst_valid_location(self):
        """
        add a field 'valid_location' in the bursts sub-group of the datatree

        Returns:
        --------

        """
        nbursts = len(self.datatree["bursts"].ds["burst"])
        burst_firstValidSample = self.datatree["bursts"].ds["firstValidSample"].values
        burst_lastValidSample = self.datatree["bursts"].ds["lastValidSample"].values
        valid_locations = np.empty((nbursts, 4), dtype="int32")
        line_per_burst = len(self.datatree["bursts"].ds["line"])
        for ibur in range(nbursts):
            fvs = burst_firstValidSample[ibur, :]
            lvs = burst_lastValidSample[ibur, :]
            # valind = np.where((fvs != -1) | (lvs != -1))[0]
            valind = np.where(np.isfinite(fvs) | np.isfinite(lvs))[0]
            valloc = [
                ibur * line_per_burst + valind.min(),
                fvs[valind].min(),
                ibur * line_per_burst + valind.max(),
                lvs[valind].max(),
            ]
            valid_locations[ibur, :] = valloc
        tmpda = xr.DataArray(
            dims=["burst", "limits"],
            coords={
                "burst": self.datatree["bursts"].ds["burst"].values,
                "limits": np.arange(4),
            },
            data=valid_locations,
            name="valid_location",
            attrs={
                "description": "start line index, start sample index, stop line index, stop sample index"
            },
        )
        # self.datatree['bursts'].ds['valid_location'] = tmpda
        tmpds = xr.merge([self.datatree["bursts"].ds, tmpda])
        self.datatree["bursts"] = self.datatree["bursts"].assign(tmpds)

    def get_bursts_polygons(self, only_valid_location=True):
        """
        get the polygons of radar bursts in the image geometry

        Parameters
        ----------
        only_valid_location : bool
            [True] -> polygons of the TOPS SLC bursts are cropped using valid location index
            False -> polygons of the TOPS SLC bursts are aligned with azimuth time start/stop index

        Returns
        -------
        geopandas.GeoDataframe
            polygons of the burst in the image (ie line/sample) geometry
            'geometry' is the polygon

        """
        if self.resolution is not None:
            # eg 2.3/100
            factor_range = self.dataset["sampleSpacing"].values / self.resolution
            factor_azimuth = self.dataset["lineSpacing"].values / self.resolution
        else:
            factor_range = 1
            factor_azimuth = 1
        # compute resolution factor if any
        if self.sar_meta.multidataset:
            blocks_list = []
            # for subswath in self.subdatasets.index:
            for submeta in self.sar_meta._submeta:
                block = submeta.bursts(only_valid_location=only_valid_location)
                block["subswath"] = submeta.dsid
                block = block.set_index("subswath", append=True).reorder_levels(
                    ["subswath", "burst"]
                )
                blocks_list.append(block)
            blocks = pd.concat(blocks_list)
        else:
            # burst_list = self._bursts
            self.get_burst_valid_location()
            burst_list = self.datatree["bursts"].ds
            nb_samples = int(
                self.datatree["image"].ds["numberOfSamples"] * factor_range
            )
            if burst_list["burst"].size == 0:
                blocks = gpd.GeoDataFrame()
            else:
                bursts = []
                bursts_az_inds = {}
                inds_burst, geoloc_azitime, geoloc_iburst, geoloc_line = (
                    self._get_indices_bursts()
                )
                for burst_ind, uu in enumerate(np.unique(inds_burst)):
                    if only_valid_location:
                        extent = np.copy(
                            burst_list["valid_location"].values[burst_ind, :]
                        )
                        area = box(
                            int(extent[0] * factor_azimuth),
                            int(extent[1] * factor_range),
                            int(extent[2] * factor_azimuth),
                            int(extent[3] * factor_range),
                        )
                    else:
                        inds_one_val = np.where(inds_burst == uu)[0]
                        bursts_az_inds[uu] = inds_one_val
                        area = box(
                            bursts_az_inds[burst_ind][0],
                            0,
                            bursts_az_inds[burst_ind][-1],
                            nb_samples,
                        )
                    burst = pd.Series(dict([("geometry_image", area)]))
                    bursts.append(burst)
                # to geopandas
                blocks = pd.concat(bursts, axis=1).T
                blocks = gpd.GeoDataFrame(blocks)
                blocks["geometry"] = blocks["geometry_image"].apply(self.coords2ll)
                blocks.index.name = "burst"
        return blocks

    def _get_indices_bursts(self):
        """

        Returns
        -------
        ind np.array
            index of the burst start in the line coordinates
        geoloc_azitime np.array
            azimuth time at the middle of the image from geolocation grid (low resolution)
        geoloc_iburst np.array

        """
        ind = None
        geoloc_azitime = None
        geoloc_iburst = None
        geoloc_line = None
        if self.sar_meta.product == "SLC" and "WV" not in self.sar_meta.swath:
            # if self.datatree.attrs['product'] == 'SLC' and 'WV' not in self.datatree.attrs['swath']:
            burst_nlines = int(self.sar_meta._bursts["linesPerBurst"])
            # burst_nlines = self.datatree['bursts'].ds['line'].size

            geoloc_line = self.sar_meta.geoloc["line"].values
            # geoloc_line = self.datatree['geolocation_annotation'].ds['line'].values
            # find the indice of the bursts in the geolocation grid
            geoloc_iburst = np.floor(geoloc_line / float(burst_nlines)).astype("int32")
            # find the indices of the bursts in the high resolution grid
            line = np.arange(0, self.sar_meta.image["numberOfLines"])
            # line = np.arange(0, self.datatree.attrs['numberOfLines'])
            iburst = np.floor(line / float(burst_nlines)).astype("int32")
            # find the indices of the burst transitions
            ind = np.searchsorted(geoloc_iburst, iburst, side="left")
            n_pixels = int((len(self.sar_meta.geoloc["sample"]) - 1) / 2)
            geoloc_azitime = self.sar_meta.geoloc["azimuthTime"].values[:, n_pixels]
            # security check for unrealistic line_values exceeding the image extent
            if ind.max() >= len(geoloc_azitime):
                ind[ind >= len(geoloc_azitime)] = len(geoloc_azitime) - 1
        return ind, geoloc_azitime, geoloc_iburst, geoloc_line

    @timing
    def load_rasterized_masks(self):
        """
        Load rasterized masks

        Returns
        -------
        xarray.Dataset
            Contains rasterized masks dataset
        """

        def _rasterize_mask_by_chunks(line, sample, mask="land"):
            chunk_coords = bbox_coords(line, sample, pad=None)
            # chunk footprint polygon, in dataset coordinates (with buffer, to enlarge a little the footprint)
            chunk_footprint_coords = Polygon(chunk_coords).buffer(10)
            # chunk footprint polygon, in lon/lat
            chunk_footprint_ll = self.sar_meta.coords2ll(chunk_footprint_coords)

            # get vector mask over chunk, in lon/lat
            vector_mask_ll = self.sar_meta.get_mask(mask).intersection(
                chunk_footprint_ll
            )

            if vector_mask_ll.is_empty:
                # no intersection with mask, return zeros
                return np.zeros((line.size, sample.size))

            # vector mask, in line/sample coordinates
            vector_mask_coords = self.ll2coords(vector_mask_ll)

            # shape of the returned chunk
            out_shape = (line.size, sample.size)

            # transform * (x, y) -> (line, sample)
            # (where (x, y) are index in out_shape)
            # Affine.permutation() is used because (line, sample) is transposed from geographic

            transform = (
                Affine.translation(*chunk_coords[0])
                * Affine.scale(*[np.unique(np.diff(c))[0] for c in [line, sample]])
                * Affine.permutation()
            )

            raster_mask = rasterio.features.rasterize(
                [vector_mask_coords],
                out_shape=out_shape,
                all_touched=False,
                transform=transform,
            )
            return raster_mask

        da_list = []
        for mask in self.sar_meta.mask_names:
            da_mask = map_blocks_coords(
                self._da_tmpl, _rasterize_mask_by_chunks, func_kwargs={"mask": mask}
            )
            name = "%s_mask" % mask
            da_mask.attrs["history"] = yaml.safe_dump(
                {name: self.sar_meta.get_mask(mask, describe=True)}
            )
            da_mask.attrs["meaning"] = "0: ocean , 1: land"
            da_list.append(da_mask.to_dataset(name=name))

        return xr.merge(da_list)

    def ll2coords_SLC(self, *args):
        """
        for SLC product with irregular projected pixel spacing in range Affine transformation are not relevant
        :return:
        """
        stride_dataset_line = 10  # stride !=1 to save computation time, hard coded here to avoid issues of between KDtree serialized and possible different stride usage
        stride_dataset_sample = 30
        # stride_dataset_line = 1 # stride !=1 to save computation time, hard coded here to avoid issues of between KDtree serialized and possible different stride usage
        # stride_dataset_sample = 1
        lon, lat = args
        subset_lon = self.dataset["longitude"].isel(
            {
                "line": slice(None, None, stride_dataset_line),
                "sample": slice(None, None, stride_dataset_sample),
            }
        )
        subset_lat = self.dataset["latitude"].isel(
            {
                "line": slice(None, None, stride_dataset_line),
                "sample": slice(None, None, stride_dataset_sample),
            }
        )
        if self.geoloc_tree is None:
            t0 = time.time()
            lontmp = subset_lon.values.ravel()
            lattmp = subset_lat.values.ravel()
            self.geoloc_tree = KDTree(np.c_[lontmp, lattmp])
            logger.debug("tree ready in %1.2f sec" % (time.time() - t0))
        ll = np.vstack([lon, lat]).T
        dd, ii = self.geoloc_tree.query(ll, k=1)
        line, sample = np.unravel_index(ii, subset_lat.shape)
        return line * stride_dataset_line, sample * stride_dataset_sample

    def coords2ll_SLC(self, *args):
        """
            for SLC product with irregular projected pixel spacing in range Affine transformation are not relevant

        Returns
        -------
        """
        lines, samples = args
        if isinstance(lines, list) or isinstance(lines, np.ndarray):
            pass
        else:  # in case of a single point,
            # to avoid error when declaring the da_line and da_sample below
            lines = [lines]
            samples = [samples]
        da_line = xr.DataArray(lines, dims="points")
        da_sample = xr.DataArray(samples, dims="points")
        lon = self.dataset["longitude"].sel(line=da_line, sample=da_sample)
        lat = self.dataset["latitude"].sel(line=da_line, sample=da_sample)
        return lon, lat

    def land_mask_slc_per_bursts(self, lazy_loading=True):
        """
        1) loop on burst polygons to get rasterized landmask
        2) merge the landmask pieces into a single Dataset to replace existing 'land_mask' is any

        Parameters
        ----------

        lazy_loading bool

        Returns
        -------

        """
        # TODO: add a prior step to compute the intersection between the self.dataset (could be a subset) and the different bursts
        # if 'land_mask' in self.dataset:
        #     self.datatree['measurement'] = self.datatree['measurement'].assign(self.datatree['measurement'].to_dataset().drop('land_mask'))
        logger.debug("start land_mask_slc_per_bursts()")

        def _rasterize_mask_by_chunks(line, sample, mask="land"):
            """
            copy/pasted from load_rasterized_masks()
            :param line:
            :param sample:
            :param mask:
            :return:
            """
            chunk_coords = bbox_coords(line, sample, pad=None)
            # chunk footprint polygon, in dataset coordinates (with buffer, to enlarge a little the footprint)
            # .buffer(1) #no buffer-> corruption of the polygon
            chunk_footprint_coords = Polygon(chunk_coords)
            assert chunk_footprint_coords.is_valid
            lines, samples = chunk_footprint_coords.exterior.xy
            lines = np.array([hh for hh in lines]).astype(int)
            lines = np.clip(lines, a_min=0, a_max=self.dataset["line"].max().values)
            samples = np.array([hh for hh in samples]).astype(int)
            samples = np.clip(
                samples, a_min=0, a_max=self.dataset["sample"].max().values
            )
            chunk_footprint_lon, chunk_footprint_lat = self.coords2ll_SLC(
                lines, samples
            )
            chunk_footprint_ll = Polygon(
                np.vstack([chunk_footprint_lon, chunk_footprint_lat]).T
            )
            if chunk_footprint_ll.is_valid is False:
                chunk_footprint_ll = make_valid(chunk_footprint_ll)
            # get vector mask over chunk, in lon/lat
            vector_mask_ll = self.sar_meta.get_mask(mask).intersection(
                chunk_footprint_ll
            )
            if vector_mask_ll.is_empty:
                # no intersection with mask, return zeros
                return np.zeros((line.size, sample.size))

            # vector mask, in line/sample coordinates
            if isinstance(vector_mask_ll, shapely.geometry.Polygon):
                lons_ma, lats_ma = vector_mask_ll.exterior.xy
                lons = np.array([hh for hh in lons_ma])
                lats = np.array([hh for hh in lats_ma])
                vector_mask_coords_lines, vector_mask_coords_samples = (
                    self.ll2coords_SLC(lons, lats)
                )
                vector_mask_coords = [
                    Polygon(
                        np.vstack(
                            [vector_mask_coords_lines, vector_mask_coords_samples]
                        ).T
                    )
                ]
            else:  # multipolygon
                vector_mask_coords = []  # to store polygons in image coordinates
                for iio, onepoly in enumerate(vector_mask_ll.geoms):
                    lons_ma, lats_ma = onepoly.exterior.xy
                    lons = np.array([hh for hh in lons_ma])
                    lats = np.array([hh for hh in lats_ma])
                    vector_mask_coords_lines, vector_mask_coords_samples = (
                        self.ll2coords_SLC(lons, lats)
                    )
                    vector_mask_coords.append(
                        Polygon(
                            np.vstack(
                                [vector_mask_coords_lines, vector_mask_coords_samples]
                            ).T
                        )
                    )

            # shape of the returned chunk
            out_shape = (line.size, sample.size)

            # transform * (x, y) -> (line, sample)
            # (where (x, y) are index in out_shape)
            # Affine.permutation() is used because (line, sample) is transposed from geographic
            # this transform Affine seems to be sufficient approx for SLC -> curvilinear could be even better?
            transform = (
                Affine.translation(*chunk_coords[0])
                * Affine.scale(*[np.unique(np.diff(c))[0] for c in [line, sample]])
                * Affine.permutation()
            )

            raster_mask = rasterio.features.rasterize(
                vector_mask_coords,
                out_shape=out_shape,
                all_touched=False,
                transform=transform,
            )

            return raster_mask

        # all_bursts = self.datatree['bursts']
        all_bursts = self.get_bursts_polygons(only_valid_location=False)
        da_dict = {}  # dictionnary to store the DataArray of each mask and each burst
        for burst_id in range(len(all_bursts["geometry_image"])):
            a_burst_bbox = all_bursts["geometry_image"].iloc[burst_id]
            line_index = np.array([int(jj) for jj in a_burst_bbox.exterior.xy[0]])
            sample_index = np.array([int(jj) for jj in a_burst_bbox.exterior.xy[1]])
            logger.debug(
                "line_index : %s %s %s", line_index, line_index.min(), line_index.max()
            )
            logger.debug("dataset shape %s", self.dataset.digital_number.shape)
            a_burst_subset = self.dataset.isel(
                {
                    "line": slice(line_index.min(), line_index.max()),
                    "sample": slice(sample_index.min(), sample_index.max()),
                    "pol": 0,
                }
            )
            logger.debug("a_burst_subset %s", a_burst_subset.digital_number.shape)
            # logging.info('burst : %s lines: %s samples: %s',burst_id,a_burst_subset.digital_number.line,a_burst_subset.digital_number.sample)
            da_tmpl = xr.DataArray(
                dask.array.empty_like(
                    np.empty(
                        (
                            len(a_burst_subset.digital_number.line),
                            len(a_burst_subset.digital_number.sample),
                        )
                    ),
                    dtype=np.int8,
                    name="empty_var_tmpl-%s" % dask.base.tokenize(self.sar_meta.name),
                ),
                dims=("line", "sample"),
                coords={
                    "line": a_burst_subset.digital_number.line,
                    "sample": a_burst_subset.digital_number.sample,
                    # 'line_time': line_time.astype(float),
                },
            )

            for mask in self.sar_meta.mask_names:
                logger.debug("mask: %s", mask)
                if lazy_loading:
                    da_mask = map_blocks_coords(
                        da_tmpl, _rasterize_mask_by_chunks, func_kwargs={"mask": mask}
                    )
                else:
                    tmpmask_val = _rasterize_mask_by_chunks(
                        a_burst_subset.digital_number.line,
                        a_burst_subset.digital_number.sample,
                        mask=mask,
                    )
                    da_mask = xr.DataArray(
                        tmpmask_val,
                        dims=("line", "sample"),
                        coords={
                            "line": a_burst_subset.digital_number.line,
                            "sample": a_burst_subset.digital_number.sample,
                        },
                    )
                name = "%s_maskv2" % mask
                da_mask.attrs["history"] = yaml.safe_dump(
                    {name: self.sar_meta.get_mask(mask, describe=True)}
                )
                da_mask.attrs["meaning"] = "0: ocean , 1: land"
                da_mask = da_mask.fillna(0)  # zero -> ocean
                da_mask = da_mask.astype(np.int8)
                logger.debug("%s -> %s", mask, da_mask.attrs["history"])
                # da_list.append(da_mask.to_dataset(name=name))
                if mask not in da_dict:
                    da_dict[mask] = [da_mask.to_dataset(name=name)]
                else:
                    da_dict[mask].append(da_mask.to_dataset(name=name))
            logger.debug("da_dict[mask] = %s %s", mask, da_dict[mask])
        # merge with existing dataset
        all_masks = []
        for kk in da_dict:
            logger.debug("da_dict[kk] %s", da_dict[kk])
            complet_mask = xr.combine_by_coords(da_dict[kk])
            all_masks.append(complet_mask)
        tmpds = self.datatree["measurement"].to_dataset()
        tmpds.attrs = self.dataset.attrs
        tmpmerged = xr.merge([tmpds] + all_masks)
        tmpmerged = tmpmerged.drop("land_mask")
        logger.debug("rename land_maskv2 -> land_mask")
        tmpmerged = tmpmerged.rename({"land_maskv2": "land_mask"})
        tmpmerged.attrs["land_mask_computed_by_burst"] = True
        self.dataset = tmpmerged
        self.datatree["measurement"] = self.datatree["measurement"].assign(tmpmerged)
        self.datatree["measurement"].attrs = tmpmerged.attrs

    @property
    def footprint(self):
        """alias for `xsar.BaseDataset.geometry`"""
        return self.geometry

    @timing
    def map_raster(self, raster_ds):
        """
        Map a raster onto xsar grid

        Parameters
        ----------
        raster_ds: xarray.Dataset or xarray.DataArray
            The dataset we want to project onto xsar grid. The `raster_ds.rio` accessor must be valid.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The projected dataset, with 'line' and 'sample' coordinate (same size as xsar dataset), and with valid `.rio` accessor.


        """
        if not raster_ds.rio.crs.is_geographic:
            raster_ds = raster_ds.rio.reproject(4326)

        if self.sar_meta.cross_antemeridian:
            raise NotImplementedError("Antimeridian crossing not yet checked")

        # get lon/lat box for xsar dataset
        lon1, lat1, lon2, lat2 = self.sar_meta.footprint.exterior.bounds
        lon_range = [lon1, lon2]
        lat_range = [lat1, lat2]

        # ensure dims ordering
        raster_ds = raster_ds.transpose("y", "x")

        # ensure coords are increasing ( for RectBiVariateSpline )
        for coord in ["x", "y"]:
            if raster_ds[coord].values[-1] < raster_ds[coord].values[0]:
                raster_ds = raster_ds.reindex({coord: raster_ds[coord][::-1]})

        # from lon/lat box in xsar dataset, get the corresponding box in raster_ds (by index)
        """
        ilon_range = [
            np.searchsorted(raster_ds.x.values, lon_range[0]),
            np.searchsorted(raster_ds.x.values, lon_range[1])
        ]
        ilat_range = [
            np.searchsorted(raster_ds.y.values, lat_range[0]),
            np.searchsorted(raster_ds.y.values, lat_range[1])
        ]
        """  # for incomplete raster (not global like hwrf)
        ilon_range = [
            np.max([1, np.searchsorted(raster_ds.x.values, lon_range[0])]),
            np.min(
                [np.searchsorted(raster_ds.x.values, lon_range[1]), raster_ds.x.size]
            ),
        ]
        ilat_range = [
            np.max([1, np.searchsorted(raster_ds.y.values, lat_range[0])]),
            np.min(
                [np.searchsorted(raster_ds.y.values, lat_range[1]), raster_ds.y.size]
            ),
        ]

        # enlarge the raster selection range, for correct interpolation
        ilon_range, ilat_range = [
            [rg[0] - 1, rg[1] + 1] for rg in (ilon_range, ilat_range)
        ]

        # select the xsar box in the raster
        raster_ds = raster_ds.isel(x=slice(*ilon_range), y=slice(*ilat_range))

        # upscale coordinates, in original projection
        # 1D array of lons/lats, trying to have same spacing as dataset (if not to high)
        num = min((self._dataset.sample.size + self._dataset.line.size) // 2, 1000)
        lons = np.linspace(*lon_range, num=num)
        lats = np.linspace(*lat_range, num=num)

        name = None
        if isinstance(raster_ds, xr.DataArray):
            # convert to temporary dataset
            name = raster_ds.name or "_tmp_name"
            raster_ds = raster_ds.to_dataset(name=name)

        mapped_ds_list = []
        for var in raster_ds:
            raster_da = raster_ds[var].chunk(raster_ds[var].shape)
            # upscale in original projection using interpolation
            # in most cases, RectBiVariateSpline give better results, but can't handle Nans
            if np.any(np.isnan(raster_da)):
                upscaled_da = raster_da.interp(x=lons, y=lats)
            else:
                upscaled_da = map_blocks_coords(
                    xr.DataArray(dims=["y", "x"], coords={"x": lons, "y": lats}).chunk(
                        1000
                    ),
                    RectBivariateSpline(
                        raster_da.y.values,
                        raster_da.x.values,
                        raster_da.values,
                        kx=3,
                        ky=3,
                    ),
                )
            upscaled_da.name = var
            # interp upscaled_da on sar grid
            mapped_ds_list.append(
                upscaled_da.interp(
                    x=self._dataset.longitude, y=self._dataset.latitude
                ).drop_vars(["x", "y"])
            )
        mapped_ds = xr.merge(mapped_ds_list)

        if name is not None:
            # convert back to dataArray
            mapped_ds = mapped_ds[name]
            if name == "_tmp_name":
                mapped_ds.name = None
        return self._set_rio(mapped_ds)

    @timing
    def _load_rasters_vars(self):
        # load and map variables from rasterfile (like ecmwf) on dataset
        if self.sar_meta.rasters.empty:
            return None
        else:
            logger.warning("Raster variable are experimental")

        if self.sar_meta.cross_antemeridian:
            raise NotImplementedError("Antimeridian crossing not yet checked")

        # will contain xr.DataArray to merge
        da_var_list = []

        for name, infos in self.sar_meta.rasters.iterrows():
            # read the raster file using helpers functions
            read_function = infos["read_function"]
            get_function = infos["get_function"]
            resource = infos["resource"]

            kwargs_get = {
                "date": datetime.strptime(
                    self.sar_meta.start_date, "%Y-%m-%d %H:%M:%S.%f"
                ),
                "footprint": self.sar_meta.footprint,
            }

            logger.debug(
                'adding raster "%s" from resource "%s"' % (name, str(resource))
            )
            if get_function is not None:
                try:
                    resource_dec = get_function(resource, **kwargs_get)
                except TypeError:
                    resource_dec = get_function(resource)

            kwargs_read = {"date": np.datetime64(resource_dec[0])}

            if read_function is None:
                raster_ds = xr.open_dataset(resource_dec[1], chunk=1000)
            else:
                # read_function should return a chunked dataset (so it's fast)
                raster_ds = read_function(resource_dec[1], **kwargs_read)

            # add globals raster attrs to globals dataset attrs
            hist_res = {"resource": resource_dec[1]}
            if get_function is not None:
                hist_res.update({"resource_decoded": resource_dec[1]})

            reprojected_ds = self.map_raster(raster_ds).rename(
                {v: "%s_%s" % (name, v) for v in raster_ds}
            )

            for v in reprojected_ds:
                reprojected_ds[v].attrs["history"] = yaml.safe_dump({v: hist_res})

            da_var_list.append(reprojected_ds)
        return xr.merge(da_var_list)

    def _get_lut(self, var_name):
        """
        Get lut for `var_name`

        Parameters
        ----------
        var_name: str

        Returns
        -------
        xarray.DataArray
            lut for `var_name`
        """
        try:
            lut_name = self._map_var_lut[var_name]
        except KeyError:
            raise ValueError("can't find lut name for var '%s'" % var_name)
        try:
            lut = self._luts[lut_name]
        except KeyError:
            raise ValueError(
                "can't find lut from name '%s' for variable '%s' "
                % (lut_name, var_name)
            )
        return lut
