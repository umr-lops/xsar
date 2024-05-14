![Install test](https://github.com/umr-lops/xsar/actions/workflows/install-test.yml/badge.svg)
# xsar

Synthetic Aperture Radar (SAR) Level-1 GRD python mapper for efficient xarray/dask based processing

This python library allow to apply different operation on SAR images such as:
 - calibration
 - de-noising
 - re-sampling

The library is working regardless it is a **Sentinel-1**, a **RadarSAT-2** or a **RCM** product.

The library is providing variables such as `longitude` , `latitude`, `incidence_angle` or `sigma0` at native product resolution or coarser resolution.

The library perform resampling that are suitable for GRD (i.e. ground projected) SAR images. The same method is used for WV SLC, and one can consider the approximation still valid because the WV image is only 20 km X 20 km.

But for TOPS (IW or EW) SLC products we recommend to use [xsarslc](https://github.com/umr-lops/xsar_slc.git)

# Install

## Conda

1) Install `xsar` (without the readers)

For a faster installation and less conflicts between packages, it is better
to make the installation with `micromamba`

```bash
conda install -c conda-forge mamba
```

2) install `xsar` (without the readers)

```bash
micromamba install -c conda-forge xsar
```
3) Add optional dependencies

- Add use of Radarsat-2 :

```bash
micromamba install -c conda-forge xradarsat2
```

- Add use of RCM (RadarSat Constellation Mission)

```bash
pip install xarray-safe-rcm
```

- Add use of Sentinel-1

```bash
micromamba install -c conda-forge xarray-safe-s1
```

## Pypi

1) install `xsar` (this will only allow to use Sentinel-1)

```bash
pip install xsar
```
2) install `xsar` with optional dependencies (to use Radarsat-2, RCM...)

- install `xsar` including Sentinel-1 :

```bash
pip install xsar[S1]
```

- install `xsar` including Radarsat-2 :

```bash
pip install xsar[RS2]
```

- install `xsar` including RCM :

```bash
pip install xsar[RCM]
```

- install `xsar` including multiple readers (here Radarsat-2 and RCM):

```bash
pip install xsar[RS2,RCM]
```


```python
>>> import xsar
>>> import xarray
>>> xarray.open_dataset('S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_Z010.SAFE')

<xarray.Dataset>
Dimensions:               (atrack: 16778, pol: 2, xtrack: 25187)
Coordinates:
  * atrack                (atrack) int64 0 1 2 3 4 ... 16774 16775 16776 16777
  * pol                   (pol) object 'VV' 'VH'
  * xtrack                (xtrack) int64 0 1 2 3 4 ... 25183 25184 25185 25186
    spatial_ref           int64 ...
Data variables: (12/19)
    time                  (atrack) timedelta64[ns] ...
    digital_number        (pol, atrack, xtrack) uint16 ...
    land_mask             (atrack, xtrack) int8 ...
    ground_heading        (atrack, xtrack) float32 ...
    sigma0_raw            (pol, atrack, xtrack) float64 ...
    nesz                  (pol, atrack, xtrack) float64 ...
    ...                    ...
    longitude             (atrack, xtrack) float64 ...
    latitude              (atrack, xtrack) float64 ...
    velocity              (atrack) float64 ...
    range_ground_spacing  (xtrack) float64 ...
    sigma0                (pol, atrack, xtrack) float64 ...
    gamma0                (pol, atrack, xtrack) float64 ...
Attributes: (12/14)
    ipf:               2.84
    platform:          SENTINEL-1A
    swath:             IW
    product:           GRDH
    pols:              VV VH
    name:              SENTINEL1_DS:/home/oarcher/SAFE/S1A_IW_GRDH_1SDV_20170...
    ...                ...
    footprint:         POLYGON ((-67.84221143971432 20.72564283093837, -70.22...
    coverage:          170km * 251km (atrack * xtrack )
    pixel_atrack_m:    10.152619433217325
    pixel_xtrack_m:    9.986179379582332
    orbit_pass:        Descending
    platform_heading:  -167.7668824808032

```



# More information

For more install options and to use xsar, see [documentation](https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsar/)
