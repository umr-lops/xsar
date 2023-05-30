![Install test](https://github.com/umr-lops/xsar/actions/workflows/install-test.yml/badge.svg)
# xsar

Sentinel1 Level 1 python reader for efficient xarray/dask based processor

 

# Install

## Conda

1) Install xsar (without the readers)

For a faster installation and less conflicts between packages, it is better
to make the installation with mamba

```
conda install -c conda-forge mamba
```

2) Install xsar (without the readers)

```
mamba install -c conda-forge xsar
```
3) Add optional dependencies

- Add use of Radarsat2 :

```
mamba install -c conda-forge xradarsat2
```

- Add use of RCM

```
pip install xarray-safe-rcm
```

- Add use of Sentinel1

```
mamba install -c conda-forge xarray-safe-s1
```

## Pypi

1) Install xsar (this will only permit to use Sentinel1)

```
pip install xsar
```
2) install xsar with optional dependencies (to use Radarsat2, RCM...)

- Install xsar including Sentinel1 :

```
pip install xsar[S1]
```

- Install xsar including Radarsat2 :

```
pip install xsar[RS2]
```

- Install xsar including RCM :

```
pip install xsar[RCM]
```

- Install xsar including multiple readers (here Radarsat2 and RCM):

```
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



# More infos

For more install options and to use xsar, see [documentation](https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsar/)

