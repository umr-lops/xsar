Introduction
============

Python Sar Library with L1 Reader for efficent xarray/dask based processor

.. code-block:: python

    import xsar
    safe_file = 'S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_992F.SAFE'
    ds = xsar.open_dataset(safe_file)

`ds` is a `dask.DataSet`, with very little memory footprint:

.. code-block:: text

    Out[1]:
    <xarray.Dataset>
    Dimensions:         (atrack: 16778, pol: 2, xtrack: 25187)
    Coordinates:
      * pol             (pol) object 'VV' 'VH'
      * atrack          (atrack) float64 0.5 1.5 2.5 ... 1.678e+04 1.678e+04
      * xtrack          (xtrack) float64 0.5 1.5 2.5 ... 2.519e+04 2.519e+04
    Data variables:
        digital_number  (pol, xtrack, atrack) uint16 dask.array<chunksize=(2, 1000, 1000), meta=np.ndarray>
        longitude       (xtrack, atrack) float64 dask.array<chunksize=(1000, 1000), meta=np.ndarray>
        latitude        (xtrack, atrack) float64 dask.array<chunksize=(1000, 1000), meta=np.ndarray>
        sigma0          (pol, xtrack, atrack) float64 dask.array<chunksize=(2, 1000, 1000), meta=np.ndarray>
        gamma0          (pol, xtrack, atrack) float64 dask.array<chunksize=(2, 1000, 1000), meta=np.ndarray>
    Attributes:
        ipf_version:     2.84
        swath_type:      IW
        footprint:       POLYGON ((-68.15836299999999 19.215193, -70.514343 19.64...
        product_class:   S
        product_type:    GRD
        mission:         SENTINEL-1
        satellite:       A
        start_date:      2017-09-07 10:30:20.936409
        stop_date:       2017-09-07 10:30:20.936409
        coverage:        251km * 170km (xtrack * atrack )
        pixel_xtrack_m:  10.0
        pixel_atrack_m:  10.2



