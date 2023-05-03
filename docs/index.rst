##################################################
xsar: efficient level 1 sar reader for xarray/dask
##################################################

**xsar** is a distributed level 1 SAR file reader designed to write efficient distributed processing algorhitm with `xarray`_ and `dask`_.

It currently handles Level-1 Sentinel-1 and Radarsat-2 data in `SAFE format`_, as found on `scihub`_ or `PEPS`_.

**xsar** is as simple to use as the well known `xarray.open_dataset`_ : simply give the dataset path, and :meth:`xsar.open_dataset` will return an `datatree.DataTree`:

.. jupyter-execute:: examples/intro.py


Documentation
-------------

Overview
........

    **xsar** rely on `xarray.open_rasterio`_, `rasterio`_ and `GDAL`_ to read *digital_number* from SAFE
    product to return an `xarray.Dataset`_ object with `dask`_ chunks.

    Luts are decoded from xml files and applied to *digital_number*, following official `ESA thermal denoising document`_ and `ESA Sentinel-1 Product Specification`_.

    So end user can directly use for example *sigma0* variable, this is the denoised sigma0 computed from *digital_number* and by applying relevants luts.

    Because **xsar** rely on `dask`_, it have a small memory footprint: variables are read from file and computed only if needed.


    :meth:`xsar.open_dataset` is very close to `xarray.open_dataset`_, but in the followings examples, you will find some additional keywords and classes that allow to:

    * `open a dataset at lower resolution`_
    * `convert lon/lat to dataset coordinates`_

Examples
........

.. note::
    With `recommended installation`_ you will be able to download and execute those examples in `jupyter notebook`_.

    Those examples will automatically download test data from https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata/

    Those file are not official ones: they are resampled to a lower resoltion and compressed to avoid big network transfert and disk usage.

    Don't use them for real science !

* :doc:`examples/xsar`

* :doc:`examples/xsar_advanced`

* :doc:`examples/radarsat2`

* :doc:`examples/rcm`

* :doc:`examples/projections`

* :doc:`examples/mask`

* :doc:`examples/xsar_multiple`

* :doc:`examples/xsar_batch_datarmor`

* :doc:`examples/xsar_tops_slc`

UML Description
...............

* :doc:`uml`

Reference
.........

* :doc:`basic_api`

Get in touch
------------

- Report bugs, suggest features or view the source code `on github`_.

----------------------------------------------

Last documentation build: |today|

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   installing


.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/xsar
   examples/xsar_advanced
   examples/radarsat2
   examples/rcm
   examples/projections
   examples/mask
   examples/xsar_multiple
   examples/xsar_batch_datarmor
   examples/xsar_tops_slc

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: UML Description

   uml

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   basic_api

.. _on github: https://github.com/umr-lops/xsar
.. _xarray: http://xarray.pydata.org
.. _dask: http://dask.org
.. _xarray.open_dataset: http://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html
.. _scihub: https://scihub.copernicus.eu/
.. _PEPS: https://peps.cnes.fr/rocket/
.. _rasterio: https://rasterio.readthedocs.io/en/latest/
.. _GDAL: https://gdal.org/
.. _xarray.open_rasterio: http://xarray.pydata.org/en/stable/generated/xarray.open_rasterio.html
.. _ESA thermal denoising document: https://sentinel.esa.int/documents/247904/2142675/Thermal-Denoising-of-Products-Generated-by-Sentinel-1-IPF
.. _ESA Sentinel-1 Product Specification: https://earth.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification
.. _xarray.Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html
.. _open a dataset at lower resolution: examples/xsar_advanced.ipynb#Open-a-dataset-at-lower-resolution
.. _convert lon/lat to dataset coordinates: examples/xsar_advanced.ipynb#Convert-(lon,lat)-to-(line,-sample)
.. _`recommended installation`: installing.rst#recommended-packages
.. _SAFE format: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/data-formats
.. _jupyter notebook: https://jupyter.readthedocs.io/en/latest/running.html#running