# CI / CD XSAR

Using Github action workflow.

For more information of github axction workflow you can read the official [tutorial](https://docs.github.com/en/actions).

We have 4 workflows for testing installation on multi OS and multi version of python.

## Workflow 1: Check Gdal 3.3 is available

Gdal is a depencies package of xsar.

We are testing if gdal version 3.3 is available in the official conda repository

## Workflow 2,3,4: Testing Xsar Installation 

Testing official xsar [installation](https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsar/installing.html) on ubuntu, macOS and windows

The jobs contains 5 steps:

- Setup conda and create environment : using a community github action package [conda-incubator/setup-miniconda@v2](https://github.com/marketplace/actions/setup-miniconda)
- Install xsar dependencies
- Check xsar environment: you can see in a debug job the version of conda, python, rasterio, gdal, cartopy and dask
- Install xsar
- Testing xsar : run the script test `test/test_xsar.py`

### MacOS particularity 

- Github action used macos 10.15 Cattalina for testing not the lastest version 11.X Bigsur
- In Python 3.6 the job stuck so we do not test in python. 3.6
- When Install xsar dependencies for python < 3.9, you must install tbb package 

### Windows particularity

- Github action used windows server 2019 for testing not Windows 10
- Windows have not rasterio 1.2.6 in python < 3.7
- When Install xsar dependencies, you must install fiona package 
