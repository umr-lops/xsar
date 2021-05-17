from setuptools import setup, find_packages
import glob

setup(
    name='xsar',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    scripts=glob.glob('src/scripts/*.py'),
    url='https://github.com/oarcher/xsar',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=[
        'GDAL',
        'dask[array]',
        'dask[distributed]',
        'xarray',
        'affine',
        'rasterio',
        'cartopy',
        'fiona',
        'pyproj',
        'jinja2',
        'lxml',
        'numpy',
        'scipy',
        'shapely',
        'jmespath',
        'geopandas',
        'more_itertools',
        'importlib-resources',
        'pyyaml',
        'fsspec',
        'aiohttp',
    ],
    entry_points={
        "xarray.backends": ["xsar=xsar.xarray_backends:XsarXarrayBackend"]
    },
    license='MIT',
    author='Olivier Archer, Alexandre Levieux',
    author_email='Olivier.Archer@ifremer.fr, Alexandre.Levieux@ifremer.fr',
    description='xarray/dask distributed L1 sar file reader'
)
