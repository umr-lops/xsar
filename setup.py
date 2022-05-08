from setuptools import setup, find_packages
import glob

setup(
    name='xsar',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    scripts=glob.glob('src/scripts/*.py'),
    url='https://github.com/umr-lops/xsar',
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
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
        'jinja2<=3.0.3',
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
        'pytz',
        'psutil'
    ],
    entry_points={
        "xarray.backends": ["xsar=xsar.xarray_backends:XsarXarrayBackend"]
    },
    license='MIT',
    author='Olivier Archer, Alexandre Levieux',
    author_email='Olivier.Archer@ifremer.fr, Alexandre.Levieux@gmail.com',
    description='xarray/dask distributed L1 sar file reader'
)
