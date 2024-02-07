from setuptools import setup, find_packages
import glob

setup(
    name='xsar',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    scripts=glob.glob('src/scripts/*.py'),
    url='https://github.com/umr-lops/xsar',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=[
        'GDAL',
        'dask[array]',
        'distributed',
        'xarray',
        'affine',
        'rasterio',
        'cartopy',
        'fiona',
        'pyproj',
        'xarray-datatree>=0.0.9',
        'numpy',
        'scipy',
        'shapely',
        'geopandas',
        'fsspec',
        'aiohttp',
        'pytz',
        'psutil',
        'jinja2',
        'rioxarray',
        'lxml'
    ],
    extras_require={
        "RS2": ["xradarsat2"],
        "RCM": ["xarray-safe-rcm"],
        "S1": ["xarray-safe-s1"]
    },
    entry_points={
        "xarray.backends": ["xsar=xsar.xarray_backends:XsarXarrayBackend"]
    },
    license='MIT',
    author='Olivier Archer, Alexandre Levieux, Antoine Grouazel',
    author_email='Olivier.Archer@ifremer.fr, Alexandre.Levieux@gmail.com, Antoine.Grouazel@ifremer.fr',
    description='xarray L1 SAR mapper',
    summary='Python xarray library to use Level-1 GRD SAR products',
    long_description_content_type='text/x-rst',
    long_description = 'Python xarray library to use Level-1 GRD Synthetic Aperture Radar products'
)
