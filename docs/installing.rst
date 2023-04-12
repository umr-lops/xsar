.. _installing:

************
Installation
************

`xsar` relies on gdal_ and other shared libs, that are not provided by pip.
So insallation in a conda_ environement is recommended.


conda install
#############

1) Install xsar (this will only permit to use Sentinel1)

.. code-block::

    conda create -n xsar
    conda activate xsar
    conda install -c conda-forge xsar


2) Add optional dependencies

- Add use of Radarsat2 :

.. code-block::

    conda install -c conda-forge xradarsat2


- Add use of RCM

.. code-block::

    pip install git+https://github.com/umr-lops/xarray-safe-rcm.git


pip install
###########

1) Install xsar (this will only permit to use Sentinel1)

.. code-block::

    conda create -n xsar
    conda activate xsar
    pip install xsar


2) install xsar with optional dependencies (to use Radarsat2, RCM...)

- Install xsar including Radarsat2 :

.. code-block::

    pip install xsar[RS2]


- Install xsar including RCM :

.. code-block::

    pip install git+https://github.com/umr-lops/xarray-safe-rcm.git
    pip install xsar[RCM]


- Install xsar including Radarsat2 and RCM:

.. code-block::

    pip install git+https://github.com/umr-lops/xarray-safe-rcm.git
    pip install xsar[RS2,RCM]


recommended packages
....................

Default installation is minimal, and should be used in non-interactive environment.


Xsar can be used in jupyter, with holoviews and geoviews. To install aditional dependancies, run:

.. code-block::

    pip install -r https://raw.githubusercontent.com/umr-lops/xsar/develop/requirements.txt
    pip install git+https://github.com/umr-lops/xsarsea.git


Update xsar to the latest version
#################################

xsar conda package can be quite old:

.. image:: https://anaconda.org/conda-forge/xsar/badges/latest_release_relative_date.svg

To be up to date with the developpement team, it's recommended to update the installation using pip:

.. code-block::

    pip install git+https://github.com/umr-lops/xsar.git



Developement  installation
..........................

.. code-block::

    git clone https://github.com/umr-lops/xsar
    cd xsar
    # this is needed to register git filters
    git config --local include.path ../.gitconfig
    pip install -e .
    pip install -r requirements.txt


.. _conda: https://docs.anaconda.com/anaconda/install/
.. _gdal: https://gdal.org/
.. _xsarsea: https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsarsea