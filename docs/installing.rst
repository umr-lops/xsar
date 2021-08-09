.. _installing:

************
Installation
************

`xsar` relies on gdal_ and other shared libs, that are not provided by pip.
So insallation in a conda_ environement is recommended.


conda setup
###########

First, create and activate a conda environment:

.. literalinclude:: scripts/conda_create_activate
    :language: shell

You can either install recommended packages or minimal packages.

.. note::
    for macOS with python<3.9 you must install tbb package with conda install

.. note::
    for windows you must install fiona package with conda install

recommended packages
....................

.. note::
    by using recommended install, you will be able to download examples and run them 'as is'.


.. literalinclude:: scripts/conda_install_recommended
    :language: shell


minimal install
...............

.. literalinclude:: scripts/conda_install_minimal
    :language: shell

Install xsar
############

Once conda environment is created and activated, **xsar** can be installed by `pip` for a normal user, or with `git clone` for a developper.

As a normal user
................

.. literalinclude:: scripts/pip_install
    :language: shell

for development installation
.............................

.. literalinclude:: scripts/git_install
    :language: shell


--------------------------------------

.. note::
    While you are here, why not install also `xsarsea`_ ?

    .. code-block:: shell

        pip install git+https://gitlab.ifremer.fr/sarlib/xsarsea.git

Update xsar
###########

To update xsar installation, just rerun `pip install`, in your already activated conda environment.

.. literalinclude:: scripts/pip_install
    :language: shell

.. note::
    if you have some problems after updating **xsar**, you might have to update also your conda environment:

    .. literalinclude:: scripts/conda_install_minimal
        :language: shell


.. _conda: https://docs.anaconda.com/anaconda/install/
.. _gdal: https://gdal.org/
.. _xsarsea: https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsarsea