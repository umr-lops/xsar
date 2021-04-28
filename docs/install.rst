Installation
============


`xsar` rely on gdal shared libs and zlib, that are not provided by pip.
So insallation in a conda_ environement is recommended.


.. code-block:: shell

    conda create -n xsar python=3
    conda activate xsar
    conda install gdal zlib
    pip install git+https://gitlab.ifremer.fr/sarlib/xsar.git


or , for developement installation:

.. code-block:: shell

    conda create -n xsar python=3
    conda install gdal zlib graphviz
    git clone https://gitlab.ifremer.fr/sarlib/xsar.git
    cd xsar
    pip install -r requirements.txt
    pip install -e .


.. _conda: https://docs.anaconda.com/anaconda/install/
