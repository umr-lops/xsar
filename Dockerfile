FROM ubuntu:20.04

# BASIC USAGE
#
# user must be in docker group (if not prefix sudo when you used docker)
# sudo gpasswd -a $USER docker
# newgrp docker
# logout to your session 
# 
# docker build -f Dockerfile . -t xsar_image
# docker run -i -t -d --name='container_xsar_image' xsar_image:latest /bin/bash
# docker attach container_xsar_image

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

## for apt to be noninteractive
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NONINTERACTIVE_SEEN=true

RUN echo Install Basic Command
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 git

RUN echo Install Conda
# miniconda install
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# create an empty conda env that is pre-activated
RUN conda create -n xsar
RUN echo "conda activate xsar" >> ~/.bashrc

RUN conda install -c conda-forge  'python<3.9' gdal pyarrow rasterio mkl pyarrow 'llvmlite<0.32' dask distributed

RUN pip install git+https://gitlab.ifremer.fr/sarlib/saroumane.git

# image is now created
