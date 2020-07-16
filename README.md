Code release for the paper "TAASRAD19, a high-resolution weather radar reflectivity dataset for precipitation nowcasting"
---
Franch, G., Maggio, V., Coviello, L. et al. TAASRAD19, a high-resolution weather radar reflectivity dataset for precipitation nowcasting. Sci Data 7, 234 (2020).

https://doi.org/10.1038/s41597-020-0574-8


The code includes the scripts for sequence extraction, a deep learning model for precipitation nowcasting and an online visualization build on TAASRAD19 dataset.
The dataset can be downloaded from the following repositories:
- Radar scans years 2010 - 2016:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3577451.svg)](https://doi.org/10.5281/zenodo.3577451)
- Radar scans years 2017 - 2019:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3591396.svg)](https://doi.org/10.5281/zenodo.3591396)
- Radar sequences years 2010 - 2019:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3591404.svg)](https://doi.org/10.5281/zenodo.3591404)

A NETCDF version of the radar sequences (not needed for the code in this repository) is available here:
- Radar sequences years 2010 - 2019 (NETCDF):
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3866204.svg)](https://doi.org/10.5281/zenodo.3866204)

All the code was developed and tested on Ubuntu 18.04 with python 3.6+.

Create a new virtualenv (for example with `venv`):
```
python3 -m venv .venv 
source .venv/bin/ctivate
```

Install all required packages in the virtualenv:
```
pip install \
   opencv-python PyYAML pandas \
   numba numpy scipy tqdm imageio \
   Pillow jupyterlab h5py umap-learn \
   joblib matplotlib
```

The [nowcasting deep learning model](deep_learning_nowcasting) was tested with mxnet 1.5.1
that can be installed with CPU support:
```
pip install mxnet==1.5.1.post0
```
or for CUDA 10.1 GPUs (see other versions at
[mxnet website](https://mxnet.apache.org/get_started/?version=v1.5.1)) :
```
pip install mxnet-cu101==1.5.1.post0
```
