# Data processing code for TAASRAD19 radar scans

This folder contains the code for generating the HDF5 archive and
the outlier mask from the original radar scans in TAR GZ format.
The data can be downloaded from the following repositories:

RADAR SCANS YEARS 2010 - 2016:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3577451.svg)](https://doi.org/10.5281/zenodo.3577451)


RADAR SCANS YEARS 2017 - 2019:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3591396.svg)](https://doi.org/10.5281/zenodo.3591396)

The following steps expect that in `[data_dir]` are present the unzipped files:
1. Original data folders of years 2010-2019 
2. Daily weather descriptions file (i.e. `daily_weather_report.csv`)

```
DATA_DIR/   [data_dir]
  |
  +-- 2010/
  |     |
  |     +-- 20100601.tar
  |     +-- ...
  |     +-- 20101231.tar
  +-- 2011/
  |     |
  |     +-- ...
  +-- ...
  +-- daily_weather_report.csv
```

## Step-by-step HDF5 archive generation instructions:

##### Weak labels extraction from daily descriptions:
`python data_processing/01_extract_weak_labels.py [data_dir]`

##### Extracts rainy sequences in HDF5 format from TAR GZ archives:
Modify the `settings.py` file to change the parameters for the HDF5 generation
`python data_processing/02_extract_sequences.py [data_dir]`

##### Radar outlier pixels mask computation (computed on sequences):
`python data_processing/03_generate_mask.py [data_dir]`

##### Generate a single NPZ file from the whole HDF5 archive, downscaled to 64x64 pixel resolution (for UMAP):
`python data_processing/04_generate_single_npz_resize.py [data_dir]`
