Nowcasting Deep Leraning Model for TAASRAD19 dataset
-----

Deep Learning Nowcasting Model for TAASRAD19 dataset.
The model code is a based on the original release from: https://github.com/sxjscience/HKO-7

The precomputed dataset sequences in HDF5 format for training/test can be downloaded from here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3591404.svg)](https://doi.org/10.5281/zenodo.3591404)

The data directory (`data_dir` argument) used for both training ([train.py](train.py))
and prediction ([predict.py](predict.py)) scripts must respect the following structure:

```
DATA_DIR/   [data_dir]
  |
  +-- hdf_archives/ 
  |     |
  |     +-- 20100601.hdf
  |     +-- ...
  |     +-- all_data.hdf
  +-- hdf_metadata.csv
  +-- mask.png
```

#### TRAIN / VALIDATE MODEL
Training the model using the included configuration requires either one GPU with 16GB RAM or two GPU with 8GB RAM.
To train and validate the model on the years 2010 to 2016 with two GPUs run:
```
python train.py \
    --data_dir  /path/to/tassrad19/sequences \
    --save_dir  /trained/model/save/dir \
    --cfg  configurations/trajgru_55_55_33_1_64_1_192_1_192_13_13_9_b4.yml \
    --ctx  gpu0,gpu1 \
    --date_start 2010-06-01 \
    --date_end   2017-01-01
```

Use `python train.py --help` to see all options

#### GENERATE PREDICTIONS
To generate predictions using the pretrained model weights on CPU:
```
python predict.py \
    --model_cfg  pretrained_model/cfg0.yml \
    --model_dir  pretrained_model \
    --model_iter 99999 \
    --save_dir  /path/output \
    --data_dir  /path/to/tassrad19/sequences \
    --date_start 2017-01-01 \
    --date_end   2017-03-01 \
    --ctx cpu \
    --batch_size 4
```

Use `python predict.py --help` to see all options

Predictions are saved in as numpy array in npz format. 
For each TAASRAD19 sequence 3 files are generated in `save_dir`:
input (`in`), ground truth (`gt`) and prediction (`pred`) sequences.

For example for the following TAASRAD19 sequence:
```
start_datetime    2017-01-12 14:20:00
end_datetime      2017-01-12 23:55:00
run_length                        116
```

The generated output is:
```
-201701121440_in_92.npz
|____________|__|__| 
       |      |   |
  start time  |   |
              |   |
            type  |
                  |
         nr. of subsequences
           (5 frames each)

-201701121440_pred_92.npz
|____________|____|__| 
       |       |    |
  start time   |    |
               |    |
             type   |
                    |
           nr. of subsequences
            (20 frames each)

-201701121440_gt_92.npz
|____________|__|__| 
       |      |   |
  start time  |   |
              |   |
            type  |
                  |
         nr. of subsequences
          (20 frames each)
```