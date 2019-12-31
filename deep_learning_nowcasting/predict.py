import cv2
from nowcasting.config import cfg
from nowcasting.utils import logging_config, parse_ctx
import os
import argparse
import logging
import mxnet as mx
from datetime import timedelta, datetime
from taasrad19 import HDFIterator
import h5py
import pandas as pd
import numpy as np
import random
from nowcasting.radar_factory import NowcastingPredictor

random.seed(123)
mx.random.seed(9302)
np.random.seed(9212)


class DateWriter(object):
    """
    Class used to save input/ground truth/prediction results as NPZ
    """
    def __init__(self, path: str, delta: timedelta = timedelta(minutes=5), fname_format="%Y%m%d%H%M_pred_{sl}.npz"):
        self.path = path
        self.delta = delta
        self.fname_format = fname_format
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.chunk = {}
        self.ts_start = None

    def _save(self):
        path = os.path.join(self.path, self.ts_start.strftime(self.fname_format.format(sl=len(self.chunk.keys()))))
        np.savez_compressed(path, **self.chunk)

    def push(self, arr: np.array, timestamp: datetime = None):
        chunk_len = len(self.chunk.keys())
        if timestamp is None:
            timestamp = self.ts_start + self.delta * chunk_len

        if chunk_len == 0:
            self.ts_start = timestamp
            self.chunk[timestamp.strftime("%Y%m%d%H%M")] = arr
        # if the gap is more than timedelta or it's a new day, save npz and restart
        elif timestamp != self.ts_start + self.delta * chunk_len or \
             timestamp.date() != (self.ts_start + self.delta * (chunk_len - 1)).date():
            self._save()
            self.ts_start = timestamp
            self.chunk = {timestamp.strftime("%Y%m%d%H%M"): arr}
        else:
            self.chunk[timestamp.strftime("%Y%m%d%H%M")] = arr

    def close(self):
        self._save()


def radar_circle_mask(img_size=480):
    radius = img_size / 2
    X, Y = np.ogrid[:img_size, :img_size]
    mask = np.sqrt(np.square(X - img_size / 2) + np.square(
        Y - img_size / 2)) <= radius
    return mask


def main(args):
    save_dir = args.save_dir
    batch_size = args.batch_size
    logging_config(folder=save_dir, name="predict")
    predictor = NowcastingPredictor(args.model_dir, args.model_iter, args.model_cfg, batch_size, args.ctx)
    pred_saver = DateWriter(save_dir)
    in_saver = DateWriter(save_dir, fname_format="%Y%m%d%H%M_in_{sl}.npz")
    gt_saver = DateWriter(save_dir, fname_format="%Y%m%d%H%M_gt_{sl}.npz")
    metadata_file = os.path.join(args.data_dir, 'hdf_metadata.csv') if args.data_csv is None else args.data_csv

    metadata = pd.read_csv(metadata_file, index_col='id')
    metadata['start_datetime'] = pd.to_datetime(metadata['start_datetime'])
    metadata['end_datetime'] = pd.to_datetime(metadata['end_datetime'])
    if args.date_start is not None:
        metadata = metadata.loc[metadata['start_datetime'] >= args.date_start]
    if args.date_end is not None:
        metadata = metadata.loc[metadata['start_datetime'] < args.date_end]
    all_data = h5py.File(os.path.join(args.data_dir, 'hdf_archives', 'all_data.hdf5'), 'r', libver='latest')
    outlier_mask = cv2.imread(os.path.join(args.data_dir, 'mask.png'), 0)

    radar_mask = radar_circle_mask()
    batcher = HDFIterator(all_data, metadata, outlier_mask, batch_size=batch_size,
                          shuffle=False, filter_threshold=0, sort_by='id', ascending=True, return_mask=False)

    mask_out = np.tile(radar_mask, (cfg.MODEL.OUT_LEN, batch_size, 1, 1))
    mask_in = np.tile(radar_mask, (cfg.MODEL.IN_LEN, batch_size, 1, 1))
    # index_df = pd.DataFrame(columns=['filename', 'index', 'timestamp'])

    j = 0
    while True:
        try:
            frame_dat, datetime_clip = next(batcher)
            # datetime_clip = datetime_clip[0]
            logging.info("Iteration {}: [{}] {}x{} clips".format(j, ", ".join([str(x) for x in datetime_clip]),
                                                                 len(datetime_clip), len(frame_dat)))
        except StopIteration:
            break

        # (seq_len, bs, ch, h, w)
        in_frame = frame_dat[:cfg.MODEL.IN_LEN, ...]
        out_frame = frame_dat[cfg.MODEL.IN_LEN:, ...]
        pred_frame = predictor.predict(in_frame)

        pred_frame[pred_frame < 0.001] = 0
        out_frame[out_frame < 0.001] = 0
        in_frame[in_frame < 0.001] = 0

        # (seq_len, bs, h, w)
        pred_frame = np.around(np.squeeze(pred_frame), decimals=3).astype(np.float32)
        out_frame = np.around(np.squeeze(out_frame), decimals=3).astype(np.float32)
        in_frame = np.around(np.squeeze(in_frame), decimals=3).astype(np.float32)

        pred_frame[~mask_out] = np.nan
        out_frame[~mask_out] = np.nan
        in_frame[~mask_in] = np.nan

        # (bs, seq_len, h, w)
        pred_frame = pred_frame.swapaxes(0, 1)
        out_frame = out_frame.swapaxes(0, 1)
        in_frame = in_frame.swapaxes(0, 1)

        for i, dc in enumerate(datetime_clip):
            dtclip = dc + timedelta(minutes=5 * (cfg.MODEL.IN_LEN - 1))
            pred_saver.push(pred_frame[i], dtclip)
            in_saver.push(in_frame[i], dtclip)
            gt_saver.push(out_frame[i], dtclip)
            # index_df.loc[j+i] = {'filename': "{:06d}".format(chunk), 'index': (j + i) % split, 'timestamp': dtclip}

        j += batch_size

    pred_saver.close()
    in_saver.close()
    gt_saver.close()
    # index_df.to_csv(os.path.join(save_dir, 'metadata.csv'), index_label='id')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Predictions using TrajGRU model on TAASRAD19 dataset.')
    parser.add_argument('--model_cfg', required=True,
                        help='Model Configuration file (yaml)')
    parser.add_argument('--model_dir', required=True,
                        help='Folder with model weights')
    parser.add_argument('--model_iter', required=True, type=int, default=99999,
                        help='Model itartion to load (default: %(default)s)')
    parser.add_argument('--data_dir', help='The data directory with hdf_archives folder, hdf_metadata.csv and mask.png',
                        required=True, type=str)
    parser.add_argument('--save_dir', default='.',
                        help='Output folder for npz files')
    parser.add_argument('--batch_size', type=int,  default=1,
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--ctx', default='cpu',
                        help='Running Context. (default: %(default)s): `--ctx gpu` '
                             'or `--ctx gpu0,gpu1` for GPU(s). `--ctx cpu` for CPU')
    parser.add_argument('--data_csv', default=None,
                        help='alternate metadata CSV file (default: [data_dir]/hdf_metadata.csv)')
    parser.add_argument('--date_start', help='Start date to filter the sequences (e.g. 2017-01-01)',
                        default=None, type=lambda s: datetime.strptime(s, '%Y-%m-%d'))
    parser.add_argument('--date_end', help='End date to filter the sequence (e.g. 2017-03-01)',
                        default=None, type=lambda s: datetime.strptime(s, '%Y-%m-%d'))

    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)
    logging.info(args)
    return args


if __name__ == "__main__":
    main(parse_args())

"""
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
"""
