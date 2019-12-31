import argparse
import random
import mxnet as mx
import mxnet.ndarray as nd
import h5py
import pandas as pd
import os
import numpy as np
import logging
from nowcasting.radar_factory import RadarNowcastingFactory
from nowcasting.config import cfg, cfg_from_file, save_cfg
from nowcasting.my_module import MyModule
from nowcasting.radar_evaluation import RadarEvaluation
from nowcasting.encoder_forecaster import encoder_forecaster_build_networks, train_step, EncoderForecasterStates,\
    load_encoder_forecaster_params
from nowcasting.utils import parse_ctx, logging_config
from taasrad19 import HDFIterator, infinite_batcher
from datetime import timedelta, datetime
import cv2

# Uncomment to try different seeds

# random.seed(12345)
# mx.random.seed(930215)
# np.random.seed(921206)

# random.seed(1234)
# mx.random.seed(93021)
# np.random.seed(92120)

random.seed(123)
mx.random.seed(9302)
np.random.seed(9212)


def frame_skip_reshape(dat, frame_skip):
    """Reshape (seq_len, B, C, H, W) to (seq_len // frame_skip, B * frame_skip, C, H, W)
    
    Parameters
    ----------
    dat : np.ndarray
    frame_skip : int

    Returns
    -------
    ret : np.ndarray
    """
    seq_len, B, C, H, W = dat.shape
    assert seq_len % frame_skip == 0
    ret = dat.reshape((seq_len // frame_skip, frame_skip, B, -1)).transpose((0, 2, 1, 3))\
             .reshape((seq_len // frame_skip, B * frame_skip, C, H, W))
    return ret


def frame_skip_reshape_back(dat, frame_skip):
    """Reshape (seq_len, B, C, H, W) to (seq_len * frame_skip, B // frame_skip, C, H, W)

    It's the reverse operation of frame_skip_reshape
    Parameters
    ----------
    dat : np.ndarray
    frame_skip : int

    Returns
    -------
    ret : np.ndarray
    """
    seq_len, B, C, H, W = dat.shape
    assert B % frame_skip == 0
    ret = dat.reshape((seq_len, B // frame_skip, frame_skip, -1)).transpose((0, 2, 1, 3))\
             .reshape((seq_len * frame_skip, B // frame_skip, C, H, W))
    return ret


def run_benchmark(model_factory, context, encoder_net, forecaster_net, batcher,
                  sample_num=1, finetune=False, mode="fixed", save_dir="hko7_rnn"):

    """Run the HKO7 Benchmark given the training sequences
    
    Parameters
    ----------
    model_factory :
    context : mx.ctx
    encoder_net : MyModule
    forecaster_net : MyModule
    sample_num : int
    finetune : bool
    batcher : HDFIterator
    mode : str
    save_dir : str
    gif_dir : str

    Returns
    -------

    """
    logging.info("Begin Evaluation, mode=%s, finetune=%s, sample_num=%d,"
                 " results will be saved to %s" % (mode, str(finetune), sample_num, save_dir))
    states = EncoderForecasterStates(factory=model_factory, ctx=context)
    evaluator = RadarEvaluation(seq_len=cfg.MODEL.OUT_LEN, use_central=False)
    while True:

        states.reset_all()
        try:
            frame_dat, datetime_clip, masks = next(batcher)
            datetime_clip = datetime_clip[0]
        except StopIteration:
            break

        in_frame = frame_dat[:cfg.MODEL.IN_LEN, ...]
        in_masks = masks[:cfg.MODEL.IN_LEN, ...]
        in_datetime_clips = [datetime_clip + (timedelta(minutes=5 * i)) for i in range(cfg.MODEL.IN_LEN)]

        out_frame = frame_dat[cfg.MODEL.IN_LEN:, ...]
        out_masks = masks[cfg.MODEL.IN_LEN:, ...]
        out_datetime_clips = [in_datetime_clips[-1] + (timedelta(minutes=5 * (i+1))) for i in range(cfg.MODEL.OUT_LEN)]

        in_frame_nd = nd.array(in_frame, ctx=context)
        encoder_net.forward(is_train=False,data_batch=mx.io.DataBatch(data=[in_frame_nd] + states.get_encoder_states()))
        outputs = encoder_net.get_outputs()
        states.update(states_nd=outputs)

        forecaster_net.forward(is_train=False, data_batch=mx.io.DataBatch(data=states.get_forecaster_state()))
        pred_nd = forecaster_net.get_outputs()
        pred_nd = pred_nd[0]
        pred_nd = nd.clip(pred_nd, a_min=0, a_max=1)

        pred_frame = pred_nd.asnumpy()

        evaluator.update(gt=out_frame,
                         pred=pred_frame,
                         mask=out_masks[:, 0:1, ...],
                         start_datetimes=[out_datetime_clips])

    evaluator.save(prefix=os.path.join(save_dir, "eval_all"))


def main(args):
    assert cfg.MODEL.FRAME_STACK == 1 and cfg.MODEL.FRAME_SKIP == 1
    base_dir = args.save_dir
    logging_config(folder=base_dir, name="train")
    save_cfg(dir_path=base_dir, source=cfg.MODEL)
    metadata_file = os.path.join(args.data_dir, 'hdf_metadata.csv') if args.data_csv is None else args.data_csv

    all_data = h5py.File(os.path.join(args.data_dir, 'hdf_archives', 'all_data.hdf5'), 'r', libver='latest')
    outlier_mask = cv2.imread(os.path.join(args.data_dir, 'mask.png'), 0)

    metadata = pd.read_csv(metadata_file, index_col='id')
    metadata['start_datetime'] = pd.to_datetime(metadata['start_datetime'])
    metadata['end_datetime'] = pd.to_datetime(metadata['end_datetime'])
    if args.date_start is not None:
        metadata = metadata.loc[metadata['start_datetime'] >= args.date_start]
    if args.date_end is not None:
        metadata = metadata.loc[metadata['start_datetime'] < args.date_end]
    sort_meta = metadata.sample(frac=1)
    split_idx = int(len(sort_meta) * 0.95)
    train_meta = sort_meta.iloc[:split_idx]
    test_meta = sort_meta.iloc[split_idx:]

    logging.info("Initializing data iterator with filter threshold %s" % cfg.HKO.ITERATOR.FILTER_RAINFALL_THRESHOLD)
    train_model_iter = infinite_batcher(all_data, train_meta, outlier_mask, shuffle=False,
                                      filter_threshold=cfg.HKO.ITERATOR.FILTER_RAINFALL_THRESHOLD)

    model_nowcasting = RadarNowcastingFactory(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE // len(args.ctx),
                                                ctx_num=len(args.ctx),
                                                in_seq_len=cfg.MODEL.IN_LEN,
                                                out_seq_len=cfg.MODEL.OUT_LEN)
    model_nowcasting_online = RadarNowcastingFactory(batch_size=1,
                                                       in_seq_len=cfg.MODEL.IN_LEN,
                                                       out_seq_len=cfg.MODEL.OUT_LEN)
    encoder_net, forecaster_net, loss_net = \
        encoder_forecaster_build_networks(
            factory=model_nowcasting,
            context=args.ctx)
    t_encoder_net, t_forecaster_net, t_loss_net = \
        encoder_forecaster_build_networks(
            factory=model_nowcasting_online,
            context=args.ctx[0],
            shared_encoder_net=encoder_net,
            shared_forecaster_net=forecaster_net,
            shared_loss_net=loss_net,
            for_finetune=True)
    encoder_net.summary()
    forecaster_net.summary()
    loss_net.summary()
    # Begin to load the model if load_dir is not empty
    if len(cfg.MODEL.LOAD_DIR) > 0:
        load_encoder_forecaster_params(load_dir=cfg.MODEL.LOAD_DIR, load_iter=cfg.MODEL.LOAD_ITER,
                                       encoder_net=encoder_net, forecaster_net=forecaster_net)
    states = EncoderForecasterStates(factory=model_nowcasting, ctx=args.ctx[0])
    for info in model_nowcasting.init_encoder_state_info:
        assert info["__layout__"].find('N') == 0, "Layout=%s is not supported!" % info["__layout__"]
    for info in model_nowcasting.init_forecaster_state_info:
        assert info["__layout__"].find('N') == 0, "Layout=%s is not supported!" % info["__layout__"]
    test_mode = "online" if cfg.MODEL.TRAIN.TBPTT else "fixed"
    iter_id = 0
    while iter_id < cfg.MODEL.TRAIN.MAX_ITER:
        # sample a random minibatch
        try:
            frame_dat, _, mask_dat = next(train_model_iter)
        except StopIteration:
            break
        else:
            states.reset_all()
            data_nd = mx.nd.array(frame_dat[0:cfg.MODEL.IN_LEN, ...], ctx=args.ctx[0])
            target_nd = mx.nd.array(
                frame_dat[cfg.MODEL.IN_LEN:(cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN), ...],
                ctx=args.ctx[0])
            mask_nd = mx.nd.array(
                mask_dat[cfg.MODEL.IN_LEN:(cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN), ...],
                ctx=args.ctx[0])
            states, _ = train_step(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                                   encoder_net=encoder_net, forecaster_net=forecaster_net,
                                   loss_net=loss_net, init_states=states,
                                   data_nd=data_nd, gt_nd=target_nd, mask_nd=mask_nd,
                                   iter_id=iter_id)
            if (iter_id + 1) % cfg.MODEL.SAVE_ITER == 0:
                encoder_net.save_checkpoint(prefix=os.path.join(base_dir, "encoder_net"), epoch=iter_id)
                forecaster_net.save_checkpoint(prefix=os.path.join(base_dir, "forecaster_net"), epoch=iter_id)
            if (iter_id + 1) % cfg.MODEL.VALID_ITER == 0:
                test_model_iter = HDFIterator(all_data, test_meta, outlier_mask, batch_size=1, shuffle=False,
                                              filter_threshold=cfg.HKO.ITERATOR.FILTER_RAINFALL_THRESHOLD)
                run_benchmark(model_factory=model_nowcasting_online,
                              context=args.ctx[0],
                              encoder_net=t_encoder_net,
                              forecaster_net=t_forecaster_net,
                              save_dir=os.path.join(base_dir, "iter%d_valid" % (iter_id + 1)),
                              mode=test_mode,
                              batcher=test_model_iter)
            iter_id += 1


def parse_args():
    parser = argparse.ArgumentParser(description='Train the Meteotn nowcasting model')
    parser.add_argument('--batch_size', dest='batch_size', help="batchsize of the training process",
                        default=None, type=int)
    parser.add_argument('--cfg', dest='cfg_file', help='Configuration file', required=True, type=str)
    parser.add_argument('--save_dir', help='The saving directory', required=True, type=str)
    parser.add_argument('--data_dir', help='The data directory with hdf_archives folder, hdf_metadata.csv and mask.png',
                        required=True, type=str)
    parser.add_argument('--data_csv', help='alternate metadata CSV file (default: [data_dir]/hdf_metadata.csv)',
                        default=None, type=str)
    parser.add_argument('--date_start', help='Start date to filter the sequences (e.g. 2010-12-31)',
                        default=None, type=lambda s: datetime.strptime(s, '%Y-%m-%d'))
    parser.add_argument('--date_end', help='End date to filter the sequences (e.g. 2016-12-31)',
                        default=None, type=lambda s: datetime.strptime(s, '%Y-%m-%d'))
    parser.add_argument('--ctx', default='cpu', help='Running Context. (default: %(default)s): `--ctx gpu` '
                                                     'or `--ctx gpu0,gpu1` for GPU(s). `--ctx cpu` for CPU')
    parser.add_argument('--threshold', dest='threshold', help='rainfall filter threshold (default 0.28)',
                        default=None, type=float)
    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file, target=cfg.MODEL)
    if args.batch_size is not None:
        cfg.MODEL.TRAIN.BATCH_SIZE = args.batch_size
    if args.threshold is not None:
        cfg.HKO.ITERATOR.FILTER_RAINFALL_THRESHOLD = float(args.threshold)
    cfg.MODEL.SAVE_DIR = args.save_dir
    logging.info(args)
    return args


if __name__ == "__main__":
    main(parse_args())


"""
python train.py \
    --data_dir  /path/to/tassrad19/sequences \
    --save_dir  /my/model/save/dir \
    --cfg  configurations/trajgru_55_55_33_1_64_1_192_1_192_13_13_9_b4.yml \
    --ctx  gpu0,gpu1 \
    --date_start 2010-06-01 \
    --date_end   2017-01-01
"""