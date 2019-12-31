import mxnet as mx
import numpy as np
import os
import logging


class SaveNpyOp(mx.operator.CustomOp):
    def __init__(self, save_name="op", save_dir=None):
        super(SaveNpyOp, self).__init__()
        self._save_name = save_name
        self._save_dir = '.' if save_dir is None else save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self._input_save_path = os.path.join(self._save_dir, '{}.npy'.format(save_name))
        self._grad_save_path = os.path.join(self._save_dir, '{}_grad.npy'.format(save_name))

    def forward(self, is_train, req, in_data, out_data, aux):
        logging.info("Saving Input to {}".format(os.path.realpath(self._input_save_path)))
        np.save(self._input_save_path, in_data[0].asnumpy())
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        logging.info("Saving Gradient to {}".format(os.path.realpath(self._input_save_path)))
        np.save(self._grad_save_path, out_grad[0].asnumpy())
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("save_npy")
class SaveNpyOpProp(mx.operator.CustomOpProp):
    def __init__(self, save_name="op", save_dir="."):
        super(SaveNpyOpProp, self).__init__(need_top_grad=True)
        self._save_name = save_name
        self._save_dir = save_dir

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return SaveNpyOp(save_name=self._save_name,
                         save_dir=self._save_dir)


def save_npy(data, save_name="op", save_dir="."):
    return mx.symbol.Custom(data=data,
                            save_name=save_name,
                            save_dir=save_dir,
                            op_type="save_npy")


def grid_generator(batch_size, height, width, normalize=True):
    """Generate the grid based on width and height

    Parameters
    ----------
    batch_size : int
    width : int
    height : int
    normalize : bool
        Whether to normalize the grid elements into [-1, 1]

    Returns
    -------
    ret : mx.sym.Symbol
        Shape : (batch_size, 2, height, width), the channel contains (x, y)
    """
    x = mx.sym.arange(start=0, stop=width)
    y = mx.sym.arange(start=0, stop=height)
    x = mx.sym.broadcast_to(mx.sym.Reshape(x, shape=(1, 1, 1, width)),
                            shape=(batch_size, 1, height, width))
    y = mx.sym.broadcast_to(mx.sym.Reshape(y, shape=(1, 1, height, 1)),
                            shape=(batch_size, 1, height, width))
    if normalize:
        x = x / float(width - 1) * 2.0 - 1.0
        y = y / float(height - 1) * 2.0 - 1.0
    ret = mx.sym.Concat(x, y, num_args=2, dim=1)
    return ret


def group_add(lhs, rhs):
    """

    Parameters
    ----------
    lhs : list of mx.sym.Symbol
    rhs : list of mx.sym.Symbol

    Returns
    -------
    ret : list of mx.sym.Symbol
    """
    if isinstance(lhs, mx.sym.Symbol):
        return lhs + rhs
    assert len(lhs) == len(rhs)
    ret = []
    for i in range(len(lhs)):
        if isinstance(lhs[i], list):
            ret.append(group_add(lhs[i], rhs[i]))
        else:
            ret.append(lhs[i] + rhs[i])
    return ret


def one_step_diff(dat, axis):
    """

    Parameters
    ----------
    dat : mx.sym.Symbol
    axes : tuple

    Returns
    -------

    """
    return mx.sym.slice_axis(dat, axis=axis, begin=0, end=-1) - \
           mx.sym.slice_axis(dat, axis=axis, begin=1, end=None)


def masked_gdl_loss(pred, gt, mask):
    """

    Parameters
    ----------
    pred : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)
    gt : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)
    mask : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)

    Returns
    -------
    gdl : mx.sym.Symbol
        Shape: (seq_len, batch_size)
    """
    valid_mask_h = mx.sym.slice_axis(mask, axis=3, begin=0, end=-1) *\
                   mx.sym.slice_axis(mask, axis=3, begin=1, end=None)
    valid_mask_w = mx.sym.slice_axis(mask, axis=4, begin=0, end=-1) *\
                   mx.sym.slice_axis(mask, axis=4, begin=1, end=None)
    pred_diff_h = mx.sym.abs(one_step_diff(pred, axis=3))
    pred_diff_w = mx.sym.abs(one_step_diff(pred, axis=4))
    gt_diff_h = mx.sym.abs(one_step_diff(gt, axis=3))
    gt_diff_w = mx.sym.abs(one_step_diff(gt, axis=4))
    gd_h = mx.sym.abs(pred_diff_h - gt_diff_h)
    gd_w = mx.sym.abs(pred_diff_w - gt_diff_w)
    gdl = mx.sym.sum(valid_mask_h * gd_h, axis=(2, 3, 4)) +\
          mx.sym.sum(valid_mask_w * gd_w, axis=(2, 3, 4))
    return gdl


def weighted_l2(pred, gt, weight):
    """
    
    Parameters
    ----------
    pred : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)
    gt : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)
    weight : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)

    Returns
    -------
    l2 : mx.nd.NDArray
        Shape: (seq_len, batch_size)
    """
    l2 = weight * mx.sym.square(pred - gt)
    l2 = mx.sym.sum(l2, axis=(2, 3, 4))
    return l2


def weighted_mse(pred, gt, weight):
    return weighted_l2(pred, gt, weight)


def weighted_l1(pred, gt, weight):
    l1 = weight * mx.sym.abs(pred - gt)
    l1 = mx.sym.sum(l1, axis=(2, 3, 4))
    return l1


def weighted_mae(pred, gt, weight):
    return weighted_l1(pred, gt, weight)


# def masked_hit_miss_counts(pred, gt, mask, thresholds):
#     """
#
#     Parameters
#     ----------
#     pred : mx.sym.Symbol
#         Shape: (seq_len, batch_size, 1, H, W)
#     gt : mx.sym.Symbol
#         Shape: (seq_len, batch_size, 1, H, W)
#     mask : mx.sym.Symbol
#         Shape: (seq_len, batch_size, 1, H, W)
#     thresholds : list
#
#     Returns
#     -------
#     hits : mx.nd.NDArray
#         Shape: (seq_len, batch_size, len(thresholds))
#     misses : mx.nd.NDArray
#         Shape: (seq_len, batch_size, len(thresholds))
#     false_alarms : mx.nd.NDArray
#         Shape: (seq_len, batch_size, len(thresholds))
#     correct_negatives : mx.nd.NDArray
#         Shape: (seq_len, batch_size, len(thresholds))
#     """
#     from nowcasting.hko_evaluation import rainfall_to_pixel
#     thresholds = [rainfall_to_pixel(threshold) for threshold in thresholds]
#     hits = []
#     misses = []
#     false_alarms = []
#     correct_negatives = []
#     for threshold in thresholds:
#         pred_rain_mask = pred > threshold
#         gt_rain_mask = gt > threshold
#         hits_ele = pred_rain_mask * gt_rain_mask * mask
#         misses_ele = (1 - pred_rain_mask) * gt_rain_mask * mask
#         false_alarms_ele = pred_rain_mask * (1 - gt_rain_mask) * mask
#         correct_negatives_ele = (1 - pred_rain_mask) * (1 - gt_rain_mask) * mask
#         hits.append(mx.sym.sum(hits_ele, axis=(3, 4)))
#         misses.append(mx.sym.sum(misses_ele, axis=(3, 4)))
#         false_alarms.append(mx.sym.sum(false_alarms_ele, axis=(3, 4)))
#         correct_negatives.append(mx.sym.sum(correct_negatives_ele, axis=(3, 4)))
#     hits = mx.sym.concat(*hits, dim=2, num_args=len(thresholds))
#     misses = mx.sym.concat(*misses, dim=2, num_args=len(thresholds))
#     false_alarms = mx.sym.concat(*false_alarms, dim=2, num_args=len(thresholds))
#     correct_negatives = mx.sym.concat(*correct_negatives, dim=2, num_args=len(thresholds))
#     return hits, misses, false_alarms, correct_negatives
