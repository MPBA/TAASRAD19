import mxnet as mx
from mxnet import nd
from nowcasting.config import cfg_from_file
from nowcasting.encoder_forecaster import EncoderForecasterBaseFactory, init_optimizer_using_cfg, \
    EncoderForecasterStates
from nowcasting.my_module import MyModule
from nowcasting.operators import *
from nowcasting.ops import *
from nowcasting.radar_evaluation import rainfall_to_pixel


def load_params(prefix, epoch):
    """

    Parameters
    ----------
    prefix : str
    epoch : int

    Returns
    -------
    arg_params : dict
    aux_params : dict
    """
    import mxnet.ndarray as nd
    save_dict = nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def load_encoder_forecaster_params(load_dir, load_iter, encoder_net, forecaster_net):
    logging.info("Loading parameters from {}, Iter = {}"
                 .format(os.path.realpath(load_dir), load_iter))
    encoder_arg_params, encoder_aux_params = load_params(prefix=os.path.join(load_dir, "encoder_net"),
                                                         epoch=load_iter)
    encoder_net.init_params(arg_params=encoder_arg_params, aux_params=encoder_aux_params,
                            allow_missing=False, force_init=True)
    forecaster_arg_params, forecaster_aux_params = load_params(prefix=os.path.join(load_dir, "forecaster_net"),
                                                               epoch=load_iter)
    forecaster_net.init_params(arg_params=forecaster_arg_params,
                               aux_params=forecaster_aux_params,
                               allow_missing=False,
                               force_init=True)
    logging.info("Loading Complete!")


def encoder_forecaster_build_networks(factory, context,
                                      shared_encoder_net=None,
                                      shared_forecaster_net=None,
                                      shared_loss_net=None,
                                      for_finetune=False):
    """

    Parameters
    ----------
    factory : EncoderForecasterBaseFactory
    context : list
    shared_encoder_net : MyModule or None
    shared_forecaster_net : MyModule or None
    shared_loss_net : MyModule or None
    for_finetune : bool

    Returns
    -------

    """
    encoder_net = MyModule(factory.encoder_sym(),
                           data_names=[ele.name for ele in factory.encoder_data_desc()],
                           label_names=[],
                           context=context,
                           name="encoder_net")
    encoder_net.bind(data_shapes=factory.encoder_data_desc(),
                     label_shapes=None,
                     inputs_need_grad=True,
                     shared_module=shared_encoder_net)
    if shared_encoder_net is None:
        encoder_net.init_params(mx.init.MSRAPrelu(slope=0.2))
        init_optimizer_using_cfg(encoder_net, for_finetune=for_finetune)
    forecaster_net = MyModule(factory.forecaster_sym(),
                              data_names=[ele.name for ele in
                                          factory.forecaster_data_desc()],
                              label_names=[],
                              context=context,
                              name="forecaster_net")
    forecaster_net.bind(data_shapes=factory.forecaster_data_desc(),
                        label_shapes=None,
                        inputs_need_grad=True,
                        shared_module=shared_forecaster_net)
    if shared_forecaster_net is None:
        forecaster_net.init_params(mx.init.MSRAPrelu(slope=0.2))
        init_optimizer_using_cfg(forecaster_net, for_finetune=for_finetune)

    loss_net = MyModule(factory.loss_sym(),
                        data_names=[ele.name for ele in
                                    factory.loss_data_desc()],
                        label_names=[ele.name for ele in
                                     factory.loss_label_desc()],
                        context=context,
                        name="loss_net")
    loss_net.bind(data_shapes=factory.loss_data_desc(),
                  label_shapes=factory.loss_label_desc(),
                  inputs_need_grad=True,
                  shared_module=shared_loss_net)
    if shared_loss_net is None:
        loss_net.init_params()
    return encoder_net, forecaster_net, loss_net


def get_loss_weight_symbol(data, mask, seq_len):
    if cfg.MODEL.USE_BALANCED_LOSS:
        balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        weights = mx.sym.ones_like(data) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (data >= threshold)
        weights = weights * mask
    else:
        weights = mask
    if cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "same":
        return weights
    elif cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "linear":
        upper = cfg.MODEL.TEMPORAL_WEIGHT_UPPER
        assert upper >= 1.0
        temporal_mult = 1 + \
                        mx.sym.arange(start=0, stop=seq_len) * (upper - 1.0) / (seq_len - 1.0)
        temporal_mult = mx.sym.reshape(temporal_mult, shape=(seq_len, 1, 1, 1, 1))
        weights = mx.sym.broadcast_mul(weights, temporal_mult)
        return weights
    elif cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "exponential":
        upper = cfg.MODEL.TEMPORAL_WEIGHT_UPPER
        assert upper >= 1.0
        base_factor = np.log(upper) / (seq_len - 1.0)
        temporal_mult = mx.sym.exp(mx.sym.arange(start=0, stop=seq_len) * base_factor)
        temporal_mult = mx.sym.reshape(temporal_mult, shape=(seq_len, 1, 1, 1, 1))
        weights = mx.sym.broadcast_mul(weights, temporal_mult)
        return weights
    else:
        raise NotImplementedError


class RadarNowcastingFactory(EncoderForecasterBaseFactory):
    def __init__(self,
                 batch_size,
                 in_seq_len,
                 out_seq_len,
                 ctx_num=1,
                 name="hko_nowcasting"):
        super(RadarNowcastingFactory, self).__init__(batch_size=batch_size,
                                                     in_seq_len=in_seq_len,
                                                     out_seq_len=out_seq_len,
                                                     ctx_num=ctx_num,
                                                     height=cfg.HKO.ITERATOR.HEIGHT,
                                                     width=cfg.HKO.ITERATOR.WIDTH,
                                                     name=name)
        self._central_region = cfg.HKO.EVALUATION.CENTRAL_REGION

    def _slice_central(self, data):
        """Slice the central region in the given symbol

        Parameters
        ----------
        data : mx.sym.Symbol

        Returns
        -------
        ret : mx.sym.Symbol
        """
        x_begin, y_begin, x_end, y_end = self._central_region
        return mx.sym.slice(data,
                            begin=(0, 0, 0, y_begin, x_begin),
                            end=(None, None, None, y_end, x_end))

    def _concat_month_code(self):
        # TODO
        raise NotImplementedError

    def loss_sym(self,
                 pred=mx.sym.Variable('pred'),
                 mask=mx.sym.Variable('mask'),
                 target=mx.sym.Variable('target')):
        """Construct loss symbol.

        Optional args:
            pred: Shape (out_seq_len, batch_size, C, H, W)
            mask: Shape (out_seq_len, batch_size, C, H, W)
            target: Shape (out_seq_len, batch_size, C, H, W)
        """
        self.reset_all()
        weights = get_loss_weight_symbol(data=target, mask=mask, seq_len=self._out_seq_len)
        mse = weighted_mse(pred=pred, gt=target, weight=weights)
        mae = weighted_mae(pred=pred, gt=target, weight=weights)
        gdl = masked_gdl_loss(pred=pred, gt=target, mask=mask)
        avg_mse = mx.sym.mean(mse)
        avg_mae = mx.sym.mean(mae)
        avg_gdl = mx.sym.mean(gdl)
        global_grad_scale = cfg.MODEL.NORMAL_LOSS_GLOBAL_SCALE
        if cfg.MODEL.L2_LAMBDA > 0:
            avg_mse = mx.sym.MakeLoss(avg_mse,
                                      grad_scale=global_grad_scale * cfg.MODEL.L2_LAMBDA,
                                      name="mse")
        else:
            avg_mse = mx.sym.BlockGrad(avg_mse, name="mse")
        if cfg.MODEL.L1_LAMBDA > 0:
            avg_mae = mx.sym.MakeLoss(avg_mae,
                                      grad_scale=global_grad_scale * cfg.MODEL.L1_LAMBDA,
                                      name="mae")
        else:
            avg_mae = mx.sym.BlockGrad(avg_mae, name="mae")
        if cfg.MODEL.GDL_LAMBDA > 0:
            avg_gdl = mx.sym.MakeLoss(avg_gdl,
                                      grad_scale=global_grad_scale * cfg.MODEL.GDL_LAMBDA,
                                      name="gdl")
        else:
            avg_gdl = mx.sym.BlockGrad(avg_gdl, name="gdl")
        loss = mx.sym.Group([avg_mse, avg_mae, avg_gdl])
        return loss


class NowcastingPredictor(object):
    def __init__(self, model_dir: str, model_iter: int, model_cfg: str, batch_size: int, ctx: list):
        self.batch_size = batch_size
        self.ctx = ctx
        self.model_dir = model_dir
        self.model_iter = model_iter
        cfg_from_file(model_cfg, target=cfg.MODEL)
        assert cfg.MODEL.FRAME_STACK == 1 and cfg.MODEL.FRAME_SKIP == 1
        assert len(self.model_dir) > 0

        hko_factory = RadarNowcastingFactory(batch_size=self.batch_size, in_seq_len=cfg.MODEL.IN_LEN,
                                             out_seq_len=cfg.MODEL.OUT_LEN)
        self.encoder_net, self.forecaster_net, t_loss_net = encoder_forecaster_build_networks(factory=hko_factory,
                                                                                    context=self.ctx, for_finetune=True)
        self.encoder_net.summary()
        self.forecaster_net.summary()
        t_loss_net.summary()
        load_encoder_forecaster_params(load_dir=self.model_dir, load_iter=self.model_iter,
                                       encoder_net=self.encoder_net,
                                       forecaster_net=self.forecaster_net)

        self.states = EncoderForecasterStates(factory=hko_factory, ctx=self.ctx[0])

    def predict(self, in_frame):
        self.states.reset_all()
        in_frame_nd = nd.array(in_frame, ctx=self.ctx[0])
        self.encoder_net.forward(is_train=False,
                            data_batch=mx.io.DataBatch(data=[in_frame_nd] + self.states.get_encoder_states()))
        self.states.update(states_nd=self.encoder_net.get_outputs())
        self.forecaster_net.forward(is_train=False, data_batch=mx.io.DataBatch(data=self.states.get_forecaster_state()))

        pred_nd = self.forecaster_net.get_outputs()
        pred_nd = pred_nd[0]
        pred_nd = nd.clip(pred_nd, a_min=0, a_max=1)
        return pred_nd.asnumpy()

