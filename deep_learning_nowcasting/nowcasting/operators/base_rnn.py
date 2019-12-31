import mxnet as mx
from nowcasting.operators.common import group_add


class BaseStackRNN(object):
    def __init__(self, base_rnn_class, stack_num=1,
                 name="BaseStackRNN", residual_connection=True,
                 **kwargs):
        self._base_rnn_class = base_rnn_class
        self._residual_connection = residual_connection
        self._name = name
        self._stack_num = stack_num
        self._prefix = name + "_"
        self._rnns = [base_rnn_class(prefix=self._name + "_%d" %i, **kwargs) for i in range(stack_num)]
        self._init_counter = 0
        self._state_info = None

    def init_state_vars(self):
        """Initial state variable for this cell.

        Parameters
        ----------

        Returns
        -------
        state_vars : nested list of Symbol
            starting states for first RNN step
        """
        state_vars = []
        for i, info in enumerate(self.state_info):
            state = mx.sym.var(name='%s_begin_state_%s' % (self._name, self.state_postfix[i]), **info)
            state_vars.append(state)
        return state_vars

    def concat_to_split(self, concat_states):
        assert len(concat_states) == len(self.state_info)
        split_states = [[] for i in range(self._stack_num)]
        for i in range(len(self.state_info)):
            channel_axis = self.state_info[i]['__layout__'].lower().find('c')
            ele = mx.sym.split(concat_states[i], num_outputs=self._stack_num, axis=channel_axis)
            for j in range(self._stack_num):
                split_states[j].append(ele[j])
        return split_states

    def split_to_concat(self, split_states):
        # Concat the states together
        concat_states = []
        for i in range(len(self.state_info)):
            channel_axis = self.state_info[i]['__layout__'].lower().find('c')
            concat_states.append(mx.sym.concat(*[ele[i] for ele in split_states],
                                               dim=channel_axis))
        return concat_states

    def check_concat(self, states):
        ret = not isinstance(states[0], list)
        return ret

    def to_concat(self, states):
        if not self.check_concat(states):
            states = self.split_to_concat(states)
        return states

    def to_split(self, states):
        if self.check_concat(states):
            states = self.concat_to_split(states)
        return states

    @property
    def state_postfix(self):
        return self._rnns[0].state_postfix

    @property
    def state_info(self):
        if self._state_info is None:
            info = []
            for i in range(len(self._rnns[0].state_info)):
                ele = {}
                for rnn in self._rnns:
                    if 'shape' not in ele:
                        ele['shape'] = list(rnn.state_info[i]['shape'])
                    else:
                        channel_dim = rnn.state_info[i]['__layout__'].lower().find('c')
                        ele['shape'][channel_dim] += rnn.state_info[i]['shape'][channel_dim]
                    if '__layout__' not in ele:
                        ele['__layout__'] = rnn.state_info[i]['__layout__'].upper()
                    else:
                        assert rnn.state_info[i]['__layout__'] == ele['__layout__'].upper()
                ele['shape'] = tuple(ele['shape'])
                info.append(ele)
            self._state_info = info
            return info
        else:
            return self._state_info

    def flatten_add_layout(self, states, blocked=False):
        """
        
        Parameters
        ----------
        states : list of list or list

        Returns
        -------
        ret : list
        """
        states = self.to_concat(states)
        assert self.check_concat(states)
        ret = []
        for i, ele in enumerate(states):
            if blocked:
                ret.append(mx.sym.BlockGrad(ele, __layout__=self.state_info[i]['__layout__']))
            else:
                ele._set_attr(__layout__=self.state_info[i]['__layout__'])
                ret.append(ele)
        return ret

    def reset(self):
        for i in range(len(self._rnns)):
            self._rnns[i].reset()

    def unroll(self, length, inputs=None, begin_states=None, ret_mid=False):
        if begin_states is None:
            begin_states = self.init_state_vars()
        begin_states = self.to_split(begin_states)
        assert len(begin_states) == self._stack_num, len(begin_states)
        for ele in begin_states:
            assert len(ele) == len(self.state_info)
        outputs = []
        final_states = []
        mid_infos = []
        for i in range(len(self._rnns)):
            rnn_out_list, rnn_final_states, rnn_mid_infos =\
                self._rnns[i].unroll(length=length, inputs=inputs,
                                     begin_state=begin_states[i],
                                     layout="TC",
                                     ret_mid=True)
            if self._residual_connection and i > 0:
                # Use residual connections
                rnn_out_list = group_add(lhs=rnn_out_list, rhs=inputs)
            inputs = rnn_out_list
            outputs.append(rnn_out_list)
            final_states.append(rnn_final_states)
            mid_infos.append(rnn_mid_infos)
        if ret_mid:
            return outputs, final_states, mid_infos
        else:
            return outputs, final_states
