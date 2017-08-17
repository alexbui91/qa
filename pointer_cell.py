import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.rnn_cell_impl import _linear

"""
    custom gru cell to pointer cell
    contexts: tensors with shape B x L x D is to sum up to B x D then be the input of next RNN step
"""


class PointerCell(RNNCell):

    def __init__(self,
                num_units,
                contexts,
                concat_context=None,
                # use_prev=True,
                result_type=None,
                activation=None,
                reuse=None,
                kernel_initializer=None,
                bias_initializer=None):
        super(PointerCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self.contexts = contexts
        self.concat_context = concat_context
        self.result_type = result_type
        self.reuse = reuse
        # self.use_prev = use_prev

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs=None, state=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope("gate_pointer_cell"):  # Reset gate and update gate.
            a_, c_ = self.get_max_pooling(self.contexts, inputs, state)
            if self.concat_context:
                inputs = tf.concat([inputs, c_])
            else:
                inputs = c_
        # We start with bias of 1.0 to not reset and not update.
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                dtype = [a.dtype for a in [inputs, state]][0]
                bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
            value = math_ops.sigmoid(
                _linear([inputs, state], 2 * self._num_units, True, bias_ones, self._kernel_initializer))
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
            with vs.variable_scope("candidate"):
                c = self._activation(
                    _linear([inputs, r * state], self._num_units, True,
                        self._bias_initializer, self._kernel_initializer))
            new_h = u * state + (1 - u) * c
            if self.result_type is 'pred':
                outputs = a_
            else:
                outputs = new_h
        return outputs, new_h

    def get_max_pooling(self, context, time_step_vector=None, prev_=None):
        """ sum over context to single vectzor"""
        context_ = tf.unstack(context, axis=1)
        if not time_step_vector is None or not prev_ is None:
            tmp = list()
            for c_ in context_:
                input_vector = list()
                input_vector.append(c_)
                if not time_step_vector is None:
                    input_vector.append(time_step_vector)
                if not prev_ is None:
                    input_vector.append(prev_)
                tmp.append(tf.concat(input_vector, axis=1))
            # B x L x D
            s_ = tf.transpose(tf.stack(tmp), perm=[1,0,2])
        else:
            s_ = context
        # # B x L x D
        # s_ = tf.layers.dense(s_,
        #                     self._num_units,
        #                     activation=tf.nn.tanh,
        #                     name="attention_question",
        #                     reuse=self.reuse)
        # To B x L x 1
        s_ = tf.layers.dense(s_,
                            1,
                            activation=None,
                            name="attention_question_softmax",
                            reuse=self.reuse)
        # To B x L
        # BxLxD => BxDxL x BxLx1 => BxD
        u_ = tf.squeeze(tf.matmul(tf.transpose(context, perm=[0,2,1]), s_))
        # return prediction and agg vector
        # expect a_: BxL, agg: BxD
        return tf.squeeze(s_), u_

class GateAttentionBase(PointerCell):
    
    def call(self, inputs=None, state=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        a_, c_ = self.get_max_pooling(self.contexts, inputs, state)
        with vs.variable_scope("sigmoid_gate"): 
            if self.concat_context:
                inputs = tf.concat([inputs, c_])
                g_ = tf.nn.sigmoid(_linear([inputs], self._num_units * 2, False))
            else:
                inputs = c_
                g_ = tf.nn.sigmoid(_linear([inputs], self._num_units, False))
        inputs = tf.multiply(inputs, g_)
        # We start with bias of 1.0 to not reset and not update.
        bias_ones = self._bias_initializer
        if self._bias_initializer is None:
            dtype = [a.dtype for a in [inputs, state]][0]
            bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
        value = math_ops.sigmoid(
            _linear([inputs, state], 2 * self._num_units, True, bias_ones, self._kernel_initializer))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        with vs.variable_scope("candidate"):
            c = self._activation(
                _linear([inputs, r * state], self._num_units, True,
                    self._bias_initializer, self._kernel_initializer))
        new_h = u * state + (1 - u) * c
        if self.result_type is 'pred':
            outputs = a_
        else:
            outputs = new_h
        return outputs, new_h