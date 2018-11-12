# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import array_ops


"""Base RNN
"""
class BaseRNN(object):
    pass


helper_doc="""\n[artf]> Implement Multi-layers RNN (vanilla version)

Args:
    RNN: 
        - num_units: int | required
        - batch_size: int | required
        - input_size: int/placeholder | required
        - num_layers: int
            layers of rnn 
        - dropout: float/placeholder
            dropout rate
        - kernel: enum lstm(default) / gru

    __call__:
        - inputs: required
        - seq_len: default None
        - batch_first: bool
            True if the shape is (batch_size, seq_len, embed_size)
            False if the shape is (seq_len, batch_size, embed_size)
        - reuse:

Usage:
    rnn = RNN(num_units, batch_size, input_size,
              num_layers=1, dropout=0.0, kernel='lstm')
    (value1, ...) = rnn(inputs, 
                        seq_len=None, 
                        batch_first=True,
                        scope='rnn',
                        reuse=None)


Input:
    inputs: (batch_size, seq_len, embed_size)

Output:
    lstm: return 3 values that are output, c, h
        - output:  (batch_size, seq_len, num_layers * num_units)
            the product of the concatenation of each layer
        - c: (batch_size, num_units)
            last context state
        - h: (batch_size, num_units)
            last hidden state

    gru: return 2 values that are output, c
        - output:  (batch_size, seq_len, num_layers * num_units)
            the product of the concatenation of each layer
        - c: (batch_size, num_units)
            last context state
"""
class RNN(BaseRNN):

    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, num_units, batch_size, input_size,
                 num_layers=1, dropout=0.0, kernel='lstm'):
        if kernel == 'gru':
            self.rnn_cell = tf.nn.rnn_cell.GRUCell
        else:
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell

        self.kernel = kernel
        self.num_layers = num_layers
        self.rnns = []
        self.dropout_mask = []

        for layer in range(num_layers):
            in_size = input_size if layer == 0 else num_units
            rnn_fw = self.rnn_cell(num_units)
            mask_fw = tf.nn.dropout(tf.ones([batch_size, 1, in_size], dtype=tf.float32), 1.0 - dropout)

            self.rnns.append(rnn_fw)
            self.dropout_mask.append(mask_fw)

    def __call__(self, inputs, 
                 seq_len=None, 
                 batch_first=True,
                 scope='rnn',
                 reuse=None):

        if not batch_first:
            # transpose to batch first
            inputs = tf.transpose(inputs, [1, 0, 2])
        outputs = [inputs]
        with tf.variable_scope(scope, reuse=reuse):
            for layer in range(self.num_layers):
                rnn_fw = self.rnns[layer]
                mask_fw = self.dropout_mask[layer]
                # forward
                with tf.variable_scope('rnn_forward_%d' % layer):
                    out_fw, state_fw = tf.nn.dynamic_rnn(
                                                cell=rnn_fw, 
                                                inputs=outputs[-1]*mask_fw,
                                                sequence_length=seq_len,
                                                dtype=tf.float32)
                outputs.append(out_fw)

            res = tf.concat(outputs[1:], axis=2)
            if not batch_first:
                res = tf.transpose(res, [1, 0, 2])

            if self.kernel == 'lstm':
                return res, state_fw.c, state_fw.h
            return res, state_fw



helper_doc="""\n[artf]> Implement Multi-layers BiDirectional RNN  (vanilla version)

Args:
    RNN: 
        - num_units: int | required
        - batch_size: int | required
        - input_size: int/placeholder | required
        - num_layers: int
            layers of rnn 
        - dropout: float/placeholder
            dropout rate
        - kernel: enum lstm(default) / gru

    __call__:
        - inputs: required
        - seq_len: default None
        - batch_first: bool
            True if the shape is (batch_size, seq_len, embed_size)
            False if the shape is (seq_len, batch_size, embed_size)
        - scope: str
        - reuse:

Usage:
    rnn = BiRNN(num_units, batch_size, input_size,
              num_layers=1, dropout=0.0, kernel='lstm')
    (value1, ...) = rnn(inputs, 
                        seq_len=None, 
                        batch_first=True,
                        scope='bidirection_rnn',
                        reuse=None)


Input:
    inputs: (batch_size, seq_len, embed_size)

Output:
    lstm: return 3 values that are output, c, h
        - output:  (batch_size, seq_len, 2 * num_layers * num_units)
            the product of the concatenation of each layer
        - c: (batch_size, 2 * num_units)
            last context state
        - h: (batch_size, 2 * num_units)
            last hidden state

    gru: return 2 values that are output, c
        - output:  (batch_size, seq_len, 2 * num_layers * num_units)
            the product of the concatenation of each layer
        - c: (batch_size, 2 * num_units)
            last context state
"""
class BiRNN(BaseRNN):
    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, num_units, batch_size, input_size, 
                 num_layers=1, dropout=0.0, kernel='lstm'):
        
        if kernel == 'gru':
            self.rnn_cell = tf.nn.rnn_cell.GRUCell
        else:
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell

        self.kernel = kernel
        self.num_layers = num_layers
        self.rnns = []
        self.dropout_mask = []

        # init layers
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else 2 * num_units
            rnn_fw = self.rnn_cell(num_units)
            rnn_bw = self.rnn_cell(num_units)
            mask = tf.nn.dropout(tf.ones([batch_size, 1, in_size], dtype=tf.float32), 1.0 - dropout)

            self.rnns.append((rnn_fw, rnn_bw))
            self.dropout_mask.append(mask)

    def __call__(self, inputs, 
                 seq_len=None, 
                 batch_first=True,
                 scope='bidirection_rnn',
                 reuse=None):

        if not batch_first:
            # transpose to batch first
            inputs = tf.transpose(inputs, [1, 0, 2])
        outputs = [inputs]
        with tf.variable_scope(scope, reuse=reuse):
            for layer in range(self.num_layers):
                rnn_fw, rnn_bw = self.rnns[layer]
                dropout_mask = self.dropout_mask[layer]

                output, state = tf.nn.bidirectional_dynamic_rnn(
                                                cell_fw=rnn_fw,
                                                cell_bw=rnn_bw,
                                                inputs=outputs[-1]*dropout_mask,
                                                sequence_length=seq_len,
                                                dtype=tf.float32,
                                                scope='dynamic_rnn_%d' % layer)

                outputs.append(tf.concat([output[0], output[1]], axis=2))
            res = tf.concat(outputs[1:], axis=2)
            if not batch_first:
                res = tf.transpose(res, [1, 0, 2])
            if self.kernel == 'lstm':
                C = tf.concat([state[0].c, state[1].c], axis=1)
                H = tf.concat([state[0].h, state[1].h], axis=1)
                return res, C, H
            C = tf.concat(state[:], axis=1)
            return res, C

helper_doc="""\n[artf]> Implement Multi-layers CudnnRNN

CudnnRNN is more effective than vanilla RNN on GPU

Args:
    RNN: 
        - num_units: int | required
        - batch_size: int | required
        - input_size: int/placeholder | required
        - num_layers: int
            layers of rnn 
        - dropout: float/placeholder
            dropout rate
        - kernel: enum lstm(default) / gru

    __call__:
        - inputs: required
        - seq_len: default None
        - batch_first: bool
            True if the shape is (batch_size, seq_len, embed_size)
            False if the shape is (seq_len, batch_size, embed_size)
        - scope: str
        - reuse:

Usage:
    rnn = CudnnRNN(num_units, batch_size, input_size,
              num_layers=1, dropout=0.0, kernel='lstm')
    (value1, ...) = rnn(inputs, 
                        seq_len=None, 
                        batch_first=True,
                        scope='cudnn_rnn',
                        reuse=None)


Input:
    inputs: (batch_size, seq_len, embed_size)

Output:
    lstm: return 3 values that are output, c, h
        - output:  (batch_size, seq_len, num_layers * num_units)
            the product of the concatenation of each layer
        - c: (batch_size, num_units)
            last context state
        - h: (batch_size, num_units)
            last hidden state

    gru: return 2 values that are output, c
        - output:  (batch_size, seq_len, num_layers * num_units)
            the product of the concatenation of each layer
        - c: (batch_size, num_units)
            last context state
"""
class CudnnRNN(BaseRNN):

    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, num_units, batch_size, input_size,
                 num_layers=1, dropout=0.0, kernel='lstm'):
        if kernel == 'gru':
            self.rnn_cell = tf.contrib.cudnn_rnn.CudnnGRU
        else:
            self.rnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM

        self.kernel = kernel
        self.num_layers = num_layers
        self.rnns = []
        self.inits = []
        self.dropout_mask = []

        for layer in range(num_layers):
            in_size = input_size if layer == 0 else num_units

            init_c = tf.tile(tf.get_variable('init_c', 
                                             shape=[1, 1, num_units],
                                             initializer=tf.zeros_initializer()), 
                            [1, batch_size, 1])

            init_h = tf.tile(tf.get_variable('init_h', 
                                             shape=[1, 1, num_units],
                                             initializer=tf.zeros_initializer()), 
                            [1, batch_size, 1])
            
            rnn_fw = self.rnn_cell(1, num_units)
            mask_fw = tf.nn.dropout(tf.ones([1, batch_size, in_size], dtype=tf.float32), 1.0 - dropout)

            self.inits.append((init_c, init_h))
            self.rnns.append(rnn_fw)
            self.dropout_mask.append(mask_fw)

    def __call__(self, inputs, 
                 seq_len=None, 
                 batch_first=True,
                 scope='cudnn_rnn',
                 reuse=None):

        if batch_first:
            # if batch_first
            inputs = tf.transpose(inputs, [1, 0, 2])
        outputs = [inputs]
        with tf.variable_scope(scope, reuse=reuse):
            for layer in range(self.num_layers):
                init_c, init_h = self.inits[layer]
                rnn_fw = self.rnns[layer]
                mask_fw = self.dropout_mask[layer]
                # forward
                with tf.variable_scope('rnn_forward_%d' % layer):
                    if self.kernel == 'lstm':
                        initial_state=(init_c, init_h)
                    else:
                        initial_state=(init_c, )
                    out_fw, state_fw = rnn_fw(outputs[-1]*mask_fw, initial_state=initial_state)

                outputs.append(out_fw)

            res = tf.concat(outputs[1:], axis=2)
            if batch_first:
                # if batch_first
                res = tf.transpose(res, [1, 0, 2])

            shape = state_fw[0].get_shape()
            if self.kernel == 'lstm':
                C = tf.reshape(state_fw[0], [shape[1], shape[2]])
                H = tf.reshape(state_fw[1], [shape[1], shape[2]])
                return res, C, H
            C = tf.reshape(state_fw[0], [shape[1], shape[2]])
            return res, C

helper_doc="""\n[artf]> Implement Multi-layers BiDirectional CudnnRNN

CudnnRNN is more effective than vanilla RNN on GPU

Args:
    RNN: 
        - num_units: int | required
        - batch_size: int | required
        - input_size: int/placeholder | required
        - num_layers: int
            layers of rnn 
        - dropout: float/placeholder
            dropout rate
        - kernel: enum lstm(default) / gru

    __call__:
        - inputs: required
        - seq_len: default None
        - batch_first: bool
            True if the shape is (batch_size, seq_len, embed_size)
            False if the shape is (seq_len, batch_size, embed_size)
        - scope: str
        - reuse:

Usage:
    rnn = BiCudnnRNN(num_units, batch_size, input_size,
              num_layers=1, dropout=0.0, kernel='lstm', scope='rnn')
    (value1, ...) = rnn(inputs, 
                        seq_len=None, 
                        batch_first=True,
                        scope="bidirection_cudnn_rnn",
                        reuse=None)


Input:
    inputs: (batch_size, seq_len, embed_size)

Output:
    lstm: return 3 values that are output, c, h
        - output:  (batch_size, seq_len, 2 * num_layers * num_units)
            the product of the concatenation of each layer
        - c: (batch_size, 2 * num_units)
            last context state
        - h: (batch_size, 2 * num_units)
            last hidden state

    gru: return 2 values that are output, c
        - output:  (batch_size, seq_len, 2 * num_layers * num_units)
            the product of the concatenation of each layer
        - c: (batch_size, 2 * num_units)
            last context state
"""
class BiCudnnRNN(BaseRNN):
    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, num_units, batch_size, input_size, 
                 num_layers=1, dropout=0.0, kernel='lstm'):
        
        if kernel == 'gru':
            self.rnn_cell = tf.contrib.cudnn_rnn.CudnnGRU
        else:
            self.rnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM

        self.kernel = kernel
        self.num_layers = num_layers
        self.rnns = []
        self.inits = []
        self.dropout_mask = []

        # init layers
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else 2 * num_units

            init_rw_c = tf.tile(tf.get_variable('init_rw_c', 
                                             shape=[1, 1, num_units],
                                             initializer=tf.zeros_initializer()), 
                            [1, batch_size, 1])

            init_rw_h = tf.tile(tf.get_variable('init_rw_h', 
                                             shape=[1, 1, num_units],
                                             initializer=tf.zeros_initializer()), 
                            [1, batch_size, 1])

            init_bw_c = tf.tile(tf.get_variable('init_bw_c', 
                                             shape=[1, 1, num_units],
                                             initializer=tf.zeros_initializer()), 
                            [1, batch_size, 1])

            init_bw_h = tf.tile(tf.get_variable('init_bw_h', 
                                             shape=[1, 1, num_units],
                                             initializer=tf.zeros_initializer()), 
                            [1, batch_size, 1])

            rnn_fw = self.rnn_cell(1, num_units)
            rnn_bw = self.rnn_cell(1, num_units)

            mask_fw = tf.nn.dropout(tf.ones([1, batch_size, in_size], dtype=tf.float32), 1.0 - dropout)
            mask_bw = tf.nn.dropout(tf.ones([1, batch_size, in_size], dtype=tf.float32), 1.0 - dropout)

            self.inits.append(((init_rw_c, init_rw_h), (init_bw_c, init_bw_h)))
            self.rnns.append((rnn_fw, rnn_bw))
            self.dropout_mask.append((mask_fw, mask_bw))

    def __call__(self, inputs, 
                 seq_len=None, 
                 batch_first=True,
                 scope='bidirection_cudnn_rnn',
                 reuse=None):
        if batch_first:
            # transpose to batch second
            inputs = tf.transpose(inputs, [1, 0, 2])
        outputs = [inputs]

        with tf.variable_scope(scope, reuse=reuse):
            for layer in range(self.num_layers):
                (init_rw_c, init_rw_h), (init_bw_c, init_bw_h) = self.inits[layer]
                rnn_fw, rnn_bw = self.rnns[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]

                # forward
                with tf.variable_scope("fw_{}".format(layer)):
                    if self.kernel == 'lstm':
                        initial_state = (init_rw_c, init_rw_h)
                    else:
                        initial_state = (init_rw_c, )

                    out_fw, state_fw = rnn_fw(
                        outputs[-1] * mask_fw, initial_state=initial_state)

                # backword
                with tf.variable_scope("bw_{}".format(layer)):
                    if self.kernel == 'lstm':
                        initial_state = (init_rw_c, init_rw_h)
                    else:
                        initial_state = (init_rw_c, )
                    if seq_len is not None:
                        inputs_bw = tf.reverse_sequence(
                            outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    else:
                        inputs_bw = array_ops.reverse(outputs[-1] * mask_bw, axis=[0])

                    out_bw, state_bw = rnn_bw(inputs_bw, initial_state=initial_state)
                    if seq_len is not None:
                        out_bw = tf.reverse_sequence(
                            out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    else:
                        out_bw = array_ops.reverse(out_bw, axis=[0])

                outputs.append(tf.concat([out_fw, out_bw], axis=2)) 

            res = tf.concat(outputs[1:], axis=2)

            if batch_first:
                # transpose back
                res = tf.transpose(res, [1, 0, 2])

            
            if self.kernel == 'lstm':
                C = tf.concat([state_fw[0][0], state_bw[0][0]], axis=1)
                H = tf.concat([state_fw[1][0], state_bw[1][0]], axis=1)
                return res, C, H
            C = tf.concat([state_fw[0][0], state_bw[0][0]], axis=1)
            return res, C
