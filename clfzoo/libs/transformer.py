# -*- coding: utf-8 -*-

import tensorflow as tf
import clfzoo.libs as libs
from clfzoo.libs.attention.multihead_attention import MultiheadAttention

helper_doc="""\n[libs]> Implement Transformer Encoder

Args:
    Encoder:
        - num_heads: int
            heads of multihead attention, default 8
        - num_blocks: int
            default 4
        - activation: non-linear activation function
            default tf.nn.relu
        - bias: bool
            whether to use bias
        - dropout: float
            dropout rate

    __call__:
        - inputs: tensor
        - num_units: int
        - input_mask: 
        - scope: str
        - reuse: bool

Outputs:
    the same shape as inputs
Usage:
    transEnc = transformer.Encoder(num_heads=8,
                                   num_blocks=4,
                                   activation=tf.nn.relu,
                                   dropout=0.0,
                                   bias=False)
    output = transEnc(inputs, num_units, 
                      input_mask=None, 
                      scope='transformer_encoder', 
                      reuse=None)

"""

class Encoder(object):
    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, 
                 num_heads=8,
                 num_blocks=4,
                 activation=tf.nn.relu,
                 dropout=0.0,   
                 bias=False):

        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.activation = activation
        self.bias = bias
        self.dropout = dropout

    def feedforward(self, inputs, num_units):
        outputs = tf.layers.conv1d(inputs=inputs, filters=num_units[0], kernel_size=1,
                                  activation=self.activation, use_bias=self.bias)

        outputs = tf.layers.conv1d(inputs=outputs, filters=num_units[1], kernel_size=1,
                                   activation=None, use_bias=self.bias)
        # residual
        outputs += inputs

        # normalization
        #outputs = libs.layer_norm(outputs)
        outputs = tf.contrib.layers.layer_norm(outputs) 
        return outputs

    def __call__(self, inputs, num_units, input_mask=None, 
                 scope="transformer_encoder", reuse=None):

        with tf.variable_scope(scope, reuse = reuse):

            enc = inputs

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):                
                    # multihead_attention
                    enc = MultiheadAttention(num_heads=self.num_heads, 
                                             dropout=0.0)(enc, enc, enc, 
                                                                   num_units=num_units,
                                                                   query_mask=input_mask,
                                                                   value_mask=input_mask)

                    # feed_forward
                    enc = self.feedforward(enc, [4*num_units, num_units])

            return enc


helper_doc="""\n[libs]> Implement Transformer Decoder

Args:
    Encoder:
        - num_heads: int
            heads of multihead attention, default 8
        - num_blocks: int
            default 4
        - activation: non-linear activation function
            default tf.nn.relu
        - bias: bool
            whether to use bias
        - dropout: float
            dropout rate

    __call__:
        - inputs: tensor
        - encoder: tensor
            the output of transformer.Encoder
        - num_units: int
        - input_mask: 
        - encoder_mask: 
        - scope: str
        - reuse: bool

Outputs:
    the same shape as inputs
Usage:
    transDec = transformer.Decoder(num_heads=8,
                                   num_blocks=4,
                                   activation=tf.nn.relu,
                                   dropout=0.0,
                                   bias=False)
    output = transDec(inputs, encoder, num_units, 
                      input_mask=None, 
                      encoder_mask=None, 
                      scope='transformer_encoder', 
                      reuse=None)

"""

class Decoder(object):
    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, 
                 num_heads=8,
                 num_blocks=4,
                 activation=tf.nn.relu,
                 dropout=0.0,   
                 bias=False):

        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.activation = activation
        self.bias = bias
        self.dropout = dropout

    def feedforward(self, inputs, num_units):
        outputs = tf.layers.conv1d(inputs=inputs, filters=num_units[0], kernel_size=1,
                                  activation=None, use_bias=self.bias)

        outputs = tf.layers.conv1d(inputs=outputs, filters=num_units[1], kernel_size=1,
                                   activation=self.activation, use_bias=self.bias)

        # residual connection
        outputs += inputs

        # normalization
        outputs = tf.contrib.layers.layer_norm(outputs)

        return outputs

    def __call__(self, inputs, encoder, num_units, 
                 input_mask=None, encoder_mask=None,
                 scope="transformer_decoder", reuse=None):

        with tf.variable_scope(scope, reuse = reuse):

            dec = inputs

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):                
                    # multihead_attention
                    dec = MultiheadAttention(num_heads=self.num_heads, 
                                             dropout=self.dropout)(dec, dec, dec, 
                                                                   num_units=num_units, 
                                                                   query_mask=input_mask,
                                                                   value_mask=input_mask,
                                                                   scope="self_attention")

                    dec = MultiheadAttention(num_heads=self.num_heads, 
                                             dropout=self.dropout)(dec, encoder, encoder, 
                                                                   num_units=num_units, 
                                                                   query_mask=input_mask,
                                                                   value_mask=encoder_mask,
                                                                   scope="vanilla_attention")
                    # feed_forward
                    dec = self.feedforward(dec, [4*num_units, num_units])

            return dec
