# -*- coding: utf-8 -*-

import tensorflow as tf
import clfzoo.libs as libs

helper_doc="""\n[libs]> Implement Convolution

Args:
    Conv:
        - activation: 
        - kernel_size: int
            default 1
        - bias: bool
        - initializer:
        - regularizer:

    __call__:
        inputs: tensor
        output_size: int
        scope: str
        reuse: bool

Usage:
    conv = Conv(activation=None, kernel_size=1, bias=None)
    output = conv(inputs, output_size, scope='conv_encoder', reuse=None)

"""

class Conv(object):
    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, 
                 activation=None,   
                 kernel_size=1,
                 bias=None,
                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32),
                 regularizer=tf.contrib.layers.l2_regularizer(scale = 3e-7)):
        self.activation = activation
        self.kernel_size = kernel_size
        self.bias = bias
        self.initializer = initializer
        self.regularizer = regularizer

    def __call__(self, inputs, output_size, 
                 scope="conv_encoder", reuse=None):

        with tf.variable_scope(scope, reuse = reuse):
            shapes = inputs.shape.as_list()
            if len(shapes) > 4:
                raise NotImplementedError
            elif len(shapes) == 4:
                filter_shape = [1, self.kernel_size, shapes[-1], output_size]
                bias_shape = [1, 1, 1, output_size]
                strides = [1, 1, 1, 1]
            else:
                filter_shape = [self.kernel_size, shapes[-1], output_size]
                bias_shape = [1, 1, output_size]
                strides = 1
            conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
            kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=self.regularizer,
                        initializer=self.initializer)
            outputs = conv_func(inputs, kernel_, strides, "SAME")

            if self.bias:
                outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=self.regularizer,
                        initializer=tf.zeros_initializer())

            if self.activation is not None:
                return self.activation(outputs)
            return outputs 
