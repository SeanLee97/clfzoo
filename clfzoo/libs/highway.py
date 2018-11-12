# -*- coding: utf-8 -*-

import tensorflow as tf
from clfzoo.libs.conv import Conv

helper_doc="""\n[libs]> Implement Highway

Args:
    Highway:
        - kernel: enum
            fcn | fcn3d | conv(default)
        - activation: 
        - num_layers: int
            default 2
        - dropout: float
            default 0.0

    __call__:
        inputs: tensor
        scope: str
        reuse:
        

Usage:
    highway = Highway(activation=None, kernel='conv', num_layers=2, dropout=0.0)
    output = highway(inputs, scope='highway', reuse=None)

"""

class Highway(object):

    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, kernel='conv', activation=None,
                 num_layers=2, dropout=0.0):
        self.kernel = kernel

        self.activation = activation
        if activation is None:
            self.activation = tf.nn.relu

        self.num_layers = num_layers
        self.dropout = dropout

    def __call__(self, inputs, scope='highway', reuse=None):
        self.scope = scope
        self.reuse = reuse

        if 'conv' == self.kernel:
            return self._conv(inputs)
        elif 'fcn' == self.kernel:
            return self._fcn(inputs)
        elif 'fcn3d' == self.kernel:
            return self._fcn3d(inputs)
        else:
            raise ValueError('%s is a invalid kernel, kernel only support conv | fcn | fcn3d')

    def _fcn(self, inputs):
        """
        implement highway by fully-connection
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):
            size = inputs.shape.as_list()[-1]
            
            curr_x = inputs
            curr_x = tf.reshape(curr_x, (-1, size))
            
            for i in range(self.num_layers):
                # init
                W = tf.get_variable('weight_%d' % i,
                                    shape=[size, size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))

                b = tf.get_variable('bias_%d' % i,
                                    shape=[size],
                                    initializer=tf.constant_initializer(0.1))

                W_T = tf.get_variable('weight_transform_%d' % i,
                                    shape=[size, size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))

                b_T = tf.get_variable('bias_transform_%d' % i,
                                    shape=[size],
                                    initializer=tf.constant_initializer(-0.1))

                H = self.activation(tf.matmul(curr_x, W)+b, name='activation_%d' % i)
                T = tf.sigmoid(tf.matmul(curr_x, W_T)+b_T, name='transorm_%d' % i)
                C = tf.subtract(tf.constant(1.0), T, name='gate_%d' % i)

                H = tf.nn.dropout(H, 1.0 - self.dropout)
                # curr_x = (H * T) + (x * C)
                curr_x = tf.add(tf.multiply(H, T), tf.multiply(curr_x, C))

            curr_x = tf.reshape(curr_x, tf.shape(inputs))
            return curr_x

    def _fcn3d(self, inputs):
        """
        If the dimension of x is 3, you can use the function instead of fcn.
        Of course fcn is also okay.
        """
        # check shape
        shapes = inputs.shape.as_list()
        if len(shapes) != 3:
            raise ValueError("""Error: the dimension of input shouble be 3, but got %s 
                             """ % len(shapes))

        size = inputs.shape.as_list()[-1]
        
        with tf.variable_scope(self.scope, reuse=self.reuse):
            for i in range(self.num_layers):

                W = tf.get_variable('weight_%d' % i,
                                    shape=[size, size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))

                b = tf.get_variable('bias_%d' % i,
                                    shape=[size],
                                    initializer=tf.constant_initializer(0.1))

                W_T = tf.get_variable('weight_transform_%d' % i,
                                    shape=[size, size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))

                b_T = tf.get_variable('bias_transform_%d' % i,
                                    shape=[size],
                                    initializer=tf.constant_initializer(-0.1))

                shape = [tf.shape(inputs)[0], tf.shape(W)[0],tf.shape(W)[1]]
                W_ = tf.tile(W, [tf.shape(inputs)[0], 1])  
                W = tf.reshape(W_, shape) 
                W_T_ = tf.tile(W_T, [tf.shape(inputs)[0], 1])  
                W_T = tf.reshape(W_T_, shape)   

                H = self.activation(tf.matmul(inputs, W) + b, name='activation_%d' % i)
                T = tf.sigmoid(tf.matmul(inputs, W_T) + b_T, name='transform_%d' % i)
                C = tf.subtract(tf.constant(1.0), T, name='gate_%d' % i)
                H = tf.nn.dropout(H, 1.0 - self.dropout)

                inputs = tf.add(tf.multiply(H, T), tf.multiply(inputs, C)) # y = (H * T) + (inputs * C)
        return inputs
    
    def _conv(self, inputs):

        with tf.variable_scope(self.scope, reuse=self.reuse):
            size = inputs.shape.as_list()[-1]

            inputs = Conv()(inputs, size, scope="input_projection")

            for i in range(self.num_layers):
                H = Conv(activation=self.activation, bias=True)(inputs, size,
                                                           scope="activation_%d"%i)
                T = Conv(bias=True, activation=tf.sigmoid)(inputs, size, 
                                                           scope="gate_%d"%i)
                H = tf.nn.dropout(H, 1.0 - self.dropout)

                inputs = H * T + inputs * (1.0 - T)
            return inputs

