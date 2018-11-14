# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops

from functools import reduce
from operator import mul

def dropout(inputs, dropout_prob=None):
    if dropout_prob is None or dropout_prob == 0.0:
      return inputs

    output = tf.nn.dropout(inputs, 1.0 - dropout_prob)
    return output

def mask_logits(inputs, mask, mask_value=-1e30):
    shape = inputs.get_shape().as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs * mask + mask_value * (1. - mask)

def gelu(inputs, scope='gelu', reuse=None):
    """Gaussian Error Linear Unit.
    
    This is a smoother version of the ReLU.
    Paper: https://arxiv.org/abs/1606.08415

    Args:
        - inputs: float Tensor
        - scope: scope name
        - reuse: whether to reuse

    Returns:
        `inputs` with the gelu activation applied.
    """
    with tf.variable_scope(scope, reuse=reuse):
        alpha = 0.5 * (1.0 + tf.erf(inputs / tf.sqrt(2.0)))
        return inputs * alpha

def glu(inputs, scope='glu', reuse=None):
    """Gated Linear Units
    Split x into two parts along last dimension
    """
    with tf.variable_scope(scope, reuse=reuse):
        x1, x2 = tf.split(inputs, 2, axis=-1)
        return tf.sigmoid(x1) * x2 

def leaky_relu(inputs, alpha=0.2, scope='leaky_relu', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        return tf.maximum(alpha * inputs, inputs)

def position_embedding(inputs, position_dim, scope='position_embedding', scale=True, reuse=None):
    """position embedding
    inputs: (batch_size, seq_len, word_dim)
    outputs: (batch_size, seq_len, position_dim)
    """
    with tf.variable_scope(scope, reuse=reuse):
        batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
        pos_j = 1. / tf.pow(10000.0, 
                        2 * tf.range(position_dim / 2, dtype=tf.float32 
                        ) / position_dim)
        pos_j = tf.expand_dims(pos_j, 0)
        pos_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
        pos_i = tf.expand_dims(pos_i, 1)
        pos_ij = tf.matmul(pos_i, pos_j)
        pos_ij = tf.concat([tf.cos(pos_ij), tf.sin(pos_ij)], 1)
        outputs = tf.expand_dims(pos_ij, 0) \
                         + tf.zeros((batch_size, seq_len, position_dim))
        if scale:
            outputs = outputs * position_dim**0.5
        return outputs

def layer_norm(inputs, episilon=1e-8, scope="layer_norm", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        filters = inputs.get_shape()[-1]

        scale = tf.get_variable(
            "scale", [filters],  
            initializer=tf.ones_initializer())
        gamma = tf.get_variable(
            "gamma", [filters], 
            initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, axis=-1, keep_dims=True)
        var = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keep_dims=True)

        inputs_ = (inputs-mean) * tf.rsqrt(var + episilon)
        return inputs_ * scale + gamma

def bilinear(p_enc, q_enc):
    """
    Args:
        p_enc: (batch_size, p_len, embed_size)
        q_enc: (batch_size, q_len, embed_size)
    
    Ouput: 
        (batch_size, p_len, q_len)
    """
    p = tf.transpose(p_enc, (0, 2, 1))
    hidden_dim = q_enc.get_shape()[-1]

    attn_W = tf.get_variable("AttnW",
                            shape=[hidden_dim, hidden_dim],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                     mode='FAN_AVG',
                                                     uniform=True,
                                                     dtype=tf.float32))

    w_q = tf.tensordot(q_enc, attn_W, axes=[[2], [0]])
    out = tf.matmul(w_q, p)  # batch x q_len x p_len
    return out

def trilinear(args, output_size = 1, bias = True, 
              squeeze=False, wd=0.0, 
              dropout= 0.0, scope = "trilinear"):
    """
    Args:
        args: tuple | list
            input list, which have the same shape like (batch_size, p_len, q_len, embed_size)
        output_size: int
            the last dim of output
        bias: bool
        squeeze: bool
        wd: float
        dropout: float
        scope: str

    Output:
        (batch_size, p_len, q_len, output_size)        
    """
    def _reconstruct(tensor, ref, keep):
        ref_shape = ref.get_shape().as_list()
        tensor_shape = tensor.get_shape().as_list()
        ref_stop = len(ref_shape) - keep
        tensor_start = len(tensor_shape) - keep
        pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
        keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
        target_shape = pre_shape + keep_shape
        out = tf.reshape(tensor, target_shape)
        return out

    def _flatten(tensor, keep):
        fixed_shape = tensor.get_shape().as_list()
        start = len(fixed_shape) - keep
        left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
        out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
        flat = tf.reshape(tensor, out_shape)
        return flat

    def _linear(args, output_size, bias, bias_initializer=tf.zeros_initializer(), 
                scope = None, kernel_initializer=initializer(), reuse = None):

        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            if shape[1].value is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                                 "but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        with tf.variable_scope(scope, reuse = reuse) as outer_scope:
            weights = tf.get_variable(
                    "linear_kernel", [total_arg_size, output_size],
                    dtype=dtype,
                    regularizer=regularizer,
                    initializer=kernel_initializer)
            if len(args) == 1:
                res = math_ops.matmul(args[0], weights)
            else:
                res = math_ops.matmul(array_ops.concat(args, 1), weights)
            if not bias:
                return res
            with tf.variable_scope(outer_scope) as inner_scope:
                inner_scope.set_partitioner(None)
                biases = tf.get_variable(
                        "linear_bias", [output_size],
                        dtype=dtype,
                        regularizer=regularizer,
                        initializer=bias_initializer)
            return nn_ops.bias_add(res, biases)

    with tf.variable_scope(scope):
        flat_args = [_flatten(arg, 1) for arg in args]
        flat_args = [tf.nn.dropout(arg, 1.0 - dropout) for arg in flat_args]
        flat_out = _linear(flat_args, output_size, bias, scope=scope)
        out = _reconstruct(flat_out, args[0], 1)
        if squeeze:
            return tf.squeeze(out, -1)
        return out

def total_params(variables):
    total_parameters = 0
    for variable in variables:
        try:
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
        except:
            pass
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))
    return total_parameters
