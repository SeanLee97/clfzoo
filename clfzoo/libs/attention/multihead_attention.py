# -*- coding: utf-8 -*-

import clfzoo.libs as libs
import tensorflow as tf

helper_doc = """\n[libs]>  Implement Multihead Attention.
    
Args:
    MultiheadAttention:
        - num_heads: int (default 8)
            number of heads
        - dropout: float
            dropout rate
    __call__:
        - queries: tensor
            (batch_size, q_len, embed_size)
        - keys: tensor
            (batch_size, k_len, embed_size)
        - values: tensor
            (batch_size, v_len, embed_size)
        - num_units: int (default None)
            the last dim of queries
        - query_mask:
        - value_mask:
        - residual: bool
            whether to use residual connection
        - scope: str
        - reuse: bool
            whether to reuse the weights of a previous layer
        
Usage:
    attention = MultiheadAttention(num_heads=8, dropout=0.0, )
    output = attention(query, key, values, num_units=None,
                       query_mask=None, value_mask=None, residual=True, 
                       scope="multihead_attention", reuse=None)
Outputs:
    (batch_size, q_len , embed_size)  
"""

class MultiheadAttention(object):
    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, 
                 num_heads=8,
                 dropout=0.0):
        self.num_heads = num_heads
        self.dropout = dropout

    def __call__(self,
                 queries,
                 keys,
                 values,
                 num_units=None,
                 query_mask=None,
                 value_mask=None,
                 residual=True,
                 scope="multihead_attention",
                 reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
                
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            # alias
            nh = self.num_heads
            nu = num_units

            # projection
            Q = tf.layers.dense(queries, nh * nu)
            Q = tf.transpose(
                    tf.reshape(Q, [-1, Q.get_shape()[1], nh, nu]), [0, 2, 1, 3])
            K = tf.layers.dense(queries, nh * nu)
            K = tf.transpose(
                    tf.reshape(K, [-1, K.get_shape()[1], nh, nu]), [0, 2, 1, 3])      
            V = tf.layers.dense(values, nh * nu)
            V = tf.transpose(
                    tf.reshape(V, [-1, V.get_shape()[1], nh, nu]), [0, 2, 1, 3])
            
            # Multiplication & Scale
            S = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(nu))
            S = tf.transpose(S, [0, 3, 2, 1])
            if value_mask is not None:
                for _ in range(len(S.shape) - 2):
                    value_mask = tf.expand_dims(value_mask, 2)
                S = libs.mask_logits(S, value_mask)
            S = tf.transpose(S, [0, 3, 2, 1])

            # Activation
            S = tf.nn.softmax(S)
            S = tf.nn.dropout(S, 1.0-self.dropout)

            # Output
            output = tf.matmul(S, V)
            output = tf.transpose(output, [0, 2, 1, 3])
            output = tf.reshape(output, [-1, output.get_shape()[1], nh*nu])
            if query_mask is not None:
                for _ in range(len(output.shape) - 2):
                    query_mask = tf.expand_dims(query_mask, 2)
                output = output * tf.cast(query_mask, tf.float32)

            # residual connection
            if residual:
                output = tf.layers.dense(output, nu)
                output += queries

            # Normalize
            output = tf.contrib.layers.layer_norm(output)

            return output
