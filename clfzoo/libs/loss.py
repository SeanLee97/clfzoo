# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import array_ops

def spread_loss(labels, activations, margin):
    activations_shape = activations.get_shape().as_list()
    mask_t = tf.equal(labels, 1)
    mask_i = tf.equal(labels, 0)    
    activations_t = tf.reshape(
      tf.boolean_mask(activations, mask_t), [activations_shape[0], 1]
    )    
    activations_i = tf.reshape(
      tf.boolean_mask(activations, mask_i), [activations_shape[0], activations_shape[1] - 1]
    )    
    gap_mit = tf.reduce_sum(tf.square(tf.nn.relu(margin - (activations_t - activations_i))))
    return gap_mit        

def cross_entropy(y, logits):    
    #y = tf.argmax(y, axis=1)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)                                               
    loss = tf.reduce_mean(loss) 
    return loss

def margin_loss(y, logits):    
    y = tf.cast(y,tf.float32)
    loss = y * tf.square(tf.maximum(0., 0.9 - logits)) + \
        0.25 * (1.0 - y) * tf.square(tf.maximum(0., logits - 0.1))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    #loss = tf.reduce_mean(loss)    
    return loss

"""
binary label focal loss
"""
def bin_focal_loss(y, logits, weights=None, alpha=0.5, gamma=2):
    sigmoid_p = tf.nn.sigmoid(logits)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = array_ops.where(y >= sigmoid_p, y - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(y > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_mean(per_entry_cross_ent)

"""
Muti-label focal loss 
"""
def focal_loss(y, logits, gamma=2, epsilon=1e-10):
    y = tf.cast(tf.expand_dims(y, -1), tf.int32)

    predictions = tf.nn.softmax(logits)
    batch_idxs = tf.range(0, tf.shape(y)[0])
    batch_idxs = tf.expand_dims(batch_idxs, 1)

    idxs = tf.concat([batch_idxs, y], 1)
    y_true_pred = tf.gather_nd(predictions, idxs)

    y = tf.cast(tf.squeeze(y, axis=-1), tf.float32)
    losses =  tf.log(y_true_pred+epsilon) * tf.pow(1-y_true_pred, gamma)

    return -tf.reduce_mean(losses)
