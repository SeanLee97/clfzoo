# -*- coding: utf-8 -*-

import six
import math
import tensorflow as tf
import clfzoo.libs as libs

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)

def assert_rank(tensor, expected_rank, name=None):

    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
                "For the tensor `%s` in scope `%s`, the actual rank "
                "`%d` (shape = %s) is not equal to the expected rank `%s`" %
                (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):

    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    
    return shape

def reshape_to_matrix(inputs):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = inputs.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (inputs.shape))
    if ndims == 2:
        return inputs

    width = inputs.shape[-1]
    output_tensor = tf.reshape(inputs, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    dropout=0.0,
                    initializer_range=0.02,
                    return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):

    def transpose_for_scores(inputs, batch_size, num_attention_heads,
                           seq_length, width):
        output_tensor = tf.reshape(
            inputs, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
  
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = libs.dropout(attention_probs, dropout)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer

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
        - initializer_range: float
        - return_all_layers: bool

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
                 activation=libs.gelu,
                 dropout=0.0,  
                 initializer_range=0.02, 
                 return_all_layers=False,
                 bias=False):

        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.activation = activation
        self.bias = bias
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.return_all_layers = return_all_layers

    def __call__(self, inputs, num_units, input_mask=None, 
                 scope="transformer_encoder", reuse=None):

        with tf.variable_scope(scope, reuse = reuse):

            if num_units % self.num_heads != 0:
                raise ValueError(
                    "The hidden size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (num_units, self.num_heads))

            attention_head_size = int(num_units / self.num_heads)
            input_shape = get_shape_list(inputs, expected_rank=3)
            batch_size = input_shape[0]
            seq_length = input_shape[1]
            input_width = input_shape[2]


            # get attention mask
            if input_mask is not None:
                to_mask = tf.cast(
                    tf.reshape(input_mask, [batch_size, 1, seq_length]), tf.float32)
                broadcast_ones = tf.ones(
                    shape=[batch_size, seq_length, 1], dtype=tf.float32)

                # Here we broadcast along two dimensions to create the mask.
                attention_mask = broadcast_ones * to_mask            
            else:
                attention_mask = None

            prev_output = reshape_to_matrix(inputs)

            all_layer_outputs = []

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):  
                    layer_input = prev_output

                    with tf.variable_scope("self_multihead_attention"):   
                        attention_heads = []

                        attention_head = attention_layer(
                                            from_tensor=layer_input,
                                            to_tensor=layer_input,
                                            attention_mask=attention_mask,
                                            num_attention_heads=self.num_heads,
                                            size_per_head=attention_head_size,
                                            dropout=self.dropout,
                                            initializer_range=self.initializer_range,
                                            return_2d_tensor=True,
                                            batch_size=batch_size,
                                            from_seq_length=seq_length,
                                            to_seq_length=seq_length)

                        attention_heads.append(attention_head)

                    attention_output = None

                    if len(attention_heads) == 1:
                        attention_output = attention_heads[0]
                    else:
                        # In the case where we have other sequences, we just concatenate
                        # them to the self-attention head before the projection.
                        attention_output = tf.concat(attention_heads, axis=-1)


                    # feed_forward
                    with tf.variable_scope("feedforward"):

                        attention_output = tf.layers.dense(
                            attention_output,
                            num_units,
                            kernel_initializer=create_initializer(self.initializer_range))
                      
                        attention_output = libs.dropout(attention_output, self.dropout)
                        attention_output = tf.contrib.layers.layer_norm(attention_output + layer_input)

                        hidden_output = tf.layers.dense(
                            attention_output,
                            num_units*2,
                            activation=self.activation,
                            kernel_initializer=create_initializer(self.initializer_range))

                        layer_output = tf.layers.dense(
                            hidden_output,
                            num_units,
                            kernel_initializer=create_initializer(self.initializer_range))
                    
                        layer_output = libs.dropout(layer_output, self.dropout)
                        layer_output = tf.contrib.layers.layer_norm(layer_output + attention_output)

                        prev_output = layer_output
                        all_layer_outputs.append(layer_output)

            if self.return_all_layers:
                final_outputs = []
                for layer_output in all_layer_outputs:
                    final_output = reshape_from_matrix(layer_output, input_shape)
                    final_outputs.append(final_output)
                return final_outputs
            else:
                final_output = reshape_from_matrix(prev_output, input_shape)

            return final_output

helper_doc="""\n[libs]> Implement Transformer Decoder

Args:
    Decoder:
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
        - initializer_range: float
        - return_all_layers: bool

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
                 initializer_range=0.02, 
                 return_all_layers=False,
                 bias=False):

        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.activation = activation
        self.bias = bias
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.return_all_layers = return_all_layers

    def __call__(self, inputs, encoder, num_units, input_mask=None, 
                 scope="transformer_encoder", reuse=None):

        with tf.variable_scope(scope, reuse = reuse):

            if num_units % self.num_heads != 0:
                raise ValueError(
                    "The hidden size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (num_units, self.num_heads))

            attention_head_size = int(num_units / self.num_heads)
            input_shape = get_shape_list(inputs, expected_rank=3)
            batch_size = input_shape[0]
            seq_length = input_shape[1]
            input_width = input_shape[2]


            # get attention mask
            if input_mask is not None:
                to_mask = tf.cast(
                    tf.reshape(input_mask, [batch_size, 1, seq_length]), tf.float32)
                broadcast_ones = tf.ones(
                    shape=[batch_size, seq_length, 1], dtype=tf.float32)

                # Here we broadcast along two dimensions to create the mask.
                attention_mask = broadcast_ones * to_mask 
            else:
                attention_mask = None           

            prev_output = reshape_to_matrix(inputs)

            all_layer_outputs = []

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):  
                    layer_input = prev_output

                    with tf.variable_scope("self_multihead_attention"):  
                        attention_heads = [] 

                        attention_dec = attention_layer(
                                            from_tensor=layer_input,
                                            to_tensor=layer_input,
                                            attention_mask=attention_mask,
                                            num_attention_heads=self.num_heads,
                                            size_per_head=attention_head_size,
                                            dropout=self.dropout,
                                            initializer_range=self.initializer_range,
                                            return_2d_tensor=True,
                                            batch_size=batch_size,
                                            from_seq_length=seq_length,
                                            to_seq_length=seq_length)

                        attention_head = attention_layer(
                                            from_tensor=layer_input,
                                            to_tensor=encoder,
                                            attention_mask=attention_mask,
                                            num_attention_heads=self.num_heads,
                                            size_per_head=attention_head_size,
                                            dropout=self.dropout,
                                            initializer_range=self.initializer_range,
                                            return_2d_tensor=True,
                                            batch_size=batch_size,
                                            from_seq_length=seq_length,
                                            to_seq_length=seq_length)

                        attention_heads.append(attention_head)

                    attention_output = None

                    if len(attention_heads) == 1:
                        attention_output = attention_heads[0]
                    else:
                        # In the case where we have other sequences, we just concatenate
                        # them to the self-attention head before the projection.
                        attention_output = tf.concat(attention_heads, axis=-1)


                    # feed_forward
                    with tf.variable_scope("feedforward"):

                        attention_output = tf.layers.dense(
                            attention_output,
                            num_units,
                            kernel_initializer=create_initializer(self.initializer_range))
                      
                        attention_output = libs.dropout(attention_output, self.dropout)
                        attention_output = tf.contrib.layers.layer_norm(attention_output + layer_input)

                        hidden_output = tf.layers.dense(
                            attention_output,
                            num_units*2,
                            activation=self.activation,
                            kernel_initializer=create_initializer(self.initializer_range))

                        layer_output = tf.layers.dense(
                            hidden_output,
                            num_units,
                            kernel_initializer=create_initializer(self.initializer_range))
                    
                        layer_output = libs.dropout(layer_output, self.dropout)
                        layer_output = tf.contrib.layers.layer_norm(layer_output + attention_output)

                        prev_output = layer_output
                        all_layer_outputs.append(layer_output)

            if self.return_all_layers:
                final_outputs = []
                for layer_output in all_layer_outputs:
                    final_output = reshape_from_matrix(layer_output, input_shape)
                    final_outputs.append(final_output)
                return final_outputs
            else:
                final_output = reshape_from_matrix(prev_output, input_shape)

            return final_output