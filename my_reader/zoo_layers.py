# -*- coding:utf8 -*-

import tensorflow as tf

#
def create_dense_vars(input_size, output_size, weight_mat=None,
                      use_bias=True, bias_init_value=0.1, scope="dense"):
    """
    """
    with tf.variable_scope(scope):
        if weight_mat is None:
            W = tf.get_variable("kernel", [input_size, output_size],
                                initializer = tf.variance_scaling_initializer(),
                                dtype = tf.float32)
        else:
            W = weight_mat
        if use_bias:
            b = tf.get_variable("bias", [output_size],
                                initializer = tf.constant_initializer(bias_init_value),
                                dtype = tf.float32)
        else:
            b = None
    #
    return W, b

def dense_with_vars(inputs, Wb, transpose_b=False):
    """
    """
    shape_list = inputs.get_shape().as_list()
    if len(shape_list) == 2:
        out = tf.matmul(inputs, Wb[0], transpose_b=transpose_b)
        if Wb[1] is not None: out = tf.nn.bias_add(out, Wb[1])
        return out
    #
    input_size = shape_list[-1]
    shape = tf.shape(inputs)
    if transpose_b:
        output_size = Wb[0].get_shape().as_list()[0]
    else:
        output_size = Wb[0].get_shape().as_list()[1]
    #
    out_shape = [shape[idx] for idx in range(len(shape_list) - 1)] + [output_size]
    flat_inputs = tf.reshape(inputs, [-1, input_size])
    out = tf.matmul(flat_inputs, Wb[0], transpose_b=transpose_b)
    if Wb[1] is not None: out = tf.nn.bias_add(out, Wb[1])
    out = tf.reshape(out, out_shape)
    return out


def dense(x, output_size, weight_mat=None, transpose_b=False,
          use_bias=True, bias_init_value=0.1, scope="dense"):
    """
    """
    input_size = x.get_shape().as_list()[-1]
    #
    wb = create_dense_vars(input_size, output_size,
                           weight_mat=weight_mat, use_bias=use_bias,
                           bias_init_value=bias_init_value, scope=scope)
    #
    out = dense_with_vars(x, wb, transpose_b=transpose_b)
    return out

#
def rnn_layer(input_sequence, sequence_length, rnn_size,
              keep_prob = 1.0, activation = None, scope = 'bi-lstm'):
    '''build bidirectional lstm layer'''
    #
    # time_major = False
    #
    input_sequence = tf.nn.dropout(input_sequence, keep_prob)
    #
    weight_initializer = tf.truncated_normal_initializer(stddev = 0.01)
    act = activation or tf.nn.tanh
    #
    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, activation = act,
                                      initializer = weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, activation = act,
                                      initializer = weight_initializer)
    #    
    #cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=dropout_rate)
    #cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=dropout_rate)
    #
    #cell_fw = MyLSTMCell(rnn_size, keep_prob, initializer = weight_initializer)
    #cell_bw = MyLSTMCell(rnn_size, keep_prob, initializer = weight_initializer)
    #
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = False,
                                                    dtype = tf.float32,
                                                    scope = scope)
    #
    rnn_output = tf.concat(rnn_output, 2, name = 'output')
    return rnn_output
    #

def gru_layer(input_sequence, sequence_length, rnn_size,
              keep_prob = 1.0, activation = None, scope = 'bi-gru'):
    '''build bidirectional gru layer'''
    #
    # time_major = False
    #
    input_sequence = tf.nn.dropout(input_sequence, keep_prob)
    #
    act = activation or tf.nn.tanh
    #
    cell_fw = tf.nn.rnn_cell.GRUCell(rnn_size, activation = act)
    cell_bw = tf.nn.rnn_cell.GRUCell(rnn_size, activation = act)
    #
    # cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=dropout_rate)
    # cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=dropout_rate)
    #
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = False,
                                                    dtype = tf.float32,
                                                    scope = scope)
    #
    rnn_output = tf.concat(rnn_output, 2, name = 'output')
    return rnn_output
    #     

