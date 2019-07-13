"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""

import tensorflow as tf

slim = tf.contrib.slim

def bp(input, out_size, a_f='tanh'):
    """a bp neruel network
    Args:
        input: normally, a tensor with the shape (bs, n_features)
        is_training: whether train or not
    Return:
        a tensor with the shape(bs, 1)
    """
    n_featues = input.get_shape().as_list()[1]
    hidden1 = tf.layers.dense(input, n_featues*2)
    out1 = tf.nn.leaky_relu(hidden1)
    # # out1 = slim.batch_norm(hidden1, is_training=is_training, activation_fn=tf.nn.leaky_relu)
    #

    # hidden2 = tf.layers.dense(out1, n_featues)
    # out2 = tf.nn.leaky_relu(hidden2)
    # out2 = slim.batch_norm(hidden2, is_training=is_training, activation_fn=tf.nn.leaky_relu)

    # hidden3 = tf.layers.dense(out2, n_featues//2)
    # out3 = tf.nn.leaky_relu(hidden3)
    # out3 = slim.batch_norm(hidden3, is_training=is_training, activation_fn=tf.nn.leaky_relu)

    hidden3 = tf.layers.dense(out1, out_size)
    if a_f == 'tanh':
        output = tf.nn.tanh(hidden3)
    elif a_f == None:
        output = hidden3
    else:
        raise  ValueError('!!')

    return output