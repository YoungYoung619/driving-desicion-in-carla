"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""

import tensorflow as tf

slim = tf.contrib.slim

def bp(input, out_size):
    """a bp neruel network
    Args:
        input: normally, a tensor with the shape (bs, n_features)
        is_training: whether train or not
    Return:
        a tensor with the shape(bs, 1)
    """
    n_featues = input.get_shape().as_list()[1]
    hidden1 = tf.layers.dense(input, n_featues*2)
    # bn1_act = slim.batch_norm(hidden1, is_training=is_training, activation_fn=tf.nn.leaky_relu)

    hidden2 = tf.layers.dense(hidden1, n_featues)
    # bn2_act = slim.batch_norm(hidden2, is_training=is_training, activation_fn=tf.nn.leaky_relu)

    hidden3 = tf.layers.dense(hidden2, n_featues//2)
    # bn2_act = slim.batch_norm(hidden3, is_training=is_training, activation_fn=tf.nn.leaky_relu)

    hidden4 = tf.layers.dense(hidden3, out_size)
    output = tf.nn.tanh(hidden4)

    return output