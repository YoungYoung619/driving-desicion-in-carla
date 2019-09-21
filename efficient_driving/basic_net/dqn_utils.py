"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf
import numpy as np

from RL.backbone.vgg16 import *
from RL.backbone.mobilenet_v2 import mobilenetv2
from RL.backbone.bp_net import bp

slim = tf.contrib.slim

class action_value_net(object):
    def __init__(self):
        pass

    def build_graph(self, img_state, n_action, is_training, var_scope):
        """build a action-value function approximation Q(s,a)
        Args:
            state: generally, a tensor with the shape(bs, h, w, c)
            n_action: total action number
            is_training: indicate whether training or not
            var_scope: tensorflow scope
        Return:
            a tensor represents the action state value
            a list of tensor represents all the trainable vars in this network
        """
        with tf.variable_scope(var_scope) as scope:
            ## get the abstract of state
            # feat, endpoints = mobilenetv2(inputs=img_state, n_dims=10*n_action, is_training=is_training)

            self.action_value, endpoints = mobilenetv2(inputs=img_state, n_dims=n_action, is_training=is_training)

            ## action state value
            # self.action_value = bp(feat, out_size=n_action, a_f=None)

            ## get trainable vars
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            return self.action_value, self.trainable_vars


if __name__ == '__main__':
    img = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
    action = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    n_action_space = 21
    # ac = action_value()
    # aa = ac.build_graph(img, n_action_space, True, 'online_q_val')
    pass