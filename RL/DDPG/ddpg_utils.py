"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf
import numpy as np

from RL.backbone.mobilenet_v2 import mobilenetv2
from RL.backbone.bp_net import bp

class critic(object):
    def __init__(self):
        pass


    def build_graph(self, state, action, is_training, var_scope):
        """build a action-value function approximation Q(s,a)
        Args:
            state: generally, a tensor with the shape(bs, h, w, c)
            action: generally, a tensor with the shape(bs, n_action), n_action is the total number of action space
            is_training: indicate whether training or not
            var_scope: tensorflow scope
        Return:
            a tensor represents the action state value
            a list of tensor represents all the trainable vars in this network
        """
        with tf.variable_scope(var_scope) as scope:
            ## get the abstract of state
            feat, endpoints = mobilenetv2(inputs=state, is_training=is_training)

            ## get flaten state abstract
            feat_flat = tf.reduce_max(feat, axis=[1,2])

            ## combine the state and action
            state_action = tf.concat([feat_flat, action], axis=-1)

            ## action state value
            self.action_state_value = tf.reshape(bp(state_action, out_size=1), shape=[-1])

            ## get trainable vars
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            return self.action_state_value, self.trainable_vars


class actor(object):
    def __init__(self):
        pass


    def build_graph(self, state, n_action_space, is_training, var_scope, action_range=None):
        """build a deterministic policy net π(s)
        Args:
            state: generally, a tensor with the shape(bs, h, w, c)
            n_action_space: an int, represents the number of action space
            is_training: indicate whether training or not
            var_scope: tensorflow scope
            action_range: a list indicates the output action range, default, all actions output are in [-1,1]
        Return:
            a tensor represents the action, with the shape (bs, n_action)
            a list of tensor represents all the trainable vars in this network
        """
        if action_range:
            assert len(action_range) == n_action_space

        with tf.variable_scope(var_scope) as scope:
            bs = tf.shape(state)[0]

            ## get the abstract of state
            feat, endpoints = mobilenetv2(inputs=state, is_training=is_training)

            ## get flaten state abstract
            feat_flat = tf.reduce_max(feat, axis=[1, 2])

            ## action
            self.action = tf.reshape(bp(feat_flat, out_size=n_action_space), shape=[bs, n_action_space])

            if action_range:
                min_a = np.expand_dims(np.array(action_range)[:,0], axis=0)
                max_a = np.expand_dims(np.array(action_range)[:,1], axis=0)
                ## when the raw action range is [-1,1], use following formula to transform.
                self.action = (self.action+1.)*(max_a-min_a)/2.+min_a

            ## get trainable vars
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            return self.action, self.trainable_vars

