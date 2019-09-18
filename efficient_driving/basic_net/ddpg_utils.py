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

class critic(object):
    def __init__(self, max_abs_q_val = 45):
        self.max_abs_q_val = max_abs_q_val
        pass

    def build_graph(self, img_state, action, is_training, var_scope):
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
            feat, endpoints = mobilenetv2(inputs=img_state, n_dims=20, is_training=is_training)

            ## combine the state and action
            state_action = tf.concat([slim.softmax(feat), action], axis=-1)

            ## action state value
            self.action_state_value = self.max_abs_q_val*tf.reshape(bp(state_action, out_size=1, a_f='tanh'), shape=[-1])

            ## get trainable vars
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            return self.action_state_value, self.trainable_vars


class actor(object):
    def __init__(self):
        pass

    def build_graph(self, img_state, n_action_space, is_training, var_scope, action_range=None):
        """build a deterministic policy net π(s)
        Args:
            img_state: generally, a tensor with the shape(bs, h, w, c)
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
            bs = tf.shape(img_state)[0]

            ## get the abstract of state
            outputs, end_points = mobilenetv2(inputs=img_state, n_dims=20, is_training=is_training)

            ## get flaten state abstract
            # feat_flat = tf.reduce_max(outputs, axis=[1, 2])

            feat = slim.softmax(outputs)

            ## action
            self.action = tf.reshape(bp(feat, out_size=n_action_space), shape=[bs, n_action_space])

            if action_range:
                min_a = np.expand_dims(np.array(action_range)[:,0], axis=0)
                max_a = np.expand_dims(np.array(action_range)[:,1], axis=0)
                ## when the raw action range is [-1,1], use following formula to transform.
                self.action = (self.action+1.)*(max_a-min_a)/2.+min_a

            ## get trainable vars
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            # return outputs, self.action, self.trainable_vars ## this return is for imitator
            return self.action, self.trainable_vars

if __name__ == '__main__':
    img = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
    action = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    n_action_space = 1
    ac = actor()
    ct = critic()

    # ac.build_graph(img, n_action_space, True, 'actor_online', action_range=[[-1, -1]])
    ct.build_graph(img, action, True, 'critic_online')