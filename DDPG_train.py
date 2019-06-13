"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import sys, os, glob
import tensorflow as tf

import RL.DDPG.ddpg_utils as ddpg
import RL.rl_utils as rl_tools

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla
except:
    raise ImportError('Please check your carla file')
from carla_utils.world_ops import *
from carla_utils.sensor_ops import *

tf.app.flags.DEFINE_string(
    'checkpoint_dir', '',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'train_dir', './checkpoint/',
    'Directory where checkpoints are written to.')

FLAGS = tf.app.flags.FLAGS

## carla config ##
bgr_camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': 800,
                     'image_size_y': 512, 'fov': 110, 'sensor_tick': 0.02,
                     'transform': carla.Transform(carla.Location(x=0.8, z=1.7)),
                     'attach_to':None}
collision_sensor_config = {'data_type': 'sensor.other.collision','attach_to': None}
invasion_sensor_config = {'data_type': 'sensor.other.lane_detector', 'attach_to': None}


if __name__ == '__main__':
    ########################### TENSORFLOW GRAPH ######################################
    logger.info('Tensorflow graph buliding...')
    ## target input ##
    target_state = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
    target_action = tf.placeholder(shape=[None, 3], dtype=tf.float32) ## steer, accel, brake

    ## online input ##
    online_state = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
    online_action = tf.placeholder(shape=[None, 3], dtype=tf.float32)

    ## other input ##
    reward = tf.placeholder(shape=[None], dtype=tf.float32)
    whether_end = tf.placeholder(shape=[None], dtype=tf.float32) ##1 is end ,0 is continue
    gamma = 0.98 ##for gamma discount reward
    tau = 1e-3 ##for soft update

    ## action range config ##
    action_range_ = [[-1.,1.], [0.,1.], [0.,1.]] ## steer, accel, brake range

    ###############
    ## target AC ##
    ###############
    target_actor = ddpg.actor()
    action_target, target_actor_vars = target_actor.build_graph(state=target_state, n_action_space=3, is_training=True,
                                                                action_range=action_range_, var_scope='target_actor')

    target_critic = ddpg.critic()
    q_target, target_critic_vars = target_critic.build_graph(state=target_state, action=target_action, is_training=True,
                                                             var_scope='target_critic')


    ###############
    ## online AC ##
    ###############
    online_actor = ddpg.actor()
    action_online, online_actor_vars = online_actor.build_graph(state=online_state, n_action_space=3, is_training=True,
                                                                action_range=action_range_, var_scope='online_actor')

    online_critic = ddpg.critic()
    q_online, online_critic_vars = online_critic.build_graph(state=online_state, action=online_action, is_training=True,
                                                             var_scope='current_critic')


    ###############
    ## optimizer ##
    ###############
    optimizer = tf.train.AdamOptimizer(learning_rate=3e-4, epsilon=1e-8)


    ###################################
    ### hard copy ops for first init###
    ###################################
    actor_hard_copy_ops = rl_tools.copy_a2b(online_actor_vars, target_actor_vars)
    critic_hard_copy_ops = rl_tools.copy_a2b(online_critic_vars, target_critic_vars)


    ####################################
    ### Onlien Critic Q(s,a) update ####
    ####################################
    """
    sess.run(online_critic_update, feed_dict={reward:(from memory), whether_end:(from memory), target_state:(next_state), 
            target_action:(next_action from target_actor(next_state)), online_state:(current_state), online_action:(current_action)})
    """
    q_loss = tf.reduce_sum(reward + (1.-whether_end)*gamma*q_target - q_online)
    q_gradients_vars = optimizer.compute_gradients(q_loss, var_list=online_critic_vars)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in q_gradients_vars]  ## clip the gradients
    online_critic_update = optimizer.apply_gradients(capped_gvs)


    #################################
    ### Online Actor π(s) update ####
    #################################
    """
    sess.run(online_actor_update, feed_dict={online_action:(current_action), online_state:(current_state)})
    """
    q_a_gradients = tf.gradients(-q_online, online_action)[0]
    determinstic_policy_gradients_vars = optimizer.compute_gradients(action_online*q_a_gradients, online_actor_vars)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in determinstic_policy_gradients_vars]  ## clip the gradients
    online_actor_update = optimizer.apply_gradients(capped_gvs)


    ###################
    ### soft update ###
    ###################
    actor_soft_copy_ops = rl_tools.soft_copy_a2b(online_actor_vars, target_actor_vars)
    critic_soft_copy_ops = rl_tools.soft_copy_a2b(online_critic_vars, target_critic_vars)


    ##########################
    ### init, saver, ckpt ####
    ##########################
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    logger.info('Tensorflow graph bulid success...')
    ########################### TENSORFLOW GRAPH ######################################

    #### carla world init ####
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)  # seconds
    logger.info('Carla connect success...')

    logger.info('Carla world initing...')
    world = client.get_world()
    ##  spawn vehicles in carla world
    spawn_vehicles(world, n_autopilots=40, n_egopilots=3)

    egopilots = get_all_egopilots(world)

    cameras = []
    lane_invasions = []
    obj_collisions = []
    logger.info('Adding some sensors to egopilots...')
    for egopilot in egopilots:
        ## attach a camera to egopilot ##
        bgr_camera_config['attach_to'] = egopilot
        bgr_sensor = bgr_camera(world, bgr_camera_config)
        cameras.append(bgr_camera)

        ## attach collision sensor to egopilot ##
        collision_sensor_config['attach_to'] = egopilot
        collision_sensor = collision_query(world, collision_sensor_config)
        obj_collisions.append(collision_sensor)

        ## attach line invasion sensor to egopilot ##
        invasion_sensor_config['attach_to'] = egopilot
        lane_invasion_sensor = lane_invasion_query(world, invasion_sensor_config)
        lane_invasions.append(lane_invasion_sensor)
    logger.info('Adding some sensors to egopilots success')

    memory_pool = rl_tools.memory_pooling(maxlen=300)

    actor_list = world.get_actors()
    vehicles = list(actor_list.filter('vehicle*'))
    for vehicle in vehicles:
        vehicle.destroy()
    logger.info('Destroy all vehicles...')

    sensors = list(actor_list.filter('sensor*'))
    for sensor in sensors:
        sensor.destroy()
    logger.info('Destroy all sensors...')
