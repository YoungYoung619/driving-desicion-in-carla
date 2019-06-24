"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
train a ddpg model for driving decision.

Author：Team Li
"""
import sys, os, glob, threading, math, time
import tensorflow as tf

import RL.DDPG.ddpg_utils as ddpg
import RL.rl_utils as rl_tools

from multiprocessing import Process

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

tf.app.flags.DEFINE_integer(
    'batch_size', 30, 'The number of samples in each batch.')

tf.app.flags.DEFINE_float('learning_rate', 3e-4, 'Initial learning rate.')

FLAGS = tf.app.flags.FLAGS

## carla config ##
semantic_camera_config = {'data_type': 'sensor.camera.semantic_segmentation', 'image_size_x': 209,
                     'image_size_y': 139, 'fov': 110, 'sensor_tick': 0.02,
                     'transform': carla.Transform(carla.Location(x=0.8, z=1.7)),
                     'attach_to':None}
collision_sensor_config = {'data_type': 'sensor.other.collision','attach_to': None}
invasion_sensor_config = {'data_type': 'sensor.other.lane_detector', 'attach_to': None}
obstacle_sensor_config = {'data_type': 'sensor.other.obstacle', 'sensor_tick': 0.02,
                          'distance': 3, 'attach_to': None}

def gaussian_r(val, mu=65., sigma=30.):
    """calculate the reward of velocity
    Args:
        vel: velocity, km/h
    Return:
        a reward
    """
    # if vel > 80:
    #     return 5.

    r = math.exp(-((val - mu) ** 2) / (2 * sigma ** 2))
    return 5 * r


def check_whether_respawn_actors(world, vehicles):
    """check whether to respawn the static acotors in a frequency"""
    while True:
        if carla_actors_static(vehicles, bigger_than=0.8):
            respawn_static_actors(world, vehicles)
        time.sleep(20)


def single_execuate(target, args):
    threading.Thread(target=target, args=args).start()


def target_thread(sess, online_begin_signal):
    """a thread for target nets in DDPG"""
    begin = True
    while True:
        t0 = time.time()
        ## get current state
        imgs = []
        for camera_sensor in cameras:
            img = camera_sensor.get()
            imgs.append(img)
        current_state = np.array(imgs)
        current_state = current_state*2./255. - 1.

        ## get current action and control the egopilots
        current_action = sess.run(action_target, feed_dict={target_state: current_state})

        std_steers = []
        ## control the egopilots ##
        for egopilot, c_a in zip(egopilots, current_action):
            egopilot.set_autopilot(enabled=True)
            std_steer = egopilot.get_control().steer
            std_steers.append(std_steer)
            egopilot.set_autopilot(enabled=False)

            steer = float(c_a[0])
            throttle = float(c_a[1])
            brake = float(c_a[2])

            throttle += random.uniform(-0.2, 0.2)
            if throttle > 1.:
                throttle = 1.
            elif throttle < 0.:
                throttle = 0.

            steer += random.uniform(-0.2, 0.2)
            if steer > 1.:
                steer = 1.
            elif steer < -1.:
                steer = -1.

            egopilot.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0.))
        time.sleep(0.2) ## sleep for a while, let the action control the egopilots to next state

        ## reward calculation
        r_s = np.zeros(shape=(len(egopilots)))
        v_s = []
        for i, egopilot in enumerate(egopilots):
            v = egopilot.get_velocity()
            v = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) ##km/h
            v_s.append(v)
            r_s[i] += gaussian_r(v, mu=60., sigma=30.)
            r_s[i] += gaussian_r(current_action[i][0], mu=std_steers[i], sigma=0.1)/5.
        # print('std_steer:', std_steers)
        # print('ours:', current_action[:,0])
        # print('phsic_r:', r_s)

        end = np.zeros(len(egopilots)).astype(np.float32)
        i = 0
        for egopilot, v, lane_invasion, obj_collision, obstacle_ahead in zip(egopilots, v_s, lane_invasions, obj_collisions, obstacle_aheads):
            on_collision = obj_collision.get()
            on_invasion = lane_invasion.get()
            has_obstacle_ahead = obstacle_ahead.get()

            if on_collision:
                r_s[i] -= 10.
                end[i] = 1.
                single_execuate(target=respawn_actors, args=(world, [egopilot],))
                obj_collision.clear()
            elif on_invasion:
                r_s[i] -= 7.
                # respawn_actors(world, [autopilots[0]])
                lane_invasion.clear()
            elif has_obstacle_ahead:
                if v < 5:
                    r_s[i] -= 5.
                obstacle_ahead.clear()
            elif not has_obstacle_ahead:
                if v < 5:
                    r_s[i] -= 7.
            i += 1

        # print(round(time.time()-t0, 2))
        # print('final:',r_s)
        # print('---')

        ## get next state
        imgs = []
        for camera_sensor in cameras:
            img = camera_sensor.get()
            imgs.append(img)
        next_state = np.array(imgs)
        next_state = next_state * 2. / 255. - 1.

        ## put the memory in pooling
        for c_state, c_action, n_state, c_r, end_f, std_s in zip(current_state, current_action, next_state, r_s, end, std_steers):
            memory_pool.put(memory=[c_state.astype(np.float32), c_action, n_state.astype(np.float32), c_r, end_f, std_s])

        if begin and memory_pool.capacity_bigger_than(FLAGS.batch_size):
            begin = False
            online_begin_signal.set()


def online_thread(sess, online_begin_signal):
    """update the online net thread"""
    online_begin_signal.wait()
    logger.info('Begin online nets...')
    while True:
        #### prepare memory data ####
        batch_memorys = memory_pool.get(batch_size=FLAGS.batch_size)
        ## calculate the norm_rewards and replace raw rewards with them.
        # raw_rewards = [m[3] for m in batch_memorys]
        # r = rl_tools.normalize_rewards(raw_rewards)
        # rl_tools.replace(batch_memorys, r)

        current_state = []
        current_action = []
        next_state = []
        current_reward = []
        end_flag = []
        std_steer_ops = []
        for a_memory in batch_memorys:
            current_state.append(a_memory[0])
            current_action.append(a_memory[1])
            next_state.append(a_memory[2])
            current_reward.append(a_memory[3])
            end_flag.append(a_memory[4])
            std_steer_ops.append(a_memory[5])

        current_state = np.array(current_state)
        current_action = np.array(current_action)
        next_state = np.array(next_state)
        current_reward = np.array(current_reward)
        end_flag = np.array(end_flag)
        std_steer_ops = np.array(std_steer_ops)
        # print(current_reward[:10])

        ## update the Online Critic Q(s,a) ##
        next_action = sess.run(action_target, feed_dict={target_state:next_state})
        sess.run(online_critic_update, feed_dict={reward: current_reward, whether_end: end_flag, target_state: next_state,
                                                  target_action: next_action, online_state: current_state,
                                                  online_action: current_action, lr:FLAGS.learning_rate})


        ## update the Online Actor π(s) ##
        sess.run(online_actor_update, feed_dict={online_state: current_state, online_action: current_action, lr:FLAGS.learning_rate})

        ## update steer action ##
        sess.run(online_actor_steer_update, feed_dict={online_action: current_action, online_state: current_state, std_steer: std_steer_ops,
                                                 lr:100.*FLAGS.learning_rate})

        ## soft update the Online nets to Target nets
        sess.run([actor_soft_copy_ops, critic_soft_copy_ops])
        # logger.info('ones')
        pass


if __name__ == '__main__':
    ########################### TENSORFLOW GRAPH ######################################
    logger.info('Tensorflow graph buliding...')
    ## target input ##
    target_state = tf.placeholder(shape=[None, 139, 209, 3], dtype=tf.float32)
    target_action = tf.placeholder(shape=[None, 3], dtype=tf.float32) ## steer, accel, brake

    ## online input ##
    online_state = tf.placeholder(shape=[None, 139, 209, 3], dtype=tf.float32)
    online_action = tf.placeholder(shape=[None, 3], dtype=tf.float32)

    ## other input ##
    reward = tf.placeholder(shape=[None], dtype=tf.float32)
    whether_end = tf.placeholder(shape=[None], dtype=tf.float32) ##True is end ,False is continue
    std_steer = tf.placeholder(shape=[None], dtype=tf.float32)

    lr = tf.placeholder(dtype=tf.float32)

    gamma = 0.98 ##for gamma discount reward
    tau = 1e-2 ##for soft update

    ## action range config ##
    action_range = [[-1.,1.], [0.,1.], [0.,1.]] ## steer, accel, brake range

    ###############
    ## target AC ##
    ###############
    target_actor = ddpg.actor()
    action_target, target_actor_vars = target_actor.build_graph(state=target_state, n_action_space=3, is_training=True,
                                                                action_range=action_range, var_scope='target_actor')

    target_critic = ddpg.critic()
    q_target, target_critic_vars = target_critic.build_graph(state=target_state, action=target_action, is_training=True,
                                                             var_scope='target_critic')


    ###############
    ## online AC ##
    ###############
    online_actor = ddpg.actor()
    action_online, online_actor_vars = online_actor.build_graph(state=online_state, n_action_space=3, is_training=True,
                                                                action_range=action_range, var_scope='online_actor')

    online_critic = ddpg.critic()
    q_online, online_critic_vars = online_critic.build_graph(state=online_state, action=online_action, is_training=True,
                                                             var_scope='online_critic')

    ###################################
    ### hard copy ops for first init###
    ###################################
    actor_hard_copy_ops = rl_tools.copy_a2b(online_actor_vars, target_actor_vars)
    critic_hard_copy_ops = rl_tools.copy_a2b(online_critic_vars, target_critic_vars)

    ###################
    ### soft update ###
    ###################
    actor_soft_copy_ops = rl_tools.soft_copy_a2b(online_actor_vars, target_actor_vars)
    critic_soft_copy_ops = rl_tools.soft_copy_a2b(online_critic_vars, target_critic_vars)


    ###############
    ## optimizer ##
    ###############
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    target_actor_bn_ops = []
    target_critic_bn_ops = []
    online_actor_bn_ops = []
    online_critic_bn_ops = []
    for update_op in update_ops:
        if update_op.name.startswith('target_actor'):
            target_actor_bn_ops.append(update_op)
        elif update_op.name.startswith('target_critic'):
            target_critic_bn_ops.append(update_op)
        elif update_op.name.startswith('online_actor'):
            online_actor_bn_ops.append(update_op)
        elif update_op.name.startswith('online_critic'):
            online_critic_bn_ops.append(update_op)

    ####################################
    ### Onlien Critic Q(s,a) update ####
    ####################################
    """
        sess.run(online_critic_update, feed_dict={reward:(from memory), whether_end:(from memory), target_state:(next_state), 
                target_action:(next_action from target_actor(next_state)), online_state:(current_state), online_action:(current_action)})
    """
    online_critic_update_ops = target_critic_bn_ops + online_critic_bn_ops
    with tf.control_dependencies(online_critic_update_ops):
        optimizer_for_online_critic = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
        q_loss = tf.reduce_sum(reward + (1.-whether_end)*gamma*q_target - q_online)
        q_gradients_vars = optimizer_for_online_critic.compute_gradients(q_loss, var_list=online_critic_vars)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in q_gradients_vars]  ## clip the gradients
        online_critic_update = optimizer_for_online_critic.apply_gradients(capped_gvs)


    #################################
    ### Online Actor π(s) update ####
    #################################
    """
        sess.run(online_actor_update, feed_dict={online_action:(current_action), online_state:(current_state)})
    """
    online_actor_update_ops = online_critic_bn_ops + online_actor_bn_ops
    with tf.control_dependencies(online_actor_update_ops):
        optimizer_for_online_actor = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
        q_a_gradients = tf.gradients(-q_online, online_action)[0]
        determinstic_policy_gradients_vars = optimizer_for_online_actor.compute_gradients(action_online*q_a_gradients, online_actor_vars)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in determinstic_policy_gradients_vars]  ## clip the gradients
        online_actor_update = optimizer_for_online_actor.apply_gradients(capped_gvs)


    ########################################
    ### Online Actor π(s) steer update ####
    ########################################
    """
        sess.run(online_actor_update, feed_dict={online_action:(current_action), online_state:(current_state), std_steer:~~})
    """
    with tf.control_dependencies(online_actor_bn_ops):
        optimizer_for_imitate = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
        steer_online = action_online[:,0]
        steer_loss = tf.reduce_sum(steer_online-std_steer)
        online_actor_steer_update = optimizer_for_imitate.minimize(steer_loss)


    ##########################
    ### init, saver, ckpt ####
    ##########################
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    logger.info('Tensorflow graph bulid success...')
    logger.info('Total trainable parameters:%s' %
                str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    ########################### TENSORFLOW GRAPH ######################################

    #### carla world init ####
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)  # seconds
    logger.info('Carla connect success...')

    logger.info('Carla world initing...')
    world = client.get_world()
    destroy_all_actors(world)

    ##  spawn vehicles in carla world
    spawn_vehicles(world, n_autopilots=40, n_egopilots=5)
    time.sleep(2) ## sometimes unstale

    autopilots = get_all_autopilots(world)
    egopilots = get_all_egopilots(world)

    cameras = []
    lane_invasions = []
    obj_collisions = []
    # obstacle_aheads = []
    logger.info('Adding some sensors to egopilots...')
    for egopilot in egopilots:
        ## attach a camera to egopilot ##
        semantic_camera_config['attach_to'] = egopilot
        semantic_sensor = semantic_camera(world, semantic_camera_config)
        cameras.append(semantic_sensor)

        ## attach collision sensor to egopilot ##
        collision_sensor_config['attach_to'] = egopilot
        collision_sensor = collision_query(world, collision_sensor_config)
        obj_collisions.append(collision_sensor)

        ## attach line invasion sensor to egopilot ##
        invasion_sensor_config['attach_to'] = egopilot
        lane_invasion_sensor = lane_invasion_query(world, invasion_sensor_config)
        lane_invasions.append(lane_invasion_sensor)

        # ## attach obstacle sensor to egopilot
        # obstacle_sensor_config['attach_to'] = egopilot
        # obstacle_sensor = obstacle_ahead_query(world, obstacle_sensor_config)
        # obstacle_aheads.append(obstacle_sensor)
    logger.info('Adding some sensors to egopilots success')

    memory_pool = rl_tools.memory_pooling(maxlen=500)
    online_begin_signal = threading.Event()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        if ckpt:
            logger.info('loading %s...' % str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('Load checkpoint success...')
        else:
            sess.run(init)
            sess.run([actor_hard_copy_ops, critic_hard_copy_ops])
            logger.info('DDPG all network variables init success...')

        check_t = threading.Thread(target=check_whether_respawn_actors, args=(world, autopilots,))
        target_t = threading.Thread(target=target_thread, args=(sess,online_begin_signal,))
        online_t = threading.Thread(target=online_thread, args=(sess,online_begin_signal,))

        # target_t.daemon = True
        # check_t.daemon = True
        # online_t.daemon = True

        check_t.start()
        target_t.start()
        online_t.start()
        while True:
            pass
    # destroy_all_actors(world)
