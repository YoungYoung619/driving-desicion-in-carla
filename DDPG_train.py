"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
train a ddpg model for driving decision.

Author：Team Li
"""
import sys, os, glob, threading, math, time
import tensorflow as tf
import cv2

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
    'imitator_checkpoint_dir', '',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'train_dir', './checkpoint/',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Initial learning rate.')

tf.app.flags.DEFINE_integer(
    'img_height', 416,
    'raw image height')

tf.app.flags.DEFINE_integer(
    'img_width', 626,
    'raw image width')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 20,
'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'n_egopilots', 1,
'the number of egopilots')

FLAGS = tf.app.flags.FLAGS

## carla config ##
semantic_camera_config = {'data_type': 'sensor.camera.semantic_segmentation', 'image_size_x': FLAGS.img_width,
                     'image_size_y': FLAGS.img_height, 'fov': 110, 'sensor_tick': 0.02,
                     'transform': carla.Transform(carla.Location(x=0., z=1.7)),
                     'attach_to':None}
bgr_camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': FLAGS.img_width,
                     'image_size_y': FLAGS.img_height, 'fov': 110, 'sensor_tick': 0.02,
                     'transform': carla.Transform(carla.Location(x=0., z=1.7)),
                     'attach_to':None}
collision_sensor_config = {'data_type': 'sensor.other.collision','attach_to': None}
invasion_sensor_config = {'data_type': 'sensor.other.lane_detector', 'attach_to': None}
obstacle_sensor_config = {'data_type': 'sensor.other.obstacle', 'sensor_tick': 0.02,
                          'distance': 3, 'attach_to': None}

light_state_encode = {'Green':np.array([1., 0., 0., 0.]).astype(np.float32),
                      'Red':np.array([0., 1., 0., 0.]).astype(np.float32),
                      'Yellow':np.array([0., 0., 1., 0.]).astype(np.float32),
                      'Unkown':np.array([0., 0., 0., 1.]).astype(np.float32)}

def gaussian_r(val, mu=30., sigma=10.):
    """calculate the reward of velocity
    Args:
        vel: velocity, km/h
    Return:
        a reward
    """
    # if vel > 80:
    #     return 5.

    r = math.exp(-((val - mu) ** 2) / (2 * sigma ** 2))
    return r


def check_whether_respawn_actors(world, vehicles):
    """check whether to respawn the static acotors in a frequency"""
    while True:
        if carla_actors_static(vehicles, bigger_than=0.65):
            respawn_static_actors(world, vehicles)
        time.sleep(20)


def single_execuate(target, args):
    threading.Thread(target=target, args=args).start()


def target_thread(sess, online_begin_signal):
    """a thread for target nets in DDPG"""
    begin = True

    whether_wait_red_light = {}
    for egopilot in egopilots:
        whether_wait_red_light[egopilot] = False

    exploration_noise = rl_tools.exploration_noise(theta=10.)

    avg_r = 0.
    episode = 0

    prev_prev_contrl = np.zeros(shape=[FLAGS.n_egopilots, 3])
    prev_contrl = np.zeros(shape=[FLAGS.n_egopilots, 3])
    while True:
        ## get current state
        imgs = []
        other_states = []
        for camera_sensor, egopilot in zip(cameras, egopilots):
            img = camera_sensor.get()
            img = img[int(FLAGS.img_height*1.8//5):, :, :] ## corp the ROI

            img = cv2.resize(img, dsize=(418, 418))
            cv2.imshow('test', img)
            imgs.append(img)

            if egopilot.is_at_traffic_light() and not whether_wait_red_light[egopilot]:
                ## mean first go into traffic light area
                if str(egopilot.get_traffic_light_state()) == 'Green':
                    pass
                elif str(egopilot.get_traffic_light_state()) == 'Red':
                    whether_wait_red_light[egopilot] = True
                light_state = light_state_encode[str(egopilot.get_traffic_light_state())]
            elif whether_wait_red_light[egopilot]:
                if str(egopilot.get_traffic_light_state()) == 'Green':
                    whether_wait_red_light[egopilot] = False
                elif str(egopilot.get_traffic_light_state()) == 'Red':
                    pass
                light_state = light_state_encode[str(egopilot.get_traffic_light_state())]
            else:
                light_state = light_state_encode['Unkown']
            ego_v = egopilot.get_velocity()
            ego_v = math.sqrt(ego_v.x ** 2 + ego_v.y ** 2 + ego_v.z ** 2)
            ego_v_ = ego_v / 10. ## normally, max velocity is about 10m/s
            other_states.append(np.concatenate([light_state, np.array([ego_v_])], axis=-1))

        current_img_state = np.array(imgs)
        current_img_state = current_img_state*2./255. - 1.
        current_other_states = np.array(other_states)

        ## get current action and control the egopilots
        current_action = sess.run(action_target, feed_dict={target_img_state: current_img_state, target_other_state:current_other_states})

        ## soft
        current_action = 0.4 * current_action + 0.3 * prev_contrl + 0.3 * prev_prev_contrl

        ## control the egopilots ##
        for egopilot, c_a in zip(egopilots, current_action):
            ## add exploration noise
            c_a = np.clip(c_a+np.expand_dims(exploration_noise.generate(episode//FLAGS.n_egopilots), axis=0), a_min=[-1., 0., 0.], a_max=[1., 1., 1.])[0]

            steer = float(c_a[0])
            throttle = float(c_a[1])
            brake = float(c_a[2])

            ## avoid carla bug
            if throttle >= brake:
                brake = 0.
            else:
                throttle = 0.

            ego_v = egopilot.get_velocity()
            ego_v = math.sqrt(ego_v.x ** 2 + ego_v.y ** 2 + ego_v.z ** 2)
            if ego_v > 8. and throttle > 0.5:
                throttle = 0.5 ## avoid velocity too big

            ## apply control
            egopilot.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            c_a[0] = steer
            c_a[1] = throttle
            c_a[2] = brake
        prev_prev_contrl = prev_contrl
        prev_contrl = current_action
        cv2.waitKey(100)
        # time.sleep(0.1) ## sleep for a while, let the action control the egopilots to next state

        ## reward calculation
        r_s = np.zeros(shape=(len(egopilots)))
        v_s = []
        ## about the velocity
        for i, egopilot in enumerate(egopilots):
            v = egopilot.get_velocity()
            v = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

            if v <= 6:
                r_s[i] += v**2/6.
                r_s[i] += 6*(gaussian_r(egopilot.get_control().steer, mu=0., sigma=0.2))
            elif v <= 8:
                r_s[i] += 3 * (8 - v)
            else:
                r_s[i] -= 2 * (v - 8) ** 2

        ## about the traffic light
        i = 0
        for egopilot in egopilots:
            if whether_wait_red_light[egopilot]:
                if str(egopilot.get_traffic_light_state()) == 'Green':
                    if egopilot.get_control().brake <= egopilot.get_control().throttle:
                        r_s[i] += 10
                    else:
                        r_s[i] -= 10
                elif str(egopilot.get_traffic_light_state()) == 'Red':
                    ## still red
                    if egopilot.get_control().brake <= egopilot.get_control().throttle:
                        r_s[i] -= 10
                    else:
                        r_s[i] += 10
            else:
                pass
            i += 1
        # print('l_r:',r_s)

        ## about the collision and lane invasion
        end = np.zeros(len(egopilots)).astype(np.float32)
        i = 0
        for egopilot, lane_invasion, obj_collision in zip(egopilots, lane_invasions, obj_collisions):
            on_collision = obj_collision.get()
            on_invasion = lane_invasion.get()
            if on_collision:
                r_s[i] -= 30
                end[i] = 1.
                episode += 1
                single_execuate(target=respawn_actors, args=(world, [egopilot],))
                obj_collision.clear()
            elif on_invasion:
                r_s[i] -= 15
                lane_invasion.clear()
            i += 1
        print('a_r:', r_s)

        ## get next state
        imgs = []
        other_states = []
        for camera_sensor, egopilot in zip(cameras, egopilots):
            img = camera_sensor.get()
            img = img[int(FLAGS.img_height*1.8//5):, :, :] ## corp the ROI
            img = cv2.resize(img, dsize=(418, 418))
            imgs.append(img)

            if egopilot.is_at_traffic_light() and not whether_wait_red_light[egopilot]:
                ## mean first go into traffic light area
                if str(egopilot.get_traffic_light_state()) == 'Green':
                    pass
                elif str(egopilot.get_traffic_light_state()) == 'Red':
                    # print('Red, Stop')
                    whether_wait_red_light[egopilot] = True
                light_state = light_state_encode[str(egopilot.get_traffic_light_state())]
            elif whether_wait_red_light[egopilot]:
                if str(egopilot.get_traffic_light_state()) == 'Green':
                    whether_wait_red_light[egopilot] = False
                elif str(egopilot.get_traffic_light_state()) == 'Red':
                    pass
                light_state = light_state_encode[str(egopilot.get_traffic_light_state())]
            else:
                light_state = light_state_encode['Unkown']

            ego_v = egopilot.get_velocity()
            ego_v_ = math.sqrt(ego_v.x ** 2 + ego_v.y ** 2 + ego_v.z ** 2)
            ego_v = ego_v_ / 10.
            other_states.append(np.concatenate([light_state, np.array([ego_v])], axis=-1))

        next_img_state = np.array(imgs)
        next_img_state = next_img_state * 2. / 255. - 1.
        next_other_state = np.array(other_states)

        ## put the memory in pooling
        for c_img_state, c_other_state, c_action, n_img_state, n_other_state, c_r, end_f in zip(current_img_state, current_other_states,
                                                                                                       current_action, next_img_state,
                                                                                                       next_other_state, r_s, end):
            memory_pool.put(memory=[c_img_state.astype(np.float32), c_other_state, c_action, n_img_state.astype(np.float32), n_other_state,
                                    c_r, end_f])

        if begin and memory_pool.capacity_bigger_than(FLAGS.batch_size):
            begin = False
            online_begin_signal.set()

        current_step = sess.run(global_step)
        if FLAGS.log_every_n_steps != None:
            ## caculate average loss ##
            step = current_step % FLAGS.log_every_n_steps
            avg_r = (avg_r * step + np.mean(np.array(r_s))) / (step + 1.)
            if step == FLAGS.log_every_n_steps - 1:
                logger.info('Step-%s:Reward:%s' % (str(current_step), str(round(avg_r,3))))


def online_thread(sess, online_begin_signal):
    """update the online net thread"""
    online_begin_signal.wait()
    logger.info('Begin online nets...')
    avg_loss = 0.
    while True:
        #### prepare memory data ####
        batch_memorys = memory_pool.get(batch_size=FLAGS.batch_size)
        ## calculate the norm_rewards and replace raw rewards with them.
        # raw_rewards = [m[3] for m in batch_memorys]
        # r = rl_tools.normalize_rewards(raw_rewards)
        # rl_tools.replace(batch_memorys, r)

        current_img_state = []
        current_other_state = []
        current_action = []
        next_img_state = []
        next_other_state = []
        current_reward = []
        end_flag = []
        for a_memory in batch_memorys:
            current_img_state.append(a_memory[0])
            current_other_state.append(a_memory[1])
            current_action.append(a_memory[2])
            next_img_state.append(a_memory[3])
            next_other_state.append(a_memory[4])
            current_reward.append(a_memory[5])
            end_flag.append(a_memory[6])

        current_img_state = np.array(current_img_state)
        current_other_state = np.array(current_other_state)
        current_action = np.array(current_action)
        next_img_state = np.array(next_img_state)
        next_other_state = np.array(next_other_state)
        current_reward = np.array(current_reward)
        end_flag = np.array(end_flag)
        # print(current_reward[:10])

        ## update the Online Critic Q(s,a) ##
        next_action = sess.run(action_target, feed_dict={target_img_state:next_img_state, target_other_state:next_other_state})
        up, q_l = sess.run([online_critic_update, q_loss], feed_dict={reward: current_reward, whether_end: end_flag, target_img_state: next_img_state, target_other_state:next_other_state,
                                                  target_action: next_action, online_img_state: current_img_state, online_other_state:current_other_state,
                                                  online_action: current_action, lr:FLAGS.learning_rate})


        ## update the Online Actor π(s) ##
        sess.run(online_actor_update, feed_dict={online_img_state: current_img_state, online_other_state:current_other_state, online_action: current_action,
                                                 lr:FLAGS.learning_rate})

        ## update steer action ##
        # sess.run(online_actor_steer_update, feed_dict={online_action: current_action, online_img_state: current_state, std_steer: std_steer_ops,
        #                                          lr:100.*FLAGS.learning_rate})

        ## soft update the Online nets to Target nets
        sess.run([actor_soft_copy_ops, critic_soft_copy_ops])
        # logger.info('ones')

        current_step = sess.run(global_step)
        if FLAGS.log_every_n_steps != None:
            ## caculate average loss ##
            step = current_step % FLAGS.log_every_n_steps
            avg_loss = (avg_loss * step + q_l) / (step + 1.)
            if step == FLAGS.log_every_n_steps - 1:
                logger.info('Step-%s:Q_loss:%s' % (str(current_step), str(round(avg_loss, 3))))
        pass


if __name__ == '__main__':
    ########################### TENSORFLOW GRAPH ######################################
    logger.info('Tensorflow graph buliding...')
    ## target input ##
    target_img_state = tf.placeholder(shape=[None, 418, 418, 3], dtype=tf.float32)
    target_other_state = tf.placeholder(shape=[None, 5], dtype=tf.float32)
    target_action = tf.placeholder(shape=[None, 3], dtype=tf.float32) ## steer, accel, brake

    ## online input ##
    online_img_state = tf.placeholder(shape=[None, 418, 418, 3], dtype=tf.float32)
    online_other_state = tf.placeholder(shape=[None, 5], dtype=tf.float32)
    online_action = tf.placeholder(shape=[None, 3], dtype=tf.float32)

    ## other input ##
    reward = tf.placeholder(shape=[None], dtype=tf.float32)
    whether_end = tf.placeholder(shape=[None], dtype=tf.float32) ##True is end ,False is continue
    std_steer = tf.placeholder(shape=[None], dtype=tf.float32)

    lr = tf.placeholder(dtype=tf.float32)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    gamma = 0.98 ##for gamma discount reward
    tau = 1e-2 ##for soft update

    ## action range config ##
    action_range = [[-1.,1.], [0.,1.], [0.,1.]] ## steer, accel, brake range

    ###############
    ## target AC ##
    ###############
    target_actor = ddpg.actor()
    action_target, target_actor_vars = target_actor.build_graph(img_state=target_img_state, other_state=target_other_state, n_action_space=3, is_training=True,
                                                                action_range=action_range, var_scope='target_actor')

    target_critic = ddpg.critic()
    q_target, target_critic_vars = target_critic.build_graph(img_state=target_img_state, other_state=target_other_state, action=target_action, is_training=True,
                                                             var_scope='target_critic')


    ###############
    ## online AC ##
    ###############
    online_actor = ddpg.actor()
    action_online, online_actor_vars = online_actor.build_graph(img_state=online_img_state, other_state=online_other_state, n_action_space=3, is_training=True,
                                                                action_range=action_range, var_scope='online_actor')

    online_critic = ddpg.critic()
    q_online, online_critic_vars = online_critic.build_graph(img_state=online_img_state, other_state=online_other_state, action=online_action, is_training=True,
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
        q_loss = tf.reduce_sum(tf.square(reward + (1.-whether_end)*gamma*q_target - q_online))
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
        q_a_gradients = tf.gradients(q_online, online_action)[0]
        determinstic_policy_gradients_vars = optimizer_for_online_actor.compute_gradients(tf.reduce_sum(-action_online*q_a_gradients), online_actor_vars)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in determinstic_policy_gradients_vars]  ## clip the gradients
        online_actor_update = optimizer_for_online_actor.apply_gradients(capped_gvs, global_step=global_step)


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
    imitator_ckpt = tf.train.get_checkpoint_state(FLAGS.imitator_checkpoint_dir)
    logger.info('Tensorflow graph bulid success...')
    logger.info('Total trainable parameters:%s' %
                str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

    imitator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope="online_actor.+")  ##filter the refine model's vars
    imitator_restore_saver = tf.train.Saver(imitator_train_vars)
    ########################### TENSORFLOW GRAPH ######################################

    #### carla world init ####
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)  # seconds
    logger.info('Carla connect success...')

    logger.info('Carla world initing...')
    world = client.get_world()
    destroy_all_actors(world)

    ##  spawn vehicles in carla world
    spawn_vehicles(world, n_autopilots=0, n_egopilots=FLAGS.n_egopilots)
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
        # semantic_camera_config['attach_to'] = egopilot
        # semantic_sensor = semantic_camera(world, semantic_camera_config)
        # cameras.append(semantic_sensor)

        bgr_camera_config['attach_to'] = egopilot
        bgr_sensor = bgr_camera(world, bgr_camera_config)
        cameras.append(bgr_sensor)

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

    memory_pool = rl_tools.memory_pooling(maxlen=1000)
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

            if imitator_ckpt:
                imitator_restore_saver.restore(sess, imitator_ckpt.model_checkpoint_path)
                logger.info('load base imitator success...')

            sess.run([actor_hard_copy_ops, critic_hard_copy_ops])
            logger.info('DDPG all network variables init success...')

        check_t = threading.Thread(target=check_whether_respawn_actors, args=(world, autopilots+egopilots,))
        target_t = threading.Thread(target=target_thread, args=(sess,online_begin_signal,))
        online_t = threading.Thread(target=online_thread, args=(sess,online_begin_signal,))

        # target_t.daemon = True
        # check_t.daemon = True
        # online_t.daemon = True

        # check_t.start()
        target_t.start()
        # online_t.start()
        while True:
            pass
    # destroy_all_actors(world)
