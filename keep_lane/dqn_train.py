"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
dqn algorithm used for controling the steer to make a vehicle keep lane.

Author：Team Li
"""
import tensorflow as tf
import cv2, math, sys, random, threading

from keep_lane.basic_net.dqn_utils import action_value_net
import RL.rl_utils as rl_tools

try:
    sys.path.append('F:\my_project\driving-desicion-in-carla\dist/carla-0.9.4-py3.7-win-amd64.egg')
    import carla
except:
    raise ImportError('Please check your carla file')
from carla_utils.world_ops import *
from carla_utils.sensor_ops import *

tf.app.flags.DEFINE_string(
    'checkpoint_dir', '../checkpoint/keep_lane_world',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'train_dir', '../checkpoint/',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer(
    'batch_size', 20, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'total_epoches', 2000, 'The number of total epoches.')

tf.app.flags.DEFINE_integer(
    'max_interations', 1000, 'The number of max interations in each epoches')

tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 20,
'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'f_save_step', 4000,
    'The frequency with which summaries are saved, in step.')

tf.app.flags.DEFINE_integer(
    'n_egopilots', 5, 'the number of egopilots')

tf.app.flags.DEFINE_integer(
    'img_height', 416,
    'raw image height captured from carla')

tf.app.flags.DEFINE_integer(
    'img_width', 626,
    'raw image width captured from carla')

tf.app.flags.DEFINE_integer(
    'net_img_height', 224,
    'image height of network input')

tf.app.flags.DEFINE_integer(
    'net_img_width', 224,
    'raw image width of network input')

tf.app.flags.DEFINE_integer(
    'n_action', 21,
    'total discrete action in steer')

tf.app.flags.DEFINE_integer(
    'e_desent_max_step', 50000,
    '')

tf.app.flags.DEFINE_float(
    'e_min_val', 0.01,
    '')

tf.app.flags.DEFINE_integer(
    'target_update_f', 1000,
    '')

FLAGS = tf.app.flags.FLAGS

## carla config ##
semantic_camera_config = {'data_type': 'sensor.camera.semantic_segmentation', 'image_size_x': FLAGS.img_width,
                     'image_size_y': FLAGS.img_height, 'fov': 110, 'sensor_tick': 0.02,
                     'transform': carla.Transform(carla.Location(x=0.5, z=1.6)),
                     'attach_to':None}
bgr_camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': FLAGS.img_width,
                     'image_size_y': FLAGS.img_height, 'fov': 110, 'sensor_tick': 0.02,
                     'transform': carla.Transform(carla.Location(x=0.5, z=1.6)),
                     'attach_to':None}
collision_sensor_config = {'data_type': 'sensor.other.collision','attach_to': None}
invasion_sensor_config = {'data_type': 'sensor.other.lane_detector', 'attach_to': None}
obstacle_sensor_config = {'data_type': 'sensor.other.obstacle', 'sensor_tick': 0.02,
                          'distance': 3, 'attach_to': None}


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


def e_greedy(step, action_index):
    r = random.uniform(0., 1.)
    if step <= FLAGS.e_desent_max_step:
        e = 1. - step*(1-FLAGS.e_min_val)/FLAGS.e_desent_max_step
        if r <= e:
            action_index = random.randint(0, FLAGS.n_action - 1)
            return action_index
        else:
            return action_index
    else:
        if r <= 0.1:
            action_index = random.randint(0, FLAGS.n_action - 1)
            return action_index
        else:
            return action_index


def action_index_2_steer(action_index):
    """ change the action index to steer val
    Args:
        action_index: an int between [0, n_action-1]
    Return:
        a steer val in [-1, 1]
    """
    steer = action_index * 2 / float(FLAGS.n_action - 1) - 1.  ## range is [-1, 1]
    return steer


def single_execuate(target, args):
    """ single thread execuate
    Args:
        target: a func
        args: args in target
    """
    threading.Thread(target=target, args=args).start()


def check_whether_respawn_actors(world, vehicles):
    """check whether to respawn the static acotors in a frequency"""
    while True:
        if carla_actors_static(vehicles, bigger_than=0.75):
            respawn_static_actors(world, vehicles)
        time.sleep(5)


def target_thread(sess, online_begin_signal):
    """a thread for target nets in DDPG"""
    begin = True

    avg_r = 0.

    while True:
        ## get current state
        imgs = []
        for camera_sensor, lane_invasion in zip(cameras, lane_invasions):
            img = camera_sensor.get()
            img = img[int(FLAGS.img_height*1.8//5):, :, :] ## corp the ROI

            img = cv2.resize(img, dsize=(FLAGS.net_img_height, FLAGS.net_img_width))
            # cv2.imshow('test', img)
            imgs.append(img)
            lane_invasion.clear()

        current_img_state = np.array(imgs)
        current_img_state = current_img_state*2./255. - 1.

        ## get current action and control the egopilots
        current_action, current_step = sess.run([max_action_index_online, global_step], feed_dict={online_img_state: current_img_state})

        ## control the egopilots ##
        i = 0
        for egopilot, c_a in zip(egopilots, current_action):
            ## e-greedy
            current_action_index = e_greedy(current_step, c_a)
            current_action[i] = current_action_index

            steer = action_index_2_steer(current_action_index)
            throttle = 0.5
            brake = 0.

            ego_v = egopilot.get_velocity()
            ego_v = math.sqrt(ego_v.x ** 2 + ego_v.y ** 2 + ego_v.z ** 2)
            if ego_v > 8. and throttle > 0.5:
                throttle = 0.5 ## avoid velocity too big

            ## apply control
            egopilot.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            i += 1

        # cv2.waitKey(1000)
        time.sleep(0.5) ## sleep for a while, let the action control the egopilots to next state

        ## reward calculation
        r_s = np.zeros(shape=(len(egopilots)))  ## init is 0 reward
        ## about the velocity and steer
        for i, egopilot in enumerate(egopilots):
            v = egopilot.get_velocity()
            v = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
            #
            # if v <= 6:
            #     r_s[i] += v**2/6.
            # elif v <= 8:
            #     r_s[i] += 3 * (8 - v)
            # else:
            #     r_s[i] -= 2 * (v - 8) ** 2
            #
            # if egopilot.get_control().steer > 0.1 :
            #     r_s[i] = 0

            ## make steer small as possible
            if v >= 0.1: ##m/s
                r_s[i] += (10*(gaussian_r(egopilot.get_control().steer, mu=0., sigma=0.1)) - 5)
            else:
                r_s[i] = 0.

        ## about the collision and lane invasion
        end = np.zeros(len(egopilots)).astype(np.float32)
        i = 0
        for egopilot, lane_invasion, obj_collision in zip(egopilots, lane_invasions, obj_collisions):
            on_collision = obj_collision.get()
            on_invasion = lane_invasion.get()
            # if on_collision:
            #     r_s[i] -= 30
            #     end[i] = 1.

            if on_invasion:
                r_s[i] -= 30
                end[i] = 1.
            i += 1
        # print('a_r:', r_s)

        ## get next state
        imgs = []
        for camera_sensor, egopilot in zip(cameras, egopilots):
            img = camera_sensor.get()
            img = img[int(FLAGS.img_height*1.8//5):, :, :] ## corp the ROI
            img = cv2.resize(img, dsize=(FLAGS.net_img_height, FLAGS.net_img_width))
            imgs.append(img)

        next_img_state = np.array(imgs)
        next_img_state = next_img_state * 2. / 255. - 1.

        ## put the memory in pooling
        for c_img_state, c_action, n_img_state, c_r, end_f in zip(current_img_state,current_action,
                                                                                 next_img_state, r_s, end):
            if c_r > 1.:
                c = 1
            else:
                c = 0

            memory_pool.put(memory=[c_img_state.astype(np.float32), c_action, n_img_state.astype(np.float32),
                                    c_r, end_f])

        ## check whether end.
        for egopilot, lane_invasion, obj_collision in zip(egopilots, lane_invasions, obj_collisions):
            on_collision = obj_collision.get()
            on_invasion = lane_invasion.get()
            # if on_collision:
            #     obj_collision.clear()
            #     single_execuate(target=respawn_actors, args=(world, [egopilot],))

            if on_invasion:
                respawn_actors(world, [egopilot])
                # time.sleep(2.)

        if begin and memory_pool.capacity_bigger_than(val=2000):
            begin = False
            online_begin_signal.set()

        # print(memory_pool.get_propotion())
        if FLAGS.log_every_n_steps != None:
            ## caculate average loss ##
            step = current_step % FLAGS.log_every_n_steps
            avg_r = (avg_r * step + np.mean(np.array(r_s))) / (step + 1.)
            if step == FLAGS.log_every_n_steps - 1:
                logger.info('Step-%s:Reward:%s' % (str(current_step), str(round(avg_r,3))))
                avg_r = 0.



def update_thread(sess, online_begin_signal):
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

        current_img_states = []
        current_actions = []
        next_img_states = []
        current_rewards = []
        end_flags = []
        for a_memory in batch_memorys:
            current_img_states.append(a_memory[0])
            current_actions.append(a_memory[1])
            next_img_states.append(a_memory[2])
            current_rewards.append(a_memory[3])
            end_flags.append(a_memory[4])

        current_img_states = np.array(current_img_states)
        current_actions = np.array(current_actions)
        next_img_states = np.array(next_img_states)
        current_rewards = np.array(current_rewards)
        end_flags = np.array(end_flags)

        q_l, up = sess.run([q_loss, online_update], feed_dict={online_img_state: current_img_states, real_action_index: current_actions,
                                                           reward: current_rewards, target_img_state: next_img_states, whether_end: end_flags,
                                                               lr: FLAGS.learning_rate})

        current_step = sess.run(global_step)
        if FLAGS.log_every_n_steps != None:
            ## caculate average loss ##
            step = current_step % FLAGS.log_every_n_steps
            avg_loss = (avg_loss * step + q_l) / (step + 1.)
            if step == FLAGS.log_every_n_steps - 1:
                logger.info('Step-%s:Q_loss:%s' % (str(current_step), str(round(avg_loss, 3))))
                avg_loss = 0.

        if FLAGS.f_save_step != None:
            step = current_step % FLAGS.f_save_step
            if step == FLAGS.f_save_step - 1:
                ## save model ##
                logger.info('Saving model...')
                model_name = os.path.join(FLAGS.train_dir, 'dqn_keep_lane')
                saver.save(sess, model_name, global_step=current_step)
                logger.info('Save model sucess...')

        if current_step % FLAGS.target_update_f == FLAGS.target_update_f - 1:
            sess.run(update_target_ops)


def vis_memory_thread():
    while True:
        if memory_pool.capacity_bigger_than(val=20):
            #### prepare memory data ####
            batch_memorys = memory_pool.get(batch_size=15)
            ## calculate the norm_rewards and replace raw rewards with them.
            raw_rewards = [m[3] for m in batch_memorys]
            r = rl_tools.normalize_rewards(raw_rewards)
            rl_tools.replace(batch_memorys, r)

            current_img_states = []
            current_actions = []
            next_img_states = []
            current_rewards = []
            end_flags = []
            for a_memory in batch_memorys:
                current_img_states.append(a_memory[0])
                current_actions.append(a_memory[1])
                next_img_states.append(a_memory[2])
                current_rewards.append(a_memory[3])
                end_flags.append(a_memory[4])
            for current_img_state, current_action, next_img_state, current_reward, end_flag, in zip(current_img_states, current_actions,
                                                                                                    next_img_states, current_rewards, end_flags):
                current_img_state = np.array(current_img_state)
                current_action = np.array(current_action)
                next_img_state = np.array(next_img_state)
                current_reward = np.array(current_reward)
                end_flag = np.array(end_flag)

                current_img_state = np.uint8((current_img_state + 1.)*255./2.)
                next_img_state = np.uint8((next_img_state + 1.)*255./2.)
                real_steer = action_index_2_steer(current_action)
                logger.info('end: %s, Current steer is %s, and reward is %s'%(str(end_flag), str(real_steer), str(current_reward)))
                logger.info('------------------------------------------------')
                cv2.imshow('current state', current_img_state)
                cv2.imshow('next state', next_img_state)
                cv2.waitKey()
                cv2.destroyAllWindows()

if __name__ == '__main__':
    online_img_state = tf.placeholder(shape=[None, FLAGS.net_img_height, FLAGS.net_img_width, 3], dtype=tf.float32)
    target_img_state = tf.placeholder(shape=[None, FLAGS.net_img_height, FLAGS.net_img_width, 3], dtype=tf.float32)

    ## other input ##
    reward = tf.placeholder(shape=[None], dtype=tf.float32)
    whether_end = tf.placeholder(shape=[None], dtype=tf.float32)  ##True is end ,False is continue
    real_action_index = tf.placeholder(shape=[None], dtype=tf.int64)
    lr = tf.placeholder(dtype=tf.float32)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    act_val_net_online = action_value_net()
    act_val_online, vars_online = act_val_net_online.build_graph(img_state=online_img_state, n_action=FLAGS.n_action, is_training=True,
                                                                      var_scope='online_act_val')

    act_val_net_target = action_value_net()
    act_val_target, vars_target = act_val_net_online.build_graph(img_state=online_img_state, n_action=21,
                                                                      is_training=True,
                                                                      var_scope='target_act_val')
    #########################################
    ## the best action ops in current step ##
    #########################################
    max_action_index_online = tf.argmax(act_val_online, axis=-1)
    max_action_index_target = tf.argmax(act_val_target, axis=-1)

    ###################################
    ### hard copy ops for first init###
    ###################################
    update_target_ops = rl_tools.copy_a2b(vars_a=vars_online, vars_b=vars_target)

    ###########
    ## q loss##
    ###########
    max_q_val_target = tf.reduce_sum(act_val_target * tf.one_hot(max_action_index_target, FLAGS.n_action), axis=-1) ## need img_state_target
    q_val_online = tf.reduce_sum(act_val_online * tf.one_hot(real_action_index, FLAGS.n_action), axis=-1) ## need img_state_online, real_action_index
    q_loss = tf.reduce_mean(tf.square(reward + (1.-whether_end)*max_q_val_target - q_val_online)) ## need reward,  whether_end

    ###############
    ## update #####
    ############
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer_for_online = tf.train.RMSPropOptimizer(learning_rate=lr)
        q_gradients_vars = optimizer_for_online.compute_gradients(q_loss, var_list=vars_online)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in q_gradients_vars]  ## clip the gradients
        online_update = optimizer_for_online.apply_gradients(capped_gvs, global_step=global_step)

    ####
    ## sess.run([q_loss, online_update], feed_dict={img_state_online: current_img, real_action_index: current_action, reward: current_reward,
    #                                       img_state_target: next_img, whether_end: end_flags})

    ##########################
    ### init, saver, ckpt ####
    ##########################
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
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
    # spawn_points = list(world.get_map().get_spawn_points())
    # spawn_egopilot_at(world, spawn_points[45])
    spawn_vehicles(world, n_autopilots=0, n_egopilots=FLAGS.n_egopilots)
    time.sleep(2)  ## sometimes unstale

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

    memory_pool = rl_tools.memory_pooling(maxlen=4000)
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
            sess.run(update_target_ops)
            logger.info('DQN all network variables init success...')

        check_t = threading.Thread(target=check_whether_respawn_actors, args=(world, autopilots + egopilots,))
        target_t = threading.Thread(target=target_thread, args=(sess, online_begin_signal,))
        # # respwan_v_t = threading.Thread(target=respwan_vehicles_in_traffic_light)
        update_t = threading.Thread(target=update_thread, args=(sess, online_begin_signal,))

        target_t.daemon = True
        check_t.daemon = True
        update_t.daemon = True

        check_t.start()
        # # respwan_v_t.start()
        target_t.start()
        update_t.start()
        # vis_memory_thread()
        while True:
            pass