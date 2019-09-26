"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
dqn algorithm used for controling the steer to make a vehicle keep lane.

Author：Team Li
"""
import tensorflow as tf
import cv2, math, sys, random, threading

from efficient_driving.basic_net.dqn_utils import action_value_net
import RL.rl_utils as rl_tools

from efficient_driving.scenario_generate_utils import *

try:
    sys.path.append('F:\my_project\driving-desicion-in-carla\dist/carla-0.9.4-py3.7-win-amd64.egg')
    import carla
except:
    raise ImportError('Please check your carla file')
from carla_utils.world_ops import *
from carla_utils.sensor_ops import *

from heat_map import produce_heat_map

tf.app.flags.DEFINE_string(
    'checkpoint_dir', '../checkpoint',
    'The path to a checkpoint from which to fine-tune.')


tf.app.flags.DEFINE_integer(
    'img_height', 416,
    'raw image height captured from carla')

tf.app.flags.DEFINE_integer(
    'img_width', 626,
    'raw image width captured from carla')

tf.app.flags.DEFINE_integer(
    'net_img_height', 200,
    'image height of network input')

tf.app.flags.DEFINE_integer(
    'net_img_width', 200,
    'raw image width of network input')

tf.app.flags.DEFINE_integer(
    'n_action', 7,
    'total discrete action in steer')


FLAGS = tf.app.flags.FLAGS

## carla config ##
semantic_camera_config = {'data_type': 'sensor.camera.semantic_segmentation', 'image_size_x': FLAGS.img_width,
                     'image_size_y': FLAGS.img_height, 'fov': 110, 'sensor_tick': 0.02,
                     'transform': carla.Transform(carla.Location(x=0.5, z=1.6)),
                     'attach_to':None}
# bgr_camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': FLAGS.img_width,
#                      'image_size_y': FLAGS.img_height, 'fov': 110, 'sensor_tick': 0.02,
#                      'transform': carla.Transform(carla.Location(x=0.5, z=1.6)),
#                      'attach_to':None}

bgr_camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': FLAGS.img_width,
                     'image_size_y': FLAGS.img_height, 'fov': 110, 'sensor_tick': 0.02,
                     'transform': carla.Transform(carla.Location(x=-0.6, z=2)),
                     'attach_to':None}

collision_sensor_config = {'data_type': 'sensor.other.collision','attach_to': None}
invasion_sensor_config = {'data_type': 'sensor.other.lane_detector', 'attach_to': None}
obstacle_sensor_config = {'data_type': 'sensor.other.obstacle', 'sensor_tick': 0.02,
                          'distance': 3, 'attach_to': None}


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


def check_whether_respawn_ego(world, vehicle):
    """check whether to respawn the static acotors in a frequency"""
    while True:
        time.sleep(20)
        if actor_static(vehicle):
            respawn_actor_at(world, vehicle, init_point)


def target_thread(sess):
    """a thread for target nets in DQN"""
    global obstacles

    while True:
        ## get current state
        for camera_sensor, lane_invasion, obj_collision in zip(cameras, lane_invasions, obj_collisions):
            img = camera_sensor.get()
            img = img[int(FLAGS.img_height*2.3//5):, :, :] ## corp the ROI

            img = cv2.resize(img, dsize=(FLAGS.net_img_height, FLAGS.net_img_width))
            # # cv2.imshow('test', img)
            # imgs.append(img)

            img_1 = cv2.resize(img, dsize=(FLAGS.net_img_height, FLAGS.net_img_width))
            s_hm = produce_heat_map(egopilots[0], obstacles, hm_size=(FLAGS.net_img_height, FLAGS.net_img_width),
                                    h_type='safe', consider_range=15)
            a_hm = produce_heat_map(egopilots[0], obstacles, hm_size=(FLAGS.net_img_height, FLAGS.net_img_width),
                                    h_type='attentive', consider_range=15)
            d_hm = produce_heat_map(egopilots[0], obstacles, hm_size=(FLAGS.net_img_height, FLAGS.net_img_width),
                                    h_type='danger', consider_range=15)
            img_2 = np.uint8(np.minimum(np.stack([a_hm, s_hm, d_hm], axis=-1) * 255, 255))
            img = np.concatenate([img_1, img_2], axis=-1)

            lane_invasion.clear()
            obj_collision.clear()

        # s_hm = produce_heat_map(egopilots[0], obstacles, hm_size=(FLAGS.net_img_height, FLAGS.net_img_width), h_type='safe')
        # a_hm = produce_heat_map(egopilots[0], obstacles, hm_size=(FLAGS.net_img_height, FLAGS.net_img_width), h_type='attentive')
        # d_hm = produce_heat_map(egopilots[0], obstacles, hm_size=(FLAGS.net_img_height, FLAGS.net_img_width), h_type='danger')
        # img = np.uint8(np.minimum(np.stack([a_hm, s_hm, d_hm], axis=-1)*255, 255))
        # cv2.imshow('test', img)
        current_img_state = np.array([img])
        current_img_state = current_img_state*2./255. - 1.

        ## get current action and control the egopilots
        current_action, current_step = sess.run([max_action_index_online, global_step], feed_dict={online_img_state: current_img_state})

        ## control the egopilots ##
        i = 0
        for egopilot, c_a in zip(egopilots, current_action):
            ## e-greedy

            current_action[i] = c_a

            steer = action_index_2_steer(c_a)
            throttle = 0.5
            brake = 0.

            ego_v = egopilot.get_velocity()
            ego_v = math.sqrt(ego_v.x ** 2 + ego_v.y ** 2 + ego_v.z ** 2)
            if ego_v > 8. and throttle > 0.5:
                throttle = 0.5 ## avoid velocity too big

            ## apply control
            egopilot.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            i += 1

        # cv2.waitKey(30)
        time.sleep(0.5) ## sleep for a while, let the action control the egopilots to next state

        ## check whether end.
        for egopilot, lane_invasion, obj_collision in zip(egopilots, lane_invasions, obj_collisions):
            on_collision = obj_collision.get()
            on_invasion = lane_invasion.get()

            if on_invasion or on_collision:
                destroy(obstacles)
                respawn_actor_at(world, egopilot, transform=init_point)
                obstacles = random_spawn_obstacles_in_specific_area(world)


if __name__ == '__main__':
    online_img_state = tf.placeholder(shape=[None, FLAGS.net_img_height, FLAGS.net_img_width, 6], dtype=tf.float32)

    ## other input ##
    reward = tf.placeholder(shape=[None], dtype=tf.float32)
    whether_end = tf.placeholder(shape=[None], dtype=tf.float32)  ##True is end ,False is continue
    real_action_index = tf.placeholder(shape=[None], dtype=tf.int64)
    lr = tf.placeholder(dtype=tf.float32)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    act_val_net_online = action_value_net()
    act_val_online, vars_online = act_val_net_online.build_graph(img_state=online_img_state, n_action=FLAGS.n_action, is_training=False,
                                                                      var_scope='online_act_val')

    #########################################
    ## the best action ops in current step ##
    #########################################
    max_action_index_online = tf.argmax(act_val_online, axis=-1)


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
    init_point = carla.Transform()
    init_point.location.x = -40
    init_point.location.y = random.sample(road_range['y'], 1)[0]
    init_point.location.z = 0.3
    init_point.rotation.yaw = -0.142975
    vehicle = spawn_egopilot_at(world, init_point)
    obstacles = random_spawn_obstacles_in_specific_area(world)
    # spawn_vehicles(world, n_autopilots=0, n_egopilots=FLAGS.n_egopilots)
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

    spawn_points = list(world.get_map().get_spawn_points())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if ckpt:
            logger.info('loading %s...' % str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('Load checkpoint success...')
        else:
            raise ValueError('must provide checkpoint...')

        check_t = threading.Thread(target=check_whether_respawn_ego, args=(world, egopilots[0],))
        target_t = threading.Thread(target=target_thread, args=(sess,))


        target_t.daemon = True
        check_t.daemon = True

        check_t.start()
        # # respwan_v_t.start()
        target_t.start()
        # vis_memory_thread()
        while True:
            pass