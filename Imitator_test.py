"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
train a imitaor using the carla standard control data.

Author：Team Li
"""
import tensorflow as tf
import numpy as np

from carla_utils.logging import logger
from carla_utils.world_ops import *
from carla_utils.sensor_ops import *

from RL.DDPG.ddpg_utils import actor
import RL.rl_utils as rl_tools

import threading
import math
import cv2

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla
except:
    raise ImportError('Please check your carla file')


tf.app.flags.DEFINE_string(
    'checkpoint_dir', './checkpoint/imitator',
    'The path to a checkpoint from which to fine-tune.')


tf.app.flags.DEFINE_integer(
    'img_height', 416,
    'raw image height')

tf.app.flags.DEFINE_integer(
    'img_width', 626,
    'raw image width')

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim


bgr_camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': FLAGS.img_width,
                     'image_size_y': FLAGS.img_height, 'fov': 110, 'sensor_tick': 0.02,
                     'transform': carla.Transform(carla.Location(x=1.0, z=1.8)),
                     'attach_to':None}
collision_sensor_config = {'data_type': 'sensor.other.collision','attach_to': None}

light_state_encode = {'Green':np.array([1., 0., 0., 0.]).astype(np.float32),
                      'Red':np.array([0., 1., 0., 0.]).astype(np.float32),
                      'Yellow':np.array([0., 0., 1., 0.]).astype(np.float32),
                      'Unkown':np.array([0., 0., 0., 1.]).astype(np.float32)}





def model(input, is_training):
    """user define model
    Args:
        input: an input tensor
        is_training: indicate whether train or not
    Return:
        output tensor of which the meaning is defined by user.
        in this script, means [steer, throttle, brake]
    """
    net = actor()
    scene_logits, action, net_vars = net.build_graph(input, other_state=other_state, n_action_space=3, is_training=is_training,
                                                      action_range=[[-1., 1.], [0., 1.], [0., 1.]], var_scope='online_actor')
    return scene_logits, action


def single_execuate(target, args):
    threading.Thread(target=target, args=args).start()


def check_whether_respawn_actors(world, vehicles):
    """check whether to respawn the static acotors in a frequency"""
    while True:
        if carla_actors_static(vehicles, bigger_than=0.8):
            respawn_static_actors(world, vehicles)
        time.sleep(20)


def control_thread(sess):
    """a thread used to control the vehicles"""

    while True:
        imgs = []
        other_states = []
        for camera, egopilot in zip(cameras, egopilots):
            img = camera.get()

            img = img[FLAGS.img_height//2:, FLAGS.img_width//5:4*FLAGS.img_width//5, :] ## corp the ROI
            img = img*2./255. - 1.
            img = cv2.resize(img, dsize=(224, 224))

            ego_v = egopilot.get_velocity()
            ego_v = math.sqrt(ego_v.x**2 + ego_v.x**2 + ego_v.x**2) / egopilot.get_speed_limit()

            if egopilot.is_at_traffic_light():
                light_state = light_state_encode[str(egopilot.get_traffic_light_state())]
            else:
                light_state = light_state_encode['Unkown']

            vehicle_state = np.concatenate([light_state, np.array([ego_v])], axis=-1)

            imgs.append(img)
            other_states.append(vehicle_state)


        acts = sess.run(action, feed_dict={input: np.array(imgs), other_state: np.array(other_states)})

        for egopilot, act in zip(egopilots, acts):
            if act[0] > act[2]:
                act[2] = 0.
            else:
                act[0] = 0.
            print(act)
            egopilot.apply_control(carla.VehicleControl(throttle=float(act[1]), steer=float(act[0]), brake=float(act[2])))

        time.sleep(0.2)

        for obj_collision, egopilot in zip(obj_collisions, egopilots):
            if obj_collision.get():
                obj_collision.clear()
                single_execuate(target=respawn_actors, args=(world, [egopilot],))



if __name__ == '__main__':
    input = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
    other_state = tf.placeholder(shape=[None, 5], dtype=tf.float32)
    std_action = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    scene_label = tf.placeholder(shape=[None, 10], dtype=tf.int32)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    lr = tf.placeholder(dtype=tf.float32)

    ## TF GRAPH ##
    scene_logits, action = model(input, is_training=True)

    # clf_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=scene_label, logits=scene_logits))
    #
    # ops_loss = tf.reduce_sum(tf.square(action - std_action))
    #
    # loss = clf_loss + 10.*ops_loss

    # ## UPDATE OPS ##
    # bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(bn_ops):
    #     optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
    #     grads_and_vars = optimizer.compute_gradients(loss)
    #     ## clip the gradients ##
    #     capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var)
    #                   for grad, var in grads_and_vars if grad != None]
    #     update_ops = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    logger.info('Build tensorflow graph finish...')
    logger.info('Total trainable parameters:%s' %
                str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))


    #### carla world init ####
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)  # seconds
    logger.info('Carla connect success...')

    logger.info('Carla world initing...')
    world = client.get_world()
    destroy_all_actors(world)

    ##  spawn vehicles in carla world
    spawn_vehicles(world, n_autopilots=1, n_egopilots=5)
    time.sleep(10)

    autopilots = get_all_autopilots(world)
    egopilots = get_all_egopilots(world)

    cameras = []
    obj_collisions = []
    logger.info('Adding some sensors to egopilots...')
    for egopilot in egopilots:
        ## attach a camera to egopilot ##
        bgr_camera_config['attach_to'] = egopilot
        bgr_sensor = bgr_camera(world, bgr_camera_config)
        cameras.append(bgr_sensor)

        ## attach collision sensor to egopilot ##
        collision_sensor_config['attach_to'] = egopilot
        collision_sensor = collision_query(world, collision_sensor_config)
        obj_collisions.append(collision_sensor)

    logger.info('Adding some sensors to egopilots success')

    # memory_pool = rl_tools.memory_pooling(maxlen=1000)
    memory_pool = rl_tools.balance_memory_pooling(max_capacity=1000, n_class=10)
    update_event = threading.Event()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if ckpt:
            logger.info('loading %s...' % str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('Load checkpoint success...')
        else:
            raise ValueError('You must provide true checkpoint...')


        a_t = threading.Thread(target=control_thread, args=(sess,))
        c_t = threading.Thread(target=check_whether_respawn_actors, args=(world, autopilots+egopilots,))
        a_t.daemon = True
        c_t.daemon = True
        a_t.start()

        time.sleep(10)
        logger.info('check thread start ...')
        c_t.start()
        a_t.join()