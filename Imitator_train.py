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
    'checkpoint_dir', '',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'train_dir', './checkpoint/imitator',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_float('learning_rate', 3e-2, 'Initial learning rate.')

tf.app.flags.DEFINE_integer(
    'f_log_step', 50,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'f_save_step', 10000,
    'The frequency with which summaries are saved, in step.')

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
obstacle_sensor_config = {'data_type': 'sensor.other.obstacle', 'sensor_tick': 0.02,
                          'distance': 3, 'attach_to': None}

light_state_encode = {'Green':np.array([1., 0., 0., 0.]).astype(np.float32),
                      'Red':np.array([0., 1., 0., 0.]).astype(np.float32),
                      'Yellow':np.array([0., 0., 1., 0.]).astype(np.float32),
                      'Unkown':np.array([0., 0., 0., 1.]).astype(np.float32)}


def steer_scene_classify(steer):
    """generate the winding one-hot groundtruth"""
    if steer >= -1. and steer < -0.5:
        return 0, np.eye(1, 11, k=0)[0].astype(np.int32)
    elif steer >= -0.5 and steer < -0.2:
        return 1, np.eye(1, 11, k=1)[0].astype(np.int32)
    elif steer >= -0.2 and steer < -0.1:
        return 2, np.eye(1, 11, k=2)[0].astype(np.int32)
    elif steer >= -0.1 and steer < -0.05:
        return 3, np.eye(1, 11, k=3)[0].astype(np.int32)
    elif steer >= -0.05 and steer < 0.:
        return 4, np.eye(1, 11, k=4)[0].astype(np.int32)
    elif steer >= 0. and steer < 0.05:
        return 5, np.eye(1, 11, k=5)[0].astype(np.int32)
    elif steer >= 0.05 and steer < 0.1:
        return 6, np.eye(1, 11, k=6)[0].astype(np.int32)
    elif steer >= 0.1 and steer < 0.2:
        return 7, np.eye(1, 11, k=7)[0].astype(np.int32)
    elif steer >= 0.2 and steer < 0.5:
        return 8, np.eye(1, 11, k=8)[0].astype(np.int32)
    elif steer >= 0.5 and steer <= 1.:
        return 9, np.eye(1, 11, k=9)[0].astype(np.int32)


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


def sample_thread(sess):
    """a thread used to collect the data from carla"""
    begin = True

    ## set all the egopilots to autopilot and init whether_wait_red_light
    whether_wait_red_light = {}
    for egopilot in egopilots:
        whether_wait_red_light[egopilot] = False
        egopilot.set_autopilot(enabled=True)


    while True:
        for camera, obj_collision, egopilot in zip(cameras, obj_collisions, egopilots):
            img = camera.get()
            collision = obj_collision.get()

            if collision:
                obj_collision.clear()
                ## if collision skip this memory
                single_execuate(target=respawn_actors, args=(world, [egopilot],))
                continue

            # cv2.imshow('test', img)

            # img = img[FLAGS.img_height//2:, FLAGS.img_width//5:4*FLAGS.img_width//5, :] ## corp the ROI
            img = img*2./255. - 1.
            img = cv2.resize(img, dsize=(224, 224))

            std_steer = egopilot.get_control().steer
            std_throttle = egopilot.get_control().throttle
            std_brake = egopilot.get_control().brake

            scene_class, steer_scene_encode = steer_scene_classify(steer=std_steer)

            # if std_steer<0.05 and std_steer>-0.05:
            #     a = random.uniform(0,1)
            #     if a < 0.8:
            #         continue

            ego_v = egopilot.get_velocity()
            ego_v_ = math.sqrt(ego_v.x ** 2 + ego_v.x ** 2 + ego_v.x ** 2)
            ego_v = ego_v_ / egopilot.get_speed_limit()

            if egopilot.is_at_traffic_light() and not whether_wait_red_light[egopilot]:
                ## mean first go into traffic light area
                if str(egopilot.get_traffic_light_state()) == 'Green':
                    # print('Green, straght forward')
                    scene_class = 10 ## mean straght forward
                elif str(egopilot.get_traffic_light_state()) == 'Red':
                    # print('Red, Stop')
                    whether_wait_red_light[egopilot] = True
                    scene_class = 11 ## mean need to stop
                light_state = light_state_encode[str(egopilot.get_traffic_light_state())]
            elif whether_wait_red_light[egopilot]:
                if str(egopilot.get_traffic_light_state()) == 'Green':
                    whether_wait_red_light[egopilot] = False
                    # print('Red to Green, I can go now')
                    std_throttle = 1.
                    std_brake = 0.
                    # print('throttle:', std_throttle)
                    # print('brake:', std_brake)
                    scene_class = 12 ## mean red to green
                elif str(egopilot.get_traffic_light_state()) == 'Red':
                    # print('Still Red, wait for green')
                    whether_wait_red_light[egopilot] = True
                    scene_class = 11 ## mean need to stop
                light_state = light_state_encode[str(egopilot.get_traffic_light_state())]
            else:
                if ego_v_ < 1e-3:   ## imitate ahead obstcle
                    steer_scene_encode = np.eye(1, 11, k=10)[0].astype(np.int32)
                    # print("stop!!!")
                # else:
                #     print('go forward')
                light_state = light_state_encode['Unkown']

            std_action = np.array([std_steer, std_throttle, std_brake])
            other_state = np.concatenate([light_state, np.array([ego_v])], axis=-1)

            memory_pool.put(memory=[img.astype(np.float32), other_state.astype(np.float32), std_action.astype(np.float32), steer_scene_encode],
                            class_index=scene_class)
        time.sleep(0.2)

        # print(memory_pool.get_propotion())

        if begin and memory_pool.is_balance():
            begin = False
            update_event.set()


def update_thread(sess):
    """a thread used to train an actor net"""
    update_event.wait()
    logger.info('Begin update the actor...')

    avg_clf_loss = 0.
    avg_ops_loss = 0.
    current_step = 0
    while True:
        memorys = memory_pool.get(batch_size=FLAGS.batch_size)
        imgs = []
        other_states = []
        actions = []
        scene_labels = []

        # img = memorys[0][0]
        # img = np.uint8((img+1.)*255./2)
        # print('state:', memorys[0][1])
        # print('action:', memorys[0][2])
        #
        # cv2.imshow('test', img)
        # cv2.waitKey()

        for memory in memorys:
            imgs.append(memory[0])
            other_states.append(memory[1])
            actions.append(memory[2])
            scene_labels.append(memory[3])

        if current_step < 60000:
            op, clf_l, ops_l, current_step = sess.run([update_ops, clf_loss, ops_loss, global_step], feed_dict={input: np.array(imgs),
                                                                                              other_state: np.array(
                                                                                                  other_states),
                                                                                             std_action: np.array(
                                                                                                 actions),
                                                                                              scene_label: np.array(
                                                                                                  scene_labels),
                                                                                              lr:FLAGS.learning_rate})
            # op, net_loss, current_step = sess.run([update_ops, clf_loss, global_step], feed_dict={input: np.array(imgs),
            #                                                                                   scene_label:np.array(scene_labels),
            #                                                                                   lr: FLAGS.learning_rate})
        elif current_step < 120000:
            op, clf_l, ops_l, current_step = sess.run([update_ops, clf_loss, ops_loss, global_step], feed_dict={input: np.array(imgs),
                                                                                              other_state: np.array(
                                                                                                  other_states),
                                                                                              std_action: np.array(
                                                                                                  actions),
                                                                                              scene_label: np.array(
                                                                                                  scene_labels),
                                                                                              lr: FLAGS.learning_rate/10})
            # op, net_loss, current_step = sess.run([update_ops, clf_loss, global_step], feed_dict={input: np.array(imgs),
            #                                                                                   scene_label: np.array(
            #                                                                                       scene_labels),
            #                                                                                   lr: FLAGS.learning_rate/10})
        elif current_step < 8000000:
            op, clf_l, ops_l, current_step = sess.run([update_ops, clf_loss, ops_loss, global_step], feed_dict={input: np.array(imgs),
                                                                                                                other_state: np.array(other_states),
                                                                                                                std_action: np.array(actions),
                                                                                                                scene_label: np.array(scene_labels),
                                                                                                                lr: FLAGS.learning_rate/100})
            # op, net_loss, current_step = sess.run([update_ops, clf_loss, global_step], feed_dict={input: np.array(imgs),
            #                                                                                   scene_label: np.array(
            #                                                                                       scene_labels),
            #                                                                                   lr: FLAGS.learning_rate/100})
        else:
            break

        if FLAGS.f_log_step != None:
            ## caculate average loss ##
            step = current_step % FLAGS.f_log_step
            avg_ops_loss = (avg_ops_loss * step + ops_l) / (step + 1.)
            avg_clf_loss = (avg_clf_loss * step + clf_l) / (step + 1.)
            if current_step % FLAGS.f_log_step == FLAGS.f_log_step - 1:
                logger.info('Step%s ops_loss:%s clf_loss:%s' % (str(current_step), str(avg_ops_loss), str(avg_clf_loss)))

        if FLAGS.f_save_step != None:
            if current_step % FLAGS.f_save_step == FLAGS.f_save_step - 1:
                ## save model ##
                logger.info('Saving model...')
                model_name = os.path.join(FLAGS.train_dir, 'imitator.model')
                saver.save(sess, model_name)
                logger.info('Save model sucess...')


if __name__ == '__main__':
    input = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
    other_state = tf.placeholder(shape=[None, 5], dtype=tf.float32)
    std_action = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    scene_label = tf.placeholder(shape=[None, 11], dtype=tf.int32)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    lr = tf.placeholder(dtype=tf.float32)

    ## TF GRAPH ##
    scene_logits, action = model(input, is_training=True)

    clf_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=scene_label, logits=scene_logits))

    ops_loss = tf.reduce_sum(tf.square(action - std_action))

    loss = clf_loss + 100.*ops_loss

    ## UPDATE OPS ##
    bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(bn_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
        grads_and_vars = optimizer.compute_gradients(loss)
        ## clip the gradients ##
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var)
                      for grad, var in grads_and_vars if grad != None]
        update_ops = optimizer.apply_gradients(capped_gvs, global_step=global_step)

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
    spawn_vehicles(world, n_autopilots=40, n_egopilots=10)
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
    memory_pool = rl_tools.balance_memory_pooling(max_capacity=1300, n_class=13)
    update_event = threading.Event()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if ckpt:
            logger.info('loading %s...' % str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('Load checkpoint success...')
        else:
            sess.run(init)
            logger.info('Actor variables init success...')


        s_t = threading.Thread(target=sample_thread, args=(sess,))
        u_t = threading.Thread(target=update_thread, args=(sess,))
        c_t = threading.Thread(target=check_whether_respawn_actors, args=(world, autopilots+egopilots,))
        s_t.daemon = True
        u_t.daemon = True
        c_t.daemon = True
        s_t.start()
        c_t.start()
        u_t.start()

        c_t.join()