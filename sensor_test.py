"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import glob, os, sys, time
from threading import Thread
import cv2

from utils.world_ops import *
from utils.sensor_ops import *

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla
except:
    raise ImportError('Please check your carla file')


def check_whether_respawn_actors(world, vehicles):
    """check whether to respawn the static acotors in a frequency"""
    while True:
        if carla_actors_static(vehicles, bigger_than=0.8):
            respawn_static_actors(world, vehicles)
        time.sleep(20)

if __name__ == '__main__':

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0) # seconds
    logger.info('connect success...')

    world = client.get_world()
    ##  spawn vehicles in carla world
    spawn_vehicles(world, number=5)

    actor_list = world.get_actors()
    vehicles = list(actor_list.filter('vehicle*'))

    bgr_camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': 418,
                        'image_size_y': 278, 'fov': 110, 'sensor_tick': 0.02,
                        'transform': carla.Transform(carla.Location(x=0.8, z=1.7)),
                        'attach_to':vehicles[0]}
    bgr_sensor = bgr_camera(world, bgr_camera_config)

    depth_camera_config = {'data_type': 'sensor.camera.depth', 'image_size_x': 418,
                        'image_size_y': 278, 'fov': 110, 'sensor_tick': 0.02,
                        'transform': carla.Transform(carla.Location(x=0.8, z=1.7)),
                        'attach_to':vehicles[0]}
    depth_sensor = depth_camera(world, depth_camera_config)


    semantic_camera_config = {'data_type': 'sensor.camera.semantic_segmentation', 'image_size_x': 418,
                            'image_size_y': 278, 'fov': 110, 'sensor_tick': 0.02,
                            'transform': carla.Transform(carla.Location(x=0.8, z=1.7)),
                            'attach_to':vehicles[0]}
    semantic_sensor = semantic_camera(world, semantic_camera_config)

    ## start check
    t = Thread(target=check_whether_respawn_actors, args=(world, vehicles))
    t.daemon = True
    t.start()

    logger.info('Press key a to stop...')
    while True:
        bgr = bgr_sensor.get()
        depth = depth_sensor.get()
        semantic = semantic_sensor.get()

        cv2.imshow('Color', bgr)
        cv2.imshow('Depth', depth)
        cv2.imshow('Semantic', semantic)
        a = cv2.waitKey(10)
        if a == 97:
            cv2.destroyAllWindows()
            break

    vehicles = list(actor_list.filter('vehicle*'))
    for vehicle in vehicles:
        vehicle.destroy()
    logger.info('Destroy all vehicles...')

    sensors = list(actor_list.filter('sensor*'))
    for sensor in sensors:
        sensor.destroy()
    logger.info('Destroy all sensors...')
