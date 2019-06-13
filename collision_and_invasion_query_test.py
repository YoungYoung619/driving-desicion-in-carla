"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import glob, os, sys, time, random
from threading import Thread
import cv2

from carla_utils.world_ops import *
from carla_utils.sensor_ops import *

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


def make_vehicle_random_drive(vehicles):
    """a thread make vehicle random drive 3s and autopilot 10s"""
    vehicle = vehicles[0]
    while True:
        time.sleep(10)
        logger.info('Start random drive...')
        vehicle.set_autopilot(False)
        steer = random.uniform(-0.2,0.2)
        throttle = random.uniform(0,1)
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
        time.sleep(3)
        logger.info('Start autopilot...')
        vehicle.set_autopilot(True)


if __name__ == '__main__':

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0) # seconds
    logger.info('connect success...')

    world = client.get_world()
    ##  spawn vehicles in carla world
    spawn_vehicles(world, n_autopilots=10, n_egopilots=3)

    actor_list = world.get_actors()
    vehicles = list(actor_list.filter('vehicle*'))

    autopilots = get_all_autopilots(world)
    egopilots = get_all_egopilots(world)
    a = autopilots + egopilots

    bgr_camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': 800,
                        'image_size_y': 512, 'fov': 110, 'sensor_tick': 0.02,
                        'transform': carla.Transform(carla.Location(x=0.8, z=1.7)),
                        'attach_to':autopilots[0]}
    bgr_sensor = bgr_camera(world, bgr_camera_config)


    collision_sensor_config = {'data_type': 'sensor.other.collision','attach_to': autopilots[0]}
    collision_q = collision_query(world, collision_sensor_config)

    invasion_sensor_config = {'data_type': 'sensor.other.lane_detector', 'attach_to': autopilots[0]}
    lane_invasion_q = lane_invasion_query(world, invasion_sensor_config)


    ## start check
    t1 = Thread(target=check_whether_respawn_actors, args=(world, autopilots+egopilots))
    t1.daemon = True
    t1.start()

    t2 = Thread(target=make_vehicle_random_drive, args=(autopilots,))
    t2.daemon = True
    t2.start()

    logger.info('Press key a to stop...')
    while True:
        bgr = bgr_sensor.get()
        on_collision = collision_q.get()
        on_invasion = lane_invasion_q.get()

        if on_collision:
            respawn_actors(world, [autopilots[0]])
            collision_q.clear()
        elif on_invasion:
            respawn_actors(world, [autopilots[0]])
            lane_invasion_q.clear()

        cv2.imshow('Color', bgr)
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
