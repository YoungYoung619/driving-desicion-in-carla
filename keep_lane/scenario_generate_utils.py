"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""

import os, sys, glob
try:
    sys.path.append('F:\my_project\driving-desicion-in-carla\dist/carla-0.9.4-py3.7-win-amd64.egg')
    import carla
except:
    raise ImportError('Please check your carla file')

from carla_utils.logging import logger
from carla_utils.world_ops import *
from carla_utils.sensor_ops import *

road_range = {'x':(-40, 154), 'y':(204, 207)}


def random_spawn_autopilot_at(world, distance_freq):
    """ random spawn an autopilot at a transform in carla world
    Args:
        world: carla world instance
        distance_freq: a distance control the vehicle number in this road area
    """
    x_min = road_range['x'][0]
    x_max = road_range['x'][1]
    x_dis = x_max - x_min
    freq = x_dis // distance_freq

    vehicles = []
    for i in range(freq-1):
        x = random.randint(x_min + 10 + i*distance_freq, x_min + 10 + (i+1)*distance_freq - 8)
        y = random.sample(road_range['y'], 1)[0]
        if x > 120:
            z = 3
        else:
            z = 1.81
        point = carla.Transform()
        point.location.x = x
        point.location.y = y
        point.location.z = z
        point.rotation.yaw = -0.142975
        vehicle = spawn_autopilot_at(world, point)
        if not vehicle:
            return None
        vehicles.append(vehicle)

    return vehicles


def control_vehicles_go_straight_task(vehicles):
    """make vehicles go straight"""
    for vehicle in vehicles:
        vehicle.apply_control(carla.VehicleControl(throttle=random.uniform(0.1, 0.3), steer=0., brake=0.))


def random_spawn_obstacles_in_specific_area(world):

    vehicles = None
    while not vehicles:
        vehicles = random_spawn_autopilot_at(world, 20)

    return vehicles

if __name__ == '__main__':
    #### carla world init ####
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)  # seconds
    logger.info('Carla connect success...')

    logger.info('Carla world initing...')
    world = client.get_world()

    destroy_all_actors(world)

    while True:
        v = random_spawn_obstacles_in_specific_area(world)
        time.sleep(1)
        destroy_all_actors(world)

    # while True:
    #     vehicles = random_spawn_autopilot_at(world, 20)
    #     for i in range(10):
    #         control_vehicles_go_straight_task(vehicles)
    #         time.sleep(1)
    #
    #     ## sample trajectory
    #     destroy_all_actors(world)
    #
    # pass