"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import sys, glob, os
from carla_utils.world_ops import *

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla
except:
    raise ImportError('Please check your carla file')

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0) # seconds
logger.info('connect success...')

world = client.get_world()
actors = world.get_actors()

## get all the vehicles in carla world
vehicles = list(actors.filter('vehicle*'))

if carla_actors_static(vehicles, bigger_than=0.8):
    ## if more that 80% of vehicles are in static state:
    respawn_static_actors(world, vehicles) ##respawn all the static actors
