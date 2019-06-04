"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
some world query and operation tools

Author：Team Li
"""
import random, time
from utils.logging import logger

"""
#####  User Recommended Functions  ####
--------------respawn_static_actors
"""

def actor_static(actor):
    """judge whether an actor is in static state
    Args:
        actor: an actor of vehicle
    Return:
        a bool val respresents whether the actor is in static state
    """
    vel = actor.get_velocity()
    if abs(vel.x) <= 1e-3 and abs(vel.y) <= 1e-3 and abs(vel.z) <= 1e-3:
        return True
    else:
        return False


def static_rate(actors):
    """get the static rate in carla world
    Args:
        actors: all the vehicle actor list
    Return:
        a rate respresents how many vehicle in static state
    """
    static_n = 0
    for vehicle in actors:
        if actor_static(vehicle):
            static_n += 1
    rate = static_n / len(actors)
    return rate


def carla_actors_static(actors, bigger_than):
    """judge whether all the actors are in static state
    Return:
        a bool val.
    """
    if static_rate(actors) > bigger_than:
        return True
    else:
        return False


def respawn_static_actors(world, actors):
    """re-spawn the static actors in the carla world
    Args:
        world: client.get_world()
        actors:world.get_actors()
    Example:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0) # seconds

        actor_list = world.get_actors()
        vehicles = list(actor_list.filter('vehicle*'))
        if carla_actors_static(vehicles):
            respawn_static_actors(vehicles)
    """
    for vehicle in actors:
        if actor_static(vehicle):
            spawn_points = list(world.get_map().get_spawn_points())
            index = random.randint(0, (len(spawn_points))-1)
            vehicle.set_transform(spawn_points[index])
            logger.info('Respawn '+str(vehicle)+' in '+str(spawn_points[index]))


def respawn_actors(world, actors):
    """re-spawn all the actors in the carla world
    Args:
        world: client.get_world()
        actors:world.get_actors()
    Example:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0) # seconds

        actor_list = world.get_actors()
        vehicles = list(actor_list.filter('vehicle*'))
        if carla_actors_static(vehicles):
            respawn_static_actors(vehicles)
    """
    for vehicle in actors:
        spawn_points = list(world.get_map().get_spawn_points())
        index = random.randint(0, (len(spawn_points))-1)
        vehicle.set_transform(spawn_points[index])
        logger.info('Respawn '+str(vehicle)+' in '+str(spawn_points[index]))


def try_spawn_random_vehicle_at(world, transform):
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprint = random.choice(blueprints)
    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
        vehicle.set_autopilot()
        logger.info('spawned %r at %s' % (vehicle.type_id, transform.location))
        return True
    return False

def spawn_vehicles(world, number):
    count = number
    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)
    for spawn_point in spawn_points:
        if try_spawn_random_vehicle_at(world, spawn_point):
            count -= 1
        if count <= 0:
            break

    while count > 0:
        time.sleep(500)
        if try_spawn_random_vehicle_at(world, random.choice(spawn_points)):
            count -= 1