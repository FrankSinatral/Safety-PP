
import importlib
import json
import gymnasium as gym

from configuration import Configurable

import numpy as np
import torch


DEVICE = torch.device('cuda:{}'.format(0))

def agent_factory(environment, config):
    """
        Handles creation of agents.

    :param environment: the environment
    :param config: configuration of the agent, must contain a '__class__' key
    :return: a new agent
    """
    if "__class__" in config:
        path = config['__class__'].split("'")[1]
        module_name, class_name = path.rsplit(".", 1)
        agent_class = getattr(importlib.import_module(module_name), class_name)
        agent = agent_class(environment, config)
        return agent
    else:
        raise ValueError("The configuration should specify the agent __class__")

def load_agent(agent_config, env):
    """
        Load an agent from a configuration file.

    :param agent_config: dict or the path to the agent configuration file
    :param env: the environment with which the agent interacts
    :return: the agent
    """
    # Load config from file
    if not isinstance(agent_config, dict):
        agent_config = load_agent_config(agent_config)
    #print(agent_config)
    return agent_factory(env, agent_config)


def load_agent_config(config_path):
    """
        Load an agent configuration from file, with inheritance.
    :param config_path: path to a json config file
    :return: the configuration dict
    """
    with open(config_path) as f:
        agent_config = json.loads(f.read())
    if "base_config" in agent_config:
        base_config = load_agent_config(agent_config["base_config"])
        del agent_config["base_config"]
        agent_config = Configurable.rec_update(base_config, agent_config)
    return agent_config


def load_environment(env_config):
    """
        Load an environment from a configuration file.

    :param env_config: the configuration, or path to the environment configuration file
    :return: the environment
    """
    # Load the environment config from file
    if not isinstance(env_config, dict):
        with open(env_config) as f:
            env_config = json.loads(f.read())
    
    '''
    if env_config['id'] == 'highway':
        from highway_env.envs.highway_env import HighwayEnv
        env = HighwayEnv()
    elif env_config['id'] == 'intersection':
        from highway_env.envs.intersection_env import IntersectionEnv
        env = IntersectionEnv()
    env.unwrapped.configure(env_config)
    env.reset()
    return env
    '''

    # Make the environment
    if env_config.get("import_module", None):
        __import__(env_config["import_module"])
    try:
        env = gym.make(env_config['id'], render_mode='rgb_array')
        # Save env module in order to be able to import it again
        env.import_module = env_config.get("import_module", None)
    except KeyError:
        raise ValueError("The gym register id of the environment must be provided")
    except gym.error.UnregisteredEnv:
        # The environment is unregistered.
        print("import_module", env_config["import_module"])
        raise gym.error.UnregisteredEnv('Environment {} not registered. The environment module should be specified by '
                                        'the "import_module" key of the environment configuration'.format(
                                            env_config['id']))

    # Configure the environment, if supported
    try:
        env.unwrapped.configure(env_config)
        # Reset the environment to ensure configuration is applied
        env.reset()
    except AttributeError as e:
        print("This environment does not support configuration. {}".format(e))
    return env

