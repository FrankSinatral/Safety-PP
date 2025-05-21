import numpy as np
import torch
import os
import random
from types import SimpleNamespace as SN
import json
from collections.abc import Mapping
import yaml
import os
import datetime
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from utils_classic_control import *
from trainer_classic_control import Trainer

from algos.dqn import DQN
from algos.rewsdqn import RewsDQN
from algos.rcdqn import RCDQN
from algos.recdqn import RecDQN
from algos.sqdqn import SQDQN
from algos.sp2dqn import SP2DQN
from algos.sp3dqn import SP3DQN
from algos.priordqn import PriorDQN
from algos.priorrewsdqn import PriorRewsDQN


def get_config(config_name, subfolder):
    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "configs_classic_control", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def set_arg(config_dict, params, arg_name, arg_type):
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            arg_name = _v.split("=")[0].replace("--", "")
            arg_value = _v.split("=")[1]
            config_dict[arg_name] = arg_type(arg_value)
            del params[_i]
            return config_dict


def recursive_dict_update(d, u):
    '''update dict d with items in u recursively. if key present in both d and u, 
    value in d takes precedence. 
    '''
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            if k not in d.keys():
                d[k] = v                
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)




if __name__ == '__main__':
    # Get the defaults from default.yaml
    config_dict = get_config("default",  "")

    # Load env base configs when considering difficulties
    env_config = get_config(config_dict['env'], "envs")

    # Load algorithm configs
    alg_config = get_config(config_dict['algo'], "algs")

    # update env_config and alg_config with values in config dict 
    # copy modified env args for logging purpose 
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config = config_copy(config_dict)

    # set up random seed
    os.environ['PYTHONHASHSEED'] = str(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])

    # save config and creat result path
    current_path = config['result_path'] + '/' + config['env']
    current_path = current_path + '/' + config['algo'] +'/seed_' + str(config['seed'])
    current_path = current_path + '_' + str(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    if not os.path.exists(current_path):
        os.makedirs(current_path)
    config['tb_logs'] = current_path
    writer=None
    if config["use_tb"]:
        writer = SummaryWriter(config['tb_logs'])
    print('Config: ', config)
    config_file = open(current_path+'/config.json', 'w')
    b = json.dump(config, config_file, indent=6)
    config_file.close()

    args = SN(**config)    

    # create env
    if args.env == 'acc':
        from envs.acc import ACC
        env = ACC(args)
    elif args.env == 'circle':
        from envs.circle import Circle
        env = Circle(args)  
    else:
        raise NotImplementedError("Environment not implemented")
    test_env = deepcopy(env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # create agent
    if args.algo == 'dqn':
        agent = DQN(state_size, action_size, args)
    elif args.algo == 'rewsdqn':
        agent = RewsDQN(state_size, action_size, args)
    elif args.algo=='rcdqn':
        agent = RCDQN(state_size, action_size, args)
    elif args.algo == 'sp2dqn':
        agent= SP2DQN(state_size, action_size, args)
    elif args.algo == 'sp3dqn':
        agent= SP3DQN(state_size, action_size, args)
    elif args.algo == 'priordqn':
        agent= PriorDQN(state_size, action_size, args)
    elif args.algo == 'priorrewsdqn':
        agent= PriorRewsDQN(state_size, action_size, args)
    elif args.algo == 'recdqn':
        agent= RecDQN(state_size, action_size, args)
    elif args.algo == 'sqdqn':
        agent= SQDQN(state_size, action_size, args)

    # train
    q_trainer = Trainer(agent, env, test_env, args, writer=writer)
    q_trainer.train(int(args.total_steps * 1000))
    




    
