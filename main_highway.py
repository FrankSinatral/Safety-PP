from trainer import Trainer
from utils import load_agent, load_environment
import argparse
import highway_env
from highway_env import register_highway_envs

register_highway_envs()


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch RL example')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--num_episodes', type=int, default=4000, help='number of training episodes')
parser.add_argument('--total_steps', type=int, default=100, help='number of total steps (k)')
parser.add_argument('--use_total_steps', type=str2bool, default=True, help='whether use_total_steps')
parser.add_argument('--env_config', type=str, default='configs/MergeEnv/env.json')
parser.add_argument('--agent_config', type=str, default='configs/MergeEnv/agents/sac_lag.json')
parser.add_argument('--exp_dir', type=str, default=None)
args = parser.parse_args()

env = load_environment(args.env_config)
agent = load_agent(args.agent_config, env)
env.unwrapped.config["offscreen_rendering"] = False
display_agent = False
display_env = False
trainer = Trainer(env, agent, num_episodes=args.num_episodes, 
                  use_total_steps = args.use_total_steps,
                  total_steps = int(args.total_steps * 1000),
                  display_env=display_env,
                  display_agent= display_agent, sim_seed=args.seed,
                  directory= args.exp_dir)
print(f"Ready to train {agent} on {env}")

trainer.train()