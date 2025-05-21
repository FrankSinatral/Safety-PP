import torch
import torch.nn as nn
from gymnasium import spaces
import random
import numpy as np
import math
import copy
from torch.distributions import Categorical

from models.factory import loss_function_factory, optimizer_factory, \
	  model_factory, size_model_config, trainable_parameters
from utils import DEVICE
from agents.base import AbstractStochasticAgent

class RolloutBuffer:
	def __init__(self):
		self.actions = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.next_states = []
		self.is_terminals = []
	
	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.next_states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.is_terminals[:]

class ActorCritic(nn.Module):
	def __init__(self, config):
		super(ActorCritic, self).__init__()

		def straight_through(x):
			return x
		if config["common"] is None:
			self.common = straight_through
		else:
			self.common = model_factory(config["common"])
		self.actor = model_factory(config["actor"])
		self.critic = model_factory(config["critic"])
		self.actor_final=nn.Softmax(dim=-1)
	
	def actor_forward(self, state):
		x = self.common(state)
		logit = self.actor(x)
		return self.actor_final(logit)
		#return logit

	def critic_forward(self, state):
		x = self.common(state)
		v = self.critic(x)
		return v


class PPOAgent(AbstractStochasticAgent):
	def __init__(self, env, config=None):
		super(PPOAgent, self).__init__(config)
		self.env = env
		assert isinstance(env.action_space, spaces.Discrete), \
			"Only compatible with Discrete action spaces."
		self.training = True
		self.previous_state = None

		if self.config["common"] is not None:
			size_model_config(self.env, self.config["common"])
		size_model_config(self.env, self.config["actor"])
		size_model_config(self.env, self.config["critic"])
		self.device = DEVICE

		self.actor_critic = ActorCritic(self.config).to(self.device)
		print("Number of trainable parameters: {}".format(trainable_parameters(self.actor_critic)))
		self.buffer = RolloutBuffer()
		
		self.loss_function = loss_function_factory(self.config["loss_function"])
		self.optimizer = optimizer_factory(self.config["optimizer"]["type"],
										   self.actor_critic.parameters(),
										   **self.config["optimizer"])
		self.timestep = 0
		self.optim_batch_size = self.config["batch_size"]
		self.entropy_coef=self.config["entropy_coef"]
		self.action_size = env.action_space.n
		
	@classmethod
	def default_config(cls):
		return dict(common=None,
					actor = dict(type ="MultiLayerPerceptron"),
					critic = dict(type = "MultiLayerPerceptron"),
					optimizer=dict(type="ADAM",
								   lr=2e-3,
								   weight_decay=0,
								   k=5),
					loss_function="l2",
					batch_size=64,
					gamma=0.99,
					lambd=0.97,
					clip_rate=0.2,
					K_epochs=10,
					T_horizon = 128,
					entropy_coef = 0.01,
					adv_normalization = True,
					entropy_coef_decay = 0.99)
	
	def act(self, state):
		self.previous_state = state
		state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
		if self.training:
			with torch.no_grad():
				pi = self.actor_critic.actor_forward(state)
				pi = pi.squeeze(0)
				distr = Categorical(probs= pi) # assume pi is positive
				action = distr.sample()
				action_logprob = distr.log_prob(action)
			self.buffer.states.append(state)
			self.buffer.actions.append(torch.as_tensor(np.array([action.cpu().numpy()]), dtype=torch.int64).to(self.device))
			self.buffer.logprobs.append(torch.as_tensor(np.array([action_logprob.cpu().numpy()]), dtype=torch.float32).to(self.device))
		else:
			with torch.no_grad():
				pi = self.actor_critic.actor_forward(state)
				pi = pi.squeeze(0)
				action = torch.argmax(pi)

		return action.item()

	def action_distribution(self, state):
		state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
		with torch.no_grad():
			pi = self.actor_critic.actor_forward(state)
			pi = pi.squeeze(0)
		values = pi.cpu().numpy()
		distribution = {action: values[action] for action in range(self.action_size)}
		return values, distribution
	
	def record(self, state, action, reward, next_state, done, end, info):
		if not self.training:
			return
		# Single-agent setting
		self.buffer.rewards.append(torch.as_tensor(np.array([reward]), dtype=torch.float32).to(self.device))
		self.buffer.next_states.append(torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device))
		self.buffer.is_terminals.append(torch.as_tensor(np.array([done]), dtype=torch.float32).to(self.device))
		self.timestep += 1

		if self.timestep % self.config["T_horizon"] == 0:
			loss, entropy = self.update()
			self.writer.add_scalar("agent/loss", loss, self.timestep)
			self.writer.add_scalar("agent/entropy", entropy, self.timestep)
	
	def update(self):
		# convert list to tensor
		s = torch.cat(self.buffer.states, dim=0).detach()
		a = torch.cat(self.buffer.actions, dim=0).detach().unsqueeze(-1)
		old_logprob_a = torch.cat(self.buffer.logprobs, dim=0).detach().unsqueeze(-1)
		r = torch.cat(self.buffer.rewards, dim=0).detach().unsqueeze(-1)
		s_prime = torch.cat(self.buffer.next_states, dim=0).detach()
		done_mask = torch.cat(self.buffer.is_terminals, dim=0).detach().unsqueeze(-1)

		#print(s.shape, a.shape, old_logprob_a.shape, r.shape, s_prime.shape, done_mask.shape)

		"""PPO update"""
		batch_size = s.shape[0]
		#Slice long trajectopy into short trajectory and perform mini-batch PPO update
		optim_iter_num = int(math.ceil(batch_size / self.optim_batch_size))

		for _ in range(self.config["K_epochs"]):

			''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
			with torch.no_grad():
				vs = self.actor_critic.critic_forward(s)
				vs_ = self.actor_critic.critic_forward(s_prime)

				'''dw(dead and win) for TD_target and Adv'''
				deltas = r + self.config["gamma"] * vs_ * (1 - done_mask) - vs
				deltas = deltas.cpu().flatten().numpy()
				adv = [0]

				'''done for GAE'''
				for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
					advantage = dlt + self.config["gamma"] * self.config["lambd"] * adv[-1] * (1 - mask)
					adv.append(advantage)
				adv.reverse()
				adv = copy.deepcopy(adv[0:-1])
				adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
				td_target = adv + vs
				if self.config["adv_normalization"]:
					adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  #useful in some envs

			#Shuffle the trajectory, Good for training
			perm = np.arange(batch_size)
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.device)
			per_s, per_a, per_td_target, per_adv, per_old_logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_logprob_a[perm].clone()

			'''mini-batch PPO update'''
			for i in range(optim_iter_num):
				index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, per_s.shape[0]))

				'''actor update'''
				#prob = self.actor_critic.actor_forward(per_s[index])
				logit = self.actor_critic.actor_forward(per_s[index])
				distr = Categorical(probs = logit) # logit are positive
				entropy = distr.entropy().sum(0, keepdim=True)
				prob_a = distr.probs.gather(1, per_a[index])
				#prob_a = prob.gather(1, per_a[index])
				ratio = torch.exp(torch.log(prob_a) - per_old_logprob_a[index])  # a/b == exp(log(a)-log(b))

				surr1 = ratio * per_adv[index]
				surr2 = torch.clamp(ratio, 1 - self.config["clip_rate"], 1 + self.config["clip_rate"]) * per_adv[index]
				per_critic = self.actor_critic.critic_forward(per_s[index])
				
				
				# final loss of clipped objective PPO
				loss = -torch.min(surr1, surr2)\
					  + 0.5* self.loss_function(per_critic, per_td_target[index]) \
						  - self.entropy_coef * entropy
				# take gradient step
				self.optimizer.zero_grad()
				loss.mean().backward()
				self.optimizer.step()

		# clear buffer
		self.buffer.clear()
		return loss.mean().detach().cpu(), entropy.detach().cpu()

	def save(self, filename):
		state = {'state_dict': self.actor_critic.state_dict(),
				 'optimizer': self.optimizer.state_dict()}
		torch.save(state, filename)
		return filename

	def load(self, filename):
		checkpoint = torch.load(filename, map_location=self.device)
		self.actor_critic.load_state_dict(checkpoint['state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		return filename

	def initialize_model(self):
		pass

	def set_writer(self, writer):
		super().set_writer(writer)
		self.writer.add_scalar("agent/trainable_parameters", trainable_parameters(self.actor_critic), 0)
	
	def eval(self):
		self.training = False

	def train(self):
		self.training = True

	def seed(self, seed):
		np.random.seed(seed)    
		random.seed(seed)    
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = True 

	def reset(self):
		pass

	def schedule_explore(self):
		self.entropy_coef *= self.config["entropy_coef_decay"] #exploring decay