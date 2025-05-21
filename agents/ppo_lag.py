import torch
import torch.nn as nn
from gymnasium import spaces
import random
import numpy as np
import math
import copy
from torch.distributions import Categorical

from torch.nn.functional import softplus
from torch.optim import Adam

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
		self.costs = []
		self.next_states = []
		self.is_terminals = []
		self.is_ends = []
	
	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.next_states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.costs[:]
		del self.is_terminals[:]
		del self.is_ends[:]

class ActorCriticCost(nn.Module):
	def __init__(self, config):
		super(ActorCriticCost, self).__init__()

		def straight_through(x):
			return x
		if config["common"] is None:
			self.common = straight_through
		else:
			self.common = model_factory(config["common"])
		self.actor = model_factory(config["actor"])
		self.critic = model_factory(config["critic"])
		self.cost_critic = model_factory(config["cost_critic"])
		self.actor_final=nn.Softmax(dim=-1)
	
	def actor_forward(self, state):
		x = self.common(state)
		logit = self.actor(x)
		return self.actor_final(logit)
		#return logit

	def critic_forward(self, state):
		x = self.common(state)
		v = self.critic(x)
		vc = self.cost_critic(x)
		return v, vc


class PPOLagAgent(AbstractStochasticAgent):
	def __init__(self, env, config=None):
		super(PPOLagAgent, self).__init__(config)
		self.env = env
		assert isinstance(env.action_space, spaces.Discrete), \
			"Only compatible with Discrete action spaces."
		self.training = True
		self.previous_state = None

		if self.config["common"] is not None:
			size_model_config(self.env, self.config["common"])
		size_model_config(self.env, self.config["actor"])
		size_model_config(self.env, self.config["critic"])
		size_model_config(self.env, self.config["cost_critic"]) # cost
		self.device = DEVICE

		self.actor_critic = ActorCriticCost(self.config).to(self.device)
		print("Number of trainable parameters: {}".format(trainable_parameters(self.actor_critic)))
		self.buffer = RolloutBuffer()
		
		self.loss_function = loss_function_factory(self.config["loss_function"])
		self.optimizer = optimizer_factory(self.config["optimizer"]["type"],
										   self.actor_critic.parameters(),
										   **self.config["optimizer"])
		
		
		self.penalty_param = torch.tensor(self.config['penalty_init']).float().to(self.device)
		self.penalty_param.requires_grad=True
		#penalty = softplus(self.penalty_param)
		penalty_lr = self.config['penalty_lr']
		self.penalty_optimizer = Adam([self.penalty_param], lr=penalty_lr)
		
		self.timestep = 0
		self.optim_batch_size = self.config["batch_size"]
		self.entropy_coef=self.config["entropy_coef"]
		self.action_size = env.action_space.n
		
	@classmethod
	def default_config(cls):
		return dict(common=dict(type ="MultiLayerPerceptron"),
					actor = dict(type ="MultiLayerPerceptron"),
					critic = dict(type = "MultiLayerPerceptron"),
					cost_critic = dict(type = "MultiLayerPerceptron"),
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
					entropy_coef_decay = 0.99,
					penalty_lr = 5e-5,
					penalty_init = 0.25)
	
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
		self.buffer.costs.append(torch.as_tensor(np.array([info['cost']]), dtype=torch.float32).to(self.device)) # cost
		self.buffer.next_states.append(torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device))
		self.buffer.is_terminals.append(torch.as_tensor(np.array([done]), dtype=torch.float32).to(self.device))
		self.buffer.is_ends.append(torch.as_tensor(np.array([end]), dtype=torch.float32).to(self.device))
		self.timestep += 1

		if self.timestep % self.config["T_horizon"] == 0:
			loss, entropy = self.update()
			self.writer.add_scalar("agent/loss", loss, self.timestep)
			self.writer.add_scalar("agent/entropy", entropy, self.timestep)
			self.writer.add_scalar("agent/penalty_param", self.penalty_param.detach().item(), self.timestep)
	
	def update(self):
		# convert list to tensor
		s = torch.cat(self.buffer.states, dim=0).detach()
		a = torch.cat(self.buffer.actions, dim=0).detach().unsqueeze(-1)
		old_logprob_a = torch.cat(self.buffer.logprobs, dim=0).detach().unsqueeze(-1)
		r = torch.cat(self.buffer.rewards, dim=0).detach().unsqueeze(-1)
		c = torch.cat(self.buffer.costs, dim=0).detach().unsqueeze(-1) # cost
		s_prime = torch.cat(self.buffer.next_states, dim=0).detach()
		done_mask = torch.cat(self.buffer.is_terminals, dim=0).detach().unsqueeze(-1)
		self.buffer.is_ends[-1]= torch.as_tensor(np.array([1.]), dtype=torch.float32).to(self.device)
		end_mask = torch.cat(self.buffer.is_ends, dim=0).detach().unsqueeze(-1)

		#print(s.shape, a.shape, old_logprob_a.shape, r.shape, s_prime.shape, done_mask.shape)

		cur_cost = torch.sum(c) / torch.clamp(torch.sum(end_mask.float()), min=1.0)
		cost_limit = 0
		cost_deviation = cur_cost - cost_limit
		loss_penalty = -self.penalty_param*cost_deviation
		self.penalty_optimizer.zero_grad()
		loss_penalty.backward()
		self.penalty_optimizer.step()

		p = softplus(self.penalty_param)
		penalty_item = p.item()

		"""PPO update"""
		batch_size = s.shape[0]
		#Slice long trajectopy into short trajectory and perform mini-batch PPO update
		optim_iter_num = int(math.ceil(batch_size / self.optim_batch_size))

		for _ in range(self.config["K_epochs"]):

			''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
			with torch.no_grad():
				vs, cvs = self.actor_critic.critic_forward(s)
				vs_, cvs_ = self.actor_critic.critic_forward(s_prime)

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
				adv = (adv - adv.mean()) / ((adv.std() + 1e-10))  #useful in some envs
			
			'''For Cost Critic: dw(dead and win) for TD_target and Adv'''
			cdeltas = c + self.config["gamma"] * cvs_ * (1 - done_mask) - cvs
			cdeltas = cdeltas.cpu().flatten().numpy()
			cadv = [0]
			for cdlt, mask in zip(cdeltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
				cadvantage = cdlt + self.config["gamma"] * self.config["lambd"] * cadv[-1] * (1 - mask)
				cadv.append(cadvantage)
			cadv.reverse()
			cadv = copy.deepcopy(cadv[0:-1])
			cadv = torch.tensor(cadv).unsqueeze(1).float().to(self.device)
			ctd_target = cadv + cvs
			if self.config["adv_normalization"]:
				cadv = (cadv - cadv.mean()) / ((cadv.std() + 1e-10))  #useful in some envs

			#Shuffle the trajectory, Good for training
			perm = np.arange(batch_size)
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.device)
			per_s, per_a, per_td_target, per_adv, per_old_logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_logprob_a[perm].clone()
			per_ctd_target = ctd_target[perm].clone()
			per_cadv = cadv[perm].clone()

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

				loss_rpi = torch.min(surr1, surr2)
				loss_cpi = ratio* per_cadv[index] * penalty_item
				pi_objective = (loss_rpi - loss_cpi ) / (1+penalty_item)

				per_critic, per_cost_critic = self.actor_critic.critic_forward(per_s[index])
				
				# final loss of clipped objective PPO
				loss = - pi_objective \
					  + 0.25* self.loss_function(per_critic, per_td_target[index]) \
						+ 0.25* self.loss_function(per_cost_critic, per_ctd_target[index]) - self.entropy_coef * entropy
				# take gradient step
				self.optimizer.zero_grad()
				loss.mean().backward()
				self.optimizer.step()

		# clear buffer
		self.buffer.clear()
		return loss.mean().detach().cpu(), entropy.detach().cpu()

	def save(self, filename):
		state = {'state_dict': self.actor_critic.state_dict(),
				 'optimizer': self.optimizer.state_dict(),
				 'penalty_param': self.penalty_param}
		torch.save(state, filename)
		return filename

	def load(self, filename):
		checkpoint = torch.load(filename, map_location=self.device)
		self.actor_critic.load_state_dict(checkpoint['state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		self.penalty_param = checkpoint['penalty_param']
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