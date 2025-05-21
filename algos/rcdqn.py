import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from network import FeedForwardNN
from replay_buffer import ReplayBuffer
from utils_classic_control import device


class RCDQN:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, state_size, action_size, args):
		self.args = args
		
		self.state_size = state_size
		self.action_size = action_size

		self.device = device
		self.TAU = args.tau
		self.GAMMA = args.gamma
		self.UPDATE_EVERY = args.update_every
		self.NUPDATES = args.n_updates
		self.BATCH_SIZE = args.batch_size
		self.clip_grad = args.clip_grad
		self.eps_start = args.eps_start
		self.eps_decay = args.eps_decay
		self.eps_min = args.eps_min
		self.double = args.double

		self.lambd = args.lambd_init
		self.lambd_lr = args.lambd_lr
		self.lambd_update_interval = args.lambd_update_interval
		self.costs = []
		self.cost_lim = 0
		self.total_cost = 0

		# Q-Network
		self.qnetwork_local = FeedForwardNN(self.state_size, self.action_size, args.hidden_dim).to(self.device)
		self.qnetwork_target = FeedForwardNN(self.state_size, self.action_size, args.hidden_dim).to(self.device)
		self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

		self.optimizer = Adam(self.qnetwork_local.parameters(), lr=args.lr)
		self.loss = nn.MSELoss()

		self.replay_buffer = ReplayBuffer(buffer_size=args.buffer_size,
									   batch_size=args.batch_size,
									   seed= args.seed, device = self.device)
		self.t_step = 0
		self.eps=self.eps_start
		self.unsafe_data=[]
		self.writer = None

	def get_action(self, state, deterministic=False):
		state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
		#self.qnetwork_local.eval()
		with torch.no_grad():
			action_values = self.qnetwork_local(state)
		#self.qnetwork_local.train()

		# Epsilon-greedy action selection
		if deterministic:
			return np.argmax(action_values.cpu().data.numpy())
		if random.random() > self.eps:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.action_size))
	
	def step(self, state, action, reward, next_state, done, cost, end):
		self.total_cost += cost
		#reward = reward - self.lambd * cost
		self.replay_buffer.add(state, action, reward, next_state, done, cost)
		Q_loss, Q_target = 0, 0
		if end: 
			self.costs.append(self.total_cost)
			self.total_cost = 0
		
		# Learn every UPDATE_EVERY time steps.
		self.t_step += 1
		if self.t_step % self.UPDATE_EVERY == 0:
			# If enough samples are available in replay_buffer, get random subset and learn
			if len(self.replay_buffer) > self.BATCH_SIZE:
				Q_losses = []
				Q_targets=[]
				for _ in range(self.NUPDATES):
					experiences = self.replay_buffer.sample()
					loss,q_target, _= self.update(experiences)
					Q_losses.append(loss)
					Q_targets.append(q_target)
				Q_loss = np.mean(Q_losses)
				Q_target = np.mean(Q_targets)
				if self.t_step % self.lambd_update_interval == 0:
					if len(self.costs)>0:
						self.lambd += self.lambd_lr * max(0, np.mean(self.costs) - self.cost_lim)
					self.costs = []	
				if (self.writer is not None) and ((self.t_step - self.BATCH_SIZE) % 100 == 0):
					self.writer.add_scalar("agent/unsafe_portion", np.mean(self.unsafe_data), self.t_step)
					self.unsafe_data =[]
		return Q_loss, Q_target, 0
	
	def update(self, experiences):
		states, actions, rewards, next_states, dones, costs = experiences
		unsafe_portion = torch.mean(costs).item()
		self.unsafe_data.append(unsafe_portion)
		rewards = rewards - self.lambd * costs

		with torch.no_grad():
			if self.double:
				_, best_actions = self.qnetwork_local(next_states).max(1)
				q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions.unsqueeze(1))
			else:
				q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
			q_targets = rewards + self.GAMMA * q_targets_next * (1 - dones)

		q_expected = self.qnetwork_local(states).gather(1, actions)

		# Compute loss
		loss = self.loss(q_expected, q_targets) 
		
		# Minimize the loss
		self.optimizer.zero_grad()
		loss.backward()
		clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad)
		self.optimizer.step()
		if self.args.hard_update_step >0:
			if self.t_step % self.args.hard_update_step == 0:
				self.hard_update(self.qnetwork_local, self.qnetwork_target)
		else:
			self.soft_update(self.qnetwork_local, self.qnetwork_target)
			
		return loss.detach().cpu().numpy(), torch.mean(q_targets, 0).cpu().numpy(), 0

	def soft_update(self, local_model, target_model):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)
	
	def hard_update(self, local_model, target_model):
		target_model.load_state_dict(local_model.state_dict())
	
	def save(self, filename):
		state = {'state_dict': self.qnetwork_local.state_dict(),
				 'optimizer': self.optimizer.state_dict()}
		torch.save(state, filename)
		return filename

	def load(self, filename):
		checkpoint = torch.load(filename, map_location=self.device)
		self.qnetwork_local.load_state_dict(checkpoint['state_dict'])
		self.qnetwork_target.load_state_dict(checkpoint['state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		return filename
	
	def exploration_schedule(self):
		self.eps = max(self.eps * self.eps_decay, self.eps_min)
