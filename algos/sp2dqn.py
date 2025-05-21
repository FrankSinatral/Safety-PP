import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from network import FeedForwardNN
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils_classic_control import device


class SP2DQN:
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
		self.xp_order = args.xp_order
		self.crisp = args.crisp
		self.crisp_threshold = args.crisp_threshold

		if args.use_log:
			self.polarized_function = torch.log
		else:
			self.polarized_function = self.xp

		# Q-Network
		self.qnetwork_local = FeedForwardNN(self.state_size, self.action_size, args.hidden_dim).to(self.device)
		self.qnetwork_target = FeedForwardNN(self.state_size, self.action_size, args.hidden_dim).to(self.device)
		self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

		# Phi-Network
		self.phi_local = FeedForwardNN(self.state_size, self.action_size, args.hidden_dim, True).to(self.device)
		self.phi_target = FeedForwardNN(self.state_size, self.action_size, args.hidden_dim, True).to(self.device)
		self.phi_target.load_state_dict(self.phi_local.state_dict())

		self.optimizer = Adam(self.qnetwork_local.parameters(), lr=args.lr)
		self.phi_optimizer = Adam(self.phi_local.parameters(), lr=args.phi_lr)
		self.loss = nn.MSELoss()

		self.replay_buffer = ReplayBuffer(buffer_size=args.buffer_size,
									   batch_size=args.batch_size,
									   seed= args.seed, device = self.device)
		self.phi_buffer = PrioritizedReplayBuffer(state_size, buffer_size=args.phi_buffer_size,
									   batch_size=args.batch_size,
									   seed= args.seed, device = self.device,
									   alpha=self.args.alpha, beta=self.args.beta)
		self.t_step = 0
		self.eps=self.eps_start
		self.unsafe_data=[]
		self.writer = None
	
	def xp(self, input_):
		return 1.0 - 1.0/ (input_ ** self.xp_order)

	def get_action(self, state, deterministic=False):
		state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
		#self.qnetwork_local.eval()
		with torch.no_grad():
			qsa_values = self.qnetwork_local(state)
			if self.crisp:
				psafe = torch.clamp(1- self.phi_local.crisp_forward(state, self.crisp_threshold), 
						min = 0.00001, max= 1.0)
			else:
				psafe = torch.clamp(1- self.phi_local(state), min = 0.00001, max= 1.0)
			action_values = qsa_values + self.args.p_weight_a * self.polarized_function(psafe)
		#self.qnetwork_local.train()

		# Epsilon-greedy action selection
		if deterministic:
			return np.argmax(action_values.cpu().data.numpy())
		if random.random() > self.eps:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.action_size))
	
	def step(self, state, action, reward, next_state, done, cost, end):
		# Save experience in replay replay_buffer
		self.replay_buffer.add(state, action, reward, next_state, done, cost)
		self.phi_buffer.add(state, action, reward, next_state, done, cost)
		Q_loss, Q_target, phi_loss = 0, 0, 0
		
		# Learn every UPDATE_EVERY time steps.
		self.t_step += 1
		if self.t_step % self.UPDATE_EVERY == 0:
			# If enough samples are available in replay_buffer, get random subset and learn
			if len(self.replay_buffer) > self.args.start_learning:
				Q_losses = []
				Q_targets=[]
				vp_losses = []
				for _ in range(self.NUPDATES):
					experiences = self.replay_buffer.sample()
					loss,q_target, vp_loss = self.update(experiences)
					Q_losses.append(loss)
					Q_targets.append(q_target)
					vp_losses.append(vp_loss)
				Q_loss = np.mean(Q_losses)
				Q_target = np.mean(Q_targets)
				phi_loss = np.mean(vp_losses)
				if (self.writer is not None) and ((self.t_step - self.BATCH_SIZE) % 100 == 0):
					self.writer.add_scalar("agent/unsafe_portion", np.mean(self.unsafe_data), self.t_step)
					self.unsafe_data =[]
		return Q_loss, Q_target, phi_loss
			
	
	def update(self, experiences):
		states, actions, rewards, next_states, dones, costs = experiences
		unsafe_portion = torch.mean(costs).item()
		self.unsafe_data.append(unsafe_portion)

		with torch.no_grad():
			if self.double:
				qsa_values = self.qnetwork_local(next_states) # (batch, act)
				if self.crisp:
					psafe =1- self.phi_local.crisp_forward(next_states, self.crisp_threshold)
				else:
					psafe =1- self.phi_local(next_states)
				psafe = torch.clamp(psafe, min = 0.00001, max= 1.0)
				sm_weights = qsa_values + self.args.p_weight_a * self.polarized_function(psafe)
				_, best_actions = sm_weights.max(1)
				q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions.unsqueeze(1))
			else:
				qsa_values = self.qnetwork_target(next_states) # (batch, act)
				if self.crisp:
					psafe =1- self.phi_target.crisp_forward(next_states, self.crisp_threshold)
				else:
					psafe =1- self.phi_target(next_states)
				psafe = torch.clamp(psafe, min = 0.00001, max= 1.0)
				sm_weights = qsa_values + self.args.p_weight_a * self.polarized_function(psafe)
				_, best_actions = sm_weights.max(1)
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

		# use prioritized here for phi
		states, actions, rewards, next_states, dones, costs, weights, sample_idxs=self.phi_buffer.sample()

		vp_loss, td_error = self.compute_vp_loss(states, actions, next_states, dones, costs, weights)
		self.phi_optimizer.zero_grad()
		vp_loss.backward()
		clip_grad_norm_(self.phi_local.parameters(), self.clip_grad)
		self.phi_optimizer.step()

		self.phi_buffer.update_batch_priorities(sample_idxs, td_error.squeeze(-1).cpu().numpy())

		if self.args.hard_update_step >0:
			if self.t_step % self.args.hard_update_step == 0:
				self.hard_update(self.qnetwork_local, self.qnetwork_target)
				self.hard_update(self.phi_local, self.phi_target)
		else:
			self.soft_update(self.qnetwork_local, self.qnetwork_target)
			self.soft_update(self.phi_local, self.phi_target)
		
		return loss.detach().cpu().numpy(), torch.mean(q_targets, 0).cpu().numpy(), vp_loss.detach().cpu().numpy()
	
	def compute_vp_loss(self, states, actions, next_states, dones, costs, weights=None):
		# jinpeng:
		# assume cost is either 0 or 1
		# so we do not operate the costs here
		with torch.no_grad():
			if self.args.use_min:
				target_vps = self.phi_target(next_states)

				if self.args.double_phi:
					_, best_actions = self.phi_local(next_states).min(1)
				else:
					_, best_actions = target_vps.min(1)
				
				target_vp = target_vps.gather(1, best_actions.unsqueeze(1))
			else:
				qsa_values =self.qnetwork_local(next_states)
				if self.crisp:
					psafe =1- self.phi_local.crisp_forward(next_states, self.crisp_threshold)
				else:
					psafe =1- self.phi_local(next_states)
				psafe = torch.clamp(psafe, min = 0.00001, max= 1.0)
				sm_weights = qsa_values + self.args.p_weight_a * self.polarized_function(psafe)
				_, best_actions = sm_weights.max(1)
				target_vp = self.qnetwork_target(next_states).gather(1, best_actions.unsqueeze(1))
		target_next_vp = torch.max(costs, self.GAMMA * (1-dones)* target_vp)
		current_vp = self.phi_local(states).gather(1, actions)

		if weights is None:
			weights = torch.ones_like(current_vp).to(self.device)

		td_error = torch.abs(current_vp - target_next_vp).detach()

		vp_loss = torch.mean((current_vp- target_next_vp.detach())**2 * weights)
		return vp_loss, td_error


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
		   		 'phi_state_dict': self.phi_local.state_dict(),
				 'optimizer': self.optimizer.state_dict()}
		torch.save(state, filename)
		return filename

	def load(self, filename):
		checkpoint = torch.load(filename, map_location=self.device)
		self.qnetwork_local.load_state_dict(checkpoint['state_dict'])
		self.qnetwork_target.load_state_dict(checkpoint['state_dict'])
		self.phi_local.load_state_dict(checkpoint['phi_state_dict'])
		self.phi_target.load_state_dict(checkpoint['phi_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		return filename
	
	def exploration_schedule(self):
		self.eps = max(self.eps * self.eps_decay, self.eps_min)
