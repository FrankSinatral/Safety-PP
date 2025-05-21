import torch
import numpy as np
import random 
from collections import deque, namedtuple

from utils_classic_control import SumTree

class ReplayBuffer():
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, buffer_size, batch_size, seed, device):
		"""Initialize a ReplayBuffer object.

		Params
		======
			action_size (int): dimension of each action
			buffer_size (int): maximum size of buffer
			batch_size (int): size of each training batch
			seed (int): random seed
		"""
		self.memory = deque(maxlen=buffer_size)  
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "cost"])
		self.seed = random.seed(seed)
		self.device = device
	
	def add(self, state, action, reward, next_state, done, cost):
		"""Add a new experience to memory."""
		e = self.experience(state, action, reward, next_state, done, cost)
		self.memory.append(e)
	
	def sample(self):
		"""Randomly sample a batch of experiences from memory."""
		experiences = random.sample(self.memory, k=self.batch_size)

		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]).astype(np.float32)).float().to(self.device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None]).astype(np.float32)).float().to(self.device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32)).float().to(self.device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
		costs = torch.from_numpy(np.vstack([e.cost for e in experiences if e is not None]).astype(np.float32)).float().to(self.device)
  
		return (states, actions, rewards, next_states, dones, costs)

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)



class SafetyReplayBuffer():
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, buffer_size, batch_size, seed, device, safe_mix=0.75):
		"""Initialize a ReplayBuffer object.

		Params
		======
			action_size (int): dimension of each action
			buffer_size (int): maximum size of buffer
			batch_size (int): size of each training batch
			seed (int): random seed
		"""
		self.memory = deque(maxlen=buffer_size)  
		self.safe_memory = deque(maxlen=buffer_size)  
		self.unsafe_memory = deque(maxlen=buffer_size)  
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "cost"])
		self.seed = random.seed(seed)
		self.device = device
		self.safe_mix=safe_mix
		self.safe_portion = max(1, int(self.batch_size * safe_mix))
		self.unsafe_portion = max(1, self.batch_size - self.safe_portion)
	
	def add(self, state, action, reward, next_state, done, cost):
		"""Add a new experience to memory."""
		e = self.experience(state, action, reward, next_state, done, cost)
		self.memory.append(e)
		if int(cost) <=0: self.safe_memory.append(e)
		else: self.unsafe_memory.append(e)
	
	def sample(self):
		"""Randomly sample a batch of experiences from memory."""
		experiences = random.sample(self.memory, k=self.batch_size)

		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]).astype(np.float32)).float().to(self.device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None]).astype(np.float32)).float().to(self.device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32)).float().to(self.device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
		costs = torch.from_numpy(np.vstack([e.cost for e in experiences if e is not None]).astype(np.float32)).float().to(self.device)
  
		return (states, actions, rewards, next_states, dones, costs)
	
	def safety_sample(self):
		
		safe_exp = random.sample(self.safe_memory, k=min(len(self.safe_memory), self.safe_portion))

		states = torch.from_numpy(np.vstack([e.state for e in safe_exp if e is not None]).astype(np.float32)).float().to(self.device)
		actions = torch.from_numpy(np.vstack([e.action for e in safe_exp if e is not None])).long().to(self.device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in safe_exp if e is not None]).astype(np.float32)).float().to(self.device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in safe_exp if e is not None]).astype(np.float32)).float().to(self.device)
		dones = torch.from_numpy(np.vstack([e.done for e in safe_exp if e is not None]).astype(np.uint8)).float().to(self.device)
		costs = torch.from_numpy(np.vstack([e.cost for e in safe_exp if e is not None]).astype(np.float32)).float().to(self.device)

		if len(self.unsafe_memory) < self.unsafe_portion:
			return (states, actions, rewards, next_states, dones, costs)
	
		unsafe_exp = random.sample(self.unsafe_memory, k=self.unsafe_portion)

		ustates = torch.from_numpy(np.vstack([e.state for e in unsafe_exp if e is not None]).astype(np.float32)).float().to(self.device)
		uactions = torch.from_numpy(np.vstack([e.action for e in unsafe_exp if e is not None])).long().to(self.device)
		urewards = torch.from_numpy(np.vstack([e.reward for e in unsafe_exp if e is not None]).astype(np.float32)).float().to(self.device)
		unext_states = torch.from_numpy(np.vstack([e.next_state for e in unsafe_exp if e is not None]).astype(np.float32)).float().to(self.device)
		udones = torch.from_numpy(np.vstack([e.done for e in unsafe_exp if e is not None]).astype(np.uint8)).float().to(self.device)
		ucosts = torch.from_numpy(np.vstack([e.cost for e in unsafe_exp if e is not None]).astype(np.float32)).float().to(self.device)

		states = torch.cat([states, ustates], dim=0)
		actions = torch.cat([actions, uactions], dim=0)
		rewards = torch.cat([rewards, urewards], dim=0)
		next_states = torch.cat([next_states, unext_states], dim=0)
		dones = torch.cat([dones, udones], dim=0)
		costs = torch.cat([costs, ucosts], dim=0)
		return (states, actions, rewards, next_states, dones, costs)

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)
	
	

class PrioritizedReplayBuffer():
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, state_size, buffer_size, batch_size, seed, device, alpha=0.1, beta=0.1):
		self.tree = SumTree(buffer_size)

		# PER params
		#self.eps = eps  # minimal priority, prevents zero probabilities
		self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
		self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
		#self.max_priority = eps  # priority for new samples, init as eps


		self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
		self.action = torch.empty(buffer_size, 1, dtype=torch.int64)
		self.reward = torch.empty(buffer_size, 1, dtype=torch.float)
		self.cost = torch.empty(buffer_size, 1, dtype=torch.float)
		self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
		self.done = torch.empty(buffer_size, 1, dtype=torch.int)


		self.batch_size = batch_size
		self.seed = random.seed(seed)
		self.device = device
		self.size = buffer_size
		self.real_size = 0
		self.count = 0
	
	def add(self, state, action, reward, next_state, done, cost):
		"""Add a new experience to memory."""
		self.state[self.count] = torch.as_tensor(state)
		self.action[self.count] = torch.as_tensor(action, dtype=torch.int64)
		self.reward[self.count] = torch.as_tensor(reward)
		self.next_state[self.count] = torch.as_tensor(next_state)
		self.done[self.count] = torch.as_tensor(done)
		self.cost[self.count] = torch.as_tensor(cost)

		# 如果是第一条经验，初始化优先级为1.0；否则，对于新存入的经验，指定为当前最大的优先级
		priority = 1.0 if self.real_size == 0 else self.tree.priority_max
		self.tree.update_priority(data_index=self.count, priority=priority)  # 更新当前经验在sum_tree中的优先级
		self.real_size = min(self.size, self.real_size + 1)
		self.count = (self.count+1) % self.size
	
	def sample(self):

		sample_idxs, Normed_IS_weight = self.tree.prioritized_sample(N=self.real_size, batch_size=self.batch_size, beta=self.beta)

		states = self.state[sample_idxs].to(self.device)
		actions = self.action[sample_idxs].to(self.device)
		rewards = self.reward[sample_idxs].to(self.device)
		next_states = self.next_state[sample_idxs].to(self.device)
		dones = self.done[sample_idxs].to(self.device)
		costs = self.cost[sample_idxs].to(self.device)
		weights = torch.as_tensor(Normed_IS_weight, dtype= torch.float32).to(self.device)
  
		return states, actions, rewards, next_states, dones, costs, weights, sample_idxs
	
	def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
		priorities = (np.abs(td_errors) + 0.01) ** self.alpha
		for index, priority in zip(batch_index, priorities):
			self.tree.update_priority(data_index=index, priority=priority)

	def __len__(self):
		"""Return the current size of internal memory."""
		return self.real_size
