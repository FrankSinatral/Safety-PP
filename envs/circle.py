import numpy as np 
import random

from gym import spaces

from utils_classic_control import clip_norm

class Circle():
	def __init__(self, args):
		self.current_step = 0
		self.ep_len = args.ep_len
		self.delta_t = args.delta_t # 0.1
		self.v_max = args.v_max
		self.m = args.m
		self.a_const = args.a_const
		self.R = args.R
		self.x_max = args.x_max
		self.y_max = args.y_max # not use
		self.penalty = args.penalty
		self.terminate_if_crash = args.terminate_if_crash

		self.low = np.array([-float('inf')] * 4)
		self.high = self.low = np.array([float('inf')]*4)

		self.observation_space = spaces.Box(low = self.low,high = self.high)
		self.action_space = spaces.Discrete(9)
		# 0: no op
		# 1: up; 2: down; 3: left; 4: right
		# 5: up left; 6: up right
		# 7: down left; 8: down right

		self.a_list = [(0.,0.),  
				 (0., self.a_const), (0., -self.a_const), (-self.a_const, 0.), (self.a_const, 0.),
				 (self.a_const * np.cos(3*np.pi/4), self.a_const * np.sin(3*np.pi/4)), 
				 (self.a_const * np.cos(np.pi/4), self.a_const * np.sin(np.pi/4)),
				 (self.a_const * np.cos(5*np.pi/4), self.a_const * np.sin(5*np.pi/4)),
				 (self.a_const * np.cos(7*np.pi/4), self.a_const * np.sin(7*np.pi/4))]

		self.speed = np.array([0., 0.])
		self.pos = np.array([0., 0.])
		self.state = np.concatenate([self.pos, self.speed])
	
	def reset(self, seed=0):
		np.random.seed(seed)
		random.seed(seed)
		speed = np.random.uniform(low=-0.5, high=0.5, size=2)
		#speed = np.random.uniform(low=-self.v_max, high=self.v_max, size=2) # old
		self.speed = clip_norm(speed, self.v_max)
		self.pos = np.random.uniform(low=-1, high= 1, size = 2)
		self.state = np.concatenate([self.pos, self.speed])
		self.current_step = 0
		return self.state, {'crashed': False, 'cost': 0}

	def step(self, action):
		self.current_step += 1
		a_x, a_y = self.a_list[action]
		act = np.array([a_x, a_y])
		next_pos = self.pos + self.speed * self.delta_t + 0.5 * (self.delta_t**2) * act
		next_pos = np.clip(next_pos, a_min = -self.y_max, a_max=self.y_max)
		next_speed = clip_norm(self.speed + act * self.delta_t, self.v_max)
		reward = (-next_pos[1] * next_speed[0] + next_pos[0] * next_speed[1]) / (1. + np.abs(np.linalg.norm(next_pos) - self.R))
		done = False
		crash = False
		cost = 0
		if (np.abs(next_pos[0]) > self.x_max):
			cost = 1.
			crash=True
			if self.terminate_if_crash:
				done = True
			reward= self.penalty
		self.pos = next_pos
		self.speed = next_speed
		self.state = np.concatenate([self.pos, self.speed])
		info = {'cost': cost, 'crashed': crash}
		
		truncated = (self.current_step == self.ep_len)
		return self.state, reward, done, truncated, info
