from gym import spaces

import numpy as np
import random

class ACC():
	'''
	Adaptive cruise control: We use a relavice reference frame in which the
	ego car gets a reward for being close to the lead car.
	
	Observation space Box(3): negative distance between the cars, speed of ego car, speed of leading car
	Discrete action space (3): decelerate (0), do nothing (1), accelerate (2)

	'''
	def __init__(self, args):
		self.args=args
		self.ep_len = args.ep_len # 100
		self.mu = args.mu # 0
		self.sigma = args.sigma # 0.2
		self.penalty = args.penalty # 0
		self.speed_min = 0.0 #speed_min
		self.speed_max = 10.0 #speed_max
		self.a_min = -1
		self.a_max = 1

		self.delta_t = args.delta_t
		self.current_step = 0

		self.state = np.array([0.,0.,0.])
		self.speed=0.
		self.r_pos = -5
		self.lead_speed = 0.


		self.low = np.array([-10, self.speed_min, self.speed_min])
		self.high = np.array([0, self.speed_max, self.speed_max])

		self.observation_space = spaces.Box(low = self.low,high = self.high)
		self.action_space = spaces.Discrete(3)

	def reset(self, seed=0):
		np.random.seed(seed)
		random.seed(seed)
		self.speed=0.
		self.r_pos = -5 - np.random.rand()
		self.lead_speed = 0.
		self.state = np.array([self.r_pos, self.speed, self.lead_speed])
		self.current_step = 0
		return self.state, {"crashed": False, "cost": 0}

	def sample_lead_a(self):
		sample = np.random.normal(self.mu, self.sigma)
		lead_a = max(-1, min(1, sample))
		return lead_a
	
	def scaler(self, state):
		return 2 * (state - self.low) / (self.high - self.low) - 1.

	def step(self, action):
		self.current_step += 1

		u = action -1 + 0.1* (np.random.rand() - 0.5) # add mild noise here
		speed = max(self.speed_min, min(self.speed_max, self.speed + u * self.delta_t))
		sample = self.sample_lead_a()
		lead_speed = max(self.speed_min, min(self.speed_max, self.lead_speed + sample * self.delta_t))

		relative_pos = self.r_pos + (speed - lead_speed) * self.delta_t

		reward = (10 + relative_pos) * 0.1
		done = False
		crash = False
		cost = 0
		if relative_pos <= -10 or relative_pos >= 0:
			done=True
			crash = True
			cost = 1.
			reward = self.penalty
		truncated = (self.current_step == self.ep_len)
		self.speed = speed; self.lead_speed = lead_speed; self.r_pos=relative_pos
		self.state = np.array([self.r_pos, self.speed, self.lead_speed])
		info = {"crashed": crash, "cost": cost}
		return self.state, reward, done, truncated, info

