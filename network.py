import torch
from torch import nn
import torch.nn.functional as F

import torch.nn as nn
import torch


class FeedForwardNN(nn.Module):
	def __init__(self, in_dim, out_dim, layer_dim=64, out_sigmoid=False):
		super(FeedForwardNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, layer_dim)
		self.layer2 = nn.Linear(layer_dim, layer_dim)
		self.layer3 = nn.Linear(layer_dim, out_dim)
		self.out_sigmoid = out_sigmoid
		self.out_dim = out_dim
		#nn.init.xavier_uniform_(self.layer1.weight)
		

	def forward(self, obs):
		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)
		if self.out_sigmoid:
			output = F.sigmoid(output)
		return output
	
	def crisp_forward(self, obs, threshold=0.5):
		x = self.forward(obs)
		x = torch.where(x<threshold, 0, x)
		x = torch.where(x>=threshold, 1, x)
		return x
