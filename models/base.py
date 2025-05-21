import torch
from torch.nn import functional as F
import torch.nn as nn

from configuration import Configurable

def base_model_factory(config: dict) -> nn.Module:
	if config["type"] == "MultiLayerPerceptron":
		return MultiLayerPerceptron(config)
	else:
		raise ValueError("Unknown model type")

def activation_factory(activation_type):
	if activation_type == "RELU":
		return F.relu
	elif activation_type == "TANH":
		return torch.tanh
	elif activation_type == 'SIGMOID':
		return torch.sigmoid
	else:
		raise ValueError("Unknown activation_type: {}".format(activation_type))

class BaseModule(torch.nn.Module):
	"""
		Base torch.nn.Module implementing basic features:
			- initialization factory
			- normalization parameters
	"""
	def __init__(self, activation_type="RELU", reset_type="XAVIER", normalize=None):
		super().__init__()
		self.activation = activation_factory(activation_type)
		self.reset_type = reset_type
		self.normalize = normalize
		self.mean = None
		self.std = None

	def _init_weights(self, m):
		if hasattr(m, 'weight'):
			if self.reset_type == "XAVIER":
				torch.nn.init.xavier_uniform_(m.weight.data)
			elif self.reset_type == "ZEROS":
				torch.nn.init.constant_(m.weight.data, 0.)
			else:
				raise ValueError("Unknown reset type")
		if hasattr(m, 'bias') and m.bias is not None:
			torch.nn.init.constant_(m.bias.data, 0.)

	def set_normalization_params(self, mean, std):
		if self.normalize:
			std[std == 0.] = 1.
		self.std = std
		self.mean = mean

	def reset(self):
		self.apply(self._init_weights)

	def forward(self, *input):
		if self.normalize:
			input = (input.float() - self.mean.float()) / self.std.float()
		return NotImplementedError

class MultiLayerPerceptron(BaseModule, Configurable):
	def __init__(self, config):
		super().__init__()
		Configurable.__init__(self, config)
		sizes = [self.config["in"]] + self.config["layers"]
		self.activation = activation_factory(self.config["activation"])
		if self.config.get("output_activate", None) is not None:
			self.output_activation = activation_factory(self.config["output_activation"])
		layers_list = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
		self.layers = nn.ModuleList(layers_list)
		if self.config.get("out", None):
			self.predict = nn.Linear(sizes[-1], self.config["out"])

	@classmethod
	def default_config(cls):
		return {"in": None,
				"layers": [64, 64],
				"activation": "RELU",
				"reshape": "True",
				"output_activation": None,
				"out": None}

	def forward(self, x):
		if self.config["reshape"]:
			x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
		for layer in self.layers:
			x = self.activation(layer(x))
		if self.config.get("out", None):
			x = self.predict(x)
		if self.config.get("output_activate", None) is not None:
			x = self.output_activation(x)
		return x


class MLPWithVp(BaseModule, Configurable):
	def __init__(self, config):
		super().__init__()
		Configurable.__init__(self, config)
		self.config["output_layer"]["in"] = config["in"]
		self.config["output_layer"]["out"] = config["out"]
		self.config["vp_layer"]["in"] = config["in"]
		self.config["vp_layer"]["out"] = config["out"]
		self.output_layer = MultiLayerPerceptron(self.config['output_layer'])
		self.vp_layer = MultiLayerPerceptron(self.config['vp_layer'])
		print('Using MLP with vp...')
	
	@classmethod
	def default_config(cls):
		return {"in": None,
				"output_layer": {
					"type": "MultiLayerPerceptron",
					"layers": [128,128],
					"reshape": True
				},
				"vp_layer": {
					"type": "MultiLayerPerceptron",
					"layers": [128,128],
					"reshape": True
				},
				"out": None}
	
	def output_forward(self, x):
		return self.output_layer.forward(x)
	
	def vp_forward(self, x):
		return self.vp_layer.forward(x)
	
	def crisp_forward(self,x, threshold=0.5):
		x = self.vp_forward(x)
		x = torch.where(x<threshold, 0, x)
		x = torch.where(x>=threshold, 1, x)
		return x
		
