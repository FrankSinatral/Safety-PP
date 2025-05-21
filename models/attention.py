import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from torch.nn.functional import one_hot

from configuration import Configurable
from models.base import BaseModule, base_model_factory


class EgoAttention(BaseModule, Configurable):
	def __init__(self, config):
		super().__init__()
		Configurable.__init__(self, config)
		self.features_per_head = int(self.config["feature_size"] / self.config["heads"])

		self.value_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
		self.key_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
		self.query_ego = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
		self.attention_combine = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)

	@classmethod
	def default_config(cls):
		return {
			"feature_size": 64,
			"heads": 4,
			"dropout_factor": 0,
			"k_attn": -1,
		}

	def forward(self, ego, others, mask=None):
		batch_size = others.shape[0]
		n_entities = others.shape[1] + 1
		input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)
		# Dimensions: Batch, entity, head, feature_per_head
		key_all = self.key_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
		value_all = self.value_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
		query_ego = self.query_ego(ego).view(batch_size, 1, self.config["heads"], self.features_per_head)

		# Dimensions: Batch, head, entity, feature_per_head
		key_all = key_all.permute(0, 2, 1, 3)
		value_all = value_all.permute(0, 2, 1, 3)
		query_ego = query_ego.permute(0, 2, 1, 3)
		if mask is not None:
			mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.config["heads"], 1, 1))
		value, attention_matrix = attention(query_ego, key_all, value_all, mask,
											nn.Dropout(self.config["dropout_factor"]), 
											self.config["k_attn"])
		# Jinpeng 202305
		result = (self.attention_combine(value.reshape((batch_size, self.config["feature_size"]))) + ego.squeeze(1))/2
		return result, attention_matrix


class SelfAttention(BaseModule, Configurable):
	def __init__(self, config):
		super().__init__()
		Configurable.__init__(self, config)
		self.features_per_head = int(self.config["feature_size"] / self.config["heads"])

		self.value_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
		self.key_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
		self.query_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
		self.attention_combine = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)

	@classmethod
	def default_config(cls):
		return {
			"feature_size": 64,
			"heads": 4,
			"dropout_factor": 0,
			"k_attn": -1,
		}

	def forward(self, ego, others, mask=None):
		batch_size = others.shape[0]
		n_entities = others.shape[1] + 1
		input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)
		# Dimensions: Batch, entity, head, feature_per_head
		key_all = self.key_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
		value_all = self.value_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
		query_all = self.query_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)

		# Dimensions: Batch, head, entity, feature_per_head
		key_all = key_all.permute(0, 2, 1, 3)
		value_all = value_all.permute(0, 2, 1, 3)
		query_all = query_all.permute(0, 2, 1, 3)
		if mask is not None:
			mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.config["heads"], 1, 1))
		value, attention_matrix = attention(query_all, key_all, value_all, mask,
											nn.Dropout(self.config["dropout_factor"]),
											self.config["k_attn"])
		result = (self.attention_combine(value.reshape((batch_size, n_entities, self.config["feature_size"]))) + input_all)/2
		return result, attention_matrix


class EgoAttentionNetwork(BaseModule, Configurable):
	def __init__(self, config):
		super().__init__()
		Configurable.__init__(self, config)
		print("Using MLP ego attention...")
		self.config = config
		if not self.config["embedding_layer"]["in"]:
			self.config["embedding_layer"]["in"] = self.config["in"]
		if not self.config["others_embedding_layer"]["in"]:
			self.config["others_embedding_layer"]["in"] = self.config["in"]
		if self.config["output_layer"]:
			self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
			self.config["output_layer"]["out"] = self.config["out"]

		self.ego_embedding = base_model_factory(self.config["embedding_layer"])
		self.others_embedding = base_model_factory(self.config["others_embedding_layer"])
		self.self_attention_layer = None
		if self.config["self_attention_layer"]:
			self.self_attention_layer = SelfAttention(self.config["self_attention_layer"])
		self.attention_layer = EgoAttention(self.config["attention_layer"])
		if self.config["output_layer"]:
			self.output_layer = base_model_factory(self.config["output_layer"])

	@classmethod
	def default_config(cls):
		return {
			"in": None,
			"out": None,
			"presence_feature_idx": 0,
			"embedding_layer": {
				"type": "MultiLayerPerceptron",
				"layers": [128, 128, 128],
				"reshape": False
			},
			"others_embedding_layer": {
				"type": "MultiLayerPerceptron",
				"layers": [128, 128, 128],
				"reshape": False
			},
			"self_attention_layer": {
				"type": "SelfAttention",
				"feature_size": 128,
				"heads": 4,
				"k_attn": -1,
			},
			"attention_layer": {
				"type": "EgoAttention",
				"feature_size": 128,
				"heads": 4,
				"k_attn": -1,
			},
			"output_layer": {
				"type": "MultiLayerPerceptron",
				"layers": [128, 128, 128],
				"reshape": False
			},
		}

	def forward(self, x):
		# Jinpeng 202305
		ego_embedded_att, _ = self.forward_attention(x)
		if self.config["output_layer"]:
			return self.output_layer(ego_embedded_att)
		else:
			return ego_embedded_att

	def split_input(self, x, mask=None):
		# Dims: batch, entities, features
		ego = x[:, 0:1, :]
		others = x[:, 1:, :]
		if mask is None:
			mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
		return ego, others, mask

	def forward_attention(self, x):
		ego, others, mask = self.split_input(x)
		ego, others = self.ego_embedding(ego), self.others_embedding(others)
		if self.self_attention_layer:
			self_att, _ = self.self_attention_layer(ego, others, mask)
			ego, others, mask = self.split_input(self_att, mask=mask)
		return self.attention_layer(ego, others, mask)

	def get_attention_matrix(self, x):
		_, attention_matrix = self.forward_attention(x)
		return attention_matrix


class AttentionNetwork(BaseModule, Configurable):
	def __init__(self, config):
		super().__init__()
		Configurable.__init__(self, config)
		self.config = config
		if not self.config["embedding_layer"]["in"]:
			self.config["embedding_layer"]["in"] = self.config["in"]
		self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
		self.config["output_layer"]["out"] = self.config["out"]

		self.embedding = base_model_factory(self.config["embedding_layer"])
		self.attention_layer = SelfAttention(self.config["attention_layer"])
		self.output_layer = base_model_factory(self.config["output_layer"])

	@classmethod
	def default_config(cls):
		return {
			"in": None,
			"out": None,
			"presence_feature_idx": 0,
			"embedding_layer": {
				"type": "MultiLayerPerceptron",
				"layers": [128, 128, 128],
				"reshape": False
			},
			"attention_layer": {
				"type": "SelfAttention",
				"feature_size": 128,
				"heads": 4,
				"k_attn": -1,
			},
			"output_layer": {
				"type": "MultiLayerPerceptron",
				"layers": [128, 128, 128],
				"reshape": False
			},
		}

	def forward(self, x):
		ego, others, mask = self.split_input(x)
		ego_embedded_att, _ = self.attention_layer(self.embedding(ego), self.others_embedding(others), mask)
		return self.output_layer(ego_embedded_att)

	def split_input(self, x):
		# Dims: batch, entities, features
		ego = x[:, 0:1, :]
		others = x[:, 1:, :]
		mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
		return ego, others, mask

	def get_attention_matrix(self, x):
		ego, others, mask = self.split_input(x)
		_, attention_matrix = self.attention_layer(self.embedding(ego), self.others_embedding(others), mask)
		return attention_matrix


def attention(query, key, value, mask=None, dropout=None, k_attn=-1):
	"""
		Compute a Scaled Dot Product Attention.
	:param query: size: batch, head, 1 (ego-entity), features
	:param key:  size: batch, head, entities, features
	:param value: size: batch, head, entities, features
	:param mask: size: batch,  head, 1 (absence feature), 1 (ego-entity)
	:param dropout:
	:return: the attention softmax(QK^T/sqrt(dk))V
	"""
	# Jinpeng 202305
	# Modify here: choose the first k-th large scores
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask, -1e9)
	p_attn = F.softmax(scores, dim=-1) # p_attn: size: batch, head, 1, entities
	if dropout is not None:
		p_attn = dropout(p_attn)
	if k_attn > 0:
		_, idx = torch.topk(p_attn, k=k_attn, dim = -1) 
		mask_attn = one_hot(idx, num_classes = p_attn.shape[-1]).sum(dim=-2)
		p_attn = p_attn * mask_attn
	
	output = torch.matmul(p_attn, value)
	return output, p_attn



########################################


class EgoAttentionNetworkWithVp(BaseModule, Configurable):
	def __init__(self, config):
		super().__init__()
		Configurable.__init__(self, config)
		print("Using MLP ego attention with vp...")
		self.config = config
		if not self.config["embedding_layer"]["in"]:
			self.config["embedding_layer"]["in"] = self.config["in"]
		if not self.config["others_embedding_layer"]["in"]:
			self.config["others_embedding_layer"]["in"] = self.config["in"]
		if self.config["output_layer"]:
			self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
			self.config["output_layer"]["out"] = self.config["out"]
		if self.config["vp_layer"]:
			self.config["vp_layer"]["in"] = self.config["attention_layer"]["feature_size"]
			self.config["vp_layer"]["out"] = self.config["out"]

		self.ego_embedding = base_model_factory(self.config["embedding_layer"])
		self.others_embedding = base_model_factory(self.config["others_embedding_layer"])
		self.self_attention_layer = None
		if self.config["self_attention_layer"]:
			self.self_attention_layer = SelfAttention(self.config["self_attention_layer"])
		self.attention_layer = EgoAttention(self.config["attention_layer"])
		if self.config["output_layer"]:
			self.output_layer = base_model_factory(self.config["output_layer"])
		if self.config["vp_layer"]:
			self.vp_layer = base_model_factory(self.config["vp_layer"])

	@classmethod
	def default_config(cls):
		return {
			"in": None,
			"out": None,
			"presence_feature_idx": 0,
			"embedding_layer": {
				"type": "MultiLayerPerceptron",
				"layers": [128, 128, 128],
				"reshape": False
			},
			"others_embedding_layer": {
				"type": "MultiLayerPerceptron",
				"layers": [128, 128, 128],
				"reshape": False
			},
			"self_attention_layer": {
				"type": "SelfAttention",
				"feature_size": 128,
				"heads": 4,
				"k_attn": -1,
			},
			"attention_layer": {
				"type": "EgoAttention",
				"feature_size": 128,
				"heads": 4,
				"k_attn": -1,
			},
			"output_layer": {
				"type": "MultiLayerPerceptron",
				"layers": [128, 128, 128],
				"reshape": False
			},
			"vp_layer": {
				"type": "MultiLayerPerceptron",
				"layers": [128, 128, 128],
				"reshape": False,
				"output_activation": "SIGMOID"
			},
		}

	def output_forward(self, x):
		# Jinpeng 202305
		ego_embedded_att, _ = self.forward_attention(x)
		if self.config["output_layer"]:
			return self.output_layer(ego_embedded_att)
		else:
			return ego_embedded_att
		
	def vp_forward(self, x):
		ego_embedded_att, _ = self.forward_attention(x)
		if self.config["vp_layer"]:
			return self.vp_layer(ego_embedded_att)
		else:
			return ego_embedded_att
	
	def crisp_forward(self,x, threshold=0.5):
		x = self.vp_forward(x)
		x = torch.where(x<threshold, 0, x)
		x = torch.where(x>=threshold, 1, x)
		return x


	def split_input(self, x, mask=None):
		# Dims: batch, entities, features
		ego = x[:, 0:1, :]
		others = x[:, 1:, :]
		if mask is None:
			mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
		return ego, others, mask

	def forward_attention(self, x):
		ego, others, mask = self.split_input(x)
		ego, others = self.ego_embedding(ego), self.others_embedding(others)
		if self.self_attention_layer:
			self_att, _ = self.self_attention_layer(ego, others, mask)
			ego, others, mask = self.split_input(self_att, mask=mask)
		return self.attention_layer(ego, others, mask)

	def get_attention_matrix(self, x):
		_, attention_matrix = self.forward_attention(x)
		return attention_matrix