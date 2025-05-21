import torch
from torch.nn import functional as F
import torch.nn as nn
from gymnasium import spaces
import numpy as np

from models.base import MultiLayerPerceptron, MLPWithVp
from models.attention import EgoAttentionNetwork, EgoAttentionNetworkWithVp


def model_factory(config: dict) -> nn.Module:
    if config["type"] == "MultiLayerPerceptron":
        return MultiLayerPerceptron(config)

    elif config["type"] == "EgoAttentionNetwork":
        return EgoAttentionNetwork(config)
    
    elif config["type"] == "EgoAttentionNetworkWithVp":
        return EgoAttentionNetworkWithVp(config)
    
    elif config["type"] == "MLPWithVp":
        return MLPWithVp(config)

    else:
        raise ValueError("Unknown model type")
    

def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def size_model_config(env, model_config):
    """
        Update the configuration of a model depending on the environment observation/action spaces

        Typically, the input/output sizes.

    :param env: an environment
    :param model_config: a model configuration
    """

    if isinstance(env.observation_space, spaces.Box):
        obs_shape = env.observation_space.shape
    elif isinstance(env.observation_space, spaces.Tuple):
        obs_shape = env.observation_space.spaces[0].shape
    if model_config["type"] == "ConvolutionalNetwork":  # Assume CHW observation space
        model_config["in_channels"] = int(obs_shape[0])
        model_config["in_height"] = int(obs_shape[1])
        model_config["in_width"] = int(obs_shape[2])
    else:
        if not model_config.get("in", None):
            model_config["in"] = int(np.prod(obs_shape))

    if not model_config.get("out", None):
        if isinstance(env.action_space, spaces.Discrete):
            model_config["out"] = int(env.action_space.n)
        elif isinstance(env.action_space, spaces.Tuple):
            model_config["out"] = int(env.action_space.spaces[0].n)


def loss_function_factory(loss_function):
    if loss_function == "l2":
        return F.mse_loss
    elif loss_function == "l1":
        return F.l1_loss
    elif loss_function == "smooth_l1":
        return F.smooth_l1_loss
    elif loss_function == "bce":
        return F.binary_cross_entropy
    else:
        raise ValueError("Unknown loss function : {}".format(loss_function))


def optimizer_factory(optimizer_type, params, lr=None, weight_decay=None, k=None, **kwargs):
    if optimizer_type == "ADAM":
        return torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "RMS_PROP":
        return torch.optim.RMSprop(params=params, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer type: {}".format(optimizer_type))
    
def optimizer_factory_new(optimizer_type, param_groups, **kwargs):
    if optimizer_type == "ADAM":
        return torch.optim.Adam(param_groups, **kwargs)
    elif optimizer_type == "RMS_PROP":
        return torch.optim.RMSprop(param_groups, **kwargs)
    else:
        raise ValueError("Unknown optimizer type: {}".format(optimizer_type))