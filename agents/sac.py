import torch
from gymnasium import spaces
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn
from models.memory import Transition, ReplayMemory
from models.factory import loss_function_factory, optimizer_factory, \
      model_factory, size_model_config, trainable_parameters
from utils import DEVICE
from agents.base import AbstractStochasticAgent


def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * s_param.data + (1 - tau) * t_param.data)


class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()

        def straight_through(x):
            return x
        if config["common"] is None:
            self.common = straight_through
        else:
            self.common = model_factory(config["common"])
        self.actor = model_factory(config["actor"])
        self.critic1 = model_factory(config["critic"])
        self.critic2 = model_factory(config["critic"])

        self.target_critic1 = model_factory(config["critic"])
        self.target_critic2 = model_factory(config["critic"])
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_final = nn.Softmax(dim=-1)

    def actor_forward(self, state):
        x = self.common(state)
        logit = self.actor(x)
        return self.actor_final(logit)

    def critic_forward(self, state):
        x = self.common(state)
        return self.critic1(x), self.critic2(x)

    def target_critic_forward(self, state):
        x = self.common(state)
        return self.target_critic1(x), self.target_critic2(x)


class SACAgent(AbstractStochasticAgent):
    def __init__(self, env, config=None):
        super(SACAgent, self).__init__(config)
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete), \
            "Only compatible with Discrete action spaces."
        self.training = True
        self.previous_state = None
        self.action_size = env.action_space.n
        self.memory = ReplayMemory(self.config)

        self.target_entropy = self.config.get("target_entropy")
        if self.target_entropy is None:
            self.target_entropy = -np.log(self.action_size) * 0.98

        self.log_alpha = torch.tensor(np.log(self.config.get("alpha", 1.0)), dtype=torch.float32, requires_grad=True, device=DEVICE)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.get("alpha_lr", 1e-4))

        self.gamma = self.config.get("gamma", 0.99)
        self.target_update = self.config.get("target_update", 1)
        self.use_double = self.config.get("double", False)
        self.soft_tau = self.config.get("soft_tau", 0.005)

        if self.config["common"] is not None:
            size_model_config(self.env, self.config["common"])
        size_model_config(self.env, self.config["actor"])
        size_model_config(self.env, self.config["critic"])
        self.device = DEVICE
        self.actor_critic = ActorCritic(self.config).to(self.device)

        print("Number of trainable parameters: {}".format(trainable_parameters(self.actor_critic)))

        self.loss_function = loss_function_factory(self.config["loss_function"])
        self.optimizer = optimizer_factory(self.config["optimizer"]["type"],
                                           self.actor_critic.parameters(),
                                           **self.config["optimizer"])
        self.steps = 0

    def act(self, state):
        self.previous_state = state
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pi = self.actor_critic.actor_forward(state)
            pi = pi.squeeze(0)
            if self.training:
                distr = Categorical(probs=pi)
                return distr.sample().item()
            else:
                return torch.argmax(pi).item()

    @classmethod
    def default_config(cls):
        return dict(common=dict(type="MultiLayerPerceptron"),
                    actor=dict(type="MultiLayerPerceptron"),
                    critic=dict(type="MultiLayerPerceptron"),
                    optimizer=dict(type="ADAM",
                                   lr=1e-4,
                                   weight_decay=0,
                                   k=5),
                    loss_function="l2",
                    memory_capacity=50000,
                    batch_size=100,
                    gamma=0.99,
                    device="cuda:best",
                    alpha=1.0,
                    alpha_lr=1e-4,
                    target_entropy=None,
                    target_update=1,
                    soft_tau=0.005,
                    double=True)

    def record(self, state, action, reward, next_state, done, end, info):
        if not self.training:
            return

        self.memory.push(state, action, reward, next_state, done, info)
        self.steps += 1
        batch = self.sample_minibatch()
        if batch and self.steps >= 256:
            q_loss, actor_loss, entropy = self.compute_loss(batch)

            self.optimizer.zero_grad()
            (q_loss + actor_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=10.0)
            self.optimizer.step()

            alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            
            self.update_target_network()

            if self.writer:
                self.writer.add_scalar("loss/q_loss", q_loss.item(), self.steps)
                self.writer.add_scalar("loss/actor_loss", actor_loss.item(), self.steps)
                self.writer.add_scalar("loss/alpha_loss", alpha_loss.item(), self.steps)
                self.writer.add_scalar("entropy/alpha", self.log_alpha.exp().item(), self.steps)
                self.writer.add_scalar("entropy/policy_entropy", entropy.item(), self.steps)

    def sample_minibatch(self):
        if len(self.memory) < self.config["batch_size"]:
            return None
        transitions = self.memory.sample(self.config["batch_size"])
        return Transition(*zip(*transitions))

    def compute_loss(self, batch):
        if not isinstance(batch.state, torch.Tensor):
            state = torch.cat(tuple(torch.tensor(np.array([batch.state]), dtype=torch.float))).to(self.device)
            action = torch.tensor(np.array(batch.action), dtype=torch.long).to(self.device)
            reward = torch.tensor(np.array(batch.reward), dtype=torch.float).to(self.device)
            next_state = torch.cat(tuple(torch.tensor(np.array([batch.next_state]), dtype=torch.float))).to(self.device)
            terminal = torch.tensor(np.array(batch.terminal), dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, batch.info)

        state = batch.state
        action = batch.action.unsqueeze(1)
        reward = batch.reward.unsqueeze(1)
        next_state = batch.next_state
        terminal = batch.terminal.unsqueeze(1).float()

        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_probs = self.actor_critic.actor_forward(next_state)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1, next_q2 = self.actor_critic.target_critic_forward(next_state)
            next_q = torch.min(next_q1, next_q2) if self.use_double else next_q1
            next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = reward + (1 - terminal) * self.gamma * next_v
            target_q = target_q.clamp(min=-10.0, max=10.0)

        q1, q2 = self.actor_critic.critic_forward(state)
        q1_pred = q1.gather(1, action)
        q2_pred = q2.gather(1, action)
        q_loss = self.loss_function(q1_pred, target_q.detach()) + self.loss_function(q2_pred, target_q.detach())

        probs = self.actor_critic.actor_forward(state)
        log_probs = torch.log(probs + 1e-8)
        q = torch.min(q1, q2) if self.use_double else q1
        actor_loss = (probs * (alpha * log_probs - q)).sum(dim=1).mean()
        entropy = -(probs * log_probs).sum(dim=1).mean()

        return q_loss, actor_loss, entropy

    def update_target_network(self):
        soft_update(self.actor_critic.target_critic1, self.actor_critic.critic1, self.soft_tau)
        soft_update(self.actor_critic.target_critic2, self.actor_critic.critic2, self.soft_tau)

    def get_batch_state_values(self, states):
        raise NotImplementedError

    def get_batch_state_action_values(self, states):
        raise NotImplementedError

    def get_state_value(self, state):
        raise NotImplementedError

    def get_state_action_values(self, state):
        raise NotImplementedError

    def save(self, filename):
        state = {'state_dict': self.actor_critic.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'log_alpha': self.log_alpha.detach().cpu().item(),
                 'alpha_optimizer': self.alpha_optimizer.state_dict()}
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.log_alpha = torch.tensor(np.log(checkpoint['log_alpha']), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
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