import torch
from gymnasium import spaces
import random
import numpy as np

from models.memory import Transition, ReplayMemory
from models.factory import loss_function_factory, optimizer_factory, \
      model_factory, size_model_config, trainable_parameters
from utils import DEVICE
from agents.base import AbstractStochasticAgent


class DQNAgent(AbstractStochasticAgent):
    def __init__(self, env, config=None):
        super(DQNAgent, self).__init__(config)
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete), \
            "Only compatible with Discrete action spaces."
        self.memory = ReplayMemory(self.config)
        self.training = True
        self.previous_state = None

        self.eps_start = self.config["eps_start"]
        self.eps = self.eps_start
        self.eps_decay = self.config["eps_decay"]
        self.eps_min = self.config["eps_min"]
        self.explore_time = 0
        self.action_size = env.action_space.n

        size_model_config(self.env, self.config["model"])
        self.value_net = model_factory(self.config["model"])
        self.target_net = model_factory(self.config["model"])
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        print("Number of trainable parameters: {}".format(trainable_parameters(self.value_net)))
        self.device = DEVICE
        self.value_net.to(self.device)
        self.target_net.to(self.device)
        self.loss_function = loss_function_factory(self.config["loss_function"])
        self.optimizer = optimizer_factory(self.config["optimizer"]["type"],
                                           self.value_net.parameters(),
                                           **self.config["optimizer"])
        self.steps = 0

    def act(self, state):
        """
            Act according to the state-action value model and an exploration policy
        :param state: current state
        :param step_exploration_time: step the exploration schedule
        :return: an action
        """
        self.previous_state = state
        # Single-agent setting
        with torch.no_grad():
            values = self.get_state_action_values(state)
        if self.training:
            # Epsilon-greedy action selection
            if random.random() > self.eps:
                return np.argmax(values)
            else:
                return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(values)       

    
    @classmethod
    def default_config(cls):
        return dict(model=dict(type="DuelingNetwork"),
                    optimizer=dict(type="ADAM",
                                   lr=5e-4,
                                   weight_decay=0,
                                   k=5),
                    loss_function="l2",
                    memory_capacity=50000,
                    batch_size=100,
                    gamma=0.99,
                    device="cuda:best",
                    eps_start = 1.0,
                    eps_decay = 0.995,
                    eps_min = 0.01,
                    target_update=4,
                    double=True)
    
    def action_distribution(self, state):
        values = self.get_state_action_values(state)
        distribution = {action: 0 for action in range(self.action_size)}
        optimal_action = np.argmax(values)
        distribution[optimal_action] += 1
        return distribution
    
    def record(self, state, action, reward, next_state, done, end, info):
        """
            Record a transition by performing a Deep Q-Network iteration

            - push the transition into memory
            - sample a minibatch
            - compute the bellman residual loss over the minibatch
            - perform one gradient descent step
            - slowly track the policy network with the target network
        :param state: a state
        :param action: an action
        :param reward: a reward
        :param next_state: a next state
        :param done: whether state is terminal
        """
        if not self.training:
            return
        # Single-agent setting
        # this is set for heter-highway-env
        self.memory.push(state, action, reward, next_state, done, info)
        batch = self.sample_minibatch()
        if batch:
            loss, _, _ = self.compute_bellman_residual(batch)
            self.step_optimizer(loss)
            self.update_target_network()
    
    def sample_minibatch(self):
        if len(self.memory) < self.config["batch_size"]:
            return None
        transitions = self.memory.sample(self.config["batch_size"])
        return Transition(*zip(*transitions))
    
    def update_target_network(self):
        self.steps += 1
        if self.steps % self.config["target_update"] == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())

    def step_optimizer(self, loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def compute_bellman_residual(self, batch, target_state_action_value=None):
        # Compute concatenate the batch elements
        if not isinstance(batch.state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(tuple(torch.tensor(np.array([batch.state]), dtype=torch.float))).to(self.device)
            action = torch.tensor(np.array(batch.action), dtype=torch.long).to(self.device)
            reward = torch.tensor(np.array(batch.reward), dtype=torch.float).to(self.device)
            next_state = torch.cat(tuple(torch.tensor(np.array([batch.next_state]), dtype=torch.float))).to(self.device)
            terminal = torch.tensor(np.array(batch.terminal), dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.value_net(batch.state)
        state_action_values = state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(batch.reward.shape).to(self.device)
                if self.config["double"]:
                    # Double Q-learning: pick best actions from policy network
                    _, best_actions = self.value_net(batch.next_state).max(1)
                    # Double Q-learning: estimate action values from target network
                    best_values = self.target_net(batch.next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    best_values, _ = self.target_net(batch.next_state).max(1)
                next_state_values[~batch.terminal] = best_values[~batch.terminal]
                # Compute the expected Q values
                target_state_action_value = batch.reward + self.config["gamma"] * next_state_values

        # Compute loss
        loss = self.loss_function(state_action_values, target_state_action_value.detach())
        return loss, target_state_action_value, batch

    def get_batch_state_values(self, states):
        values, actions = self.value_net(torch.tensor(np.array(states), dtype=torch.float).to(self.device)).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        # Jinpeng 202305
        # if want to use sparse attention, then add choosing attention here
        return self.value_net(torch.tensor(np.array(states), dtype=torch.float).to(self.device)).data.cpu().numpy()
    
    def get_state_value(self, state):
        """
        :param state: s, an environment state
        :return: V, its state-value
        """
        values, actions = self.get_batch_state_values([state])
        return values[0], actions[0]

    def get_state_action_values(self, state):
        """
        :param state: s, an environment state
        :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions
        """
        return self.get_batch_state_action_values([state])[0]

    def save(self, filename):
        state = {'state_dict': self.value_net.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return filename

    def initialize_model(self):
        self.value_net.reset()

    def set_writer(self, writer):
        super().set_writer(writer)
        self.writer.add_scalar("agent/trainable_parameters", trainable_parameters(self.value_net), 0)
    
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

    def schedule_explore(self):
        self.explore_time += 1
        self.eps = max(self.eps * self.eps_decay, self.eps_min)
