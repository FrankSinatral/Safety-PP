import random
from collections.__init__ import namedtuple

from configuration import Configurable

import torch
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal', 'cost', 'info'))
# Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal', 'info'))


class ReplayMemory(Configurable):
    """
        Container that stores and samples transitions.
    """
    def __init__(self, config=None, transition_type=Transition):
        super(ReplayMemory, self).__init__(config)
        self.capacity = int(self.config['memory_capacity'])
        self.transition_type = transition_type
        self.memory = []
        self.position = 0

    @classmethod
    def default_config(cls):
        return dict(memory_capacity=10000,
                    n_steps=1,
                    gamma=0.99)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.position = len(self.memory) - 1
        elif len(self.memory) > self.capacity:
            self.memory = self.memory[:self.capacity]
        # Faster than append and pop
        self.memory[self.position] = self.transition_type(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, collapsed=True):
        """
            Sample a batch of transitions.

            If n_steps is greater than one, the batch will be composed of lists of successive transitions.
        :param batch_size: size of the batch
        :param collapsed: whether successive transitions must be collapsed into one n-step transition.
        :return: the sampled batch
        """
        # TODO: use agent's np_random for seeding
        if self.config["n_steps"] == 1:
            # Directly sample transitions
            return random.sample(self.memory, batch_size)
        else:
            # Sample initial transition indexes
            indexes = random.sample(range(len(self.memory)), batch_size)
            # Get the batch of n-consecutive-transitions starting from sampled indexes
            all_transitions = [self.memory[i:i+self.config["n_steps"]] for i in indexes]
            # Collapse transitions
            return map(self.collapse_n_steps, all_transitions) if collapsed else all_transitions

    def collapse_n_steps(self, transitions):
        """
            Collapse n transitions <s,a,r,s',t> of a trajectory into one transition <s0, a0, Sum(r_i), sp, tp>.

            We start from the initial state, perform the first action, and then the return estimate is formed by
            accumulating the discounted rewards along the trajectory until a terminal state or the end of the
            trajectory is reached.
        :param transitions: A list of n successive transitions
        :return: The corresponding n-step transition
        """
        state, action, cumulated_reward, next_state, done, info = transitions[0]
        discount = 1
        for transition in transitions[1:]:
            if done:
                break
            else:
                _, _, reward, next_state, done, info = transition
                discount *= self.config['gamma']
                cumulated_reward += discount*reward
        return state, action, cumulated_reward, next_state, done, info

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) == self.capacity

    def is_empty(self):
        return len(self.memory) == 0





class PrioritizedReplayMemory(Configurable):
    """
        Container that stores and samples transitions.
    """
    def __init__(self, config=None, transition_type=Transition):
        super(PrioritizedReplayMemory, self).__init__(config)
        
        self.transition_type = transition_type
        self.memory = []
        self.position = 0

        self.capacity = int(self.config['memory_capacity'])
        self.real_size = 0
        # self.count = 0
        self.tree = SumTree(self.capacity)

    @classmethod
    def default_config(cls):
        return dict(memory_capacity=100000,
                    n_steps=1,
                    gamma=0.99,
                    alpha=0.6,
                    beta=0.4)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.position = len(self.memory) - 1
        elif len(self.memory) > self.capacity:
            self.memory = self.memory[:self.capacity]
        # Faster than append and pop
        self.memory[self.position] = self.transition_type(*args)

        # 如果是第一条经验，初始化优先级为1.0；否则，对于新存入的经验，指定为当前最大的优先级
        priority = 1.0 if self.real_size == 0 else self.tree.priority_max
        # self.tree.update_priority(data_index=self.count, priority=priority)  # 更新当前经验在sum_tree中的优先级
        self.tree.update_priority(data_index=self.position, priority=priority)  # 更新当前经验在sum_tree中的优先级
        self.real_size = min(self.capacity, self.real_size + 1)
        # self.count = (self.count+1) % self.capacity
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sample_idxs, Normed_IS_weight = self.tree.prioritized_sample(N=self.real_size, batch_size=batch_size, beta=self.config["beta"])
        batch = [self.memory[a] for a in list(sample_idxs)]
        return batch, Normed_IS_weight, sample_idxs
    
    def uniform_sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
        priorities = (np.abs(td_errors) + 0.01) ** self.config["alpha"]
        for index, priority in zip(batch_index, priorities):
            self.tree.update_priority(data_index=index, priority=priority)

    def __len__(self):
        return self.real_size

    def is_full(self):
        return len(self.memory) == self.capacity

    def is_empty(self):
        return len(self.memory) == 0





class SumTree(object):
    """
    Story data with its priority in the tree.
    Tree structure and array storage:

    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions

    Array type for storing:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity  # buffer的容量
        self.tree_capacity = 2 * buffer_capacity - 1  # sum_tree的容量
        self.tree = np.zeros(self.tree_capacity)

    def update_priority(self, data_index, priority):
        ''' Update the priority for one transition according to its index in buffer '''
        # data_index表示当前数据在buffer中的index
        # tree_index表示当前数据在sum_tree中的index
        tree_index = data_index + self.buffer_capacity - 1  # 把当前数据在buffer中的index转换为在sum_tree中的index
        change = priority - self.tree[tree_index]  # 当前数据的priority的改变量
        self.tree[tree_index] = priority  # 更新树的最后一层叶子节点的优先级
        # then propagate the change through the tree
        while tree_index != 0:  # 更新上层节点的优先级，一直传播到最顶端
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def prioritized_sample(self, N, batch_size, beta):
        ''' sample a batch of index and normlized IS weight according to priorites '''
        batch_index = np.zeros(batch_size, dtype=np.int32)
        IS_weight = torch.zeros(batch_size, dtype=torch.float32)
        segment = self.priority_sum / batch_size  # 把[0,priority_sum]等分成batch_size个区间，在每个区间均匀采样一个数
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            buffer_index, priority = self._get_index(v)
            batch_index[i] = buffer_index
            prob = priority / self.priority_sum  # 当前数据被采样的概率
            IS_weight[i] = (N * prob) ** (-beta)
        Normed_IS_weight = IS_weight / IS_weight.max()  # normalization

        return batch_index, Normed_IS_weight

    def _get_index(self, v):
        ''' sample a index '''
        parent_idx = 0  # 从树的顶端开始
        while True:
            child_left_idx = 2 * parent_idx + 1  # 父节点下方的左右两个子节点的index
            child_right_idx = child_left_idx + 1
            if child_left_idx >= self.tree_capacity:  # reach bottom, end search
                tree_index = parent_idx  # tree_index表示采样到的数据在sum_tree中的index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[child_left_idx]:
                    parent_idx = child_left_idx
                else:
                    v -= self.tree[child_left_idx]
                    parent_idx = child_right_idx

        data_index = tree_index - self.buffer_capacity + 1  # tree_index->data_index
        return data_index, self.tree[tree_index]  # 返回采样到的data在buffer中的index,以及相对应的priority

    @property
    def priority_sum(self):
        return self.tree[0]  # 树的顶端保存了所有priority之和

    @property
    def priority_max(self):
        return self.tree[self.buffer_capacity - 1:].max()  # 树的最后一层叶节点，保存的才是每个数据对应的priority
