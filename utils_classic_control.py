
import torch
from torch.distributions import Normal
import numpy as np

device = "cuda:0"


def clip_norm(u, c):
	u_norm = np.linalg.norm(u)
	if u_norm > c:
		u = u * c / (u_norm + 1e-10)
	return u


def gaussian_kl(mean1, std1, mean2, std2):
    """
    Calculate KL-divergence between two Gaussian distributions N(mu1, sigma1) and N(mu2, sigma2)
    """
    normal1 = Normal(mean1, std1)
    normal2 = Normal(mean2, std2)
    return torch.distributions.kl.kl_divergence(normal1,normal2).sum(-1, keepdim=True)


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
