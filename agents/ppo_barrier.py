import torch
import torch.nn as nn
from gymnasium import spaces
import random
import numpy as np
import math
import copy
from torch.distributions import Categorical

from torch.nn.functional import softplus
from torch.optim import Adam

from models.factory import loss_function_factory, optimizer_factory, \
      model_factory, size_model_config, trainable_parameters
from utils import DEVICE
from agents.base import AbstractStochasticAgent

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.costs = []
        self.next_states = []
        self.is_terminals = []
        self.is_ends = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.costs[:]
        del self.is_terminals[:]
        del self.is_ends[:]
    
class ActorCriticBarrier(nn.Module):
    def __init__(self, config):
        super(ActorCriticBarrier, self).__init__()

        def straight_through(x):
            return x
        if config["common"] is None:
            self.common = straight_through
        else:
            self.common = model_factory(config["common"])
        self.actor = model_factory(config["actor"])
        self.critic = model_factory(config["critic"])
        
        self.barrier = model_factory(config["barrier"])
        self.actor_final=nn.Softmax(dim=-1)
    
    def actor_forward(self, state):
        x = self.common(state)
        logit = self.actor(x)
        return self.actor_final(logit)
        #return logit

    def critic_forward(self, state):
        x = self.common(state)
        v = self.critic(x)
        
        return v
    
    def barrier_forward(self, state):
        x = self.common(state)
        barrier = self.barrier(x)
        return barrier


class PPOBarrierAgent(AbstractStochasticAgent):
    def __init__(self, env, config=None):
        super(PPOBarrierAgent, self).__init__(config)
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete), \
            "Only compatible with Discrete action spaces."
        self.training = True
        self.previous_state = None

        if self.config["common"] is not None:
            size_model_config(self.env, self.config["common"])
        size_model_config(self.env, self.config["actor"])
        size_model_config(self.env, self.config["critic"])
        size_model_config(self.env, self.config["barrier"]) # TODO: barrier
        self.device = DEVICE

        self.actor_critic = ActorCriticBarrier(self.config).to(self.device)
        print("Number of trainable parameters: {}".format(trainable_parameters(self.actor_critic)))
        self.buffer = RolloutBuffer()
        
        self.loss_function = loss_function_factory(self.config["loss_function"])
        self.optimizer = optimizer_factory(self.config["optimizer"]["type"],
                                           self.actor_critic.parameters(),
                                           **self.config["optimizer"])
        
        
        self.penalty_param = torch.tensor(self.config['penalty_init']).float().to(self.device)
        self.penalty_param.requires_grad=True
        #penalty = softplus(self.penalty_param)
        penalty_lr = self.config['penalty_lr']
        self.penalty_optimizer = Adam([self.penalty_param], lr=penalty_lr)
        
        self.timestep = 0
        self.optim_batch_size = self.config["batch_size"]
        self.entropy_coef=self.config["entropy_coef"]
        self.action_size = env.action_space.n
        
    @classmethod
    def default_config(cls):
        return dict(common=dict(type ="MultiLayerPerceptron"),
                    actor = dict(type ="MultiLayerPerceptron"),
                    critic = dict(type = "MultiLayerPerceptron"),
                    barrier = dict(type = "MultiLayerPerceptron"), 
                    optimizer=dict(type="ADAM",
                                   lr=2e-3,
                                   weight_decay=0,
                                   k=5),
                    loss_function="l2",
                    batch_size=64,
                    gamma=0.99,
                    lambd=0.97,
                    clip_rate=0.2,
                    K_epochs=10,
                    T_horizon = 128,
                    entropy_coef = 0.01,
                    adv_normalization = True,
                    entropy_coef_decay = 0.99,
                    penalty_lr = 5e-5,
                    penalty_init = 0.25)
    
    def act(self, state):
        self.previous_state = state
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if self.training:
            with torch.no_grad():
                pi = self.actor_critic.actor_forward(state)
                pi = pi.squeeze(0)
                distr = Categorical(probs= pi) # assume pi is positive
                action = distr.sample()
                action_logprob = distr.log_prob(action)
            self.buffer.states.append(state)
            self.buffer.actions.append(torch.as_tensor(np.array([action.cpu().numpy()]), dtype=torch.int64).to(self.device))
            self.buffer.logprobs.append(torch.as_tensor(np.array([action_logprob.cpu().numpy()]), dtype=torch.float32).to(self.device))
        else:
            with torch.no_grad():
                pi = self.actor_critic.actor_forward(state)
                pi = pi.squeeze(0)
                action = torch.argmax(pi)

        return action.item()

    def action_distribution(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pi = self.actor_critic.actor_forward(state)
            pi = pi.squeeze(0)
        values = pi.cpu().numpy()
        distribution = {action: values[action] for action in range(self.action_size)}
        return values, distribution
    
    def record(self, state, action, reward, next_state, done, end, info):
        if not self.training:
            return
        # Single-agent setting
        self.buffer.rewards.append(torch.as_tensor(np.array([reward]), dtype=torch.float32).to(self.device))
        self.buffer.costs.append(torch.as_tensor(np.array([info['cost']]), dtype=torch.float32).to(self.device)) # cost
        self.buffer.next_states.append(torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device))
        self.buffer.is_terminals.append(torch.as_tensor(np.array([done]), dtype=torch.float32).to(self.device))
        self.buffer.is_ends.append(torch.as_tensor(np.array([end]), dtype=torch.float32).to(self.device))
        self.timestep += 1

        if self.timestep % self.config["T_horizon"] == 0:
            loss, entropy, certificate_loss = self.update()
            self.writer.add_scalar("agent/loss", loss, self.timestep)
            self.writer.add_scalar("agent/entropy", entropy, self.timestep)
            self.writer.add_scalar("agent/certificate_loss", certificate_loss, self.timestep)
            self.writer.add_scalar("agent/penalty_param", self.penalty_param.detach().item(), self.timestep)
    
    

    def compute_invariance_loss(
        self,
        certificate: torch.Tensor,             # shape: [T] or [T, 1]
        episode_mask: torch.Tensor,            # shape: [T], 0/1 mask
        epsilon: float,
        barrier_lambda: float,
        gamma: float,
        num_barrier_step: int
    ) -> torch.Tensor:
        # 确保 certificate 是 [T]
        if certificate.ndim == 2 and certificate.shape[1] == 1:
            certificate = certificate.squeeze(-1)

        # 初始 mask：去掉最后一个时间步（对齐 certificate[:-1]）
        mask = episode_mask[:-1].bool()

        invariance_loss = 0.0
        T = certificate.shape[0]

        for i in range(num_barrier_step):
            k = i + 1
            decay = (1 - barrier_lambda) ** k
            eps_k = epsilon * (1 - decay) / barrier_lambda

            # B_t and B_{t+k}
            cert_t  = certificate[:-k]
            cert_tk = certificate[k:]

            # loss: max(ε_k + B_{t+k} - decay * B_t, 0)
            inv_loss = eps_k + cert_tk - decay * cert_t
            inv_loss = torch.clamp(inv_loss, min=0.0)

            # mask: 对当前的 loss 做遮罩
            valid_mask = ~mask[:T - k]
            inv_loss = inv_loss * valid_mask.float()

            # 平均这个 step 的 loss，加上折扣
            step_loss = inv_loss.mean()
            invariance_loss += (gamma ** i) * step_loss

            # 更新 mask: mask = mask[:-1] | mask[1:]
            if mask.shape[0] > 1:
                mask = mask[:-1] | mask[1:]

        # 归一化
        norm = (1 - gamma) / (1 - gamma ** num_barrier_step)
        invariance_loss = invariance_loss * norm

        return invariance_loss
    
    def update(self):
        # convert list to tensor
        # this is a batch of trajectory
        s = torch.cat(self.buffer.states, dim=0).detach()
        a = torch.cat(self.buffer.actions, dim=0).detach().unsqueeze(-1)
        old_logprob_a = torch.cat(self.buffer.logprobs, dim=0).detach().unsqueeze(-1)
        r = torch.cat(self.buffer.rewards, dim=0).detach().unsqueeze(-1)
        c = torch.cat(self.buffer.costs, dim=0).detach().unsqueeze(-1) # cost
        feasible = (c != 1) # (batch_size, 1)
        infeasible = ~feasible
        s_prime = torch.cat(self.buffer.next_states, dim=0).detach()
        done_mask = torch.cat(self.buffer.is_terminals, dim=0).detach().unsqueeze(-1)
        self.buffer.is_ends[-1]= torch.as_tensor(np.array([1.]), dtype=torch.float32).to(self.device)
        end_mask = torch.cat(self.buffer.is_ends, dim=0).detach().unsqueeze(-1)

        #print(s.shape, a.shape, old_logprob_a.shape, r.shape, s_prime.shape, done_mask.shape)

        ## TODO:Update penalty parameter
        with torch.no_grad():
            certificate = self.actor_critic.barrier_forward(s) # (batch_size, 1)
        penalty_margin = torch.zeros_like(certificate)
        for i in range(self.config["num_barrier_step"]):
            k = i + 1
            decay = (1 - self.config["barrier_lambda"]) ** k
            epsilon_k = self.config["epsilon"] * (1 - decay) / self.config["barrier_lambda"]

            # 对齐当前 certificate[t] 和 future certificate[t+k]
            cert_t     = certificate[:-k]            # B_t
            cert_tk    = certificate[k:]             # B_{t+k}

            # 计算 pm（正项惩罚）
            pm = epsilon_k + cert_tk - decay * cert_t
            pm = torch.clamp(pm, min=self.config["penalty_margin_clip"])

            # 累加到 penalty_margin 的前 T-k 项
            penalty_margin[:-k] += (self.config["gamma"] ** i) * pm

        # 时间折扣归一化
        norm = (1 - self.config["gamma"]) / (1 - self.config["gamma"] ** self.config["num_barrier_step"])
        penalty_margin = penalty_margin * norm
        if self.config["penalty_margin_normalization"]:
            penalty_margin = (penalty_margin - penalty_margin.mean()) / ((penalty_margin.std() + 1e-10))
        with torch.no_grad():
            logit = self.actor_critic.actor_forward(s)
        distr = Categorical(probs = logit) # logit are positive
        prob_a = distr.probs.gather(1, a)
        #prob_a = prob.gather(1, per_a[index])
        ratio = torch.exp(torch.log(prob_a) - old_logprob_a)  # a/b == exp(log(a)-log(b))  
        loss_penalty = (-self.penalty_param*ratio*penalty_margin).mean()
        self.penalty_optimizer.zero_grad()
        loss_penalty.backward()
        self.penalty_optimizer.step()

        p = softplus(self.penalty_param) #TODO
        penalty_item = p.item()

        """PPO update"""
        batch_size = s.shape[0]
        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(batch_size / self.optim_batch_size))

        for _ in range(self.config["K_epochs"]):
            
            '''Calculate invariance loss outside the loop, with grad tracking'''
            for i in range(optim_iter_num):
                certificate_full = self.actor_critic.barrier_forward(s)
                invariance_loss = self.compute_invariance_loss(
                    certificate=certificate_full.squeeze(-1),
                    episode_mask=done_mask.squeeze(-1),
                    epsilon=self.config["epsilon"],
                    barrier_lambda=self.config["barrier_lambda"],
                    gamma=self.config["gamma"],
                    num_barrier_step=self.config["num_barrier_step"]
                )
                '''Calculate Certificated Loss'''
                feasible_loss = feasible * torch.clamp(self.config["epsilon"] + certificate_full, min=0.0)
                feasible_loss = torch.sum(feasible_loss) / torch.clamp(torch.sum(feasible), min=1.0)
                infeasible_loss = infeasible * torch.clamp(self.config["epsilon"] - certificate_full, min=0.0)
                infeasible_loss = torch.sum(infeasible_loss) / torch.clamp(torch.sum(infeasible), min=1.0)
                certificate_loss = feasible_loss + infeasible_loss + invariance_loss
                self.optimizer.zero_grad()
                certificate_loss.backward()
                self.optimizer.step()
                
            ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
            with torch.no_grad():
                vs = self.actor_critic.critic_forward(s)
                vs_ = self.actor_critic.critic_forward(s_prime)
                
            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.config["gamma"] * vs_ * (1 - done_mask) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.config["gamma"] * self.config["lambd"] * adv[-1] * (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
            td_target = adv + vs # (total_batch_size, 1)
            if self.config["adv_normalization"]:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-10))  #useful in some envs
            
            '''Calculate Certificate using Barrier'''
            with torch.no_grad():
                certificate = self.actor_critic.barrier_forward(s) # (batch_size, 1)
            '''Calculate the similar GAE for Barrier (without tracking gradient)'''
            penalty_margin = torch.zeros_like(certificate)
            for i in range(self.config["num_barrier_step"]):
                k = i + 1
                decay = (1 - self.config["barrier_lambda"]) ** k
                epsilon_k = self.config["epsilon"] * (1 - decay) / self.config["barrier_lambda"]

                # 对齐当前 certificate[t] 和 future certificate[t+k]
                cert_t = certificate[:-k]            # B_t
                cert_tk = certificate[k:]             # B_{t+k}

                # 计算 pm（正项惩罚）
                pm = epsilon_k + cert_tk - decay * cert_t
                pm = torch.clamp(pm, min=self.config["penalty_margin_clip"])

                # 累加到 penalty_margin 的前 T-k 项
                penalty_margin[:-k] += (self.config["gamma"] ** i) * pm

            # 时间折扣归一化
            norm = (1 - self.config["gamma"]) / (1 - self.config["gamma"] ** self.config["num_barrier_step"])
            penalty_margin = penalty_margin * norm
            if self.config["penalty_margin_normalization"]:
                penalty_margin = (penalty_margin - penalty_margin.mean()) / ((penalty_margin.std() + 1e-10)) 
            
            #Shuffle the trajectory, Good for training
            perm = np.arange(batch_size)
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)
            per_s, per_a, per_td_target, per_adv, per_old_logprob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_logprob_a[perm].clone()
            per_penalty_margin = penalty_margin[perm].clone()
            
            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, per_s.shape[0]))

                '''actor update'''
                #prob = self.actor_critic.actor_forward(per_s[index])
                logit = self.actor_critic.actor_forward(per_s[index])
                distr = Categorical(probs = logit) # logit are positive
                entropy = distr.entropy().sum(0, keepdim=True)
                prob_a = distr.probs.gather(1, per_a[index])
                #prob_a = prob.gather(1, per_a[index])
                ratio = torch.exp(torch.log(prob_a) - per_old_logprob_a[index])  # a/b == exp(log(a)-log(b))

                surr1 = ratio * per_adv[index]
                surr2 = torch.clamp(ratio, 1 - self.config["clip_rate"], 1 + self.config["clip_rate"]) * per_adv[index]

                loss_rpi = torch.min(surr1, surr2) # (batch_size, 1), batch_size here is 32
                
                # TODO: here we needed to postprocess the cost signal
                loss_surrogate_cpi = ratio * per_penalty_margin[index]
                pi_objective = (loss_rpi - loss_surrogate_cpi) / (1+penalty_item)
                
                per_critic = self.actor_critic.critic_forward(per_s[index])
            
                # final loss of clipped objective PPO
                loss = - pi_objective \
                      + 0.25* self.loss_function(per_critic, per_td_target[index]) \
                         - self.entropy_coef * entropy
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # clear buffer
        self.buffer.clear()
        return loss.mean().detach().cpu(), entropy.detach().cpu(), certificate_loss.detach().cpu()

    def save(self, filename):
        state = {'state_dict': self.actor_critic.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'penalty_param': self.penalty_param}
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.penalty_param = checkpoint['penalty_param']
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

    def schedule_explore(self):
        self.entropy_coef *= self.config["entropy_coef_decay"] #exploring decay