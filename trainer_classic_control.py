import numpy as np


class Trainer():
    def __init__(self, 
                 agent, 
                 env, 
                 test_env, 
                 args,
                 writer = None):
        
        self.agent=agent
        self.env = env
        self.test_env = test_env
        self.args = args
        self.writer = writer
        self.agent.writer = self.writer
        

    def train(self, total_steps):
        step = 0
        episode = 0
        obs, info = self.env.reset(seed=episode * (self.args.seed +1))
        ep_rews = []
        ep_costs = []
        ep_step = 0
        total_rew=0
        total_cost=0
        while step <= total_steps:

            if step % self.args.eval_freq == 0:
                if self.args.enable_test:
                    ep_rew, ep_cost = self.evaluate()
                    if self.writer is not None:
                        self.writer.add_scalar('test/ep_rew', ep_rew, step)
                        self.writer.add_scalar('test/ep_cost', ep_cost, step)
                if ep_rews != []:
                    if self.writer is not None:
                        self.writer.add_scalar('train/ep_rew', np.mean(ep_rews), step)
                        self.writer.add_scalar('train/ep_cost', np.mean(ep_costs), step)
                    print(f"At step {step}, mean ep reward = {np.mean(ep_rews)}, mean ep cost = {np.mean(ep_costs)}")
                    ep_rews = []
                    ep_costs = []
            if step % self.args.save_freq == 0:
                self.agent.save(self.args.tb_logs +'/model_' + str(step))

            act = self.agent.get_action(obs, False)
            next_obs, rew, done, truncated, info = self.env.step(act)
            ep_step +=1
            truncated = truncated or (ep_step==self.args.ep_len)
            cost= info['cost']
            Q_loss, Q_target, phi_loss = self.agent.step(obs,act, rew, next_obs, done, cost, done or truncated)
            obs = next_obs
            step += 1

            total_cost += cost
            total_rew += rew

            if self.writer is not None:
                self.writer.add_scalar('train/Q_loss', Q_loss, step)
                self.writer.add_scalar('train/Q_target', Q_target, step)
                self.writer.add_scalar('train/phi_loss', phi_loss, step)
            
            if done or truncated:
                ep_costs.append(total_cost)
                ep_rews.append(total_rew)
                total_cost = 0
                total_rew = 0
                episode += 1
                obs, info = self.env.reset(seed=episode * (self.args.seed +1))
                self.agent.exploration_schedule()
                ep_step = 0
        self.agent.save(self.args.tb_logs +'/model_final')

    def evaluate(self):
        ep_costs = []; ep_rews = []
        for i in range(self.args.eval_times):
            done = False; truncated=False
            total_cost=0; total_rew = 0
            obs, info = self.test_env.reset(seed=i * (self.args.seed +99999))
            while not (done or truncated):
                act = self.agent.get_action(obs, True)
                next_obs, rew, done, truncated, info = self.test_env.step(act)
                total_cost += info['cost']
                total_rew += rew
                obs = next_obs
            ep_costs.append(total_cost); ep_rews.append(total_rew)
        return np.mean(ep_rews), np.mean(ep_costs)


    
