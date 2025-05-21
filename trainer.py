import datetime
import json
import os
import time
import random
from pathlib import Path
import numpy as np
from tensorboardX import SummaryWriter
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, capped_cubic_video_schedule

from configuration import serialize
from agents.graphics import AgentGraphics
import pickle
GAP = 500


class Trainer(object):
    """
        The trainer of an agent interacting with an environment to maximize its expected reward.
    """

    OUTPUT_FOLDER = 'out'
    SAVED_MODELS_FOLDER = 'saved_models'
    RUN_FOLDER = 'run_{}_{}'
    METADATA_FILE = 'metadata.json'
    LOGGING_FILE = 'logging.{}.log'

    def __init__(self,
                 env,
                 agent,
                 directory=None,
                 run_directory=None,
                 num_episodes=1000,
                 use_total_steps = False,
                 total_steps = 100000,
                 training=True,
                 sim_seed=0,
                 recover=None,
                 display_env=True,
                 display_agent=False,
                 display_rewards=False,
                 close_env=True,
                 step_callback_fn=None):
        """

        :param env: The environment to be solved, possibly wrapping an AbstractEnv environment
        :param AbstractAgent agent: The agent solving the environment
        :param Path directory: Workspace directory path
        :param Path run_directory: Run directory path
        :param int num_episodes: Number of episodes run
        !param training: Whether the agent is being trained or tested
        :param sim_seed: The seed used for the environment/agent randomness source
        :param recover: Recover the agent parameters from a file.
                        - If True, it the default latest save will be used.
                        - If a string, it will be used as a path.
        :param display_env: Render the environment, and have a monitor recording its videos
        :param display_agent: Add the agent graphics to the environment viewer, if supported
        :param display_rewards: Display the performances of the agent through the episodes
        :param close_env: Should the environment be closed when the evaluation is closed
        :param step_callback_fn: A callback function called after every environment step. It takes the following
               arguments: (episode, env, agent, transition, writer).

        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.total_steps = total_steps
        self.use_total_steps = use_total_steps
        self.training = training
        self.sim_seed = sim_seed if sim_seed is not None else np.random.randint(0, 1e6)
        self.close_env = close_env
        self.display_env = display_env
        self.display_agent = display_agent and display_env
        self.step_callback_fn = step_callback_fn

        self.directory = Path(directory or self.default_directory)
        self.run_directory = self.directory / (run_directory or self.default_run_directory)
        self.wrapped_env = RecordVideo(env,
                                       self.run_directory,
                                       episode_trigger=(None if self.display_env else lambda e: False))
        try:
            self.wrapped_env.unwrapped.set_record_video_wrapper(self.wrapped_env)
        except AttributeError:
            pass
        self.wrapped_env = RecordEpisodeStatistics(self.wrapped_env)
        self.episode = 0
        self.ep_step = 0
        self.writer = SummaryWriter(str(self.run_directory))
        self.agent.set_writer(self.writer)
        #self.agent.evaluation = self
        self.write_metadata()
        self.filtered_agent_stats = 0
        self.best_agent_stats = -np.infty, 0

        self.recover = recover
        if self.recover:
            self.load_agent_model(self.recover)
        
        

        self.reward_viewer = None
        self.observation = None

    @property
    def default_directory(self):
        return Path(self.OUTPUT_FOLDER) / self.env.unwrapped.__class__.__name__ / self.agent.__class__.__name__

    @property
    def default_run_directory(self):
        return self.RUN_FOLDER.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid())
    
    def train(self):
        self.training = True
        try:
            self.agent.train()
        except AttributeError:
            print("No train() method.")
            pass
        if self.use_total_steps:
            self.run_total_steps()
        else:
            self.run_episodes()
        self.close()

    def test(self):
        """
        Test the agent.

        If applicable, the agent model should be loaded before using the recover option.
        """
        self.training = False
        # image = self.env.render()
        if self.display_env:
            self.wrapped_env.episode_trigger = lambda e: True
        try:
            self.agent.eval()
        except AttributeError:
            print("No eval() method.")
            pass
        self.run_episodes()
        # self.run_episodes_heter()
        self.close()

    def run_total_steps(self):
        crash_times = 0
        success_times = 0
        current_step = 0

        terminal = False
        rewards = []
        speeds = []
        start_time = time.time()
        self.ep_step = 0
        self.reset(seed=self.episode * (self.sim_seed + 1))
        ep_rewards= []; ep_crashes=[]
        while current_step <= self.total_steps:
            current_step += 1
            # Run episode
            self.ep_step += 1
            # Step until a terminal step is reached
            reward, terminal, info = self.step()
            rewards.append(reward)
            if isinstance(info['speed'], list):
                # fank: change for multi-agent info
                speeds.append(info['speed'][0])
                if info['crashed'][0]: crash_times += 1
            else:
                speeds.append(info['speed'])    
                if info['crashed']: crash_times += 1        
            success_times += float(info.get('success', 0))

            if current_step % GAP == 0:
                if len(ep_rewards)>0:
                    self.writer.add_scalar("eval/ep_rewards", np.mean(ep_rewards), current_step)
                    self.writer.add_scalar("eval/crash_ratio", np.mean(ep_crashes), current_step)
                    ep_rewards=[]; ep_crashes=[]
            if terminal:
                if isinstance(info['crashed'], list):
                    ep_crashes.append(float(info['crashed'][0]))
                else:
                    ep_crashes.append(float(info['crashed']))
                ep_rewards.append(sum(rewards))
                # End of episode
                duration = time.time() - start_time
                self.writer.add_scalar("agent/success", success_times, current_step)
                self.writer.add_scalar("agent/crash", crash_times, current_step)
                self.writer.add_scalar("agent/speed", np.mean(speeds), current_step)
                self.after_all_episodes(current_step, rewards, duration) # logger
                self.after_some_episodes(self.episode, rewards)
                self.episode += 1

                terminal = False
                rewards = []
                speeds = []
                start_time = time.time()
                self.ep_step = 0
                self.reset(seed=self.episode * (self.sim_seed + 1))   
    
    def run_episodes(self):
        crash_times = 0
        success_times = 0
        crash_ids = []
        # with open('crash_ids.pickle', 'rb') as f:
        #     lists = pickle.load(f)
        
        # for episode_number in range(self.num_episodes):
        #     if self.display_agent:
        #         try:
        #             # Render the agent within the environment viewer, if supported
        #             self.env.render()
        #             self.env.unwrapped.viewer.directory = self.run_directory
        #             self.env.unwrapped.viewer.set_agent_display(
        #                 lambda agent_surface, sim_surface: AgentGraphics.display(self.agent, agent_surface, sim_surface))
        #             self.env.unwrapped.viewer.directory = self.run_directory
        #         except AttributeError:
        #             print("The environment viewer doesn't support agent rendering.") 
        #     # Run episode
        #     terminal = False
        #     self.reset(seed=episode_number * (self.sim_seed + 1))
        #     rewards = []
        #     speeds = []
        #     start_time = time.time()
        #     self.ep_step = 0
        #     while not terminal:
        #         self.ep_step += 1
        #         # Step until a terminal step is reached
        #         reward, terminal, info = self.step()
        #         rewards.append(reward)
        #         if isinstance(info['speed'], list):
        #             speeds.append(info['speed'][0])
        #             if info['crashed'][0]: 
        #                 crash_times += 1 
        #         else:
        #             speeds.append(info['speed'])  
        #             if info['crashed']: 
        #                 crash_times += 1
        #         # why there is success???      
        #         success_times += float(info.get('success', 0))

        #         # Catch interruptions
        #         try:
        #             if self.env.unwrapped.done:
        #                 break
        #         except AttributeError:
        #             pass
        #     # End of episode
        #     duration = time.time() - start_time
        #     self.writer.add_scalar("agent/success", success_times, self.episode)
        #     self.writer.add_scalar("agent/crash", crash_times, self.episode)
        #     self.writer.add_scalar("agent/speed", np.mean(speeds), self.episode)
        #     self.after_all_episodes(self.episode, rewards, duration)
        #     self.after_some_episodes(self.episode, rewards)
            
                
        # this is the original version of test script   
        for self.episode in range(self.num_episodes):
            episode_crash_times = 0
            if capped_cubic_video_schedule(self.episode) and self.display_agent:
                try:
                    # Render the agent within the environment viewer, if supported
                    self.env.render()
                    self.env.unwrapped.viewer.directory = self.run_directory
                    self.env.unwrapped.viewer.set_agent_display(
                        lambda agent_surface, sim_surface: AgentGraphics.display(self.agent, agent_surface, sim_surface))
                    self.env.unwrapped.viewer.directory = self.run_directory
                except AttributeError:
                    print("The environment viewer doesn't support agent rendering.")
            # Run episode
            terminal = False
            self.reset(seed=self.episode * (self.sim_seed + 1))
            rewards = []
            speeds = []
            start_time = time.time()
            self.ep_step = 0
            while not terminal:
                self.ep_step += 1
                # Step until a terminal step is reached
                reward, terminal, info = self.step()
                rewards.append(reward)
                if isinstance(info['speed'], list):
                    speeds.append(info['speed'][0])
                    if info['crashed'][0]: 
                        crash_times += 1 
                        episode_crash_times += 1
                else:
                    speeds.append(info['speed'])  
                    if info['crashed']: 
                        crash_times += 1  
                        episode_crash_times += 1
                # why there is success???      
                success_times += float(info.get('success', 0))

                # Catch interruptions
                try:
                    if self.env.unwrapped.done:
                        break
                except AttributeError:
                    pass
            if episode_crash_times > 0:
                crash_ids.append(self.episode)
            # End of episode
            duration = time.time() - start_time
            self.writer.add_scalar("agent/success", success_times, self.episode)
            self.writer.add_scalar("agent/crash", crash_times, self.episode)
            self.writer.add_scalar("agent/speed", np.mean(speeds), self.episode)
            self.after_all_episodes(self.episode, rewards, duration)
            self.after_some_episodes(self.episode, rewards)
        # with open("crash_ids.pickle", "wb") as f:
        #     pickle.dump(crash_ids, f)
    
    def step(self):
        """
            Plan a sequence of actions according to the agent policy, and step the environment accordingly.
        """
        # Query agent for actions sequence
        actions = self.agent.plan(self.observation)
        if not actions:
            raise Exception("The agent did not plan any action")

        # Forward the actions to the environment viewer
        try:
            self.env.unwrapped.viewer.set_agent_action_sequence(actions)
        except AttributeError:
            pass

        # Step the environment
        previous_observation, action = self.observation, actions[0]
        transition = self.wrapped_env.step(action)
        self.observation, reward, done, truncated, info = transition
        feasible, infeasible = False, False
        if isinstance(info.get('cost', 0), list):
            if info.get('cost', 0)[0] > 0: infeasible = True # fank: change for multi-agent info
        else:
            if info.get('cost', 0) > 0: infeasible = True # jinpeng
        if self.ep_step == 1: feasible = True 
        info['feasible'] = feasible
        info['infeasible'] = infeasible

        #print(info)
        terminal = done or truncated

        # Call callback
        if self.step_callback_fn is not None:
            self.step_callback_fn(self.episode, self.wrapped_env, self.agent, transition, self.writer)

        # Record the experience.
        try:
            self.agent.record(previous_observation, action, reward, self.observation, done, terminal, info)
        except NotImplementedError:
            pass

        return reward, terminal, info
    
    def after_all_episodes(self, episode, rewards, duration):
        rewards = np.array(rewards)
        gamma = self.agent.config.get("gamma", 1)
        self.writer.add_scalar('episode/length', len(rewards), episode)
        self.writer.add_scalar('episode/total_reward', sum(rewards), episode)
        #self.writer.add_scalar('episode/return', sum(r*gamma**t for t, r in enumerate(rewards)), episode)
        #self.writer.add_scalar('episode/fps', len(rewards) / max(duration, 1e-6), episode)
        #self.writer.add_histogram('episode/rewards', rewards, episode)
        print("Episode {} score: {:.1f}".format(episode, sum(rewards)))
        if self.training:
            try:
                self.agent.schedule_explore()
            except AttributeError:
                pass

    def after_some_episodes(self, episode, rewards,
                            best_increase=1.1,
                            episodes_window=50):
        if capped_cubic_video_schedule(episode):
            # Save the model
            if self.training:
                self.save_agent_model(episode)

        if self.training:
            # Save best model so far, averaged on a window
            best_reward, best_episode = self.best_agent_stats
            self.filtered_agent_stats += 1 / episodes_window * (np.sum(rewards) - self.filtered_agent_stats)
            if self.filtered_agent_stats > best_increase * best_reward \
                    and episode >= best_episode + episodes_window:
                self.best_agent_stats = (self.filtered_agent_stats, episode)
                self.save_agent_model("best")

    def save_agent_model(self, identifier, do_save=True):
        # Create the folder if it doesn't exist
        permanent_folder = self.directory / self.SAVED_MODELS_FOLDER
        os.makedirs(permanent_folder, exist_ok=True)

        episode_path = None
        if do_save:
            episode_path = Path(self.run_directory) / "checkpoint-{}.tar".format(identifier)
            try:
                self.agent.save(filename=permanent_folder / "latest.tar")
                episode_path = self.agent.save(filename=episode_path)
                if episode_path:
                    print("Saved {} model to {}".format(self.agent.__class__.__name__, episode_path))
            except NotImplementedError:
                pass
        return episode_path

    def load_agent_model(self, model_path):
        if model_path is True:
            model_path = self.directory / self.SAVED_MODELS_FOLDER / "latest.tar"
        if isinstance(model_path, str):
            model_path = Path(model_path)
            if not model_path.exists():
                model_path = self.directory / self.SAVED_MODELS_FOLDER / model_path
        try:
            model_path = self.agent.load(filename=model_path)
            if model_path:
                print("Loaded {} model from {}".format(self.agent.__class__.__name__, model_path))
        except FileNotFoundError:
            print("No pre-trained model found at the desired location.")
        except NotImplementedError:
            pass

    def write_metadata(self):
        metadata = dict(env=serialize(self.env), agent=serialize(self.agent), sim_seed = self.sim_seed)
        #file_infix = '{}.{}'.format(id(self.wrapped_env), os.getpid())
        file = self.run_directory / self.METADATA_FILE
        with file.open('w') as f:
            json.dump(metadata, f, sort_keys=True, indent=4)

    def reset(self, seed=0):
        seed = self.sim_seed + seed if self.sim_seed is not None else None
        np.random.seed(seed)    
        random.seed(seed)   
        self.observation, info = self.wrapped_env.reset(seed=seed)
        self.agent.seed(seed)  # Seed the agent with the main environment seed
        self.agent.reset()

    def close(self):
        """
            Close the evaluation.
        """
        if self.training:
            self.save_agent_model("final")
        self.wrapped_env.close()
        self.writer.close()
        if self.close_env:
            self.env.close()