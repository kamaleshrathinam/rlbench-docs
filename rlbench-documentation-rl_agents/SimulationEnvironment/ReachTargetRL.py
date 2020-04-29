from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.tasks import ReachTarget
from typing import List
from quaternion import from_rotation_matrix, quaternion
import scipy as sp
import numpy as np
from gym import spaces
from .Environment import SimulationEnvironment
from .Environment import image_types,DEFAULT_ACTION_MODE,ArmActionMode
from .RewardFunctions import RewardFunction

import sys
sys.path.append('..')
from models.Agent import LearningAgent,RLAgent
import logger
from Utilities.core import *
class ReplayBuffer():
    
    def __init__(self):
        self.observations = []
        self.rewards = []
        self.actions = []
        self.total_reward = 0

    def store(self,observation:Observation,action,reward:int):
        self.observation.append(observation)
        self.actions.append(action)
        self.observation.append(reward)

DEFAULT_ACTION_MODE = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
DEFAULT_TASK = ReachTarget
         
class ReachTargetRLEnvironment(SimulationEnvironment):
    
    def __init__(self,
                reward_function:RewardFunction, 
                action_mode=DEFAULT_ACTION_MODE,\
                task=DEFAULT_TASK,\
                headless=True,
                num_episodes=100, 
                episode_length=70, 
                dataset_root=''):

        super(ReachTargetRLEnvironment,self).__init__(action_mode=action_mode, task=ReachTarget, headless=headless,dataset_root=dataset_root)
        # training parameters
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.logger = logger.create_logger(__class__.__name__)
        self.logger.propagate = 0
        self.reward_function = reward_function

  
    #IMP
    def _get_state(self, obs:Observation,check_images=True):
        # _get_state function is present so that some alterations can be made to observations so that
        # dimensionality management is handled from lower level. 

        if not check_images: # This is set so that image loading can be avoided
            return obs

        for state_type in image_types:    # changing axis of images in `Observation`
            image = getattr(obs, state_type)
            if image is None:
                continue
            if len(image.shape) == 2:
                # Depth and Mask can be single channel.Hence we Reshape the image from (width,height) -> (width,height,1)
                image = image.reshape(*image.shape,1)
            # self.logger.info("Shape of : %s Before Move Axis %s" % (state_type,str(image.shape)))
            image=np.moveaxis(image, 2, 0)  # change (H, W, C) to (C, H, W) for torch
            # self.logger.info("After Moveaxis :: %s" % str(image.shape))
            setattr(obs,state_type,image)

        return obs
   
    
    #IMP
    def step(self, action):
        error = None
        state_obs = None
        obs_, rl_bench_reward, terminate = self.task.step(action)  # reward in original rlbench is binary for success or not
        state_obs = self._get_state(obs_)
        shaped_reward = self.reward_function(self,state_obs,action,rl_bench_reward)
        return state_obs, shaped_reward, terminate,rl_bench_reward

    # IMP
    def train_rl_agent(self,agent:RLAgent,eval_mode=False):
        replay_buffer = RLMetrics()
        total_steps = 0 # Total Steps of all Episodes.
        valid_steps = 0 # Valid steps predicted by the agent

        for episode_i in range(self.num_episodes):
            descriptions, obs = self.task.reset()
            prev_obs = obs
            agent.reset([self._get_state(obs)]) # Reset to Rescue for resetting s_t in agent on failure
            step_counter = 0 # A counter of valid_steps within episiode
            reward_buffer = replay_buffer.get_new_reward_buffer()
            while step_counter < self.episode_length:
            # for step_counter in range(self.episode_length): # Iterate for each timestep in Episode length
                total_steps+=1
                action = agent.act([prev_obs],timestep=step_counter) # Provide state s_t to agent.
                selected_action = action
                new_obs, reward, terminate,rl_bench_reward = self.step(selected_action)
                reward_buffer.add(reward)
                if rl_bench_reward == 1:
                    self.logger.info("Reached Goal!!")
                    reward_buffer.completed = True
                step_counter+=1                
                agent.observe([new_obs],action,reward,terminate) # s_t+1,a_t,reward_t : This should also be thought out.
                prev_obs = new_obs           
                if step_counter == self.episode_length-1:
                    self.logger.info("Terminating on Max Steps!!")
                    terminate = True # setting termination here becuase failed trajectory. 
                
                if valid_steps > agent.warmup and not eval_mode:
                    agent.update()
                if terminate:
                    break
            self.logger.info("Total Reward Gain For all Epsiodes : %d"%replay_buffer.total_reward)
        
        return replay_buffer