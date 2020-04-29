# Import Absolutes deps
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.tasks import *
from typing import List
import numpy as np
from gym import spaces
# Import Relative deps
import sys
sys.path.append('..')
from models.Agent import LearningAgent
import logger
# list of state types
state_types = [ 'left_shoulder_rgb',
                'left_shoulder_depth',
                'left_shoulder_mask',
                'right_shoulder_rgb',
                'right_shoulder_depth',
                'right_shoulder_mask',
                'wrist_rgb',
                'wrist_depth',
                'wrist_mask',
                'joint_velocities',
                'joint_velocities_noise',
                'joint_positions',
                'joint_positions_noise',
                'joint_forces',
                'joint_forces_noise',
                'gripper_pose',
                'gripper_touch_forces',
                'task_low_dim_state']

# The CameraConfig controls if mask values will be RGB or 1 channel 
# https://github.com/stepjam/RLBench/blob/master/rlbench/observation_config.py#L5
# The depth Values are also single channel. Where depth will be of dims (width,height)
# https://github.com/stepjam/PyRep/blob/4a61f6756c3827db66423409632358de312b97e4/pyrep/objects/vision_sensor.py#L128
image_types=[ 
    'left_shoulder_rgb', # (width,height,channel)
    'left_shoulder_depth', # Depth is in Black and White : (width,height)
    'left_shoulder_mask', # Mask can be single channel
    'right_shoulder_rgb',
    'right_shoulder_depth', # Depth is in Black and White
    'right_shoulder_mask', # Mask can be single channel
    'wrist_rgb',
    'wrist_depth', # Depth is in Black and White
    'wrist_mask' # Mask can be single channel
]



DEFAULT_ACTION_MODE = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
DEFAULT_TASK = ReachTarget
class SimulationEnvironment():
    """
    This can be a parent class from which we can have multiple child classes that 
    can diversify for different tasks and deeper functions within the tasks.
    """
    def __init__(self,\
                action_mode=DEFAULT_ACTION_MODE,\
                task=DEFAULT_TASK,\
                dataset_root='',\
                headless=True):
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        action_mode = action_mode
        self.env = Environment(
            action_mode,dataset_root,obs_config=obs_config, headless=headless)
        # Dont need to call launch as task.get_task can launch env. 
        self.task = self.env.get_task(task)
        _, obs = self.task.reset()
        self.action_space =  spaces.Box(low=-1.0, high=1.0, shape=(self.env.action_size,), dtype=np.float32)
        self.logger = logger.create_logger(__class__.__name__)
        self.logger.propagate = 0

    def _get_state(self, obs:Observation):
        """
        This will be a hook over the environment to get the state. 
        This will ensure that if changes need to be made to attributes of `Observation` then they can be made
        in a modular way. 
        """
        raise NotImplementedError()

    
    def reset(self):
        descriptions, obs = self.task.reset()
        return self._get_state(obs)

    def step(self, action):
        obs_, reward, terminate = self.task.step(action)  # reward in original rlbench is binary for success or not
        return self._get_state(obs_), reward, terminate

    def shutdown(self):
        self.logger.info("Environment Shutdown! Create New Instance If u want to start again")
        self.logger.handlers.pop()
        self.env.shutdown()
    
    def get_demos(self,num_demos):
       """
       Should be implemented by child classes because other extra properties needed
       before training can be accessed via that. 
       """
       raise NotImplementedError()


class ReachTargetSimulationEnv(SimulationEnvironment):
    """
    Inherits the `SimulationEnvironment` class. 
    This environment is specially ment for running traing agent for ReachTarget Task. 
    This can be inherited for different ways of doing learning. 
    
    :param num_episodes : Get the total Epochs needed for the Simulation
    """
    def __init__(self, action_mode=DEFAULT_ACTION_MODE, headless=True,num_episodes = 120,task=ReachTarget,episode_length = 40,dataset_root=''):
        super(ReachTargetSimulationEnv,self).__init__(action_mode=action_mode, task=task, headless=headless,dataset_root=dataset_root)
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.logger = logger.create_logger(__class__.__name__)
        self.logger.propagate = 0

    def get_goal_poistion(self):
        """
        This will return the postion of the target for the ReachTarget Task. 
        """
        return np.array(self.task.target.get_position())


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

    def run_trained_agent(self,agent:LearningAgent):
        simulation_analytics = {
            'total_epochs_allowed':self.num_episodes,
            'max_steps_per_episode': self.episode_length,
            'convergence_metrics':[],
            'distance_metrics':[]
        }
        rest_step_counter = 0
        total_steps = 0
        for i in range(self.num_episodes):
            rest_step_counter=0
            self.logger.info('Reset Episode %d'% i)
            obs,descriptions = self.task_reset()
            self.logger.info(descriptions)

            distance_per_step = []
            for _ in  range(self.episode_length): # Iterate for each timestep in Episode length
                action = agent.predict_action([obs])
                distance = agent.get_distance([obs])
                distance_per_step.append(distance)
                selected_action = action
                # print(selected_action,"selected_action")
                obs, reward, terminate = self.step(selected_action)


                rest_step_counter+=1
                if reward == 1:
                    self.logger.info("Reward Of 1 Achieved. Task Completed By Agent In steps : %d"%rest_step_counter)
                    simulation_analytics['convergence_metrics'].append({
                        'steps_to_convergence': rest_step_counter,
                        'epoch_num':i
                    })

                if terminate:
                    break

            min_dis = min(distance_per_step)
            simulation_analytics['distance_metrics'].append({
                'distance': min_dis,
                'episode_num': i
            })
                
        self.shutdown()
        return simulation_analytics

    def task_reset(self):
        descriptions, obs = self.task.reset()
        obs = self._get_state(obs)
        return obs,descriptions


    def get_demos(self,num_demos,live_demos=True,image_paths_output=False):
        """
        :param: live_demos : If live_demos=True,
        :param: image_paths_output : Useful set to True when used with live_demos=False. If set True then the dataset loaded from FS will not load the images but will load the paths to the images. 
        """
        self.logger.info("Creating Demos")
        demos = self.task.get_demos(num_demos, live_demos=live_demos,image_paths=image_paths_output)  # -> List[List[Observation]]
        self.logger.info("Created Demos")
        demos = np.array(demos).flatten()
        self.shutdown()
        new_demos = []
        for episode in demos:
            new_episode = []
            for step in episode:
                # Only transform images to in `Observation` object if its a live_demo or when image_out_path=False with live_demo=False
                new_episode.append(self._get_state(step,check_images=not live_demos and not image_paths_output)) 
            new_demos.append(new_episode)
        return new_demos