# Import Absolutes deps
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from rlbench.backend.observation import Observation
from typing import List
import numpy as np
from torch.utils.data.dataset import Dataset
# Import Relative deps
import quaternion
import sys
sys.path.append('..')
from models.CupboardAgents.DDPG.ddpg import DDPG,DDPGArgs
from models.Agent import TorchRLAgent
import logger

class GripperPoseDDPGNN():
    """
    GripperPoseDDPGNN : 
        - 2 Sub DDPG Agents.
            - Gripper Pose Agent
            - Final XYZ Agent
        $ CHANGES
            $ 1. 1 DDPG Agent with Normalised Quaternion : Raate
            $ 2. Reward function setup : @kadu 
                $ 2.1 Integrate Task into PutGrocceriesEnv. 
            $ 3. Mix and Experment. 
                $ 3.1 4000 Episodes. : MAX_STEPS 
    """
    def __init__(self):
        self.postion_predictor = DDPG(7,8)
        pose_args = DDPGArgs()
        pose_args.bounding_min = 0
        self.pose_predictor = DDPG(7,5,args=pose_args)
    
    def update(self):
        self.postion_predictor.update_policy()
        self.pose_predictor.update_policy()
    
    def observe(self, r_t, s_t1, done):
        self.postion_predictor.observe(r_t,s_t1,done)    
        self.pose_predictor.observe(r_t,s_t1,done)

    def random_action(self):
        postion_pred = self.postion_predictor.random_action()
        pose_pred = self.pose_predictor.random_action()
        quaternion_vals = quaternion.as_float_array(quaternion.as_quat_array(pose_pred[:-1]))
        action = np.array(list(postion_pred)+list(quaternion_vals)+[pose_pred[-1]])
        return action 

    def select_action(self,s_t):
        postion_pred = self.postion_predictor.select_action(s_t)
        pose_pred = self.pose_predictor.select_action(s_t)
        quaternion_vals = np.array(quaternion.as_float_array(quaternion.as_quat_array(pose_pred[:-1])))
        quaternion_vals = quaternion_vals / np.linalg.norm(quaternion_vals)
        action = np.array(list(postion_pred)+list(quaternion_vals)+[pose_pred[-1]])
        return action 
 
class GripperPoseRLAgent(TorchRLAgent):
    """
    GripperPoseRLAgent
    -----------------------
    Algo Of Choice : https://spinningup.openai.com/en/latest/algorithms/ddpg.html

    Why DDPG : 
    - Our choice of input and output makes it ideal as
        1. GripperPoseRLAgent's input 
            1. `object_pose`
        2. Output is : `EE_pose` : This is the final pose. which is CONTINOUS but it can even be seen as deterministic. Our output will be the same 
        
        So as its a continous actionspace one can try and use DDPG 
        https://math.stackexchange.com/questions/3179912/policy-gradient-reinforcement-learning-for-continuous-state-and-action-space
        https://ai.stackexchange.com/questions/4085/how-can-policy-gradients-be-applied-in-the-case-of-multiple-continuous-actions

    TODO : FIX/Transform DDPG Action PREDS FOR Supporting allowed Quartienion Predictions
    TODO : FIX DDPG To Support ACTION SPACE FOR XYZ Based on Robot's Workspace. : self.task._task.boundary._boundaries[0]._boundary_bbox.min_x 
    TODO 
    """
    def __init__(self,learning_rate = 0.001,batch_size=64,collect_gradients=False,warmup=10):
        super(GripperPoseRLAgent,self).__init__(collect_gradients=collect_gradients,warmup=warmup)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = DDPG(7,8) # 1 DDPG Setup with Different Predictors. 
        self.agent_name ="DDPG__AGENT"
        self.logger = logger.create_logger(self.agent_name)
        self.logger.propagate = 0
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None
        self.batch_size =batch_size
        self.desired_obj = 'soup_grasp_point'
        self.print_every = 40
   
    
    def get_information_vector(self,demonstration_episode:List[Observation]):
        return demonstration_episode[0].object_poses[self.desired_obj]

    def observe(self,state_t1:List[Observation],action_t,reward_t:int,done:bool):
        """
        State s_t1 can be None because of errors thrown by policy. 
        """
        state_t1 = None if state_t1[0] is None else self.get_information_vector(state_t1)
        self.neural_network.observe(reward_t,state_t1,done)
    
    def update(self):
        self.neural_network.update_policy()

    def reset(self,state:List[Observation]):
        self.neural_network.reset(self.get_information_vector(state))

    def act(self,state:List[Observation],timestep=0):
        """
        ACTION PREDICTION : ABS_EE_POSE_PLAN
        """
        # agent pick action ...
        if timestep <= self.warmup:
            action = self.neural_network.random_action()
        else:
            state = self.get_information_vector(state)
            action = self.neural_network.select_action(state) # 8 Dim Vector
        
        quaternion_vals = np.array(action[3:-1]) # Normalising Quaternion values. 
        quaternion_vals = quaternion_vals / np.linalg.norm(quaternion_vals)
        action = np.array(list(action[:3])+list(quaternion_vals)+[action[-1]])
        return action