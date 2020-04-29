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
# Import Relative deps
import sys
sys.path.append('..')
from models.Agent import TorchAgent
import logger
from scipy.spatial import distance


class FullyConnectedPolicyEstimator(nn.Module):
    
    def __init__(self,num_states,num_actions):
        super(FullyConnectedPolicyEstimator, self).__init__()
        self.fc1 = nn.Linear(num_states, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_actions)

    # x is input to the network.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class ImmitationLearningAgent(TorchAgent):
    """
    ImmitationLearningAgent
    -----------------------

    Dumb Learning Agent that will try to eastimate an action given a state. 
    This will Not be considering Past actions/states while making these predictions. 
    
    todo : Make LSTM Based Networks that can remember over a batch of given observations. 
    https://stackoverflow.com/a/27516930 : For LSTM Array Stacking
    """
    def __init__(self,learning_rate = 0.01,batch_size=64,collect_gradients=False):
        super(TorchAgent,self).__init__(collect_gradients=collect_gradients)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = FullyConnectedPolicyEstimator(7,8)
        self.optimizer = optim.SGD(self.neural_network.parameters(), lr=learning_rate, momentum=0.9)
        self.loss_function = nn.SmoothL1Loss()
        self.training_data = None
        self.logger = logger.create_logger(__name__)
        self.logger.propagate = 0
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None
        self.batch_size =batch_size

    def injest_demonstrations(self,demos:List[List[Observation]],**kwargs):
        # For this Agent, Put all experiences in one huge dump from where you sample state->action 
        # https://stats.stackexchange.com/questions/187591/when-the-data-set-size-is-not-a-multiple-of-the-mini-batch-size-should-the-last
        # $ CREATE Matrix of shape (total_step_from_all_demos,shape_of_observation)
        # $ This is done because we are training a dumb agent to estimate a policy based on just dumb current state
        # $ So for training we will use a 2D Matrix. If we were doing LSTM based training then the data modeling will change. 
        joint_position_train_vector = torch.from_numpy(self.get_train_vectors(demos))
        self.total_train_size = len(joint_position_train_vector)
        # $ First Extract the output_action. Meaning the action that will control the kinematics of the robot. 
        ground_truth_velocities = np.array([getattr(observation,'joint_velocities') for episode in demos for observation in episode]) #
        # $ Create a matrix for gripper position vectors.                                                                                                                                                     
        ground_truth_gripper_positions = np.array([getattr(observation,'gripper_open') for episode in demos for observation in episode])
        # $ Final Ground truth Tensor will be [joint_velocities_0,...joint_velocities_6,gripper_open]
        ground_truth_gripper_positions = ground_truth_gripper_positions.reshape(len(ground_truth_gripper_positions),1)
        ground_truth = torch.from_numpy(np.concatenate((ground_truth_velocities,ground_truth_gripper_positions),axis=1))
        
        # demos[0][0].task_low_dim_state contains all target's coordinates
        self.logger.info("Creating Tensordata for Pytorch of Size : %s"%str(joint_position_train_vector.size()))
        self.dataset = torch.utils.data.TensorDataset(joint_position_train_vector, ground_truth)
        
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_vectors(self,demos:List[List[Observation]]):
        return np.array([getattr(observation,'joint_positions') for episode in demos for observation in episode])

    def get_distance_vectors(self,demos:List[List[Observation]]):
        endeff_pos_arr = np.array(
            [getattr(observation, 'gripper_pose') for episode in demos for observation in episode])
        target_pos_arr = np.array(
            [getattr(observation, 'task_low_dim_state') for episode in demos for observation in episode])
        return target_pos_arr,endeff_pos_arr

    def train_agent(self,epochs:int):
        if not self.dataset:
            raise Exception("No Training Data Set to Train Agent. Please set Training Data using ImmitationLearningAgent.injest_demonstrations")
        
        self.logger.info("Starting Training of Agent ")
        self.neural_network.train()
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                network_pred = self.neural_network(data.float()) 
                loss = self.loss_function(network_pred,target.float())
                loss.backward()
                if self.collect_gradients:
                    self.set_gradients(self.neural_network.named_parameters())
                self.optimizer.step()
                running_loss += loss.item()
                steps+=1

            self.logger.info('[%d] loss: %.6f' % (epoch + 1, running_loss / (steps+1)))

    def predict_action(self, demonstration_episode:List[Observation],**kwargs) -> np.array:
        self.neural_network.eval()
        train_vectors = self.get_train_vectors([demonstration_episode])
        input_val = Variable(torch.from_numpy(train_vectors[0]))
        output = self.neural_network(input_val.float())
        return output.data.cpu().numpy()
        # return np.random.uniform(size=(len(batch), 7))

    def get_distance(self, demonstration_episode:List[Observation],**kwargs) -> np.array:
        target_pos_arr, endeff_pos_arr = self.get_distance_vectors([demonstration_episode])
        endeff_pos_arr=endeff_pos_arr[:,:3]
        dis=distance.euclidean(target_pos_arr,endeff_pos_arr)

        return dis