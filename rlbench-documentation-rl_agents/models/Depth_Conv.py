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
import sys

sys.path.append('..')
from models.Agent import TorchAgent
import logger
from scipy.spatial import distance
import torch

device = torch.device("cuda:0")

class ConvolutionalPolicyEstimator(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 64, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        #nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        #nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 5)
        #nn.BatchNorm2d(256)
        x = torch.randn(128,128).view(-1,1,128,128)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 200)
        self.fc3 = nn.Linear(200, 8)# 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) # bc this is our output layer. No activation here.
        x=self.fc3(x)
        return x


class ModularPolicyImagesDataset(Dataset):
    def __init__(self, joint_pos, target_pos, wrist_depth, actions):
        self.joint_pos = joint_pos
        self.target_pos = target_pos
        self.actions = actions

        self.wrist_depth_vector = wrist_depth

    def __getitem__(self, index):
        return (self.joint_pos[index], self.target_pos[index],self.wrist_depth_vector[index], self.actions[index])

    def __len__(self):
        return len(self.joint_pos)  # of how many examples(images?) you have


class Imitation_Depth(TorchAgent):
  
    def __init__(self, learning_rate=0.01, batch_size=64, collect_gradients=False):
        super(TorchAgent, self).__init__(collect_gradients=collect_gradients)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = ConvolutionalPolicyEstimator().to(device)
        self.optimizer = optim.SGD(self.neural_network.parameters(), lr=learning_rate, momentum=0.9)
        self.loss_function = nn.MSELoss()
        self.training_data = None
        self.logger = logger.create_logger(__class__.__name__)
        self.logger.propagate = 0
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None
        self.batch_size = batch_size

    def injest_demonstrations(self, demos: List[List[Observation]], **kwargs):

        joint_pos_arr, target_pos_arr,  wrist_depth = self.get_train_vectors(demos)
        joint_position_train_vector = torch.from_numpy(joint_pos_arr)
        target_position_train_vector = torch.from_numpy(target_pos_arr)

        wrist_depth = torch.from_numpy(wrist_depth)
        print("Wrist Depth Shape", wrist_depth.shape)
        self.total_train_size = len(target_position_train_vector)
        # Output Action Tensors
        ground_truth_velocities = np.array(
            [getattr(observation, 'joint_velocities') for episode in demos for observation in episode])  #
        ground_truth_gripper_positions = np.array(
            [getattr(observation, 'gripper_open') for episode in demos for observation in episode])
        ground_truth_gripper_positions = ground_truth_gripper_positions.reshape(len(ground_truth_gripper_positions), 1)
        ground_truth = torch.from_numpy(
            np.concatenate((ground_truth_velocities, ground_truth_gripper_positions), axis=1))

        self.logger.info("Creating Tensordata for Pytorch of Size : %s %s " % (
        str(joint_position_train_vector.size()), str(target_position_train_vector.size())))
        self.dataset = ModularPolicyImagesDataset(joint_position_train_vector, \
                                                  target_position_train_vector, \
                                                  wrist_depth, \
                                                  ground_truth)

        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_vectors(self, demos: List[List[Observation]]):
        joint_pos_arr = np.array(
            [getattr(observation, 'joint_positions') for episode in demos for observation in episode])
        target_pos_arr = np.array(
            [getattr(observation, 'task_low_dim_state') for episode in demos for observation in episode])

        wrist_depth = np.array([getattr(observation, 'wrist_depth') for episode in demos for observation in episode])
        return joint_pos_arr, target_pos_arr, wrist_depth

    def get_distance_vectors(self,demos:List[List[Observation]]):

        target_pos_arr = np.array([getattr(observation, 'task_low_dim_state') for episode in demos for observation in episode])
        endeff_pos_arr = np.array([getattr(observation, 'gripper_pose') for episode in demos for observation in episode])

        return target_pos_arr,endeff_pos_arr

    def train_agent(self, epochs: int):
        if not self.dataset:
            raise Exception(
                "No Training Data Set to Train Agent. Please set Training Data using ImmitationLearningAgent.injest_demonstrations")

        self.logger.info("Starting Training of Agent ")
        self.neural_network.train().to(device)
        final_loss = []
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0
            for batch_idx, (jointpos, targetpos, wrist_depth, output) in enumerate(self.data_loader):
                jointpos, targetpos, wrist_depth, output = Variable(jointpos), Variable(
                    targetpos),Variable(wrist_depth), Variable(output)
                self.optimizer.zero_grad()
                wrist_depth=(wrist_depth.float()).to(device)
                output=(output.float()).to(device)
                network_pred = self.neural_network(wrist_depth.float()).to(device)
                loss = self.loss_function(network_pred, output.float())
                loss.backward()

                if self.collect_gradients:
                    self.set_gradients(self.neural_network.named_parameters())
                self.optimizer.step()
                running_loss += loss.item() * jointpos.size(0)
                steps += 1


            self.logger.info('[%d] loss: %.6f' % (epoch + 1, running_loss / (steps + 1)))
            final_loss.append(float(running_loss))

        return final_loss

    def predict_action(self, demonstration_episode: List[Observation], **kwargs) -> np.array:
        self.neural_network.eval()
        joint_pos_arr, target_pos_arr,  wrist_depth_arr = self.get_train_vectors(
            [demonstration_episode])

        wrist_depth = Variable(torch.from_numpy(wrist_depth_arr))
        wrist_depth = (wrist_depth.float()).to(device)
        output = self.neural_network(wrist_depth.float()).to(device)
        op = output.data.cpu().numpy()
        return op[0]  # Because there is only one action as output.

    def get_distance(self, demonstration_episode:List[Observation],**kwargs) -> np.array:
        target_pos_arr, endeff_pos_arr = self.get_distance_vectors([demonstration_episode])
        endeff_pos_arr=endeff_pos_arr[:,:3]
        dis=distance.euclidean(target_pos_arr,endeff_pos_arr)

        return dis