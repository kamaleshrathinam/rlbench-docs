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


class SimpleFullyConnectedPolicyEstimator(nn.Module):
    
    def __init__(self,num_states,num_actions,num_layers=1):
        super(SimpleFullyConnectedPolicyEstimator, self).__init__()
        self.fc0 = nn.Linear(num_states, 200)

        self.layer_names = ['fc'+str(i+1) for i in range(num_layers)]
        for layer_name in self.layer_names:
            setattr(self,layer_name,nn.Linear(200,200))
        self.fc_op = nn.Linear(200, num_actions)

    # x is input to the network.
    def forward(self, x):
        x = F.relu(self.fc0(x))
        for layer_name in self.layer_names:
            layer = getattr(self,layer_name)
            x = F.relu(layer(x))
        x = self.fc_op(x)
        return x


class SimplePolicyDataset(Dataset):
    def __init__(self, final_vector,actions):
        self.final_vector = final_vector
        self.actions = actions

        
    def __getitem__(self, index):
        return (self.final_vector[index], self.actions[index])

    def __len__(self):
        return len(self.final_vector) # of how many examples(images?) you have

class SimpleImmitationLearningAgent(TorchAgent):
    """
    SimpleImmitationLearningAgent
    -----------------------
    Ment for the ReachTarget Task. 

    Slightly Smarter Agent than its predecessor as it will try to eastimate an action given a joint positions + final coordinates as a single vector. 
    This will Not be considering Past actions/states while making these predictions. 
    
    - [joint_positions_1,....joint_position_7,x,y,z] : Tensor Representation
    """
    def __init__(self,learning_rate = 0.01,batch_size=64,num_layers=4,collect_gradients=False):
        super(TorchAgent,self).__init__(collect_gradients=collect_gradients)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = SimpleFullyConnectedPolicyEstimator(10,8,num_layers=num_layers)
        self.optimizer = optim.SGD(self.neural_network.parameters(), lr=learning_rate, momentum=0.9)
        self.loss_function = nn.SmoothL1Loss()
        self.training_data = None
        agent_name = __class__.__name__+'__Layer__'+str(num_layers)
        self.logger = logger.create_logger(agent_name)
        self.logger.propagate = 0
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None
        self.batch_size =batch_size
        

    def injest_demonstrations(self,demos:List[List[Observation]],**kwargs):
        """
        take demos and make one tensor from
            - joint_positions
            - target_positions
        Output Labels are : Joint velcoities
        """
        # Input State Tensors
        final_vector = self.get_train_vectors(demos)    
        final_torch_tensor = torch.from_numpy(final_vector)
        self.total_train_size = len(final_torch_tensor)
        # Output Action Tensors
        ground_truth_velocities = np.array([getattr(observation,'joint_velocities') for episode in demos for observation in episode]) #
        ground_truth_gripper_positions = np.array([getattr(observation,'gripper_open') for episode in demos for observation in episode])
        ground_truth_gripper_positions = ground_truth_gripper_positions.reshape(len(ground_truth_gripper_positions),1)
        ground_truth = torch.from_numpy(np.concatenate((ground_truth_velocities,ground_truth_gripper_positions),axis=1))
        
        self.logger.info("Creating Tensordata for Pytorch of Size : %s" % (str(final_torch_tensor.size())))
        self.dataset = SimplePolicyDataset(final_torch_tensor, ground_truth)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_vectors(self,demos:List[List[Observation]]):
        joint_pos_arr = np.array([getattr(observation,'joint_positions') for episode in demos for observation in episode])
        target_pos_arr = np.array([getattr(observation,'task_low_dim_state') for episode in demos for observation in episode])
        final_vector = np.concatenate((joint_pos_arr,target_pos_arr),axis=1)
        return final_vector

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
        final_loss = []
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0
            for batch_idx, (final_torch_tensor,output) in enumerate(self.data_loader):
                final_torch_tensor,output = Variable(final_torch_tensor),Variable(output) 
                self.optimizer.zero_grad()
                network_pred = self.neural_network(final_torch_tensor.float()) 
                loss = self.loss_function(network_pred,output.float())
                loss.backward()
                if self.collect_gradients:
                    self.set_gradients(self.neural_network.named_parameters())
                self.optimizer.step()
                running_loss += loss.item()*final_torch_tensor.size(0)
                steps+=1

            self.logger.info('[%d] loss: %.6f' % (epoch + 1, running_loss / (steps+1)))
            final_loss.append(float(running_loss))
        
        return final_loss

    def predict_action(self, demonstration_episode:List[Observation],**kwargs) -> np.array:
        self.neural_network.eval()
        final_vector = self.get_train_vectors([demonstration_episode])
        final_vector = Variable(torch.from_numpy(final_vector))
        output = self.neural_network(final_vector.float())
        op = output.data.cpu().numpy()
        return op[0] # Because there is only one action as output. 

    def get_distance(self, demonstration_episode: List[Observation], **kwargs) -> np.array:
        target_pos_arr, endeff_pos_arr = self.get_distance_vectors([demonstration_episode])
        endeff_pos_arr = endeff_pos_arr[:, :3]
        dis = distance.euclidean(target_pos_arr, endeff_pos_arr)

        return dis
