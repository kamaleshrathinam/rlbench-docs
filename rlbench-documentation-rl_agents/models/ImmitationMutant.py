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

class FullyConnectedPolicyEstimator(nn.Module):
    
    def __init__(self,num_states,num_actions):
        super(FullyConnectedPolicyEstimator, self).__init__()
        self.fc1 = nn.Linear(num_states, 200)
        self.fc3 = nn.Linear(200, num_actions)

    # x is input to the network.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

class ModularPolicyEstimator(nn.Module):
    
    def __init__(self,action_dims=8):
        super(ModularPolicyEstimator, self).__init__()
        modular_policy_op_dims = 10
        # Define Modular Policy dimes
        self.joint_pos_policy = FullyConnectedPolicyEstimator(7,modular_policy_op_dims)
        self.target_pos_policy = FullyConnectedPolicyEstimator(3,modular_policy_op_dims)

        # Define connecting Fc Linear layers.
        self.fc1 = nn.Linear(20, 200)
        self.fc3 = nn.Linear(200,action_dims)

    def forward(self,joint_pos,target_pos):
        joint_pos_op = self.joint_pos_policy(joint_pos)
        target_pos_op = self.target_pos_policy(target_pos)
        
        # option 1
        # stacked_tensor : shape (batch_size,10+10)
        # Output is combined into a single tensor.
        stacked_tensor = torch.cat((joint_pos_op,target_pos_op),1)
        op = F.relu(self.fc1(stacked_tensor))
        op = self.fc3(op)

        return op



class ModularPolicyDataset(Dataset):
    def __init__(self, joint_pos,target_pos,actions):
        self.joint_pos = joint_pos
        self.target_pos = target_pos
        self.actions = actions

        
    def __getitem__(self, index):
        return (self.joint_pos[index], self.target_pos[index],self.actions[index])

    def __len__(self):
        return len(self.joint_pos) # of how many examples(images?) you have

class ImmitationLearningMutantAgent(TorchAgent):
    """
    ImmitationLearningMutantAgent
    -----------------------
    Ment for the ReachTarget Task. 

    Slightly Smarter Agent than its predecessor as it will try to eastimate an action given a joint positions + final coordinates of the Object. 
    This will Not be considering Past actions/states while making these predictions. 
    
    - Uses a Parent NN which has modular sub NNs which deal with the joint_position and final_coords. 
    - Parent NN will then concat the output on the axis=1
        Eg, 
            x = tensor([[1,2,3]])
            y = tensor([[1,2,3]])
            z  = torch.cat((x,y),1)
            
            z is tensor([[1,2,3,1,2,3]])
    - The concat vector is passed through fully connected layers to finally yeild output vector.
    """
    def __init__(self,learning_rate = 0.01,batch_size=64,collect_gradients=False):
        super(TorchAgent,self).__init__(collect_gradients=collect_gradients)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = ModularPolicyEstimator()
        self.optimizer = optim.SGD(self.neural_network.parameters(), lr=learning_rate, momentum=0.9)
        self.loss_function = nn.SmoothL1Loss()
        self.training_data = None
        self.logger = logger.create_logger(__class__.__name__)
        self.logger.propagate = 0
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None
        self.batch_size =batch_size

    def injest_demonstrations(self,demos:List[List[Observation]],**kwargs):
        """
        take demos and make two tensors
            - joint_positions
            - target_positions
        Output Labels are : 
        """
        # Input State Tensors
        joint_pos_arr,target_pos_arr = self.get_train_vectors(demos)    
        joint_position_train_vector = torch.from_numpy(joint_pos_arr)
        target_position_train_vector = torch.from_numpy(target_pos_arr)
        self.total_train_size = len(joint_position_train_vector)
        # Output Action Tensors
        ground_truth_velocities = np.array([getattr(observation,'joint_velocities') for episode in demos for observation in episode]) #
        ground_truth_gripper_positions = np.array([getattr(observation,'gripper_open') for episode in demos for observation in episode])
        ground_truth_gripper_positions = ground_truth_gripper_positions.reshape(len(ground_truth_gripper_positions),1)
        ground_truth = torch.from_numpy(np.concatenate((ground_truth_velocities,ground_truth_gripper_positions),axis=1))
        
        self.logger.info("Creating Tensordata for Pytorch of Size : %s %s " % (str(joint_position_train_vector.size()),str(target_position_train_vector.size())))
        self.dataset = ModularPolicyDataset(joint_position_train_vector,target_position_train_vector, ground_truth)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_vectors(self,demos:List[List[Observation]]):
        joint_pos_arr = np.array([getattr(observation,'joint_positions') for episode in demos for observation in episode])
        target_pos_arr = np.array([getattr(observation,'task_low_dim_state') for episode in demos for observation in episode])
        return joint_pos_arr,target_pos_arr

    def get_distance_vectors(self, demos: List[List[Observation]]):

        target_pos_arr = np.array([getattr(observation, 'task_low_dim_state') for episode in demos for observation in episode])
        endeff_pos_arr = np.array([getattr(observation, 'gripper_pose') for episode in demos for observation in episode])

        return target_pos_arr, endeff_pos_arr

    def train_agent(self,epochs:int):
        if not self.dataset:
            raise Exception("No Training Data Set to Train Agent. Please set Training Data using ImmitationLearningAgent.injest_demonstrations")
        
        self.logger.info("Starting Training of Agent ")
        self.neural_network.train()
        final_loss = []
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0
            for batch_idx, (jointpos, targetpos,output) in enumerate(self.data_loader):
                jointpos, targetpos,output = Variable(jointpos), Variable(targetpos),Variable(output) 
                self.optimizer.zero_grad()
                network_pred = self.neural_network(jointpos.float(),targetpos.float()) 
                loss = self.loss_function(network_pred,output.float())
                loss.backward()
                if self.collect_gradients:
                    self.set_gradients(self.neural_network.named_parameters())
                self.optimizer.step()
                running_loss += loss.item()*jointpos.size(0)
                steps+=1

            self.logger.info('[%d] loss: %.6f' % (epoch + 1, running_loss / (steps+1)))
            final_loss.append(float(running_loss))
        
        return final_loss

    def predict_action(self, demonstration_episode:List[Observation],**kwargs) -> np.array:
        self.neural_network.eval()
        joint_pos_arr,target_pos_arr = self.get_train_vectors([demonstration_episode])
        joint_pos = Variable(torch.from_numpy(joint_pos_arr))
        target_pos = Variable(torch.from_numpy(target_pos_arr))
        output = self.neural_network(joint_pos.float(),target_pos.float())
        op = output.data.cpu().numpy()
        return op[0] # Because there is only one action as output.

    def get_distance(self, demonstration_episode:List[Observation],**kwargs) -> np.array:
        target_pos_arr, endeff_pos_arr = self.get_distance_vectors([demonstration_episode])
        endeff_pos_arr=endeff_pos_arr[:,:3]
        dis=distance.euclidean(target_pos_arr,endeff_pos_arr)

        return dis
