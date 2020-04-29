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


class ConvolutionalPolicyEstimator(nn.Module):

    def __init__(self,num_actions):
        super(ConvolutionalPolicyEstimator, self).__init__()
        # Image will be 128 * 128
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(4*32*32, 200) # Input Dims Related to Op Dims of cnn_layer
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_actions)

    # x is input to the network.
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModularConvolutionalPolicyEstimator(nn.Module):

    def __init__(self,action_dims=8):
        super(ModularConvolutionalPolicyEstimator, self).__init__()
        modular_policy_op_dims = 10
        # Define Modular Policy dimes for joint vectors 
        self.joint_pos_policy = FullyConnectedPolicyEstimator(7,modular_policy_op_dims)
        self.target_pos_policy = FullyConnectedPolicyEstimator(3,modular_policy_op_dims)
        # Define modular policy dimes for image based ConvolutionalPolicyEstimator
        self.left_rgb_policy = ConvolutionalPolicyEstimator(modular_policy_op_dims)
        self.right_rgb_policy = ConvolutionalPolicyEstimator(modular_policy_op_dims)
        self.wrist_rgb_policy = ConvolutionalPolicyEstimator(modular_policy_op_dims)

        # Define connecting Fc Linear layers.
        self.fc1 = nn.Linear(50, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200,action_dims)

    def forward(self,joint_pos,target_pos,left_rgb,right_rgb,wrist_rgb):
        joint_pos_op = self.joint_pos_policy(joint_pos)
        target_pos_op = self.target_pos_policy(target_pos)
        
        left_rgb_op = self.left_rgb_policy(left_rgb)
        right_rgb_op = self.right_rgb_policy(right_rgb)
        wrist_rgb_op = self.wrist_rgb_policy(wrist_rgb)
        # option 1
        # stacked_tensor : shape (batch_size,10+10)
        # Output is combined into a single tensor.
        stacked_tensor = torch.cat((joint_pos_op,target_pos_op,left_rgb_op,right_rgb_op,wrist_rgb_op),1)
        op = F.relu(self.fc1(stacked_tensor))
        op = F.relu(self.fc2(op))
        op = self.fc3(op)

        return op


class ModularPolicyImagesDataset(Dataset):
    def __init__(self, joint_pos,target_pos,left_shoulder_rgb_vector,right_shoulder_rgb_vector,wrist_rgb,actions):
        self.joint_pos = joint_pos
        self.target_pos = target_pos
        self.actions = actions
        self.left_shoulder_rgb_vector = left_shoulder_rgb_vector
        self.right_shoulder_rgb_vector = right_shoulder_rgb_vector
        self.wrist_rgb_vector = wrist_rgb
        
    def __getitem__(self, index):
        return (self.joint_pos[index], self.target_pos[index],self.left_shoulder_rgb_vector[index],self.right_shoulder_rgb_vector[index],self.wrist_rgb_vector[index],self.actions[index])

    def __len__(self):
        return len(self.joint_pos) # of how many examples(images?) you have

class ImmitationLearningConvolvingMutantAgent(TorchAgent):
    """
    ImmitationLearningConvolvingMutantAgent
    -----------------------
    Ment for the ReachTarget Task. 

    Slightly Smarter Agent than its predecessor `ImmitationLearningMutantAgent` as it will try to eastimate an action given a joint positions + final coordinates + image capture of the Object. 
    This will Not be considering Past actions/states while making these predictions. 
    But it will have a lot more information to help it guide decision making. 
    
    ModularConvolutionalPolicyEstimator is not working out so well. 

    The Parent NN takes 4 tensors as input. Output is the actions. 
        - Modular child NN based on type of tensor. 
        - If tensor is like and image then Policy has conv NN otherwise just dense sequential NN

    """
    def __init__(self,learning_rate = 0.01,batch_size=64,collect_gradients=False):
        super(TorchAgent,self).__init__(collect_gradients=collect_gradients)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = ModularConvolutionalPolicyEstimator()
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
        take demos and make 5 tensors
            - joint_positions
            - target_positions
            - left_camera_rgb
            - right_camera_rgb
            - wrist_rgb

        Output Labels are : The velocities
        """
        # Input State Tensors
        joint_pos_arr,target_pos_arr,left_shoulder_rgb,right_shoulder_rgb,wrist_rgb = self.get_train_vectors(demos)    
        joint_position_train_vector = torch.from_numpy(joint_pos_arr)
        target_position_train_vector = torch.from_numpy(target_pos_arr)
        left_shoulder_rgb_vector = torch.from_numpy(left_shoulder_rgb)
        right_shoulder_rgb_vector = torch.from_numpy(right_shoulder_rgb)
        wrist_rgb  = torch.from_numpy(wrist_rgb)
        print("Wrist RGB Shape",wrist_rgb.shape,right_shoulder_rgb_vector.shape,left_shoulder_rgb_vector.shape)
        self.total_train_size = len(target_position_train_vector)
        # Output Action Tensors
        ground_truth_velocities = np.array([getattr(observation,'joint_velocities') for episode in demos for observation in episode]) #
        ground_truth_gripper_positions = np.array([getattr(observation,'gripper_open') for episode in demos for observation in episode])
        ground_truth_gripper_positions = ground_truth_gripper_positions.reshape(len(ground_truth_gripper_positions),1)
        ground_truth = torch.from_numpy(np.concatenate((ground_truth_velocities,ground_truth_gripper_positions),axis=1))
        
        self.logger.info("Creating Tensordata for Pytorch of Size : %s %s " % (str(joint_position_train_vector.size()),str(target_position_train_vector.size())))
        self.dataset = ModularPolicyImagesDataset(joint_position_train_vector,\
                                                target_position_train_vector,\
                                                left_shoulder_rgb_vector,\
                                                right_shoulder_rgb_vector,\
                                                wrist_rgb,\
                                                ground_truth)
                                                
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_vectors(self,demos:List[List[Observation]]):
        joint_pos_arr = np.array([getattr(observation,'joint_positions') for episode in demos for observation in episode])
        target_pos_arr = np.array([getattr(observation,'task_low_dim_state') for episode in demos for observation in episode])
        left_shoulder_rgb = np.array([getattr(observation,'left_shoulder_rgb') for episode in demos for observation in episode])
        right_shoulder_rgb = np.array([getattr(observation,'right_shoulder_rgb') for episode in demos for observation in episode])
        wrist_rgb = np.array([getattr(observation,'wrist_rgb') for episode in demos for observation in episode])
        return joint_pos_arr,target_pos_arr,left_shoulder_rgb,right_shoulder_rgb,wrist_rgb

    def train_agent(self,epochs:int):
        if not self.dataset:
            raise Exception("No Training Data Set to Train Agent. Please set Training Data using ImmitationLearningAgent.injest_demonstrations")
        
        self.logger.info("Starting Training of Agent ")
        self.neural_network.train()
        final_loss = []
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0
            for batch_idx, (jointpos, targetpos,left_rgb,right_rgb,wrist_rgb,output) in enumerate(self.data_loader):
                jointpos, targetpos,left_rgb,right_rgb,wrist_rgb, output = Variable(jointpos), Variable(targetpos), Variable(left_rgb),Variable(right_rgb),Variable(wrist_rgb),Variable(output) 
                self.optimizer.zero_grad()
                network_pred = self.neural_network(jointpos.float(),targetpos.float(),left_rgb.float(),right_rgb.float(),wrist_rgb.float()) 
                loss = self.loss_function(network_pred,output.float())
                loss.backward()
                if self.collect_gradients:
                    self.set_gradients(self.neural_network.named_parameters())
                self.optimizer.step()
                running_loss += loss.item()*jointpos.size(0)
                steps+=1
                if steps % 10 == 0:
                    self.logger.info('[%d][%d] loss: %.6f' % (epoch + 1,steps + 1, running_loss / (steps+1)))

            self.logger.info('[%d] loss: %.6f' % (epoch + 1, running_loss / (steps+1)))
            final_loss.append(float(running_loss))
        
        return final_loss

    def predict_action(self, demonstration_episode:List[Observation],**kwargs) -> np.array:
        self.neural_network.eval()
        joint_pos_arr,target_pos_arr,left_shoulder_rgb_arr,right_shoulder_rgb_arr,wrist_rgb_arr = self.get_train_vectors([demonstration_episode])
        
        joint_pos = Variable(torch.from_numpy(joint_pos_arr))
        target_pos = Variable(torch.from_numpy(target_pos_arr))
        left_shoulder_rgb = Variable(torch.from_numpy(left_shoulder_rgb_arr))
        right_shoulder_rgb  = Variable(torch.from_numpy(right_shoulder_rgb_arr))
        wrist_rgb = Variable(torch.from_numpy(wrist_rgb_arr))
        output = self.neural_network(joint_pos.float(),target_pos.float(),left_shoulder_rgb.float(),right_shoulder_rgb.float(),wrist_rgb.float())
        op = output.data.cpu().numpy()
        return op[0] # Because there is only one action as output. 
