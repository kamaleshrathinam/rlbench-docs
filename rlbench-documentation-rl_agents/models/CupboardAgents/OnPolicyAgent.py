# Import Absolutes deps
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data.dataset import Dataset
from rlbench.backend.observation import Observation
from typing import List
import numpy as np
# Import Relative deps
import sys
sys.path.append('..')
from models.Agent import TorchAgent
import logger
 


class FullyConnectedPolicyEstimator(nn.Module):
    
    def __init__(self,num_states,num_actions,hidden_dims=200):
        super(FullyConnectedPolicyEstimator, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, num_actions)

    # x is input to the network.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


class ConvolutionalPolicyEstimator(nn.Module):

    def __init__(self,num_actions,hidden_dims=50):
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
        self.fc1 = nn.Linear(4*32*32, hidden_dims) # Input Dims Related to Op Dims of cnn_layer
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, num_actions)

    # x is input to the network.
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GripperOpenEstimator(nn.Module):
    def __init__(self,image_impact_dims=2,ee_impact_dims=4,hidden_dims=30):
        super(GripperOpenEstimator,self).__init__()
        # Wrist Camera NN
        self.wrist_camera_rgb_convolution_NN = ConvolutionalPolicyEstimator(image_impact_dims) # Hypothesis : Output of this network will 
        # EE Pose NN
        self.end_effector_pose_NN = FullyConnectedPolicyEstimator(7,ee_impact_dims)

        self.fc1 = nn.Linear(ee_impact_dims+image_impact_dims,hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 1)

    def forward(self,EE_Pose,wrist_rgb):
        wrist_op = self.wrist_camera_rgb_convolution_NN(wrist_rgb)
        ee_op = self.end_effector_pose_NN(EE_Pose)
        stacked_tensor = torch.cat((wrist_op,ee_op),1)
        x = F.relu(self.fc1(stacked_tensor))
        x = self.fc3(x) # Todo : Check for softmax . 
        return x


class TargetPositionPolicy(nn.Module):

    def __init__(self,image_impact_dims=2,pred_hidden_dims=50):
        super(TargetPositionPolicy,self).__init__()
        self.l_rgb_conversion = ConvolutionalPolicyEstimator(image_impact_dims)
        self.r_rgb_conversion = ConvolutionalPolicyEstimator(image_impact_dims)
        self.pre_pred_layer = nn.Linear(image_impact_dims*2+1,pred_hidden_dims) # Because of gripper open 
        self.output_layer = nn.Linear(pred_hidden_dims,3) # For X Y Z 

    def forward(self,left_rgb,right_rgb,gripper_open):
        left_rgb_op = self.l_rgb_conversion(left_rgb)
        right_rgb_op = self.r_rgb_conversion(right_rgb)
        stacked_tensor = torch.cat((left_rgb_op,right_rgb_op,gripper_open),1)
        x = F.relu(self.pre_pred_layer(stacked_tensor))
        x = self.output_layer(x)
        return x


class ActionModeEstimator(nn.Module):
    
    def __init__(self,action_dims=7,hidden_pre_pred_dims=30,joint_pos_policy_hidden=20):
        super(ActionModeEstimator, self).__init__()
        modular_policy_op_dims = 10
        # Define Modular Policy dimes
        self.joint_pos_policy = FullyConnectedPolicyEstimator(7,modular_policy_op_dims,hidden_dims=joint_pos_policy_hidden)
        # Define connecting Fc Linear layers.
        input_tensor_dims = modular_policy_op_dims + 3 + 1
        self.pre_pred_layer = nn.Linear(input_tensor_dims, hidden_pre_pred_dims)
        self.output_layer = nn.Linear(hidden_pre_pred_dims,action_dims)

    def forward(self,joint_pos,target_pos,gripper_open):
        joint_pos_op = self.joint_pos_policy(joint_pos)
        # option 1
        # stacked_tensor : shape (batch_size,)
        # Output is combined into a single tensor.
        stacked_tensor = torch.cat((joint_pos_op,target_pos,gripper_open),1)
        op = F.relu(self.pre_pred_layer(stacked_tensor))
        op = self.output_layer(op)
        return op




class ModularPolicyEstimator(nn.Module):
    def __init__(self):
        super(ModularPolicyEstimator,self).__init__()
        self.gripper_open_estimator = GripperOpenEstimator()
        self.action_mode_estimator = ActionModeEstimator()
        self.target_position_estimator = TargetPositionPolicy()
    
    def forward(self,EE_Pose,wrist_rgb,left_rgb,right_rgb,joint_positions):
        gripper_open = self.gripper_open_estimator(EE_Pose,wrist_rgb)
        target_position = self.target_position_estimator(left_rgb,right_rgb,gripper_open)
        pred_action = self.action_mode_estimator(joint_positions,target_position,gripper_open)        
        return gripper_open,pred_action,target_position



class ModularPolicyEstimator(nn.Module):
    def __init__(self):
        super(ModularPolicyEstimator,self).__init__()
        self.gripper_open_estimator = GripperOpenEstimator()
        self.action_mode_estimator = ActionModeEstimator()
        self.target_position_estimator = TargetPositionPolicy()
    
    def forward(self,EE_Pose,wrist_rgb,left_rgb,right_rgb,joint_positions):
        gripper_open = self.gripper_open_estimator(EE_Pose,wrist_rgb)
        target_position = self.target_position_estimator(left_rgb,right_rgb,gripper_open)
        pred_action = self.action_mode_estimator(joint_positions,target_position,gripper_open)        
        return gripper_open,pred_action,target_position

class ModularPolicyDataset(Dataset):
    def __init__(self,EE_Pose_x,wrist_rgb_x,left_rgb_x,right_rgb_x,joint_positions_x,gripper_open,pred_action,target_position):
        self.EE_Pose_x = EE_Pose_x
        self.wrist_rgb_x = wrist_rgb_x
        self.left_rgb_x = left_rgb_x
        self.right_rgb_x = right_rgb_x
        self.joint_positions_x = joint_positions_x
        self.gripper_open = gripper_open
        self.pred_action = pred_action
        self.target_position = target_position
        
        
    def __getitem__(self, index):
        return (
        self.EE_Pose_x[index],\
        self.wrist_rgb_x[index],\
        self.left_rgb_x[index],\
        self.right_rgb_x[index],\
        self.joint_positions_x[index],\
        self.gripper_open[index],\
        self.pred_action[index],\
        self.target_position[index])

    def __len__(self):
        return len(self.EE_Pose_x) # of how many examples(images?) you have


class OnPolicyAgent(TorchAgent):
    """
    OnPolicyAgent
    -----------------------

    
    """
    def __init__(self,learning_rate = 0.01,batch_size=64,collect_gradients=False):
        super(TorchAgent,self).__init__(collect_gradients=collect_gradients)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = ModularPolicyEstimator()
        self.optimizer = optim.SGD(self.neural_network.parameters(), lr=learning_rate, momentum=0.9)
        self.loss_function = nn.SmoothL1Loss()
        self.agent_name = "GROCCERY_PICKING_COMPLEX_"+__class__.__name__
        self.training_data = None
        self.logger = logger.create_logger(__class__.__name__)
        self.logger.propagate = 0
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None
        self.batch_size =batch_size

    def injest_demonstrations(self,demos:List[List[Observation]],**kwargs):
        joint_positions_x,left_rgb_x,right_rgb_x,wrist_rgb_x,EE_Pose_x = self.get_train_vectors(demos)
        joint_positions_x = torch.from_numpy(joint_positions_x)
        left_rgb_x = torch.from_numpy(left_rgb_x)
        right_rgb_x = torch.from_numpy(right_rgb_x)
        wrist_rgb_x = torch.from_numpy(wrist_rgb_x)
        EE_Pose_x = torch.from_numpy(EE_Pose_x)

        self.total_train_size = len(EE_Pose_x)
        
        # $ First Extract the output_action. Meaning the action that will control the kinematics of the robot. 
        ground_truth_velocities = torch.from_numpy(np.array([getattr(observation,'joint_velocities') for episode in demos for observation in episode]))
        # $ Create a matrix for gripper position vectors.                                                                                                                                                     
        ground_truth_gripper_open = np.array([getattr(observation,'gripper_open') for episode in demos for observation in episode])
        ground_truth_gripper_open = torch.from_numpy(ground_truth_gripper_open.reshape(len(ground_truth_gripper_open),1))

        # todo : Calculate Target pos Arra on basis  of the policy vector. 
        target_pos_vector = None
        for demo in demos:
            add_vector = self.get_target_coord_tensor(demo)
            if target_pos_vector is None:
                target_pos_vector = add_vector
            else:
                target_pos_vector = np.concatenate((target_pos_vector,add_vector),axis=0)
        gt_target_position = torch.from_numpy(target_pos_vector)
        
        # demos[0][0].task_low_dim_state contains all target's coordinates
        self.logger.info("Creating Tensordata for Pytorch of Size : %s"%str(EE_Pose_x.size()))
        
        self.dataset = ModularPolicyDataset(
            EE_Pose_x, \
            wrist_rgb_x, \
            left_rgb_x, \
            right_rgb_x, \
            joint_positions_x, \
            ground_truth_gripper_open,\
            ground_truth_velocities,\
            gt_target_position \
        )
        
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_vectors(self,demos:List[List[Observation]]):
        joint_positions_x = np.array([getattr(observation,'joint_positions') for episode in demos for observation in episode])
        left_rgb_x = np.array([getattr(observation,'left_shoulder_rgb') for episode in demos for observation in episode])
        right_rgb_x = np.array([getattr(observation,'right_shoulder_rgb') for episode in demos for observation in episode])
        wrist_rgb_x = np.array([getattr(observation,'wrist_rgb') for episode in demos for observation in episode])
        EE_Pose_x = np.array([getattr(observation,'gripper_pose') for episode in demos for observation in episode])
        return joint_positions_x,left_rgb_x,right_rgb_x,wrist_rgb_x,EE_Pose_x


    def train_agent(self,epochs:int):
        if not self.dataset:
            raise Exception("No Training Data Set to Train Agent. Please set Training Data using ImmitationLearningAgent.injest_demonstrations")
        
        self.logger.info("Starting Training of Agent ")
        self.neural_network.train()
        final_loss = []
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0
            for batch_idx, \
                (EE_Pose_x,\
                wrist_rgb_x,\
                left_rgb_x,\
                right_rgb_x,\
                joint_positions_x, \
                gripper_open_y,\
                pred_action_y,\
                target_position_y) in enumerate(self.data_loader):

                self.optimizer.zero_grad()
                gripper_open,pred_action,target_position = self.neural_network(
                            EE_Pose_x.float(),\
                            wrist_rgb_x.float(),\
                            left_rgb_x.float(),\
                            right_rgb_x.float(),\
                            joint_positions_x.float(),\
                    )
                gripper_open_criterion = self.loss_function(gripper_open,gripper_open_y.float())
                pred_action_criterion = self.loss_function(pred_action,pred_action_y.float())
                target_position_criterion = self.loss_function(target_position,target_position_y.float())
                total_loss = gripper_open_criterion + pred_action_criterion + target_position_criterion
                total_loss.backward()
                if self.collect_gradients:
                    self.set_gradients(self.neural_network.named_parameters())
                self.optimizer.step()
                running_loss += total_loss.item()
                steps+=1
            
            final_loss.append(float(running_loss))
            self.logger.info('[%d] loss: %.6f' % (epoch + 1, running_loss / (steps+1)))
        
        return final_loss

    def get_target_coord_tensor(self,demo:List[Observation]):
        """
        Finds Tensor which will hold XYZ coords according to where the robot needs to go. 
        Eg. the function will return [[1,2,3],[1,2,3],[1,2,3],[5,6,7],[5,6,7],[5,6,7]] which denotes the final XYZ each observation needed to reach. 
        """
        prev = None
        loading_traj_change_index = None
        unloading_traj_change_index = None
        for i,obs in enumerate(demo):
            if prev is None:
                prev = obs
                continue
            if obs.gripper_open!=prev.gripper_open:
                if loading_traj_change_index is None:
                    loading_traj_change_index = i
                elif unloading_traj_change_index is None:
                    unloading_traj_change_index = i 
                # print("Gripper Pos Change",i)
                # print(obs.gripper_pose)
            prev = obs
        
        loading_traj_coords = np.array(demo[loading_traj_change_index].gripper_pose[:3])
        loading_traj_coords = np.reshape(loading_traj_coords,(1,3))
        loading_traj_coords = np.repeat(loading_traj_coords,loading_traj_change_index,axis=0)
        
        unloading_traj_coords = np.array(demo[unloading_traj_change_index].gripper_pose[:3])
        unloading_traj_coords = np.repeat(np.reshape(unloading_traj_coords,(1,3)),len(demo)-loading_traj_change_index,axis=0)

        target_pos_vector = np.concatenate((loading_traj_coords,unloading_traj_coords),axis=0)
        return target_pos_vector

    def predict_action(self, demonstration_episode:List[Observation],**kwargs) -> np.array:
        self.neural_network.eval()
        train_vectors = self.get_train_vectors([demonstration_episode])
        joint_positions_x,left_rgb_x,right_rgb_x,wrist_rgb_x,EE_Pose_x = self.get_train_vectors([demonstration_episode])
        joint_positions_x = torch.from_numpy(joint_positions_x)
        left_rgb_x = torch.from_numpy(left_rgb_x)
        right_rgb_x = torch.from_numpy(right_rgb_x)
        wrist_rgb_x = torch.from_numpy(wrist_rgb_x)
        EE_Pose_x = torch.from_numpy(EE_Pose_x)
        
        gripper_open,pred_action,target_position = self.neural_network(
            EE_Pose_x.float(),\
            wrist_rgb_x.float(),\
            left_rgb_x.float(),\
            right_rgb_x.float(),\
            joint_positions_x.float(),\
        )
        gripper_open = gripper_open.data.cpu().numpy()
        pred_action = pred_action.data.cpu().numpy()
        action = np.concatenate((pred_action,gripper_open),1)
        return action[0]
        # return np.random.uniform(size=(len(batch), 7))