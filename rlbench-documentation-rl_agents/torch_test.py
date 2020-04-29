# coding: utf-8
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch.optim as optim
import logger as L
from torch.utils.data.dataset import Dataset
logger = L.create_logger('NN_Visualiser')
import numpy as np

def plot_bar_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

class ModularPolicyDataset(Dataset):
    def __init__(self, joint_pos,target_pos,actions):
        self.joint_pos = joint_pos
        self.target_pos = target_pos
        self.actions = actions

        
    def __getitem__(self, index):
        return (self.joint_pos[index], self.target_pos[index],self.actions[index])

    def __len__(self):
        return len(self.joint_pos) # of how many examples(images?) you have


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

  
x1 = torch.rand(1000,7)
x2 = torch.rand(1000,3)
y = torch.rand(1000,8)
batch_size = 64
epochs = 100
learning_rate = 0.01

dataset = ModularPolicyDataset(x1,x2,y)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
neural_network = ModularPolicyEstimator()
optimizer = optim.SGD(neural_network.parameters(), lr=learning_rate, momentum=0.9)
loss_function = nn.SmoothL1Loss()

neural_network.train()
final_loss = []
for epoch in range(epochs):
    running_loss = 0.0
    steps = 0
    for batch_idx, (jointpos, targetpos,output) in enumerate(data_loader):
        jointpos, targetpos,output = torch.Tensor(jointpos), torch.Tensor(targetpos),torch.Tensor(output) 
        optimizer.zero_grad()
        network_pred = neural_network(jointpos.float(),targetpos.float()) 
        loss = loss_function(network_pred,output.float())
        loss.backward()
        plot_grad_flow(neural_network.named_parameters())
        optimizer.step()
        running_loss += loss.item()*jointpos.size(0)
        steps+=1

    logger.info('[%d] loss: %.6f' % (epoch + 1, running_loss / (steps+1)))
    final_loss.append(float(running_loss))

plt.show()