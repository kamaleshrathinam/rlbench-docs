
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from .model import (Actor, Critic)
from .memory import SequentialMemory
from .random_process import OrnsteinUhlenbeckProcess
from .util import *

# from ipdb import set_trace as debug
class DDPGArgs():
    def __init__(self):
        self.seed = 0
        self.hidden1 = 400 # hidden num of first fully connect layer
        self.hidden2 = 300 # hidden num of second fully connect layer
        self.init_w = 0.003 # 
        self.prate = 0.0001 # policy net learning rate
        self.rate = 0.001 # kearning rate
        self.window_length = 1 
        self.bsize = 64 # minibatch size
        self.tau = 0.001 # moving average for target network
        self.discount = 0.99
        self.epsilon = 50000  # linear decay of exploration policy
        self.ou_theta = 0.15 # noise theta
        self.ou_mu = 0.0 # noise mu
        self.ou_sigma= 0.2 # noises sigma
        self.rmsize = 6000000 # memory size
        self.bounding_min=-1
        self.bounding_max=1



class DDPG(object):
    def __init__(self, nb_states, nb_actions, args=DDPGArgs(),USE_CUDA=False):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.criterion = nn.MSELoss()

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        self.bounding_min = args.bounding_min # To cap the action output of critique
        self.bounding_max = args.bounding_max # To cap the action output of critique

        # 
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        state_tensor = to_tensor(np.array(state_batch),dtype=torch.float)
        action_tensor =  to_tensor(np.array(action_batch),dtype=torch.float)
        # Prepare for the target q batch Or Expected State-Action Value
        next_state_tensor = to_tensor(next_state_batch)
        actor_batch_op = self.actor_target(next_state_tensor)
        next_q_values = self.critic_target([
            next_state_tensor,
            actor_batch_op.detach(), #  It detaches the output from the computational graph. So no gradient will be backpropagated along this variable.
        ])
        terminal_tensor = to_tensor(terminal_batch.astype(np.float),requires_grad=False)
        expected_q_values = to_tensor(reward_batch) + \
            self.discount*terminal_tensor*next_q_values

        # Critic update Basis Value loss from Expected Q Values with Computed Q Values
        self.critic_optim.zero_grad()
        computed_q_values= self.critic([state_tensor,action_tensor ])
        value_loss = self.criterion(computed_q_values, expected_q_values)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor_optim.zero_grad()
        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        # TODO : This is a problem. Need to Fix Action Spaces Here. 
        action = np.random.uniform(self.bounding_min,self.bounding_max,self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        # todo : Actions being damped here. Need to think and figure more. 
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, self.bounding_min, self.bounding_max)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
