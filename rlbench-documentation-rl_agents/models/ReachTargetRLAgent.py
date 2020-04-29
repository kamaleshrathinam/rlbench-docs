# Import Absolutes deps
import torch.nn as nn
import torch as T
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
from models.Agent import TorchRLAgent
import logger
from models.CupboardAgents.DDPG.ddpg import DDPGArgs
 
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        #f2 = 0.002
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        #self.q.weight.data.uniform_(-f3, f3)
        #self.q.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #f2 = 0.002
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        #f3 = 0.004
        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        #self.mu.weight.data.uniform_(-f3, f3)
        #self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x


class DDPG(object):
    def __init__(self,arguements=DDPGArgs(),input_dims=[13],n_actions=8):
        self.gamma = arguements.discount
        self.tau = arguements.tau
        self.memory = ReplayBuffer(arguements.rmsize, input_dims, n_actions)
        self.batch_size = arguements.bsize

        self.actor = ActorNetwork(arguements.prate, input_dims, arguements.hidden1,
                                  arguements.hidden2, n_actions=n_actions,
                                  name='Actor')
        self.critic = CriticNetwork(arguements.rate, input_dims, arguements.hidden1,
                                    arguements.hidden2, n_actions=n_actions,
                                    name='Critic')

        self.target_actor = ActorNetwork(arguements.prate, input_dims, arguements.hidden1,
                                         arguements.hidden2, n_actions=n_actions,
                                         name='TargetActor')
        self.target_critic = CriticNetwork(arguements.rate, input_dims, arguements.hidden1,
                                           arguements.hidden2, n_actions=n_actions,
                                           name='TargetCritic')

        self.noise = OUActionNoise(mu=np.zeros(n_actions), sigma=arguements.ou_sigma, theta=arguements.ou_theta)

        self.update_network_parameters(tau=1)

    def to_object(self):
        networks = {
            'target':{
                'actor':self.target_actor.state_dict(),
                'critic':self.target_critic.state_dict()
            },
            'pred':{
               'actor':self.actor.state_dict(),
                'critic':self.critic.state_dict()
            }
        }
        return networks
    
    def from_object(self,nw_object):
        if 'target' not in nw_object or 'pred' not in nw_object:
            raise Exception("No Target Or Prediction Network as Properties")
        self.target_actor.load_state_dict(nw_object['target']['actor'])
        self.target_critic.load_state_dict(nw_object['target']['critic'])
        self.actor.load_state_dict(nw_object['pred']['actor'])
        self.critic.load_state_dict(nw_object['pred']['critic'])
        print("Model Loaded!")
        


    def select_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        #print(mu_prime.cpu().detach().numpy()[0])
        return mu_prime.cpu().detach().numpy()[0]
  
    def observe(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_policy(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

STATE_DIM_MAP = {
    'joint_velocities':7,
    'joint_velocities_noise':7,
    'joint_positions':7,
    'joint_positions_noise':7,
    'joint_forces':7,
    'joint_forces_noise':7,
    'gripper_pose':7,
    'gripper_touch_forces':7,
    'task_low_dim_state':3
}   

class ReachTargetRLAgent(TorchRLAgent):
    """
    ReachTargetRLAgent
    -----------------------
    Algo Of Choice : https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    
    OUTPUT ACTION MODE : ABS_JOINT_VELOCITY
    
    Why DDPG : 
   
        So as its a continous actionspace one can try and use DDPG 
        https://math.stackexchange.com/questions/3179912/policy-gradient-reinforcement-learning-for-continuous-state-and-action-space
        https://ai.stackexchange.com/questions/4085/how-can-policy-gradients-be-applied-in-the-case-of-multiple-continuous-actions

    TODO : ADD SAVE AND LOAD MODEL METHODS TO ACCOMODATE DDPG.
    """
    def __init__(self,collect_gradients=False,warmup=50,ddpg_args=DDPGArgs(),input_states=['joint_velocities','task_low_dim_state']):
        super(ReachTargetRLAgent,self).__init__(collect_gradients=collect_gradients,warmup=warmup)
        # action should contain 1 extra value for gripper open close state
        
        input_dims = [0 if st not in STATE_DIM_MAP else STATE_DIM_MAP[st] for st in input_states]
        input_dims = [sum(input_dims)]
        self.neural_network = DDPG(arguements=ddpg_args,input_dims=input_dims,n_actions=8) # 1 DDPG Setup with Different Predictors. 
        self.agent_name ="DDPG__AGENT"
        self.logger = logger.create_logger(self.agent_name)
        self.input_states = [st for st in input_states if st in STATE_DIM_MAP]
        self.logger.propagate = 0
        self.data_loader = None
        self.dataset = None
        self.print_every = 40
        self.curr_state = None
        self.logger.info("Agent Wired With Input States : %s",','.join(self.input_states))
   
    
    def get_information_vector(self,demonstration_episode:List[Observation]):
        final_arrs = [np.array([getattr(observation,input_state) for observation in demonstration_episode]) for input_state in self.input_states]
        final_vector = np.concatenate(tuple(final_arrs),axis=1)
        return final_vector

    def observe(self,state_t1:List[Observation],action_t,reward_t:int,done:bool):
        """
        State s_t1 can be None because of errors thrown by policy. 
        """
        state_t1 = None if state_t1[0] is None else self.get_information_vector(state_t1)
        self.neural_network.observe(self.curr_state,action_t,reward_t,state_t1,done)
    
    def update(self):
        self.neural_network.update_policy()

    def reset(self,state:List[Observation]):
        self.curr_state = self.get_information_vector(state)
        # self.neural_network.reset(self.get_information_vector(state))

    def load_model_from_object(self,state_dict):
        self.neural_network.from_object(state_dict)

    def get_model(self):
        return self.neural_network.to_object()

    def act(self,state:List[Observation],timestep=0):
        """
        ACTION PREDICTION : ABS_JOINT_VELOCITY
        """
        state = self.get_information_vector(state)
        self.curr_state = state
        action = self.neural_network.select_action(state) # 8 Dim Vector
        # action = list(action)
        return action