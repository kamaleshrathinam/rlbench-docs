import numpy as np
from rlbench.backend.observation import Observation
'''
Reward Functions
'''
class RewardFunction():
    def __call__(self,env, state:Observation, action, rl_bench_reward):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__

class MahattanReward(RewardFunction):
    """
    reward_function : Reward function tanh of manhattan distance. 
    Input Parameters
    ---------- 
    state : state observation.
    action: action taken  by the agent in case needed for reward shaping=
    
    Using the manhattan distance
    manhattan distace= |x1-x2|+|y1-y2|+|z1-z2|
    """
    def __call__(self,env, state:Observation, action, rl_bench_reward):
        ee_pose=np.array([getattr(state,'gripper_pose')[:3]])
        target_pose=np.array([getattr(state,'task_low_dim_state')])
        distance= abs(target_pose[0][0]-ee_pose[0][0])+abs(target_pose[0][1]-ee_pose[0][1])\
            +abs(target_pose[0][2]-ee_pose[0][2])
        reward=np.tanh(distance)+rl_bench_reward
        return reward

class ExponentialMahattanReward(RewardFunction):
    """
    reward_function : Reward function negative exponential of manhatten distance. 
    Input Parameters
    ---------- 
    state : state observation.
    action: action taken  by the agent in case needed for reward shaping
    
    Using the manhattan distance
    manhattan distace= |x1-x2|+|y1-y2|+|z1-z2|
    """
    def __call__(self,env, state:Observation, action, rl_bench_reward):
        ee_pose=np.array([getattr(state,'gripper_pose')[:3]])
        target_pose=np.array([getattr(state,'task_low_dim_state')])
        distance= abs(target_pose[0,0]-ee_pose[0,0])+abs(target_pose[0,1]-ee_pose[0,1])\
            +abs(target_pose[0,2]-ee_pose[0,2])

        reward=np.exp(-distance)
        return reward

class EuclideanReward(RewardFunction):
    """
    reward_function : Reward function tanh of euclidean distance. 
    Input Parameters
    ---------- 
    state : state observation.
    action: action taken  by the agent in case needed for reward shaping=
    """
    def __call__(self,env, state:Observation, action, rl_bench_reward):
        ee_pose=np.array([getattr(state,'gripper_pose')[:3]])
        target_pose=np.array([getattr(state,'task_low_dim_state')])

        distance =np.sqrt((target_pose[0,0]-ee_pose[0,0])**2+(target_pose[0,1]-ee_pose[0,1])**2+(target_pose[0,2]-ee_pose[0,2])**2)
        reward=np.tanh(1/distance)
        return reward

class ExponentialEuclideanReward(RewardFunction):
    """
    reward_function : Reward function exponential of euclidean distance. 
    Input Parameters
    ---------- 
    state : state observation.
    action: action taken  by the agent in case needed for reward shaping=
    """
    def __call__(self,env, state:Observation, action, rl_bench_reward):
        ee_pose=np.array([getattr(state,'gripper_pose')[:3]])
        target_pose=np.array([getattr(state,'task_low_dim_state')])
        distance =np.sqrt((target_pose[0,0]-ee_pose[0,0])**2+(target_pose[0,1]-ee_pose[0,1])**2+(target_pose[0,2]-ee_pose[0,2])**2)
        reward=np.exp(1/distance)
        return reward

class HuberReward(RewardFunction):
    '''
    .. math::
        \text{loss}(x, y) = \frac{1}{n} \sum_{i} z_{i}

        where :math:`z_{i}` is given by:

    .. math::
        z_{i} =
        \begin{cases}
        0.5 (x_i - y_i)^2, & \text{if } |x_i - y_i| < 1 \\
        |x_i - y_i| - 0.5, & \text{otherwise }
        \end{cases}
    '''
    def __call__(self, env, state, action, rl_bench_reward):
        ee_pose=np.array([getattr(state,'gripper_pose')[:3]])
        target_pose=np.array([getattr(state,'task_low_dim_state')]) 
        # reward = -c1*huberloss(x,y)-c2*action'*action
        huber_loss= self.HuberLoss(target_pose[0],ee_pose[0])
        reward= -0.1*huber_loss-0.1*np.dot(action[0:6].T,action[0:6])
        return reward
    
    @staticmethod
    def HuberLoss(x,y):
        loss=np.zeros(len(x))
        for i in range(len(x)):
            if np.abs(x[i]-y[i]) < 1:
                loss[i]=0.5*(x[i]-y[i])**2
            else:
                loss[i]=np.abs(x[i]-y[i])-0.5
        return np.mean(loss)
