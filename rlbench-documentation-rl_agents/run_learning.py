# import deep_learning_rl as Simulator  
from SimulationEnvironment.ReachTargetRL import ReachTargetRLEnvironment
from models.ReachTargetRLAgent import ReachTargetRLAgent    
from SimulationEnvironment.RewardFunctions import EuclideanReward
import time
curr_env = ReachTargetRLEnvironment(EuclideanReward(),dataset_root='/home/valay/Documents/robotics-data/rlbench_data',headless=True)
agent = ReachTargetRLAgent() 
curr_env.train_rl_agent(agent)