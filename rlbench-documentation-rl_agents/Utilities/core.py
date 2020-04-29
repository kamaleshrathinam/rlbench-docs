"""
This will hold all the Core Reporting Structures 
"""

"""
METAFLOW REPORTING STRUCTURES
"""
class ModelAnalytics():
    def __init__(self):
        self.simulation_analytics = []
        self.reward_function = None # see RewardFunctions in SimulationEnvironment.RewardFunctions
        self.training_analytics = SimulationAnalytics()
        self.input_states = None
        self.episode_length = None
        self.warmup = None
        self.num_episodes = None
        self.collect_gradients = None


class StoredModel():
    def __init__(self):
        self.model = None
        self.reward_function = None # see RewardFunctions in SimulationEnvironment.RewardFunctions
        self.training_analytics = SimulationAnalytics()
        self.input_states = None
        self.model_args = None
        self.episode_length = None
        self.warmup = None
        self.num_episodes = None
        self.collect_gradients = None


class SimulationAnalytics():
    def __init__(self):
        self.analytics = None # RLMetrics
        self.simulation_environment = None # Simulation Environment Name 


class EpisodeRewardBuffer():
    def __init__(self,rewards = [],completed = False):
        self.rewards = rewards
        self.completed = completed

    def add(self,reward):
        self.rewards.append(reward)        

    @property
    def total(self):
        return sum(self.rewards)

    def to_json(self):
        return {
            'rewards':self.rewards,
            'completed':self.completed
        }

class RLMetrics():
    def __init__(self,episiode_rewards = []):
        self.episode_rewards = episiode_rewards # [EpisodeRewardBuffer]
    
    def get_new_reward_buffer(self):
        buffer = EpisodeRewardBuffer()
        self.episode_rewards.append(buffer)
        return buffer
    
    def __len__(self):
        return len(self.episode_rewards)
    
    @property
    def total_reward(self):
        return sum([e.total for e in self.episode_rewards])
    
    def to_json(self):
        return [e.to_json() for e in self.episode_rewards]

    def __str__(self):
        total_rewards = sum([e.total for e in self.episode_rewards])
        print_args = {
            'total_rewards' : str(total_rewards),
            'num_episodes':str(len(self.episode_rewards)),
            'avg_reward':str(total_rewards/len(self.episode_rewards))
        }
        return '''
        Total Rewards : {total_rewards}

        Number of Episodes : {num_episodes}

        Average Reward/Episode : {avg_reward}
        '''.format(**print_args)
