from metaflow import FlowSpec, step,retry
import json

class ReinforcementLearningSimulatorFlow(FlowSpec):
    '''
    Train RL Agents With Different Input States and Reward Functions. 
    Simulate the Same Trained Agent in Simulations of 
        1. Left Environment with all left data. 
        2. Right Environment with all Right data
        3. Environment
    '''
    @step
    def start(self):
        from models.CupboardAgents.DDPG.ddpg import DDPGArgs
        # io_array will hold [[input][ouput]]
        self.possible_states= [
            [['joint_velocities','joint_positions','gripper_pose','task_low_dim_state'],['joint_velocities']],
            # [['joint_velocities','gripper_pose','task_low_dim_state'],['joint_velocities']],
            # [['joint_positions','gripper_pose','task_low_dim_state'],['joint_velocities']],
            # [['gripper_pose','task_low_dim_state'],['joint_velocities']]
        ]
        self.reward_functions = [
            'MahattanReward',
            # 'ExponentialMahattanReward',
            # 'EuclideanReward',
            # 'ExponentialEuclideanReward',
            # 'HuberReward'
        ]
        self.simulation_environments = [
            'ReachTarget',
            # 'LeftTarget',
            # 'RightTarget'
        ]
        self.training_task = 'ReachTarget'
        self.model_args = DDPGArgs()
        self.episode_length=100 # Length of the Episode
        self.warmup = 50 # Warmup Steps for RL Algo
        self.num_episodes=2 # Simulated Testing Epochs.
        self.collect_gradients = False                                                                                                                                                                            
        self.next(self.reward_split_placeholder,foreach='possible_states')

    @step
    def reward_split_placeholder(self):
        '''
        Split the training for different Rewards functions
        '''
        self.input_state = self.input[0]
        self.next(self.train,foreach='reward_functions')


    @retry(times=4)
    @step
    def train(self):
        '''
        Train Reinforcement Learning agent according to an input_state and Reward Function
        `todo` : Create a docker image for training 
        '''
        self.input_value = self.input_state
        self.reward_function = self.input
        from SimulationEnvironment.ReachTargetRL import ReachTargetRLEnvironment
        from models.ReachTargetRLAgent import ReachTargetRLAgent
        
        agent = ReachTargetRLAgent(warmup=self.warmup,ddpg_args=self.model_args,input_states=self.input_state,collect_gradients=self.collect_gradients)
        import importlib
        reward_module = importlib.import_module('SimulationEnvironment.RewardFunctions')
        reward_function = getattr(reward_module,self.reward_function)()
        
        curr_env = ReachTargetRLEnvironment(
                reward_function, 
                num_episodes=self.num_episodes, 
                episode_length=self.episode_length)

        env_metrics = curr_env.train_rl_agent(agent)
        self.training_metrics = env_metrics
        # TODO : Add Model Saving Code here. 
        self.model = agent.get_model()
        self.next(self.simulate_split)
    
    @step
    def simulate_split(self):
        '''
        This step will parallelise the training of the individual simulated models. 
        '''
        self.next(self.simulate,foreach='simulation_environments')
    
    @retry(times=4)
    @step
    def simulate(self):
        '''
        Simulate The Model To Different Versions of the Task such as Training on Left/Right/Everywhere.
        '''
        self.simulation_environment = self.input
        from SimulationEnvironment.ReachTargetRL import ReachTargetRLEnvironment
        from models.ReachTargetRLAgent import ReachTargetRLAgent
        
        # todo : Add         
        agent = ReachTargetRLAgent(warmup=self.warmup,ddpg_args=self.model_args,input_states=self.input_state,collect_gradients=self.collect_gradients)
        import importlib
        reward_module = importlib.import_module('SimulationEnvironment.RewardFunctions')
        reward_function = getattr(reward_module,self.reward_function)()
        task_module = importlib.import_module('rlbench.tasks')
        task = getattr(task_module,self.simulation_environment)

        curr_env = ReachTargetRLEnvironment(
                reward_function,
                task=task, 
                num_episodes=self.num_episodes, 
                episode_length=self.episode_length)        

        agent.load_model_from_object(self.model)
        simulation_analytics = curr_env.train_rl_agent(agent,eval_mode=True)
        self.simulation_analytics = simulation_analytics
        self.next(self.join_simulations)

    @step
    def join_simulations(self,inputs):
        from Utilities.core import SimulationAnalytics,RLMetrics
        from Utilities.core import ModelAnalytics,RLMetrics,StoredModel
        self.model_analytics = []
        self.models = []
        simulation_analytics = []
        for task_data in inputs:
            data = SimulationAnalytics()
            data.analytics = task_data.simulation_analytics
            data.simulation_environment = task_data.simulation_environment            
            simulation_analytics.append(data)

        data = ModelAnalytics()
        data.simulation_analytics = simulation_analytics
        data.reward_function = inputs[0].reward_function
        data.episode_length = inputs[0].episode_length 
        data.warmup = inputs[0].warmup 
        data.num_episodes = inputs[0].num_episodes 
        data.collect_gradients = inputs[0].collect_gradients 
        data.training_analytics.analytics = inputs[0].training_metrics
        data.training_analytics.simulation_environment = inputs[0].training_task
        data.input_states = inputs[0].input_value 
        self.model_analytics.append(data)
        
        data = StoredModel()
        data.episode_length = inputs[0].episode_length 
        data.warmup = inputs[0].warmup 
        data.num_episodes = inputs[0].num_episodes
        data.reward_function = inputs[0].reward_function
        data.model_args = inputs[0].model_args 
        data.collect_gradients = inputs[0].collect_gradients 
        data.training_analytics.analytics = inputs[0].training_metrics
        data.training_analytics.simulation_environment = inputs[0].training_task
        data.input_states = inputs[0].input_value 
        self.models.append(data)
        
        self.next(self.join_reward)

    @step
    def join_reward(self,inputs):
        '''
        Collect all trained Models for all Reward branches of the training 
        '''
        from Utilities.core import ModelAnalytics,RLMetrics,StoredModel
        self.model_analytics = []
        self.models = []
        for task_data in inputs:
            self.models+=task_data.models
            self.model_analytics+=task_data.model_analytics
        
        self.next(self.join_input_states)

    @step
    def join_input_states(self,inputs):
        from Utilities.core import ModelAnalytics,RLMetrics,StoredModel
        self.model_analytics = []
        self.models = []
        for task_data in inputs:
            self.models+=task_data.models
            self.model_analytics+=task_data.model_analytics
        
        self.next(self.end)

    @step
    def end(self):
        print("Done Computation")

if __name__ == '__main__':
    ReinforcementLearningSimulatorFlow()