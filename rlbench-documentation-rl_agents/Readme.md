# Deep Learning On RLBench 
Use RLBench to simulate/train robot agents. The robot agents will use Neural networks to run. 


# Relevant Modules

## Environment
- The `Environment` can be imported. Every environment takes an `ActionMode` and `ObservationConfig` which will help determine the inputs actions and the observations the environment will make. 
- Every `Observation` consists of the below data points which are captured in the demos. 
    ```python
    class Observation(object):
        """Storage for both visual and low-dimensional observations."""

        def __init__(self,
                    left_shoulder_rgb: np.ndarray,
                    left_shoulder_depth: np.ndarray,
                    left_shoulder_mask: np.ndarray,
                    right_shoulder_rgb: np.ndarray,
                    right_shoulder_depth: np.ndarray,
                    right_shoulder_mask: np.ndarray,
                    wrist_rgb: np.ndarray,
                    wrist_depth: np.ndarray,
                    wrist_mask: np.ndarray,
                    joint_velocities: np.ndarray,
                    joint_positions: np.ndarray,
                    joint_forces: np.ndarray,
                    gripper_open_amount: float,
                    gripper_pose: np.ndarray,
                    gripper_joint_positions: np.ndarray,
                    gripper_touch_forces: np.ndarray,
                    task_low_dim_state: np.ndarray):
    ```

- Observations for an experiment can be configured using `ObservationConfig`.
    ```python 
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.left_shoulder_camera.rgb = True
    obs_config.right_shoulder_camera.rgb = True
    ```
- `ArmActionMode.ABS_JOINT_VELOCITY` helps specify the what the action will mean. More action Modes available [here](https://github.com/stepjam/RLBench/blob/9f3bf886ce5d59d2eff8d9ec93ac49cb2b816b2f/rlbench/action_modes.py#L7). This basically means that when U provide an action it will expect it to mean what is selected in ArmActionMode.
    ```python
    from rlbench.environment import Environment
    from rlbench.action_modes import ArmActionMode, ActionMode
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
        action_mode, DATASET, obs_config, False)
    # for obs_config refer to Tasks 
    env.launch()
    ```

## Task
- Every `Environment` consists of tasks which can use Different Robots to do different things. 
    ```python
    from rlbench.tasks import ReachTarget
    task = env.get_task(ReachTarget)
    ```
- Every `Task` consists of a method called `get_demos`. The `get_demos` function will perform the `Task` and get the obs
ervations which achieve the task.

    ```python
    demos = task.get_demos(2, live_demos=True)  # -> List[List[Observation]] -> List[Observation] represents a individual Demonstration with every item in that List represnts a step in that Demonstration
    ```

## Agent Setup 
- Use the `models.Agent` to create new agents that will implement data and observations from the environments.
    ```python
    class LearningAgent():
        def __init__(self):
            self.learning_rate = None
            self.neural_network = None
            self.optimizer = None
            self.loss_function = None
            self.training_data = None
            self.logger = None
            self.input_state = None
            self.output_action = None

        def injest_demonstrations(self,demos:List[List[Observation]]):
            raise NotImplementedError()

        
        def train_agent(self,epochs:int):
            raise NotImplementedError()
        
        def predict_action(self, demonstration_episode:List[Observation]):
            """
            This should Use model.eval() in Pytoch to do prediction for an action
            This is ment for using saved model
            """
            raise NotImplementedError()

        def act(self,state:List[Observation]):
            """
            This will be used by the RL agents and Learn from feadback from the environment. 
            This will let pytorch hold gradients when running the network. 
            """
            raise NotImplementedError()

        def save_model(self,file_path):
            if not self.neural_network:
                return
            self.neural_network.to('cpu')
            torch.save(self.neural_network.state_dict(), file_path)

        def load_model(self,file_path):
            if not self.neural_network:
                return
            # $ this will load a model from file path.
            self.neural_network.load_state_dict(torch.load(file_path))
    ```

- The `deep_learning_rl.py` file contains methods that can directly get demonstrations and for immitation learning. The file also contains a method(`simulate_trained_agent`) which takes a learned `Agent` and runs it in simulation without headless mode for quick evaluation of model.
- Example of Dumb Agent Learning a Stupid Policy(state->neural net->action) from demostration can be found in the `models.ImmitationLearning`. 

## Observations / Available Data

`Observation.joint_positions`: The intrinsic position of a joint. This is a one-dimensional value: if the joint is revolute, the rotation angle is returned, if the joint is prismatic, the translation amount is returned, etc. 

# Running the Code 

1. Prerequisites PyRep and RLBench
    1. Run `sh setup_pyrep_ubuntu18.04.sh` on Ubuntu18.04 for installing PyRep + RLBench + Downloading the Dataset
2. Running Immitation Learning Agent Training / Simulation:
    ```sh
    python run_learning.py
    ```
3. Running Dataset Creator for RL bench to create dataset in `/tmp/rlbench_datasets`:
    ```
    python RLBench/tools/dataset_generator.py --tasks reach_target --episodes_per_task 2 --processes 2
    ```

# TODO 
- [ ] Document Agents
- [ ] Document Observations/Available data
- [x] Add Complete Package Install Scripts
- [ ] Add More deep learning approaches for task based training for agents. 
