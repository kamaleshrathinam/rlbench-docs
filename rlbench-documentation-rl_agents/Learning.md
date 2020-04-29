# Agent Design Intuition 

## Input Param Intuition
- To build truely Model Free RL for the current agent we will need following information on which agent learns :
    - **RGB values of the image showcasing robot from multiple positions at each timestep**:
        - Need this because we want the robot to learn to corelate from visual features with its own joint positions. 
            - `Observation.left_shoulder_rgb: np.ndarray`
            - `Observation.left_shoulder_depth: np.ndarray`
            - `Observation.left_shoulder_mask: np.ndarray`
            - `Observation.right_shoulder_rgb: np.ndarray`
            - `Observation.right_shoulder_depth: np.ndarray`
            - `Observation.right_shoulder_mask: np.ndarray`
    - **Joint positions**:
        - This is basic. `Observation.joint_position()`
    - **Final Position Coords**: 
        - This is because the agent should learn to make decisions knowing whats the final goal positions. `task.target.get_position()`

## NN Architecture Intuition 

### Immitation Learning NN 
todo 

### Immitatioon Learning as base with RL
todo

## Experimantation Metrics

1. Number of Steps to Goal Position post Learning. 
2. Rewards in RL setup
