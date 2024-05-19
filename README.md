# CS258_FinalProject
# Network Resource Spectrum Allocation

## Project Overview

This project focuses on efficiently solving the Routing and Spectrum Allocation (RSA) problem in optical communication networks. The goal is to maximize resource utilization, improve the quality of service, and reduce operational costs. The project employs both reinforcement learning (RL) algorithms and heuristic methods to achieve these objectives. Project Description is availabe at https://github.com/sjsu-interconnect/cs258/blob/main/projects/rsa.md


## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/rajivkarthikreddy/CS258_FinalProject.git
    cd CS258_FinalProject
    ```


2. **Install Required Libraries**:
    ```bash
    pip install "ray[rllib]"
    pip install gymnasium
    pip install networkx
    ```

## Input Data

We are generating 100 requests each for two cases:
- **Case I**: Randomly selected source and target nodes.
- **Case II**: Fixed source and target nodes ('San Diego Supercomputer Center' to 'Jon Von Neumann Center, Princeton, NJ').

These requests are stored in `request_c1` and `request_c2` files within the same folder. These input files are used for both the RL algorithms and the First-Fit Shortest Path heuristic.

## Running the Code

### Generating Requests

To generate requests and store them:
```bash
python src/generate_requests.py
```


## Training and Evaluation
To run the RL algorithms (DQN and PPO) and evaluate their performance:

```bash
python src/dqn_random_run.py
python src/dqn_fixed_run.py
python src/ppo_random_run.py
python src/ppo_fixed_run.py
```

## First-Fit Shortest Path Heuristic
To run the heuristic algorithm and evaluate its performance:

```bash
python src/rsa_shortest.py
```

## Code Explanation
### net_env_fixed.py/net_env_random.py
Defines the NetworkEnv class which represents the environment for the RL algorithms. It includes:

State space represents the network's nodes, edges, and spectrum slots.
Action space representing possible actions (edge and color allocation).
Reward calculation based on network utilization.
### dqn_fixed_run.py and other similar runs
Configures and trains the RL algorithms (DQN and PPO). It also logs network utilization and plots the performance over time.

### generate_requests.py
Generates Case I and II requests and stores them in the same folder.

### CustomMetricsCallback
A custom callback to log and calculate custom metrics like network utilization during training.

## Results
Mean Reward and Network Utilization per episode are plotted for the RL algorithms. Network Utilization over T requests is plotted for the heuristic First-Fit algorithm.


## Results
Mean Reward and Network Utilization per episode are plotted for the RL algorithms. Network Utilization over T requests is plotted for the heuristic First-Fit algorithm.

### DQN Mean Reward for Fixed Nodes
![dqn_fixed_reward](https://github.com/rajivkarthikreddy/CS258_FinalProject/assets/170271928/9844ee17-5aca-4174-ba4f-4528fcd51032)

### DQN Mean Reward for Random Nodes
<img width="986" alt="dqn_random_reward" src="https://github.com/rajivkarthikreddy/CS258_FinalProject/assets/170271928/5b00b858-a2fd-4310-aefb-4f5f8173f069">

### PPO Mean Reward for Fixed Nodes
![ppo_fixed_reward](https://github.com/rajivkarthikreddy/CS258_FinalProject/assets/170271928/67c02d53-b151-4c81-9827-067c534ded84)

### PPO Mean Reward for Random Nodes
![ppo_random_reward](https://github.com/rajivkarthikreddy/CS258_FinalProject/assets/170271928/07ba0677-7a33-41fc-841c-f5605955f86a)

### DQN Network Utilization for Fixed Nodes
![dqn_fixed_ut](https://github.com/rajivkarthikreddy/CS258_FinalProject/assets/170271928/c4b9d3fa-90f8-47ca-bb56-43162707f068)

### DQN Network Utilization for Random Nodes
![dqn_random_uti](https://github.com/rajivkarthikreddy/CS258_FinalProject/assets/170271928/c1236ce7-7d0a-4be4-bba2-40e2ab7fb558)

### PPO Network Utilization for Fixed Nodes
![ppo_fixed_uti](https://github.com/rajivkarthikreddy/CS258_FinalProject/assets/170271928/0a495648-4260-4a2f-8068-e6684a379288)

### PPO Network Utilization for Random Nodes
![ppo_random_uti](https://github.com/rajivkarthikreddy/CS258_FinalProject/assets/170271928/e6bfec8a-af20-4603-a0ba-965eabd3fa73)

### First-Fit Utilization Curve- Case 1
![FirstFit_ut1](https://github.com/rajivkarthikreddy/CS258_FinalProject/assets/170271928/bf3eba0b-dbf3-4ec2-a197-d4818536a2f9)

### First-Fit Utilization Curve- Case 2
![FirstFit_ut2](https://github.com/rajivkarthikreddy/CS258_FinalProject/assets/170271928/5bf3622f-bfd9-422f-b6c9-ead7248d8461)



## Conclusion and Future Work
The current results show a need to refine the reward function and tune hyperparameters to improve performance. Future work will explore dynamic action spaces and other improvements to optimize resource allocation.
