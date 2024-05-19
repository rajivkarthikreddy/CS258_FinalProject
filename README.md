# CS258_FinalProject
# Optical Network Resource Allocation

## Project Overview

This project focuses on efficiently solving the Routing and Spectrum Allocation (RSA) problem in optical communication networks. The goal is to maximize resource utilization, improve the quality of service, and reduce operational costs. The project employs both reinforcement learning (RL) algorithms and heuristic methods to achieve these objectives. Project Description is availabe at https://github.com/sjsu-interconnect/cs258/blob/main/projects/rsa.md


## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/rajivkarthikreddy/CS258_FinalProject.git
    cd CS258_FinalProject
    ```


2. **Install Ray**:
    ```bash
    pip install ray[rllib]
    ```

## Input Data

We are generating 100 requests each for two cases:
- **Case I**: Randomly selected source and target nodes.
- **Case II**: Fixed source and target nodes ('San Diego Supercomputer Center' to 'Jon Von Neumann Center, Princeton, NJ').

These requests are stored in `request_c1` and `request_c2` files within the `input_data` folder. These input files are used for both the RL algorithms and the First-Fit Shortest Path heuristic.

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

State space representing the network's nodes, edges, and spectrum slots.
Action space representing possible actions (edge and color allocation).
Reward calculation based on network utilization.
### dqn_fixed_run.py and other similar runs
Configures and trains the RL algorithms (DQN and PPO). It also logs network utilization and plots the performance over time.

### generate_requests.py
Generates requests for Case I and Case II and stores them in the same folder.

###CustomMetricsCallback
A custom callback to log and calculate custom metrics like network utilization during training.

## Results
The results of the RL algorithms and heuristic methods will be logged and plotted, showing the trends in network utilization and blocked requests over time.

## Conclusion and Future Work
The current results show a need to refine the reward function and tune hyperparameters to improve performance. Future work will explore dynamic action spaces and other improvements to optimize resource allocation.
