import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkEnv(gym.Env):
    def __init__(self, config=None):
        super(NetworkEnv, self).__init__()

        self.link_capacity = 10

        # Read the network graph from a GML file
        self.graph = nx.read_gml('nsfnet.gml')
        
        self.num_nodes = len(self.graph.nodes)
        self.num_edges = len(self.graph.edges)
        
        # Define the observation space: (nodes, edges, link_capacity)
        self.observation_space = spaces.Box(
            low=0,
            high=20,
            shape=(self.num_nodes, self.num_edges, self.link_capacity),
            dtype=np.int32
        )
        
        # Define the action space: flattened (edge * color)
        self.action_space = spaces.Discrete(self.num_edges * self.link_capacity)
        
        # Initialize state and total utilization only once
        self.state = np.zeros((self.num_nodes, self.num_edges, self.link_capacity), dtype=np.int32)
        self.total_utilization = np.zeros((self.num_edges, self.link_capacity))
        
        self.max_steps = 200  # Allow multiple steps per episode
        
        # Load pre-generated requests from a file
        with open ('request_c1', 'rb') as fp:
            self.requests = pickle.load(fp)
        self.current_request_idx = 0  # Track the current request index
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Check if all requests have been processed
        if self.current_request_idx >= len(self.requests):
            self.current_request_idx = 0  # Reset to first request if all requests have been processed
            self.state = np.zeros((self.num_nodes, self.num_edges, self.link_capacity), dtype=np.int32)
            self.total_utilization = np.zeros((self.num_edges, self.link_capacity))
        
        # Set the current request
        self.current_request = self.requests[self.current_request_idx]
        self.current_request_idx += 1
        
        # Decrement the holding time for all edges by 1
        pre_decrement_state = self.state.copy()
        self.state = np.maximum(self.state - 1, 0)
        
        # Update utilization for slots where holding time has become zero
        for node in range(self.num_nodes):
            for edge in range(self.num_edges):
                for color in range(self.link_capacity):
                    if pre_decrement_state[node, edge, color] == 1 and self.state[node, edge, color] == 0:
                        self.total_utilization[edge, color] = 0


        # Do not reset state and total_utilization
        self.previous_edge = None
        self.previous_color = None
        self.steps_taken = 0  # Reset steps taken for the new episode
        
        return self.state, {}


    def step(self, action):
        edge_idx, color = self._unflatten_action(action)
        edge = list(self.graph.edges)[edge_idx]
        
        done = False
        
        source, target, holding_time = self.current_request
        
        
        
        # Check if the selected edge is connected
        if edge in self.graph.edges:
            if self.previous_edge is None or (color == self.previous_color and self.state[source, edge_idx, color] == 0 and 
                                              (self.previous_edge[0] in edge or self.previous_edge[1] in edge)):
                self.state[source, edge_idx, color] = holding_time
                self.previous_edge = edge
                self.previous_color = color
                # Update total utilization for the edge
                self.total_utilization[edge_idx, color] = 1
                if self.previous_edge[1] == target or self.previous_edge[0] == target:
                    done = True
                    reward = 20  # Assign a high reward for reaching the target node
                else:
                    reward = 5
            else:
                reward = -1
        else:
            reward = -1
        
        self.steps_taken += 1  # Increment steps taken

        if self.steps_taken >= self.max_steps:  # Check if the max steps are reached
            done = True
        
        # Calculate average utilization U(e) for each edge
        average_utilization = np.sum(self.total_utilization, axis=1) / self.link_capacity
        
        # Calculate network-wide utilization U
        network_utilization = np.mean(average_utilization)

        reward += network_utilization  # Include network utilization in the reward
        
        # logger.info(f"Action: {action}, Unflattened: (edge_idx={edge_idx}, color={color}), Reward: {reward}, Done: {done}")
        logger.info(f"Action: {action}, Unflattened: (edge_idx={edge_idx}, color={color}), Reward: {reward}, Done: {done}, Network Utilization: {network_utilization}")

        # return self.state, reward, done, False, info
        return self.state, reward, done, False, {'network_utilization': network_utilization}

    def _flatten_action(self, edge_idx, color):
        return edge_idx * self.link_capacity + color

    def _unflatten_action(self, action):
        edge_idx = action // self.link_capacity
        color = action % self.link_capacity
        return edge_idx, color