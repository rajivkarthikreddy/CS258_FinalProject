from ray.rllib.algorithms.dqn import DQNConfig
import ray
from ray.tune.registry import register_env
import warnings
import logging
import matplotlib.pyplot as plt
import time
from custom_metrics_callback import CustomMetricsCallback

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import the custom environment
from net_env_fixed import NetworkEnv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
start_time = time.time() # Track the start time of the run

# graph = nx.read_gml('nsfnet.gml')

# num_nodes = len(graph.nodes)

# Initialize Ray with adequate object store memory
ray.init(object_store_memory=78643200)  # Setting to minimum allowed (75MB)

# Register the custom environment
def env_creator(env_config):
    return NetworkEnv()

register_env('netenv-v0', env_creator)

# Define convolutional filters for the model
conv_filters = [
    [512, [11, 13], 11]
]

# Configure the RL algorithm
config = (DQNConfig()
          .training(gamma=0.999, lr=0.001, model={"conv_filters": conv_filters, "fcnet_hiddens": [512, 512]})
          .environment(env='netenv-v0')
          .resources(num_gpus=0)
          .rollouts(num_rollout_workers=0, num_envs_per_worker=1)
          .evaluation(evaluation_interval=1)  # Add evaluation to ensure metrics are logged
          .callbacks(CustomMetricsCallback)
        )

# Build the algorithm
algo = config.build()

# List to store network utilizations for plotting
network_utilizations = []
num_episodes = 100

# Train the agent for a specified number of episodes
for episode in range(num_episodes):
    result = algo.train()
    
    custom_metrics = result.get('custom_metrics', {})
    if custom_metrics:
        logger.info(f"Custom Metrics: {custom_metrics}")
        network_utilization = custom_metrics.get('network_utilization_mean', None)
    
        if network_utilization is not None:
            network_utilizations.append(network_utilization)
            logger.info(f"Episode {episode + 1}/{num_episodes}: Network Utilization {network_utilization}")
        else:
            logger.warning(f"Episode {episode + 1}/{num_episodes}: No network utilization found in result")
    else:
        logger.warning(f"Episode {episode + 1}/{num_episodes}: No custom metrics found in result")

   
    
# Check if network_utilizations has been populated
if len(network_utilizations) == 0:
    logger.error("No network utilizations were recorded. Please check the training process.")

# Save the trained model
algo.save('dqn_network_env')

# logger.info("################################");
# logger.info(network_utilizations)

# Plot the network utilization vs. episode
if network_utilizations:
    plt.plot(range(1, len(network_utilizations) + 1), network_utilizations)
    plt.xlabel('Episode')
    plt.ylabel('Network Utilization')
    plt.title('Network Utilization vs. Episode')
    plt.grid(True)
    plt.show()
else:
    logger.error("Network utilization data is empty. Cannot plot.")


end_time = time.time() 
# Track the end time of the run
elapsed_time = end_time - start_time

logger.info(f"Run started at {start_time} and ended at {end_time}. Elapsed time: {elapsed_time} seconds")

# Test the trained agent
env = NetworkEnv()
state, _ = env.reset()  # Unpack the observation from the tuple
done = False
for i in range(100):
    # Ensure the state is passed as a dictionary
    action = algo.compute_single_action(state)
    state, reward, done, _, info = env.step(action)  # Unpack the observation from the tuple
    logger.info(f"Step {i}: Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
    if done:
        break
print("Test completed.")