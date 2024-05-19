from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray.tune.registry import register_env
import warnings
import logging
import time
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import the custom environment
from net_env_random import NetworkEnv
from custom_metrics_callback import CustomMetricsCallback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
start_time = time.time() # Track the start time of the run
ray.init()

# Register the custom environment
def env_creator(env_config):
    return NetworkEnv()

register_env('netenv-v0', env_creator)

config = (PPOConfig()
          .training(model={"fcnet_hiddens": [512, 512], "conv_filters": [[16, [3, 3], 2], [32, [3, 3], 2], [64, [3, 3], 2]]}, 
                   lambda_=0.95, 
                   clip_param=0.2, 
                   kl_target=0.01, 
                   kl_coeff=1.0)
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

# Save the trained models
algo.save('ppo_network_env')

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

# Load the trained models for evaluation
algo = config.build()
algo.restore('ppo_network_env')

# Test the trained agent
env = NetworkEnv()
state,_ = env.reset()
for i in range(100):
    action = algo.compute_single_action(state)
    state, reward, done,_, info = env.step(action)
    logger.info(f"Step {i}: Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
    if done:
        break

print("Test completed.")