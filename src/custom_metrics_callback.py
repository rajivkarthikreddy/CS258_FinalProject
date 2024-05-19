import logging
from ray.rllib.algorithms.callbacks import DefaultCallbacks

logger = logging.getLogger(__name__)

class CustomMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        env = base_env.get_sub_environments()[0]  # Get the underlying environment
        info = episode.last_info_for()  # Access the info dictionary
        if 'network_utilization' in info:
            network_utilization = info['network_utilization']
            episode.custom_metrics['network_utilization'] = network_utilization
            logger.info(f"Episode {episode.episode_id} ended with network_utilization: {network_utilization}")
