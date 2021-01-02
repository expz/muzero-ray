# From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/evaluation/sampler.py
# and https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/evaluation/rollout_metrics.py

import collections

class PerfStats:
    """Sampler perf stats that will be included in rollout metrics."""

    def __init__(self):
        self.iters = 0
        self.env_wait_time = 0.0
        self.raw_obs_processing_time = 0.0
        self.inference_time = 0.0
        self.action_processing_time = 0.0

    def get(self):
        # Mean multiplicator (1000 = ms -> sec).
        factor = 1000 / self.iters
        return {
            # Waiting for environment (during poll).
            "mean_env_wait_ms": self.env_wait_time * factor,
            # Raw observation preprocessing.
            "mean_raw_obs_processing_ms": self.raw_obs_processing_time *
            factor,
            # Computing actions through policy.
            "mean_inference_ms": self.inference_time * factor,
            # Processing actions (to be sent to env, e.g. clipping).
            "mean_action_processing_ms": self.action_processing_time * factor,
        }

# Define this in its own file, see #5125
RolloutMetrics = collections.namedtuple("RolloutMetrics", [
    "episode_length", "episode_reward", "agent_rewards", "custom_metrics",
    "perf_stats", "hist_data"
])
RolloutMetrics.__new__.__defaults__ = (0, 0, {}, {}, {}, {})
