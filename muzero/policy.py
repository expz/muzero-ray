from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from muzero.sample_batch import SampleBatch


AgentID = str
TensorType = Any

LEARNER_STATS_KEY = "learner_stats"
STEPS_SAMPLED_COUNTER = "num_steps_sampled"
STEPS_TRAINED_COUNTER = "num_steps_trained"

GRAD_WAIT_TIMER = "grad_wait"
SAMPLE_TIMER = "sample"

# Instant metrics (keys for metrics.info).
LEARNER_INFO = "learner"

VALUE_TARGETS = "value_targets"


class Policy:

    def __init__(
            self,
            obs_space,
            action_space,
            config):
        self.observation_space = obs_space
        self.action_space = action_space
        self.config = config

    def postprocess_trajectory(
            self,
            sample_batch: SampleBatch,
            other_agent_batches: Optional[Dict[AgentID, Tuple[Policy, SampleBatch]]] = None,
            episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:
        raise NotImplementedError()

    def compute_actions(self,
                        obs_batch: TensorType,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        raise NotImplementedError()

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        grads_and_vars, stats = self.compute_gradients(samples)
        self.apply_gradients(grads_and_vars)
        return stats

    def compute_gradients(self, samples):
        raise NotImplementedError()

    def apply_gradients(self, grads_and_vars):
        raise NotImplementedError()


class TFPolicy(Policy):
    pass


class TorchPolicy(Policy):
    pass


def _convert_to_tf(x):
    """
    From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/policy/eager_tf_policy.py
    """
    if isinstance(x, SampleBatch):
        x = {k: v for k, v in x.items() if k != SampleBatch.INFOS}
        return tf.nest.map_structure(_convert_to_tf, x)
    if isinstance(x, Policy):
        return x

    if x is not None:
        x = tf.nest.map_structure(
            lambda f: tf.convert_to_tensor(f) if f is not None else None, x)
    return x


def traced_eager_policy(eager_policy_cls):
    """
    Wrapper that enables tracing for all eager policy methods.
    This is enabled by the --trace / "eager_tracing" config.

    From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/policy/eager_tf_policy.py
    """

    class TracedEagerPolicy(eager_policy_cls):
        def __init__(self, *args, **kwargs):
            self._traced_learn_on_batch = None
            self._traced_compute_actions = None
            self._traced_compute_gradients = None
            self._traced_apply_gradients = None
            super(TracedEagerPolicy, self).__init__(*args, **kwargs)

        def learn_on_batch(self, samples):

            if self._traced_learn_on_batch is None:
                self._traced_learn_on_batch = tf.function(
                    super(TracedEagerPolicy, self).learn_on_batch,
                    autograph=False)

            return self._traced_learn_on_batch(samples)

        def compute_actions(self,
                            obs_batch,
                            state_batches=None,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            info_batch=None,
                            episodes=None,
                            explore=None,
                            timestep=None,
                            **kwargs):

            obs_batch = tf.convert_to_tensor(obs_batch)
            state_batches = _convert_to_tf(state_batches)
            prev_action_batch = _convert_to_tf(prev_action_batch)
            prev_reward_batch = _convert_to_tf(prev_reward_batch)

            if self._traced_compute_actions is None:
                self._traced_compute_actions = tf.function(
                    super(TracedEagerPolicy, self).compute_actions,
                    autograph=False)

            return self._traced_compute_actions(
                obs_batch, state_batches, prev_action_batch, prev_reward_batch,
                info_batch, episodes, explore, timestep, **kwargs)

        def compute_gradients(self, samples):

            if self._traced_compute_gradients is None:
                self._traced_compute_gradients = tf.function(
                    super(TracedEagerPolicy, self).compute_gradients,
                    autograph=False)

            return self._traced_compute_gradients(samples)

        def apply_gradients(self, grads):

            if self._traced_apply_gradients is None:
                self._traced_apply_gradients = tf.function(
                    super(TracedEagerPolicy, self).apply_gradients,
                    autograph=False)

            return self._traced_apply_gradients(grads)

    TracedEagerPolicy.__name__ = eager_policy_cls.__name__
    TracedEagerPolicy.__qualname__ = eager_policy_cls.__qualname__
    return TracedEagerPolicy
