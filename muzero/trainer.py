import collections
import copy
import datetime
import os
import tempfile
from typing import Callable, Tuple

import ray
from ray.actor import ActorHandle
from ray.rllib.agents.dqn.learner_thread import LearnerThread
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, STEPS_TRAINED_COUNTER
from ray.rllib.execution.concurrency_ops import Concurrently, Enqueue, Dequeue
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.actors import create_colocated
from ray.tune import Trainable
from ray.tune.logger import Logger, UnifiedLogger
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.util.iter import LocalIterator
import tensorflow as tf

from muzero.replay_buffer import ReplayActor, PRIO_WEIGHTS


# Update worker weights as they finish generating experiences.
class BroadcastUpdateLearnerWeights:
    """
    Adapted from https://github.com/ray-project/ray/blob/5acd3e66ddc1d7a1af6590567fcc3df95169d8a2/rllib/agents/impala/impala.py#L177
    """
    def __init__(self, learner_thread, workers, broadcast_interval):
        self.learner_thread = learner_thread
        self.steps_since_broadcast = collections.defaultdict(int)
        self.broadcast_interval = broadcast_interval
        self.workers = workers
        self.weights = workers.local_worker().get_weights()

    def __call__(self, item):
        actor, batch = item
        self.steps_since_broadcast[actor] += 1
        if (self.steps_since_broadcast[actor] >= self.broadcast_interval
                and self.learner_thread.weights_updated):
            self.weights = ray.put(self.workers.local_worker().get_weights())
            self.steps_since_broadcast[actor] = 0
            self.learner_thread.weights_updated = False
            # Update metrics.
            metrics = LocalIterator.get_metrics()
            metrics.counters["num_weight_broadcasts"] += 1
        global_vars = {"timestep": LocalIterator.get_metrics().counters[STEPS_SAMPLED_COUNTER]}
        actor.set_weights.remote(self.weights, global_vars)
        # Also update global vars of the local worker.
        self.workers.local_worker().set_global_vars(global_vars)


class MuZeroTrainer(Trainable):
    def __init__(self,
                 config: dict = None,
                 env: str = None,
                 logger_creator: Callable[[], Logger] = None):
        """
        From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/agents/trainer.py
        """
        self._env_id = env or config['env']

        if logger_creator is None:
            timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
            logdir_prefix = "{}_{}_{}".format(self._name, self._env_id,
                                              timestr)

            def default_logger_creator(config):
                """Creates a Unified logger with a default logdir prefix
                containing the agent name and the env id
                """
                if not os.path.exists(DEFAULT_RESULTS_DIR):
                    os.makedirs(DEFAULT_RESULTS_DIR)
                logdir = tempfile.mkdtemp(
                    prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR)
                return UnifiedLogger(config, logdir, loggers=None)
            
            logger_creator = default_logger_creator

        super().__init__(config, logger_creator)

    def _calculate_rr_weights(self):
        if not self.config["training_intensity"]:
            return [1, 1]
        # e.g., 32 / 4 -> native ratio of 8.0
        native_ratio = (
            self.config["train_batch_size"] / self.config["rollout_fragment_length"])
        # Training intensity is specified in terms of
        # (steps_replayed / steps_sampled), so adjust for the native ratio.
        weights = [1, self.config["training_intensity"] / native_ratio]
        return weights

    def _build_global_op(self, workers):
        # Create a number of replay buffer actors.
        num_replay_buffer_shards = self.config["optimizer"]["num_replay_buffer_shards"]

        tensor_spec = (
            tf.TensorSpec((96, 96, 4), tf.float32, SampleBatch.CUR_OBS),
            tf.TensorSpec((1,), tf.int32, SampleBatch.EPS_ID),
            tf.TensorSpec((1,), tf.int32, SampleBatch.UNROLL_ID),
            tf.TensorSpec((1,), tf.int32, SampleBatch.ACTIONS),
            tf.TensorSpec((4,), tf.float32, SampleBatch.ACTION_DIST_INPUTS),
            tf.TensorSpec((1,), tf.float32, SampleBatch.ACTION_PROB),
            tf.TensorSpec((1,), tf.float32, SampleBatch.ACTION_LOGP),
            tf.TensorSpec((1,), tf.bool, SampleBatch.DONES),
            tf.TensorSpec((1,), tf.float32, SampleBatch.REWARDS),
            tf.TensorSpec((5, 4), tf.float32, 'rollout_policies'),
            tf.TensorSpec((5,), tf.float32, 'rollout_rewards'),
            tf.TensorSpec((5,), tf.float32, 'rollout_values'),
        )

        # There's no way to control where the ReplayActors are located in a multi-node setup, so
        # this function spins up more and more actors until it finds enough
        # which are on the local host.
        replay_actors = create_colocated(ReplayActor, [
            tensor_spec,
            num_replay_buffer_shards,
            self.config["learning_starts"],
            self.config["buffer_size"],
            self.config["train_batch_size"],
            self.config["prioritized_replay_alpha"],
            self.config["prioritized_replay_beta"],
            self.config["prioritized_replay_eps"],
            self.config["multiagent"]["replay_mode"],
            self.config["replay_sequence_length"],
        ], num_replay_buffer_shards)
        
        # Start the learner thread on the local host (where the ReplayActors are located).
        learner_thread = LearnerThread(
            workers.local_worker())
            #minibatch_buffer_size = config['minibatch_buffer_size'],
            #num_sgd_iter=config["num_sgd_iter"],
            #learner_queue_size=config["learner_queue_size"],
            #learner_queue_timeout=config["learner_queue_timeout"])
        learner_thread.start()

        # Update experience priorities post learning.
        def update_prio_and_stats(item: Tuple[ActorHandle, dict, int]) -> None:
            actor, prio_dict, count = item
            #print('actor', type(actor), actor)
            #print('prio_dict:', type(prio_dict), prio_dict)
            #print('count:', type(count), count)
            actor.update_priorities.remote(prio_dict)
            metrics = LocalIterator.get_metrics()
            # Manually update the steps trained counter since the learner thread
            # is executing outside the pipeline.
            metrics.counters[STEPS_TRAINED_COUNTER] += count
            metrics.timers["learner_dequeue"] = learner_thread.queue_timer
            metrics.timers["learner_grad"] = learner_thread.grad_timer
            metrics.timers["learner_overall"] = learner_thread.overall_timer
            return metrics
        
        # We execute the following steps concurrently:
        # (1) Generate rollouts and store them in our replay buffer actors. Update
        # the weights of the worker that generated the batch.
        rollouts = ParallelRollouts(
            workers,
            mode="async",
            num_async=self.config['max_sample_requests_in_flight_per_worker'])
        store_op = rollouts.for_each(StoreToReplayBuffer(actors=replay_actors))

        if workers.remote_workers():
            store_op = store_op.zip_with_source_actor() \
                .for_each(BroadcastUpdateLearnerWeights(
                    learner_thread,
                    workers,
                    broadcast_interval=self.config["broadcast_interval"]))

        # (2) Read experiences from the replay buffer actors and send to the
        # learner thread via its in-queue.
        replay_op = Replay(actors=replay_actors, num_async=4) \
            .zip_with_source_actor() \
            .for_each(Enqueue(learner_thread.inqueue))

            #.filter(WaitUntilTimestepsElapsed(config["learning_starts"])) \
        # (3) Get priorities back from learner thread and apply them to the
        # replay buffer actors.
        update_op = Dequeue(
            learner_thread.outqueue, check=learner_thread.is_alive) \
            .for_each(update_prio_and_stats)
        
        if self.config["training_intensity"]:
            # Execute (1), (2) with a fixed intensity ratio.
            rr_weights = self._calculate_rr_weights() + ["*"]
            merged_op = Concurrently(
                [store_op, replay_op, update_op],
                mode="round_robin",
                output_indexes=[2],
                round_robin_weights=rr_weights)
        else:
            # Execute (1), (2), (3) asynchronously as fast as possible. Only output
            # items from (3) since metrics aren't available before then.
            merged_op = Concurrently(
                [store_op, replay_op, update_op], mode="async", output_indexes=[2])

        def add_muzero_metrics(result):
            replay_stats = ray.get(replay_actors[0].stats.remote(
                    self.config["optimizer"].get("debug")))
            result["info"].update({
                "learner_queue": learner_thread.learner_queue_size.stats(),
                "learner": copy.deepcopy(learner_thread.stats),
                "replay_shard_0": replay_stats,
            })
            return result

        return StandardMetricsReporting(merged_op, workers, self.config) \
            .for_each(add_muzero_metrics)

    def setup(self, config):
        """
        Adapted from https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/agents/trainer.py
        """
        self.config = config

        env = self._env_id
        if env:
            config["env"] = env
            # An already registered env.
            if _global_registry.contains(ENV_CREATOR, env):
                self.env_creator = _global_registry.get(ENV_CREATOR, env)
            # Try gym.
            else:
                import gym  # soft dependency
                self.env_creator = lambda env_config: gym.make(env)
        else:
            self.env_creator = lambda env_config: None

        if config['framework'] == 'tf' or config['framework'] == 'tfe':
            from muzero.muzero_tf_policy import MuZeroTFPolicy
            self._policy = MuZeroTFPolicy
        else:
            raise NotImplementedError(
                f'Framework "{config["framework"]} not supported')

        self.workers = WorkerSet(
            self.env_creator,
            self._policy,
            self.config,
            num_workers=self.config['num_workers'],
            logdir=self.logdir)

        self._global_op = self._build_global_op(self.workers)

    def step(self):
        return next(self._global_op)