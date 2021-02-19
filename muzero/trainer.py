from __future__ import annotations

import collections
import copy
from datetime import datetime
import os
import pickle
import tempfile
import time
from typing import Callable, Tuple

import numpy as np
import ray
from ray.actor import ActorHandle
from ray.tune import Trainable
from ray.tune.logger import Logger, UnifiedLogger
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.tune.resources import Resources
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.util.iter import LocalIterator
import tensorflow as tf

from muzero.global_vars import GlobalVars
from muzero.learner_thread import LearnerThread
from muzero.muzero import ATARI_DEFAULT_CONFIG, BOARD_DEFAULT_CONFIG
from muzero.muzero_tf_policy import MuZeroTFPolicy
from muzero.ops.concurrency_ops import Concurrently, Enqueue, Dequeue
from muzero.ops.metric_ops import StandardMetricsReporting
from muzero.ops.replay_ops import Replay, StoreToReplayBuffer, CalculatePriorities
from muzero.ops.rollout_ops import ParallelRollouts
from muzero.policy import STEPS_SAMPLED_COUNTER, STEPS_TRAINED_COUNTER
from muzero.replay_buffer import ReplayActor, PRIO_WEIGHTS
from muzero.sample_batch import SampleBatch
from muzero.structure_list import ArraySpec
from muzero.util import create_colocated
from muzero.worker_set import WorkerSet


# Update worker weights as they finish generating experiences.
class BroadcastUpdateLearnerWeights:
    """
    Adapted from https://github.com/ray-project/ray/blob/5acd3e66ddc1d7a1af6590567fcc3df95169d8a2/rllib/agents/impala/impala.py#L177
    """
    def __init__(
        self,
        learner_thread: LearnerThread,
        workers: WorkerSet,
        broadcast_interval: int):

        self.learner_thread = learner_thread
        self.steps_since_broadcast = collections.defaultdict(int)
        self.broadcast_interval = broadcast_interval
        self.workers = workers
        self.weights = workers.local_worker().get_weights()

    def __call__(self, item):
        actor, batch = item
        timestep = LocalIterator.get_metrics().counters[STEPS_SAMPLED_COUNTER]
        self.steps_since_broadcast[actor] += 1
        if self.steps_since_broadcast[actor] >= self.broadcast_interval:
            if self.learner_thread.weights_updated:
                self.weights = ray.put(self.workers.local_worker().get_weights())
                self.steps_since_broadcast[actor] = 0
                self.learner_thread.weights_updated = False
            # Update metrics.
            metrics = LocalIterator.get_metrics()
            metrics.counters["num_weight_broadcasts"] += 1
            actor.set_weights.remote(self.weights, timestep)
        # Also update global vars of the local worker.
        self.workers.local_worker().set_timestep(timestep)


class MuZeroTrainer(Trainable):
    def __init__(self,
                 config: dict = None,
                 env: str = None,
                 logger_creator: Callable[[], Logger] = None):
        """
        From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/agents/trainer.py
        """
        self._env_id = env or config['env']
        self._name = 'MuZeroTrainer'
        self._last_mem_reset = 0
        self._global_vars = GlobalVars.remote()

        self.learner_thread = None
        self.replay_actors = None
        self.store_op = None
        self.replay_op = None
        self.update_op = None

        tf.get_logger().setLevel(config['log_level'])

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

        n_channels = self.config['n_channels']
        frame_shape = self.config['frame_shape']
        loss_steps = self.config['loss_steps']
        action_count = self.env_creator({}).action_space.n
        tensor_spec = (
            ArraySpec(frame_shape + (n_channels,), np.float32, SampleBatch.CUR_OBS),
            ArraySpec((1,), np.int32, SampleBatch.EPS_ID),
            ArraySpec((1,), np.int32, SampleBatch.UNROLL_ID),
            ArraySpec((loss_steps,), np.int32, SampleBatch.ACTIONS),
            ArraySpec((action_count,), np.float32, 'action_dist_probs'),
            ArraySpec((1,), np.float32, SampleBatch.ACTION_PROB),
            ArraySpec((1,), np.float32, SampleBatch.ACTION_LOGP),
            ArraySpec((1,), np.bool, SampleBatch.DONES),
            ArraySpec((1,), np.float32, SampleBatch.REWARDS),
            ArraySpec((loss_steps, action_count), np.float32, 'rollout_policies'),
            ArraySpec((loss_steps,), np.float32, 'rollout_rewards'),
            ArraySpec((loss_steps,), np.float32, 'rollout_values'),
        )

        # There's no way to control where the ReplayActors are located in a multi-node setup, so
        # this function spins up more and more actors until it finds enough
        # which are on the local host.
        self.replay_actors = create_colocated(ReplayActor, [
            tensor_spec,
            num_replay_buffer_shards,
            self.config["learning_starts"],
            self.config["buffer_size"],
            self.config["train_batch_size"],
            self.config["prioritized_replay_alpha"],
            self.config["prioritized_replay_beta"],
            self.config["prioritized_replay_eps"],
            self.config["multiagent"]["replay_mode"],
            1,  # replay_sequence_length,
            self.config['input_steps'],
        ], num_replay_buffer_shards)
        
        # Start the learner thread on the local host (where the ReplayActors are located).
        self.learner_thread = LearnerThread(workers.local_worker())
        self.learner_thread.start()
        
        # We execute the following steps concurrently:
        # (1) Generate rollouts and store them in our replay buffer actors. Update
        # the weights of the worker that generated the batch.
        self.store_op = self._build_store_op(workers, self.replay_actors, self.learner_thread)

        # (2) Read experiences from the replay buffer actors and send to the
        # learner thread via its in-queue.
        self.replay_op = self._build_replay_op(self.replay_actors, self.learner_thread)

        # (3) Get priorities back from learner thread and apply them to the
        # replay buffer actors.
        self.update_op = self._build_update_op(self.learner_thread)
        
        return self._build_merged_op(
            self.store_op,
            self.replay_op,
            self.update_op,
            workers,
            self.replay_actors,
            self.learner_thread)

    def _build_store_op(self, workers, replay_actors, learner_thread):
        rollouts = ParallelRollouts(
            workers,
            self._global_vars,
            mode='async',
            num_async=self.config['max_sample_requests_in_flight_per_worker'])
        store_op = rollouts.for_each(CalculatePriorities(self.config['n_step'], self.config['gamma'])) \
            .for_each(StoreToReplayBuffer(actors=replay_actors))

        if workers.remote_workers():
            store_op = store_op.zip_with_source_actor() \
                .for_each(BroadcastUpdateLearnerWeights(
                    learner_thread,
                    workers,
                    broadcast_interval=self.config['broadcast_interval']))

        return store_op

    def _build_replay_op(self, replay_actors, learner_thread):
        return Replay(actors=replay_actors, num_async=4) \
                    .zip_with_source_actor() \
                    .for_each(Enqueue(learner_thread.inqueue))

    def _build_update_op(self, learner_thread):
        # Update experience priorities post learning.
        def update_prio_and_stats(item: Tuple[ActorHandle, dict, int]) -> None:
            actor, prio_dict, count = item
            actor.update_priorities.remote(prio_dict)
            metrics = LocalIterator.get_metrics()
            # Manually update the steps trained counter since the learner thread
            # is executing outside the pipeline.
            metrics.counters[STEPS_TRAINED_COUNTER] += count
            metrics.timers['learner_dequeue'] = learner_thread.queue_timer
            metrics.timers['learner_grad'] = learner_thread.grad_timer
            metrics.timers['learner_overall'] = learner_thread.overall_timer
            return metrics

        return Dequeue(learner_thread.outqueue, check=learner_thread.is_alive) \
            .for_each(update_prio_and_stats)

    def _build_merged_op(self, store_op, replay_op, update_op, workers, replay_actors, learner_thread):
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

        return StandardMetricsReporting(merged_op, workers, self.config, self._global_vars) \
            .for_each(add_muzero_metrics)

    def _build_workers(self, global_vars):
        return WorkerSet(
            self.env_creator,
            self._policy,
            self.config,
            global_vars,
            num_workers=self.config['num_workers'],
            logdir=self.logdir)

    def setup(self, config):
        self.config = config

        env = self._env_id
        if env is not None:
            config["env"] = env
            if _global_registry.contains(ENV_CREATOR, env):
                self.env_creator = _global_registry.get(ENV_CREATOR, env)
            else:
                import gym
                self.env_creator = lambda env_config: gym.make(env)
        else:
            raise Exception('self._env_id should not be None.')

        self._policy = MuZeroTFPolicy

        self.workers = self._build_workers(self._global_vars)

        self._global_op = self._build_global_op(self.workers)

    def step(self):
        ts = self.workers.local_worker().timestep
        interval = self.config['memory_reset_interval']
        if interval > 0 and (ts - self._last_mem_reset) >= interval:
            # Reset memory by killing workers and respawning.
            self._last_mem_reset = ts
            self.workers.remove_workers(self.config['num_workers'])
            self.workers.add_workers(self.config['num_workers'])
            self.store_op = self._build_store_op(self.workers, self.replay_actors, self.learner_thread)
            self._global_op = self._build_merged_op(
                self.store_op,
                self.replay_op,
                self.update_op,
                self.workers,
                self.replay_actors,
                self.learner_thread)

        return next(self._global_op)

    def cleanup(self):
        if hasattr(self, "workers"):
            self.workers.stop()
        if hasattr(self, "optimizer") and self.optimizer:
            self.optimizer.stop()

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        checkpoint_path = os.path.join(checkpoint_dir,
                                       "checkpoint-{}".format(self.iteration))
        pickle.dump(self.__getstate__(), open(checkpoint_path, "wb"))

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        extra_data = pickle.load(open(checkpoint_path, "rb"))
        self.__setstate__(extra_data)

    def __getstate__(self) -> dict:
        state = {}
        if hasattr(self, "workers"):
            state["worker"] = self.workers.local_worker().save()
        if hasattr(self, "optimizer") and hasattr(self.optimizer, "save"):
            state["optimizer"] = self.optimizer.save()
        return state

    def __setstate__(self, state: dict):
        if "worker" in state:
            self.workers.local_worker().restore(state["worker"])
            remote_state = ray.put(state["worker"])
            for r in self.workers.remote_workers():
                r.restore.remote(remote_state)
        if "optimizer" in state:
            self.optimizer.restore(state["optimizer"])

    def get_policy(self):
        return self.workers.local_worker().policy

    @classmethod
    def default_resource_request(
            cls, config: dict) -> Resources:
        cf = dict(cls._default_config(config.get('action_type', None)), **config)
        num_workers = cf["num_workers"]
        return Resources(
            cpu=cf["num_cpus_for_driver"],
            gpu=cf["num_gpus"],
            memory=cf["memory"],
            object_store_memory=cf["object_store_memory"],
            extra_cpu=cf["num_cpus_per_worker"] * num_workers,
            extra_gpu=cf["num_gpus_per_worker"] * num_workers,
            extra_memory=cf["memory_per_worker"] * num_workers,
            extra_object_store_memory=cf["object_store_memory_per_worker"] *
            num_workers)

    @classmethod
    def _default_config(cls, action_type):
        if action_type == 'atari' or action_type is None:
            return ATARI_DEFAULT_CONFIG
        elif action_type == 'board':
            return BOARD_DEFAULT_CONFIG
        else:
            raise Exception(f"Unknown action type: {action_type}")
        
