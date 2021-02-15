from typing import Any, TypeVar, Callable, List, Union

import gym
import ray

from muzero.global_vars import GlobalVars
from muzero.policy import Policy
from muzero.rollout_worker import RolloutWorker


TrainerConfigDict = dict
EnvType = gym.Env


class WorkerSet:
    """Represents a set of RolloutWorkers.

    There must be one local worker copy, and zero or more remote workers.
    """

    def __init__(self,
                 env_creator: Callable[[Any], EnvType],
                 policy: Policy,
                 trainer_config: TrainerConfigDict,
                 global_vars: GlobalVars,
                 num_workers: int = 0,
                 logdir: str = None):
        """Create a new WorkerSet and initialize its workers.

        Arguments:
            env_creator (func): Function that returns env given env config.
            policy (cls): rllib.policy.Policy class.
            trainer_config (dict): Optional dict that extends the common
                config of the Trainer class.
            num_workers (int): Number of remote rollout workers to create.
            logdir (str): Optional logging directory for workers.
            _setup (bool): Whether to setup workers. This is only for testing.
        """

        self._env_creator = env_creator
        self._policy = policy
        self._config = trainer_config
        self._global_vars = global_vars
        self._logdir = logdir

        # Always create a local worker
        self._local_worker = RolloutWorker(
            env_creator, policy, self._config, 1, 0, global_vars)

        # Create a number of remote workers
        self._remote_workers = []
        self._num_workers = 0
        self.add_workers(num_workers)

    def local_worker(self) -> RolloutWorker:
        return self._local_worker

    def remote_workers(self) -> List[RolloutWorker]:
        return self._remote_workers

    def sync_weights(self) -> None:
        """Syncs weights of remote workers with the local worker."""
        if self.remote_workers():
            weights = ray.put(self.local_worker().get_weights())
            for e in self.remote_workers():
                e.set_weights.remote(weights)

    def stop(self) -> None:
        """Stop all rollout workers."""
        self.local_worker().stop()
        for w in self.remote_workers():
            w.stop.remote()
            w.__ray_terminate__.remote()

    def add_workers(self, num_workers: int) -> None:
        """Creates and add a number of remote workers to this worker set.

        Args:
            num_workers (int): The number of remote Workers to add to this
                WorkerSet.
        """
        cls = ray.remote(
            num_cpus=self._config["num_cpus_per_worker"],
            num_gpus=self._config["num_gpus_per_worker"],
            memory=self._config["memory_per_worker"],
            object_store_memory=self._config["object_store_memory_per_worker"],
            resources=self._config["custom_resources_per_worker"]
        )(RolloutWorker).remote
        self._remote_workers.extend([
            cls(
                self._env_creator,
                self._policy,
                self._config,
                self._num_workers + num_workers,
                self._num_workers + i + 1,
                self._global_vars)
            for i in range(num_workers)
        ])
        self._num_workers += num_workers

    def remove_workers(self, num_workers: int) -> None:
        while num_workers > 0:
            if not self._remote_workers:
                break
            worker = self._remote_workers.pop()
            worker.shutdown.remote()
            self._num_workers -= 1
            num_workers -= 1