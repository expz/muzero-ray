"""
Some code from https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/evaluation/rollout_worker.py
"""

from __future__ import annotations

import gc
import os
import pickle
import psutil
import queue
import resource
import time
from typing import Callable, TypeVar, List

import numpy as np
from ray.util.iter import ParallelIteratorWorker

from muzero.sample_batch import SampleBatch
from muzero.rollout_metrics import PerfStats, RolloutMetrics
from muzero.util import getsize

T = TypeVar("T")

_global_worker = None


def get_global_worker():
    """Returns a handle to the active rollout worker in this process."""

    global _global_worker
    return _global_worker


class RolloutWorker(ParallelIteratorWorker):
    def __init__(
            self,
            env_creator,
            policy_cls,
            config,
            num_workers,
            worker_index):

        global _global_worker
        _global_worker = self

        self._env_creator = env_creator
        self.policy_cls = policy_cls
        self.config = config
        self._num_workers = num_workers
        self._worker_index = worker_index
        self.global_vars: dict = {}

        self._n = -1
        self._m = -1

        def rollout():
            while True:
                yield self.sample()

        ParallelIteratorWorker.__init__(self, rollout, False)

        assert config['envs_per_worker'] > 0
        self.envs = [
            env_creator({})
            for _ in range(config['envs_per_worker'])
        ]
        self.obs = [env.reset() for env in self.envs]
        self.step_id = [0 for _ in self.envs]
        self.eps_id = [self._generate_eps_id() for _ in self.envs]
        self.total_reward = [0 for _ in self.envs]

        self.metrics_queue = queue.Queue()
        self.perf_stats = [PerfStats() for _ in self.envs]

        self.policy = policy_cls(
            self.envs[0].observation_space,
            self.envs[0].action_space,
            config)
        print(f'Worker {self._worker_index} initialized.')

    def learn_on_batch(self, batch):
        return self.policy.learn_on_batch(batch)

    def _generate_eps_id(self):
        self._n += 1
        return self._n * self._num_workers + self._worker_index

    def _generate_unroll_id(self):
        self._m += 1
        return self._m * self._num_workers + self._worker_index

    def sample(self):
        N = self.config['replay_batch_size']
        batch = {}
        batch[SampleBatch.CUR_OBS] = []
        batch[SampleBatch.ACTIONS] = []
        batch['t'] = []
        batch[SampleBatch.EPS_ID] = []
        batch[SampleBatch.REWARDS] = []
        batch[SampleBatch.DONES] = []
        batch[SampleBatch.INFOS] = []
        batch[SampleBatch.UNROLL_ID] = [self._generate_unroll_id()] * N
        i = 0
        while i < N:
            obs = self.obs[:N - i]  # Usually no-op except maybe for final iter
            batch[SampleBatch.CUR_OBS].extend(obs)

            t2 = time.time()
            actions, _, b = self.policy.compute_actions(np.array(obs))
            inference_time = time.time() - t2
            for ps in self.perf_stats:
                ps.inference_time += inference_time

            t3 = time.time()
            for key in b:
                if key not in batch:
                    batch[key] = b[key]
                else:
                    batch[key] = np.concatenate([batch[key], b[key]], axis=0)
            batch[SampleBatch.ACTIONS].extend(actions.tolist())
            action_processing_time = time.time() - t3
            for ps in self.perf_stats:
                ps.action_processing_time += action_processing_time

            for (j, env), action in zip(enumerate(self.envs), actions.tolist()):
                if i + j >= N:
                    break
                self.perf_stats[j].iters += 1
                t0 = time.time()
                self.obs[j], reward, done, info = env.step(action)
                self.perf_stats[j].env_wait_time += time.time() - t0

                t1 = time.time()
                self.step_id[j] += 1
                self.total_reward[j] += reward
                batch['t'].append(self.step_id[j])
                batch[SampleBatch.EPS_ID].append(self.eps_id[j])
                batch[SampleBatch.REWARDS].append(reward)
                batch[SampleBatch.DONES].append(done)
                batch[SampleBatch.INFOS].append(info)
                self.perf_stats[j].raw_obs_processing_time += time.time() - t1

                if done:
                    self.metrics_queue.put(RolloutMetrics(
                        episode_length=self.step_id[j] + 1,
                        episode_reward=self.total_reward[j],
                        perf_stats=self.perf_stats[j].get()  
                    ))
                    self.perf_stats[j] = PerfStats()
                    self.total_reward[j] = 0.0
                    self.obs[j] = self.envs[j].reset()
                    self.eps_id[j] = self._generate_eps_id()
                    self.step_id[j] = 0
            i += len(self.envs)
        sample_batch = self.policy.postprocess_trajectory(SampleBatch(batch))
        del batch
        gc.collect()
        # print(f'worker_{self._worker_index}:', self.mem_stats())
        return sample_batch

    def get_weights(self):
        return self.policy.get_weights()

    def set_weights(self, weights, global_vars = None):
        self.policy.set_weights(weights)
        if global_vars:
            self.set_global_vars(global_vars)

    def set_global_vars(self, global_vars):
        self.global_vars = global_vars
        self.policy.set_global_vars(global_vars)

    def mem_stats(self):
        #rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        #shr = resource.getrusage(resource.RUSAGE_SELF).ru_ixrss / 1024.0
        process = psutil.Process(os.getpid())
        mi = process.memory_info()
        rss = mi.rss / 1024.0 / 1024.0
        shr = mi.shared / 1024.0 / 1024.0
        return {
            'worker_mem_mib': round(rss - shr, 2),
            'worker_shared_mib': round(shr, 2),
            'mcts_size_mib': round(getsize(self.policy.model.mcts) / 1024.0 / 1024.0, 2),
            'policy_size_mib': round(getsize(self.policy) / 1024.0 / 1024.0, 2),
            'worker_size_mib': round(getsize(self) / 1024.0 / 1024.0, 2)
        }

    def get_metrics(self):
        stats = {
            'mcts': {},
            'mem': {},
        }
        stats['mcts'].update(self.policy.model.mcts.stats())
        stats['mem'].update(self.mem_stats())
        return stats

    def get_episode_metrics(self) -> List[RolloutMetrics]:
        """Returns a list of new RolloutMetric objects from evaluation."""
        out = []
        while True:
            try:
                out.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return out

    def apply(self, func: Callable[["RolloutWorker"], T], *args) -> T:
        """Apply the given function to this rollout worker instance."""

        return func(self, *args)

    def save(self) -> str:
        state = self.policy.get_state()
        return pickle.dumps(state)

    def restore(self, objs: str) -> None:
        state = pickle.loads(objs)
        self.policy.set_state(state)