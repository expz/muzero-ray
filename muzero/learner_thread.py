"""
Code from https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/agents/dqn/learner_thread.py
and https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/utils/window_stat.py
"""
from __future__ import annotations

import queue
import threading

from ray.util.timer import _Timer as TimerStat

from muzero.metrics import get_learner_stats
from muzero.policy import LEARNER_STATS_KEY
from muzero.sample_batch import DEFAULT_POLICY_ID

LEARNER_QUEUE_MAX_SIZE = 16


import numpy as np


class WindowStat:
    def __init__(self, name, n):
        self.name = name
        self.items = [None] * n
        self.idx = 0
        self.count = 0

    def push(self, obj):
        self.items[self.idx] = obj
        self.idx += 1
        self.count += 1
        self.idx %= len(self.items)

    def stats(self):
        if not self.count:
            _quantiles = []
        else:
            _quantiles = np.nanpercentile(self.items[:self.count],
                                          [0, 10, 50, 90, 100]).tolist()
        return {
            self.name + "_count": int(self.count),
            self.name + "_mean": float(np.nanmean(self.items[:self.count])),
            self.name + "_std": float(np.nanstd(self.items[:self.count])),
            self.name + "_quantiles": _quantiles,
        }


class LearnerThread(threading.Thread):
    """Background thread that updates the local model from replay data.
    The learner thread communicates with the main thread through Queues. This
    is needed since Ray operations can only be run on the main thread. In
    addition, moving heavyweight gradient ops session runs off the main thread
    improves overall throughput.
    """

    def __init__(self, local_worker):
        threading.Thread.__init__(self)
        self.learner_queue_size = WindowStat("size", 50)
        self.local_worker = local_worker
        self.inqueue = queue.Queue(maxsize=LEARNER_QUEUE_MAX_SIZE)
        self.outqueue = queue.Queue()
        self.queue_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.overall_timer = TimerStat()
        self.daemon = True
        self.weights_updated = False
        self.stopped = False
        self.stats = {}

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        with self.overall_timer:
            with self.queue_timer:
                ra, batch = self.inqueue.get()
            if batch is not None:
                prio_dict = {}
                with self.grad_timer:
                    info = self.local_worker.learn_on_batch(batch)
                    pid = DEFAULT_POLICY_ID
                    p = info.get(
                        "replay_p",
                        info[LEARNER_STATS_KEY].get("replay_p"))
                    prio_dict[pid] = (batch.data.get("batch_indexes"), p)
                    self.stats[pid] = get_learner_stats(info)
                    self.grad_timer.push_units_processed(batch.count)
                self.outqueue.put((ra, prio_dict, batch.count))
            self.learner_queue_size.push(self.inqueue.qsize())
            self.weights_updated = True
            self.overall_timer.push_units_processed(batch and batch.count
                                                    or 0)