import logging
from typing import Any, List
import time

import ray
from ray.util.iter import LocalIterator

from muzero.global_vars import GlobalVars
from muzero.metrics import collect_episodes, summarize_episodes
from muzero.policy import STEPS_SAMPLED_COUNTER
from muzero.worker_set import WorkerSet


logger = logging.getLogger(__name__)


class RecordWorkerStats:
    TIMEOUT_SECONDS = 18.0

    def __init__(self, workers):
        self.workers = workers.remote_workers()
    
    def __call__(self, batch):
        pending = [
            worker.get_metrics.remote() for worker in self.workers
        ]
        collected, to_be_collected = ray.wait(
            pending, num_returns=len(pending), timeout=self.TIMEOUT_SECONDS)
        if pending and len(collected) == 0:
            logger.warning(
                "WARNING: collected no worker metrics in {} seconds".format(
                    self.TIMEOUT_SECONDS))
            return batch
        metrics = ray.get(collected)
        stats = {
            'mcts': {},
            'mem': {},
        }
        stats['mcts'].update(metrics[0]['mcts'])
        action_space_size = 0
        for key in stats['mcts']:
            if 'action_' in key and len(key) > 12:
                # Count action_{i}_count keys and skip action_count key
                action_space_size += 1
        for metric in metrics[1:]:
            d = metric['mcts']
            for key in d:
                stats['mcts'][key] += d[key]
        for i in range(action_space_size):
            stats['mcts'][f'action_{i}_count_pct'] = stats['mcts'][f'action_{i}_count'] / stats['mcts']['action_count']
        for i, metric in enumerate(metrics):
            for key in metric['mem']:
                stats['mem'][f'worker_{i}_{key}'] = metric['mem'][key]
        LocalIterator.get_metrics().info.update(stats)
        return batch


def StandardMetricsReporting(
        train_op: LocalIterator[Any],
        workers: WorkerSet,
        config: dict,
        global_vars: GlobalVars,
        selected_workers: List["ActorHandle"] = None) -> LocalIterator[dict]:
    """Operator to periodically collect and report metrics.

    Arguments:
        train_op (LocalIterator): Operator for executing training steps.
            We ignore the output values.
        workers (WorkerSet): Rollout workers to collect metrics from.
        config (dict): Trainer configuration, used to determine the frequency
            of stats reporting.
        selected_workers (list): Override the list of remote workers
            to collect metrics from.

    Returns:
        A local iterator over training results.

    Examples:
        >>> train_op = ParallelRollouts(...).for_each(TrainOneStep(...))
        >>> metrics_op = StandardMetricsReporting(train_op, workers, config)
        >>> next(metrics_op)
        {"episode_reward_max": ..., "episode_reward_mean": ..., ...}

    Code From:
        https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/execution/metric_ops.py
    """
    output_op = train_op \
        .filter(OncePerTimestepsElapsed(config["timesteps_per_iteration"])) \
        .filter(OncePerTimeInterval(config["min_iter_time_s"])) \
        .for_each(CollectMetrics(
            workers, global_vars,
            min_history=config["metrics_smoothing_episodes"],
            timeout_seconds=config["collect_metrics_timeout"],
            selected_workers=selected_workers))
    return output_op


class CollectMetrics:
    """Callable that collects metrics from workers.

    The metrics are smoothed over a given history window.

    This should be used with the .for_each() operator. For a higher level
    API, consider using StandardMetricsReporting instead.

    Examples:
        >>> output_op = train_op.for_each(CollectMetrics(workers))
        >>> print(next(output_op))
        {"episode_reward_max": ..., "episode_reward_mean": ..., ...}

    Code From:
        https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/execution/metric_ops.py
    """

    def __init__(self,
                 workers: WorkerSet,
                 global_vars: GlobalVars,
                 min_history: int = 100,
                 timeout_seconds: float = 180,
                 selected_workers: List["ActorHandle"] = None):
        self.workers = workers
        self.global_vars = global_vars
        self.episode_history = []
        self.to_be_collected = []
        self.min_history = min_history
        self.timeout_seconds = timeout_seconds
        self.selected_workers = selected_workers

    def __call__(self, _):
        # Collect worker metrics.
        episodes, self.to_be_collected = collect_episodes(
            self.workers.local_worker(),
            self.selected_workers or self.workers.remote_workers(),
            self.to_be_collected,
            timeout_seconds=self.timeout_seconds)
        orig_episodes = list(episodes)
        missing = self.min_history - len(episodes)
        if missing > 0:
            episodes.extend(self.episode_history[-missing:])
            assert len(episodes) <= self.min_history
        self.episode_history.extend(orig_episodes)
        self.episode_history = self.episode_history[-self.min_history:]
        res = summarize_episodes(episodes, orig_episodes)

        # Add in iterator metrics.
        metrics = LocalIterator.get_metrics()
        metrics.counters[STEPS_SAMPLED_COUNTER] = ray.get(self.global_vars.get_count.remote('timestep'))

        timers = {}
        counters = {}
        info = {}
        info.update(metrics.info)
        for k, counter in metrics.counters.items():
            counters[k] = counter
        for k, timer in metrics.timers.items():
            timers["{}_time_ms".format(k)] = round(timer.mean * 1000, 3)
            if timer.has_units_processed():
                timers["{}_throughput".format(k)] = round(
                    timer.mean_throughput, 3)
        res.update({
            "num_healthy_workers": len(self.workers.remote_workers()),
            "timesteps_total": metrics.counters[STEPS_SAMPLED_COUNTER],
        })
        res["timers"] = timers
        res["info"] = info
        res["info"].update(counters)
        return res


class OncePerTimeInterval:
    """Callable that returns True once per given interval.

    This should be used with the .filter() operator to throttle / rate-limit
    metrics reporting. For a higher-level API, consider using
    StandardMetricsReporting instead.

    Examples:
        >>> throttled_op = train_op.filter(OncePerTimeInterval(5))
        >>> start = time.time()
        >>> next(throttled_op)
        >>> print(time.time() - start)
        5.00001  # will be greater than 5 seconds

    Code From:
        https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/execution/metric_ops.py
    """

    def __init__(self, delay):
        self.delay = delay
        self.last_called = 0

    def __call__(self, item):
        if self.delay <= 0.0:
            return True
        now = time.time()
        if now - self.last_called > self.delay:
            self.last_called = now
            return True
        return False


class OncePerTimestepsElapsed:
    """Callable that returns True once per given number of timesteps.

    This should be used with the .filter() operator to throttle / rate-limit
    metrics reporting. For a higher-level API, consider using
    StandardMetricsReporting instead.

    Examples:
        >>> throttled_op = train_op.filter(OncePerTimestepsElapsed(1000))
        >>> next(throttled_op)
        # will only return after 1000 steps have elapsed

    Code From:
        https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/execution/metric_ops.py
    """

    def __init__(self, delay_steps):
        self.delay_steps = delay_steps
        self.last_called = 0

    def __call__(self, item):
        if self.delay_steps <= 0:
            return True
        metrics = LocalIterator.get_metrics()
        now = metrics.counters[STEPS_SAMPLED_COUNTER]
        if now - self.last_called >= self.delay_steps:
            self.last_called = now
            return True
        return False