from __future__ import annotations

import logging
from typing import List, Tuple, Any

import ray
from ray.util.iter import from_actors, LocalIterator
from ray.util.iter_metrics import SharedMetrics

from muzero.global_vars import GlobalVars
from muzero.policy import STEPS_SAMPLED_COUNTER
from muzero.sample_batch import SampleBatch
from muzero.worker_set import WorkerSet

logger = logging.getLogger(__name__)

PolicyID = str
ModelGradients = List[Tuple[Any, Any]]


def standardized(array):
    """Normalize the values in an array.
    Arguments:
        array (np.ndarray): Array of values to normalize.
    Returns:
        array with zero mean and unit standard deviation.

    Code From:
        https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/execution/rollout_ops.py
    """
    return (array - array.mean()) / max(1e-4, array.std())


def ParallelRollouts(workers: WorkerSet,
                     global_vars: GlobalVars,
                     *,
                     mode: str = "bulk_sync",
                     num_async: int = 1) -> LocalIterator[SampleBatch]:
    """Operator to collect experiences in parallel from rollout workers.

    If there are no remote workers, experiences will be collected serially from
    the local worker instance instead.

    Arguments:
        workers (WorkerSet): set of rollout workers to use.
        mode (str): One of {'async', 'bulk_sync', 'raw'}.
            - In 'async' mode, batches are returned as soon as they are
              computed by rollout workers with no order guarantees.
            - In 'bulk_sync' mode, we collect one batch from each worker
              and concatenate them together into a large batch to return.
            - In 'raw' mode, the ParallelIterator object is returned directly
              and the caller is responsible for implementing gather and
              updating the timesteps counter.
        num_async (int): In async mode, the max number of async
            requests in flight per actor.

    Returns:
        A local iterator over experiences collected in parallel.

    Examples:
        >>> rollouts = ParallelRollouts(workers, mode="async")
        >>> batch = next(rollouts)
        >>> print(batch.count)
        50  # config.rollout_fragment_length

        >>> rollouts = ParallelRollouts(workers, mode="bulk_sync")
        >>> batch = next(rollouts)
        >>> print(batch.count)
        200  # config.rollout_fragment_length * config.num_workers

    Code From:
        https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/execution/rollout_ops.py

    Updates the STEPS_SAMPLED_COUNTER counter in the local iterator context.
    """

    # Ensure workers are initially in sync.
    workers.sync_weights()

    def report_timesteps(batch):
        ray.get(global_vars.add.remote('timestep', batch.count))
        metrics = LocalIterator.get_metrics()
        metrics.counters[STEPS_SAMPLED_COUNTER] += batch.count
        return batch

    if not workers.remote_workers():
        # Handle the serial sampling case.
        def sampler(_):
            while True:
                yield workers.local_worker().sample()

        return (LocalIterator(sampler, SharedMetrics())
                .for_each(report_timesteps))

    # Create a parallel iterator over generated experiences.
    rollouts = from_actors(workers.remote_workers())

    if mode == "bulk_sync":
        return rollouts \
            .batch_across_shards() \
            .for_each(lambda batches: SampleBatch.concat_samples(batches)) \
            .for_each(report_timesteps)
    elif mode == "async":
        return rollouts.gather_async(
            num_async=num_async).for_each(report_timesteps)
    elif mode == "raw":
        return rollouts
    else:
        raise ValueError("mode must be one of 'bulk_sync', 'async', 'raw', "
                         "got '{}'".format(mode))
