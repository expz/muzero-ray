from __future__ import annotations

import queue
import random
from typing import Dict, List

from ray.util.iter import from_actors, LocalIterator, _NextValueNotReady
from ray.util.iter_metrics import SharedMetrics

from muzero.replay_buffer import LocalReplayBuffer, PRIORITIES
from muzero.sample_batch import SampleBatch


class CalculatePriorities:
    def __init__(self, bootstrap_steps, gamma):
        self.n = bootstrap_steps
        self.gamma = gamma
        self.episodes: Dict[int, queue.Queue] = {}

    def get_next_batch(self, q: queue.Queue) -> SampleBatch:
        search_value = 0
        observed_return = 0
        for m in range(self.n):
            b, k = q.get_nowait()
            if m < self.n - 1:
                observed_return += self.gamma**m * b[SampleBatch.REWARDS][k]
            else:
                observed_return += b[SampleBatch.VF_PREDS][k]
            q.put((b, k))
        b, k = q.get_nowait()
        search_value = b[SampleBatch.VF_PREDS][k]
        b_frame = b.slice(k, k + 1)
        b_frame[PRIORITIES] = [abs(search_value - observed_return)]
        b_frame[SampleBatch.VF_PREDS] = [observed_return]
        return b_frame

    def __call__(self, batch: SampleBatch):
        """
        Assumes that batch is ordered by step number, although
        different episodes can be intermixed.
        """
        next_batches = []
        for i, eps_id in enumerate(batch[SampleBatch.EPS_ID]):
            if eps_id not in self.episodes:
                assert not batch[SampleBatch.DONES][i]
                self.episodes[eps_id] = queue.Queue(maxsize=self.n)
            q = self.episodes[eps_id]
            q.put_nowait((batch, i))
            if batch[SampleBatch.DONES][i]:
                # Force calculation of search value with less than n steps
                s = q.qsize()
                for _ in range(s):
                    while q.qsize() < self.n:
                        q.put((batch, i), timeout=0.1)
                    next_batches.append(self.get_next_batch(q))
                del self.episodes[eps_id]
            elif q.qsize() == self.n:
                next_batches.append(self.get_next_batch(q))
        if not next_batches:
            return None
        metrics = LocalIterator.get_metrics()
        metrics.info['calculate_priorities_queue_count'] = len(self.episodes)
        return SampleBatch.concat_samples(next_batches)


class StoreToReplayBuffer:
    """Callable that stores data into replay buffer actors.

    If constructed with a local replay actor, data will be stored into that
    buffer. If constructed with a list of replay actor handles, data will
    be stored randomly among those actors.

    This should be used with the .for_each() operator on a rollouts iterator.
    The batch that was stored is returned.

    Examples:
        >>> actors = [ReplayActor.remote() for _ in range(4)]
        >>> rollouts = ParallelRollouts(...)
        >>> store_op = rollouts.for_each(StoreToReplayActors(actors=actors))
        >>> next(store_op)
        SampleBatch(...)

    Code From:
        https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/execution/replay_ops.py
    """

    def __init__(self,
                 *,
                 local_buffer: LocalReplayBuffer = None,
                 actors: List["ActorHandle"] = None):
        if bool(local_buffer) == bool(actors):
            raise ValueError(
                "Exactly one of local_buffer and replay_actors must be given.")

        if local_buffer:
            self.local_actor = local_buffer
            self.replay_actors = None
        else:
            self.local_actor = None
            self.replay_actors = actors

    def __call__(self, batch: SampleBatch):
        if batch is not None:
            if self.local_actor:
                self.local_actor.add_batch(batch)
            else:
                actor = random.choice(self.replay_actors)
                actor.add_batch.remote(batch)
        #return batch


def Replay(*,
           local_buffer: LocalReplayBuffer = None,
           actors: List["ActorHandle"] = None,
           num_async=4):
    """Replay experiences from the given buffer or actors.

    This should be combined with the StoreToReplayActors operation using the
    Concurrently() operator.

    Arguments:
        local_buffer (LocalReplayBuffer): Local buffer to use. Only one of this
            and replay_actors can be specified.
        actors (list): List of replay actors. Only one of this and
            local_buffer can be specified.
        num_async (int): In async mode, the max number of async
            requests in flight per actor.

    Examples:
        >>> actors = [ReplayActor.remote() for _ in range(4)]
        >>> replay_op = Replay(actors=actors)
        >>> next(replay_op)
        SampleBatch(...)

    Code From:
        https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/execution/replay_ops.py
    """

    if bool(local_buffer) == bool(actors):
        raise ValueError(
            "Exactly one of local_buffer and replay_actors must be given.")

    if actors:
        replay = from_actors(actors)
        return replay.gather_async(
            num_async=num_async).filter(lambda x: x is not None)

    def gen_replay(_):
        while True:
            item = local_buffer.replay()
            if item is None:
                yield _NextValueNotReady()
            else:
                yield item

    return LocalIterator(gen_replay, SharedMetrics())
