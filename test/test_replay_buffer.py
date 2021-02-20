from collections import defaultdict

import numpy as np
import pytest

from .conftest import random_sample_batch
from muzero.replay_buffer import PrioritizedReplayBuffer
from muzero.sample_batch import SampleBatch
from muzero.structure_list import ArraySpec

def test_probabilities(atari_tensor_spec):
    batch_size = 8
    frames_per_obs = 4
    buffer = PrioritizedReplayBuffer(64, atari_tensor_spec, frames_per_obs=frames_per_obs)
    batch = random_sample_batch(atari_tensor_spec, batch_size)
    first = True
    for s in batch.timeslices(1):
        p = (batch_size - 1) if first else 1.0
        buffer.add(s, p)
        first = False
    assert buffer.stats()['frame_count'] == 8
    counts = defaultdict(int)
    for _ in range(1000):
        s = buffer.sample(1)
        counts[(s[SampleBatch.EPS_ID][0], s['t'][0])] += 1
    print(counts)
    assert 0.75 * 500 < counts[(0, 0)] < 1.25 * 500, 'This can fail by chance. Run again to see if it succeeds.'
    assert 0.75 * 72 < counts[(0, 1)] < 1.25 * 72, 'This can fail by chance. Run again to see if it succeeds.'

def test_recall():
    batch_size = 8
    frames_per_obs = 5
    n_channels = 3
    tensor_spec = (
        ArraySpec((96, 96, n_channels), np.float32, name=SampleBatch.CUR_OBS),
        ArraySpec((1,), np.int32, name=SampleBatch.EPS_ID),
    )
    buffer = PrioritizedReplayBuffer(64, tensor_spec, frames_per_obs=frames_per_obs)
    batch = random_sample_batch(tensor_spec, batch_size)
    for s in batch.timeslices(1):
        buffer.add(s, 1.0)
    
    # Check shape of sample.
    s = buffer.sample(1)
    assert s[SampleBatch.CUR_OBS].shape == (1, 96, 96, n_channels * frames_per_obs)

    # Check shape of episode and step.
    assert s[SampleBatch.EPS_ID].shape == (1,)
    eps_id = s[SampleBatch.EPS_ID][0]
    assert s['t'].shape == (1,)
    step = s['t'][0]

    # Build expected value of sampled observation.
    frames = {}
    for b in batch.timeslices(1):
        frames[(b[SampleBatch.EPS_ID][0], b['t'][0])] = b[SampleBatch.CUR_OBS]
    f = step
    target_obs = []
    for _ in range(frames_per_obs):
        target_obs.append(frames[(eps_id, f)])
        if f > 0:
            f -= 1
    target_obs = np.array(target_obs)
    full_target_obs = np.transpose(np.array(target_obs), (1, 2, 3, 0, 4))
    full_target_obs = np.reshape(full_target_obs, (1, 96, 96, n_channels * frames_per_obs))

    # Check sampled observation matches expectations.
    assert np.allclose(s[SampleBatch.CUR_OBS], full_target_obs)

    # Check sampled first frame matches expectations.
    assert np.allclose(s[SampleBatch.CUR_OBS][:,:,:,:n_channels], target_obs[0])

def test_empty(atari_tensor_spec):
    buffer = PrioritizedReplayBuffer(64, atari_tensor_spec)
    assert buffer.stats()['frame_count'] == 0
    with pytest.raises(Exception):
        buffer.sample(1)

def test_update(atari_tensor_spec):
    batch_size = 8
    frames_per_obs = 4
    buffer = PrioritizedReplayBuffer(64, atari_tensor_spec, frames_per_obs=frames_per_obs)
    batch = random_sample_batch(atari_tensor_spec, batch_size)
    for s in batch.timeslices(1):
        buffer.add(s, 1.0)
    ps = [2.0, 3.0]
    buffer.update_priorities([1, 3], ps)
    assert buffer.get_priorities([1, 3]) == ps
    assert buffer.get_priorities([0, 0]) == [1.0, 1.0]
